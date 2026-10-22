//! Explicit Runge–Kutta stepping of a method-of-lines system `y' = F(t, y)`.
//!
//! Every scheme is stored as data in Shu–Osher form
//!
//! ```text
//! Y_0 = y_n
//! Y_i = Σ_{j<i} ( α_ij Y_j + Δt β_ij F(t_n + c_j Δt, Y_j) ),   i = 1..s
//! y_{n+1} = Y_s
//! ```
//!
//! with its declared strong-stability-preserving (SSP) coefficient
//! `C = min α_ij / β_ij`: when forward Euler is strongly stable (TVD,
//! positive, ...) for `Δt ≤ Δt_FE`, an SSP scheme is for `Δt ≤ C·Δt_FE`.
//!
//! Summation order is fixed and documented so results are reproducible bit
//! for bit: each component of a stage state is accumulated left to right over
//! `j = 0, 1, ..., i-1`, adding `α_ij·Y_j[m]` and then `(β_ij·Δt)·F_j[m]`,
//! skipping exactly-zero coefficients, the first term initializing the
//! accumulator. Forward Euler is therefore exactly `y + Δt·F`. Linear
//! invariants `w·y` are summed in increasing component order. No fused
//! multiply-add or reassociation is used.
//!
//! # Embedded error estimates and step control
//!
//! A scheme may declare embedded weights `b̂_j` of a lower order `p̂`; the
//! local error estimate is `e = Σ_j ((b_j − b̂_j)·Δt)·F_j` (Butcher weights
//! `b_j` of the scheme, `j` ascending, zero differences skipped). Under
//! [`StepPolicy::ErrorEstimate`] the step is accepted when the weighted RMS
//! norm `‖e‖ = sqrt(1/n Σ_m (e_m / (atol + rtol·max(|y_n,m|, |y_{n+1},m|)))²)`
//! is at most one and rejected otherwise; the next step is `Δt·f` with the
//! PI factor `f = safety · ‖e‖^(−0.7/k) · ‖e_prev‖^(0.4/k)`, `k = p̂ + 1`
//! (the plain `safety · ‖e‖^(−1/k)` on the first controlled step and after a
//! rejection, where no growth is allowed), clamped to
//! `[min_factor, max_factor]`. The controller memory travels in
//! [`ExplicitState::error_control`], so a restart replays bit for bit.
//!
//! # Low-storage execution
//!
//! [`ExplicitScheme::SspRk104LowStorage`] runs the same SSP-RK(10,4)
//! scheme in Ketcheson's two-register form (three live vectors of the
//! state dimension: two registers and one right-hand-side buffer, counted
//! in [`ExplicitStepReport::storage_vectors`]) instead of the `2s + 1`
//! vectors of the reference path. The nine forward-Euler substeps are
//! operation-for-operation identical to the reference; the two convex
//! recombinations are associated differently, so the results agree to a
//! small multiple of `ε·max|Y|` rather than bit for bit (see the test
//! `low_storage_ssp_rk104_matches_the_reference_within_the_declared_bound`
//! for the operation-count bound).

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{EvaluationContext, NumericError, SolveError};

/// Whether a state lies in the operator's admissible set (for example
/// positive density). Checked for every stage state an explicit step forms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Admissibility {
    Admissible,
    /// The state is outside the admissible set; `index` names the offending
    /// component when the owner knows it.
    Inadmissible {
        index: Option<usize>,
        reason: String,
    },
}

/// Method-of-lines right-hand side `F(t, y)` over flat `f64` state.
pub trait ExplicitOperator: Send + Sync {
    fn dimension(&self) -> usize;

    /// `output = F(time, state)`.
    fn rhs(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;

    /// The largest step for which forward Euler is stable for the owner's
    /// spatial operator at `state` (the step at CFL number 1), or `None`
    /// when the owner declares none. `f64::INFINITY` means unconstrained
    /// (for example zero wave speed). Must be positive.
    fn stable_step(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
    ) -> Result<Option<f64>, NumericError> {
        Ok(None)
    }

    /// Stage-wise admissibility of a state (for example positivity). An
    /// inadmissible stage rejects the step; it is a step-size question, not
    /// an evaluation failure.
    fn admissibility(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
    ) -> Result<Admissibility, NumericError> {
        Ok(Admissibility::Admissible)
    }

    /// Weight vectors `w_k` of linear invariants `w_k·y` (for example cell
    /// volumes per conserved component), each of length `dimension()`.
    fn linear_invariants(&self) -> &[Vec<f64>] {
        &[]
    }

    /// Net boundary/source rate `d(w_k·y)/dt` of each invariant at
    /// `(time, state)`, written to `output` (one entry per invariant).
    /// Returns `false` when the owner declares none, in which case the
    /// invariants are treated as closed (zero rate).
    fn invariant_rates(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
        _output: &mut [f64],
    ) -> Result<bool, NumericError> {
        Ok(false)
    }
}

/// An explicit Runge–Kutta scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplicitScheme {
    /// SSP-RK1, order 1, `C = 1`.
    ForwardEuler,
    /// SSP-RK2 (Heun), order 2, two stages, `C = 1`.
    SspRk2,
    /// SSP-RK3 (Shu–Osher 1988), order 3, three stages, `C = 1`.
    SspRk3,
    /// SSP-RK(10,4) (Ketcheson 2008), order 4, ten stages, `C = 6`.
    SspRk104,
    /// Classical RK4, order 4, four stages. Not SSP: no strong-stability
    /// guarantee at any step; for smooth problems only.
    Rk4,
    /// Bogacki–Shampine 3(2) (1989), order 3, four stages, embedded order 2.
    /// Not SSP. The fourth evaluation is at the step result (first-same-
    /// as-last) and feeds only the error estimate; it is not reused.
    BogackiShampine32,
    /// SSP-RK(10,4) executed in Ketcheson's two-register low-storage form;
    /// the same tableau, order, stages and SSP coefficient as
    /// [`ExplicitScheme::SspRk104`].
    SspRk104LowStorage,
}

impl ExplicitScheme {
    /// Classical order of accuracy.
    #[must_use]
    pub const fn order(self) -> u32 {
        match self {
            Self::ForwardEuler => 1,
            Self::SspRk2 => 2,
            Self::SspRk3 | Self::BogackiShampine32 => 3,
            Self::SspRk104 | Self::SspRk104LowStorage | Self::Rk4 => 4,
        }
    }

    /// Number of right-hand-side evaluations per step.
    #[must_use]
    pub const fn stages(self) -> usize {
        match self {
            Self::ForwardEuler => 1,
            Self::SspRk2 => 2,
            Self::SspRk3 => 3,
            Self::Rk4 | Self::BogackiShampine32 => 4,
            Self::SspRk104 | Self::SspRk104LowStorage => 10,
        }
    }

    /// Declared SSP coefficient, `None` for a scheme that is not SSP.
    #[must_use]
    pub const fn ssp_coefficient(self) -> Option<f64> {
        match self {
            Self::ForwardEuler | Self::SspRk2 | Self::SspRk3 => Some(1.0),
            Self::SspRk104 | Self::SspRk104LowStorage => Some(6.0),
            Self::Rk4 | Self::BogackiShampine32 => None,
        }
    }

    /// Order `p̂` of the embedded error estimate, `None` for a scheme that
    /// declares none (which [`StepPolicy::ErrorEstimate`] refuses).
    #[must_use]
    pub const fn embedded_order(self) -> Option<u32> {
        match self {
            Self::SspRk2 => Some(1),
            Self::SspRk3 | Self::BogackiShampine32 => Some(2),
            Self::ForwardEuler | Self::SspRk104 | Self::SspRk104LowStorage | Self::Rk4 => None,
        }
    }

    /// Whether the scheme runs on the two-register low-storage path.
    #[must_use]
    pub const fn low_storage(self) -> bool {
        matches!(self, Self::SspRk104LowStorage)
    }

    /// The scheme's Shu–Osher coefficients (and embedded weights, if any).
    #[must_use]
    pub fn tableau(self) -> ShuOsherTableau {
        let third = 1.0 / 3.0;
        let sixth = 1.0 / 6.0;
        let mut embedded = None;
        let (alpha, beta): (Vec<Vec<f64>>, Vec<Vec<f64>>) = match self {
            Self::ForwardEuler => (vec![vec![1.0]], vec![vec![1.0]]),
            Self::SspRk2 => {
                embedded = Some(vec![1.0, 0.0]);
                (
                    vec![vec![1.0], vec![0.5, 0.5]],
                    vec![vec![1.0], vec![0.0, 0.5]],
                )
            }
            Self::SspRk3 => {
                embedded = Some(vec![0.5, 0.5, 0.0]);
                (
                    vec![vec![1.0], vec![0.75, 0.25], vec![third, 0.0, 2.0 * third]],
                    vec![vec![1.0], vec![0.0, 0.25], vec![0.0, 0.0, 2.0 * third]],
                )
            }
            Self::Rk4 => (
                vec![
                    vec![1.0],
                    vec![1.0, 0.0],
                    vec![1.0, 0.0, 0.0],
                    vec![1.0, 0.0, 0.0, 0.0],
                ],
                vec![
                    vec![0.5],
                    vec![0.0, 0.5],
                    vec![0.0, 0.0, 1.0],
                    vec![sixth, third, third, sixth],
                ],
            ),
            Self::BogackiShampine32 => {
                embedded = Some(vec![7.0 / 24.0, 0.25, third, 0.125]);
                (
                    vec![
                        vec![1.0],
                        vec![1.0, 0.0],
                        vec![1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                    ],
                    vec![
                        vec![0.5],
                        vec![0.0, 0.75],
                        vec![2.0 / 9.0, third, 4.0 / 9.0],
                        vec![0.0, 0.0, 0.0, 0.0],
                    ],
                )
            }
            Self::SspRk104 | Self::SspRk104LowStorage => ketcheson_10_4(),
        };
        ShuOsherTableau {
            alpha,
            beta,
            embedded,
        }
    }
}

/// Ketcheson's low-storage SSP-RK(10,4) written in Shu–Osher form: five
/// forward-Euler substeps of `Δt/6`, the convex combination
/// `Y_5 = 3/5 Y_0 + 2/5 (Y_4 + Δt/6 F_4)`, four more substeps, and
/// `y_{n+1} = 1/25 Y_0 + 9/25 (Y_4 + Δt/6 F_4) + 3/5 Y_9 + Δt/10 F_9`.
fn ketcheson_10_4() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let sixth = 1.0 / 6.0;
    let mut alpha = Vec::with_capacity(10);
    let mut beta = Vec::with_capacity(10);
    for row in 1..=10_usize {
        let mut a = vec![0.0; row];
        let mut b = vec![0.0; row];
        match row {
            5 => {
                a[0] = 3.0 / 5.0;
                a[4] = 2.0 / 5.0;
                b[4] = 1.0 / 15.0;
            }
            10 => {
                a[0] = 1.0 / 25.0;
                a[4] = 9.0 / 25.0;
                a[9] = 3.0 / 5.0;
                b[4] = 3.0 / 50.0;
                b[9] = 1.0 / 10.0;
            }
            _ => {
                a[row - 1] = 1.0;
                b[row - 1] = sixth;
            }
        }
        alpha.push(a);
        beta.push(b);
    }
    (alpha, beta)
}

/// Shu–Osher coefficients of an explicit scheme: row `i - 1` holds
/// `α_ij`/`β_ij` for `j < i`, `i = 1..s`.
#[derive(Clone, Debug, PartialEq)]
pub struct ShuOsherTableau {
    alpha: Vec<Vec<f64>>,
    beta: Vec<Vec<f64>>,
    /// Butcher weights `b̂_j` of the embedded lower-order estimate.
    embedded: Option<Vec<f64>>,
}

/// The equivalent Butcher form `Y_i = y_n + Δt Σ_j a_ij F_j`,
/// `y_{n+1} = y_n + Δt Σ_j b_j F_j`, `c_i = Σ_j a_ij`.
#[derive(Clone, Debug, PartialEq)]
pub struct ButcherForm {
    pub a: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
}

impl ShuOsherTableau {
    #[must_use]
    pub fn stages(&self) -> usize {
        self.alpha.len()
    }

    /// `α_ij` for `j < i` (`i = 1..=stages()`).
    #[must_use]
    pub fn alpha(&self, stage: usize) -> &[f64] {
        &self.alpha[stage - 1]
    }

    /// `β_ij` for `j < i` (`i = 1..=stages()`).
    #[must_use]
    pub fn beta(&self, stage: usize) -> &[f64] {
        &self.beta[stage - 1]
    }

    /// Butcher weights `b̂_j` (one per right-hand-side evaluation) of the
    /// embedded lower-order estimate, when the scheme declares one.
    #[must_use]
    pub fn embedded_weights(&self) -> Option<&[f64]> {
        self.embedded.as_deref()
    }

    /// The SSP coefficient computed from the coefficients:
    /// `min α_ij/β_ij` over `β_ij > 0`, zero if any coefficient is negative
    /// or any `β_ij > 0` has `α_ij = 0`.
    #[must_use]
    pub fn ssp_coefficient_from_coefficients(&self) -> f64 {
        let mut coefficient = f64::INFINITY;
        for (alpha, beta) in self.alpha.iter().zip(&self.beta) {
            for (&a, &b) in alpha.iter().zip(beta) {
                if a < 0.0 || b < 0.0 {
                    return 0.0;
                }
                if b > 0.0 {
                    if a == 0.0 {
                        return 0.0;
                    }
                    coefficient = coefficient.min(a / b);
                }
            }
        }
        coefficient
    }

    /// The equivalent Butcher form, by the recursion
    /// `v_i = Σ_k α_ik v_k + β_i` on the coefficient vectors of `Δt F_j`.
    #[must_use]
    pub fn butcher_form(&self) -> ButcherForm {
        let stages = self.stages();
        let mut rows: Vec<Vec<f64>> = vec![vec![0.0; stages]];
        for i in 1..=stages {
            let mut row = vec![0.0; stages];
            for (k, (&a, &b)) in self.alpha(i).iter().zip(self.beta(i)).enumerate() {
                for (entry, &previous) in row.iter_mut().zip(&rows[k]) {
                    *entry += a * previous;
                }
                row[k] += b;
            }
            rows.push(row);
        }
        let b = rows.pop().expect("at least one stage");
        let c = rows.iter().map(|row| row.iter().sum()).collect();
        ButcherForm { a: rows, b, c }
    }
}

/// How the step size of each explicit step is chosen.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StepPolicy {
    /// A constant step (shortened only to land on the final time or by a
    /// caller's retry ceiling).
    Fixed { step: f64 },
    /// `Δt = target · stable_step(y_n)`, clipped to `max_step`. A stage
    /// whose own CFL number `Δt / stable_step(Y_i)` exceeds `max` rejects
    /// the step; a selected step below `min_step` is refused as
    /// [`SolveError::StepSizeCollapse`].
    Cfl {
        target: f64,
        max: f64,
        min_step: f64,
        max_step: f64,
    },
    /// Embedded error-estimate control (see the module documentation): the
    /// first step is `initial_step`, later steps are the PI controller's
    /// proposal carried in [`ExplicitState::error_control`], every step is
    /// clipped to `max_step`; a selected step below `min_step` is refused as
    /// [`SolveError::StepSizeCollapse`]. Requires a scheme with an embedded
    /// estimate ([`ExplicitScheme::embedded_order`]).
    ErrorEstimate {
        absolute_tolerance: f64,
        relative_tolerance: f64,
        /// Safety factor on the controller's step factor, in `(0, 1]`.
        safety: f64,
        min_step: f64,
        max_step: f64,
        initial_step: f64,
        /// Smallest step factor after an accepted or rejected step, in `(0, 1]`.
        min_factor: f64,
        /// Largest step factor after an accepted step, at least one.
        max_factor: f64,
    },
}

impl StepPolicy {
    pub(crate) fn validate(&self) -> Result<(), SolveError> {
        let invalid = |reason: &str| {
            Err(SolveError::InvalidConfiguration {
                reason: reason.into(),
            })
        };
        match *self {
            StepPolicy::Fixed { step } => {
                if !(step.is_finite() && step > 0.0) {
                    return invalid("fixed explicit step must be finite and positive");
                }
            }
            StepPolicy::Cfl {
                target,
                max,
                min_step,
                max_step,
            } => {
                if !(target.is_finite() && target > 0.0 && max.is_finite() && target <= max) {
                    return invalid("CFL target must be finite, positive and at most max");
                }
                if !(min_step.is_finite() && min_step > 0.0 && max_step >= min_step) {
                    return invalid(
                        "CFL minimum step must be finite and positive and at most the maximum step",
                    );
                }
            }
            StepPolicy::ErrorEstimate {
                absolute_tolerance,
                relative_tolerance,
                safety,
                min_step,
                max_step,
                initial_step,
                min_factor,
                max_factor,
            } => {
                let nonnegative = |value: f64| value.is_finite() && value >= 0.0;
                if !(nonnegative(absolute_tolerance)
                    && nonnegative(relative_tolerance)
                    && absolute_tolerance + relative_tolerance > 0.0)
                {
                    return invalid(
                        "error-estimate tolerances must be finite and nonnegative, not both zero",
                    );
                }
                if !(safety.is_finite() && safety > 0.0 && safety <= 1.0) {
                    return invalid("error-estimate safety factor must lie in (0, 1]");
                }
                if !(min_step.is_finite() && min_step > 0.0 && max_step >= min_step) {
                    return invalid(
                        "error-estimate minimum step must be finite and positive and at most the \
                         maximum step",
                    );
                }
                if !(initial_step.is_finite() && initial_step >= min_step) {
                    return invalid(
                        "error-estimate initial step must be finite and at least min_step",
                    );
                }
                if !(min_factor.is_finite()
                    && min_factor > 0.0
                    && min_factor <= 1.0
                    && max_factor.is_finite()
                    && max_factor >= 1.0)
                {
                    return invalid(
                        "error-estimate step factors must satisfy 0 < min_factor <= 1 <= max_factor",
                    );
                }
            }
        }
        Ok(())
    }

    /// The minimum step the policy refuses to go below, if it declares one.
    pub(crate) fn min_step(&self) -> Option<f64> {
        match *self {
            StepPolicy::Fixed { .. } => None,
            StepPolicy::Cfl { min_step, .. } | StepPolicy::ErrorEstimate { min_step, .. } => {
                Some(min_step)
            }
        }
    }

    /// The retry ceiling an inadmissible stage suggests: half the step under
    /// an adaptive policy, none under `Fixed`.
    pub(crate) fn halve_on_rejection(&self, step: f64) -> Option<f64> {
        match self {
            StepPolicy::Fixed { .. } => None,
            StepPolicy::Cfl { .. } | StepPolicy::ErrorEstimate { .. } => Some(0.5 * step),
        }
    }
}

/// Scheme and step policy of an explicit integration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplicitConfig {
    pub scheme: ExplicitScheme,
    pub step_policy: StepPolicy,
}

impl ExplicitConfig {
    fn validate(&self) -> Result<(), SolveError> {
        self.step_policy.validate()?;
        if matches!(self.step_policy, StepPolicy::ErrorEstimate { .. })
            && self.scheme.embedded_order().is_none()
        {
            return Err(SolveError::InvalidConfiguration {
                reason: format!(
                    "{:?} declares no embedded error estimate; StepPolicy::ErrorEstimate needs one",
                    self.scheme
                ),
            });
        }
        Ok(())
    }
}

/// The PI step controller's memory between accepted error-controlled steps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorControlMemory {
    /// The controller's proposal for the next step (before `max_step`, the
    /// retry ceiling and the final time are applied).
    pub next_step: f64,
    /// The accepted step's error norm, floored at `1e-4` (Hairer's
    /// `facold`), used by the PI term of the next step.
    pub previous_error_norm: f64,
}

/// Committed explicit-integration state. A rejected attempt never changes it.
///
/// Also the committed state of an IMEX integration
/// ([`crate::imex_step`]); `error_control` is `Some` only after a step
/// accepted under [`StepPolicy::ErrorEstimate`] and is omitted from the
/// serialized form when `None`, so states written before it existed read
/// back unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ExplicitStateData")]
pub struct ExplicitState {
    pub time: f64,
    pub values: Vec<f64>,
    pub accepted_steps: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_control: Option<ErrorControlMemory>,
}

#[derive(Deserialize)]
struct ExplicitStateData {
    time: f64,
    values: Vec<f64>,
    accepted_steps: u64,
    #[serde(default)]
    error_control: Option<ErrorControlMemory>,
}

impl TryFrom<ExplicitStateData> for ExplicitState {
    type Error = String;

    fn try_from(data: ExplicitStateData) -> Result<Self, Self::Error> {
        if !data.time.is_finite() {
            return Err("explicit state time must be finite".into());
        }
        NumericError::require_finite("explicit state", &data.values)
            .map_err(|error| error.to_string())?;
        if let Some(memory) = &data.error_control {
            if !(memory.next_step.is_finite() && memory.next_step > 0.0) {
                return Err("explicit error-control next step must be finite and positive".into());
            }
            if !(memory.previous_error_norm.is_finite() && memory.previous_error_norm >= 0.0) {
                return Err(
                    "explicit error-control previous error norm must be finite and nonnegative"
                        .into(),
                );
            }
        }
        Ok(Self {
            time: data.time,
            values: data.values,
            accepted_steps: data.accepted_steps,
            error_control: data.error_control,
        })
    }
}

impl ExplicitState {
    /// Validate an initial state: finite, of the operator's dimension, and
    /// admissible (an inadmissible initial state is
    /// [`SolveError::InadmissibleState`]).
    pub fn initialize(
        operator: &(impl ExplicitOperator + ?Sized),
        context: &EvaluationContext,
        time: f64,
        values: Vec<f64>,
    ) -> Result<Self, SolveError> {
        Self::initialize_with(operator.dimension(), time, values, |values| {
            operator.admissibility(context, time, values)
        })
    }

    pub(crate) fn initialize_with(
        dimension: usize,
        time: f64,
        values: Vec<f64>,
        admissibility: impl FnOnce(&[f64]) -> Result<Admissibility, NumericError>,
    ) -> Result<Self, SolveError> {
        if !time.is_finite() {
            return Err(NumericError::InvalidInput {
                message: "initial explicit time must be finite".into(),
            }
            .into());
        }
        NumericError::require_len("initial explicit state", values.len(), dimension)?;
        NumericError::require_finite("initial explicit state", &values)?;
        if let Admissibility::Inadmissible { index, reason } = admissibility(&values)? {
            return Err(SolveError::InadmissibleState { index, reason });
        }
        Ok(Self {
            time,
            values,
            accepted_steps: 0,
            error_control: None,
        })
    }

    pub(crate) fn validate(&self, dimension: usize, final_time: f64) -> Result<(), SolveError> {
        NumericError::require_len("explicit state", self.values.len(), dimension)?;
        NumericError::require_finite("explicit state", &self.values)?;
        if !self.time.is_finite() || !final_time.is_finite() || final_time <= self.time {
            return Err(NumericError::InvalidInput {
                message: "explicit final time must be finite and after the state time".into(),
            }
            .into());
        }
        if let Some(memory) = &self.error_control {
            if !(memory.next_step.is_finite() && memory.next_step > 0.0)
                || !(memory.previous_error_norm.is_finite() && memory.previous_error_norm >= 0.0)
            {
                return Err(NumericError::InvalidInput {
                    message: "explicit error-control memory must be finite and positive".into(),
                }
                .into());
            }
        }
        Ok(())
    }

    pub(crate) fn next_accepted_steps(&self) -> Result<u64, NumericError> {
        self.accepted_steps
            .checked_add(1)
            .ok_or_else(|| NumericError::InvalidInput {
                message: "accepted explicit step count overflowed".into(),
            })
    }
}

/// What determined the size of an accepted step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepLimit {
    /// `StepPolicy::Fixed`'s step.
    Fixed,
    /// `target · stable_step`.
    Cfl,
    /// `StepPolicy::Cfl`'s or `StepPolicy::ErrorEstimate`'s `max_step`.
    MaximumStep,
    /// The caller's retry ceiling (after a rejection).
    RetryCeiling,
    /// Shortened (or stretched by at most rounding) to land on the final time.
    FinalTime,
    /// `StepPolicy::ErrorEstimate`'s `initial_step` (no controller memory yet).
    InitialStep,
    /// The PI controller's proposal from the previous accepted step.
    ErrorControl,
}

/// The step one attempt takes, chosen by [`choose_step`].
pub(crate) struct StepChoice {
    pub(crate) step: f64,
    pub(crate) new_time: f64,
    pub(crate) limit: StepLimit,
}

/// Select the step of one attempt from the policy, the explicit stable-step
/// hint at the current state, the retry ceiling and the final time.
pub(crate) fn choose_step(
    policy: &StepPolicy,
    state: &ExplicitState,
    final_time: f64,
    retry_ceiling: Option<f64>,
    hint0: Option<f64>,
) -> Result<StepChoice, SolveError> {
    if retry_ceiling.is_some_and(|ceiling| !(ceiling.is_finite() && ceiling > 0.0)) {
        return Err(SolveError::InvalidConfiguration {
            reason: "explicit retry ceiling must be finite and positive".into(),
        });
    }
    let time = state.time;
    let (mut candidate, mut limit) = match *policy {
        StepPolicy::Fixed { step } => (step, StepLimit::Fixed),
        StepPolicy::Cfl {
            target, max_step, ..
        } => {
            let hint = hint0.ok_or_else(|| SolveError::InvalidConfiguration {
                reason: "a CFL step policy requires the operator to declare a stable step".into(),
            })?;
            let step = target * hint;
            if step > max_step {
                (max_step, StepLimit::MaximumStep)
            } else {
                (step, StepLimit::Cfl)
            }
        }
        StepPolicy::ErrorEstimate {
            max_step,
            initial_step,
            ..
        } => {
            let (step, limit) = match &state.error_control {
                Some(memory) => (memory.next_step, StepLimit::ErrorControl),
                None => (initial_step, StepLimit::InitialStep),
            };
            if step > max_step {
                (max_step, StepLimit::MaximumStep)
            } else {
                (step, limit)
            }
        }
    };
    if let Some(ceiling) = retry_ceiling.filter(|&ceiling| ceiling < candidate) {
        candidate = ceiling;
        limit = StepLimit::RetryCeiling;
    }
    if let Some(min_step) = policy.min_step() {
        if candidate < min_step {
            return Err(SolveError::StepSizeCollapse {
                time,
                step: candidate,
                minimum_step: min_step,
            });
        }
    }
    let remaining = final_time - time;
    // Each accepted step's `t + Δt` rounds by at most ε/2·|t|, so after `k`
    // steps the time coordinate carries up to `k·ε/2·max|t|` of rounding.
    let rounding =
        (state.accepted_steps as f64 + 4.0) * f64::EPSILON * time.abs().max(final_time.abs());
    let lands = candidate >= remaining || remaining - candidate <= rounding;
    let step = if lands {
        limit = StepLimit::FinalTime;
        remaining
    } else {
        candidate
    };
    let new_time = if lands { final_time } else { time + step };
    if new_time <= time {
        return Err(SolveError::StepSizeCollapse {
            time,
            step,
            minimum_step: 0.0,
        });
    }
    Ok(StepChoice {
        step,
        new_time,
        limit,
    })
}

/// Telemetry of the embedded error estimate of one accepted step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorControlReport {
    /// The weighted RMS error norm; accepted steps have `error_norm <= 1`.
    pub error_norm: f64,
    /// The PI controller's factor, clamped to `[min_factor, max_factor]`.
    pub step_factor: f64,
    /// `step · step_factor` clipped to `max_step`: the next proposal.
    pub next_step: f64,
    /// `k = p̂ + 1`, the exponent base of the controller.
    pub controller_order: u32,
}

/// `sqrt(1/n Σ (e_m / (atol + rtol·max(|y_m|, |z_m|)))²)`, summed in
/// component order; zero for an empty vector.
pub(crate) fn error_norm(
    absolute_tolerance: f64,
    relative_tolerance: f64,
    before: &[f64],
    after: &[f64],
    estimate: &[f64],
) -> f64 {
    if estimate.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for ((&y, &z), &e) in before.iter().zip(after).zip(estimate) {
        let scale = absolute_tolerance + relative_tolerance * y.abs().max(z.abs());
        let ratio = e / scale;
        sum += ratio * ratio;
    }
    (sum / estimate.len() as f64).sqrt()
}

/// The controller's decision after measuring `error_norm` on a step of
/// size `step`: `Ok((report, memory))` for an accepted step, `Err(ceiling)`
/// with the retry ceiling for a rejected one.
pub(crate) fn control_step(
    policy: &StepPolicy,
    embedded_order: u32,
    step: f64,
    error_norm: f64,
    previous: Option<&ErrorControlMemory>,
) -> Result<(ErrorControlReport, ErrorControlMemory), f64> {
    let StepPolicy::ErrorEstimate {
        safety,
        max_step,
        min_factor,
        max_factor,
        ..
    } = *policy
    else {
        unreachable!("control_step is only called under StepPolicy::ErrorEstimate")
    };
    let k = embedded_order + 1;
    let exponent = 1.0 / f64::from(k);
    if error_norm > 1.0 {
        let factor = (safety * error_norm.powf(-exponent)).clamp(min_factor, 1.0);
        return Err(step * factor);
    }
    let raw = if error_norm == 0.0 {
        max_factor
    } else {
        match previous {
            Some(memory) if memory.previous_error_norm > 0.0 => {
                safety
                    * error_norm.powf(-0.7 * exponent)
                    * memory.previous_error_norm.powf(0.4 * exponent)
            }
            _ => safety * error_norm.powf(-exponent),
        }
    };
    let step_factor = raw.clamp(min_factor, max_factor);
    let next_step = (step * step_factor).min(max_step);
    Ok((
        ErrorControlReport {
            error_norm,
            step_factor,
            next_step,
            controller_order: k,
        },
        ErrorControlMemory {
            next_step,
            previous_error_norm: error_norm.max(1.0e-4),
        },
    ))
}

/// Discrete balance of one linear invariant over one step.
///
/// `predicted` integrates the declared rates with the scheme's own
/// coefficients (`P_i = Σ α_ij P_j + Δt β_ij r_j`, `P_0 = before`), so for an
/// exactly conservative operator `balance_residual = after - predicted` is
/// pure round-off, bounded by a small multiple of `ε · scale`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InvariantBalance {
    pub before: f64,
    pub after: f64,
    pub predicted: f64,
    pub balance_residual: f64,
    /// `max(Σ|w_m y_m|)` over the step's start and end states.
    pub scale: f64,
}

/// Telemetry of an accepted explicit step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplicitStepReport {
    pub step: f64,
    pub step_limit: StepLimit,
    pub scheme: ExplicitScheme,
    pub stages: usize,
    pub ssp_coefficient: Option<f64>,
    /// `max_i Δt / stable_step(Y_i)` over the stage-input states, when the
    /// operator declares a stable step.
    pub cfl_realized: Option<f64>,
    /// Whether the invariant balances are corrected by declared rates.
    pub balance_corrected: bool,
    pub invariants: Vec<InvariantBalance>,
    /// The embedded error estimate, under `StepPolicy::ErrorEstimate` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_control: Option<ErrorControlReport>,
    /// Live vectors of the state dimension the step allocated, reported by
    /// the low-storage path only (the reference path keeps `2s + 1`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_vectors: Option<usize>,
}

/// Why an explicit step attempt was rejected.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExplicitRejection {
    /// Stage `stage` (`1..=s`, `s` being the step result) was inadmissible.
    InadmissibleStage {
        stage: usize,
        time: f64,
        index: Option<usize>,
        reason: String,
    },
    /// Stage `stage`'s CFL number exceeded the policy maximum.
    CflExceeded { stage: usize, cfl: f64, max: f64 },
    /// The embedded error estimate's weighted norm exceeded one.
    ErrorEstimateExceeded { error_norm: f64 },
}

impl fmt::Display for ExplicitRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InadmissibleStage {
                stage,
                time,
                index,
                reason,
            } => {
                write!(formatter, "stage {stage} at time {time} is inadmissible")?;
                if let Some(index) = index {
                    write!(formatter, " at component {index}")?;
                }
                write!(formatter, ": {reason}")
            }
            Self::CflExceeded { stage, cfl, max } => {
                write!(formatter, "stage {stage} CFL number {cfl} exceeds {max}")
            }
            Self::ErrorEstimateExceeded { error_norm } => {
                write!(formatter, "embedded error norm {error_norm} exceeds 1")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplicitAcceptedStep {
    pub state: ExplicitState,
    pub report: ExplicitStepReport,
}

/// A rejected attempt. The committed state is the caller's input, unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplicitRejectedStep {
    pub attempted_step: f64,
    pub rejection: ExplicitRejection,
    /// The retry ceiling to pass to the next attempt; `None` under
    /// `StepPolicy::Fixed`, which never shrinks its step on its own.
    pub suggested_step: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExplicitStepOutcome {
    Accepted(ExplicitAcceptedStep),
    Rejected(ExplicitRejectedStep),
}

pub(crate) fn checked_hint(stage: usize, hint: Option<f64>) -> Result<Option<f64>, NumericError> {
    match hint {
        Some(value) if value.is_nan() || value <= 0.0 => Err(NumericError::InvalidInput {
            message: format!("stable step at stage {stage} must be positive, got {value}"),
        }),
        other => Ok(other),
    }
}

pub(crate) fn weighted_sum(weights: &[f64], values: &[f64]) -> (f64, f64) {
    let mut sum = 0.0;
    let mut scale = 0.0;
    for (&weight, &value) in weights.iter().zip(values) {
        let term = weight * value;
        sum += term;
        scale += term.abs();
    }
    (sum, scale)
}

pub(crate) fn check_invariant_weights(
    invariants: &[Vec<f64>],
    dimension: usize,
) -> Result<(), NumericError> {
    for (k, weights) in invariants.iter().enumerate() {
        NumericError::require_len(&format!("invariant {k} weights"), weights.len(), dimension)?;
        NumericError::require_finite(&format!("invariant {k} weights"), weights)?;
    }
    Ok(())
}

/// Per-stage bookkeeping shared by the reference path, the low-storage path
/// and the IMEX stepper: finiteness and admissibility of every stage state,
/// the CFL number of every stage input, and the declared invariant rates.
pub(crate) struct StageMonitor<'a> {
    policy: &'a StepPolicy,
    step: f64,
    hint0: Option<f64>,
    cfl_realized: Option<f64>,
    cfl_threshold: Option<f64>,
    minimum_hint: Option<f64>,
    balance_corrected: Option<bool>,
    /// One rate vector per evaluated stage (the IMEX stepper stores the
    /// explicit rates followed by the implicit ones).
    rates: Vec<Vec<f64>>,
}

impl<'a> StageMonitor<'a> {
    pub(crate) fn new(policy: &'a StepPolicy, step: f64, hint0: Option<f64>) -> Self {
        let cfl_realized = hint0.map(|hint| step / hint);
        let cfl_threshold = match policy {
            StepPolicy::Cfl { max, .. } => Some(max.max(cfl_realized.unwrap_or(0.0))),
            StepPolicy::Fixed { .. } | StepPolicy::ErrorEstimate { .. } => None,
        };
        Self {
            policy,
            step,
            hint0,
            cfl_realized,
            cfl_threshold,
            minimum_hint: hint0,
            balance_corrected: None,
            rates: Vec::new(),
        }
    }

    pub(crate) fn cfl_realized(&self) -> Option<f64> {
        self.cfl_realized
    }

    pub(crate) fn balance_corrected(&self) -> bool {
        self.balance_corrected.unwrap_or(false)
    }

    pub(crate) fn rates(&self) -> &[Vec<f64>] {
        &self.rates
    }

    /// A newly formed stage state: finite (else an error) and admissible
    /// (else a rejection suggesting half the step under an adaptive policy).
    pub(crate) fn check_state(
        &self,
        stage: usize,
        stage_time: f64,
        values: &[f64],
        admissibility: Result<Admissibility, NumericError>,
    ) -> Result<Option<ExplicitRejectedStep>, SolveError> {
        if let Admissibility::Inadmissible { index, reason } = admissibility? {
            return Ok(Some(ExplicitRejectedStep {
                attempted_step: self.step,
                rejection: ExplicitRejection::InadmissibleStage {
                    stage,
                    time: stage_time,
                    index,
                    reason,
                },
                suggested_step: self.policy.halve_on_rejection(self.step),
            }));
        }
        let _ = values;
        Ok(None)
    }

    /// The stable-step hint at a stage input (stages after the first):
    /// consistency with stage 0, CFL bookkeeping, and the CFL rejection.
    pub(crate) fn check_hint(
        &mut self,
        stage: usize,
        hint: Option<f64>,
    ) -> Result<Option<ExplicitRejectedStep>, SolveError> {
        let hint = checked_hint(stage, hint)?;
        if hint.is_some() != self.hint0.is_some() {
            return Err(NumericError::InvalidInput {
                message: format!(
                    "stable step declared at stage 0 but not at stage {stage}, or the reverse"
                ),
            }
            .into());
        }
        if let Some(hint) = hint {
            let cfl = self.step / hint;
            self.cfl_realized = self.cfl_realized.map(|realized| realized.max(cfl));
            self.minimum_hint = self.minimum_hint.map(|minimum| minimum.min(hint));
            if let (Some(threshold), StepPolicy::Cfl { target, max, .. }) =
                (self.cfl_threshold, self.policy)
            {
                if cfl > threshold {
                    return Ok(Some(ExplicitRejectedStep {
                        attempted_step: self.step,
                        rejection: ExplicitRejection::CflExceeded {
                            stage,
                            cfl,
                            max: *max,
                        },
                        suggested_step: self.minimum_hint.map(|minimum| target * minimum),
                    }));
                }
            }
        }
        Ok(None)
    }

    /// Record the declared invariant rates of one evaluated stage;
    /// `declared == false` records zero rates. Declaring at some stages
    /// but not others is an error.
    pub(crate) fn record_rates(
        &mut self,
        stage: usize,
        declared: bool,
        mut rate: Vec<f64>,
    ) -> Result<(), SolveError> {
        if *self.balance_corrected.get_or_insert(declared) != declared {
            return Err(NumericError::InvalidInput {
                message: format!("invariant rates declared inconsistently at stage {stage}"),
            }
            .into());
        }
        if declared {
            NumericError::require_finite(&format!("invariant rates at stage {stage}"), &rate)?;
        } else {
            rate.fill(0.0);
        }
        self.rates.push(rate);
        Ok(())
    }
}

/// Attempt one explicit step from `state` toward `final_time`.
///
/// The step is chosen by `config.step_policy`, never exceeds
/// `retry_ceiling` (pass a rejection's `suggested_step`), and lands exactly
/// on `final_time` when the remaining interval is at most the chosen step
/// (or exceeds it only by `(k + 4)·ε·max(|t|, |final_time|)` after `k`
/// accepted steps, the accumulated rounding of the time coordinate, which
/// would otherwise force a degenerate sliver step).
///
/// Every stage state is checked finite and then admissible; an inadmissible
/// stage, or under `StepPolicy::Cfl` a stage whose CFL number exceeds
/// `max(max, cfl at stage 0)`, or under `StepPolicy::ErrorEstimate` an
/// error norm above one, is an [`ExplicitStepOutcome::Rejected`].
/// Operator failures, non-finite stage values or right-hand sides, and
/// malformed hints or invariant weights are errors, never NaN results.
pub fn explicit_step(
    operator: &(impl ExplicitOperator + ?Sized),
    context: &EvaluationContext,
    state: &ExplicitState,
    final_time: f64,
    config: &ExplicitConfig,
    retry_ceiling: Option<f64>,
) -> Result<ExplicitStepOutcome, SolveError> {
    config.validate()?;
    let dimension = operator.dimension();
    state.validate(dimension, final_time)?;
    let invariants = operator.linear_invariants();
    check_invariant_weights(invariants, dimension)?;

    let time = state.time;
    let hint0 = checked_hint(0, operator.stable_step(context, time, &state.values)?)?;
    let StepChoice {
        step,
        new_time,
        limit,
    } = choose_step(&config.step_policy, state, final_time, retry_ceiling, hint0)?;

    let scheme = config.scheme;
    let tableau = scheme.tableau();
    let stage_times = tableau.butcher_form().c;
    let stages = tableau.stages();
    let mut monitor = StageMonitor::new(&config.step_policy, step, hint0);
    let stage_time = |i: usize| {
        if i == stages {
            new_time
        } else if i == 0 {
            time
        } else {
            time + stage_times[i] * step
        }
    };

    let mut registers = Registers::default();
    let (new_values, estimate) = if scheme.low_storage() {
        match ssp_rk104_low_storage(
            operator,
            context,
            state,
            step,
            &stage_time,
            &mut monitor,
            &mut registers,
        )? {
            Ok(values) => (values, None),
            Err(rejected) => return Ok(ExplicitStepOutcome::Rejected(rejected)),
        }
    } else {
        match reference_stages(
            operator,
            context,
            state,
            step,
            &tableau,
            &stage_time,
            &mut monitor,
            matches!(config.step_policy, StepPolicy::ErrorEstimate { .. }),
        )? {
            Ok(result) => result,
            Err(rejected) => return Ok(ExplicitStepOutcome::Rejected(rejected)),
        }
    };

    let mut error_control = None;
    let mut memory = None;
    if let StepPolicy::ErrorEstimate {
        absolute_tolerance,
        relative_tolerance,
        ..
    } = config.step_policy
    {
        let estimate = estimate.expect("the reference path forms the estimate under ErrorEstimate");
        NumericError::require_finite("explicit error estimate", &estimate)?;
        let norm = error_norm(
            absolute_tolerance,
            relative_tolerance,
            &state.values,
            &new_values,
            &estimate,
        );
        NumericError::require_finite("explicit error norm", &[norm])?;
        let embedded_order = scheme.embedded_order().expect("validated");
        match control_step(
            &config.step_policy,
            embedded_order,
            step,
            norm,
            state.error_control.as_ref(),
        ) {
            Ok((report, next)) => {
                error_control = Some(report);
                memory = Some(next);
            }
            Err(ceiling) => {
                return Ok(ExplicitStepOutcome::Rejected(ExplicitRejectedStep {
                    attempted_step: step,
                    rejection: ExplicitRejection::ErrorEstimateExceeded { error_norm: norm },
                    suggested_step: Some(ceiling),
                }));
            }
        }
    }

    let balances = invariant_balances(invariants, &state.values, &new_values, |k, before| {
        let mut predicted = vec![before];
        for i in 1..=stages {
            let mut value: Option<f64> = None;
            for (j, (&a, &b)) in tableau.alpha(i).iter().zip(tableau.beta(i)).enumerate() {
                if a != 0.0 {
                    let term = a * predicted[j];
                    value = Some(value.map_or(term, |sum| sum + term));
                }
                if b != 0.0 {
                    let term = (b * step) * monitor.rates()[j][k];
                    value = Some(value.map_or(term, |sum| sum + term));
                }
            }
            predicted.push(value.unwrap_or(0.0));
        }
        predicted[stages]
    });
    let accepted_steps = state.next_accepted_steps()?;
    Ok(ExplicitStepOutcome::Accepted(ExplicitAcceptedStep {
        state: ExplicitState {
            time: new_time,
            values: new_values,
            accepted_steps,
            error_control: memory,
        },
        report: ExplicitStepReport {
            step,
            step_limit: limit,
            scheme,
            stages,
            ssp_coefficient: scheme.ssp_coefficient(),
            cfl_realized: monitor.cfl_realized(),
            balance_corrected: monitor.balance_corrected(),
            invariants: balances,
            error_control,
            storage_vectors: scheme.low_storage().then_some(registers.count()),
        },
    }))
}

/// The balance of every invariant over one step; `predicted(k, before)`
/// integrates invariant `k`'s declared rates with the scheme's coefficients.
pub(crate) fn invariant_balances(
    invariants: &[Vec<f64>],
    before_values: &[f64],
    after_values: &[f64],
    predicted: impl Fn(usize, f64) -> f64,
) -> Vec<InvariantBalance> {
    invariants
        .iter()
        .enumerate()
        .map(|(k, weights)| {
            let (before, scale_before) = weighted_sum(weights, before_values);
            let (after, scale_after) = weighted_sum(weights, after_values);
            let predicted = predicted(k, before);
            InvariantBalance {
                before,
                after,
                predicted,
                balance_residual: after - predicted,
                scale: scale_before.max(scale_after),
            }
        })
        .collect()
}

type StageResult<T> = Result<Result<T, ExplicitRejectedStep>, SolveError>;

/// Evaluate the right-hand side and the declared invariant rates at one
/// stage input (after its CFL check), recording the rates in the monitor.
fn evaluate_stage(
    operator: &(impl ExplicitOperator + ?Sized),
    context: &EvaluationContext,
    stage: usize,
    stage_time: f64,
    values: &[f64],
    output: &mut [f64],
    monitor: &mut StageMonitor<'_>,
) -> Result<(), SolveError> {
    operator.rhs(context, stage_time, values, output)?;
    NumericError::require_finite(&format!("explicit stage {stage} right-hand side"), output)?;
    let invariant_count = operator.linear_invariants().len();
    if invariant_count > 0 {
        let mut rate = vec![0.0; invariant_count];
        let declared = operator.invariant_rates(context, stage_time, values, &mut rate)?;
        monitor.record_rates(stage, declared, rate)?;
    }
    Ok(())
}

/// The reference path: every stage state and right-hand side kept
/// (`2s + 1` vectors). Returns the step result and, when requested, the
/// embedded error estimate `Σ_j ((b_j − b̂_j)·Δt)·F_j`.
#[allow(clippy::too_many_arguments)]
fn reference_stages(
    operator: &(impl ExplicitOperator + ?Sized),
    context: &EvaluationContext,
    state: &ExplicitState,
    step: f64,
    tableau: &ShuOsherTableau,
    stage_time: &dyn Fn(usize) -> f64,
    monitor: &mut StageMonitor<'_>,
    estimate: bool,
) -> StageResult<(Vec<f64>, Option<Vec<f64>>)> {
    let dimension = state.values.len();
    let stages = tableau.stages();
    let mut stage_values: Vec<Vec<f64>> = Vec::with_capacity(stages + 1);
    let mut stage_rhs: Vec<Vec<f64>> = Vec::with_capacity(stages);
    stage_values.push(state.values.clone());
    for i in 0..=stages {
        let stage_time = stage_time(i);
        if i > 0 {
            let values = combine_stage(
                tableau.alpha(i),
                tableau.beta(i),
                step,
                &stage_values,
                &stage_rhs,
                dimension,
            );
            NumericError::require_finite(&format!("explicit stage {i} state"), &values)?;
            if let Some(rejected) = monitor.check_state(
                i,
                stage_time,
                &values,
                operator.admissibility(context, stage_time, &values),
            )? {
                return Ok(Err(rejected));
            }
            stage_values.push(values);
        }
        if i == stages {
            break;
        }
        let values = &stage_values[i];
        if i > 0 {
            if let Some(rejected) =
                monitor.check_hint(i, operator.stable_step(context, stage_time, values)?)?
            {
                return Ok(Err(rejected));
            }
        }
        let mut output = vec![0.0; dimension];
        evaluate_stage(
            operator,
            context,
            i,
            stage_time,
            values,
            &mut output,
            monitor,
        )?;
        stage_rhs.push(output);
    }
    let estimate = if estimate {
        let embedded = tableau
            .embedded_weights()
            .expect("validated: the scheme declares embedded weights");
        let weights = tableau.butcher_form().b;
        let terms: Vec<(usize, f64)> = weights
            .iter()
            .zip(embedded)
            .enumerate()
            .filter(|(_, (b, b_hat))| *b != *b_hat)
            .map(|(j, (b, b_hat))| (j, (b - b_hat) * step))
            .collect();
        Some(
            (0..dimension)
                .map(|m| {
                    let mut sum: Option<f64> = None;
                    for (j, coefficient) in &terms {
                        let term = coefficient * stage_rhs[*j][m];
                        sum = Some(sum.map_or(term, |sum| sum + term));
                    }
                    sum.unwrap_or(0.0)
                })
                .collect(),
        )
    } else {
        None
    };
    let new_values = stage_values.pop().expect("final stage state");
    Ok(Ok((new_values, estimate)))
}

/// Counts the vectors of the state dimension a stepping path allocates.
#[derive(Default)]
struct Registers {
    count: usize,
}

impl Registers {
    fn allocate(&mut self, dimension: usize) -> Vec<f64> {
        self.count += 1;
        vec![0.0; dimension]
    }

    fn clone_of(&mut self, values: &[f64]) -> Vec<f64> {
        self.count += 1;
        values.to_vec()
    }

    const fn count(&self) -> usize {
        self.count
    }
}

/// Ketcheson's two-register SSP-RK(10,4):
///
/// ```text
/// q1 = q2 = y_n
/// q1 += Δt/6 F(q1)                      five times      (Y_1 .. Y_4, Y_5')
/// q2 = 1/25 q2 + 9/25 q1;  q1 = 15 q2 − 5 q1              (Y_5)
/// q1 += Δt/6 F(q1)                      four times      (Y_6 .. Y_9)
/// y_{n+1} = q2 + 3/5 q1 + Δt/10 F(q1)
/// ```
///
/// Each forward-Euler substep is computed as `q1[m] + (Δt/6)·F[m]`, the
/// same operations as the reference path's `1·Y + (β·Δt)·F`. The
/// recombinations are summed left to right as written.
fn ssp_rk104_low_storage(
    operator: &(impl ExplicitOperator + ?Sized),
    context: &EvaluationContext,
    state: &ExplicitState,
    step: f64,
    stage_time: &dyn Fn(usize) -> f64,
    monitor: &mut StageMonitor<'_>,
    registers: &mut Registers,
) -> StageResult<Vec<f64>> {
    let dimension = state.values.len();
    let sixth_step = (1.0 / 6.0) * step;
    let mut q1 = registers.clone_of(&state.values);
    let mut q2 = registers.clone_of(&state.values);
    let mut f = registers.allocate(dimension);
    // Stage inputs Y_0..Y_9 are evaluated at `stage_time(i)`; the state
    // formed after evaluating Y_i is checked as stage `i + 1`.
    for i in 0..10_usize {
        let t = stage_time(i);
        if i > 0 {
            if let Some(rejected) = monitor.check_hint(i, operator.stable_step(context, t, &q1)?)? {
                return Ok(Err(rejected));
            }
        }
        evaluate_stage(operator, context, i, t, &q1, &mut f, monitor)?;
        let formed = i + 1;
        let formed_time = stage_time(formed);
        if formed == 10 {
            for m in 0..dimension {
                q2[m] = q2[m] + 0.6 * q1[m] + (0.1 * step) * f[m];
            }
            NumericError::require_finite("explicit stage 10 state", &q2)?;
            if let Some(rejected) = monitor.check_state(
                formed,
                formed_time,
                &q2,
                operator.admissibility(context, formed_time, &q2),
            )? {
                return Ok(Err(rejected));
            }
            return Ok(Ok(q2));
        }
        for m in 0..dimension {
            q1[m] += sixth_step * f[m];
        }
        if formed == 5 {
            for m in 0..dimension {
                q2[m] = (1.0 / 25.0) * q2[m] + (9.0 / 25.0) * q1[m];
                q1[m] = 15.0 * q2[m] - 5.0 * q1[m];
            }
        }
        NumericError::require_finite(&format!("explicit stage {formed} state"), &q1)?;
        if let Some(rejected) = monitor.check_state(
            formed,
            formed_time,
            &q1,
            operator.admissibility(context, formed_time, &q1),
        )? {
            return Ok(Err(rejected));
        }
    }
    unreachable!("the tenth stage returns")
}

/// `Y_i[m] = Σ_j α_ij Y_j[m] + (β_ij Δt) F_j[m]` in the documented order.
fn combine_stage(
    alpha: &[f64],
    beta: &[f64],
    step: f64,
    stage_values: &[Vec<f64>],
    stage_rhs: &[Vec<f64>],
    dimension: usize,
) -> Vec<f64> {
    enum Source {
        Value,
        Rate,
    }
    let mut terms = Vec::new();
    for (j, (&a, &b)) in alpha.iter().zip(beta).enumerate() {
        if a != 0.0 {
            terms.push((Source::Value, j, a));
        }
        if b != 0.0 {
            terms.push((Source::Rate, j, b * step));
        }
    }
    (0..dimension)
        .map(|m| {
            let mut sum: Option<f64> = None;
            for (source, j, coefficient) in &terms {
                let operand = match source {
                    Source::Value => stage_values[*j][m],
                    Source::Rate => stage_rhs[*j][m],
                };
                let term = coefficient * operand;
                sum = Some(sum.map_or(term, |sum| sum + term));
            }
            sum.unwrap_or(0.0)
        })
        .collect()
}

/// Every accepted step and every rejected attempt of [`integrate_explicit`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplicitIntegrationReport {
    pub state: ExplicitState,
    pub steps: Vec<ExplicitStepReport>,
    pub rejections: Vec<ExplicitRejectedStep>,
}

/// Step from `state` to exactly `final_time`, retrying a rejected attempt at
/// its suggested step. A rejection without a suggestion (`StepPolicy::Fixed`)
/// is refused as [`SolveError::ExplicitStepRejected`]; a suggestion below the
/// CFL policy's `min_step` is refused as [`SolveError::StepSizeCollapse`];
/// more than `max_attempts` attempts is
/// [`SolveError::ExplicitStepBudgetExhausted`].
pub fn integrate_explicit(
    operator: &(impl ExplicitOperator + ?Sized),
    context: &EvaluationContext,
    state: &ExplicitState,
    final_time: f64,
    config: &ExplicitConfig,
    max_attempts: usize,
) -> Result<ExplicitIntegrationReport, SolveError> {
    let mut current = state.clone();
    let mut steps = Vec::new();
    let mut rejections = Vec::new();
    let mut ceiling = None;
    let mut attempts = 0_usize;
    while current.time < final_time {
        if attempts == max_attempts {
            return Err(SolveError::ExplicitStepBudgetExhausted {
                time: current.time,
                attempts,
            });
        }
        attempts += 1;
        match explicit_step(operator, context, &current, final_time, config, ceiling)? {
            ExplicitStepOutcome::Accepted(accepted) => {
                current = accepted.state;
                steps.push(accepted.report);
                ceiling = None;
            }
            ExplicitStepOutcome::Rejected(rejected) => {
                let Some(suggested) = rejected.suggested_step else {
                    return Err(SolveError::ExplicitStepRejected {
                        time: current.time,
                        rejection: rejected.rejection,
                    });
                };
                ceiling = Some(suggested);
                rejections.push(rejected);
            }
        }
    }
    Ok(ExplicitIntegrationReport {
        state: current,
        steps,
        rejections,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [ExplicitScheme; 7] = [
        ExplicitScheme::ForwardEuler,
        ExplicitScheme::SspRk2,
        ExplicitScheme::SspRk3,
        ExplicitScheme::SspRk104,
        ExplicitScheme::SspRk104LowStorage,
        ExplicitScheme::Rk4,
        ExplicitScheme::BogackiShampine32,
    ];

    fn dot(left: &[f64], right: &[f64]) -> f64 {
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter().map(|row| dot(row, x)).collect()
    }

    /// Order conditions through order 4 on the Butcher form, as
    /// `(order, computed, exact)`.
    fn order_conditions(form: &ButcherForm) -> Vec<(u32, f64, f64)> {
        let c = &form.c;
        let b = &form.b;
        let c2: Vec<f64> = c.iter().map(|x| x * x).collect();
        let c3: Vec<f64> = c.iter().map(|x| x * x * x).collect();
        let ac = mat_vec(&form.a, c);
        let ac2 = mat_vec(&form.a, &c2);
        let aac = mat_vec(&form.a, &ac);
        let c_ac: Vec<f64> = c.iter().zip(&ac).map(|(x, y)| x * y).collect();
        vec![
            (1, b.iter().sum(), 1.0),
            (2, dot(b, c), 0.5),
            (3, dot(b, &c2), 1.0 / 3.0),
            (3, dot(b, &ac), 1.0 / 6.0),
            (4, dot(b, &c3), 0.25),
            (4, dot(b, &c_ac), 0.125),
            (4, dot(b, &ac2), 1.0 / 12.0),
            (4, dot(b, &aac), 1.0 / 24.0),
        ]
    }

    #[test]
    fn tableaux_satisfy_order_conditions_exactly_to_their_declared_order() {
        for scheme in ALL {
            let tableau = scheme.tableau();
            assert_eq!(tableau.stages(), scheme.stages(), "{scheme:?}");
            for i in 1..=tableau.stages() {
                let sum: f64 = tableau.alpha(i).iter().sum();
                assert!((sum - 1.0).abs() <= 1e-15, "{scheme:?} row {i} alpha sum");
            }
            let form = tableau.butcher_form();
            let conditions = order_conditions(&form);
            for (order, computed, exact) in &conditions {
                if *order <= scheme.order() {
                    assert!(
                        (computed - exact).abs() <= 1e-14,
                        "{scheme:?} order-{order} condition {computed} vs {exact}"
                    );
                }
            }
            if scheme.order() < 4 {
                assert!(
                    conditions
                        .iter()
                        .filter(|(order, ..)| *order == scheme.order() + 1)
                        .any(|(_, computed, exact)| (computed - exact).abs() > 1e-3),
                    "{scheme:?} must not satisfy the next order"
                );
            }
        }
    }

    #[test]
    fn declared_ssp_coefficients_match_the_coefficients() {
        for scheme in ALL {
            let computed = scheme.tableau().ssp_coefficient_from_coefficients();
            match scheme.ssp_coefficient() {
                Some(declared) => assert!(
                    (computed - declared).abs() <= 1e-14 * declared,
                    "{scheme:?}: {computed} vs {declared}"
                ),
                None => assert_eq!(computed, 0.0, "{scheme:?}"),
            }
        }
        let c = ExplicitScheme::SspRk104.tableau().butcher_form().c;
        let expected = [
            0.0,
            1.0 / 6.0,
            2.0 / 6.0,
            3.0 / 6.0,
            4.0 / 6.0,
            1.0 / 3.0,
            0.5,
            2.0 / 3.0,
            5.0 / 6.0,
            1.0,
        ];
        for (computed, expected) in c.iter().zip(expected) {
            assert!((computed - expected).abs() <= 1e-15);
        }
    }

    #[test]
    fn config_state_and_reports_round_trip_and_invalid_state_is_refused() {
        let config = ExplicitConfig {
            scheme: ExplicitScheme::SspRk104,
            step_policy: StepPolicy::Cfl {
                target: 5.5,
                max: 6.0,
                min_step: 1e-9,
                max_step: f64::MAX,
            },
        };
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<ExplicitConfig>(&json).unwrap(),
            config
        );
        let state = ExplicitState {
            time: 0.25,
            values: vec![1.0, -0.0, 1e-300],
            accepted_steps: 7,
            error_control: None,
        };
        let json = serde_json::to_string(&state).unwrap();
        let back: ExplicitState = serde_json::from_str(&json).unwrap();
        assert_eq!(back, state);
        assert!(back.values[1].is_sign_negative());
        assert!(
            serde_json::from_str::<ExplicitState>(
                r#"{"time":0.0,"values":[1.0,null],"accepted_steps":0}"#
            )
            .is_err()
        );
    }

    #[test]
    fn malformed_configurations_are_refused() {
        let bad = [
            StepPolicy::Fixed { step: 0.0 },
            StepPolicy::Fixed { step: f64::NAN },
            StepPolicy::Cfl {
                target: 1.5,
                max: 1.0,
                min_step: 1e-9,
                max_step: 1.0,
            },
            StepPolicy::Cfl {
                target: 0.5,
                max: 1.0,
                min_step: 0.0,
                max_step: 1.0,
            },
            StepPolicy::Cfl {
                target: 0.5,
                max: 1.0,
                min_step: 2.0,
                max_step: 1.0,
            },
        ];
        for step_policy in bad {
            let config = ExplicitConfig {
                scheme: ExplicitScheme::SspRk3,
                step_policy,
            };
            assert!(matches!(
                config.validate(),
                Err(SolveError::InvalidConfiguration { .. })
            ));
        }
    }
}
