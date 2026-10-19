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
}

impl ExplicitScheme {
    /// Classical order of accuracy.
    #[must_use]
    pub const fn order(self) -> u32 {
        match self {
            Self::ForwardEuler => 1,
            Self::SspRk2 => 2,
            Self::SspRk3 => 3,
            Self::SspRk104 | Self::Rk4 => 4,
        }
    }

    /// Number of right-hand-side evaluations per step.
    #[must_use]
    pub const fn stages(self) -> usize {
        match self {
            Self::ForwardEuler => 1,
            Self::SspRk2 => 2,
            Self::SspRk3 => 3,
            Self::Rk4 => 4,
            Self::SspRk104 => 10,
        }
    }

    /// Declared SSP coefficient, `None` for a scheme that is not SSP.
    #[must_use]
    pub const fn ssp_coefficient(self) -> Option<f64> {
        match self {
            Self::ForwardEuler | Self::SspRk2 | Self::SspRk3 => Some(1.0),
            Self::SspRk104 => Some(6.0),
            Self::Rk4 => None,
        }
    }

    /// The scheme's Shu–Osher coefficients.
    #[must_use]
    pub fn tableau(self) -> ShuOsherTableau {
        let third = 1.0 / 3.0;
        let sixth = 1.0 / 6.0;
        let (alpha, beta): (Vec<Vec<f64>>, Vec<Vec<f64>>) = match self {
            Self::ForwardEuler => (vec![vec![1.0]], vec![vec![1.0]]),
            Self::SspRk2 => (
                vec![vec![1.0], vec![0.5, 0.5]],
                vec![vec![1.0], vec![0.0, 0.5]],
            ),
            Self::SspRk3 => (
                vec![vec![1.0], vec![0.75, 0.25], vec![third, 0.0, 2.0 * third]],
                vec![vec![1.0], vec![0.0, 0.25], vec![0.0, 0.0, 2.0 * third]],
            ),
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
            Self::SspRk104 => ketcheson_10_4(),
        };
        ShuOsherTableau { alpha, beta }
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
}

/// Scheme and step policy of an explicit integration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplicitConfig {
    pub scheme: ExplicitScheme,
    pub step_policy: StepPolicy,
}

impl ExplicitConfig {
    fn validate(&self) -> Result<(), SolveError> {
        let invalid = |reason: &str| {
            Err(SolveError::InvalidConfiguration {
                reason: reason.into(),
            })
        };
        match self.step_policy {
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
        }
        Ok(())
    }
}

/// Committed explicit-integration state. A rejected attempt never changes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ExplicitStateData")]
pub struct ExplicitState {
    pub time: f64,
    pub values: Vec<f64>,
    pub accepted_steps: u64,
}

#[derive(Deserialize)]
struct ExplicitStateData {
    time: f64,
    values: Vec<f64>,
    accepted_steps: u64,
}

impl TryFrom<ExplicitStateData> for ExplicitState {
    type Error = String;

    fn try_from(data: ExplicitStateData) -> Result<Self, Self::Error> {
        if !data.time.is_finite() {
            return Err("explicit state time must be finite".into());
        }
        NumericError::require_finite("explicit state", &data.values)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            time: data.time,
            values: data.values,
            accepted_steps: data.accepted_steps,
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
        if !time.is_finite() {
            return Err(NumericError::InvalidInput {
                message: "initial explicit time must be finite".into(),
            }
            .into());
        }
        NumericError::require_len("initial explicit state", values.len(), operator.dimension())?;
        NumericError::require_finite("initial explicit state", &values)?;
        if let Admissibility::Inadmissible { index, reason } =
            operator.admissibility(context, time, &values)?
        {
            return Err(SolveError::InadmissibleState { index, reason });
        }
        Ok(Self {
            time,
            values,
            accepted_steps: 0,
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
    /// `StepPolicy::Cfl`'s `max_step`.
    MaximumStep,
    /// The caller's retry ceiling (after a rejection).
    RetryCeiling,
    /// Shortened (or stretched by at most rounding) to land on the final time.
    FinalTime,
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

fn checked_hint(stage: usize, hint: Option<f64>) -> Result<Option<f64>, NumericError> {
    match hint {
        Some(value) if value.is_nan() || value <= 0.0 => Err(NumericError::InvalidInput {
            message: format!("stable step at stage {stage} must be positive, got {value}"),
        }),
        other => Ok(other),
    }
}

fn weighted_sum(weights: &[f64], values: &[f64]) -> (f64, f64) {
    let mut sum = 0.0;
    let mut scale = 0.0;
    for (&weight, &value) in weights.iter().zip(values) {
        let term = weight * value;
        sum += term;
        scale += term.abs();
    }
    (sum, scale)
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
/// `max(max, cfl at stage 0)`, is an [`ExplicitStepOutcome::Rejected`].
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
    NumericError::require_len("explicit state", state.values.len(), dimension)?;
    NumericError::require_finite("explicit state", &state.values)?;
    if !state.time.is_finite() || !final_time.is_finite() || final_time <= state.time {
        return Err(NumericError::InvalidInput {
            message: "explicit final time must be finite and after the state time".into(),
        }
        .into());
    }
    if retry_ceiling.is_some_and(|ceiling| !(ceiling.is_finite() && ceiling > 0.0)) {
        return Err(SolveError::InvalidConfiguration {
            reason: "explicit retry ceiling must be finite and positive".into(),
        });
    }
    let invariants = operator.linear_invariants();
    for (k, weights) in invariants.iter().enumerate() {
        NumericError::require_len(&format!("invariant {k} weights"), weights.len(), dimension)?;
        NumericError::require_finite(&format!("invariant {k} weights"), weights)?;
    }

    let time = state.time;
    let hint0 = checked_hint(0, operator.stable_step(context, time, &state.values)?)?;
    let (mut candidate, mut limit) = match config.step_policy {
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
    };
    if let Some(ceiling) = retry_ceiling.filter(|&ceiling| ceiling < candidate) {
        candidate = ceiling;
        limit = StepLimit::RetryCeiling;
    }
    if let StepPolicy::Cfl { min_step, .. } = config.step_policy {
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

    let scheme = config.scheme;
    let tableau = scheme.tableau();
    let stage_times = tableau.butcher_form().c;
    let stages = tableau.stages();
    let cfl_cap = match config.step_policy {
        StepPolicy::Cfl { max, .. } => Some(max),
        StepPolicy::Fixed { .. } => None,
    };
    let mut cfl_realized = hint0.map(|hint| step / hint);
    let cfl_threshold = cfl_cap.map(|max| max.max(cfl_realized.unwrap_or(0.0)));
    let mut minimum_hint = hint0;

    let invariant_count = invariants.len();
    let mut rates: Vec<Vec<f64>> = Vec::with_capacity(stages);
    let mut balance_corrected = None;
    let mut stage_values: Vec<Vec<f64>> = Vec::with_capacity(stages + 1);
    let mut stage_rhs: Vec<Vec<f64>> = Vec::with_capacity(stages);
    stage_values.push(state.values.clone());

    for i in 0..=stages {
        let stage_time = if i == stages {
            new_time
        } else if i == 0 {
            time
        } else {
            time + stage_times[i] * step
        };
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
            if let Admissibility::Inadmissible { index, reason } =
                operator.admissibility(context, stage_time, &values)?
            {
                let suggested_step = match config.step_policy {
                    StepPolicy::Cfl { .. } => Some(0.5 * step),
                    StepPolicy::Fixed { .. } => None,
                };
                return Ok(ExplicitStepOutcome::Rejected(ExplicitRejectedStep {
                    attempted_step: step,
                    rejection: ExplicitRejection::InadmissibleStage {
                        stage: i,
                        time: stage_time,
                        index,
                        reason,
                    },
                    suggested_step,
                }));
            }
            stage_values.push(values);
        }
        if i == stages {
            break;
        }
        let values = &stage_values[i];
        if i > 0 {
            let hint = checked_hint(i, operator.stable_step(context, stage_time, values)?)?;
            if hint.is_some() != hint0.is_some() {
                return Err(NumericError::InvalidInput {
                    message: format!(
                        "stable step declared at stage 0 but not at stage {i}, or the reverse"
                    ),
                }
                .into());
            }
            if let Some(hint) = hint {
                let cfl = step / hint;
                cfl_realized = cfl_realized.map(|realized| realized.max(cfl));
                minimum_hint = minimum_hint.map(|minimum| minimum.min(hint));
                if let (Some(threshold), StepPolicy::Cfl { target, max, .. }) =
                    (cfl_threshold, &config.step_policy)
                {
                    if cfl > threshold {
                        return Ok(ExplicitStepOutcome::Rejected(ExplicitRejectedStep {
                            attempted_step: step,
                            rejection: ExplicitRejection::CflExceeded {
                                stage: i,
                                cfl,
                                max: *max,
                            },
                            suggested_step: minimum_hint.map(|minimum| target * minimum),
                        }));
                    }
                }
            }
        }
        let mut output = vec![0.0; dimension];
        operator.rhs(context, stage_time, values, &mut output)?;
        NumericError::require_finite(&format!("explicit stage {i} right-hand side"), &output)?;
        stage_rhs.push(output);
        if invariant_count > 0 {
            let mut rate = vec![0.0; invariant_count];
            let declared = operator.invariant_rates(context, stage_time, values, &mut rate)?;
            if *balance_corrected.get_or_insert(declared) != declared {
                return Err(NumericError::InvalidInput {
                    message: format!("invariant rates declared inconsistently at stage {i}"),
                }
                .into());
            }
            if declared {
                NumericError::require_finite(&format!("invariant rates at stage {i}"), &rate)?;
            } else {
                rate.fill(0.0);
            }
            rates.push(rate);
        }
    }

    let new_values = stage_values.pop().expect("final stage state");
    let balances = invariants
        .iter()
        .enumerate()
        .map(|(k, weights)| {
            let (before, scale_before) = weighted_sum(weights, &state.values);
            let (after, scale_after) = weighted_sum(weights, &new_values);
            let mut predicted = vec![before];
            for i in 1..=stages {
                let mut value: Option<f64> = None;
                for (j, (&a, &b)) in tableau.alpha(i).iter().zip(tableau.beta(i)).enumerate() {
                    if a != 0.0 {
                        let term = a * predicted[j];
                        value = Some(value.map_or(term, |sum| sum + term));
                    }
                    if b != 0.0 {
                        let term = (b * step) * rates[j][k];
                        value = Some(value.map_or(term, |sum| sum + term));
                    }
                }
                predicted.push(value.unwrap_or(0.0));
            }
            let predicted = predicted[stages];
            InvariantBalance {
                before,
                after,
                predicted,
                balance_residual: after - predicted,
                scale: scale_before.max(scale_after),
            }
        })
        .collect();
    let accepted_steps =
        state
            .accepted_steps
            .checked_add(1)
            .ok_or_else(|| NumericError::InvalidInput {
                message: "accepted explicit step count overflowed".into(),
            })?;
    Ok(ExplicitStepOutcome::Accepted(ExplicitAcceptedStep {
        state: ExplicitState {
            time: new_time,
            values: new_values,
            accepted_steps,
        },
        report: ExplicitStepReport {
            step,
            step_limit: limit,
            scheme,
            stages,
            ssp_coefficient: scheme.ssp_coefficient(),
            cfl_realized,
            balance_corrected: balance_corrected.unwrap_or(false),
            invariants: balances,
        },
    }))
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

    const ALL: [ExplicitScheme; 5] = [
        ExplicitScheme::ForwardEuler,
        ExplicitScheme::SspRk2,
        ExplicitScheme::SspRk3,
        ExplicitScheme::SspRk104,
        ExplicitScheme::Rk4,
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
