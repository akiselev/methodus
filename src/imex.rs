//! Implicit–explicit (IMEX) additive Runge–Kutta stepping of
//! `y' = F_E(t, y) + F_I(t, y)`: the nonstiff part `F_E` explicit, the stiff
//! part `F_I` implicit.
//!
//! Every pair is stored as additive Butcher data
//!
//! ```text
//! Y_0 = y_n
//! Y_i = y_n + Δt Σ_{j<i} a^E_ij F_E(t_j, Y_j) + Δt Σ_{j≤i} a^I_ij F_I(t_j, Y_j),   i = 1..s-1
//! y_{n+1} = y_n + Δt Σ_j b^E_j F_E(t_j, Y_j) + Δt Σ_j b^I_j F_I(t_j, Y_j)
//! ```
//!
//! with shared nodes `t_j = t_n + c_j Δt`, a declared order, an optional
//! embedded pair `(b̂^E, b̂^I)` and a stiff-accuracy flag (`b^I` equals the
//! last row of `a^I`). When additionally `b^E` equals the last row of
//! `a^E` the step result is the last stage itself, taken as solved; the
//! Ascher–Ruuth–Spiteri pairs have this property and the Kennedy–Carpenter
//! pair does not.
//!
//! Each implicit stage solves `G(Y_i) = Y_i − (a^I_ii Δt) F_I(t_i, Y_i) − R_i = 0`
//! with `R_i` the explicit sum, either by the owner's own linear solve
//! ([`ImexOperator::solve_implicit_stage`], for a linear `F_I`) or by
//! [`crate::solve_newton_krylov`] over the owner's Jacobian–vector product
//! ([`ImexOperator::implicit_jacobian_vector_product`]) from the initial
//! guess `R_i`. After either solve `F_I(t_i, Y_i)` is re-evaluated (never
//! recovered algebraically as `(Y_i − R_i)/(a^I_ii Δt)`) and the stage
//! residual `‖G(Y_i)‖₂` is reported.
//!
//! Summation order is fixed and documented: each component of `R_i` (and
//! of a `b`-weighted result or error estimate) starts from `y_n[m]` and
//! adds, for `j` ascending, `(a^E_ij·Δt)·F_E_j[m]` then
//! `(a^I_ij·Δt)·F_I_j[m]`, skipping exactly-zero coefficients; no fused
//! multiply-add or reassociation. Linear invariants are summed in component
//! order as in [`crate::explicit_step`].

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use crate::explicit::{
    Admissibility, ErrorControlReport, ExplicitRejection, ExplicitState, InvariantBalance,
    StageMonitor, StepChoice, StepLimit, StepPolicy, check_invariant_weights, checked_hint,
    choose_step, control_step, error_norm, invariant_balances,
};
use crate::krylov_method::KrylovMethod;
use crate::linear::l2;
use crate::newton_krylov::{NewtonKrylovConfig, PreconditionerFactory, solve_newton_krylov};
use crate::operator::{LinearOperator, NonlinearOperator, OperatorProperties, Preconditioner};
use crate::{EvaluationContext, NumericError, SolveError};

/// Additively split method-of-lines right-hand side `F_E(t, y) + F_I(t, y)`
/// over flat `f64` state.
pub trait ImexOperator: Send + Sync {
    fn dimension(&self) -> usize;

    /// `output = F_E(time, state)`, the part integrated explicitly.
    fn explicit_rhs(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;

    /// `output = F_I(time, state)`, the part integrated implicitly.
    fn implicit_rhs(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;

    /// The largest step for which forward Euler on the *explicit part* is
    /// stable at `state` (the step at CFL number 1), or `None` when the
    /// owner declares none; `f64::INFINITY` means unconstrained. Must be
    /// positive. The implicit part imposes no step restriction here.
    fn stable_step(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
    ) -> Result<Option<f64>, NumericError> {
        Ok(None)
    }

    /// Stage-wise admissibility of a state; see
    /// [`crate::ExplicitOperator::admissibility`].
    fn admissibility(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
    ) -> Result<Admissibility, NumericError> {
        Ok(Admissibility::Admissible)
    }

    /// Weight vectors `w_k` of linear invariants `w_k·y`, each of length
    /// `dimension()`.
    fn linear_invariants(&self) -> &[Vec<f64>] {
        &[]
    }

    /// Net boundary/source rates `d(w_k·y)/dt` of each invariant at
    /// `(time, state)`, split by part: `explicit_output[k] = w_k·F_E` and
    /// `implicit_output[k] = w_k·F_I` as the owner declares them (one
    /// entry per invariant each). Returns `false` when the owner declares
    /// none, in which case both parts are treated as closed (zero rate).
    fn invariant_rates(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
        _explicit_output: &mut [f64],
        _implicit_output: &mut [f64],
    ) -> Result<bool, NumericError> {
        Ok(false)
    }

    /// Declared properties of the stage Jacobian `I − σ ∂F_I/∂y` for every
    /// `σ > 0` a step may form, valid at every admitted state (used by the
    /// Newton–Krylov stage solver to admit or refuse its Krylov method,
    /// exactly as [`crate::NonlinearOperator::jacobian_properties`]). The
    /// default makes no claim.
    fn implicit_stage_properties(&self) -> OperatorProperties {
        OperatorProperties::default()
    }

    /// `output = (∂F_I/∂y)(time, state) · direction`. Returns `false` when
    /// the owner declares no Jacobian–vector product, which refuses the
    /// Newton–Krylov stage solver as an invalid configuration.
    fn implicit_jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
        _direction: &[f64],
        _output: &mut [f64],
    ) -> Result<bool, NumericError> {
        Ok(false)
    }

    /// A preconditioner approximating the inverse of `I − shift·∂F_I/∂y` at
    /// `state`, for the Newton–Krylov stage solver; `None` runs it
    /// unpreconditioned.
    fn implicit_stage_preconditioner<'a>(
        &'a self,
        _context: &EvaluationContext,
        _time: f64,
        _shift: f64,
        _state: &[f64],
    ) -> Result<Option<Box<dyn Preconditioner + 'a>>, NumericError> {
        Ok(None)
    }

    /// The owner's own stage solve: write to `output` the `Y` with
    /// `Y − shift·F_I(time, Y) = right_hand_side` (`shift = a^I_ii Δt > 0`),
    /// for example a direct solve of a linear implicit part. Returns
    /// `false` when the owner provides none, which refuses
    /// [`ImplicitStageSolver::Owner`] as an invalid configuration. The
    /// stepper re-evaluates `F_I(time, Y)` and checks the residual against
    /// the configured tolerance.
    fn solve_implicit_stage(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _shift: f64,
        _right_hand_side: &[f64],
        _output: &mut [f64],
    ) -> Result<bool, NumericError> {
        Ok(false)
    }
}

/// An additive (IMEX) Runge–Kutta pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImexScheme {
    /// ARS(2,2,2) (Ascher–Ruuth–Spiteri 1997): order 2, three stages, two
    /// implicit solves, stiffly accurate, L-stable implicit part. No
    /// embedded estimate.
    Ars222,
    /// ARS(4,4,3) (Ascher–Ruuth–Spiteri 1997): order 3, five stages, four
    /// implicit solves, stiffly accurate, L-stable implicit part. No
    /// embedded estimate.
    Ars443,
    /// ARK3(2)4L\[2\]SA (Kennedy–Carpenter 2003): order 3, four stages,
    /// three implicit solves, stiffly accurate L-stable ESDIRK implicit
    /// part, embedded order 2.
    Ark324L2Sa,
}

impl ImexScheme {
    /// Classical order of accuracy.
    #[must_use]
    pub const fn order(self) -> u32 {
        match self {
            Self::Ars222 => 2,
            Self::Ars443 | Self::Ark324L2Sa => 3,
        }
    }

    /// Number of Butcher stages including `Y_0 = y_n`.
    #[must_use]
    pub const fn stages(self) -> usize {
        match self {
            Self::Ars222 => 3,
            Self::Ars443 => 5,
            Self::Ark324L2Sa => 4,
        }
    }

    /// Number of implicit stage solves per step.
    #[must_use]
    pub const fn implicit_solves(self) -> usize {
        match self {
            Self::Ars222 => 2,
            Self::Ars443 => 4,
            Self::Ark324L2Sa => 3,
        }
    }

    /// Order of the embedded estimate, `None` when the pair declares none
    /// (which [`StepPolicy::ErrorEstimate`] refuses).
    #[must_use]
    pub const fn embedded_order(self) -> Option<u32> {
        match self {
            Self::Ars222 | Self::Ars443 => None,
            Self::Ark324L2Sa => Some(2),
        }
    }

    /// The pair's coefficients.
    #[must_use]
    pub fn tableau(self) -> ArkTableau {
        match self {
            Self::Ars222 => {
                let gamma = 1.0 - 1.0 / 2.0_f64.sqrt();
                let delta = 1.0 - 1.0 / (2.0 * gamma);
                ArkTableau {
                    explicit: vec![
                        vec![0.0, 0.0, 0.0],
                        vec![gamma, 0.0, 0.0],
                        vec![delta, 1.0 - delta, 0.0],
                    ],
                    implicit: vec![
                        vec![0.0, 0.0, 0.0],
                        vec![0.0, gamma, 0.0],
                        vec![0.0, 1.0 - gamma, gamma],
                    ],
                    weights_explicit: vec![delta, 1.0 - delta, 0.0],
                    weights_implicit: vec![0.0, 1.0 - gamma, gamma],
                    nodes: vec![0.0, gamma, 1.0],
                    embedded_explicit: None,
                    embedded_implicit: None,
                    order: 2,
                    embedded_order: None,
                }
            }
            Self::Ars443 => ArkTableau {
                explicit: vec![
                    vec![0.0; 5],
                    vec![0.5, 0.0, 0.0, 0.0, 0.0],
                    vec![11.0 / 18.0, 1.0 / 18.0, 0.0, 0.0, 0.0],
                    vec![5.0 / 6.0, -5.0 / 6.0, 0.5, 0.0, 0.0],
                    vec![0.25, 1.75, 0.75, -1.75, 0.0],
                ],
                implicit: vec![
                    vec![0.0; 5],
                    vec![0.0, 0.5, 0.0, 0.0, 0.0],
                    vec![0.0, 1.0 / 6.0, 0.5, 0.0, 0.0],
                    vec![0.0, -0.5, 0.5, 0.5, 0.0],
                    vec![0.0, 1.5, -1.5, 0.5, 0.5],
                ],
                weights_explicit: vec![0.25, 1.75, 0.75, -1.75, 0.0],
                weights_implicit: vec![0.0, 1.5, -1.5, 0.5, 0.5],
                nodes: vec![0.0, 0.5, 2.0 / 3.0, 0.5, 1.0],
                embedded_explicit: None,
                embedded_implicit: None,
                order: 3,
                embedded_order: None,
            },
            Self::Ark324L2Sa => {
                let gamma = 1_767_732_205_903.0 / 4_055_673_282_236.0;
                let b = vec![
                    1_471_266_399_579.0 / 7_840_856_788_654.0,
                    -4_482_444_167_858.0 / 7_529_755_066_697.0,
                    11_266_239_266_428.0 / 11_593_286_722_821.0,
                    gamma,
                ];
                let b_hat = vec![
                    2_756_255_671_327.0 / 12_835_298_489_170.0,
                    -10_771_552_573_575.0 / 22_201_958_757_719.0,
                    9_247_589_265_047.0 / 10_645_013_368_117.0,
                    2_193_209_047_091.0 / 5_459_859_503_100.0,
                ];
                ArkTableau {
                    explicit: vec![
                        vec![0.0; 4],
                        vec![1_767_732_205_903.0 / 2_027_836_641_118.0, 0.0, 0.0, 0.0],
                        vec![
                            5_535_828_885_825.0 / 10_492_691_773_637.0,
                            788_022_342_437.0 / 10_882_634_858_940.0,
                            0.0,
                            0.0,
                        ],
                        vec![
                            6_485_989_280_629.0 / 16_251_701_735_622.0,
                            -4_246_266_847_089.0 / 9_704_473_918_619.0,
                            10_755_448_449_292.0 / 10_357_097_424_841.0,
                            0.0,
                        ],
                    ],
                    implicit: vec![
                        vec![0.0; 4],
                        vec![gamma, gamma, 0.0, 0.0],
                        vec![
                            2_746_238_789_719.0 / 10_658_868_560_708.0,
                            -640_167_445_237.0 / 6_845_629_431_997.0,
                            gamma,
                            0.0,
                        ],
                        b.clone(),
                    ],
                    weights_explicit: b.clone(),
                    weights_implicit: b,
                    nodes: vec![0.0, 1_767_732_205_903.0 / 2_027_836_641_118.0, 0.6, 1.0],
                    embedded_explicit: Some(b_hat.clone()),
                    embedded_implicit: Some(b_hat),
                    order: 3,
                    embedded_order: Some(2),
                }
            }
        }
    }
}

/// Coefficients of an additive Runge–Kutta pair: `s × s` explicit
/// (strictly lower-triangular) and implicit (lower-triangular, first
/// diagonal entry zero) matrices, weights, shared nodes, and an optional
/// embedded pair.
#[derive(Clone, Debug, PartialEq)]
pub struct ArkTableau {
    explicit: Vec<Vec<f64>>,
    implicit: Vec<Vec<f64>>,
    weights_explicit: Vec<f64>,
    weights_implicit: Vec<f64>,
    nodes: Vec<f64>,
    embedded_explicit: Option<Vec<f64>>,
    embedded_implicit: Option<Vec<f64>>,
    order: u32,
    embedded_order: Option<u32>,
}

impl ArkTableau {
    #[must_use]
    pub fn stages(&self) -> usize {
        self.nodes.len()
    }

    /// Row `i` of `a^E` (`j = 0..s`).
    #[must_use]
    pub fn explicit(&self, stage: usize) -> &[f64] {
        &self.explicit[stage]
    }

    /// Row `i` of `a^I` (`j = 0..s`).
    #[must_use]
    pub fn implicit(&self, stage: usize) -> &[f64] {
        &self.implicit[stage]
    }

    #[must_use]
    pub fn weights_explicit(&self) -> &[f64] {
        &self.weights_explicit
    }

    #[must_use]
    pub fn weights_implicit(&self) -> &[f64] {
        &self.weights_implicit
    }

    #[must_use]
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    #[must_use]
    pub fn embedded_explicit(&self) -> Option<&[f64]> {
        self.embedded_explicit.as_deref()
    }

    #[must_use]
    pub fn embedded_implicit(&self) -> Option<&[f64]> {
        self.embedded_implicit.as_deref()
    }

    #[must_use]
    pub const fn order(&self) -> u32 {
        self.order
    }

    #[must_use]
    pub const fn embedded_order(&self) -> Option<u32> {
        self.embedded_order
    }

    /// Whether the implicit part is stiffly accurate: `b^I` equals the
    /// last row of `a^I`.
    #[must_use]
    pub fn stiffly_accurate(&self) -> bool {
        self.implicit.last() == Some(&self.weights_implicit)
    }

    /// Whether the step result is the last stage itself (`b^E` and `b^I`
    /// both equal the last rows), in which case the stepper takes the
    /// solved stage as `y_{n+1}` instead of re-summing.
    #[must_use]
    pub fn final_stage_is_solution(&self) -> bool {
        self.stiffly_accurate() && self.explicit.last() == Some(&self.weights_explicit)
    }

    fn validate(&self) -> Result<(), SolveError> {
        let s = self.stages();
        let invalid = |reason: String| SolveError::InvalidConfiguration { reason };
        let square =
            |matrix: &Vec<Vec<f64>>| matrix.len() == s && matrix.iter().all(|row| row.len() == s);
        if !(square(&self.explicit)
            && square(&self.implicit)
            && self.weights_explicit.len() == s
            && self.weights_implicit.len() == s
            && self.embedded_explicit.as_ref().is_none_or(|b| b.len() == s)
            && self.embedded_implicit.as_ref().is_none_or(|b| b.len() == s)
            && self.embedded_explicit.is_some() == self.embedded_implicit.is_some()
            && self.embedded_explicit.is_some() == self.embedded_order.is_some())
        {
            return Err(invalid("ARK tableau dimensions are inconsistent".into()));
        }
        for i in 0..s {
            if self.explicit[i][i..].iter().any(|&a| a != 0.0)
                || self.implicit[i][i + 1..].iter().any(|&a| a != 0.0)
            {
                return Err(invalid(format!(
                    "ARK tableau row {i} is not explicit/diagonally implicit"
                )));
            }
            if !(self.implicit[i][i].is_finite() && self.implicit[i][i] >= 0.0) {
                return Err(invalid(format!(
                    "ARK tableau implicit diagonal at stage {i} must be nonnegative"
                )));
            }
        }
        if self.implicit[0][0] != 0.0 {
            return Err(invalid("ARK tableau stage 0 must be the step start".into()));
        }
        Ok(())
    }
}

/// How each implicit stage equation is solved.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImplicitStageSolver {
    /// [`crate::solve_newton_krylov`] over the owner's implicit
    /// Jacobian–vector product and optional preconditioner, from the
    /// initial guess `R_i`. A solve that does not converge rejects the step.
    NewtonKrylov {
        method: KrylovMethod,
        newton: NewtonKrylovConfig,
    },
    /// The owner's own [`ImexOperator::solve_implicit_stage`]; the stage
    /// residual must satisfy `‖G‖₂ ≤ absolute_tolerance + relative_tolerance·‖R_i‖₂`
    /// or the step is rejected.
    Owner {
        absolute_tolerance: f64,
        relative_tolerance: f64,
    },
}

/// Scheme, step policy and stage solver of an IMEX integration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImexConfig {
    pub scheme: ImexScheme,
    pub step_policy: StepPolicy,
    pub stage_solver: ImplicitStageSolver,
}

impl ImexConfig {
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
        if let ImplicitStageSolver::Owner {
            absolute_tolerance,
            relative_tolerance,
        } = self.stage_solver
        {
            let nonnegative = |value: f64| value.is_finite() && value >= 0.0;
            if !(nonnegative(absolute_tolerance) && nonnegative(relative_tolerance)) {
                return Err(SolveError::InvalidConfiguration {
                    reason: "owner stage-solve tolerances must be finite and nonnegative".into(),
                });
            }
        }
        self.scheme.tableau().validate()
    }
}

impl ExplicitState {
    /// Validate an initial IMEX state exactly as [`ExplicitState::initialize`]
    /// does for an explicit operator.
    pub fn initialize_imex(
        operator: &(impl ImexOperator + ?Sized),
        context: &EvaluationContext,
        time: f64,
        values: Vec<f64>,
    ) -> Result<Self, SolveError> {
        Self::initialize_with(operator.dimension(), time, values, |values| {
            operator.admissibility(context, time, values)
        })
    }
}

/// Evidence about one implicit stage solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImplicitStageSolve {
    Owner {
        /// `‖Y_i − shift·F_I(t_i, Y_i) − R_i‖₂` after the owner's solve.
        residual_norm: f64,
        tolerance: f64,
    },
    NewtonKrylov {
        iterations: usize,
        linear_iterations: usize,
        /// `‖G(Y_i)‖₂` re-evaluated at the returned stage state.
        residual_norm: f64,
        /// The Newton–Krylov threshold `atol + rtol·‖G(R_i)‖₂` it met.
        tolerance: f64,
    },
}

/// One implicit stage of an accepted IMEX step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImplicitStageReport {
    pub stage: usize,
    pub time: f64,
    /// `a^I_ii · Δt`.
    pub shift: f64,
    pub solve: ImplicitStageSolve,
}

/// Telemetry of an accepted IMEX step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImexStepReport {
    pub step: f64,
    pub step_limit: StepLimit,
    pub scheme: ImexScheme,
    pub stages: usize,
    pub implicit_stages: Vec<ImplicitStageReport>,
    /// `max_i Δt / stable_step(Y_i)` of the explicit part over the stage
    /// states, when the operator declares a stable step.
    pub cfl_realized: Option<f64>,
    /// Whether the invariant balances are corrected by declared rates.
    pub balance_corrected: bool,
    pub invariants: Vec<InvariantBalance>,
    /// The embedded error estimate, under `StepPolicy::ErrorEstimate` only.
    pub error_control: Option<ErrorControlReport>,
}

/// Why an IMEX step attempt was rejected.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImexRejection {
    /// Stage `stage` (`1..=s`; `s` names a re-summed step result) was
    /// inadmissible.
    InadmissibleStage {
        stage: usize,
        time: f64,
        index: Option<usize>,
        reason: String,
    },
    /// Stage `stage`'s explicit-part CFL number exceeded the policy maximum.
    CflExceeded { stage: usize, cfl: f64, max: f64 },
    /// The embedded error estimate's weighted norm exceeded one.
    ErrorEstimateExceeded { error_norm: f64 },
    /// The implicit stage solve did not meet its tolerance.
    ImplicitStageNotConverged {
        stage: usize,
        time: f64,
        residual_norm: f64,
        tolerance: f64,
    },
}

impl From<ExplicitRejection> for ImexRejection {
    fn from(rejection: ExplicitRejection) -> Self {
        match rejection {
            ExplicitRejection::InadmissibleStage {
                stage,
                time,
                index,
                reason,
            } => Self::InadmissibleStage {
                stage,
                time,
                index,
                reason,
            },
            ExplicitRejection::CflExceeded { stage, cfl, max } => {
                Self::CflExceeded { stage, cfl, max }
            }
            ExplicitRejection::ErrorEstimateExceeded { error_norm } => {
                Self::ErrorEstimateExceeded { error_norm }
            }
        }
    }
}

impl fmt::Display for ImexRejection {
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
            Self::ImplicitStageNotConverged {
                stage,
                time,
                residual_norm,
                tolerance,
            } => write!(
                formatter,
                "implicit stage {stage} at time {time} did not converge: residual {residual_norm} \
                 exceeds {tolerance}"
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImexAcceptedStep {
    pub state: ExplicitState,
    pub report: ImexStepReport,
}

/// A rejected attempt. The committed state is the caller's input, unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImexRejectedStep {
    pub attempted_step: f64,
    pub rejection: ImexRejection,
    /// The retry ceiling for the next attempt; `None` under
    /// `StepPolicy::Fixed`.
    pub suggested_step: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImexStepOutcome {
    Accepted(ImexAcceptedStep),
    Rejected(ImexRejectedStep),
}

/// `G(Y) = Y − shift·F_I(t, Y) − R` as a [`NonlinearOperator`] for the
/// Newton–Krylov stage solver.
struct StageSystem<'a, O: ImexOperator + ?Sized> {
    operator: &'a O,
    time: f64,
    shift: f64,
    right_hand_side: &'a [f64],
    jvp_undeclared: AtomicBool,
}

impl<O: ImexOperator + ?Sized> NonlinearOperator for StageSystem<'_, O> {
    fn dimension(&self) -> usize {
        self.operator.dimension()
    }

    fn jacobian_properties(&self) -> OperatorProperties {
        self.operator.implicit_stage_properties()
    }

    fn residual(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.operator
            .implicit_rhs(context, self.time, state, output)?;
        stage_residual_in_place(state, self.shift, self.right_hand_side, output);
        Ok(())
    }

    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let declared = self
            .operator
            .implicit_jacobian_vector_product(context, self.time, state, direction, output)?;
        if !declared {
            self.jvp_undeclared.store(true, Ordering::Relaxed);
            return Err(NumericError::Operator {
                message: "ImexOperator declares no implicit Jacobian-vector product".into(),
            });
        }
        for (out, &d) in output.iter_mut().zip(direction) {
            *out = d - self.shift * *out;
        }
        Ok(())
    }
}

/// `output[m] = (state[m] − shift·output[m]) − rhs[m]` where `output` holds
/// `F_I(t, state)` on entry.
fn stage_residual_in_place(state: &[f64], shift: f64, right_hand_side: &[f64], output: &mut [f64]) {
    for ((out, &y), &r) in output.iter_mut().zip(state).zip(right_hand_side) {
        *out = (y - shift * *out) - r;
    }
}

struct StagePreconditioners<'a, O: ImexOperator + ?Sized> {
    operator: &'a O,
    time: f64,
    shift: f64,
}

impl<O: ImexOperator + ?Sized> PreconditionerFactory for StagePreconditioners<'_, O> {
    fn build<'b>(
        &'b self,
        context: &EvaluationContext,
        _jacobian: &dyn LinearOperator,
        state: &[f64],
    ) -> Result<Option<Box<dyn Preconditioner + 'b>>, NumericError> {
        self.operator
            .implicit_stage_preconditioner(context, self.time, self.shift, state)
    }
}

/// Solve one implicit stage. On success `stage_values` holds `Y_i` and
/// `implicit_rhs` holds the re-evaluated `F_I(t_i, Y_i)`.
#[allow(clippy::too_many_arguments)]
fn solve_stage(
    operator: &(impl ImexOperator + ?Sized),
    context: &EvaluationContext,
    solver: &ImplicitStageSolver,
    stage: usize,
    time: f64,
    shift: f64,
    right_hand_side: &[f64],
    stage_values: &mut Vec<f64>,
    implicit_rhs: &mut [f64],
) -> Result<Result<ImplicitStageSolve, ImexRejection>, SolveError> {
    let dimension = right_hand_side.len();
    let residual_of = |values: &[f64], implicit_rhs: &mut [f64]| -> Result<f64, SolveError> {
        operator.implicit_rhs(context, time, values, implicit_rhs)?;
        NumericError::require_finite(
            &format!("implicit stage {stage} right-hand side"),
            implicit_rhs,
        )?;
        let mut residual = implicit_rhs.to_vec();
        stage_residual_in_place(values, shift, right_hand_side, &mut residual);
        Ok(l2(&residual)?)
    };
    match solver {
        ImplicitStageSolver::NewtonKrylov { method, newton } => {
            let system = StageSystem {
                operator,
                time,
                shift,
                right_hand_side,
                jvp_undeclared: AtomicBool::new(false),
            };
            let factory = StagePreconditioners {
                operator,
                time,
                shift,
            };
            let report = match solve_newton_krylov(
                &system,
                context,
                right_hand_side,
                method,
                Some(&factory),
                None,
                newton,
            ) {
                Ok(report) => report,
                Err(error) => {
                    if system.jvp_undeclared.load(Ordering::Relaxed) {
                        return Err(SolveError::InvalidConfiguration {
                            reason: "ImplicitStageSolver::NewtonKrylov requires \
                                     ImexOperator::implicit_jacobian_vector_product"
                                .into(),
                        });
                    }
                    return Err(error);
                }
            };
            let initial_norm = report
                .trace
                .first()
                .map_or(0.0, |entry| entry.residual_norm);
            let tolerance = newton.absolute_tolerance + newton.relative_tolerance * initial_norm;
            let iterations = report.trace.len().saturating_sub(1);
            let linear_iterations = report.linear_iterations();
            *stage_values = report.state;
            NumericError::require_finite(&format!("implicit stage {stage} state"), stage_values)?;
            let residual_norm = residual_of(stage_values, implicit_rhs)?;
            if !report.converged {
                return Ok(Err(ImexRejection::ImplicitStageNotConverged {
                    stage,
                    time,
                    residual_norm,
                    tolerance,
                }));
            }
            Ok(Ok(ImplicitStageSolve::NewtonKrylov {
                iterations,
                linear_iterations,
                residual_norm,
                tolerance,
            }))
        }
        ImplicitStageSolver::Owner {
            absolute_tolerance,
            relative_tolerance,
        } => {
            stage_values.clear();
            stage_values.resize(dimension, 0.0);
            let declared = operator.solve_implicit_stage(
                context,
                time,
                shift,
                right_hand_side,
                stage_values,
            )?;
            if !declared {
                return Err(SolveError::InvalidConfiguration {
                    reason:
                        "ImplicitStageSolver::Owner requires ImexOperator::solve_implicit_stage"
                            .into(),
                });
            }
            NumericError::require_finite(&format!("implicit stage {stage} state"), stage_values)?;
            let residual_norm = residual_of(stage_values, implicit_rhs)?;
            let tolerance = absolute_tolerance + relative_tolerance * l2(right_hand_side)?;
            if residual_norm > tolerance {
                return Ok(Err(ImexRejection::ImplicitStageNotConverged {
                    stage,
                    time,
                    residual_norm,
                    tolerance,
                }));
            }
            Ok(Ok(ImplicitStageSolve::Owner {
                residual_norm,
                tolerance,
            }))
        }
    }
}

/// `out[m] = base[m] + Σ_terms coefficient·vector[m]` in term order.
fn combine(base: &[f64], terms: &[(f64, &[f64])]) -> Vec<f64> {
    base.iter()
        .enumerate()
        .map(|(m, &start)| {
            let mut sum = start;
            for (coefficient, vector) in terms {
                sum += coefficient * vector[m];
            }
            sum
        })
        .collect()
}

/// The `(coefficient·Δt, F_j)` terms of a weighted sum over the evaluated
/// stage right-hand sides: `j` ascending, explicit then implicit, zero
/// coefficients skipped.
fn weighted_terms<'a>(
    explicit_weights: &[f64],
    implicit_weights: &[f64],
    step: f64,
    explicit_rhs: &'a [Option<Vec<f64>>],
    implicit_rhs: &'a [Option<Vec<f64>>],
    count: usize,
) -> Result<Vec<(f64, &'a [f64])>, NumericError> {
    let mut terms = Vec::new();
    for j in 0..count {
        for (weight, rhs, part) in [
            (explicit_weights[j], &explicit_rhs[j], "explicit"),
            (implicit_weights[j], &implicit_rhs[j], "implicit"),
        ] {
            if weight != 0.0 {
                let vector = rhs.as_deref().ok_or_else(|| NumericError::InvalidInput {
                    message: format!("{part} right-hand side of stage {j} was not evaluated"),
                })?;
                terms.push((weight * step, vector));
            }
        }
    }
    Ok(terms)
}

/// Attempt one IMEX step from `state` toward `final_time`.
///
/// Step selection, landing on `final_time`, the retry ceiling, stage
/// finiteness and admissibility, the CFL rejection (on the explicit part's
/// stable step) and the embedded error control follow
/// [`crate::explicit_step`] exactly. In addition every implicit stage solve
/// that does not meet its tolerance is an
/// [`ImexRejection::ImplicitStageNotConverged`] (retry ceiling `Δt/2` under
/// an adaptive policy). Operator failures, non-finite values, a Newton
/// line-search failure or Krylov breakdown, and a configured stage solver
/// the operator does not support are errors.
pub fn imex_step(
    operator: &(impl ImexOperator + ?Sized),
    context: &EvaluationContext,
    state: &ExplicitState,
    final_time: f64,
    config: &ImexConfig,
    retry_ceiling: Option<f64>,
) -> Result<ImexStepOutcome, SolveError> {
    config.validate()?;
    let dimension = operator.dimension();
    state.validate(dimension, final_time)?;
    let invariants = operator.linear_invariants();
    check_invariant_weights(invariants, dimension)?;
    let invariant_count = invariants.len();

    let time = state.time;
    let hint0 = checked_hint(0, operator.stable_step(context, time, &state.values)?)?;
    let StepChoice {
        step,
        new_time,
        limit,
    } = choose_step(&config.step_policy, state, final_time, retry_ceiling, hint0)?;

    let scheme = config.scheme;
    let tableau = scheme.tableau();
    let stages = tableau.stages();
    let reject = |rejection: ImexRejection, suggested_step: Option<f64>| {
        Ok(ImexStepOutcome::Rejected(ImexRejectedStep {
            attempted_step: step,
            rejection,
            suggested_step,
        }))
    };
    let estimate_wanted = matches!(config.step_policy, StepPolicy::ErrorEstimate { .. });
    // Which right-hand sides later stages, the weights or the estimate use.
    let needed = |matrix: &[Vec<f64>], weights: &[f64], embedded: Option<&[f64]>| {
        (0..stages)
            .map(|j| {
                weights[j] != 0.0
                    || (estimate_wanted && embedded.is_some_and(|b| b[j] != 0.0))
                    || (j + 1..stages).any(|i| matrix[i][j] != 0.0)
            })
            .collect::<Vec<bool>>()
    };
    let need_explicit = needed(
        &tableau.explicit,
        tableau.weights_explicit(),
        tableau.embedded_explicit(),
    );
    let need_implicit = needed(
        &tableau.implicit,
        tableau.weights_implicit(),
        tableau.embedded_implicit(),
    );

    let mut monitor = StageMonitor::new(&config.step_policy, step, hint0);
    let mut stage_values: Vec<Vec<f64>> = Vec::with_capacity(stages);
    let mut explicit_rhs: Vec<Option<Vec<f64>>> = Vec::with_capacity(stages);
    let mut implicit_rhs: Vec<Option<Vec<f64>>> = Vec::with_capacity(stages);
    let mut implicit_stages = Vec::with_capacity(scheme.implicit_solves());

    for i in 0..stages {
        let stage_time = if i == 0 {
            time
        } else if i == stages - 1 && tableau.nodes()[i] == 1.0 {
            new_time
        } else {
            time + tableau.nodes()[i] * step
        };
        let mut implicit_here: Option<Vec<f64>> = None;
        let values = if i == 0 {
            state.values.clone()
        } else {
            let terms = weighted_terms(
                tableau.explicit(i),
                tableau.implicit(i),
                step,
                &explicit_rhs,
                &implicit_rhs,
                i,
            )?;
            let right_hand_side = combine(&state.values, &terms);
            NumericError::require_finite(
                &format!("IMEX stage {i} explicit sum"),
                &right_hand_side,
            )?;
            let diagonal = tableau.implicit(i)[i];
            if diagonal != 0.0 {
                let shift = diagonal * step;
                let mut values = Vec::new();
                let mut rhs = vec![0.0; dimension];
                match solve_stage(
                    operator,
                    context,
                    &config.stage_solver,
                    i,
                    stage_time,
                    shift,
                    &right_hand_side,
                    &mut values,
                    &mut rhs,
                )? {
                    Ok(solve) => implicit_stages.push(ImplicitStageReport {
                        stage: i,
                        time: stage_time,
                        shift,
                        solve,
                    }),
                    Err(rejection) => {
                        return reject(rejection, config.step_policy.halve_on_rejection(step));
                    }
                }
                implicit_here = Some(rhs);
                values
            } else {
                right_hand_side
            }
        };
        if i > 0 {
            NumericError::require_finite(&format!("IMEX stage {i} state"), &values)?;
            if let Some(rejected) = monitor.check_state(
                i,
                stage_time,
                &values,
                operator.admissibility(context, stage_time, &values),
            )? {
                return reject(rejected.rejection.into(), rejected.suggested_step);
            }
            if let Some(rejected) =
                monitor.check_hint(i, operator.stable_step(context, stage_time, &values)?)?
            {
                return reject(rejected.rejection.into(), rejected.suggested_step);
            }
        }
        explicit_rhs.push(if need_explicit[i] {
            let mut output = vec![0.0; dimension];
            operator.explicit_rhs(context, stage_time, &values, &mut output)?;
            NumericError::require_finite(&format!("explicit stage {i} right-hand side"), &output)?;
            Some(output)
        } else {
            None
        });
        implicit_rhs.push(if implicit_here.is_some() {
            implicit_here
        } else if need_implicit[i] {
            let mut output = vec![0.0; dimension];
            operator.implicit_rhs(context, stage_time, &values, &mut output)?;
            NumericError::require_finite(&format!("implicit stage {i} right-hand side"), &output)?;
            Some(output)
        } else {
            None
        });
        if invariant_count > 0 {
            let mut rate = vec![0.0; 2 * invariant_count];
            let (explicit_rate, implicit_rate) = rate.split_at_mut(invariant_count);
            let declared = operator.invariant_rates(
                context,
                stage_time,
                &values,
                explicit_rate,
                implicit_rate,
            )?;
            monitor.record_rates(i, declared, rate)?;
        }
        stage_values.push(values);
    }

    let new_values = if tableau.final_stage_is_solution() {
        stage_values[stages - 1].clone()
    } else {
        let terms = weighted_terms(
            tableau.weights_explicit(),
            tableau.weights_implicit(),
            step,
            &explicit_rhs,
            &implicit_rhs,
            stages,
        )?;
        let values = combine(&state.values, &terms);
        NumericError::require_finite("IMEX step result", &values)?;
        if let Some(rejected) = monitor.check_state(
            stages,
            new_time,
            &values,
            operator.admissibility(context, new_time, &values),
        )? {
            return reject(rejected.rejection.into(), rejected.suggested_step);
        }
        values
    };

    let mut error_control = None;
    let mut memory = None;
    if let StepPolicy::ErrorEstimate {
        absolute_tolerance,
        relative_tolerance,
        ..
    } = config.step_policy
    {
        let embedded_explicit = tableau.embedded_explicit().expect("validated");
        let embedded_implicit = tableau.embedded_implicit().expect("validated");
        let difference = |weights: &[f64], embedded: &[f64]| -> Vec<f64> {
            weights
                .iter()
                .zip(embedded)
                .map(|(b, b_hat)| if b == b_hat { 0.0 } else { b - b_hat })
                .collect()
        };
        let explicit_difference = difference(tableau.weights_explicit(), embedded_explicit);
        let implicit_difference = difference(tableau.weights_implicit(), embedded_implicit);
        let terms = weighted_terms(
            &explicit_difference,
            &implicit_difference,
            step,
            &explicit_rhs,
            &implicit_rhs,
            stages,
        )?;
        let estimate = combine(&vec![0.0; dimension], &terms);
        NumericError::require_finite("IMEX error estimate", &estimate)?;
        let norm = error_norm(
            absolute_tolerance,
            relative_tolerance,
            &state.values,
            &new_values,
            &estimate,
        );
        NumericError::require_finite("IMEX error norm", &[norm])?;
        match control_step(
            &config.step_policy,
            scheme.embedded_order().expect("validated"),
            step,
            norm,
            state.error_control.as_ref(),
        ) {
            Ok((report, next)) => {
                error_control = Some(report);
                memory = Some(next);
            }
            Err(ceiling) => {
                return reject(
                    ImexRejection::ErrorEstimateExceeded { error_norm: norm },
                    Some(ceiling),
                );
            }
        }
    }

    let balances = invariant_balances(invariants, &state.values, &new_values, |k, before| {
        let rates = monitor.rates();
        let mut predicted = before;
        for j in 0..stages {
            let explicit_weight = tableau.weights_explicit()[j];
            if explicit_weight != 0.0 {
                predicted += (explicit_weight * step) * rates[j][k];
            }
            let implicit_weight = tableau.weights_implicit()[j];
            if implicit_weight != 0.0 {
                predicted += (implicit_weight * step) * rates[j][invariant_count + k];
            }
        }
        predicted
    });
    let accepted_steps = state.next_accepted_steps()?;
    Ok(ImexStepOutcome::Accepted(ImexAcceptedStep {
        state: ExplicitState {
            time: new_time,
            values: new_values,
            accepted_steps,
            error_control: memory,
        },
        report: ImexStepReport {
            step,
            step_limit: limit,
            scheme,
            stages,
            implicit_stages,
            cfl_realized: monitor.cfl_realized(),
            balance_corrected: monitor.balance_corrected(),
            invariants: balances,
            error_control,
        },
    }))
}

/// Every accepted step and every rejected attempt of [`integrate_imex`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImexIntegrationReport {
    pub state: ExplicitState,
    pub steps: Vec<ImexStepReport>,
    pub rejections: Vec<ImexRejectedStep>,
}

/// Step from `state` to exactly `final_time`, retrying a rejected attempt at
/// its suggested step. A rejection without a suggestion (`StepPolicy::Fixed`)
/// is refused as [`SolveError::ImexStepRejected`]; a suggestion below the
/// policy's `min_step` is refused as [`SolveError::StepSizeCollapse`]; more
/// than `max_attempts` attempts is [`SolveError::ImexStepBudgetExhausted`].
pub fn integrate_imex(
    operator: &(impl ImexOperator + ?Sized),
    context: &EvaluationContext,
    state: &ExplicitState,
    final_time: f64,
    config: &ImexConfig,
    max_attempts: usize,
) -> Result<ImexIntegrationReport, SolveError> {
    let mut current = state.clone();
    let mut steps = Vec::new();
    let mut rejections = Vec::new();
    let mut ceiling = None;
    let mut attempts = 0_usize;
    while current.time < final_time {
        if attempts == max_attempts {
            return Err(SolveError::ImexStepBudgetExhausted {
                time: current.time,
                attempts,
            });
        }
        attempts += 1;
        match imex_step(operator, context, &current, final_time, config, ceiling)? {
            ImexStepOutcome::Accepted(accepted) => {
                current = accepted.state;
                steps.push(accepted.report);
                ceiling = None;
            }
            ImexStepOutcome::Rejected(rejected) => {
                let Some(suggested) = rejected.suggested_step else {
                    return Err(SolveError::ImexStepRejected {
                        time: current.time,
                        rejection: rejected.rejection,
                    });
                };
                ceiling = Some(suggested);
                rejections.push(rejected);
            }
        }
    }
    Ok(ImexIntegrationReport {
        state: current,
        steps,
        rejections,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [ImexScheme; 3] = [
        ImexScheme::Ars222,
        ImexScheme::Ars443,
        ImexScheme::Ark324L2Sa,
    ];

    fn dot(left: &[f64], right: &[f64]) -> f64 {
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter().map(|row| dot(row, x)).collect()
    }

    /// Additive order conditions through order 3 for weights `(b^E, b^I)`
    /// sharing the nodes `c`: `(order, computed, exact)`.
    fn additive_order_conditions(
        tableau: &ArkTableau,
        weights_explicit: &[f64],
        weights_implicit: &[f64],
    ) -> Vec<(u32, f64, f64)> {
        let c = tableau.nodes();
        let c2: Vec<f64> = c.iter().map(|x| x * x).collect();
        let ae_c = mat_vec(&tableau.explicit, c);
        let ai_c = mat_vec(&tableau.implicit, c);
        let mut conditions = Vec::new();
        for b in [weights_explicit, weights_implicit] {
            conditions.push((1, b.iter().sum(), 1.0));
            conditions.push((2, dot(b, c), 0.5));
            conditions.push((3, dot(b, &c2), 1.0 / 3.0));
            conditions.push((3, dot(b, &ae_c), 1.0 / 6.0));
            conditions.push((3, dot(b, &ai_c), 1.0 / 6.0));
        }
        conditions
    }

    #[test]
    fn tableaux_are_consistent_and_satisfy_the_additive_order_conditions() {
        for scheme in ALL {
            let tableau = scheme.tableau();
            tableau.validate().unwrap();
            assert_eq!(tableau.stages(), scheme.stages(), "{scheme:?}");
            assert_eq!(tableau.order(), scheme.order());
            assert_eq!(tableau.embedded_order(), scheme.embedded_order());
            assert!(tableau.stiffly_accurate(), "{scheme:?}");
            let solves = (0..tableau.stages())
                .filter(|&i| tableau.implicit(i)[i] != 0.0)
                .count();
            assert_eq!(solves, scheme.implicit_solves(), "{scheme:?}");
            for i in 0..tableau.stages() {
                let explicit_sum: f64 = tableau.explicit(i).iter().sum();
                let implicit_sum: f64 = tableau.implicit(i).iter().sum();
                let node = tableau.nodes()[i];
                assert!(
                    (explicit_sum - node).abs() <= 1e-15 && (implicit_sum - node).abs() <= 1e-15,
                    "{scheme:?} row {i}: {explicit_sum} {implicit_sum} vs {node}"
                );
            }
            for (order, computed, exact) in additive_order_conditions(
                &tableau,
                tableau.weights_explicit(),
                tableau.weights_implicit(),
            ) {
                assert!(
                    order > scheme.order() || (computed - exact).abs() <= 1e-14,
                    "{scheme:?} order-{order} condition {computed} vs {exact}"
                );
            }
            if let (Some(b_hat_e), Some(b_hat_i), Some(embedded_order)) = (
                tableau.embedded_explicit(),
                tableau.embedded_implicit(),
                scheme.embedded_order(),
            ) {
                let conditions = additive_order_conditions(&tableau, b_hat_e, b_hat_i);
                for (order, computed, exact) in &conditions {
                    assert!(
                        *order > embedded_order || (computed - exact).abs() <= 1e-14,
                        "{scheme:?} embedded order-{order} condition {computed} vs {exact}"
                    );
                }
                assert!(
                    conditions
                        .iter()
                        .filter(|(order, ..)| *order == embedded_order + 1)
                        .any(|(_, computed, exact)| (computed - exact).abs() > 1e-3),
                    "{scheme:?} embedded weights must not satisfy the next order"
                );
            }
        }
        assert!(ImexScheme::Ars222.tableau().final_stage_is_solution());
        assert!(ImexScheme::Ars443.tableau().final_stage_is_solution());
        assert!(!ImexScheme::Ark324L2Sa.tableau().final_stage_is_solution());
    }

    #[test]
    fn config_round_trips_and_malformed_configurations_are_refused() {
        let config = ImexConfig {
            scheme: ImexScheme::Ark324L2Sa,
            step_policy: StepPolicy::ErrorEstimate {
                absolute_tolerance: 1e-8,
                relative_tolerance: 1e-6,
                safety: 0.9,
                min_step: 1e-12,
                max_step: 1.0,
                initial_step: 1e-3,
                min_factor: 0.2,
                max_factor: 5.0,
            },
            stage_solver: ImplicitStageSolver::Owner {
                absolute_tolerance: 1e-14,
                relative_tolerance: 1e-12,
            },
        };
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(serde_json::from_str::<ImexConfig>(&json).unwrap(), config);
        config.validate().unwrap();
        let no_embedded = ImexConfig {
            scheme: ImexScheme::Ars443,
            ..config.clone()
        };
        assert!(matches!(
            no_embedded.validate(),
            Err(SolveError::InvalidConfiguration { .. })
        ));
        let bad_tolerance = ImexConfig {
            stage_solver: ImplicitStageSolver::Owner {
                absolute_tolerance: -1.0,
                relative_tolerance: 0.0,
            },
            ..config
        };
        assert!(matches!(
            bad_tolerance.validate(),
            Err(SolveError::InvalidConfiguration { .. })
        ));
    }
}
