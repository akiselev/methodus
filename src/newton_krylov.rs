//! E7/SC-W3 (SV7-F3 subset, pulled forward): an inexact Newton–Krylov
//! driver over a matrix-free Jacobian action.
//!
//! [`solve_newton_krylov`] solves `F(x) = 0` by, at each iterate, solving
//! the Newton system `J(x) s = −F(x)` *inexactly* with a caller-selected
//! [`KrylovMethod`] to the forcing tolerance `‖F + J s‖ ≤ η_k ‖F‖`, then
//! taking a backtracked step satisfying the Eisenstat–Walker sufficient-
//! decrease condition. The Jacobian is never assembled: [`JacobianOperator`]
//! exposes [`NonlinearOperator::jacobian_vector_product`] at a frozen state
//! as a [`LinearOperator`] whose declared properties come from
//! [`NonlinearOperator::jacobian_properties`], so the Krylov method's own
//! admission applies unchanged (conjugate gradient refuses a Jacobian
//! declared `Nonsymmetric`, MINRES refuses one not declared `Symmetric`,
//! GMRES/BiCGSTAB admit any declaration). No method is ever silently
//! substituted.
//!
//! Two hooks make the driver composable without teaching it any physics:
//! a [`PreconditionerFactory`] builds (or refreshes) a preconditioner for
//! the Jacobian at each new iterate, and a [`crate::NullspaceProjector`]
//! keeps a singular-Jacobian system (a declared constant mode, say) on its
//! pseudo-solution branch as [`crate::solve_krylov`] documents. Telemetry
//! is typed and deterministic: every outer iteration records the residual
//! norm, the forcing term used, and the inner linear solve's method,
//! iteration count, verdict and final residual. A linear solve that ran
//! out of budget is recorded as such and its step still tried, because
//! inexact Newton needs only a descent direction; a refusal or breakdown
//! is an error.

use serde::{Deserialize, Serialize};

use crate::context::EvaluationContext;
use crate::error::{NumericError, SolveError};
use crate::krylov_method::{KrylovMethod, KrylovMethodKind, solve_krylov};
use crate::linear::l2;
use crate::nonlinear::{IterationTrace, NonlinearSolver, SolveReport};
use crate::nullspace::NullspaceProjector;
use crate::operator::{
    LinearOperator, NonlinearOperator, OperatorProperties, OperatorSymmetry, Preconditioner,
};

/// The Jacobian `∂F/∂x` of a [`NonlinearOperator`] at one frozen state, as
/// a square matrix-free [`LinearOperator`].
#[derive(Debug)]
pub struct JacobianOperator<'a, F: NonlinearOperator + ?Sized> {
    operator: &'a F,
    state: &'a [f64],
    properties: OperatorProperties,
}

impl<'a, F: NonlinearOperator + ?Sized> JacobianOperator<'a, F> {
    /// Freezes `operator`'s Jacobian at `state`.
    ///
    /// # Errors
    /// Refuses a state whose length is not the operator dimension or that
    /// contains non-finite values.
    pub fn new(operator: &'a F, state: &'a [f64]) -> Result<Self, NumericError> {
        NumericError::require_len("Jacobian state", state.len(), operator.dimension())?;
        NumericError::require_finite("Jacobian state", state)?;
        Ok(Self {
            operator,
            state,
            properties: operator.jacobian_properties(),
        })
    }

    /// The state the Jacobian is frozen at.
    #[must_use]
    pub const fn state(&self) -> &'a [f64] {
        self.state
    }
}

impl<F: NonlinearOperator + ?Sized> LinearOperator for JacobianOperator<'_, F> {
    fn rows(&self) -> usize {
        self.operator.dimension()
    }

    fn columns(&self) -> usize {
        self.operator.dimension()
    }

    fn symmetry(&self) -> OperatorSymmetry {
        self.properties.symmetry()
    }

    fn properties(&self) -> OperatorProperties {
        self.properties.clone()
    }

    fn apply(
        &self,
        context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        NumericError::require_len("Jacobian direction", input.len(), self.rows())?;
        NumericError::require_len("Jacobian action output", output.len(), self.rows())?;
        NumericError::require_finite("Jacobian direction", input)?;
        self.operator
            .jacobian_vector_product(context, self.state, input, output)?;
        NumericError::require_finite("Jacobian action", output)
    }
}

/// Builds the preconditioner used for the Newton system at each iterate.
pub trait PreconditionerFactory: Send + Sync {
    /// Builds (or refreshes) a preconditioner approximating the inverse of
    /// `jacobian`, the Jacobian frozen at `state`. Returning `None` runs the
    /// linear solve unpreconditioned at this iterate.
    ///
    /// # Errors
    /// Propagates construction failures; the Newton solve stops on them.
    fn build<'a>(
        &'a self,
        context: &EvaluationContext,
        jacobian: &dyn LinearOperator,
        state: &[f64],
    ) -> Result<Option<Box<dyn Preconditioner + 'a>>, NumericError>;
}

/// How the forcing term `η_k` (the relative tolerance of the `k`-th inner
/// linear solve, `‖F_k + J_k s_k‖ ≤ η_k ‖F_k‖`) is chosen.
///
/// Both policies are floored at half the outer convergence threshold
/// relative to `‖F_k‖`, so the last inner solves are never asked for more
/// accuracy than the outer test can observe (Eisenstat–Walker's
/// oversolving safeguard).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForcingPolicy {
    /// The same `η` at every iteration. A tiny value reproduces exact
    /// Newton (quadratic convergence) at the cost of full inner solves.
    Constant { forcing: f64 },
    /// Eisenstat–Walker choice 2:
    /// `η_k = γ (‖F_k‖ / ‖F_{k−1}‖)^α`, with `η_0 = initial`, capped at
    /// `maximum`, and safeguarded from below by `γ η_{k−1}^α` whenever that
    /// exceeds 0.1 so the sequence cannot collapse prematurely. With
    /// `α = 2` the outer iteration converges superlinearly.
    EisenstatWalker {
        initial: f64,
        gamma: f64,
        alpha: f64,
        maximum: f64,
    },
}

impl ForcingPolicy {
    fn validate(&self) -> Result<(), SolveError> {
        let valid = match self {
            Self::Constant { forcing } => forcing.is_finite() && *forcing > 0.0 && *forcing < 1.0,
            Self::EisenstatWalker {
                initial,
                gamma,
                alpha,
                maximum,
            } => {
                initial.is_finite()
                    && *initial > 0.0
                    && maximum.is_finite()
                    && *maximum > 0.0
                    && *maximum < 1.0
                    && *initial <= *maximum
                    && gamma.is_finite()
                    && *gamma > 0.0
                    && *gamma <= 1.0
                    && alpha.is_finite()
                    && *alpha > 1.0
                    && *alpha <= 2.0
            }
        };
        if valid {
            Ok(())
        } else {
            Err(SolveError::InvalidConfiguration {
                reason: "forcing terms must be finite and lie in (0, 1); Eisenstat–Walker needs \
                         0 < initial <= maximum < 1, 0 < gamma <= 1, and 1 < alpha <= 2"
                    .into(),
            })
        }
    }
}

/// Controls the outer inexact Newton iteration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewtonKrylovConfig {
    pub max_iterations: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub forcing: ForcingPolicy,
    /// Sufficient-decrease constant `t` in
    /// `‖F(x + λ s)‖ ≤ (1 − t λ (1 − η)) ‖F(x)‖`.
    pub sufficient_decrease: f64,
    pub initial_damping: f64,
    pub minimum_damping: f64,
    pub max_line_search_steps: usize,
}

impl Default for NewtonKrylovConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-8,
            forcing: ForcingPolicy::EisenstatWalker {
                initial: 0.1,
                gamma: 0.9,
                alpha: 2.0,
                maximum: 0.9,
            },
            sufficient_decrease: 1.0e-4,
            initial_damping: 1.0,
            minimum_damping: 1.0e-4,
            max_line_search_steps: 12,
        }
    }
}

/// Evidence about one inner linear solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearStepSummary {
    pub method: KrylovMethodKind,
    /// The forcing term `η_k` the solve was asked to meet.
    pub forcing: f64,
    pub iterations: usize,
    /// The inner solver's own verdict against `η_k ‖F_k‖`.
    pub converged: bool,
    /// The inner solver's final reported residual norm (its semantics).
    pub residual_norm: f64,
    pub restart_cycles: Option<usize>,
}

/// One outer iteration of an inexact Newton–Krylov solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewtonKrylovIteration {
    pub iteration: usize,
    /// `‖F(x_k)‖` at the start of the iteration.
    pub residual_norm: f64,
    /// `None` on the terminal (converged or exhausted) observation.
    pub linear: Option<LinearStepSummary>,
    pub accepted_damping: Option<f64>,
}

/// Final state and convergence evidence from an inexact Newton–Krylov solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewtonKrylovReport {
    pub state: Vec<f64>,
    pub converged: bool,
    pub trace: Vec<NewtonKrylovIteration>,
}

impl NewtonKrylovReport {
    /// Total inner Krylov iterations over the whole solve.
    #[must_use]
    pub fn linear_iterations(&self) -> usize {
        self.trace
            .iter()
            .filter_map(|entry| entry.linear.as_ref())
            .map(|linear| linear.iterations)
            .sum()
    }
}

/// Solve `F(x) = 0` by inexact Newton with a Krylov inner solve.
///
/// # Errors
/// Refuses invalid configuration, dimension mismatches and non-finite
/// states; propagates the selected Krylov method's admission refusals and
/// breakdowns, preconditioner construction failures, and a line search
/// that cannot achieve sufficient decrease (`LineSearchFailed`). Running
/// out of outer iterations is reported as `converged == false`, not as an
/// error.
pub fn solve_newton_krylov(
    operator: &(impl NonlinearOperator + ?Sized),
    context: &EvaluationContext,
    initial_state: &[f64],
    method: &KrylovMethod,
    preconditioner: Option<&dyn PreconditionerFactory>,
    nullspace_projector: Option<&dyn NullspaceProjector>,
    config: &NewtonKrylovConfig,
) -> Result<NewtonKrylovReport, SolveError> {
    validate_config(config)?;
    let dimension = operator.dimension();
    NumericError::require_len("initial nonlinear state", initial_state.len(), dimension)?;
    NumericError::require_finite("initial nonlinear state", initial_state)?;

    let mut state = initial_state.to_vec();
    let mut residual = vec![0.0; dimension];
    operator.residual(context, &state, &mut residual)?;
    NumericError::require_finite("initial residual", &residual)?;
    let mut residual_norm = l2(&residual)?;
    let threshold = config.absolute_tolerance + config.relative_tolerance * residual_norm;
    let mut trace = Vec::with_capacity(config.max_iterations + 1);
    let mut previous: Option<(f64, f64)> = None; // (‖F_{k−1}‖, η_{k−1})

    for iteration in 0..=config.max_iterations {
        if residual_norm <= threshold {
            trace.push(NewtonKrylovIteration {
                iteration,
                residual_norm,
                linear: None,
                accepted_damping: None,
            });
            return Ok(NewtonKrylovReport {
                state,
                converged: true,
                trace,
            });
        }
        if iteration == config.max_iterations {
            trace.push(NewtonKrylovIteration {
                iteration,
                residual_norm,
                linear: None,
                accepted_damping: None,
            });
            return Ok(NewtonKrylovReport {
                state,
                converged: false,
                trace,
            });
        }

        let forcing = forcing_term(&config.forcing, residual_norm, threshold, previous);
        let jacobian = JacobianOperator::new(operator, &state)?;
        let built = match preconditioner {
            Some(factory) => factory.build(context, &jacobian, &state)?,
            None => None,
        };
        let rhs: Vec<f64> = residual.iter().map(|value| -value).collect();
        let linear = solve_krylov(
            &method.with_tolerances(0.0, forcing),
            &jacobian,
            built.as_deref(),
            nullspace_projector,
            context,
            &rhs,
            &vec![0.0; dimension],
        )?;
        drop(built);
        let step = linear.solution;
        NumericError::require_finite("Newton–Krylov step", &step)?;
        let summary = LinearStepSummary {
            method: linear.method,
            forcing,
            iterations: linear.trace.len().saturating_sub(1),
            converged: linear.converged,
            residual_norm: linear
                .trace
                .last()
                .map_or(f64::NAN, |entry| entry.residual_norm),
            restart_cycles: linear.restart_cycles,
        };

        let (next_state, next_residual, next_norm, damping) = backtrack(
            operator,
            context,
            &state,
            &step,
            residual_norm,
            forcing,
            config,
        )?;
        trace.push(NewtonKrylovIteration {
            iteration,
            residual_norm,
            linear: Some(summary),
            accepted_damping: Some(damping),
        });
        previous = Some((residual_norm, forcing));
        state = next_state;
        residual = next_residual;
        residual_norm = next_norm;
    }
    unreachable!("iteration loop always returns")
}

fn forcing_term(
    policy: &ForcingPolicy,
    residual_norm: f64,
    threshold: f64,
    previous: Option<(f64, f64)>,
) -> f64 {
    let raw = match policy {
        ForcingPolicy::Constant { forcing } => *forcing,
        ForcingPolicy::EisenstatWalker {
            initial,
            gamma,
            alpha,
            maximum,
        } => match previous {
            None => *initial,
            Some((previous_norm, previous_forcing)) => {
                let mut eta = gamma * (residual_norm / previous_norm).powf(*alpha);
                let safeguard = gamma * previous_forcing.powf(*alpha);
                if safeguard > 0.1 {
                    eta = eta.max(safeguard);
                }
                eta.min(*maximum)
            }
        },
    };
    // Oversolving safeguard: never demand more than the outer test can see.
    let floor = 0.5 * threshold / residual_norm;
    let forcing = raw.max(floor);
    if forcing.is_finite() {
        forcing.min(0.999)
    } else {
        raw
    }
}

#[allow(clippy::too_many_arguments)]
fn backtrack(
    operator: &(impl NonlinearOperator + ?Sized),
    context: &EvaluationContext,
    state: &[f64],
    step: &[f64],
    current_norm: f64,
    forcing: f64,
    config: &NewtonKrylovConfig,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64), SolveError> {
    let mut damping = config.initial_damping;
    let mut candidate = vec![0.0; state.len()];
    let mut residual = vec![0.0; state.len()];
    for _ in 0..config.max_line_search_steps {
        for ((candidate, state), step) in candidate.iter_mut().zip(state).zip(step) {
            *candidate = state + damping * step;
        }
        operator.residual(context, &candidate, &mut residual)?;
        if residual.iter().all(|value| value.is_finite()) {
            let norm = l2(&residual)?;
            let bound =
                (1.0 - config.sufficient_decrease * damping * (1.0 - forcing)) * current_norm;
            if norm <= bound {
                return Ok((candidate, residual, norm, damping));
            }
        }
        damping *= 0.5;
        if damping < config.minimum_damping {
            break;
        }
    }
    Err(SolveError::LineSearchFailed)
}

fn validate_config(config: &NewtonKrylovConfig) -> Result<(), SolveError> {
    config.forcing.validate()?;
    let tolerances_valid = config.absolute_tolerance.is_finite()
        && config.absolute_tolerance >= 0.0
        && config.relative_tolerance.is_finite()
        && config.relative_tolerance >= 0.0
        && (config.absolute_tolerance > 0.0 || config.relative_tolerance > 0.0);
    let damping_valid = config.initial_damping.is_finite()
        && config.initial_damping > 0.0
        && config.initial_damping <= 1.0
        && config.minimum_damping.is_finite()
        && config.minimum_damping > 0.0
        && config.minimum_damping <= config.initial_damping;
    let decrease_valid = config.sufficient_decrease.is_finite()
        && config.sufficient_decrease > 0.0
        && config.sufficient_decrease < 0.5;
    if config.max_iterations == 0
        || config.max_iterations.checked_add(1).is_none()
        || config.max_line_search_steps == 0
        || !tolerances_valid
        || !damping_valid
        || !decrease_valid
    {
        return Err(SolveError::InvalidConfiguration {
            reason: "Newton–Krylov limits, tolerances, damping, and sufficient-decrease constant \
                     must be positive, finite, and ordered"
                .into(),
        });
    }
    Ok(())
}

/// [`solve_newton_krylov`] as a [`NonlinearSolver`], so time integrators
/// (`bdf_step_with`) and block drivers can run inexact Newton over a
/// matrix-free step Jacobian.
pub struct NewtonKrylovSolver<'a> {
    method: &'a KrylovMethod,
    preconditioner: Option<&'a dyn PreconditionerFactory>,
    nullspace_projector: Option<&'a dyn NullspaceProjector>,
    config: &'a NewtonKrylovConfig,
}

impl std::fmt::Debug for NewtonKrylovSolver<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NewtonKrylovSolver")
            .field("method", self.method)
            .field("preconditioner", &self.preconditioner.is_some())
            .field("nullspace_projector", &self.nullspace_projector.is_some())
            .field("config", self.config)
            .finish()
    }
}

impl<'a> NewtonKrylovSolver<'a> {
    #[must_use]
    pub const fn new(
        method: &'a KrylovMethod,
        preconditioner: Option<&'a dyn PreconditionerFactory>,
        nullspace_projector: Option<&'a dyn NullspaceProjector>,
        config: &'a NewtonKrylovConfig,
    ) -> Self {
        Self {
            method,
            preconditioner,
            nullspace_projector,
            config,
        }
    }
}

impl NonlinearSolver for NewtonKrylovSolver<'_> {
    fn solve(
        &self,
        operator: &dyn NonlinearOperator,
        context: &EvaluationContext,
        initial_state: &[f64],
    ) -> Result<SolveReport, SolveError> {
        let report = solve_newton_krylov(
            operator,
            context,
            initial_state,
            self.method,
            self.preconditioner,
            self.nullspace_projector,
            self.config,
        )?;
        Ok(SolveReport {
            state: report.state,
            converged: report.converged,
            trace: report
                .trace
                .into_iter()
                .map(|entry| IterationTrace {
                    iteration: entry.iteration,
                    residual_norm: entry.residual_norm,
                    scaled_residual_norm: entry.residual_norm,
                    block_residual_norms: Vec::new(),
                    accepted_damping: entry.accepted_damping,
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_and_forcing_are_validated() {
        let bad_forcing = NewtonKrylovConfig {
            forcing: ForcingPolicy::Constant { forcing: 1.0 },
            ..NewtonKrylovConfig::default()
        };
        assert!(matches!(
            validate_config(&bad_forcing),
            Err(SolveError::InvalidConfiguration { .. })
        ));
        let bad_walker = NewtonKrylovConfig {
            forcing: ForcingPolicy::EisenstatWalker {
                initial: 0.95,
                gamma: 0.9,
                alpha: 2.0,
                maximum: 0.9,
            },
            ..NewtonKrylovConfig::default()
        };
        assert!(matches!(
            validate_config(&bad_walker),
            Err(SolveError::InvalidConfiguration { .. })
        ));
        let bad_decrease = NewtonKrylovConfig {
            sufficient_decrease: 0.5,
            ..NewtonKrylovConfig::default()
        };
        assert!(matches!(
            validate_config(&bad_decrease),
            Err(SolveError::InvalidConfiguration { .. })
        ));
        validate_config(&NewtonKrylovConfig::default()).unwrap();
    }

    #[test]
    fn eisenstat_walker_forcing_shrinks_with_the_residual_and_is_floored() {
        let policy = ForcingPolicy::EisenstatWalker {
            initial: 0.1,
            gamma: 0.9,
            alpha: 2.0,
            maximum: 0.9,
        };
        assert_eq!(forcing_term(&policy, 1.0, 1.0e-12, None), 0.1);
        let next = forcing_term(&policy, 0.01, 1.0e-12, Some((1.0, 0.1)));
        assert!((next - 0.9 * 1.0e-4).abs() < 1.0e-15);
        // Near convergence the floor takes over: half the threshold relative
        // to the residual.
        let floored = forcing_term(&policy, 2.0e-12, 1.0e-12, Some((1.0e-6, 1.0e-3)));
        assert!((floored - 0.25).abs() < 1.0e-12);
        // The safeguard keeps η from collapsing when the previous η was large.
        let safeguarded = forcing_term(&policy, 0.5, 1.0e-12, Some((1.0, 0.9)));
        assert!((safeguarded - 0.9 * 0.81).abs() < 1.0e-12);
    }
}
