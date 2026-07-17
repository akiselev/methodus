//! Result, status, and error types for the DAE/ODE time integrator.
//!
//! [`integrate_dae`](super::integrate_dae) never panics and never returns a bare
//! `Result`; it always returns a [`Trajectory`] that carries the *partial* history
//! accumulated up to the point of failure plus an [`IntegrateStatus`]. A caller that
//! wants the strict success/failure split reads [`Trajectory::error`] (or matches on
//! `status`). Non-convergence of the per-step Newton solve is reported as a typed
//! [`IntegrateError::NonConvergence`], never a panic.

use std::fmt;

use numeric_contracts::NumericError;

/// The time/state history produced by [`integrate_dae`](super::integrate_dae).
///
/// `times[k]` is the wall-clock time at sample `k` and `states[k]` is the state
/// vector `x` at that time; the two vectors always have equal length. `times[0]`
/// is the initial time and `states[0]` is the initial condition. On failure the
/// vectors still hold every accepted step up to the failing step, and
/// [`status`](Trajectory::status) is [`IntegrateStatus::Failed`].
#[derive(Clone, Debug)]
pub struct Trajectory {
    /// Sample times, ascending, starting at `t_span.0`.
    pub times: Vec<f64>,
    /// Sampled states, one per entry in [`times`](Trajectory::times).
    pub states: Vec<Vec<f64>>,
    /// How the integration ended.
    pub status: IntegrateStatus,
    /// Aggregate counters over the whole run.
    pub stats: IntegrateStats,
}

impl Trajectory {
    /// Whether the integration reached the end of the span (or a terminal event)
    /// without a failure.
    #[must_use]
    pub fn is_completed(&self) -> bool {
        matches!(
            self.status,
            IntegrateStatus::Completed | IntegrateStatus::Terminated
        )
    }

    /// The final state, if any sample was recorded.
    #[must_use]
    pub fn last_state(&self) -> Option<&[f64]> {
        self.states.last().map(Vec::as_slice)
    }

    /// The final time, if any sample was recorded.
    #[must_use]
    pub fn last_time(&self) -> Option<f64> {
        self.times.last().copied()
    }

    /// The number of recorded samples (accepted steps + 1 for the initial point).
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether no samples were recorded (only possible on an input-validation failure).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// The failure that ended the run, if it failed.
    #[must_use]
    pub fn error(&self) -> Option<&IntegrateError> {
        match &self.status {
            IntegrateStatus::Failed(e) => Some(e),
            _ => None,
        }
    }
}

/// How an integration run terminated.
#[derive(Clone, Debug)]
pub enum IntegrateStatus {
    /// Reached the end of the requested time span.
    Completed,
    /// Stopped early on a terminal event (reserved for the event loop, a deferred
    /// seam — see the module docs). Not produced by the current integrators.
    Terminated,
    /// Ended on a typed failure. The trajectory still holds the partial history.
    Failed(IntegrateError),
}

/// Aggregate diagnostics accumulated over an integration run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntegrateStats {
    /// Steps whose local-error estimate was accepted.
    pub accepted_steps: usize,
    /// Step attempts rejected (error over tolerance, or a Newton failure that was
    /// retried with a smaller step).
    pub rejected_steps: usize,
    /// Total per-step Newton iterations across all attempts.
    pub newton_iterations: usize,
    /// Total residual (`residual_at` + `charge`/`mass_apply`) evaluations.
    pub residual_evals: usize,
    /// Total iteration-matrix assemblies.
    pub jacobian_evals: usize,
}

/// A typed, non-panicking integrator failure.
#[derive(Clone, Debug, PartialEq)]
pub enum IntegrateError {
    /// The per-step Newton solve failed to converge and the step could not be
    /// reduced any further (fixed-step mode, or step size hit `min_step`).
    NonConvergence {
        /// Time at the start of the failing step.
        t: f64,
        /// Step size of the failing attempt.
        h: f64,
        /// Last residual norm the Newton solve reached.
        residual_norm: f64,
    },
    /// The adaptive controller drove the step below `min_step` while still failing
    /// the local-error test — the problem is too stiff/ill-scaled for the tolerances.
    StepSizeUnderflow {
        /// Time at which the step underflowed.
        t: f64,
        /// The step size that underflowed.
        h: f64,
    },
    /// A step produced a non-finite state (NaN/Inf) that could not be recovered.
    NonFiniteState {
        /// Time at which the non-finite state appeared.
        t: f64,
    },
    /// The step budget (`max_steps`) was exhausted before reaching the span end.
    StepLimitReached {
        /// Time reached when the budget ran out.
        t: f64,
        /// The budget that was hit.
        steps: usize,
    },
    /// The underlying [`DaeResidual`](numeric_contracts::DaeResidual) seam returned
    /// a [`NumericError`] (e.g. a matrix-free block with no assembled
    /// `iteration_matrix`, or a dimension mismatch).
    Seam(NumericError),
    /// The inputs were inconsistent (dimension mismatch, non-positive step, empty
    /// or reversed span).
    InvalidInput {
        /// A short, human-legible reason.
        reason: String,
    },
}

impl fmt::Display for IntegrateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegrateError::NonConvergence {
                t,
                h,
                residual_norm,
            } => write!(
                f,
                "per-step Newton did not converge at t={t:e} (h={h:e}, residual {residual_norm:e})"
            ),
            IntegrateError::StepSizeUnderflow { t, h } => {
                write!(f, "step size underflow at t={t:e} (h={h:e})")
            }
            IntegrateError::NonFiniteState { t } => {
                write!(f, "non-finite state at t={t:e}")
            }
            IntegrateError::StepLimitReached { t, steps } => {
                write!(f, "step limit ({steps}) reached at t={t:e}")
            }
            IntegrateError::Seam(e) => write!(f, "DAE seam error: {e}"),
            IntegrateError::InvalidInput { reason } => write!(f, "invalid input: {reason}"),
        }
    }
}

impl std::error::Error for IntegrateError {}

impl From<NumericError> for IntegrateError {
    fn from(e: NumericError) -> Self {
        IntegrateError::Seam(e)
    }
}
