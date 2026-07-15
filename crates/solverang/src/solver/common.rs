//! Validation and classification helpers shared by every solver entry point.
//!
//! Each root-finding solver performs the same pre-flight checks (dimensions,
//! initial-point length and finiteness) and the same in-loop finiteness
//! guards. Centralizing them keeps failure policy in one place.

use crate::problem::Problem;
use crate::solver::auto::SolverChoice;
use crate::solver::config::SolverConfig;
use crate::solver::levenberg_marquardt::LMSolver;
use crate::solver::lm_config::LMConfig;
use crate::solver::newton_raphson::Solver;
use crate::solver::result::{SolveError, SolveResult};

/// Pre-flight validation shared by all solvers: dimension checks plus a
/// finiteness check on the initial point.
pub(crate) fn validate_problem<P: Problem + ?Sized>(
    problem: &P,
    x0: &[f64],
) -> Result<(), SolveError> {
    let n = problem.variable_count();
    let m = problem.residual_count();

    if n == 0 {
        return Err(SolveError::NoVariables);
    }
    if m == 0 {
        return Err(SolveError::NoEquations);
    }
    if x0.len() != n {
        return Err(SolveError::DimensionMismatch {
            expected: n,
            got: x0.len(),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(SolveError::NonFiniteResiduals);
    }
    Ok(())
}

/// Guard against NaN/infinity in residuals.
pub(crate) fn check_residuals_finite(residuals: &[f64]) -> Result<(), SolveError> {
    if residuals.iter().any(|r| !r.is_finite()) {
        Err(SolveError::NonFiniteResiduals)
    } else {
        Ok(())
    }
}

/// Guard against NaN/infinity in sparse Jacobian triplets.
pub(crate) fn check_jacobian_finite(entries: &[(usize, usize, f64)]) -> Result<(), SolveError> {
    if entries.iter().any(|(_, _, v)| !v.is_finite()) {
        Err(SolveError::NonFiniteJacobian)
    } else {
        Ok(())
    }
}

/// Resolve [`SolverChoice::Auto`] to a concrete algorithm from the problem
/// shape: square systems get Newton-Raphson, everything else
/// Levenberg-Marquardt.
pub(crate) fn resolve_choice<P: Problem + ?Sized>(
    choice: SolverChoice,
    problem: &P,
) -> SolverChoice {
    match choice {
        SolverChoice::Auto => {
            if problem.is_square() {
                SolverChoice::NewtonRaphson
            } else {
                SolverChoice::LevenbergMarquardt
            }
        }
        concrete => concrete,
    }
}

/// Dispatch a solve to Newton-Raphson or Levenberg-Marquardt according to
/// `choice` (resolving `Auto` from the problem shape first). This is the one
/// place that maps a [`SolverChoice`] to an actual solver run.
pub(crate) fn solve_with_choice<P: Problem + ?Sized>(
    choice: SolverChoice,
    nr_config: &SolverConfig,
    lm_config: &LMConfig,
    problem: &P,
    x0: &[f64],
) -> SolveResult {
    match resolve_choice(choice, problem) {
        SolverChoice::NewtonRaphson => Solver::new(nr_config.clone()).solve(problem, x0),
        SolverChoice::LevenbergMarquardt | SolverChoice::Auto => {
            LMSolver::new(lm_config.clone()).solve(problem, x0)
        }
    }
}
