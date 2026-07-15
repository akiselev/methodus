//! Result types for optimization solvers.

use super::multiplier_store::MultiplierStore;

/// KKT residual components for convergence assessment.
#[derive(Debug, Clone)]
pub struct KktResidual {
    /// Primal feasibility: `max(||g(x)||, max(h_j(x), 0))`.
    pub primal: f64,
    /// Dual feasibility: `||∇_x L||`.
    pub dual: f64,
    /// Complementarity: `max |μ_j * h_j(x)|`.
    pub complementarity: f64,
}

impl KktResidual {
    /// Check if all KKT components are within their tolerances.
    pub fn is_within_tolerance(&self, primal_tol: f64, dual_tol: f64, comp_tol: f64) -> bool {
        self.primal < primal_tol && self.dual < dual_tol && self.complementarity < comp_tol
    }
}

/// Why a line search gave up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSearchError {
    /// The supplied direction is not a descent direction (`∇f·d ≥ 0`).
    NotDescentDirection,
    /// The objective or gradient produced a non-finite value.
    NonFiniteValue,
    /// The step shrank below `line_search_min_step` without satisfying Armijo.
    StepTooSmall,
    /// The evaluation budget (`line_search_max_evals`) was exhausted.
    EvaluationBudgetExceeded,
    /// No feasible step exists along the direction (`alpha_max ≤ 0`).
    InfeasibleDirection,
}

/// A failed line search, with the evaluations consumed before giving up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineSearchFailure {
    /// The failure reason.
    pub reason: LineSearchError,
    /// Number of objective evaluations consumed.
    pub f_evals: usize,
    /// Number of gradient evaluations consumed.
    pub grad_evals: usize,
}

/// Status of an optimization solve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationStatus {
    /// Solver converged: KKT conditions satisfied within tolerance.
    Converged,
    /// Maximum iterations reached without convergence.
    MaxIterationsReached,
    /// Problem is infeasible (constraints cannot be simultaneously satisfied).
    Infeasible,
    /// Solver diverged (multipliers or objective exploded).
    Diverged,
    /// The penalty parameter saturated without further feasibility progress.
    Stalled,
    /// A line search failed to make progress; the reported point is the best
    /// found. Carries the failure reason and evaluation counts.
    LineSearchFailed(LineSearchFailure),
    /// The selected algorithm cannot solve the registered problem structure
    /// (e.g. BFGS explicitly selected while equality constraints exist), or
    /// the configuration itself is invalid.
    UnsupportedProblemStructure {
        /// Human-readable description of the mismatch.
        reason: String,
    },
}

impl OptimizationStatus {
    /// Whether the solver converged successfully.
    pub fn is_converged(&self) -> bool {
        matches!(self, Self::Converged)
    }
}

/// Result of an optimization solve.
///
/// Fields are public for direct access; [`is_converged`](Self::is_converged)
/// and [`iterations`](Self::iterations) mirror the accessor vocabulary of
/// [`SolveResult`](crate::solver::SolveResult) so the two solver families
/// read uniformly at call sites.
#[derive(Debug)]
pub struct OptimizationResult {
    /// Final objective value f(x*).
    pub objective_value: f64,
    /// Solve status.
    pub status: OptimizationStatus,
    /// Total outer iterations (ALM outer loop, or BFGS iterations).
    pub outer_iterations: usize,
    /// Total inner iterations (ALM inner NR/LM solves, summed).
    pub inner_iterations: usize,
    /// Final KKT residual (primal, dual, complementarity).
    pub kkt_residual: KktResidual,
    /// Lagrange multipliers for sensitivity analysis.
    pub multipliers: MultiplierStore,
    /// Per-constraint violation values (positive = violated).
    pub constraint_violations: Vec<f64>,
    /// Wall-clock duration of the solve.
    pub duration: std::time::Duration,
}

impl OptimizationResult {
    /// Whether the solve converged (KKT conditions within tolerance).
    pub fn is_converged(&self) -> bool {
        self.status.is_converged()
    }

    /// Headline iteration count: outer iterations (ALM outer loop, or
    /// BFGS/trust-region iterations). Matches the semantics of
    /// `SolveResult::iterations()`.
    pub fn iterations(&self) -> usize {
        self.outer_iterations
    }
}
