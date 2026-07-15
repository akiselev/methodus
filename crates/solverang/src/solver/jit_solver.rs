//! JIT-enabled solver implementation.
//!
//! This module provides a solver that can use JIT-compiled constraint evaluation
//! for improved performance on large constraint systems.

use crate::jit::{CompiledConstraints, JITCompiler, JITConfig, JITError, JITFunction, JitMode};
use crate::problem::Problem;
use crate::solver::result::{SolveError, SolveResult};
use nalgebra::{DMatrix, DVector};

/// Why a [`JITSolver::solve`] call used interpreted evaluation instead of JIT.
#[derive(Clone, Debug)]
pub enum JitFallback {
    /// The configuration forces interpreted evaluation ([`JitMode::ForceInterpreted`]).
    ForcedInterpreted,
    /// The problem does not implement `lower_to_compiled_constraints()`.
    NoCompiledConstraints,
    /// JIT is not supported on this platform.
    PlatformUnavailable,
    /// The estimated work was below `jit_threshold`.
    BelowThreshold,
    /// Compilation was attempted but failed.
    CompilationFailed(JITError),
}

/// A solver that uses JIT compilation for large constraint systems.
///
/// For problems above a configurable threshold, this solver compiles the
/// constraint evaluation to native code, eliminating virtual function call
/// overhead during the iterative solve loop.
///
/// # Example
///
/// ```rust,ignore
/// use solverang::solver::JITSolver;
/// use solverang::jit::JITConfig;
///
/// let solver = JITSolver::new(JITConfig::default());
/// let result = solver.solve(&problem, &initial_point);
/// ```
pub struct JITSolver {
    /// Configuration.
    config: JITConfig,

    /// Cached JIT compiler.
    compiler: Option<JITCompiler>,

    /// Why the most recent `solve()` fell back to interpreted evaluation
    /// (`None` when it used JIT).
    last_fallback: Option<JitFallback>,
}

impl JITSolver {
    /// Create a new JIT-enabled solver.
    pub fn new(config: JITConfig) -> Self {
        Self {
            config,
            compiler: None,
            last_fallback: None,
        }
    }

    /// Create a solver with default configuration.
    pub fn default_solver() -> Self {
        Self::new(JITConfig::default())
    }

    /// Get the solver configuration.
    pub fn config(&self) -> &JITConfig {
        &self.config
    }

    /// Why the most recent [`solve`](Self::solve) used interpreted evaluation.
    ///
    /// Returns `None` if the last solve ran JIT-compiled, or if no solve has
    /// happened yet. Use this to distinguish an intentional fallback (below
    /// threshold, forced interpreted) from an unexpected one (compilation
    /// failure, unsupported platform).
    pub fn last_jit_fallback(&self) -> Option<&JitFallback> {
        self.last_fallback.as_ref()
    }

    /// Check if JIT will be used for the given problem.
    pub fn will_use_jit<P: Problem>(&self, problem: &P) -> bool {
        match self.config.mode {
            JitMode::ForceInterpreted => false,
            JitMode::ForceJit => crate::jit::jit_available(),
            JitMode::Auto => {
                let estimated_work = problem.residual_count() * self.config.estimated_iterations;
                estimated_work > self.config.jit_threshold && crate::jit::jit_available()
            }
        }
    }

    /// Check if JIT should be used for the given compiled constraints.
    ///
    /// Uses `total_ops()` from the compiled constraints for a more accurate
    /// estimate of computation cost than `will_use_jit()`.
    pub fn should_jit(&self, cc: &CompiledConstraints) -> bool {
        match self.config.mode {
            JitMode::ForceInterpreted => false,
            JitMode::ForceJit => crate::jit::jit_available(),
            JitMode::Auto => {
                let estimated_work = cc.total_ops() * self.config.estimated_iterations;
                estimated_work > self.config.jit_threshold && crate::jit::jit_available()
            }
        }
    }

    /// Solve a problem using either JIT or interpreted evaluation.
    pub fn solve<P: Problem>(&mut self, problem: &P, x0: &[f64]) -> SolveResult {
        if let Err(error) = super::common::validate_problem(problem, x0) {
            return error.into();
        }

        // Try JIT path if the problem can produce CompiledConstraints. Record
        // why we fall back so callers can distinguish an intentional
        // interpreted solve from an unexpected one (see `last_jit_fallback`).
        self.last_fallback = if self.config.mode == JitMode::ForceInterpreted {
            Some(JitFallback::ForcedInterpreted)
        } else if let Some(compiled) = problem.lower_to_compiled_constraints() {
            if !crate::jit::jit_available() {
                Some(JitFallback::PlatformUnavailable)
            } else if !self.should_jit(&compiled) {
                Some(JitFallback::BelowThreshold)
            } else {
                match self.compile(&compiled) {
                    Ok(jit_fn) => {
                        self.last_fallback = None;
                        return self.solve_with_jit(&jit_fn, x0);
                    }
                    Err(e) => Some(JitFallback::CompilationFailed(e)),
                }
            }
        } else {
            Some(JitFallback::NoCompiledConstraints)
        };

        self.solve_interpreted(problem, x0)
    }

    /// Solve using JIT-compiled evaluation.
    ///
    /// This method is called when a compiled JIT function is provided directly.
    pub fn solve_with_jit(&self, jit_fn: &JITFunction, x0: &[f64]) -> SolveResult {
        let n = jit_fn.variable_count();
        let m = jit_fn.residual_count();

        if x0.len() != n {
            return SolveResult::Failed {
                error: SolveError::DimensionMismatch {
                    expected: n,
                    got: x0.len(),
                },
            };
        }

        let mut x = DVector::from_column_slice(x0);
        let mut residuals = vec![0.0; m];
        let mut j = DMatrix::zeros(m, n);

        for iteration in 0..self.config.max_iterations {
            // Evaluate both residuals and dense Jacobian in a single fused pass.
            // JIT writes directly into DMatrix column-major storage — no COO copy.
            jit_fn.evaluate_both_dense(x.as_slice(), &mut residuals, j.as_mut_slice());

            if let Err(error) = super::common::check_residuals_finite(&residuals) {
                return error.into();
            }

            let r = DVector::from_column_slice(&residuals);
            let norm = r.norm();

            // Check convergence
            if norm < self.config.tolerance {
                return SolveResult::Converged {
                    solution: x.as_slice().to_vec(),
                    iterations: iteration,
                    residual_norm: norm,
                };
            }

            // Check for non-finite Jacobian entries
            if j.as_slice().iter().any(|v| !v.is_finite()) {
                return SolveResult::Failed {
                    error: SolveError::NonFiniteJacobian,
                };
            }

            // Solve J * delta = -r for the Newton step
            let delta = match self.solve_linear(&j, &(-&r)) {
                Some(d) => d,
                None => {
                    return SolveResult::Failed {
                        error: SolveError::SingularJacobian,
                    };
                }
            };

            // Update solution
            x += delta;
        }

        // Did not converge within max iterations
        let norm: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

        SolveResult::NotConverged {
            solution: x.as_slice().to_vec(),
            iterations: self.config.max_iterations,
            residual_norm: norm,
            residuals: residuals.clone(),
        }
    }

    /// Solve using interpreted (non-JIT) evaluation.
    fn solve_interpreted<P: Problem>(&self, problem: &P, x0: &[f64]) -> SolveResult {
        let n = problem.variable_count();
        let m = problem.residual_count();

        let mut x = DVector::from_column_slice(x0);

        for iteration in 0..self.config.max_iterations {
            // Compute residuals
            let residuals = problem.residuals(x.as_slice());

            if let Err(error) = super::common::check_residuals_finite(&residuals) {
                return error.into();
            }

            let r = DVector::from_column_slice(&residuals);
            let norm = r.norm();

            // Check convergence
            if norm < self.config.tolerance {
                return SolveResult::Converged {
                    solution: x.as_slice().to_vec(),
                    iterations: iteration,
                    residual_norm: norm,
                };
            }

            // Compute Jacobian
            let jac_entries = problem.jacobian(x.as_slice());

            if let Err(error) = super::common::check_jacobian_finite(&jac_entries) {
                return error.into();
            }

            let mut j = DMatrix::zeros(m, n);
            for (row, col, val) in jac_entries {
                if row < m && col < n {
                    j[(row, col)] = val;
                }
            }

            // Solve J * delta = -r for the Newton step
            let delta = match self.solve_linear(&j, &(-&r)) {
                Some(d) => d,
                None => {
                    return SolveResult::Failed {
                        error: SolveError::SingularJacobian,
                    };
                }
            };

            // Update solution
            x += delta;
        }

        // Did not converge within max iterations
        let residuals = problem.residuals(x.as_slice());
        let norm: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

        SolveResult::NotConverged {
            solution: x.as_slice().to_vec(),
            iterations: self.config.max_iterations,
            residual_norm: norm,
            residuals,
        }
    }

    /// Solve the linear system J * delta = rhs.
    fn solve_linear(&self, j: &DMatrix<f64>, rhs: &DVector<f64>) -> Option<DVector<f64>> {
        let n_rows = j.nrows();
        let n_cols = j.ncols();

        if n_rows == n_cols {
            // Square system: try LU decomposition first
            if let Some(solution) = j.clone().lu().solve(rhs) {
                return Some(solution);
            }
        }

        // Rectangular or singular: use SVD-based pseudoinverse
        let svd = j.clone().svd(true, true);
        svd.solve(rhs, 1e-10).ok()
    }

    /// Compile constraints for JIT evaluation.
    ///
    /// Returns a JIT function that can be reused for multiple solves.
    pub fn compile(&mut self, constraints: &CompiledConstraints) -> Result<JITFunction, JITError> {
        // Ensure compiler is initialized
        if self.compiler.is_none() {
            self.compiler = Some(JITCompiler::new()?);
        }

        let compiler = self.compiler.as_mut().ok_or(JITError::NotAvailable)?;
        compiler.compile(constraints)
    }
}

impl Default for JITSolver {
    fn default() -> Self {
        Self::default_solver()
    }
}

/// Result of JIT compilation attempt.
#[derive(Debug)]
pub enum JITCompilationResult {
    /// Successfully compiled.
    Compiled(JITFunction),

    /// JIT is not available on this platform.
    NotAvailable,

    /// Compilation failed.
    Failed(JITError),

    /// Problem is too small for JIT to be beneficial.
    TooSmall,
}

/// Try to compile a problem for JIT evaluation.
pub fn try_compile(constraints: &CompiledConstraints, threshold: usize) -> JITCompilationResult {
    if !crate::jit::jit_available() {
        return JITCompilationResult::NotAvailable;
    }

    let estimated_work = constraints.n_residuals * 50;
    if estimated_work < threshold {
        return JITCompilationResult::TooSmall;
    }

    match JITCompiler::new() {
        Ok(mut compiler) => match compiler.compile(constraints) {
            Ok(jit_fn) => JITCompilationResult::Compiled(jit_fn),
            Err(e) => JITCompilationResult::Failed(e),
        },
        Err(e) => JITCompilationResult::Failed(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleProblem;

    impl Problem for SimpleProblem {
        fn name(&self) -> &str {
            "simple"
        }

        fn residual_count(&self) -> usize {
            1
        }

        fn variable_count(&self) -> usize {
            1
        }

        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] - 2.0]
        }

        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 2.0 * x[0])]
        }

        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![1.0 * factor]
        }
    }

    #[test]
    fn test_jit_solver_interpreted() {
        let mut solver = JITSolver::new(JITConfig::always_interpreted());
        let result = solver.solve(&SimpleProblem, &[1.5]);

        assert!(result.is_converged());
        let solution = result.solution().expect("should have solution");
        assert!(
            (solution[0] - std::f64::consts::SQRT_2).abs() < 1e-6,
            "solution should be sqrt(2), got {}",
            solution[0]
        );
    }

    #[test]
    fn test_jit_solver_config() {
        let config = JITConfig::default();
        assert_eq!(config.jit_threshold, 1000);
        assert_eq!(config.max_iterations, 200);
        assert_eq!(config.mode, JitMode::Auto);
    }

    #[test]
    fn test_will_use_jit() {
        // Small problem should not use JIT
        let solver = JITSolver::new(JITConfig::default());
        assert!(!solver.will_use_jit(&SimpleProblem));

        // Force JIT should use JIT (if available)
        let solver_jit = JITSolver::new(JITConfig::always_jit());
        assert_eq!(
            solver_jit.will_use_jit(&SimpleProblem),
            crate::jit::jit_available()
        );

        // Force interpreted should not use JIT
        let solver_interp = JITSolver::new(JITConfig::always_interpreted());
        assert!(!solver_interp.will_use_jit(&SimpleProblem));
    }

    #[test]
    fn test_jit_solver_dimension_mismatch() {
        let mut solver = JITSolver::default();
        let result = solver.solve(&SimpleProblem, &[1.0, 2.0]);

        assert!(!result.is_converged());
        assert!(!result.is_completed());
        assert!(matches!(
            result.error(),
            Some(SolveError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_should_jit_uses_total_ops() {
        let solver = JITSolver::new(JITConfig::default());

        // Small problem: few ops → should not JIT
        let mut small_cc = CompiledConstraints::new(2, 1);
        small_cc.residual_ops = vec![
            crate::jit::ConstraintOp::LoadVar {
                dst: crate::jit::Reg::new(0),
                var_idx: 0,
            },
            crate::jit::ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: crate::jit::Reg::new(0),
            },
        ];
        assert!(
            !solver.should_jit(&small_cc),
            "small problem should not use JIT"
        );

        // Large problem: many ops → should JIT
        let mut large_cc = CompiledConstraints::new(100, 100);
        // Add 200 ops to exceed threshold
        for i in 0..200 {
            large_cc
                .residual_ops
                .push(crate::jit::ConstraintOp::LoadVar {
                    dst: crate::jit::Reg::new(i as u16),
                    var_idx: i.min(99),
                });
        }
        assert!(
            solver.should_jit(&large_cc) == crate::jit::jit_available(),
            "large problem should use JIT if available"
        );
    }

    #[test]
    fn test_should_jit_force_flags() {
        let small_cc = CompiledConstraints::new(1, 1);

        // Force JIT
        let solver_jit = JITSolver::new(JITConfig::always_jit());
        assert_eq!(
            solver_jit.should_jit(&small_cc),
            crate::jit::jit_available()
        );

        // Force interpreted
        let solver_interp = JITSolver::new(JITConfig::always_interpreted());
        assert!(!solver_interp.should_jit(&small_cc));
    }
}
