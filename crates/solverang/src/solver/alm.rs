//! Augmented Lagrangian Method (ALM) for constrained optimization.
//!
//! Solves `min f(x) s.t. g(x) = 0, h(x) ≤ 0` by minimizing a sequence of
//! augmented Lagrangian subproblems with an inner quasi-Newton solver
//! (L-BFGS, or L-BFGS-B when free parameters carry finite bounds).
//!
//! The augmented Lagrangian is:
//!
//! ```text
//! L_A(x, λ, μ, ρ) = f(x) + λᵀg(x) + (ρ/2)‖g(x)‖²
//!                 + Σ_j (ρ/2)[max(0, h_j(x) + μ_j/ρ)² − (μ_j/ρ)²]
//! ```
//!
//! The outer loop updates equality multipliers `λ ← λ + ρ g(x)`, inequality
//! multipliers `μ ← max(0, μ + ρ h(x))` (kept nonnegative for dual
//! feasibility), and grows the penalty ρ when the combined equality +
//! inequality violation does not shrink fast enough.
//!
//! Convergence requires all KKT components: primal feasibility, stationarity
//! of the original Lagrangian `∇f + J_gᵀλ + J_hᵀμ`, dual feasibility
//! (`μ ≥ 0`, enforced by construction), and complementarity `|μ_j h_j| ≈ 0`.
//! Contradictory constraints are detected as a saturated penalty with
//! stagnant violation and reported as `Infeasible` rather than a generic
//! iteration limit.

use std::time::Instant;

use crate::constraint::Constraint;
use crate::optimization::{
    InequalityFn, KktResidual, MultiplierId, MultiplierInitStrategy, MultiplierStore, Objective,
    OptimizationConfig, OptimizationResult, OptimizationStatus,
};
use crate::param::ParamStore;

fn compute_complementarity(
    inequalities: &[&dyn InequalityFn],
    mu: &[f64],
    store: &ParamStore,
) -> f64 {
    let mut comp = 0.0_f64;
    let mut idx = 0;
    for h in inequalities {
        for v in h.values(store) {
            comp = comp.max((mu[idx] * v).abs());
            idx += 1;
        }
    }
    comp
}

/// Equality residuals followed by inequality values, in registration order.
fn collect_violations(
    constraints: &[&dyn Constraint],
    inequalities: &[&dyn InequalityFn],
    store: &ParamStore,
) -> Vec<f64> {
    let mut v: Vec<f64> = constraints
        .iter()
        .flat_map(|c| c.residuals(store))
        .collect();
    v.extend(inequalities.iter().flat_map(|h| h.values(store)));
    v
}

/// Combined primal violation: `sqrt(‖g‖² + ‖max(0, h)‖²)`.
fn combined_violation(
    constraints: &[&dyn Constraint],
    inequalities: &[&dyn InequalityFn],
    store: &ParamStore,
) -> f64 {
    let eq_sq: f64 = constraints
        .iter()
        .flat_map(|c| c.residuals(store))
        .map(|r| r * r)
        .sum();
    let ineq_sq: f64 = inequalities
        .iter()
        .flat_map(|h| h.values(store))
        .map(|v| v.max(0.0).powi(2))
        .sum();
    (eq_sq + ineq_sq).sqrt()
}

/// Stationarity of the original Lagrangian: `‖∇f + J_gᵀλ + J_hᵀμ‖`.
fn lagrangian_stationarity(
    objective: &dyn Objective,
    constraints: &[&dyn Constraint],
    inequalities: &[&dyn InequalityFn],
    lambda: &[f64],
    mu: &[f64],
    param_ids: &[crate::id::ParamId],
    store: &ParamStore,
) -> f64 {
    let n = param_ids.len();
    let mut grad_l = vec![0.0; n];
    for (pid, val) in objective.gradient(store) {
        if let Some(col) = param_ids.iter().position(|&p| p == pid) {
            grad_l[col] = val;
        }
    }
    let mut eq_offset = 0;
    for c in constraints {
        for (row, pid, val) in c.jacobian(store) {
            if let Some(col) = param_ids.iter().position(|&p| p == pid) {
                grad_l[col] += lambda[eq_offset + row] * val;
            }
        }
        eq_offset += c.equation_count();
    }
    let mut ineq_offset = 0;
    for h in inequalities {
        for (row, pid, val) in h.jacobian(store) {
            if let Some(col) = param_ids.iter().position(|&p| p == pid) {
                grad_l[col] += mu[ineq_offset + row] * val;
            }
        }
        ineq_offset += h.inequality_count();
    }
    grad_l.iter().map(|g| g * g).sum::<f64>().sqrt()
}

fn build_multiplier_store(
    constraints: &[&dyn Constraint],
    inequalities: &[&dyn InequalityFn],
    lambda: &[f64],
    mu: &[f64],
) -> MultiplierStore {
    let mut store = MultiplierStore::new();
    let mut eq_idx = 0;
    for c in constraints {
        for row in 0..c.equation_count() {
            store.set(MultiplierId::new(c.id(), row), lambda[eq_idx]);
            eq_idx += 1;
        }
    }
    let mut ineq_idx = 0;
    for h in inequalities {
        for row in 0..h.inequality_count() {
            store.set(MultiplierId::new(h.id(), row), mu[ineq_idx]);
            ineq_idx += 1;
        }
    }
    store
}

/// Augmented Lagrangian Method solver.
pub struct AlmSolver;

impl AlmSolver {
    /// Solve a constrained optimization problem.
    ///
    /// Minimizes `objective.value(store)` subject to equality constraints
    /// (existing `Constraint` objects where residuals should be zero) and
    /// inequality constraints `h(x) ≤ 0`.
    ///
    /// # Algorithm
    ///
    /// 1. Minimize the augmented Lagrangian with L-BFGS (or L-BFGS-B when
    ///    free parameters have finite bounds).
    /// 2. Update multipliers: `λ ← λ + ρ g(x)`, `μ ← max(0, μ + ρ h(x))`.
    /// 3. Grow ρ if the combined equality + inequality violation stagnates.
    /// 4. Check the full KKT system (primal, stationarity, complementarity;
    ///    dual feasibility `μ ≥ 0` holds by construction).
    /// 5. Report `Infeasible` when ρ saturates without feasibility progress,
    ///    and propagate inner-solver failures instead of iterating blindly.
    pub fn solve(
        objective: &dyn Objective,
        constraints: &[&dyn Constraint],
        inequalities: &[&dyn InequalityFn],
        store: &mut ParamStore,
        config: &OptimizationConfig,
        warm_start: Option<&MultiplierStore>,
    ) -> OptimizationResult {
        let start = Instant::now();
        let mapping = store.build_solver_mapping();
        let n = mapping.len();

        // Count total equality / inequality equations.
        let total_eq: usize = constraints.iter().map(|c| c.equation_count()).sum();
        let total_ineq: usize = inequalities.iter().map(|h| h.inequality_count()).sum();

        if n == 0 {
            // No free variables: nothing to optimize, but the fixed point
            // must still satisfy the constraints to count as converged.
            let violations = collect_violations(constraints, inequalities, store);
            let primal = combined_violation(constraints, inequalities, store);
            let status = if primal <= config.outer_tolerance {
                OptimizationStatus::Converged
            } else {
                OptimizationStatus::Infeasible
            };
            return OptimizationResult {
                objective_value: objective.value(store),
                status,
                outer_iterations: 0,
                inner_iterations: 0,
                kkt_residual: KktResidual {
                    primal,
                    dual: 0.0,
                    complementarity: 0.0,
                },
                multipliers: MultiplierStore::new(),
                constraint_violations: violations,
                duration: start.elapsed(),
            };
        }

        let param_ids = mapping.col_to_param.clone();

        // Dispatch inner loop to BFGS-B if any free parameter has finite bounds.
        let use_bfgs_b = store.any_free_finite_bounds();

        // Initialize multipliers (warm-start if requested and available).
        let mut lambda = match (&config.multiplier_init, &warm_start) {
            (MultiplierInitStrategy::WarmStart, Some(ms)) => ms.extract_equality_vec(constraints),
            _ => vec![0.0; total_eq],
        };
        let mut mu = match (&config.multiplier_init, &warm_start) {
            (MultiplierInitStrategy::WarmStart, Some(ms)) => {
                ms.extract_inequality_vec(inequalities)
            }
            _ => vec![0.0; total_ineq],
        };
        // Dual feasibility requires μ ≥ 0: clamp warm-started (or otherwise
        // invalid) inequality multipliers.
        for m in &mut mu {
            *m = m.max(0.0).min(config.max_multiplier);
        }
        for l in &mut lambda {
            *l = l.clamp(-config.max_multiplier, config.max_multiplier);
        }

        let mut rho = config.rho_init;
        let mut prev_violation = f64::INFINITY;
        let mut stalled_outers = 0usize;
        let mut total_inner_iters = 0;

        for outer_iter in 0..config.max_outer_iterations {
            // Build the augmented Lagrangian as an Objective for the inner solver.
            let alm_objective = AugmentedLagrangianObjective {
                objective,
                constraints,
                inequalities,
                param_ids: &param_ids,
                lambda: &lambda,
                mu: &mu,
                rho,
            };

            let inner_config = OptimizationConfig {
                max_outer_iterations: config.max_inner_iterations,
                dual_tolerance: config.inner_tolerance,
                ..config.clone()
            };
            let inner_result = if use_bfgs_b {
                super::bfgs_b::BfgsBSolver::solve(&alm_objective, store, &inner_config)
            } else {
                super::bfgs::BfgsSolver::solve(&alm_objective, store, &inner_config)
            };

            total_inner_iters += inner_result.outer_iterations;

            // Propagate hard inner-solver failures instead of continuing to
            // update multipliers against a bogus iterate. An inner iteration
            // limit is expected (subproblems are solved loosely early on),
            // and a line-search failure still leaves the store at the best
            // point found (handled by the stall counter below); divergence
            // or non-finite values are fatal.
            if matches!(inner_result.status, OptimizationStatus::Diverged)
                || !inner_result.objective_value.is_finite()
            {
                return OptimizationResult {
                    objective_value: objective.value(store),
                    status: OptimizationStatus::Diverged,
                    outer_iterations: outer_iter + 1,
                    inner_iterations: total_inner_iters,
                    kkt_residual: KktResidual {
                        primal: combined_violation(constraints, inequalities, store),
                        dual: f64::INFINITY,
                        complementarity: compute_complementarity(inequalities, &mu, store),
                    },
                    multipliers: build_multiplier_store(constraints, inequalities, &lambda, &mu),
                    constraint_violations: collect_violations(constraints, inequalities, store),
                    duration: start.elapsed(),
                };
            }
            let inner_line_search_failed =
                matches!(inner_result.status, OptimizationStatus::LineSearchFailed);

            // First-order multiplier updates (Hestenes-Powell):
            //   λ ← λ + ρ g(x),  μ ← max(0, μ + ρ h(x)).
            // The updated multipliers are also the ones for which the inner
            // solve's stationary point satisfies ∇f + J_gᵀλ + J_hᵀμ ≈ 0, so
            // KKT stationarity is checked against them below.
            let mut lambda_new = lambda.clone();
            let mut eq_idx = 0;
            for c in constraints {
                for r in c.residuals(store) {
                    lambda_new[eq_idx] = (lambda_new[eq_idx] + rho * r)
                        .clamp(-config.max_multiplier, config.max_multiplier);
                    eq_idx += 1;
                }
            }
            let mut mu_new = mu.clone();
            let mut ineq_idx = 0;
            for h in inequalities {
                for v in h.values(store) {
                    mu_new[ineq_idx] = (mu_new[ineq_idx] + rho * v)
                        .max(0.0)
                        .min(config.max_multiplier);
                    ineq_idx += 1;
                }
            }

            // --- Full KKT assessment at the current iterate ---

            // Primal feasibility (combined equality + inequality violation).
            let violation_norm = combined_violation(constraints, inequalities, store);

            // Stationarity of the original Lagrangian with the updated
            // multipliers.
            let dual_norm = lagrangian_stationarity(
                objective,
                constraints,
                inequalities,
                &lambda_new,
                &mu_new,
                &param_ids,
                store,
            );

            // Complementarity with the updated (dual-feasible) multipliers.
            let complementarity = compute_complementarity(inequalities, &mu_new, store);

            let (primal_check, dual_check, comp_check) = if config.relative_tolerance {
                let total_constraints = total_eq + total_ineq;
                let primal_scale = (1.0_f64).max((total_constraints as f64).sqrt());
                let dual_scale = (1.0_f64).max((n as f64).sqrt());
                (
                    violation_norm / primal_scale,
                    dual_norm / dual_scale,
                    complementarity,
                )
            } else {
                (violation_norm, dual_norm, complementarity)
            };
            if primal_check < config.outer_tolerance
                && dual_check < config.dual_tolerance
                && comp_check < config.dual_tolerance
            {
                return OptimizationResult {
                    objective_value: objective.value(store),
                    status: OptimizationStatus::Converged,
                    outer_iterations: outer_iter + 1,
                    inner_iterations: total_inner_iters,
                    kkt_residual: KktResidual {
                        primal: violation_norm,
                        dual: dual_norm,
                        complementarity,
                    },
                    multipliers: build_multiplier_store(
                        constraints,
                        inequalities,
                        &lambda_new,
                        &mu_new,
                    ),
                    constraint_violations: collect_violations(constraints, inequalities, store),
                    duration: start.elapsed(),
                };
            }

            // Infeasibility / stall detection: penalty saturated (or the
            // inner solver can no longer make progress) while the combined
            // violation refuses to shrink toward tolerance.
            if (rho >= config.rho_max || inner_line_search_failed)
                && violation_norm > config.outer_tolerance
            {
                if violation_norm > 0.9 * prev_violation {
                    stalled_outers += 1;
                } else {
                    stalled_outers = 0;
                }
                if stalled_outers >= 3 {
                    // A saturated penalty with materially non-zero violation
                    // is evidence of contradictory constraints; a stall at
                    // small violation (or before ρ saturates) is a stall.
                    let status = if rho >= config.rho_max
                        && violation_norm > config.outer_tolerance.max(1e-4)
                    {
                        OptimizationStatus::Infeasible
                    } else {
                        OptimizationStatus::Stalled
                    };
                    return OptimizationResult {
                        objective_value: objective.value(store),
                        status,
                        outer_iterations: outer_iter + 1,
                        inner_iterations: total_inner_iters,
                        kkt_residual: KktResidual {
                            primal: violation_norm,
                            dual: dual_norm,
                            complementarity,
                        },
                        multipliers: build_multiplier_store(
                            constraints,
                            inequalities,
                            &lambda_new,
                            &mu_new,
                        ),
                        constraint_violations: collect_violations(
                            constraints,
                            inequalities,
                            store,
                        ),
                        duration: start.elapsed(),
                    };
                }
            }

            // Commit the multiplier updates.
            lambda = lambda_new;
            mu = mu_new;

            // Increase penalty if the combined violation didn't decrease
            // enough. Using the combined measure means inequality-only
            // problems also drive ρ growth.
            if violation_norm > 0.25 * prev_violation {
                rho = (rho * config.rho_growth).min(config.rho_max);
            }
            prev_violation = violation_norm;
        }

        // Max outer iterations.
        let violation_norm = combined_violation(constraints, inequalities, store);
        let dual_norm = lagrangian_stationarity(
            objective,
            constraints,
            inequalities,
            &lambda,
            &mu,
            &param_ids,
            store,
        );
        let complementarity = compute_complementarity(inequalities, &mu, store);

        OptimizationResult {
            objective_value: objective.value(store),
            status: OptimizationStatus::MaxIterationsReached,
            outer_iterations: config.max_outer_iterations,
            inner_iterations: total_inner_iters,
            kkt_residual: KktResidual {
                primal: violation_norm,
                dual: dual_norm,
                complementarity,
            },
            multipliers: build_multiplier_store(constraints, inequalities, &lambda, &mu),
            constraint_violations: collect_violations(constraints, inequalities, store),
            duration: start.elapsed(),
        }
    }
}

// ---------------------------------------------------------------------------
// Augmented Lagrangian as an Objective (for BFGS inner loop)
// ---------------------------------------------------------------------------

/// Wraps an objective + equality + inequality constraints as a scalar Objective.
///
/// L_A(x) = f(x) + λ^T g(x) + (ρ/2) ||g(x)||^2
///         + Σ_ineq (ρ/2) [max(0, h_j + μ_j/ρ)² - (μ_j/ρ)²]
///
/// The gradient is:
/// ∇L_A = ∇f + J_g^T (λ + ρ g(x)) + J_h^T max(0, μ + ρ h(x))
///
/// BFGS minimizes this using only value + gradient — no Hessian needed.
struct AugmentedLagrangianObjective<'a> {
    objective: &'a dyn Objective,
    constraints: &'a [&'a dyn Constraint],
    inequalities: &'a [&'a dyn InequalityFn],
    param_ids: &'a [crate::id::ParamId],
    lambda: &'a [f64],
    mu: &'a [f64],
    rho: f64,
}

impl Objective for AugmentedLagrangianObjective<'_> {
    fn id(&self) -> crate::optimization::ObjectiveId {
        crate::optimization::ObjectiveId::new(u32::MAX, 0)
    }

    fn name(&self) -> &str {
        "augmented_lagrangian"
    }

    fn param_ids(&self) -> &[crate::id::ParamId] {
        self.param_ids
    }

    fn value(&self, store: &ParamStore) -> f64 {
        let mut val = self.objective.value(store);

        let mut eq_offset = 0;
        for c in self.constraints {
            let resid = c.residuals(store);
            for (row, ri) in resid.iter().enumerate() {
                // λ^T g + (ρ/2) ||g||^2
                val += self.lambda[eq_offset + row] * ri + 0.5 * self.rho * ri * ri;
            }
            eq_offset += resid.len();
        }

        // Inequality terms: (ρ/2) [max(0, h_j + μ_j/ρ)² - (μ_j/ρ)²]
        let mut ineq_offset = 0;
        for h in self.inequalities {
            let vals = h.values(store);
            for (row, hi) in vals.iter().enumerate() {
                let mu_over_rho = self.mu[ineq_offset + row] / self.rho;
                let shifted = hi + mu_over_rho;
                if shifted > 0.0 {
                    val += 0.5 * self.rho * shifted * shifted;
                }
                val -= 0.5 * self.rho * mu_over_rho * mu_over_rho;
            }
            ineq_offset += vals.len();
        }

        val
    }

    fn gradient(&self, store: &ParamStore) -> Vec<(crate::id::ParamId, f64)> {
        let n = self.param_ids.len();

        // Start with objective gradient
        let obj_grad = self.objective.gradient(store);
        let mut grad = vec![0.0; n];
        for (pid, val) in obj_grad {
            if let Some(col) = self.param_ids.iter().position(|&p| p == pid) {
                grad[col] = val;
            }
        }

        // Add J_g^T (λ + ρ g(x))
        let mut eq_offset = 0;
        for c in self.constraints {
            let resid = c.residuals(store);
            let jac = c.jacobian(store);
            for (row, pid, val) in jac {
                if let Some(col) = self.param_ids.iter().position(|&p| p == pid) {
                    let dual_coeff = self.lambda[eq_offset + row] + self.rho * resid[row];
                    grad[col] += dual_coeff * val;
                }
            }
            eq_offset += resid.len();
        }

        // Add J_h^T * max(0, μ + ρ·h(x))
        let mut ineq_offset = 0;
        for h in self.inequalities {
            let vals = h.values(store);
            let jac = h.jacobian(store);
            for (row, pid, val) in jac {
                if let Some(col) = self.param_ids.iter().position(|&p| p == pid) {
                    let shifted = self.mu[ineq_offset + row] + self.rho * vals[row];
                    if shifted > 0.0 {
                        grad[col] += shifted * val;
                    }
                }
            }
            ineq_offset += vals.len();
        }

        // Return sparse (only non-zero entries)
        grad.into_iter()
            .enumerate()
            .filter(|(_, v)| v.abs() > 1e-30)
            .map(|(i, v)| (self.param_ids[i], v))
            .collect()
    }
}
