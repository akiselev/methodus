//! L-BFGS-B solver for box-constrained optimization.
//!
//! Minimizes a scalar objective `f(x)` subject to box constraints
//! `lower_i ≤ x_i ≤ upper_i`, using the projected L-BFGS method.
//!
//! The algorithm:
//! 1. Project the initial point onto the feasible box.
//! 2. Compute gradient and check convergence via projected gradient norm.
//! 3. Compute an L-BFGS search direction (reusing `bfgs::lbfgs_direction`).
//! 4. Project the candidate step to stay feasible.
//! 5. Run line search on the projected step.
//! 6. Update L-BFGS history with curvature pair (s, y).

use std::collections::VecDeque;
use std::time::Instant;

use crate::optimization::{
    KktResidual, MultiplierStore, Objective, OptimizationConfig, OptimizationResult,
    OptimizationStatus,
};
use crate::param::ParamStore;
use crate::solver::bfgs::{
    dense_gradient, dot, lbfgs_direction, update_lbfgs_history, vec_norm, write_x_to_store,
};
use crate::solver::line_search;

/// L-BFGS-B solver for box-constrained optimization.
pub struct BfgsBSolver;

impl BfgsBSolver {
    /// Solve a box-constrained optimization problem.
    ///
    /// Minimizes `objective.value(store)` subject to the bounds stored in
    /// `store` for each free parameter. Returns when the projected gradient
    /// norm is below tolerance or max iterations are reached.
    pub fn solve(
        objective: &dyn Objective,
        store: &mut ParamStore,
        config: &OptimizationConfig,
    ) -> OptimizationResult {
        let start = Instant::now();
        let mapping = store.build_solver_mapping();
        let n = mapping.len();

        if n == 0 {
            let f = objective.value(store);
            return OptimizationResult {
                objective_value: f,
                status: OptimizationStatus::Converged,
                outer_iterations: 0,
                inner_iterations: 0,
                kkt_residual: KktResidual {
                    primal: 0.0,
                    dual: 0.0,
                    complementarity: 0.0,
                },
                multipliers: MultiplierStore::new(),
                constraint_violations: Vec::new(),
                duration: start.elapsed(),
            };
        }

        let param_ids = &mapping.col_to_param;

        // Extract bounds for each free parameter in solver column order.
        let lower: Vec<f64> = param_ids.iter().map(|&pid| store.bounds(pid).0).collect();
        let upper: Vec<f64> = param_ids.iter().map(|&pid| store.bounds(pid).1).collect();

        // Extract initial point and project it onto the feasible box.
        let mut x: Vec<f64> = param_ids.iter().map(|&pid| store.get(pid)).collect();
        project(&mut x, &lower, &upper);
        write_x_to_store(store, param_ids, &x);

        // L-BFGS memory.
        let m = config.lbfgs_memory;
        let mut s_history: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);
        let mut y_history: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);

        let mut f = objective.value(store);
        let mut grad = dense_gradient(objective, store, param_ids, n);

        for iter in 0..config.max_outer_iterations {
            // Convergence check via projected gradient norm.
            let pg_norm = projected_gradient_norm(&x, &grad, &lower, &upper);
            let dual_check = if config.relative_tolerance {
                pg_norm / (1.0_f64).max(f.abs())
            } else {
                pg_norm
            };
            if dual_check < config.dual_tolerance {
                write_x_to_store(store, param_ids, &x);
                return OptimizationResult {
                    objective_value: f,
                    status: OptimizationStatus::Converged,
                    outer_iterations: iter,
                    inner_iterations: 0,
                    kkt_residual: KktResidual {
                        primal: 0.0,
                        dual: pg_norm,
                        complementarity: 0.0,
                    },
                    multipliers: MultiplierStore::new(),
                    constraint_violations: Vec::new(),
                    duration: start.elapsed(),
                };
            }

            // Compute search direction via Generalized Cauchy Point + subspace minimization.
            let gamma = compute_gamma(&s_history, &y_history);
            let (x_cauchy, active_set) = generalized_cauchy_point(&x, &grad, &lower, &upper, gamma);
            let correction = subspace_minimization(
                &x_cauchy,
                &grad,
                &active_set,
                &lower,
                &upper,
                &s_history,
                &y_history,
            );
            let mut direction: Vec<f64> = x_cauchy
                .iter()
                .zip(&correction)
                .zip(&x)
                .map(|((c, d), xi)| c + d - xi)
                .collect();

            // Fallback: if GCP produces a degenerate zero direction, use
            // projected steepest descent.
            if vec_norm(&direction) < 1e-15 {
                direction = project_direction(
                    &grad.iter().map(|g| -g).collect::<Vec<_>>(),
                    &x,
                    &lower,
                    &upper,
                );
                // If projected steepest descent is also zero, we are at a
                // corner of the feasible box — converged.
                if vec_norm(&direction) < 1e-15 {
                    write_x_to_store(store, param_ids, &x);
                    return OptimizationResult {
                        objective_value: f,
                        status: OptimizationStatus::Converged,
                        outer_iterations: iter,
                        inner_iterations: 0,
                        kkt_residual: KktResidual {
                            primal: 0.0,
                            dual: pg_norm,
                            complementarity: 0.0,
                        },
                        multipliers: MultiplierStore::new(),
                        constraint_violations: Vec::new(),
                        duration: start.elapsed(),
                    };
                }
                s_history.clear();
                y_history.clear();
            }

            // Ensure descent direction.
            let dg = dot(&grad, &direction);
            if dg >= 0.0 {
                s_history.clear();
                y_history.clear();
                direction = project_direction(
                    &grad.iter().map(|g| -g).collect::<Vec<_>>(),
                    &x,
                    &lower,
                    &upper,
                );
                if vec_norm(&direction) < 1e-15 {
                    write_x_to_store(store, param_ids, &x);
                    return OptimizationResult {
                        objective_value: f,
                        status: OptimizationStatus::Converged,
                        outer_iterations: iter,
                        inner_iterations: 0,
                        kkt_residual: KktResidual {
                            primal: 0.0,
                            dual: pg_norm,
                            complementarity: 0.0,
                        },
                        multipliers: MultiplierStore::new(),
                        constraint_violations: Vec::new(),
                        duration: start.elapsed(),
                    };
                }
            }

            // Bound-aware line search: cap the step at the first bound
            // crossing so the objective is never evaluated outside the box.
            let alpha_max = line_search::max_feasible_step(&x, &direction, &lower, &upper);
            let step = match line_search::bounded_line_search(
                objective, store, param_ids, &x, &direction, f, &grad, alpha_max, config,
            ) {
                Ok(step) => step,
                Err(_) => {
                    // Store was restored to x; report the best point found.
                    write_x_to_store(store, param_ids, &x);
                    return OptimizationResult {
                        objective_value: f,
                        status: OptimizationStatus::LineSearchFailed,
                        outer_iterations: iter,
                        inner_iterations: 0,
                        kkt_residual: KktResidual {
                            primal: 0.0,
                            dual: pg_norm,
                            complementarity: 0.0,
                        },
                        multipliers: MultiplierStore::new(),
                        constraint_violations: Vec::new(),
                        duration: start.elapsed(),
                    };
                }
            };
            let (alpha, f_new) = (step.alpha, step.f);

            // Compute new iterate. The step is feasible by construction;
            // clamp only to absorb floating-point rounding at the boundary.
            let mut x_new: Vec<f64> = x
                .iter()
                .zip(&direction)
                .map(|(xi, di)| xi + alpha * di)
                .collect();
            project(&mut x_new, &lower, &upper);
            debug_assert!(
                {
                    let unclamped: Vec<f64> = x
                        .iter()
                        .zip(&direction)
                        .map(|(xi, di)| xi + alpha * di)
                        .collect();
                    unclamped
                        .iter()
                        .zip(&x_new)
                        .all(|(u, c)| (u - c).abs() < 1e-9)
                },
                "BfgsB accepted a step materially outside the feasible box"
            );

            write_x_to_store(store, param_ids, &x_new);
            debug_assert!(
                (f_new - objective.value(store)).abs()
                    <= 1e-9 * (1.0 + f_new.abs()),
                "BfgsB reported objective does not match the accepted iterate"
            );
            let grad_new = dense_gradient(objective, store, param_ids, n);

            // L-BFGS update: s = x_new - x (unprojected), y = P(g_new) - P(g_old).
            let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();

            // Project gradients: zero out components where the iterate is at a bound
            // and the gradient would push further into the bound
            let pg_new: Vec<f64> = grad_new
                .iter()
                .enumerate()
                .map(|(i, &g)| {
                    if x_new[i] <= lower[i] && g > 0.0 {
                        0.0
                    } else if x_new[i] >= upper[i] && g < 0.0 {
                        0.0
                    } else {
                        g
                    }
                })
                .collect();
            let pg_old: Vec<f64> = grad
                .iter()
                .enumerate()
                .map(|(i, &g)| {
                    if x[i] <= lower[i] && g > 0.0 {
                        0.0
                    } else if x[i] >= upper[i] && g < 0.0 {
                        0.0
                    } else {
                        g
                    }
                })
                .collect();
            let y: Vec<f64> = pg_new.iter().zip(&pg_old).map(|(a, b)| a - b).collect();

            update_lbfgs_history(&mut s_history, &mut y_history, s, y, m);

            x = x_new;
            f = f_new;
            grad = grad_new;
        }

        // Max iterations reached.
        write_x_to_store(store, param_ids, &x);
        let pg_norm = projected_gradient_norm(&x, &grad, &lower, &upper);
        OptimizationResult {
            objective_value: f,
            status: OptimizationStatus::MaxIterationsReached,
            outer_iterations: config.max_outer_iterations,
            inner_iterations: 0,
            kkt_residual: KktResidual {
                primal: 0.0,
                dual: pg_norm,
                complementarity: 0.0,
            },
            multipliers: MultiplierStore::new(),
            constraint_violations: Vec::new(),
            duration: start.elapsed(),
        }
    }
}

/// Project each component of `x` onto the interval `[lower[i], upper[i]]`.
fn project(x: &mut [f64], lower: &[f64], upper: &[f64]) {
    for i in 0..x.len() {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }
}

/// Projected gradient norm: `||P(x - g) - x||` where P is the box projection.
///
/// This is the standard convergence metric for box-constrained problems.
/// It equals zero if and only if x is a KKT point.
fn projected_gradient_norm(x: &[f64], grad: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let mut norm_sq = 0.0;
    for i in 0..x.len() {
        let pg = (x[i] - grad[i]).clamp(lower[i], upper[i]) - x[i];
        norm_sq += pg * pg;
    }
    norm_sq.sqrt()
}

/// Project a direction vector: zero out components that would push further into
/// an already-active bound.
fn project_direction(dir: &[f64], x: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    let mut d = dir.to_vec();
    for i in 0..d.len() {
        if x[i] <= lower[i] && d[i] < 0.0 {
            d[i] = 0.0;
        }
        if x[i] >= upper[i] && d[i] > 0.0 {
            d[i] = 0.0;
        }
    }
    d
}

/// Compute the scaled-identity Hessian parameter γ = yᵀy / sᵀy from L-BFGS history.
fn compute_gamma(s_history: &VecDeque<Vec<f64>>, y_history: &VecDeque<Vec<f64>>) -> f64 {
    let k = s_history.len();
    if k == 0 {
        return 1.0;
    }
    let last = k - 1;
    let sy = dot(&s_history[last], &y_history[last]);
    let yy = dot(&y_history[last], &y_history[last]);
    if sy.abs() > 1e-30 {
        yy / sy
    } else {
        1.0
    }
}

/// Generalized Cauchy Point (Byrd-Lu-Nocedal-Zhu 1995).
///
/// Finds the minimizer of the quadratic model `q(t) = f + gᵀ(x(t)-x) + γ/2 ||x(t)-x||²`
/// along the piecewise-linear path `x(t) = P[x - t·g]`.
///
/// Returns `(cauchy_point, active_set)` where `active_set[i] = true` means
/// variable `i` is at a bound at the Cauchy point.
fn generalized_cauchy_point(
    x: &[f64],
    grad: &[f64],
    lower: &[f64],
    upper: &[f64],
    gamma: f64,
) -> (Vec<f64>, Vec<bool>) {
    let n = x.len();
    let inf = f64::INFINITY;

    // Compute breakpoints: t_i where x_i(t) hits a bound along x(t) = P[x - t*g].
    // For variable i with g[i] < 0 (moves toward upper bound):  t_i = (x[i] - upper[i]) / g[i]
    // For variable i with g[i] > 0 (moves toward lower bound):  t_i = (x[i] - lower[i]) / g[i]
    let mut breakpoints: Vec<(f64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let t_i = if grad[i] < 0.0 && upper[i] < inf {
            (x[i] - upper[i]) / grad[i]
        } else if grad[i] > 0.0 && lower[i] > f64::NEG_INFINITY {
            (x[i] - lower[i]) / grad[i]
        } else {
            inf
        };
        if t_i > 1e-30 {
            breakpoints.push((t_i, i));
        }
    }
    breakpoints.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Track which variables are active (at a bound and gradient pointing out of feasible set).
    let mut active = vec![false; n];
    for i in 0..n {
        if grad[i] > 0.0 && x[i] <= lower[i] {
            active[i] = true;
        } else if grad[i] < 0.0 && x[i] >= upper[i] {
            active[i] = true;
        }
    }

    // Path derivative at t=0: fp = gᵀ d where d[i] = -g[i] for free vars.
    // fp = -sum_{free} g[i]^2
    // fpp = gamma * sum_{free} g[i]^2 = -gamma * fp
    let mut fp: f64 = -(0..n)
        .filter(|&i| !active[i])
        .map(|i| grad[i] * grad[i])
        .sum::<f64>();
    let mut fpp: f64 = -gamma * fp; // = gamma * sum g_i^2

    if fpp < 1e-30 || fp >= 0.0 {
        // Degenerate or already optimal: return x clamped.
        let mut x_c = x.to_vec();
        project(&mut x_c, lower, upper);
        return (x_c, active);
    }

    let mut t_prev = 0.0_f64;

    for &(t_i, coord) in &breakpoints {
        // Already active?  Skip.
        if active[coord] {
            continue;
        }

        // Minimum of 1D quadratic in current segment [t_prev, t_i]:
        // t* relative to segment start = -fp / fpp  (absolute: t_prev + dt_opt)
        let dt_opt = -fp / fpp;
        if dt_opt <= 1e-30 {
            break;
        }
        let t_star = t_prev + dt_opt;

        if t_star <= t_i {
            // Minimum is inside this segment.
            let mut x_c = x.to_vec();
            for i in 0..n {
                if !active[i] {
                    x_c[i] = (x[i] - t_star * grad[i]).clamp(lower[i], upper[i]);
                }
            }
            return (x_c, active);
        }

        // Advance to breakpoint: update fp and fpp for the now-active variable.
        let dt = t_i - t_prev;
        // fp advances along the segment: fp += fpp * dt (path derivative at t_i)
        fp += fpp * dt;
        // Remove coord's contribution from future derivative (it's now fixed at bound).
        let g_j = grad[coord];
        fp -= gamma * g_j * g_j; // d[coord] = -g_j, so gamma * d_j^2 = gamma * g_j^2
        fpp -= gamma * g_j * g_j;
        if fpp < 1e-30 {
            fpp = 1e-30;
        }

        active[coord] = true;
        t_prev = t_i;
    }

    // Minimum is beyond all breakpoints (or no breakpoints).
    // Use t* from remaining free variables.
    let mut x_c = x.to_vec();
    if fpp > 1e-30 && fp < 0.0 {
        let dt_opt = -fp / fpp;
        let t_abs = t_prev + dt_opt;
        for i in 0..n {
            if !active[i] {
                x_c[i] = (x[i] - t_abs * grad[i]).clamp(lower[i], upper[i]);
            }
        }
    } else {
        // No further improvement; use last breakpoint position.
        for i in 0..n {
            if !active[i] {
                x_c[i] = (x[i] - t_prev * grad[i]).clamp(lower[i], upper[i]);
            }
        }
    }

    project(&mut x_c, lower, upper);
    (x_c, active)
}

/// Subspace minimization: refine the Cauchy point using L-BFGS restricted to
/// the free variables (those not at bounds at the Cauchy point).
///
/// Returns a correction vector (same size as x); active-set components are zero.
fn subspace_minimization(
    x_cauchy: &[f64],
    grad: &[f64],
    active_set: &[bool],
    lower: &[f64],
    upper: &[f64],
    s_history: &VecDeque<Vec<f64>>,
    y_history: &VecDeque<Vec<f64>>,
) -> Vec<f64> {
    let n = x_cauchy.len();
    let free_indices: Vec<usize> = (0..n).filter(|&i| !active_set[i]).collect();
    let nf = free_indices.len();

    if nf == 0 || s_history.is_empty() {
        return vec![0.0; n];
    }

    // Gradient at x_cauchy restricted to free variables.
    let grad_free: Vec<f64> = free_indices.iter().map(|&i| grad[i]).collect();

    // Run L-BFGS two-loop recursion on the free subspace.
    // Build reduced s/y histories.
    let s_reduced: VecDeque<Vec<f64>> = s_history
        .iter()
        .map(|s| free_indices.iter().map(|&i| s[i]).collect())
        .collect();
    let y_reduced: VecDeque<Vec<f64>> = y_history
        .iter()
        .map(|y| free_indices.iter().map(|&i| y[i]).collect())
        .collect();

    let dir_free = lbfgs_direction(&grad_free, &s_reduced, &y_reduced);

    // Embed free-variable result back to full space.
    let mut correction = vec![0.0; n];
    for (k, &i) in free_indices.iter().enumerate() {
        // correction[i] = x_cauchy[i] + dir_free[k] - x_cauchy[i] = dir_free[k]
        // Clamp to stay inside bounds.
        let x_new = (x_cauchy[i] + dir_free[k]).clamp(lower[i], upper[i]);
        correction[i] = x_new - x_cauchy[i];
    }

    correction
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{EntityId, ParamId};
    use crate::optimization::{Objective, ObjectiveId, OptimizationStatus};

    /// f(x) = -ln(x0) + x0; undefined (NaN) for x0 <= 0, minimum at x0 = 1.
    struct LogBarrier {
        params: Vec<ParamId>,
    }

    impl Objective for LogBarrier {
        fn id(&self) -> ObjectiveId {
            ObjectiveId::new(0, 0)
        }
        fn name(&self) -> &str {
            "log_barrier"
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn value(&self, store: &ParamStore) -> f64 {
            let x = store.get(self.params[0]);
            if x <= 0.0 {
                f64::NAN
            } else {
                -x.ln() + x
            }
        }
        fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
            let x = store.get(self.params[0]);
            vec![(self.params[0], -1.0 / x + 1.0)]
        }
    }

    /// f(x, y) = (x + 2)^2 + (y + 2)^2; unconstrained minimum at (-2, -2).
    struct ShiftedBowl {
        params: Vec<ParamId>,
    }

    impl Objective for ShiftedBowl {
        fn id(&self) -> ObjectiveId {
            ObjectiveId::new(1, 0)
        }
        fn name(&self) -> &str {
            "shifted_bowl"
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn value(&self, store: &ParamStore) -> f64 {
            self.params
                .iter()
                .map(|&p| (store.get(p) + 2.0).powi(2))
                .sum()
        }
        fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
            self.params
                .iter()
                .map(|&p| (p, 2.0 * (store.get(p) + 2.0)))
                .collect()
        }
    }

    #[test]
    fn objective_undefined_outside_bounds_never_evaluated_there() {
        // min -ln(x) + x for x in [0.05, 10]; solution x = 1. Any trial
        // point at x <= 0 would return NaN and fail the solve.
        let mut store = ParamStore::new();
        let owner = EntityId::new(0, 0);
        let p = store.alloc(0.1, owner);
        store.set_bounds(p, 0.05, 10.0);

        let obj = LogBarrier { params: vec![p] };
        let config = OptimizationConfig::default();
        let result = BfgsBSolver::solve(&obj, &mut store, &config);

        assert_eq!(result.status, OptimizationStatus::Converged);
        assert!(
            (store.get(p) - 1.0).abs() < 1e-4,
            "expected x = 1, got {}",
            store.get(p)
        );
        assert!(result.objective_value.is_finite());
    }

    #[test]
    fn corner_solution_with_multiple_active_bounds() {
        // Bowl centered at (-2, -2), box [0, 5] x [0, 5]: solution is the
        // corner (0, 0) with both bounds simultaneously active.
        let mut store = ParamStore::new();
        let owner = EntityId::new(0, 0);
        let px = store.alloc(3.0, owner);
        let py = store.alloc(4.0, owner);
        store.set_bounds(px, 0.0, 5.0);
        store.set_bounds(py, 0.0, 5.0);

        let obj = ShiftedBowl {
            params: vec![px, py],
        };
        let config = OptimizationConfig::default();
        let result = BfgsBSolver::solve(&obj, &mut store, &config);

        assert_eq!(result.status, OptimizationStatus::Converged);
        assert!(store.get(px).abs() < 1e-6, "px = {}", store.get(px));
        assert!(store.get(py).abs() < 1e-6, "py = {}", store.get(py));
        // Reported objective must match the returned iterate.
        assert!((result.objective_value - obj.value(&store)).abs() < 1e-9);
    }

    #[test]
    fn boundary_solution_single_active_bound() {
        // Bowl centered at (-2, -2), box [0, 5] x [-5, 5]: solution (0, -2)
        // with only the x lower bound active.
        let mut store = ParamStore::new();
        let owner = EntityId::new(0, 0);
        let px = store.alloc(3.0, owner);
        let py = store.alloc(4.0, owner);
        store.set_bounds(px, 0.0, 5.0);
        store.set_bounds(py, -5.0, 5.0);

        let obj = ShiftedBowl {
            params: vec![px, py],
        };
        let config = OptimizationConfig::default();
        let result = BfgsBSolver::solve(&obj, &mut store, &config);

        assert_eq!(result.status, OptimizationStatus::Converged);
        assert!(store.get(px).abs() < 1e-6, "px = {}", store.get(px));
        assert!(
            (store.get(py) + 2.0).abs() < 1e-4,
            "py = {}",
            store.get(py)
        );
    }
}
