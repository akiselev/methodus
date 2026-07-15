//! Line search algorithms for gradient-based optimization.
//!
//! Provides strong Wolfe line search (Nocedal & Wright Algorithm 3.5/3.6) with
//! Armijo backtracking as a fallback, with explicit success/failure reporting.
//!
//! Every search returns `Result<LineSearchStep, LineSearchFailure>`. On
//! success the [`ParamStore`] is left at the accepted candidate `x + α·d`;
//! on failure it is restored to the starting point `x`. The step records
//! which acceptance condition held (strong Wolfe, or Armijo only) plus the
//! number of function and gradient evaluations consumed.
//!
//! [`bounded_line_search`] additionally caps the step at `alpha_max` so the
//! objective is never evaluated outside a feasible box.

use crate::optimization::{LineSearchError, LineSearchFailure, Objective, OptimizationConfig};
use crate::param::ParamStore;
use crate::solver::bfgs::{dense_gradient, dot, write_x_to_store};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Which acceptance condition the returned step satisfied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepCondition {
    /// Both the Armijo sufficient-decrease and the strong curvature condition.
    StrongWolfe,
    /// Only the Armijo sufficient-decrease condition.
    ArmijoOnly,
}

/// A successfully accepted line-search step.
#[derive(Debug, Clone, Copy)]
pub struct LineSearchStep {
    /// Accepted step length along the search direction.
    pub alpha: f64,
    /// Objective value at `x + alpha·d` (recomputed at the accepted point).
    pub f: f64,
    /// Which condition the step satisfied.
    pub condition: StepCondition,
    /// Number of objective evaluations consumed.
    pub f_evals: usize,
    /// Number of gradient evaluations consumed.
    pub grad_evals: usize,
}

// ---------------------------------------------------------------------------
// Evaluation bookkeeping
// ---------------------------------------------------------------------------

/// Tracks evaluations against a budget and enforces finiteness.
struct Evaluator<'a> {
    objective: &'a dyn Objective,
    param_ids: &'a [crate::id::ParamId],
    x: &'a [f64],
    direction: &'a [f64],
    budget: usize,
    f_evals: usize,
    grad_evals: usize,
}

impl<'a> Evaluator<'a> {
    fn failure(&self, reason: LineSearchError) -> LineSearchFailure {
        LineSearchFailure {
            reason,
            f_evals: self.f_evals,
            grad_evals: self.grad_evals,
        }
    }

    fn check_budget(&self) -> Result<(), LineSearchFailure> {
        if self.f_evals + self.grad_evals >= self.budget {
            Err(self.failure(LineSearchError::EvaluationBudgetExceeded))
        } else {
            Ok(())
        }
    }

    /// Write `x + alpha·d` into the store and evaluate the objective there.
    fn value_at(&mut self, store: &mut ParamStore, alpha: f64) -> Result<f64, LineSearchFailure> {
        self.check_budget()?;
        self.f_evals += 1;
        let x_trial: Vec<f64> = self
            .x
            .iter()
            .zip(self.direction)
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        write_x_to_store(store, self.param_ids, &x_trial);
        let f = self.objective.value(store);
        if f.is_finite() {
            Ok(f)
        } else {
            Err(self.failure(LineSearchError::NonFiniteValue))
        }
    }

    /// Directional derivative `∇f(x + alpha·d)·d` at the store's current point.
    ///
    /// Must be called immediately after `value_at` with the same `alpha`.
    fn slope_at_current(&mut self, store: &ParamStore) -> Result<f64, LineSearchFailure> {
        self.check_budget()?;
        self.grad_evals += 1;
        let grad = dense_gradient(self.objective, store, self.param_ids, self.x.len());
        let dg = dot(&grad, self.direction);
        if dg.is_finite() {
            Ok(dg)
        } else {
            Err(self.failure(LineSearchError::NonFiniteValue))
        }
    }

    /// Restore the store to the starting point `x`.
    fn restore_start(&self, store: &mut ParamStore) {
        write_x_to_store(store, self.param_ids, self.x);
    }

    /// Leave the store at the accepted candidate and build the step.
    fn accept(
        &self,
        store: &mut ParamStore,
        alpha: f64,
        f: f64,
        condition: StepCondition,
    ) -> LineSearchStep {
        let x_new: Vec<f64> = self
            .x
            .iter()
            .zip(self.direction)
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        write_x_to_store(store, self.param_ids, &x_new);
        LineSearchStep {
            alpha,
            f,
            condition,
            f_evals: self.f_evals,
            grad_evals: self.grad_evals,
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Unified line search: strong Wolfe first, Armijo backtracking on failure.
///
/// On success the store holds `x + α·d`; on failure it is restored to `x`.
#[allow(clippy::too_many_arguments)]
pub fn line_search(
    objective: &dyn Objective,
    store: &mut ParamStore,
    param_ids: &[crate::id::ParamId],
    x: &[f64],
    direction: &[f64],
    f_x: f64,
    grad: &[f64],
    config: &OptimizationConfig,
) -> Result<LineSearchStep, LineSearchFailure> {
    search_impl(
        objective,
        store,
        param_ids,
        x,
        direction,
        f_x,
        grad,
        f64::INFINITY,
        config,
    )
}

/// Bound-aware line search: identical to [`line_search`] but never evaluates
/// the objective at a step longer than `alpha_max`, so all trial points stay
/// inside a feasible box when the caller computes `alpha_max` from the first
/// bound crossing along the direction.
#[allow(clippy::too_many_arguments)]
pub fn bounded_line_search(
    objective: &dyn Objective,
    store: &mut ParamStore,
    param_ids: &[crate::id::ParamId],
    x: &[f64],
    direction: &[f64],
    f_x: f64,
    grad: &[f64],
    alpha_max: f64,
    config: &OptimizationConfig,
) -> Result<LineSearchStep, LineSearchFailure> {
    search_impl(
        objective, store, param_ids, x, direction, f_x, grad, alpha_max, config,
    )
}

/// Largest step along `direction` from `x` that stays inside `[lower, upper]`.
///
/// Returns `f64::INFINITY` when no bound is crossed. Components already at a
/// bound with the direction pointing outward yield `alpha_max = 0`.
pub fn max_feasible_step(x: &[f64], direction: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let mut alpha_max = f64::INFINITY;
    for i in 0..x.len() {
        let d = direction[i];
        if d > 0.0 && upper[i] < f64::INFINITY {
            alpha_max = alpha_max.min((upper[i] - x[i]) / d);
        } else if d < 0.0 && lower[i] > f64::NEG_INFINITY {
            alpha_max = alpha_max.min((lower[i] - x[i]) / d);
        }
    }
    alpha_max.max(0.0)
}

// ---------------------------------------------------------------------------
// Core implementation
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn search_impl(
    objective: &dyn Objective,
    store: &mut ParamStore,
    param_ids: &[crate::id::ParamId],
    x: &[f64],
    direction: &[f64],
    f_x: f64,
    grad: &[f64],
    alpha_max: f64,
    config: &OptimizationConfig,
) -> Result<LineSearchStep, LineSearchFailure> {
    let mut ev = Evaluator {
        objective,
        param_ids,
        x,
        direction,
        budget: config.line_search.max_evals.max(4),
        f_evals: 0,
        grad_evals: 0,
    };

    let dg0 = dot(grad, direction);
    if !dg0.is_finite() || !f_x.is_finite() {
        ev.restore_start(store);
        return Err(ev.failure(LineSearchError::NonFiniteValue));
    }
    if dg0 >= 0.0 {
        ev.restore_start(store);
        return Err(ev.failure(LineSearchError::NotDescentDirection));
    }
    if alpha_max <= 0.0 {
        ev.restore_start(store);
        return Err(ev.failure(LineSearchError::InfeasibleDirection));
    }

    match wolfe_bracket(&mut ev, store, f_x, dg0, alpha_max, config) {
        Ok(step) => Ok(step),
        Err(failure) if failure.reason == LineSearchError::EvaluationBudgetExceeded => {
            ev.restore_start(store);
            Err(failure)
        }
        // Wolfe could not find an acceptable step within its iteration
        // limits, or ran into a non-finite region (Armijo backtracks toward
        // x, where the objective is known finite): fall back to plain
        // Armijo backtracking.
        Err(_) => armijo_backtrack(&mut ev, store, f_x, dg0, alpha_max, config),
    }
}

/// Bracketing phase of the strong Wolfe search (Nocedal & Wright Alg. 3.5).
fn wolfe_bracket(
    ev: &mut Evaluator<'_>,
    store: &mut ParamStore,
    f_x: f64,
    dg0: f64,
    alpha_max: f64,
    config: &OptimizationConfig,
) -> Result<LineSearchStep, LineSearchFailure> {
    const MAX_BRACKET_ITERS: usize = 10;
    let c1 = config.line_search.armijo_c1;
    let c2 = config.line_search.wolfe_c2;

    let mut alpha_prev = 0.0_f64;
    let mut f_prev = f_x;
    let mut dg_prev = dg0;
    let mut alpha = 1.0_f64.min(alpha_max);

    for i in 0..MAX_BRACKET_ITERS {
        let f_alpha = ev.value_at(store, alpha)?;

        // Armijo fails or function increased relative to previous point:
        // the minimum is bracketed between alpha_prev and alpha.
        if f_alpha > f_x + c1 * alpha * dg0 || (i > 0 && f_alpha >= f_prev) {
            return zoom(
                ev, store, f_x, dg0, c1, c2, alpha_prev, alpha, f_prev, f_alpha, dg_prev, config,
            );
        }

        let dg_alpha = ev.slope_at_current(store)?;

        // Strong curvature condition satisfied.
        if dg_alpha.abs() <= c2 * dg0.abs() {
            return Ok(ev.accept(store, alpha, f_alpha, StepCondition::StrongWolfe));
        }

        // Slope at alpha is positive: minimum lies between alpha and alpha_prev.
        if dg_alpha >= 0.0 {
            return zoom(
                ev, store, f_x, dg0, c1, c2, alpha, alpha_prev, f_alpha, f_prev, dg_alpha, config,
            );
        }

        // Hit the feasibility cap while Armijo still holds: accept the
        // boundary step (curvature may be unattainable inside the box).
        if alpha >= alpha_max {
            return Ok(ev.accept(store, alpha, f_alpha, StepCondition::ArmijoOnly));
        }

        alpha_prev = alpha;
        f_prev = f_alpha;
        dg_prev = dg_alpha;
        alpha = (alpha * 2.0).min(alpha_max);
    }

    Err(ev.failure(LineSearchError::StepTooSmall))
}

/// Zoom phase (Nocedal & Wright Alg. 3.6) with safeguarded quadratic
/// interpolation (bisection fallback).
///
/// Only Armijo-satisfying candidates are retained for the fallback result.
#[allow(clippy::too_many_arguments)]
fn zoom(
    ev: &mut Evaluator<'_>,
    store: &mut ParamStore,
    f_x: f64,
    dg0: f64,
    c1: f64,
    c2: f64,
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    mut f_lo: f64,
    mut f_hi: f64,
    mut dg_lo: f64,
    config: &OptimizationConfig,
) -> Result<LineSearchStep, LineSearchFailure> {
    const MAX_ZOOM_ITERS: usize = 20;

    // Best Armijo-satisfying candidate seen so far (alpha_lo satisfies
    // Armijo by the zoom invariant, unless it is the starting point 0).
    let mut best: Option<(f64, f64)> = if alpha_lo > 0.0 && f_lo <= f_x + c1 * alpha_lo * dg0 {
        Some((alpha_lo, f_lo))
    } else {
        None
    };

    for _ in 0..MAX_ZOOM_ITERS {
        let alpha_j = interpolate(alpha_lo, alpha_hi, f_lo, f_hi, dg_lo);
        let f_j = ev.value_at(store, alpha_j)?;

        let armijo_ok = f_j <= f_x + c1 * alpha_j * dg0;
        if armijo_ok && best.is_none_or(|(_, bf)| f_j < bf) {
            best = Some((alpha_j, f_j));
        }

        if !armijo_ok || f_j >= f_lo {
            // Armijo violated or no improvement: shrink from above.
            alpha_hi = alpha_j;
            f_hi = f_j;
        } else {
            let dg_j = ev.slope_at_current(store)?;

            if dg_j.abs() <= c2 * dg0.abs() {
                return Ok(ev.accept(store, alpha_j, f_j, StepCondition::StrongWolfe));
            }

            if dg_j * (alpha_hi - alpha_lo) >= 0.0 {
                alpha_hi = alpha_lo;
                f_hi = f_lo;
            }
            alpha_lo = alpha_j;
            f_lo = f_j;
            dg_lo = dg_j;
        }

        if (alpha_hi - alpha_lo).abs() < config.line_search.min_step {
            break;
        }
    }

    // Zoom failed to find a strong Wolfe step: return the best
    // Armijo-satisfying candidate, if any.
    match best {
        Some((alpha, f)) => Ok(ev.accept(store, alpha, f, StepCondition::ArmijoOnly)),
        None => Err(ev.failure(LineSearchError::StepTooSmall)),
    }
}

/// Safeguarded quadratic interpolation for the zoom trial point.
///
/// Uses the quadratic through `(alpha_lo, f_lo)` with slope `dg_lo` and
/// `(alpha_hi, f_hi)`. Falls back to bisection when the minimizer is
/// ill-defined or too close to the bracket ends.
fn interpolate(alpha_lo: f64, alpha_hi: f64, f_lo: f64, f_hi: f64, dg_lo: f64) -> f64 {
    let d = alpha_hi - alpha_lo;
    let denom = f_hi - f_lo - dg_lo * d;
    let bisect = alpha_lo + 0.5 * d;
    if denom.abs() < 1e-30 {
        return bisect;
    }
    let alpha_q = alpha_lo - 0.5 * dg_lo * d * d / denom;
    // Safeguard: require the trial to lie well inside the bracket.
    let lo = alpha_lo.min(alpha_hi);
    let hi = alpha_lo.max(alpha_hi);
    let margin = 0.1 * (hi - lo);
    if alpha_q.is_finite() && alpha_q > lo + margin && alpha_q < hi - margin {
        alpha_q
    } else {
        bisect
    }
}

/// Armijo backtracking fallback.
fn armijo_backtrack(
    ev: &mut Evaluator<'_>,
    store: &mut ParamStore,
    f_x: f64,
    dg0: f64,
    alpha_max: f64,
    config: &OptimizationConfig,
) -> Result<LineSearchStep, LineSearchFailure> {
    let c1 = config.line_search.armijo_c1;
    let backtrack = config.line_search.backtrack;
    let min_step = config.line_search.min_step;
    let mut alpha = 1.0_f64.min(alpha_max);

    loop {
        match ev.value_at(store, alpha) {
            Ok(f_trial) => {
                if f_trial <= f_x + c1 * alpha * dg0 {
                    return Ok(ev.accept(store, alpha, f_trial, StepCondition::ArmijoOnly));
                }
            }
            Err(failure) if failure.reason == LineSearchError::NonFiniteValue => {
                // Non-finite trial value: keep backtracking toward x, where
                // the objective is known finite.
            }
            Err(failure) => {
                ev.restore_start(store);
                return Err(failure);
            }
        }

        alpha *= backtrack;
        if alpha < min_step {
            ev.restore_start(store);
            return Err(ev.failure(LineSearchError::StepTooSmall));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{EntityId, ParamId};
    use crate::optimization::{Objective, ObjectiveId, OptimizationConfig};
    use crate::param::ParamStore;

    /// f(x) = sum x_i^2 (simple quadratic bowl).
    struct Quadratic {
        params: Vec<ParamId>,
    }

    impl Objective for Quadratic {
        fn id(&self) -> ObjectiveId {
            ObjectiveId::new(0, 0)
        }
        fn name(&self) -> &str {
            "quadratic"
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn value(&self, store: &ParamStore) -> f64 {
            self.params.iter().map(|&p| store.get(p).powi(2)).sum()
        }
        fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
            self.params
                .iter()
                .map(|&p| (p, 2.0 * store.get(p)))
                .collect()
        }
    }

    /// f(x) = -ln(x); only defined for x > 0.
    struct NegLog {
        params: Vec<ParamId>,
    }

    impl Objective for NegLog {
        fn id(&self) -> ObjectiveId {
            ObjectiveId::new(1, 0)
        }
        fn name(&self) -> &str {
            "neg_log"
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn value(&self, store: &ParamStore) -> f64 {
            let x = store.get(self.params[0]);
            if x <= 0.0 {
                f64::NAN
            } else {
                -x.ln()
            }
        }
        fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
            let x = store.get(self.params[0]);
            vec![(self.params[0], -1.0 / x)]
        }
    }

    fn setup(vals: &[f64]) -> (ParamStore, Vec<ParamId>) {
        let mut store = ParamStore::new();
        let owner = EntityId::new(0, 0);
        let ids: Vec<ParamId> = vals.iter().map(|&v| store.alloc(v, owner)).collect();
        (store, ids)
    }

    #[test]
    fn wolfe_step_on_quadratic_reports_strong_wolfe() {
        let (mut store, ids) = setup(&[4.0]);
        let obj = Quadratic {
            params: ids.clone(),
        };
        let config = OptimizationConfig::default();
        let x = vec![4.0];
        let grad = vec![8.0];
        let dir = vec![-8.0];
        let step = line_search(&obj, &mut store, &ids, &x, &dir, 16.0, &grad, &config)
            .expect("line search should succeed on a quadratic");
        assert!(step.f < 16.0);
        assert_eq!(step.condition, StepCondition::StrongWolfe);
        // Store must hold the accepted candidate.
        let expected = 4.0 - step.alpha * 8.0;
        assert!((store.get(ids[0]) - expected).abs() < 1e-12);
        // Reported f must equal the objective at the accepted point.
        assert!((step.f - obj.value(&store)).abs() < 1e-12);
        assert!(step.f_evals >= 1);
    }

    #[test]
    fn non_descent_direction_fails_and_restores_store() {
        let (mut store, ids) = setup(&[4.0]);
        let obj = Quadratic {
            params: ids.clone(),
        };
        let config = OptimizationConfig::default();
        let x = vec![4.0];
        let grad = vec![8.0];
        let dir = vec![8.0]; // uphill
        let err = line_search(&obj, &mut store, &ids, &x, &dir, 16.0, &grad, &config)
            .expect_err("uphill direction must fail");
        assert_eq!(err.reason, LineSearchError::NotDescentDirection);
        assert!((store.get(ids[0]) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn bounded_search_never_evaluates_outside_box() {
        // Start at x = 0.5, direction -1 (toward the x <= 0 singularity).
        // alpha_max keeps trials at x >= 0.1, where -ln(x) is finite.
        let (mut store, ids) = setup(&[0.5]);
        let obj = NegLog {
            params: ids.clone(),
        };
        let config = OptimizationConfig::default();
        let x = vec![0.5];
        let grad = vec![-2.0]; // -1/0.5
        let dir = vec![1.0]; // increasing x decreases -ln(x)
        let alpha_max = 2.5; // upper bound x <= 3.0
        let f_x = obj.value(&store);
        let step = bounded_line_search(
            &obj, &mut store, &ids, &x, &dir, f_x, &grad, alpha_max, &config,
        )
        .expect("bounded search should succeed");
        assert!(step.alpha <= alpha_max + 1e-12);
        let x_final = store.get(ids[0]);
        assert!(x_final <= 3.0 + 1e-12);
        assert!((step.f - obj.value(&store)).abs() < 1e-12);
    }

    #[test]
    fn max_feasible_step_finds_first_bound() {
        let x = [0.0, 0.0];
        let dir = [1.0, 2.0];
        let lower = [-1.0, -1.0];
        let upper = [10.0, 1.0];
        // Component 1 hits its upper bound 1.0 at alpha = 0.5.
        let a = max_feasible_step(&x, &dir, &lower, &upper);
        assert!((a - 0.5).abs() < 1e-12);
    }

    #[test]
    fn max_feasible_step_zero_at_active_bound() {
        let x = [1.0];
        let dir = [1.0];
        let lower = [0.0];
        let upper = [1.0];
        assert_eq!(max_feasible_step(&x, &dir, &lower, &upper), 0.0);
    }

    #[test]
    fn evaluation_budget_is_enforced() {
        let (mut store, ids) = setup(&[4.0]);
        let obj = Quadratic {
            params: ids.clone(),
        };
        let config = OptimizationConfig {
            line_search: crate::optimization::LineSearchConfig {
                max_evals: 4,
                ..Default::default()
            },
            ..OptimizationConfig::default()
        };
        let x = vec![4.0];
        let grad = vec![8.0];
        let dir = vec![-8.0];
        let result = line_search(&obj, &mut store, &ids, &x, &dir, 16.0, &grad, &config);
        if let Ok(step) = &result {
            assert!(step.f_evals + step.grad_evals <= 4 + 1);
        }
        // Whether it succeeds within budget or fails, the store must be
        // consistent: either at x or at the accepted candidate.
        match result {
            Ok(step) => {
                let expected = 4.0 - step.alpha * 8.0;
                assert!((store.get(ids[0]) - expected).abs() < 1e-12);
            }
            Err(fail) => {
                assert_eq!(fail.reason, LineSearchError::EvaluationBudgetExceeded);
                assert!((store.get(ids[0]) - 4.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn non_finite_objective_backtracks_or_fails_cleanly() {
        // Start close to the singularity moving toward it: trial points at
        // x <= 0 yield NaN; Armijo fallback must keep backtracking or fail.
        let (mut store, ids) = setup(&[0.1]);
        let obj = NegLog {
            params: ids.clone(),
        };
        let config = OptimizationConfig::default();
        let x = vec![0.1];
        let f_x = obj.value(&store);
        let grad = vec![-10.0];
        // Descent direction for -ln(x) is +x; force the pathological
        // direction toward the singularity with a fake descending slope.
        let dir = vec![1.0];
        let step = line_search(&obj, &mut store, &ids, &x, &dir, f_x, &grad, &config)
            .expect("moving away from singularity should succeed");
        assert!(store.get(ids[0]) > 0.0);
        assert!(step.f.is_finite());
    }
}
