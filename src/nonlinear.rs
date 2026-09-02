use serde::{Deserialize, Serialize};

use crate::{
    BlockLayout, BlockNonlinearOperator, EvaluationContext, NonlinearOperator, NumericError,
    SolveError,
};

/// Coupling policy for a block nonlinear problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockStrategy {
    Monolithic,
    GaussSeidel,
    Jacobi,
}

/// Controls Newton updates and backtracking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewtonConfig {
    pub max_iterations: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub initial_damping: f64,
    pub minimum_damping: f64,
    pub max_line_search_steps: usize,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-8,
            initial_damping: 1.0,
            minimum_damping: 1.0e-4,
            max_line_search_steps: 12,
        }
    }
}

/// One residual evaluation recorded by a nonlinear solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IterationTrace {
    pub iteration: usize,
    pub residual_norm: f64,
    pub scaled_residual_norm: f64,
    pub block_residual_norms: Vec<(String, f64)>,
    pub accepted_damping: Option<f64>,
}

/// Final state and convergence evidence from a nonlinear solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolveReport {
    pub state: Vec<f64>,
    pub converged: bool,
    pub trace: Vec<IterationTrace>,
}

/// A nonlinear solve strategy that time integrators and other drivers can
/// be handed instead of being bound to one algorithm.
///
/// [`DenseNewton`] wraps [`solve_newton`]; [`crate::NewtonKrylovSolver`]
/// wraps [`crate::solve_newton_krylov`]. `bdf_step_with` consumes either.
pub trait NonlinearSolver: Send + Sync {
    /// Solves `F(x) = 0` from `initial_state`.
    ///
    /// # Errors
    /// Propagates the wrapped algorithm's refusals and failures unchanged.
    fn solve(
        &self,
        operator: &dyn NonlinearOperator,
        context: &EvaluationContext,
        initial_state: &[f64],
    ) -> Result<SolveReport, SolveError>;
}

/// [`solve_newton`] as a [`NonlinearSolver`].
#[derive(Clone, Debug, PartialEq)]
pub struct DenseNewton<'a> {
    config: &'a NewtonConfig,
}

impl<'a> DenseNewton<'a> {
    #[must_use]
    pub const fn new(config: &'a NewtonConfig) -> Self {
        Self { config }
    }
}

impl NonlinearSolver for DenseNewton<'_> {
    fn solve(
        &self,
        operator: &dyn NonlinearOperator,
        context: &EvaluationContext,
        initial_state: &[f64],
    ) -> Result<SolveReport, SolveError> {
        solve_newton(operator, context, initial_state, self.config)
    }
}

/// Solve an unpartitioned nonlinear system with dense Newton updates.
pub fn solve_newton(
    operator: &(impl NonlinearOperator + ?Sized),
    context: &EvaluationContext,
    initial_state: &[f64],
    config: &NewtonConfig,
) -> Result<SolveReport, SolveError> {
    solve(
        operator,
        context,
        initial_state,
        config,
        None,
        BlockStrategy::Monolithic,
    )
}

/// Solve a partitioned nonlinear system with a selected coupling policy.
pub fn solve_blocks(
    operator: &(impl BlockNonlinearOperator + ?Sized),
    context: &EvaluationContext,
    initial_state: &[f64],
    strategy: BlockStrategy,
    config: &NewtonConfig,
) -> Result<SolveReport, SolveError> {
    if operator.dimension() != operator.block_layout().dimension() {
        return Err(SolveError::InvalidLayout {
            reason: format!(
                "operator dimension {} differs from layout dimension {}",
                operator.dimension(),
                operator.block_layout().dimension()
            ),
        });
    }
    solve(
        operator,
        context,
        initial_state,
        config,
        Some(operator.block_layout()),
        strategy,
    )
}

fn solve(
    operator: &(impl NonlinearOperator + ?Sized),
    context: &EvaluationContext,
    initial_state: &[f64],
    config: &NewtonConfig,
    layout: Option<&BlockLayout>,
    strategy: BlockStrategy,
) -> Result<SolveReport, SolveError> {
    validate_config(config)?;
    let dimension = operator.dimension();
    NumericError::require_len("initial nonlinear state", initial_state.len(), dimension)?;
    NumericError::require_finite("initial nonlinear state", initial_state)?;

    let mut state = initial_state.to_vec();
    let mut residual = vec![0.0; dimension];
    operator.residual(context, &state, &mut residual)?;
    NumericError::require_finite("initial residual", &residual)?;
    let initial_norm = scaled_norm(layout, &residual);
    let convergence_threshold =
        config.absolute_tolerance + config.relative_tolerance * initial_norm;
    let trace_capacity =
        config
            .max_iterations
            .checked_add(1)
            .ok_or_else(|| SolveError::InvalidConfiguration {
                reason: "Newton iteration trace capacity overflows usize".into(),
            })?;
    let mut trace = Vec::with_capacity(trace_capacity);

    for iteration in 0..=config.max_iterations {
        operator.residual(context, &state, &mut residual)?;
        NumericError::require_finite("nonlinear residual", &residual)?;
        let scaled_residual_norm = scaled_norm(layout, &residual);
        let residual_norm = l2(&residual);
        let mut current_trace = IterationTrace {
            iteration,
            residual_norm,
            scaled_residual_norm,
            block_residual_norms: block_norms(layout, &residual),
            accepted_damping: None,
        };

        if scaled_residual_norm <= convergence_threshold {
            trace.push(current_trace);
            return Ok(SolveReport {
                state,
                converged: true,
                trace,
            });
        }
        if iteration == config.max_iterations {
            trace.push(current_trace);
            return Ok(SolveReport {
                state,
                converged: false,
                trace,
            });
        }

        let update = match (strategy, layout) {
            (BlockStrategy::Monolithic, _) => {
                dense_newton_update(operator, context, &state, &residual)?
            }
            (BlockStrategy::GaussSeidel, Some(layout)) => {
                staggered_update(operator, context, &state, layout, false)?
            }
            (BlockStrategy::Jacobi, Some(layout)) => {
                staggered_update(operator, context, &state, layout, true)?
            }
            (_, None) => unreachable!("unpartitioned solves are always monolithic"),
        };
        NumericError::require_finite("Newton update", &update)?;

        let (next_state, damping) = backtrack(
            operator,
            context,
            &state,
            &update,
            scaled_residual_norm,
            layout,
            config,
        )?;
        current_trace.accepted_damping = Some(damping);
        trace.push(current_trace);
        state = next_state;
    }
    unreachable!("iteration loop always returns")
}

fn validate_config(config: &NewtonConfig) -> Result<(), SolveError> {
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
    if config.max_iterations == 0
        || config.max_iterations.checked_add(1).is_none()
        || config.max_line_search_steps == 0
        || !tolerances_valid
        || !damping_valid
    {
        return Err(SolveError::InvalidConfiguration {
            reason: "Newton limits, tolerances, and damping must be positive and finite".into(),
        });
    }
    Ok(())
}

fn dense_newton_update(
    operator: &(impl NonlinearOperator + ?Sized),
    context: &EvaluationContext,
    state: &[f64],
    residual: &[f64],
) -> Result<Vec<f64>, SolveError> {
    let dimension = state.len();
    let mut jacobian = vec![vec![0.0; dimension]; dimension];
    let mut direction = vec![0.0; dimension];
    let mut column = vec![0.0; dimension];
    for column_index in 0..dimension {
        direction[column_index] = 1.0;
        operator.jacobian_vector_product(context, state, &direction, &mut column)?;
        NumericError::require_finite("Jacobian column", &column)?;
        for row in 0..dimension {
            jacobian[row][column_index] = column[row];
        }
        direction[column_index] = 0.0;
    }
    solve_dense(jacobian, residual.iter().map(|value| -value).collect())
}

fn staggered_update(
    operator: &(impl NonlinearOperator + ?Sized),
    context: &EvaluationContext,
    state: &[f64],
    layout: &BlockLayout,
    jacobi: bool,
) -> Result<Vec<f64>, SolveError> {
    let original = state.to_vec();
    let mut working = original.clone();
    for block in layout.blocks() {
        let base = if jacobi { &original } else { &working }.clone();
        let mut residual = vec![0.0; layout.dimension()];
        operator.residual(context, &base, &mut residual)?;
        let range = block.range();
        let width = block.length();
        let mut jacobian = vec![vec![0.0; width]; width];
        let mut direction = vec![0.0; layout.dimension()];
        let mut column = vec![0.0; layout.dimension()];
        for local_column in 0..width {
            direction[range.start + local_column] = 1.0;
            operator.jacobian_vector_product(context, &base, &direction, &mut column)?;
            for local_row in 0..width {
                jacobian[local_row][local_column] = column[range.start + local_row];
            }
            direction[range.start + local_column] = 0.0;
        }
        let right_hand_side = range.clone().map(|index| -residual[index]).collect();
        let block_update = solve_dense(jacobian, right_hand_side)?;
        for (index, delta) in range.zip(block_update) {
            working[index] = base[index] + delta;
        }
    }
    Ok(working
        .iter()
        .zip(original)
        .map(|(next, current)| next - current)
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn backtrack(
    operator: &(impl NonlinearOperator + ?Sized),
    context: &EvaluationContext,
    state: &[f64],
    update: &[f64],
    current_norm: f64,
    layout: Option<&BlockLayout>,
    config: &NewtonConfig,
) -> Result<(Vec<f64>, f64), SolveError> {
    let mut damping = config.initial_damping;
    let mut candidate = state.to_vec();
    let mut residual = vec![0.0; state.len()];
    for _ in 0..config.max_line_search_steps {
        for ((candidate_value, state_value), update_value) in
            candidate.iter_mut().zip(state).zip(update)
        {
            *candidate_value = state_value + damping * update_value;
        }
        operator.residual(context, &candidate, &mut residual)?;
        if residual.iter().all(|value| value.is_finite())
            && scaled_norm(layout, &residual) < current_norm
        {
            return Ok((candidate, damping));
        }
        damping *= 0.5;
        if damping < config.minimum_damping {
            break;
        }
    }
    Err(SolveError::LineSearchFailed)
}

fn block_norms(layout: Option<&BlockLayout>, residual: &[f64]) -> Vec<(String, f64)> {
    layout.map_or_else(Vec::new, |layout| {
        layout
            .blocks()
            .iter()
            .map(|block| {
                let norm = residual[block.range()]
                    .iter()
                    .map(|value| (value / block.residual_scale()).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (block.name().to_owned(), norm)
            })
            .collect()
    })
}

fn scaled_norm(layout: Option<&BlockLayout>, residual: &[f64]) -> f64 {
    match layout {
        Some(layout) => block_norms(Some(layout), residual)
            .iter()
            .map(|(_, norm)| norm * norm)
            .sum::<f64>()
            .sqrt(),
        None => l2(residual),
    }
}

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

pub(crate) fn solve_dense(
    mut matrix: Vec<Vec<f64>>,
    mut right_hand_side: Vec<f64>,
) -> Result<Vec<f64>, SolveError> {
    let dimension = right_hand_side.len();
    for pivot_column in 0..dimension {
        let pivot_row = (pivot_column..dimension)
            .max_by(|&left, &right| {
                matrix[left][pivot_column]
                    .abs()
                    .total_cmp(&matrix[right][pivot_column].abs())
            })
            .ok_or(SolveError::Singular)?;
        if matrix[pivot_row][pivot_column].abs() < 1.0e-14 {
            return Err(SolveError::Singular);
        }
        matrix.swap(pivot_column, pivot_row);
        right_hand_side.swap(pivot_column, pivot_row);
        let diagonal = matrix[pivot_column][pivot_column];
        for value in &mut matrix[pivot_column][pivot_column..] {
            *value /= diagonal;
        }
        right_hand_side[pivot_column] /= diagonal;
        let pivot_tail = matrix[pivot_column][pivot_column..].to_vec();
        for row in 0..dimension {
            if row == pivot_column {
                continue;
            }
            let factor = matrix[row][pivot_column];
            for (value, pivot_value) in matrix[row][pivot_column..].iter_mut().zip(&pivot_tail) {
                *value -= factor * pivot_value;
            }
            right_hand_side[row] -= factor * right_hand_side[pivot_column];
        }
    }
    NumericError::require_finite("dense solve", &right_hand_side)?;
    Ok(right_hand_side)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_rejects_iteration_count_overflow() {
        let config = NewtonConfig {
            max_iterations: usize::MAX,
            ..NewtonConfig::default()
        };
        assert!(matches!(
            validate_config(&config),
            Err(SolveError::InvalidConfiguration { .. })
        ));
    }
}
