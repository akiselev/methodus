use serde::{Deserialize, Serialize};

use crate::{EvaluationContext, NumericError, SolveError, nonlinear::solve_dense};

/// A rectangular residual system for nonlinear least-squares algorithms.
pub trait LeastSquaresOperator: Send + Sync {
    fn variable_count(&self) -> usize;
    fn residual_count(&self) -> usize;
    fn residual(
        &self,
        context: &EvaluationContext,
        variables: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
    /// Fill a row-major `residual_count * variable_count` Jacobian.
    fn jacobian(
        &self,
        context: &EvaluationContext,
        variables: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeastSquaresConfig {
    pub max_iterations: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub initial_damping: f64,
    pub minimum_damping: f64,
    pub maximum_damping: f64,
}

impl Default for LeastSquaresConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-8,
            initial_damping: 1.0e-3,
            minimum_damping: 1.0e-12,
            maximum_damping: 1.0e12,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeastSquaresIteration {
    pub iteration: usize,
    pub residual_norm: f64,
    pub damping: f64,
    pub accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeastSquaresReport {
    pub variables: Vec<f64>,
    pub converged: bool,
    pub trace: Vec<LeastSquaresIteration>,
}

/// Solve a rectangular residual system with a deterministic damped Gauss-Newton baseline.
pub fn solve_least_squares(
    operator: &(impl LeastSquaresOperator + ?Sized),
    context: &EvaluationContext,
    initial_variables: &[f64],
    config: &LeastSquaresConfig,
) -> Result<LeastSquaresReport, SolveError> {
    validate_config(config)?;
    let variables_count = operator.variable_count();
    let residual_count = operator.residual_count();
    NumericError::require_len(
        "initial least-squares variables",
        initial_variables.len(),
        variables_count,
    )?;
    if variables_count == 0 || residual_count == 0 {
        return Err(SolveError::InvalidConfiguration {
            reason: "least-squares systems require variables and residuals".into(),
        });
    }
    NumericError::require_finite("initial least-squares variables", initial_variables)?;

    let mut variables = initial_variables.to_vec();
    let mut residual = evaluate_residual(operator, context, &variables)?;
    let initial_norm = l2(&residual);
    let threshold = config.absolute_tolerance + config.relative_tolerance * initial_norm;
    let mut damping = config.initial_damping;
    let mut trace = Vec::with_capacity(config.max_iterations.saturating_add(1));

    for iteration in 0..=config.max_iterations {
        let residual_norm = l2(&residual);
        if residual_norm <= threshold {
            trace.push(LeastSquaresIteration {
                iteration,
                residual_norm,
                damping,
                accepted: true,
            });
            return Ok(LeastSquaresReport {
                variables,
                converged: true,
                trace,
            });
        }
        if iteration == config.max_iterations {
            trace.push(LeastSquaresIteration {
                iteration,
                residual_norm,
                damping,
                accepted: false,
            });
            break;
        }

        let jacobian = evaluate_jacobian(operator, context, &variables)?;
        let (normal, right_hand_side) = normal_equations(
            &jacobian,
            &residual,
            residual_count,
            variables_count,
            damping,
        );
        let update = solve_dense(normal, right_hand_side)?;
        let candidate: Vec<_> = variables
            .iter()
            .zip(&update)
            .map(|(value, delta)| value + delta)
            .collect();
        NumericError::require_finite("least-squares candidate", &candidate)?;
        let candidate_residual = evaluate_residual(operator, context, &candidate)?;
        let accepted = l2(&candidate_residual) < residual_norm;
        trace.push(LeastSquaresIteration {
            iteration,
            residual_norm,
            damping,
            accepted,
        });
        if accepted {
            variables = candidate;
            residual = candidate_residual;
            damping = (damping * 0.3).max(config.minimum_damping);
        } else {
            damping = (damping * 10.0).min(config.maximum_damping);
        }
    }

    Ok(LeastSquaresReport {
        variables,
        converged: false,
        trace,
    })
}

/// Compare a supplied row-major Jacobian with centered finite differences.
pub fn verify_least_squares_jacobian(
    operator: &(impl LeastSquaresOperator + ?Sized),
    context: &EvaluationContext,
    variables: &[f64],
    epsilon: f64,
) -> Result<f64, NumericError> {
    let variable_count = operator.variable_count();
    let residual_count = operator.residual_count();
    NumericError::require_len("least-squares variables", variables.len(), variable_count)?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(NumericError::InvalidInput {
            message: "verification epsilon must be finite and positive".into(),
        });
    }
    let analytic = evaluate_jacobian(operator, context, variables)?;
    let mut discrepancy = 0.0_f64;
    for column in 0..variable_count {
        let mut plus = variables.to_vec();
        let mut minus = variables.to_vec();
        plus[column] += epsilon;
        minus[column] -= epsilon;
        let residual_plus = evaluate_residual(operator, context, &plus)?;
        let residual_minus = evaluate_residual(operator, context, &minus)?;
        for row in 0..residual_count {
            let numeric = (residual_plus[row] - residual_minus[row]) / (2.0 * epsilon);
            discrepancy =
                discrepancy.max((analytic[row * variable_count + column] - numeric).abs());
        }
    }
    Ok(discrepancy)
}

fn validate_config(config: &LeastSquaresConfig) -> Result<(), SolveError> {
    let tolerances_valid = config.absolute_tolerance.is_finite()
        && config.absolute_tolerance >= 0.0
        && config.relative_tolerance.is_finite()
        && config.relative_tolerance >= 0.0
        && (config.absolute_tolerance > 0.0 || config.relative_tolerance > 0.0);
    let damping_valid = config.minimum_damping.is_finite()
        && config.minimum_damping > 0.0
        && config.initial_damping.is_finite()
        && config.initial_damping >= config.minimum_damping
        && config.maximum_damping.is_finite()
        && config.maximum_damping >= config.initial_damping;
    if config.max_iterations == 0 || !tolerances_valid || !damping_valid {
        return Err(SolveError::InvalidConfiguration {
            reason: "least-squares limits, tolerances, and damping are invalid".into(),
        });
    }
    Ok(())
}

fn evaluate_residual(
    operator: &(impl LeastSquaresOperator + ?Sized),
    context: &EvaluationContext,
    variables: &[f64],
) -> Result<Vec<f64>, NumericError> {
    let mut output = vec![0.0; operator.residual_count()];
    operator.residual(context, variables, &mut output)?;
    NumericError::require_finite("least-squares residual", &output)?;
    Ok(output)
}

fn evaluate_jacobian(
    operator: &(impl LeastSquaresOperator + ?Sized),
    context: &EvaluationContext,
    variables: &[f64],
) -> Result<Vec<f64>, NumericError> {
    let length = operator
        .residual_count()
        .checked_mul(operator.variable_count())
        .ok_or_else(|| NumericError::InvalidInput {
            message: "least-squares Jacobian dimensions overflow usize".into(),
        })?;
    let mut output = vec![0.0; length];
    operator.jacobian(context, variables, &mut output)?;
    NumericError::require_finite("least-squares Jacobian", &output)?;
    Ok(output)
}

fn normal_equations(
    jacobian: &[f64],
    residual: &[f64],
    rows: usize,
    columns: usize,
    damping: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut normal = vec![vec![0.0; columns]; columns];
    let mut right_hand_side = vec![0.0; columns];
    for row in 0..rows {
        for left in 0..columns {
            let left_value = jacobian[row * columns + left];
            right_hand_side[left] -= left_value * residual[row];
            for right in 0..columns {
                normal[left][right] += left_value * jacobian[row * columns + right];
            }
        }
    }
    for (index, row) in normal.iter_mut().enumerate() {
        row[index] += damping;
    }
    (normal, right_hand_side)
}

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LineFit;

    impl LeastSquaresOperator for LineFit {
        fn variable_count(&self) -> usize {
            2
        }
        fn residual_count(&self) -> usize {
            3
        }
        fn residual(
            &self,
            _: &EvaluationContext,
            x: &[f64],
            out: &mut [f64],
        ) -> Result<(), NumericError> {
            for (row, sample) in [0.0, 1.0, 2.0].into_iter().enumerate() {
                out[row] = x[0] * sample + x[1] - (2.0 * sample + 1.0);
            }
            Ok(())
        }
        fn jacobian(
            &self,
            _: &EvaluationContext,
            _: &[f64],
            out: &mut [f64],
        ) -> Result<(), NumericError> {
            out.copy_from_slice(&[0.0, 1.0, 1.0, 1.0, 2.0, 1.0]);
            Ok(())
        }
    }

    #[test]
    fn overdetermined_line_fit_converges_and_jacobian_is_independent() {
        let context = EvaluationContext::reproducible();
        let report = solve_least_squares(
            &LineFit,
            &context,
            &[0.0, 0.0],
            &LeastSquaresConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
        assert!((report.variables[0] - 2.0).abs() < 1.0e-8);
        assert!((report.variables[1] - 1.0).abs() < 1.0e-8);
        assert!(
            verify_least_squares_jacobian(&LineFit, &context, &[0.5, 0.5], 1.0e-6).unwrap()
                < 1.0e-8
        );
    }
}
