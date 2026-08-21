use serde::{Deserialize, Serialize};

use crate::{EvaluationContext, LinearOperator, NumericError, Preconditioner, SolveError};

/// Convergence policy for a conjugate-gradient solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConjugateGradientConfig {
    pub max_iterations: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}

impl Default for ConjugateGradientConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-10,
        }
    }
}

/// One deterministic residual observation from a linear solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearIteration {
    pub iteration: usize,
    pub residual_norm: f64,
}

/// Final state and convergence evidence from a linear solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearSolveReport {
    pub solution: Vec<f64>,
    pub converged: bool,
    pub trace: Vec<LinearIteration>,
}

/// Solve a symmetric positive-definite system through operator and optional preconditioner
/// actions. The implementation fixes the reduction and update order so reproducible downstream
/// operators yield a reproducible reference solve.
pub fn solve_conjugate_gradient(
    operator: &(impl LinearOperator + ?Sized),
    preconditioner: Option<&dyn Preconditioner>,
    context: &EvaluationContext,
    right_hand_side: &[f64],
    initial_solution: &[f64],
    config: &ConjugateGradientConfig,
) -> Result<LinearSolveReport, SolveError> {
    validate_config(config)?;
    let dimension = operator.rows();
    if operator.columns() != dimension {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "conjugate gradient requires a square operator, got {}x{}",
                operator.rows(),
                operator.columns()
            ),
        });
    }
    NumericError::require_len("linear right-hand side", right_hand_side.len(), dimension)?;
    NumericError::require_len("initial linear solution", initial_solution.len(), dimension)?;
    NumericError::require_finite("linear right-hand side", right_hand_side)?;
    NumericError::require_finite("initial linear solution", initial_solution)?;
    if let Some(preconditioner) = preconditioner
        && preconditioner.dimension() != dimension
    {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "preconditioner dimension {} differs from operator dimension {dimension}",
                preconditioner.dimension()
            ),
        });
    }

    let mut solution = initial_solution.to_vec();
    let mut action = vec![0.0; dimension];
    operator.apply(context, &solution, &mut action)?;
    NumericError::require_finite("initial linear operator action", &action)?;
    let mut residual = right_hand_side
        .iter()
        .zip(&action)
        .map(|(rhs, action)| rhs - action)
        .collect::<Vec<_>>();
    let initial_norm = l2(&residual)?;
    let threshold = config.absolute_tolerance + config.relative_tolerance * initial_norm;
    let mut trace = vec![LinearIteration {
        iteration: 0,
        residual_norm: initial_norm,
    }];
    if initial_norm <= threshold {
        return Ok(LinearSolveReport {
            solution,
            converged: true,
            trace,
        });
    }

    let mut preconditioned = vec![0.0; dimension];
    apply_preconditioner(preconditioner, context, &residual, &mut preconditioned)?;
    let mut search = preconditioned.clone();
    let mut residual_product = dot(&residual, &preconditioned)?;
    if residual_product <= 0.0 {
        return Err(SolveError::KrylovBreakdown { iteration: 0 });
    }

    for iteration in 1..=config.max_iterations {
        operator.apply(context, &search, &mut action)?;
        NumericError::require_finite("linear operator action", &action)?;
        let curvature = dot(&search, &action)?;
        if curvature <= 0.0 {
            return Err(SolveError::KrylovBreakdown { iteration });
        }
        let alpha = residual_product / curvature;
        NumericError::require_finite("conjugate-gradient step", &[alpha])?;
        for index in 0..dimension {
            solution[index] += alpha * search[index];
            residual[index] -= alpha * action[index];
        }
        NumericError::require_finite("linear solution", &solution)?;
        NumericError::require_finite("linear residual", &residual)?;
        let residual_norm = l2(&residual)?;
        trace.push(LinearIteration {
            iteration,
            residual_norm,
        });
        if residual_norm <= threshold {
            return Ok(LinearSolveReport {
                solution,
                converged: true,
                trace,
            });
        }

        apply_preconditioner(preconditioner, context, &residual, &mut preconditioned)?;
        let next_product = dot(&residual, &preconditioned)?;
        if next_product <= 0.0 {
            return Err(SolveError::KrylovBreakdown { iteration });
        }
        let beta = next_product / residual_product;
        NumericError::require_finite("conjugate-gradient recurrence", &[beta])?;
        for index in 0..dimension {
            search[index] = preconditioned[index] + beta * search[index];
        }
        residual_product = next_product;
    }

    Ok(LinearSolveReport {
        solution,
        converged: false,
        trace,
    })
}

fn validate_config(config: &ConjugateGradientConfig) -> Result<(), SolveError> {
    let tolerances_valid = config.absolute_tolerance.is_finite()
        && config.absolute_tolerance >= 0.0
        && config.relative_tolerance.is_finite()
        && config.relative_tolerance >= 0.0
        && (config.absolute_tolerance > 0.0 || config.relative_tolerance > 0.0);
    if config.max_iterations == 0 || !tolerances_valid {
        return Err(SolveError::InvalidConfiguration {
            reason: "linear iteration limit and tolerances must be positive and finite".into(),
        });
    }
    Ok(())
}

fn apply_preconditioner(
    preconditioner: Option<&dyn Preconditioner>,
    context: &EvaluationContext,
    input: &[f64],
    output: &mut [f64],
) -> Result<(), NumericError> {
    if let Some(preconditioner) = preconditioner {
        preconditioner.apply_inverse(context, input, output)?;
    } else {
        output.copy_from_slice(input);
    }
    NumericError::require_finite("preconditioner action", output)
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, NumericError> {
    let value = left
        .iter()
        .zip(right)
        .fold(0.0, |sum, (left, right)| sum + left * right);
    NumericError::require_finite("linear inner product", &[value])?;
    Ok(value)
}

fn l2(values: &[f64]) -> Result<f64, NumericError> {
    let norm = dot(values, values)?.sqrt();
    NumericError::require_finite("linear residual norm", &[norm])?;
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrMatrix;

    struct DiagonalInverse(Vec<f64>);

    impl Preconditioner for DiagonalInverse {
        fn dimension(&self) -> usize {
            self.0.len()
        }

        fn apply_inverse(
            &self,
            _context: &EvaluationContext,
            right_hand_side: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            for ((output, value), diagonal) in output.iter_mut().zip(right_hand_side).zip(&self.0) {
                *output = value / diagonal;
            }
            Ok(())
        }
    }

    #[test]
    fn conjugate_gradient_uses_operator_and_preconditioner_actions() {
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, 4.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 4.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 3.0),
            ],
        )
        .unwrap();
        let preconditioner = DiagonalInverse(vec![4.0, 4.0, 3.0]);
        let report = solve_conjugate_gradient(
            &matrix,
            Some(&preconditioner),
            &EvaluationContext::reproducible(),
            &[15.0, 10.0, 10.0],
            &[0.0; 3],
            &ConjugateGradientConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
        for (actual, expected) in report.solution.iter().zip([5.0; 3]) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        assert!(report.trace.last().unwrap().residual_norm < 1.0e-12);
    }

    #[test]
    fn conjugate_gradient_refuses_rectangular_operators() {
        let matrix = CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![1.0]).unwrap();
        let error = solve_conjugate_gradient(
            &matrix,
            None,
            &EvaluationContext::default(),
            &[1.0],
            &[0.0, 0.0],
            &ConjugateGradientConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
    }
}
