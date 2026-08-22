//! Consumer-neutral numerical verification utilities.

use serde::{Deserialize, Serialize};

use crate::NumericError;

/// Absolute and relative tolerances used by comparison checkers.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComparisonTolerance {
    pub absolute: f64,
    pub relative: f64,
}

impl ComparisonTolerance {
    pub fn validate(self) -> Result<Self, NumericError> {
        if self.absolute.is_finite()
            && self.absolute >= 0.0
            && self.relative.is_finite()
            && self.relative >= 0.0
            && (self.absolute > 0.0 || self.relative > 0.0)
        {
            Ok(self)
        } else {
            Err(invalid(
                "comparison tolerances must be finite, nonnegative, and not both zero",
            ))
        }
    }
}

/// Error summary for a vector comparison.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub maximum_absolute_error: f64,
    pub maximum_relative_error: f64,
    pub accepted: bool,
}

/// Compare two solve strategies component by component.
///
/// A component passes when `|left-right| <= absolute + relative * max(|left|, |right|)`.
pub fn check_solve_strategy_agreement(
    left: &[f64],
    right: &[f64],
    tolerance: ComparisonTolerance,
) -> Result<ComparisonReport, NumericError> {
    tolerance.validate()?;
    NumericError::require_len("solve-strategy result", right.len(), left.len())?;
    if left.is_empty() {
        return Err(invalid("solve-strategy results must be nonempty"));
    }
    NumericError::require_finite("left solve-strategy result", left)?;
    NumericError::require_finite("right solve-strategy result", right)?;
    let mut maximum_absolute_error = 0.0_f64;
    let mut maximum_relative_error = 0.0_f64;
    let mut accepted = true;
    for (&left, &right) in left.iter().zip(right) {
        let absolute_error = (left - right).abs();
        if !absolute_error.is_finite() {
            return Err(invalid("solve-strategy discrepancy is not finite"));
        }
        let scale = left.abs().max(right.abs());
        let relative_error = if scale == 0.0 {
            0.0
        } else {
            absolute_error / scale
        };
        maximum_absolute_error = maximum_absolute_error.max(absolute_error);
        maximum_relative_error = maximum_relative_error.max(relative_error);
        accepted &= absolute_error <= tolerance.absolute
            || (scale > 0.0 && (absolute_error - tolerance.absolute) / scale <= tolerance.relative);
    }
    Ok(ComparisonReport {
        maximum_absolute_error,
        maximum_relative_error,
        accepted,
    })
}

/// Evaluation counts that are stable across machines and wall-clock conditions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkCount {
    pub operator_evaluations: u64,
    pub linear_iterations: u64,
    pub nonlinear_iterations: u64,
    pub accepted_steps: u64,
    pub rejected_steps: u64,
}

/// Inclusive deterministic limits for [`WorkCount`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkBudget {
    pub operator_evaluations: u64,
    pub linear_iterations: u64,
    pub nonlinear_iterations: u64,
    pub accepted_steps: u64,
    pub rejected_steps: u64,
}

/// Per-category result of checking deterministic work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkBudgetReport {
    pub observed: WorkCount,
    pub budget: WorkBudget,
    pub accepted: bool,
}

pub fn check_work_budget(observed: WorkCount, budget: WorkBudget) -> WorkBudgetReport {
    let accepted = observed.operator_evaluations <= budget.operator_evaluations
        && observed.linear_iterations <= budget.linear_iterations
        && observed.nonlinear_iterations <= budget.nonlinear_iterations
        && observed.accepted_steps <= budget.accepted_steps
        && observed.rejected_steps <= budget.rejected_steps;
    WorkBudgetReport {
        observed,
        budget,
        accepted,
    }
}

/// A positive resolution and its positive error.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceSample {
    pub resolution: f64,
    pub error: f64,
}

/// Adjacent-pair and least-squares fitted convergence orders.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceOrderReport {
    pub pair_orders: Vec<f64>,
    pub fitted_order: f64,
    pub minimum_pair_order: f64,
}

/// Estimate convergence order from at least two samples.
///
/// Resolutions must be strictly decreasing and errors strictly positive. Exact zero errors carry
/// no finite logarithmic order and are therefore refused rather than assigned a sentinel value.
pub fn estimate_convergence_order(
    samples: &[ConvergenceSample],
) -> Result<ConvergenceOrderReport, NumericError> {
    if samples.len() < 2 {
        return Err(invalid("convergence order requires at least two samples"));
    }
    for (index, sample) in samples.iter().enumerate() {
        if !sample.resolution.is_finite() || sample.resolution <= 0.0 {
            return Err(invalid(format!(
                "sample {index} resolution must be finite and positive"
            )));
        }
        if !sample.error.is_finite() || sample.error <= 0.0 {
            return Err(invalid(format!(
                "sample {index} error must be finite and positive"
            )));
        }
    }
    if samples
        .windows(2)
        .any(|pair| pair[1].resolution >= pair[0].resolution)
    {
        return Err(invalid(
            "convergence resolutions must be strictly decreasing",
        ));
    }

    let pair_orders: Vec<_> = samples
        .windows(2)
        .map(|pair| {
            (pair[1].error.ln() - pair[0].error.ln())
                / (pair[1].resolution.ln() - pair[0].resolution.ln())
        })
        .collect();
    let minimum_pair_order = pair_orders.iter().copied().fold(f64::INFINITY, f64::min);

    let count = samples.len() as f64;
    let mean_x = samples
        .iter()
        .map(|sample| sample.resolution.ln())
        .sum::<f64>()
        / count;
    let mean_y = samples.iter().map(|sample| sample.error.ln()).sum::<f64>() / count;
    let numerator = samples
        .iter()
        .map(|sample| (sample.resolution.ln() - mean_x) * (sample.error.ln() - mean_y))
        .sum::<f64>();
    let denominator = samples
        .iter()
        .map(|sample| (sample.resolution.ln() - mean_x).powi(2))
        .sum::<f64>();
    if denominator == 0.0 {
        return Err(invalid(
            "convergence resolutions have no logarithmic spread",
        ));
    }
    let fitted_order = numerator / denominator;
    if !fitted_order.is_finite() || pair_orders.iter().any(|order| !order.is_finite()) {
        return Err(invalid("convergence order is not finite"));
    }
    Ok(ConvergenceOrderReport {
        pair_orders,
        fitted_order,
        minimum_pair_order,
    })
}

/// Weighted trajectory discrepancy over a common, strictly increasing time grid.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryNormReport {
    pub l_infinity: f64,
    pub l2_time: f64,
}

/// Compute the maximum-in-time Euclidean state error and its trapezoidal time integral.
pub fn trajectory_error_norms(
    times: &[f64],
    reference: &[Vec<f64>],
    candidate: &[Vec<f64>],
) -> Result<TrajectoryNormReport, NumericError> {
    if times.is_empty() {
        return Err(invalid("trajectory requires at least one time sample"));
    }
    NumericError::require_len("reference trajectory", reference.len(), times.len())?;
    NumericError::require_len("candidate trajectory", candidate.len(), times.len())?;
    NumericError::require_finite("trajectory times", times)?;
    if times.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(invalid("trajectory times must be strictly increasing"));
    }
    let dimension = reference[0].len();
    if dimension == 0 {
        return Err(invalid("trajectory states must be nonempty"));
    }
    let mut squared_errors = Vec::with_capacity(times.len());
    let mut l_infinity = 0.0_f64;
    for (index, (reference, candidate)) in reference.iter().zip(candidate).enumerate() {
        NumericError::require_len("trajectory state", reference.len(), dimension)?;
        NumericError::require_len("trajectory state", candidate.len(), dimension)?;
        NumericError::require_finite(&format!("reference trajectory state {index}"), reference)?;
        NumericError::require_finite(&format!("candidate trajectory state {index}"), candidate)?;
        let squared = reference
            .iter()
            .zip(candidate)
            .map(|(reference, candidate)| {
                let error = reference - candidate;
                error * error
            })
            .sum::<f64>();
        if !squared.is_finite() {
            return Err(invalid("trajectory discrepancy is not finite"));
        }
        l_infinity = l_infinity.max(squared.sqrt());
        squared_errors.push(squared);
    }
    let integral = times
        .windows(2)
        .zip(squared_errors.windows(2))
        .map(|(time, error)| 0.5 * (time[1] - time[0]) * (error[0] + error[1]))
        .sum::<f64>();
    if !integral.is_finite() {
        return Err(invalid("trajectory time integral is not finite"));
    }
    Ok(TrajectoryNormReport {
        l_infinity,
        l2_time: integral.sqrt(),
    })
}

/// Error at one perturbation size in a directional derivative check.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DerivativeSample {
    pub step: f64,
    pub maximum_absolute_error: f64,
}

/// A sequence of derivative or Taylor-remainder errors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DerivativeCheckReport {
    pub samples: Vec<DerivativeSample>,
}

/// Check a directional derivative using centered finite differences.
pub fn check_centered_difference<F>(
    state: &[f64],
    direction: &[f64],
    analytic_derivative: &[f64],
    output_dimension: usize,
    steps: &[f64],
    mut evaluate: F,
) -> Result<DerivativeCheckReport, NumericError>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<(), NumericError>,
{
    validate_derivative_inputs(
        state,
        direction,
        analytic_derivative,
        output_dimension,
        steps,
    )?;
    let mut samples = Vec::with_capacity(steps.len());
    for &step in steps {
        let plus: Vec<_> = state
            .iter()
            .zip(direction)
            .map(|(x, d)| x + step * d)
            .collect();
        let minus: Vec<_> = state
            .iter()
            .zip(direction)
            .map(|(x, d)| x - step * d)
            .collect();
        NumericError::require_finite("positive centered-difference probe", &plus)?;
        NumericError::require_finite("negative centered-difference probe", &minus)?;
        let mut plus_output = vec![0.0; output_dimension];
        let mut minus_output = vec![0.0; output_dimension];
        evaluate(&plus, &mut plus_output)?;
        evaluate(&minus, &mut minus_output)?;
        NumericError::require_finite("positive centered-difference output", &plus_output)?;
        NumericError::require_finite("negative centered-difference output", &minus_output)?;
        let mut maximum_absolute_error = 0.0_f64;
        for (analytic, (plus, minus)) in analytic_derivative
            .iter()
            .zip(plus_output.iter().zip(minus_output))
        {
            let error = (analytic - (plus - minus) / (2.0 * step)).abs();
            if !error.is_finite() {
                return Err(invalid("centered-difference discrepancy is not finite"));
            }
            maximum_absolute_error = maximum_absolute_error.max(error);
        }
        samples.push(DerivativeSample {
            step,
            maximum_absolute_error,
        });
    }
    Ok(DerivativeCheckReport { samples })
}

/// Check a directional derivative from the imaginary response of a complex-step evaluation.
///
/// The callback evaluates `f(state + i * step * direction)` and writes only its imaginary part.
/// This keeps Methodus independent of the caller's complex scalar representation.
pub fn check_complex_step<F>(
    state: &[f64],
    direction: &[f64],
    analytic_derivative: &[f64],
    output_dimension: usize,
    steps: &[f64],
    mut evaluate_imaginary: F,
) -> Result<DerivativeCheckReport, NumericError>
where
    F: FnMut(&[f64], &[f64], f64, &mut [f64]) -> Result<(), NumericError>,
{
    validate_derivative_inputs(
        state,
        direction,
        analytic_derivative,
        output_dimension,
        steps,
    )?;
    let mut samples = Vec::with_capacity(steps.len());
    for &step in steps {
        let mut imaginary = vec![0.0; output_dimension];
        evaluate_imaginary(state, direction, step, &mut imaginary)?;
        NumericError::require_finite("complex-step imaginary output", &imaginary)?;
        let mut maximum_absolute_error = 0.0_f64;
        for (analytic, imaginary) in analytic_derivative.iter().zip(imaginary) {
            let error = (analytic - imaginary / step).abs();
            if !error.is_finite() {
                return Err(invalid("complex-step discrepancy is not finite"));
            }
            maximum_absolute_error = maximum_absolute_error.max(error);
        }
        samples.push(DerivativeSample {
            step,
            maximum_absolute_error,
        });
    }
    Ok(DerivativeCheckReport { samples })
}

/// Check the first-order Taylor remainder `f(x+h d) - f(x) - h Df(x)d`.
pub fn check_taylor_remainder<F>(
    state: &[f64],
    direction: &[f64],
    analytic_derivative: &[f64],
    output_dimension: usize,
    steps: &[f64],
    mut evaluate: F,
) -> Result<DerivativeCheckReport, NumericError>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<(), NumericError>,
{
    validate_derivative_inputs(
        state,
        direction,
        analytic_derivative,
        output_dimension,
        steps,
    )?;
    let mut baseline = vec![0.0; output_dimension];
    evaluate(state, &mut baseline)?;
    NumericError::require_finite("Taylor baseline output", &baseline)?;
    let mut samples = Vec::with_capacity(steps.len());
    for &step in steps {
        let shifted: Vec<_> = state
            .iter()
            .zip(direction)
            .map(|(x, d)| x + step * d)
            .collect();
        NumericError::require_finite("Taylor shifted probe", &shifted)?;
        let mut output = vec![0.0; output_dimension];
        evaluate(&shifted, &mut output)?;
        NumericError::require_finite("Taylor shifted output", &output)?;
        let mut maximum_absolute_error = 0.0_f64;
        for (output, (baseline, derivative)) in
            output.iter().zip(baseline.iter().zip(analytic_derivative))
        {
            let error = (output - baseline - step * derivative).abs();
            if !error.is_finite() {
                return Err(invalid("Taylor remainder is not finite"));
            }
            maximum_absolute_error = maximum_absolute_error.max(error);
        }
        samples.push(DerivativeSample {
            step,
            maximum_absolute_error,
        });
    }
    Ok(DerivativeCheckReport { samples })
}

fn validate_derivative_inputs(
    state: &[f64],
    direction: &[f64],
    analytic_derivative: &[f64],
    output_dimension: usize,
    steps: &[f64],
) -> Result<(), NumericError> {
    NumericError::require_len("derivative direction", direction.len(), state.len())?;
    NumericError::require_len(
        "analytic derivative",
        analytic_derivative.len(),
        output_dimension,
    )?;
    NumericError::require_finite("derivative state", state)?;
    NumericError::require_finite("derivative direction", direction)?;
    NumericError::require_finite("analytic derivative", analytic_derivative)?;
    if state.is_empty() || output_dimension == 0 {
        return Err(invalid(
            "derivative checks require nonempty input and output dimensions",
        ));
    }
    if steps.is_empty() {
        return Err(invalid("derivative check requires at least one step"));
    }
    if steps.iter().any(|step| !step.is_finite() || *step <= 0.0) {
        return Err(invalid("derivative steps must be finite and positive"));
    }
    if steps.windows(2).any(|pair| pair[1] >= pair[0]) {
        return Err(invalid("derivative steps must be strictly decreasing"));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> NumericError {
    NumericError::InvalidInput {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(input: &[f64], output: &mut [f64]) -> Result<(), NumericError> {
        output[0] = input[0] * input[0];
        output[1] = input[1] * input[1];
        Ok(())
    }

    #[test]
    fn centered_difference_and_taylor_have_expected_orders() {
        let state = [2.0, -1.0];
        let direction = [0.5, 2.0];
        let derivative = [2.0, -4.0];
        let steps = [0.2, 0.1, 0.05];
        let centered =
            check_centered_difference(&state, &direction, &derivative, 2, &steps, square).unwrap();
        assert!(
            centered
                .samples
                .iter()
                .all(|sample| sample.maximum_absolute_error < 1.0e-12)
        );

        let taylor =
            check_taylor_remainder(&state, &direction, &derivative, 2, &steps, square).unwrap();
        let convergence: Vec<_> = taylor
            .samples
            .iter()
            .map(|sample| ConvergenceSample {
                resolution: sample.step,
                error: sample.maximum_absolute_error,
            })
            .collect();
        let report = estimate_convergence_order(&convergence).unwrap();
        assert!((report.fitted_order - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn complex_step_uses_caller_imaginary_response() {
        let report = check_complex_step(
            &[2.0],
            &[3.0],
            &[12.0],
            1,
            &[1.0e-10, 1.0e-20],
            |state, direction, step, output| {
                // Imaginary part of (x + i h d)^2 is exactly 2 x h d.
                output[0] = 2.0 * state[0] * step * direction[0];
                Ok(())
            },
        )
        .unwrap();
        assert!(
            report
                .samples
                .iter()
                .all(|sample| sample.maximum_absolute_error == 0.0)
        );
    }

    #[test]
    fn convergence_and_trajectory_reports_match_analytic_values() {
        let report = estimate_convergence_order(&[
            ConvergenceSample {
                resolution: 0.5,
                error: 0.25,
            },
            ConvergenceSample {
                resolution: 0.25,
                error: 0.0625,
            },
            ConvergenceSample {
                resolution: 0.125,
                error: 0.015625,
            },
        ])
        .unwrap();
        assert_eq!(report.pair_orders, vec![2.0, 2.0]);
        assert!((report.fitted_order - 2.0).abs() < 1.0e-14);

        let norms = trajectory_error_norms(
            &[0.0, 1.0, 2.0],
            &[vec![0.0], vec![0.0], vec![0.0]],
            &[vec![1.0], vec![1.0], vec![1.0]],
        )
        .unwrap();
        assert_eq!(norms.l_infinity, 1.0);
        assert!((norms.l2_time - 2.0_f64.sqrt()).abs() < 1.0e-14);

        let vector_norms = trajectory_error_norms(
            &[0.0, 1.0],
            &[vec![0.0, 0.0], vec![0.0, 0.0]],
            &[vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();
        assert!((vector_norms.l_infinity - 2.0_f64.sqrt()).abs() < 1.0e-14);
        assert!((vector_norms.l2_time - 2.0_f64.sqrt()).abs() < 1.0e-14);
    }

    #[test]
    fn agreement_and_work_budget_have_inclusive_boundaries() {
        let agreement = check_solve_strategy_agreement(
            &[1.0, 100.0],
            &[1.000_001, 100.01],
            ComparisonTolerance {
                absolute: 1.0e-5,
                relative: 1.0e-4,
            },
        )
        .unwrap();
        assert!(agreement.accepted);
        let observed = WorkCount {
            operator_evaluations: 10,
            linear_iterations: 5,
            nonlinear_iterations: 2,
            accepted_steps: 4,
            rejected_steps: 1,
        };
        assert!(
            check_work_budget(
                observed,
                WorkBudget {
                    operator_evaluations: 10,
                    linear_iterations: 5,
                    nonlinear_iterations: 2,
                    accepted_steps: 4,
                    rejected_steps: 1,
                }
            )
            .accepted
        );
        assert!(
            !check_work_budget(
                observed,
                WorkBudget {
                    operator_evaluations: 9,
                    linear_iterations: 5,
                    nonlinear_iterations: 2,
                    accepted_steps: 4,
                    rejected_steps: 1,
                }
            )
            .accepted
        );
    }

    #[test]
    fn hostile_inputs_are_refused() {
        assert!(
            estimate_convergence_order(&[
                ConvergenceSample {
                    resolution: 0.5,
                    error: 0.25
                },
                ConvergenceSample {
                    resolution: 0.5,
                    error: 0.1
                },
            ])
            .is_err()
        );
        assert!(
            estimate_convergence_order(&[
                ConvergenceSample {
                    resolution: 0.5,
                    error: 0.0
                },
                ConvergenceSample {
                    resolution: 0.25,
                    error: 0.0
                },
            ])
            .is_err()
        );
        assert!(
            trajectory_error_norms(
                &[0.0, 0.0],
                &[vec![0.0], vec![0.0]],
                &[vec![0.0], vec![0.0]]
            )
            .is_err()
        );
        assert!(check_centered_difference(&[1.0], &[1.0], &[2.0], 1, &[0.1, 0.2], square).is_err());
        assert!(
            check_solve_strategy_agreement(
                &[f64::NAN],
                &[0.0],
                ComparisonTolerance {
                    absolute: 1.0,
                    relative: 0.0
                },
            )
            .is_err()
        );
        assert!(
            check_solve_strategy_agreement(
                &[f64::MAX],
                &[-f64::MAX],
                ComparisonTolerance {
                    absolute: 1.0,
                    relative: 1.0,
                },
            )
            .is_err()
        );
        assert!(trajectory_error_norms(&[0.0], &[vec![f64::MAX]], &[vec![-f64::MAX]]).is_err());
        assert!(
            check_centered_difference(&[f64::MAX], &[f64::MAX], &[0.0], 1, &[2.0], |_, output| {
                output[0] = 0.0;
                Ok(())
            },)
            .is_err()
        );
    }
}
