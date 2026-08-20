use serde::{Deserialize, Serialize};

use crate::{
    DaeOperator, EvaluationContext, NewtonConfig, NonlinearOperator, NumericError, SolveError,
    solve_newton,
};

/// Backward differentiation formula order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BdfOrder {
    One,
    Two,
}

/// Error control and nonlinear solve policy for one BDF attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BdfConfig {
    pub order: BdfOrder,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub minimum_step: f64,
    pub maximum_step: f64,
    pub newton: NewtonConfig,
}

impl Default for BdfConfig {
    fn default() -> Self {
        Self {
            order: BdfOrder::Two,
            absolute_tolerance: 1.0e-7,
            relative_tolerance: 1.0e-5,
            minimum_step: 1.0e-10,
            maximum_step: f64::INFINITY,
            newton: NewtonConfig::default(),
        }
    }
}

/// Committed BDF history. Rejected attempts never mutate this value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BdfState {
    pub time: f64,
    pub values: Vec<f64>,
    pub previous_values: Option<Vec<f64>>,
    pub accepted_steps: u64,
}

impl BdfState {
    pub fn initialize(
        operator: &(impl DaeOperator + ?Sized),
        context: &EvaluationContext,
        time: f64,
        mut values: Vec<f64>,
    ) -> Result<Self, NumericError> {
        if !time.is_finite() {
            return Err(NumericError::InvalidInput {
                message: "initial DAE time must be finite".into(),
            });
        }
        NumericError::require_len("initial DAE state", values.len(), operator.dimension())?;
        NumericError::require_finite("initial DAE state", &values)?;
        operator.make_initial_state_consistent(context, time, &mut values)?;
        NumericError::require_finite("consistent initial DAE state", &values)?;
        Ok(Self {
            time,
            values,
            previous_values: None,
            accepted_steps: 0,
        })
    }
}

/// A zero crossing located by interpolation across an accepted step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocatedEvent {
    pub index: usize,
    pub time: f64,
    pub value_before: f64,
    pub value_after: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcceptedStep {
    pub state: BdfState,
    pub suggested_step: f64,
    pub error_estimate: f64,
    pub events: Vec<LocatedEvent>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RejectedStep {
    pub committed_state: BdfState,
    pub suggested_step: f64,
    pub error_estimate: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StepOutcome {
    Accepted(AcceptedStep),
    Rejected(RejectedStep),
}

/// Attempt one implicit BDF1 or BDF2 step.
pub fn bdf_step(
    operator: &(impl DaeOperator + ?Sized),
    context: &EvaluationContext,
    state: &BdfState,
    step: f64,
    config: &BdfConfig,
) -> Result<StepOutcome, SolveError> {
    validate_step(operator, state, step, config)?;
    let (candidate, error_estimate) = match (config.order, &state.previous_values) {
        (BdfOrder::Two, Some(previous)) => {
            let second = implicit_step(operator, context, state, Some(previous), step, 2, config)?;
            let first = implicit_step(operator, context, state, None, step, 1, config)?;
            let error = scaled_error(&second, &first, config);
            (second, error)
        }
        _ => (
            implicit_step(operator, context, state, None, step, 1, config)?,
            0.0,
        ),
    };
    let suggested_step = adapt_step(step, error_estimate, config);
    if error_estimate > 1.0 && step > config.minimum_step {
        return Ok(StepOutcome::Rejected(RejectedStep {
            committed_state: state.clone(),
            suggested_step,
            error_estimate,
        }));
    }

    let next_time = state.time + step;
    let mut before = vec![0.0; operator.event_count()];
    let mut after = vec![0.0; operator.event_count()];
    operator.event_values(context, state.time, &state.values, &mut before)?;
    operator.event_values(context, next_time, &candidate, &mut after)?;
    NumericError::require_finite("DAE events before step", &before)?;
    NumericError::require_finite("DAE events after step", &after)?;
    let events = locate_events(state.time, next_time, &before, &after);
    Ok(StepOutcome::Accepted(AcceptedStep {
        state: BdfState {
            time: next_time,
            values: candidate,
            previous_values: Some(state.values.clone()),
            accepted_steps: state.accepted_steps + 1,
        },
        suggested_step,
        error_estimate,
        events,
    }))
}

fn validate_step(
    operator: &(impl DaeOperator + ?Sized),
    state: &BdfState,
    step: f64,
    config: &BdfConfig,
) -> Result<(), SolveError> {
    NumericError::require_len(
        "committed DAE state",
        state.values.len(),
        operator.dimension(),
    )?;
    NumericError::require_finite("committed DAE state", &state.values)?;
    if let Some(previous) = &state.previous_values {
        NumericError::require_len("previous DAE state", previous.len(), operator.dimension())?;
        NumericError::require_finite("previous DAE state", previous)?;
    }
    if !state.time.is_finite() || !step.is_finite() || step <= 0.0 {
        return Err(SolveError::InvalidConfiguration {
            reason: "DAE time and attempted step must be finite, with a positive step".into(),
        });
    }
    let tolerances_valid = config.absolute_tolerance.is_finite()
        && config.absolute_tolerance >= 0.0
        && config.relative_tolerance.is_finite()
        && config.relative_tolerance >= 0.0
        && (config.absolute_tolerance > 0.0 || config.relative_tolerance > 0.0);
    let limits_valid = config.minimum_step.is_finite()
        && config.minimum_step > 0.0
        && !config.maximum_step.is_nan()
        && config.maximum_step >= config.minimum_step;
    if !tolerances_valid || !limits_valid {
        return Err(SolveError::InvalidConfiguration {
            reason: "BDF tolerances and step limits must be positive and ordered".into(),
        });
    }
    Ok(())
}

fn implicit_step(
    operator: &(impl DaeOperator + ?Sized),
    context: &EvaluationContext,
    state: &BdfState,
    previous: Option<&Vec<f64>>,
    step: f64,
    order: u8,
    config: &BdfConfig,
) -> Result<Vec<f64>, SolveError> {
    struct ImplicitOperator<'a, Operator: DaeOperator + ?Sized> {
        operator: &'a Operator,
        state: &'a BdfState,
        previous: Option<&'a Vec<f64>>,
        step: f64,
        order: u8,
    }

    impl<Operator: DaeOperator + ?Sized> NonlinearOperator for ImplicitOperator<'_, Operator> {
        fn dimension(&self) -> usize {
            self.operator.dimension()
        }

        fn residual(
            &self,
            context: &EvaluationContext,
            values: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            let rate = bdf_derivative(values, self.state, self.previous, self.step, self.order);
            self.operator
                .residual(context, self.state.time + self.step, values, &rate, output)
        }

        fn jacobian_vector_product(
            &self,
            context: &EvaluationContext,
            values: &[f64],
            direction: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            let rate = bdf_derivative(values, self.state, self.previous, self.step, self.order);
            let alpha = if self.order == 2 { 1.5 } else { 1.0 } / self.step;
            let rate_direction = direction
                .iter()
                .map(|value| alpha * value)
                .collect::<Vec<_>>();
            self.operator.jacobian_vector_product(
                context,
                self.state.time + self.step,
                values,
                &rate,
                direction,
                &rate_direction,
                output,
            )
        }
    }

    let implicit = ImplicitOperator {
        operator,
        state,
        previous,
        step,
        order,
    };
    let report = solve_newton(&implicit, context, &state.values, &config.newton)?;
    if report.converged {
        Ok(report.state)
    } else {
        Err(SolveError::NotConverged)
    }
}

fn bdf_derivative(
    values: &[f64],
    state: &BdfState,
    previous: Option<&Vec<f64>>,
    step: f64,
    order: u8,
) -> Vec<f64> {
    if order == 2 {
        let previous = previous.expect("BDF2 requires a previous state");
        values
            .iter()
            .zip(&state.values)
            .zip(previous)
            .map(|((next, current), earlier)| (1.5 * next - 2.0 * current + 0.5 * earlier) / step)
            .collect()
    } else {
        values
            .iter()
            .zip(&state.values)
            .map(|(next, current)| (next - current) / step)
            .collect()
    }
}

fn scaled_error(second: &[f64], first: &[f64], config: &BdfConfig) -> f64 {
    second
        .iter()
        .zip(first)
        .map(|(higher_order, lower_order)| {
            (higher_order - lower_order).abs()
                / (config.absolute_tolerance
                    + config.relative_tolerance * higher_order.abs().max(lower_order.abs()))
        })
        .fold(0.0, f64::max)
}

fn adapt_step(step: f64, error: f64, config: &BdfConfig) -> f64 {
    let factor = if error <= f64::EPSILON {
        2.0
    } else {
        (0.9 / error.sqrt()).clamp(0.2, 2.0)
    };
    (step * factor).clamp(config.minimum_step, config.maximum_step)
}

fn locate_events(
    start_time: f64,
    end_time: f64,
    before: &[f64],
    after: &[f64],
) -> Vec<LocatedEvent> {
    before
        .iter()
        .zip(after)
        .enumerate()
        .filter_map(|(index, (&value_before, &value_after))| {
            let crosses = value_before == 0.0
                || value_after == 0.0
                || value_before.is_sign_positive() != value_after.is_sign_positive();
            if !crosses || (value_before == 0.0 && value_after == 0.0) {
                return None;
            }
            let denominator = value_before.abs() + value_after.abs();
            let fraction = if denominator == 0.0 {
                0.0
            } else {
                value_before.abs() / denominator
            };
            Some(LocatedEvent {
                index,
                time: start_time + fraction * (end_time - start_time),
                value_before,
                value_after,
            })
        })
        .collect()
}
