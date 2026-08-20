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
#[serde(try_from = "BdfStateData")]
pub struct BdfState {
    pub time: f64,
    pub values: Vec<f64>,
    pub previous_values: Option<Vec<f64>>,
    /// Size of the step between `previous_values` and `values`.
    pub previous_step: Option<f64>,
    pub accepted_steps: u64,
}

#[derive(Deserialize)]
struct BdfStateData {
    time: f64,
    values: Vec<f64>,
    previous_values: Option<Vec<f64>>,
    previous_step: Option<f64>,
    accepted_steps: u64,
}

impl TryFrom<BdfStateData> for BdfState {
    type Error = String;

    fn try_from(data: BdfStateData) -> Result<Self, Self::Error> {
        if !data.time.is_finite() {
            return Err("BDF state time must be finite".into());
        }
        NumericError::require_finite("BDF state", &data.values)
            .map_err(|error| error.to_string())?;
        match (&data.previous_values, data.previous_step) {
            (Some(previous), Some(previous_step)) => {
                NumericError::require_len("previous BDF state", previous.len(), data.values.len())
                    .map_err(|error| error.to_string())?;
                NumericError::require_finite("previous BDF state", previous)
                    .map_err(|error| error.to_string())?;
                if !previous_step.is_finite() || previous_step <= 0.0 {
                    return Err("previous BDF step must be finite and positive".into());
                }
            }
            (None, None) => {}
            _ => {
                return Err(
                    "previous BDF values and their step size must either both be present or both be absent"
                        .into(),
                );
            }
        }
        Ok(Self {
            time: data.time,
            values: data.values,
            previous_values: data.previous_values,
            previous_step: data.previous_step,
            accepted_steps: data.accepted_steps,
        })
    }
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
            previous_step: None,
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
    let next_time = state.time + step;
    let (candidate, error_estimate) = match (
        config.order,
        state.previous_values.as_deref(),
        state.previous_step,
    ) {
        (BdfOrder::Two, Some(previous), Some(previous_step)) => {
            let second = implicit_step(
                operator,
                context,
                state,
                Some((previous, previous_step)),
                step,
                next_time,
                config,
            )?;
            let first = implicit_step(operator, context, state, None, step, next_time, config)?;
            let error = scaled_error(&second, &first, config)?;
            (second, error)
        }
        _ => (
            implicit_step(operator, context, state, None, step, next_time, config)?,
            0.0,
        ),
    };
    let suggested_step = adapt_step(step, error_estimate, config)?;
    if error_estimate > 1.0 && step > config.minimum_step {
        return Ok(StepOutcome::Rejected(RejectedStep {
            committed_state: state.clone(),
            suggested_step,
            error_estimate,
        }));
    }

    let mut before = vec![0.0; operator.event_count()];
    let mut after = vec![0.0; operator.event_count()];
    operator.event_values(context, state.time, &state.values, &mut before)?;
    operator.event_values(context, next_time, &candidate, &mut after)?;
    NumericError::require_finite("DAE events before step", &before)?;
    NumericError::require_finite("DAE events after step", &after)?;
    let events = locate_events(state.time, next_time, &before, &after);
    let accepted_steps =
        state
            .accepted_steps
            .checked_add(1)
            .ok_or_else(|| SolveError::InvalidConfiguration {
                reason: "accepted BDF step count overflowed u64".into(),
            })?;
    Ok(StepOutcome::Accepted(AcceptedStep {
        state: BdfState {
            time: next_time,
            values: candidate,
            previous_values: Some(state.values.clone()),
            previous_step: Some(step),
            accepted_steps,
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
    match (&state.previous_values, state.previous_step) {
        (Some(previous), Some(previous_step)) => {
            NumericError::require_len("previous DAE state", previous.len(), operator.dimension())?;
            NumericError::require_finite("previous DAE state", previous)?;
            if !previous_step.is_finite() || previous_step <= 0.0 {
                return Err(SolveError::InvalidConfiguration {
                    reason: "previous BDF step must be finite and positive".into(),
                });
            }
        }
        (None, None) => {}
        _ => {
            return Err(SolveError::InvalidConfiguration {
                reason:
                    "previous BDF values and their step size must both be present or both be absent"
                        .into(),
            });
        }
    }
    if !state.time.is_finite() || !step.is_finite() || step <= 0.0 {
        return Err(SolveError::InvalidConfiguration {
            reason: "DAE time and attempted step must be finite, with a positive step".into(),
        });
    }
    let next_time = state.time + step;
    if !next_time.is_finite() || next_time <= state.time {
        return Err(SolveError::InvalidConfiguration {
            reason: "attempted BDF step must advance to a finite representable time".into(),
        });
    }
    if state.accepted_steps == u64::MAX {
        return Err(SolveError::InvalidConfiguration {
            reason: "accepted BDF step count cannot be incremented".into(),
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
    previous: Option<(&[f64], f64)>,
    step: f64,
    next_time: f64,
    config: &BdfConfig,
) -> Result<Vec<f64>, SolveError> {
    struct ImplicitOperator<'a, Operator: DaeOperator + ?Sized> {
        operator: &'a Operator,
        state: &'a BdfState,
        previous: Option<(&'a [f64], f64)>,
        step: f64,
        next_time: f64,
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
            let (rate, _) = bdf_derivative(values, self.state, self.previous, self.step)?;
            self.operator
                .residual(context, self.next_time, values, &rate, output)
        }

        fn jacobian_vector_product(
            &self,
            context: &EvaluationContext,
            values: &[f64],
            direction: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            let (rate, alpha) = bdf_derivative(values, self.state, self.previous, self.step)?;
            let rate_direction = direction
                .iter()
                .map(|value| alpha * value)
                .collect::<Vec<_>>();
            NumericError::require_finite("BDF rate direction", &rate_direction)?;
            self.operator.jacobian_vector_product(
                context,
                self.next_time,
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
        next_time,
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
    previous: Option<(&[f64], f64)>,
    step: f64,
) -> Result<(Vec<f64>, f64), NumericError> {
    let (next_coefficient, current_coefficient, previous_coefficient) =
        bdf_coefficients(step, previous.map(|(_, previous_step)| previous_step))?;
    let derivative: Vec<f64> = if let Some((previous, _)) = previous {
        values
            .iter()
            .zip(&state.values)
            .zip(previous)
            .map(|((next, current), earlier)| {
                next_coefficient * next
                    + current_coefficient * current
                    + previous_coefficient * earlier
            })
            .collect()
    } else {
        values
            .iter()
            .zip(&state.values)
            .map(|(next, current)| next_coefficient * next + current_coefficient * current)
            .collect()
    };
    NumericError::require_finite("BDF derivative", &derivative)?;
    Ok((derivative, next_coefficient))
}

fn bdf_coefficients(
    step: f64,
    previous_step: Option<f64>,
) -> Result<(f64, f64, f64), NumericError> {
    let coefficients = if let Some(previous_step) = previous_step {
        let ratio = step / previous_step;
        let denominator = 1.0 + ratio;
        let ratio_fraction = ratio / denominator;
        (
            (1.0 + ratio_fraction) / step,
            -denominator / step,
            ratio * ratio_fraction / step,
        )
    } else {
        (1.0 / step, -1.0 / step, 0.0)
    };
    if [coefficients.0, coefficients.1, coefficients.2]
        .iter()
        .all(|coefficient| coefficient.is_finite())
    {
        Ok(coefficients)
    } else {
        Err(NumericError::InvalidInput {
            message: "BDF step ratio produced non-finite derivative coefficients".into(),
        })
    }
}

fn scaled_error(second: &[f64], first: &[f64], config: &BdfConfig) -> Result<f64, NumericError> {
    second
        .iter()
        .zip(first)
        .try_fold(0.0_f64, |maximum, (higher_order, lower_order)| {
            let numerator = (higher_order - lower_order).abs();
            let denominator = config.absolute_tolerance
                + config.relative_tolerance * higher_order.abs().max(lower_order.abs());
            let error = numerator / denominator;
            if numerator.is_finite() && denominator.is_finite() && error.is_finite() {
                Ok(maximum.max(error))
            } else {
                Err(NumericError::InvalidInput {
                    message: "BDF error estimate overflowed".into(),
                })
            }
        })
}

fn adapt_step(step: f64, error: f64, config: &BdfConfig) -> Result<f64, SolveError> {
    let factor = if error <= f64::EPSILON {
        2.0
    } else {
        (0.9 / error.sqrt()).clamp(0.2, 2.0)
    };
    let scaled = step * factor;
    if !scaled.is_finite() {
        return Err(SolveError::InvalidConfiguration {
            reason: "adaptive BDF step calculation overflowed".into(),
        });
    }
    Ok(scaled.clamp(config.minimum_step, config.maximum_step))
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
