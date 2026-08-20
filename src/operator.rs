use crate::{EvaluationContext, NumericError};

/// Matrix-free action of a rectangular linear operator.
pub trait LinearOperator: Send + Sync {
    fn rows(&self) -> usize;
    fn columns(&self) -> usize;
    fn apply(
        &self,
        context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
}

/// Approximate inverse action used by iterative linear solvers.
pub trait Preconditioner: Send + Sync {
    fn dimension(&self) -> usize;
    fn apply_inverse(
        &self,
        context: &EvaluationContext,
        right_hand_side: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
}

/// In-place residual and Jacobian-vector products for `F(x) = 0`.
pub trait NonlinearOperator: Send + Sync {
    fn dimension(&self) -> usize;
    fn residual(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
}

/// In-place residual and directional derivatives for `F(t, y, ydot) = 0`.
pub trait DaeOperator: Send + Sync {
    fn dimension(&self) -> usize;
    fn residual(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
    #[allow(clippy::too_many_arguments)]
    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;

    fn make_initial_state_consistent(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &mut [f64],
    ) -> Result<(), NumericError> {
        Ok(())
    }

    fn event_count(&self) -> usize {
        0
    }

    fn event_values(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        NumericError::require_len("DAE event output", output.len(), 0)
    }
}

/// Maximum absolute discrepancy between an analytic nonlinear JVP and a centered difference.
pub fn verify_jvp(
    operator: &(impl NonlinearOperator + ?Sized),
    context: &EvaluationContext,
    state: &[f64],
    direction: &[f64],
    epsilon: f64,
) -> Result<f64, NumericError> {
    let dimension = operator.dimension();
    validate_probe(dimension, state, direction, epsilon)?;
    let mut analytic = vec![0.0; dimension];
    operator.jacobian_vector_product(context, state, direction, &mut analytic)?;
    let plus = shifted(state, direction, epsilon);
    let minus = shifted(state, direction, -epsilon);
    let mut residual_plus = vec![0.0; dimension];
    let mut residual_minus = vec![0.0; dimension];
    operator.residual(context, &plus, &mut residual_plus)?;
    operator.residual(context, &minus, &mut residual_minus)?;
    discrepancy(&analytic, &residual_plus, &residual_minus, epsilon)
}

/// Maximum absolute discrepancy between an analytic DAE JVP and a centered difference.
#[allow(clippy::too_many_arguments)]
pub fn verify_dae_jvp(
    operator: &(impl DaeOperator + ?Sized),
    context: &EvaluationContext,
    time: f64,
    state: &[f64],
    state_rate: &[f64],
    state_direction: &[f64],
    rate_direction: &[f64],
    epsilon: f64,
) -> Result<f64, NumericError> {
    let dimension = operator.dimension();
    validate_probe(dimension, state, state_direction, epsilon)?;
    validate_probe(dimension, state_rate, rate_direction, epsilon)?;
    let mut analytic = vec![0.0; dimension];
    operator.jacobian_vector_product(
        context,
        time,
        state,
        state_rate,
        state_direction,
        rate_direction,
        &mut analytic,
    )?;
    let state_plus = shifted(state, state_direction, epsilon);
    let state_minus = shifted(state, state_direction, -epsilon);
    let rate_plus = shifted(state_rate, rate_direction, epsilon);
    let rate_minus = shifted(state_rate, rate_direction, -epsilon);
    let mut residual_plus = vec![0.0; dimension];
    let mut residual_minus = vec![0.0; dimension];
    operator.residual(context, time, &state_plus, &rate_plus, &mut residual_plus)?;
    operator.residual(
        context,
        time,
        &state_minus,
        &rate_minus,
        &mut residual_minus,
    )?;
    discrepancy(&analytic, &residual_plus, &residual_minus, epsilon)
}

fn validate_probe(
    dimension: usize,
    values: &[f64],
    direction: &[f64],
    epsilon: f64,
) -> Result<(), NumericError> {
    NumericError::require_len("verification state", values.len(), dimension)?;
    NumericError::require_len("verification direction", direction.len(), dimension)?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(NumericError::InvalidInput {
            message: "verification epsilon must be finite and positive".into(),
        });
    }
    Ok(())
}

fn shifted(values: &[f64], direction: &[f64], factor: f64) -> Vec<f64> {
    values
        .iter()
        .zip(direction)
        .map(|(value, delta)| value + factor * delta)
        .collect()
}

fn discrepancy(
    analytic: &[f64],
    residual_plus: &[f64],
    residual_minus: &[f64],
    epsilon: f64,
) -> Result<f64, NumericError> {
    NumericError::require_finite("analytic JVP", analytic)?;
    NumericError::require_finite("positive residual probe", residual_plus)?;
    NumericError::require_finite("negative residual probe", residual_minus)?;
    Ok(analytic
        .iter()
        .zip(residual_plus.iter().zip(residual_minus))
        .map(|(analytic, (plus, minus))| (analytic - (plus - minus) / (2.0 * epsilon)).abs())
        .fold(0.0, f64::max))
}
