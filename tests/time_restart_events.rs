use solverang::{
    BdfConfig, BdfOrder, BdfState, DaeOperator, EvaluationContext, NumericError, StepOutcome,
    bdf_step,
};

struct DecayWithEvent;

impl DaeOperator for DecayWithEvent {
    fn dimension(&self) -> usize {
        1
    }

    fn residual(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state_rate[0] + state[0];
        Ok(())
    }

    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
        _state_rate: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = rate_direction[0] + state_direction[0];
        Ok(())
    }

    fn event_count(&self) -> usize {
        1
    }

    fn event_values(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state[0] - 0.5;
        Ok(())
    }
}

fn config(step: f64) -> BdfConfig {
    BdfConfig {
        order: BdfOrder::Two,
        relative_tolerance: 1.0e12,
        absolute_tolerance: 1.0e12,
        minimum_step: step,
        maximum_step: step,
        ..BdfConfig::default()
    }
}

fn march(mut state: BdfState, steps: usize, step: f64) -> (BdfState, Vec<f64>) {
    let context = EvaluationContext::reproducible();
    let operator = DecayWithEvent;
    let config = config(step);
    let mut events = Vec::new();
    for _ in 0..steps {
        match bdf_step(&operator, &context, &state, step, &config).unwrap() {
            StepOutcome::Accepted(step) => {
                events.extend(step.events.iter().map(|event| event.time));
                state = step.state;
            }
            StepOutcome::Rejected(step) => panic!("fixed-step acceptance run rejected: {step:?}"),
        }
    }
    (state, events)
}

#[test]
fn serialized_checkpoint_restart_is_trajectory_identical() {
    let context = EvaluationContext::reproducible();
    let operator = DecayWithEvent;
    let initial = BdfState::initialize(&operator, &context, 0.0, vec![1.0]).unwrap();
    let (continuous, continuous_events) = march(initial.clone(), 10, 0.1);

    let (midpoint, first_events) = march(initial, 4, 0.1);
    let bytes = serde_json::to_vec(&midpoint).unwrap();
    let restarted: BdfState = serde_json::from_slice(&bytes).unwrap();
    let (restarted, second_events) = march(restarted, 6, 0.1);

    assert_eq!(
        serde_json::to_vec(&continuous).unwrap(),
        serde_json::to_vec(&restarted).unwrap()
    );
    let mut restart_events = first_events;
    restart_events.extend(second_events);
    assert_eq!(continuous_events, restart_events);
}

#[test]
fn event_location_preserves_bdf_history() {
    let context = EvaluationContext::reproducible();
    let operator = DecayWithEvent;
    let mut state = BdfState::initialize(&operator, &context, 0.0, vec![1.0]).unwrap();
    let config = config(0.1);
    let mut saw_event = false;

    for _ in 0..10 {
        let previous_values = state.values.clone();
        let accepted = match bdf_step(&operator, &context, &state, 0.1, &config).unwrap() {
            StepOutcome::Accepted(step) => step,
            StepOutcome::Rejected(step) => panic!("fixed step rejected: {step:?}"),
        };
        if !accepted.events.is_empty() {
            saw_event = true;
            assert_eq!(
                accepted.state.previous_values.as_deref(),
                Some(previous_values.as_slice())
            );
            assert!(
                accepted
                    .events
                    .iter()
                    .all(|event| { event.time >= state.time && event.time <= accepted.state.time })
            );
        }
        state = accepted.state;
    }
    assert!(saw_event, "decay trajectory must cross y=0.5");
}
