use solverang_contracts::{Ctx, NumericError};
use solverang_scientific::{bdf_step, BdfConfig, BdfOrder, BdfState, DaeOperator, StepOutcome};

struct DecayWithEvent;
impl DaeOperator for DecayWithEvent {
    fn dimension(&self) -> usize {
        1
    }

    fn residual(
        &self,
        _ctx: &Ctx,
        _t: f64,
        y: &[f64],
        ydot: &[f64],
        out: &mut [f64],
    ) -> Result<(), NumericError> {
        out[0] = ydot[0] + y[0];
        Ok(())
    }

    fn jvp(
        &self,
        _ctx: &Ctx,
        _t: f64,
        _y: &[f64],
        _ydot: &[f64],
        dy: &[f64],
        dydot: &[f64],
        out: &mut [f64],
    ) -> Result<(), NumericError> {
        out[0] = dydot[0] + dy[0];
        Ok(())
    }

    fn event_values(&self, _ctx: &Ctx, _t: f64, y: &[f64]) -> Result<Vec<f64>, NumericError> {
        Ok(vec![y[0] - 0.5])
    }
}

fn config(dt: f64) -> BdfConfig {
    BdfConfig {
        order: BdfOrder::Two,
        relative_error_tolerance: 1.0e12,
        absolute_error_tolerance: 1.0e12,
        min_dt: dt,
        max_dt: dt,
        ..BdfConfig::default()
    }
}

fn march(mut state: BdfState, steps: usize, dt: f64) -> (BdfState, Vec<f64>) {
    let ctx = Ctx::reproducible();
    let operator = DecayWithEvent;
    let config = config(dt);
    let mut events = Vec::new();
    for _ in 0..steps {
        match bdf_step(&operator, &ctx, &state, dt, &config).unwrap() {
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
    let ctx = Ctx::reproducible();
    let operator = DecayWithEvent;
    let initial = BdfState::initialize(&operator, &ctx, 0.0, vec![1.0]).unwrap();
    let (continuous, events_continuous) = march(initial.clone(), 10, 0.1);

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
    assert_eq!(events_continuous, restart_events);
}

#[test]
fn event_location_preserves_bdf_history() {
    let ctx = Ctx::reproducible();
    let operator = DecayWithEvent;
    let initial = BdfState::initialize(&operator, &ctx, 0.0, vec![1.0]).unwrap();
    let mut state = initial;
    let config = config(0.1);
    let mut saw_event = false;

    for _ in 0..10 {
        let previous_y = state.y.clone();
        let accepted = match bdf_step(&operator, &ctx, &state, 0.1, &config).unwrap() {
            StepOutcome::Accepted(step) => step,
            StepOutcome::Rejected(step) => panic!("fixed step rejected: {step:?}"),
        };
        if !accepted.events.is_empty() {
            saw_event = true;
            assert_eq!(
                accepted.state.previous.as_deref(),
                Some(previous_y.as_slice())
            );
            assert!(accepted
                .events
                .iter()
                .all(|event| { event.time >= state.t && event.time <= accepted.state.t }));
        }
        state = accepted.state;
    }
    assert!(saw_event, "decay trajectory must cross y=0.5");
}
