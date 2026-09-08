use methodus::{
    BdfConfig, BdfOrder, BdfState, DaeOperator, EvaluationContext, NumericError, StepOutcome,
    bdf_step,
};

struct Decay;

impl DaeOperator for Decay {
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
}

struct Quadratic;

impl DaeOperator for Quadratic {
    fn dimension(&self) -> usize {
        1
    }

    fn residual(
        &self,
        _context: &EvaluationContext,
        time: f64,
        _state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state_rate[0] - 2.0 * time;
        Ok(())
    }

    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
        _state_rate: &[f64],
        _state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = rate_direction[0];
        Ok(())
    }
}

fn integrate(order: BdfOrder, step: f64) -> f64 {
    let context = EvaluationContext::reproducible();
    let operator = Decay;
    let config = BdfConfig {
        order,
        // Disable adaptive rejection for this fixed-step convergence experiment.
        relative_tolerance: 1.0e12,
        absolute_tolerance: 1.0e12,
        minimum_step: step,
        maximum_step: step,
        ..BdfConfig::default()
    };
    let mut state = BdfState::initialize(&operator, &context, 0.0, vec![1.0]).unwrap();
    let steps = (1.0 / step).round() as usize;
    for _ in 0..steps {
        state = match bdf_step(&operator, &context, &state, step, &config).unwrap() {
            StepOutcome::Accepted(step) => step.state,
            StepOutcome::Rejected(step) => panic!("fixed-step convergence run rejected: {step:?}"),
        };
    }
    state.values[0]
}

#[test]
fn bdf1_has_first_order_temporal_convergence() {
    let exact = (-1.0_f64).exp();
    let coarse = (integrate(BdfOrder::One, 0.1) - exact).abs();
    let fine = (integrate(BdfOrder::One, 0.05) - exact).abs();
    assert!(coarse / fine > 1.7, "BDF1 ratio {}", coarse / fine);
}

#[test]
fn bdf2_has_second_order_temporal_convergence_after_startup() {
    let exact = (-1.0_f64).exp();
    let coarse = (integrate(BdfOrder::Two, 0.1) - exact).abs();
    let fine = (integrate(BdfOrder::Two, 0.05) - exact).abs();
    assert!(coarse / fine > 3.0, "BDF2 ratio {}", coarse / fine);
}

#[test]
fn bdf2_uses_the_previous_step_size_for_variable_steps() {
    let context = EvaluationContext::reproducible();
    let operator = Quadratic;
    let state = BdfState {
        time: 1.0,
        values: vec![1.0],
        previous_values: Some(vec![0.0]),
        previous_step: Some(1.0),
        accepted_steps: 1,
    };
    let config = BdfConfig {
        order: BdfOrder::Two,
        relative_tolerance: 1.0e12,
        absolute_tolerance: 1.0e12,
        minimum_step: 2.0,
        maximum_step: 2.0,
        ..BdfConfig::default()
    };

    let StepOutcome::Accepted(accepted) =
        bdf_step(&operator, &context, &state, 2.0, &config).unwrap()
    else {
        panic!("variable-step BDF2 attempt was rejected");
    };
    assert!((accepted.state.values[0] - 9.0).abs() < 1.0e-12);
    assert_eq!(accepted.state.previous_step, Some(2.0));
}

#[test]
fn rejected_step_returns_bit_identical_committed_state() {
    let context = EvaluationContext::reproducible();
    let operator = Decay;
    let initial = BdfState::initialize(&operator, &context, 0.0, vec![1.0]).unwrap();
    let config = BdfConfig {
        order: BdfOrder::Two,
        relative_tolerance: 1.0e-16,
        absolute_tolerance: 1.0e-16,
        minimum_step: 1.0e-6,
        maximum_step: 1.0,
        ..BdfConfig::default()
    };

    let seeded = BdfState {
        time: 0.1,
        values: vec![1.0 / 1.1],
        previous_values: Some(initial.values.clone()),
        previous_step: Some(0.1),
        accepted_steps: 1,
    };
    let before = serde_json::to_vec(&seeded).unwrap();
    let outcome = bdf_step(&operator, &context, &seeded, 0.1, &config).unwrap();
    let StepOutcome::Rejected(rejected) = outcome else {
        panic!("expected strict tolerance to reject the BDF2 attempt");
    };
    assert_eq!(
        serde_json::to_vec(&rejected.committed_state).unwrap(),
        before
    );
}

#[test]
fn deserialization_rejects_incomplete_bdf_history() {
    let malformed = r#"{
        "time": 1.0,
        "values": [1.0],
        "previous_values": [0.0],
        "previous_step": null,
        "accepted_steps": 1
    }"#;
    assert!(serde_json::from_str::<BdfState>(malformed).is_err());
}

#[test]
fn candidate_rate_uses_configured_order_and_unequal_step_history() {
    let state = BdfState {
        time: 1.0,
        values: vec![1.0],
        previous_values: Some(vec![0.25]),
        previous_step: Some(0.5),
        accepted_steps: 1,
    };
    let candidate = [1.5625]; // t² at t = 1.25
    let rate = methodus::bdf_candidate_rate(&state, &candidate, 0.25, BdfOrder::Two).unwrap();
    assert!((rate[0] - 2.5).abs() < 1e-14);
    let first = methodus::bdf_candidate_rate(&state, &candidate, 0.25, BdfOrder::One).unwrap();
    assert_eq!(first, vec![2.25]);
    let bootstrap = BdfState {
        previous_values: None,
        previous_step: None,
        ..state.clone()
    };
    assert_eq!(
        methodus::bdf_candidate_rate(&bootstrap, &candidate, 0.25, BdfOrder::Two).unwrap(),
        first
    );
    let mut bad = state.clone();
    bad.previous_step = None;
    assert!(methodus::bdf_candidate_rate(&bad, &candidate, 0.25, BdfOrder::Two).is_err());
    for step in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(methodus::bdf_candidate_rate(&state, &candidate, step, BdfOrder::Two).is_err());
    }
    assert!(methodus::bdf_candidate_rate(&state, &[], 0.25, BdfOrder::Two).is_err());
    assert!(methodus::bdf_candidate_rate(&state, &[f64::NAN], 0.25, BdfOrder::Two).is_err());
    bad = state.clone();
    bad.previous_values = Some(vec![]);
    assert!(methodus::bdf_candidate_rate(&bad, &candidate, 0.25, BdfOrder::Two).is_err());
    bad = state.clone();
    bad.previous_values = Some(vec![f64::INFINITY]);
    assert!(methodus::bdf_candidate_rate(&bad, &candidate, 0.25, BdfOrder::Two).is_err());
    bad = state;
    bad.time = f64::MAX;
    assert!(methodus::bdf_candidate_rate(&bad, &candidate, 0.25, BdfOrder::Two).is_err());
}

#[test]
fn accepted_candidate_rate_satisfies_the_actual_implicit_solve() {
    let context = EvaluationContext::reproducible();
    for order in [BdfOrder::One, BdfOrder::Two] {
        let config = BdfConfig {
            order,
            absolute_tolerance: 1e12,
            relative_tolerance: 1e12,
            ..BdfConfig::default()
        };
        let mut state = BdfState::initialize(&Decay, &context, 0.0, vec![1.0]).unwrap();
        for step in [0.1, 0.15, 0.07] {
            let StepOutcome::Accepted(accepted) =
                bdf_step(&Decay, &context, &state, step, &config).unwrap()
            else {
                panic!("test disables rejection")
            };
            let rate =
                methodus::bdf_candidate_rate(&state, &accepted.state.values, step, order).unwrap();
            let mut residual = vec![0.0];
            Decay
                .residual(
                    &context,
                    accepted.state.time,
                    &accepted.state.values,
                    &rate,
                    &mut residual,
                )
                .unwrap();
            assert!(residual[0].abs() < 1e-12, "{order:?}: {residual:?}");
            state = accepted.state;
        }
    }
}
