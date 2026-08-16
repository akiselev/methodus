use solverang_contracts::{Ctx, NumericError};
use solverang_scientific::{
    BdfConfig, BdfOrder, BdfState, DaeOperator, StepOutcome, bdf_step,
};

struct Decay;
impl DaeOperator for Decay {
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
}

fn integrate(order: BdfOrder, dt: f64) -> f64 {
    let ctx = Ctx::reproducible();
    let operator = Decay;
    let config = BdfConfig {
        order,
        // Disable adaptive rejection for this fixed-step convergence experiment.
        relative_error_tolerance: 1.0e12,
        absolute_error_tolerance: 1.0e12,
        min_dt: dt,
        max_dt: dt,
        ..BdfConfig::default()
    };
    let mut state = BdfState::initialize(&operator, &ctx, 0.0, vec![1.0]).unwrap();
    let steps = (1.0 / dt).round() as usize;
    for _ in 0..steps {
        state = match bdf_step(&operator, &ctx, &state, dt, &config).unwrap() {
            StepOutcome::Accepted(step) => step.state,
            StepOutcome::Rejected(step) => panic!("fixed-step convergence run rejected: {step:?}"),
        };
    }
    state.y[0]
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
fn rejected_step_returns_bit_identical_committed_state() {
    let ctx = Ctx::reproducible();
    let operator = Decay;
    let state = BdfState::initialize(&operator, &ctx, 0.0, vec![1.0]).unwrap();
    let config = BdfConfig {
        order: BdfOrder::Two,
        relative_error_tolerance: 1.0e-16,
        absolute_error_tolerance: 1.0e-16,
        min_dt: 1.0e-6,
        max_dt: 1.0,
        ..BdfConfig::default()
    };

    // Seed a prior state so BDF2's embedded BDF1/BDF2 difference can force rejection.
    let seeded = BdfState {
        t: 0.1,
        y: vec![1.0 / 1.1],
        previous: Some(state.y.clone()),
        accepted_steps: 1,
        rejected_steps: 0,
    };
    let before = serde_json::to_vec(&seeded).unwrap();
    let outcome = bdf_step(&operator, &ctx, &seeded, 0.1, &config).unwrap();
    let StepOutcome::Rejected(rejected) = outcome else {
        panic!("expected strict tolerance to reject the BDF2 attempt");
    };
    assert_eq!(serde_json::to_vec(&rejected.unchanged_state).unwrap(), before);
}
