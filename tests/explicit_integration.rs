use methodus::{
    Admissibility, ConvergenceSample, EvaluationContext, ExplicitConfig, ExplicitOperator,
    ExplicitRejection, ExplicitScheme, ExplicitState, ExplicitStepOutcome, NumericError,
    SolveError, StepLimit, StepPolicy, estimate_convergence_order, explicit_step,
    integrate_explicit,
};

const SCHEMES: [ExplicitScheme; 5] = [
    ExplicitScheme::ForwardEuler,
    ExplicitScheme::SspRk2,
    ExplicitScheme::SspRk3,
    ExplicitScheme::SspRk104,
    ExplicitScheme::Rk4,
];

fn context() -> EvaluationContext {
    EvaluationContext::reproducible()
}

fn fixed(scheme: ExplicitScheme, step: f64) -> ExplicitConfig {
    ExplicitConfig {
        scheme,
        step_policy: StepPolicy::Fixed { step },
    }
}

fn cfl(scheme: ExplicitScheme, target: f64, max: f64, min_step: f64) -> ExplicitConfig {
    ExplicitConfig {
        scheme,
        step_policy: StepPolicy::Cfl {
            target,
            max,
            min_step,
            max_step: f64::INFINITY,
        },
    }
}

fn accepted(outcome: ExplicitStepOutcome) -> methodus::ExplicitAcceptedStep {
    match outcome {
        ExplicitStepOutcome::Accepted(step) => step,
        ExplicitStepOutcome::Rejected(rejected) => panic!("rejected: {rejected:?}"),
    }
}

fn rejected(outcome: ExplicitStepOutcome) -> methodus::ExplicitRejectedStep {
    match outcome {
        ExplicitStepOutcome::Rejected(step) => step,
        ExplicitStepOutcome::Accepted(step) => panic!("accepted: {step:?}"),
    }
}

/// Linear damped oscillator `y' = [[-0.1, 1], [-1, -0.1]] y`.
struct DampedOscillator;

impl ExplicitOperator for DampedOscillator {
    fn dimension(&self) -> usize {
        2
    }

    fn rhs(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = -0.1 * state[0] + state[1];
        output[1] = -state[0] - 0.1 * state[1];
        Ok(())
    }
}

/// Nonlinear, non-autonomous `y' = -2 t y²`, `y(0) = 1`, exact `1/(1+t²)`.
struct Riccati;

impl ExplicitOperator for Riccati {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = -2.0 * time * state[0] * state[0];
        Ok(())
    }
}

fn final_error(
    operator: &dyn ExplicitOperator,
    initial: Vec<f64>,
    exact: &[f64],
    scheme: ExplicitScheme,
    steps: usize,
) -> f64 {
    let final_time = 2.0;
    let state = ExplicitState::initialize(operator, &context(), 0.0, initial).unwrap();
    let report = integrate_explicit(
        operator,
        &context(),
        &state,
        final_time,
        &fixed(scheme, final_time / steps as f64),
        steps + 1,
    )
    .unwrap();
    assert_eq!(
        report.steps.len(),
        steps,
        "{scheme:?} took a degenerate extra step"
    );
    assert_eq!(report.state.time, final_time);
    assert!(report.rejections.is_empty());
    report
        .state
        .values
        .iter()
        .zip(exact)
        .map(|(value, exact)| (value - exact).abs())
        .fold(0.0, f64::max)
}

#[test]
fn observed_order_matches_each_scheme_on_linear_and_nonlinear_odes() {
    let decay = (-0.2_f64).exp();
    let oscillator_exact = [decay * 2.0_f64.cos(), -decay * 2.0_f64.sin()];
    let riccati_exact = [1.0 / 5.0];
    for scheme in SCHEMES {
        for (name, operator, initial, exact) in [
            (
                "oscillator",
                &DampedOscillator as &dyn ExplicitOperator,
                vec![1.0, 0.0],
                &oscillator_exact[..],
            ),
            ("riccati", &Riccati, vec![1.0], &riccati_exact[..]),
        ] {
            let samples: Vec<_> = [20, 40, 80, 160]
                .into_iter()
                .map(|steps| ConvergenceSample {
                    resolution: 2.0 / steps as f64,
                    error: final_error(operator, initial.clone(), exact, scheme, steps),
                })
                .collect();
            let report = estimate_convergence_order(&samples).unwrap();
            let order = f64::from(scheme.order());
            println!(
                "{scheme:?} {name}: fitted {:.4}, pairs {:?}, errors {:?}",
                report.fitted_order,
                report.pair_orders,
                samples
                    .iter()
                    .map(|sample| sample.error)
                    .collect::<Vec<_>>()
            );
            assert!(
                (report.fitted_order - order).abs() <= 0.1,
                "{scheme:?} {name}: fitted order {}",
                report.fitted_order
            );
        }
    }
}

#[test]
fn a_fixed_step_lands_exactly_on_the_final_time_without_a_sliver_step() {
    let state = ExplicitState::initialize(&Riccati, &context(), 0.0, vec![1.0]).unwrap();
    let report = integrate_explicit(
        &Riccati,
        &context(),
        &state,
        1.0,
        &fixed(ExplicitScheme::SspRk3, 0.1),
        100,
    )
    .unwrap();
    assert_eq!(report.steps.len(), 10);
    assert_eq!(report.state.time, 1.0);
    assert_eq!(report.state.accepted_steps, 10);
    assert_eq!(report.steps[9].step_limit, StepLimit::FinalTime);
    assert!(
        report.steps[..9]
            .iter()
            .all(|step| step.step_limit == StepLimit::Fixed)
    );
}

/// First-order upwind advection `u_i' = -(a/h_i)(u_i - u_{i-1})` with
/// an optional inflow value instead of periodic wrap-around, and an optional
/// leak on cell 0 (a deliberately nonconservative mutation).
struct Upwind {
    widths: Vec<f64>,
    speed: f64,
    inflow: Option<fn(f64) -> f64>,
    declare_rates: bool,
    leak: f64,
    invariants: Vec<Vec<f64>>,
}

impl Upwind {
    fn new(widths: Vec<f64>, inflow: Option<fn(f64) -> f64>, declare_rates: bool) -> Self {
        Self {
            invariants: vec![widths.clone()],
            widths,
            speed: 1.0,
            inflow,
            declare_rates,
            leak: 0.0,
        }
    }

    fn uniform(cells: usize) -> Self {
        Self::new(vec![1.0 / cells as f64; cells], None, true)
    }
}

impl ExplicitOperator for Upwind {
    fn dimension(&self) -> usize {
        self.widths.len()
    }

    fn rhs(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let n = state.len();
        for i in 0..n {
            let upstream = match (i, self.inflow) {
                (0, Some(inflow)) => inflow(time),
                (0, None) => state[n - 1],
                _ => state[i - 1],
            };
            output[i] = -(self.speed / self.widths[i]) * (state[i] - upstream);
        }
        output[0] -= self.leak * state[0];
        Ok(())
    }

    fn stable_step(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
    ) -> Result<Option<f64>, NumericError> {
        let narrowest = self.widths.iter().copied().fold(f64::INFINITY, f64::min);
        Ok(Some(narrowest / self.speed))
    }

    fn linear_invariants(&self) -> &[Vec<f64>] {
        &self.invariants
    }

    fn invariant_rates(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<bool, NumericError> {
        if !self.declare_rates {
            return Ok(false);
        }
        output[0] = match self.inflow {
            Some(inflow) => self.speed * inflow(time) - self.speed * state[state.len() - 1],
            None => 0.0,
        };
        Ok(true)
    }
}

fn total_variation(values: &[f64]) -> f64 {
    let n = values.len();
    (0..n)
        .map(|i| (values[i] - values[(i + n - 1) % n]).abs())
        .sum()
}

fn square_wave(cells: usize) -> Vec<f64> {
    (0..cells)
        .map(|i| {
            let x = (i as f64 + 0.5) / cells as f64;
            if (0.25..0.5).contains(&x) { 1.0 } else { 0.0 }
        })
        .collect()
}

#[test]
fn ssp_schemes_at_their_ssp_cfl_are_tvd_and_bounded_on_upwind_advection() {
    let cells = 100;
    let operator = Upwind::uniform(cells);
    for scheme in [
        ExplicitScheme::ForwardEuler,
        ExplicitScheme::SspRk2,
        ExplicitScheme::SspRk3,
        ExplicitScheme::SspRk104,
    ] {
        let coefficient = scheme.ssp_coefficient().unwrap();
        let config = cfl(scheme, coefficient, coefficient, 1e-12);
        let mut state =
            ExplicitState::initialize(&operator, &context(), 0.0, square_wave(cells)).unwrap();
        let mut variation = total_variation(&state.values);
        assert_eq!(variation, 2.0);
        let mut steps = 0;
        while state.time < 0.5 {
            let step =
                accepted(explicit_step(&operator, &context(), &state, 0.5, &config, None).unwrap());
            let next = total_variation(&step.state.values);
            assert!(
                next <= variation + 1e-13,
                "{scheme:?} step {steps}: TV {variation} -> {next}"
            );
            let (low, high) = step
                .state
                .values
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            assert!(
                low >= -1e-15 && high <= 1.0 + 1e-15,
                "{scheme:?}: [{low}, {high}]"
            );
            assert_eq!(step.report.ssp_coefficient, Some(coefficient));
            if step.report.step_limit == StepLimit::Cfl {
                let realized = step.report.cfl_realized.unwrap();
                assert!((realized - coefficient).abs() <= 1e-14 * coefficient);
            }
            variation = next;
            state = step.state;
            steps += 1;
        }
        if scheme != ExplicitScheme::ForwardEuler {
            // Forward Euler at CFL 1 is an exact shift; the others diffuse.
            assert!(variation < 2.0, "{scheme:?} must diffuse the square wave");
        }
        println!("{scheme:?}: {steps} steps to t=0.5 at CFL {coefficient}, TV {variation}");
    }
}

#[test]
fn forward_euler_beyond_its_cfl_limit_increases_total_variation() {
    let cells = 100;
    let operator = Upwind::uniform(cells);
    let state = ExplicitState::initialize(&operator, &context(), 0.0, square_wave(cells)).unwrap();
    let config = fixed(ExplicitScheme::ForwardEuler, 1.5 / cells as f64);
    let step = accepted(explicit_step(&operator, &context(), &state, 1.0, &config, None).unwrap());
    assert_eq!(step.report.cfl_realized, Some(1.5));
    assert!(total_variation(&step.state.values) > 2.0 + 0.1);
}

fn nonuniform_widths(cells: usize) -> Vec<f64> {
    (0..cells)
        .map(|i| {
            let x = (i as f64 + 0.5) / cells as f64;
            (1.0 + 0.5 * (2.0 * std::f64::consts::PI * x).sin()) / cells as f64
        })
        .collect()
}

fn smooth_profile(cells: usize) -> Vec<f64> {
    (0..cells)
        .map(|i| 1.0 + 0.5 * (2.0 * std::f64::consts::PI * (i as f64 + 0.5) / cells as f64).cos())
        .collect()
}

#[test]
fn conservative_operator_invariant_drift_is_round_off_per_step_and_cumulatively() {
    let cells = 64;
    let operator = Upwind::new(nonuniform_widths(cells), None, true);
    for scheme in SCHEMES {
        let state =
            ExplicitState::initialize(&operator, &context(), 0.0, smooth_profile(cells)).unwrap();
        let config = cfl(
            scheme,
            0.9 * scheme.ssp_coefficient().unwrap_or(1.0),
            6.0,
            1e-12,
        );
        let report =
            integrate_explicit(&operator, &context(), &state, 2.0, &config, 10_000).unwrap();
        // A priori round-off budget per step: one ε unit per summand of the
        // two invariant sums plus four per stage of stage arithmetic.
        let k = (cells + 4 * scheme.stages()) as f64;
        let mut worst = 0.0_f64;
        let mut cumulative_bound = 0.0;
        for step in &report.steps {
            assert!(step.balance_corrected);
            let balance = &step.invariants[0];
            let bound = k * f64::EPSILON * balance.scale;
            assert!(
                balance.balance_residual.abs() <= bound,
                "{scheme:?}: {balance:?} exceeds {bound}"
            );
            worst = worst.max(balance.balance_residual.abs() / (f64::EPSILON * balance.scale));
            cumulative_bound += bound;
        }
        let first = report.steps[0].invariants[0].before;
        let last = report.steps.last().unwrap().invariants[0].after;
        assert!((last - first).abs() <= cumulative_bound);
        println!(
            "{scheme:?}: {} steps, worst per-step |residual| = {worst:.2} ε·scale (budget {k}), \
             total drift {:.3e} (budget {cumulative_bound:.3e})",
            report.steps.len(),
            (last - first).abs()
        );
    }
}

#[test]
fn a_leaking_operator_fails_the_balance_by_orders_of_magnitude() {
    let cells = 64;
    let mut operator = Upwind::new(nonuniform_widths(cells), None, true);
    operator.leak = 1e-3;
    let state =
        ExplicitState::initialize(&operator, &context(), 0.0, smooth_profile(cells)).unwrap();
    let config = cfl(ExplicitScheme::SspRk3, 0.9, 1.0, 1e-12);
    let step = accepted(explicit_step(&operator, &context(), &state, 2.0, &config, None).unwrap());
    let balance = &step.report.invariants[0];
    let bound = (cells + 12) as f64 * f64::EPSILON * balance.scale;
    assert!(balance.balance_residual.abs() > 1e6 * bound, "{balance:?}");
}

fn inflow(time: f64) -> f64 {
    1.0 + 0.5 * (3.0 * time).sin()
}

#[test]
fn declared_boundary_rates_correct_the_balance_of_an_open_system() {
    let cells = 64;
    let corrected = Upwind::new(nonuniform_widths(cells), Some(inflow), true);
    let uncorrected = Upwind::new(nonuniform_widths(cells), Some(inflow), false);
    let config = cfl(ExplicitScheme::SspRk3, 0.9, 1.0, 1e-12);
    for (operator, declared) in [(&corrected, true), (&uncorrected, false)] {
        let state =
            ExplicitState::initialize(operator, &context(), 0.0, smooth_profile(cells)).unwrap();
        let report =
            integrate_explicit(operator, &context(), &state, 1.0, &config, 10_000).unwrap();
        let worst = report
            .steps
            .iter()
            .map(|step| {
                assert_eq!(step.balance_corrected, declared);
                let balance = &step.invariants[0];
                balance.balance_residual.abs() / (f64::EPSILON * balance.scale)
            })
            .fold(0.0, f64::max);
        if declared {
            assert!(
                worst <= (cells + 12) as f64,
                "corrected worst {worst} ε·scale"
            );
        } else {
            assert!(worst > 1e6, "uncorrected worst {worst} ε·scale");
        }
    }
}

/// `y' = -1` with admissible set `y ≥ 0` and a constant stable step 0.4.
struct Drain;

impl ExplicitOperator for Drain {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = -1.0;
        Ok(())
    }

    fn stable_step(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
    ) -> Result<Option<f64>, NumericError> {
        Ok(Some(0.4))
    }

    fn admissibility(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        state: &[f64],
    ) -> Result<Admissibility, NumericError> {
        Ok(if state[0] >= 0.0 {
            Admissibility::Admissible
        } else {
            Admissibility::Inadmissible {
                index: Some(0),
                reason: "negative".into(),
            }
        })
    }
}

#[test]
fn an_inadmissible_stage_rejects_the_step_and_a_retry_at_the_suggestion_is_accepted() {
    let state = ExplicitState::initialize(&Drain, &context(), 0.0, vec![0.25]).unwrap();
    let config = cfl(ExplicitScheme::ForwardEuler, 1.0, 1.0, 0.05);
    let first = rejected(explicit_step(&Drain, &context(), &state, 2.0, &config, None).unwrap());
    assert_eq!(first.attempted_step, 0.4);
    assert_eq!(first.suggested_step, Some(0.2));
    assert!(matches!(
        first.rejection,
        ExplicitRejection::InadmissibleStage {
            stage: 1,
            index: Some(0),
            ..
        }
    ));
    let retry = accepted(
        explicit_step(
            &Drain,
            &context(),
            &state,
            2.0,
            &config,
            first.suggested_step,
        )
        .unwrap(),
    );
    assert_eq!(retry.report.step_limit, StepLimit::RetryCeiling);
    assert_eq!(retry.state.values, vec![0.25 - 0.2]);
    assert_eq!(retry.state.accepted_steps, 1);

    // RK4 forms an intermediate stage at y - Δt (stage 3) that is inadmissible
    // although stages 1 and 2 are not.
    let rk4 = cfl(ExplicitScheme::Rk4, 1.0, 1.0, 0.05);
    let stage = rejected(explicit_step(&Drain, &context(), &state, 2.0, &rk4, None).unwrap());
    assert!(matches!(
        stage.rejection,
        ExplicitRejection::InadmissibleStage { stage: 3, .. }
    ));
}

#[test]
fn repeated_inadmissible_stages_end_in_a_typed_step_collapse() {
    let state = ExplicitState::initialize(&Drain, &context(), 0.0, vec![0.25]).unwrap();
    let config = cfl(ExplicitScheme::ForwardEuler, 1.0, 1.0, 0.05);
    let error = integrate_explicit(&Drain, &context(), &state, 2.0, &config, 1000).unwrap_err();
    assert!(
        matches!(error, SolveError::StepSizeCollapse { minimum_step, step, .. }
            if minimum_step == 0.05 && step < 0.05),
        "{error:?}"
    );

    let fixed_error = integrate_explicit(
        &Drain,
        &context(),
        &state,
        2.0,
        &fixed(ExplicitScheme::ForwardEuler, 0.4),
        1000,
    )
    .unwrap_err();
    assert!(matches!(
        fixed_error,
        SolveError::ExplicitStepRejected {
            time: 0.0,
            rejection: ExplicitRejection::InadmissibleStage { stage: 1, .. },
        }
    ));
    let budget = integrate_explicit(
        &Drain,
        &context(),
        &state,
        2.0,
        &cfl(ExplicitScheme::ForwardEuler, 1.0, 1.0, 1e-9),
        3,
    )
    .unwrap_err();
    assert_eq!(
        budget,
        SolveError::ExplicitStepBudgetExhausted {
            time: 0.2,
            attempts: 3
        }
    );
    assert!(matches!(
        ExplicitState::initialize(&Drain, &context(), 0.0, vec![-1.0]),
        Err(SolveError::InadmissibleState { index: Some(0), .. })
    ));
}

/// `y' = y` with a state-dependent stable step `1/|y|`.
struct Growth;

impl ExplicitOperator for Growth {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state[0];
        Ok(())
    }

    fn stable_step(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        state: &[f64],
    ) -> Result<Option<f64>, NumericError> {
        Ok(Some(1.0 / state[0].abs()))
    }
}

#[test]
fn a_stage_cfl_above_the_policy_maximum_rejects_with_a_smaller_suggestion() {
    let state = ExplicitState::initialize(&Growth, &context(), 0.0, vec![1.0]).unwrap();
    let config = cfl(ExplicitScheme::SspRk2, 1.0, 1.0, 1e-6);
    let first = rejected(explicit_step(&Growth, &context(), &state, 10.0, &config, None).unwrap());
    assert_eq!(
        first.rejection,
        ExplicitRejection::CflExceeded {
            stage: 1,
            cfl: 2.0,
            max: 1.0
        }
    );
    assert_eq!(first.suggested_step, Some(0.5));
    let retry = accepted(
        explicit_step(
            &Growth,
            &context(),
            &state,
            10.0,
            &config,
            first.suggested_step,
        )
        .unwrap(),
    );
    let realized = retry.report.cfl_realized.unwrap();
    assert!((realized - 0.75).abs() <= 1e-15, "{realized}");
    assert_eq!(retry.state.values, vec![1.625]);

    // Under a fixed step the same CFL is reported but not enforced.
    let free = accepted(
        explicit_step(
            &Growth,
            &context(),
            &state,
            10.0,
            &fixed(ExplicitScheme::SspRk2, 1.0),
            None,
        )
        .unwrap(),
    );
    assert_eq!(free.report.cfl_realized, Some(2.0));
}

struct Broken {
    rhs_value: f64,
    hint: Option<f64>,
}

impl ExplicitOperator for Broken {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = self.rhs_value;
        Ok(())
    }

    fn stable_step(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
    ) -> Result<Option<f64>, NumericError> {
        Ok(self.hint)
    }
}

#[test]
fn nonfinite_values_and_malformed_hints_are_refused_not_propagated() {
    let state = ExplicitState {
        time: 0.0,
        values: vec![1.0],
        accepted_steps: 0,
    };
    let fixed_config = fixed(ExplicitScheme::SspRk3, 0.1);
    let nan = Broken {
        rhs_value: f64::NAN,
        hint: None,
    };
    assert!(matches!(
        explicit_step(&nan, &context(), &state, 1.0, &fixed_config, None),
        Err(SolveError::Numeric(NumericError::NonFinite { .. }))
    ));
    let overflow = Broken {
        rhs_value: f64::MAX,
        hint: None,
    };
    assert!(matches!(
        explicit_step(
            &overflow,
            &context(),
            &state,
            10.0,
            &fixed(ExplicitScheme::SspRk2, 10.0),
            None
        ),
        Err(SolveError::Numeric(NumericError::NonFinite { .. }))
    ));
    let no_hint = Broken {
        rhs_value: 0.0,
        hint: None,
    };
    let cfl_config = cfl(ExplicitScheme::SspRk3, 0.5, 1.0, 1e-9);
    assert!(matches!(
        explicit_step(&no_hint, &context(), &state, 1.0, &cfl_config, None),
        Err(SolveError::InvalidConfiguration { .. })
    ));
    let negative_hint = Broken {
        rhs_value: 0.0,
        hint: Some(-1.0),
    };
    assert!(matches!(
        explicit_step(&negative_hint, &context(), &state, 1.0, &cfl_config, None),
        Err(SolveError::Numeric(NumericError::InvalidInput { .. }))
    ));
    let unconstrained = Broken {
        rhs_value: 0.0,
        hint: Some(f64::INFINITY),
    };
    let landed = accepted(
        explicit_step(&unconstrained, &context(), &state, 1.0, &cfl_config, None).unwrap(),
    );
    assert_eq!(landed.state.time, 1.0);
    assert_eq!(landed.report.step_limit, StepLimit::FinalTime);
    assert!(matches!(
        explicit_step(&no_hint, &context(), &state, 0.0, &fixed_config, None),
        Err(SolveError::Numeric(NumericError::InvalidInput { .. }))
    ));
}
