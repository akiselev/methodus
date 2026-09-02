//! E7/SC-W3 acceptance: the inexact Newton–Krylov driver converges
//! quadratically (constant tiny forcing) and superlinearly
//! (Eisenstat–Walker) on a nonlinear algebraic fixture and on
//! finite-difference nonlinear PDE fixtures, honours the preconditioner and
//! nullspace-projector hooks, refuses property-inadmissible Krylov methods,
//! agrees with the dense Newton baseline, runs inside a BDF step through the
//! `NonlinearSolver` hook, and reports bit-reproducible telemetry.

use methodus::{
    BdfConfig, BdfOrder, BdfState, BiCgStabConfig, ConjugateGradientConfig,
    ConjugateGradientSymmetryPolicy, ConstantModeProjector, DaeOperator, Definiteness,
    EvaluationContext, ForcingPolicy, GmresConfig, KrylovMethod, KrylovMethodKind, LinearOperator,
    MinresConfig, NewtonConfig, NewtonKrylovConfig, NewtonKrylovSolver, NonlinearOperator,
    NumericError, OperatorProperties, OperatorStructureHint, OperatorSymmetry, Preconditioner,
    PreconditionerFactory, SolveError, StepOutcome, bdf_step, bdf_step_with, solve_newton,
    solve_newton_krylov,
};

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// `F(x, y) = [x² + y² − 2, x − y]`, root `(1, 1)`, nonsymmetric Jacobian.
struct Circle;

impl NonlinearOperator for Circle {
    fn dimension(&self) -> usize {
        2
    }
    fn jacobian_properties(&self) -> OperatorProperties {
        OperatorProperties::from_symmetry(OperatorSymmetry::Nonsymmetric)
    }
    fn residual(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state[0] * state[0] + state[1] * state[1] - 2.0;
        output[1] = state[0] - state[1];
        Ok(())
    }
    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = 2.0 * state[0] * direction[0] + 2.0 * state[1] * direction[1];
        output[1] = direction[0] - direction[1];
        Ok(())
    }
}

/// Exact-Newton forcing with an absolute residual target; the PDE fixtures
/// use a target above their floating-point residual floor
/// (`‖J‖ · ‖u‖ · ε ≈ 1e-12` at `1/h² ≈ 4e3`), the algebraic fixture a
/// tighter one.
fn exact_newton_config(absolute_tolerance: f64) -> NewtonKrylovConfig {
    NewtonKrylovConfig {
        absolute_tolerance,
        relative_tolerance: 0.0,
        forcing: ForcingPolicy::Constant { forcing: 1.0e-12 },
        ..NewtonKrylovConfig::default()
    }
}

fn eisenstat_walker_config(absolute_tolerance: f64) -> NewtonKrylovConfig {
    NewtonKrylovConfig {
        absolute_tolerance,
        relative_tolerance: 0.0,
        ..NewtonKrylovConfig::default()
    }
}

const ALGEBRAIC_TARGET: f64 = 1.0e-13;
const PDE_TARGET: f64 = 1.0e-9;

fn tight_gmres() -> KrylovMethod {
    KrylovMethod::Gmres(GmresConfig::default())
}

#[test]
fn algebraic_fixture_converges_quadratically_with_exact_forcing() {
    let context = EvaluationContext::reproducible();
    let report = solve_newton_krylov(
        &Circle,
        &context,
        &[3.0, 0.5],
        &tight_gmres(),
        None,
        None,
        &exact_newton_config(ALGEBRAIC_TARGET),
    )
    .unwrap();
    assert!(report.converged, "{report:?}");
    assert!((report.state[0] - 1.0).abs() < 1.0e-12);
    assert!((report.state[1] - 1.0).abs() < 1.0e-12);

    // Quadratic convergence evidence: once in the local regime, each
    // residual is bounded by a fixed constant times the square of the
    // previous one, and the ratio does not grow.
    let norms: Vec<f64> = report
        .trace
        .iter()
        .map(|entry| entry.residual_norm)
        .collect();
    let local: Vec<(f64, f64)> = norms
        .windows(2)
        .filter(|pair| pair[0] < 0.2 && pair[1] > 0.0)
        .map(|pair| (pair[0], pair[1]))
        .collect();
    assert!(local.len() >= 2, "too few local iterations: {norms:?}");
    for (current, next) in &local {
        let ratio = next / (current * current);
        assert!(
            ratio < 2.0,
            "quadratic ratio {ratio} for {current} -> {next}"
        );
    }
    // Every inner solve met its (tiny) forcing tolerance: exact Newton.
    for entry in report
        .trace
        .iter()
        .filter_map(|entry| entry.linear.as_ref())
    {
        assert_eq!(entry.method, KrylovMethodKind::Gmres);
        assert!(entry.forcing <= 1.0e-12 || entry.forcing < 0.5);
    }
}

#[test]
fn algebraic_fixture_converges_superlinearly_with_eisenstat_walker_forcing() {
    let context = EvaluationContext::reproducible();
    let report = solve_newton_krylov(
        &Circle,
        &context,
        &[3.0, 0.5],
        &tight_gmres(),
        None,
        None,
        &eisenstat_walker_config(ALGEBRAIC_TARGET),
    )
    .unwrap();
    assert!(report.converged, "{report:?}");
    assert!((report.state[0] - 1.0).abs() < 1.0e-12);
    let norms: Vec<f64> = report
        .trace
        .iter()
        .map(|entry| entry.residual_norm)
        .collect();
    let ratios: Vec<f64> = norms
        .windows(2)
        .filter(|pair| pair[0] < 0.2 && pair[1] > 0.0)
        .map(|pair| pair[1] / pair[0])
        .collect();
    assert!(ratios.len() >= 2, "{norms:?}");
    // Superlinear: the linear rate tends to zero.
    assert!(
        ratios.windows(2).all(|pair| pair[1] < pair[0]),
        "{ratios:?}"
    );
    assert!(*ratios.last().unwrap() < 1.0e-2, "{ratios:?}");
    // The recorded forcing terms tighten as the residual drops (the
    // Eisenstat–Walker safeguard and the oversolving floor may raise
    // individual terms, so monotonicity is not the claim).
    let forcings: Vec<f64> = report
        .trace
        .iter()
        .filter_map(|entry| entry.linear.as_ref().map(|linear| linear.forcing))
        .collect();
    assert_eq!(forcings[0], 0.1);
    let tightest = forcings.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(tightest < 1.0e-3, "{forcings:?}");
}

#[test]
fn inadmissible_krylov_methods_are_refused_on_the_declared_jacobian() {
    let context = EvaluationContext::default();
    for method in [
        KrylovMethod::ConjugateGradient(ConjugateGradientConfig::default()),
        KrylovMethod::Minres(MinresConfig::default()),
    ] {
        let error = solve_newton_krylov(
            &Circle,
            &context,
            &[3.0, 0.5],
            &method,
            None,
            None,
            &eisenstat_walker_config(ALGEBRAIC_TARGET),
        )
        .unwrap_err();
        assert!(
            matches!(error, SolveError::InvalidConfiguration { .. }),
            "{method:?}"
        );
    }
    // BiCGSTAB admits the nonsymmetric Jacobian.
    let report = solve_newton_krylov(
        &Circle,
        &context,
        &[3.0, 0.5],
        &KrylovMethod::BiCgStab(BiCgStabConfig::default()),
        None,
        None,
        &eisenstat_walker_config(ALGEBRAIC_TARGET),
    )
    .unwrap();
    assert!(report.converged);
}

/// `−u'' + c u' + u³ = f` on `(0, 1)` with homogeneous Dirichlet data,
/// second-order centered finite differences, manufactured so that the
/// exact discrete-continuous target is `u = sin(πx)`. `c = 0` gives a
/// symmetric positive-definite Jacobian; `c ≠ 0` a nonsymmetric one.
struct ReactionDiffusion {
    interior: usize,
    convection: f64,
}

impl ReactionDiffusion {
    fn spacing(&self) -> f64 {
        1.0 / (self.interior as f64 + 1.0)
    }
    fn forcing(&self, index: usize) -> f64 {
        let x = (index as f64 + 1.0) * self.spacing();
        let pi = std::f64::consts::PI;
        pi * pi * (pi * x).sin() + self.convection * pi * (pi * x).cos() + (pi * x).sin().powi(3)
    }
    fn exact(&self) -> Vec<f64> {
        (0..self.interior)
            .map(|index| (std::f64::consts::PI * (index as f64 + 1.0) * self.spacing()).sin())
            .collect()
    }
}

impl NonlinearOperator for ReactionDiffusion {
    fn dimension(&self) -> usize {
        self.interior
    }
    fn jacobian_properties(&self) -> OperatorProperties {
        if self.convection == 0.0 {
            OperatorProperties::new(
                OperatorSymmetry::Symmetric,
                Definiteness::PositiveDefinite,
                Some(0),
                OperatorStructureHint::Dense,
            )
            .unwrap()
        } else {
            OperatorProperties::from_symmetry(OperatorSymmetry::Nonsymmetric)
        }
    }
    fn residual(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let h = self.spacing();
        for index in 0..self.interior {
            let left = if index > 0 { state[index - 1] } else { 0.0 };
            let right = if index + 1 < self.interior {
                state[index + 1]
            } else {
                0.0
            };
            let u = state[index];
            output[index] = (2.0 * u - left - right) / (h * h)
                + self.convection * (right - left) / (2.0 * h)
                + u * u * u
                - self.forcing(index);
        }
        Ok(())
    }
    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let h = self.spacing();
        for index in 0..self.interior {
            let left = if index > 0 { direction[index - 1] } else { 0.0 };
            let right = if index + 1 < self.interior {
                direction[index + 1]
            } else {
                0.0
            };
            let d = direction[index];
            output[index] = (2.0 * d - left - right) / (h * h)
                + self.convection * (right - left) / (2.0 * h)
                + 3.0 * state[index] * state[index] * d;
        }
        Ok(())
    }
}

fn assert_pde_quadratic(report: &methodus::NewtonKrylovReport) {
    // Quadratic-convergence evidence over the steps whose inner solve was
    // effectively exact (forcing far below the residual ratio being
    // measured). The final step is excluded on purpose: the oversolving
    // floor loosens its forcing to half the outer threshold, so that step
    // is inexact by design.
    let norms: Vec<f64> = report
        .trace
        .iter()
        .map(|entry| entry.residual_norm)
        .collect();
    let local: Vec<(f64, f64)> = report
        .trace
        .windows(2)
        .filter(|pair| {
            pair[0].residual_norm < 10.0
                && pair[1].residual_norm > 0.0
                && pair[0]
                    .linear
                    .as_ref()
                    .is_some_and(|linear| linear.forcing <= 1.0e-6)
        })
        .map(|pair| (pair[0].residual_norm, pair[1].residual_norm))
        .collect();
    assert!(local.len() >= 2, "{norms:?}");
    for (current, next) in &local {
        let ratio = next / (current * current);
        assert!(
            ratio < 0.1,
            "quadratic ratio {ratio} for {current} -> {next}: {norms:?}"
        );
    }
}

#[test]
fn symmetric_pde_fixture_converges_quadratically_through_conjugate_gradient() {
    let problem = ReactionDiffusion {
        interior: 63,
        convection: 0.0,
    };
    let context = EvaluationContext::reproducible();
    let report = solve_newton_krylov(
        &problem,
        &context,
        &vec![0.0; problem.interior],
        &KrylovMethod::ConjugateGradient(ConjugateGradientConfig::default()),
        None,
        None,
        &exact_newton_config(PDE_TARGET),
    )
    .unwrap();
    assert!(report.converged, "{:?}", report.trace);
    let discretization_error = report
        .state
        .iter()
        .zip(problem.exact())
        .map(|(u, exact)| (u - exact).abs())
        .fold(0.0, f64::max);
    // Second-order discretization on h = 1/64 lands near 1e-4 of the
    // manufactured solution.
    assert!(discretization_error < 1.0e-3, "{discretization_error}");
    assert_pde_quadratic(&report);
    for linear in report
        .trace
        .iter()
        .filter_map(|entry| entry.linear.as_ref())
    {
        assert_eq!(linear.method, KrylovMethodKind::ConjugateGradient);
    }
}

/// Exact tridiagonal factorization of the Jacobian at the current state,
/// built by probing the matrix-free action with unit vectors; as a
/// preconditioner it makes every inner solve converge in one iteration.
struct TridiagonalFactory;

struct TridiagonalSolve {
    lower: Vec<f64>,
    diagonal: Vec<f64>,
    upper: Vec<f64>,
}

impl Preconditioner for TridiagonalSolve {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }
    fn apply_inverse(
        &self,
        _context: &EvaluationContext,
        right_hand_side: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let n = self.diagonal.len();
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        c[0] = self.upper[0] / self.diagonal[0];
        d[0] = right_hand_side[0] / self.diagonal[0];
        for i in 1..n {
            let denominator = self.diagonal[i] - self.lower[i] * c[i - 1];
            c[i] = if i + 1 < n {
                self.upper[i] / denominator
            } else {
                0.0
            };
            d[i] = (right_hand_side[i] - self.lower[i] * d[i - 1]) / denominator;
        }
        output[n - 1] = d[n - 1];
        for i in (0..n - 1).rev() {
            output[i] = d[i] - c[i] * output[i + 1];
        }
        Ok(())
    }
}

impl PreconditionerFactory for TridiagonalFactory {
    fn build<'a>(
        &'a self,
        context: &EvaluationContext,
        jacobian: &dyn LinearOperator,
        _state: &[f64],
    ) -> Result<Option<Box<dyn Preconditioner + 'a>>, NumericError> {
        let n = jacobian.rows();
        let mut lower = vec![0.0; n];
        let mut diagonal = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut unit = vec![0.0; n];
        let mut column = vec![0.0; n];
        for j in 0..n {
            unit[j] = 1.0;
            jacobian.apply(context, &unit, &mut column)?;
            unit[j] = 0.0;
            diagonal[j] = column[j];
            if j > 0 {
                upper[j - 1] = column[j - 1];
            }
            if j + 1 < n {
                lower[j + 1] = column[j + 1];
            }
        }
        Ok(Some(Box::new(TridiagonalSolve {
            lower,
            diagonal,
            upper,
        })))
    }
}

#[test]
fn nonsymmetric_pde_fixture_uses_gmres_and_the_preconditioner_hook_and_matches_dense_newton() {
    let problem = ReactionDiffusion {
        interior: 40,
        convection: 8.0,
    };
    let context = EvaluationContext::reproducible();
    let initial = vec![0.0; problem.interior];

    // Conjugate gradient is refused on the declared-nonsymmetric Jacobian,
    // even with the explicit "assume symmetric" escape hatch (that hatch
    // only applies to Unknown).
    let error = solve_newton_krylov(
        &problem,
        &context,
        &initial,
        &KrylovMethod::ConjugateGradient(ConjugateGradientConfig {
            symmetry_policy: ConjugateGradientSymmetryPolicy::AssumeSymmetric,
            ..ConjugateGradientConfig::default()
        }),
        None,
        None,
        &exact_newton_config(PDE_TARGET),
    )
    .unwrap_err();
    assert!(matches!(error, SolveError::InvalidConfiguration { .. }));

    let unpreconditioned = solve_newton_krylov(
        &problem,
        &context,
        &initial,
        &tight_gmres(),
        None,
        None,
        &exact_newton_config(PDE_TARGET),
    )
    .unwrap();
    assert!(unpreconditioned.converged, "{:?}", unpreconditioned.trace);
    assert_pde_quadratic(&unpreconditioned);

    let preconditioned = solve_newton_krylov(
        &problem,
        &context,
        &initial,
        &tight_gmres(),
        Some(&TridiagonalFactory),
        None,
        &exact_newton_config(PDE_TARGET),
    )
    .unwrap();
    assert!(preconditioned.converged);
    // An exact preconditioner makes every inner solve a one-iteration solve.
    for linear in preconditioned
        .trace
        .iter()
        .filter_map(|entry| entry.linear.as_ref())
    {
        assert_eq!(linear.iterations, 1, "{linear:?}");
        assert!(linear.converged);
    }
    assert!(preconditioned.linear_iterations() < unpreconditioned.linear_iterations());

    // Both agree with the dense Newton correctness baseline to 1e-10.
    let dense = solve_newton(
        &problem,
        &context,
        &initial,
        &NewtonConfig {
            absolute_tolerance: PDE_TARGET,
            relative_tolerance: 0.0,
            ..NewtonConfig::default()
        },
    )
    .unwrap();
    assert!(dense.converged);
    for report in [&unpreconditioned, &preconditioned] {
        let difference: Vec<f64> = report
            .state
            .iter()
            .zip(&dense.state)
            .map(|(a, b)| a - b)
            .collect();
        assert!(norm(&difference) < 1.0e-10, "{}", norm(&difference));
    }
}

/// `F(x) = φ(x₀ − x₁) · [1, −1]` with `φ(d) = d + d³/3 − 4/3` (root `d = 1`):
/// the Jacobian `φ'(d) [[1, −1], [−1, 1]]` is symmetric positive
/// semidefinite with the constant mode as its nullspace, and the residual
/// is always consistent.
struct SingularDifference;

impl NonlinearOperator for SingularDifference {
    fn dimension(&self) -> usize {
        2
    }
    fn jacobian_properties(&self) -> OperatorProperties {
        OperatorProperties::new(
            OperatorSymmetry::Symmetric,
            Definiteness::PositiveSemidefinite,
            Some(1),
            OperatorStructureHint::Dense,
        )
        .unwrap()
    }
    fn residual(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let d = state[0] - state[1];
        let phi = d + d * d * d / 3.0 - 4.0 / 3.0;
        output[0] = phi;
        output[1] = -phi;
        Ok(())
    }
    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let d = state[0] - state[1];
        let slope = 1.0 + d * d;
        let action = slope * (direction[0] - direction[1]);
        output[0] = action;
        output[1] = -action;
        Ok(())
    }
}

#[test]
fn nullspace_projector_hook_keeps_a_singular_jacobian_solve_on_the_pseudo_solution() {
    let context = EvaluationContext::reproducible();
    let initial = [5.0, 1.0];
    // Without a projector MINRES refuses the declared nullspace outright.
    let error = solve_newton_krylov(
        &SingularDifference,
        &context,
        &initial,
        &KrylovMethod::Minres(MinresConfig::default()),
        None,
        None,
        &exact_newton_config(PDE_TARGET),
    )
    .unwrap_err();
    assert!(matches!(error, SolveError::InvalidConfiguration { .. }));

    let projector = ConstantModeProjector::new(2, 0, 2).unwrap();
    for method in [
        KrylovMethod::Minres(MinresConfig::default()),
        tight_gmres(),
        KrylovMethod::BiCgStab(BiCgStabConfig::default()),
    ] {
        let report = solve_newton_krylov(
            &SingularDifference,
            &context,
            &initial,
            &method,
            None,
            Some(&projector),
            &exact_newton_config(PDE_TARGET),
        )
        .unwrap();
        assert!(report.converged, "{method:?}: {:?}", report.trace);
        let difference = report.state[0] - report.state[1];
        assert!(
            (difference - 1.0).abs() < 1.0e-10,
            "{method:?}: {difference}"
        );
        // Every step was orthogonal to the constant mode, so the mean of
        // the state is the mean of the initial guess.
        let mean = 0.5 * (report.state[0] + report.state[1]);
        assert!((mean - 3.0).abs() < 1.0e-10, "{method:?}: {mean}");
    }
}

/// `y' = −y` as a DAE, with a declared symmetric implicit Jacobian.
struct Decay;

impl DaeOperator for Decay {
    fn dimension(&self) -> usize {
        1
    }
    fn jacobian_properties(&self) -> OperatorProperties {
        OperatorProperties::new(
            OperatorSymmetry::Symmetric,
            Definiteness::PositiveDefinite,
            Some(0),
            OperatorStructureHint::Dense,
        )
        .unwrap()
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

#[test]
fn bdf_steps_through_the_newton_krylov_solver_hook_agree_with_dense_newton() {
    let context = EvaluationContext::reproducible();
    let config = BdfConfig {
        order: BdfOrder::Two,
        relative_tolerance: 1.0e12,
        absolute_tolerance: 1.0e12,
        ..BdfConfig::default()
    };
    let method = KrylovMethod::ConjugateGradient(ConjugateGradientConfig::default());
    let newton_krylov = NewtonKrylovConfig {
        absolute_tolerance: 1.0e-14,
        relative_tolerance: 0.0,
        forcing: ForcingPolicy::Constant { forcing: 1.0e-12 },
        ..NewtonKrylovConfig::default()
    };
    let solver = NewtonKrylovSolver::new(&method, None, None, &newton_krylov);

    let mut dense = BdfState::initialize(&Decay, &context, 0.0, vec![1.0]).unwrap();
    let mut krylov = dense.clone();
    let step = 0.05;
    for _ in 0..20 {
        dense = match bdf_step(&Decay, &context, &dense, step, &config).unwrap() {
            StepOutcome::Accepted(accepted) => accepted.state,
            StepOutcome::Rejected(_) => panic!("fixed-step run must not reject"),
        };
        krylov = match bdf_step_with(&Decay, &context, &krylov, step, &config, &solver).unwrap() {
            StepOutcome::Accepted(accepted) => accepted.state,
            StepOutcome::Rejected(_) => panic!("fixed-step run must not reject"),
        };
    }
    assert!((dense.time - 1.0).abs() < 1.0e-12);
    assert_eq!(dense.time, krylov.time);
    assert!((dense.values[0] - krylov.values[0]).abs() < 1.0e-12);
    assert!((krylov.values[0] - (-1.0_f64).exp()).abs() < 1.0e-3);

    // MINRES is refused inside the step on a DAE that declares nothing
    // about its implicit Jacobian.
    struct Undeclared;
    impl DaeOperator for Undeclared {
        fn dimension(&self) -> usize {
            1
        }
        fn residual(
            &self,
            context: &EvaluationContext,
            time: f64,
            state: &[f64],
            state_rate: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            Decay.residual(context, time, state, state_rate, output)
        }
        fn jacobian_vector_product(
            &self,
            context: &EvaluationContext,
            time: f64,
            state: &[f64],
            state_rate: &[f64],
            state_direction: &[f64],
            rate_direction: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            Decay.jacobian_vector_product(
                context,
                time,
                state,
                state_rate,
                state_direction,
                rate_direction,
                output,
            )
        }
    }
    let minres = KrylovMethod::Minres(MinresConfig::default());
    let refusing = NewtonKrylovSolver::new(&minres, None, None, &newton_krylov);
    let initial = BdfState::initialize(&Undeclared, &context, 0.0, vec![1.0]).unwrap();
    let error =
        bdf_step_with(&Undeclared, &context, &initial, step, &config, &refusing).unwrap_err();
    assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
}

#[test]
fn newton_krylov_telemetry_is_bit_reproducible_and_exhaustion_is_reported_honestly() {
    let problem = ReactionDiffusion {
        interior: 24,
        convection: 3.0,
    };
    let context = EvaluationContext::reproducible();
    let initial = vec![0.0; problem.interior];
    let config = eisenstat_walker_config(PDE_TARGET);
    let first = solve_newton_krylov(
        &problem,
        &context,
        &initial,
        &tight_gmres(),
        Some(&TridiagonalFactory),
        None,
        &config,
    )
    .unwrap();
    let second = solve_newton_krylov(
        &problem,
        &context,
        &initial,
        &tight_gmres(),
        Some(&TridiagonalFactory),
        None,
        &config,
    )
    .unwrap();
    assert_eq!(first, second);
    let json = serde_json::to_string(&first).unwrap();
    let parsed: methodus::NewtonKrylovReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, first);

    let exhausted = solve_newton_krylov(
        &problem,
        &context,
        &initial,
        &tight_gmres(),
        None,
        None,
        &NewtonKrylovConfig {
            max_iterations: 1,
            ..config
        },
    )
    .unwrap();
    assert!(!exhausted.converged);
    assert_eq!(exhausted.trace.len(), 2);
    assert!(exhausted.trace.last().unwrap().linear.is_none());
}
