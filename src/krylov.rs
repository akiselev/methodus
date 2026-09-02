//! SV2-B6: MINRES and GMRES (plus E7's BiCGSTAB), physics-neutral iterative
//! linear solvers over the existing Methodus operator traits.
//!
//! Both solvers mirror [`crate::solve_conjugate_gradient`]'s refusal
//! machinery — typed refusal codes, never panics, never a silent fallback —
//! but admit a different shape of operator:
//!
//! - [`solve_minres`] requires a declared-[`OperatorSymmetry::Symmetric`]
//!   operator (indefinite allowed, unlike conjugate gradient — that is
//!   MINRES's point, e.g. a saddle-point Stokes system) and refuses both
//!   `Nonsymmetric` and `Unknown` declarations outright, with no
//!   caller-assumption escape hatch. A declared positive nullspace dimension
//!   is refused unless the caller supplies a
//!   [`crate::NullspaceProjector`], which is applied to keep every Krylov
//!   vector and the returned solution orthogonal to the declared nullspace
//!   (see [`crate::nullspace`]).
//! - [`solve_gmres`] accepts any declared symmetry and refuses only a
//!   genuinely unsupported shape (a non-square operator).
//! - [`solve_bicgstab`] (E7/SV1-D1) is the short-recurrence companion to
//!   GMRES for nonsymmetric adjoint and Jacobian actions; it admits any
//!   declared symmetry, refuses only a non-square operator, and reports
//!   Lanczos-type breakdowns as typed errors.
//!
//! Both report deterministic, bit-reproducible telemetry as typed
//! [`crate::LinearIteration`] traces, following a fixed reduction and update
//! order.

use serde::{Deserialize, Serialize};

use crate::context::EvaluationContext;
use crate::error::{NumericError, SolveError};
use crate::linear::{LinearIteration, LinearSolveReport, apply_preconditioner, dot, l2};
use crate::nullspace::NullspaceProjector;
use crate::operator::{LinearOperator, OperatorSymmetry, Preconditioner};

/// Convergence policy for a MINRES solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinresConfig {
    pub max_iterations: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}

impl Default for MinresConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-10,
        }
    }
}

/// Solve a symmetric (possibly indefinite) system through operator,
/// optional preconditioner, and optional nullspace-projector actions.
///
/// `residual_norm` in the returned trace is the Euclidean residual norm
/// when `preconditioner` is `None`; with a preconditioner it is the
/// preconditioner-weighted (Mⁿ¹) residual norm produced by the classical
/// Choi/Paige/Saunders MINRES recurrence, consistent with the algorithm's
/// internal convergence estimate.
///
/// # Errors
/// Refuses a non-square operator, a declared `Nonsymmetric` or `Unknown`
/// symmetry, a declared positive nullspace dimension with no supplied
/// [`NullspaceProjector`], dimension mismatches, non-finite input, and a
/// non-positive Lanczos inner product (`SolveError::KrylovBreakdown`).
pub fn solve_minres(
    operator: &(impl LinearOperator + ?Sized),
    preconditioner: Option<&dyn Preconditioner>,
    nullspace_projector: Option<&dyn NullspaceProjector>,
    context: &EvaluationContext,
    right_hand_side: &[f64],
    initial_solution: &[f64],
    config: &MinresConfig,
) -> Result<LinearSolveReport, SolveError> {
    validate_minres_config(config)?;
    let dimension = operator.rows();
    if operator.columns() != dimension {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "minres requires a square operator, got {}x{}",
                operator.rows(),
                operator.columns()
            ),
        });
    }
    match operator.symmetry() {
        OperatorSymmetry::Symmetric => {}
        OperatorSymmetry::Nonsymmetric => {
            return Err(SolveError::InvalidConfiguration {
                reason: "minres refuses an operator declared nonsymmetric".into(),
            });
        }
        OperatorSymmetry::Unknown => {
            return Err(SolveError::InvalidConfiguration {
                reason: "minres requires a declared Symmetric operator; unknown symmetry is \
                         refused outright, with no caller-assumption escape hatch"
                    .into(),
            });
        }
    }
    let declared_nullspace = operator.properties().nullspace_dimension().unwrap_or(0);
    if declared_nullspace > 0 && nullspace_projector.is_none() {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "minres refuses an operator with a declared nullspace dimension of \
                 {declared_nullspace} without a nullspace projector; supply one via \
                 NullspaceProjector to proceed"
            ),
        });
    }
    if let Some(projector) = nullspace_projector
        && projector.dimension() != dimension
    {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "nullspace projector dimension {} differs from operator dimension {dimension}",
                projector.dimension()
            ),
        });
    }
    NumericError::require_len("minres right-hand side", right_hand_side.len(), dimension)?;
    NumericError::require_len("initial minres solution", initial_solution.len(), dimension)?;
    NumericError::require_finite("minres right-hand side", right_hand_side)?;
    NumericError::require_finite("initial minres solution", initial_solution)?;
    if let Some(preconditioner) = preconditioner
        && preconditioner.dimension() != dimension
    {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "preconditioner dimension {} differs from operator dimension {dimension}",
                preconditioner.dimension()
            ),
        });
    }

    let mut x = initial_solution.to_vec();
    let mut r1 = vec![0.0; dimension];
    operator.apply(context, &x, &mut r1)?;
    NumericError::require_finite("initial minres operator action", &r1)?;
    for (residual, rhs) in r1.iter_mut().zip(right_hand_side) {
        *residual = rhs - *residual;
    }
    project(nullspace_projector, context, &mut r1)?;

    let mut y = vec![0.0; dimension];
    apply_preconditioner(preconditioner, context, &r1, &mut y)?;
    project(nullspace_projector, context, &mut y)?;

    let beta1_squared = dot(&r1, &y)?;
    if beta1_squared < 0.0 {
        return Err(SolveError::KrylovBreakdown { iteration: 0 });
    }
    let beta1 = beta1_squared.sqrt();
    let threshold = config.absolute_tolerance + config.relative_tolerance * beta1;
    let mut trace = vec![LinearIteration {
        iteration: 0,
        residual_norm: beta1,
    }];
    if beta1 <= threshold {
        return Ok(LinearSolveReport {
            solution: x,
            converged: true,
            trace,
        });
    }

    let mut r2 = r1.clone();
    let mut beta = beta1;
    let mut oldb = 0.0_f64;
    let mut dbar = 0.0_f64;
    let mut epsln = 0.0_f64;
    let mut cs = -1.0_f64;
    let mut sn = 0.0_f64;
    let mut phibar = beta1;
    let mut v = vec![0.0; dimension];
    let mut w = vec![0.0; dimension];
    let mut w1 = vec![0.0; dimension];
    let mut w2 = vec![0.0; dimension];

    for iteration in 1..=config.max_iterations {
        let s = 1.0 / beta;
        for (v, y) in v.iter_mut().zip(&y) {
            *v = s * y;
        }

        operator.apply(context, &v, &mut y)?;
        NumericError::require_finite("minres operator action", &y)?;
        project(nullspace_projector, context, &mut y)?;
        if iteration >= 2 {
            let ratio = beta / oldb;
            for (y, r1) in y.iter_mut().zip(&r1) {
                *y -= ratio * r1;
            }
        }
        let alfa = dot(&v, &y)?;
        for (y, r2) in y.iter_mut().zip(&r2) {
            *y -= (alfa / beta) * r2;
        }
        r1.copy_from_slice(&r2);
        r2.copy_from_slice(&y);
        apply_preconditioner(preconditioner, context, &r2, &mut y)?;
        project(nullspace_projector, context, &mut y)?;
        oldb = beta;
        let beta_squared = dot(&r2, &y)?;
        if beta_squared < 0.0 {
            return Err(SolveError::KrylovBreakdown { iteration });
        }
        beta = beta_squared.sqrt();

        let oldeps = epsln;
        let delta = cs * dbar + sn * alfa;
        let gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;

        let mut gamma = gbar.hypot(beta);
        if gamma < f64::EPSILON {
            gamma = f64::EPSILON;
        }
        cs = gbar / gamma;
        sn = beta / gamma;
        let phi = cs * phibar;
        phibar *= sn;

        let denom = 1.0 / gamma;
        std::mem::swap(&mut w1, &mut w2);
        std::mem::swap(&mut w2, &mut w);
        for (((w, v), w1), w2) in w.iter_mut().zip(&v).zip(&w1).zip(&w2) {
            *w = (v - oldeps * w1 - delta * w2) * denom;
        }
        for (x, w) in x.iter_mut().zip(&w) {
            *x += phi * w;
        }
        NumericError::require_finite("minres solution", &x)?;

        let residual_norm = phibar.abs();
        trace.push(LinearIteration {
            iteration,
            residual_norm,
        });
        if residual_norm <= threshold {
            project(nullspace_projector, context, &mut x)?;
            return Ok(LinearSolveReport {
                solution: x,
                converged: true,
                trace,
            });
        }
        if beta.abs() < 1.0e-13 {
            // Lanczos breakdown: the current Krylov subspace already spans
            // every direction reachable from the initial residual.
            break;
        }
    }

    project(nullspace_projector, context, &mut x)?;
    Ok(LinearSolveReport {
        solution: x,
        converged: false,
        trace,
    })
}

fn project(
    projector: Option<&dyn NullspaceProjector>,
    context: &EvaluationContext,
    vector: &mut [f64],
) -> Result<(), NumericError> {
    if let Some(projector) = projector {
        projector.project(context, vector)?;
    }
    Ok(())
}

fn validate_minres_config(config: &MinresConfig) -> Result<(), SolveError> {
    let tolerances_valid = config.absolute_tolerance.is_finite()
        && config.absolute_tolerance >= 0.0
        && config.relative_tolerance.is_finite()
        && config.relative_tolerance >= 0.0
        && (config.absolute_tolerance > 0.0 || config.relative_tolerance > 0.0);
    if config.max_iterations == 0 || !tolerances_valid {
        return Err(SolveError::InvalidConfiguration {
            reason: "minres iteration limit and tolerances must be positive and finite".into(),
        });
    }
    Ok(())
}

/// Convergence and restart policy for a GMRES solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GmresConfig {
    pub max_iterations: usize,
    pub restart: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}

impl Default for GmresConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            restart: 30,
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-10,
        }
    }
}

/// Final state and convergence evidence from a GMRES solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GmresReport {
    pub solution: Vec<f64>,
    pub converged: bool,
    pub trace: Vec<LinearIteration>,
    pub restart_cycles: usize,
}

/// Solve a general (possibly nonsymmetric) square system through operator
/// and optional left-preconditioner actions, using restarted GMRES with
/// modified Gram-Schmidt Arnoldi and incremental Givens-rotation QR.
///
/// `residual_norm` in the returned trace is the Euclidean residual norm
/// when `preconditioner` is `None`; with a preconditioner it is the
/// preconditioner-weighted (left-preconditioned) residual norm.
///
/// # Errors
/// Refuses a non-square operator (the only shape GMRES does not support),
/// invalid configuration, dimension mismatches, and non-finite input.
pub fn solve_gmres(
    operator: &(impl LinearOperator + ?Sized),
    preconditioner: Option<&dyn Preconditioner>,
    context: &EvaluationContext,
    right_hand_side: &[f64],
    initial_solution: &[f64],
    config: &GmresConfig,
) -> Result<GmresReport, SolveError> {
    validate_gmres_config(config)?;
    let dimension = operator.rows();
    if operator.columns() != dimension {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "gmres requires a square operator, got {}x{}",
                operator.rows(),
                operator.columns()
            ),
        });
    }
    NumericError::require_len("gmres right-hand side", right_hand_side.len(), dimension)?;
    NumericError::require_len("initial gmres solution", initial_solution.len(), dimension)?;
    NumericError::require_finite("gmres right-hand side", right_hand_side)?;
    NumericError::require_finite("initial gmres solution", initial_solution)?;
    if let Some(preconditioner) = preconditioner
        && preconditioner.dimension() != dimension
    {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "preconditioner dimension {} differs from operator dimension {dimension}",
                preconditioner.dimension()
            ),
        });
    }

    let restart = config.restart.min(dimension);
    let mut x = initial_solution.to_vec();
    let mut trace = Vec::new();
    let mut global_iteration = 0usize;
    let mut restart_cycles = 0usize;
    let mut threshold = None;
    let mut action = vec![0.0; dimension];
    let mut residual = vec![0.0; dimension];
    let mut preconditioned_residual = vec![0.0; dimension];

    loop {
        operator.apply(context, &x, &mut action)?;
        NumericError::require_finite("gmres operator action", &action)?;
        for ((residual, rhs), action) in residual.iter_mut().zip(right_hand_side).zip(&action) {
            *residual = rhs - action;
        }
        apply_preconditioner(
            preconditioner,
            context,
            &residual,
            &mut preconditioned_residual,
        )?;
        let beta = l2(&preconditioned_residual)?;
        let threshold =
            *threshold.get_or_insert(config.absolute_tolerance + config.relative_tolerance * beta);
        if global_iteration == 0 {
            trace.push(LinearIteration {
                iteration: 0,
                residual_norm: beta,
            });
        }
        if beta <= threshold {
            return Ok(GmresReport {
                solution: x,
                converged: true,
                trace,
                restart_cycles,
            });
        }
        if global_iteration >= config.max_iterations {
            return Ok(GmresReport {
                solution: x,
                converged: false,
                trace,
                restart_cycles,
            });
        }
        restart_cycles += 1;

        let mut basis: Vec<Vec<f64>> = Vec::with_capacity(restart + 1);
        let inv_beta = 1.0 / beta;
        basis.push(
            preconditioned_residual
                .iter()
                .map(|value| value * inv_beta)
                .collect(),
        );
        let mut hessenberg: Vec<Vec<f64>> = Vec::with_capacity(restart);
        let mut cs: Vec<f64> = Vec::with_capacity(restart);
        let mut sn: Vec<f64> = Vec::with_capacity(restart);
        let mut g = vec![0.0; restart + 1];
        g[0] = beta;
        let mut completed = 0usize;

        for j in 0..restart {
            if global_iteration >= config.max_iterations {
                break;
            }
            let mut action_j = vec![0.0; dimension];
            operator.apply(context, &basis[j], &mut action_j)?;
            NumericError::require_finite("gmres operator action", &action_j)?;
            let mut w = vec![0.0; dimension];
            apply_preconditioner(preconditioner, context, &action_j, &mut w)?;

            let mut column = vec![0.0; j + 2];
            for (i, basis_vector) in basis.iter().enumerate().take(j + 1) {
                let projection = dot(basis_vector, &w)?;
                column[i] = projection;
                for (w, basis) in w.iter_mut().zip(basis_vector) {
                    *w -= projection * basis;
                }
            }
            let h_next = l2(&w)?;
            column[j + 1] = h_next;

            for i in 0..j {
                let temp = cs[i] * column[i] + sn[i] * column[i + 1];
                column[i + 1] = -sn[i] * column[i] + cs[i] * column[i + 1];
                column[i] = temp;
            }

            let (rotation_cos, rotation_sin) = givens_rotation(column[j], column[j + 1]);
            let new_diagonal = rotation_cos * column[j] + rotation_sin * column[j + 1];
            let new_subdiagonal = -rotation_sin * column[j] + rotation_cos * column[j + 1];
            column[j] = new_diagonal;
            column[j + 1] = new_subdiagonal;
            cs.push(rotation_cos);
            sn.push(rotation_sin);
            hessenberg.push(column);

            let g_j = g[j];
            g[j] = rotation_cos * g_j;
            g[j + 1] = -rotation_sin * g_j;

            global_iteration += 1;
            let residual_norm = g[j + 1].abs();
            trace.push(LinearIteration {
                iteration: global_iteration,
                residual_norm,
            });
            completed = j + 1;

            if h_next.abs() < 1.0e-13 {
                // Arnoldi breakdown: the current Krylov subspace already
                // spans every direction reachable from the initial residual.
                break;
            }
            if residual_norm <= threshold {
                break;
            }
            if j + 1 < restart {
                let inv_h_next = 1.0 / h_next;
                basis.push(w.iter().map(|value| value * inv_h_next).collect());
            }
        }

        let mut y = vec![0.0; completed];
        for i in (0..completed).rev() {
            let mut sum = g[i];
            for (l, y_l) in y.iter().enumerate().take(completed).skip(i + 1) {
                sum -= hessenberg[l][i] * y_l;
            }
            y[i] = sum / hessenberg[i][i];
        }
        for (i, y_i) in y.iter().enumerate() {
            for (x, basis) in x.iter_mut().zip(&basis[i]) {
                *x += y_i * basis;
            }
        }
        NumericError::require_finite("gmres solution", &x)?;
    }
}

/// Computes `(cos, sin)` such that `cos*a + sin*b = hypot(a, b)` and
/// `-sin*a + cos*b = 0`, the standard Givens rotation used to eliminate one
/// Hessenberg subdiagonal entry.
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if a == 0.0 && b == 0.0 {
        (1.0, 0.0)
    } else {
        let r = a.hypot(b);
        (a / r, b / r)
    }
}

fn validate_gmres_config(config: &GmresConfig) -> Result<(), SolveError> {
    let tolerances_valid = config.absolute_tolerance.is_finite()
        && config.absolute_tolerance >= 0.0
        && config.relative_tolerance.is_finite()
        && config.relative_tolerance >= 0.0
        && (config.absolute_tolerance > 0.0 || config.relative_tolerance > 0.0);
    if config.max_iterations == 0 || config.restart == 0 || !tolerances_valid {
        return Err(SolveError::InvalidConfiguration {
            reason: "gmres iteration limit, restart length, and tolerances must be positive and \
                     finite"
                .into(),
        });
    }
    Ok(())
}

/// Convergence policy for a BiCGSTAB solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BiCgStabConfig {
    pub max_iterations: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}

impl Default for BiCgStabConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-10,
        }
    }
}

/// Solve a general (possibly nonsymmetric) square system with the
/// stabilized bi-conjugate gradient method (van der Vorst's BiCGSTAB), using
/// right preconditioning so that `residual_norm` in the returned trace is
/// always the true Euclidean residual `‖b − A x‖`, with or without a
/// preconditioner.
///
/// BiCGSTAB admits any declared symmetry and, like GMRES, refuses only a
/// non-square operator. It is a short-recurrence alternative to restarted
/// GMRES for adjoint and Newton–Krylov solves whose transpose or Jacobian
/// action is nonsymmetric. The Lanczos-type breakdowns of the method (a
/// vanishing shadow inner product `<r̂, r>`, `<r̂, v>`, or stabilization
/// denominator `<t, t>`) are reported as [`SolveError::KrylovBreakdown`],
/// never silently restarted.
///
/// # Errors
/// Refuses a non-square operator, invalid configuration, dimension
/// mismatches, non-finite input, and Lanczos breakdown.
pub fn solve_bicgstab(
    operator: &(impl LinearOperator + ?Sized),
    preconditioner: Option<&dyn Preconditioner>,
    context: &EvaluationContext,
    right_hand_side: &[f64],
    initial_solution: &[f64],
    config: &BiCgStabConfig,
) -> Result<LinearSolveReport, SolveError> {
    validate_bicgstab_config(config)?;
    let dimension = operator.rows();
    if operator.columns() != dimension {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "bicgstab requires a square operator, got {}x{}",
                operator.rows(),
                operator.columns()
            ),
        });
    }
    NumericError::require_len("bicgstab right-hand side", right_hand_side.len(), dimension)?;
    NumericError::require_len(
        "initial bicgstab solution",
        initial_solution.len(),
        dimension,
    )?;
    NumericError::require_finite("bicgstab right-hand side", right_hand_side)?;
    NumericError::require_finite("initial bicgstab solution", initial_solution)?;
    if let Some(preconditioner) = preconditioner
        && preconditioner.dimension() != dimension
    {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "preconditioner dimension {} differs from operator dimension {dimension}",
                preconditioner.dimension()
            ),
        });
    }

    let mut x = initial_solution.to_vec();
    let mut r = vec![0.0; dimension];
    operator.apply(context, &x, &mut r)?;
    NumericError::require_finite("initial bicgstab operator action", &r)?;
    for (residual, rhs) in r.iter_mut().zip(right_hand_side) {
        *residual = rhs - *residual;
    }
    let initial_norm = l2(&r)?;
    let threshold = config.absolute_tolerance + config.relative_tolerance * initial_norm;
    let mut trace = vec![LinearIteration {
        iteration: 0,
        residual_norm: initial_norm,
    }];
    if initial_norm <= threshold {
        return Ok(LinearSolveReport {
            solution: x,
            converged: true,
            trace,
        });
    }

    // The shadow residual is fixed at the initial residual, the standard
    // deterministic choice.
    let shadow = r.clone();
    let mut rho_old = 1.0_f64;
    let mut alpha = 1.0_f64;
    let mut omega = 1.0_f64;
    let mut p = vec![0.0; dimension];
    let mut v = vec![0.0; dimension];
    let mut p_hat = vec![0.0; dimension];
    let mut s = vec![0.0; dimension];
    let mut s_hat = vec![0.0; dimension];
    let mut t = vec![0.0; dimension];

    for iteration in 1..=config.max_iterations {
        let rho = dot(&shadow, &r)?;
        if rho == 0.0 {
            return Err(SolveError::KrylovBreakdown { iteration });
        }
        let beta = (rho / rho_old) * (alpha / omega);
        NumericError::require_finite("bicgstab recurrence", &[beta])?;
        for ((p, r), v) in p.iter_mut().zip(&r).zip(&v) {
            *p = r + beta * (*p - omega * v);
        }
        apply_preconditioner(preconditioner, context, &p, &mut p_hat)?;
        operator.apply(context, &p_hat, &mut v)?;
        NumericError::require_finite("bicgstab operator action", &v)?;
        let shadow_v = dot(&shadow, &v)?;
        if shadow_v == 0.0 {
            return Err(SolveError::KrylovBreakdown { iteration });
        }
        alpha = rho / shadow_v;
        NumericError::require_finite("bicgstab step", &[alpha])?;
        for ((s, r), v) in s.iter_mut().zip(&r).zip(&v) {
            *s = r - alpha * v;
        }
        for (x, p_hat) in x.iter_mut().zip(&p_hat) {
            *x += alpha * p_hat;
        }
        let half_norm = l2(&s)?;
        if half_norm <= threshold {
            NumericError::require_finite("bicgstab solution", &x)?;
            trace.push(LinearIteration {
                iteration,
                residual_norm: half_norm,
            });
            return Ok(LinearSolveReport {
                solution: x,
                converged: true,
                trace,
            });
        }
        apply_preconditioner(preconditioner, context, &s, &mut s_hat)?;
        operator.apply(context, &s_hat, &mut t)?;
        NumericError::require_finite("bicgstab operator action", &t)?;
        let t_t = dot(&t, &t)?;
        if t_t == 0.0 {
            return Err(SolveError::KrylovBreakdown { iteration });
        }
        omega = dot(&t, &s)? / t_t;
        NumericError::require_finite("bicgstab stabilization", &[omega])?;
        for (x, s_hat) in x.iter_mut().zip(&s_hat) {
            *x += omega * s_hat;
        }
        for ((r, s), t) in r.iter_mut().zip(&s).zip(&t) {
            *r = s - omega * t;
        }
        NumericError::require_finite("bicgstab solution", &x)?;
        let residual_norm = l2(&r)?;
        trace.push(LinearIteration {
            iteration,
            residual_norm,
        });
        if residual_norm <= threshold {
            return Ok(LinearSolveReport {
                solution: x,
                converged: true,
                trace,
            });
        }
        if omega == 0.0 {
            return Err(SolveError::KrylovBreakdown { iteration });
        }
        rho_old = rho;
    }

    Ok(LinearSolveReport {
        solution: x,
        converged: false,
        trace,
    })
}

fn validate_bicgstab_config(config: &BiCgStabConfig) -> Result<(), SolveError> {
    let tolerances_valid = config.absolute_tolerance.is_finite()
        && config.absolute_tolerance >= 0.0
        && config.relative_tolerance.is_finite()
        && config.relative_tolerance >= 0.0
        && (config.absolute_tolerance > 0.0 || config.relative_tolerance > 0.0);
    if config.max_iterations == 0 || !tolerances_valid {
        return Err(SolveError::InvalidConfiguration {
            reason: "bicgstab iteration limit and tolerances must be positive and finite".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CsrMatrix, Definiteness, OperatorProperties, OperatorStructureHint};

    struct DeclaredOperator {
        matrix: CsrMatrix,
        properties: OperatorProperties,
    }

    impl LinearOperator for DeclaredOperator {
        fn rows(&self) -> usize {
            self.matrix.rows()
        }

        fn columns(&self) -> usize {
            self.matrix.columns()
        }

        fn symmetry(&self) -> OperatorSymmetry {
            self.properties.symmetry()
        }

        fn properties(&self) -> OperatorProperties {
            self.properties.clone()
        }

        fn apply(
            &self,
            context: &EvaluationContext,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            self.matrix.apply(context, input, output)
        }
    }

    /// The classic minimal saddle-point system: a 2x2 SPD velocity block
    /// coupled to a single scalar constraint, giving an indefinite
    /// symmetric 3x3 system with a known solution.
    fn saddle_point_matrix() -> CsrMatrix {
        CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, 2.0),
                (0, 2, 1.0),
                (1, 1, 2.0),
                (1, 2, 1.0),
                (2, 0, 1.0),
                (2, 1, 1.0),
            ],
        )
        .unwrap()
    }

    #[test]
    fn minres_solves_a_symmetric_indefinite_saddle_point_system() {
        let matrix = saddle_point_matrix();
        assert_eq!(matrix.symmetry(), OperatorSymmetry::Symmetric);
        let report = solve_minres(
            &matrix,
            None,
            None,
            &EvaluationContext::reproducible(),
            &[3.0, 5.0, 4.0],
            &[0.0; 3],
            &MinresConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
        let expected = [1.5, 2.5, 0.0];
        for (actual, expected) in report.solution.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-8, "{actual} vs {expected}");
        }
    }

    #[test]
    fn minres_agrees_with_conjugate_gradient_on_an_spd_system() {
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, 4.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 4.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 3.0),
            ],
        )
        .unwrap();
        let context = EvaluationContext::reproducible();
        let rhs = [15.0, 10.0, 10.0];
        let cg_report = crate::solve_conjugate_gradient(
            &matrix,
            None,
            &context,
            &rhs,
            &[0.0; 3],
            &crate::ConjugateGradientConfig::default(),
        )
        .unwrap();
        let minres_report = solve_minres(
            &matrix,
            None,
            None,
            &context,
            &rhs,
            &[0.0; 3],
            &MinresConfig::default(),
        )
        .unwrap();
        assert!(minres_report.converged);
        for (cg, minres) in cg_report.solution.iter().zip(&minres_report.solution) {
            assert!((cg - minres).abs() < 1.0e-8, "{cg} vs {minres}");
        }
    }

    #[test]
    fn minres_refuses_declared_nonsymmetric_and_unknown_operators() {
        let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0), (1, 1, 2.0)]).unwrap();
        for symmetry in [OperatorSymmetry::Nonsymmetric, OperatorSymmetry::Unknown] {
            let operator = DeclaredOperator {
                matrix: CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0), (1, 1, 2.0)]).unwrap(),
                properties: OperatorProperties::from_symmetry(symmetry),
            };
            let error = solve_minres(
                &operator,
                None,
                None,
                &EvaluationContext::default(),
                &[1.0, 1.0],
                &[0.0; 2],
                &MinresConfig::default(),
            )
            .unwrap_err();
            assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
        }
        // Symmetric, indefinite is admitted (the whole point of MINRES).
        let operator = DeclaredOperator {
            matrix,
            properties: OperatorProperties::new(
                OperatorSymmetry::Symmetric,
                Definiteness::Indefinite,
                None,
                OperatorStructureHint::Dense,
            )
            .unwrap(),
        };
        let report = solve_minres(
            &operator,
            None,
            None,
            &EvaluationContext::default(),
            &[2.0, 4.0],
            &[0.0; 2],
            &MinresConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
    }

    #[test]
    fn minres_refuses_a_declared_nullspace_without_a_projector() {
        let operator = DeclaredOperator {
            matrix: CsrMatrix::from_triplets(
                2,
                2,
                vec![(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)],
            )
            .unwrap(),
            properties: OperatorProperties::new(
                OperatorSymmetry::Symmetric,
                Definiteness::PositiveSemidefinite,
                Some(1),
                OperatorStructureHint::Dense,
            )
            .unwrap(),
        };
        let error = solve_minres(
            &operator,
            None,
            None,
            &EvaluationContext::default(),
            &[1.0, -1.0],
            &[0.0; 2],
            &MinresConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
    }

    #[test]
    fn minres_projects_a_singular_consistent_system_onto_the_pseudo_solution() {
        use crate::ConstantModeProjector;

        let operator = DeclaredOperator {
            matrix: CsrMatrix::from_triplets(
                2,
                2,
                vec![(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)],
            )
            .unwrap(),
            properties: OperatorProperties::new(
                OperatorSymmetry::Symmetric,
                Definiteness::PositiveSemidefinite,
                Some(1),
                OperatorStructureHint::Dense,
            )
            .unwrap(),
        };
        let projector = ConstantModeProjector::new(2, 0, 2).unwrap();
        let report = solve_minres(
            &operator,
            None,
            Some(&projector),
            &EvaluationContext::default(),
            &[1.0, -1.0],
            &[0.0; 2],
            &MinresConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
        assert!((report.solution[0] - 0.5).abs() < 1.0e-8);
        assert!((report.solution[1] + 0.5).abs() < 1.0e-8);
    }

    #[test]
    fn minres_telemetry_is_bit_reproducible() {
        let matrix = saddle_point_matrix();
        let context = EvaluationContext::reproducible();
        let first = solve_minres(
            &matrix,
            None,
            None,
            &context,
            &[3.0, 5.0, 4.0],
            &[0.0; 3],
            &MinresConfig::default(),
        )
        .unwrap();
        let second = solve_minres(
            &matrix,
            None,
            None,
            &context,
            &[3.0, 5.0, 4.0],
            &[0.0; 3],
            &MinresConfig::default(),
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn minres_refuses_rectangular_operators() {
        let matrix = CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![1.0]).unwrap();
        let error = solve_minres(
            &matrix,
            None,
            None,
            &EvaluationContext::default(),
            &[1.0],
            &[0.0, 0.0],
            &MinresConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
    }

    fn nonsymmetric_matrix_with_known_solution() -> (CsrMatrix, Vec<f64>, Vec<f64>) {
        // A x = b for x = [1, 2, 3].
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, 4.0),
                (0, 1, 1.0),
                (1, 1, 3.0),
                (1, 2, 1.0),
                (2, 0, 1.0),
                (2, 2, 2.0),
            ],
        )
        .unwrap();
        (matrix, vec![6.0, 9.0, 7.0], vec![1.0, 2.0, 3.0])
    }

    #[test]
    fn gmres_solves_a_nonsymmetric_fixture() {
        let (matrix, rhs, expected) = nonsymmetric_matrix_with_known_solution();
        assert_eq!(matrix.symmetry(), OperatorSymmetry::Nonsymmetric);
        let report = solve_gmres(
            &matrix,
            None,
            &EvaluationContext::reproducible(),
            &rhs,
            &[0.0; 3],
            &GmresConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
        for (actual, expected) in report.solution.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1.0e-8, "{actual} vs {expected}");
        }
    }

    #[test]
    fn gmres_accepts_any_declared_symmetry_and_refuses_only_rectangular_operators() {
        let (matrix, rhs, _) = nonsymmetric_matrix_with_known_solution();
        for symmetry in [
            OperatorSymmetry::Unknown,
            OperatorSymmetry::Symmetric,
            OperatorSymmetry::Nonsymmetric,
        ] {
            let operator = DeclaredOperator {
                matrix: CsrMatrix::from_triplets(
                    3,
                    3,
                    vec![
                        (0, 0, 4.0),
                        (0, 1, 1.0),
                        (1, 1, 3.0),
                        (1, 2, 1.0),
                        (2, 0, 1.0),
                        (2, 2, 2.0),
                    ],
                )
                .unwrap(),
                properties: OperatorProperties::from_symmetry(symmetry),
            };
            let report = solve_gmres(
                &operator,
                None,
                &EvaluationContext::default(),
                &rhs,
                &[0.0; 3],
                &GmresConfig::default(),
            )
            .unwrap();
            assert!(report.converged);
        }
        let _ = matrix;

        let rectangular = CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![1.0]).unwrap();
        let error = solve_gmres(
            &rectangular,
            None,
            &EvaluationContext::default(),
            &[1.0],
            &[0.0, 0.0],
            &GmresConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
    }

    #[test]
    fn gmres_telemetry_is_bit_reproducible() {
        let (matrix, rhs, _) = nonsymmetric_matrix_with_known_solution();
        let context = EvaluationContext::reproducible();
        let first = solve_gmres(
            &matrix,
            None,
            &context,
            &rhs,
            &[0.0; 3],
            &GmresConfig::default(),
        )
        .unwrap();
        let second = solve_gmres(
            &matrix,
            None,
            &context,
            &rhs,
            &[0.0; 3],
            &GmresConfig::default(),
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn gmres_restarts_when_the_budget_exceeds_the_restart_length() {
        let (matrix, rhs, expected) = nonsymmetric_matrix_with_known_solution();
        let config = GmresConfig {
            restart: 1,
            ..GmresConfig::default()
        };
        let report = solve_gmres(
            &matrix,
            None,
            &EvaluationContext::reproducible(),
            &rhs,
            &[0.0; 3],
            &config,
        )
        .unwrap();
        assert!(report.converged);
        assert!(report.restart_cycles > 1);
        for (actual, expected) in report.solution.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1.0e-8, "{actual} vs {expected}");
        }
    }

    #[test]
    fn bicgstab_solves_a_nonsymmetric_fixture_and_reports_true_residuals() {
        let (matrix, rhs, expected) = nonsymmetric_matrix_with_known_solution();
        let context = EvaluationContext::reproducible();
        let report = solve_bicgstab(
            &matrix,
            None,
            &context,
            &rhs,
            &[0.0; 3],
            &BiCgStabConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
        for (actual, expected) in report.solution.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1.0e-8, "{actual} vs {expected}");
        }
        // The reported final residual is the true residual `‖b − A x‖`.
        let mut action = vec![0.0; 3];
        matrix
            .apply(&context, &report.solution, &mut action)
            .unwrap();
        let true_norm = rhs
            .iter()
            .zip(&action)
            .map(|(b, a)| (b - a).powi(2))
            .sum::<f64>()
            .sqrt();
        let reported = report.trace.last().unwrap().residual_norm;
        assert!(
            (true_norm - reported).abs() < 1.0e-9,
            "{true_norm} vs {reported}"
        );
    }

    #[test]
    fn bicgstab_with_a_preconditioner_still_reports_true_residuals() {
        struct DiagonalInverse(Vec<f64>);
        impl Preconditioner for DiagonalInverse {
            fn dimension(&self) -> usize {
                self.0.len()
            }
            fn apply_inverse(
                &self,
                _context: &EvaluationContext,
                right_hand_side: &[f64],
                output: &mut [f64],
            ) -> Result<(), NumericError> {
                for ((output, value), diagonal) in
                    output.iter_mut().zip(right_hand_side).zip(&self.0)
                {
                    *output = value / diagonal;
                }
                Ok(())
            }
        }
        let (matrix, rhs, expected) = nonsymmetric_matrix_with_known_solution();
        let context = EvaluationContext::reproducible();
        let preconditioner = DiagonalInverse(vec![4.0, 3.0, 2.0]);
        let report = solve_bicgstab(
            &matrix,
            Some(&preconditioner),
            &context,
            &rhs,
            &[0.0; 3],
            &BiCgStabConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
        for (actual, expected) in report.solution.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1.0e-8, "{actual} vs {expected}");
        }
        let mut action = vec![0.0; 3];
        matrix
            .apply(&context, &report.solution, &mut action)
            .unwrap();
        let true_norm = rhs
            .iter()
            .zip(&action)
            .map(|(b, a)| (b - a).powi(2))
            .sum::<f64>()
            .sqrt();
        let reported = report.trace.last().unwrap().residual_norm;
        assert!(
            (true_norm - reported).abs() < 1.0e-9,
            "{true_norm} vs {reported}"
        );
    }

    #[test]
    fn bicgstab_telemetry_is_bit_reproducible_and_refuses_rectangular_operators() {
        let (matrix, rhs, _) = nonsymmetric_matrix_with_known_solution();
        let context = EvaluationContext::reproducible();
        let first = solve_bicgstab(
            &matrix,
            None,
            &context,
            &rhs,
            &[0.0; 3],
            &BiCgStabConfig::default(),
        )
        .unwrap();
        let second = solve_bicgstab(
            &matrix,
            None,
            &context,
            &rhs,
            &[0.0; 3],
            &BiCgStabConfig::default(),
        )
        .unwrap();
        assert_eq!(first, second);

        let rectangular = CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![1.0]).unwrap();
        let error = solve_bicgstab(
            &rectangular,
            None,
            &EvaluationContext::default(),
            &[1.0],
            &[0.0, 0.0],
            &BiCgStabConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
    }
}
