//! E7/SV1-D1: one typed selector over the Krylov solvers Methodus ships,
//! consumed by the adjoint driver ([`crate::solve_adjoint`]) and the inexact
//! Newton–Krylov driver ([`crate::solve_newton_krylov`]).
//!
//! [`KrylovMethod`] carries the concrete per-method configuration so a
//! caller (Sinbad policy, Krasis transactions) selects the algorithm and its
//! tolerances in one serializable value. [`solve_krylov`] dispatches to the
//! underlying solver **without** loosening any of that solver's admission
//! rules: conjugate gradient still refuses declared-nonsymmetric operators,
//! MINRES still refuses anything not declared `Symmetric`, and GMRES/BiCGSTAB
//! still refuse only a non-square operator. The optional
//! [`NullspaceProjector`] hook is honoured natively by MINRES; for GMRES and
//! BiCGSTAB it selects the representative of a singular consistent system
//! by projecting the initial guess and the returned solution (the true
//! residual is unchanged by that projection, so acceptance stays honest);
//! conjugate gradient takes no projector and refuses one.

use serde::{Deserialize, Serialize};

use crate::context::EvaluationContext;
use crate::error::SolveError;
use crate::krylov::{
    BiCgStabConfig, GmresConfig, MinresConfig, solve_bicgstab, solve_gmres, solve_minres,
};
use crate::linear::{ConjugateGradientConfig, LinearIteration, solve_conjugate_gradient};
use crate::nullspace::NullspaceProjector;
use crate::operator::{LinearOperator, Preconditioner};

/// Which Krylov solver a [`KrylovMethod`] selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KrylovMethodKind {
    ConjugateGradient,
    Minres,
    Gmres,
    BiCgStab,
}

/// A Krylov solver selection together with its full configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KrylovMethod {
    ConjugateGradient(ConjugateGradientConfig),
    Minres(MinresConfig),
    Gmres(GmresConfig),
    BiCgStab(BiCgStabConfig),
}

impl KrylovMethod {
    #[must_use]
    pub const fn kind(&self) -> KrylovMethodKind {
        match self {
            Self::ConjugateGradient(_) => KrylovMethodKind::ConjugateGradient,
            Self::Minres(_) => KrylovMethodKind::Minres,
            Self::Gmres(_) => KrylovMethodKind::Gmres,
            Self::BiCgStab(_) => KrylovMethodKind::BiCgStab,
        }
    }

    /// The `(absolute, relative)` convergence tolerances of the selected
    /// solver.
    #[must_use]
    pub const fn tolerances(&self) -> (f64, f64) {
        match self {
            Self::ConjugateGradient(config) => {
                (config.absolute_tolerance, config.relative_tolerance)
            }
            Self::Minres(config) => (config.absolute_tolerance, config.relative_tolerance),
            Self::Gmres(config) => (config.absolute_tolerance, config.relative_tolerance),
            Self::BiCgStab(config) => (config.absolute_tolerance, config.relative_tolerance),
        }
    }

    /// The same selection with replaced convergence tolerances; every other
    /// field (iteration budget, restart length, symmetry policy) is kept.
    /// Inexact Newton uses this to impose its forcing term.
    #[must_use]
    pub fn with_tolerances(&self, absolute_tolerance: f64, relative_tolerance: f64) -> Self {
        match self {
            Self::ConjugateGradient(config) => Self::ConjugateGradient(ConjugateGradientConfig {
                absolute_tolerance,
                relative_tolerance,
                ..config.clone()
            }),
            Self::Minres(config) => Self::Minres(MinresConfig {
                absolute_tolerance,
                relative_tolerance,
                ..config.clone()
            }),
            Self::Gmres(config) => Self::Gmres(GmresConfig {
                absolute_tolerance,
                relative_tolerance,
                ..config.clone()
            }),
            Self::BiCgStab(config) => Self::BiCgStab(BiCgStabConfig {
                absolute_tolerance,
                relative_tolerance,
                ..config.clone()
            }),
        }
    }
}

/// Final state and convergence evidence from a dispatched Krylov solve.
///
/// `converged` and `trace` are the underlying solver's own verdict and
/// telemetry; their `residual_norm` semantics follow that solver (true
/// residual for conjugate gradient and BiCGSTAB, preconditioner-weighted
/// residual for left-preconditioned MINRES/GMRES). Drivers that need a
/// method-independent acceptance recompute the true residual themselves.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KrylovSolveReport {
    pub method: KrylovMethodKind,
    pub solution: Vec<f64>,
    pub converged: bool,
    pub trace: Vec<LinearIteration>,
    /// Restart cycles consumed; `Some` only for GMRES.
    pub restart_cycles: Option<usize>,
}

impl KrylovSolveReport {
    /// Number of Krylov iterations performed (trace entries after the
    /// initial residual observation).
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.trace.len().saturating_sub(1)
    }
}

/// Dispatch one linear solve to the selected Krylov method.
///
/// # Errors
/// Propagates every refusal of the selected solver unchanged, refuses a
/// projector whose dimension differs from the operator's, and refuses a
/// projector paired with conjugate gradient (which admits no projection or
/// deflation hook by contract).
pub fn solve_krylov(
    method: &KrylovMethod,
    operator: &(impl LinearOperator + ?Sized),
    preconditioner: Option<&dyn Preconditioner>,
    nullspace_projector: Option<&dyn NullspaceProjector>,
    context: &EvaluationContext,
    right_hand_side: &[f64],
    initial_solution: &[f64],
) -> Result<KrylovSolveReport, SolveError> {
    if let Some(projector) = nullspace_projector
        && projector.dimension() != operator.columns()
    {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "nullspace projector dimension {} differs from operator column count {}",
                projector.dimension(),
                operator.columns()
            ),
        });
    }
    match method {
        KrylovMethod::ConjugateGradient(config) => {
            if nullspace_projector.is_some() {
                return Err(SolveError::InvalidConfiguration {
                    reason: "conjugate gradient takes no nullspace projector; select minres \
                             (symmetric) or gmres/bicgstab (general) for a projected solve"
                        .into(),
                });
            }
            let report = solve_conjugate_gradient(
                operator,
                preconditioner,
                context,
                right_hand_side,
                initial_solution,
                config,
            )?;
            Ok(KrylovSolveReport {
                method: KrylovMethodKind::ConjugateGradient,
                solution: report.solution,
                converged: report.converged,
                trace: report.trace,
                restart_cycles: None,
            })
        }
        KrylovMethod::Minres(config) => {
            let report = solve_minres(
                operator,
                preconditioner,
                nullspace_projector,
                context,
                right_hand_side,
                initial_solution,
                config,
            )?;
            Ok(KrylovSolveReport {
                method: KrylovMethodKind::Minres,
                solution: report.solution,
                converged: report.converged,
                trace: report.trace,
                restart_cycles: None,
            })
        }
        KrylovMethod::Gmres(config) => {
            let initial = projected(nullspace_projector, context, initial_solution)?;
            let report = solve_gmres(
                operator,
                preconditioner,
                context,
                right_hand_side,
                &initial,
                config,
            )?;
            let solution = projected(nullspace_projector, context, &report.solution)?;
            Ok(KrylovSolveReport {
                method: KrylovMethodKind::Gmres,
                solution,
                converged: report.converged,
                trace: report.trace,
                restart_cycles: Some(report.restart_cycles),
            })
        }
        KrylovMethod::BiCgStab(config) => {
            let initial = projected(nullspace_projector, context, initial_solution)?;
            let report = solve_bicgstab(
                operator,
                preconditioner,
                context,
                right_hand_side,
                &initial,
                config,
            )?;
            let solution = projected(nullspace_projector, context, &report.solution)?;
            Ok(KrylovSolveReport {
                method: KrylovMethodKind::BiCgStab,
                solution,
                converged: report.converged,
                trace: report.trace,
                restart_cycles: None,
            })
        }
    }
}

fn projected(
    projector: Option<&dyn NullspaceProjector>,
    context: &EvaluationContext,
    vector: &[f64],
) -> Result<Vec<f64>, SolveError> {
    let mut projected = vector.to_vec();
    if let Some(projector) = projector {
        projector.project(context, &mut projected)?;
    }
    Ok(projected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstantModeProjector, CsrMatrix};

    fn nonsymmetric() -> (CsrMatrix, Vec<f64>, Vec<f64>) {
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
    fn dispatch_reports_the_selected_kind_and_keeps_each_solver_admission() {
        let (matrix, rhs, expected) = nonsymmetric();
        let context = EvaluationContext::reproducible();
        for method in [
            KrylovMethod::Gmres(GmresConfig::default()),
            KrylovMethod::BiCgStab(BiCgStabConfig::default()),
        ] {
            let report =
                solve_krylov(&method, &matrix, None, None, &context, &rhs, &[0.0; 3]).unwrap();
            assert_eq!(report.method, method.kind());
            assert!(report.converged);
            assert!(report.iterations() >= 1);
            for (actual, expected) in report.solution.iter().zip(&expected) {
                assert!((actual - expected).abs() < 1.0e-8);
            }
        }
        for method in [
            KrylovMethod::ConjugateGradient(ConjugateGradientConfig::default()),
            KrylovMethod::Minres(MinresConfig::default()),
        ] {
            let error =
                solve_krylov(&method, &matrix, None, None, &context, &rhs, &[0.0; 3]).unwrap_err();
            assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
        }
    }

    #[test]
    fn conjugate_gradient_refuses_a_projector_and_dimension_mismatches_are_refused() {
        let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0), (1, 1, 2.0)]).unwrap();
        let projector = ConstantModeProjector::new(2, 0, 2).unwrap();
        let error = solve_krylov(
            &KrylovMethod::ConjugateGradient(ConjugateGradientConfig::default()),
            &matrix,
            None,
            Some(&projector),
            &EvaluationContext::default(),
            &[1.0, 1.0],
            &[0.0; 2],
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));

        let wrong = ConstantModeProjector::new(3, 0, 3).unwrap();
        let error = solve_krylov(
            &KrylovMethod::Gmres(GmresConfig::default()),
            &matrix,
            None,
            Some(&wrong),
            &EvaluationContext::default(),
            &[1.0, 1.0],
            &[0.0; 2],
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
    }

    #[test]
    fn gmres_with_a_projector_returns_the_pseudo_solution_of_a_singular_system() {
        // Singular, consistent, symmetric: the constant mode is the nullspace.
        let matrix = CsrMatrix::from_triplets(
            2,
            2,
            vec![(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)],
        )
        .unwrap();
        let projector = ConstantModeProjector::new(2, 0, 2).unwrap();
        let report = solve_krylov(
            &KrylovMethod::Gmres(GmresConfig::default()),
            &matrix,
            None,
            Some(&projector),
            &EvaluationContext::reproducible(),
            &[1.0, -1.0],
            &[3.0, 3.0],
        )
        .unwrap();
        assert!(report.converged);
        assert!((report.solution[0] - 0.5).abs() < 1.0e-8);
        assert!((report.solution[1] + 0.5).abs() < 1.0e-8);
    }

    #[test]
    fn with_tolerances_replaces_only_the_tolerances() {
        let method = KrylovMethod::Gmres(GmresConfig {
            restart: 7,
            max_iterations: 11,
            ..GmresConfig::default()
        });
        let tightened = method.with_tolerances(1.0e-3, 1.0e-2);
        assert_eq!(tightened.tolerances(), (1.0e-3, 1.0e-2));
        match tightened {
            KrylovMethod::Gmres(config) => {
                assert_eq!(config.restart, 7);
                assert_eq!(config.max_iterations, 11);
            }
            _ => unreachable!(),
        }
        let json = serde_json::to_string(&method).unwrap();
        let parsed: KrylovMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, method);
    }
}
