//! E7/SV1-D1: the adjoint linear solve `Aᵀ λ = g` under the C5
//! `TransposableOperator`/`OperatorProperties` contracts.
//!
//! The driver never approximates a transpose. The transpose action comes
//! from a [`TransposeOperator`] built either by symmetric delegation
//! ([`TransposeOperator::new`], `A = Aᵀ` under a `Symmetric` declaration) or
//! from an explicit [`crate::TransposableOperator::apply_transpose`]
//! ([`TransposeOperator::explicit`], the assembled-operator path). An
//! operator that offers neither is refused at construction, before any
//! solve; the driver therefore cannot be handed a transpose it cannot
//! compute.
//!
//! Method admission is property-aware and refuses rather than falls back:
//! conjugate gradient and MINRES are refused on a transpose whose declared
//! symmetry is not `Symmetric` (conjugate gradient additionally keeps its
//! own explicit-assumption policy for `Unknown`), while GMRES and BiCGSTAB
//! admit any declared symmetry. Acceptance is residual-based and
//! method-independent: after the Krylov solve the driver recomputes the true
//! residual `g − Aᵀ λ` through the transpose action and accepts only when
//! its Euclidean norm meets the caller's [`ResidualAcceptance`], regardless
//! of what the inner solver's (possibly preconditioner-weighted) estimate
//! claimed. Telemetry is typed and deterministic.

use serde::{Deserialize, Serialize};

use crate::context::EvaluationContext;
use crate::error::{NumericError, SolveError};
use crate::krylov_method::{KrylovMethod, KrylovMethodKind, KrylovSolveReport, solve_krylov};
use crate::linear::{LinearIteration, l2};
use crate::nullspace::NullspaceProjector;
use crate::operator::{LinearOperator, OperatorSymmetry, Preconditioner};
use crate::transpose::{TransposeOperator, TransposeSource};

/// Method-independent acceptance test on the true adjoint residual
/// `‖g − Aᵀ λ‖ ≤ absolute_tolerance + relative_tolerance · ‖g‖`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualAcceptance {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}

impl Default for ResidualAcceptance {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-10,
        }
    }
}

impl ResidualAcceptance {
    fn validate(&self) -> Result<(), SolveError> {
        let valid = self.absolute_tolerance.is_finite()
            && self.absolute_tolerance >= 0.0
            && self.relative_tolerance.is_finite()
            && self.relative_tolerance >= 0.0
            && (self.absolute_tolerance > 0.0 || self.relative_tolerance > 0.0);
        if valid {
            Ok(())
        } else {
            Err(SolveError::InvalidConfiguration {
                reason: "adjoint acceptance tolerances must be finite, nonnegative, and not both \
                         zero"
                    .into(),
            })
        }
    }

    #[must_use]
    fn threshold(&self, reference_norm: f64) -> f64 {
        self.absolute_tolerance + self.relative_tolerance * reference_norm
    }
}

/// Configuration of one adjoint solve: which Krylov method computes `λ`
/// and how the true residual is accepted afterwards.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdjointConfig {
    pub method: KrylovMethod,
    #[serde(default)]
    pub acceptance: ResidualAcceptance,
}

/// Final state and evidence of an adjoint solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdjointSolveReport {
    /// The adjoint state `λ` (projected when a nullspace projector was
    /// supplied).
    pub adjoint: Vec<f64>,
    /// `true` iff the recomputed true residual met the acceptance test;
    /// the inner solver's own verdict is `solver_converged`.
    pub converged: bool,
    pub method: KrylovMethodKind,
    pub transpose_source: TransposeSource,
    /// The inner Krylov solver's own convergence verdict.
    pub solver_converged: bool,
    /// The inner solver's telemetry (its `residual_norm` semantics).
    pub trace: Vec<LinearIteration>,
    pub restart_cycles: Option<usize>,
    /// Euclidean norm of the adjoint right-hand side `g`.
    pub gradient_norm: f64,
    /// Euclidean norm of the recomputed true residual `g − Aᵀ λ`.
    pub residual_norm: f64,
}

/// Solve `Aᵀ λ = g` through a [`TransposeOperator`] view of `A`.
///
/// `preconditioner`, when supplied, must approximate the inverse of `Aᵀ`
/// (not of `A`); it is the caller's obligation to transpose an approximate
/// inverse of `A` where the two differ. `nullspace_projector`, when
/// supplied, is applied as [`crate::solve_krylov`] documents.
///
/// # Errors
/// Refuses a non-square transpose, conjugate gradient or MINRES on a
/// transpose not declared `Symmetric` (with conjugate gradient's own
/// `Unknown` policy retained), invalid acceptance tolerances, and every
/// refusal of the selected inner solver. A solve that ran but did not meet
/// the acceptance test is *not* an error: it returns `converged == false`
/// with the measured residual so the caller can decide.
pub fn solve_adjoint<T: LinearOperator + ?Sized>(
    transpose: &TransposeOperator<'_, T>,
    preconditioner: Option<&dyn Preconditioner>,
    nullspace_projector: Option<&dyn NullspaceProjector>,
    context: &EvaluationContext,
    gradient: &[f64],
    initial_adjoint: &[f64],
    config: &AdjointConfig,
) -> Result<AdjointSolveReport, SolveError> {
    config.acceptance.validate()?;
    let dimension = transpose.rows();
    if transpose.columns() != dimension {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "adjoint solve requires a square transpose, got {}x{}",
                transpose.rows(),
                transpose.columns()
            ),
        });
    }
    let symmetry = transpose.symmetry();
    match (config.method.kind(), symmetry) {
        (KrylovMethodKind::ConjugateGradient, OperatorSymmetry::Nonsymmetric) => {
            return Err(SolveError::InvalidConfiguration {
                reason: "adjoint solve refuses conjugate gradient on a transpose declared \
                         Nonsymmetric; select gmres or bicgstab"
                    .into(),
            });
        }
        (KrylovMethodKind::Minres, OperatorSymmetry::Nonsymmetric | OperatorSymmetry::Unknown) => {
            return Err(SolveError::InvalidConfiguration {
                reason: format!(
                    "adjoint solve refuses minres on a transpose declared {symmetry:?}; select \
                     gmres or bicgstab"
                ),
            });
        }
        _ => {}
    }
    NumericError::require_len("adjoint right-hand side", gradient.len(), dimension)?;
    NumericError::require_len("initial adjoint state", initial_adjoint.len(), dimension)?;
    NumericError::require_finite("adjoint right-hand side", gradient)?;
    NumericError::require_finite("initial adjoint state", initial_adjoint)?;

    let KrylovSolveReport {
        method,
        solution,
        converged: solver_converged,
        trace,
        restart_cycles,
    } = solve_krylov(
        &config.method,
        transpose,
        preconditioner,
        nullspace_projector,
        context,
        gradient,
        initial_adjoint,
    )?;

    let mut action = vec![0.0; dimension];
    transpose.apply(context, &solution, &mut action)?;
    NumericError::require_finite("adjoint transpose action", &action)?;
    let residual: Vec<f64> = gradient.iter().zip(&action).map(|(g, a)| g - a).collect();
    let residual_norm = l2(&residual)?;
    let gradient_norm = l2(gradient)?;
    let converged = residual_norm <= config.acceptance.threshold(gradient_norm);

    Ok(AdjointSolveReport {
        adjoint: solution,
        converged,
        method,
        transpose_source: transpose.source(),
        solver_converged,
        trace,
        restart_cycles,
        gradient_norm,
        residual_norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConjugateGradientConfig, CsrMatrix, GmresConfig, MinresConfig};

    #[test]
    fn acceptance_tolerances_are_validated() {
        let config = AdjointConfig {
            method: KrylovMethod::Gmres(GmresConfig::default()),
            acceptance: ResidualAcceptance {
                absolute_tolerance: 0.0,
                relative_tolerance: 0.0,
            },
        };
        let matrix = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 2.0)]).unwrap();
        let transpose = TransposeOperator::explicit(&matrix);
        let error = solve_adjoint(
            &transpose,
            None,
            None,
            &EvaluationContext::default(),
            &[1.0],
            &[0.0],
            &config,
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
    }

    #[test]
    fn symmetric_delegation_admits_conjugate_gradient_and_minres() {
        let matrix = CsrMatrix::from_triplets(
            2,
            2,
            vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)],
        )
        .unwrap();
        let transpose = TransposeOperator::new(&matrix).unwrap();
        for method in [
            KrylovMethod::ConjugateGradient(ConjugateGradientConfig::default()),
            KrylovMethod::Minres(MinresConfig::default()),
        ] {
            let report = solve_adjoint(
                &transpose,
                None,
                None,
                &EvaluationContext::reproducible(),
                &[5.0, 4.0],
                &[0.0; 2],
                &AdjointConfig {
                    method,
                    acceptance: ResidualAcceptance::default(),
                },
            )
            .unwrap();
            assert!(report.converged);
            assert_eq!(
                report.transpose_source,
                TransposeSource::SymmetricDelegation
            );
            assert!((report.adjoint[0] - 1.0).abs() < 1.0e-9);
            assert!((report.adjoint[1] - 1.0).abs() < 1.0e-9);
        }
    }
}
