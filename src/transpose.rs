//! SV1-C5/SV1-D1: transpose-operator and adjoint-solve contracts.
//!
//! [`TransposeOperator`] adapts any symmetric-declared [`LinearOperator`] into
//! its algebraic transpose by direct delegation — for symmetric operators the
//! transpose action equals the primal action, so the wrapper is exact and free.
//! Nonsymmetric or evidence-free operators are refused rather than silently
//! approximated, because computing a genuine matrix-free transpose requires
//! column access the [`LinearOperator`] contract does not expose.
//!
//! [`verify_adjoint_identity`] checks `<A u, v> == <u, Aᵀ v>` on caller-chosen
//! probes; [`transpose_view`] is the entry point adjoint solves use.

use crate::context::EvaluationContext;
use crate::error::NumericError;
use crate::operator::{LinearOperator, OperatorSymmetry};

/// Algebraic transpose view of a symmetric-declared linear operator.
#[derive(Debug)]
pub struct TransposeOperator<'a, T: LinearOperator + ?Sized> {
    inner: &'a T,
}

impl<'a, T: LinearOperator + ?Sized> TransposeOperator<'a, T> {
    /// Wraps one borrowed operator as its transpose.
    ///
    /// # Errors
    /// Refuses operators whose declared symmetry does not certify `A = Aᵀ`,
    /// because the matrix-free contract cannot compute genuine transposes.
    pub fn new(inner: &'a T) -> Result<Self, NumericError> {
        match inner.symmetry() {
            OperatorSymmetry::Symmetric => Ok(Self { inner }),
            other => Err(NumericError::InvalidInput {
                message: format!(
                    "transpose view requires a Symmetric declaration, got {other:?}; \
                     matrix-free transposes of nonsymmetric operators need column access"
                ),
            }),
        }
    }
}

impl<T: LinearOperator + ?Sized> LinearOperator for TransposeOperator<'_, T> {
    fn rows(&self) -> usize {
        self.inner.columns()
    }

    fn columns(&self) -> usize {
        self.inner.rows()
    }

    fn symmetry(&self) -> OperatorSymmetry {
        OperatorSymmetry::Symmetric
    }

    fn apply(
        &self,
        context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        // A = Aᵀ by the admitted declaration, so the transpose action IS the
        // primal action; dimensions coincide for square symmetric operators.
        self.inner.apply(context, input, output)
    }
}

/// Symmetric-declared transpose view; see [`TransposeOperator::new`].
///
/// # Errors
/// Refuses nonsymmetric or evidence-free operators.
pub fn transpose_view<T: LinearOperator + ?Sized>(
    operator: &T,
) -> Result<TransposeOperator<'_, T>, NumericError> {
    TransposeOperator::new(operator)
}

/// Verifies `<A u, v> == <u, Aᵀ v>` on one caller-chosen probe pair.
///
/// Returns the absolute inner-product discrepancy; refuses the operator when
/// it exceeds `tolerance`.
///
/// # Errors
/// Propagates operator failures and dimension mismatches.
pub fn verify_adjoint_identity<T: LinearOperator + ?Sized>(
    operator: &T,
    transpose: &TransposeOperator<'_, T>,
    context: &EvaluationContext,
    u: &[f64],
    v: &[f64],
    tolerance: f64,
) -> Result<f64, NumericError> {
    if u.len() != operator.columns() || v.len() != operator.rows() {
        return Err(NumericError::DimensionMismatch {
            operation: "adjoint identity probes".into(),
            expected: operator.columns(),
            actual: u.len(),
        });
    }
    let mut a_u = vec![0.0_f64; operator.rows()];
    operator.apply(context, u, &mut a_u)?;
    let mut at_v = vec![0.0_f64; operator.columns()];
    transpose.apply(context, v, &mut at_v)?;

    let left: f64 = a_u.iter().zip(v).map(|(a, b)| a * b).sum();
    let right: f64 = u.iter().zip(&at_v).map(|(a, b)| a * b).sum();
    let discrepancy = (left - right).abs();
    if discrepancy > tolerance {
        return Err(NumericError::Operator {
            message: format!(
                "adjoint identity violated: <Au,v> = {left}, <u,transpose v> = {right}"
            ),
        });
    }
    Ok(discrepancy)
}
