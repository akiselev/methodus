//! SV1-C5/SV1-D1 and GX-D2/SV2-A4: transpose-operator and adjoint-solve contracts.
//!
//! [`TransposeOperator`] adapts a linear operator into its algebraic
//! transpose in one of two ways. [`TransposeOperator::new`] accepts any
//! symmetric-declared [`LinearOperator`] by direct delegation — for
//! symmetric operators the transpose action equals the primal action, so the
//! wrapper is exact and free. [`TransposeOperator::explicit`] accepts any
//! operator that implements [`TransposableOperator`], using its explicit
//! transpose action; this covers `Nonsymmetric`/`Unknown` operators that
//! carry a genuine transpose. Matrix-free `Nonsymmetric`/`Unknown` operators
//! without that trait are still refused rather than silently approximated,
//! because computing a genuine matrix-free transpose requires column access
//! the [`LinearOperator`] contract does not otherwise expose.
//!
//! [`verify_adjoint_identity`] checks `<A u, v> == <u, Aᵀ v>` on caller-chosen
//! probes; [`transpose_view`] is the entry point adjoint solves use.

use serde::{Deserialize, Serialize};

use crate::context::EvaluationContext;
use crate::error::NumericError;
use crate::operator::{
    LinearOperator, OperatorProperties, OperatorStructureHint, OperatorSymmetry,
};

/// How a [`TransposeOperator`] obtains its transpose action; reported by
/// adjoint solves so evidence records which path produced `Aᵀ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransposeSource {
    /// `A = Aᵀ` under the admitted `Symmetric` declaration.
    SymmetricDelegation,
    /// An explicit [`TransposableOperator::apply_transpose`] action.
    ExplicitTranspose,
}

/// Matrix-free operators whose transpose (column-space) action can be
/// computed directly, independent of any symmetry declaration.
pub trait TransposableOperator: LinearOperator {
    /// Applies the transpose action `Aᵀ x`.
    ///
    /// # Errors
    /// Propagates dimension mismatches and non-finite values the same way
    /// [`LinearOperator::apply`] does.
    fn apply_transpose(
        &self,
        context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
}

/// Function-pointer shape of [`TransposableOperator::apply_transpose`], used
/// to store an explicit transpose action without boxing.
type ExplicitTransposeFn<T> =
    fn(&T, &EvaluationContext, &[f64], &mut [f64]) -> Result<(), NumericError>;

/// Where a [`TransposeOperator`]'s action is sourced from.
#[derive(Debug)]
enum TransposeAction<T: LinearOperator + ?Sized> {
    /// `A = Aᵀ` under the admitted `Symmetric` declaration; the transpose
    /// action is the primal action.
    Delegated,
    /// An explicit transpose action supplied by
    /// [`TransposableOperator::apply_transpose`].
    Explicit(ExplicitTransposeFn<T>),
}

/// Algebraic transpose view of a linear operator.
///
/// See [`TransposeOperator::new`] (symmetric delegation) and
/// [`TransposeOperator::explicit`] (explicit transpose action).
#[derive(Debug)]
pub struct TransposeOperator<'a, T: LinearOperator + ?Sized> {
    inner: &'a T,
    action: TransposeAction<T>,
}

impl<'a, T: LinearOperator + ?Sized> TransposeOperator<'a, T> {
    /// Wraps one borrowed operator as its transpose by symmetric delegation.
    ///
    /// # Errors
    /// Refuses operators whose declared symmetry does not certify `A = Aᵀ`,
    /// because the matrix-free contract cannot compute genuine transposes
    /// without either a `Symmetric` declaration or [`TransposableOperator`]
    /// (see [`TransposeOperator::explicit`]).
    pub fn new(inner: &'a T) -> Result<Self, NumericError> {
        match inner.symmetry() {
            OperatorSymmetry::Symmetric => Ok(Self {
                inner,
                action: TransposeAction::Delegated,
            }),
            other => Err(NumericError::InvalidInput {
                message: format!(
                    "transpose view requires a Symmetric declaration, got {other:?}; \
                     matrix-free transposes of nonsymmetric operators need column access"
                ),
            }),
        }
    }

    /// Which path supplies this view's transpose action.
    #[must_use]
    pub const fn source(&self) -> TransposeSource {
        match self.action {
            TransposeAction::Delegated => TransposeSource::SymmetricDelegation,
            TransposeAction::Explicit(_) => TransposeSource::ExplicitTranspose,
        }
    }

    /// The wrapped primal operator.
    #[must_use]
    pub const fn inner(&self) -> &T {
        self.inner
    }

    /// Wraps one borrowed [`TransposableOperator`] using its explicit
    /// transpose action.
    ///
    /// Unlike [`TransposeOperator::new`], this works for `Nonsymmetric` and
    /// `Unknown` declarations, because [`TransposableOperator::apply_transpose`]
    /// supplies the genuine transpose action directly rather than relying on
    /// `A = Aᵀ`.
    #[must_use]
    pub fn explicit(inner: &'a T) -> Self
    where
        T: TransposableOperator,
    {
        Self {
            inner,
            action: TransposeAction::Explicit(|inner, context, input, output| {
                inner.apply_transpose(context, input, output)
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
        // Symmetry is a self-transpose property (`A = Aᵀ`), so whatever the
        // wrapped operator declares also holds for its transpose, regardless
        // of which action mode produced this view.
        self.inner.symmetry()
    }

    /// Properties of `Aᵀ` derived from those declared for `A`: symmetry and
    /// definiteness are transpose-invariant (`xᵀAᵀx = xᵀAx`), and for a
    /// square operator so are the nullspace dimension (equal rank) and the
    /// block structure (input and output share one partition). For a
    /// rectangular operator the nullspace dimensions of `A` and `Aᵀ` differ
    /// and the block structure does not transfer, so both are dropped.
    fn properties(&self) -> OperatorProperties {
        let inner = self.inner.properties();
        if self.inner.rows() == self.inner.columns() {
            inner
        } else {
            OperatorProperties::new(
                inner.symmetry(),
                inner.definiteness(),
                None,
                OperatorStructureHint::Dense,
            )
            .unwrap_or_else(|_| OperatorProperties::from_symmetry(inner.symmetry()))
        }
    }

    fn apply(
        &self,
        context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        match self.action {
            TransposeAction::Delegated => self.inner.apply(context, input, output),
            TransposeAction::Explicit(apply_transpose) => {
                apply_transpose(self.inner, context, input, output)
            }
        }
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
