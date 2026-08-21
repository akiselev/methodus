use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Failure reported while evaluating a numerical operator.
#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum NumericError {
    #[error("{operation}: expected length {expected}, got {actual}")]
    DimensionMismatch {
        operation: String,
        expected: usize,
        actual: usize,
    },
    #[error("{operation}: non-finite value at index {index}")]
    NonFinite { operation: String, index: usize },
    #[error("invalid numerical input: {message}")]
    InvalidInput { message: String },
    #[error("operator evaluation failed: {message}")]
    Operator { message: String },
}

impl NumericError {
    pub(crate) fn require_len(operation: &str, actual: usize, expected: usize) -> Result<(), Self> {
        if actual == expected {
            Ok(())
        } else {
            Err(Self::DimensionMismatch {
                operation: operation.into(),
                expected,
                actual,
            })
        }
    }

    pub(crate) fn require_finite(operation: &str, values: &[f64]) -> Result<(), Self> {
        match values.iter().position(|value| !value.is_finite()) {
            Some(index) => Err(Self::NonFinite {
                operation: operation.into(),
                index,
            }),
            None => Ok(()),
        }
    }
}

/// Failure reported by a Methodus algorithm.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum SolveError {
    #[error(transparent)]
    Numeric(#[from] NumericError),
    #[error("invalid block layout: {reason}")]
    InvalidLayout { reason: String },
    #[error("invalid solver configuration: {reason}")]
    InvalidConfiguration { reason: String },
    #[error("singular dense Newton system")]
    Singular,
    #[error("line search could not reduce the residual")]
    LineSearchFailed,
    #[error("nonlinear solve did not converge")]
    NotConverged,
    #[error("conjugate-gradient iteration {iteration} encountered a non-positive search curvature")]
    KrylovBreakdown { iteration: usize },
}
