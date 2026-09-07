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
    /// A typed failure raised inside an operator callback (a constitutive law, an external
    /// input, a sampled datum). `code` is the producer's own refusal code and `origin` names
    /// the input, slot or expression it came from; both are carried verbatim through every
    /// Methodus algorithm so the consumer reports the original cause, never "non-finite".
    #[error("{code} at {origin}: {message}")]
    Evaluation {
        code: String,
        origin: String,
        message: String,
    },
}

impl NumericError {
    /// The producer's refusal code when this is a typed [`NumericError::Evaluation`].
    pub fn evaluation_code(&self) -> Option<&str> {
        match self {
            Self::Evaluation { code, .. } => Some(code),
            _ => None,
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluation_failure_keeps_its_code_and_origin_through_display_serde_and_solve_error() {
        let error = NumericError::Evaluation {
            code: "RUN_TANGENT_UNAVAILABLE".into(),
            origin: "provider/diffusivity".into(),
            message: "no tangent for an analytic_provided law".into(),
        };
        assert_eq!(
            error.to_string(),
            "RUN_TANGENT_UNAVAILABLE at provider/diffusivity: no tangent for an analytic_provided law"
        );
        assert_eq!(error.evaluation_code(), Some("RUN_TANGENT_UNAVAILABLE"));
        let json = serde_json::to_string(&error).expect("serialize");
        let back: NumericError = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, error);
        let solve: SolveError = error.clone().into();
        assert_eq!(solve.to_string(), error.to_string());
        assert_eq!(
            NumericError::Operator {
                message: "x".into()
            }
            .evaluation_code(),
            None
        );
    }
}
