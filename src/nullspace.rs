//! SV2-B6: typed nullspace-projection hook consumed by projected Krylov
//! solvers over singular, consistent symmetric systems (e.g. a saddle-point
//! Stokes discretization's constant-pressure mode).
//!
//! [`crate::solve_conjugate_gradient`] refuses any operator carrying a
//! declared positive nullspace dimension outright. [`crate::solve_minres`]
//! may instead proceed when the caller supplies a [`NullspaceProjector`]
//! whose [`NullspaceProjector::project`] removes the declared nullspace
//! component from a vector. [`ConstantModeProjector`] is a bounded reference
//! implementation for the single most common case: a constant mode confined
//! to one contiguous coordinate range.

use crate::context::EvaluationContext;
use crate::error::NumericError;

/// Removes a declared operator nullspace component from a vector, in place.
///
/// Implementations must be idempotent (`project` applied twice equals
/// applied once) and must return finite output for finite input.
pub trait NullspaceProjector: Send + Sync {
    /// The dimension of vectors this projector accepts.
    fn dimension(&self) -> usize;

    /// Projects `vector` onto the orthogonal complement of the declared
    /// nullspace, in place.
    ///
    /// # Errors
    /// Returns a typed error on a dimension mismatch or non-finite result.
    fn project(&self, context: &EvaluationContext, vector: &mut [f64]) -> Result<(), NumericError>;
}

/// Projects out one constant mode confined to a contiguous coordinate range
/// (e.g. the constant-pressure nullspace of a saddle-point Stokes system),
/// by subtracting the mean of that range from each of its entries.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstantModeProjector {
    dimension: usize,
    start: usize,
    length: usize,
}

impl ConstantModeProjector {
    /// Builds a validated constant-mode projector over `[start, start +
    /// length)` within an ambient vector of size `dimension`.
    ///
    /// # Errors
    /// Refuses a zero-length range or a range that does not fit within
    /// `dimension`.
    pub fn new(dimension: usize, start: usize, length: usize) -> Result<Self, NumericError> {
        if length == 0 {
            return Err(NumericError::InvalidInput {
                message: "constant-mode projector range must not be empty".into(),
            });
        }
        let end = start
            .checked_add(length)
            .ok_or_else(|| NumericError::InvalidInput {
                message: "constant-mode projector range overflows usize".into(),
            })?;
        if end > dimension {
            return Err(NumericError::InvalidInput {
                message: format!(
                    "constant-mode projector range {start}..{end} exceeds dimension {dimension}"
                ),
            });
        }
        Ok(Self {
            dimension,
            start,
            length,
        })
    }
}

impl NullspaceProjector for ConstantModeProjector {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn project(
        &self,
        _context: &EvaluationContext,
        vector: &mut [f64],
    ) -> Result<(), NumericError> {
        NumericError::require_len(
            "constant-mode projector input",
            vector.len(),
            self.dimension,
        )?;
        NumericError::require_finite("constant-mode projector input", vector)?;
        let range = &mut vector[self.start..self.start + self.length];
        let mean = range.iter().sum::<f64>() / self.length as f64;
        for value in range.iter_mut() {
            *value -= mean;
        }
        NumericError::require_finite("constant-mode projector output", vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projector_removes_the_mean_over_its_range_only() {
        let projector = ConstantModeProjector::new(4, 1, 2).unwrap();
        let mut vector = [10.0, 3.0, 5.0, -7.0];
        projector
            .project(&EvaluationContext::default(), &mut vector)
            .unwrap();
        assert_eq!(vector[0], 10.0);
        assert_eq!(vector[3], -7.0);
        assert!((vector[1] - (-1.0)).abs() < 1e-14);
        assert!((vector[2] - 1.0).abs() < 1e-14);
        assert!((vector[1] + vector[2]).abs() < 1e-14);
    }

    #[test]
    fn projection_is_idempotent() {
        let projector = ConstantModeProjector::new(3, 0, 3).unwrap();
        let mut vector = [1.0, 2.0, 6.0];
        projector
            .project(&EvaluationContext::default(), &mut vector)
            .unwrap();
        let once = vector;
        projector
            .project(&EvaluationContext::default(), &mut vector)
            .unwrap();
        assert_eq!(vector, once);
    }

    #[test]
    fn out_of_range_construction_is_refused() {
        assert!(ConstantModeProjector::new(3, 2, 2).is_err());
        assert!(ConstantModeProjector::new(3, 0, 0).is_err());
    }

    #[test]
    fn dimension_mismatch_is_refused() {
        let projector = ConstantModeProjector::new(3, 0, 3).unwrap();
        let mut vector = [1.0, 2.0];
        let error = projector
            .project(&EvaluationContext::default(), &mut vector)
            .unwrap_err();
        assert!(matches!(error, NumericError::DimensionMismatch { .. }));
    }
}
