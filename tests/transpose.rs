//! SV1-C5/SV1-D1 acceptance: symmetric-declared transpose view delegates
//! exactly, refuses nonsymmetric declarations, and satisfies the adjoint
//! identity on probe vectors.

use methodus::{
    EvaluationContext, LinearOperator, NumericError, OperatorSymmetry, TransposeOperator,
    transpose_view, verify_adjoint_identity,
};

/// Diagonal operator with configurable symmetry declaration.
#[derive(Debug)]
struct Diagonal {
    values: Vec<f64>,
    declared: OperatorSymmetry,
}

impl LinearOperator for Diagonal {
    fn rows(&self) -> usize {
        self.values.len()
    }
    fn columns(&self) -> usize {
        self.values.len()
    }
    fn symmetry(&self) -> OperatorSymmetry {
        self.declared
    }
    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        if input.len() != self.values.len() || output.len() != self.values.len() {
            return Err(NumericError::DimensionMismatch {
                operation: "diagonal apply".into(),
                expected: self.values.len(),
                actual: input.len(),
            });
        }
        for (output, (input, scale)) in output.iter_mut().zip(input.iter().zip(&self.values)) {
            *output = input * scale;
        }
        Ok(())
    }
}

#[test]
fn symmetric_transpose_delegates_exactly_and_satisfies_identity() {
    let context = EvaluationContext::reproducible();
    let diagonal = Diagonal {
        values: vec![2.0, -3.0, 0.5],
        declared: OperatorSymmetry::Symmetric,
    };
    let transpose = transpose_view(&diagonal).unwrap();
    assert_eq!(transpose.rows(), 3);
    assert_eq!(transpose.columns(), 3);
    assert_eq!(transpose.symmetry(), OperatorSymmetry::Symmetric);

    let u = [1.5, -2.0, 4.0];
    let v = [-1.0, 0.25, 3.0];
    let discrepancy =
        verify_adjoint_identity(&diagonal, &transpose, &context, &u, &v, 1.0e-14).unwrap();
    assert!(discrepancy < 1.0e-14);

    // The transpose action equals the primal action element-for-element.
    let mut primal = vec![0.0; 3];
    let mut transposed = vec![0.0; 3];
    diagonal.apply(&context, &u, &mut primal).unwrap();
    transpose.apply(&context, &u, &mut transposed).unwrap();
    assert_eq!(primal, transposed);

    // A dimension mismatch in either probe is refused outright.
    let short = [1.0, 2.0];
    assert!(matches!(
        verify_adjoint_identity(&diagonal, &transpose, &context, &short, &v, 1.0e-14),
        Err(NumericError::DimensionMismatch { .. })
    ));
}

#[test]
fn nonsymmetric_and_unknown_declarations_are_refused() {
    for declared in [OperatorSymmetry::Nonsymmetric, OperatorSymmetry::Unknown] {
        let diagonal = Diagonal {
            values: vec![1.0, 2.0],
            declared,
        };
        let error = TransposeOperator::new(&diagonal).unwrap_err();
        assert!(
            error.to_string().contains("Symmetric declaration"),
            "{error}"
        );
    }
}
