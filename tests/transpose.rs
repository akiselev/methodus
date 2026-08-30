//! SV1-C5/SV1-D1 acceptance: symmetric-declared transpose view delegates
//! exactly, refuses nonsymmetric declarations, and satisfies the adjoint
//! identity on probe vectors.

use methodus::{
    EvaluationContext, LinearOperator, NumericError, OperatorSymmetry, TransposableOperator,
    TransposeOperator, transpose_view, verify_adjoint_identity,
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

/// Row-major dense operator with an explicit, genuinely matrix-free
/// transpose action, used to exercise `TransposeOperator::explicit` on a
/// `Nonsymmetric`-declared, non-square operator.
struct DenseMatrix {
    rows: usize,
    columns: usize,
    data: Vec<f64>,
}

impl LinearOperator for DenseMatrix {
    fn rows(&self) -> usize {
        self.rows
    }
    fn columns(&self) -> usize {
        self.columns
    }
    fn symmetry(&self) -> OperatorSymmetry {
        OperatorSymmetry::Nonsymmetric
    }
    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        for (row, output_value) in output.iter_mut().enumerate() {
            *output_value = (0..self.columns)
                .map(|column| self.data[row * self.columns + column] * input[column])
                .sum();
        }
        Ok(())
    }
}

impl TransposableOperator for DenseMatrix {
    fn apply_transpose(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        for (column, output_value) in output.iter_mut().enumerate() {
            *output_value = (0..self.rows)
                .map(|row| self.data[row * self.columns + column] * input[row])
                .sum();
        }
        Ok(())
    }
}

#[test]
fn explicit_transpose_works_for_a_matrix_free_nonsymmetric_operator() {
    let context = EvaluationContext::reproducible();
    // A 2x3 rectangular, declared-Nonsymmetric matrix-free operator.
    let matrix = DenseMatrix {
        rows: 2,
        columns: 3,
        data: vec![1.0, 2.0, -1.0, 0.0, 3.0, 4.0],
    };

    // Symmetric delegation is refused for a Nonsymmetric declaration.
    assert!(TransposeOperator::new(&matrix).is_err());

    let transpose = TransposeOperator::explicit(&matrix);
    assert_eq!(transpose.rows(), 3);
    assert_eq!(transpose.columns(), 2);

    let input = [1.0, -1.0];
    let mut via_transpose_operator = vec![0.0; 3];
    transpose
        .apply(&context, &input, &mut via_transpose_operator)
        .unwrap();

    let mut via_direct_action = vec![0.0; 3];
    matrix
        .apply_transpose(&context, &input, &mut via_direct_action)
        .unwrap();
    assert_eq!(via_transpose_operator, via_direct_action);

    let discrepancy = verify_adjoint_identity(
        &matrix,
        &transpose,
        &context,
        &[2.0, -3.0, 0.5],
        &input,
        1.0e-12,
    )
    .unwrap();
    assert!(discrepancy < 1.0e-12);
}
