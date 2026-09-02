//! E7/SV1-D1 acceptance: the adjoint solve `Aᵀ λ = g` on nonsymmetric dense
//! and sparse operators under the C5 `TransposableOperator` contract
//! satisfies the adjoint-solve consistency identity `<λ, b> = <g, u>` for
//! `A u = b`, refuses property-inadmissible methods and transpose-less
//! operators, reports true residuals, and is bit-reproducible.

use methodus::{
    AdjointConfig, BiCgStabConfig, ConjugateGradientConfig, CsrMatrix, EvaluationContext,
    GmresConfig, KrylovMethod, KrylovMethodKind, LinearOperator, MinresConfig, NumericError,
    OperatorSymmetry, Preconditioner, ResidualAcceptance, SolveError, TransposableOperator,
    TransposeOperator, TransposeSource, solve_adjoint, solve_krylov,
};

fn require_len(operation: &str, actual: usize, expected: usize) -> Result<(), NumericError> {
    if actual == expected {
        Ok(())
    } else {
        Err(NumericError::DimensionMismatch {
            operation: operation.into(),
            expected,
            actual,
        })
    }
}

/// Row-major dense operator with an explicit transpose action.
struct DenseMatrix {
    dimension: usize,
    data: Vec<f64>,
}

impl DenseMatrix {
    fn entry(&self, row: usize, column: usize) -> f64 {
        self.data[row * self.dimension + column]
    }
}

impl LinearOperator for DenseMatrix {
    fn rows(&self) -> usize {
        self.dimension
    }
    fn columns(&self) -> usize {
        self.dimension
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
        require_len("dense input", input.len(), self.dimension)?;
        require_len("dense output", output.len(), self.dimension)?;
        for (row, output_value) in output.iter_mut().enumerate() {
            *output_value = (0..self.dimension)
                .map(|column| self.entry(row, column) * input[column])
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
        require_len("dense transpose input", input.len(), self.dimension)?;
        require_len("dense transpose output", output.len(), self.dimension)?;
        for (column, output_value) in output.iter_mut().enumerate() {
            *output_value = (0..self.dimension)
                .map(|row| self.entry(row, column) * input[row])
                .sum();
        }
        Ok(())
    }
}

/// A well-conditioned, clearly nonsymmetric 6x6 dense fixture (diagonally
/// dominant with an asymmetric off-diagonal pattern).
fn dense_fixture() -> DenseMatrix {
    let n = 6;
    let mut data = vec![0.0; n * n];
    for row in 0..n {
        for column in 0..n {
            let value = if row == column {
                10.0 + row as f64
            } else if column == row + 1 {
                -2.5
            } else if column + 2 == row {
                1.75
            } else if column > row {
                0.3 * (row as f64 + 1.0)
            } else {
                -0.2 * (column as f64 + 1.0)
            };
            data[row * n + column] = value;
        }
    }
    DenseMatrix { dimension: n, data }
}

/// A nonsymmetric sparse fixture: a 1-D upwind convection–diffusion
/// stencil (diffusion 1, convection 0.6) on 12 unknowns.
fn sparse_fixture() -> CsrMatrix {
    let n = 12;
    let (diffusion, convection) = (1.0, 0.6);
    let mut triplets = Vec::new();
    for row in 0..n {
        triplets.push((row, row, 2.0 * diffusion + convection));
        if row > 0 {
            triplets.push((row, row - 1, -diffusion - convection));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, -diffusion));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets).unwrap()
}

fn probe(dimension: usize, seed: f64) -> Vec<f64> {
    (0..dimension)
        .map(|index| (seed * (index as f64 + 1.0)).sin() + 0.25 * (index as f64 - 2.0))
        .collect()
}

fn inner(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// `<λ, b> = <g, u>` for `A u = b` and `Aᵀ λ = g`, through every admissible
/// nonsymmetric method, on one primal/transpose operator pair.
fn assert_adjoint_consistency<T: LinearOperator + TransposableOperator>(operator: &T) {
    let context = EvaluationContext::reproducible();
    let n = operator.rows();
    let b = probe(n, 1.3);
    let g = probe(n, 0.7);

    let primal = solve_krylov(
        &KrylovMethod::Gmres(GmresConfig {
            absolute_tolerance: 1.0e-14,
            relative_tolerance: 1.0e-14,
            ..GmresConfig::default()
        }),
        operator,
        None,
        None,
        &context,
        &b,
        &vec![0.0; n],
    )
    .unwrap();
    assert!(primal.converged);
    let u = primal.solution;

    let transpose = TransposeOperator::explicit(operator);
    assert_eq!(transpose.source(), TransposeSource::ExplicitTranspose);
    let acceptance = ResidualAcceptance {
        absolute_tolerance: 1.0e-14,
        relative_tolerance: 1.0e-13,
    };
    for method in [
        KrylovMethod::Gmres(GmresConfig::default()),
        KrylovMethod::BiCgStab(BiCgStabConfig::default()),
    ] {
        let report = solve_adjoint(
            &transpose,
            None,
            None,
            &context,
            &g,
            &vec![0.0; n],
            &AdjointConfig {
                method: method.clone(),
                acceptance: acceptance.clone(),
            },
        )
        .unwrap();
        assert!(
            report.converged,
            "{method:?} did not meet acceptance: {report:?}"
        );
        assert!(report.solver_converged);
        assert_eq!(report.method, method.kind());
        assert_eq!(report.transpose_source, TransposeSource::ExplicitTranspose);
        assert!(
            report.residual_norm
                <= acceptance.absolute_tolerance
                    + acceptance.relative_tolerance * report.gradient_norm
        );

        let lambda_b = inner(&report.adjoint, &b);
        let g_u = inner(&g, &u);
        let discrepancy = (lambda_b - g_u).abs();
        assert!(
            discrepancy < 1.0e-10,
            "{method:?}: <λ,b> = {lambda_b}, <g,u> = {g_u}, discrepancy {discrepancy}"
        );
    }
}

#[test]
fn adjoint_solve_is_consistent_on_a_nonsymmetric_dense_fixture() {
    let matrix = dense_fixture();
    assert_eq!(matrix.symmetry(), OperatorSymmetry::Nonsymmetric);
    assert_adjoint_consistency(&matrix);
}

#[test]
fn adjoint_solve_is_consistent_on_a_nonsymmetric_sparse_fixture() {
    let matrix = sparse_fixture();
    assert_eq!(matrix.symmetry(), OperatorSymmetry::Nonsymmetric);
    assert_adjoint_consistency(&matrix);
}

#[test]
fn conjugate_gradient_and_minres_are_refused_on_a_nonsymmetric_transpose() {
    let matrix = sparse_fixture();
    let transpose = TransposeOperator::explicit(&matrix);
    let n = matrix.rows();
    for method in [
        KrylovMethod::ConjugateGradient(ConjugateGradientConfig::default()),
        KrylovMethod::Minres(MinresConfig::default()),
    ] {
        let error = solve_adjoint(
            &transpose,
            None,
            None,
            &EvaluationContext::default(),
            &vec![1.0; n],
            &vec![0.0; n],
            &AdjointConfig {
                method,
                acceptance: ResidualAcceptance::default(),
            },
        )
        .unwrap_err();
        match error {
            SolveError::InvalidConfiguration { reason } => {
                assert!(reason.contains("adjoint solve refuses"), "{reason}");
            }
            other => panic!("expected a typed refusal, got {other:?}"),
        }
    }
}

/// A matrix-free operator without any transpose action and without a
/// `Symmetric` declaration: no honest transpose exists, so the transpose
/// view (and therefore any adjoint solve) is refused before any iteration.
#[derive(Debug)]
struct MatrixFreeNonsymmetric;

impl LinearOperator for MatrixFreeNonsymmetric {
    fn rows(&self) -> usize {
        2
    }
    fn columns(&self) -> usize {
        2
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
        output[0] = 2.0 * input[0] + input[1];
        output[1] = 3.0 * input[1];
        Ok(())
    }
}

#[test]
fn an_operator_without_a_transpose_is_refused_before_any_solve() {
    let error = TransposeOperator::new(&MatrixFreeNonsymmetric).unwrap_err();
    assert!(matches!(error, NumericError::InvalidInput { .. }));
    // `TransposeOperator::explicit` is not even constructible here: the
    // operator does not implement `TransposableOperator`, which the type
    // system enforces. The only remaining path is symmetric delegation,
    // refused above.
}

#[test]
fn a_rectangular_transpose_is_refused() {
    let rectangular = CsrMatrix::from_triplets(2, 3, vec![(0, 0, 1.0), (1, 2, 1.0)]).unwrap();
    let transpose = TransposeOperator::explicit(&rectangular);
    let error = solve_adjoint(
        &transpose,
        None,
        None,
        &EvaluationContext::default(),
        &[1.0, 1.0, 1.0],
        &[0.0, 0.0],
        &AdjointConfig {
            method: KrylovMethod::Gmres(GmresConfig::default()),
            acceptance: ResidualAcceptance::default(),
        },
    )
    .unwrap_err();
    assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
}

/// Diagonal preconditioner used as an approximate inverse of `Aᵀ`.
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
        for ((output, value), diagonal) in output.iter_mut().zip(right_hand_side).zip(&self.0) {
            *output = value / diagonal;
        }
        Ok(())
    }
}

#[test]
fn acceptance_is_on_the_true_residual_even_with_a_left_preconditioner() {
    let matrix = dense_fixture();
    let n = matrix.rows();
    let transpose = TransposeOperator::explicit(&matrix);
    let context = EvaluationContext::reproducible();
    let g = probe(n, 2.1);
    let preconditioner = DiagonalInverse((0..n).map(|index| matrix.entry(index, index)).collect());
    let report = solve_adjoint(
        &transpose,
        Some(&preconditioner),
        None,
        &context,
        &g,
        &vec![0.0; n],
        &AdjointConfig {
            method: KrylovMethod::Gmres(GmresConfig::default()),
            acceptance: ResidualAcceptance::default(),
        },
    )
    .unwrap();
    assert!(report.converged);

    // Independently recompute `‖g − Aᵀ λ‖` and compare with the report.
    let mut action = vec![0.0; n];
    matrix
        .apply_transpose(&context, &report.adjoint, &mut action)
        .unwrap();
    let true_norm = g
        .iter()
        .zip(&action)
        .map(|(g, a)| (g - a).powi(2))
        .sum::<f64>()
        .sqrt();
    assert_eq!(true_norm, report.residual_norm);
    assert!(true_norm <= 1.0e-12 + 1.0e-10 * report.gradient_norm);
}

#[test]
fn an_exhausted_budget_reports_non_acceptance_instead_of_claiming_convergence() {
    let matrix = sparse_fixture();
    let n = matrix.rows();
    let transpose = TransposeOperator::explicit(&matrix);
    let report = solve_adjoint(
        &transpose,
        None,
        None,
        &EvaluationContext::reproducible(),
        &probe(n, 0.9),
        &vec![0.0; n],
        &AdjointConfig {
            method: KrylovMethod::Gmres(GmresConfig {
                max_iterations: 2,
                restart: 2,
                ..GmresConfig::default()
            }),
            acceptance: ResidualAcceptance::default(),
        },
    )
    .unwrap();
    assert!(!report.converged);
    assert!(!report.solver_converged);
    assert!(report.residual_norm > 1.0e-12 + 1.0e-10 * report.gradient_norm);
    assert_eq!(report.method, KrylovMethodKind::Gmres);
    assert_eq!(report.restart_cycles, Some(1));
}

#[test]
fn adjoint_telemetry_is_bit_reproducible() {
    let matrix = sparse_fixture();
    let n = matrix.rows();
    let transpose = TransposeOperator::explicit(&matrix);
    let context = EvaluationContext::reproducible();
    let g = probe(n, 0.4);
    let config = AdjointConfig {
        method: KrylovMethod::BiCgStab(BiCgStabConfig::default()),
        acceptance: ResidualAcceptance::default(),
    };
    let first =
        solve_adjoint(&transpose, None, None, &context, &g, &vec![0.0; n], &config).unwrap();
    let second =
        solve_adjoint(&transpose, None, None, &context, &g, &vec![0.0; n], &config).unwrap();
    assert_eq!(first, second);
    let json = serde_json::to_string(&first).unwrap();
    let parsed: methodus::AdjointSolveReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, first);
}

#[test]
fn transpose_view_carries_the_primal_properties_for_square_operators() {
    use methodus::{Definiteness, OperatorProperties, OperatorStructureHint};

    struct Declared(CsrMatrix, OperatorProperties);
    impl LinearOperator for Declared {
        fn rows(&self) -> usize {
            self.0.rows()
        }
        fn columns(&self) -> usize {
            self.0.columns()
        }
        fn symmetry(&self) -> OperatorSymmetry {
            self.1.symmetry()
        }
        fn properties(&self) -> OperatorProperties {
            self.1.clone()
        }
        fn apply(
            &self,
            context: &EvaluationContext,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            self.0.apply(context, input, output)
        }
    }
    impl TransposableOperator for Declared {
        fn apply_transpose(
            &self,
            context: &EvaluationContext,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            self.0.apply_transpose(context, input, output)
        }
    }

    let properties = OperatorProperties::new(
        OperatorSymmetry::Nonsymmetric,
        Definiteness::PositiveDefinite,
        Some(0),
        OperatorStructureHint::Dense,
    )
    .unwrap();
    let operator = Declared(sparse_fixture(), properties.clone());
    let transpose = TransposeOperator::explicit(&operator);
    assert_eq!(transpose.properties(), properties);

    let rectangular = Declared(
        CsrMatrix::from_triplets(2, 3, vec![(0, 0, 1.0), (1, 2, 1.0)]).unwrap(),
        OperatorProperties::new(
            OperatorSymmetry::Nonsymmetric,
            Definiteness::Unknown,
            Some(1),
            OperatorStructureHint::Dense,
        )
        .unwrap(),
    );
    let transpose = TransposeOperator::explicit(&rectangular);
    assert_eq!(transpose.properties().nullspace_dimension(), None);
}
