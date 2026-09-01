use serde::{Deserialize, Serialize};

use crate::{
    Definiteness, EvaluationContext, LinearOperator, NumericError, OperatorSymmetry,
    Preconditioner, SolveError,
};

/// How conjugate gradient handles an operator whose symmetry is not declared.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConjugateGradientSymmetryPolicy {
    /// Refuse an operator without an affirmative symmetry declaration.
    #[default]
    RequireDeclared,
    /// Record an explicit caller assertion that an otherwise-unknown operator is symmetric.
    AssumeSymmetric,
}

/// Convergence policy for a conjugate-gradient solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConjugateGradientConfig {
    pub max_iterations: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    #[serde(default)]
    pub symmetry_policy: ConjugateGradientSymmetryPolicy,
}

impl Default for ConjugateGradientConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-10,
            symmetry_policy: ConjugateGradientSymmetryPolicy::RequireDeclared,
        }
    }
}

/// One deterministic residual observation from a linear solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearIteration {
    pub iteration: usize,
    pub residual_norm: f64,
}

/// Final state and convergence evidence from a linear solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearSolveReport {
    pub solution: Vec<f64>,
    pub converged: bool,
    pub trace: Vec<LinearIteration>,
}

/// Solve a symmetric positive-definite system through operator and optional preconditioner
/// actions. The implementation fixes the reduction and update order so reproducible downstream
/// operators yield a reproducible reference solve.
pub fn solve_conjugate_gradient(
    operator: &(impl LinearOperator + ?Sized),
    preconditioner: Option<&dyn Preconditioner>,
    context: &EvaluationContext,
    right_hand_side: &[f64],
    initial_solution: &[f64],
    config: &ConjugateGradientConfig,
) -> Result<LinearSolveReport, SolveError> {
    validate_config(config)?;
    let dimension = operator.rows();
    if operator.columns() != dimension {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "conjugate gradient requires a square operator, got {}x{}",
                operator.rows(),
                operator.columns()
            ),
        });
    }
    match operator.symmetry() {
        OperatorSymmetry::Nonsymmetric => {
            return Err(SolveError::InvalidConfiguration {
                reason: "conjugate gradient refuses an operator declared nonsymmetric".into(),
            });
        }
        OperatorSymmetry::Unknown
            if config.symmetry_policy == ConjugateGradientSymmetryPolicy::RequireDeclared =>
        {
            return Err(SolveError::InvalidConfiguration {
                reason:
                    "conjugate gradient requires declared symmetry or an explicit caller assumption"
                        .into(),
            });
        }
        OperatorSymmetry::Unknown | OperatorSymmetry::Symmetric => {}
    }
    let properties = operator.properties();
    if properties.definiteness() == Definiteness::Indefinite {
        return Err(SolveError::InvalidConfiguration {
            reason: "conjugate gradient refuses an operator declared Indefinite; no projection \
                     or deflation is supplied"
                .into(),
        });
    }
    if matches!(properties.nullspace_dimension(), Some(dimension) if dimension > 0) {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "conjugate gradient refuses an operator with a declared nullspace dimension of \
                 {:?}; no projection or deflation is supplied",
                properties.nullspace_dimension()
            ),
        });
    }
    NumericError::require_len("linear right-hand side", right_hand_side.len(), dimension)?;
    NumericError::require_len("initial linear solution", initial_solution.len(), dimension)?;
    NumericError::require_finite("linear right-hand side", right_hand_side)?;
    NumericError::require_finite("initial linear solution", initial_solution)?;
    if let Some(preconditioner) = preconditioner
        && preconditioner.dimension() != dimension
    {
        return Err(SolveError::InvalidConfiguration {
            reason: format!(
                "preconditioner dimension {} differs from operator dimension {dimension}",
                preconditioner.dimension()
            ),
        });
    }

    let mut solution = initial_solution.to_vec();
    let mut action = vec![0.0; dimension];
    operator.apply(context, &solution, &mut action)?;
    NumericError::require_finite("initial linear operator action", &action)?;
    let mut residual = right_hand_side
        .iter()
        .zip(&action)
        .map(|(rhs, action)| rhs - action)
        .collect::<Vec<_>>();
    let initial_norm = l2(&residual)?;
    let threshold = config.absolute_tolerance + config.relative_tolerance * initial_norm;
    let mut trace = vec![LinearIteration {
        iteration: 0,
        residual_norm: initial_norm,
    }];
    if initial_norm <= threshold {
        return Ok(LinearSolveReport {
            solution,
            converged: true,
            trace,
        });
    }

    let mut preconditioned = vec![0.0; dimension];
    apply_preconditioner(preconditioner, context, &residual, &mut preconditioned)?;
    let mut search = preconditioned.clone();
    let mut residual_product = dot(&residual, &preconditioned)?;
    if residual_product <= 0.0 {
        return Err(SolveError::KrylovBreakdown { iteration: 0 });
    }

    for iteration in 1..=config.max_iterations {
        operator.apply(context, &search, &mut action)?;
        NumericError::require_finite("linear operator action", &action)?;
        let curvature = dot(&search, &action)?;
        if curvature <= 0.0 {
            return Err(SolveError::KrylovBreakdown { iteration });
        }
        let alpha = residual_product / curvature;
        NumericError::require_finite("conjugate-gradient step", &[alpha])?;
        for index in 0..dimension {
            solution[index] += alpha * search[index];
            residual[index] -= alpha * action[index];
        }
        NumericError::require_finite("linear solution", &solution)?;
        NumericError::require_finite("linear residual", &residual)?;
        let residual_norm = l2(&residual)?;
        trace.push(LinearIteration {
            iteration,
            residual_norm,
        });
        if residual_norm <= threshold {
            return Ok(LinearSolveReport {
                solution,
                converged: true,
                trace,
            });
        }

        apply_preconditioner(preconditioner, context, &residual, &mut preconditioned)?;
        let next_product = dot(&residual, &preconditioned)?;
        if next_product <= 0.0 {
            return Err(SolveError::KrylovBreakdown { iteration });
        }
        let beta = next_product / residual_product;
        NumericError::require_finite("conjugate-gradient recurrence", &[beta])?;
        for index in 0..dimension {
            search[index] = preconditioned[index] + beta * search[index];
        }
        residual_product = next_product;
    }

    Ok(LinearSolveReport {
        solution,
        converged: false,
        trace,
    })
}

fn validate_config(config: &ConjugateGradientConfig) -> Result<(), SolveError> {
    let tolerances_valid = config.absolute_tolerance.is_finite()
        && config.absolute_tolerance >= 0.0
        && config.relative_tolerance.is_finite()
        && config.relative_tolerance >= 0.0
        && (config.absolute_tolerance > 0.0 || config.relative_tolerance > 0.0);
    if config.max_iterations == 0 || !tolerances_valid {
        return Err(SolveError::InvalidConfiguration {
            reason: "linear iteration limit and tolerances must be positive and finite".into(),
        });
    }
    Ok(())
}

pub(crate) fn apply_preconditioner(
    preconditioner: Option<&dyn Preconditioner>,
    context: &EvaluationContext,
    input: &[f64],
    output: &mut [f64],
) -> Result<(), NumericError> {
    if let Some(preconditioner) = preconditioner {
        preconditioner.apply_inverse(context, input, output)?;
    } else {
        output.copy_from_slice(input);
    }
    NumericError::require_finite("preconditioner action", output)
}

pub(crate) fn dot(left: &[f64], right: &[f64]) -> Result<f64, NumericError> {
    let value = left
        .iter()
        .zip(right)
        .fold(0.0, |sum, (left, right)| sum + left * right);
    NumericError::require_finite("linear inner product", &[value])?;
    Ok(value)
}

pub(crate) fn l2(values: &[f64]) -> Result<f64, NumericError> {
    let norm = dot(values, values)?.sqrt();
    NumericError::require_finite("linear residual norm", &[norm])?;
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrMatrix;

    struct DiagonalInverse(Vec<f64>);

    /// A declared-symmetric identity operator carrying additional
    /// [`crate::OperatorProperties`] metadata, used to exercise the
    /// definiteness/nullspace refusal paths independent of the symmetry check.
    struct SymmetricWithProperties {
        properties: crate::OperatorProperties,
    }

    impl LinearOperator for SymmetricWithProperties {
        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            2
        }

        fn symmetry(&self) -> OperatorSymmetry {
            OperatorSymmetry::Symmetric
        }

        fn properties(&self) -> crate::OperatorProperties {
            self.properties.clone()
        }

        fn apply(
            &self,
            _context: &EvaluationContext,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            output.copy_from_slice(input);
            Ok(())
        }
    }

    struct UnknownIdentity;

    impl LinearOperator for UnknownIdentity {
        fn rows(&self) -> usize {
            1
        }

        fn columns(&self) -> usize {
            1
        }

        fn apply(
            &self,
            _context: &EvaluationContext,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            output.copy_from_slice(input);
            Ok(())
        }
    }

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
    fn conjugate_gradient_uses_operator_and_preconditioner_actions() {
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, 4.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 4.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 3.0),
            ],
        )
        .unwrap();
        let preconditioner = DiagonalInverse(vec![4.0, 4.0, 3.0]);
        let report = solve_conjugate_gradient(
            &matrix,
            Some(&preconditioner),
            &EvaluationContext::reproducible(),
            &[15.0, 10.0, 10.0],
            &[0.0; 3],
            &ConjugateGradientConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
        for (actual, expected) in report.solution.iter().zip([5.0; 3]) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        assert!(report.trace.last().unwrap().residual_norm < 1.0e-12);
    }

    #[test]
    fn conjugate_gradient_refuses_rectangular_operators() {
        let matrix = CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![1.0]).unwrap();
        let error = solve_conjugate_gradient(
            &matrix,
            None,
            &EvaluationContext::default(),
            &[1.0],
            &[0.0, 0.0],
            &ConjugateGradientConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
    }

    #[test]
    fn conjugate_gradient_refuses_declared_nonsymmetric_operators() {
        let matrix =
            CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0), (0, 1, 1.0), (1, 1, 2.0)]).unwrap();
        assert_eq!(matrix.symmetry(), OperatorSymmetry::Nonsymmetric);
        let error = solve_conjugate_gradient(
            &matrix,
            None,
            &EvaluationContext::default(),
            &[1.0, 1.0],
            &[0.0; 2],
            &ConjugateGradientConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
    }

    #[test]
    fn conjugate_gradient_requires_an_explicit_unknown_symmetry_assumption() {
        let error = solve_conjugate_gradient(
            &UnknownIdentity,
            None,
            &EvaluationContext::default(),
            &[2.0],
            &[0.0],
            &ConjugateGradientConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));

        let config = ConjugateGradientConfig {
            symmetry_policy: ConjugateGradientSymmetryPolicy::AssumeSymmetric,
            ..ConjugateGradientConfig::default()
        };
        let report = solve_conjugate_gradient(
            &UnknownIdentity,
            None,
            &EvaluationContext::default(),
            &[2.0],
            &[0.0],
            &config,
        )
        .unwrap();
        assert!(report.converged);
        assert_eq!(report.solution, [2.0]);
    }

    #[test]
    fn conjugate_gradient_refuses_declared_indefinite_operators() {
        let operator = SymmetricWithProperties {
            properties: crate::OperatorProperties::new(
                OperatorSymmetry::Symmetric,
                Definiteness::Indefinite,
                None,
                crate::OperatorStructureHint::Dense,
            )
            .unwrap(),
        };
        let error = solve_conjugate_gradient(
            &operator,
            None,
            &EvaluationContext::default(),
            &[1.0, 1.0],
            &[0.0; 2],
            &ConjugateGradientConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
    }

    #[test]
    fn conjugate_gradient_refuses_a_declared_positive_nullspace_dimension() {
        let operator = SymmetricWithProperties {
            properties: crate::OperatorProperties::new(
                OperatorSymmetry::Symmetric,
                Definiteness::PositiveSemidefinite,
                Some(1),
                crate::OperatorStructureHint::Dense,
            )
            .unwrap(),
        };
        let error = solve_conjugate_gradient(
            &operator,
            None,
            &EvaluationContext::default(),
            &[1.0, 1.0],
            &[0.0; 2],
            &ConjugateGradientConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));
    }

    #[test]
    fn conjugate_gradient_accepts_a_declared_zero_nullspace_dimension() {
        let operator = SymmetricWithProperties {
            properties: crate::OperatorProperties::new(
                OperatorSymmetry::Symmetric,
                Definiteness::PositiveDefinite,
                Some(0),
                crate::OperatorStructureHint::Dense,
            )
            .unwrap(),
        };
        let report = solve_conjugate_gradient(
            &operator,
            None,
            &EvaluationContext::default(),
            &[2.0, -3.0],
            &[0.0; 2],
            &ConjugateGradientConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
        assert_eq!(report.solution, [2.0, -3.0]);
    }
}
