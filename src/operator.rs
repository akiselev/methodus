use serde::{Deserialize, Serialize};

use crate::{BlockLayout, EvaluationContext, NumericError};

/// Declared symmetry of a real linear operator action.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorSymmetry {
    /// The implementation does not carry enough evidence to make a symmetry claim.
    #[default]
    Unknown,
    /// The action is known to be symmetric under the Euclidean inner product.
    Symmetric,
    /// The action is known not to be symmetric.
    Nonsymmetric,
}

/// Declared definiteness of a real linear operator action.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Definiteness {
    /// The implementation does not carry enough evidence to make a definiteness claim.
    #[default]
    Unknown,
    /// The operator's quadratic form is strictly positive away from the origin.
    PositiveDefinite,
    /// The operator's quadratic form is nonnegative and may vanish off the origin.
    PositiveSemidefinite,
    /// The operator's quadratic form takes both signs.
    Indefinite,
}

/// Coarse structural hint over a linear operator's block partition.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorStructureHint {
    /// No declared block partition.
    #[default]
    Dense,
    /// A declared block partition, optionally with a missing diagonal block
    /// coupled through off-diagonal terms (a saddle-point structure).
    Block {
        layout: BlockLayout,
        saddle_point: bool,
    },
}

/// Full operator-property metadata used by callers to decide algorithm admissibility.
///
/// Construction is validated: declaring [`Definiteness::PositiveDefinite`]
/// together with a nullspace dimension greater than zero is inconsistent,
/// because a positive-definite operator is nonsingular by definition, so it
/// is refused rather than accepted silently.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "OperatorPropertiesData")]
pub struct OperatorProperties {
    symmetry: OperatorSymmetry,
    definiteness: Definiteness,
    nullspace_dimension: Option<usize>,
    structure: OperatorStructureHint,
}

#[derive(Deserialize)]
struct OperatorPropertiesData {
    symmetry: OperatorSymmetry,
    definiteness: Definiteness,
    nullspace_dimension: Option<usize>,
    structure: OperatorStructureHint,
}

impl TryFrom<OperatorPropertiesData> for OperatorProperties {
    type Error = NumericError;

    fn try_from(data: OperatorPropertiesData) -> Result<Self, Self::Error> {
        Self::new(
            data.symmetry,
            data.definiteness,
            data.nullspace_dimension,
            data.structure,
        )
    }
}

impl OperatorProperties {
    /// Builds validated operator-property metadata.
    ///
    /// # Errors
    /// Refuses [`Definiteness::PositiveDefinite`] paired with a declared
    /// nullspace dimension greater than zero, because a positive-definite
    /// operator is nonsingular by definition.
    pub fn new(
        symmetry: OperatorSymmetry,
        definiteness: Definiteness,
        nullspace_dimension: Option<usize>,
        structure: OperatorStructureHint,
    ) -> Result<Self, NumericError> {
        if definiteness == Definiteness::PositiveDefinite
            && matches!(nullspace_dimension, Some(dimension) if dimension > 0)
        {
            return Err(NumericError::InvalidInput {
                message: format!(
                    "operator properties declare PositiveDefinite with nullspace_dimension \
                     {nullspace_dimension:?}; a positive-definite operator is nonsingular"
                ),
            });
        }
        Ok(Self {
            symmetry,
            definiteness,
            nullspace_dimension,
            structure,
        })
    }

    /// Derives properties from a symmetry declaration alone: unknown
    /// definiteness, no declared nullspace, dense structure.
    #[must_use]
    pub const fn from_symmetry(symmetry: OperatorSymmetry) -> Self {
        Self {
            symmetry,
            definiteness: Definiteness::Unknown,
            nullspace_dimension: None,
            structure: OperatorStructureHint::Dense,
        }
    }

    #[must_use]
    pub const fn symmetry(&self) -> OperatorSymmetry {
        self.symmetry
    }

    #[must_use]
    pub const fn definiteness(&self) -> Definiteness {
        self.definiteness
    }

    #[must_use]
    pub const fn nullspace_dimension(&self) -> Option<usize> {
        self.nullspace_dimension
    }

    #[must_use]
    pub fn structure(&self) -> &OperatorStructureHint {
        &self.structure
    }
}

impl Default for OperatorProperties {
    fn default() -> Self {
        Self::from_symmetry(OperatorSymmetry::default())
    }
}

/// Matrix-free action of a rectangular linear operator.
pub trait LinearOperator: Send + Sync {
    fn rows(&self) -> usize;
    fn columns(&self) -> usize;
    /// Report symmetry evidence for algorithms whose validity depends on it.
    ///
    /// Implementations must return [`OperatorSymmetry::Nonsymmetric`] when their construction is
    /// known to destroy symmetry. The default makes no claim.
    ///
    /// Retained for existing callers; new property queries should prefer
    /// [`LinearOperator::properties`]. Implementations that override either
    /// method must keep `properties().symmetry()` consistent with
    /// `symmetry()` — [`check_properties_consistency`] verifies this for a
    /// given operator.
    fn symmetry(&self) -> OperatorSymmetry {
        OperatorSymmetry::Unknown
    }
    /// Reports full operator-property metadata for algorithm admissibility decisions.
    ///
    /// Defaults to [`OperatorProperties::from_symmetry`] over
    /// [`LinearOperator::symmetry`]. See the consistency invariant documented
    /// on [`LinearOperator::symmetry`].
    fn properties(&self) -> OperatorProperties {
        OperatorProperties::from_symmetry(self.symmetry())
    }
    fn apply(
        &self,
        context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
}

/// Debug/validation helper verifying that `properties().symmetry()` agrees
/// with `symmetry()` for one operator.
///
/// Implementations must keep the two consistent; see the trait-level
/// invariant documented on [`LinearOperator::symmetry`]. Intended for tests
/// and debug assertions, not hot solver paths.
///
/// # Errors
/// Returns a typed error describing the mismatch when the two disagree.
pub fn check_properties_consistency(
    operator: &(impl LinearOperator + ?Sized),
) -> Result<(), NumericError> {
    let declared = operator.symmetry();
    let derived = operator.properties().symmetry();
    if declared == derived {
        Ok(())
    } else {
        Err(NumericError::InvalidInput {
            message: format!(
                "operator symmetry() = {declared:?} disagrees with properties().symmetry() \
                 = {derived:?}; the two must remain consistent"
            ),
        })
    }
}

/// Approximate inverse action used by iterative linear solvers.
pub trait Preconditioner: Send + Sync {
    fn dimension(&self) -> usize;
    fn apply_inverse(
        &self,
        context: &EvaluationContext,
        right_hand_side: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
}

/// In-place residual and Jacobian-vector products for `F(x) = 0`.
pub trait NonlinearOperator: Send + Sync {
    fn dimension(&self) -> usize;
    /// Declared properties of the Jacobian `∂F/∂x`, valid at every state
    /// the operator admits, used by Newton–Krylov to admit or refuse a
    /// linear method exactly as [`LinearOperator::properties`] does. The
    /// default makes no claim (`Unknown` symmetry), which conjugate gradient
    /// refuses without an explicit assumption and MINRES refuses outright.
    fn jacobian_properties(&self) -> OperatorProperties {
        OperatorProperties::default()
    }
    fn residual(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
}

/// In-place residual and directional derivatives for `F(t, y, ydot) = 0`.
pub trait DaeOperator: Send + Sync {
    fn dimension(&self) -> usize;
    /// Declared properties of the implicit-step Jacobian
    /// `∂F/∂y + α ∂F/∂ẏ` for every `α > 0` a BDF step may form, valid at
    /// every admitted state; see [`NonlinearOperator::jacobian_properties`].
    /// The default makes no claim.
    fn jacobian_properties(&self) -> OperatorProperties {
        OperatorProperties::default()
    }
    fn residual(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
    #[allow(clippy::too_many_arguments)]
    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;

    fn make_initial_state_consistent(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &mut [f64],
    ) -> Result<(), NumericError> {
        Ok(())
    }

    fn event_count(&self) -> usize {
        0
    }

    fn event_values(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        NumericError::require_len("DAE event output", output.len(), 0)
    }
}

/// Maximum absolute discrepancy between an analytic nonlinear JVP and a centered difference.
pub fn verify_jvp(
    operator: &(impl NonlinearOperator + ?Sized),
    context: &EvaluationContext,
    state: &[f64],
    direction: &[f64],
    epsilon: f64,
) -> Result<f64, NumericError> {
    let dimension = operator.dimension();
    validate_probe(dimension, state, direction, epsilon)?;
    let mut analytic = vec![0.0; dimension];
    operator.jacobian_vector_product(context, state, direction, &mut analytic)?;
    let plus = shifted(state, direction, epsilon);
    let minus = shifted(state, direction, -epsilon);
    let mut residual_plus = vec![0.0; dimension];
    let mut residual_minus = vec![0.0; dimension];
    operator.residual(context, &plus, &mut residual_plus)?;
    operator.residual(context, &minus, &mut residual_minus)?;
    discrepancy(&analytic, &residual_plus, &residual_minus, epsilon)
}

/// Maximum absolute discrepancy between an analytic DAE JVP and a centered difference.
#[allow(clippy::too_many_arguments)]
pub fn verify_dae_jvp(
    operator: &(impl DaeOperator + ?Sized),
    context: &EvaluationContext,
    time: f64,
    state: &[f64],
    state_rate: &[f64],
    state_direction: &[f64],
    rate_direction: &[f64],
    epsilon: f64,
) -> Result<f64, NumericError> {
    let dimension = operator.dimension();
    validate_probe(dimension, state, state_direction, epsilon)?;
    validate_probe(dimension, state_rate, rate_direction, epsilon)?;
    let mut analytic = vec![0.0; dimension];
    operator.jacobian_vector_product(
        context,
        time,
        state,
        state_rate,
        state_direction,
        rate_direction,
        &mut analytic,
    )?;
    let state_plus = shifted(state, state_direction, epsilon);
    let state_minus = shifted(state, state_direction, -epsilon);
    let rate_plus = shifted(state_rate, rate_direction, epsilon);
    let rate_minus = shifted(state_rate, rate_direction, -epsilon);
    let mut residual_plus = vec![0.0; dimension];
    let mut residual_minus = vec![0.0; dimension];
    operator.residual(context, time, &state_plus, &rate_plus, &mut residual_plus)?;
    operator.residual(
        context,
        time,
        &state_minus,
        &rate_minus,
        &mut residual_minus,
    )?;
    discrepancy(&analytic, &residual_plus, &residual_minus, epsilon)
}

fn validate_probe(
    dimension: usize,
    values: &[f64],
    direction: &[f64],
    epsilon: f64,
) -> Result<(), NumericError> {
    NumericError::require_len("verification state", values.len(), dimension)?;
    NumericError::require_len("verification direction", direction.len(), dimension)?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(NumericError::InvalidInput {
            message: "verification epsilon must be finite and positive".into(),
        });
    }
    Ok(())
}

fn shifted(values: &[f64], direction: &[f64], factor: f64) -> Vec<f64> {
    values
        .iter()
        .zip(direction)
        .map(|(value, delta)| value + factor * delta)
        .collect()
}

fn discrepancy(
    analytic: &[f64],
    residual_plus: &[f64],
    residual_minus: &[f64],
    epsilon: f64,
) -> Result<f64, NumericError> {
    NumericError::require_finite("analytic JVP", analytic)?;
    NumericError::require_finite("positive residual probe", residual_plus)?;
    NumericError::require_finite("negative residual probe", residual_minus)?;
    Ok(analytic
        .iter()
        .zip(residual_plus.iter().zip(residual_minus))
        .map(|(analytic, (plus, minus))| (analytic - (plus - minus) / (2.0 * epsilon)).abs())
        .fold(0.0, f64::max))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedSymmetry(OperatorSymmetry);

    impl LinearOperator for FixedSymmetry {
        fn rows(&self) -> usize {
            1
        }

        fn columns(&self) -> usize {
            1
        }

        fn symmetry(&self) -> OperatorSymmetry {
            self.0
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

    /// Deliberately violates the `symmetry()`/`properties().symmetry()`
    /// consistency invariant so [`check_properties_consistency`] can be
    /// exercised against a genuine mismatch.
    struct InconsistentOperator;

    impl LinearOperator for InconsistentOperator {
        fn rows(&self) -> usize {
            1
        }

        fn columns(&self) -> usize {
            1
        }

        fn symmetry(&self) -> OperatorSymmetry {
            OperatorSymmetry::Symmetric
        }

        fn properties(&self) -> OperatorProperties {
            OperatorProperties::from_symmetry(OperatorSymmetry::Nonsymmetric)
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

    #[test]
    fn properties_default_is_all_unknown_and_dense() {
        let properties = OperatorProperties::default();
        assert_eq!(properties.symmetry(), OperatorSymmetry::Unknown);
        assert_eq!(properties.definiteness(), Definiteness::Unknown);
        assert_eq!(properties.nullspace_dimension(), None);
        assert_eq!(*properties.structure(), OperatorStructureHint::Dense);
    }

    #[test]
    fn from_symmetry_derives_unknown_definiteness_and_dense_structure() {
        let properties = OperatorProperties::from_symmetry(OperatorSymmetry::Symmetric);
        assert_eq!(properties.symmetry(), OperatorSymmetry::Symmetric);
        assert_eq!(properties.definiteness(), Definiteness::Unknown);
        assert_eq!(properties.nullspace_dimension(), None);
        assert_eq!(*properties.structure(), OperatorStructureHint::Dense);
    }

    #[test]
    fn linear_operator_default_properties_matches_from_symmetry() {
        for symmetry in [
            OperatorSymmetry::Unknown,
            OperatorSymmetry::Symmetric,
            OperatorSymmetry::Nonsymmetric,
        ] {
            let operator = FixedSymmetry(symmetry);
            assert_eq!(
                operator.properties(),
                OperatorProperties::from_symmetry(symmetry)
            );
        }
    }

    #[test]
    fn positive_definite_with_positive_nullspace_dimension_is_rejected() {
        let error = OperatorProperties::new(
            OperatorSymmetry::Symmetric,
            Definiteness::PositiveDefinite,
            Some(2),
            OperatorStructureHint::Dense,
        )
        .unwrap_err();
        assert!(matches!(error, NumericError::InvalidInput { .. }));
    }

    #[test]
    fn positive_definite_with_zero_or_absent_nullspace_dimension_is_accepted() {
        for nullspace_dimension in [None, Some(0)] {
            OperatorProperties::new(
                OperatorSymmetry::Symmetric,
                Definiteness::PositiveDefinite,
                nullspace_dimension,
                OperatorStructureHint::Dense,
            )
            .unwrap();
        }
    }

    #[test]
    fn positive_semidefinite_permits_a_positive_nullspace_dimension() {
        OperatorProperties::new(
            OperatorSymmetry::Symmetric,
            Definiteness::PositiveSemidefinite,
            Some(1),
            OperatorStructureHint::Dense,
        )
        .unwrap();
    }

    #[test]
    fn deserialization_revalidates_operator_properties_invariants() {
        let malformed = r#"{
            "symmetry": "symmetric",
            "definiteness": "positive_definite",
            "nullspace_dimension": 3,
            "structure": "dense"
        }"#;
        assert!(serde_json::from_str::<OperatorProperties>(malformed).is_err());

        let well_formed = r#"{
            "symmetry": "symmetric",
            "definiteness": "positive_definite",
            "nullspace_dimension": null,
            "structure": "dense"
        }"#;
        let properties: OperatorProperties = serde_json::from_str(well_formed).unwrap();
        assert_eq!(properties.definiteness(), Definiteness::PositiveDefinite);
    }

    #[test]
    fn check_properties_consistency_accepts_matching_operators() {
        for symmetry in [
            OperatorSymmetry::Unknown,
            OperatorSymmetry::Symmetric,
            OperatorSymmetry::Nonsymmetric,
        ] {
            check_properties_consistency(&FixedSymmetry(symmetry)).unwrap();
        }
    }

    #[test]
    fn check_properties_consistency_rejects_a_genuine_mismatch() {
        let error = check_properties_consistency(&InconsistentOperator).unwrap_err();
        assert!(matches!(error, NumericError::InvalidInput { .. }));
    }
}
