//! SC-W3 (SV7-F3 subset): consumer-neutral acceleration of a fixed-point
//! iterate sequence `x_{k+1} = G(x_k)` over flat `f64` slices, the shape a
//! partitioned coupled solve (e.g. Krasis's block Gauss-Seidel/Jacobi
//! transaction) converges by. Methodus never sees what `G` computes; it
//! only ever consumes one evaluation (or, inside [`crate::solve_blocks`],
//! one partitioned Newton correction) per iteration and the vectors that
//! evaluation produced.
//!
//! Two accelerators are provided, both applied to the fixed-point residual
//! `r_k = G(x_k) - x_k`:
//! - fixed under-relaxation: `x_{k+1} = x_k + factor * r_k` for a constant
//!   `factor`;
//! - Aitken `Δ²` dynamic relaxation (vector form): the same update with a
//!   `factor` recomputed every iteration from the two most recent
//!   residuals.
//!
//! Every outcome is typed: a nonfinite evaluation or update is refused
//! through [`NumericError`], a degenerate Aitken denominator is refused as
//! [`SolveError::AccelerationBreakdown`], and exhausting `max_iterations`
//! without meeting tolerance is refused as
//! [`SolveError::AccelerationNotConverged`] carrying the full iteration
//! trace — never a silently returned, unconverged iterate.

use serde::{Deserialize, Serialize};

use crate::context::EvaluationContext;
use crate::error::{NumericError, SolveError};
use crate::linear::l2;

/// One fixed-point map evaluation `G(x)` an accelerator drives, over flat
/// state and image vectors.
pub trait FixedPointOperator: Send + Sync {
    fn dimension(&self) -> usize;
    /// Evaluates `G(state)` into `output`.
    ///
    /// # Errors
    /// Propagates the evaluation's own failure, including a typed
    /// producer [`NumericError::Evaluation`].
    fn evaluate(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
}

/// Relaxation policy for [`accelerate_fixed_point`] and the
/// [`crate::solve_blocks`] partitioned driver's optional acceleration.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerationMethod {
    /// `x_{k+1} = x_k + factor * (G(x_k) - x_k)` with a constant `factor`.
    FixedRelaxation { factor: f64 },
    /// Vector Aitken `Δ²` dynamic relaxation (Irons-Tuck): `initial_factor`
    /// is used for the first step; every later step recomputes the factor
    /// from the two most recent fixed-point residuals.
    Aitken { initial_factor: f64 },
}

impl AccelerationMethod {
    /// The relaxation factor to start from: the constant factor for fixed
    /// relaxation, the first-step factor for Aitken (recomputed from the
    /// second step on).
    pub(crate) fn initial_factor(self) -> f64 {
        match self {
            Self::FixedRelaxation { factor } => factor,
            Self::Aitken { initial_factor } => initial_factor,
        }
    }

    /// Whether the declared/initial relaxation factor is a finite value in
    /// `(0, 1]`.
    pub(crate) fn is_valid(self) -> bool {
        let factor = self.initial_factor();
        factor.is_finite() && factor > 0.0 && factor <= 1.0
    }
}

/// Convergence policy for a standalone [`accelerate_fixed_point`] run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AccelerationConfig {
    pub method: AccelerationMethod,
    pub max_iterations: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}

/// One accelerated iteration's evidence: the fixed-point residual norm at
/// this iterate and the relaxation factor applied to it to produce the
/// next iterate — `None` on the terminal entry (converged, or budget
/// exhausted), where no update was applied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AccelerationIteration {
    pub iteration: usize,
    pub residual_norm: f64,
    pub relaxation_factor: Option<f64>,
}

/// Converged state and per-iteration evidence from an accelerated
/// fixed-point solve. Only returned once tolerance is met: a run that
/// exhausts its iteration budget is refused as
/// [`SolveError::AccelerationNotConverged`], which carries the same trace,
/// never an unconverged report (see [`accelerate_fixed_point`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AccelerationReport {
    pub state: Vec<f64>,
    pub trace: Vec<AccelerationIteration>,
}

/// Drives `x_{k+1} = G(x_k)` under fixed or Aitken relaxation to
/// `‖G(x_k) - x_k‖ ≤ absolute_tolerance + relative_tolerance * ‖G(x_0) -
/// x_0‖`. Convergence is tested before any relaxation factor is computed,
/// so a converged sequence is reported converged, and an Aitken factor is
/// only ever computed for an update that is then applied (and recorded on
/// that iterate's trace entry).
///
/// # Errors
/// Refuses invalid configuration (non-finite/out-of-`(0, 1]` relaxation
/// factors, a zero iteration budget or non-finite/non-positive
/// tolerances), a dimension mismatch or non-finite `initial_state`, any
/// non-finite value the operator or the update produces, a degenerate
/// Aitken denominator ([`SolveError::AccelerationBreakdown`]), and
/// exhausting `max_iterations` without meeting tolerance
/// ([`SolveError::AccelerationNotConverged`], carrying the trace).
pub fn accelerate_fixed_point(
    operator: &(impl FixedPointOperator + ?Sized),
    context: &EvaluationContext,
    initial_state: &[f64],
    config: &AccelerationConfig,
) -> Result<AccelerationReport, SolveError> {
    validate_acceleration_config(config)?;
    let dimension = operator.dimension();
    NumericError::require_len("initial acceleration state", initial_state.len(), dimension)?;
    NumericError::require_finite("initial acceleration state", initial_state)?;

    let mut state = initial_state.to_vec();
    let mut image = vec![0.0; dimension];
    let mut factor = config.method.initial_factor();
    let mut previous_residual: Option<Vec<f64>> = None;
    let mut threshold = None;
    let trace_capacity =
        config
            .max_iterations
            .checked_add(1)
            .ok_or_else(|| SolveError::InvalidConfiguration {
                reason: "acceleration iteration trace capacity overflows usize".into(),
            })?;
    let mut trace = Vec::with_capacity(trace_capacity);

    for iteration in 0..=config.max_iterations {
        operator.evaluate(context, &state, &mut image)?;
        NumericError::require_finite("fixed-point evaluation", &image)?;
        let residual = image
            .iter()
            .zip(&state)
            .map(|(g, x)| g - x)
            .collect::<Vec<_>>();
        let residual_norm = l2(&residual)?;
        let threshold = *threshold
            .get_or_insert(config.absolute_tolerance + config.relative_tolerance * residual_norm);

        // Terminal entries (converged, or budget exhausted) apply no update
        // and so record no factor; the convergence test precedes any
        // Aitken factor computation, which only ever sees a residual pair
        // that is about to be used.
        if residual_norm <= threshold {
            trace.push(AccelerationIteration {
                iteration,
                residual_norm,
                relaxation_factor: None,
            });
            return Ok(AccelerationReport { state, trace });
        }
        if iteration == config.max_iterations {
            trace.push(AccelerationIteration {
                iteration,
                residual_norm,
                relaxation_factor: None,
            });
            return Err(SolveError::AccelerationNotConverged { trace });
        }

        if let (AccelerationMethod::Aitken { .. }, Some(previous)) =
            (&config.method, &previous_residual)
        {
            factor = aitken_relaxation_factor(iteration, factor, previous, &residual)?;
        }
        trace.push(AccelerationIteration {
            iteration,
            residual_norm,
            relaxation_factor: Some(factor),
        });
        for (value, delta) in state.iter_mut().zip(&residual) {
            *value += factor * delta;
        }
        NumericError::require_finite("accelerated fixed-point state", &state)?;
        previous_residual = Some(residual);
    }
    unreachable!("iteration loop always returns")
}

/// Relative floor below which an Aitken denominator `‖Δr‖²` is degenerate:
/// `‖Δr‖ ≤ ε · max(‖r_k‖, ‖r_{k-1}‖)` with this `ε`, i.e. the two residuals
/// agree to eight digits and their difference is dominated by rounding.
const AITKEN_DEGENERACY_EPSILON: f64 = 1.0e-8;

/// Vector Aitken `Δ²` factor from the previous and current fixed-point
/// residuals: `ω_k = -ω_{k-1} · (r_{k-1} · Δr_{k-1}) / ‖Δr_{k-1}‖²` where
/// `Δr_{k-1} = r_k - r_{k-1}`.
///
/// # Errors
/// Refuses a degenerate denominator — exactly zero, non-finite, or
/// negligible relative to the residuals themselves,
/// `‖Δr‖² ≤ ε²·max(‖r_k‖², ‖r_{k-1}‖²)` with `ε = 1e-8`, so the judgement
/// is scale-invariant (a sequence at scale `1e-9` is treated exactly as one
/// at scale `1`) — or a resulting non-finite factor, as
/// [`SolveError::AccelerationBreakdown`].
pub(crate) fn aitken_relaxation_factor(
    iteration: usize,
    previous_factor: f64,
    previous_residual: &[f64],
    current_residual: &[f64],
) -> Result<f64, SolveError> {
    let delta = current_residual
        .iter()
        .zip(previous_residual)
        .map(|(current, previous)| current - previous)
        .collect::<Vec<_>>();
    let squared_norm = |values: &[f64]| values.iter().map(|value| value * value).sum::<f64>();
    let denominator = squared_norm(&delta);
    let residual_scale = squared_norm(current_residual).max(squared_norm(previous_residual));
    if !denominator.is_finite() || denominator <= AITKEN_DEGENERACY_EPSILON.powi(2) * residual_scale
    {
        return Err(SolveError::AccelerationBreakdown {
            iteration,
            reason: "Aitken denominator ‖Δr‖² is degenerate (zero, non-finite, or at most \
                     1e-16 of the residual scale max(‖r_k‖², ‖r_{k-1}‖²))"
                .into(),
        });
    }
    let numerator = previous_residual
        .iter()
        .zip(&delta)
        .map(|(previous, delta)| previous * delta)
        .sum::<f64>();
    let factor = -previous_factor * numerator / denominator;
    if !factor.is_finite() {
        return Err(SolveError::AccelerationBreakdown {
            iteration,
            reason: "Aitken factor is non-finite".into(),
        });
    }
    Ok(factor)
}

fn validate_acceleration_config(config: &AccelerationConfig) -> Result<(), SolveError> {
    let tolerances_valid = config.absolute_tolerance.is_finite()
        && config.absolute_tolerance >= 0.0
        && config.relative_tolerance.is_finite()
        && config.relative_tolerance >= 0.0
        && (config.absolute_tolerance > 0.0 || config.relative_tolerance > 0.0);
    if config.max_iterations == 0 || !tolerances_valid || !config.method.is_valid() {
        return Err(SolveError::InvalidConfiguration {
            reason: "acceleration iteration limit, tolerances and relaxation factor must be \
                     positive, finite and (for the factor) at most 1"
                .into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `G(x) = matrix * x`, a linear fixed-point map with a known spectrum.
    struct LinearMap {
        matrix: Vec<Vec<f64>>,
    }

    impl FixedPointOperator for LinearMap {
        fn dimension(&self) -> usize {
            self.matrix.len()
        }

        fn evaluate(
            &self,
            _context: &EvaluationContext,
            state: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            for (row, value) in self.matrix.iter().zip(output.iter_mut()) {
                *value = row.iter().zip(state).map(|(a, b)| a * b).sum();
            }
            Ok(())
        }
    }

    #[test]
    fn configuration_rejects_out_of_bounds_relaxation_factor() {
        for factor in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            let config = AccelerationConfig {
                method: AccelerationMethod::FixedRelaxation { factor },
                max_iterations: 10,
                absolute_tolerance: 1.0e-10,
                relative_tolerance: 0.0,
            };
            assert!(matches!(
                validate_acceleration_config(&config),
                Err(SolveError::InvalidConfiguration { .. })
            ));
        }
    }

    /// A decoupled, diagonally contractive map: eigenvalues 0.5 and 0.9.
    fn contractive_map() -> LinearMap {
        LinearMap {
            matrix: vec![vec![0.5, 0.0], vec![0.0, 0.9]],
        }
    }

    fn aitken_config(absolute_tolerance: f64) -> AccelerationConfig {
        AccelerationConfig {
            method: AccelerationMethod::Aitken {
                initial_factor: 1.0,
            },
            max_iterations: 500,
            absolute_tolerance,
            relative_tolerance: 0.0,
        }
    }

    #[test]
    fn aitken_converges_a_contractive_two_block_map_in_fewer_iterations_than_plain_iteration() {
        // Full-step (factor = 1) plain iteration converges at rate 0.9 per
        // step; Aitken adapts its scalar relaxation factor from the
        // residual history and needs far fewer iterations.
        let map = contractive_map();
        let context = EvaluationContext::reproducible();
        let plain_config = AccelerationConfig {
            method: AccelerationMethod::FixedRelaxation { factor: 1.0 },
            max_iterations: 500,
            absolute_tolerance: 1.0e-6,
            relative_tolerance: 0.0,
        };
        let plain = accelerate_fixed_point(&map, &context, &[1.0, 1.0], &plain_config).unwrap();
        let aitken =
            accelerate_fixed_point(&map, &context, &[1.0, 1.0], &aitken_config(1.0e-6)).unwrap();

        assert_eq!((plain.trace.len(), aitken.trace.len()), (111, 8));
        assert!(aitken.state.iter().all(|value| value.abs() < 1.0e-4));
    }

    #[test]
    fn aitken_converges_at_a_tight_tolerance_instead_of_a_false_breakdown() {
        // At absolute_tolerance 1e-9 the first cut (24f9c7d) refused
        // `AccelerationBreakdown { iteration: 8 }`: it computed a factor
        // from an already-converged residual pair before testing
        // convergence, against an absolute 1e-14 floor. The convergence
        // test now precedes the factor, and the floor is relative.
        let report = accelerate_fixed_point(
            &contractive_map(),
            &EvaluationContext::reproducible(),
            &[1.0, 1.0],
            &aitken_config(1.0e-9),
        )
        .unwrap();
        assert!(report.state.iter().all(|value| value.abs() < 1.0e-7));
        assert_eq!(report.trace.len(), 10);
    }

    #[test]
    fn aitken_degeneracy_test_is_scale_invariant() {
        // The same contractive sequence started at scale 1e-9, with the
        // tolerance scaled alike, converges in exactly as many iterations
        // as at scale 1 and with the same factors: the Aitken denominator
        // is judged relative to the residuals, not against an absolute
        // floor (which refused the small-scale run at its first factor).
        let map = contractive_map();
        let context = EvaluationContext::reproducible();
        let unit =
            accelerate_fixed_point(&map, &context, &[1.0, 1.0], &aitken_config(1.0e-6)).unwrap();
        let small =
            accelerate_fixed_point(&map, &context, &[1.0e-9, 1.0e-9], &aitken_config(1.0e-15))
                .unwrap();
        assert_eq!(small.trace.len(), unit.trace.len());
        for (small, unit) in small.trace.iter().zip(&unit.trace) {
            match (small.relaxation_factor, unit.relaxation_factor) {
                (Some(small), Some(unit)) => assert!((small - unit).abs() < 1.0e-6 * unit.abs()),
                (None, None) => {}
                other => panic!("factor presence differs between scales: {other:?}"),
            }
        }
    }

    #[test]
    fn trace_records_each_factor_on_the_entry_it_was_applied_to() {
        // Replaying the recorded factors from the initial state reproduces
        // the returned state bit for bit; the terminal (converged) entry
        // applied no update and carries no factor.
        let map = contractive_map();
        let context = EvaluationContext::reproducible();
        let report =
            accelerate_fixed_point(&map, &context, &[1.0, 1.0], &aitken_config(1.0e-6)).unwrap();
        let (terminal, applied) = report.trace.split_last().unwrap();
        assert_eq!(terminal.relaxation_factor, None);
        assert_eq!(applied[0].relaxation_factor, Some(1.0));
        assert!(
            applied
                .iter()
                .all(|entry| entry.relaxation_factor.is_some())
        );
        assert!(
            report
                .trace
                .iter()
                .enumerate()
                .all(|(index, entry)| entry.iteration == index)
        );
        let mut state = vec![1.0, 1.0];
        let mut image = vec![0.0; 2];
        for entry in applied {
            map.evaluate(&context, &state, &mut image).unwrap();
            let factor = entry.relaxation_factor.unwrap();
            for (value, g) in state.iter_mut().zip(&image) {
                *value += factor * (g - *value);
            }
        }
        assert_eq!(state, report.state);

        // Fixed relaxation records its constant factor on every applied
        // entry and nothing on the terminal one.
        let fixed = accelerate_fixed_point(
            &map,
            &context,
            &[1.0, 1.0],
            &AccelerationConfig {
                method: AccelerationMethod::FixedRelaxation { factor: 0.5 },
                max_iterations: 500,
                absolute_tolerance: 1.0e-6,
                relative_tolerance: 0.0,
            },
        )
        .unwrap();
        let (terminal, applied) = fixed.trace.split_last().unwrap();
        assert_eq!(terminal.relaxation_factor, None);
        assert!(
            applied
                .iter()
                .all(|entry| entry.relaxation_factor == Some(0.5))
        );
    }

    #[test]
    fn divergent_explicit_iteration_is_refused_as_predicted_with_its_trace() {
        // Full-step (factor = 1) iteration on a map with an eigenvalue
        // greater than one in magnitude diverges; it must never be
        // silently reported as converged or return a garbage iterate, and
        // the refusal carries the evidence: every evaluation's residual
        // norm, strictly growing, with no factor on the terminal entry.
        let map = LinearMap {
            matrix: vec![vec![1.5, 0.0], vec![0.0, 0.5]],
        };
        let config = AccelerationConfig {
            method: AccelerationMethod::FixedRelaxation { factor: 1.0 },
            max_iterations: 20,
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 0.0,
        };
        let error = accelerate_fixed_point(
            &map,
            &EvaluationContext::reproducible(),
            &[1.0, 1.0],
            &config,
        )
        .unwrap_err();
        let SolveError::AccelerationNotConverged { trace } = error else {
            panic!("expected a typed non-convergence refusal, got {error:?}");
        };
        assert_eq!(trace.len(), 21);
        assert_eq!(trace.last().unwrap().relaxation_factor, None);
        assert!(
            trace
                .windows(2)
                .all(|pair| pair[1].residual_norm > pair[0].residual_norm)
        );
    }

    #[test]
    fn aitken_refuses_a_degenerate_denominator() {
        // G(x) = x + 1: a constant residual of 1 every iteration, so Δr
        // between any two iterations is exactly zero.
        struct StallingMap;
        impl FixedPointOperator for StallingMap {
            fn dimension(&self) -> usize {
                1
            }

            fn evaluate(
                &self,
                _context: &EvaluationContext,
                state: &[f64],
                output: &mut [f64],
            ) -> Result<(), NumericError> {
                output[0] = state[0] + 1.0;
                Ok(())
            }
        }
        let config = AccelerationConfig {
            method: AccelerationMethod::Aitken {
                initial_factor: 1.0,
            },
            max_iterations: 10,
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 0.0,
        };
        let error = accelerate_fixed_point(
            &StallingMap,
            &EvaluationContext::reproducible(),
            &[0.0],
            &config,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            SolveError::AccelerationBreakdown { iteration: 1, .. }
        ));
    }

    #[test]
    fn aitken_refuses_a_near_stalling_sequence_at_any_scale() {
        // G(x) = x + scale + 1e-10·x: the residual changes by ~1e-10 of
        // itself per iteration, so ‖Δr‖² ≈ 1e-20·‖r‖² sits below the
        // relative floor at every scale. An absolute 1e-14 floor would have
        // admitted the scale-1e4 run (‖Δr‖² ≈ 1e-12) and produced a factor
        // of magnitude ~1e10.
        struct NearlyStallingMap {
            scale: f64,
        }
        impl FixedPointOperator for NearlyStallingMap {
            fn dimension(&self) -> usize {
                1
            }

            fn evaluate(
                &self,
                _context: &EvaluationContext,
                state: &[f64],
                output: &mut [f64],
            ) -> Result<(), NumericError> {
                output[0] = state[0] + self.scale + 1.0e-10 * state[0];
                Ok(())
            }
        }
        for scale in [1.0e-9, 1.0, 1.0e4] {
            let config = AccelerationConfig {
                method: AccelerationMethod::Aitken {
                    initial_factor: 1.0,
                },
                max_iterations: 10,
                absolute_tolerance: 1.0e-30,
                relative_tolerance: 0.0,
            };
            let error = accelerate_fixed_point(
                &NearlyStallingMap { scale },
                &EvaluationContext::reproducible(),
                &[0.0],
                &config,
            )
            .unwrap_err();
            assert!(
                matches!(
                    error,
                    SolveError::AccelerationBreakdown { iteration: 1, .. }
                ),
                "scale {scale}: {error:?}"
            );
        }
    }

    #[test]
    fn refuses_a_nonfinite_evaluation() {
        struct NonFiniteMap;
        impl FixedPointOperator for NonFiniteMap {
            fn dimension(&self) -> usize {
                1
            }

            fn evaluate(
                &self,
                _context: &EvaluationContext,
                _state: &[f64],
                output: &mut [f64],
            ) -> Result<(), NumericError> {
                output[0] = f64::NAN;
                Ok(())
            }
        }
        let config = AccelerationConfig {
            method: AccelerationMethod::FixedRelaxation { factor: 1.0 },
            max_iterations: 10,
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 0.0,
        };
        let error = accelerate_fixed_point(
            &NonFiniteMap,
            &EvaluationContext::reproducible(),
            &[0.0],
            &config,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            SolveError::Numeric(NumericError::NonFinite { .. })
        ));
    }
}
