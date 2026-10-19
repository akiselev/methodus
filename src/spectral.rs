//! Spectral-radius bounds of a small dense matrix, for `SpectralBound`
//! obligations and explicit stable-step hints.

use serde::{Deserialize, Serialize};

use crate::NumericError;

/// Spectral-radius evidence for a dense row-major `n × n` matrix.
///
/// Rigour, stated exactly:
/// - `gershgorin_upper_bound` is a rigorous upper bound on `ρ(A)`, rounding
///   included: `min(max row |·|-sum, max column |·|-sum)` computed in
///   floating point and inflated by `1 + 2(n+1)ε`, which dominates the
///   relative rounding of an `n`-term nonnegative sum for any `n·ε < 0.01`.
/// - `rayleigh_lower_bound` (exactly symmetric matrices only) is a rigorous
///   lower bound on `ρ(A)`: the largest `|xᵀAx| / xᵀx` over the power
///   iterates, deflated by a rounding bound on the dot products
///   (`2(n+1)ε·|x|ᵀ|A||x|` on the numerator, `1 + (n+1)ε` on the
///   denominator). For a symmetric matrix the Rayleigh quotient never
///   exceeds `ρ(A)`.
/// - `power_estimate` is a heuristic, with no bound in either direction:
///   for a symmetric matrix the last `|Rayleigh quotient|`; otherwise the
///   growth rate `(‖A^k x‖ / ‖A^{k/2} x‖)^{2/k}` over the second half of the
///   iterations, which tends to `ρ(A)` for a start vector with a component
///   along the dominant eigenspace, also for a complex dominant pair.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpectralRadiusReport {
    pub dimension: usize,
    pub symmetric: bool,
    pub gershgorin_upper_bound: f64,
    pub rayleigh_lower_bound: Option<f64>,
    pub power_estimate: f64,
    pub iterations: usize,
}

/// Bound and estimate the spectral radius of a dense row-major matrix with
/// `iterations` power iterations from the deterministic start vector
/// `x_i = 1 + frac(i·0.6180339887498949)`.
pub fn estimate_spectral_radius(
    matrix: &[f64],
    dimension: usize,
    iterations: usize,
) -> Result<SpectralRadiusReport, NumericError> {
    if dimension == 0 || iterations == 0 {
        return Err(NumericError::InvalidInput {
            message: "spectral radius needs a positive dimension and iteration count".into(),
        });
    }
    let entries = dimension
        .checked_mul(dimension)
        .ok_or_else(|| NumericError::InvalidInput {
            message: "spectral radius dimension overflows".into(),
        })?;
    NumericError::require_len("spectral radius matrix", matrix.len(), entries)?;
    NumericError::require_finite("spectral radius matrix", matrix)?;
    let n = dimension;
    let entry = |i: usize, j: usize| matrix[i * n + j];
    let symmetric = (0..n).all(|i| (0..i).all(|j| entry(i, j) == entry(j, i)));

    let max_row = (0..n)
        .map(|i| (0..n).map(|j| entry(i, j).abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let max_column = (0..n)
        .map(|j| (0..n).map(|i| entry(i, j).abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let slack = (n as f64 + 1.0) * f64::EPSILON;
    let gershgorin_upper_bound = max_row.min(max_column) * (1.0 + 2.0 * slack);

    let golden = 0.618_033_988_749_894_9_f64;
    let mut x: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64 * golden).fract()).collect();
    normalize(&mut x);
    let mut y = vec![0.0; n];
    let mut log_growth = Vec::with_capacity(iterations);
    let mut rayleigh_lower_bound: Option<f64> = None;
    let mut last_rayleigh = 0.0;
    for _ in 0..iterations {
        for (i, out) in y.iter_mut().enumerate() {
            *out = (0..n).map(|j| entry(i, j) * x[j]).sum();
        }
        if symmetric {
            let numerator: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
            let denominator: f64 = x.iter().map(|a| a * a).sum();
            let magnitude: f64 = (0..n)
                .map(|i| x[i].abs() * (0..n).map(|j| entry(i, j).abs() * x[j].abs()).sum::<f64>())
                .sum();
            last_rayleigh = numerator.abs() / denominator;
            let bound = ((numerator.abs() - 2.0 * slack * magnitude).max(0.0)
                / (denominator * (1.0 + slack)))
                * (1.0 - 4.0 * f64::EPSILON);
            rayleigh_lower_bound = Some(rayleigh_lower_bound.map_or(bound, |best| best.max(bound)));
        }
        let norm = y.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm == 0.0 {
            log_growth.push(f64::NEG_INFINITY);
            break;
        }
        log_growth.push(norm.ln());
        for (target, value) in x.iter_mut().zip(&y) {
            *target = value / norm;
        }
    }
    let power_estimate = if symmetric {
        last_rayleigh
    } else if log_growth.last() == Some(&f64::NEG_INFINITY) {
        0.0
    } else {
        let half = log_growth.len() / 2;
        let tail = &log_growth[half..];
        (tail.iter().sum::<f64>() / tail.len() as f64).exp()
    };
    Ok(SpectralRadiusReport {
        dimension: n,
        symmetric,
        gershgorin_upper_bound,
        rayleigh_lower_bound,
        power_estimate,
        iterations: log_growth.len(),
    })
}

fn normalize(values: &mut [f64]) {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    for value in values {
        *value /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_tridiagonal_bounds_bracket_the_exact_radius() {
        let n = 12;
        let mut matrix = vec![0.0; n * n];
        for i in 0..n {
            matrix[i * n + i] = 2.0;
            if i + 1 < n {
                matrix[i * n + i + 1] = -1.0;
                matrix[(i + 1) * n + i] = -1.0;
            }
        }
        let exact = 2.0 - 2.0 * (n as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
        let report = estimate_spectral_radius(&matrix, n, 2000).unwrap();
        assert!(report.symmetric);
        let lower = report.rayleigh_lower_bound.unwrap();
        assert!(lower <= exact && exact <= report.gershgorin_upper_bound);
        assert!(report.gershgorin_upper_bound <= 4.0 * (1.0 + 1e-14));
        assert!((report.power_estimate - exact).abs() <= 1e-6 * exact);
        assert!(exact - lower <= 1e-6 * exact);
    }

    #[test]
    fn upwind_circulant_estimate_tends_to_two_below_the_rigorous_bound() {
        // Periodic first-order upwind difference u_i - u_{i-1}: eigenvalues
        // 1 - e^{-iθ}, radius exactly 2 for an even size (θ = π, simple).
        let n = 16;
        let mut matrix = vec![0.0; n * n];
        for i in 0..n {
            matrix[i * n + i] = 1.0;
            matrix[i * n + (i + n - 1) % n] = -1.0;
        }
        let report = estimate_spectral_radius(&matrix, n, 4000).unwrap();
        assert!(!report.symmetric);
        assert_eq!(report.rayleigh_lower_bound, None);
        assert!(report.gershgorin_upper_bound >= 2.0);
        assert!(report.gershgorin_upper_bound <= 2.0 * (1.0 + 1e-14));
        assert!(
            (report.power_estimate - 2.0).abs() <= 1e-2,
            "{}",
            report.power_estimate
        );
    }

    #[test]
    fn rotation_with_a_complex_dominant_pair_and_a_nilpotent_matrix() {
        // 0.5 * rotation by 1 radian: eigenvalues 0.5 e^{±i}, radius 0.5.
        let (s, c) = 1.0_f64.sin_cos();
        let rotation = [0.5 * c, -0.5 * s, 0.5 * s, 0.5 * c];
        let report = estimate_spectral_radius(&rotation, 2, 50).unwrap();
        assert!((report.power_estimate - 0.5).abs() <= 1e-12);
        assert!(report.gershgorin_upper_bound >= 0.5);
        let nilpotent = [0.0, 1.0, 0.0, 0.0];
        let report = estimate_spectral_radius(&nilpotent, 2, 50).unwrap();
        assert_eq!(report.power_estimate, 0.0);
        assert!(report.iterations < 50);
    }

    #[test]
    fn malformed_inputs_are_refused() {
        assert!(estimate_spectral_radius(&[], 0, 10).is_err());
        assert!(estimate_spectral_radius(&[1.0], 1, 0).is_err());
        assert!(matches!(
            estimate_spectral_radius(&[1.0, 2.0], 2, 10),
            Err(NumericError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            estimate_spectral_radius(&[1.0, f64::NAN, 0.0, 1.0], 2, 10),
            Err(NumericError::NonFinite { index: 1, .. })
        ));
    }
}
