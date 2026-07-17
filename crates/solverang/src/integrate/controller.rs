//! The PI (predictive) step-size controller.
//!
//! Implements the standard proportional-integral step-size controller
//! (Gustafsson 1988; Hairer–Wanner *Solving ODEs I* §II.4). Given a weighted
//! local-error norm `err` (target `1`), the controller proposes a new step size
//! `h_new = h · f` where
//!
//! ```text
//! f = clamp( safety · err^{−α} · err_prev^{β},  facmin, facmax )
//! α = i_gain / (p + 1),   β = p_gain / (p + 1)
//! ```
//!
//! for a method of order `p`. The `β` (proportional) term uses the *previous*
//! step's error to damp the oscillatory step sequence that a bare integral
//! controller (`f = safety · err^{−α}`) produces on mildly stiff problems.

use super::options::PiControllerConfig;

/// Stateful PI step-size controller. One instance per integration run; it remembers
/// the previous accepted error to feed the proportional term.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PiController {
    cfg: PiControllerConfig,
    /// `α = i_gain / (p+1)` — the integral exponent.
    alpha: f64,
    /// `β = p_gain / (p+1)` — the proportional exponent.
    beta: f64,
    /// The last *accepted* error norm (starts neutral at `1`, so the first step is a
    /// pure integral update).
    err_prev: f64,
}

impl PiController {
    /// Build a controller for a method of order `order`.
    pub(crate) fn new(cfg: PiControllerConfig, order: usize) -> Self {
        let k = (order + 1) as f64;
        PiController {
            cfg,
            alpha: cfg.i_gain / k,
            beta: cfg.p_gain / k,
            err_prev: 1.0,
        }
    }

    /// The growth/shrink factor for an **accepted** step with error `err`. Also
    /// records `err` as the previous error for the next step's proportional term.
    pub(crate) fn accept_factor(&mut self, err: f64) -> f64 {
        let f = self.raw_factor(err);
        self.err_prev = err.max(1e-10);
        f.clamp(self.cfg.facmin, self.cfg.facmax)
    }

    /// The shrink factor for a **rejected** step with error `err`. Rejections may only
    /// shrink (`≤ 1`) and do not update the proportional-term history.
    pub(crate) fn reject_factor(&self, err: f64) -> f64 {
        // Pure integral shrink on reject — the proportional term can point the wrong
        // way right after an overshoot.
        let f = self.cfg.safety * err.powf(-self.alpha);
        f.clamp(self.cfg.facmin, 1.0)
    }

    fn raw_factor(&self, err: f64) -> f64 {
        if err <= 1e-10 {
            // Essentially zero error — grow by the maximum.
            return self.cfg.facmax;
        }
        self.cfg.safety * err.powf(-self.alpha) * self.err_prev.powf(self.beta)
    }
}

/// The weighted RMS local-error norm used by the controller: `err ≤ 1` means the
/// estimated local error is within the mixed absolute/relative tolerance band.
///
/// ```text
/// err = sqrt( (1/N) Σ_i ( le_i / (atol + rtol·max(|x_old_i|, |x_new_i|)) )² )
/// ```
///
/// `suppress[i] == true` excludes component `i` from the norm (the *suppress-alg*
/// remedy for structurally algebraic components — see
/// [`algebraic_mask`](super::step::algebraic_mask)); if every component is suppressed
/// the norm falls back to including all of them.
pub(crate) fn wrms_norm(
    le: &[f64],
    x_old: &[f64],
    x_new: &[f64],
    atol: f64,
    rtol: f64,
    suppress: &[bool],
) -> f64 {
    let n = le.len();
    if n == 0 {
        return 0.0;
    }
    let all_suppressed = suppress.len() == n && suppress.iter().all(|&b| b);
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        if !all_suppressed && suppress.get(i).copied().unwrap_or(false) {
            continue;
        }
        let scale = atol + rtol * x_old[i].abs().max(x_new[i].abs());
        if scale > 0.0 {
            let r = le[i] / scale;
            sum += r * r;
        }
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    (sum / count as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_error_grows_large_error_shrinks() {
        let mut c = PiController::new(PiControllerConfig::default(), 2);
        // Comfortably under tolerance -> grow.
        assert!(c.accept_factor(0.1) > 1.0);
        // Over tolerance on a rejected step -> shrink below 1.
        assert!(c.reject_factor(8.0) < 1.0);
    }

    #[test]
    fn wrms_zero_error_is_zero() {
        let e = [0.0, 0.0];
        let x = [1.0, 2.0];
        assert_eq!(wrms_norm(&e, &x, &x, 1e-8, 1e-6, &[false, false]), 0.0);
    }

    #[test]
    fn wrms_scales_with_tolerance() {
        let e = [1e-3];
        let x = [1.0];
        let tight = wrms_norm(&e, &x, &x, 1e-10, 1e-10, &[false]);
        let loose = wrms_norm(&e, &x, &x, 1e-2, 1e-2, &[false]);
        assert!(tight > loose);
    }

    #[test]
    fn wrms_suppresses_algebraic_component() {
        // A huge error on component 1 is ignored when it is suppressed.
        let e = [1e-9, 1e6];
        let x = [1.0, 1.0];
        let with = wrms_norm(&e, &x, &x, 1e-9, 1e-6, &[false, false]);
        let without = wrms_norm(&e, &x, &x, 1e-9, 1e-6, &[false, true]);
        assert!(with > 1.0);
        assert!(without < 1.0);
    }

    #[test]
    fn reject_factor_never_grows() {
        let c = PiController::new(PiControllerConfig::default(), 1);
        // Even a tiny error on reject must not propose growth.
        assert!(c.reject_factor(1.0) <= 1.0);
    }
}
