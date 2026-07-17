//! Integrator method selection, tolerances, and the PI step-size controller config.

/// The time-integration method used by [`integrate_dae`](super::integrate_dae).
///
/// All three form the per-step nonlinear system from the
/// [`DaeResidual`](numeric_contracts::DaeResidual) seam and solve it by reusing
/// solverang's globalized Newton (see the [module docs](super)). They differ in the
/// time-discretization of the reactive charge `d/dt q(x,t)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// **Implicit Euler (BDF-1).** First-order, L-stable, the index-1 first-order-DAE
    /// bootstrap. Per-step residual `(q^{n+1} − q^n)/h + g^{n+1} = 0`.
    ImplicitEuler,
    /// **Variable-step BDF-2.** Second-order, the index-1 first-order-DAE workhorse.
    /// Starts with one implicit-Euler step (no two-point history yet) then switches
    /// to the three-point variable-step BDF-2 stencil.
    Bdf2,
    /// **Generalized-α** (Jansen–Whiting–Hulbert first-order form). Second-order with
    /// tunable high-frequency damping via `ρ∞ ∈ [0, 1]` — the shape `cardan` uses for
    /// the index-3 mechanical path. `rho_inf = 1.0` is non-dissipative (trapezoidal),
    /// `rho_inf = 0.0` is asymptotically annihilating.
    GeneralizedAlpha {
        /// The spectral radius at infinite frequency, `ρ∞ ∈ [0, 1]`.
        rho_inf: f64,
    },
}

impl Method {
    /// The convergence order used to size the PI-controller exponent (error `~ h^{p+1}`).
    #[must_use]
    pub fn order(&self) -> usize {
        match self {
            Method::ImplicitEuler => 1,
            Method::Bdf2 | Method::GeneralizedAlpha { .. } => 2,
        }
    }
}

/// The PI (predictive / proportional-integral) step-size controller gains.
///
/// After a step with weighted local-error norm `err` (target `1`), the step size is
/// scaled by `safety · err^{−α} · err_prev^{β}` clamped to `[facmin, facmax]`, where
/// `α = i_gain / (p+1)` and `β = p_gain / (p+1)` for a method of order `p`. This is
/// the standard Gustafsson PI controller (Hairer–Wanner II.4): the `β` term damps the
/// oscillatory step-size sequence a pure integral (elementary) controller produces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PiControllerConfig {
    /// Multiplicative safety factor (`< 1`, typically `0.9`) — bias toward acceptance.
    pub safety: f64,
    /// Smallest allowed step-size shrink factor per step.
    pub facmin: f64,
    /// Largest allowed step-size growth factor per step.
    pub facmax: f64,
    /// Integral (`I`) gain numerator (`0.7` in the classic PI controller).
    pub i_gain: f64,
    /// Proportional (`P`) gain numerator (`0.4` in the classic PI controller).
    pub p_gain: f64,
}

impl Default for PiControllerConfig {
    fn default() -> Self {
        PiControllerConfig {
            safety: 0.9,
            facmin: 0.2,
            facmax: 5.0,
            i_gain: 0.7,
            p_gain: 0.4,
        }
    }
}

/// Options controlling an [`integrate_dae`](super::integrate_dae) run.
///
/// Construct with [`IntegratorOptions::fixed`] (constant step, no error control) or
/// [`IntegratorOptions::adaptive`] (PI step-size control), then tweak fields.
#[derive(Clone, Debug)]
pub struct IntegratorOptions {
    /// Which integrator to run.
    pub method: Method,
    /// Initial step (adaptive) or the fixed step (non-adaptive). Must be `> 0`.
    pub h0: f64,
    /// Whether the PI controller adapts the step; `false` = constant step.
    pub adaptive: bool,
    /// Relative local-error tolerance.
    pub rtol: f64,
    /// Absolute local-error tolerance (floors the weighting for near-zero components).
    pub atol: f64,
    /// Smallest step the controller may take before declaring
    /// [`StepSizeUnderflow`](super::IntegrateError::StepSizeUnderflow).
    pub min_step: f64,
    /// Largest step the controller may grow to.
    pub max_step: f64,
    /// Step budget — an upper bound on accepted steps before
    /// [`StepLimitReached`](super::IntegrateError::StepLimitReached).
    pub max_steps: usize,
    /// Maximum Newton iterations per step attempt.
    pub max_newton_iters: usize,
    /// **Absolute** floor of the per-step Newton convergence test (residual norm).
    ///
    /// The per-step Newton is judged converged when `‖G(x)‖ ≤ newton_tol +
    /// newton_rtol · scale`, where `scale` is the magnitude of the stage residual's
    /// constituent terms (the SUNDIALS-style mixed absolute+relative form). `newton_tol`
    /// is the absolute part — it dominates for well-scaled problems (where `scale` is
    /// `O(‖G‖)`), keeping the classic tight behaviour; `newton_rtol · scale` dominates
    /// for high-magnitude / high-conductivity states, where it sits safely above the
    /// per-step cancellation floor `‖a₀·M + K‖·‖x‖·ε` that an absolute-only tolerance
    /// spuriously fails against (collapsing the step to `min_step`).
    pub newton_tol: f64,
    /// **Relative** part of the per-step Newton convergence test — see [`newton_tol`].
    /// Scales with the problem magnitude so the test never falls below the residual
    /// cancellation floor. `0.0` recovers a pure absolute tolerance.
    ///
    /// [`newton_tol`]: IntegratorOptions::newton_tol
    pub newton_rtol: f64,
    /// Exclude structurally algebraic components (zero mass rows) from the local-error
    /// test — the *suppress-alg* remedy (SUNDIALS IDA). On by default; without it the
    /// O(h) predictor error on an index-1 DAE's algebraic rows collapses the step size.
    pub suppress_algebraic: bool,
    /// PI-controller gains (ignored when `adaptive == false`).
    pub controller: PiControllerConfig,
}

impl IntegratorOptions {
    /// A **constant-step** configuration: no error control, one step of size `h`
    /// until the span end (the last step is shortened to land exactly on it). Newton
    /// non-convergence is a hard failure (the step is not reduced). This is the mode
    /// the order-of-convergence tests use.
    #[must_use]
    pub fn fixed(method: Method, h: f64) -> Self {
        IntegratorOptions {
            method,
            h0: h,
            adaptive: false,
            rtol: 1e-6,
            atol: 1e-9,
            min_step: 1e-14,
            max_step: f64::INFINITY,
            max_steps: 10_000_000,
            max_newton_iters: 50,
            newton_tol: 1e-11,
            // ~7 orders above machine epsilon: `newton_rtol · scale` sits well above the
            // residual cancellation floor (`~scale · ε`) for any state magnitude, while
            // still resolving the stage residual to ~9 relative digits.
            newton_rtol: 1e-9,
            suppress_algebraic: true,
            controller: PiControllerConfig::default(),
        }
    }

    /// An **adaptive** configuration: the PI controller adapts the step to keep the
    /// weighted local-error norm near `1`, rejecting and retrying overshooting steps.
    /// `h0` is the initial guess.
    #[must_use]
    pub fn adaptive(method: Method, h0: f64) -> Self {
        IntegratorOptions {
            adaptive: true,
            ..Self::fixed(method, h0)
        }
    }

    /// Set relative and absolute tolerances (builder-style).
    #[must_use]
    pub fn with_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.rtol = rtol;
        self.atol = atol;
        self
    }
}
