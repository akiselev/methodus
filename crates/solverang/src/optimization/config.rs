//! Configuration types for optimization solvers.
//!
//! [`OptimizationConfig`] holds the knobs shared by every algorithm
//! (iteration limits, tolerances, L-BFGS memory) plus per-concern sub-configs:
//! [`LineSearchConfig`], [`AlmConfig`], and [`TrustRegionConfig`]. Fields in a
//! sub-config only matter when the corresponding machinery runs.

/// Which optimization algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationAlgorithm {
    /// Automatically select based on problem structure.
    /// Unconstrained with bounds → BfgsB, unconstrained → BFGS, equality-constrained → ALM.
    #[default]
    Auto,
    /// L-BFGS for unconstrained optimization (gradient-only).
    Bfgs,
    /// L-BFGS-B for box-constrained optimization (projected gradient).
    BfgsB,
    /// Augmented Lagrangian Method for constrained optimization.
    /// Uses existing NR/LM as inner solver.
    Alm,
    /// Trust-region method with dogleg (n < threshold) or Steihaug-CG (n >= threshold).
    TrustRegion,
}

/// Strategy for initializing Lagrange multipliers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiplierInitStrategy {
    /// Initialize all multipliers to zero (simplest, always works).
    Zero,
    /// Warm-start from previous solve's multipliers.
    WarmStart,
}

/// Line-search parameters (used by BFGS, BFGS-B, and the ALM inner loop).
#[derive(Debug, Clone)]
pub struct LineSearchConfig {
    /// Armijo sufficient-decrease parameter c₁.
    pub armijo_c1: f64,
    /// Wolfe curvature condition parameter c₂ (strong curvature):
    /// `|∇f(x+αd)·d| ≤ c₂|∇f(x)·d|`.
    pub wolfe_c2: f64,
    /// Backtracking factor applied when a candidate step is rejected.
    pub backtrack: f64,
    /// Minimum step size before declaring failure.
    pub min_step: f64,
    /// Maximum combined function + gradient evaluations per line search.
    pub max_evals: usize,
}

impl Default for LineSearchConfig {
    fn default() -> Self {
        Self {
            armijo_c1: 1e-4,
            wolfe_c2: 0.9,
            backtrack: 0.5,
            min_step: 1e-12,
            max_evals: 100,
        }
    }
}

/// Augmented-Lagrangian parameters (used when equality or inequality
/// constraints are present).
#[derive(Debug, Clone)]
pub struct AlmConfig {
    /// Initial penalty parameter ρ.
    pub rho_init: f64,
    /// Penalty growth factor: `ρ_{k+1} = min(ρ_k * growth, ρ_max)`.
    pub rho_growth: f64,
    /// Maximum penalty parameter.
    pub rho_max: f64,
    /// Maximum absolute value for multipliers (divergence guard).
    pub max_multiplier: f64,
    /// Strategy for initializing multipliers.
    pub multiplier_init: MultiplierInitStrategy,
}

impl Default for AlmConfig {
    fn default() -> Self {
        Self {
            rho_init: 1.0,
            rho_growth: 10.0,
            rho_max: 1e6,
            max_multiplier: 1e8,
            multiplier_init: MultiplierInitStrategy::Zero,
        }
    }
}

/// Trust-region parameters (used by [`OptimizationAlgorithm::TrustRegion`]).
#[derive(Debug, Clone)]
pub struct TrustRegionConfig {
    /// Initial trust-region radius.
    pub initial_radius: f64,
    /// Maximum trust-region radius.
    pub max_radius: f64,
    /// Dimension threshold: dogleg for n < threshold, Steihaug-CG for n >= threshold.
    pub subproblem_threshold: usize,
}

impl Default for TrustRegionConfig {
    fn default() -> Self {
        Self {
            initial_radius: 1.0,
            max_radius: 100.0,
            subproblem_threshold: 100,
        }
    }
}

/// Configuration for optimization solvers.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Which algorithm to use.
    pub algorithm: OptimizationAlgorithm,
    /// Maximum outer iterations (ALM outer loop, or BFGS total iterations).
    pub max_outer_iterations: usize,
    /// Maximum inner iterations (ALM inner BFGS/BFGS-B solve).
    pub max_inner_iterations: usize,
    /// Outer tolerance: primal feasibility `||g(x)|| < tol`.
    pub outer_tolerance: f64,
    /// Inner tolerance: inner solver convergence criterion.
    pub inner_tolerance: f64,
    /// Dual feasibility tolerance: `||∇_x L|| < tol`.
    pub dual_tolerance: f64,
    /// L-BFGS memory size (number of past gradient pairs to store).
    pub lbfgs_memory: usize,
    /// Use relative tolerances for convergence checks.
    ///
    /// When `true`, BFGS scales the gradient norm by `max(1.0, |f|)` and ALM
    /// scales norms by the square root of the problem dimension, making
    /// convergence criteria independent of problem size and objective magnitude.
    /// When `false`, absolute tolerances are used (backward-compatible behavior).
    pub relative_tolerance: bool,
    /// Line-search parameters.
    pub line_search: LineSearchConfig,
    /// Augmented-Lagrangian parameters.
    pub alm: AlmConfig,
    /// Trust-region parameters.
    pub trust_region: TrustRegionConfig,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            algorithm: OptimizationAlgorithm::Auto,
            max_outer_iterations: 100,
            max_inner_iterations: 200,
            outer_tolerance: 1e-6,
            inner_tolerance: 1e-8,
            dual_tolerance: 1e-6,
            lbfgs_memory: 10,
            relative_tolerance: true,
            line_search: LineSearchConfig::default(),
            alm: AlmConfig::default(),
            trust_region: TrustRegionConfig::default(),
        }
    }
}

impl OptimizationConfig {
    /// Check the configuration for values that would make a solve meaningless
    /// (non-positive tolerances, non-finite penalties, out-of-range line-search
    /// constants). Returns a description of the first problem found.
    pub fn validate(&self) -> Result<(), String> {
        fn positive_finite(name: &str, v: f64) -> Result<(), String> {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(format!("{name} must be positive and finite, got {v}"))
            }
        }
        positive_finite("outer_tolerance", self.outer_tolerance)?;
        positive_finite("inner_tolerance", self.inner_tolerance)?;
        positive_finite("dual_tolerance", self.dual_tolerance)?;
        positive_finite("line_search.min_step", self.line_search.min_step)?;
        positive_finite("alm.rho_init", self.alm.rho_init)?;
        positive_finite(
            "trust_region.initial_radius",
            self.trust_region.initial_radius,
        )?;
        if self.max_outer_iterations == 0 {
            return Err("max_outer_iterations must be at least 1".to_string());
        }
        if !(0.0..1.0).contains(&self.line_search.armijo_c1) {
            return Err(format!(
                "line_search.armijo_c1 must be in (0, 1), got {}",
                self.line_search.armijo_c1
            ));
        }
        if !(self.line_search.armijo_c1..1.0).contains(&self.line_search.wolfe_c2) {
            return Err(format!(
                "line_search.wolfe_c2 must be in (armijo_c1, 1), got {}",
                self.line_search.wolfe_c2
            ));
        }
        if !(0.0..1.0).contains(&self.line_search.backtrack) {
            return Err(format!(
                "line_search.backtrack must be in (0, 1), got {}",
                self.line_search.backtrack
            ));
        }
        if self.alm.rho_growth <= 1.0 {
            return Err(format!(
                "alm.rho_growth must be > 1, got {}",
                self.alm.rho_growth
            ));
        }
        Ok(())
    }
}
