//! [`ConstraintSystem`] — the top-level coordinator for entity/constraint solving.
//!
//! This module provides the main entry point for building and solving constraint
//! systems. It manages entities, constraints, parameters, and the solve pipeline:
//!
//! 1. Entities are added (each owns parameters in the [`ParamStore`]).
//! 2. Constraints are added between entities.
//! 3. On [`solve()`](ConstraintSystem::solve), the system delegates to a
//!    [`SolvePipeline`] which decomposes into independent clusters, analyzes,
//!    reduces, and solves each one.
//! 4. Solutions are written back to the `ParamStore`.
//!
//! # Example
//!
//! ```
//! use solverang::ConstraintSystem;
//! use solverang::system::SystemStatus;
//!
//! let mut system = ConstraintSystem::new();
//!
//! // Allocate an entity ID first, then its parameters. A geometry layer
//! // (e.g. `sketch2d::Sketch2DBuilder`) normally does this for you and
//! // also adds the entity and constraints.
//! let entity = system.alloc_entity_id();
//! let px = system.alloc_param(3.0, entity);
//!
//! let result = system.solve();
//! assert!(matches!(result.status, SystemStatus::Solved));
//! assert_eq!(system.get_param(px), 3.0);
//! ```

use crate::constraint::Constraint;
use crate::dataflow::{ChangeTracker, SolutionCache};
use crate::entity::Entity;
use crate::id::{ConstraintId, EntityId, ParamId};
use crate::optimization::{
    InequalityFn, MultiplierStore, Objective, OptimizationConfig, OptimizationResult,
    OptimizationStatus,
};
use crate::param::ParamStore;
use crate::pipeline::SolvePipeline;
use crate::time::{SolveClock, StdClock};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the constraint system and its solver pipeline.
#[derive(Clone, Debug)]
pub struct SystemConfig {
    /// Configuration for the Levenberg-Marquardt solver.
    pub lm_config: crate::solver::LMConfig,
    /// Configuration for the Newton-Raphson solver (used by AutoSolver).
    pub solver_config: crate::solver::SolverConfig,
    /// Maximum permitted final residual norm before a system can report solved.
    ///
    /// This is a final certification pass over every live constraint after the
    /// pipeline writes its solution back into the parameter store. It prevents
    /// cached, skipped, or incorrectly decomposed clusters from masking an
    /// unsatisfied constraint.
    pub final_residual_tolerance: f64,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            lm_config: crate::solver::LMConfig::default(),
            solver_config: crate::solver::SolverConfig::default(),
            final_residual_tolerance: 1e-8,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Overall result of solving the entire constraint system.
pub struct SystemResult {
    /// High-level status of the solve.
    pub status: SystemStatus,
    /// Per-cluster results (one entry per independent cluster).
    pub clusters: Vec<ClusterResult>,
    /// Total solver iterations summed across all clusters.
    pub iterations: usize,
    /// Duration reported by the solve clock.
    pub duration: std::time::Duration,
}

/// High-level status of the entire system solve.
#[derive(Debug)]
pub enum SystemStatus {
    /// All clusters converged.
    Solved,
    /// Some clusters converged but at least one did not.
    PartiallySolved,
    /// Structural issues detected before or after solving.
    DiagnosticFailure(Vec<DiagnosticIssue>),
}

/// Result of solving a single cluster.
pub struct ClusterResult {
    /// Which cluster this result belongs to.
    pub cluster_id: crate::id::ClusterId,
    /// Solve status for this cluster.
    pub status: ClusterSolveStatus,
    /// Number of solver iterations for this cluster.
    pub iterations: usize,
    /// Final residual norm for this cluster.
    pub residual_norm: f64,
}

/// Solve status for a single cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterSolveStatus {
    /// The solver converged within tolerance.
    Converged,
    /// The solver ran but did not converge.
    NotConverged,
    /// The cluster was skipped (e.g., no free variables).
    Skipped,
}

/// Why an entity or constraint removal was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemovalError {
    /// The ID is unknown to this system or its generation is stale
    /// (the slot was reused after an earlier removal).
    StaleId,
    /// Live constraints still reference the entity; remove them first.
    HasDependentConstraints {
        /// The constraints that reference the entity.
        constraints: Vec<ConstraintId>,
    },
    /// Another live entity shares one of the entity's parameters (e.g. a
    /// line segment sharing a point's coordinates); remove it first.
    SharedParams {
        /// The entities sharing parameters.
        entities: Vec<EntityId>,
    },
}

impl std::fmt::Display for RemovalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaleId => write!(f, "stale or unknown ID"),
            Self::HasDependentConstraints { constraints } => write!(
                f,
                "{} constraint(s) still reference the entity",
                constraints.len()
            ),
            Self::SharedParams { entities } => write!(
                f,
                "{} other entity(ies) share the entity's parameters",
                entities.len()
            ),
        }
    }
}

impl std::error::Error for RemovalError {}

/// A diagnostic issue detected in the constraint system.
#[derive(Debug, Clone)]
pub enum DiagnosticIssue {
    /// A constraint is redundant (implied by others).
    RedundantConstraint {
        /// The redundant constraint.
        constraint: ConstraintId,
        /// Constraints that already imply it (empty when unknown).
        implied_by: Vec<ConstraintId>,
    },
    /// Two or more constraints conflict (cannot be simultaneously satisfied).
    ConflictingConstraints {
        /// The mutually conflicting constraints.
        constraints: Vec<ConstraintId>,
    },
    /// An entity has unconstrained directions.
    UnderConstrained {
        /// The under-constrained entity.
        entity: EntityId,
        /// Number of unconstrained directions (degrees of freedom).
        free_directions: usize,
    },
    /// One or more constraints still have residuals above certification
    /// tolerance after the pipeline wrote its final parameter values.
    UnsatisfiedConstraints {
        /// Constraints whose residuals exceeded the final tolerance.
        constraints: Vec<ConstraintId>,
        /// Largest absolute residual observed across all constraints.
        max_residual: f64,
        /// Certification tolerance used for the check.
        tolerance: f64,
    },
}

// ---------------------------------------------------------------------------
// Objective model
// ---------------------------------------------------------------------------

/// One coherent objective installed on the system.
///
/// Either a first-order objective (value + gradient) or a second-order one
/// that additionally exposes an exact Hessian. Replacing the objective
/// replaces the whole model, so a stale Hessian can never survive a
/// first-order replacement.
enum ObjectiveModel {
    FirstOrder(Box<dyn Objective>),
    SecondOrder(Box<dyn crate::optimization::ObjectiveHessian>),
}

impl ObjectiveModel {
    fn objective(&self) -> &dyn Objective {
        match self {
            Self::FirstOrder(o) => o.as_ref(),
            Self::SecondOrder(o) => o.as_ref(),
        }
    }

    fn hessian(&self) -> Option<&dyn crate::optimization::ObjectiveHessian> {
        match self {
            Self::FirstOrder(_) => None,
            Self::SecondOrder(o) => Some(o.as_ref()),
        }
    }
}

// ---------------------------------------------------------------------------
// ConstraintSystem
// ---------------------------------------------------------------------------

/// The top-level constraint system coordinator.
///
/// Manages entities, constraints, and parameters. Provides a `solve()` method
/// that delegates to a [`SolvePipeline`] which decomposes the system into
/// independent clusters, analyzes, reduces, and solves each one.
pub struct ConstraintSystem {
    params: ParamStore,
    entities: Vec<Option<Box<dyn Entity>>>,
    constraints: Vec<Option<Box<dyn Constraint>>>,
    config: SystemConfig,
    pipeline: SolvePipeline,
    change_tracker: ChangeTracker,
    solution_cache: SolutionCache,
    /// Per-slot generation counters for entity IDs.
    entity_generations: Vec<u32>,
    /// Free list of reusable entity slots.
    entity_free_list: Vec<u32>,
    /// Per-slot generation counters for constraint IDs.
    constraint_generations: Vec<u32>,
    /// Free list of reusable constraint slots.
    constraint_free_list: Vec<u32>,
    // --- Optimization extension ---
    /// Objective to minimize (None = constraint-satisfaction only).
    objective: Option<ObjectiveModel>,
    /// Inequality constraints h(x) ≤ 0.
    inequalities: Vec<Option<Box<dyn InequalityFn>>>,
    /// Configuration for optimization solvers.
    opt_config: OptimizationConfig,
    /// Multipliers from the last optimization solve.
    last_multipliers: MultiplierStore,
    /// Structural fingerprint of the problem the last multipliers belong to.
    /// Warm-start multipliers are only reused when the fingerprint matches.
    last_problem_fingerprint: Option<u64>,
}

impl Default for ConstraintSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintSystem {
    /// Create a new, empty constraint system with default configuration.
    pub fn new() -> Self {
        Self {
            params: ParamStore::new(),
            entities: Vec::new(),
            constraints: Vec::new(),
            config: SystemConfig::default(),
            pipeline: SolvePipeline::default(),
            change_tracker: ChangeTracker::new(),
            solution_cache: SolutionCache::new(),
            entity_generations: Vec::new(),
            entity_free_list: Vec::new(),
            constraint_generations: Vec::new(),
            constraint_free_list: Vec::new(),
            objective: None,
            inequalities: Vec::new(),
            opt_config: OptimizationConfig::default(),
            last_multipliers: MultiplierStore::new(),
            last_problem_fingerprint: None,
        }
    }

    /// Create a new constraint system with the given configuration.
    pub fn with_config(config: SystemConfig) -> Self {
        let mut s = Self::new();
        s.config = config;
        s
    }

    /// Returns the constraint-system solve configuration.
    pub fn config(&self) -> &SystemConfig {
        &self.config
    }

    /// Returns mutable access to the constraint-system solve configuration.
    pub fn config_mut(&mut self) -> &mut SystemConfig {
        &mut self.config
    }

    // -------------------------------------------------------------------
    // Parameter access
    // -------------------------------------------------------------------

    /// Allocate a new parameter with the given initial value, owned by `owner`.
    ///
    /// This is the primary way entities obtain `ParamId`s before being added.
    pub fn alloc_param(&mut self, value: f64, owner: EntityId) -> ParamId {
        self.params.alloc(value, owner)
    }

    /// Shared reference to the parameter store.
    pub fn params(&self) -> &ParamStore {
        &self.params
    }

    /// Mutable reference to the parameter store.
    pub fn params_mut(&mut self) -> &mut ParamStore {
        &mut self.params
    }

    /// Get the current value of a parameter.
    pub fn get_param(&self, id: ParamId) -> f64 {
        self.params.get(id)
    }

    /// Set the value of a parameter.
    pub fn set_param(&mut self, id: ParamId, value: f64) {
        self.params.set(id, value);
        self.change_tracker.mark_param_dirty(id);
    }

    /// Mark a parameter as fixed (excluded from solving).
    pub fn fix_param(&mut self, id: ParamId) {
        self.params.fix(id);
        self.change_tracker.mark_param_dirty(id);
        self.pipeline.invalidate();
    }

    /// Mark a parameter as free (included in solving).
    pub fn unfix_param(&mut self, id: ParamId) {
        self.params.unfix(id);
        self.change_tracker.mark_param_dirty(id);
        self.pipeline.invalidate();
    }

    // -------------------------------------------------------------------
    // Entity management
    // -------------------------------------------------------------------

    /// Add an entity to the system.
    ///
    /// The entity must already have its `EntityId` and `ParamId`s allocated
    /// (via [`alloc_entity_id`](Self::alloc_entity_id) and
    /// [`alloc_param`](Self::alloc_param)).
    ///
    /// Returns the entity's ID.
    ///
    /// # Panics
    ///
    /// Panics if the entity's ID was not allocated by this system's
    /// [`alloc_entity_id`](Self::alloc_entity_id) (unknown slot, stale
    /// generation, or a slot that is already occupied) — that is a
    /// programming error, not a recoverable condition.
    pub fn add_entity(&mut self, entity: Box<dyn Entity>) -> EntityId {
        let id = entity.id();
        let idx = id.raw_index() as usize;

        assert!(
            idx < self.entities.len() && self.entity_generations[idx] == id.generation,
            "add_entity: {id:?} was not allocated by this system's alloc_entity_id"
        );
        assert!(
            self.entities[idx].is_none(),
            "add_entity: slot for {id:?} is already occupied"
        );
        self.entities[idx] = Some(entity);
        self.change_tracker.mark_entity_added(id);
        id
    }

    /// Allocate a new [`EntityId`] for constructing an entity.
    ///
    /// Call this first, then use the returned ID to allocate parameters
    /// via [`alloc_param`](Self::alloc_param), build the entity, and finally
    /// call [`add_entity`](Self::add_entity).
    pub fn alloc_entity_id(&mut self) -> EntityId {
        if let Some(index) = self.entity_free_list.pop() {
            let gen = self.entity_generations[index as usize] + 1;
            self.entity_generations[index as usize] = gen;
            // Clear the slot for reuse
            self.entities[index as usize] = None;
            EntityId::new(index, gen)
        } else {
            let index = self.entities.len() as u32;
            self.entities.push(None);
            self.entity_generations.push(0);
            EntityId::new(index, 0)
        }
    }

    /// Remove an entity and free its parameters.
    ///
    /// Removal is refused (with a descriptive error) instead of corrupting
    /// the system when:
    ///
    /// - the ID is stale or unknown ([`RemovalError::StaleId`]),
    /// - live constraints still reference the entity
    ///   ([`RemovalError::HasDependentConstraints`]) — remove them first,
    /// - another live entity shares one of its parameters
    ///   ([`RemovalError::SharedParams`]) — e.g. a line segment sharing an
    ///   endpoint's coordinates. Remove the sharing entity first.
    pub fn remove_entity(&mut self, id: EntityId) -> Result<(), RemovalError> {
        let idx = id.raw_index() as usize;
        let live = idx < self.entities.len()
            && idx < self.entity_generations.len()
            && self.entity_generations[idx] == id.generation
            && self.entities[idx].is_some();
        if !live {
            return Err(RemovalError::StaleId);
        }

        // Refuse while constraints still reference the entity.
        let dependents: Vec<ConstraintId> = self
            .constraints
            .iter()
            .filter_map(|c| c.as_deref())
            .filter(|c| c.entity_ids().contains(&id))
            .map(|c| c.id())
            .collect();
        if !dependents.is_empty() {
            return Err(RemovalError::HasDependentConstraints {
                constraints: dependents,
            });
        }

        // Refuse while another live entity shares one of its parameters
        // (freeing them would leave that entity with dangling ParamIds).
        let params = self.entities[idx].as_ref().expect("checked live").params();
        let sharing: Vec<EntityId> = self
            .entities
            .iter()
            .filter_map(|e| e.as_deref())
            .filter(|e| e.id() != id && e.params().iter().any(|p| params.contains(p)))
            .map(|e| e.id())
            .collect();
        if !sharing.is_empty() {
            return Err(RemovalError::SharedParams { entities: sharing });
        }

        let entity = self.entities[idx].take().expect("checked live");
        for &pid in entity.params() {
            self.params.free(pid);
        }
        self.entity_free_list.push(idx as u32);
        self.change_tracker.mark_entity_removed(id);
        self.pipeline.invalidate();
        Ok(())
    }

    // -------------------------------------------------------------------
    // Constraint management
    // -------------------------------------------------------------------

    /// Allocate a new [`ConstraintId`] for constructing a constraint.
    pub fn alloc_constraint_id(&mut self) -> ConstraintId {
        if let Some(index) = self.constraint_free_list.pop() {
            let gen = self.constraint_generations[index as usize] + 1;
            self.constraint_generations[index as usize] = gen;
            self.constraints[index as usize] = None;
            ConstraintId::new(index, gen)
        } else {
            let index = self.constraints.len() as u32;
            self.constraints.push(None);
            self.constraint_generations.push(0);
            ConstraintId::new(index, 0)
        }
    }

    /// Add a constraint to the system.
    ///
    /// The constraint must already have its `ConstraintId` set (via
    /// [`alloc_constraint_id`](Self::alloc_constraint_id)).
    ///
    /// Returns the constraint's ID.
    ///
    /// # Panics
    ///
    /// Panics if the constraint's ID was not allocated by this system's
    /// [`alloc_constraint_id`](Self::alloc_constraint_id) (unknown slot,
    /// stale generation, or an occupied slot) — that is a programming error,
    /// not a recoverable condition.
    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint>) -> ConstraintId {
        let id = constraint.id();
        let idx = id.raw_index() as usize;

        assert!(
            idx < self.constraints.len() && self.constraint_generations[idx] == id.generation,
            "add_constraint: {id:?} was not allocated by this system's alloc_constraint_id"
        );
        assert!(
            self.constraints[idx].is_none(),
            "add_constraint: slot for {id:?} is already occupied"
        );
        self.constraints[idx] = Some(constraint);
        self.change_tracker.mark_constraint_added(id);
        self.pipeline.invalidate();
        id
    }

    /// Remove a constraint from the system.
    ///
    /// Returns [`RemovalError::StaleId`] when the ID is stale or unknown
    /// instead of silently doing nothing.
    pub fn remove_constraint(&mut self, id: ConstraintId) -> Result<(), RemovalError> {
        let idx = id.raw_index() as usize;
        if idx < self.constraints.len()
            && idx < self.constraint_generations.len()
            && self.constraint_generations[idx] == id.generation
            && self.constraints[idx].is_some()
        {
            self.constraints[idx] = None;
            self.constraint_free_list.push(idx as u32);
            self.change_tracker.mark_constraint_removed(id);
            self.pipeline.invalidate();
            Ok(())
        } else {
            Err(RemovalError::StaleId)
        }
    }

    /// Debug-mode referential integrity sweep, run before every solve.
    ///
    /// Panics when a live constraint references a dead parameter or a dead
    /// entity, or a live entity owns a dead parameter — all states that the
    /// removal guards should make unreachable.
    #[cfg(debug_assertions)]
    fn debug_validate_integrity(&self) {
        for entity in self.entities.iter().filter_map(|e| e.as_deref()) {
            for &pid in entity.params() {
                debug_assert!(
                    self.params.is_alive(pid),
                    "integrity: entity {:?} owns dead param {pid:?}",
                    entity.id()
                );
            }
        }
        for constraint in self.constraints.iter().filter_map(|c| c.as_deref()) {
            for &pid in constraint.param_ids() {
                debug_assert!(
                    self.params.is_alive(pid),
                    "integrity: constraint {:?} references dead param {pid:?}",
                    constraint.id()
                );
            }
            for &eid in constraint.entity_ids() {
                let idx = eid.raw_index() as usize;
                let live = idx < self.entities.len()
                    && self.entity_generations[idx] == eid.generation
                    && self.entities[idx].is_some();
                debug_assert!(
                    live,
                    "integrity: constraint {:?} references dead entity {eid:?}",
                    constraint.id()
                );
            }
        }
    }

    // -------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------

    /// Number of independent clusters in the current decomposition.
    pub fn cluster_count(&self) -> usize {
        self.pipeline.cluster_count()
    }

    /// Degrees of freedom: (free params) - (total equation count).
    ///
    /// A positive DOF means under-constrained; zero means well-constrained;
    /// negative means over-constrained.
    pub fn degrees_of_freedom(&self) -> i32 {
        let free_params = self.params.free_param_count() as i32;
        let equations: i32 = self
            .constraints
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|c| c.equation_count() as i32)
            .sum();
        free_params - equations
    }

    /// Number of alive entities.
    pub fn entity_count(&self) -> usize {
        self.entities.iter().filter(|e| e.is_some()).count()
    }

    /// Number of alive constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraints.iter().filter(|c| c.is_some()).count()
    }

    // -------------------------------------------------------------------
    // Solving
    // -------------------------------------------------------------------

    /// Solve the constraint system.
    ///
    /// Delegates to the [`SolvePipeline`] which handles decomposition,
    /// analysis, reduction, per-cluster solving, and post-processing.
    pub fn solve(&mut self) -> SystemResult {
        self.solve_with_clock(&StdClock)
    }

    /// Solve the constraint system using a host-provided clock.
    ///
    /// Embedders that own determinism or platform policy should use this entry
    /// point instead of [`solve`](Self::solve), then route elapsed-time capture
    /// through their own platform abstraction.
    pub fn solve_with_clock<C: SolveClock>(&mut self, clock: &C) -> SystemResult {
        #[cfg(debug_assertions)]
        self.debug_validate_integrity();
        let start = clock.mark();
        let mut result = self.pipeline.run_with_clock(
            &self.constraints,
            &self.entities,
            &mut self.params,
            &self.config,
            &mut self.change_tracker,
            &mut self.solution_cache,
            clock,
        );
        result.duration = clock.elapsed(&start);
        self.certify_final_residuals(&mut result);
        result
    }

    /// Solve only clusters affected by parameter changes since the last solve.
    /// Falls back to full solve on structural changes.
    pub fn solve_incremental(&mut self) -> SystemResult {
        // Same as solve() -- the pipeline already handles incremental logic.
        self.solve()
    }

    /// Incremental solve using a host-provided clock.
    pub fn solve_incremental_with_clock<C: SolveClock>(&mut self, clock: &C) -> SystemResult {
        // Same as solve_with_clock() -- the pipeline already handles incremental logic.
        self.solve_with_clock(clock)
    }

    fn certify_final_residuals(&self, result: &mut SystemResult) {
        let tolerance = self.config.final_residual_tolerance;
        let constraint_clusters = self.pipeline.constraint_cluster_map();
        // Residual norm each cluster claims for its solution — but only for
        // clusters claiming success. A NotConverged cluster admits failure,
        // so its constraints are certified against the tolerance alone.
        let reported: std::collections::HashMap<crate::id::ClusterId, f64> = result
            .clusters
            .iter()
            .filter(|c| {
                matches!(
                    c.status,
                    ClusterSolveStatus::Converged | ClusterSolveStatus::Skipped
                )
            })
            .map(|c| (c.cluster_id, c.residual_norm))
            .collect();

        let mut max_residual = 0.0_f64;
        let mut failing_constraints = Vec::new();

        for (idx, constraint) in self
            .constraints
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_deref().map(|c| (i, c)))
        {
            let local_max = constraint
                .residuals(&self.params)
                .into_iter()
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            max_residual = max_residual.max(local_max);

            // A constraint fails certification when its residual exceeds the
            // tolerance AND exceeds what its cluster's solve reported. The
            // second condition keeps legitimate least-squares minima (over-
            // determined clusters converge with non-zero residual, which the
            // cluster reports) while still catching cached, skipped, or
            // stale-cascade solutions whose store state is worse than the
            // reported result.
            let reported_norm = constraint_clusters
                .get(&idx)
                .and_then(|cid| reported.get(cid))
                .copied()
                .unwrap_or(0.0);
            if local_max > tolerance && local_max > reported_norm * (1.0 + 1e-9) + tolerance {
                failing_constraints.push(constraint.id());
            }
        }

        if failing_constraints.is_empty() {
            return;
        }

        let issue = DiagnosticIssue::UnsatisfiedConstraints {
            constraints: failing_constraints,
            max_residual,
            tolerance: self.config.final_residual_tolerance,
        };

        result.status = match &mut result.status {
            SystemStatus::DiagnosticFailure(issues) => {
                issues.push(issue);
                return;
            }
            _ => SystemStatus::DiagnosticFailure(vec![issue]),
        };
    }

    /// Project a drag displacement onto the constraint manifold.
    pub fn drag(&mut self, displacements: &[(ParamId, f64)]) -> crate::solve::drag::DragResult {
        use crate::solve::drag::{apply_drag, project_drag};

        // Build constraint refs and mapping for affected params.
        let constraint_refs: Vec<&dyn Constraint> = self
            .constraints
            .iter()
            .filter_map(|c| c.as_deref())
            .collect();
        let mapping = self.params.build_solver_mapping();

        let result = project_drag(
            &constraint_refs,
            &self.params,
            &mapping,
            displacements,
            1e-10,
        );

        apply_drag(&mut self.params, &mapping, &result);

        // Mark dragged params dirty for subsequent solve.
        for &(pid, _) in displacements {
            self.change_tracker.mark_param_dirty(pid);
        }

        result
    }

    /// Analyze redundancy in the constraint system.
    pub fn analyze_redundancy(&self) -> crate::graph::redundancy::RedundancyAnalysis {
        let constraint_refs: Vec<(usize, &dyn Constraint)> = self
            .constraints
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_deref().map(|c| (i, c as &dyn Constraint)))
            .collect();
        let mapping = self.params.build_solver_mapping();
        crate::graph::redundancy::analyze_redundancy(
            &constraint_refs,
            &self.params,
            &mapping,
            1e-10,
        )
    }

    /// Analyze degrees of freedom per entity.
    pub fn analyze_dof(&self) -> crate::graph::dof::DofAnalysis {
        let entity_refs: Vec<&dyn Entity> =
            self.entities.iter().filter_map(|e| e.as_deref()).collect();
        let constraint_refs: Vec<(usize, &dyn Constraint)> = self
            .constraints
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_deref().map(|c| (i, c as &dyn Constraint)))
            .collect();
        let mapping = self.params.build_solver_mapping();
        crate::graph::dof::analyze_dof(&entity_refs, &constraint_refs, &self.params, &mapping)
    }

    /// Run full diagnostics (redundancy + DOF analysis).
    pub fn diagnose(&self) -> Vec<DiagnosticIssue> {
        let mut issues = Vec::new();

        let redundancy = self.analyze_redundancy();
        for r in &redundancy.redundant {
            issues.push(DiagnosticIssue::RedundantConstraint {
                constraint: r.id,
                implied_by: vec![],
            });
        }
        for g in &redundancy.conflicts {
            issues.push(DiagnosticIssue::ConflictingConstraints {
                constraints: g.constraint_ids.clone(),
            });
        }

        let dof = self.analyze_dof();
        for e in &dof.entities {
            if e.dof > 0 {
                issues.push(DiagnosticIssue::UnderConstrained {
                    entity: e.entity_id,
                    free_directions: e.dof,
                });
            }
        }

        issues
    }

    // -------------------------------------------------------------------
    // Optimization
    // -------------------------------------------------------------------

    /// Set the objective function to minimize.
    ///
    /// Only one objective is supported at a time. Setting a new objective
    /// replaces the previous one entirely, including any exact Hessian
    /// installed via [`set_objective_with_hessian`](Self::set_objective_with_hessian).
    pub fn set_objective(&mut self, objective: Box<dyn Objective>) {
        self.objective = Some(ObjectiveModel::FirstOrder(objective));
    }

    /// Set an objective that also provides an exact Hessian.
    ///
    /// This installs one coherent objective used for both first- and
    /// second-order evaluation: [`optimize`](Self::optimize) minimizes it
    /// like any other objective, and the trust-region solver additionally
    /// uses its exact Hessian instead of the L-BFGS approximation, giving
    /// quadratic convergence near the solution.
    pub fn set_objective_with_hessian(
        &mut self,
        objective: Box<dyn crate::optimization::ObjectiveHessian>,
    ) {
        self.objective = Some(ObjectiveModel::SecondOrder(objective));
    }

    /// Remove the objective function (revert to constraint-satisfaction only).
    pub fn clear_objective(&mut self) {
        self.objective = None;
    }

    /// Whether an objective function is set (first- or second-order).
    pub fn has_objective(&self) -> bool {
        self.objective.is_some()
    }

    /// Add an inequality constraint h(x) ≤ 0.
    ///
    /// The inequality must already have its `ConstraintId` set (via
    /// [`alloc_constraint_id`](Self::alloc_constraint_id) — shares ID space
    /// with equality constraints).
    pub fn add_inequality(&mut self, inequality: Box<dyn InequalityFn>) -> ConstraintId {
        let id = inequality.id();
        let idx = id.raw_index() as usize;

        if idx >= self.inequalities.len() {
            self.inequalities.resize_with(idx + 1, || None);
        }
        self.inequalities[idx] = Some(inequality);
        id
    }

    /// Set optimization configuration.
    pub fn set_opt_config(&mut self, config: OptimizationConfig) {
        self.opt_config = config;
    }

    /// Get the current optimization configuration.
    pub fn opt_config(&self) -> &OptimizationConfig {
        &self.opt_config
    }

    /// Structural fingerprint of the optimization problem: which objective
    /// and which equality/inequality constraints it consists of. Used to
    /// prevent warm-start multipliers from being reused across a different
    /// problem.
    fn problem_fingerprint(&self) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let model = self.objective.as_ref()?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let oid = model.objective().id();
        oid.raw_index().hash(&mut hasher);
        matches!(model, ObjectiveModel::SecondOrder(_)).hash(&mut hasher);
        for c in self.constraints.iter().filter_map(|c| c.as_deref()) {
            c.id().raw_index().hash(&mut hasher);
            c.id().generation.hash(&mut hasher);
            c.equation_count().hash(&mut hasher);
        }
        0xFFu8.hash(&mut hasher); // separator between equalities and inequalities
        for h in self.inequalities.iter().filter_map(|h| h.as_deref()) {
            h.id().raw_index().hash(&mut hasher);
            h.id().generation.hash(&mut hasher);
            h.inequality_count().hash(&mut hasher);
        }
        Some(hasher.finish())
    }

    fn unsupported_structure(reason: impl Into<String>) -> OptimizationResult {
        OptimizationResult {
            objective_value: f64::NAN,
            status: OptimizationStatus::UnsupportedProblemStructure {
                reason: reason.into(),
            },
            outer_iterations: 0,
            inner_iterations: 0,
            kkt_residual: crate::optimization::KktResidual {
                primal: f64::INFINITY,
                dual: f64::INFINITY,
                complementarity: f64::INFINITY,
            },
            multipliers: MultiplierStore::new(),
            constraint_violations: Vec::new(),
            duration: std::time::Duration::ZERO,
        }
    }

    /// Run constrained optimization: `min f(x) s.t. constraints`.
    ///
    /// Requires an objective to be set via [`set_objective`](Self::set_objective)
    /// or [`set_objective_with_hessian`](Self::set_objective_with_hessian).
    /// Existing [`Constraint`] objects serve as equality constraints (`g(x) = 0`).
    /// [`InequalityFn`] objects serve as inequality constraints (`h(x) ≤ 0`).
    ///
    /// # Algorithm Selection
    ///
    /// - Equality or inequality constraints → ALM (with BFGS/BFGS-B inner loop)
    /// - Finite bounds only → BFGS-B
    /// - Otherwise → BFGS
    ///
    /// Explicitly selecting an algorithm that cannot honor the registered
    /// problem structure (e.g. BFGS with equality constraints present)
    /// returns [`OptimizationStatus::UnsupportedProblemStructure`] without
    /// modifying any parameters, instead of silently solving a different
    /// problem.
    pub fn optimize(&mut self) -> OptimizationResult {
        self.optimize_with_clock(&StdClock)
    }

    /// Like [`optimize`](Self::optimize), but with a host-provided clock for
    /// duration reporting (mirrors [`solve_with_clock`](Self::solve_with_clock)).
    pub fn optimize_with_clock<C: SolveClock>(&mut self, clock: &C) -> OptimizationResult {
        let model = match &self.objective {
            Some(model) => model,
            None => {
                // Nothing is infeasible here — the problem is malformed.
                return Self::unsupported_structure(
                    "no objective set; call set_objective or set_objective_with_hessian first",
                );
            }
        };
        let objective = model.objective();

        // Reject nonsensical configurations before touching any parameters.
        if let Err(reason) = self.opt_config.validate() {
            return Self::unsupported_structure(format!("invalid optimization config: {reason}"));
        }

        // Classify: check if we have equality constraints
        let eq_constraints: Vec<&dyn Constraint> = self
            .constraints
            .iter()
            .filter_map(|c| c.as_deref())
            .collect();

        let has_equalities = !eq_constraints.is_empty();

        // Algorithm selection
        use crate::optimization::OptimizationAlgorithm;

        // Check if any free parameter has finite bounds.
        let has_finite_bounds = self
            .params
            .free_param_ids()
            .any(|pid| self.params.has_finite_bounds(pid));

        let ineq_constraints: Vec<&dyn InequalityFn> = self
            .inequalities
            .iter()
            .filter_map(|h| h.as_deref())
            .collect();

        let has_inequalities = !ineq_constraints.is_empty();

        // Validate algorithm/problem compatibility before any solver touches
        // parameter values.
        match self.opt_config.algorithm {
            OptimizationAlgorithm::Bfgs => {
                if has_equalities || has_inequalities {
                    return Self::unsupported_structure(
                        "BFGS cannot honor equality/inequality constraints; use Auto or Alm",
                    );
                }
                if has_finite_bounds {
                    return Self::unsupported_structure(
                        "BFGS ignores parameter bounds; use Auto or BfgsB",
                    );
                }
            }
            OptimizationAlgorithm::BfgsB => {
                if has_equalities || has_inequalities {
                    return Self::unsupported_structure(
                        "BFGS-B cannot honor equality/inequality constraints; use Auto or Alm",
                    );
                }
            }
            OptimizationAlgorithm::TrustRegion => {
                if has_equalities || has_inequalities {
                    return Self::unsupported_structure(
                        "trust region is unconstrained-only; use Auto or Alm",
                    );
                }
                if has_finite_bounds {
                    return Self::unsupported_structure(
                        "trust region ignores parameter bounds; use Auto or BfgsB",
                    );
                }
            }
            OptimizationAlgorithm::Auto | OptimizationAlgorithm::Alm => {}
        }

        // Invalidate warm-start multipliers when the problem structure
        // changed since they were produced.
        let fingerprint = self.problem_fingerprint();
        if fingerprint != self.last_problem_fingerprint {
            self.last_multipliers = MultiplierStore::new();
        }

        let algorithm = match self.opt_config.algorithm {
            OptimizationAlgorithm::Auto => {
                if has_equalities || has_inequalities {
                    OptimizationAlgorithm::Alm
                } else if has_finite_bounds {
                    OptimizationAlgorithm::BfgsB
                } else {
                    OptimizationAlgorithm::Bfgs
                }
            }
            other => other,
        };

        let result = match algorithm {
            OptimizationAlgorithm::Bfgs => crate::solver::BfgsSolver::new(self.opt_config.clone())
                .solve_with_clock(objective, &mut self.params, clock),
            OptimizationAlgorithm::BfgsB => crate::solver::BfgsBSolver::new(
                self.opt_config.clone(),
            )
            .solve_with_clock(objective, &mut self.params, clock),
            OptimizationAlgorithm::Alm => {
                let warm = match self.opt_config.alm.multiplier_init {
                    crate::optimization::MultiplierInitStrategy::WarmStart => {
                        Some(&self.last_multipliers)
                    }
                    _ => None,
                };
                crate::solver::AlmSolver::new(self.opt_config.clone()).solve_with_clock(
                    objective,
                    &eq_constraints,
                    &ineq_constraints,
                    &mut self.params,
                    warm,
                    clock,
                )
            }
            OptimizationAlgorithm::TrustRegion => {
                let solver = crate::solver::TrustRegionSolver::new(self.opt_config.clone());
                if let Some(hess_obj) = model.hessian() {
                    solver.solve_with_hessian_and_clock(hess_obj, &mut self.params, clock)
                } else {
                    solver.solve_with_clock(objective, &mut self.params, clock)
                }
            }
            OptimizationAlgorithm::Auto => {
                unreachable!("Auto is resolved to a concrete algorithm before this match")
            }
        };

        // Store multipliers (and the problem they belong to) for post-solve
        // access and warm starting.
        self.last_multipliers = MultiplierStore::new();
        for (mid, val) in result.multipliers.iter() {
            self.last_multipliers.set(mid, val);
        }
        self.last_problem_fingerprint = fingerprint;

        result
    }

    /// Get the Lagrange multipliers from the last optimization solve for a
    /// specific constraint.
    ///
    /// Returns `None` if no optimization has been run or if the constraint
    /// has no multipliers.
    pub fn multiplier(&self, constraint_id: ConstraintId) -> Option<Vec<f64>> {
        self.last_multipliers.lambda_for_constraint(constraint_id)
    }

    /// Get the full multiplier store from the last optimization solve.
    pub fn multipliers(&self) -> &MultiplierStore {
        &self.last_multipliers
    }

    // -------------------------------------------------------------------
    // Pipeline
    // -------------------------------------------------------------------

    /// Set a custom pipeline for this system.
    pub fn set_pipeline(&mut self, pipeline: SolvePipeline) {
        self.pipeline = pipeline;
    }

    /// Access the change tracker.
    pub fn change_tracker(&self) -> &ChangeTracker {
        &self.change_tracker
    }

    // -----------------------------------------------------------------
    // Convenience methods (useful for testing and geometry plugins)
    // -----------------------------------------------------------------

    /// Total number of scalar equations across all constraints.
    pub fn equation_count(&self) -> usize {
        self.constraints
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|c| c.equation_count())
            .sum()
    }

    /// Evaluate all constraint residuals at the current parameter values.
    pub fn compute_residuals(&self) -> Vec<f64> {
        let mut residuals = Vec::new();
        for c in &self.constraints {
            if let Some(c) = c.as_ref() {
                residuals.extend(c.residuals(&self.params));
            }
        }
        residuals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::Constraint;
    use crate::entity::Entity;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;
    use crate::solver::{LMConfig, SolverConfig};

    // -------------------------------------------------------------------
    // Test entity: a 2D point with two parameters (x, y).
    // -------------------------------------------------------------------
    struct TestPoint {
        id: EntityId,
        params: Vec<ParamId>,
    }

    impl Entity for TestPoint {
        fn id(&self) -> EntityId {
            self.id
        }
        fn params(&self) -> &[ParamId] {
            &self.params
        }
        fn name(&self) -> &str {
            "TestPoint"
        }
    }

    // -------------------------------------------------------------------
    // Test constraint: distance between two 1D points equals target.
    //   residual = (a - b)^2 - d^2  (single equation)
    //   Using squared form to keep it simple. For tests with small values
    //   we use the linear form: residual = a - target.
    // -------------------------------------------------------------------
    struct FixValueConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        param: ParamId,
        target: f64,
    }

    impl Constraint for FixValueConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "FixValue"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entity_ids
        }
        fn param_ids(&self) -> &[ParamId] {
            std::slice::from_ref(&self.param)
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.param) - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.param, 1.0)]
        }
    }

    // -------------------------------------------------------------------
    // Test constraint: a + b = target  (sum constraint).
    // -------------------------------------------------------------------
    struct SumConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        params: Vec<ParamId>,
        target: f64,
    }

    impl Constraint for SumConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "Sum"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entity_ids
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            let a = store.get(self.params[0]);
            let b = store.get(self.params[1]);
            vec![a + b - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.params[0], 1.0), (0, self.params[1], 1.0)]
        }
    }

    /// Helper to build a point entity in the system.
    fn add_test_point(
        system: &mut ConstraintSystem,
        x: f64,
        y: f64,
    ) -> (EntityId, ParamId, ParamId) {
        let eid = system.alloc_entity_id();
        let px = system.alloc_param(x, eid);
        let py = system.alloc_param(y, eid);
        let point = TestPoint {
            id: eid,
            params: vec![px, py],
        };
        system.add_entity(Box::new(point));
        (eid, px, py)
    }

    #[test]
    fn test_empty_system() {
        let system = ConstraintSystem::new();
        assert_eq!(system.entity_count(), 0);
        assert_eq!(system.constraint_count(), 0);
        assert_eq!(system.degrees_of_freedom(), 0);
    }

    #[test]
    fn test_add_entity() {
        let mut system = ConstraintSystem::new();
        let (eid, _px, _py) = add_test_point(&mut system, 1.0, 2.0);

        assert_eq!(system.entity_count(), 1);
        assert_eq!(system.params().alive_count(), 2);
        assert_eq!(system.degrees_of_freedom(), 2); // 2 free params, 0 constraints

        // Verify param values
        let _ = eid; // used for ownership
    }

    #[test]
    fn test_add_and_remove_entity() {
        let mut system = ConstraintSystem::new();
        let (eid, px, py) = add_test_point(&mut system, 3.0, 4.0);

        assert_eq!(system.entity_count(), 1);
        assert_eq!(system.params().alive_count(), 2);

        system.remove_entity(eid).unwrap();
        assert_eq!(system.entity_count(), 0);
        // Params should be freed
        assert_eq!(system.params().alive_count(), 0);

        // Suppress unused variable warnings
        let _ = (px, py);
    }

    #[test]
    fn test_add_constraint() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

        let cid = system.alloc_constraint_id();
        let constraint = FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 5.0,
        };
        system.add_constraint(Box::new(constraint));

        assert_eq!(system.constraint_count(), 1);
        // DOF = 2 free params - 1 equation = 1
        assert_eq!(system.degrees_of_freedom(), 1);
    }

    #[test]
    fn test_remove_constraint() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

        let cid = system.alloc_constraint_id();
        let constraint = FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 5.0,
        };
        system.add_constraint(Box::new(constraint));
        assert_eq!(system.constraint_count(), 1);

        system.remove_constraint(cid).unwrap();
        assert_eq!(system.constraint_count(), 0);
        assert_eq!(system.degrees_of_freedom(), 2);
    }

    #[test]
    fn test_fix_unfix_param() {
        let mut system = ConstraintSystem::new();
        let (_eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

        assert_eq!(system.degrees_of_freedom(), 2);

        system.fix_param(px);
        assert_eq!(system.degrees_of_freedom(), 1); // one param fixed

        system.unfix_param(px);
        assert_eq!(system.degrees_of_freedom(), 2);
    }

    #[test]
    fn remove_entity_stale_id_errors() {
        let mut system = ConstraintSystem::new();
        let eid = system.alloc_entity_id();
        // Never added: removal must report a stale/unknown ID.
        assert_eq!(system.remove_entity(eid), Err(RemovalError::StaleId));
    }

    #[test]
    fn remove_entity_with_dependent_constraint_is_refused() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 1.0,
        }));

        match system.remove_entity(eid) {
            Err(RemovalError::HasDependentConstraints { constraints }) => {
                assert_eq!(constraints, vec![cid]);
            }
            other => panic!("expected HasDependentConstraints, got {other:?}"),
        }

        // After removing the constraint, entity removal succeeds.
        system.remove_constraint(cid).unwrap();
        system.remove_entity(eid).unwrap();
    }

    #[test]
    fn remove_constraint_stale_id_errors() {
        let mut system = ConstraintSystem::new();
        let cid = system.alloc_constraint_id();
        assert_eq!(system.remove_constraint(cid), Err(RemovalError::StaleId));
    }

    #[test]
    fn test_solve_empty_system() {
        let mut system = ConstraintSystem::new();
        let result = system.solve();

        assert!(matches!(result.status, SystemStatus::Solved));
        assert_eq!(result.clusters.len(), 0);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn solve_with_zero_clock_reports_deterministic_duration() {
        let mut system = ConstraintSystem::new();
        let result = system.solve_with_clock(&crate::time::ZeroClock);

        assert!(matches!(result.status, SystemStatus::Solved));
        assert_eq!(result.duration, std::time::Duration::ZERO);
    }

    #[test]
    fn final_residual_certification_rejects_unsatisfied_constraints() {
        let mut system = ConstraintSystem::new();
        system.config_mut().final_residual_tolerance = 1e-9;
        let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);

        system.fix_param(px);
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 7.0,
        }));

        let result = system.solve_with_clock(&crate::time::ZeroClock);

        let SystemStatus::DiagnosticFailure(issues) = result.status else {
            panic!("expected diagnostic failure, got {:?}", result.status);
        };
        assert!(issues.iter().any(|issue| matches!(
            issue,
            DiagnosticIssue::UnsatisfiedConstraints {
                constraints,
                max_residual,
                tolerance,
            } if constraints == &vec![cid]
                && (*max_residual - 7.0).abs() < 1e-12
                && (*tolerance - 1e-9).abs() < 1e-15
        )));
    }

    #[test]
    fn test_solve_single_fix_constraint() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);

        let cid = system.alloc_constraint_id();
        let constraint = FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 7.0,
        };
        system.add_constraint(Box::new(constraint));

        let result = system.solve();
        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Expected Solved or PartiallySolved, got {:?}",
            result.status
        );

        // px should now be close to 7.0
        let val = system.get_param(px);
        assert!((val - 7.0).abs() < 1e-6, "Expected px ~ 7.0, got {}", val);
    }

    #[test]
    fn test_solve_two_independent_clusters() {
        let mut system = ConstraintSystem::new();
        let (eid1, px1, _py1) = add_test_point(&mut system, 0.0, 0.0);
        let (eid2, px2, _py2) = add_test_point(&mut system, 0.0, 0.0);

        // Constraint on px1 -> target 3.0
        let cid1 = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid1,
            entity_ids: vec![eid1],
            param: px1,
            target: 3.0,
        }));

        // Constraint on px2 -> target 5.0 (independent cluster)
        let cid2 = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid2,
            entity_ids: vec![eid2],
            param: px2,
            target: 5.0,
        }));

        let result = system.solve();

        // Should be 2 clusters
        assert_eq!(result.clusters.len(), 2);

        // Both should converge
        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Expected Solved or PartiallySolved, got {:?}",
            result.status
        );

        assert!((system.get_param(px1) - 3.0).abs() < 1e-6);
        assert!((system.get_param(px2) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_solve_coupled_constraints() {
        let mut system = ConstraintSystem::new();
        let (eid, px, py) = add_test_point(&mut system, 0.0, 0.0);

        // Fix px = 3.0
        let cid1 = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid1,
            entity_ids: vec![eid],
            param: px,
            target: 3.0,
        }));

        // px + py = 10.0  =>  py = 7.0
        let cid2 = system.alloc_constraint_id();
        system.add_constraint(Box::new(SumConstraint {
            id: cid2,
            entity_ids: vec![eid],
            params: vec![px, py],
            target: 10.0,
        }));

        let result = system.solve();

        // These two constraints share px, so they should be in the same cluster
        assert_eq!(
            result.clusters.len(),
            1,
            "Coupled constraints should form 1 cluster"
        );

        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Solve status: {:?}",
            result.status
        );

        assert!(
            (system.get_param(px) - 3.0).abs() < 1e-6,
            "px = {}, expected 3.0",
            system.get_param(px)
        );
        assert!(
            (system.get_param(py) - 7.0).abs() < 1e-6,
            "py = {}, expected 7.0",
            system.get_param(py)
        );
    }

    #[test]
    fn test_solve_with_fixed_param_cluster_skipped() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 5.0, 0.0);

        // Fix px so it cannot move
        system.fix_param(px);

        // Constraint wants px = 5.0 (already satisfied since px is fixed at 5.0)
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 5.0,
        }));

        let result = system.solve();

        // The cluster should be skipped (no free variables)
        assert_eq!(result.clusters.len(), 1);
        assert_eq!(result.clusters[0].status, ClusterSolveStatus::Skipped);
        // Residual should be ~0 since the constraint is already satisfied
        assert!(result.clusters[0].residual_norm < 1e-10);
    }

    #[test]
    fn test_get_set_param() {
        let mut system = ConstraintSystem::new();
        let eid = system.alloc_entity_id();
        let p = system.alloc_param(42.0, eid);

        assert!((system.get_param(p) - 42.0).abs() < 1e-12);

        system.set_param(p, 99.0);
        assert!((system.get_param(p) - 99.0).abs() < 1e-12);
    }

    #[test]
    fn test_with_config() {
        let config = SystemConfig {
            lm_config: LMConfig::robust(),
            solver_config: SolverConfig::fast(),
            final_residual_tolerance: 1e-9,
        };
        let system = ConstraintSystem::with_config(config);
        assert_eq!(system.entity_count(), 0);
        assert!((system.config().final_residual_tolerance - 1e-9).abs() < 1e-15);
    }

    #[test]
    fn test_system_result_duration() {
        let mut system = ConstraintSystem::new();
        let result = system.solve();
        // Duration should be non-negative (trivially true but checks the field exists)
        let _duration = result.duration;
    }

    #[test]
    fn test_structural_change_triggers_redecompose() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

        // First solve works fine
        let _ = system.solve();

        // Adding a constraint is a structural change
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 5.0,
        }));

        // The change tracker should have structural changes
        assert!(system.change_tracker().has_structural_changes());

        // Solve again should succeed (triggers re-decompose)
        let result = system.solve();
        assert!(matches!(
            result.status,
            SystemStatus::Solved | SystemStatus::PartiallySolved
        ));

        // After solve, change tracker is cleared
        assert!(!system.change_tracker().has_any_changes());

        // Removing the constraint is also a structural change
        system.remove_constraint(cid).unwrap();
        assert!(system.change_tracker().has_structural_changes());
    }

    // -------------------------------------------------------------------
    // Objective/Hessian API tests
    // -------------------------------------------------------------------

    use crate::optimization::{
        Objective, ObjectiveHessian, ObjectiveId, OptimizationAlgorithm, OptimizationStatus,
    };

    /// f(x) = (x - target)^2
    struct QuadObjective {
        id_index: u32,
        param: ParamId,
        target: f64,
    }

    impl Objective for QuadObjective {
        fn id(&self) -> ObjectiveId {
            ObjectiveId::new(self.id_index, 0)
        }
        fn name(&self) -> &str {
            "quad"
        }
        fn param_ids(&self) -> &[ParamId] {
            std::slice::from_ref(&self.param)
        }
        fn value(&self, store: &ParamStore) -> f64 {
            (store.get(self.param) - self.target).powi(2)
        }
        fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
            vec![(self.param, 2.0 * (store.get(self.param) - self.target))]
        }
    }

    impl ObjectiveHessian for QuadObjective {
        fn hessian_entries(&self, _store: &ParamStore) -> Vec<(ParamId, ParamId, f64)> {
            vec![(self.param, self.param, 2.0)]
        }
    }

    fn quad_system(target: f64) -> (ConstraintSystem, ParamId) {
        let mut system = ConstraintSystem::new();
        let eid = system.alloc_entity_id();
        let p = system.alloc_param(0.0, eid);
        let _ = target;
        (system, p)
    }

    #[test]
    fn hessian_objective_alone_is_optimizable() {
        let (mut system, p) = quad_system(4.0);
        system.set_objective_with_hessian(Box::new(QuadObjective {
            id_index: 0,
            param: p,
            target: 4.0,
        }));
        assert!(system.has_objective());
        let result = system.optimize();
        assert_eq!(result.status, OptimizationStatus::Converged);
        assert!((system.get_param(p) - 4.0).abs() < 1e-4);
    }

    #[test]
    fn set_objective_replaces_hessian_objective() {
        let (mut system, p) = quad_system(4.0);
        system.set_objective_with_hessian(Box::new(QuadObjective {
            id_index: 0,
            param: p,
            target: 4.0,
        }));
        // Replace with a first-order objective at a different target: the
        // stale Hessian objective must not survive.
        system.set_objective(Box::new(QuadObjective {
            id_index: 1,
            param: p,
            target: -2.0,
        }));
        let result = system.optimize();
        assert_eq!(result.status, OptimizationStatus::Converged);
        assert!(
            (system.get_param(p) + 2.0).abs() < 1e-4,
            "expected new objective's minimum -2, got {}",
            system.get_param(p)
        );

        // And the other replacement order: first-order then second-order.
        system.set_param(p, 0.0);
        system.set_objective_with_hessian(Box::new(QuadObjective {
            id_index: 2,
            param: p,
            target: 7.0,
        }));
        let result = system.optimize();
        assert_eq!(result.status, OptimizationStatus::Converged);
        assert!((system.get_param(p) - 7.0).abs() < 1e-4);
    }

    #[test]
    fn clear_objective_removes_both_orders() {
        let (mut system, p) = quad_system(4.0);
        system.set_objective_with_hessian(Box::new(QuadObjective {
            id_index: 0,
            param: p,
            target: 4.0,
        }));
        system.clear_objective();
        assert!(!system.has_objective());
        let result = system.optimize();
        assert!(matches!(
            result.status,
            OptimizationStatus::UnsupportedProblemStructure { .. }
        ));
    }

    #[test]
    fn trust_region_uses_installed_hessian_objective() {
        let (mut system, p) = quad_system(9.0);
        system.set_objective_with_hessian(Box::new(QuadObjective {
            id_index: 0,
            param: p,
            target: 9.0,
        }));
        let config = crate::optimization::OptimizationConfig {
            algorithm: OptimizationAlgorithm::TrustRegion,
            ..Default::default()
        };
        system.set_opt_config(config);
        let result = system.optimize();
        assert_eq!(result.status, OptimizationStatus::Converged);
        assert!((system.get_param(p) - 9.0).abs() < 1e-4);
    }

    #[test]
    fn explicit_bfgs_with_equality_constraints_is_rejected() {
        let (mut system, p) = quad_system(4.0);
        system.set_objective(Box::new(QuadObjective {
            id_index: 0,
            param: p,
            target: 4.0,
        }));
        let eid = system.alloc_entity_id();
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: p,
            target: 1.0,
        }));

        let before = system.get_param(p);
        let config = crate::optimization::OptimizationConfig {
            algorithm: OptimizationAlgorithm::Bfgs,
            ..Default::default()
        };
        system.set_opt_config(config);
        let result = system.optimize();
        assert!(
            matches!(
                result.status,
                OptimizationStatus::UnsupportedProblemStructure { .. }
            ),
            "expected UnsupportedProblemStructure, got {:?}",
            result.status
        );
        // Parameters must be untouched.
        assert_eq!(system.get_param(p), before);
    }

    #[test]
    fn explicit_trust_region_with_constraints_is_rejected() {
        let (mut system, p) = quad_system(4.0);
        system.set_objective(Box::new(QuadObjective {
            id_index: 0,
            param: p,
            target: 4.0,
        }));
        let eid = system.alloc_entity_id();
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: p,
            target: 1.0,
        }));

        let config = crate::optimization::OptimizationConfig {
            algorithm: OptimizationAlgorithm::TrustRegion,
            ..Default::default()
        };
        system.set_opt_config(config);
        let result = system.optimize();
        assert!(matches!(
            result.status,
            OptimizationStatus::UnsupportedProblemStructure { .. }
        ));
    }

    #[test]
    fn explicit_bfgs_with_bounds_is_rejected() {
        let (mut system, p) = quad_system(4.0);
        system.set_objective(Box::new(QuadObjective {
            id_index: 0,
            param: p,
            target: 4.0,
        }));
        system.params_mut().set_bounds(p, 0.0, 1.0);

        let config = crate::optimization::OptimizationConfig {
            algorithm: OptimizationAlgorithm::Bfgs,
            ..Default::default()
        };
        system.set_opt_config(config);
        let result = system.optimize();
        assert!(matches!(
            result.status,
            OptimizationStatus::UnsupportedProblemStructure { .. }
        ));
    }

    #[test]
    fn replacing_objective_invalidates_warm_start_multipliers() {
        // Solve a constrained problem, then swap the objective. The stored
        // multipliers must not be handed to the next solve as a warm start
        // for a structurally different problem.
        let (mut system, p) = quad_system(4.0);
        system.set_objective(Box::new(QuadObjective {
            id_index: 0,
            param: p,
            target: 4.0,
        }));
        let eid = system.alloc_entity_id();
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: p,
            target: 1.0,
        }));
        let config = crate::optimization::OptimizationConfig {
            alm: crate::optimization::AlmConfig {
                multiplier_init: crate::optimization::MultiplierInitStrategy::WarmStart,
                ..Default::default()
            },
            ..Default::default()
        };
        system.set_opt_config(config);

        let r1 = system.optimize();
        assert_eq!(r1.status, OptimizationStatus::Converged);
        assert!(system.multiplier(cid).is_some());

        // New objective (different id) — same constraint set. The next
        // solve must start from clean multipliers and still converge.
        system.set_objective(Box::new(QuadObjective {
            id_index: 5,
            param: p,
            target: -10.0,
        }));
        let r2 = system.optimize();
        assert_eq!(r2.status, OptimizationStatus::Converged);
        assert!((system.get_param(p) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_cluster_count_after_solve() {
        let mut system = ConstraintSystem::new();
        let (eid1, px1, _) = add_test_point(&mut system, 0.0, 0.0);
        let (eid2, px2, _) = add_test_point(&mut system, 0.0, 0.0);

        let cid1 = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid1,
            entity_ids: vec![eid1],
            param: px1,
            target: 1.0,
        }));

        let cid2 = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid2,
            entity_ids: vec![eid2],
            param: px2,
            target: 2.0,
        }));

        let _ = system.solve();
        assert_eq!(system.cluster_count(), 2);
    }
}
