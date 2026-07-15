## Assessment

Solverang already contains enough algorithms. The next phase should be **correctness, architectural consolidation, and a defensible public API**, not another solver implementation.

The repository currently spans:

* nonlinear root finding and least squares;
* dense, sparse, parallel, and JIT execution;
* an entity/constraint graph with reduction and incremental solving;
* 2D/3D geometry and rigid-body assembly;
* unconstrained, bounded, equality-constrained, and inequality-constrained optimization;
* symbolic gradients, Jacobians, and Hessians.

That breadth is impressive, but it has produced three problems:

1. Some important execution paths are not yet mathematically or state-machine correct.
2. Documentation and planning files no longer agree with the implementation.
3. The project’s product identity is unclear: generic numerical solver, CAD constraint kernel, or optimization toolkit.

My recommendation is to make Solverang an **interactive geometric constraint and optimization kernel that also exposes a clean domain-independent numerical layer**. The geometric system, incremental architecture, symbolic derivatives, and JIT are a more distinctive combination than another general-purpose Rust optimizer.

## P0: Correctness work to do immediately

### 1. Fix decomposition cascading and globally validate every solve

This is the highest-priority defect. `STATUS.md` already documents that an earlier cluster’s solution may not propagate to dependent clusters, allowing `SystemStatus::Solved` with non-zero cross-cluster residuals.

The pipeline currently determines the final status solely from per-cluster statuses. It does not reevaluate all constraints after all cluster solutions have been written back.

* [ ] Reevaluate every active constraint after the complete pipeline finishes.
* [ ] Never return `Solved` unless every hard constraint satisfies its configured tolerance.
* [ ] Model dependencies between clusters created by reduction/fixed-parameter substitution.
* [ ] Propagate dirty state downstream after a cluster changes shared or substituted parameters.
* [ ] Iterate dependent clusters to a fixed point, with a bounded pass count.
* [ ] Return the globally unsatisfied constraints in `DiagnosticFailure`.
* [ ] Add regression tests reproducing the documented cascading failure.
* [ ] Change `param_to_cluster` from a single-cluster mapping to a one-to-many dependency mapping where necessary.

This should be the release blocker for 0.1.

### 2. Repair the objective/Hessian API

`ConstraintSystem` stores first-order and second-order objectives in separate optional fields. `set_objective_with_hessian()` sets only the Hessian field, while `optimize()` requires the regular objective field to be populated. Consequently, calling the Hessian setter by itself cannot successfully optimize. A previously installed Hessian can also remain attached after replacing the ordinary objective.

* [ ] Make `set_objective_with_hessian()` install one coherent objective used for both first- and second-order evaluation.
* [ ] Clear any stale Hessian when `set_objective()` replaces the objective.
* [ ] Make `has_objective()` recognize a second-order objective.
* [ ] Prefer one stored abstraction, such as an `ObjectiveModel` enum or an object exposing optional Hessian-vector products.
* [ ] Add tests for replacing objectives in every possible order.
* [ ] Add a structural generation/fingerprint so warm-start multipliers cannot be reused with a different objective.

There is also an algorithm-selection problem: explicitly selecting BFGS, BFGS-B, or trust region currently bypasses registered equality and inequality constraints rather than rejecting an incompatible configuration.

* [ ] Validate algorithm/problem compatibility before changing parameters.
* [ ] Return a structured `UnsupportedProblemStructure` error instead of silently solving a different problem.
* [ ] Validate that BFGS-B is selected only where bounds are meaningful.
* [ ] Decide whether constrained trust region is unsupported or implement a genuine constrained method later.

### 3. Fix the L-BFGS-B line search

The bounded solver calls the ordinary unconstrained line search. That line search may try `α > 1`, which can evaluate the objective outside the box. The solver then projects the accepted point back into the box but retains the objective value calculated at the unprojected point. This can leave `x` and `f` inconsistent.

* [ ] Implement a bound-aware projected line search.
* [ ] Calculate `alpha_max` from the first bound encountered along the direction.
* [ ] Never evaluate outside the feasible box.
* [ ] Recompute the objective after any projection.
* [ ] Assert that the reported objective equals `objective.value(store)` at every accepted iterate.
* [ ] Return a line-search failure rather than treating the minimum step as success.
* [ ] Test objectives whose values are undefined outside their bounds, such as `ln(x)` with `x > 0`.
* [ ] Test a solution on a boundary, at a corner, and with several simultaneously active bounds.

### 4. Finish the ALM implementation mathematically

The ALM code is considerably further along than the documentation suggests, but several correctness gaps remain.

A zero-variable problem is immediately reported as converged without evaluating whether fixed parameters satisfy its equality or inequality constraints.

Penalty growth is based only on the equality residual norm. For an inequality-only problem that norm remains zero, so `rho` may never increase even when inequalities remain violated.

Complementarity is computed, but convergence only checks primal and dual residuals.

* [ ] Evaluate feasibility before returning from the zero-free-variable path.
* [ ] Base penalty updates on the combined equality and positive inequality violation.
* [ ] Check all KKT components: primal feasibility, stationarity, dual feasibility, and complementarity.
* [ ] Compute stationarity for the original Lagrangian, `∇f + Jgᵀλ + Jhᵀμ`, rather than using the gradient of the current penalized subproblem as the final KKT residual.
* [ ] Require `μ ≥ 0` and include explicit dual-feasibility diagnostics.
* [ ] Propagate inner BFGS/BFGS-B failures instead of continuing multiplier updates blindly.
* [ ] Detect infeasibility or stalled penalty progression.
* [ ] Include inequality values in `constraint_violations`.
* [ ] Clamp or validate warm-started inequality multipliers.
* [ ] Add mixed bounds + equalities + inequalities tests.
* [ ] Add contradictory-constraint tests and require an `Infeasible` or `Stalled` result rather than generic maximum-iterations termination.
* [ ] Update the ALM module documentation, which still describes an equality-only LM-based implementation even though the code now uses BFGS/BFGS-B and supports inequalities.

### 5. Make line-search failure explicit

The Wolfe implementation is present, but its zoom fallback tracks the lowest objective value rather than specifically the best Armijo-satisfying point. The unified wrapper accepts any objective decrease, even if the declared line-search conditions were not satisfied.

* [ ] Return `Result<LineSearchStep, LineSearchFailure>`.
* [ ] Record whether the accepted step satisfied Wolfe, Armijo only, or neither.
* [ ] Only retain fallback candidates that satisfy Armijo.
* [ ] Restore `ParamStore` to the returned candidate before exiting.
* [ ] Add function and gradient evaluation counts.
* [ ] Add a configurable evaluation budget.
* [ ] Detect non-finite objective and gradient values.
* [ ] Add interpolation to zoom after correctness is established; bisection can remain the dependable fallback.

## P1: Consolidate solver architecture

### 6. Use the correct active parameter set

BFGS and the other optimizers currently build a mapping from every free parameter in the entire `ParamStore`, even when the objective and constraints reference only a subset. The repository already has `build_solver_mapping_for()`, but the optimization solvers do not consistently use it.

This can introduce unrelated zero-gradient dimensions and unnecessary work.

* [ ] Construct the active set as the union of objective, equality, inequality, and bound-relevant parameter IDs.
* [ ] Use `SolverMapping::param_to_col` instead of repeated `param_ids.iter().position(...)`.
* [ ] Pass the mapping through gradient, Jacobian, Hessian, and multiplier assembly.
* [ ] Reject derivative entries referencing undeclared parameters.
* [ ] Add contract tests ensuring `param_ids()` accurately describes every derivative entry.

This removes numerous current `O(nnz × n)` lookups and makes large sparse problems much more credible.

### 7. Turn JIT into an evaluator backend, not a separate numerical solver

Several items in `TODO.md` are already complete. JIT auto-detection, fused residual/Jacobian evaluation, and direct dense output are present in `JITSolver`.

The more important architectural issue is that `JITSolver` is its own plain Newton solver. Enabling JIT therefore changes both the evaluation mechanism and the numerical algorithm.

* [ ] Define an evaluator interface for residuals, Jacobians, and fused evaluation.
* [ ] Implement interpreted, finite-difference, symbolic, and JIT evaluators.
* [ ] Let Newton, LM, robust, sparse, and future solvers consume any evaluator.
* [ ] Preserve identical convergence and fallback behavior when switching evaluator backends.
* [ ] Cache compiled functions by a stable structural fingerprint.
* [ ] Expose compilation failures in diagnostics, while retaining configurable fallback.
* [ ] Calibrate compilation thresholds for one-shot solves versus interactive repeated solves.
* [ ] Benchmark compile cost, evaluation cost, total solve time, and cache reuse separately.
* [ ] Remove completed items from `TODO.md`.

That makes JIT a genuine acceleration layer instead of an alternative, less robust solver.

### 8. Make sparse caching real

`SparseSolver` records a sparsity pattern, but it still reconstructs CSR, creates faer triplets, and performs a fresh sparse decomposition in each nonlinear iteration.

* [ ] Cache the mapping from Jacobian entries to numeric matrix storage.
* [ ] Reuse symbolic LU/QR analysis and ordering while only updating numeric values.
* [ ] Detect structural pattern changes and invalidate the symbolic cache.
* [ ] Support sparse minimum-norm solutions for underdetermined systems.
* [ ] Add sparse LM or sparse Gauss-Newton; large least-squares problems should not be forced through sparse Newton.
* [ ] Verify whether `linear_tolerance` affects any current factorization path; remove or implement it.
* [ ] Benchmark dense/sparse crossover rather than documenting a fixed “1000+ variables” rule.

### 9. Either finish trust region or mark it experimental

The approximate trust-region path does not currently use a coherent L-BFGS quadratic model. Its Hessian-vector product is merely a scaled identity, while its Newton point uses the inverse L-BFGS recursion; predicted reduction is then reduced to the linear term.

The exact-Hessian path constructs a dense `n × n` matrix even when dispatching to the nominally large-problem Steihaug-CG path.

* [ ] Expose `hessian_vector_product()` as the primary second-order interface.
* [ ] Use sparse Hessian entries or matrix-free HVPs for Steihaug-CG.
* [ ] Implement a mathematically consistent compact L-BFGS Hessian product.
* [ ] Use the same model for step construction and predicted reduction.
* [ ] Add a minimum trust radius and stagnation termination.
* [ ] Handle indefinite Hessians and failed Newton solves explicitly.
* [ ] Report accepted/rejected steps and radius history.
* [ ] Keep `TrustRegion` out of stable public guarantees until these invariants are tested.

## P2: Build a stable geometric modeling API

### 10. Replace raw IDs and panics with typed handles

`Sketch2DBuilder` accepts raw `EntityId`s and discovers their types through internal enum matching. Wrong or unknown entity types panic.

Use types such as:

```rust
PointHandle
LineHandle
CircleHandle
ArcHandle
ConstraintHandle<C>
```

* [ ] Make invalid combinations unrepresentable where practical.
* [ ] Return `Result` for dynamic operations rather than indexing maps and panicking.
* [ ] Introduce a public `BuilderError`.
* [ ] Validate stale generational IDs at API boundaries.
* [ ] Add batch-edit transactions so failed edits can roll back.
* [ ] Define entity/constraint removal semantics.

Removal is particularly important because line entities share point parameters. Removing a point can free parameters that another entity still references. The current system explicitly leaves referring constraints in place when an entity is removed.

* [ ] Track reverse references.
* [ ] Refuse removal while dependents exist, cascade removal explicitly, or use reference-counted parameter ownership.
* [ ] Add integrity validation before every solve in debug mode.

### 11. Complete the sketch domain

The builder’s entity metadata currently supports points, line segments, and circles, despite the lower-level module and documentation referring to arcs.

* [ ] Add `ArcHandle` and builder support for arcs.
* [ ] Add point-on-arc and tangent-to-arc constraints.
* [ ] Add internal as well as external circle tangency.
* [ ] Add point-to-line and line-to-line distance constraints.
* [ ] Add radius and diameter constraints.
* [ ] Add equal-angle and angle-bisector constraints.
* [ ] Add orientation-aware directed angles.
* [ ] Add construction geometry and reference-only entities.
* [ ] Add soft/reference dimensions that report values without constraining them.
* [ ] Add constraint priority or weighting for graceful overconstraint handling.

### 12. Add interactive CAD behavior

This is where Solverang can become distinctive.

* [ ] Continuation/homotopy between the previous solution and a dragged target.
* [ ] Branch continuity so sketches do not unexpectedly flip.
* [ ] Temporary drag constraints with configurable priority.
* [ ] Null-space motion for underconstrained sketches.
* [ ] Stable solution selection based on distance from the previous state.
* [ ] Incremental conflict explanation after each edit.
* [ ] Minimal conflicting constraint sets, not merely broad redundancy groups.
* [ ] Edit transactions, undo/redo-friendly change sets, and deterministic replay.
* [ ] Serialization with stable external IDs independent of arena generations.
* [ ] A small WASM sketch demo that exercises interactive re-solving.

## P3: General numerical capabilities still worth adding

These should follow the correctness work:

* [ ] Residual weighting and variable scaling.
* [ ] Automatic scaling based on Jacobian column norms.
* [ ] Robust least-squares losses: Huber, Cauchy, soft-L1, and Tukey.
* [ ] Per-residual tolerances and units.
* [ ] Iteration callbacks and structured tracing.
* [ ] Cancellation and wall-clock/evaluation budgets.
* [ ] Covariance and rank estimates for least-squares solutions.
* [ ] Better rank-deficiency diagnostics and null-space bases.
* [ ] A `ProblemBuilder` for composing residual functions without manually implementing `Problem`; this remains a valid item from the existing TODO.
* [ ] A detailed result containing each residual’s name, value, scale, and associated constraint.
* [ ] Deterministic solver policies for reproducible applications and tests.

## Repository and release work

The default feature set includes JIT, sparse algebra, parallel execution, macros, and NIST problems, making the default build substantially heavier than necessary. Package metadata is also minimal.

* [ ] Make the default feature set small, probably `std` plus optionally `macros`.
* [ ] Make `jit`, `parallel`, `sparse`, and `nist` opt-in.
* [ ] Decide whether `std` is a real portability boundary. The code currently uses `std` directly in core solver modules, so the empty `std` feature is not presently a meaningful `no_std` design.
* [ ] Add `repository`, `documentation`, `homepage`, `readme`, `keywords`, `categories`, and `rust-version`.
* [ ] Add a changelog and semver policy.
* [ ] Generate API documentation showing stable versus experimental modules.
* [ ] Replace stale `TODO.md`, `STATUS.md`, `NEXT-PLAN.md`, and `OPT-PLAN.md` with a single current roadmap or GitHub milestones.

CI currently runs on Ubuntu stable and checks builds, tests, and documentation, but it does not run formatting, Clippy, feature powersets, MSRV, or cross-platform jobs.

* [ ] Add `cargo fmt --check`.
* [ ] Add Clippy with warnings denied.
* [ ] Add `cargo hack` feature-powerset checks.
* [ ] Add Linux, macOS, and Windows jobs.
* [ ] Add x86-64 and ARM coverage where JIT behavior differs.
* [ ] Establish and test an MSRV.
* [ ] Add `cargo llvm-cov`.
* [ ] Run `cargo mutants` on the solver and reduction logic.
* [ ] Fuzz derivative macros, malformed constraint graphs, stale IDs, and degenerate geometry.
* [ ] Add benchmark regression tracking for dense, sparse, JIT, incremental, and geometric workloads.
* [ ] Run release validation through `cargo publish --dry-run` before tag-triggered publication.

## Recommended implementation sequence

I would organize the next work into four PRs:

1. **Correctness regression suite:** reproduce decomposition cascading, Hessian-objective state, bounded line-search inconsistency, and inequality-only ALM penalty behavior before modifying implementation.
2. **Correctness fixes:** global residual audit, objective ownership cleanup, bounded line search, and full ALM KKT checks.
3. **Core performance cleanup:** active parameter mappings, constant-time ParamId lookup, sparse symbolic reuse, and evaluator abstraction.
4. **Public API stabilization:** typed geometry handles, lifecycle validation, result diagnostics, feature cleanup, CI expansion, and updated documentation.

After those four, Solverang would have a credible foundation for a 0.1 release. Until then, additional algorithms would increase surface area faster than reliability. This review was based on the current repository contents and code paths; I did not execute the test suite locally.
