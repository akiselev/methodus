# Solverang TODO

*Consolidated 2026-07-10 from `OPT-PLAN.md`, `NEXT-PLAN.md`, the previous `TODO.md`, and the
independent review (`NEXT.md`, archived at `docs/notes/2026-07-10-independent-review.md`).
Every "done" claim below was verified against the code, not the plans.*

## Direction

No new algorithms for now. The next phase is **correctness, architectural consolidation,
and a defensible public API**. Product identity: an **interactive geometric constraint and
optimization kernel** that also exposes a clean domain-independent numerical layer — the
combination of geometric system + incremental architecture + symbolic derivatives + JIT is
more distinctive than another general-purpose Rust optimizer.

Suggested PR sequence:
1. **Correctness regression suite** — reproduce the P0 defects as failing tests before touching implementation.
2. **Correctness fixes** — global residual audit, objective ownership, bounded line search, full ALM KKT checks.
3. **Core performance cleanup** — active parameter mappings, sparse symbolic reuse, evaluator abstraction.
4. **Public API stabilization** — typed handles, lifecycle validation, diagnostics, feature cleanup, CI, docs.

---

## P0: Correctness (release blockers for 0.1)

### 1. Decomposition cascading + global solve validation

Highest-priority defect: an earlier cluster's solution may not propagate to dependent
clusters. **Partially addressed** (uncommitted): final residual certification now prevents
`SystemStatus::Solved` when post-solve residuals exceed tolerance
(`DiagnosticIssue::UnsatisfiedConstraints`), with regression tests. Remaining:

- [ ] Model dependencies between clusters created by reduction/fixed-parameter substitution.
- [ ] Propagate dirty state downstream after a cluster changes shared or substituted parameters.
- [ ] Iterate dependent clusters to a fixed point, with a bounded pass count.
- [ ] Change `param_to_cluster` from single-cluster mapping to one-to-many dependency mapping where necessary.
- [ ] Add regression tests reproducing the documented cascading failure (not just the certification backstop).

### 2. Repair the objective/Hessian API

`set_objective_with_hessian()` sets only the Hessian field while `optimize()` requires the
regular objective field; a stale Hessian can survive objective replacement.

- [ ] Make `set_objective_with_hessian()` install one coherent objective for both first- and second-order evaluation.
- [ ] Clear any stale Hessian when `set_objective()` replaces the objective.
- [ ] Make `has_objective()` recognize a second-order objective.
- [ ] Prefer one stored abstraction (e.g. `ObjectiveModel` enum or object exposing optional Hessian-vector products).
- [ ] Add tests for replacing objectives in every possible order.
- [ ] Add a structural generation/fingerprint so warm-start multipliers can't be reused with a different objective.

Algorithm-selection validation (explicitly selecting BFGS/BFGS-B/trust region currently
bypasses registered constraints instead of rejecting the configuration):

- [ ] Validate algorithm/problem compatibility before changing parameters.
- [ ] Return a structured `UnsupportedProblemStructure` error instead of silently solving a different problem.
- [ ] Validate that BFGS-B is selected only where bounds are meaningful.
- [ ] Decide whether constrained trust region is unsupported or implement a genuine constrained method later.

### 3. Fix the L-BFGS-B line search

The bounded solver calls the unconstrained line search, which may evaluate outside the box
(`α > 1`), then projects the point but keeps the unprojected objective value.

- [ ] Implement a bound-aware projected line search; compute `alpha_max` from the first bound along the direction.
- [ ] Never evaluate the objective outside the feasible box.
- [ ] Recompute the objective after any projection; assert reported `f == objective.value(store)` at every accepted iterate.
- [ ] Return a line-search failure rather than treating the minimum step as success.
- [ ] Test objectives undefined outside bounds (e.g. `ln(x)`, `x > 0`); test boundary/corner solutions and multiple simultaneously active bounds.

### 4. Finish the ALM implementation mathematically

- [ ] Evaluate feasibility before returning from the zero-free-variable path (currently reports converged without checking constraints on fixed params).
- [ ] Base penalty (ρ) updates on combined equality + positive inequality violation (inequality-only problems currently never grow ρ).
- [ ] Check all KKT components: primal feasibility, stationarity, dual feasibility, and complementarity.
- [ ] Compute stationarity for the original Lagrangian `∇f + Jgᵀλ + Jhᵀμ`, not the gradient of the penalized subproblem.
- [ ] Require `μ ≥ 0` with explicit dual-feasibility diagnostics; clamp/validate warm-started inequality multipliers.
- [ ] Propagate inner BFGS/BFGS-B failures instead of continuing multiplier updates blindly.
- [ ] Detect infeasibility or stalled penalty progression; contradictory-constraint tests must yield `Infeasible`/`Stalled`, not generic max-iterations.
- [ ] Include inequality values in `constraint_violations`.
- [ ] Add mixed bounds + equalities + inequalities tests.
- [ ] Update ALM module docs (still describe an equality-only LM-based implementation; code now uses BFGS/BFGS-B with inequalities).

### 5. Make line-search failure explicit

- [ ] Return `Result<LineSearchStep, LineSearchFailure>`; record whether the step satisfied Wolfe, Armijo only, or neither.
- [ ] Only retain zoom-fallback candidates that satisfy Armijo (currently tracks lowest f regardless).
- [ ] Restore `ParamStore` to the returned candidate before exiting.
- [ ] Add function/gradient evaluation counts and a configurable evaluation budget.
- [ ] Detect non-finite objective and gradient values.
- [ ] Add interpolation to zoom after correctness is established (bisection stays as fallback).

---

## P1: Consolidate solver architecture

### 6. Use the correct active parameter set

Optimizers build mappings from every free parameter in the `ParamStore` even when the
problem references a subset; `build_solver_mapping_for()` exists but isn't used consistently.

- [ ] Construct the active set as the union of objective, equality, inequality, and bound-relevant parameter IDs.
- [ ] Use `SolverMapping::param_to_col` instead of repeated `param_ids.iter().position(...)` (removes `O(nnz × n)` lookups).
- [ ] Pass the mapping through gradient, Jacobian, Hessian, and multiplier assembly.
- [ ] Reject derivative entries referencing undeclared parameters; add contract tests that `param_ids()` describes every derivative entry.

### 7. Turn JIT into an evaluator backend, not a separate solver

`JITSolver` is its own plain Newton solver, so enabling JIT changes both evaluation
mechanism and numerical algorithm. (Auto-detection, fused eval, and direct dense output are
already done — see "Done" below.)

- [ ] Define an evaluator interface for residuals, Jacobians, and fused evaluation.
- [ ] Implement interpreted, finite-difference, symbolic, and JIT evaluators.
- [ ] Let Newton, LM, robust, sparse, and future solvers consume any evaluator, with identical convergence/fallback behavior across backends.
- [ ] Cache compiled functions by a stable structural fingerprint.
- [ ] Expose compilation failures in diagnostics, with configurable fallback.
- [ ] Calibrate `jit_threshold` (still hardcoded 1000/500) empirically; consider two-tier thresholds for one-shot vs. repeated interactive solves.
- [ ] Benchmark compile cost, evaluation cost, total solve time, and cache reuse separately.

### 8. Make sparse caching real

`SparseSolver` records a sparsity pattern but reconstructs CSR, faer triplets, and a fresh
decomposition every nonlinear iteration.

- [ ] Cache the mapping from Jacobian entries to numeric matrix storage.
- [ ] Reuse symbolic LU/QR analysis and ordering; only update numeric values; invalidate on structural change.
- [ ] Support sparse minimum-norm solutions for underdetermined systems.
- [ ] Add sparse LM or sparse Gauss-Newton (large least-squares shouldn't go through sparse Newton).
- [ ] Verify whether `linear_tolerance` affects any factorization path; remove or implement it.
- [ ] Benchmark the dense/sparse crossover instead of documenting a fixed "1000+ variables" rule.

### 9. Either finish trust region or mark it experimental

The approximate path's HVP is a scaled identity while its Newton point uses the inverse
L-BFGS recursion; predicted reduction reduces to the linear term. The exact path builds a
dense `n × n` Hessian even on the Steihaug-CG (large-n) path.

- [ ] Expose `hessian_vector_product()` as the primary second-order interface.
- [ ] Use sparse Hessian entries or matrix-free HVPs for Steihaug-CG.
- [ ] Implement a mathematically consistent compact L-BFGS Hessian product; use the same model for step construction and predicted reduction.
- [ ] Add a minimum trust radius and stagnation termination.
- [ ] Handle indefinite Hessians and failed Newton solves explicitly.
- [ ] Report accepted/rejected steps and radius history.
- [ ] Keep `TrustRegion` out of stable public guarantees until these invariants are tested.

---

## P2: Stable geometric modeling API

### 10. Replace raw IDs and panics with typed handles

`Sketch2DBuilder` takes raw `EntityId`s and panics on wrong/unknown entity types.

- [ ] Typed handles: `PointHandle`, `LineHandle`, `CircleHandle`, `ArcHandle`, `ConstraintHandle<C>`; make invalid combinations unrepresentable where practical.
- [ ] Return `Result` with a public `BuilderError` instead of panicking.
- [ ] Validate stale generational IDs at API boundaries.
- [ ] Add batch-edit transactions so failed edits can roll back.
- [ ] Define entity/constraint removal semantics — lines share point parameters; removal currently leaves referring constraints in place:
  - [ ] Track reverse references.
  - [ ] Refuse removal while dependents exist, cascade explicitly, or use reference-counted parameter ownership.
  - [ ] Add integrity validation before every solve in debug mode.

### 11. Complete the sketch domain

Builder metadata supports points, lines, circles — but not arcs, despite the lower-level module supporting them.

- [ ] `ArcHandle` and builder support for arcs; point-on-arc and tangent-to-arc constraints.
- [ ] Internal as well as external circle tangency.
- [ ] Point-to-line and line-to-line distance constraints.
- [ ] Radius and diameter constraints.
- [ ] Equal-angle and angle-bisector constraints; orientation-aware directed angles.
- [ ] Construction geometry and reference-only entities.
- [ ] Soft/reference dimensions that report values without constraining.
- [ ] Constraint priority or weighting for graceful overconstraint handling.

### 12. Interactive CAD behavior (the distinctive part)

- [ ] Continuation/homotopy between the previous solution and a dragged target; branch continuity so sketches don't flip.
- [ ] Temporary drag constraints with configurable priority.
- [ ] Null-space motion for underconstrained sketches.
- [ ] Stable solution selection based on distance from the previous state.
- [ ] Incremental conflict explanation after each edit; minimal conflicting constraint sets (not just broad redundancy groups).
- [ ] Edit transactions, undo/redo-friendly change sets, deterministic replay.
- [ ] Serialization with stable external IDs independent of arena generations.
- [ ] A small WASM sketch demo exercising interactive re-solving.

---

## P3: General numerical capabilities (after correctness work)

- [ ] Residual weighting and variable scaling; automatic scaling from Jacobian column norms.
- [ ] Robust least-squares losses: Huber, Cauchy, soft-L1, Tukey.
- [ ] Per-residual tolerances and units.
- [ ] Iteration callbacks and structured tracing.
- [ ] Cancellation and wall-clock/evaluation budgets.
- [ ] Covariance and rank estimates for least-squares solutions.
- [ ] Better rank-deficiency diagnostics and null-space bases.
- [ ] A detailed result containing each residual's name, value, scale, and associated constraint.
- [ ] Deterministic solver policies for reproducible applications and tests.

*(The review also listed `ProblemBuilder` here, but it already exists — `problem.rs`, exported from `lib.rs`.)*

---

## Release & repository work

### Packaging / features

- [ ] Shrink the default feature set (probably `std` + optionally `macros`); make `jit`, `parallel`, `sparse`, `nist` opt-in.
- [ ] Decide whether `std` is a real portability boundary (core solver modules use `std` directly; the empty `std` feature isn't a meaningful `no_std` design today).
- [ ] Add `repository`, `documentation`, `homepage`, `readme`, `keywords`, `categories`, `rust-version` to package metadata.
- [ ] Changelog and semver policy.
- [ ] API docs distinguishing stable vs. experimental modules.
- [ ] Audit `pub` vs `pub(crate)` surface (beyond the already-hidden `__jit_reexports`).
- [ ] `cargo publish --dry-run` in release validation before tag-triggered publication.

### CI

- [ ] `cargo fmt --check`; Clippy with warnings denied.
- [ ] `cargo hack` feature-powerset checks.
- [ ] Linux, macOS, Windows jobs; x86-64 and ARM coverage where JIT behavior differs.
- [ ] Establish and test an MSRV.
- [ ] `cargo llvm-cov` coverage; `cargo mutants` on solver and reduction logic.
- [ ] Fuzz derivative macros, malformed constraint graphs, stale IDs, degenerate geometry.
- [ ] Benchmark regression tracking for dense, sparse, JIT, incremental, and geometric workloads.

### Documentation

- [ ] End-to-end tutorial (headline demo): define a physics equation, annotate with `#[auto_jacobian]`, JIT-compile, call from an autorouter loop.
- [ ] Keep crate-level docs on the two paths (`Problem` trait vs `ConstraintSystem`) in sync as APIs stabilize.

---

## Downstream (lives in `altium-cli-simplified`, not this repo)

Carried from NEXT-PLAN M8–M11; blocked on solverang 0.1 stability:

- [ ] DRC repair via solverang (`autopcb-router/src/drc/repair.rs` stub).
- [ ] Rubber-banding with clearance constraints (`autopcb-router/src/optimize/rubber_band.rs`).
- [ ] Spec language `minimize` / `subject_to` blocks + sensitivity report (altium-format-spec).
- [ ] Channel MCP server (`autopcb-channel`) for agent feedback loops.

---

## Done (verified in code, 2026-07-10)

Recorded so the deleted plan files aren't missed.

**Phase 1 optimization (NEXT-PLAN M1–M7, M12)** — `optimization/` module (`Objective`,
`ObjectiveHessian`, `InequalityFn`, `MultiplierStore`, `OptimizationConfig`,
`OptimizationResult`), `ConstraintSystem::optimize()` with classify/dispatch (inline in
`system.rs` rather than separate pipeline phases), L-BFGS solver, ALM solver,
`#[auto_diff]` / `#[objective]` / `#[inequality]` macros, `LeastSquaresObjective` adapter,
MINPACK optimization tests, optimization README/CLAUDE.md.

**Optimizer enhancements (OPT-PLAN M1–M8)** — asin/acos/sinh/cosh/tanh in the macro expr
engine; strong Wolfe line search with Armijo fallback (`solver/line_search.rs`);
`relative_tolerance` convergence scaling; `ParamStore` bounds + L-BFGS-B
(`solver/bfgs_b.rs`); ALM inequality constraints with μ multipliers and complementarity;
opt-in `#[hessian]` generation; trust-region solver with dogleg + Steihaug-CG
(`solver/trust_region.rs`); solver docs. *(P0 items above track known defects in several of
these.)*

**Old TODO JIT/ergonomics items** — multi-residual `#[auto_jacobian]`; JIT auto-detection
via `lower_to_compiled_constraints()` in `JITSolver`; fused residual+Jacobian
(`evaluate_both_dense`); direct dense Jacobian assembly (dense-offset rewrite in
`jit/opcodes.rs`); dead `StoreJacobian` opcode removed (only `StoreJacobianIndexed`
remains); compiled Newton steps (`jit/compiled_newton.rs`); `ProblemBuilder`; solve failure
diagnostics via `DiagnosticIssue::UnsatisfiedConstraints` + final residual certification
(uncommitted at time of writing). Still open from that list: JIT threshold calibration
(P1 §7) and the tutorial (Documentation).
