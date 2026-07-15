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

**All P0 sections completed 2026-07-10** (uncommitted at time of writing). Full workspace
test suite passes with `--all-features` (1205 tests). Details below retained with notes.

### 1. Decomposition cascading + global solve validation — DONE

- [x] Model dependencies between clusters created by reduction/fixed-parameter substitution
      (`param_to_clusters` map built from each cluster's *constraint* param_ids, not just owned params).
- [x] Propagate dirty state downstream after a cluster changes shared or substituted parameters
      (per-solve parameter snapshot + dependent-cluster marking in `SolvePipeline::run_with_clock`).
- [x] Iterate dependent clusters to a fixed point, bounded by the cluster count.
- [x] `param_to_cluster` changed to one-to-many (`HashMap<ParamId, Vec<ClusterId>>`);
      `ChangeTracker::compute_dirty_clusters` updated accordingly.
- [x] Regression tests: `cross_cluster_cascade_propagates_shared_param_changes`,
      `cross_cluster_cascade_pass_count_is_bounded` (pipeline tests with an overlapping custom decompose).
- [x] Final residual certification refined: a constraint fails certification when its residual
      exceeds tolerance *and* what its (Converged/Skipped) cluster reported — this keeps
      legitimate least-squares minima (Bard, FreudensteinRoth) while still catching stale
      cached/skipped/cascade-masked solutions. Requires `SolvePipeline::constraint_cluster_map()`.

### 2. Repair the objective/Hessian API — DONE

- [x] `ObjectiveModel` enum (FirstOrder/SecondOrder) stores one coherent objective;
      `set_objective_with_hessian()` installs it for both first- and second-order evaluation.
- [x] `set_objective()` replaces the whole model — no stale Hessian can survive.
- [x] `has_objective()` recognizes second-order objectives.
- [x] Tests for replacing objectives in both orders (`set_objective_replaces_hessian_objective`, etc.).
- [x] Structural problem fingerprint (`problem_fingerprint()`): warm-start multipliers are cleared
      when the objective or constraint set changes.

Algorithm-selection validation:

- [x] Compatibility validated in `optimize()` before any solver touches parameters.
- [x] `OptimizationStatus::UnsupportedProblemStructure { reason }` returned for incompatible configs.
- [x] Explicit BFGS/trust region with finite bounds or eq/ineq constraints rejected; explicit
      BFGS-B with eq/ineq constraints rejected (BFGS-B without bounds is allowed — it degenerates
      correctly to BFGS).
- [x] Decision recorded: constrained trust region is **unsupported** (rejected with
      `UnsupportedProblemStructure`); a genuine constrained method is future work.

### 3. Fix the L-BFGS-B line search — DONE

- [x] `bounded_line_search()` + `max_feasible_step()` compute `alpha_max` from the first bound crossing.
- [x] The objective is never evaluated outside the feasible box (test with NaN outside bounds).
- [x] Reported `f` always matches the accepted iterate (debug assertions in `BfgsBSolver`).
- [x] Line-search failure propagates as `OptimizationStatus::LineSearchFailed` instead of a fake step.
- [x] Tests: `-ln(x)+x` (NaN outside box), boundary solutions, multi-active-bound corner solutions.

### 4. Finish the ALM implementation mathematically — DONE

- [x] Zero-free-variable path evaluates feasibility; violated constraints → `Infeasible`.
- [x] ρ updates use combined `sqrt(‖g‖² + ‖max(0,h)‖²)` — inequality-only problems grow ρ.
- [x] Convergence requires primal + stationarity + complementarity (dual feasibility μ ≥ 0 by construction).
- [x] Stationarity computed for the original Lagrangian `∇f + Jgᵀλ + Jhᵀμ` with the updated multipliers.
- [x] Warm-started μ clamped to `[0, max_multiplier]`; λ clamped symmetrically.
- [x] Inner-solver divergence/non-finite values propagate as `Diverged`; inner line-search failure
      feeds the stall counter instead of being ignored.
- [x] Saturated-ρ stagnation → `Infeasible` (materially violated) or `Stalled`; contradictory-constraint
      test asserts this.
- [x] `constraint_violations` includes inequality values (equalities first, then inequalities).
- [x] Mixed bounds + equality + inequality test (`alm_mixed_bounds_equalities_inequalities`).
- [x] Module docs rewritten for the BFGS/BFGS-B inner loop with inequalities.

### 5. Make line-search failure explicit — DONE

- [x] `Result<LineSearchStep, LineSearchFailure>`; `StepCondition::{StrongWolfe, ArmijoOnly}` recorded.
- [x] Zoom fallback retains only Armijo-satisfying candidates.
- [x] `ParamStore` left at the accepted candidate on success, restored to the start point on failure.
- [x] Function/gradient evaluation counts on both success and failure; `line_search_max_evals`
      budget in `OptimizationConfig` (default 100).
- [x] Non-finite objective/gradient detection; Armijo fallback backtracks through non-finite
      regions toward the (finite) start point (required by Osborne 1).
- [x] Safeguarded quadratic interpolation in zoom with bisection fallback.
- [x] BFGS retries once along steepest descent (with L-BFGS memory reset) before reporting
      `LineSearchFailed`.

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
- [x] Expose compilation failures in diagnostics: `JitFallback` + `JITSolver::last_jit_fallback()` report why a solve fell back (forced, no lowering, platform, threshold, compile error) *(2026-07-15)*.
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

### 10. Replace raw IDs and panics with typed handles — DONE 2026-07-15 (except transactions/arcs)

- [x] Typed handles: `PointHandle`, `LineHandle`, `CircleHandle` on `Sketch2DBuilder`; wrong
      entity kinds are now unrepresentable at compile time. (`ArcHandle` waits on §11 arc support;
      `ConstraintHandle<C>` not yet needed — constraints keep plain `ConstraintId`.)
- [x] `Result` + public `BuilderError` instead of panicking (`UnknownEntity` for foreign handles).
- [x] Stale generational IDs validated at API boundaries: `add_entity`/`add_constraint` assert the
      ID came from this system's allocator; `remove_entity`/`remove_constraint` return
      `RemovalError::StaleId` instead of silently no-opping; `ParamStore::is_alive()` added.
- [ ] Add batch-edit transactions so failed edits can roll back.
- [x] Entity/constraint removal semantics defined: `remove_entity` refuses while dependent
      constraints reference the entity (`HasDependentConstraints`) or another entity shares its
      params (`SharedParams`) — no more dangling ParamIds from shared line-segment endpoints.
- [x] Debug-mode integrity validation before every solve
      (`ConstraintSystem::debug_validate_integrity`).

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
- [ ] Per-residual tolerances and units. *(Parasolid RE: role-aware linear vs angular —
  default `1e-8` linear / `1e-11` angular session precision + relative ε-multiples; mixed
  mm/rad residuals under one tolerance is dimensionally wrong. See
  `docs/notes/parasolid-kernel-lessons.md` #2.)*
- [ ] Iteration callbacks and structured tracing.
- [ ] Cancellation and wall-clock/evaluation budgets.
- [ ] Covariance and rank estimates for least-squares solutions.
- [ ] Better rank-deficiency diagnostics and null-space bases.
- [ ] A detailed result containing each residual's name, value, scale, and associated constraint.
- [ ] Deterministic solver policies for reproducible applications and tests. *(Parasolid RE:
  ships scalar-deterministic, non-SIMD math (~178:1) for bit-identical cross-platform solves;
  our `rayon`/`faer` parallel reductions break this. Add a single- vs multi-thread reproducibility
  regression test. Lessons doc #5.)*

**From the Parasolid numeric-core RE** (`docs/notes/parasolid-kernel-lessons.md` — evidence
from `../parasolid-re/`; new items not already covered above):
- [ ] Per-component residual **noise-gating / dead-zone** (floor `|rᵢ| < ε` to 0 before `‖r‖²`) —
  the most-repeated robustness trick in the binary; a few lines in `newton_raphson.rs` + the
  BFGS gradient norm. Lessons doc #1.
- [ ] **Damped-Newton-Cramer micro-cluster fast path** for decomposed `n ≤ 3` square systems:
  direct Cramer step + monotone-residual-decrease acceptance + ≤10 step-halvings (×0.5) +
  determinant singular-guard, cap 20; route tiny clusters here before the general NR/SVD path.
  Pairs with a cheap `ParamStore` checkpoint/rollback around the speculative step. Lessons doc #3, #7.
- [ ] **Typed failure + non-poisonous exit**: extend `OptimizationStatus` with `SingularJacobian`
  / `Diverged` / `MaxItersUnconverged`; a diverged solve must leave params in a documented state,
  never a half-applied step. Lessons doc #4.
- [ ] **Solver↔kernel constraint-evaluation seam** (before wiring cadabra2): geometric constraints
  (`PointOnSurface`, `TangentToSurface`, …) take residual + Jacobian rows from the kernel's surface
  derivatives, and share ONE tolerance model with the kernel; budget for the nested inner
  projection (warm-start it across outer iterations). Lessons doc "solver ↔ kernel contract".

*(The review also listed `ProblemBuilder` here, but it already exists — `problem.rs`, exported from `lib.rs`.)*

---

## Release & repository work

### Packaging / features

- [x] Default features shrunk to `std` + `macros`; `jit`, `parallel`, `sparse`, `nist` opt-in; CI main job runs `--all-features` *(2026-07-15)*.
- [ ] Decide whether `std` is a real portability boundary (core solver modules use `std` directly; the empty `std` feature isn't a meaningful `no_std` design today).
- [x] Package metadata added: `repository`, `readme`, `keywords`, `categories`, `rust-version` (`documentation`/`homepage` default to docs.rs) *(2026-07-15)*.
- [ ] Changelog and semver policy.
- [ ] API docs distinguishing stable vs. experimental modules.
- [x] `pub` vs `pub(crate)` surface audited: solver algorithm modules (`alm`, `bfgs`, `bfgs_b`, `line_search`, `trust_region`) are `pub(crate)` behind curated re-exports; `#![warn(missing_docs)]` enabled on both crates with zero gaps *(2026-07-15)*.
- [ ] `cargo publish --dry-run` in release validation before tag-triggered publication.

### CI

- [x] `cargo fmt --check` + `cargo clippy --workspace --all-targets --all-features -- -D warnings` CI job; workspace is warning-free *(2026-07-15)*.
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
