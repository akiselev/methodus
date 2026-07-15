# Code Quality & API Design Review — 2026-07-15

> **Status update (same day):** all eight items in the action list below were implemented
> on 2026-07-15. See `CHANGELOG.md` (Unreleased), `STATUS.md`, and the checked-off entries
> in `TODO.md` for what changed. This document is kept as the point-in-time review.

**Scope:** entire workspace (~73k lines of Rust: `crates/solverang` ~49k src + ~13k tests, `crates/macros` ~1.4k).
**Method:** a 25-category code-smell sweep (naming, types, structure, error handling, duplication, module boundaries, docs, tests), a manual read of the public API surface (`Problem`, `ConstraintSystem`, `Sketch2DBuilder`, `Entity`/`Constraint`, configs, results), and objective tooling signals (`cargo clippy`, `cargo fmt --check`, CI config). Findings below were verified against the code; agent-reported claims that didn't survive verification were dropped or corrected.
**Relationship to prior review:** complements the correctness-focused [2026-07-10 independent review](2026-07-10-independent-review.md). Its P0 items are marked done in `TODO.md` (verified claims); this review does not re-audit them. It focuses on what the previous review touched only lightly: code quality, consistency, and API design.

## Verdict

This is a well-built codebase for its age (first commit 2026-01-25, 73 commits) — honestly, well above the norm for a solo project this broad. The core "solver-first" design is genuinely good and consistently executed: the solver core imports no geometry types, the `Entity`/`Constraint` extension traits are minimal and documented down to *why* things are excluded from them, and the test culture (property tests, MINPACK/NIST oracles, contract tests, error-path tests) is real. The 2026-07-10 review was acted on within days, with regression tests first — that responsiveness is itself a quality signal.

The honest criticism is structural, and most of it traces to one root cause: **the optimization stack was grown alongside the root-finding stack without ever deciding they were the same product.** Nearly every inconsistency below — two solve-entry idioms, two result idioms, three iteration-count vocabularies, seven config structs, a clock abstraction wired into one world but not the other — is a symptom of that. Secondary themes: the geometry-facing API still has no error contract (panics and silent no-ops), rich failure diagnostics get built at the leaves and thrown away one caller up, a superseded graph layer ships as dead code, and there is no mechanical hygiene enforcement (clippy currently *fails* the workspace; five files fail `rustfmt`).

None of this is crisis-level. All of it is the right agenda for the "public API stabilization" phase that `TODO.md` already names as PR #4.

---

## What's genuinely good

Worth stating explicitly, because an honest review cuts both ways:

- **The extension-trait core is exemplary.** `constraint/mod.rs` and `entity/mod.rs` are small, complete, and document their *negative space* ("What's NOT on this trait") — the single best documentation habit in the repo. Jacobians keyed by `ParamId` rather than column index is the right decoupling, and the module docs explain why.
- **Generational IDs everywhere** (`ParamId`, `EntityId`, `ConstraintId`, `ClusterId`) with free-list reuse — the correct foundation for an interactive kernel with entity removal.
- **Test discipline.** 1205 tests across unit, property (proptest), oracle (MINPACK, NIST StRD), contract, error-path, and macro UI tests (`trybuild`). Recent bug fixes landed with named regression tests (`cross_cluster_cascade_*`). `TESTING_STRATEGY.md` exists and is mostly followed.
- **Zero TODO/FIXME/HACK comments** in source. Plan files carry the plans; code doesn't rot with stale markers.
- **Low panic density:** 99 `unwrap`/`expect`/`panic!` across ~50k src lines, many inside `#[cfg(test)]` modules. For numerical Rust, that's clean.
- **Documentation volume and CI:** ~3.8k doc-comment lines over 583 public functions; CI builds docs with `-D warnings` and runs doc-tests, plus a minimal-features build job. `CHANGELOG.md` is maintained. Per-module `CLAUDE.md` index files are current.
- **Honest planning docs:** `TODO.md` states "Every 'done' claim below was verified against the code, not the plans" — and spot-checks bear it out.

---

## Findings

### 1. Two solver worlds that don't know about each other — the central API-design problem

**Severity: high (API design) · the root cause behind findings 1a–1e**

The root-finding stack and the optimization stack are two dialects:

| Seam | Root-finding world | Optimization world |
|---|---|---|
| Entry point | `Solver::solve(&self, problem: &P, x0: &[f64])` — instance method, config held by the solver | `BfgsSolver::solve(objective, &mut ParamStore, &OptimizationConfig)` — static fn, state threaded through params |
| Result | `SolveResult` enum, consumed via accessors (`solution()`, `iterations()`) | `OptimizationResult` struct with public fields + a separate `OptimizationStatus` enum |
| Iterations | `iterations` | `outer_iterations` + `inner_iterations`; `SystemResult` adds a third name, `total_iterations` |
| Tolerances | `tolerance` (NR), `ftol`/`xtol`/`gtol` (LM) | `outer_tolerance`/`inner_tolerance`/`dual_tolerance` |
| Time | `SolveClock` injected (`solve_with_clock`) | `Instant::now()` hardcoded in all five solvers (`bfgs.rs:35`, `bfgs_b.rs:41`, `trust_region.rs:37,162`, `alm.rs:174`) |
| Polymorphism | `AutoSolver`/`RobustSolver` can swap NR↔LM | impossible — `AlmSolver::solve` takes different parameters than the others |

Specific consequences:

- **1a.** `ConstraintSystem::optimize()` cannot be made deterministic the way `solve_with_clock()` can — the `SolveClock` abstraction (built precisely for host-provided timing) stops at the optimization boundary. There is no `optimize_with_clock()`.
- **1b.** Solver-selection logic is re-implemented: `m == n` / `m > n` raw comparisons appear in `auto.rs:181-190,261-268`, `parallel.rs:323,413`, `sparse_solver.rs:315,401`, while `Problem::is_least_squares()` exists and is **never called**. The `SolverChoice` dispatch match is duplicated between `auto.rs` and two places in `parallel.rs`.
- **1c.** `OptimizationConfig` is a 21-field grab-bag mixing ALM penalties (`rho_*`), L-BFGS memory, Wolfe constants, trust-region radii, and line-search budgets. Which fields matter depends entirely on `algorithm`, and nothing validates them (negative tolerances, `wolfe_c2 < armijo_c1`, etc. are representable). Meanwhile the workspace has **seven** config structs total (`SolverConfig`, `LMConfig`, `ParallelSolverConfig`, `SparseSolverConfig`, `OptimizationConfig`, `JITConfig`, `SystemConfig`) with overlapping-but-differently-named knobs.
- **1d.** The two Jacobian triplet shapes — `Vec<(usize, usize, f64)>` on `Problem`, `Vec<(usize, ParamId, f64)>` on `Constraint` — are each fine in isolation, but there's no named type or alias for either, and the raw-tuple shape crosses the crate's most central trait boundary at 100+ call sites.
- **1e.** Naming: the plain name `Solver` is bound to Newton-Raphson — the *most fragile* algorithm gets the most generic name. The README even jokes about it ("fast and fragile"). Newcomers reaching for "the solver" get the one that needs the best initial guess. Pre-0.1 is the only cheap time to rename (`NewtonRaphson` or `NewtonSolver`, keep `Solver` as a deprecated alias).

**Recommendation.** This doesn't require merging the two stacks algorithmically. It requires one decision — "these are one product family" — and then mechanical alignment: pick the instance-based entry idiom for both worlds; share the iteration/tolerance vocabulary; give `OptimizationResult` the same accessor discipline as `SolveResult` (or make both plain structs + status — either, but one); thread `SolveClock` through `optimize()`; split `OptimizationConfig` into per-algorithm sub-structs (or validate on use). Most of finding 5's duplication falls out of this for free. TODO items 6/7 (active parameter sets, evaluator abstraction) will be much cheaper after this convergence than before.

### 2. Diagnostics are built at the leaves and discarded one level up

**Severity: high (quality/debuggability)**

The §5 line-search work (commit `0703cb0`) built exactly the right thing: `Result<LineSearchStep, LineSearchFailure>` with the satisfied condition, failure reason, and evaluation counts. Then its callers throw it away:

- `bfgs.rs:124` and `bfgs_b.rs:196` match `Err(_)` and report a bare `OptimizationStatus::LineSearchFailed` — reason and eval counts gone. The five-variant `LineSearchFailure` never reaches a user.
- `jit_solver.rs:117`: JIT compilation errors are swallowed with `Err(_) => { /* fall through to interpreted */ }`. A user cannot tell whether they got a compiled or interpreted solve, nor why compilation failed. (The silent fallback also interacts badly with the `force_jit`/`force_interpreted` bool pair in `JITConfig` — two mutually exclusive flags whose `(true, true)` state is resolved only by check ordering in `jit_solver.rs:55-78`; that's an enum, `JitMode { Auto, ForceJit, ForceInterpreted }`.)
- `sparse_solver.rs:311,328,368`: matrix construction and LU/QR factorization failures collapse to `None` or silently reroute (LU → QR), with the actual faer error discarded.
- `system.rs:794`: `optimize()` with **no objective set** returns `OptimizationStatus::Infeasible` — semantically wrong (nothing is infeasible; the problem is malformed) and it aliases a status that elsewhere means "your constraints genuinely contradict each other." `UnsupportedProblemStructure { reason }` already exists and fits.
- There is no tracing/logging facility anywhere in the crate, so none of this can be observed even in debug runs.

**Recommendation.** Make statuses carry their leaf payload (`LineSearchFailed(LineSearchFailure)` — it's `Clone`-able data, not an excuse for `Box<dyn Error>`); return the no-objective case as `UnsupportedProblemStructure`; record "JIT fell back: <reason>" in the result or a diagnostics vec; consider an optional `tracing` feature. This is mostly plumbing, and it multiplies the value of diagnostic work already done.

### 3. The geometry-facing API has no error contract

**Severity: high (API design) · acknowledged as TODO P2 item 10 — this review adds specifics**

- `Sketch2DBuilder` panics on type mismatches (`builder.rs:94-113`: `panic!("Entity {:?} is not a Point2D")`) and on unknown IDs (raw `HashMap` index). Every `constrain_*` method takes untyped `EntityId`s, so `constrain_parallel(point, circle)` compiles and dies at runtime.
- The lower-level constraint constructors amplify this: the worst clippy offenders are **15-parameter constructors** in `sketch3d/constraints.rs:598,755,897` taking long positional `ParamId` runs — swap two of them and you get wrong geometry that compiles cleanly. Typed handles (`PointHandle`, `LineHandle`, …) fix both layers at once.
- `ConstraintSystem::add_entity`/`add_constraint` (`system.rs:316-327,395-406`) trust the caller-provided ID and resize storage to fit it — a fabricated ID silently creates phantom `None` slots. `remove_entity`/`remove_constraint` silently no-op on stale generations (no error, no bool). `remove_entity` frees params that surviving entities may still reference (line segments share endpoint params), and constraints referencing the removed entity are left in place — both documented, neither guarded.
- `ParamStore::free`/`set`/etc. panic via `expect("free: invalid ParamId")` (`param/store.rs:76-116`) — public methods on the most-central store.

**Recommendation.** The TODO's plan (typed handles + `BuilderError` + reverse-reference tracking) is right. Two cheap interim guards: return `bool`/`Result` from the removal paths instead of silently no-opping, and add a debug-mode integrity check before solve (every constraint's `param_ids()` alive, every entity's params alive).

### 4. A superseded graph layer ships as dead code

**Severity: medium (quality)**

`graph::cluster::{RigidCluster, ClusterStatus}`, `graph::decompose::decompose_clusters`, `graph::bipartite::ConstraintGraph`, `graph::dof::quick_dof`, and `reduce::apply_eliminations` are used **nowhere** outside their own modules (verified by grep and by clippy's unused-import warnings on the `graph/mod.rs` and `reduce/mod.rs` re-exports). The pipeline's decompose phase replaced them. Meanwhile 19 `#[allow(dead_code)]` attributes — mostly on `sketch3d`/`assembly` constraint helpers — suppress the compiler's ability to report exactly this kind of rot.

Side effect: the dead `ClusterStatus` (Fresh/Dirty/Solved/Failed) sits one confusing name away from the live `ClusterSolveStatus` (Converged/NotConverged/Skipped) in `system.rs:107`. Deleting the dead enum dissolves the naming problem.

**Recommendation.** Delete the superseded layer (it's in git history if ever needed), then audit each `#[allow(dead_code)]`: delete what's dead, un-allow what's actually used, and keep only the handful with a written justification.

### 5. Copy-paste solver plumbing

**Severity: medium (quality)**

- The validation preamble (`n == 0` → `NoVariables`, `m == 0` → `NoEquations`, `x0.len() != n` → `DimensionMismatch`) is duplicated across six solver entry points (`newton_raphson.rs`, `levenberg_marquardt.rs`, `sparse_solver.rs`, `jit_solver.rs`, twice in `parallel.rs`).
- Non-finite guards (`.iter().any(|v| !v.is_finite())` → `NonFiniteResiduals`/`NonFiniteJacobian`) appear at ~37 sites in 5+ files.
- The projected-gradient computation in `bfgs_b.rs:253-278` is duplicated verbatim for `pg_new`/`pg_old` (also the source of clippy's identical-branches warnings — the math is correct; the duplication is the smell).

**Recommendation.** One `validate_problem(problem, x0) -> Result<(), SolveError>` helper plus `ensure_finite(...)` helpers; a `project_gradient(x, g, lower, upper)` function. Best done *after* finding 1's entry-point decision so the helpers land in their final home.

### 6. Public API surface is wider than intended

**Severity: medium (API design)**

- `solver/mod.rs:51-85` declares `alm`, `bfgs`, `bfgs_b`, `line_search`, `trust_region` as `pub mod` — internal algorithms, their static `solve` functions, and shared helpers (`dot`, `vec_norm`, `dense_gradient`, `write_x_to_store`) are all reachable and appear in rustdoc, with no stability intent. Compare `lib.rs`, which correctly makes `graph`, `reduce`, `dataflow` `pub(crate)` — the discipline exists, it just wasn't applied inside `solver/`.
- There is no `#![warn(missing_docs)]`, so surface growth is invisible.
- Default features are heavyweight: `default = ["std", "macros", "jit", "nist", "sparse", "parallel"]` means every downstream user compiles Cranelift, faer, rayon, **and the NIST regression test problems**. `nist` in particular is test fixture data shipping as default product. (Flagged in the 07-10 review; still current.)
- `Cargo.toml` still lacks `repository`, `keywords`, `categories`, `readme`, `rust-version` — blocking a credible crates.io page.

**Recommendation.** `pub(crate)` the algorithm modules and re-export only intended types from `solver/mod.rs`; add `#![warn(missing_docs)]`; slim defaults to `["std", "macros"]` (or at minimum drop `nist`); fill in package metadata.

### 7. Hygiene exists but isn't enforced — and has started to drift

**Severity: medium (quality) · cheapest fixes in this review**

Current, verified state:

- **`cargo clippy --workspace --all-targets` fails.** `crates/macros/src/codegen_opcodes.rs:345` uses `Expr::Const(3.14)` in a test, tripping the deny-by-default `approx_constant` lint. One-character fix (use `2.5`); until then no clippy gate can be added.
- **~55 clippy warnings** on the main lib: 20× too-many-arguments (up to 15/7 — see finding 3), 8× unused imports (all from finding 4's dead layer), 4× very-complex-type, `ptHp`/`gtP` non-snake-case, 3× identical-if-blocks (benign, see finding 5), assorted `clone`-on-`Copy` in macros.
- **`cargo fmt --check` fails on 5 files** (15 hunks): `pipeline/mod.rs`, `sketch2d/builder.rs`, `solver/alm.rs`, `solver/bfgs_b.rs`, `solver/line_search.rs` — i.e., the recent P0 work landed unformatted.
- **CI** (`ci.yml`) runs build, tests, minimal-features build, and docs with `-D warnings` — good bones — but no fmt gate, no clippy gate, no feature-powerset, no MSRV.

**Recommendation, in order:** fix the `3.14` constant → `cargo fmt` the five files → burn down the warning list (finding 4's deletion removes ~10 of them) → add `cargo fmt --check` and `cargo clippy -- -D warnings` jobs to CI so the ratchet holds.

### 8. Documentation drift

**Severity: medium (quality)**

- `TESTING_STRATEGY.md:38-50` references a **non-existent `geometry` feature** (`cargo mutants --features geometry,parallel,sparse`) and a **non-existent `src/geometry/constraints/` path** — the modules are `sketch2d`/`sketch3d`/`assembly`. Anyone following the doc gets a cargo error. The same stale feature name recurs in several `docs/plans/` files.
- `STATUS.md` says "Last updated 2026-05-03" and predates the entire P0 correctness campaign — the repo's headline status file describes a two-month-old codebase. (It's also currently modified in the working tree; worth finishing that update.)
- The `system.rs:15-22` module example is ` ```ignore `-fenced and wrong as written (uses an undeclared `entity_id`; `alloc_param` before any entity exists inverts the documented order). Ignored examples are exactly where errors hide — this one would be better as a compiling example built on `alloc_entity_id()`.
- `README.md` and `lib.rs` duplicate substantial content (solver table, quick-start, feature table) with no single source of truth; they will drift.

**Recommendation.** Fix `TESTING_STRATEGY.md` mechanically; refresh `STATUS.md` (or fold it into README + CHANGELOG and delete it, per the 07-10 review's "single current roadmap" advice); convert `ignore` doc examples to compiling ones where feasible.

---

## Minor observations (no dedicated section warranted)

- `dataflow/tracker.rs:41`: `structural_change: bool` duplicates state derivable from the added/removed vecs — derive it.
- `solve_from_initial(&self, problem, factor: f64)` next to `solve(&self, problem, x0: &[f64])` (`newton_raphson.rs:170`) reads as two unrelated concepts sharing a prefix; a doc line clarifying "factor scales `Problem::initial_point`" would do.
- Nine `filter().map()` chains that could be `filter_map` — purely stylistic; fine to leave.
- Test-problem structs in `tests/` use `//` comments instead of `///` docs — trivial, only worth fixing opportunistically.
- `Constraint::weight()`/`is_soft()` defaults exist but no soft-constraint machinery consumes them yet — aspirational API; fine pre-0.1, but they should either gain semantics or a doc note saying they're reserved.

## What this review did *not* find

Also worth recording: no god-objects (the largest file, `system.rs` at 1750 lines, is a legitimately central orchestrator with clear section structure); no circular module dependencies; no feature-flag sprawl (`#[cfg]` usage is disciplined — dual-impl pattern used consistently); no unsafe code outside the JIT boundary where it's inherent; no stale test suites (tests were extended alongside every recent fix). Feature hygiene in particular came back *cleaner* than expected from a codebase with five optional subsystems.

## Limits

Static review plus lints only — the test suite was not executed here (TODO.md records 1205 passing with `--all-features` as of 2026-07-10). Mathematical correctness of the algorithms was in scope for the 2026-07-10 review, not this one. Smell findings were machine-assisted and individually spot-verified; line numbers refer to the working tree as of this date (including uncommitted changes).

---

## Prioritized action list

Ordered by leverage per effort; items 1–3 are near-mechanical and could land this week.

| # | Action | Findings | Complexity |
|---|--------|----------|------------|
| 1 | Fix `3.14` → clippy runs again; `cargo fmt` the 5 files; add fmt+clippy CI gates | 7 | low |
| 2 | Delete the superseded `graph` clustering layer + audit 19 `allow(dead_code)` | 4 | low |
| 3 | Fix `TESTING_STRATEGY.md` feature/paths; refresh `STATUS.md`; fix `system.rs` doc example | 8 | low |
| 4 | Status payloads: `LineSearchFailed(LineSearchFailure)`, JIT-fallback visibility, no-objective → `UnsupportedProblemStructure` | 2 | medium |
| 5 | `pub(crate)` solver internals; `#![warn(missing_docs)]`; slim default features; Cargo metadata | 6 | medium |
| 6 | Shared solver validation/classification helpers; use `is_least_squares()`; dedupe `SolverChoice` dispatch | 5 | medium |
| 7 | **Converge the two solver worlds**: one entry idiom, one result vocabulary, clock through `optimize()`, per-algorithm config split | 1 | high |
| 8 | Geometry API error contract: typed handles, `BuilderError`, removal semantics (existing TODO P2 item 10 — with this review's specifics) | 3 | high |

Item 7 is the strategic one: it should be decided (even if executed incrementally) before TODO items 6/7 (active parameter sets, evaluator abstraction), because both of those get materially cheaper once there is a single solver-entry shape to build against.
