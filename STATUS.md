# Solverang -- Repository Status

*Last updated: 2026-07-17*

## Overview

Solverang is a domain-agnostic numerical solver for nonlinear systems and least-squares problems. It has two personalities: a low-level `Problem` trait for raw equation systems, and a high-level V3 constraint system (`ConstraintSystem`, `Sketch2DBuilder`) where you describe entities and constraints and the solver figures out the rest.

**Architecture**: V3 "solver-first" -- the solver core never imports a geometry type. Domain-specific modules (sketch2d, sketch3d, assembly) implement the `Entity` and `Constraint` extension traits.

## Latest Work

**2026-07-17 — DAE/ODE time-integration stack** (`src/integrate/`, the runtime half of
the SINBAD M3 first-transient-result path). New, additive; the pre-existing ~1071
default-feature tests stay green (now 1093 with +22 integrator tests).

- **New dep:** `numeric-contracts` (pkg `sinbad-numeric-contracts`, path-dep to
  `/home/dev/sinbad`), mirroring the existing `malleus` path-dep. solverang consumes the
  `DaeResidual` / `IntegratorCoeffs` seam (types + traits only) to integrate a
  first-order DAE `d/dt q(x,t) + g(x,t) = 0` **without importing the physics-assembly
  crate**. Only `f64` slices cross the boundary, so the crate's transitive `nalgebra`
  0.33 compiles harmlessly alongside solverang's 0.34.
- **Methods:** implicit Euler (BDF-1, L-stable), variable-step **BDF-2** (the index-1
  first-order-DAE workhorse; bootstraps with one Euler step), and **generalized-α**
  (Jansen–Whiting–Hulbert first-order form, `ρ∞`-damped — the `cardan` shape). Each
  forms its per-step nonlinear system from the seam (`residual_at` + charge-difference
  time derivative via `charge`/`mass_apply`) with Newton matrix `iteration_matrix(...)`.
- **Per-step Newton reuse:** a `DaeStepProblem` adapter (in `integrate::step`)
  implements solverang's `Problem` over one stage (`residual_at`→`residuals`,
  `iteration_matrix`→COO `jacobian`) and is solved by the **existing globalized
  `Solver`** (Newton + line search). No bespoke Newton loop.
- **PI (predictive) step controller** (`integrate::controller`): weighted-RMS
  local-error estimate from the predictor/corrector difference, accept/reject/retry,
  Gustafsson PI step adaptation. Structurally algebraic components (zero mass rows) are
  excluded from the error test (*suppress-alg*, à la SUNDIALS IDA) so index-1 DAEs don't
  collapse the step size.
- **Entry point:** `integrate_dae(problem, ctx, t_span, x0, opts) -> Trajectory`
  (time/state history + `IntegrateStatus`). **Panic-free**: every failure is a typed
  `IntegrateError` carried in the status with the partial history preserved.
- **Verified** (`tests/dae_integration.rs`, 17 tests, hand-built `DaeResidual` impls):
  scalar-decay accuracy (fixed + adaptive), stiff-system A-stability at `hλ = 100`,
  order of convergence (BDF-2 ≈ 4×, implicit Euler ≈ 2×, gen-α ≈ 4× for `ρ∞ ∈ {0,½,1}`),
  a linear index-1 DAE with a singular mass row, PI accept/reject, and typed-error paths.
- **Deferred seams** (documented in the module docs + TODO §DAE): dense output, the
  event loop (`EventBearing`), higher-order/order control + Radau IIA / ESDIRK, the full
  DymNL globalization ladder (homotopy/continuation/scaling), and unified factor reuse
  (one factored Jacobian across Newton/adjoint/ROM). Today each Newton iteration
  reassembles + refactorizes, and the error estimate uses an order-1 predictor (so the
  adaptive controller is conservative on the 2nd-order methods — correct, just not yet
  step-optimal).

**2026-07-15 — code-quality and API-design overhaul** (from

**2026-07-15 — code-quality and API-design overhaul** (from
`docs/notes/2026-07-15-code-quality-review.md`; all eight review work items done):

- Deleted the superseded `graph` clustering layer (`RigidCluster`, `decompose_clusters`,
  `ConstraintGraph`) and every `#[allow(dead_code)]`; the workspace now builds and lints
  clean: `cargo clippy --workspace --all-targets --all-features -- -D warnings` passes,
  `#![warn(missing_docs)]` is on with zero gaps, and CI gates fmt + clippy.
- Failure diagnostics are preserved end-to-end: `OptimizationStatus::LineSearchFailed`
  carries the `LineSearchFailure` (reason + eval counts), JIT fallback is reportable via
  `JITSolver::last_jit_fallback()`, sparse factorization failures return
  `SolveError::LinearSolveFailed { details }`, and `optimize()` without an objective is
  `UnsupportedProblemStructure`, not `Infeasible`. `JITConfig`'s force flags became a
  `JitMode` enum.
- The two solver families converged: optimization solvers are instance-based
  (`BfgsSolver::new(config).solve(objective, store)`), accept a host `SolveClock`
  (`optimize_with_clock`), and share the result vocabulary (`SystemResult::iterations`,
  `OptimizationResult::{is_converged, iterations}` accessors). `OptimizationConfig` is
  split into `line_search`/`alm`/`trust_region` sub-configs with `validate()`.
- Shared solver plumbing extracted (`solver/common.rs`): one validation preamble, one
  finiteness guard, one `SolverChoice` dispatch; `Problem::is_square()` added.
- Geometry API error contract: `Sketch2DBuilder` uses typed `PointHandle` /
  `LineHandle` / `CircleHandle` with `Result<_, BuilderError>`; `remove_entity` /
  `remove_constraint` return `RemovalError` (refusing removal with dependent constraints
  or shared params) instead of silently no-opping; a debug integrity sweep runs before
  every solve.
- Default features slimmed to `std` + `macros`; publish metadata added; stale docs fixed.

**2026-07-10 — P0 correctness campaign** (see `docs/notes/2026-07-10-independent-review.md`):
decomposition cascading with global residual certification, coherent objective/Hessian
model, bound-aware L-BFGS-B line search, full ALM KKT checks, explicit line-search
failure reporting, and host-provided solve timing (`SolveClock`).

## Codebase

| Component | Files | Lines |
|-----------|-------|-------|
| solverang library (`crates/solverang/src/`) | ~130 | ~49,400 |
| macros crate (`crates/macros/src/`) | 5 | ~1,400 |
| Integration tests (`crates/solverang/tests/`) | 17 | ~12,600 |
| Benchmarks (`crates/solverang/benches/`) | 3 | ~1,300 |

The full workspace test suite passes with `--all-features`, with default features,
and with `--no-default-features --features std`; the workspace is clippy- and
rustfmt-clean with warnings denied.

## Module Map

### Core (Tier 1)

| Module | What it does |
|--------|-------------|
| `id` | Generational index types: ParamId, EntityId, ConstraintId, ClusterId |
| `param` | ParamStore -- parameter allocation, fixing, get/set; SolverMapping |
| `entity` | Entity trait -- named parameter groups |
| `constraint` | Constraint trait -- residuals + Jacobian entries keyed by ParamId |
| `graph` | Bipartite entity-constraint graph, clustering, decomposition, DOF analysis, redundancy detection |
| `decomposition` | Union-find component extraction for the low-level Problem API |
| `system` | ConstraintSystem -- the top-level orchestrator for V3 |
| `pipeline` | Pluggable solve pipeline: Decompose, Analyze, Reduce, Solve, PostProcess |
| `problem` | Low-level Problem trait (residuals, Jacobian, dimensions) |

### Solving Infrastructure (Tier 2)

| Module | What it does |
|--------|-------------|
| `solve` | ReducedSubProblem adapter, null-space drag, branch selection, closed-form solvers |
| `reduce` | Symbolic reduction: substitute fixed params, merge coincident, eliminate trivials |
| `dataflow` | ChangeTracker + SolutionCache for incremental re-solving with warm starts |
| `solver` | Newton-Raphson, Levenberg-Marquardt, AutoSolver, RobustSolver, ParallelSolver, SparseSolver, JITSolver |
| `jacobian` | Finite-difference Jacobian, verification, sparse patterns, CSR matrices |
| `constraints` | Inequality constraints, slack variable transforms, bounds |

### Domain Plugins (Tier 3)

| Module | What it does |
|--------|-------------|
| `sketch2d` | 2D sketch: Point2D, LineSegment2D, Circle2D, Arc2D + 16 constraint types (incl. SymmetricAboutLine) + Sketch2DBuilder (concentric, tangent_circle_circle, collinear, equal_radius, symmetric_about_line builder methods added) |
| `sketch3d` | 3D sketch: Point3D, LineSegment3D, Plane, Axis3D + 8 constraint types |
| `assembly` | Rigid-body assembly: RigidBody (quaternion orientation) + Mate, Coaxial, Insert, Gear |

### Other

| Module | What it does |
|--------|-------------|
| `test_problems` | 18 MGH least-squares + 14 nonlinear equation problems + NIST StRD (feature-gated) |
| `jit` | Cranelift JIT compilation for constraint evaluation (feature-gated) |

### Macros Crate

`solverang_macros` -- `#[auto_jacobian]` procedural macro for automatic Jacobian generation via symbolic differentiation. Supports arithmetic, trig, sqrt, pow, atan2, chain rule. No control flow.

## Feature Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `std` | yes | Standard library support |
| `macros` | yes | `#[auto_jacobian]` procedural macro |
| `sparse` | no | Sparse matrix support via faer |
| `parallel` | no | Parallel solving via rayon |
| `jit` | no | Cranelift JIT compilation |
| `nist` | no | NIST StRD regression test problems |

## Test Suite

| Category | Files | What they cover |
|----------|-------|-----------------|
| Unit tests (embedded) | — | All modules |
| Solver megatest | 1 | 100-var chains, overdetermined, sparse mega, cross-solver, robustness |
| Property tests (sketch2d) | 1 | Proptest: satisfaction, Jacobian, DOF, decomposition, coordinate invariance |
| Property tests (sketch3d) | 1 | Proptest: 3D constraint properties |
| Property tests (assembly) | 1 | Proptest: rigid body, quaternion, assembly constraints |
| Property tests (general) | 1 | Proptest: solver properties |
| Contract tests | 1 | Design-by-contract: trait compliance for all constraint/entity types |
| Solver comparison | 1 | NR vs LM vs AutoSolver consistency |
| Sparse tests | 1 | Sparse Jacobian, CSR, faer integration |
| LM tests | 1 | Levenberg-Marquardt edge cases |
| Parallel tests | 1 | Decomposition, component independence |
| Macro tests | 1 | `#[auto_jacobian]` symbolic differentiation |
| MINPACK verification | 1 | Reference validation against MINPACK |
| Solver basic tests | 1 | Basic solver functionality |
| Doc-tests | — | Inline examples in lib.rs |

## Benchmarks

| Suite | What it measures |
|-------|-----------------|
| `comprehensive.rs` | NR vs LM vs AutoSolver, sparse vs dense crossover, parallel speedup |
| `scaling.rs` | Solver scaling across problem sizes, decomposition vs monolithic |
| `nist_benchmarks.rs` | MGH problem suite performance (requires `--features nist`) |

## Known Issues

1. **Decomposition cascading** -- fixed (2026-07-10). The pipeline builds a
   one-to-many param -> clusters dependency map from each cluster's constraint
   parameters, marks dependent clusters dirty when a solve changes a shared or
   substituted parameter, and iterates to a fixed point bounded by the cluster
   count. Final residual certification remains as the backstop: `Solved` is
   never reported when a constraint's residual exceeds both the tolerance and
   what its cluster reported (the latter condition preserves legitimate
   least-squares minima on over-determined clusters).

## Documentation

- `docs/plans/solver-first-v3.md` -- V3 architecture blueprint
- `docs/plans/testing/` -- 11 testing strategy documents
- `docs/plans/jit/` -- Three-level JIT implementation plan
- `docs/notes/` -- Research notes on solvers, JIT, differential dataflow, SOTA survey
- `docs/notes/parasolid-kernel-lessons.md` -- production-hardening lessons distilled from
  reverse-engineering Parasolid's numeric core (noise-gating, role-aware linear/angular
  tolerances, damped-Newton-Cramer micro-cluster path, determinism contract, and the
  solver↔kernel evaluation seam for sitting on top of cadabra2)
- `TESTING_STRATEGY.md` -- Comprehensive testing strategy overview
- `lib.rs` -- Crate-level docs with runnable examples

## CI/CD

GitHub Actions workflows:
- `ci.yml` -- build, test, clippy, doc-tests on push/PR
- `release.yml` -- full test suite + crates.io publish on version tags
