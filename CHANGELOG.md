# Changelog

## [Unreleased]

### Changed
- **Optimization solvers are instance-based**: `BfgsSolver::new(config).solve(objective, store)`
  replaces the old static `BfgsSolver::solve(objective, store, &config)`; same for `BfgsBSolver`,
  `TrustRegionSolver`, and `AlmSolver`. Each also gains a `solve_with_clock` variant, and
  `ConstraintSystem::optimize_with_clock` threads a host `SolveClock` end-to-end.
- **`OptimizationConfig` split into sub-configs**: line-search, ALM, and trust-region knobs moved
  to `config.line_search`, `config.alm`, and `config.trust_region`; `validate()` rejects
  nonsensical values before a solve.
- **`Sketch2DBuilder` uses typed handles**: `add_point`/`add_circle`/`add_line_segment` return
  `PointHandle`/`CircleHandle`/`LineHandle`; constraint methods take handles and return
  `Result<ConstraintId, BuilderError>` instead of panicking on wrong entity kinds.
- **Removal has a contract**: `ConstraintSystem::remove_entity`/`remove_constraint` return
  `Result<(), RemovalError>` — stale IDs error instead of silently no-opping, and entity removal
  is refused while dependent constraints or param-sharing entities exist.
- `OptimizationStatus::LineSearchFailed` now carries the `LineSearchFailure` (reason and
  evaluation counts); `UnsupportedProblemStructure { reason }` is a `String` and is returned for
  `optimize()` without an objective (previously `Infeasible`).
- `JITConfig::force_jit`/`force_interpreted` replaced by a single `JitMode` enum;
  `JITSolver::last_jit_fallback()` reports why a solve used interpreted evaluation.
- Sparse/dense linear-solve failures return `SolveError::LinearSolveFailed { details }` with the
  failing factorization instead of a generic `SingularJacobian`.
- `SystemResult::total_iterations` renamed to `iterations`; `OptimizationResult` gains
  `is_converged()` / `iterations()` accessors.
- Default features slimmed to `std` + `macros`; `jit`, `sparse`, `parallel`, `nist` are opt-in.

### Removed
- Superseded `graph` clustering layer (`RigidCluster`, `ClusterStatus`, `decompose_clusters`,
  `ConstraintGraph`, `quick_dof`) and other dead code; every `#[allow(dead_code)]` is gone.
- `OptimizationStatus::NotImplemented` (nothing could return it).

### Added
- `Problem::is_square()`; `ParamStore::is_alive()`; `LineSearchConfig` / `AlmConfig` /
  `TrustRegionConfig`; `BuilderError`, `RemovalError`, `JitFallback` types.
- Debug-mode referential integrity sweep before every `ConstraintSystem` solve.
- CI gates: `cargo fmt --check` and `cargo clippy --workspace --all-targets --all-features
  -- -D warnings`; the main test job runs `--all-features`. `#![warn(missing_docs)]` on both
  crates. Publish metadata (`repository`, `keywords`, `categories`, `rust-version`).

## [0.1.0] - 2026-03-21

### Added
- Multiple solver algorithms: Newton-Raphson, Levenberg-Marquardt, AutoSolver, RobustSolver, ParallelSolver, SparseSolver
- JIT compilation via Cranelift for constraint evaluation (`jit` feature)
  - Fused residual + Jacobian evaluation in single native function
  - Direct dense Jacobian assembly (column-major, no COO copy)
  - Compiled Newton steps for small systems (N < 30)
  - Automatic JIT detection in JITSolver::solve()
- `#[auto_jacobian]` proc macro for automatic symbolic differentiation (`macros` feature)
  - Multi-residual support (multiple `#[residual]` methods)
  - JIT opcode lowering generation
- ProblemBuilder API for closure-based problem construction
- Solve failure diagnostics with per-equation residual breakdown
- Problem decomposition into independent sub-problems
- Jacobian verification via finite differences
- V3 ConstraintSystem with Sketch2D/3D builders
- 40+ MINPACK test problems
