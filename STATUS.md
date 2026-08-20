# Solverang status

Updated: 2026-08-20
Branch: `master`
Milestone: single-crate numerical core

## Current role

Solverang owns physics-neutral numerical contracts and algorithms. It operates on flat `f64` slices and explicit operator actions. It must not understand `.res`, dimensions or units, fields, materials, function spaces, meshes, element kernels, or product/runtime policy.

The repository is one root package named `solverang`. There are no subordinate contracts, scientific, or macro packages.

## Implemented surface

- In-place `LinearOperator`, `Preconditioner`, `NonlinearOperator`, and `DaeOperator` traits.
- `EvaluationContext` for explicit reproducibility policy.
- Validated contiguous `BlockLayout` and block-aware operator/preconditioner traits.
- Canonical sorted `CsrMatrix` with deterministic duplicate summation and matrix-vector action.
- Dense Newton correctness baseline with backtracking and residual traces.
- Monolithic, block Gauss-Seidel, and block Jacobi nonlinear strategies.
- Block-diagonal and block-lower-triangular preconditioners.
- BDF1/BDF2 implicit stepping with error-based rejection, consistent initialization, serializable history, restart identity, and zero-crossing events.
- Centered-difference checks for nonlinear and DAE Jacobian-vector products.

## Repository cleanup

- Removed the historical contracts alias and scientific facade.
- Removed the Malleus/JIT dependency and solver facade.
- Removed procedural macros and opcode generation.
- Removed CAD/sketch/entity/assembly, constraint-graph, pipeline/reduction, optimization, dataflow, benchmark, and bundled test-problem concerns.
- Removed stale migration, review, and implementation-plan documents. Git history is the archive.

This is an intentional API break. No compatibility types, feature aliases, or forwarding packages remain.

## Dependency contract

- Krasis implements `NonlinearOperator`, `DaeOperator`, and `BlockNonlinearOperator` for coupled state.
- Finitum may implement `LinearOperator` for realized discrete operators.
- Solverang has no dependencies on any scientific-stack repository.

## Validation

Validated locally on 2026-08-20:

- `cargo fmt --all -- --check`: passed.
- `cargo check --all-targets`: passed.
- `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo test --all-targets`: passed, 12 tests total (5 unit, 7 integration), 0 failed.

## Next concrete work

1. Add a matrix-free Krylov solver against `LinearOperator` and `Preconditioner` when Finitum supplies the first realized operator.
2. Exercise Krasis coupled-state implementations against the block and DAE acceptance tests.
3. Replace the dense Newton baseline only after representative form-compiler systems define scaling and performance requirements.

Blockers: none.
