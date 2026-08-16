# Solverang status

Updated: 2026-08-16
Branch: `agent/r13-r20-wave-a-f`
Milestone: Waves B, E, F / R13, R18-R20

## Current role

Solverang owns physics-neutral numerical contracts and algorithms. It must not understand RSL, materials, function spaces, mesh semantics, or names such as temperature/voltage/displacement/pressure.

## Stable baseline

- `solverang-contracts` provides sparse matrix/operator and DAE seams.
- Existing Newton/LM/sparse/JIT paths remain available for legacy and geometric consumers.
- Malleus is the real JIT/compiler boundary; no fake Malleus compatibility crate remains.

## Implemented on this branch

A new `solverang-scientific` workspace crate isolates the scientific numerical lane from the legacy geometry-facing API.

- R13: general implicit `DaeOperator` contract for `F(t,y,ydot)=0`, JVPs, consistent initialization, and event values.
- R18: `BlockLayout`, `BlockResidual`, `BlockLinearOperator`, and `BlockPreconditioner` contracts.
- R19: monolithic Newton, block Newton, Gauss-Seidel/staggered, and Jacobi strategies with per-block scaling and damping/line search traces.
- R20: BDF1/BDF2 implicit stepping, numerical Newton linearization via JVPs, adaptive error estimate, explicit accepted/rejected outcomes, unchanged-state rejection semantics, consistent-initial-state hook, and zero-crossing event reporting.
- Verification helper: directional JVP vs centered finite difference.

## Validation state

Local Rust validation is unavailable in the execution sandbox because rustup cannot reach its download service. GitHub Actions is the compile/test authority. This branch is not verified until CI is green.

## Cross-repository contract

Sinbad should adapt its coupled/runtime state to `solverang-scientific` contracts. Solverang must receive only numeric vectors, block layouts, residual/JVP callbacks, scaling, and algorithm configuration.

## Next

1. Run/fix workspace format, clippy, and tests in GitHub Actions.
2. Exercise electrothermal and thermoelastic block problems from Sinbad.
3. Exercise BDF1/BDF2 manufactured transient cases and rejected-step history invariants through Sinbad's transactional state adapter.
