# Solverang status

Updated: 2026-08-16
Branch: `agent/r13-r20-wave-a-f`
Milestone: Waves B, E, F / R13, R18-R20

## Current role

Solverang owns physics-neutral numerical contracts and algorithms. It must not understand RSL, materials, function spaces, mesh semantics, or names such as temperature/voltage/displacement/pressure.

## Implemented on this branch

A new `solverang-scientific` workspace crate isolates the scientific numerical lane from the legacy geometry-facing API.

- R13: general implicit `DaeOperator` contract for `F(t,y,ydot)=0`, JVPs, consistent initialization, and event values.
- R18: `BlockLayout`, `BlockResidual`, `BlockLinearOperator`, and `BlockPreconditioner` contracts.
- R19: monolithic Newton, block Newton, Gauss-Seidel/staggered, and Jacobi strategies with per-block scaling and damping/line-search traces.
- R20: BDF1/BDF2 implicit stepping, JVP-based Newton linearization, adaptive error estimate, explicit accepted/rejected outcomes, bit-identical committed-state rejection semantics, consistent-initial-state hook, and zero-crossing event reporting.
- R20 acceptance tests cover BDF1 first-order and BDF2 second-order convergence on a manufactured decay problem plus byte-identical rejected state.
- Verification helper: directional JVP vs centered finite difference.
- Existing integrator rustdoc links were repaired without changing the numerical API.

## Validation state

Local Rust installation is blocked by the execution sandbox's outbound DNS policy; GitHub-hosted Rust jobs are the validation authority.

The latest dedicated rustdoc diagnostic completed with exit code 0, confirming the earlier documentation failure is repaired. Prior normal-CI runs also had build/test, MSRV, and format/clippy green after the scientific adapter fixes. Temporary diagnostics have been removed. This user-authored status commit retriggers the complete normal CI on the current R20 convergence/rejection test tree.

Do not mark the branch fully verified until that complete run is green.

## Cross-repository contract

Sinbad consumes only numeric vectors, block layouts, residual/JVP callbacks, scaling, and algorithm configuration from this crate. Once the current complete CI is green, Sinbad's Cargo revision and `scientific-stack.lock` should pin that exact Solverang commit.

## Remaining before merge

1. Confirm the complete current normal-CI run is green, including the new R20 convergence/rejection tests.
2. Freeze the resulting revision in Sinbad's federation tuple.
3. Re-run Sinbad scientific-stack CI against that exact revision.
