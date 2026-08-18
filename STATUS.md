# Solverang status

Updated: 2026-08-16
Branch: `agent/r13-r20-wave-a-f`
Milestone: Waves B, E, F / R13, R18-R20

## Current role

Solverang owns physics-neutral numerical contracts and algorithms. It must not understand RSL, materials, function spaces, mesh semantics, or field names such as temperature/voltage/displacement/pressure.

## Implemented on this branch

- R13: generic implicit `DaeOperator` for `F(t,y,ydot)=0`, JVPs, consistent initialization, and event values.
- R18: block layout/residual/linear/preconditioner contracts.
- R19: monolithic Newton, block Newton, Gauss-Seidel and Jacobi strategies with per-block scaling, damping/line search, physics-neutral block-diagonal/lower-triangular preconditioners, and per-block residual histories.
- R19 acceptance includes a common two-block reference problem where monolithic and block Newton agree, plus a strongly coupled case where monolithic convergence is contrasted with deliberately slow staggered convergence.
- R20: BDF1/BDF2, adaptive error control, JVP-based implicit systems, bit-identical rejected committed state, consistent initialization, located zero-crossing events, and serializable history state.
- R20 acceptance: BDF1/BDF2 convergence, byte-identical rejected state, checkpoint/restart trajectory identity, and event detection without corrupting BDF history.
- Verification helper: analytic DAE JVP vs centered finite difference.

## Validation state

Local Rust installation is blocked by sandbox DNS; GitHub-hosted Rust jobs are authoritative. Earlier compile findings in the Wave layer were corrected and rustfmt applied. This user-authored update retriggers full normal CI on the R19 block-history/strategy tests and R20 restart/event tests.

## Cross-repository contract

Sinbad consumes numeric vectors, block layouts, residual/JVP callbacks, scaling, preconditioners, and algorithm configuration only. Once normal CI is green, Sinbad must pin this exact revision in Cargo metadata and `scientific-stack.lock`.

## Remaining before merge

1. Resolve any findings from the complete current CI run.
2. Pin the final green revision into Sinbad and rerun the federation acceptance suite.
3. Record final CI evidence and keep this file compact rather than appending history.
