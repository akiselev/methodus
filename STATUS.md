# Methodus status

Updated: 2026-10-19
Branch: `master`
Milestone: SV2-C4 explicit SSP Runge–Kutta / method-of-lines contract (W12
lane L2), on top of SC-W3 composite block preconditioners and acceleration.
Compatibility: purely additive. No existing type, serialized shape, default
or algorithm changed (no existing file but `lib.rs`/`error.rs` touched);
`SolveError` gained four variants (it is not `Serialize`; an exhaustive
`match` on it in a consumer would need new arms — none found in live
Krasis/Sinbad/Finitum). The 128 preexisting owner tests pass unchanged.

## SV2-C4 explicit method-of-lines stepping

`explicit.rs`. `ExplicitOperator` (style of `DaeOperator`): `dimension`,
`rhs(cx, t, y, out)`, and defaulted `stable_step` (forward-Euler-stable step
at CFL 1; `None` = undeclared, `+∞` = unconstrained, `≤0`/NaN refused),
`admissibility -> Admissibility::{Admissible, Inadmissible{index, reason}}`,
`linear_invariants -> &[Vec<f64>]` (weights `w_k`), `invariant_rates`
(declared net boundary/source rate of each `w_k·y`; `false` = closed).

`ExplicitScheme::{ForwardEuler, SspRk2, SspRk3, SspRk104, Rk4}` with
`order()`, `stages()`, declared `ssp_coefficient()` (1, 1, 1, 6, `None` —
RK4 declared non-SSP) and `tableau() -> ShuOsherTableau` (coefficients as
data; `ssp_coefficient_from_coefficients()`, `butcher_form()`). SSP-RK(10,4)
is Ketcheson's low-storage algorithm rewritten as a 10-row Shu–Osher form
(rows 5 and 10 are the convex recombinations, α/β = 6 throughout).
Summation order is fixed: each stage component accumulates `α_ij·Y_j` then
`(β_ij·Δt)·F_j` for `j` ascending, zero coefficients skipped, no FMA
(forward Euler is exactly `y + Δt·F`); invariants sum in component order.
All stage states and right-hand sides are kept (`2s` vectors; a
low-storage path is open).

`StepPolicy::{Fixed{step}, Cfl{target, max, min_step, max_step}}`,
`ExplicitConfig{scheme, step_policy}`, `ExplicitState{time, values,
accepted_steps}` (serde; non-finite refused on deserialize;
`initialize` refuses an inadmissible initial state as
`SolveError::InadmissibleState`). `explicit_step(op, cx, state,
final_time, cfg, retry_ceiling) -> ExplicitStepOutcome::{Accepted{state,
report}, Rejected{attempted_step, rejection, suggested_step}}`; the input
state is never mutated. Step = `Fixed.step` or `target·stable_step(y_n)`
clipped to `max_step`, then to the retry ceiling; `< min_step` is
`SolveError::StepSizeCollapse`; lands exactly on `final_time` when the
remainder is ≤ the step plus `(k+4)·ε·max|t|` after `k` steps (no sliver
steps). Every stage state is checked finite, then admissible:
`ExplicitRejection::InadmissibleStage{stage, time, index, reason}`
(suggestion `Δt/2` under `Cfl`, `None` under `Fixed`); under `Cfl` each
stage's `Δt/stable_step(Y_i)` above `max(max, cfl_0)` is
`CflExceeded{stage, cfl, max}` (suggestion `target·min hint`). Non-finite
stages/rhs/rates and malformed weights/hints are `NumericError`s.
`ExplicitStepReport{step, step_limit, scheme, stages, ssp_coefficient,
cfl_realized (max over stages), balance_corrected, invariants:
[InvariantBalance{before, after, predicted, balance_residual, scale}]}` —
`predicted` integrates the declared rates with the scheme's own α/β, so for
a conservative operator `balance_residual` is pure round-off relative to
`scale = max Σ|w y|`. `integrate_explicit(.., max_attempts)` retries at the
suggestion, refusing `Fixed` rejections (`ExplicitStepRejected`), collapse,
and budget exhaustion (`ExplicitStepBudgetExhausted`). Krasis's L5
`ExplicitExecution` consumes `explicit_step` (Rejected = retryable,
`Err` = not).

`spectral.rs`: `estimate_spectral_radius(matrix, n, iterations) ->
SpectralRadiusReport{symmetric, gershgorin_upper_bound,
rayleigh_lower_bound, power_estimate, iterations}` over a dense row-major
matrix. Rigour: the Gershgorin bound (min of row/column abs-sums, inflated
by `1+2(n+1)ε`) is a rigorous upper bound with rounding; the Rayleigh bound
(exactly symmetric matrices only, rounding-deflated) is a rigorous lower
bound; `power_estimate` is a heuristic (last Rayleigh quotient, or the
second-half growth rate for nonsymmetric matrices, which also handles a
complex dominant pair).

Proof (`tests/explicit_integration.rs`, `explicit.rs`/`spectral.rs` unit
tests): order conditions through order 4 hold to 1e-14 for each tableau and
the next order fails for orders 1–3; computed SSP coefficients equal the
declared ones. Observed order, fixed steps N = 20/40/80/160 on [0, 2],
least-squares fit (`estimate_convergence_order`), asserted `|fit − p| ≤ 0.1`:
damped linear oscillator / nonlinear non-autonomous `y' = −2ty²`:
FE 1.040/1.024, SSP-RK2 2.028/2.029, SSP-RK3 3.003/3.042, SSP-RK(10,4)
3.991/4.007, RK4 3.975/4.025; each run takes exactly N steps. TVD: periodic
first-order upwind (plain `&[f64]` operator), 100 cells, square wave, each
SSP scheme at CFL = its SSP coefficient to t = 0.5 — total variation
non-increasing every step (≤ +1e-13) and values in [−1e-15, 1+1e-15]
(SSP-RK(10,4): 9 steps at CFL 6); forward Euler at CFL 1.5 raises TV by
> 0.1 (negative control). Conservation: nonuniform periodic upwind, 64
cells, every scheme to t = 2 at 0.9·C: per-step `|balance_residual|` ≤
`(n + 4s)·ε·scale` (worst observed 3.5–5.0 ε·scale over 48–285 steps) and
total drift ≤ the summed budget (observed ≤ 1.6e-14); a 1e-3 leak exceeds
the bound by > 1e6×; an open inflow/outflow system is balanced by its
declared rates (≤ 76 ε·scale) and is > 1e6 ε·scale without them. Plus
rejection/retry/collapse/budget/Fixed-refusal paths, RK4 stage-3
inadmissibility, CFL-stage rejection with exact suggestion, NaN/overflow
refusals, serde round trips. Spectral: (−1,2,−1) tridiagonal bracketed with
the estimate within 1e-6; upwind circulant estimate 2 ± 1e-2 under a
bound ≤ 2(1+1e-14); scaled rotation exact; nilpotent → 0.

Open: `ImexOperator` (ARK2/3) not declared — a trait with no integrator
using it would be dead scaffolding; specify with the first IMEX case.
Low-storage (2-register) execution of SSP-RK(10,4); embedded error
estimators/adaptive accuracy control (only CFL control exists).

## SC-W3 preconditioners and acceleration (2026-09-29, review-fixed 2026-10-08)

`BlockGaussSeidelPreconditioner` (forward/backward/symmetric sweeps over an
owner `BlockLayout`, owner diagonal solves and action-based
`BlockCouplingAction`s; typed refusals, no probing). Symmetry is declared,
never assumed: `Symmetric` only for the symmetric sweep with every diagonal
solve `Symmetric` and couplings closed under owner-declared transposition
(`BlockCouplingAction::transpose_of`), else `Unknown`/`Nonsymmetric`.
`Preconditioner::symmetry()` (defaulted `Unknown`); CG and MINRES refuse a
`Nonsymmetric` preconditioner; block-diagonal declares `Symmetric`, block
lower-triangular `Nonsymmetric`, `CompositeBlockPreconditioner` follows its
blocks. Measured GMRES counts to a relative 1e-3 reduction (none, Jacobi,
GS): 2 blocks (3, 3, 2), 3 blocks (5, 4, 2).

`accelerate_fixed_point` (`FixedRelaxation`, `Aitken`) over `&[f64]`
fixed-point sequences: convergence tested before any factor, relative
scale-invariant Aitken breakdown test, per-entry factor trace that replays
bit for bit, `AccelerationBreakdown`/`AccelerationNotConverged{trace}`
refusals. Opt-in inside `solve_blocks` via `NewtonConfig.acceleration`
(`None` default, skipped when `None` so `NewtonConfig` bytes are unchanged;
pinned against the pre-field literal). Aitken counts (plain, aitken) =
(111, 8) at 1e-6. Krasis's per-sweep driver must use `accelerate_fixed_point`
(history resets per `solve_blocks` call). Details: git history of
`24f9c7d..a9d97c8`.

## Current role

Methodus owns consumer-neutral numerical contracts and algorithms over flat
`f64` slices and explicit operator actions; it must not understand
constraints, `.res`, units, fields, materials, geometry, function spaces,
meshes, element kernels, or product/runtime policy. The repository is one
root package named `methodus` with no subordinate packages.

## Implemented surface

- In-place `LinearOperator`, `Preconditioner`, `NonlinearOperator`, and `DaeOperator` traits, with
  explicit symmetric/nonsymmetric/unknown metadata on linear actions and (new) preconditioners.
- `EvaluationContext` for explicit reproducibility policy.
- Validated contiguous `BlockLayout` and block-aware operator/preconditioner traits.
- Canonical sorted `CsrMatrix` with input-order-independent duplicate summation and matrix-vector action.
- Deterministic preconditioned conjugate gradient over `LinearOperator` and `Preconditioner`, with
  residual traces, dimension/configuration validation, finite-value checks, and non-positive
  curvature refusal. CG always refuses declared-nonsymmetric operators or preconditioners and
  requires either declared symmetry or an explicit caller assumption for an unknown operator.
- Deterministic MINRES over `LinearOperator`/`Preconditioner`/`NullspaceProjector`, admitting
  declared-`Symmetric` operators of any definiteness (indefinite included, e.g. saddle-point
  Stokes) and refusing `Nonsymmetric`/`Unknown` declarations outright with no caller-assumption
  escape hatch, and (like CG) refusing a preconditioner declared `Nonsymmetric`; a declared
  positive nullspace dimension refuses unless a `NullspaceProjector` is
  supplied, in which case every Krylov vector and the returned solution stay orthogonal to the
  declared nullspace.
- Deterministic restarted GMRES over `LinearOperator`/`Preconditioner`, admitting any declared
  symmetry and refusing only a non-square operator; modified Gram-Schmidt Arnoldi with incremental
  Givens-rotation QR, left preconditioning, and per-restart-cycle telemetry.
- Deterministic right-preconditioned BiCGSTAB (`solve_bicgstab`) admitting any declared symmetry,
  refusing only a non-square operator, reporting true residuals, and typing Lanczos breakdowns.
- `KrylovMethod`/`solve_krylov`: one serializable selector over CG/MINRES/GMRES/BiCGSTAB with
  per-solver admission preserved and a nullspace-projector hook (native in MINRES; endpoint
  projection for GMRES/BiCGSTAB; refused for CG).
- `solve_adjoint`: `Aᵀ λ = g` through `TransposeOperator` (symmetric delegation or explicit
  `TransposableOperator`), property-aware method refusal, true-residual acceptance, typed
  deterministic telemetry including the `TransposeSource`.
- `solve_newton_krylov`: inexact Newton over `JacobianOperator` (matrix-free JVP with declared
  `jacobian_properties`) with any `KrylovMethod`, `Constant`/`EisenstatWalker` forcing,
  sufficient-decrease backtracking, `PreconditionerFactory` and `NullspaceProjector` hooks,
  and typed per-iteration telemetry; `NonlinearSolver` (`DenseNewton`, `NewtonKrylovSolver`,
  `BlockNewton`) and `bdf_step_with` let BDF run any of them inside a step; a fixed
  `&dyn Preconditioner` is accepted as a `PreconditionerFactory`.
- `NullspaceProjector` trait plus the bounded reference `ConstantModeProjector` (one constant mode
  over a contiguous coordinate range).
- Block preconditioners: `BlockDiagonalPreconditioner` (Jacobi),
  `BlockLowerTriangularPreconditioner` (dense off-diagonal storage),
  `BlockGaussSeidelPreconditioner` (forward/backward/symmetric, action-based
  off-diagonal couplings, owner inner solves), and `CompositeBlockPreconditioner`
  (block-diagonal composition of independent per-block preconditioners, the
  bounded reference for Schur-complement/pressure-mass saddle-point shapes).
- `accelerate_fixed_point`/`AccelerationMethod` (fixed relaxation, Aitken) over
  `&[f64]` fixed-point sequences, also selectable inside `solve_blocks`.
- Invariant-validated deserialization for CSR matrices, block layouts, preconditioners, and BDF history.
- Dense Newton correctness baseline with backtracking and residual traces
  (still the default inside `bdf_step` and `solve_blocks` when no acceleration is configured).
- Rectangular `LeastSquaresOperator`, deterministic damped Gauss-Newton solve,
  and centered-difference full-Jacobian verification.
- Monolithic, block Gauss-Seidel, and block Jacobi nonlinear strategies.
- BDF1 and variable-step BDF2 implicit stepping with error-based rejection, consistent initialization, serializable step-size history, restart identity, and zero-crossing events.
- Checked dimension, capacity, time, and accepted-step arithmetic on fallible solver paths.
- Centered-difference checks for nonlinear and DAE Jacobian-vector products.
- Verification utilities: directional Taylor-remainder, centered-difference, and
  callback-based complex-step reports; convergence-order estimation; trajectory
  max/trapezoidal-L2 norms; solve-strategy agreement; deterministic work-budget
  checks. Malformed inputs and overflowed discrepancies are refused, never
  converted into passing evidence.
- `bdf_candidate_rate` (W8): read-only reconstruction of the numerical rate an
  accepted implicit step used, from the pre-step state and accepted values.
- `NumericError::Evaluation { code, origin, message }` (W8): a producer's typed
  evaluation failure passes unchanged through every algorithm and `SolveError`.
- `ExplicitOperator`, `explicit_step`/`integrate_explicit` (SSP-RK1/2/3,
  SSP-RK(10,4), RK4) with `StepPolicy::{Fixed, Cfl}`, stage admissibility,
  and per-step linear-invariant balance telemetry (SV2-C4).
- `estimate_spectral_radius`: dense Gershgorin upper / symmetric Rayleigh
  lower bounds and a power-iteration estimate (SV2-C4 `SpectralBound`).

## Dependency contract

Krasis implements `NonlinearOperator`/`DaeOperator`/`BlockNonlinearOperator`;
Finitum implements `LinearOperator` (and `TransposableOperator` where it has a
transpose) and, for finite volume, `ExplicitOperator`; Krasis drives
`explicit_step` inside its explicit transaction; Solverang implements `LeastSquaresOperator`. Methodus depends on no
scientific-stack repository.

## Validation

2026-10-19, opt-level 2 test profile:
- `cargo test -p methodus --lib`: 99 passed, none failed or ignored (the 91
  preexisting plus 4 `explicit` and 4 `spectral` unit tests).
- `cargo test -p methodus --test <name>`: `adjoint` 9, `coupling_strategies`
  4, `newton_krylov` 12, `time_integration` 7, `time_restart_events` 2,
  `transpose` 3 (the 128 preexisting tests, unchanged and passing), and
  `explicit_integration` 11 — all passed. 147 tests total.
- `cargo fmt -p methodus -- --check`: passed.
- `cargo clippy -p methodus --all-targets --all-features -- -D warnings`: passed.
- `RUSTDOCFLAGS='-D warnings' cargo doc -p methodus --no-deps`: passed.
- `cargo test -p methodus --doc`: 0 doctests, none failed.

## Known limits

- A transpose exists only by `Symmetric` delegation or an explicit
  `TransposableOperator`; Finitum's matrix-free operators implement neither
  today, so `solve_adjoint` is usable on `CsrMatrix`-shaped assembled
  operators and on whatever Finitum's SV1-C1 lane makes `TransposableOperator`,
  not on the current Finitum matrix-free path.
- `solve_adjoint` takes a preconditioner for `Aᵀ`; Methodus offers no
  transposed-preconditioner adapter, so a caller with an approximate inverse
  of `A` must transpose it itself where the two differ.
- Block preconditioning covers block-diagonal, block-lower-triangular (dense
  or action-based via Gauss-Seidel), and symmetric/backward Gauss-Seidel
  composition; no algebraic multigrid, incomplete factorization, or
  Schur-complement/pressure-mass *computation* exists — only the composition
  contract. A caller must supply its own approximate block solves.
- Iterate-sequence acceleration covers fixed relaxation and Aitken; IQN and
  any interface (multiplier/Nitsche) acceleration are not implemented.
- BiCGSTAB has no restart or look-ahead; a Lanczos breakdown is a typed
  error, and a caller wanting robustness against it selects GMRES.
- Newton–Krylov requires the operator's own JVP; there is no
  finite-difference Jacobian-free fallback. Globalization is backtracking
  only (no trust region) unless `NewtonConfig.acceleration` selects
  relaxation/Aitken instead for a partitioned strategy; a residual already
  at its floating-point floor fails the sufficient-decrease test as
  `LineSearchFailed` rather than being declared converged — callers set the
  outer tolerance above `‖J‖·‖x‖·ε`.
- `solve_blocks` (Gauss–Seidel/Jacobi, also as `BlockNewton` inside BDF)
  still builds dense per-block Jacobians by JVP column probing; a
  block-aware Newton–Krylov (per-block Krylov solves inside the staggered
  update) is not implemented.
- `bdf_step_with` ignores `config.newton`; the nonlinear policy lives in the
  supplied `NonlinearSolver`. `BdfConfig`'s serialized shape is unchanged.
- MINRES's nullspace-projection hook ships one bounded reference
  implementation, `ConstantModeProjector` (a single constant mode over one
  contiguous coordinate range). Multi-dimensional nullspaces need a
  caller-supplied `NullspaceProjector`.

## Next concrete work

1. Block-aware Newton–Krylov inside `solve_blocks` (per-block Krylov solves
   replacing dense per-block JVP-column Jacobians) only if a Finitum/Krasis
   case demonstrates the current dense per-block cost is the bottleneck.
2. Schur-complement/pressure-mass block *computation* (not just the
   `CompositeBlockPreconditioner` composition contract already landed) only
   when a Finitum or Krasis case demonstrates the need.
3. IQN acceleration when SV7-F3's IQN item is scheduled; not part of this
   SC-W3 increment.
4. Promote the dense least-squares baseline only from representative
   Solverang constraint systems and independent numerical checks.

Blockers: none.

Final cross-repository evidence: [September 18 acceptance](../sinbad/docs/validation/2026-09-18-assembly/README.md).
