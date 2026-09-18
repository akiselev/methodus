# Methodus status

2026-09-18 implementation in acceptance:
SC-W3 candidate: optional owner-provided linear/nonlinear/DAE diagonals; Jacobian and implicit BDF adapters forward the actual state, rate and shift. JacobiFactory consumes the existing block diagonal preconditioner and refuses missing/zero/nonfinite diagonals without probing or fallback. Owner gate passed: 100 tests, formatting, strict clippy, rustdoc and doctests; final consumer acceptance pending.


Updated: 2026-09-08
Branch: `master`
Milestone: W8 accepted-candidate BDF rate reporting (published `4f52d38`); existing numerical
algorithms, serialized state and execution behavior unchanged.

## W8 BDF candidate-rate API

`bdf_candidate_rate(&pre_step_state, &candidate_values, step, order)` returns the exact
numerical rate computed by the existing implicit-step derivative routine. Save the BDF state
before stepping; pass accepted values and the same step/order afterwards. BDF2 consumes
complete unequal-step history, bootstrapping with BDF1 when history is absent. Explicit BDF1
ignores complete older history. This read-only API performs no operator callback or solve.

It validates vector dimensions, finite values/time, positive present/previous steps, complete
history, representable time advancement, and finite coefficients/results. Tests distinguish
unequal-step quadratic BDF2 from BDF1, cover bootstrap/refusals, and verify reconstructed
rates satisfy the actual accepted implicit solve at unequal steps for both configured orders.

## W8 typed evaluation failures

`NumericError::Evaluation { code, origin, message }` passes producer failures unchanged through
algorithms and `SolveError::Numeric`. `evaluation_code()` exposes the code without string
matching. Methodus does not interpret caller codes or origins; Finitum and Krasis consume the
contract. Existing configuration/nonfinite error categories remain distinct.

## Current role

Methodus owns consumer-neutral numerical contracts and algorithms over flat
`f64` slices and explicit operator actions; it must not understand
constraints, `.res`, units, fields, materials, geometry, function spaces,
meshes, element kernels, or product/runtime policy. The repository is one
root package named `methodus` with no subordinate packages.

## Implemented surface

- In-place `LinearOperator`, `Preconditioner`, `NonlinearOperator`, and `DaeOperator` traits, with
  explicit symmetric/nonsymmetric/unknown metadata on linear actions.
- `EvaluationContext` for explicit reproducibility policy.
- Validated contiguous `BlockLayout` and block-aware operator/preconditioner traits.
- Canonical sorted `CsrMatrix` with input-order-independent duplicate summation and matrix-vector action.
- Deterministic preconditioned conjugate gradient over `LinearOperator` and `Preconditioner`, with
  residual traces, dimension/configuration validation, finite-value checks, and non-positive
  curvature refusal. CG always refuses declared-nonsymmetric actions and requires either declared
  symmetry or an explicit caller assumption for unknown actions.
- Deterministic MINRES over `LinearOperator`/`Preconditioner`/`NullspaceProjector`, admitting
  declared-`Symmetric` operators of any definiteness (indefinite included, e.g. saddle-point
  Stokes) and refusing `Nonsymmetric`/`Unknown` declarations outright with no caller-assumption
  escape hatch; a declared positive nullspace dimension refuses unless a `NullspaceProjector` is
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
- `CompositeBlockPreconditioner`: block-diagonal composition of caller-supplied per-block
  `Preconditioner`s, the bounded reference implementation for Schur-complement/pressure-mass
  saddle-point block preconditioning.
- Invariant-validated deserialization for CSR matrices, block layouts, preconditioners, and BDF history.
- Dense Newton correctness baseline with backtracking and residual traces
  (still the default inside `bdf_step` and `solve_blocks`).
- Rectangular `LeastSquaresOperator`, deterministic damped Gauss-Newton solve,
  and centered-difference full-Jacobian verification.
- Monolithic, block Gauss-Seidel, and block Jacobi nonlinear strategies.
- Block-diagonal and block-lower-triangular preconditioners.
- BDF1 and variable-step BDF2 implicit stepping with error-based rejection, consistent initialization, serializable step-size history, restart identity, and zero-crossing events.
- Checked dimension, capacity, time, and accepted-step arithmetic on fallible solver paths.
- Centered-difference checks for nonlinear and DAE Jacobian-vector products.
- Verification utilities: directional Taylor-remainder, centered-difference, and
  callback-based complex-step reports; convergence-order estimation; trajectory
  max/trapezoidal-L2 norms; solve-strategy agreement; deterministic work-budget
  checks. Malformed inputs and overflowed discrepancies are refused, never
  converted into passing evidence.

## Dependency contract

Krasis implements `NonlinearOperator`/`DaeOperator`/`BlockNonlinearOperator`;
Finitum implements `LinearOperator` (and `TransposableOperator` where it has a
transpose); Solverang implements `LeastSquaresOperator`. Methodus depends on no
scientific-stack repository.

## Validation

- `cargo test -q -p methodus`: 98 tests passed (65 unit, 33 integration), none failed or ignored.
- Focused time integration: 7 tests passed (including 2 new candidate-rate tests).
- `cargo clippy -p methodus --all-targets -- -D warnings`: passed.
- `RUSTDOCFLAGS='-D warnings' cargo doc -p methodus --no-deps`: passed.
- Scoped formatting and `git diff --check`: passed.

## Known limits (updated after the W7 lane-3 slices)

- A transpose exists only by `Symmetric` delegation or an explicit
  `TransposableOperator`; Finitum's matrix-free operators implement neither
  today, so `solve_adjoint` is usable on `CsrMatrix`-shaped assembled
  operators and on whatever Finitum's SV1-C1 lane makes `TransposableOperator`
  (W7 lane 2), not on the current Finitum matrix-free path.
- `solve_adjoint` takes a preconditioner for `Aᵀ`; Methodus offers no
  transposed-preconditioner adapter, so a caller with an approximate inverse
  of `A` must transpose it itself where the two differ.
- Linear-solve *sensitivity* beyond the adjoint solve (tangent solves with a
  caller-differentiated right-hand side `∂b/∂p − (∂A/∂p) u`) needs no new
  Methodus algorithm — it is a primal `solve_krylov` — and no wrapper was
  added for it; the parameter-derivative actions are Finitum's (SV1-C3).
- Block preconditioning is limited to block-diagonal and block-lower-
  triangular composition (`BlockDiagonalPreconditioner`,
  `BlockLowerTriangularPreconditioner`, `CompositeBlockPreconditioner`); no
  algebraic multigrid, incomplete factorization, or Schur-complement
  *computation* exists — only the composition contract. A caller must supply
  its own approximate Schur-complement/pressure-mass block preconditioner.
- BiCGSTAB has no restart or look-ahead; a Lanczos breakdown is a typed
  error, and a caller wanting robustness against it selects GMRES.
- Newton–Krylov requires the operator's own JVP; there is no
  finite-difference Jacobian-free fallback (a JVP is part of every Methodus
  nonlinear contract). Globalization is backtracking only (no trust region),
  and a residual already at its floating-point floor fails the sufficient-
  decrease test as `LineSearchFailed` rather than being declared converged
  — callers set the outer tolerance above `‖J‖·‖x‖·ε`.
- `solve_blocks` (Gauss–Seidel/Jacobi, also as `BlockNewton` inside BDF)
  still builds dense per-block Jacobians by JVP column probing; a
  block-aware Newton–Krylov (per-block Krylov solves inside the staggered
  update) is not implemented.
- `bdf_step_with` ignores `config.newton`; the nonlinear policy lives in the
  supplied `NonlinearSolver`. `BdfConfig`'s serialized shape is unchanged.
- MINRES's nullspace-projection hook ships one bounded reference
  implementation, `ConstantModeProjector` (a single constant mode over one
  contiguous coordinate range). Multi-dimensional nullspaces (e.g.
  rigid-body modes) need a caller-supplied `NullspaceProjector`; no reference
  implementation exists for that shape.

## Next concrete work

1. Done (Sinbad C11.17, `35f4e2a`): MINRES/GMRES are wired into Sinbad's
   `SolvePolicy`/`LinearAlgorithm` admission; Methodus did not own that
   selection policy and did not change.
2. Promote the dense least-squares baseline only from representative Solverang
   constraint systems and independent numerical checks.
3. Replace dense Newton only after representative compiled systems define
   scaling and performance requirements.
4. Done (W7 lane 3, this slice): the inexact Newton–Krylov driver with
   `KrylovMethod`, `PreconditionerFactory`, and `NullspaceProjector` hooks,
   plus `bdf_step_with`/`NonlinearSolver` for Krasis's Newton inside BDF.
   Krasis wires `NewtonKrylovSolver` into its DAE transactions when batch P
   needs it; Sinbad resolves policy into `KrylovMethod`/`NewtonKrylovConfig`.
5. SC composition (design `sinbad/ARCHITECTURE.md` §8–9; the SV7-F3 subset
   pulled forward under its own ID): fixed-point acceleration over `&[f64]`
   iterate sequences (relaxation, Aitken; IQN later), and a block-aware
   Newton–Krylov if `solve_blocks`'s dense per-block Jacobians become the
   bottleneck. `CompositeBlockPreconditioner` is reused as it is. Methodus
   never sees instance names, outputs, or connector vocabulary; Sinbad
   resolves schedules and convergence targets to block ids.
6. Block-preconditioner contracts beyond block-diagonal composition
   (Schur-complement/pressure-mass approximations as traits with a dense
   reference) only when a Finitum or Krasis case demonstrates the need; none
   surfaced in `ARCHITECTURE.md` §6/§9 during W7.
   **Consumer named 2026-09-07:** workspace `PLAN.md` §6 "W8" decision 6 makes SC-W3 the
   consumer — Sinbad's `PreconditionerPolicy` grows beyond `None`, Sinbad consumes
   `NewtonKrylovSolver` + `PreconditionerFactory`, and the transient product path drops its dense
   monolithic Newton Jacobian (kept only as the agreement oracle). The Schur/pressure-mass
   contracts start when the SC-W3 Methodus lane is launched, not before.

Blockers: none.
