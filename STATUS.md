# Methodus status

Updated: 2026-09-01
Branch: `master`
Milestone: W7 lane 3 — E7/SV1-D1 adjoint solve on nonsymmetric operators
and the inexact Newton–Krylov driver with preconditioner/nullspace hooks
(previously: SV2-B6 MINRES/GMRES, nullspace projection, and block
preconditioner contracts; SV0-B1 checkers; SV1-C5 transposes)

## E7/SC-W3 inexact Newton–Krylov driver (W7, 2026-09-01)

`solve_newton_krylov` solves `F(x) = 0` by inexact Newton over a matrix-free
Jacobian: `JacobianOperator` exposes `NonlinearOperator::jacobian_vector_
product` at a frozen state as a square `LinearOperator` whose declared
`OperatorProperties` come from the new defaulted trait method
`NonlinearOperator::jacobian_properties` (and `DaeOperator::
jacobian_properties` for the implicit-step Jacobian `∂F/∂y + α ∂F/∂ẏ`;
both default to `Unknown`, so admission is honest by default). The inner
solve is any `KrylovMethod` through `solve_krylov`, so each method's own
admission applies unchanged (CG refuses a `Nonsymmetric` Jacobian, MINRES
refuses anything not `Symmetric`, GMRES/BiCGSTAB admit any declaration);
nothing is ever substituted. `ForcingPolicy` is `Constant { forcing }` or
`EisenstatWalker { initial, gamma, alpha, maximum }` (choice 2 with the
collapse safeguard), both floored at half the outer threshold relative to
`‖F_k‖` so the last inner solves are never oversolved. Globalization is
backtracking with the Eisenstat–Walker sufficient-decrease condition
`‖F(x+λs)‖ ≤ (1 − tλ(1−η))‖F(x)‖`. Hooks: `PreconditionerFactory::build`
is called at every iterate with the frozen Jacobian and state and may return
`None`; a `NullspaceProjector` is forwarded to `solve_krylov`. Telemetry
(`NewtonKrylovReport`/`NewtonKrylovIteration`/`LinearStepSummary`) records
per outer iteration the residual norm, the forcing used, and the inner
solve's method, iteration count, verdict, final residual, and restart cycles;
it is bit-reproducible. An inner solve that exhausts its budget is recorded
and its step still tried (inexact Newton needs only a descent direction);
a refusal, breakdown, or failed sufficient decrease is a typed error; an
exhausted outer budget is `converged == false`.

For Krasis's Newton-inside-BDF (batch P): `NonlinearSolver` is a
`Send + Sync` trait with `solve(&dyn NonlinearOperator, context, initial)`;
`DenseNewton` wraps `solve_newton`, `NewtonKrylovSolver` wraps
`solve_newton_krylov`; `bdf_step_with(operator, context, state, step,
config, solver: &dyn NonlinearSolver)` runs a BDF1/BDF2 attempt with the
supplied solver (`config.newton` is then not consulted), and the implicit
operator forwards `DaeOperator::jacobian_properties`. `bdf_step` is
unchanged in signature and behaviour (`bdf_step_with` over `DenseNewton`).

Acceptance tests (`tests/newton_krylov.rs`, 8): quadratic convergence
(`‖F_{k+1}‖/‖F_k‖² < 2` in the local regime) on a 2-D nonsymmetric
algebraic fixture with tiny constant forcing, superlinear convergence
(strictly decreasing rate reaching `< 1e-2`) with Eisenstat–Walker; CG and
MINRES refused on the declared-nonsymmetric Jacobian, BiCGSTAB admitted;
`−u'' + u³ = f` (63 unknowns, SPD-declared Jacobian) through CG with
quadratic ratios `< 0.1` on every effectively-exact step and the
manufactured `sin(πx)` recovered to discretization accuracy;
`−u'' + 8u' + u³ = f` (40 unknowns, nonsymmetric) through GMRES, CG refused
even with `AssumeSymmetric` (that hatch covers only `Unknown`), an exact
tridiagonal `PreconditionerFactory` making every inner solve one iteration
and cutting total linear iterations, and agreement with dense `solve_newton`
to 1e-10; a singular-Jacobian fixture (constant-mode nullspace) refused by
MINRES without a projector and solved to the pseudo-solution through
MINRES, GMRES and BiCGSTAB with `ConstantModeProjector`, the constant-mode
component of the state preserved; 20 BDF2 steps of `y' = −y` through
`bdf_step_with(NewtonKrylovSolver(CG))` matching `bdf_step` to 1e-12 and
MINRES refused inside the step on a DAE declaring nothing; bit-identical
and JSON-round-trip telemetry plus honest outer-budget exhaustion. Unit
tests cover configuration/forcing validation and the Eisenstat–Walker
sequence, floor, and safeguard.

## E7/SV1-D1 adjoint solve on nonsymmetric operators (W7, 2026-09-01)

`solve_adjoint` solves `Aᵀ λ = g` through a `TransposeOperator` view of `A`
and never approximates a transpose: the view is built either by symmetric
delegation (`TransposeOperator::new`, refused unless `Symmetric` is declared)
or from an explicit `TransposableOperator::apply_transpose`
(`TransposeOperator::explicit`, the assembled-operator path — `CsrMatrix`
implements it by transposed CSR traversal). An operator offering neither is
refused at construction, before any iteration. `TransposeOperator` now
reports its `TransposeSource` (`SymmetricDelegation | ExplicitTranspose`) and
carries the primal `OperatorProperties` through (definiteness and symmetry
are transpose-invariant; nullspace dimension and block structure only for
square operators).

Method selection is one serializable `KrylovMethod` value
(`ConjugateGradient | Minres | Gmres | BiCgStab`, each with its full
config) dispatched by `solve_krylov` without loosening any solver's own
admission; the adjoint driver additionally refuses conjugate gradient on a
`Nonsymmetric` transpose and MINRES on a `Nonsymmetric`/`Unknown` one with
adjoint-specific refusal text. `solve_bicgstab` is new: right-preconditioned
van der Vorst BiCGSTAB whose reported residuals are always the true
`‖b − A x‖`, with Lanczos-type breakdowns typed as `KrylovBreakdown`.
Acceptance is residual-based and method-independent: after the Krylov
solve the driver recomputes `g − Aᵀ λ` through the transpose action and sets
`converged` only from that norm against `ResidualAcceptance`, independent of
the inner solver's (possibly preconditioner-weighted) estimate, which is
reported separately as `solver_converged`. An exhausted budget returns
`converged == false` with the measured residual, never an error and never a
claim. Telemetry (`AdjointSolveReport`) is typed and bit-reproducible.

Acceptance tests (`tests/adjoint.rs`, 9): `<λ, b> = <g, u>` for `A u = b`
and `Aᵀ λ = g` within 1e-10 on a 6x6 nonsymmetric dense fixture and a 12x12
nonsymmetric upwind convection–diffusion `CsrMatrix`, through both GMRES and
BiCGSTAB; CG/MINRES refusal on a nonsymmetric transpose; refusal of a
transpose-less matrix-free operator before any solve; rectangular refusal;
acceptance measured on the true residual under a left preconditioner; budget
exhaustion reported as non-acceptance; bit-identical and JSON-round-trip
telemetry; transpose property carry-through. Unit tests cover BiCGSTAB (true
residuals with and without preconditioning, determinism, rectangular
refusal) and the dispatcher (kind reporting, per-solver admission kept, CG
refusing a projector, GMRES with a projector reaching the pseudo-solution of
a singular consistent system, tolerance override keeping other fields).

Deviation recorded: C11.16 said GMRES takes no nullspace projector. The
dispatcher gives GMRES/BiCGSTAB a projector hook that projects only the
initial guess and the returned solution (selecting the representative
orthogonal to the declared nullspace without touching the residual, so
acceptance stays honest); `solve_gmres`/`solve_bicgstab` signatures are
unchanged. Conjugate gradient still refuses a projector.

## Earlier slices (compacted; details in Git history)

- **SV2-B6 (Methodus `8de32cd`, ratified as GX-CONTRACTS C5.6):** MINRES
  (declared-`Symmetric` only, any definiteness, `NullspaceProjector` hook)
  and restarted GMRES (any declared symmetry, square only) with typed
  refusals and bit-reproducible `LinearIteration` traces;
  `ConstantModeProjector`; `CompositeBlockPreconditioner` as the
  block-diagonal composition contract for saddle-point preconditioning
  (no Schur-complement computation). Twelve acceptance tests.
- **SV1-C5 / GX-D2 (`6e7fd94`):** `OperatorProperties` (symmetry,
  definiteness, nullspace dimension, structure hint), `TransposableOperator`
  with `TransposeOperator::{new, explicit}`, `verify_adjoint_identity`,
  honest CG refusal of `Indefinite`/nullspace declarations.

## Current role

Methodus owns consumer-neutral numerical contracts and algorithms. It operates
on flat `f64` slices and explicit operator actions. It must not understand
constraints, `.res`, dimensions or units, fields, materials, geometry,
function spaces, meshes, element kernels, or product/runtime policy.

The repository is one root package named `methodus`. There are no subordinate contracts, scientific, or macro packages.

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
  and typed per-iteration telemetry; `NonlinearSolver` (`DenseNewton`, `NewtonKrylovSolver`)
  and `bdf_step_with` let BDF run either solver inside a step.
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
- Reusable directional Taylor-remainder, centered-difference, and callback-based
  complex-step reports over caller-supplied numerical evaluations.
- Convergence-order estimation from strictly refined positive samples, with
  adjacent orders and a log-space least-squares fit.
- Common-grid trajectory max/trapezoidal-L2 norms, tolerance-based
  solve-strategy agreement, and deterministic per-category work-budget checks.
- Malformed dimensions, non-monotone sample sequences, invalid tolerances,
  non-finite values, and floating-point overflow in computed discrepancies are
  refused rather than converted into passing evidence.

## Extraction

Created from Solverang history at numerical-core head `2bf2ee5` and renamed
directly, without a forwarding package or compatibility facade; Solverang
now consumes Methodus. Historical mixed CAD/scientific/JIT/pipeline code
remains in Git history only. This was an intentional API break with no
compatibility types, feature aliases, or forwarding packages.

## Dependency contract

- Krasis implements `NonlinearOperator`, `DaeOperator`, and `BlockNonlinearOperator` for coupled state.
- Finitum may implement `LinearOperator` for realized discrete operators.
- Solverang implements `LeastSquaresOperator` for its constraint graph.
- Methodus has no dependencies on any scientific-stack repository.

## Validation

Validated locally on 2026-09-01 (E7/SV1-D1 adjoint slice and the
Newton–Krylov driver):

- formatting and all-target checks passed;
- warnings-denied Clippy passed;
- 93 tests passed (64 unit, 29 integration), 0 failed;
- warnings-denied rustdoc passed; doctests passed (0 doctests present);
- `git diff --check` passed.

Prior validation (2026-08-31, SV2-B6 slice, 65 tests): formatting, locked
all-target checks, warnings-denied Clippy, warnings-denied rustdoc/doctests,
and `git diff --check` all passed.

Prior validation (2026-08-24, 31 tests): formatting, locked all-target
checks, warnings-denied Clippy, warnings-denied rustdoc/doctests, and
`git diff --check` all passed.

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
- `solve_blocks` (Gauss–Seidel/Jacobi) still builds dense per-block
  Jacobians by JVP column probing; a block-aware Newton–Krylov (per-block
  Krylov solves inside the staggered update) is not implemented.
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

Blockers: none.
