# Methodus status

Updated: 2026-09-01
Branch: `master`
Milestone: W7 lane 3 — E7/SV1-D1 adjoint solve on nonsymmetric operators
(previously: SV2-B6 MINRES/GMRES, nullspace projection, and block
preconditioner contracts; SV0-B1 checkers; SV1-C5 transposes)

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

## SV2-B6 Krylov slice

MINRES and GMRES over the existing `LinearOperator`/`Preconditioner` traits,
mirroring conjugate gradient's typed-refusal machinery (typed refusal codes,
never panics, never a silent fallback). MINRES admits any declared-
`Symmetric` operator regardless of definiteness (indefinite included — the
saddle-point Stokes case) and refuses `Nonsymmetric`/`Unknown` declarations
outright, with no caller-assumption escape hatch (unlike CG). GMRES admits
any declared symmetry and refuses only a non-square operator. A typed
`NullspaceProjector` hook lets MINRES proceed on an operator with a declared
positive nullspace dimension that CG refuses outright: the projector is
applied to every Krylov vector and to the returned solution, converging to
the pseudo-solution of a singular consistent system.
`ConstantModeProjector` is the bounded reference implementation (one
constant mode over a contiguous coordinate range, e.g. Stokes'
constant-pressure mode). `CompositeBlockPreconditioner` composes
caller-supplied per-block `Preconditioner`s block-diagonally — the bounded
reference contract for Schur-complement/pressure-mass saddle-point
preconditioning; Methodus does not implement Schur-complement approximation
itself, only the composition contract. Both solvers report deterministic,
bit-reproducible `LinearIteration` traces (typed structs, no debug strings).
Twelve acceptance tests cover the refusal matrix for both solvers, MINRES on
a symmetric-indefinite saddle-point fixture agreeing with a hand-derived
solution, MINRES/CG agreement on an SPD fixture, GMRES on a nonsymmetric
fixture with a known solution, nullspace-projection correctness (a singular
consistent system solved to the pseudo-solution), telemetry determinism (two
runs byte-identical), and GMRES restart behavior.

## SV1-C5/D1 transpose and adjoint slice

`TransposeOperator` adapts symmetric-declared linear operators into their
algebraic transpose by exact delegation (A = Aᵀ under the admitted
declaration); nonsymmetric or evidence-free declarations are refused because
the matrix-free contract cannot compute genuine transposes without column
access. `verify_adjoint_identity` checks <Au,v> == <u,transpose v> on caller
probes; `transpose_view` is the entry point adjoint solves compose with.
Two acceptance tests: delegation equality plus identity satisfaction, and
refusal of Nonsymmetric/Unknown declarations plus dimension-mismatch probes.

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
- `NullspaceProjector` trait plus the bounded reference `ConstantModeProjector` (one constant mode
  over a contiguous coordinate range).
- `CompositeBlockPreconditioner`: block-diagonal composition of caller-supplied per-block
  `Preconditioner`s, the bounded reference implementation for Schur-complement/pressure-mass
  saddle-point block preconditioning.
- Invariant-validated deserialization for CSR matrices, block layouts, preconditioners, and BDF history.
- Dense Newton correctness baseline with backtracking and residual traces.
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

- Created from Solverang history at numerical-core head `2bf2ee5` and renamed
  directly, without a forwarding package or compatibility facade.
- Solverang retains its history and now consumes Methodus while owning the
  generalized constraint engine and 2-D/3-D geometry vocabularies.
- Historical mixed CAD/scientific/JIT/pipeline code remains available in Git
  history but was not restored into Methodus.

This is an intentional API break. No compatibility types, feature aliases, or forwarding packages remain.

## Dependency contract

- Krasis implements `NonlinearOperator`, `DaeOperator`, and `BlockNonlinearOperator` for coupled state.
- Finitum may implement `LinearOperator` for realized discrete operators.
- Solverang implements `LeastSquaresOperator` for its constraint graph.
- Methodus has no dependencies on any scientific-stack repository.

## Validation

Validated locally on 2026-09-01 (E7/SV1-D1 adjoint slice):

- formatting and all-target checks passed;
- warnings-denied Clippy passed;
- 83 tests passed (62 unit, 21 integration), 0 failed;
- warnings-denied rustdoc passed; doctests passed (0 doctests present).

Prior validation (2026-08-31, SV2-B6 slice, 65 tests): formatting, locked
all-target checks, warnings-denied Clippy, warnings-denied rustdoc/doctests,
and `git diff --check` all passed.

Prior validation (2026-08-24, 31 tests): formatting, locked all-target
checks, warnings-denied Clippy, warnings-denied rustdoc/doctests, and
`git diff --check` all passed.

## Known limits (updated after the SV2-B6 slice)

- `OperatorProperties` (symmetry, definiteness, nullspace dimension, and a
  dense/block `OperatorStructureHint` with a saddle-point flag) landed under
  GX-D2 before this slice; the three-valued-symmetry-only limit an earlier
  2026-08-30 audit recorded here is resolved.
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
4. W7 lane 3, next package: an inexact Newton–Krylov driver over a
   matrix-free JVP operator with a pluggable `KrylovMethod`, a preconditioner
   hook, and a nullspace-projector hook (today `solve_newton` builds a dense
   Jacobian by JVP column probing), needed by Krasis's N-block DAE with Newton
   inside BDF (batch P) and SC-W3.
5. SC composition (design `sinbad/ARCHITECTURE.md` §8–9; the SV7-F3 subset
   pulled forward under its own ID): fixed-point acceleration over `&[f64]`
   iterate sequences (relaxation, Aitken; IQN later). `solve_blocks`
   GS/Jacobi and `CompositeBlockPreconditioner` are reused as they are.
   Methodus never sees instance names, outputs, or connector vocabulary;
   Sinbad resolves schedules and convergence targets to block ids.

Blockers: none.
