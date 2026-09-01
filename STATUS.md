# Methodus status

Updated: 2026-08-31
Branch: `master`
Milestone: SV2-B6 MINRES/GMRES, nullspace projection, and block
preconditioner contracts (previously: SV0-B1 checkers, SV1-C5/D1
transpose/adjoint)

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

Validated locally on 2026-08-31 (SV2-B6 slice):

- formatting and locked all-target checks passed;
- warnings-denied Clippy passed;
- 65 tests passed (53 unit, 12 integration), 0 failed;
- warnings-denied rustdoc passed; doctests passed (0 doctests present);
- `git diff --check` passed, including the two new source files.

Prior validation (2026-08-24, 31 tests): formatting, locked all-target
checks, warnings-denied Clippy, warnings-denied rustdoc/doctests, and
`git diff --check` all passed.

## Known limits (updated after the SV2-B6 slice)

- `OperatorProperties` (symmetry, definiteness, nullspace dimension, and a
  dense/block `OperatorStructureHint` with a saddle-point flag) landed under
  GX-D2 before this slice; the three-valued-symmetry-only limit an earlier
  2026-08-30 audit recorded here is resolved.
- The transpose contract still accepts only `Symmetric` operators via
  delegation (`TransposableOperator` covers explicit nonsymmetric
  transposes), and Finitum's matrix-free operator still never declares
  either, so `transpose_view` remains unusable on the current Finitum
  realization path — unchanged by this slice, tracked upstream.
- Block preconditioning is limited to block-diagonal and block-lower-
  triangular composition (`BlockDiagonalPreconditioner`,
  `BlockLowerTriangularPreconditioner`, `CompositeBlockPreconditioner`); no
  algebraic multigrid, incomplete factorization, or Schur-complement
  *computation* exists — only the composition contract. A caller must supply
  its own approximate Schur-complement/pressure-mass block preconditioner.
- MINRES's nullspace-projection hook ships one bounded reference
  implementation, `ConstantModeProjector` (a single constant mode over one
  contiguous coordinate range). Multi-dimensional nullspaces (e.g.
  rigid-body modes) need a caller-supplied `NullspaceProjector`; no reference
  implementation exists for that shape.

## Next concrete work

1. Wire MINRES/GMRES into Sinbad's `SolvePolicy`/`LinearAlgorithm`
   admission once Sinbad picks up SV2-B6 (Methodus does not own that
   selection policy).
2. Promote the dense least-squares baseline only from representative Solverang
   constraint systems and independent numerical checks.
3. Replace dense Newton only after representative compiled systems define
   scaling and performance requirements.

Blockers: none.
