# Methodus status

Updated: 2026-08-21
Branch: `master`
Milestone: SV0-B1 reusable numerical verification checkers

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

Validated locally on 2026-08-21:

- formatting and locked all-target checks passed;
- warnings-denied Clippy passed;
- 29 tests passed (20 unit, 9 integration), 0 failed;
- warnings-denied rustdoc and doctests passed;
- `git diff --check` passed.

## Next concrete work

1. Integrate these checkers through Finitum/Krasis and Sinbad SV0-B3/B4/B5
   without moving campaign policy into Methodus.
2. Add additional Krylov methods only when representative realized systems require them.
3. Promote the dense least-squares baseline only from representative Solverang
   constraint systems and independent numerical checks.
4. Replace dense Newton only after representative compiled systems define
   scaling and performance requirements.

Blockers: none.
