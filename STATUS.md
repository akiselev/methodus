# Methodus status

2026-09-29 SC-W3 preconditioners/acceleration landed (W9 lane): composite block
Gauss-Seidel/triangular preconditioners beyond Jacobi, and iterate-sequence
acceleration (fixed relaxation, Aitken) over `&[f64]` fixed-point sequences,
wired as an opt-in `solve_blocks` option. See "SC-W3 composite block
preconditioners" and "SC-W3 iterate-sequence acceleration" below.

2026-09-18 bounded implementation accepted: SC-W3 optional owner-provided
linear/nonlinear/DAE diagonals; `JacobiFactory` consumes the existing block
diagonal preconditioner and refuses missing/zero/nonfinite diagonals without
probing or fallback. Owner gate passed: 100 tests; final consumer acceptance
passed (210 tests across 35 targets, documented external-fixture retry).

Updated: 2026-09-29
Branch: `master`
Milestone: SC-W3 composite block preconditioners and acceleration (this lane);
existing numerical algorithms and execution behavior unchanged (default
`NewtonConfig`/`solve_blocks` output is bit-for-bit identical; acceleration
is opt-in via a new `None`-default field). Compatibility, stated exactly:
`NewtonConfig` gained a public field, so an exhaustive struct literal must
add it — a source change in consumers (Sinbad's `resolve_newton`, live
sinbad `113900e`). The serialized bytes of every existing value are
unchanged: `acceleration` is skipped while `None`
(`skip_serializing_if = "Option::is_none"`), asserted in `nonlinear.rs`
against the exact pre-field literal of `NewtonConfig::default()`; an old
payload without the key deserializes to `None`, and an Aitken payload
round-trips. (The first cut of this lane, `24f9c7d`, serialized an extra
`"acceleration":null`, which would have changed the `NewtonConfig` hash in
Krasis's consistent-initialization identities; fixed before publication.)

## SC-W3 composite block preconditioners

`BlockGaussSeidelPreconditioner` (`preconditioner.rs`): an owner-supplied
`BlockLayout`, one owner-supplied diagonal-block `Preconditioner` per block
(the existing `BlockDiagonalPreconditioner`/Jacobi, a dense per-block solve,
or an owner inner Krylov solve), and owner-supplied off-diagonal
`BlockCouplingAction`s (row/column block actions, not dense storage). One
`GaussSeidelSweep::Forward` application is a block-lower-triangular solve;
`Backward` is block-upper-triangular; `Symmetric` runs both sweeps in
sequence. Same refusal discipline as `JacobiFactory`: no probing, no zero
replacement, typed refusals for a diagonal-solve count/dimension mismatch, an
out-of-range or self-coupling block index, and nonfinite input.

Symmetry is declared, not assumed. `Forward`/`Backward` always declare
`Nonsymmetric`. The symmetric sweep's action is `(D+U)⁻¹ D (D+L)⁻¹`, which
is symmetric only when every `D_i` is symmetric AND `U = Lᵀ`, so `Symmetric`
is declared only when every diagonal solve declares `Symmetric` and the
coupling set is closed under transposition by owner declaration: each
supplied block pair `(i, j)` and its mirror `(j, i)` are supplied by exactly
one action each, both returning `BlockCouplingAction::transpose_of` = the
other pair (a new defaulted trait method; `None` declares nothing, and a
declaration naming any pair other than the mirror is refused at
construction). A symmetric sweep with a `Nonsymmetric` diagonal solve
declares `Nonsymmetric`; everything else (undeclared, one-directional or
duplicated couplings, an `Unknown` inner solve) declares `Unknown` — never a
`Symmetric` it cannot justify. (The first cut, `24f9c7d`, checked only the
inner solves and declared a lower-only coupling set — whose action is the
nonsymmetric `(D+L)⁻¹` — `Symmetric`, which conjugate gradient then admitted;
fixed before publication.) `Preconditioner`
gained a defaulted `symmetry()` method (`Unknown` unless overridden, so every
preexisting preconditioner is unaffected); `solve_conjugate_gradient` and
`solve_minres` now refuse a preconditioner explicitly declared
`Nonsymmetric` (an addition — no preexisting preconditioner declares this,
so no existing caller is affected; the MINRES twin of the refusal was
missing in `24f9c7d`, proven by
`minres_refuses_a_preconditioner_declared_nonsymmetric`).
`BlockDiagonalPreconditioner` now declares `Symmetric`;
`BlockLowerTriangularPreconditioner` now declares `Nonsymmetric`.

Proof: exact one-application solve of a block-triangular system for both
`Forward` and `Backward` (`preconditioner.rs` tests); measured GMRES
iteration counts on a 2-block and a 3-block coupled dense system, asserted
exactly — two blocks `(none, jacobi, gs) = (4, 2, 2)`, three blocks
`(5, 3, 2)` — showing both preconditioners reduce iterations over none, and
the exact per-block solve is never worse than elementwise Jacobi; the
symmetry declaration rule
(`gauss_seidel_symmetry_declaration_follows_sweep_inner_solves_and_coupling_closure`:
a lower-only symmetric sweep is numerically nonsymmetric and declared
`Unknown`, a declared transposed pair is numerically symmetric and declared
`Symmetric` and accepted by conjugate gradient, the forward sweep is refused
by conjugate gradient); a 3-block, length-2 declared-transposed fixture whose
`M⁻¹e_i·e_j` equals `M⁻¹e_j·e_i` over all unit vectors to 1e-12 and whose
undeclared twin is `Unknown`
(`symmetric_gauss_seidel_over_declared_transposed_couplings_is_numerically_symmetric`);
the symmetric sweep's action against the closed form `(D+U)⁻¹ D (D+L)⁻¹` on
a 3-block, length-2 system coupled in both directions, which also proves the
backward pass is load-bearing (`symmetric_gauss_seidel_action_matches_its_closed_form`);
the contradictory-declaration refusal; and every listed refusal.

## SC-W3 iterate-sequence acceleration

`acceleration.rs`: `FixedPointOperator` (one `G(x)` evaluation over `&[f64]`)
and `accelerate_fixed_point` drive `x_{k+1} = x_k + factor*(G(x_k) - x_k)`
under `AccelerationMethod::FixedRelaxation` (constant factor) or `::Aitken`
(vector Δ², factor recomputed each iteration from the two most recent
fixed-point residuals). Typed report: per-iteration residual norm and
relaxation factor used. Typed refusals, never NaN or a silent unconverged
iterate: nonfinite evaluation/state (`NumericError`), a degenerate Aitken
denominator (`SolveError::AccelerationBreakdown`), and exhausting
`max_iterations` without meeting tolerance (`SolveError::NotConverged`).

Integrated into `solve_blocks` as `NewtonConfig.acceleration:
Option<AccelerationMethod>` (`#[serde(default, skip_serializing_if =
"Option::is_none")]`, `None` by default): when
set, a `GaussSeidel`/`Jacobi` outer iteration relaxes its partitioned Newton
correction by the accelerator instead of backtracking (the correction vector
itself is the fixed-point residual; no extra `G` evaluation). Refused with
`Monolithic` (no partitioned fixed-point sequence exists there). No new
entry point; every existing `solve_blocks` call behaves identically because
the field defaults to `None` (exhaustive `NewtonConfig` literals must name
the field; see Milestone).

Proof: a contractive 2-block linear fixed point where Aitken converges in
provably fewer iterations than plain iteration, counts asserted exactly
(`(plain, aitken) = (111, 8)`); a divergent full-step iteration refused as
`NotConverged`; a degenerate-denominator refusal; a nonfinite-evaluation
refusal; relaxation-factor bounds `(0, 1]` refused outside that range; and,
in `solve_blocks`, a strongly-coupled (0.99) 2-block system that an
unaccelerated `GaussSeidel` fails to converge on (existing test) now
converges under Aitken acceleration, plus the `Monolithic` refusal.

Out of scope (unchanged from the brief): IQN, multiplier/Nitsche interfaces,
multilevel/ILU preconditioning, and Schur-complement/pressure-mass block
*computation* (only the composition contract exists, per "Known limits").

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

## Dependency contract

Krasis implements `NonlinearOperator`/`DaeOperator`/`BlockNonlinearOperator`;
Finitum implements `LinearOperator` (and `TransposableOperator` where it has a
transpose); Solverang implements `LeastSquaresOperator`. Methodus depends on no
scientific-stack repository.

## Validation

- `cargo test -p methodus --lib`: 79 tests passed, none failed or ignored.
- `cargo test -p methodus --test <name>` for each of `adjoint` (9),
  `coupling_strategies` (4), `newton_krylov` (12), `time_integration` (7),
  `time_restart_events` (2), `transpose` (3): all passed. 116 tests total.
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
