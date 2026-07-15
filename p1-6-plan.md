# P1.6 Implementation Plan: Use the Correct Active Parameter Set

## Goal

Optimization solvers must operate only on parameters that belong to the optimization problem,
rather than every free parameter in the `ParamStore`.

The active parameter set is the union of:

1. Parameters declared by the objective.
2. Parameters declared by equality constraints.
3. Parameters declared by inequality constraints.
4. Alive, free parameters with finite bounds when the selected algorithm honors bounds.

The implementation must also replace linear `ParamId` searches during derivative assembly with
`SolverMapping::param_to_col` lookups and reject derivative entries that violate their source's
`param_ids()` contract.

This work is limited to optimization. The whole-store mapping used by geometric solving,
decomposition, drag, and diagnostics remains unchanged.

## Current State

- `ParamStore::build_solver_mapping_for()` already builds a deduplicated mapping for a supplied
  set of parameter IDs and excludes fixed parameters.
- BFGS, BFGS-B, ALM, and both trust-region paths currently call
  `ParamStore::build_solver_mapping()`, which includes every alive, free parameter in the store.
- Gradient, Jacobian, and Hessian assembly repeatedly searches `param_ids` with
  `iter().position(...)`, making lookup `O(nnz x n)`.
- Derivative entries whose parameter is absent from the solver vector are silently dropped. This
  can conceal an invalid `Objective`, `Constraint`, or `InequalityFn` implementation.
- ALM assembles objective, equality, and inequality derivatives in several places, so its union
  must not conceal a contract violation by an individual child source.

## Proposed Design

### 1. Central optimization problem view

Add a private `optimization/active_set.rs` module containing an
`OptimizationProblemView` or equivalently named internal type. It should own:

- The single `SolverMapping` used for the solve.
- The objective's declared parameter set.
- One declared parameter set per equality constraint.
- One declared parameter set per inequality constraint.
- References or metadata needed by checked derivative assembly.

The view must not borrow the `ParamStore`; the solvers need to mutate the store while retaining
the mapping and declaration metadata.

### 2. Deterministic active-set construction

Collect candidate IDs in this order:

1. `objective.param_ids()` order.
2. Equality constraints in registration order, preserving each constraint's `param_ids()` order.
3. Inequality constraints in registration order, preserving each inequality's `param_ids()`
   order.
4. Alive, free, finite-bounded parameters in `ParamStore` order when bounds are honored.

Deduplicate by first occurrence, then pass the ordered vector to
`ParamStore::build_solver_mapping_for()`.

This gives deterministic solver columns while relying on the existing mapping implementation to
exclude fixed parameters.

Bound-only parameters are deliberately included for BFGS-B and bounded ALM. A registered bound
is part of the problem: the solver must project an initially infeasible parameter even when the
objective and nonlinear constraints do not otherwise reference it.

Algorithms use the builder as follows:

| Algorithm | Declared sources | Include finite-bound-only parameters |
| --- | --- | --- |
| BFGS | Objective | No |
| BFGS-B | Objective | Yes |
| ALM | Objective, equalities, inequalities | Yes |
| Trust region | Objective | No |

`ConstraintSystem::optimize()` continues to reject algorithm/problem combinations that cannot
honor the registered bounds or constraints.

### 3. Source-local derivative contracts

Every derivative entry must refer to a parameter declared by the same object that emitted it:

- Objective gradient entries must appear in that objective's `param_ids()`.
- Both endpoints of an objective Hessian entry must appear in that objective's `param_ids()`.
- Equality Jacobian entries must appear in that equality constraint's `param_ids()`.
- Inequality Jacobian entries must appear in that inequality's `param_ids()`.

Validation must use the source-local declaration set, not the global union. For example, a
constraint may not emit a derivative for a parameter merely because the objective declared that
parameter.

A parameter can be declared but absent from the solver mapping because it is fixed. Such a
derivative entry is valid and is omitted from free-variable assembly. Therefore the assembly
order is:

1. Check membership in the emitting source's declared set.
2. Reject the entry if it is undeclared.
3. Otherwise look it up in `SolverMapping::param_to_col`.
4. Assemble it when a free-variable column exists; skip it when the declared parameter is fixed.

Duplicate IDs in `param_ids()` remain legal and are deduplicated in the mapping. Stale
generational-ID validation is outside this item and remains part of the later lifecycle/API work.

### 4. Explicit contract failure

Add an `OptimizationStatus::InvalidProblemDefinition { reason: String }` variant. The diagnostic
should identify:

- The emitting source and its name or ID.
- Whether the failure came from a gradient, equality Jacobian, inequality Jacobian, or Hessian.
- The undeclared `ParamId`.

Checked assembly helpers should return an internal `DerivativeContractError` and convert it to
the public optimization status at solver boundaries.

Contract failure must not be reported as convergence, line-search failure, or numerical
divergence. Snapshot the initial active values and restore them before returning an invalid
problem result, including errors discovered after trial evaluations begin.

### 5. Fallible scalar evaluation for BFGS internals

The `Objective` trait currently returns derivative vectors directly, so it cannot itself report a
contract error. Introduce a small private scalar-evaluation interface for the BFGS core and line
search:

- Evaluate a scalar value.
- Assemble a checked dense gradient against a supplied `SolverMapping`.

Provide two internal adapters:

- A direct-objective adapter for BFGS, BFGS-B, and approximate trust region.
- An augmented-Lagrangian adapter that assembles the objective and child constraint derivatives
  while retaining source-local validation.

Keep this interface private and limited to the active-set work. It is not the public residual/JIT
evaluator abstraction planned in P1.7.

## Implementation Sequence

### Step 1: Add regression tests

Add tests before modifying solver behavior. Cover:

- An unrelated, unbounded free parameter is excluded and remains unchanged during BFGS.
- The ALM active set contains the union of objective, equality, and inequality parameters.
- A finite-bounded-only parameter is included and projected by BFGS-B.
- Duplicate declarations yield one solver column with stable first-occurrence ordering.
- A fixed, declared parameter is excluded from solver columns without causing a contract error.
- An objective gradient referencing an undeclared parameter is rejected.
- An equality Jacobian referencing an undeclared parameter is rejected.
- An inequality Jacobian referencing an undeclared parameter is rejected.
- A Hessian whose first or second endpoint is undeclared is rejected.
- A constraint entry remains invalid when its parameter is declared globally by the objective but
  not locally by that constraint.
- Invalid derivative rejection restores the initial active parameter values.

Test the active-set builder directly for mapping membership and ordering. Use solver-level tests
for observable projection, contract status, and store-restoration behavior.

### Step 2: Add the active-set module and status

- Add `optimization/active_set.rs` and register it as a private module.
- Implement ordered union construction and per-source declaration metadata.
- Add `InvalidProblemDefinition` to `OptimizationStatus`.
- Add a common helper for constructing an invalid-problem `OptimizationResult` without
  duplicating sentinel KKT fields across solvers.

### Step 3: Refactor BFGS and line search

- Preserve the public `BfgsSolver::solve(...)` signature.
- Have it build an objective-only problem view and delegate to a private mapping-aware core.
- Change `dense_gradient()` to accept `&SolverMapping` and return a checked result.
- Replace `param_ids.iter().position(...)` with `mapping.param_to_col.get(&pid)`.
- Pass the mapping through line-search evaluation rather than passing only `param_ids`.
- Continue using `mapping.col_to_param` for vector extraction and store writes.
- Propagate contract errors distinctly from ordinary line-search failures.

### Step 4: Refactor BFGS-B

- Preserve the public `BfgsBSolver::solve(...)` signature.
- Build the objective plus finite-bound active set.
- Extract bounds in `mapping.col_to_param` order.
- Reuse the mapping-aware BFGS core utilities and line search.
- Ensure a bound-only parameter is projected while an unrelated unbounded parameter is omitted.

### Step 5: Refactor ALM

- Build one active mapping from the objective, equalities, inequalities, and finite bounds.
- Determine BFGS-B inner dispatch by checking finite bounds only among active mapped parameters.
- Pass the same mapping into every inner solve rather than rebuilding it on each outer iteration.
- Replace linear parameter searches in augmented-gradient and Lagrangian-stationarity assembly
  with `param_to_col` lookups.
- Route augmented-Lagrangian gradient construction through checked, source-local assembly.
- Keep multiplier vector ordering and `MultiplierStore` construction unchanged; only the
  derivative-to-column assembly uses the solver mapping.
- On a contract error, stop before multiplier updates and restore the solve's initial active
  values.

### Step 6: Refactor trust region

- Preserve both public trust-region entry points.
- Build an objective-only active mapping.
- Use checked mapping-aware gradient assembly in both paths.
- Change exact Hessian construction to use `param_to_col` for both endpoints.
- Reject undeclared Hessian endpoints instead of silently skipping them.
- Do not change the trust-region model, dense Hessian policy, subproblem algorithms, or
  convergence behavior in this work.

### Step 7: Integrate with `ConstraintSystem::optimize()`

- Retain compatibility validation before parameters are mutated.
- Use the centralized active-set/bound classification when dispatching optimization solvers.
- Preserve multiplier fingerprinting and warm-start invalidation.
- Ensure direct public solver calls and `ConstraintSystem::optimize()` exercise the same active-set
  and derivative-contract logic.

### Step 8: Update API documentation

Update the documentation on:

- `Objective::param_ids()` and `Objective::gradient()`.
- `ObjectiveHessian::hessian_entries()`.
- `Constraint::param_ids()` and `Constraint::jacobian()`.
- `InequalityFn::param_ids()` and `InequalityFn::jacobian()`.

State that every emitted derivative parameter must be declared by that source and that fixed
declared parameters are legal but absent from the free-variable mapping.

## Failure Shields and Scope Boundaries

- Do not equate absence from the active mapping with an undeclared parameter; fixed parameters
  are declared but intentionally unmapped.
- Do not validate ALM derivatives only against the global union; that would mask malformed child
  sources.
- Build one mapping per solve. Rebuilding it in every ALM inner iteration would lose much of the
  architectural and performance benefit.
- Preserve `ParamStore::build_solver_mapping()` for non-optimization paths.
- Preserve public solver entry points; keep mapping-aware cores internal.
- Do not alter objective/constraint trait return types in this item.
- Do not combine this work with stale-ID lifecycle validation, warning cleanup, JIT evaluator
  restructuring, sparse factorization caching, or trust-region mathematical repairs.
- Do not change duplicate derivative-entry semantics as part of this work.

## Validation

Run the focused gates first:

```sh
cargo fmt --check
cargo test -p solverang --all-features --test optimization_solvers
cargo test -p solverang --all-features system::tests
cargo test -p solverang --no-default-features --lib
```

Then run the full workspace gate:

```sh
cargo test --workspace --all-features
```

The pre-change targeted optimization baseline is 40 passing tests with all features. Existing
compiler warnings are outside this item and should remain informational rather than turning this
PR into a warning-cleanup project.

## Completion Criteria

P1.6 is complete when:

- Every optimization solver uses a problem-scoped active mapping.
- Bound-aware solvers include finite-bound-only parameters.
- All derivative-to-column lookup in the optimization paths is constant-time through
  `SolverMapping::param_to_col`.
- The same mapping is used throughout an ALM solve, including inner solves and KKT assembly.
- Undeclared gradient, Jacobian, and Hessian parameters produce an explicit invalid-problem
  result and do not leave partial parameter mutations behind.
- Fixed declared parameters remain valid and excluded from the solver vector.
- Direct solver APIs and `ConstraintSystem::optimize()` behave consistently.
- Focused, minimal-feature, and full-workspace validation gates pass.
