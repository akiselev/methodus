# R9 numerical-contract and execution migration

Solverang no longer requires an absolute developer-local Sinbad checkout to compile its DAE/operator contracts or its JIT surface.

## Ownership

- `solverang-contracts` owns portable sparse/operator/Jacobian/DAE/integrator contracts.
- `solverang` owns numerical algorithms and the batteries-included high-level solver APIs.
- Resolvent owns mathematical/scientific compiler semantics and produces solver-facing operator and execution-plan artifacts without depending on Solverang.
- Malleus owns pointwise/kernel compilation and finite-precision code generation.
- Sinbad owns simulation orchestration, field execution, and physics composition.

The historical dependency name `numeric-contracts` remains temporarily as a Cargo alias to `solverang-contracts`, avoiding source churn while downstream Sinbad code migrates. It does not imply Sinbad ownership.

## Malleus boundary

Solverang's `jit` feature now imports the real public `malleus` crate at a pinned coordinated R9 commit. `crate::jit` remains only a source-compatibility re-export path for Solverang callers; there is no copied JIT implementation and no `sinbad-malleus` package.

The former `crates/malleus-compat` workspace member has been removed. This is deliberate: a compatibility crate that did not depend on Malleus obscured ownership and allowed the copied implementation to drift from the actual execution compiler.

Malleus can independently enable its `resolvent` feature to consume frozen Resolvent execution plans. Solverang does not need to depend on Resolvent to use Malleus, preserving the intended compiler -> execution -> numerical-solver separation without a dependency cycle.
