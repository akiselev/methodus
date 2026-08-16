# R9 numerical-contract migration

Solverang no longer requires an absolute developer-local Sinbad checkout to compile its DAE/operator contracts.

## Ownership

- `solverang-contracts` owns portable sparse/operator/Jacobian/DAE/integrator contracts.
- `solverang` owns numerical algorithms and the batteries-included high-level solver APIs.
- Resolvent owns mathematical/scientific compiler semantics and produces solver-facing operator artifacts without depending on Solverang.
- Anvil remains an execution backend rather than a solver semantic dependency.
- Sinbad owns simulation orchestration and physics composition.

The historical dependency name `numeric-contracts` remains temporarily as a Cargo alias to `solverang-contracts`, avoiding source churn while downstream Sinbad code migrates. It does not imply Sinbad ownership.

## JIT compatibility

`crates/anvil-compat` preserves Solverang's historical JIT-facing API without `/home/dev/sinbad/crates/anvil`. It is a deterministic compatibility/fallback implementation, not a claim to replace the optimized Anvil compiler. Once Anvil is consumable as a portable repository/package dependency, Solverang can point the same public seam at it without changing the numerical contracts.
