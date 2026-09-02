# Methodus

Methodus is a consumer-neutral numerical-methods library extracted from
Solverang with shared Git history. It accepts flat vectors and in-place operator
actions; it has no knowledge of constraints, source languages, units, fields,
materials, geometry, meshes, finite elements, compiled kernels, or product
orchestration.

The repository contains exactly one Rust package and one public API. There are no contract facades or scientific companion crates.

## Owned interfaces

- `LinearOperator` and `Preconditioner` for matrix-free linear actions, with explicit
  symmetric/nonsymmetric/unknown metadata.
- `NonlinearOperator` for residuals and Jacobian-vector products.
- `LeastSquaresOperator` for rectangular residuals and row-major Jacobians.
- `DaeOperator` for `F(t, y, ydot) = 0`, consistent initialization, and event values.
- `BlockLayout` plus block-aware operator and preconditioner traits.
- Canonical `CsrMatrix` storage with input-order-independent duplicate summation for small assembled baselines and interchange.

All operator outputs are written into caller-provided slices. `EvaluationContext` carries explicit numerical execution policy without importing runtime state.

## Implemented algorithms

- Dense correctness-baseline Newton solves with backtracking, and an inexact
  Newton–Krylov driver over matrix-free Jacobian actions with Eisenstat–Walker
  or constant forcing, a preconditioner factory hook, and a nullspace-projector
  hook.
- Conjugate gradient, MINRES, restarted GMRES, and BiCGSTAB over
  `LinearOperator`/`Preconditioner`, selected through one `KrylovMethod`
  value with each solver's property-aware admission preserved.
- Adjoint solves `Aᵀ λ = g` through symmetric-delegated or explicit
  (`TransposableOperator`) transposes with true-residual acceptance.
- Deterministic damped Gauss-Newton/Levenberg-Marquardt-style least-squares
  solves and centered-difference Jacobian verification.
- Monolithic, block Gauss-Seidel, and block Jacobi coupling policies.
- Block-diagonal and block-lower-triangular preconditioner actions.
- BDF1 and variable-step BDF2 stepping, adaptive rejection, checkpointable step-size history, consistent initialization, and zero-crossing events, with the implicit solve pluggable through `NonlinearSolver` (`bdf_step_with`).
- Centered-difference verification for nonlinear and DAE Jacobian-vector products.
- Consumer-neutral verification utilities for directional Taylor remainders,
  centered differences, callback-based complex-step checks, fitted convergence
  order, trajectory error norms, solve-strategy agreement, and deterministic
  work budgets.

Verification thresholds remain caller policy. Methodus validates numerical
inputs and returns measurements (or an explicit comparison/budget decision);
it does not assign scientific meaning or promote support claims. Complex-step
callbacks provide the imaginary response so consumers may use their own
complex or dual scalar representation without adding one to Methodus.

The dense Newton factorization is deliberately a correctness baseline, not the intended large-system backend; `solve_newton_krylov` is the matrix-free path. The Krylov solvers consume `LinearOperator` actions and refuse what their declared operator properties do not admit: conjugate gradient refuses declared-nonsymmetric actions and requires an explicit caller assumption when symmetry is unknown, MINRES requires a `Symmetric` declaration, GMRES and BiCGSTAB admit any declaration. Scalable sparse factorizations and multigrid can be added behind the same contracts as concrete systems require them.

## Example

```rust
use methodus::{
    EvaluationContext, NewtonConfig, NonlinearOperator, NumericError, solve_newton,
};

struct SquareRootOfTwo;

impl NonlinearOperator for SquareRootOfTwo {
    fn dimension(&self) -> usize { 1 }

    fn residual(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state[0] * state[0] - 2.0;
        Ok(())
    }

    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = 2.0 * state[0] * direction[0];
        Ok(())
    }
}

let report = solve_newton(
    &SquareRootOfTwo,
    &EvaluationContext::reproducible(),
    &[1.5],
    &NewtonConfig::default(),
)?;
assert!(report.converged);
assert!((report.state[0] - std::f64::consts::SQRT_2).abs() < 1.0e-10);
# Ok::<(), methodus::SolveError>(())
```

Solverang implements generalized constraint systems over the least-squares
contract. Krasis implements the nonlinear, DAE, and block traits for coupled
simulation state. Finitum may implement linear actions for realized discrete
operators. Methodus remains independent of all three repositories.

## Validation

```text
cargo fmt --all -- --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

See `STATUS.md` for the current verified surface and next work.
