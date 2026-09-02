//! Physics-neutral numerical contracts and algorithms.
//!
//! Methodus consumes vectors and operator actions. It does not own meshes,
//! fields, units, constitutive laws, compiled kernels, or simulation state.

#![forbid(unsafe_code)]

mod adjoint;
mod bdf;
mod block;
mod context;
mod error;
mod krylov;
mod krylov_method;
mod least_squares;
mod linear;
mod nonlinear;
mod nullspace;
mod operator;
mod preconditioner;
mod sparse;
mod transpose;
mod verification;

pub use adjoint::{AdjointConfig, AdjointSolveReport, ResidualAcceptance, solve_adjoint};
pub use bdf::{
    AcceptedStep, BdfConfig, BdfOrder, BdfState, LocatedEvent, RejectedStep, StepOutcome, bdf_step,
};
pub use block::{
    Block, BlockLayout, BlockLinearOperator, BlockNonlinearOperator, BlockPreconditioner, BlockSpec,
};
pub use context::EvaluationContext;
pub use error::{NumericError, SolveError};
pub use krylov::{
    BiCgStabConfig, GmresConfig, GmresReport, MinresConfig, solve_bicgstab, solve_gmres,
    solve_minres,
};
pub use krylov_method::{KrylovMethod, KrylovMethodKind, KrylovSolveReport, solve_krylov};
pub use least_squares::{
    LeastSquaresConfig, LeastSquaresIteration, LeastSquaresOperator, LeastSquaresReport,
    solve_least_squares, verify_least_squares_jacobian,
};
pub use linear::{
    ConjugateGradientConfig, ConjugateGradientSymmetryPolicy, LinearIteration, LinearSolveReport,
    solve_conjugate_gradient,
};
pub use nonlinear::{
    BlockStrategy, IterationTrace, NewtonConfig, SolveReport, solve_blocks, solve_newton,
};
pub use nullspace::{ConstantModeProjector, NullspaceProjector};
pub use operator::{
    DaeOperator, Definiteness, LinearOperator, NonlinearOperator, OperatorProperties,
    OperatorStructureHint, OperatorSymmetry, Preconditioner, check_properties_consistency,
    verify_dae_jvp, verify_jvp,
};
pub use preconditioner::{
    BlockDiagonalPreconditioner, BlockLowerTriangularPreconditioner, CompositeBlockPreconditioner,
    LowerBlock,
};
pub use sparse::CsrMatrix;
pub use transpose::{
    TransposableOperator, TransposeOperator, TransposeSource, transpose_view,
    verify_adjoint_identity,
};
pub use verification::{
    ComparisonReport, ComparisonTolerance, ConvergenceOrderReport, ConvergenceSample,
    DerivativeCheckReport, DerivativeSample, TrajectoryNormReport, WorkBudget, WorkBudgetReport,
    WorkCount, check_centered_difference, check_complex_step, check_solve_strategy_agreement,
    check_taylor_remainder, check_work_budget, estimate_convergence_order, trajectory_error_norms,
};
