//! Physics-neutral numerical contracts and algorithms.
//!
//! Solverang consumes vectors and operator actions. It does not own meshes,
//! fields, units, constitutive laws, compiled kernels, or simulation state.

#![forbid(unsafe_code)]

mod bdf;
mod block;
mod context;
mod error;
mod linear;
mod nonlinear;
mod operator;
mod preconditioner;
mod sparse;

pub use bdf::{
    AcceptedStep, BdfConfig, BdfOrder, BdfState, LocatedEvent, RejectedStep, StepOutcome, bdf_step,
};
pub use block::{
    Block, BlockLayout, BlockLinearOperator, BlockNonlinearOperator, BlockPreconditioner, BlockSpec,
};
pub use context::EvaluationContext;
pub use error::{NumericError, SolveError};
pub use linear::{
    ConjugateGradientConfig, ConjugateGradientSymmetryPolicy, LinearIteration, LinearSolveReport,
    solve_conjugate_gradient,
};
pub use nonlinear::{
    BlockStrategy, IterationTrace, NewtonConfig, SolveReport, solve_blocks, solve_newton,
};
pub use operator::{
    DaeOperator, LinearOperator, NonlinearOperator, OperatorSymmetry, Preconditioner,
    verify_dae_jvp, verify_jvp,
};
pub use preconditioner::{
    BlockDiagonalPreconditioner, BlockLowerTriangularPreconditioner, LowerBlock,
};
pub use sparse::CsrMatrix;
