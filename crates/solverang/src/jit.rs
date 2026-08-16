//! JIT compilation — re-export shim onto the `sinbad-malleus` crate.
//!
//! The JIT (scalar opcode IR + Cranelift codegen + interpreter oracle) was
//! **re-homed** into the standalone `sinbad-malleus` crate during M1 (a pure,
//! faithful port — no behavior change). solverang no longer owns this code; it
//! depends on malleus behind the `jit` feature.
//!
//! This module re-exports malleus's public JIT surface under the historical
//! `crate::jit::…` path so that the `#[auto_jacobian]` macro, the [`Problem`]
//! trait's `lower_to_compiled_constraints`, and [`JITSolver`] keep compiling
//! unchanged. New code may also `use malleus::…` directly.
//!
//! [`Problem`]: crate::Problem
//! [`JITSolver`]: crate::JITSolver
pub use malleus::{
    jit_available, CompiledConstraints, CompiledNewtonStep, ConstraintOp, HessianEntry,
    JITCompiler, JITConfig, JITError, JITFunction, JacobianEntry, JitMode, OpcodeEmitter, Reg,
    ValidationError,
};
