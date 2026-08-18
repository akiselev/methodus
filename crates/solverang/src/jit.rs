//! JIT compilation — re-export shim onto the standalone `malleus` crate.
//!
//! The JIT (scalar opcode IR + Cranelift codegen + interpreter oracle) was
//! re-homed into Malleus during M1. Solverang no longer owns or copies this code;
//! its `jit` feature depends directly on the public Malleus repository.
//!
//! This module re-exports Malleus' public JIT surface under the historical
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
