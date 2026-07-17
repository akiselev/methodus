//! JIT compilation — re-export shim onto the `sinbad-anvil` crate.
//!
//! The JIT (scalar opcode IR + Cranelift codegen + interpreter oracle) was
//! **re-homed** into the standalone `sinbad-anvil` crate during M1 (a pure,
//! faithful port — no behavior change). solverang no longer owns this code; it
//! depends on anvil behind the `jit` feature.
//!
//! This module re-exports anvil's public JIT surface under the historical
//! `crate::jit::…` path so that the `#[auto_jacobian]` macro, the [`Problem`]
//! trait's `lower_to_compiled_constraints`, and [`JITSolver`] keep compiling
//! unchanged. New code may also `use anvil::…` directly.
//!
//! [`Problem`]: crate::Problem
//! [`JITSolver`]: crate::JITSolver
pub use anvil::{
    jit_available, CompiledConstraints, CompiledNewtonStep, ConstraintOp, HessianEntry,
    JITCompiler, JITConfig, JITError, JITFunction, JacobianEntry, JitMode, OpcodeEmitter, Reg,
    ValidationError,
};
