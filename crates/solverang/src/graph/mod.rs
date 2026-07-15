//! Constraint graph analysis: redundancy detection, DOF computation, and
//! closed-form pattern matching.
//!
//! Decomposition into independent clusters lives in the solve pipeline
//! ([`crate::pipeline`]); this module provides the analysis passes that the
//! pipeline and [`crate::system::ConstraintSystem`] diagnostics build on.

pub mod dof;
pub mod pattern;
pub mod redundancy;
