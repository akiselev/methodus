//! Clock abstraction for solve diagnostics.
//!
//! Solverang is often embedded in host applications that own determinism,
//! tracing, and timing policy. The default [`StdClock`] keeps the standalone
//! crate ergonomic, while [`SolveClock`] lets hosts provide their own elapsed
//! time source instead of requiring direct wall-clock calls inside solve paths.

use std::time::{Duration, Instant};

/// Host-provided clock used to measure solve duration.
pub trait SolveClock {
    /// Opaque timestamp captured before a solve starts.
    type Mark;

    /// Captures the current clock mark.
    fn mark(&self) -> Self::Mark;

    /// Returns elapsed time since `mark`.
    fn elapsed(&self, mark: &Self::Mark) -> Duration;
}

/// Standalone wall-clock implementation used by default solverang APIs.
#[derive(Clone, Copy, Debug, Default)]
pub struct StdClock;

impl SolveClock for StdClock {
    type Mark = Instant;

    fn mark(&self) -> Self::Mark {
        Instant::now()
    }

    fn elapsed(&self, mark: &Self::Mark) -> Duration {
        mark.elapsed()
    }
}

/// Deterministic clock for embedders and tests that do not want wall-clock data.
#[derive(Clone, Copy, Debug, Default)]
pub struct ZeroClock;

impl SolveClock for ZeroClock {
    type Mark = ();

    fn mark(&self) -> Self::Mark {}

    fn elapsed(&self, _mark: &Self::Mark) -> Duration {
        Duration::ZERO
    }
}
