use serde::{Deserialize, Serialize};

/// Per-evaluation numerical policy passed explicitly to every operator action.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationContext {
    reproducible: bool,
}

impl EvaluationContext {
    /// Requests deterministic choices from algorithms and downstream backends.
    #[must_use]
    pub const fn reproducible() -> Self {
        Self { reproducible: true }
    }

    /// Whether reproducible execution was requested.
    #[must_use]
    pub const fn is_reproducible(&self) -> bool {
        self.reproducible
    }
}
