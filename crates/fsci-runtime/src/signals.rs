#![forbid(unsafe_code)]

//! Decision signal types for the CASP policy controller.
//!
//! Provides [`DecisionSignals`] for single evaluations and
//! [`SignalSequence`] for replay-based adversarial testing.

use serde::{Deserialize, Serialize};

/// Input triple for policy decision evaluation.
///
/// Signal values are stored as-provided at construction time.
/// Clamping occurs during logit computation, not here.
///
/// Ranges (semantic, not enforced at construction):
/// - `condition_number_log10`: `[0, +inf)`, clamped to `[0, 1]` via `/16.0`
/// - `metadata_incompatibility_score`: `[0, 1]`
/// - `input_anomaly_score`: `[0, 1]`
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecisionSignals {
    pub condition_number_log10: f64,
    pub metadata_incompatibility_score: f64,
    pub input_anomaly_score: f64,
}

impl DecisionSignals {
    #[must_use]
    pub fn new(
        condition_number_log10: f64,
        metadata_incompatibility_score: f64,
        input_anomaly_score: f64,
    ) -> Self {
        Self {
            condition_number_log10,
            metadata_incompatibility_score,
            input_anomaly_score,
        }
    }

    /// Returns `true` if all three signal components are finite (not NaN or Inf).
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.condition_number_log10.is_finite()
            && self.metadata_incompatibility_score.is_finite()
            && self.input_anomaly_score.is_finite()
    }
}

/// Ordered sequence of decision signals for replay-based testing.
///
/// Used by adversarial test suites (THREAT-003: evidence poisoning,
/// THREAT-004: NaN injection) to replay crafted signal sequences
/// through the policy controller and verify invariants at each step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignalSequence {
    /// Identifier for this sequence (e.g., test case name).
    pub id: String,
    /// Ordered signal values to replay through PolicyController::decide.
    pub signals: Vec<DecisionSignals>,
    /// Optional expected actions for each step (for assertion).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_actions: Option<Vec<String>>,
}

impl SignalSequence {
    /// Create a new empty signal sequence with the given identifier.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            signals: Vec::new(),
            expected_actions: None,
        }
    }

    /// Append a signal to the sequence.
    pub fn push(&mut self, signal: DecisionSignals) {
        self.signals.push(signal);
    }

    /// Number of signals in the sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    /// Whether the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }

    /// Iterate over signals in order.
    pub fn iter(&self) -> impl Iterator<Item = &DecisionSignals> {
        self.signals.iter()
    }
}
