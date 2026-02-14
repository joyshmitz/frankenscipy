#![forbid(unsafe_code)]

//! Bounded FIFO evidence ledger for policy decision audit trail.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::mode::RuntimeMode;
use crate::policy::{PolicyAction, RiskState};
use crate::signals::DecisionSignals;

/// Complete record of a single policy decision for audit/forensic analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionEvidenceEntry {
    pub mode: RuntimeMode,
    pub signals: DecisionSignals,
    pub logits: [f64; 3],
    pub posterior: [f64; 3],
    pub expected_losses: [f64; 3],
    pub action: PolicyAction,
    pub top_state: RiskState,
    pub reason: String,
}

/// Bounded FIFO evidence buffer recording all policy decisions.
///
/// Capacity is enforced via `capacity.max(1)` — minimum 1 entry.
/// When full, the oldest entry (front of `VecDeque`) is evicted before
/// a new entry is appended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyEvidenceLedger {
    capacity: usize,
    entries: VecDeque<DecisionEvidenceEntry>,
}

impl PolicyEvidenceLedger {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::new(),
        }
    }

    /// Append an entry, evicting the oldest if at capacity.
    pub fn record(&mut self, entry: DecisionEvidenceEntry) {
        if self.entries.len() == self.capacity {
            let _ = self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The most recently recorded entry.
    #[must_use]
    pub fn latest(&self) -> Option<&DecisionEvidenceEntry> {
        self.entries.back()
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}
