#![forbid(unsafe_code)]

//! Bounded FIFO evidence ledger for policy decision audit trail.

use blake3::hash;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::mode::RuntimeMode;
use crate::policy::{PolicyAction, RiskState, decision_loss_matrix};
use crate::signals::DecisionSignals;

/// Spec §6 decision-theory record with every model input and output surfaced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlienArtifactDecision {
    pub state_space: [RiskState; 3],
    pub state: RiskState,
    pub evidence: DecisionSignals,
    pub logits: [f64; 3],
    pub loss_matrix: [[f64; 3]; 3],
    pub posterior: [f64; 3],
    pub expected_losses: [f64; 3],
    pub action: PolicyAction,
    pub confidence: f64,
    pub calibration_fallback_trigger: bool,
    pub reason: String,
}

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

impl DecisionEvidenceEntry {
    #[must_use]
    pub fn confidence(&self) -> f64 {
        self.posterior[self.top_state.index()]
    }

    #[must_use]
    pub fn calibration_fallback_trigger(&self) -> bool {
        !self.signals.is_finite()
            || (self.action == PolicyAction::FailClosed
                && self.top_state == RiskState::IncompatibleMetadata)
    }

    #[must_use]
    pub fn alien_artifact_decision(&self) -> AlienArtifactDecision {
        AlienArtifactDecision {
            state_space: RiskState::ALL,
            state: self.top_state,
            evidence: self.signals,
            logits: self.logits,
            loss_matrix: decision_loss_matrix(self.mode),
            posterior: self.posterior,
            expected_losses: self.expected_losses,
            action: self.action,
            confidence: self.confidence(),
            calibration_fallback_trigger: self.calibration_fallback_trigger(),
            reason: self.reason.clone(),
        }
    }
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

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &DecisionEvidenceEntry> {
        self.entries.iter()
    }

    #[must_use]
    pub fn alien_artifact_decisions(&self) -> Vec<AlienArtifactDecision> {
        self.entries
            .iter()
            .map(DecisionEvidenceEntry::alien_artifact_decision)
            .collect()
    }

    #[must_use]
    pub fn latest_alien_artifact_decision(&self) -> Option<AlienArtifactDecision> {
        self.latest()
            .map(DecisionEvidenceEntry::alien_artifact_decision)
    }

    /// Serialize Spec §6 decision records as JSONL for audit artifacts.
    #[must_use]
    pub fn to_alien_artifact_jsonl(&self) -> String {
        self.entries
            .iter()
            .filter_map(|entry| serde_json::to_string(&entry.alien_artifact_decision()).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Audit actions recorded by the runtime for forensic analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AuditAction {
    ModeDecision { mode: RuntimeMode },
    BoundedRecovery { recovery_action: String },
    FailClosed { reason: String },
    // br-egba-1: `PolicyOverride { override_action: String }` was
    // defined here but never constructed by any crate in the workspace
    // (grep confirms zero call sites outside the enum definition).
    // Removed as dead code. If policy-override semantics are added
    // back, re-introduce the variant alongside at least one emission
    // site so it remains non-dead.
}

/// Single audit event entry with input fingerprint and outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp_ms: u64,
    pub input_fingerprint: String,
    pub action: AuditAction,
    pub outcome: String,
}

impl AuditEvent {
    #[must_use]
    pub fn new(
        timestamp_ms: u64,
        input_fingerprint: impl Into<String>,
        action: AuditAction,
        outcome: impl Into<String>,
    ) -> Self {
        Self {
            timestamp_ms,
            input_fingerprint: input_fingerprint.into(),
            action,
            outcome: outcome.into(),
        }
    }
}

/// Append-only ledger for audit events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditLedger {
    entries: Vec<AuditEvent>,
}

impl AuditLedger {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record an audit event (append-only).
    pub fn record(&mut self, event: AuditEvent) {
        self.entries.push(event);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn entries(&self) -> &[AuditEvent] {
        &self.entries
    }

    /// Serialize ledger as JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize ledger from JSON.
    pub fn from_json(payload: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(payload)
    }

    /// Compute a blake3 hex fingerprint for raw input bytes.
    #[must_use]
    pub fn fingerprint_bytes(bytes: &[u8]) -> String {
        hash(bytes).to_hex().to_string()
    }

    /// Construct a shared, thread-safe ledger handle.
    #[must_use]
    pub fn shared() -> SharedAuditLedger {
        Arc::new(Mutex::new(Self::new()))
    }
}

impl Default for AuditLedger {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe audit ledger handle shared by synchronous crate APIs.
pub type SharedAuditLedger = Arc<Mutex<AuditLedger>>;

/// Canonical synchronous audit ledger handle.
pub type SyncSharedAuditLedger = SharedAuditLedger;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_ledger_roundtrip_preserves_entries() {
        let mut ledger = AuditLedger::new();
        ledger.record(AuditEvent::new(
            1,
            AuditLedger::fingerprint_bytes(b"mode"),
            AuditAction::ModeDecision {
                mode: RuntimeMode::Strict,
            },
            "accepted",
        ));
        ledger.record(AuditEvent::new(
            2,
            AuditLedger::fingerprint_bytes(b"recover"),
            AuditAction::BoundedRecovery {
                recovery_action: "trim_nan".to_string(),
            },
            "recovered",
        ));
        ledger.record(AuditEvent::new(
            3,
            AuditLedger::fingerprint_bytes(b"reject"),
            AuditAction::FailClosed {
                reason: "non_finite_input".to_string(),
            },
            "rejected",
        ));

        let json = ledger.to_json().expect("serialize failed");
        let decoded = AuditLedger::from_json(&json).expect("deserialize failed");

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded, ledger);
    }

    #[test]
    fn audit_fail_closed_includes_reason() {
        let event = AuditEvent::new(
            7,
            AuditLedger::fingerprint_bytes(b"fail"),
            AuditAction::FailClosed {
                reason: "invalid_metadata".to_string(),
            },
            "rejected",
        );
        match &event.action {
            AuditAction::FailClosed { reason } => {
                assert_eq!(reason, "invalid_metadata");
            }
            other => {
                unreachable!("expected fail closed action, got {other:?}");
            }
        }
    }

    #[test]
    fn audit_bounded_recovery_includes_action_and_outcome() {
        let event = AuditEvent::new(
            9,
            AuditLedger::fingerprint_bytes(b"recover"),
            AuditAction::BoundedRecovery {
                recovery_action: "drop_outliers".to_string(),
            },
            "recovered",
        );
        match &event.action {
            AuditAction::BoundedRecovery { recovery_action } => {
                assert_eq!(recovery_action, "drop_outliers");
            }
            other => {
                unreachable!("expected bounded recovery action, got {other:?}");
            }
        }
        assert_eq!(event.outcome, "recovered");
    }

    #[test]
    fn alien_artifact_decision_surfaces_spec_fields() {
        let entry = DecisionEvidenceEntry {
            mode: RuntimeMode::Strict,
            signals: DecisionSignals::new(8.0, 0.4, 0.2),
            logits: [1.0, 0.5, -1.0],
            posterior: [0.6, 0.3, 0.1],
            expected_losses: [39.5, 14.0, 31.6],
            action: PolicyAction::FullValidate,
            top_state: RiskState::Compatible,
            reason: String::from("test"),
        };

        let decision = entry.alien_artifact_decision();

        assert_eq!(decision.state_space, RiskState::ALL);
        assert_eq!(decision.state, RiskState::Compatible);
        assert_eq!(decision.evidence, entry.signals);
        assert_eq!(
            decision.loss_matrix,
            decision_loss_matrix(RuntimeMode::Strict)
        );
        assert_eq!(decision.posterior, entry.posterior);
        assert_eq!(decision.action, PolicyAction::FullValidate);
        assert_eq!(decision.confidence, 0.6);
        assert!(!decision.calibration_fallback_trigger);
    }

    #[test]
    fn policy_evidence_ledger_emits_alien_artifact_jsonl() {
        let mut ledger = PolicyEvidenceLedger::new(2);
        ledger.record(DecisionEvidenceEntry {
            mode: RuntimeMode::Hardened,
            signals: DecisionSignals::new(f64::NAN, 0.0, 0.0),
            logits: [-1.0e30, -1.0e30, 0.0],
            posterior: [0.0, 0.0, 1.0],
            expected_losses: [180.0, 60.0, 1.0],
            action: PolicyAction::FailClosed,
            top_state: RiskState::IncompatibleMetadata,
            reason: String::from("non_finite_signals=true"),
        });

        let latest = ledger
            .latest_alien_artifact_decision()
            .expect("latest decision");
        assert_eq!(latest.confidence, 1.0);
        assert!(latest.calibration_fallback_trigger);

        let jsonl = ledger.to_alien_artifact_jsonl();
        let parsed =
            serde_json::from_str::<serde_json::Value>(&jsonl).expect("valid JSONL decision");
        assert!(parsed.get("state_space").is_some());
        assert!(parsed.get("evidence").is_some());
        assert!(parsed.get("loss_matrix").is_some());
        assert_eq!(parsed["confidence"], 1.0);
        assert_eq!(parsed["calibration_fallback_trigger"], true);
    }
}
