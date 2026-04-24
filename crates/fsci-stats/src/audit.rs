//! Audit-ledger scaffolding for fsci-stats (br-egba-3).
//!
//! Mirrors the workspace audit pattern: a shared ledger re-export, a
//! factory, and `record_*` helpers. First emission site
//! demonstrated is [`super::try_fit_with_audit`] on Normal, which
//! records a FailClosed event when input validation rejects the sample.
//!
//! At the time of this slice, `try_fit` itself already fails closed
//! via `FitError`; the audit integration just adds a forensic trail
//! for Hardened-mode deployments.

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, casp_now_unix_ms};

#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

pub fn record_fail_closed(
    ledger: &SyncSharedAuditLedger,
    input_bytes: &[u8],
    reason: &str,
    outcome: &str,
) {
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        AuditLedger::fingerprint_bytes(input_bytes),
        AuditAction::FailClosed {
            reason: reason.to_string(),
        },
        outcome.to_string(),
    );
    if let Ok(mut ledger) = ledger.lock() {
        ledger.record(event);
    }
}

pub fn record_bounded_recovery(
    ledger: &SyncSharedAuditLedger,
    input_bytes: &[u8],
    recovery_action: &str,
    outcome: &str,
) {
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        AuditLedger::fingerprint_bytes(input_bytes),
        AuditAction::BoundedRecovery {
            recovery_action: recovery_action.to_string(),
        },
        outcome.to_string(),
    );
    if let Ok(mut ledger) = ledger.lock() {
        ledger.record(event);
    }
}
