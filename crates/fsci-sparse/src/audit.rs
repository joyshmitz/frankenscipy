//! Audit-ledger scaffolding for fsci-sparse (br-egba-4).
//!
//! Type-compatible with fsci-linalg / fsci-special / fsci-stats so a
//! single ledger can be threaded across crate boundaries once ovxs
//! unifies the alias. The crate already has explicit Hardened-mode
//! validation in the format conversion routines (csr_to_csc_with_mode
//! and friends); `_with_audit` wrappers expose those rejections to a
//! forensic ledger without changing the existing error behavior.

use std::sync::{Arc, Mutex as StdMutex};

use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, casp_now_unix_ms};

pub type SyncSharedAuditLedger = Arc<StdMutex<AuditLedger>>;

#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    Arc::new(StdMutex::new(AuditLedger::new()))
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
