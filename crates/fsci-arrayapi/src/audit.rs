//! Audit-ledger helpers for Array API hardened rejection paths.

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, casp_now_unix_ms};

#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

/// Acquire the ledger guard, recovering from a poisoned mutex so
/// audit events still record after any prior thread panicked.
/// Resolves [frankenscipy-kt4od].
fn lock_or_recover(
    ledger: &SyncSharedAuditLedger,
) -> std::sync::MutexGuard<'_, AuditLedger> {
    match ledger.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            ledger.clear_poison();
            poisoned.into_inner()
        }
    }
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
    lock_or_recover(ledger).record(event);
}

pub(crate) fn record_array_api_error(
    ledger: &SyncSharedAuditLedger,
    operation: &str,
    input_bytes: &[u8],
    kind: crate::error::ArrayApiErrorKind,
) {
    record_fail_closed(
        ledger,
        input_bytes,
        &format!("{operation}::{kind:?}"),
        "rejected",
    );
}
