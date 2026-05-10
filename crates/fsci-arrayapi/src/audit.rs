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
fn lock_or_recover(ledger: &SyncSharedAuditLedger) -> std::sync::MutexGuard<'_, AuditLedger> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_after_poison_still_lands_in_ledger() {
        // /mock-code-finder regression for [frankenscipy-h5hj3]:
        // mirrors the fsci-stats poison-recovery test for fsci-arrayapi
        // so a peer agent cannot silently revert lock_or_recover to the
        // 'if let Ok(mut g) = lock()' silent-drop pattern in this crate.
        let ledger = sync_audit_ledger();

        let poisoned_thread = {
            let l = ledger.clone();
            std::thread::spawn(move || {
                let _g = l.lock().expect("acquire");
                std::panic::resume_unwind(Box::new("poison fsci-arrayapi audit ledger on purpose"));
            })
            .join()
        };
        assert!(poisoned_thread.is_err(), "thread should have panicked");
        assert!(
            ledger.lock().is_err(),
            "ledger must be poisoned after panic"
        );

        record_fail_closed(
            &ledger,
            b"shape mismatch",
            "broadcast::Incompatible",
            "rejected",
        );

        let g = ledger.lock().expect("ledger should recover after poison");
        assert_eq!(g.len(), 1, "the audit event must be recorded");
        assert!(matches!(
            &g.entries()[0].action,
            fsci_runtime::AuditAction::FailClosed { reason } if reason == "broadcast::Incompatible"
        ));
    }
}
