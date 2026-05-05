//! Audit-ledger scaffolding for fsci-sparse (br-egba-4).
//!
//! Uses the canonical `fsci_runtime::SyncSharedAuditLedger` so a
//! single ledger can be threaded across crate boundaries. The crate
//! already has explicit Hardened-mode
//! validation in the format conversion routines (csr_to_csc_with_mode
//! and friends); `_with_audit` wrappers expose those rejections to a
//! forensic ledger without changing the existing error behavior.

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, casp_now_unix_ms};

#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

/// Acquire the ledger guard, recovering from a poisoned mutex so audit
/// events still record after any prior thread panicked.
/// Resolves [frankenscipy-l2irg] for fsci-sparse.
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
    lock_or_recover(ledger).record(event);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_after_poison_still_lands_in_ledger() {
        let ledger = sync_audit_ledger();

        let poisoned_thread = {
            let l = ledger.clone();
            std::thread::spawn(move || {
                let _g = l.lock().expect("acquire");
                panic!("poison fsci-sparse audit ledger on purpose");
            })
            .join()
        };
        assert!(poisoned_thread.is_err(), "thread should have panicked");
        assert!(
            ledger.lock().is_err(),
            "ledger must be poisoned after panic"
        );

        record_fail_closed(&ledger, b"shape", "csr_to_csc::shape", "rejected");
        record_bounded_recovery(&ledger, b"recover", "duplicate_indices_dedup", "recovered");

        let g = ledger.lock().expect("ledger should recover");
        assert_eq!(g.len(), 2);
    }
}
