//! Audit-ledger scaffolding for fsci-stats (br-egba-3).
//!
//! Mirrors the workspace audit pattern: a shared ledger re-export, a
//! factory, and `record_*` helpers. First emission site
//! demonstrated is `super::try_fit_with_audit` on Normal, which
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

/// Acquire the ledger guard, recovering from a poisoned mutex.
///
/// The previous `if let Ok(mut ledger) = ledger.lock()` pattern
/// silently dropped audit events when any prior thread panicked while
/// holding the guard, violating the fail-closed audit invariant.
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
        // /deadlock-finder-and-fixer regression for frankenscipy-kt4od:
        // a prior panic while holding the audit-ledger guard must NOT
        // silently drop subsequent record_fail_closed events. Before
        // the fix, `if let Ok(mut g) = ledger.lock()` returned Err on
        // poison and the event was lost. After the fix, lock_or_recover
        // clears the poison and proceeds.
        let ledger = sync_audit_ledger();

        // Deliberately poison the ledger: panic while holding the lock.
        let poisoned_thread = {
            let l = ledger.clone();
            std::thread::spawn(move || {
                let _g = l.lock().expect("acquire");
                panic!("poison the ledger on purpose");
            })
            .join()
        };
        assert!(poisoned_thread.is_err(), "thread should have panicked");
        assert!(
            ledger.lock().is_err(),
            "ledger must be poisoned after panic"
        );

        // The fixed record_fail_closed must still land an event.
        record_fail_closed(&ledger, b"x=NaN", "non_finite_input", "rejected");
        record_bounded_recovery(&ledger, b"x=Inf", "clamp_to_max", "recovered");

        // After clear_poison + into_inner the lock is healthy again.
        let g = ledger
            .lock()
            .expect("ledger should be healthy after recovery");
        assert_eq!(g.len(), 2, "both events must be recorded");
        let kinds: Vec<_> = g
            .entries()
            .iter()
            .map(|e| match &e.action {
                fsci_runtime::AuditAction::FailClosed { reason } => format!("FC:{reason}"),
                fsci_runtime::AuditAction::BoundedRecovery { recovery_action } => {
                    format!("BR:{recovery_action}")
                }
                _ => "OTHER".to_string(),
            })
            .collect();
        assert!(kinds.contains(&"FC:non_finite_input".to_string()));
        assert!(kinds.contains(&"BR:clamp_to_max".to_string()));
    }
}
