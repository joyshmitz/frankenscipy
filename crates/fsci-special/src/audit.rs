//! Audit-ledger scaffolding for fsci-special (br-egba-2).
//!
//! Mirrors the workspace audit pattern: a `SyncSharedAuditLedger`
//! re-export, a `sync_audit_ledger()` factory, and `record_*`
//! helpers that crate-internal functions call at rejection / recovery
//! sites in Hardened mode.
//!
//! At the time of this slice, only a small subset of functions have
//! `_with_audit` variants wired up — `gamma` is the demonstrator. The
//! remaining 340+ public functions inherit a silent path until
//! dedicated wiring lands. `SyncSharedAuditLedger` comes from
//! fsci-runtime so a single ledger can be threaded across crate
//! boundaries.

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, casp_now_unix_ms};

/// Create a new shared audit ledger for synchronous contexts.
#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

/// Record a fail-closed audit event when Hardened mode rejects input.
///
/// `input_bytes` is hashed into the event fingerprint; callers with
/// numeric inputs should pass a byte representation that uniquely
/// identifies the offending value (e.g. `x.to_le_bytes()` for an f64).
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

/// Record a bounded-recovery audit event when Hardened mode falls
/// back to a degraded but safe computation path.
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
