#![forbid(unsafe_code)]

use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, RuntimeMode};

fn canonical_audit_ledger() -> AuditLedger {
    let mut ledger = AuditLedger::new();
    ledger.record(AuditEvent::new(
        1_700_000_000_000,
        "strict-policy-fingerprint",
        AuditAction::ModeDecision {
            mode: RuntimeMode::Strict,
        },
        "accepted",
    ));
    ledger.record(AuditEvent::new(
        1_700_000_000_100,
        "nan-input-fingerprint",
        AuditAction::BoundedRecovery {
            recovery_action: "trim_nan".to_string(),
        },
        "recovered",
    ));
    ledger.record(AuditEvent::new(
        1_700_000_000_200,
        "bad-metadata-fingerprint",
        AuditAction::FailClosed {
            reason: "invalid_metadata".to_string(),
        },
        "rejected",
    ));
    ledger
}

#[test]
fn audit_ledger_json_matches_golden_artifact() {
    let actual = canonical_audit_ledger()
        .to_json()
        .expect("canonical audit ledger should serialize");
    let expected = include_str!("goldens/audit_ledger.json").trim_end();

    assert_eq!(actual, expected);
}

#[test]
fn audit_ledger_golden_artifact_is_decodable() {
    let expected = include_str!("goldens/audit_ledger.json");
    let decoded =
        AuditLedger::from_json(expected).expect("audit ledger golden artifact should decode");

    assert_eq!(decoded, canonical_audit_ledger());
}
