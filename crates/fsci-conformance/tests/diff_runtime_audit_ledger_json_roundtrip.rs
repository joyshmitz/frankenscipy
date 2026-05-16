#![forbid(unsafe_code)]
//! Verify AuditLedger.to_json() / from_json() roundtrip and
//! fingerprint_bytes determinism.
//!
//! Resolves [frankenscipy-vlr6l]. Serialize a ledger populated with
//! several events, deserialize, and verify equivalence. Also confirm
//! fingerprint_bytes returns the same hex for the same input bytes.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::{AuditAction, AuditEvent, AuditLedger};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create audit_json diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

#[test]
fn diff_runtime_audit_ledger_json_roundtrip() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    // Empty ledger roundtrip
    let empty = AuditLedger::new();
    let s = empty.to_json().expect("to_json empty");
    let back = AuditLedger::from_json(&s).expect("from_json empty");
    diffs.push(CaseDiff {
        case_id: "empty_roundtrip".into(),
        pass: empty == back,
    });

    // Populated ledger with two event types
    let mut led = AuditLedger::new();
    led.record(AuditEvent::new(
        1000,
        "fingerprint-A",
        AuditAction::FailClosed {
            reason: "non_finite_input".into(),
        },
        "rejected",
    ));
    led.record(AuditEvent::new(
        2000,
        "fingerprint-B",
        AuditAction::BoundedRecovery {
            recovery_action: "restart_with_lower_step".into(),
        },
        "succeeded",
    ));
    let json = led.to_json().expect("to_json populated");
    let back2 = AuditLedger::from_json(&json).expect("from_json populated");
    diffs.push(CaseDiff {
        case_id: "populated_roundtrip".into(),
        pass: led == back2 && back2.len() == 2,
    });

    // fingerprint_bytes determinism
    let f1 = AuditLedger::fingerprint_bytes(b"alpha");
    let f2 = AuditLedger::fingerprint_bytes(b"alpha");
    let f3 = AuditLedger::fingerprint_bytes(b"beta");
    diffs.push(CaseDiff {
        case_id: "fingerprint_deterministic".into(),
        pass: f1 == f2,
    });
    diffs.push(CaseDiff {
        case_id: "fingerprint_distinguishes".into(),
        pass: f1 != f3,
    });

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_runtime_audit_ledger_json_roundtrip".into(),
        category: "AuditLedger json + fingerprint determinism".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("audit_json mismatch: {}", d.case_id);
        }
    }

    assert!(
        all_pass,
        "audit_json conformance failed: {} cases",
        diffs.len(),
    );
}
