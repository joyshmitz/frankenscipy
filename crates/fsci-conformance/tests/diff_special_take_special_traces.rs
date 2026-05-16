#![forbid(unsafe_code)]
//! Verify fsci_special::take_special_traces drains the special-trace
//! log: record → take returns ≥1 entry; subsequent take returns empty.
//!
//! Resolves [frankenscipy-f2wl4].

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::take_special_traces;
use fsci_special::types::record_special_trace;
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
    fs::create_dir_all(output_dir()).expect("create take_special_traces diff dir");
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
fn diff_special_take_special_traces() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    // Drain any pre-existing traces.
    let _ = take_special_traces();

    record_special_trace(
        "test_fn",
        RuntimeMode::Strict,
        "test_category",
        "x=1.0",
        "clamped",
        "1.5",
        true,
    );
    record_special_trace(
        "test_fn",
        RuntimeMode::Hardened,
        "test_category",
        "x=2.0",
        "ok",
        "2.5",
        false,
    );
    let drained1 = take_special_traces();
    diffs.push(CaseDiff {
        case_id: "two_records_drained".into(),
        pass: drained1.len() == 2,
    });

    // Second take returns empty
    let drained2 = take_special_traces();
    diffs.push(CaseDiff {
        case_id: "second_take_empty".into(),
        pass: drained2.is_empty(),
    });

    // Recorded entries preserve their metadata
    let pass_metadata = drained1
        .iter()
        .any(|e| e.function == "test_fn" && e.category == "test_category" && e.clamped);
    diffs.push(CaseDiff {
        case_id: "metadata_preserved".into(),
        pass: pass_metadata,
    });

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_take_special_traces".into(),
        category: "fsci_special::take_special_traces drain + metadata".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("take_special_traces mismatch: {}", d.case_id);
        }
    }

    assert!(
        all_pass,
        "take_special_traces coverage failed: {} cases",
        diffs.len(),
    );
}
