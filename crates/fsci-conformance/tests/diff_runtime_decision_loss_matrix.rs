#![forbid(unsafe_code)]
//! Property test for fsci_runtime::decision_loss_matrix.
//!
//! Resolves [frankenscipy-wkpb6]. Verify the documented Strict and
//! Hardened loss matrices are returned exactly.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::{RuntimeMode, decision_loss_matrix};
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
    fs::create_dir_all(output_dir()).expect("create dlm diff dir");
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
fn diff_runtime_decision_loss_matrix() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    let strict_expected: [[f64; 3]; 3] =
        [[0.0, 65.0, 200.0], [8.0, 4.0, 80.0], [40.0, 25.0, 1.0]];
    let hardened_expected: [[f64; 3]; 3] =
        [[0.0, 50.0, 180.0], [5.0, 3.0, 60.0], [55.0, 30.0, 1.0]];

    let strict_actual = decision_loss_matrix(RuntimeMode::Strict);
    let hardened_actual = decision_loss_matrix(RuntimeMode::Hardened);

    diffs.push(CaseDiff {
        case_id: "strict_matrix".into(),
        pass: strict_actual == strict_expected,
    });
    diffs.push(CaseDiff {
        case_id: "hardened_matrix".into(),
        pass: hardened_actual == hardened_expected,
    });

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_runtime_decision_loss_matrix".into(),
        category: "fsci_runtime::decision_loss_matrix const lookup".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("dlm mismatch: {}", d.case_id);
        }
    }

    assert!(
        all_pass,
        "decision_loss_matrix coverage failed: {} cases",
        diffs.len(),
    );
}
