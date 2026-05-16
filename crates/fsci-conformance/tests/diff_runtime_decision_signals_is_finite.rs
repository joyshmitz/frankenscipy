#![forbid(unsafe_code)]
//! Verify DecisionSignals::is_finite returns true iff all three
//! components are finite (not NaN, not ±Inf).
//!
//! Resolves [frankenscipy-5eaks].

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::DecisionSignals;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    expected: bool,
    actual: bool,
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
    fs::create_dir_all(output_dir()).expect("create ds_finite diff dir");
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
fn diff_runtime_decision_signals_is_finite() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    let probes: &[(&str, DecisionSignals, bool)] = &[
        ("all_finite", DecisionSignals::new(8.0, 0.0, 0.1), true),
        ("nan_first", DecisionSignals::new(f64::NAN, 0.0, 0.1), false),
        ("nan_second", DecisionSignals::new(8.0, f64::NAN, 0.1), false),
        ("nan_third", DecisionSignals::new(8.0, 0.0, f64::NAN), false),
        ("pos_inf", DecisionSignals::new(f64::INFINITY, 0.0, 0.1), false),
        ("neg_inf", DecisionSignals::new(8.0, 0.0, f64::NEG_INFINITY), false),
        ("zeros", DecisionSignals::new(0.0, 0.0, 0.0), true),
        ("very_large_finite", DecisionSignals::new(1.0e300, 1.0e10, 0.5), true),
    ];

    for (label, sig, expected) in probes {
        let actual = sig.is_finite();
        diffs.push(CaseDiff {
            case_id: (*label).into(),
            expected: *expected,
            actual,
            pass: actual == *expected,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_runtime_decision_signals_is_finite".into(),
        category: "DecisionSignals::is_finite branch coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "ds_finite mismatch: {} got {} (expected {})",
                d.case_id, d.actual, d.expected
            );
        }
    }

    assert!(
        all_pass,
        "DecisionSignals::is_finite coverage failed: {} cases",
        diffs.len(),
    );
}
