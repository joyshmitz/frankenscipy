#![forbid(unsafe_code)]
//! Verify fsci_fft::take_transform_traces drains the transform-trace
//! log: after fft() emits a trace, the call returns it and subsequent
//! calls return empty.
//!
//! Resolves [frankenscipy-zc64q].

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{Complex64, FftOptions, fft, take_transform_traces};
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
    fs::create_dir_all(output_dir()).expect("create take_traces diff dir");
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
fn diff_fft_take_transform_traces() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let opts = FftOptions::default();

    // Drain any pre-existing traces (sibling tests may have produced them
    // since the global trace log is process-wide).
    let _ = take_transform_traces();

    // Issue an fft call — it should record at least one trace.
    let signal: Vec<Complex64> =
        (0..16).map(|i| (i as f64, 0.0)).collect();
    let _ = fft(&signal, &opts).expect("fft");

    let traces1 = take_transform_traces();
    diffs.push(CaseDiff {
        case_id: "fft_produces_trace".into(),
        pass: !traces1.is_empty(),
    });

    // Subsequent call without intervening fft should return empty.
    let traces2 = take_transform_traces();
    diffs.push(CaseDiff {
        case_id: "second_call_empty".into(),
        pass: traces2.is_empty(),
    });

    // Issue several fft calls; trace count grows.
    let _ = fft(&signal, &opts).expect("fft1");
    let _ = fft(&signal, &opts).expect("fft2");
    let _ = fft(&signal, &opts).expect("fft3");
    let traces3 = take_transform_traces();
    diffs.push(CaseDiff {
        case_id: "three_calls_three_or_more".into(),
        pass: traces3.len() >= 3,
    });

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_fft_take_transform_traces".into(),
        category: "fsci_fft::take_transform_traces drain behavior".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("take_traces mismatch: {}", d.case_id);
        }
    }

    assert!(
        all_pass,
        "take_traces coverage failed: {} cases",
        diffs.len(),
    );
}
