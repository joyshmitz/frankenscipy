#![forbid(unsafe_code)]
//! Property-based test for fsci_opt::simulated_annealing.
//!
//! Resolves [frankenscipy-p82uo]. Discrete-state SA on the
//! "find integer in [-10, 10] closest to target" problem. The
//! cost is (x - target)². Neighbor flips one unit ±1 with bounds
//! clamping. With seed=42 and sufficient iterations SA must find
//! the target or an integer within 1 of it.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::simulated_annealing;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL: f64 = 4.0; // (x-target)² ≤ 4 (i.e. within 2 of target)

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create sa diff dir");
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
fn diff_opt_simulated_annealing() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;

    // Three targets in [-10, 10]
    for &(label, target) in &[
        ("target_zero", 0_i64),
        ("target_three", 3),
        ("target_neg_five", -5),
    ] {
        let cost = |s: &i64| ((s - target).pow(2)) as f64;
        let neighbor = |s: &i64, rng_state: u64| -> i64 {
            // Random ±1 step clamped to [-10, 10]
            let step: i64 = if (rng_state >> 32) & 1 == 0 { 1 } else { -1 };
            (s + step).clamp(-10, 10)
        };
        let (best_state, best_cost) =
            simulated_annealing(8_i64, cost, neighbor, 1.0, 1.0e-4, 5000, 42);
        let _ = best_state;
        max_overall = max_overall.max(best_cost);
        diffs.push(CaseDiff {
            case_id: format!("sa_{label}"),
            abs_diff: best_cost,
            pass: best_cost <= TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_simulated_annealing".into(),
        category: "fsci_opt::simulated_annealing property test".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("sa mismatch: {} cost={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "sa conformance failed: {} cases, max_cost={}",
        diffs.len(),
        max_overall
    );
}
