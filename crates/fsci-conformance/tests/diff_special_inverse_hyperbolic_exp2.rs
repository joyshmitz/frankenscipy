#![forbid(unsafe_code)]
//! Cover fsci_special::{arcsinh, arccosh, arctanh, exp2_iterated}.
//!
//! Resolves [frankenscipy-tfced]. These are scalar wrappers over std
//! f64 math methods (arcsinh = asinh, arccosh = acosh, arctanh = atanh)
//! plus exp2_iterated which computes exp(exp(x)) — closed-form
//! comparison against the same f64 primitives.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::convenience::exp2_iterated;
use fsci_special::{arccosh, arcsinh, arctanh};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-14;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    actual: f64,
    expected: f64,
    abs_diff: f64,
    pass: bool,
    note: String,
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
    fs::create_dir_all(output_dir()).expect("create inv_hyper diff dir");
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
fn diff_special_inverse_hyperbolic_exp2() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, actual: f64, expected: f64| {
        let abs_diff = (actual - expected).abs();
        diffs.push(CaseDiff {
            case_id: id.into(),
            actual,
            expected,
            abs_diff,
            pass: abs_diff <= ABS_TOL,
            note: String::new(),
        });
    };

    // arcsinh: defined for all real x; identity arcsinh(sinh(x)) = x
    for &x in &[-3.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 100.0] {
        check(&format!("arcsinh_{x}"), arcsinh(x), x.asinh());
    }
    // sinh-arcsinh round-trip
    for &x in &[-2.5_f64, 0.0, 1.5] {
        let s = x.sinh();
        let back = arcsinh(s);
        check(&format!("arcsinh_round_trip_{x}"), back, x);
    }

    // arccosh: defined for x >= 1
    for &x in &[1.0_f64, 1.5, 2.0, 5.0, 100.0] {
        check(&format!("arccosh_{x}"), arccosh(x), x.acosh());
    }
    // cosh-arccosh round-trip for x >= 0 (arccosh always non-negative)
    for &x in &[0.0_f64, 0.5, 2.0, 3.5] {
        let c = x.cosh();
        let back = arccosh(c);
        check(&format!("arccosh_round_trip_{x}"), back, x);
    }

    // arctanh: defined for |x| < 1
    for &x in &[-0.99_f64, -0.5, -0.1, 0.0, 0.1, 0.5, 0.99] {
        check(&format!("arctanh_{x}"), arctanh(x), x.atanh());
    }
    // tanh-arctanh round-trip
    for &x in &[-2.0_f64, -0.5, 0.0, 0.5, 2.0] {
        let t = x.tanh();
        let back = arctanh(t);
        check(&format!("arctanh_round_trip_{x}"), back, x);
    }

    // exp2_iterated(x) = exp(exp(x))
    for &x in &[-2.0_f64, -1.0, 0.0, 0.5, 1.0, 2.0] {
        check(&format!("exp2_iterated_{x}"), exp2_iterated(x), x.exp().exp());
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_special_inverse_hyperbolic_exp2".into(),
        category: "fsci_special::{arcsinh, arccosh, arctanh, exp2_iterated} coverage".into(),
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
                "inv_hyper mismatch: {} actual={} expected={} abs={}",
                d.case_id, d.actual, d.expected, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "inv_hyper/exp2 coverage failed: {} cases",
        diffs.len()
    );
}
