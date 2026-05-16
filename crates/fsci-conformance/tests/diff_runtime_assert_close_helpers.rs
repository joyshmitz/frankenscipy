#![forbid(unsafe_code)]
//! Cover fsci_runtime::{assert_close_slice, assert_close_matrix,
//! casp_now_unix_ms}.
//!
//! Resolves [frankenscipy-hx06w]. assert_close_slice/_matrix should accept
//! matching inputs and panic on either length mismatch or any element
//! outside tolerance. casp_now_unix_ms should return a nonzero
//! monotonic-ish timestamp.

use std::fs;
use std::panic;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::{assert_close_matrix, assert_close_slice, casp_now_unix_ms};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create assert_helpers diff dir");
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

/// Run f silently (no panic output noise) and return Ok if it does not panic,
/// Err with the panic message otherwise.
fn catch(f: impl FnOnce() + panic::UnwindSafe) -> Result<(), String> {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let r = panic::catch_unwind(f);
    panic::set_hook(prev);
    r.map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&'static str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "panic (unknown payload)".into())
    })
}

#[test]
fn diff_runtime_assert_close_helpers() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    // === assert_close_slice ===

    // 1. Equal-length slices within tol → ok
    {
        let a = vec![1.0_f64, 2.0, 3.0];
        let e = vec![1.0 + 1e-12, 2.0, 3.0 - 1e-12];
        let r = catch(move || assert_close_slice(&a, &e, 1e-9, 1e-9));
        diffs.push(CaseDiff {
            case_id: "slice_within_tol".into(),
            pass: r.is_ok(),
            note: r.err().unwrap_or_default(),
        });
    }

    // 2. Length mismatch → panic
    {
        let a = vec![1.0_f64, 2.0];
        let e = vec![1.0, 2.0, 3.0];
        let r = catch(move || assert_close_slice(&a, &e, 1e-9, 1e-9));
        diffs.push(CaseDiff {
            case_id: "slice_len_mismatch".into(),
            pass: r.is_err(),
            note: r.err().unwrap_or_default(),
        });
    }

    // 3. Value outside tol → panic
    {
        let a = vec![1.0_f64, 2.0, 3.0];
        let e = vec![1.0, 2.0, 3.5];
        let r = catch(move || assert_close_slice(&a, &e, 1e-9, 1e-9));
        diffs.push(CaseDiff {
            case_id: "slice_value_outside_tol".into(),
            pass: r.is_err(),
            note: r.err().unwrap_or_default(),
        });
    }

    // 4. NaN matches NaN, Inf matches Inf
    {
        let a = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let e = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let r = catch(move || assert_close_slice(&a, &e, 1e-9, 1e-9));
        diffs.push(CaseDiff {
            case_id: "slice_nan_inf_match".into(),
            pass: r.is_ok(),
            note: r.err().unwrap_or_default(),
        });
    }

    // 5. NaN vs finite → panic
    {
        let a = vec![f64::NAN];
        let e = vec![0.0];
        let r = catch(move || assert_close_slice(&a, &e, 1e-9, 1e-9));
        diffs.push(CaseDiff {
            case_id: "slice_nan_vs_finite_panics".into(),
            pass: r.is_err(),
            note: r.err().unwrap_or_default(),
        });
    }

    // 6. Empty slices both → ok
    {
        let a: Vec<f64> = vec![];
        let e: Vec<f64> = vec![];
        let r = catch(move || assert_close_slice(&a, &e, 1e-9, 1e-9));
        diffs.push(CaseDiff {
            case_id: "slice_empty_ok".into(),
            pass: r.is_ok(),
            note: r.err().unwrap_or_default(),
        });
    }

    // === assert_close_matrix ===

    // 7. Matching matrices within tol → ok
    {
        let a = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]];
        let e = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let r = catch(move || assert_close_matrix(&a, &e, 1e-9, 1e-9));
        diffs.push(CaseDiff {
            case_id: "matrix_match".into(),
            pass: r.is_ok(),
            note: r.err().unwrap_or_default(),
        });
    }

    // 8. Row count mismatch → panic
    {
        let a = vec![vec![1.0_f64, 2.0]];
        let e = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let r = catch(move || assert_close_matrix(&a, &e, 1e-9, 1e-9));
        diffs.push(CaseDiff {
            case_id: "matrix_row_mismatch".into(),
            pass: r.is_err(),
            note: r.err().unwrap_or_default(),
        });
    }

    // 9. Column count mismatch → panic
    {
        let a = vec![vec![1.0_f64, 2.0], vec![3.0]];
        let e = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let r = catch(move || assert_close_matrix(&a, &e, 1e-9, 1e-9));
        diffs.push(CaseDiff {
            case_id: "matrix_col_mismatch".into(),
            pass: r.is_err(),
            note: r.err().unwrap_or_default(),
        });
    }

    // 10. Value mismatch at single cell → panic
    {
        let a = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]];
        let e = vec![vec![1.0, 2.0], vec![3.0, 99.0]];
        let r = catch(move || assert_close_matrix(&a, &e, 1e-9, 1e-9));
        diffs.push(CaseDiff {
            case_id: "matrix_value_mismatch".into(),
            pass: r.is_err(),
            note: r.err().unwrap_or_default(),
        });
    }

    // 11. Empty matrices → ok
    {
        let a: Vec<Vec<f64>> = vec![];
        let e: Vec<Vec<f64>> = vec![];
        let r = catch(move || assert_close_matrix(&a, &e, 1e-9, 1e-9));
        diffs.push(CaseDiff {
            case_id: "matrix_empty_ok".into(),
            pass: r.is_ok(),
            note: r.err().unwrap_or_default(),
        });
    }

    // === casp_now_unix_ms ===

    // 12. Nonzero timestamp roughly matches SystemTime::now()
    let now_t = casp_now_unix_ms();
    let sys_t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    let drift_ms = now_t.abs_diff(sys_t);
    diffs.push(CaseDiff {
        case_id: "casp_now_nonzero_and_aligned".into(),
        pass: now_t > 1_700_000_000_000 && drift_ms < 5_000,
        note: format!("now_t={now_t} sys_t={sys_t} drift_ms={drift_ms}"),
    });

    // 13. Subsequent call is monotonic-ish (>= prev)
    let t1 = casp_now_unix_ms();
    sleep(Duration::from_millis(10));
    let t2 = casp_now_unix_ms();
    diffs.push(CaseDiff {
        case_id: "casp_now_monotonic_after_sleep".into(),
        pass: t2 >= t1,
        note: format!("t1={t1} t2={t2}"),
    });

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_runtime_assert_close_helpers".into(),
        category: "fsci_runtime::{assert_close_slice, assert_close_matrix, casp_now_unix_ms}"
            .into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("assert_helpers mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "assert_close helper coverage failed: {} cases",
        diffs.len(),
    );
}
