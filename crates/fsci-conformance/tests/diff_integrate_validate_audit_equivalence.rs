#![forbid(unsafe_code)]
//! Property test: fsci_integrate validation audit variants
//! (validate_first_step_with_audit, validate_max_step_with_audit,
//! validate_tol_with_audit) produce the same Result as their
//! non-audit counterparts.
//!
//! Resolves [frankenscipy-nrqqx].

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{
    ToleranceValue, sync_audit_ledger, validate_first_step, validate_first_step_with_audit,
    validate_max_step, validate_max_step_with_audit, validate_tol, validate_tol_with_audit,
};
use fsci_runtime::RuntimeMode;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create validate_audit diff dir");
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
fn diff_integrate_validate_audit_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let ledger = sync_audit_ledger();

    // validate_first_step probes
    let fs_probes: &[(&str, f64, f64, f64)] = &[
        ("ok_small", 0.01, 0.0, 1.0),
        ("ok_mid", 0.5, 0.0, 2.0),
        ("err_too_large", 10.0, 0.0, 1.0),
        ("err_negative", -0.1, 0.0, 1.0),
        ("err_zero", 0.0, 0.0, 1.0),
        ("err_inf", f64::INFINITY, 0.0, 1.0),
    ];
    for (label, fs_val, t0, tb) in fs_probes {
        let plain = validate_first_step(*fs_val, *t0, *tb);
        let audit = validate_first_step_with_audit(*fs_val, *t0, *tb, Some(&ledger));
        let pass = match (plain, audit) {
            (Ok(p), Ok(a)) => p == a,
            (Err(_), Err(_)) => true,
            _ => false,
        };
        diffs.push(CaseDiff {
            case_id: format!("first_step_{label}"),
            op: "first_step".into(),
            pass,
        });
    }

    // validate_max_step probes
    let ms_probes: &[(&str, f64)] = &[
        ("ok_finite", 1.0),
        ("ok_inf", f64::INFINITY),
        ("err_zero", 0.0),
        ("err_neg", -1.0),
        ("err_nan", f64::NAN),
    ];
    for (label, ms_val) in ms_probes {
        let plain = validate_max_step(*ms_val);
        let audit = validate_max_step_with_audit(*ms_val, Some(&ledger));
        let pass = match (plain, audit) {
            (Ok(p), Ok(a)) => p == a,
            (Err(_), Err(_)) => true,
            _ => false,
        };
        diffs.push(CaseDiff {
            case_id: format!("max_step_{label}"),
            op: "max_step".into(),
            pass,
        });
    }

    // validate_tol probes — scalar rtol/atol, varied n
    let tol_probes: &[(&str, ToleranceValue, ToleranceValue, usize, RuntimeMode)] = &[
        ("ok_strict", ToleranceValue::Scalar(1e-6), ToleranceValue::Scalar(1e-9), 3, RuntimeMode::Strict),
        ("ok_hardened", ToleranceValue::Scalar(1e-3), ToleranceValue::Scalar(1e-6), 5, RuntimeMode::Hardened),
        ("err_rtol_nan", ToleranceValue::Scalar(f64::NAN), ToleranceValue::Scalar(1e-9), 3, RuntimeMode::Strict),
        ("err_atol_nan", ToleranceValue::Scalar(1e-6), ToleranceValue::Scalar(f64::NAN), 3, RuntimeMode::Strict),
    ];
    for (label, rtol, atol, n, mode) in tol_probes {
        let plain = validate_tol(rtol.clone(), atol.clone(), *n, *mode);
        let audit = validate_tol_with_audit(rtol.clone(), atol.clone(), *n, *mode, Some(&ledger));
        let pass = match (plain, audit) {
            (Ok(p), Ok(a)) => p == a,
            (Err(_), Err(_)) => true,
            _ => false,
        };
        diffs.push(CaseDiff {
            case_id: format!("tol_{label}"),
            op: "tol".into(),
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_validate_audit_equivalence".into(),
        category: "fsci_integrate validation audit variants equivalent to non-audit".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("{} mismatch: {}", d.op, d.case_id);
        }
    }

    assert!(
        all_pass,
        "validate_audit_equiv conformance failed: {} cases",
        diffs.len(),
    );
}
