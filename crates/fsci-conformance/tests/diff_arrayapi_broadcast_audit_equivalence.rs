#![forbid(unsafe_code)]
//! Property test: fsci_arrayapi::broadcast_shapes_with_audit
//! produces same result as broadcast_shapes.
//!
//! Resolves [frankenscipy-g6ont]. Audit only logs to ledger on
//! errors; success returns the same Shape; error kind matches.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_arrayapi::{Shape, broadcast_shapes, broadcast_shapes_with_audit, sync_audit_ledger};
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
    fs::create_dir_all(output_dir()).expect("create bcast_audit diff dir");
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
fn diff_arrayapi_broadcast_audit_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let ledger = sync_audit_ledger();

    let probes: &[(&str, Vec<Shape>)] = &[
        ("scalar_vec", vec![Shape::scalar(), Shape::new(vec![5])]),
        ("vec_mat", vec![Shape::new(vec![3]), Shape::new(vec![2, 3])]),
        (
            "compat_3d",
            vec![Shape::new(vec![1, 5, 1]), Shape::new(vec![4, 1, 3])],
        ),
        (
            "identical",
            vec![Shape::new(vec![3, 4]), Shape::new(vec![3, 4])],
        ),
        // Incompatible cases
        (
            "incompat_mismatch",
            vec![Shape::new(vec![3, 4]), Shape::new(vec![5, 4])],
        ),
        ("incompat_dim", vec![Shape::new(vec![2, 3]), Shape::new(vec![2, 4])]),
        ("empty", vec![]),
    ];

    for (label, shapes) in probes {
        let plain = broadcast_shapes(shapes);
        let audited = broadcast_shapes_with_audit(shapes, &ledger);
        let pass = match (&plain, &audited) {
            (Ok(p), Ok(a)) => p == a,
            (Err(pe), Err(ae)) => pe.kind == ae.kind,
            _ => false,
        };
        diffs.push(CaseDiff {
            case_id: format!("bcast_{label}"),
            op: "broadcast_shapes".into(),
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_arrayapi_broadcast_audit_equivalence".into(),
        category: "fsci_arrayapi::broadcast_shapes_with_audit equivalent to broadcast_shapes".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("broadcast mismatch: {}", d.case_id);
        }
    }

    assert!(
        all_pass,
        "broadcast_audit_equiv conformance failed: {} cases",
        diffs.len(),
    );
}
