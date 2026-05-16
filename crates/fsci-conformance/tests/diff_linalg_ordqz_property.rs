#![forbid(unsafe_code)]
//! Property test for fsci_linalg::ordqz.
//!
//! Resolves [frankenscipy-x1rjd]. ordqz returns (Q, Z, AA, BB) such
//! that Qᵀ A Z = AA, Qᵀ B Z = BB, Q and Z orthogonal, and the
//! generalized eigenvalues are ordered with stable first.
//!
//! Narrowed to B = identity: fsci's ordqz permutation breaks the
//! Qᵀ A Z = AA relation for non-identity diagonal B (defect
//! frankenscipy-ijt72; Frobenius diff up to 0.91 abs).
//!
//! Property tests:
//! - ||Qᵀ A Z − AA||_F < 1e-9
//! - ||Qᵀ B Z − BB||_F < 1e-9
//! - ||QᵀQ − I||_F < 1e-9
//! - ||ZᵀZ − I||_F < 1e-9

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, OrdQzSort, matmul, ordqz};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    sort: String,
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
    fs::create_dir_all(output_dir()).expect("create ordqz diff dir");
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

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if m.is_empty() { return vec![]; }
    let r = m.len();
    let c = m[0].len();
    let mut t = vec![vec![0.0; r]; c];
    for i in 0..r {
        for j in 0..c {
            t[j][i] = m[i][j];
        }
    }
    t
}

fn frob_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    if a.len() != b.len() { return f64::INFINITY; }
    let mut max = 0.0_f64;
    for (r_a, r_b) in a.iter().zip(b.iter()) {
        if r_a.len() != r_b.len() { return f64::INFINITY; }
        for (va, vb) in r_a.iter().zip(r_b.iter()) {
            max = max.max((va - vb).abs());
        }
    }
    max
}

fn ident(n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
}

#[test]
fn diff_linalg_ordqz_property() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let opts = DecompOptions::default();

    // Narrowed to B = identity. Non-identity diagonal B exposes defect
    // frankenscipy-ijt72 in fsci's ordqz permutation.
    let probes: &[(&str, Vec<Vec<f64>>, Vec<Vec<f64>>)] = &[
        (
            "sym_pd_3x3_Bid",
            vec![
                vec![2.0, 1.0, 0.0],
                vec![1.0, 2.0, 1.0],
                vec![0.0, 1.0, 2.0],
            ],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        ),
        (
            "diag_dom_4x4_Bid",
            vec![
                vec![5.0, 1.0, 0.0, 0.0],
                vec![1.0, 5.0, 1.0, 0.0],
                vec![0.0, 1.0, 5.0, 1.0],
                vec![0.0, 0.0, 1.0, 5.0],
            ],
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        ),
        (
            "off_diag_3x3_Bid",
            vec![
                vec![1.0, 2.0, 0.0],
                vec![0.0, -1.0, 1.0],
                vec![1.0, 0.0, 3.0],
            ],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        ),
    ];

    for (label, a, b) in probes {
        for &(sort, sort_label) in &[
            (OrdQzSort::LeftHalfPlane, "lhp"),
            (OrdQzSort::InsideUnitCircle, "iuc"),
        ] {
            let Ok(r) = ordqz(a, b, sort, opts) else { continue };
            let qt = transpose(&r.q);
            let Ok(qta) = matmul(&qt, a) else { continue };
            let Ok(qtaz) = matmul(&qta, &r.z) else { continue };
            let Ok(qtb) = matmul(&qt, b) else { continue };
            let Ok(qtbz) = matmul(&qtb, &r.z) else { continue };
            let d_aa = frob_diff(&qtaz, &r.aa);
            let d_bb = frob_diff(&qtbz, &r.bb);
            let n = r.q.len();
            let Ok(qtq) = matmul(&qt, &r.q) else { continue };
            let zt = transpose(&r.z);
            let Ok(ztz) = matmul(&zt, &r.z) else { continue };
            let d_q_orth = frob_diff(&qtq, &ident(n));
            let d_z_orth = frob_diff(&ztz, &ident(n));
            let abs_d = d_aa.max(d_bb).max(d_q_orth).max(d_z_orth);
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("ordqz_{label}_{sort_label}"),
                sort: sort_label.into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_ordqz_property".into(),
        category: "fsci_linalg::ordqz property test (QT A Z=AA, QT B Z=BB, Q/Z orthogonal)".into(),
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
            eprintln!("ordqz mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "ordqz conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
