#![forbid(unsafe_code)]
//! Property-based coverage for fsci_ndimage::affine_transform.
//!
//! Resolves [frankenscipy-0s069]. fsci's affine_transform uses a 2×3
//! matrix and an internally-inverted mapping (output coord → input
//! coord), so direct scipy parity is brittle in the general case
//! (scipy.ndimage.affine_transform inverts the matrix internally too,
//! but treats matrix as the OUTPUT→INPUT mapping while fsci accepts
//! INPUT→OUTPUT and inverts). Property-based invariants stay clean:
//!   * Identity matrix returns input unchanged (order ∈ {0, 1})
//!   * Singular matrix (det ≈ 0) returns the all-zero array
//!   * Error: order > 5
//!   * Error: input.ndim != 2

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{BoundaryMode, NdArray, affine_transform};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;

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
    fs::create_dir_all(output_dir()).expect("create affine diff dir");
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
fn diff_ndimage_affine_transform_properties() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // Build a 5×5 test image with sin texture
    let rows = 5;
    let cols = 5;
    let data: Vec<f64> = (0..rows * cols).map(|i| (i as f64).sin() * 10.0).collect();
    let arr = NdArray::new(data.clone(), vec![rows, cols]).expect("ndarray");

    // === 1. Identity affine: matrix = [[1,0,0],[0,1,0]] ===
    // For both order=0 and order=1, output should equal input.
    {
        let identity: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        for order in [0_usize, 1] {
            let result = affine_transform(&arr, &identity, order, BoundaryMode::Constant, 0.0)
                .expect("identity affine");
            let max_diff = result
                .data
                .iter()
                .zip(data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            check(
                &format!("identity_order{order}"),
                result.shape == arr.shape && max_diff <= ABS_TOL,
                format!("max_diff={max_diff}"),
            );
        }
    }

    // === 2. Singular matrix (det=0) → all zeros ===
    {
        let singular: [[f64; 3]; 2] = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        let result = affine_transform(&arr, &singular, 1, BoundaryMode::Constant, 0.0)
            .expect("singular");
        let all_zero = result.data.iter().all(|&v| v == 0.0);
        check(
            "singular_matrix_returns_zeros",
            result.shape == arr.shape && all_zero,
            format!("shape={:?} all_zero={}", result.shape, all_zero),
        );
    }

    // === 3. Error: order > 5 ===
    {
        let identity: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = affine_transform(&arr, &identity, 6, BoundaryMode::Constant, 0.0);
        check(
            "order_too_large_errors",
            result.is_err(),
            format!("res={result:?}"),
        );
    }

    // === 4. Error: input.ndim != 2 (1D input) ===
    {
        let arr_1d = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).expect("ndarray 1d");
        let identity: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = affine_transform(&arr_1d, &identity, 0, BoundaryMode::Constant, 0.0);
        check(
            "non_2d_errors",
            result.is_err(),
            format!("res={result:?}"),
        );
    }

    // === 5. Error: input.ndim != 2 (3D input) ===
    {
        let arr_3d = NdArray::new(vec![1.0; 8], vec![2, 2, 2]).expect("ndarray 3d");
        let identity: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = affine_transform(&arr_3d, &identity, 0, BoundaryMode::Constant, 0.0);
        check(
            "ndim_3_errors",
            result.is_err(),
            format!("res={result:?}"),
        );
    }

    // === 6. Output shape always equals input shape (per fsci semantics) ===
    {
        let scale_2x: [[f64; 3]; 2] = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let result = affine_transform(&arr, &scale_2x, 1, BoundaryMode::Constant, 0.0)
            .expect("scale 2x");
        check(
            "output_shape_eq_input",
            result.shape == arr.shape,
            format!("shape={:?}", result.shape),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_ndimage_affine_transform_properties".into(),
        category: "fsci_ndimage::affine_transform property-based coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("affine_transform mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "affine_transform property coverage failed: {} cases",
        diffs.len()
    );
}
