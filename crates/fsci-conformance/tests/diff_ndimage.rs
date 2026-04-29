#![forbid(unsafe_code)]
//! Live SciPy differential coverage for FSCI-P2C-015 ndimage filters.
//!
//! The existing ndimage E2E tests exercise FrankenSciPy outputs against
//! invariant checks. This harness adds a process-based SciPy oracle for concrete
//! filter semantics across deterministic input families.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    BoundaryMode, NdArray, gaussian_filter, maximum_filter, median_filter, minimum_filter,
    uniform_filter,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct NdimageCase {
    case_id: String,
    op: String,
    shape: [usize; 2],
    data: Vec<f64>,
    size: Option<usize>,
    sigma: Option<f64>,
    mode: String,
    cval: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleCase {
    case_id: String,
    values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    shape: [usize; 2],
    mode: String,
    parameter: String,
    max_abs_diff: f64,
    tolerance: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    tolerance: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create ndimage diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ndimage diff log");
    fs::write(path, json).expect("write ndimage diff log");
}

fn deterministic_data(shape: [usize; 2], seed: usize) -> Vec<f64> {
    let [rows, cols] = shape;
    (0..rows * cols)
        .map(|idx| {
            let row = idx / cols;
            let col = idx % cols;
            let plane = (row as f64 + 1.0) * 0.5 - (col as f64) * 0.25;
            let seed_bias = (seed % 7) as f64 * 0.125 - 0.375;
            let ripple = ((idx + seed * 3) % 5) as f64 * 0.2 - 0.4;
            plane + seed_bias + ripple
        })
        .collect()
}

fn ndimage_cases() -> Vec<NdimageCase> {
    let shapes = [[1, 1], [1, 7], [2, 5], [4, 4], [5, 3]];
    let modes = ["constant", "nearest"];
    let mut cases = Vec::new();

    for (shape_idx, shape) in shapes.iter().copied().enumerate() {
        for mode in modes {
            for size in [1, 3, 5] {
                let cval = if mode == "constant" { -1.5 } else { 0.0 };
                let seed = cases.len() + shape_idx;
                cases.push(NdimageCase {
                    case_id: format!(
                        "uniform_shape{}x{}_mode_{mode}_size_{size}",
                        shape[0], shape[1]
                    ),
                    op: String::from("uniform_filter"),
                    shape,
                    data: deterministic_data(shape, seed),
                    size: Some(size),
                    sigma: None,
                    mode: mode.to_string(),
                    cval,
                });
            }
        }
    }

    for (shape_idx, shape) in shapes.iter().copied().enumerate() {
        for mode in modes {
            for sigma in [0.5, 1.0, 1.25] {
                let cval = if mode == "constant" { 2.25 } else { 0.0 };
                let seed = 100 + cases.len() + shape_idx;
                cases.push(NdimageCase {
                    case_id: format!(
                        "gaussian_shape{}x{}_mode_{mode}_sigma_{sigma:.2}",
                        shape[0], shape[1]
                    ),
                    op: String::from("gaussian_filter"),
                    shape,
                    data: deterministic_data(shape, seed),
                    size: None,
                    sigma: Some(sigma),
                    mode: mode.to_string(),
                    cval,
                });
            }
        }
    }

    cases
}

fn ndimage_rank_cases() -> Vec<NdimageCase> {
    let shapes = [[1, 1], [1, 8], [3, 4], [4, 4], [6, 2]];
    let modes = ["constant", "nearest"];
    let ops = ["median_filter", "minimum_filter", "maximum_filter"];
    let mut cases = Vec::new();

    for (shape_idx, shape) in shapes.iter().copied().enumerate() {
        for mode in modes {
            for op in ops {
                for size in [1, 3, 5] {
                    let cval = if mode == "constant" { -2.75 } else { 0.0 };
                    let seed = 500 + cases.len() + shape_idx;
                    cases.push(NdimageCase {
                        case_id: format!(
                            "{op}_shape{}x{}_mode_{mode}_size_{size}",
                            shape[0], shape[1]
                        ),
                        op: op.to_string(),
                        shape,
                        data: deterministic_data(shape, seed),
                        size: Some(size),
                        sigma: None,
                        mode: mode.to_string(),
                        cval,
                    });
                }
            }
        }
    }

    cases
}

fn boundary_mode(mode: &str) -> Option<BoundaryMode> {
    match mode {
        "constant" => Some(BoundaryMode::Constant),
        "nearest" => Some(BoundaryMode::Nearest),
        _ => None,
    }
}

fn rust_output(case: &NdimageCase) -> Option<Vec<f64>> {
    let array = NdArray::new(case.data.clone(), case.shape.to_vec()).ok()?;
    let mode = boundary_mode(&case.mode)?;
    let output = match case.op.as_str() {
        "uniform_filter" => uniform_filter(&array, case.size?, mode, case.cval),
        "gaussian_filter" => gaussian_filter(&array, case.sigma?, mode, case.cval),
        "median_filter" => median_filter(&array, case.size?, mode, case.cval),
        "minimum_filter" => minimum_filter(&array, case.size?, mode, case.cval),
        "maximum_filter" => maximum_filter(&array, case.size?, mode, case.cval),
        _ => return None,
    }
    .ok()?;
    Some(output.data)
}

fn max_abs_diff(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len(), "oracle output length mismatch");
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max)
}

fn run_scipy_oracle(cases: &[NdimageCase]) -> Option<Vec<OracleCase>> {
    let script = r#"
import json
import sys

import numpy as np
from scipy import ndimage

cases = json.load(sys.stdin)
output = []
for case in cases:
    arr = np.array(case["data"], dtype=np.float64).reshape(case["shape"])
    if case["op"] == "uniform_filter":
        values = ndimage.uniform_filter(
            arr,
            size=int(case["size"]),
            mode=case["mode"],
            cval=float(case["cval"]),
        )
    elif case["op"] == "median_filter":
        values = ndimage.median_filter(
            arr,
            size=int(case["size"]),
            mode=case["mode"],
            cval=float(case["cval"]),
        )
    elif case["op"] == "minimum_filter":
        values = ndimage.minimum_filter(
            arr,
            size=int(case["size"]),
            mode=case["mode"],
            cval=float(case["cval"]),
        )
    elif case["op"] == "maximum_filter":
        values = ndimage.maximum_filter(
            arr,
            size=int(case["size"]),
            mode=case["mode"],
            cval=float(case["cval"]),
        )
    elif case["op"] == "gaussian_filter":
        values = ndimage.gaussian_filter(
            arr,
            sigma=float(case["sigma"]),
            mode=case["mode"],
            cval=float(case["cval"]),
            truncate=4.0,
        )
    else:
        raise ValueError(f"unsupported op: {case['op']}")
    output.append({
        "case_id": case["case_id"],
        "values": [float(value) for value in values.ravel(order="C")],
    })
print(json.dumps(output))
"#;

    let mut child = Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    {
        let stdin = child.stdin.as_mut()?;
        let input = serde_json::to_vec(cases).expect("serialize ndimage oracle input");
        stdin.write_all(&input).expect("write ndimage oracle input");
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        eprintln!(
            "SciPy ndimage oracle unavailable: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    serde_json::from_slice(&output.stdout).ok()
}

fn scipy_oracle_or_skip(test_id: &str, cases: &[NdimageCase]) -> Option<Vec<OracleCase>> {
    let oracle = run_scipy_oracle(cases);
    if oracle.is_none() {
        assert!(
            std::env::var_os(REQUIRE_SCIPY_ENV).is_none(),
            "{REQUIRE_SCIPY_ENV}=1 but SciPy ndimage is unavailable for {test_id}"
        );
        eprintln!(
            "SciPy ndimage not available; skipping {test_id}. Set {REQUIRE_SCIPY_ENV}=1 to fail closed when the oracle is missing"
        );
    }
    oracle
}

#[test]
fn diff_001_ndimage_filters_live_scipy() {
    let cases = ndimage_cases();
    assert_eq!(cases.len(), 60, "ndimage diff case inventory changed");

    let start = Instant::now();
    let Some(oracle_cases) = scipy_oracle_or_skip("diff_001_ndimage_filters_live_scipy", &cases)
    else {
        return;
    };
    assert_eq!(
        oracle_cases.len(),
        cases.len(),
        "SciPy ndimage oracle case count mismatch"
    );

    let mut case_diffs = Vec::with_capacity(cases.len());
    for (case, oracle) in cases.iter().zip(oracle_cases.iter()) {
        assert_eq!(
            case.case_id, oracle.case_id,
            "ndimage oracle case id mismatch"
        );
        let actual = rust_output(case).unwrap_or_default();
        let diff = max_abs_diff(&actual, &oracle.values);
        let parameter = match case.op.as_str() {
            "uniform_filter" => format!("size={}", case.size.expect("uniform size")),
            "gaussian_filter" => format!("sigma={:.2}", case.sigma.expect("gaussian sigma")),
            "median_filter" | "minimum_filter" | "maximum_filter" => {
                format!("size={}", case.size.expect("rank filter size"))
            }
            _ => String::from("unknown"),
        };
        case_diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            shape: case.shape,
            mode: case.mode.clone(),
            parameter,
            max_abs_diff: diff,
            tolerance: TOL,
            pass: diff <= TOL,
        });
    }

    let max_diff = case_diffs
        .iter()
        .map(|case| case.max_abs_diff)
        .fold(0.0_f64, f64::max);
    let pass = case_diffs.iter().all(|case| case.pass);
    let log = DiffLog {
        test_id: String::from("diff_001_ndimage_filters_live_scipy"),
        category: String::from("live_scipy_differential"),
        case_count: cases.len(),
        max_abs_diff: max_diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: case_diffs,
    };
    emit_log(&log);

    assert!(
        pass,
        "ndimage live SciPy diff max_abs_diff={max_diff:.3e} exceeds tolerance {TOL:.3e}"
    );
}

#[test]
fn diff_002_ndimage_rank_filters_live_scipy() {
    let cases = ndimage_rank_cases();
    assert_eq!(
        cases.len(),
        90,
        "ndimage rank-filter diff case inventory changed"
    );

    let start = Instant::now();
    let Some(oracle_cases) =
        scipy_oracle_or_skip("diff_002_ndimage_rank_filters_live_scipy", &cases)
    else {
        return;
    };
    assert_eq!(
        oracle_cases.len(),
        cases.len(),
        "SciPy ndimage rank-filter oracle case count mismatch"
    );

    let mut case_diffs = Vec::with_capacity(cases.len());
    for (case, oracle) in cases.iter().zip(oracle_cases.iter()) {
        assert_eq!(
            case.case_id, oracle.case_id,
            "ndimage rank-filter oracle case id mismatch"
        );
        let actual = rust_output(case).unwrap_or_default();
        let diff = max_abs_diff(&actual, &oracle.values);
        case_diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            shape: case.shape,
            mode: case.mode.clone(),
            parameter: format!("size={}", case.size.expect("rank filter size")),
            max_abs_diff: diff,
            tolerance: TOL,
            pass: diff <= TOL,
        });
    }

    let max_diff = case_diffs
        .iter()
        .map(|case| case.max_abs_diff)
        .fold(0.0_f64, f64::max);
    let pass = case_diffs.iter().all(|case| case.pass);
    let log = DiffLog {
        test_id: String::from("diff_002_ndimage_rank_filters_live_scipy"),
        category: String::from("live_scipy_differential"),
        case_count: cases.len(),
        max_abs_diff: max_diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: case_diffs,
    };
    emit_log(&log);

    assert!(
        pass,
        "ndimage live SciPy rank-filter diff max_abs_diff={max_diff:.3e} exceeds tolerance {TOL:.3e}"
    );
}
