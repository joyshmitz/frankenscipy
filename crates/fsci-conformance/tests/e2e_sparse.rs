#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-004 (Sparse base/model invariants).
//!
//! Implements bd-3jh.15.7 acceptance criteria:
//!   Happy-path:     1-3  (construction→conversion→solve→verify)
//!   Error recovery: 4-6  (singular matrix, shape mismatch, mode switch)
//!   Adversarial:    7-10 (pathological sparsity, edge sizes, stress)
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-004/e2e/`.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use fsci_runtime::RuntimeMode;
use fsci_sparse::{
    CooMatrix, CsrMatrix, FormatConvertible, Shape2D, SolveOptions, SparseError,
    add_csr, csr_to_csc_with_mode, diags, eye,
    scale_csr, spmv_csr, spsolve, sub_csr,
};
use serde::Serialize;

// ───────────────────────── Forensic log types ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    artifacts: Vec<ArtifactRef>,
    environment: EnvironmentInfo,
    overall: OverallResult,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicStep {
    step_id: usize,
    step_name: String,
    action: String,
    input_summary: String,
    output_summary: String,
    duration_ns: u128,
    mode: String,
    outcome: String,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactRef {
    path: String,
    blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
    total_memory_mb: String,
}

#[derive(Debug, Clone, Serialize)]
struct OverallResult {
    status: String,
    total_duration_ns: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

// ───────────────────────── Helpers ─────────────────────────

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-004/e2e")
}

fn make_env() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: String::from(env!("CARGO_PKG_VERSION")),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        total_memory_mb: String::from("unknown"),
    }
}

fn replay_cmd(scenario_id: &str) -> String {
    format!("cargo test -p fsci-conformance --test e2e_sparse -- {scenario_id} --nocapture")
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("failed to create e2e dir {}: {e}", dir.display()));
    let path = dir.join(format!("{scenario_id}.json"));
    let json = serde_json::to_vec_pretty(bundle).expect("serialize bundle");
    fs::write(&path, &json).unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}

const TOL: f64 = 1e-10;

/// Build a tridiagonal matrix: sub-diagonal, main diagonal, super-diagonal.
fn make_tridiag(n: usize, sub: f64, main: f64, sup: f64) -> CsrMatrix {
    let sub_diag = vec![sub; n.saturating_sub(1)];
    let main_diag = vec![main; n];
    let sup_diag = vec![sup; n.saturating_sub(1)];
    diags(&[sub_diag, main_diag, sup_diag], &[-1, 0, 1], Some(Shape2D::new(n, n)))
        .expect("tridiag")
}

/// Build a bidiagonal matrix (main + one off-diagonal).
fn make_bidiag(n: usize, main: f64, off: f64, offset: isize) -> CsrMatrix {
    let main_diag = vec![main; n];
    let off_len = if offset.unsigned_abs() < n { n - offset.unsigned_abs() } else { 0 };
    let off_diag = vec![off; off_len];
    diags(&[main_diag, off_diag], &[0, offset], Some(Shape2D::new(n, n)))
        .expect("bidiag")
}

fn make_step(
    step_id: usize,
    name: &str,
    action: &str,
    input: &str,
    output: &str,
    dur: u128,
    outcome: &str,
) -> ForensicStep {
    ForensicStep {
        step_id,
        step_name: name.to_string(),
        action: action.to_string(),
        input_summary: input.to_string(),
        output_summary: output.to_string(),
        duration_ns: dur,
        mode: "strict".to_string(),
        outcome: outcome.to_string(),
    }
}

fn dense_from_csr(csr: &CsrMatrix) -> Vec<Vec<f64>> {
    let shape = csr.shape();
    let coo = csr.to_coo().expect("csr->coo");
    let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
    for idx in 0..coo.nnz() {
        dense[coo.row_indices()[idx]][coo.col_indices()[idx]] += coo.data()[idx];
    }
    dense
}

fn max_abs_diff_vec(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ═══════════════════════════════════════════════════════════════════
// HAPPY-PATH SCENARIOS (1-3)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 1: Construct tridiagonal → convert COO→CSR→CSC→COO → verify roundtrip
#[test]
fn e2e_001_tridiagonal_format_roundtrip() {
    let scenario_id = "e2e_sparse_001_tridiag";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 10;

    // Step 1: Construct tridiagonal matrix via diags
    let t_start = Instant::now();
    let tri = make_tridiag(n, -1.0, 2.0, -1.0);
    steps.push(make_step(
        1, "construct_tridiag", "diags",
        &format!("n={n}, diags=[-1,2,-1]"),
        &format!("csr nnz={}", tri.nnz()),
        t_start.elapsed().as_nanos(), "ok",
    ));

    // Step 2: CSR → CSC
    let t_start = Instant::now();
    let (csc, _log1) = csr_to_csc_with_mode(&tri, RuntimeMode::Strict, "e2e-conv-1")
        .expect("csr->csc");
    steps.push(make_step(
        2, "csr_to_csc", "convert", &format!("csr nnz={}", tri.nnz()),
        &format!("csc nnz={}", csc.nnz()),
        t_start.elapsed().as_nanos(), "ok",
    ));

    // Step 3: CSC → COO → CSR → verify
    let t_start = Instant::now();
    let coo = csc.to_coo().expect("csc->coo");
    let roundtrip = coo.to_csr().expect("coo->csr");
    let dense_orig = dense_from_csr(&tri);
    let dense_rt = dense_from_csr(&roundtrip);
    let mut max_diff = 0.0_f64;
    for (ro, rr) in dense_orig.iter().zip(dense_rt.iter()) {
        max_diff = max_diff.max(max_abs_diff_vec(ro, rr));
    }
    let pass = max_diff < TOL;
    steps.push(make_step(
        3, "roundtrip_verify", "compare",
        &format!("max_diff={max_diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(), if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "tridiagonal roundtrip: max_diff={max_diff:.4e}");
}

/// Scenario 2: Construct tridiag → spmv → verify against dense matvec
#[test]
fn e2e_002_spmv_verify_dense() {
    let scenario_id = "e2e_sparse_002_spmv";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 5;

    // Step 1: Build tridiagonal system
    let t_start = Instant::now();
    let a = make_tridiag(n, -1.0, 3.0, -1.0);
    let v: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
    steps.push(make_step(
        1, "build_system", "diags+vector",
        &format!("n={n}, diag=[-1,3,-1]"),
        &format!("A nnz={}, v len={}", a.nnz(), v.len()),
        t_start.elapsed().as_nanos(), "ok",
    ));

    // Step 2: spmv
    let t_start = Instant::now();
    let result = spmv_csr(&a, &v).expect("spmv");
    steps.push(make_step(
        2, "spmv", "spmv_csr",
        "A*v",
        &format!("result len={}", result.len()),
        t_start.elapsed().as_nanos(), "ok",
    ));

    // Step 3: Compare with dense matvec
    let t_start = Instant::now();
    let dense = dense_from_csr(&a);
    let expected: Vec<f64> = dense
        .iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect();
    let diff = max_abs_diff_vec(&result, &expected);
    let pass = diff < TOL;
    steps.push(make_step(
        3, "verify_dense", "compare",
        &format!("diff={diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(), if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "spmv vs dense: diff={diff:.4e}");
}

/// Scenario 3: Arithmetic pipeline: A+B, A-B, scale, spmv consistency
#[test]
fn e2e_003_arithmetic_pipeline() {
    let scenario_id = "e2e_sparse_003_arith";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 4;

    // Step 1: Build A and B
    let t_start = Instant::now();
    let a = make_bidiag(n, 2.0, -1.0, 1);
    let b = make_bidiag(n, 1.0, 1.0, -1);
    steps.push(make_step(
        1, "build_matrices", "diags",
        &format!("n={n}"),
        &format!("A nnz={}, B nnz={}", a.nnz(), b.nnz()),
        t_start.elapsed().as_nanos(), "ok",
    ));

    // Step 2: C = A + B
    let t_start = Instant::now();
    let c = add_csr(&a, &b).expect("A+B");
    steps.push(make_step(
        2, "add", "add_csr",
        "A+B",
        &format!("C nnz={}", c.nnz()),
        t_start.elapsed().as_nanos(), "ok",
    ));

    // Step 3: D = A - B
    let t_start = Instant::now();
    let d = sub_csr(&a, &b).expect("A-B");
    steps.push(make_step(
        3, "subtract", "sub_csr",
        "A-B",
        &format!("D nnz={}", d.nnz()),
        t_start.elapsed().as_nanos(), "ok",
    ));

    // Step 4: Verify (A+B) + (A-B) = 2*A via spmv
    let t_start = Instant::now();
    let v: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
    let cpd = add_csr(&c, &d).expect("C+D");
    let two_a = scale_csr(&a, 2.0).expect("2*A");
    let lhs = spmv_csr(&cpd, &v).expect("(C+D)*v");
    let rhs = spmv_csr(&two_a, &v).expect("2A*v");
    let diff = max_abs_diff_vec(&lhs, &rhs);
    let pass = diff < TOL;
    steps.push(make_step(
        4, "verify_arithmetic", "spmv+compare",
        &format!("(A+B)+(A-B) vs 2A, diff={diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(), if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "arithmetic pipeline failed: diff={diff:.4e}");
}

// ═══════════════════════════════════════════════════════════════════
// ERROR RECOVERY SCENARIOS (4-6)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 4: Non-square spsolve → shape error → switch to spmv → succeed
#[test]
fn e2e_004_nonsquare_solve_recovery() {
    let scenario_id = "e2e_sparse_004_nonsq";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: Try spsolve on non-square matrix
    let t_start = Instant::now();
    let rect = CooMatrix::from_triplets(
        Shape2D::new(3, 4),
        vec![1.0, 2.0, 3.0],
        vec![0, 1, 2],
        vec![0, 1, 2],
        false,
    )
    .expect("rect coo")
    .to_csr()
    .expect("rect csr");
    let b = vec![1.0, 2.0, 3.0];
    let opts = SolveOptions::default();
    let result = spsolve(&rect, &b, opts);
    let is_err = matches!(result, Err(SparseError::InvalidShape { .. }));
    steps.push(make_step(
        1, "solve_nonsquare", "spsolve",
        "3x4 non-square matrix",
        &format!("got_invalid_shape={is_err}"),
        t_start.elapsed().as_nanos(), if is_err { "expected_error" } else { "unexpected_ok" },
    ));

    // Step 2: Fall back to spmv which works on non-square
    let t_start = Instant::now();
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let result = spmv_csr(&rect, &v);
    let pass = result.is_ok();
    steps.push(make_step(
        2, "fallback_spmv", "spmv_csr",
        "3x4 matrix, vector len=4",
        &format!("success={pass}"),
        t_start.elapsed().as_nanos(), if pass { "ok" } else { "fail" },
    ));

    let overall_pass = is_err && pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "nonsquare solve recovery failed");
}

/// Scenario 5: Hardened mode rejects unsorted → switch to strict → succeed
#[test]
fn e2e_005_mode_switch_recovery() {
    let scenario_id = "e2e_sparse_005_mode_switch";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: Build unsorted CSR
    let t_start = Instant::now();
    let unsorted = CsrMatrix::from_components(
        Shape2D::new(2, 3),
        vec![1.0, 2.0, 3.0],
        vec![2, 0, 1],
        vec![0, 2, 3],
        false,
    )
    .expect("unsorted csr");
    steps.push(make_step(
        1, "build_unsorted", "from_components",
        "2x3 unsorted CSR",
        &format!("sorted={}", unsorted.canonical_meta().sorted_indices),
        t_start.elapsed().as_nanos(), "ok",
    ));

    // Step 2: Hardened conversion → error
    let t_start = Instant::now();
    let result = csr_to_csc_with_mode(&unsorted, RuntimeMode::Hardened, "e2e-hard");
    let is_err = result.is_err();
    steps.push(make_step(
        2, "hardened_convert", "csr_to_csc",
        "hardened mode, unsorted input",
        &format!("error={is_err}"),
        t_start.elapsed().as_nanos(), if is_err { "expected_error" } else { "unexpected_ok" },
    ));

    // Step 3: Strict mode → success
    let t_start = Instant::now();
    let (csc, _) = csr_to_csc_with_mode(&unsorted, RuntimeMode::Strict, "e2e-strict")
        .expect("strict csr->csc");
    let pass = csc.nnz() == unsorted.nnz();
    steps.push(make_step(
        3, "strict_convert", "csr_to_csc",
        "strict mode, same unsorted input",
        &format!("success={pass}, nnz={}", csc.nnz()),
        t_start.elapsed().as_nanos(), if pass { "ok" } else { "fail" },
    ));

    let overall_pass = is_err && pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "mode switch recovery failed");
}

/// Scenario 6: Vector length mismatch → diagnose → fix → succeed
#[test]
fn e2e_006_spmv_mismatch_recovery() {
    let scenario_id = "e2e_sparse_006_spmv_mismatch";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    let a = eye(4).expect("4x4 identity");

    // Step 1: Wrong vector length
    let t_start = Instant::now();
    let result = spmv_csr(&a, &[1.0, 2.0]); // expects 4
    let is_err = matches!(result, Err(SparseError::IncompatibleShape { .. }));
    steps.push(make_step(
        1, "spmv_wrong_len", "spmv_csr",
        "4x4 matrix, vector len=2",
        &format!("error={is_err}"),
        t_start.elapsed().as_nanos(), if is_err { "expected_error" } else { "unexpected_ok" },
    ));

    // Step 2: Correct vector length
    let t_start = Instant::now();
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let result = spmv_csr(&a, &v).expect("spmv");
    let diff = max_abs_diff_vec(&result, &v);
    let pass = diff < TOL;
    steps.push(make_step(
        2, "spmv_correct", "spmv_csr",
        "4x4 identity, vector len=4",
        &format!("diff={diff:.4e}, pass={pass}"),
        t_start.elapsed().as_nanos(), if pass { "ok" } else { "fail" },
    ));

    let overall_pass = is_err && pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "spmv mismatch recovery failed");
}

// ═══════════════════════════════════════════════════════════════════
// ADVERSARIAL SCENARIOS (7-10)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 7: Zero-NNZ matrix operations
#[test]
fn e2e_007_zero_nnz_operations() {
    let scenario_id = "e2e_sparse_007_zero_nnz";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 5;

    let t_start = Instant::now();
    let zero = CooMatrix::from_triplets(Shape2D::new(n, n), vec![], vec![], vec![], false)
        .expect("zero coo")
        .to_csr()
        .expect("zero csr");
    steps.push(make_step(
        1, "build_zero", "from_triplets",
        &format!("{n}x{n} zero matrix"),
        &format!("nnz={}", zero.nnz()),
        t_start.elapsed().as_nanos(), "ok",
    ));

    // spmv with zero matrix
    let t_start = Instant::now();
    let v: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
    let result = spmv_csr(&zero, &v).expect("spmv");
    let expected = vec![0.0; n];
    let diff = max_abs_diff_vec(&result, &expected);
    let pass_spmv = diff < TOL;
    steps.push(make_step(
        2, "spmv_zero", "spmv_csr",
        "zero * v",
        &format!("all_zero={pass_spmv}"),
        t_start.elapsed().as_nanos(), if pass_spmv { "ok" } else { "fail" },
    ));

    // add zero + identity
    let t_start = Instant::now();
    let id = eye(n).expect("identity");
    let sum = add_csr(&zero, &id).expect("zero+I");
    let result = spmv_csr(&sum, &v).expect("spmv");
    let diff = max_abs_diff_vec(&result, &v);
    let pass_add = diff < TOL;
    steps.push(make_step(
        3, "add_zero_identity", "add_csr+spmv",
        "0+I = I",
        &format!("pass={pass_add}"),
        t_start.elapsed().as_nanos(), if pass_add { "ok" } else { "fail" },
    ));

    let overall_pass = pass_spmv && pass_add;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "zero-nnz operations failed");
}

/// Scenario 8: Single-element 1x1 matrix through full pipeline
#[test]
fn e2e_008_single_element() {
    let scenario_id = "e2e_sparse_008_single";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    let t_start = Instant::now();
    let coo = CooMatrix::from_triplets(Shape2D::new(1, 1), vec![7.0], vec![0], vec![0], false)
        .expect("1x1 coo");
    let csr = coo.to_csr().expect("csr");
    let (csc, _) = csr_to_csc_with_mode(&csr, RuntimeMode::Strict, "e2e-1x1")
        .expect("csc");
    let back = csc.to_coo().expect("coo").to_csr().expect("csr");
    let dense_orig = dense_from_csr(&csr);
    let dense_back = dense_from_csr(&back);
    let diff = max_abs_diff_vec(&dense_orig[0], &dense_back[0]);
    let pass_rt = diff < TOL;
    steps.push(make_step(
        1, "roundtrip_1x1", "convert",
        "1x1 matrix val=7",
        &format!("pass={pass_rt}"),
        t_start.elapsed().as_nanos(), if pass_rt { "ok" } else { "fail" },
    ));

    // spmv: 7 * [2.0] = [14.0]
    let t_start = Instant::now();
    let result = spmv_csr(&csr, &[2.0]).expect("spmv");
    let pass_solve = (result[0] - 14.0).abs() < TOL;
    steps.push(make_step(
        2, "spmv_1x1", "spmv_csr",
        "7 * [2] = [14]",
        &format!("result={}, pass={pass_solve}", result[0]),
        t_start.elapsed().as_nanos(), if pass_solve { "ok" } else { "fail" },
    ));

    let overall_pass = pass_rt && pass_solve;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "single element pipeline failed");
}

/// Scenario 9: Dense-like full matrix (pathological sparsity)
#[test]
fn e2e_009_dense_fill_pattern() {
    let scenario_id = "e2e_sparse_009_dense_fill";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 8;

    // Build fully dense matrix as sparse
    let t_start = Instant::now();
    let mut data = Vec::new();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for r in 0..n {
        for c in 0..n {
            rows.push(r);
            cols.push(c);
            data.push(if r == c { (n as f64) + 1.0 } else { 1.0 });
        }
    }
    let coo = CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .expect("dense-as-sparse coo");
    let csr = coo.to_csr().expect("csr");
    steps.push(make_step(
        1, "build_dense_sparse", "from_triplets",
        &format!("{n}x{n} fully dense as sparse"),
        &format!("nnz={}", csr.nnz()),
        t_start.elapsed().as_nanos(), "ok",
    ));

    // Verify spmv against dense matvec
    let t_start = Instant::now();
    let v: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
    let sparse_result = spmv_csr(&csr, &v).expect("spmv");
    let dense = dense_from_csr(&csr);
    let expected: Vec<f64> = dense
        .iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect();
    let diff = max_abs_diff_vec(&sparse_result, &expected);
    let pass = diff < TOL;
    steps.push(make_step(
        2, "spmv_verify", "spmv_csr+dense",
        &format!("diff={diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(), if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "dense fill pattern: diff={diff:.4e}");
}

/// Scenario 10: Rapid sequential operations (no state leakage)
#[test]
fn e2e_010_rapid_sequential() {
    let scenario_id = "e2e_sparse_010_rapid";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let iterations = 50;

    let t_start = Instant::now();
    let mut all_pass = true;
    for i in 0..iterations {
        let n = 3 + (i % 5); // sizes 3..7
        let a = make_tridiag(n, -1.0, 3.0, -1.0);
        let v: Vec<f64> = (0..n).map(|j| ((i * n + j) as f64) * 0.1 + 1.0).collect();
        let result = spmv_csr(&a, &v).expect("spmv");
        let dense = dense_from_csr(&a);
        let expected: Vec<f64> = dense
            .iter()
            .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
            .collect();
        let diff = max_abs_diff_vec(&result, &expected);
        if diff > TOL {
            all_pass = false;
        }
    }
    steps.push(make_step(
        1, "rapid_spmv", "spmv+verify",
        &format!("{iterations} iterations, sizes 3-7"),
        &format!("all_pass={all_pass}"),
        t_start.elapsed().as_nanos(), if all_pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(all_pass, "rapid sequential: state leakage detected");
}
