//! bd-3jh.13.9: [FSCI-P2C-002-I] Final Evidence Pack
//!
//! Produces a self-contained evidence bundle at:
//!   fixtures/artifacts/P2C-002/evidence/
//!
//! Artifacts:
//! 1. fixture_manifest.json — all test matrices with dimensions, conditions, types
//! 2. parity_gates.json — pass/fail outcomes per operation per fixture
//! 3. risk_notes.json — singular/ill-conditioned handling documentation
//! 4. parity_report.json — max_abs_diff, max_rel_diff per operation per fixture
//! 5. evidence_bundle.raptorq.json — RaptorQ sidecar for the bundle

use fsci_conformance::{RaptorQSidecar, generate_raptorq_sidecar};
use fsci_linalg::{
    InvOptions, LstsqOptions, PinvOptions, SolveOptions, TriangularSolveOptions, det, inv, lstsq,
    pinv, solve, solve_banded, solve_triangular,
};
use fsci_runtime::RuntimeMode;
use serde::Serialize;
use std::path::Path;

// ── Fixture data structures ────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct FixtureManifest {
    packet_id: &'static str,
    generated_at: String,
    fixtures: Vec<FixtureEntry>,
}

#[derive(Debug, Serialize)]
struct FixtureEntry {
    fixture_id: String,
    dimensions: [usize; 2],
    structure: &'static str,
    condition_note: &'static str,
    operations: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct ParityGateReport {
    packet_id: &'static str,
    all_gates_pass: bool,
    gates: Vec<ParityGate>,
}

#[derive(Debug, Serialize)]
struct ParityGate {
    fixture_id: String,
    operation: &'static str,
    pass: bool,
    max_abs_diff: f64,
    max_rel_diff: f64,
    tolerance_used: f64,
    matrix_dims: [usize; 2],
}

#[derive(Debug, Serialize)]
struct RiskNote {
    category: &'static str,
    description: String,
    affected_operations: Vec<&'static str>,
    mitigation: String,
}

#[derive(Debug, Serialize)]
struct RiskNotesReport {
    packet_id: &'static str,
    notes: Vec<RiskNote>,
}

#[derive(Debug, Serialize)]
struct ParityReport {
    packet_id: &'static str,
    operation_summaries: Vec<OperationParitySummary>,
}

#[derive(Debug, Serialize)]
struct OperationParitySummary {
    operation: &'static str,
    total_fixtures: usize,
    passed: usize,
    failed: usize,
    max_abs_diff_across_all: f64,
    max_rel_diff_across_all: f64,
}

#[derive(Debug, Serialize)]
struct EvidenceBundle {
    manifest: FixtureManifest,
    parity_gates: ParityGateReport,
    risk_notes: RiskNotesReport,
    parity_report: ParityReport,
    sidecar: Option<RaptorQSidecar>,
}

// ── Matrix generators ──────────────────────────────────────────────────────────

fn make_diag_dominant(n: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i == j {
                (n as f64) * 2.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }
    a
}

fn make_upper_triangular(n: usize) -> Vec<Vec<f64>> {
    let mut a = make_diag_dominant(n);
    for (i, row) in a.iter_mut().enumerate() {
        for cell in row.iter_mut().take(i) {
            *cell = 0.0;
        }
    }
    a
}

fn make_overdetermined(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; cols]; rows];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i == j {
                (cols as f64) * 2.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }
    a
}

fn make_tridiag_banded(n: usize) -> ((usize, usize), Vec<Vec<f64>>) {
    let mut ab = vec![vec![0.0; n]; 3];
    for col in &mut ab[1] {
        *col = 4.0;
    }
    for col in ab[0].iter_mut().skip(1) {
        *col = -1.0;
    }
    for col in ab[2].iter_mut().take(n.saturating_sub(1)) {
        *col = -1.0;
    }
    ((1, 1), ab)
}

fn make_rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i + 1) as f64).collect()
}

// ── Parity checking helpers ────────────────────────────────────────────────────

fn check_solve_parity(n: usize) -> ParityGate {
    let a = make_diag_dominant(n);
    let b = make_rhs(n);
    let r = solve(&a, &b, SolveOptions::default()).unwrap();
    // Residual: max |A*x - b|
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (i, &bi) in b.iter().enumerate() {
        let ax_i: f64 = a[i].iter().zip(&r.x).map(|(&aij, &xj)| aij * xj).sum();
        let abs_diff = (ax_i - bi).abs();
        max_abs = max_abs.max(abs_diff);
        if bi.abs() > 1e-15 {
            max_rel = max_rel.max(abs_diff / bi.abs());
        }
    }
    ParityGate {
        fixture_id: format!("diag_dominant_{n}x{n}"),
        operation: "solve",
        pass: max_abs < 1e-10,
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
        tolerance_used: 1e-10,
        matrix_dims: [n, n],
    }
}

fn check_solve_triangular_parity(n: usize) -> ParityGate {
    let a = make_upper_triangular(n);
    let b = make_rhs(n);
    let r = solve_triangular(&a, &b, TriangularSolveOptions::default()).unwrap();
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (i, &bi) in b.iter().enumerate() {
        let ax_i: f64 = a[i].iter().zip(&r.x).map(|(&aij, &xj)| aij * xj).sum();
        let abs_diff = (ax_i - bi).abs();
        max_abs = max_abs.max(abs_diff);
        if bi.abs() > 1e-15 {
            max_rel = max_rel.max(abs_diff / bi.abs());
        }
    }
    ParityGate {
        fixture_id: format!("upper_tri_{n}x{n}"),
        operation: "solve_triangular",
        pass: max_abs < 1e-10,
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
        tolerance_used: 1e-10,
        matrix_dims: [n, n],
    }
}

fn check_solve_banded_parity(n: usize) -> ParityGate {
    let (l_u, ab) = make_tridiag_banded(n);
    let b = make_rhs(n);
    let banded = solve_banded(l_u, &ab, &b, SolveOptions::default()).unwrap();
    // Compare with dense solve of equivalent matrix
    let mut dense = vec![vec![0.0; n]; n];
    for j in 0..n {
        dense[j][j] = 4.0;
        if j > 0 {
            dense[j][j - 1] = -1.0;
        }
        if j + 1 < n {
            dense[j][j + 1] = -1.0;
        }
    }
    let dense_r = solve(&dense, &b, SolveOptions::default()).unwrap();
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (i, (&bi, &di)) in banded.x.iter().zip(&dense_r.x).enumerate() {
        let abs_diff = (bi - di).abs();
        max_abs = max_abs.max(abs_diff);
        if di.abs() > 1e-15 {
            max_rel = max_rel.max(abs_diff / di.abs());
        }
    }
    ParityGate {
        fixture_id: format!("tridiag_banded_{n}x{n}"),
        operation: "solve_banded",
        pass: max_abs < 1e-10,
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
        tolerance_used: 1e-10,
        matrix_dims: [n, n],
    }
}

fn check_inv_parity(n: usize) -> ParityGate {
    let a = make_diag_dominant(n);
    let r = inv(&a, InvOptions::default()).unwrap();
    // A @ inv(A) ≈ I
    let mut max_abs = 0.0_f64;
    for (i, a_row) in a.iter().enumerate() {
        for j in 0..n {
            let val: f64 = a_row
                .iter()
                .enumerate()
                .map(|(k, &aik)| aik * r.inverse[k][j])
                .sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            max_abs = max_abs.max((val - expected).abs());
        }
    }
    ParityGate {
        fixture_id: format!("diag_dominant_{n}x{n}"),
        operation: "inv",
        pass: max_abs < 1e-8,
        max_abs_diff: max_abs,
        max_rel_diff: max_abs, // identity elements are 0 or 1
        tolerance_used: 1e-8,
        matrix_dims: [n, n],
    }
}

fn check_det_parity(n: usize) -> ParityGate {
    let a = make_diag_dominant(n);
    let d1 = det(&a, RuntimeMode::Strict, true).unwrap();
    let inv_a = inv(&a, InvOptions::default()).unwrap();
    let d2 = det(&inv_a.inverse, RuntimeMode::Strict, true).unwrap();
    let product = d1 * d2;
    let abs_diff = (product - 1.0).abs();
    ParityGate {
        fixture_id: format!("diag_dominant_{n}x{n}"),
        operation: "det",
        pass: abs_diff < 1e-6,
        max_abs_diff: abs_diff,
        max_rel_diff: abs_diff,
        tolerance_used: 1e-6,
        matrix_dims: [n, n],
    }
}

fn check_lstsq_parity(n: usize) -> ParityGate {
    let rows = n * 2;
    let a = make_overdetermined(rows, n);
    let b = make_rhs(rows);
    let r = lstsq(&a, &b, LstsqOptions::default()).unwrap();
    // Normal equations: A^T A x ≈ A^T b
    let mut max_abs = 0.0_f64;
    for j in 0..n {
        let mut atax_j = 0.0_f64;
        let mut atb_j = 0.0_f64;
        for i in 0..rows {
            atb_j += a[i][j] * b[i];
            for (k, &xk) in r.x.iter().enumerate() {
                atax_j += a[i][j] * a[i][k] * xk;
            }
        }
        max_abs = max_abs.max((atax_j - atb_j).abs());
    }
    ParityGate {
        fixture_id: format!("overdetermined_{rows}x{n}"),
        operation: "lstsq",
        pass: max_abs < 1e-6,
        max_abs_diff: max_abs,
        max_rel_diff: 0.0, // normal equations residual
        tolerance_used: 1e-6,
        matrix_dims: [rows, n],
    }
}

fn check_pinv_parity(n: usize) -> ParityGate {
    let rows = n * 2;
    let a = make_overdetermined(rows, n);
    let b = make_rhs(rows);
    let r = pinv(&a, PinvOptions::default()).unwrap();
    let lstsq_r = lstsq(&a, &b, LstsqOptions::default()).unwrap();
    // pinv(A) @ b ≈ lstsq(A, b)
    let pinv_x: Vec<f64> = r
        .pseudo_inverse
        .iter()
        .map(|row| row.iter().zip(&b).map(|(&p, &bi)| p * bi).sum())
        .collect();
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (i, (&pi, &li)) in pinv_x.iter().zip(&lstsq_r.x).enumerate() {
        let abs_diff = (pi - li).abs();
        max_abs = max_abs.max(abs_diff);
        if li.abs() > 1e-15 {
            max_rel = max_rel.max(abs_diff / li.abs());
        }
    }
    ParityGate {
        fixture_id: format!("overdetermined_{rows}x{n}"),
        operation: "pinv",
        pass: max_abs < 1e-8,
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
        tolerance_used: 1e-8,
        matrix_dims: [rows, n],
    }
}

// ── Main evidence pack test ────────────────────────────────────────────────────

#[test]
fn evidence_p2c002_final_pack() {
    let evidence_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/P2C-002/evidence");
    std::fs::create_dir_all(&evidence_dir).unwrap();

    // 1. Fixture manifest
    let manifest = FixtureManifest {
        packet_id: "FSCI-P2C-002",
        generated_at: now_str(),
        fixtures: vec![
            FixtureEntry {
                fixture_id: "diag_dominant_4x4".into(),
                dimensions: [4, 4],
                structure: "diag_dominant",
                condition_note: "well-conditioned, rcond ~ O(1/n)",
                operations: vec!["solve", "inv", "det"],
            },
            FixtureEntry {
                fixture_id: "diag_dominant_16x16".into(),
                dimensions: [16, 16],
                structure: "diag_dominant",
                condition_note: "well-conditioned",
                operations: vec!["solve", "inv", "det"],
            },
            FixtureEntry {
                fixture_id: "diag_dominant_64x64".into(),
                dimensions: [64, 64],
                structure: "diag_dominant",
                condition_note: "well-conditioned",
                operations: vec!["solve", "inv", "det"],
            },
            FixtureEntry {
                fixture_id: "upper_tri_4x4".into(),
                dimensions: [4, 4],
                structure: "upper_triangular",
                condition_note: "triangular, rcond varies",
                operations: vec!["solve_triangular"],
            },
            FixtureEntry {
                fixture_id: "upper_tri_16x16".into(),
                dimensions: [16, 16],
                structure: "upper_triangular",
                condition_note: "triangular",
                operations: vec!["solve_triangular"],
            },
            FixtureEntry {
                fixture_id: "tridiag_banded_16x16".into(),
                dimensions: [16, 16],
                structure: "tridiagonal_banded",
                condition_note: "banded, l=1, u=1",
                operations: vec!["solve_banded"],
            },
            FixtureEntry {
                fixture_id: "overdetermined_32x16".into(),
                dimensions: [32, 16],
                structure: "overdetermined_rectangular",
                condition_note: "tall matrix, well-conditioned",
                operations: vec!["lstsq", "pinv"],
            },
            FixtureEntry {
                fixture_id: "overdetermined_128x64".into(),
                dimensions: [128, 64],
                structure: "overdetermined_rectangular",
                condition_note: "tall matrix",
                operations: vec!["lstsq", "pinv"],
            },
        ],
    };

    // 2. Parity gates
    let sizes = [4, 16, 64];
    let mut gates = Vec::new();

    for &n in &sizes {
        gates.push(check_solve_parity(n));
    }
    for &n in &[4, 16] {
        gates.push(check_solve_triangular_parity(n));
    }
    gates.push(check_solve_banded_parity(16));
    for &n in &sizes {
        gates.push(check_inv_parity(n));
    }
    for &n in &sizes {
        gates.push(check_det_parity(n));
    }
    for &n in &[16, 64] {
        gates.push(check_lstsq_parity(n));
    }
    for &n in &[16, 64] {
        gates.push(check_pinv_parity(n));
    }

    let all_pass = gates.iter().all(|g| g.pass);
    let parity_gates = ParityGateReport {
        packet_id: "FSCI-P2C-002",
        all_gates_pass: all_pass,
        gates,
    };

    // 3. Risk notes
    let risk_notes = RiskNotesReport {
        packet_id: "FSCI-P2C-002",
        notes: vec![
            RiskNote {
                category: "near-singular matrices",
                description: "Matrices with rcond < 1e-12 may produce inaccurate results. \
                    In Strict mode, a LinalgWarning::IllConditioned is emitted. \
                    In Hardened mode, rcond < HARDENED_RCOND_THRESHOLD (1e-14) \
                    triggers LinalgError::ConditionTooHigh."
                    .into(),
                affected_operations: vec!["solve", "inv", "det"],
                mitigation: "Use RuntimeMode::Hardened for critical paths. Monitor rcond \
                    in SolveResult/InvResult warnings. Fall back to lstsq/pinv for \
                    ill-conditioned systems."
                    .into(),
            },
            RiskNote {
                category: "non-square lstsq",
                description: "lstsq handles overdetermined (m > n) and underdetermined (m < n) \
                    systems via SVD. For underdetermined systems, the minimum-norm solution is \
                    returned. Numerical rank is estimated from singular values."
                    .into(),
                affected_operations: vec!["lstsq"],
                mitigation: "Use LstsqOptions::cond to set a condition threshold for rank \
                    determination. Default driver Gelsd (SVD divide-and-conquer) is most robust."
                    .into(),
            },
            RiskNote {
                category: "ill-conditioned pinv",
                description: "pinv relies on SVD and thresholding of small singular values. \
                    Very ill-conditioned matrices may produce inaccurate pseudoinverses. \
                    atol and rtol parameters control singular value cutoff."
                    .into(),
                affected_operations: vec!["pinv"],
                mitigation: "Set PinvOptions::atol/rtol explicitly for precision control. \
                    Verify via pinv(A) @ b ≈ lstsq(A, b) cross-check."
                    .into(),
            },
            RiskNote {
                category: "banded solver edge cases",
                description: "solve_banded converts to dense internally. For large banded systems \
                    this loses the O(n) advantage. Band storage format follows LAPACK convention: \
                    ab[nupper + i - j][j] = A[i][j]."
                    .into(),
                affected_operations: vec!["solve_banded"],
                mitigation: "For performance-critical banded systems >1000x1000, consider \
                    a dedicated banded solver. Current implementation is correctness-first."
                    .into(),
            },
        ],
    };

    // 4. Parity report (aggregated)
    let ops: Vec<&str> = vec![
        "solve",
        "solve_triangular",
        "solve_banded",
        "inv",
        "det",
        "lstsq",
        "pinv",
    ];
    let operation_summaries: Vec<OperationParitySummary> = ops
        .iter()
        .map(|&op| {
            let op_gates: Vec<_> = parity_gates
                .gates
                .iter()
                .filter(|g| g.operation == op)
                .collect();
            OperationParitySummary {
                operation: op,
                total_fixtures: op_gates.len(),
                passed: op_gates.iter().filter(|g| g.pass).count(),
                failed: op_gates.iter().filter(|g| !g.pass).count(),
                max_abs_diff_across_all: op_gates
                    .iter()
                    .map(|g| g.max_abs_diff)
                    .fold(0.0_f64, f64::max),
                max_rel_diff_across_all: op_gates
                    .iter()
                    .map(|g| g.max_rel_diff)
                    .fold(0.0_f64, f64::max),
            }
        })
        .collect();

    let parity_report = ParityReport {
        packet_id: "FSCI-P2C-002",
        operation_summaries,
    };

    // 5. Generate RaptorQ sidecar for the full bundle
    let bundle = EvidenceBundle {
        manifest,
        parity_gates,
        risk_notes,
        parity_report,
        sidecar: None, // Filled in after serialization
    };

    let bundle_json = serde_json::to_string_pretty(&bundle).unwrap();
    let sidecar = generate_raptorq_sidecar(bundle_json.as_bytes()).ok();

    // Write individual artifacts
    let write = |name: &str, data: &impl Serialize| {
        let json = serde_json::to_string_pretty(data).unwrap();
        std::fs::write(evidence_dir.join(name), &json).unwrap();
    };

    write("fixture_manifest.json", &bundle.manifest);
    write("parity_gates.json", &bundle.parity_gates);
    write("risk_notes.json", &bundle.risk_notes);
    write("parity_report.json", &bundle.parity_report);

    if let Some(ref sc) = sidecar {
        write("evidence_bundle.raptorq.json", sc);
    }

    // Write the full bundle
    let final_bundle = EvidenceBundle { sidecar, ..bundle };
    let final_json = serde_json::to_string_pretty(&final_bundle).unwrap();
    std::fs::write(evidence_dir.join("evidence_bundle.json"), &final_json).unwrap();

    // Assertions
    assert!(
        final_bundle.parity_gates.all_gates_pass,
        "All parity gates must pass"
    );

    // Print summary
    eprintln!("\n── P2C-002 Evidence Pack ──");
    eprintln!(
        "  Parity gates: {}/{} passed",
        final_bundle
            .parity_gates
            .gates
            .iter()
            .filter(|g| g.pass)
            .count(),
        final_bundle.parity_gates.gates.len()
    );
    for s in &final_bundle.parity_report.operation_summaries {
        eprintln!(
            "  {}: {}/{} pass, max_abs={:.2e}, max_rel={:.2e}",
            s.operation,
            s.passed,
            s.total_fixtures,
            s.max_abs_diff_across_all,
            s.max_rel_diff_across_all
        );
    }
    eprintln!(
        "  RaptorQ sidecar: {}",
        if final_bundle.sidecar.is_some() {
            "generated"
        } else {
            "skipped (encoder limitation)"
        }
    );
    eprintln!(
        "  Risk notes: {} categories",
        final_bundle.risk_notes.notes.len()
    );
    eprintln!("  Artifacts written to: {evidence_dir:?}");
}

fn now_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("unix:{secs}")
}
