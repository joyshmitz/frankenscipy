#![forbid(unsafe_code)]
//! bd-3jh.20: [FOUNDATION] User Workflow Scenario Corpus + Golden Journeys
//!
//! 12+ golden journey scenarios covering all 8 packets, written from user perspective.
//! Each journey exercises a realistic end-to-end workflow and emits fixture snapshots
//! at `fixtures/artifacts/golden_journeys/`.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use fsci_fft::{FftOptions, Normalization, fft, ifft, rfft, irfft};
use fsci_integrate::{SolveIvpOptions, SolverKind, ToleranceValue, solve_ivp};
use fsci_linalg::{
    InvOptions, LstsqOptions, PinvOptions, SolveOptions, det, inv, lstsq, pinv, solve,
};
use fsci_opt::{
    MinimizeOptions, OptimizeMethod, RootMethod, RootOptions,
    bfgs, bisect, brentq, cg_pr_plus, powell,
};
use fsci_runtime::{
    ConformalCalibrator, DecisionSignals, MatrixConditionState, PolicyAction, PolicyController,
    RuntimeMode, SolverPortfolio,
};
use fsci_sparse::{CsrMatrix, FormatConvertible, Shape2D, diags, eye, random, scale_csr, spmv_csr};
use fsci_special::{SpecialTensor, beta, erf, erfc, gamma, gammaln, j0, rgamma};
use serde::Serialize;

type Complex64 = (f64, f64);

#[derive(Debug, Serialize)]
struct JourneyResult {
    journey_id: String,
    description: String,
    category: String,
    packet: String,
    steps: usize,
    all_pass: bool,
    duration_ns: u128,
    snapshot: serde_json::Value,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/golden_journeys")
}

fn write_journey(result: &JourneyResult) {
    let dir = output_dir();
    fs::create_dir_all(&dir).unwrap();
    let json = serde_json::to_string_pretty(result).unwrap();
    fs::write(dir.join(format!("{}.json", result.journey_id)), &json).unwrap();
}

fn real_val(t: &SpecialTensor) -> f64 {
    match t {
        SpecialTensor::RealScalar(v) => *v,
        _ => panic!("expected RealScalar"),
    }
}

fn scalar(x: f64) -> SpecialTensor {
    SpecialTensor::RealScalar(x)
}

fn solve_opts() -> SolveOptions {
    SolveOptions::default()
}

// ═══════════════════════════════════════════════════════════════════
// Journey 1: Linear algebra pipeline (P2C-002)
// ═══════════════════════════════════════════════════════════════════

/// User constructs matrix → computes det → solves Ax=b → verifies residual
#[test]
fn journey_01_linalg_pipeline() {
    let start = Instant::now();
    // 4×4 diag-dominant matrix
    let a = vec![
        vec![10.0, 1.0, 0.0, 0.0],
        vec![1.0, 10.0, 1.0, 0.0],
        vec![0.0, 1.0, 10.0, 1.0],
        vec![0.0, 0.0, 1.0, 10.0],
    ];
    let b = vec![11.0, 12.0, 12.0, 11.0];

    // Compute determinant
    let d = det(&a, RuntimeMode::Strict, true).expect("det");
    assert!(d > 0.0, "positive definite matrix has positive det");

    // Solve
    let result = solve(&a, &b, solve_opts()).expect("solve");
    assert!(result.x.len() == 4);

    // Verify residual: ||Ax - b|| < tol
    let mut residual = vec![0.0; 4];
    for i in 0..4 {
        for j in 0..4 {
            residual[i] += a[i][j] * result.x[j];
        }
        residual[i] -= b[i];
    }
    let max_residual: f64 = residual.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
    assert!(max_residual < 1e-10, "residual too large: {max_residual}");

    write_journey(&JourneyResult {
        journey_id: "gj_01_linalg_pipeline".into(),
        description: "Construct matrix → det → solve Ax=b → verify residual".into(),
        category: "basic_usage".into(),
        packet: "P2C-002".into(),
        steps: 4,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({ "det": d, "max_residual": max_residual }),
    });
}

// ═══════════════════════════════════════════════════════════════════
// Journey 2: Optimization workflow (P2C-003)
// ═══════════════════════════════════════════════════════════════════

/// User defines objective → minimizes with BFGS → checks convergence → compares methods
#[test]
fn journey_02_optimization_workflow() {
    let start = Instant::now();

    let quadratic = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
    let x0 = vec![3.0, -2.0, 1.0];

    // Minimize with BFGS
    let bfgs_result = bfgs(&quadratic, &x0, MinimizeOptions {
        method: Some(OptimizeMethod::Bfgs),
        ..Default::default()
    }).expect("bfgs");
    assert!(bfgs_result.success, "BFGS should converge");

    // Compare with CG
    let cg_result = cg_pr_plus(&quadratic, &x0, MinimizeOptions {
        method: Some(OptimizeMethod::ConjugateGradient),
        ..Default::default()
    }).expect("cg");
    assert!(cg_result.success, "CG should converge");

    // Both should find minimum near origin
    let bfgs_err: f64 = bfgs_result.x.iter().map(|xi| xi.abs()).fold(0.0_f64, f64::max);
    let cg_err: f64 = cg_result.x.iter().map(|xi| xi.abs()).fold(0.0_f64, f64::max);
    assert!(bfgs_err < 1e-3 && cg_err < 1e-3, "both near zero");

    write_journey(&JourneyResult {
        journey_id: "gj_02_optimization_workflow".into(),
        description: "Define objective → minimize BFGS → compare CG → verify convergence".into(),
        category: "basic_usage".into(),
        packet: "P2C-003".into(),
        steps: 4,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({ "bfgs_nfev": bfgs_result.nfev, "cg_nfev": cg_result.nfev }),
    });
}

// ═══════════════════════════════════════════════════════════════════
// Journey 3: IVP solver pipeline (P2C-001)
// ═══════════════════════════════════════════════════════════════════

/// User defines ODE → sets tolerances → integrates → verifies solution
#[test]
fn journey_03_ivp_pipeline() {
    let start = Instant::now();

    let mut decay = |_t: f64, y: &[f64]| -> Vec<f64> { vec![-y[0]] };
    let result = solve_ivp(&mut decay, &SolveIvpOptions {
        t_span: (0.0, 5.0),
        y0: &[1.0],
        method: SolverKind::Rk45,
        rtol: 1e-8,
        atol: ToleranceValue::Scalar(1e-10),
        ..Default::default()
    }).expect("solve_ivp");

    assert!(result.success);
    let y_final = result.y.last().unwrap()[0];
    let expected = (-5.0_f64).exp();
    let err = (y_final - expected).abs();
    assert!(err < 1e-6, "solution error: {err}");

    write_journey(&JourneyResult {
        journey_id: "gj_03_ivp_pipeline".into(),
        description: "Define ODE → set tolerances → integrate → verify y(t)=e^(-t)".into(),
        category: "basic_usage".into(),
        packet: "P2C-001".into(),
        steps: 4,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({ "y_final": y_final, "expected": expected, "err": err, "nfev": result.nfev }),
    });
}

// ═══════════════════════════════════════════════════════════════════
// Journey 4: FFT analysis pipeline (P2C-005)
// ═══════════════════════════════════════════════════════════════════

/// User creates signal → FFT → filter → IFFT → compare
#[test]
fn journey_04_fft_analysis() {
    let start = Instant::now();
    let n = 64;
    let opts = FftOptions::default();

    // Create sinusoidal signal
    let signal: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            ((2.0 * std::f64::consts::PI * t).sin(), 0.0)
        })
        .collect();

    // Forward FFT
    let spectrum = fft(&signal, &opts).expect("fft");
    assert_eq!(spectrum.len(), n);

    // Inverse FFT (roundtrip)
    let recovered = ifft(&spectrum, &opts).expect("ifft");
    let max_err: f64 = signal
        .iter()
        .zip(&recovered)
        .map(|(a, b)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt())
        .fold(0.0_f64, f64::max);
    assert!(max_err < 1e-10, "fft-ifft roundtrip error: {max_err}");

    write_journey(&JourneyResult {
        journey_id: "gj_04_fft_analysis".into(),
        description: "Create signal → FFT → IFFT → verify roundtrip".into(),
        category: "basic_usage".into(),
        packet: "P2C-005".into(),
        steps: 4,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({ "n": n, "max_roundtrip_err": max_err }),
    });
}

// ═══════════════════════════════════════════════════════════════════
// Journey 5: Sparse matrix operations (P2C-004)
// ═══════════════════════════════════════════════════════════════════

/// User creates sparse matrix → spmv → format conversion → arithmetic
#[test]
fn journey_05_sparse_operations() {
    let start = Instant::now();
    let n = 100;

    // Create identity
    let identity = eye(n).expect("eye");
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // spmv with identity should give same vector
    let y = spmv_csr(&identity, &x).expect("spmv");
    let max_err: f64 = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(max_err < 1e-15, "identity spmv error: {max_err}");

    // Create random sparse (COO) → convert to CSR → scale
    let a = random(Shape2D(n, n), 0.05, 42).expect("random").to_csr().expect("to_csr");
    let _scaled = scale_csr(&a, 2.0).expect("scale");

    // Convert to CSC and back
    let csc = a.to_csc().expect("to_csc");
    let roundtrip = csc.to_csr().expect("roundtrip to_csr");
    assert_eq!(roundtrip.shape(), a.shape());

    write_journey(&JourneyResult {
        journey_id: "gj_05_sparse_operations".into(),
        description: "Create sparse → spmv → scale → format conversion roundtrip".into(),
        category: "basic_usage".into(),
        packet: "P2C-004".into(),
        steps: 4,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({ "n": n, "nnz": a.nnz(), "identity_err": max_err }),
    });
}

// ═══════════════════════════════════════════════════════════════════
// Journey 6: Special functions computation chain (P2C-006)
// ═══════════════════════════════════════════════════════════════════

/// User computes gamma → gammaln → erf → erfc identities
#[test]
fn journey_06_special_functions() {
    let start = Instant::now();
    let mode = RuntimeMode::Strict;

    // gamma(5) = 4! = 24
    let g5 = real_val(&gamma(&scalar(5.0), mode).unwrap());
    assert!((g5 - 24.0).abs() < 1e-10, "gamma(5) != 24: {g5}");

    // gammaln(5) = ln(24)
    let gl5 = real_val(&gammaln(&scalar(5.0), mode).unwrap());
    assert!((gl5 - 24.0_f64.ln()).abs() < 1e-10);

    // erf(x) + erfc(x) = 1
    let e = real_val(&erf(&scalar(1.5), mode).unwrap());
    let ec = real_val(&erfc(&scalar(1.5), mode).unwrap());
    assert!((e + ec - 1.0).abs() < 1e-12, "erf+erfc != 1");

    // gamma(x) * rgamma(x) = 1
    let gx = real_val(&gamma(&scalar(3.5), mode).unwrap());
    let rgx = real_val(&rgamma(&scalar(3.5), mode).unwrap());
    assert!((gx * rgx - 1.0).abs() < 1e-10, "gamma*rgamma != 1");

    write_journey(&JourneyResult {
        journey_id: "gj_06_special_functions".into(),
        description: "gamma(5)=24 → gammaln → erf+erfc=1 → gamma*rgamma=1".into(),
        category: "basic_usage".into(),
        packet: "P2C-006".into(),
        steps: 4,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({ "gamma5": g5, "erf_erfc_sum": e+ec }),
    });
}

// ═══════════════════════════════════════════════════════════════════
// Journey 7: CASP policy decision pipeline (P2C-008)
// ═══════════════════════════════════════════════════════════════════

/// User creates CASP controller → makes decisions → inspects ledger
#[test]
fn journey_07_casp_pipeline() {
    let start = Instant::now();

    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);

    // Benign decision
    let d1 = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    assert_eq!(d1.action, PolicyAction::Allow, "benign → Allow");

    // Risky decision
    let d2 = ctrl.decide(DecisionSignals::new(16.0, 0.8, 0.5));
    assert!(
        matches!(d2.action, PolicyAction::FullValidate | PolicyAction::FailClosed),
        "risky → FullValidate or FailClosed"
    );

    // Ledger has 2 entries
    assert_eq!(ctrl.ledger().len(), 2);

    // Portfolio selection
    let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let (action, _, _, _) = portfolio.select_action(&MatrixConditionState::WellConditioned);
    assert!(matches!(action, fsci_runtime::SolverAction::DirectLU));

    write_journey(&JourneyResult {
        journey_id: "gj_07_casp_pipeline".into(),
        description: "CASP controller → benign allow → risky validate → portfolio select".into(),
        category: "basic_usage".into(),
        packet: "P2C-008".into(),
        steps: 4,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({ "benign_action": format!("{:?}", d1.action), "risky_action": format!("{:?}", d2.action) }),
    });
}

// ═══════════════════════════════════════════════════════════════════
// Journey 8: Root-finding pipeline (P2C-003)
// ═══════════════════════════════════════════════════════════════════

/// User finds roots with brentq → verifies → compares with bisect
#[test]
fn journey_08_rootfinding() {
    let start = Instant::now();

    let cubic = |x: f64| -> f64 { x * x * x - 2.0 * x - 5.0 };
    let bq = brentq(cubic, (1.0, 3.0), RootOptions::default()).unwrap();
    assert!(bq.converged);
    assert!(cubic(bq.root).abs() < 1e-10);

    let bs = bisect(cubic, (1.0, 3.0), RootOptions::default()).unwrap();
    assert!(bs.converged);
    assert!((bq.root - bs.root).abs() < 1e-10);

    write_journey(&JourneyResult {
        journey_id: "gj_08_rootfinding".into(),
        description: "brentq finds root → verify → bisect agrees".into(),
        category: "basic_usage".into(),
        packet: "P2C-003".into(),
        steps: 3,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({ "root": bq.root, "f_root": cubic(bq.root) }),
    });
}

// ═══════════════════════════════════════════════════════════════════
// Journey 9: Least-squares + pseudoinverse (P2C-002, advanced)
// ═══════════════════════════════════════════════════════════════════

/// User has overdetermined system → lstsq → pinv → compare solutions
#[test]
fn journey_09_lstsq_pinv() {
    let start = Instant::now();

    // 4×2 overdetermined system
    let a = vec![
        vec![1.0, 1.0],
        vec![1.0, 2.0],
        vec![1.0, 3.0],
        vec![1.0, 4.0],
    ];
    let b = vec![2.1, 3.9, 6.1, 7.9]; // roughly y = 2x

    let ls = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq");
    assert_eq!(ls.x.len(), 2);

    let pi = pinv(&a, PinvOptions::default()).expect("pinv");
    // pinv(A) * b should give similar result to lstsq
    let mut pinv_x = vec![0.0; 2];
    for i in 0..2 {
        for j in 0..4 {
            pinv_x[i] += pi.pseudo_inverse[i][j] * b[j];
        }
    }
    let diff: f64 = ls.x.iter().zip(&pinv_x).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(diff < 1e-8, "lstsq vs pinv@b diff: {diff}");

    write_journey(&JourneyResult {
        journey_id: "gj_09_lstsq_pinv".into(),
        description: "Overdetermined system → lstsq → pinv → compare solutions".into(),
        category: "advanced".into(),
        packet: "P2C-002".into(),
        steps: 3,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({ "lstsq_x": ls.x, "pinv_x": pinv_x, "diff": diff }),
    });
}

// ═══════════════════════════════════════════════════════════════════
// Journey 10: Real FFT pipeline (P2C-005, advanced)
// ═══════════════════════════════════════════════════════════════════

/// User creates real signal → rfft → irfft → verify roundtrip
#[test]
fn journey_10_real_fft() {
    let start = Instant::now();
    let n = 128;
    let opts = FftOptions::default();

    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (6.0 * std::f64::consts::PI * t).cos()
        })
        .collect();

    let spectrum = rfft(&signal, &opts).expect("rfft");
    assert_eq!(spectrum.len(), n / 2 + 1);

    let recovered = irfft(&spectrum, Some(n), &opts).expect("irfft");
    let max_err: f64 = signal.iter().zip(&recovered).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(max_err < 1e-10, "rfft-irfft roundtrip error: {max_err}");

    write_journey(&JourneyResult {
        journey_id: "gj_10_real_fft".into(),
        description: "Real signal → rfft → irfft → verify roundtrip".into(),
        category: "advanced".into(),
        packet: "P2C-005".into(),
        steps: 3,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({ "n": n, "spectrum_len": spectrum.len(), "max_err": max_err }),
    });
}

// ═══════════════════════════════════════════════════════════════════
// Journey 11: Error recovery workflow (cross-packet)
// ═══════════════════════════════════════════════════════════════════

/// User encounters errors → handles gracefully → retries with correct params
#[test]
fn journey_11_error_recovery() {
    let start = Instant::now();

    // Bad tolerance → catch error → fix → succeed
    let bad = fsci_integrate::validate_tol(
        ToleranceValue::Vector(vec![1e-6]),
        ToleranceValue::Scalar(1e-9),
        3, // mismatch: vector len 1 for n=3
        RuntimeMode::Strict,
    );
    assert!(bad.is_err(), "should reject mismatched vector");

    // Fix and retry
    let good = fsci_integrate::validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Scalar(1e-9),
        3,
        RuntimeMode::Strict,
    );
    assert!(good.is_ok(), "scalar should work");

    // Bad bracket → catch → fix → find root
    let bad_root = brentq(|x: f64| x * x - 1.0, (2.0, 3.0), RootOptions::default());
    assert!(bad_root.is_err(), "no sign change in [2,3]");

    let good_root = brentq(|x: f64| x * x - 1.0, (0.0, 2.0), RootOptions::default());
    assert!(good_root.is_ok());
    assert!((good_root.unwrap().root - 1.0).abs() < 1e-10);

    write_journey(&JourneyResult {
        journey_id: "gj_11_error_recovery".into(),
        description: "Encounter errors → handle gracefully → retry with correct params".into(),
        category: "error_recovery".into(),
        packet: "cross-packet".into(),
        steps: 4,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({ "tol_error_caught": true, "bracket_error_caught": true }),
    });
}

// ═══════════════════════════════════════════════════════════════════
// Journey 12: Full cross-packet integration (all packets)
// ═══════════════════════════════════════════════════════════════════

/// User chains operations across multiple subsystems
#[test]
fn journey_12_cross_packet_integration() {
    let start = Instant::now();
    let mode = RuntimeMode::Strict;

    // Step 1: Generate test data with special functions
    let x_vals: Vec<f64> = (1..=5).map(|i| i as f64 * 0.5).collect();
    let gamma_vals: Vec<f64> = x_vals.iter()
        .map(|&x| real_val(&gamma(&scalar(x), mode).unwrap()))
        .collect();

    // Step 2: Solve linear system using gamma values
    let n = 4;
    let a: Vec<Vec<f64>> = (0..n).map(|i| {
        (0..n).map(|j| if i == j { gamma_vals[i] } else { 0.1 }).collect()
    }).collect();
    let b: Vec<f64> = vec![1.0; n];
    let sol = solve(&a, &b, solve_opts()).expect("solve with gamma diagonal");

    // Step 3: Use solution as FFT input
    let fft_input: Vec<Complex64> = sol.x.iter().map(|&v| (v, 0.0)).collect();
    let spectrum = fft(&fft_input, &FftOptions::default()).expect("fft");
    let recovered = ifft(&spectrum, &FftOptions::default()).expect("ifft");
    let fft_err: f64 = fft_input.iter().zip(&recovered)
        .map(|(a, b)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt())
        .fold(0.0_f64, f64::max);
    assert!(fft_err < 1e-10);

    // Step 4: CASP decision on the system
    let cond = det(&a, mode, true).expect("det").abs().log10();
    let mut ctrl = PolicyController::new(mode, 64);
    let decision = ctrl.decide(DecisionSignals::new(cond, 0.0, 0.0));
    let posterior_sum: f64 = decision.posterior.iter().sum();
    assert!((posterior_sum - 1.0).abs() < 1e-9);

    write_journey(&JourneyResult {
        journey_id: "gj_12_cross_packet".into(),
        description: "Special funcs → linalg solve → FFT roundtrip → CASP decision".into(),
        category: "advanced".into(),
        packet: "cross-packet".into(),
        steps: 4,
        all_pass: true,
        duration_ns: start.elapsed().as_nanos(),
        snapshot: serde_json::json!({
            "gamma_vals": gamma_vals,
            "fft_err": fft_err,
            "casp_action": decision.action,
        }),
    });
}
