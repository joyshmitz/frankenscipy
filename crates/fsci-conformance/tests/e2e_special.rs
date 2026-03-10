#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-006 (Special functions).
//!
//! Implements bd-3jh.17.7 acceptance criteria:
//!   Happy-path:     1-3  (function evaluation → chained computation → identity verification)
//!   Error recovery: 4-6  (pole input → handle → continue, domain errors)
//!   Adversarial:    7-10 (extreme arguments, edge values)
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-006/e2e/`.

use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use fsci_runtime::RuntimeMode;
use fsci_special::{
    SpecialError, SpecialErrorKind, SpecialTensor, beta, erf, erfc, erfinv, gamma, gammainc,
    gammaincc, gammaln, j0, j1, rgamma,
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
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-006/e2e")
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
    format!("cargo test -p fsci-conformance --test e2e_special -- {scenario_id} --nocapture")
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

fn real_val(t: &SpecialTensor) -> f64 {
    match t {
        SpecialTensor::RealScalar(v) => *v,
        _ => panic!("expected RealScalar, got {t:?}"),
    }
}

fn scalar(x: f64) -> SpecialTensor {
    SpecialTensor::RealScalar(x)
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

// ═══════════════════════════════════════════════════════════════════
// HAPPY-PATH SCENARIOS (1-3)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 1: Gamma function evaluation chain with known identities
/// gamma(n) = (n-1)!, gamma(1/2) = sqrt(pi)
#[test]
fn e2e_001_gamma_identity_chain() {
    let scenario_id = "e2e_special_001_gamma";
    let overall_start = Instant::now();
    let mode = RuntimeMode::Strict;
    let mut steps = Vec::new();
    let mut all_pass = true;

    // gamma(1) = 1
    let t_start = Instant::now();
    let result = gamma(&scalar(1.0), mode).expect("gamma(1)");
    let val = real_val(&result);
    let pass = (val - 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "gamma_1",
        "gamma",
        "x=1.0",
        &format!("result={val}, expected=1.0"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // gamma(5) = 4! = 24
    let t_start = Instant::now();
    let result = gamma(&scalar(5.0), mode).expect("gamma(5)");
    let val = real_val(&result);
    let pass = (val - 24.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "gamma_5",
        "gamma",
        "x=5.0",
        &format!("result={val}, expected=24.0"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // gamma(1/2) = sqrt(pi)
    let t_start = Instant::now();
    let result = gamma(&scalar(0.5), mode).expect("gamma(0.5)");
    let val = real_val(&result);
    let expected = PI.sqrt();
    let pass = (val - expected).abs() < 1e-8;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "gamma_half",
        "gamma",
        "x=0.5",
        &format!("result={val:.6}, expected={expected:.6}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // gammaln(5) = ln(24)
    let t_start = Instant::now();
    let result = gammaln(&scalar(5.0), mode).expect("gammaln(5)");
    let val = real_val(&result);
    let expected = 24.0_f64.ln();
    let pass = (val - expected).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "gammaln_5",
        "gammaln",
        "x=5.0",
        &format!("result={val:.6}, expected={expected:.6}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
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
    assert!(all_pass, "gamma identity chain failed");
}

/// Scenario 2: Error function chain: erf + erfc = 1, erfinv roundtrip
#[test]
fn e2e_002_error_function_chain() {
    let scenario_id = "e2e_special_002_erf";
    let overall_start = Instant::now();
    let mode = RuntimeMode::Strict;
    let mut steps = Vec::new();
    let mut all_pass = true;

    // erf(0) = 0
    let t_start = Instant::now();
    let result = erf(&scalar(0.0), mode).expect("erf(0)");
    let val = real_val(&result);
    let pass = val.abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "erf_zero",
        "erf",
        "x=0.0",
        &format!("result={val}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // erf(x) + erfc(x) = 1 for x = 1.5
    let t_start = Instant::now();
    let x = 1.5;
    let erf_val = real_val(&erf(&scalar(x), mode).expect("erf(1.5)"));
    let erfc_val = real_val(&erfc(&scalar(x), mode).expect("erfc(1.5)"));
    let sum = erf_val + erfc_val;
    let pass = (sum - 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "erf_erfc_sum",
        "erf+erfc",
        &format!("x={x}"),
        &format!("erf={erf_val:.6}+erfc={erfc_val:.6}={sum:.6}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // erfinv(erf(0.5)) ≈ 0.5
    let t_start = Instant::now();
    let erf_05 = real_val(&erf(&scalar(0.5), mode).expect("erf(0.5)"));
    let roundtrip = real_val(&erfinv(&scalar(erf_05), mode).expect("erfinv"));
    let pass = (roundtrip - 0.5).abs() < 1e-6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "erfinv_roundtrip",
        "erfinv(erf(x))",
        "x=0.5",
        &format!("erf(0.5)={erf_05:.6}, erfinv={roundtrip:.6}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
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
    assert!(all_pass, "error function chain failed");
}

/// Scenario 3: Beta-gamma relation: B(a,b) = gamma(a)*gamma(b)/gamma(a+b)
#[test]
fn e2e_003_beta_gamma_relation() {
    let scenario_id = "e2e_special_003_beta_gamma";
    let overall_start = Instant::now();
    let mode = RuntimeMode::Strict;
    let mut steps = Vec::new();

    let a = 2.0;
    let b = 3.0;

    // beta(a,b)
    let t_start = Instant::now();
    let beta_val = real_val(&beta(&scalar(a), &scalar(b), mode).expect("beta"));
    steps.push(make_step(
        1,
        "beta",
        "beta",
        &format!("a={a}, b={b}"),
        &format!("beta={beta_val:.6}"),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // gamma(a) * gamma(b) / gamma(a+b)
    let t_start = Instant::now();
    let ga = real_val(&gamma(&scalar(a), mode).expect("gamma(a)"));
    let gb = real_val(&gamma(&scalar(b), mode).expect("gamma(b)"));
    let gab = real_val(&gamma(&scalar(a + b), mode).expect("gamma(a+b)"));
    let expected = ga * gb / gab;
    let diff = (beta_val - expected).abs();
    let pass = diff < TOL;
    steps.push(make_step(
        2,
        "verify_relation",
        "gamma+compare",
        &format!("ga={ga:.4}, gb={gb:.4}, gab={gab:.4}"),
        &format!("expected={expected:.6}, diff={diff:.4e}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
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
    assert!(pass, "beta-gamma relation: diff={diff:.4e}");
}

// ═══════════════════════════════════════════════════════════════════
// ERROR RECOVERY SCENARIOS (4-6)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 4: Gamma at negative integer (pole) → NaN in strict → handle
#[test]
fn e2e_004_gamma_pole_recovery() {
    let scenario_id = "e2e_special_004_pole";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: gamma(-2) in strict mode → should return NaN/Inf
    let t_start = Instant::now();
    let result = gamma(&scalar(-2.0), RuntimeMode::Strict);
    let is_nan_or_inf = match &result {
        Ok(t) => {
            let v = real_val(t);
            v.is_nan() || v.is_infinite()
        }
        Err(_) => true,
    };
    steps.push(make_step(
        1,
        "gamma_pole_strict",
        "gamma",
        "x=-2.0, strict",
        &format!("nan_or_inf={is_nan_or_inf}"),
        t_start.elapsed().as_nanos(),
        if is_nan_or_inf {
            "expected_behavior"
        } else {
            "unexpected"
        },
    ));

    // Step 2: gamma(-2) in hardened mode → should return error
    let t_start = Instant::now();
    let result = gamma(&scalar(-2.0), RuntimeMode::Hardened);
    let is_err = result.is_err();
    steps.push(make_step(
        2,
        "gamma_pole_hardened",
        "gamma",
        "x=-2.0, hardened",
        &format!("error={is_err}"),
        t_start.elapsed().as_nanos(),
        if is_err {
            "expected_error"
        } else {
            "unexpected_ok"
        },
    ));

    // Step 3: Recover by using non-pole value
    let t_start = Instant::now();
    let result = gamma(&scalar(3.0), RuntimeMode::Strict).expect("gamma(3)");
    let val = real_val(&result);
    let pass = (val - 2.0).abs() < TOL;
    steps.push(make_step(
        3,
        "gamma_recovery",
        "gamma",
        "x=3.0 (non-pole)",
        &format!("result={val}, expected=2.0, pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let overall_pass = is_nan_or_inf && is_err && pass;
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
    assert!(overall_pass, "gamma pole recovery failed");
}

/// Scenario 5: erfinv domain error → handle → continue
#[test]
fn e2e_005_erfinv_domain_recovery() {
    let scenario_id = "e2e_special_005_erfinv_domain";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: erfinv(1.5) out of domain [-1,1]
    let t_start = Instant::now();
    let result = erfinv(&scalar(1.5), RuntimeMode::Strict);
    let is_nan = match &result {
        Ok(t) => real_val(t).is_nan(),
        Err(_) => true,
    };
    steps.push(make_step(
        1,
        "erfinv_oob_strict",
        "erfinv",
        "x=1.5 (out of [-1,1])",
        &format!("nan_or_error={is_nan}"),
        t_start.elapsed().as_nanos(),
        if is_nan {
            "expected_behavior"
        } else {
            "unexpected"
        },
    ));

    // Step 2: Hardened mode → explicit error
    let t_start = Instant::now();
    let result = erfinv(&scalar(1.5), RuntimeMode::Hardened);
    let is_err = matches!(
        result,
        Err(SpecialError {
            kind: SpecialErrorKind::DomainError,
            ..
        })
    );
    steps.push(make_step(
        2,
        "erfinv_oob_hardened",
        "erfinv",
        "x=1.5, hardened",
        &format!("domain_error={is_err}"),
        t_start.elapsed().as_nanos(),
        if is_err {
            "expected_error"
        } else {
            "unexpected"
        },
    ));

    // Step 3: Valid input succeeds
    let t_start = Instant::now();
    let result = erfinv(&scalar(0.5), RuntimeMode::Strict).expect("erfinv(0.5)");
    let val = real_val(&result);
    let pass = val.is_finite();
    steps.push(make_step(
        3,
        "erfinv_valid",
        "erfinv",
        "x=0.5 (valid)",
        &format!("result={val:.6}, finite={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let overall_pass = is_nan && is_err && pass;
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
    assert!(overall_pass, "erfinv domain recovery failed");
}

/// Scenario 6: Mode switch: strict→hardened behavior change
#[test]
fn e2e_006_mode_switch_behavior() {
    let scenario_id = "e2e_special_006_mode_switch";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // erfinv(1.1) in strict → NaN
    let t_start = Instant::now();
    let strict_result = erfinv(&scalar(1.1), RuntimeMode::Strict);
    let strict_nan = match &strict_result {
        Ok(t) => real_val(t).is_nan(),
        Err(_) => false,
    };
    steps.push(make_step(
        1,
        "strict_domain",
        "erfinv",
        "x=1.1, strict",
        &format!("returns_nan={strict_nan}"),
        t_start.elapsed().as_nanos(),
        if strict_nan { "ok" } else { "unexpected" },
    ));

    // Same input in hardened → error
    let t_start = Instant::now();
    let hardened_result = erfinv(&scalar(1.1), RuntimeMode::Hardened);
    let hardened_err = hardened_result.is_err();
    steps.push(make_step(
        2,
        "hardened_domain",
        "erfinv",
        "x=1.1, hardened",
        &format!("returns_error={hardened_err}"),
        t_start.elapsed().as_nanos(),
        if hardened_err { "ok" } else { "unexpected" },
    ));

    let pass = strict_nan && hardened_err;
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
    assert!(pass, "mode switch behavior failed");
}

// ═══════════════════════════════════════════════════════════════════
// ADVERSARIAL SCENARIOS (7-10)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 7: Bessel functions at x=0 boundary
#[test]
fn e2e_007_bessel_zero_boundary() {
    let scenario_id = "e2e_special_007_bessel";
    let overall_start = Instant::now();
    let mode = RuntimeMode::Strict;
    let mut steps = Vec::new();

    // J0(0) = 1
    let t_start = Instant::now();
    let result = j0(&scalar(0.0), mode).expect("j0(0)");
    let val = real_val(&result);
    let pass_j0 = (val - 1.0).abs() < 1e-8;
    steps.push(make_step(
        1,
        "j0_zero",
        "j0",
        "x=0",
        &format!("result={val}, expected=1.0"),
        t_start.elapsed().as_nanos(),
        if pass_j0 { "ok" } else { "fail" },
    ));

    // J1(0) = 0
    let t_start = Instant::now();
    let result = j1(&scalar(0.0), mode).expect("j1(0)");
    let val = real_val(&result);
    let pass_j1 = val.abs() < 1e-8;
    steps.push(make_step(
        2,
        "j1_zero",
        "j1",
        "x=0",
        &format!("result={val}, expected=0.0"),
        t_start.elapsed().as_nanos(),
        if pass_j1 { "ok" } else { "fail" },
    ));

    let overall_pass = pass_j0 && pass_j1;
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
    assert!(overall_pass, "bessel zero boundary failed");
}

/// Scenario 8: gamma * rgamma = 1 identity
#[test]
fn e2e_008_gamma_rgamma_identity() {
    let scenario_id = "e2e_special_008_rgamma";
    let overall_start = Instant::now();
    let mode = RuntimeMode::Strict;
    let mut steps = Vec::new();
    let mut all_pass = true;

    for &x in &[0.5, 1.0, 2.5, 4.0, 7.5] {
        let t_start = Instant::now();
        let g = real_val(&gamma(&scalar(x), mode).expect("gamma"));
        let rg = real_val(&rgamma(&scalar(x), mode).expect("rgamma"));
        let product = g * rg;
        let pass = (product - 1.0).abs() < 1e-8;
        if !pass {
            all_pass = false;
        }
        steps.push(make_step(
            steps.len() + 1,
            &format!("gamma_rgamma_x{x}"),
            "gamma*rgamma",
            &format!("x={x}"),
            &format!("product={product:.6}, pass={pass}"),
            t_start.elapsed().as_nanos(),
            if pass { "ok" } else { "fail" },
        ));
    }

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
    assert!(all_pass, "gamma*rgamma identity failed");
}

/// Scenario 9: gammainc + gammaincc = 1 identity
#[test]
fn e2e_009_incomplete_gamma_complement() {
    let scenario_id = "e2e_special_009_gammainc";
    let overall_start = Instant::now();
    let mode = RuntimeMode::Strict;
    let mut steps = Vec::new();
    let mut all_pass = true;

    for &(a, x) in &[(1.0, 1.0), (2.0, 0.5), (3.0, 2.0), (0.5, 1.5)] {
        let t_start = Instant::now();
        let inc = real_val(&gammainc(&scalar(a), &scalar(x), mode).expect("gammainc"));
        let incc = real_val(&gammaincc(&scalar(a), &scalar(x), mode).expect("gammaincc"));
        let sum = inc + incc;
        let pass = (sum - 1.0).abs() < 1e-8;
        if !pass {
            all_pass = false;
        }
        steps.push(make_step(
            steps.len() + 1,
            &format!("gammainc_a{a}_x{x}"),
            "gammainc+gammaincc",
            &format!("a={a}, x={x}"),
            &format!("sum={sum:.6}, pass={pass}"),
            t_start.elapsed().as_nanos(),
            if pass { "ok" } else { "fail" },
        ));
    }

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
    assert!(all_pass, "incomplete gamma complement identity failed");
}

/// Scenario 10: Rapid sequential evaluations (no state leakage)
#[test]
fn e2e_010_rapid_sequential() {
    let scenario_id = "e2e_special_010_rapid";
    let overall_start = Instant::now();
    let mode = RuntimeMode::Strict;
    let mut steps = Vec::new();
    let iterations = 100;

    let t_start = Instant::now();
    let mut all_pass = true;
    for i in 0..iterations {
        let x = 0.5 + (i as f64) * 0.1; // x from 0.5 to 10.4
        let g = real_val(&gamma(&scalar(x), mode).expect("gamma"));
        let lg = real_val(&gammaln(&scalar(x), mode).expect("gammaln"));
        // gammaln(x) should equal ln(gamma(x)) for x > 0
        let expected_lg = g.ln();
        if (lg - expected_lg).abs() > 1e-6 {
            all_pass = false;
        }
    }
    steps.push(make_step(
        1,
        "rapid_gamma_gammaln",
        "gamma+gammaln",
        &format!("{iterations} iterations, x=0.5..10.4"),
        &format!("all_pass={all_pass}"),
        t_start.elapsed().as_nanos(),
        if all_pass { "ok" } else { "fail" },
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
