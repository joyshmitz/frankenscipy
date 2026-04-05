#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-001 (IVP solver).
//!
//! Implements bd-3jh.12.7 acceptance criteria:
//!   Happy-path:     1-3  (full IVP workflow → tolerance cascade → multi-step validation)
//!   Error recovery: 4-6  (invalid tolerance → mode switch → boundary tolerance)
//!   Adversarial:    7-8  (rapid sequential validations → large system dimension)
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-001/e2e/`.

use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use fsci_integrate::{
    SolveIvpOptions, SolverKind, ToleranceValue, solve_ivp, validate_first_step, validate_max_step,
    validate_tol,
};
use fsci_runtime::RuntimeMode;
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
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-001/e2e")
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
    format!(
        "rch exec -- cargo test -p fsci-conformance --test e2e_ivp -- {scenario_id} --nocapture"
    )
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) -> Result<(), String> {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create e2e dir {}: {e}", dir.display()))?;
    let path = dir.join(format!("{scenario_id}.json"));
    let json = serde_json::to_vec_pretty(bundle)
        .map_err(|e| format!("failed to serialize bundle: {e}"))?;
    fs::write(&path, &json).map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    Ok(())
}

fn assert_bundle_written(scenario_id: &str, bundle: &ForensicLogBundle) {
    let bundle_write = write_bundle(scenario_id, bundle);
    assert!(
        bundle_write.is_ok(),
        "bundle write failed for {scenario_id}: {}",
        bundle_write
            .as_ref()
            .err()
            .map_or("unknown error", String::as_str)
    );
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

// ── Standard test ODEs ────────────────────────────────────────────────

/// dy/dt = -y, y(0) = 1  →  y(t) = e^(-t)
fn exponential_decay(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![-y[0]]
}

/// dy/dt = [y[1], -y[0]], harmonic oscillator
fn harmonic_oscillator(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![y[1], -y[0]]
}

/// Lotka-Volterra prey-predator
fn lotka_volterra(_t: f64, y: &[f64]) -> Vec<f64> {
    let alpha = 1.5;
    let beta = 1.0;
    let delta = 1.0;
    let gamma_val = 3.0;
    vec![
        alpha * y[0] - beta * y[0] * y[1],
        delta * y[0] * y[1] - gamma_val * y[1],
    ]
}

fn stiff_decay(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![-1000.0 * y[0]]
}

// ═══════════════════════════════════════════════════════════════════
// HAPPY-PATH SCENARIOS (1-3)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 1: Full IVP setup workflow
/// Create tolerances → validate → configure solver → solve → verify solution
#[test]
fn e2e_001_full_ivp_workflow() {
    let scenario_id = "e2e_ivp_001_full_workflow";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: Validate tolerances
    let t = Instant::now();
    let vtol = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Scalar(1e-6),
        1,
        RuntimeMode::Strict,
    )
    .expect("validate_tol");
    steps.push(make_step(
        1,
        "validate_tol",
        "validate",
        "rtol=1e-3, atol=1e-6, n=1",
        &format!("warnings={}", vtol.warnings.len()),
        t.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Validate first step
    let t = Instant::now();
    let fs = validate_first_step(0.01, 0.0, 10.0).expect("validate_first_step");
    let pass = fs == 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "validate_first_step",
        "validate",
        "first_step=0.01, t0=0.0, t_bound=10.0",
        &format!("result={fs}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Validate max step
    let t = Instant::now();
    let ms = validate_max_step(1.0).expect("validate_max_step");
    let pass = ms == 1.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "validate_max_step",
        "validate",
        "max_step=1.0",
        &format!("result={ms}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 4: Solve IVP (exponential decay)
    let t = Instant::now();
    let mut fun = exponential_decay;
    let result = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 5.0),
            y0: &[1.0],
            method: SolverKind::Rk45,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            ..Default::default()
        },
    )
    .expect("solve_ivp");
    let pass = result.success;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "solve_ivp",
        "solve",
        "dy/dt=-y, t=[0,5], y0=1",
        &format!(
            "success={}, nfev={}, t_len={}",
            result.success,
            result.nfev,
            result.t.len()
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 5: Verify solution accuracy
    let t = Instant::now();
    let t_final = *result.t.last().unwrap();
    let y_final = result.y.last().unwrap()[0];
    let expected = (-t_final).exp();
    let rel_err = (y_final - expected).abs() / expected;
    let pass = rel_err < 1e-6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "verify_solution",
        "check",
        &format!("y({t_final:.2})={y_final:.8e}"),
        &format!("expected={expected:.8e}, rel_err={rel_err:.2e}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    assert_bundle_written(scenario_id, &bundle);
    assert!(all_pass, "Scenario 1 failed");
}

/// Scenario 2: Tolerance cascade — scalar and vector tolerances propagate correctly
#[test]
fn e2e_002_tolerance_cascade() {
    let scenario_id = "e2e_ivp_002_tolerance_cascade";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: Scalar tolerances
    let t = Instant::now();
    let vtol = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Scalar(1e-9),
        3,
        RuntimeMode::Strict,
    )
    .expect("scalar validate");
    let pass = vtol.warnings.is_empty();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "scalar_tol",
        "validate",
        "rtol=1e-6, atol=1e-9, n=3",
        &format!("warnings={}", vtol.warnings.len()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 2: Vector tolerances
    let t = Instant::now();
    let vtol = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Vector(vec![1e-8, 1e-9, 1e-10]),
        3,
        RuntimeMode::Strict,
    )
    .expect("vector validate");
    let pass = vtol.warnings.is_empty();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "vector_atol",
        "validate",
        "rtol=1e-6, atol=[1e-8, 1e-9, 1e-10], n=3",
        &format!("warnings={}", vtol.warnings.len()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Solve with vector tolerances (harmonic oscillator)
    let t = Instant::now();
    let mut fun = harmonic_oscillator;
    let result = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 2.0 * PI),
            y0: &[1.0, 0.0],
            method: SolverKind::Rk45,
            rtol: 1e-8,
            atol: ToleranceValue::Vector(vec![1e-10, 1e-10]),
            ..Default::default()
        },
    )
    .expect("solve_ivp harmonic");
    let pass = result.success;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "solve_harmonic",
        "solve",
        "harmonic oscillator, t=[0,2pi], y0=[1,0]",
        &format!("success={}, nfev={}", result.success, result.nfev),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 4: Verify periodicity (y(2pi) ≈ y(0))
    let t = Instant::now();
    let y_final = result.y.last().unwrap();
    let err_x = (y_final[0] - 1.0).abs();
    let err_v = y_final[1].abs();
    let pass = err_x < 1e-3 && err_v < 1e-3;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "verify_periodicity",
        "check",
        "y(2pi) ≈ [1, 0]",
        &format!("err_x={err_x:.2e}, err_v={err_v:.2e}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    assert_bundle_written(scenario_id, &bundle);
    assert!(all_pass, "Scenario 2 failed");
}

/// Scenario 3: Multi-step validation sequence
#[test]
fn e2e_003_multi_step_validation() {
    let scenario_id = "e2e_ivp_003_multi_validation";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: validate_tol
    let t = Instant::now();
    let vtol = validate_tol(
        ToleranceValue::Scalar(1e-4),
        ToleranceValue::Scalar(1e-7),
        2,
        RuntimeMode::Strict,
    )
    .expect("validate_tol");
    steps.push(make_step(
        1,
        "validate_tol",
        "validate",
        "rtol=1e-4, atol=1e-7, n=2",
        &format!("ok, warnings={}", vtol.warnings.len()),
        t.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: validate_first_step
    let t = Instant::now();
    let fs = validate_first_step(0.001, 0.0, 100.0).expect("validate_first_step");
    let pass = (fs - 0.001).abs() < f64::EPSILON;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "validate_first_step",
        "validate",
        "first_step=0.001, t0=0.0, t_bound=100.0",
        &format!("result={fs}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: validate_max_step
    let t = Instant::now();
    let ms = validate_max_step(10.0).expect("validate_max_step");
    let pass = (ms - 10.0).abs() < f64::EPSILON;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "validate_max_step",
        "validate",
        "max_step=10.0",
        &format!("result={ms}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 4: Full solve using validated params
    let t = Instant::now();
    let mut fun = lotka_volterra;
    let result = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 10.0),
            y0: &[1.0, 0.5],
            method: SolverKind::Rk45,
            rtol: 1e-4,
            atol: ToleranceValue::Scalar(1e-7),
            first_step: Some(0.001),
            max_step: 10.0,
            ..Default::default()
        },
    )
    .expect("solve_ivp lotka-volterra");
    let pass = result.success;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "solve_lotka_volterra",
        "solve",
        "Lotka-Volterra, t=[0,10], y0=[1,0.5]",
        &format!(
            "success={}, nfev={}, t_len={}",
            result.success,
            result.nfev,
            result.t.len()
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 5: Check solutions remain positive (biological constraint)
    let t = Instant::now();
    let all_positive = result.y.iter().all(|yi| yi.iter().all(|&v| v > 0.0));
    let pass = all_positive;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "verify_positive",
        "check",
        "all y > 0 (biological constraint)",
        &format!("all_positive={all_positive}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    assert_bundle_written(scenario_id, &bundle);
    assert!(all_pass, "Scenario 3 failed");
}

// ═══════════════════════════════════════════════════════════════════
// ERROR RECOVERY SCENARIOS (4-6)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 4: Invalid tolerance recovery — submit bad tol → catch → correct → retry → succeed
#[test]
fn e2e_004_invalid_tolerance_recovery() {
    let scenario_id = "e2e_ivp_004_invalid_tol_recovery";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: Submit bad atol (wrong vector length)
    let t = Instant::now();
    let bad_result = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Vector(vec![1e-6, 1e-6]), // wrong length for n=3
        3,
        RuntimeMode::Strict,
    );
    let pass = bad_result.is_err();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "submit_bad_atol",
        "validate",
        "atol vec len=2 for n=3",
        &format!("is_err={}", bad_result.is_err()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 2: Catch error and inspect
    let t = Instant::now();
    let err_msg = format!("{:?}", bad_result.unwrap_err());
    let pass = err_msg.contains("WrongShape") || err_msg.contains("AtolWrongShape");
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "catch_error",
        "inspect",
        "error from step 1",
        &format!("err={err_msg}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Correct and retry
    let t = Instant::now();
    let good_result = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Vector(vec![1e-6, 1e-6, 1e-6]),
        3,
        RuntimeMode::Strict,
    );
    let pass = good_result.is_ok();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "correct_and_retry",
        "validate",
        "atol vec len=3 for n=3",
        &format!("is_ok={}", good_result.is_ok()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 4: Submit bad first_step (negative)
    let t = Instant::now();
    let bad_fs = validate_first_step(-0.01, 0.0, 10.0);
    let pass = bad_fs.is_err();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "bad_first_step",
        "validate",
        "first_step=-0.01",
        &format!("is_err={}", bad_fs.is_err()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 5: Correct first_step
    let t = Instant::now();
    let good_fs = validate_first_step(0.01, 0.0, 10.0);
    let pass = good_fs.is_ok();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "correct_first_step",
        "validate",
        "first_step=0.01",
        &format!("is_ok={}", good_fs.is_ok()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    assert_bundle_written(scenario_id, &bundle);
    assert!(all_pass, "Scenario 4 failed");
}

/// Scenario 5: Mode switch — run Strict validation, switch to Hardened, verify behavior changes
#[test]
fn e2e_005_mode_switch() {
    let scenario_id = "e2e_ivp_005_mode_switch";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: Validate in Strict mode
    let t = Instant::now();
    let strict_result = validate_tol(
        ToleranceValue::Scalar(1e-4),
        ToleranceValue::Scalar(1e-8),
        2,
        RuntimeMode::Strict,
    )
    .expect("strict validate");
    steps.push(make_step(
        1,
        "strict_validate",
        "validate",
        "mode=Strict, rtol=1e-4",
        &format!("ok, mode={:?}", strict_result.mode),
        t.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Validate in Hardened mode
    let t = Instant::now();
    let hardened_result = validate_tol(
        ToleranceValue::Scalar(1e-4),
        ToleranceValue::Scalar(1e-8),
        2,
        RuntimeMode::Hardened,
    )
    .expect("hardened validate");
    let pass = matches!(hardened_result.mode, RuntimeMode::Hardened);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "hardened_validate",
        "validate",
        "mode=Hardened, rtol=1e-4",
        &format!("ok, mode={:?}", hardened_result.mode),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Solve in Strict mode
    let t = Instant::now();
    let mut fun = exponential_decay;
    let strict_sol = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 2.0),
            y0: &[1.0],
            method: SolverKind::Rk45,
            mode: RuntimeMode::Strict,
            ..Default::default()
        },
    )
    .expect("strict solve");
    let pass = strict_sol.success;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "strict_solve",
        "solve",
        "mode=Strict, exponential decay",
        &format!("nfev={}", strict_sol.nfev),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 4: Solve in Hardened mode
    let t = Instant::now();
    let mut fun = exponential_decay;
    let hardened_sol = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 2.0),
            y0: &[1.0],
            method: SolverKind::Rk45,
            mode: RuntimeMode::Hardened,
            ..Default::default()
        },
    )
    .expect("hardened solve");
    let pass = hardened_sol.success;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "hardened_solve",
        "solve",
        "mode=Hardened, exponential decay",
        &format!("nfev={}", hardened_sol.nfev),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 5: Verify solutions agree
    let t = Instant::now();
    let y_strict = strict_sol.y.last().unwrap()[0];
    let y_hardened = hardened_sol.y.last().unwrap()[0];
    let diff = (y_strict - y_hardened).abs();
    let pass = diff < 1e-6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "compare_modes",
        "check",
        "strict vs hardened final y",
        &format!("diff={diff:.2e}, strict={y_strict:.8e}, hardened={y_hardened:.8e}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    assert_bundle_written(scenario_id, &bundle);
    assert!(all_pass, "Scenario 5 failed");
}

/// Scenario 6: Boundary tolerance — validate with MIN_RTOL exactly
#[test]
fn e2e_006_boundary_tolerance() {
    let scenario_id = "e2e_ivp_006_boundary_tol";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;
    let min_rtol = 100.0 * f64::EPSILON;

    // Step 1: Validate with exactly MIN_RTOL — should not warn
    let t = Instant::now();
    let vtol = validate_tol(
        ToleranceValue::Scalar(min_rtol),
        ToleranceValue::Scalar(1e-15),
        1,
        RuntimeMode::Strict,
    )
    .expect("validate at MIN_RTOL");
    let pass = vtol.warnings.is_empty();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "exact_min_rtol",
        "validate",
        &format!("rtol=MIN_RTOL={min_rtol:.2e}"),
        &format!("warnings={}", vtol.warnings.len()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 2: Below MIN_RTOL — should clamp with warning
    let t = Instant::now();
    let too_small = min_rtol / 10.0;
    let vtol = validate_tol(
        ToleranceValue::Scalar(too_small),
        ToleranceValue::Scalar(1e-15),
        1,
        RuntimeMode::Strict,
    )
    .expect("validate below MIN_RTOL");
    let pass = vtol.warnings.len() == 1; // Should have RtolClamped warning
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "below_min_rtol",
        "validate",
        &format!("rtol={too_small:.2e} (below MIN_RTOL)"),
        &format!("warnings={}", vtol.warnings.len()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Zero atol should fail
    let t = Instant::now();
    let zero_atol = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Scalar(-1.0), // negative
        1,
        RuntimeMode::Strict,
    );
    let pass = zero_atol.is_err();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "negative_atol",
        "validate",
        "atol=-1.0",
        &format!("is_err={}", zero_atol.is_err()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 4: Solve with tight tolerances
    let t = Instant::now();
    let mut fun = exponential_decay;
    let result = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &[1.0],
            method: SolverKind::Rk45,
            rtol: min_rtol,
            atol: ToleranceValue::Scalar(1e-15),
            ..Default::default()
        },
    )
    .expect("solve at tight tol");
    let y_final = result.y.last().unwrap()[0];
    let expected = (-1.0_f64).exp();
    let err = (y_final - expected).abs();
    let pass = result.success && err < 1e-10;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "solve_tight_tol",
        "solve",
        &format!("rtol=MIN_RTOL={min_rtol:.2e}"),
        &format!("err={err:.2e}, nfev={}", result.nfev),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    assert_bundle_written(scenario_id, &bundle);
    assert!(all_pass, "Scenario 6 failed");
}

// ═══════════════════════════════════════════════════════════════════
// ADVERSARIAL / EDGE SCENARIOS (7-8)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 7: Rapid sequential validations — 1000 validate_tol calls, verify no state leakage
#[test]
fn e2e_007_rapid_sequential_validations() {
    let scenario_id = "e2e_ivp_007_rapid_sequential";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: 1000 sequential validate_tol calls
    let t = Instant::now();
    let mut pass_count = 0;
    let mut fail_count = 0;
    for i in 0..1000 {
        let rtol = 1e-3 + (i as f64) * 1e-6;
        let atol = 1e-6 + (i as f64) * 1e-9;
        let n = (i % 10) + 1;
        match validate_tol(
            ToleranceValue::Scalar(rtol),
            ToleranceValue::Scalar(atol),
            n,
            RuntimeMode::Strict,
        ) {
            Ok(_) => pass_count += 1,
            Err(_) => fail_count += 1,
        }
    }
    let pass = pass_count == 1000 && fail_count == 0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "rapid_validate",
        "batch_validate",
        "1000 sequential validate_tol calls",
        &format!("pass={pass_count}, fail={fail_count}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 2: 1000 sequential validate_first_step calls
    let t = Instant::now();
    pass_count = 0;
    for i in 0..1000 {
        let first_step = 0.001 + (i as f64) * 0.0001;
        if validate_first_step(first_step, 0.0, 100.0).is_ok() {
            pass_count += 1;
        }
    }
    let pass = pass_count == 1000;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "rapid_first_step",
        "batch_validate",
        "1000 sequential validate_first_step calls",
        &format!("pass={pass_count}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Verify no state leakage — results should be deterministic
    let t = Instant::now();
    let v1 = validate_tol(
        ToleranceValue::Scalar(1e-5),
        ToleranceValue::Scalar(1e-8),
        4,
        RuntimeMode::Strict,
    )
    .expect("v1");
    let v2 = validate_tol(
        ToleranceValue::Scalar(1e-5),
        ToleranceValue::Scalar(1e-8),
        4,
        RuntimeMode::Strict,
    )
    .expect("v2");
    let pass = v1.warnings.len() == v2.warnings.len();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "determinism_check",
        "validate",
        "same inputs twice",
        &format!("warnings_match={pass}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    assert_bundle_written(scenario_id, &bundle);
    assert!(all_pass, "Scenario 7 failed");
}

/// Scenario 8: Large system dimension — n=1000, verify no performance degradation
#[test]
fn e2e_008_large_system() {
    let scenario_id = "e2e_ivp_008_large_system";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;
    let n = 1000;

    // Step 1: Validate tolerances with large vector
    let t = Instant::now();
    let atol_vec: Vec<f64> = vec![1e-6; n];
    let _vtol = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Vector(atol_vec),
        n,
        RuntimeMode::Strict,
    )
    .expect("validate large system");
    let dur = t.elapsed();
    let pass = dur.as_millis() < 100;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "validate_large_tol",
        "validate",
        &format!("n={n}, vector atol"),
        &format!("ok, duration_ms={}", dur.as_millis()),
        dur.as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 2: Solve large linear system dy/dt = -alpha * y
    let t = Instant::now();
    let y0: Vec<f64> = (0..n).map(|i| 1.0 + 0.001 * i as f64).collect();
    let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { y.iter().map(|&yi| -0.1 * yi).collect() };
    let result = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &y0,
            method: SolverKind::Rk23, // RK23 for speed on large system
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            ..Default::default()
        },
    )
    .expect("solve large system");
    let dur = t.elapsed();
    let pass = result.success && dur.as_secs() < 30;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "solve_large_system",
        "solve",
        &format!("n={n}, dy/dt=-0.1*y, t=[0,1]"),
        &format!(
            "success={}, nfev={}, duration_ms={}",
            result.success,
            result.nfev,
            dur.as_millis()
        ),
        dur.as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Verify solution accuracy for large system
    let t = Instant::now();
    let y_final = result.y.last().unwrap();
    let max_err: f64 = y_final
        .iter()
        .enumerate()
        .map(|(i, &yi)| {
            let expected = (1.0 + 0.001 * i as f64) * (-0.1_f64).exp();
            (yi - expected).abs()
        })
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    let pass = max_err < 1e-4;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "verify_accuracy",
        "check",
        &format!("n={n} components"),
        &format!("max_err={max_err:.2e}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    assert_bundle_written(scenario_id, &bundle);
    assert!(all_pass, "Scenario 8 failed");
}

/// Scenario 9: RK23 vs RK45 method comparison — same problem, verify consistency
#[test]
fn e2e_009_rk23_vs_rk45() {
    let scenario_id = "e2e_ivp_009_method_comparison";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: Solve with RK45
    let t = Instant::now();
    let mut fun = exponential_decay;
    let rk45 = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 5.0),
            y0: &[1.0],
            method: SolverKind::Rk45,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-9),
            ..Default::default()
        },
    )
    .expect("rk45 solve");
    let pass = rk45.success;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "solve_rk45",
        "solve",
        "RK45, exponential decay",
        &format!("nfev={}, t_len={}", rk45.nfev, rk45.t.len()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 2: Solve with RK23
    let t = Instant::now();
    let mut fun = exponential_decay;
    let rk23 = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 5.0),
            y0: &[1.0],
            method: SolverKind::Rk23,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-9),
            ..Default::default()
        },
    )
    .expect("rk23 solve");
    let pass = rk23.success;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "solve_rk23",
        "solve",
        "RK23, exponential decay",
        &format!("nfev={}, t_len={}", rk23.nfev, rk23.t.len()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Compare final solutions
    let t = Instant::now();
    let y45 = rk45.y.last().unwrap()[0];
    let y23 = rk23.y.last().unwrap()[0];
    let expected = (-5.0_f64).exp();
    let err45 = (y45 - expected).abs();
    let err23 = (y23 - expected).abs();
    let pass = err45 < 1e-6 && err23 < 1e-6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "compare_solutions",
        "check",
        "RK45 vs RK23 at t=5",
        &format!(
            "err45={err45:.2e}, err23={err23:.2e}, diff={:.2e}",
            (y45 - y23).abs()
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 4: RK45 should use fewer evaluations (higher order)
    let t = Instant::now();
    let pass = rk45.nfev < rk23.nfev;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "efficiency_check",
        "check",
        "RK45 fewer evals than RK23",
        &format!("rk45_nfev={}, rk23_nfev={}", rk45.nfev, rk23.nfev),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    assert_bundle_written(scenario_id, &bundle);
    assert!(all_pass, "Scenario 9 failed");
}

/// Scenario 10: First step exceeds interval — verify error handling
#[test]
fn e2e_010_first_step_exceeds_interval() {
    let scenario_id = "e2e_ivp_010_first_step_bounds";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Step 1: first_step > |t_bound - t0| should error
    let t = Instant::now();
    let result = validate_first_step(20.0, 0.0, 10.0);
    let pass = result.is_err();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "first_step_too_large",
        "validate",
        "first_step=20.0, interval=[0,10]",
        &format!("is_err={}", result.is_err()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 2: first_step exactly at boundary should succeed
    let t = Instant::now();
    let result = validate_first_step(10.0, 0.0, 10.0);
    let pass = result.is_ok();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "first_step_exact_boundary",
        "validate",
        "first_step=10.0, interval=[0,10]",
        &format!("is_ok={}", result.is_ok()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: max_step = 0 should error
    let t = Instant::now();
    let result = validate_max_step(0.0);
    let pass = result.is_err();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "max_step_zero",
        "validate",
        "max_step=0.0",
        &format!("is_err={}", result.is_err()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 4: Negative max_step should error
    let t = Instant::now();
    let result = validate_max_step(-1.0);
    let pass = result.is_err();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "max_step_negative",
        "validate",
        "max_step=-1.0",
        &format!("is_err={}", result.is_err()),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    assert_bundle_written(scenario_id, &bundle);
    assert!(all_pass, "Scenario 10 failed");
}

/// Scenario 11: BDF Newton corrector on a stiff decay problem
#[test]
fn e2e_011_bdf_newton_stiff_decay() {
    let scenario_id = "e2e_ivp_011_bdf_newton";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let mut fun = stiff_decay;
    let result = solve_ivp(
        &mut fun,
        &SolveIvpOptions {
            t_span: (0.0, 0.01),
            y0: &[1.0],
            method: SolverKind::Bdf,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            first_step: Some(1e-6),
            max_step: 1e-3,
            ..Default::default()
        },
    )
    .expect("BDF stiff solve");
    let pass = result.success;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "solve_bdf_stiff_decay",
        "solve",
        "BDF, dy/dt=-1000y, t=[0,0.01], y0=1",
        &format!(
            "success={}, nfev={}, njev={}, nlu={}, steps={}",
            result.success,
            result.nfev,
            result.njev,
            result.nlu,
            result.t.len()
        ),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let t = Instant::now();
    let final_y = result.y.last().unwrap()[0];
    let expected = (-10.0_f64).exp();
    let abs_err = (final_y - expected).abs();
    let pass = abs_err < 5e-3;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "verify_stiff_solution",
        "check",
        &format!("final_y={final_y:.8e}"),
        &format!("expected={expected:.8e}, abs_err={abs_err:.2e}"),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let t = Instant::now();
    let pass = result.njev > 0 && result.nlu > 0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "verify_newton_diagnostics",
        "check",
        "BDF should evaluate Jacobians and factorize Newton systems",
        &format!("njev={}, nlu={}", result.njev, result.nlu),
        t.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass {
                "pass".into()
            } else {
                "fail".into()
            },
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    assert_bundle_written(scenario_id, &bundle);
    assert!(all_pass, "Scenario 11 failed");
}
