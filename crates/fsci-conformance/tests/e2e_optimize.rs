#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-003 (optimize/root workflows).
//!
//! Implements 8 scenarios per bd-3jh.14.7 acceptance criteria:
//! - Happy path minimize
//! - Multi-algorithm comparison
//! - Root-finding pipeline
//! - Convergence verification chain
//! - Strict vs hardened mode switch behavior
//! - Adversarial objective behavior
//! - Callback workflow
//! - Large-dimension stress

use fsci_opt::types::OptimizeTraceEntry;
use fsci_opt::{
    ConvergenceStatus, MinimizeOptions, OptimizeMethod, RootMethod, RootOptions,
    get_optimize_traces, minimize, root_scalar,
};
use fsci_runtime::RuntimeMode;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    convergence_trace: Vec<ConvergenceTracePoint>,
    environment: EnvironmentInfo,
    overall: OverallResult,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicStep {
    scenario_id: String,
    step_name: String,
    timestamp_ms: u128,
    duration_ms: u128,
    outcome: String,
    algorithm: String,
    nfev: usize,
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct ConvergenceTracePoint {
    iter: usize,
    f_val: Option<f64>,
    grad_norm: Option<f64>,
    step_size: Option<f64>,
    algorithm: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OverallResult {
    status: String,
    total_duration_ms: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

#[derive(Debug, Clone)]
struct StepOutcome {
    detail: String,
    nfev: usize,
    trace: Vec<ConvergenceTracePoint>,
}

impl StepOutcome {
    fn new(detail: impl Into<String>, nfev: usize) -> Self {
        Self {
            detail: detail.into(),
            nfev,
            trace: Vec::new(),
        }
    }

    fn with_trace(
        detail: impl Into<String>,
        nfev: usize,
        trace: Vec<ConvergenceTracePoint>,
    ) -> Self {
        Self {
            detail: detail.into(),
            nfev,
            trace,
        }
    }
}

struct ScenarioRunner {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    convergence_trace: Vec<ConvergenceTracePoint>,
    passed: bool,
    error_chain: Option<String>,
    start: Instant,
}

impl ScenarioRunner {
    fn new(scenario_id: &str) -> Self {
        Self {
            scenario_id: scenario_id.to_owned(),
            steps: Vec::new(),
            convergence_trace: Vec::new(),
            passed: true,
            error_chain: None,
            start: Instant::now(),
        }
    }

    fn record_step(
        &mut self,
        step_name: &str,
        algorithm: &str,
        f: impl FnOnce() -> Result<StepOutcome, String>,
    ) -> bool {
        let started = Instant::now();
        let timestamp_ms = now_unix_ms();
        let result = f();
        let duration_ms = started.elapsed().as_millis();
        let (outcome, nfev, detail, trace) = match result {
            Ok(outcome) => (
                "pass".to_owned(),
                outcome.nfev,
                outcome.detail,
                outcome.trace,
            ),
            Err(error) => {
                self.passed = false;
                let hinted = format!(
                    "{error}; hint: replay with `{}` and inspect this step's inputs/outputs",
                    replay_cmd(&self.scenario_id)
                );
                if self.error_chain.is_none() {
                    self.error_chain = Some(hinted.clone());
                }
                ("fail".to_owned(), 0, hinted, Vec::new())
            }
        };

        self.convergence_trace.extend(trace);
        self.steps.push(ForensicStep {
            scenario_id: self.scenario_id.clone(),
            step_name: step_name.to_owned(),
            timestamp_ms,
            duration_ms,
            outcome: outcome.clone(),
            algorithm: algorithm.to_owned(),
            nfev,
            detail,
        });

        outcome == "pass"
    }

    fn finish(self) -> ForensicLogBundle {
        let bundle = ForensicLogBundle {
            scenario_id: self.scenario_id.clone(),
            steps: self.steps,
            convergence_trace: self.convergence_trace,
            environment: environment_info(),
            overall: OverallResult {
                status: if self.passed {
                    "pass".to_owned()
                } else {
                    "fail".to_owned()
                },
                total_duration_ms: self.start.elapsed().as_millis(),
                replay_command: replay_cmd(&self.scenario_id),
                error_chain: self.error_chain,
            },
        };
        write_bundle(&self.scenario_id, &bundle);
        bundle
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/P2C-003/e2e/runs")
}

fn replay_cmd(scenario_id: &str) -> String {
    format!("cargo test -p fsci-conformance --test e2e_optimize -- {scenario_id} --nocapture")
}

fn environment_info() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: String::from(env!("CARGO_PKG_VERSION")),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
    }
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", dir.display()));
    let path = dir.join(format!("{scenario_id}.json"));
    let bytes = serde_json::to_vec_pretty(bundle).expect("serialize forensic bundle");
    fs::write(&path, bytes)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}

fn rosenbrock(x: &[f64]) -> f64 {
    let a = 1.0 - x[0];
    let b = x[1] - x[0] * x[0];
    a * a + 100.0 * b * b
}

fn shifted_quadratic(x: &[f64]) -> f64 {
    let dx0 = x[0] - 2.0;
    let dx1 = x[1] + 3.0;
    dx0 * dx0 + dx1 * dx1
}

fn cusp_abs_objective(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).sum()
}

fn nan_barrier_objective(x: &[f64]) -> f64 {
    if x[0] < 0.0 {
        f64::NAN
    } else {
        (x[0] - 1.0).powi(2) + x[1] * x[1]
    }
}

fn cubic_root_target(x: f64) -> f64 {
    x * x * x - x - 2.0
}

fn capture_minimize_run<F>(
    algorithm: &str,
    fun: F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<(fsci_opt::OptimizeResult, Vec<ConvergenceTracePoint>), String>
where
    F: Fn(&[f64]) -> f64,
{
    let _guard = trace_capture_lock()
        .lock()
        .expect("trace capture lock poisoned");
    let _ = get_optimize_traces();
    let result =
        minimize(fun, x0, options).map_err(|error| format!("minimize call failed: {error}"))?;
    let trace = to_convergence_trace(algorithm, get_optimize_traces());
    Ok((result, trace))
}

fn trace_capture_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn to_convergence_trace(
    algorithm: &str,
    entries: Vec<OptimizeTraceEntry>,
) -> Vec<ConvergenceTracePoint> {
    entries
        .into_iter()
        .filter(|entry| entry.event == "iteration")
        .map(|entry| ConvergenceTracePoint {
            iter: entry.iter_num,
            f_val: entry.f_val,
            grad_norm: entry.grad_norm,
            step_size: entry.step_size,
            algorithm: algorithm.to_owned(),
        })
        .collect()
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn assert_bundle_pass(bundle: &ForensicLogBundle) {
    assert_eq!(
        bundle.overall.status,
        "pass",
        "scenario={} failed, error_chain={:?}, last_step={:?}",
        bundle.scenario_id,
        bundle.overall.error_chain,
        bundle.steps.last()
    );
}

#[test]
fn e2e_p2c003_01_happy_path_minimize_bfgs() {
    let mut runner = ScenarioRunner::new("p2c003_01_happy_path_minimize_bfgs");
    let x0 = vec![-1.2, 1.0];
    let mut trace_cache = Vec::new();

    runner.record_step("run_rosenbrock_bfgs", "bfgs", || {
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-8),
            maxiter: Some(400),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let (result, trace) = capture_minimize_run("bfgs", rosenbrock, &x0, opts)?;
        if !result.success {
            return Err(format!(
                "expected convergence on Rosenbrock, got status={:?} message={}",
                result.status, result.message
            ));
        }
        let distance = l2_norm(&[result.x[0] - 1.0, result.x[1] - 1.0]);
        let final_f = result.fun.unwrap_or(f64::INFINITY);
        if distance > 5.0e-2 || final_f > 1.0e-6 {
            return Err(format!(
                "solution quality too weak: distance={distance:.3e}, final_f={final_f:.3e}"
            ));
        }
        trace_cache = trace.clone();
        Ok(StepOutcome::with_trace(
            format!(
                "converged to x={:?}, f={final_f:.3e}, distance_to_opt={distance:.3e}",
                result.x
            ),
            result.nfev,
            trace,
        ))
    });

    runner.record_step("verify_convergence_trace", "bfgs", || {
        if trace_cache.is_empty() {
            return Err("missing per-iteration convergence trace".to_owned());
        }
        let first = trace_cache
            .first()
            .and_then(|entry| entry.f_val)
            .unwrap_or(f64::INFINITY);
        let last = trace_cache
            .last()
            .and_then(|entry| entry.f_val)
            .unwrap_or(f64::INFINITY);
        if !last.is_finite() || last > first {
            return Err(format!(
                "trace did not improve objective: first_f={first:.3e}, last_f={last:.3e}"
            ));
        }
        Ok(StepOutcome::new(
            format!(
                "trace length={} first_f={first:.3e} last_f={last:.3e}",
                trace_cache.len()
            ),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_02_multi_algorithm_comparison() {
    let mut runner = ScenarioRunner::new("p2c003_02_multi_algorithm_comparison");
    let mut terminal_values = Vec::<(String, f64)>::new();
    let x0 = vec![8.0, -8.0];

    for (method, algorithm) in [
        (OptimizeMethod::Bfgs, "bfgs"),
        (OptimizeMethod::ConjugateGradient, "cg_pr_plus"),
        (OptimizeMethod::Powell, "powell"),
    ] {
        runner.record_step(&format!("run_{algorithm}"), algorithm, || {
            let opts = MinimizeOptions {
                method: Some(method),
                tol: Some(1.0e-8),
                maxiter: Some(300),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            };
            let (result, trace) = capture_minimize_run(algorithm, shifted_quadratic, &x0, opts)?;
            let final_f = result.fun.unwrap_or(f64::INFINITY);
            if !final_f.is_finite() {
                return Err(format!("algorithm={algorithm} produced non-finite objective"));
            }
            if final_f > 1.0e-5 {
                return Err(format!(
                    "algorithm={algorithm} did not reach low objective; final_f={final_f:.3e}, status={:?}",
                    result.status
                ));
            }
            terminal_values.push((algorithm.to_owned(), final_f));
            Ok(StepOutcome::with_trace(
                format!(
                    "algorithm={algorithm} success={} status={:?} final_f={final_f:.3e}",
                    result.success, result.status
                ),
                result.nfev,
                trace,
            ))
        });
    }

    runner.record_step("compare_terminal_objectives", "comparison", || {
        if terminal_values.len() != 3 {
            return Err(format!(
                "expected 3 algorithm outcomes, got {}",
                terminal_values.len()
            ));
        }
        let min_f = terminal_values.iter().map(|(_, value)| *value).fold(
            f64::INFINITY,
            |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            },
        );
        let max_f = terminal_values.iter().map(|(_, value)| *value).fold(
            f64::NEG_INFINITY,
            |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            },
        );
        let spread = max_f - min_f;
        if spread > 1.0e-4 {
            return Err(format!(
                "algorithm spread too wide: {spread:.3e} ({terminal_values:?})"
            ));
        }
        Ok(StepOutcome::new(
            format!("terminal objective spread={spread:.3e}"),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_03_root_finding_pipeline() {
    let mut runner = ScenarioRunner::new("p2c003_03_root_finding_pipeline");
    let expected_root = 1.521_379_706_804_567_6;
    let mut brentq_root = 0.0;
    let mut bisect_root = 0.0;

    runner.record_step("run_brentq", "brentq", || {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = root_scalar(cubic_root_target, Some((1.0, 2.0)), None, None, options)
            .map_err(|error| format!("brentq failed: {error}"))?;
        if !result.converged {
            return Err(format!("brentq failed to converge: {}", result.message));
        }
        let residual = cubic_root_target(result.root).abs();
        if residual > 1.0e-10 {
            return Err(format!("brentq residual too large: {residual:.3e}"));
        }
        brentq_root = result.root;
        Ok(StepOutcome::with_trace(
            format!(
                "root={:.12} iterations={} calls={} residual={residual:.3e}",
                result.root, result.iterations, result.function_calls
            ),
            result.function_calls,
            vec![ConvergenceTracePoint {
                iter: result.iterations,
                f_val: Some(residual),
                grad_norm: None,
                step_size: None,
                algorithm: "brentq".to_owned(),
            }],
        ))
    });

    runner.record_step("run_bisect", "bisect", || {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = root_scalar(cubic_root_target, Some((1.0, 2.0)), None, None, options)
            .map_err(|error| format!("bisect failed: {error}"))?;
        if !result.converged {
            return Err(format!("bisect failed to converge: {}", result.message));
        }
        bisect_root = result.root;
        let residual = cubic_root_target(result.root).abs();
        Ok(StepOutcome::with_trace(
            format!(
                "root={:.12} iterations={} calls={} residual={residual:.3e}",
                result.root, result.iterations, result.function_calls
            ),
            result.function_calls,
            vec![ConvergenceTracePoint {
                iter: result.iterations,
                f_val: Some(residual),
                grad_norm: None,
                step_size: None,
                algorithm: "bisect".to_owned(),
            }],
        ))
    });

    runner.record_step("verify_root_agreement", "root_pipeline", || {
        let brentq_err = (brentq_root - expected_root).abs();
        let bisect_err = (bisect_root - expected_root).abs();
        let cross_diff = (brentq_root - bisect_root).abs();
        if brentq_err > 1.0e-9 || bisect_err > 1.0e-8 || cross_diff > 1.0e-8 {
            return Err(format!(
                "root mismatch: brentq_err={brentq_err:.3e}, bisect_err={bisect_err:.3e}, cross_diff={cross_diff:.3e}"
            ));
        }
        Ok(StepOutcome::new(
            format!(
                "root agreement verified (brentq={brentq_root:.12}, bisect={bisect_root:.12})"
            ),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_04_convergence_verification_chain() {
    let mut runner = ScenarioRunner::new("p2c003_04_convergence_verification_chain");
    let x0 = vec![4.5, -4.5];
    let mut trace_cache = Vec::new();

    runner.record_step("run_cg_with_trace_capture", "cg_pr_plus", || {
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-7),
            maxiter: Some(300),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let (result, trace) = capture_minimize_run("cg_pr_plus", shifted_quadratic, &x0, opts)?;
        let final_f = result.fun.unwrap_or(f64::INFINITY);
        if final_f > 1.0e-5 {
            return Err(format!(
                "cg_pr_plus did not converge enough: final_f={final_f:.3e}, status={:?}",
                result.status
            ));
        }
        trace_cache = trace.clone();
        Ok(StepOutcome::with_trace(
            format!(
                "captured trace points={}, final_f={final_f:.3e}",
                trace_cache.len()
            ),
            result.nfev,
            trace,
        ))
    });

    runner.record_step("verify_trace_semantics", "cg_pr_plus", || {
        if trace_cache.is_empty() {
            return Err("expected at least one convergence trace point".to_owned());
        }
        if trace_cache.iter().any(|entry| entry.iter == 0) {
            return Err("trace contains zero iteration index".to_owned());
        }
        if trace_cache
            .iter()
            .any(|entry| entry.grad_norm.is_some_and(|value| !value.is_finite()))
        {
            return Err("trace contains non-finite grad_norm values".to_owned());
        }
        let first_f = trace_cache
            .first()
            .and_then(|entry| entry.f_val)
            .unwrap_or(f64::INFINITY);
        let last_f = trace_cache
            .last()
            .and_then(|entry| entry.f_val)
            .unwrap_or(f64::INFINITY);
        if last_f > first_f {
            return Err(format!(
                "trace regression detected: first_f={first_f:.3e}, last_f={last_f:.3e}"
            ));
        }
        Ok(StepOutcome::new(
            format!(
                "trace semantics valid: points={}, first_f={first_f:.3e}, last_f={last_f:.3e}",
                trace_cache.len()
            ),
            0,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_05_mode_switch_non_finite_objective() {
    let mut runner = ScenarioRunner::new("p2c003_05_mode_switch_non_finite_objective");
    let x0 = vec![-0.5, 1.0];

    runner.record_step("strict_mode_behavior", "bfgs-strict", || {
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let (result, trace) =
            capture_minimize_run("bfgs-strict", nan_barrier_objective, &x0, opts)?;
        if result.success {
            return Err("strict mode unexpectedly reported success on NaN objective".to_owned());
        }
        if !result.message.contains("non-finite") {
            return Err(format!(
                "strict mode message missing non-finite context: {}",
                result.message
            ));
        }
        Ok(StepOutcome::with_trace(
            format!(
                "strict mode produced expected failure: status={:?}",
                result.status
            ),
            result.nfev,
            trace,
        ))
    });

    runner.record_step("hardened_mode_behavior", "bfgs-hardened", || {
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            mode: RuntimeMode::Hardened,
            ..MinimizeOptions::default()
        };
        let (result, trace) =
            capture_minimize_run("bfgs-hardened", nan_barrier_objective, &x0, opts)?;
        if result.success {
            return Err("hardened mode unexpectedly reported success on NaN objective".to_owned());
        }
        if !result
            .message
            .contains("hardened mode rejects non-finite objective values")
        {
            return Err(format!(
                "hardened mode did not report expected rejection message: {}",
                result.message
            ));
        }
        Ok(StepOutcome::with_trace(
            format!(
                "hardened mode rejected non-finite objective as expected: status={:?}",
                result.status
            ),
            result.nfev,
            trace,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_06_adversarial_nondifferentiable_objective() {
    let mut runner = ScenarioRunner::new("p2c003_06_adversarial_nondifferentiable_objective");
    let x0 = vec![3.0, -4.0];

    runner.record_step("powell_on_abs_objective", "powell", || {
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-6),
            maxiter: Some(250),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let (result, trace) = capture_minimize_run("powell", cusp_abs_objective, &x0, opts)?;
        let final_f = result.fun.unwrap_or(f64::INFINITY);
        if !final_f.is_finite() || final_f > 1.0 {
            return Err(format!(
                "adversarial objective did not reach acceptable value: final_f={final_f:.3e}, status={:?}, message={}",
                result.status, result.message
            ));
        }
        Ok(StepOutcome::with_trace(
            format!(
                "graceful handling confirmed: success={} status={:?} final_f={final_f:.3e}",
                result.success, result.status
            ),
            result.nfev,
            trace,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

static CALLBACK_INVOCATIONS: AtomicUsize = AtomicUsize::new(0);

fn stop_immediately(_: &[f64]) -> bool {
    CALLBACK_INVOCATIONS.fetch_add(1, Ordering::SeqCst);
    false
}

#[test]
fn e2e_p2c003_07_callback_workflow() {
    let mut runner = ScenarioRunner::new("p2c003_07_callback_workflow");
    let x0 = vec![2.5, -2.5];

    runner.record_step("run_with_callback_stop", "bfgs-callback", || {
        CALLBACK_INVOCATIONS.store(0, Ordering::SeqCst);
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            callback: Some(stop_immediately),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let (result, trace) = capture_minimize_run("bfgs-callback", shifted_quadratic, &x0, opts)?;
        if result.status != ConvergenceStatus::CallbackStop {
            return Err(format!(
                "expected callback stop status, got {:?} ({})",
                result.status, result.message
            ));
        }
        let invocations = CALLBACK_INVOCATIONS.load(Ordering::SeqCst);
        if invocations < 1 {
            return Err(format!(
                "callback should have been invoked at least once, got {invocations}"
            ));
        }
        Ok(StepOutcome::with_trace(
            format!(
                "callback stop confirmed with invocations={invocations}, nfev={}",
                result.nfev
            ),
            result.nfev,
            trace,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}

#[test]
fn e2e_p2c003_08_large_dimension_stress() {
    let mut runner = ScenarioRunner::new("p2c003_08_large_dimension_stress");
    let n = 64usize;
    let shift: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
    let x0 = vec![5.0; n];

    runner.record_step("cg_large_dimension", "cg_pr_plus", || {
        let start = Instant::now();
        let opts = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-7),
            maxiter: Some(500),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let objective = |x: &[f64]| {
            x.iter()
                .zip(shift.iter())
                .map(|(value, target)| {
                    let delta = value - target;
                    delta * delta
                })
                .sum::<f64>()
        };
        let (result, trace) = capture_minimize_run("cg_pr_plus", objective, &x0, opts)?;
        let elapsed_ms = start.elapsed().as_millis();
        let final_f = result.fun.unwrap_or(f64::INFINITY);
        if elapsed_ms > 15_000 {
            return Err(format!(
                "large-dimension stress exceeded time budget: {elapsed_ms}ms"
            ));
        }
        if final_f > 1.0e-4 {
            return Err(format!(
                "large-dimension objective too high: final_f={final_f:.3e}, status={:?}",
                result.status
            ));
        }
        Ok(StepOutcome::with_trace(
            format!("completed n={n} in {elapsed_ms}ms with final_f={final_f:.3e}"),
            result.nfev,
            trace,
        ))
    });

    let bundle = runner.finish();
    assert_bundle_pass(&bundle);
}
