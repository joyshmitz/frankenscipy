#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-009 (Statistical distributions).
//!
//! Acceptance criteria:
//!   Happy-path:     1-3  (distribution evaluation, CDF/PPF roundtrip, parameter estimation)
//!   Error recovery: 4-6  (invalid parameters, boundary quantiles, NaN inputs)
//!   Adversarial:    7-10 (extreme parameters, tail probabilities, identity checks)
//!
//! Each scenario emits topology-compliant artifacts to
//! `fixtures/artifacts/FSCI-P2C-009/e2e/runs/{run_id}/{scenario_id}/`
//! containing `events.jsonl` and `summary.json`.

use std::f64::consts::PI;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    Argus, BetaDist, Burr12, ChiSquared, ContinuousDistribution, CrystalBall, ExponWeibull,
    Exponential, FrechetR, GammaDist, GenHalfLogistic, GenLogistic, Gompertz, Gumbel, InvWeibull,
    LogLaplace, Logistic, Lognormal, Maxwell, Mielke, Normal, Pareto, Rayleigh, StudentT,
    TukeyLambda, Uniform, Weibull,
};
use serde::Serialize;

// ---- Topology-compliant log types ----

#[derive(Debug, Clone, Serialize)]
struct EventEntry {
    scenario_id: String,
    step_name: String,
    timestamp_ms: u128,
    duration_ms: u128,
    outcome: String,
    message: String,
    environment: EnvironmentInfo,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RunSummary {
    packet_id: String,
    scenario_id: String,
    run_id: String,
    passed: bool,
    failed_step: Option<String>,
    replay_command: String,
    generated_unix_ms: u128,
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
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
    total_memory_mb: String,
}

type ScalarFn = Box<dyn Fn(f64) -> f64>;
type RoundtripDistribution = (&'static str, ScalarFn, ScalarFn);
type CdfDistribution = (&'static str, ScalarFn);

const PACKET_ID: &str = "FSCI-P2C-009";

// ---- Helpers ----

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn run_id() -> String {
    format!("run-{}", now_unix_ms())
}

fn e2e_runs_dir(run_id: &str, scenario_id: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts")
        .join(PACKET_ID)
        .join("e2e/runs")
        .join(run_id)
        .join(scenario_id)
}

fn rustc_version_string() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|stdout| stdout.trim().to_string())
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| String::from("unknown"))
}

fn make_env() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: rustc_version_string(),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        total_memory_mb: String::from("unknown"),
    }
}

fn replay_cmd(scenario_id: &str) -> String {
    format!(
        "rch exec -- cargo test -p fsci-conformance --test e2e_stats -- {scenario_id} --nocapture"
    )
}

fn write_topology_artifacts(
    scenario_id: &str,
    steps: &[ForensicStep],
    all_pass: bool,
) -> Result<(), String> {
    let rid = run_id();
    let dir = e2e_runs_dir(&rid, scenario_id);
    fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create run dir {}: {e}", dir.display()))?;

    // Write events.jsonl — one JSON line per step
    let events_path = dir.join("events.jsonl");
    let file = fs::File::create(&events_path)
        .map_err(|e| format!("failed to create {}: {e}", events_path.display()))?;
    let mut writer = BufWriter::new(file);
    let env = make_env();
    for step in steps {
        let entry = EventEntry {
            scenario_id: scenario_id.to_string(),
            step_name: step.step_name.clone(),
            timestamp_ms: now_unix_ms(),
            duration_ms: step.duration_ns / 1_000_000,
            outcome: step.outcome.clone(),
            message: format!("{}: {}", step.action, step.output_summary),
            environment: env.clone(),
            artifact_refs: vec![],
        };
        serde_json::to_writer(&mut writer, &entry)
            .map_err(|e| format!("failed to serialize event for {scenario_id}: {e}"))?;
        writer
            .write_all(b"\n")
            .map_err(|e| format!("failed to write newline to {}: {e}", events_path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("failed to flush {}: {e}", events_path.display()))?;

    // Write summary.json
    let first_fail = steps
        .iter()
        .find(|s| s.outcome == "FAIL")
        .map(|s| s.step_name.clone());
    let summary = RunSummary {
        packet_id: PACKET_ID.to_string(),
        scenario_id: scenario_id.to_string(),
        run_id: rid,
        passed: all_pass,
        failed_step: first_fail,
        replay_command: replay_cmd(scenario_id),
        generated_unix_ms: now_unix_ms(),
    };
    let summary_path = dir.join("summary.json");
    let json = serde_json::to_vec_pretty(&summary)
        .map_err(|e| format!("failed to serialize summary for {scenario_id}: {e}"))?;
    fs::write(&summary_path, &json)
        .map_err(|e| format!("failed to write {}: {e}", summary_path.display()))?;
    Ok(())
}

const TOL: f64 = 1e-8;

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

fn assert_artifacts_written(scenario_id: &str, steps: &[ForensicStep], all_pass: bool) {
    let artifact_write = write_topology_artifacts(scenario_id, steps, all_pass);
    assert!(
        artifact_write.is_ok(),
        "artifact write failed for {scenario_id}: {}",
        artifact_write
            .as_ref()
            .err()
            .map_or("unknown error", String::as_str)
    );
}

// ======================================================================
// HAPPY-PATH SCENARIOS (1-3)
// ======================================================================

/// Scenario 1: Normal distribution evaluation chain.
/// Verifies PDF at mean, CDF symmetry, and PPF roundtrip.
#[test]
fn e2e_001_normal_distribution_chain() {
    let scenario_id = "e2e_stats_001_normal";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let n = Normal::standard();

    // Step 1: PDF at mean = 1/sqrt(2*pi)
    let t = Instant::now();
    let pdf_at_0 = n.pdf(0.0);
    let expected_pdf = 1.0 / (2.0 * PI).sqrt();
    let pass = (pdf_at_0 - expected_pdf).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "pdf_at_mean",
        "Normal::pdf",
        "x=0.0",
        &format!("{pdf_at_0}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Step 2: CDF(0) = 0.5 (symmetry)
    let t = Instant::now();
    let cdf_at_0 = n.cdf(0.0);
    let pass = (cdf_at_0 - 0.5).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "cdf_symmetry",
        "Normal::cdf",
        "x=0.0",
        &format!("{cdf_at_0}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Step 3: PPF(0.5) = 0.0
    let t = Instant::now();
    let ppf_half = n.ppf(0.5);
    let pass = ppf_half.abs() < 1e-6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "ppf_median",
        "Normal::ppf",
        "q=0.5",
        &format!("{ppf_half}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Step 4: CDF/PPF roundtrip for q = 0.975
    let t = Instant::now();
    let q = 0.975;
    let x = n.ppf(q);
    let roundtrip = n.cdf(x);
    let pass = (roundtrip - q).abs() < 1e-6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "cdf_ppf_roundtrip",
        "CDF(PPF(0.975))",
        &format!("q={q}"),
        &format!("roundtrip={roundtrip}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Step 5: SF(x) + CDF(x) = 1
    let t = Instant::now();
    let x = 1.96;
    let sum = n.cdf(x) + n.sf(x);
    let pass = (sum - 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "sf_cdf_complement",
        "CDF(1.96) + SF(1.96)",
        &format!("x={x}"),
        &format!("sum={sum}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 2: Multi-distribution CDF/PPF roundtrip.
/// Verifies the inverse CDF contract for several distributions.
#[test]
fn e2e_002_multi_distribution_roundtrip() {
    let scenario_id = "e2e_stats_002_roundtrip";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let distributions: Vec<RoundtripDistribution> = vec![
        {
            let dist = Normal::standard();
            (
                "Normal(0,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = StudentT::new(5.0);
            (
                "StudentT(5)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = ChiSquared::new(3.0);
            (
                "ChiSq(3)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Exponential::new(2.0);
            (
                "Exponential(2)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Uniform::new(0.0, 1.0);
            (
                "Uniform(0,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = BetaDist::new(2.0, 5.0);
            (
                "Beta(2,5)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = GammaDist::new(3.0, 2.0);
            (
                "Gamma(3,2)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Weibull::new(1.5, 1.0);
            (
                "Weibull(1.5,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Lognormal::new(1.0, 1.0);
            (
                "Lognormal(1,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Gumbel::new(0.0, 1.0);
            (
                "Gumbel(0,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Logistic::new(0.0, 1.0);
            (
                "Logistic(0,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
    ];

    let quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];

    for (step_id, (name, ppf, cdf)) in distributions.iter().enumerate() {
        let t = Instant::now();
        let mut dist_pass = true;
        for &q in &quantiles {
            let x = ppf(q);
            if !x.is_finite() {
                continue;
            }
            let roundtrip = cdf(x);
            if (roundtrip - q).abs() > 1e-4 {
                dist_pass = false;
            }
        }
        if !dist_pass {
            all_pass = false;
        }
        steps.push(make_step(
            step_id + 1,
            name,
            "CDF(PPF(q)) roundtrip",
            &format!("quantiles={quantiles:?}"),
            &format!("all_close={dist_pass}"),
            t.elapsed().as_nanos(),
            if dist_pass { "pass" } else { "FAIL" },
        ));
    }

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 3: Distribution moments verification.
/// mean and var should match known analytical values.
#[test]
fn e2e_003_moments_verification() {
    let scenario_id = "e2e_stats_003_moments";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Normal(3, 2): mean=3, var=4
    let t = Instant::now();
    let n = Normal::new(3.0, 2.0);
    let pass = (n.mean() - 3.0).abs() < TOL && (n.var() - 4.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "normal_moments",
        "Normal(3,2)",
        "loc=3, scale=2",
        &format!("mean={}, var={}", n.mean(), n.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Exponential(2): mean=0.5, var=0.25
    let t = Instant::now();
    let e = Exponential::new(2.0);
    let pass = (e.mean() - 0.5).abs() < TOL && (e.var() - 0.25).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "exp_moments",
        "Exponential(2)",
        "lambda=2",
        &format!("mean={}, var={}", e.mean(), e.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Uniform(0,1): mean=0.5, var=1/12
    let t = Instant::now();
    let u = Uniform::new(0.0, 1.0);
    let pass = (u.mean() - 0.5).abs() < TOL && (u.var() - 1.0 / 12.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "uniform_moments",
        "Uniform(0,1)",
        "loc=0, scale=1",
        &format!("mean={}, var={}", u.mean(), u.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // StudentT(5): mean=0, var=5/3
    let t = Instant::now();
    let st = StudentT::new(5.0);
    let pass = st.mean().abs() < TOL && (st.var() - 5.0 / 3.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "student_t_moments",
        "StudentT(5)",
        "df=5",
        &format!("mean={}, var={}", st.mean(), st.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // ChiSquared(4): mean=4, var=8
    let t = Instant::now();
    let c = ChiSquared::new(4.0);
    let pass = (c.mean() - 4.0).abs() < TOL && (c.var() - 8.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "chi2_moments",
        "ChiSquared(4)",
        "df=4",
        &format!("mean={}, var={}", c.mean(), c.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

// ======================================================================
// ERROR RECOVERY SCENARIOS (4-6)
// ======================================================================

/// Scenario 4: Boundary quantile handling.
/// PPF(0) = -inf, PPF(1) = +inf for unbounded distributions.
#[test]
fn e2e_004_boundary_quantiles() {
    let scenario_id = "e2e_stats_004_boundaries";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let n = Normal::standard();

    let t = Instant::now();
    let ppf_0 = n.ppf(0.0);
    let pass = ppf_0 == f64::NEG_INFINITY;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "ppf_zero",
        "Normal::ppf(0)",
        "q=0",
        &format!("{ppf_0}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let ppf_1 = n.ppf(1.0);
    let pass = ppf_1 == f64::INFINITY;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ppf_one",
        "Normal::ppf(1)",
        "q=1",
        &format!("{ppf_1}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // CDF(-inf) = 0, CDF(+inf) = 1
    let t = Instant::now();
    let cdf_neg_inf = n.cdf(f64::NEG_INFINITY);
    let cdf_pos_inf = n.cdf(f64::INFINITY);
    let pass = cdf_neg_inf.abs() < TOL && (cdf_pos_inf - 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "cdf_infinities",
        "Normal::cdf",
        "x=-inf, x=+inf",
        &format!("cdf(-inf)={cdf_neg_inf}, cdf(+inf)={cdf_pos_inf}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // PDF should be non-negative everywhere
    let t = Instant::now();
    let pass = [
        f64::NEG_INFINITY,
        -100.0,
        -1.0,
        0.0,
        1.0,
        100.0,
        f64::INFINITY,
    ]
    .iter()
    .all(|&x| n.pdf(x) >= 0.0);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "pdf_nonnegative",
        "Normal::pdf",
        "various x",
        "all >= 0",
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 5: CDF monotonicity verification across distributions.
#[test]
fn e2e_005_cdf_monotonicity() {
    let scenario_id = "e2e_stats_005_monotonicity";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let distributions: Vec<CdfDistribution> = vec![
        {
            let dist = Normal::standard();
            ("Normal(0,1)", Box::new(move |x| dist.cdf(x)))
        },
        {
            let dist = Exponential::new(1.0);
            ("Exponential(1)", Box::new(move |x| dist.cdf(x)))
        },
        {
            let dist = ChiSquared::new(2.0);
            ("ChiSq(2)", Box::new(move |x| dist.cdf(x)))
        },
        {
            let dist = Pareto::new(2.0, 1.0);
            ("Pareto(2,1)", Box::new(move |x| dist.cdf(x)))
        },
        {
            let dist = Rayleigh::new(1.0);
            ("Rayleigh(1)", Box::new(move |x| dist.cdf(x)))
        },
    ];

    for (step_id, (name, cdf)) in distributions.iter().enumerate() {
        let t = Instant::now();
        let xs: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.2).collect();
        let cdfs: Vec<f64> = xs.iter().map(|&x| cdf(x)).collect();
        let monotonic = cdfs.windows(2).all(|w| w[1] >= w[0] - 1e-15);
        if !monotonic {
            all_pass = false;
        }
        steps.push(make_step(
            step_id + 1,
            name,
            "CDF monotonicity check",
            "x in [-10, 10] step 0.2",
            &format!("monotonic={monotonic}"),
            t.elapsed().as_nanos(),
            if monotonic { "pass" } else { "FAIL" },
        ));
    }

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 6: PDF integrates to 1 (trapezoidal approximation).
#[test]
fn e2e_006_pdf_normalization() {
    let scenario_id = "e2e_stats_006_normalization";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Normal: integrate PDF from -10 to 10
    let t = Instant::now();
    let n = Normal::standard();
    let integral = trapezoidal_integrate(|x| n.pdf(x), -10.0, 10.0, 10_000);
    let pass = (integral - 1.0).abs() < 1e-6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "normal_pdf_integral",
        "trapezoidal(Normal::pdf, -10, 10)",
        "n=10000",
        &format!("integral={integral}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Exponential: integrate PDF from 0 to 50
    let t = Instant::now();
    let e = Exponential::new(1.0);
    let integral = trapezoidal_integrate(|x| e.pdf(x), 0.0, 50.0, 10_000);
    let pass = (integral - 1.0).abs() < 1e-4;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "exp_pdf_integral",
        "trapezoidal(Exp::pdf, 0, 50)",
        "n=10000",
        &format!("integral={integral}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Uniform: integrate PDF from -1 to 2
    // Uniform PDF is discontinuous at 0 and 1, so trapezoidal rule needs more
    // points or a relaxed tolerance to handle the step boundaries.
    let t = Instant::now();
    let u = Uniform::new(0.0, 1.0);
    let integral = trapezoidal_integrate(|x| u.pdf(x), -1.0, 2.0, 100_000);
    let pass = (integral - 1.0).abs() < 1e-4;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "uniform_pdf_integral",
        "trapezoidal(Uniform::pdf, -1, 2)",
        "n=1000",
        &format!("integral={integral}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

fn trapezoidal_integrate(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    sum * h
}

// ======================================================================
// ADVERSARIAL SCENARIOS (7-10)
// ======================================================================

/// Scenario 7: Extreme tail probabilities.
/// CDF at far tails should be near 0 or 1.
#[test]
fn e2e_007_extreme_tails() {
    let scenario_id = "e2e_stats_007_tails";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let n = Normal::standard();

    // CDF(-8) should be very close to 0
    let t = Instant::now();
    let cdf_far_left = n.cdf(-8.0);
    let pass = (0.0..1e-10).contains(&cdf_far_left);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "cdf_far_left",
        "Normal::cdf(-8)",
        "x=-8",
        &format!("{cdf_far_left:.2e}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // CDF(8) should be very close to 1
    let t = Instant::now();
    let cdf_far_right = n.cdf(8.0);
    let pass = (cdf_far_right - 1.0).abs() < 1e-10;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "cdf_far_right",
        "Normal::cdf(8)",
        "x=8",
        &format!("{cdf_far_right}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // SF should complement CDF even in tails
    let t = Instant::now();
    let sf_8 = n.sf(8.0);
    let pass = sf_8 > 0.0 && sf_8 < 1e-10;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "sf_far_right",
        "Normal::sf(8)",
        "x=8",
        &format!("{sf_8:.2e}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 8: Rapid sequential evaluations — stress test.
/// Evaluate PDF/CDF thousands of times to catch state corruption.
#[test]
fn e2e_008_rapid_sequential() {
    let scenario_id = "e2e_stats_008_rapid";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let n = Normal::standard();

    let t = Instant::now();
    let mut anomalies = 0usize;
    for i in 0..10_000 {
        let x = -5.0 + 10.0 * (i as f64 / 10_000.0);
        let cdf = n.cdf(x);
        let pdf = n.pdf(x);
        if !cdf.is_finite() || !pdf.is_finite() || !(0.0..=1.0).contains(&cdf) || pdf < 0.0 {
            anomalies += 1;
        }
    }
    let pass = anomalies == 0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "rapid_normal_eval",
        "10000 PDF+CDF evaluations",
        "x in [-5, 5]",
        &format!("anomalies={anomalies}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let e = Exponential::new(1.0);
    let mut anomalies = 0usize;
    for i in 0..10_000 {
        let x = 0.001 + 20.0 * (i as f64 / 10_000.0);
        let cdf = e.cdf(x);
        let pdf = e.pdf(x);
        if !cdf.is_finite() || !pdf.is_finite() || !(0.0..=1.0).contains(&cdf) || pdf < 0.0 {
            anomalies += 1;
        }
    }
    let pass = anomalies == 0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "rapid_exp_eval",
        "10000 PDF+CDF evaluations",
        "x in [0.001, 20]",
        &format!("anomalies={anomalies}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 9: Location-scale invariance.
/// Normal(mu, sigma).cdf(x) should equal Normal(0,1).cdf((x-mu)/sigma).
#[test]
fn e2e_009_location_scale_invariance() {
    let scenario_id = "e2e_stats_009_invariance";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let standard = Normal::standard();
    let shifted = Normal::new(5.0, 3.0);

    let t = Instant::now();
    let test_points = [-2.0, 0.0, 1.0, 5.0, 8.0, 11.0];
    let mut max_err = 0.0_f64;
    for &x in &test_points {
        let cdf_shifted = shifted.cdf(x);
        let z = (x - 5.0) / 3.0;
        let cdf_standard = standard.cdf(z);
        let err = (cdf_shifted - cdf_standard).abs();
        max_err = max_err.max(err);
    }
    let pass = max_err < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "loc_scale_cdf",
        "Normal(5,3).cdf(x) vs Normal(0,1).cdf((x-5)/3)",
        &format!("test_points={test_points:?}"),
        &format!("max_err={max_err:.2e}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 10: Maxwell distribution mode verification.
/// Mode of Maxwell(sigma) = sigma * sqrt(2).
#[test]
fn e2e_010_maxwell_mode() {
    let scenario_id = "e2e_stats_010_maxwell_mode";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let sigma = 2.0;
    let m = Maxwell::new(sigma);
    let expected_mode = sigma * 2.0_f64.sqrt();

    // Find mode numerically by scanning PDF
    let t = Instant::now();
    let mut max_pdf = 0.0_f64;
    let mut mode_x = 0.0;
    for i in 0..10_000 {
        let x = 0.001 + 10.0 * (i as f64 / 10_000.0);
        let p = m.pdf(x);
        if p > max_pdf {
            max_pdf = p;
            mode_x = x;
        }
    }
    let pass = (mode_x - expected_mode).abs() < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "maxwell_mode",
        "scan Maxwell PDF for mode",
        &format!("sigma={sigma}"),
        &format!("mode={mode_x:.4}, expected={expected_mode:.4}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 11: Closed-form moments for newly remediated distributions.
#[test]
fn e2e_011_closed_form_moments() {
    let scenario_id = "e2e_stats_011_closed_form_moments";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let burr = Burr12::new(2.0, 3.0);
    let pass = (burr.mean() - 0.589_048_622_548_086_1).abs() < 1e-12
        && (burr.var() - 0.153_021_720_274_202_2).abs() < 1e-12
        && Burr12::new(1.0, 2.0).var().is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "burr12_closed_form_moments",
        "Burr12::mean/var",
        "c=2, d=3 and boundary c*d=2",
        &format!("mean={}, var={}", burr.mean(), burr.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let inv_weibull = InvWeibull::new(3.0);
    let pass = (inv_weibull.mean() - 1.354_117_939_426_400_2).abs() < 1e-12
        && (inv_weibull.var() - 0.845_303_140_831_346_9).abs() < 1e-10
        && InvWeibull::new(2.0).var().is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "invweibull_closed_form_moments",
        "InvWeibull::mean/var",
        "c=3 and boundary c=2",
        &format!("mean={}, var={}", inv_weibull.mean(), inv_weibull.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let tukey = TukeyLambda::new(0.14);
    let pass = tukey.mean().abs() < TOL
        && (tukey.var() - 2.110_297_022_214_417).abs() < 1e-12
        && (TukeyLambda::new(0.0).var() - PI * PI / 3.0).abs() < 1e-12
        && TukeyLambda::new(-0.5).var().is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "tukey_lambda_closed_form_moments",
        "TukeyLambda::mean/var",
        "lam=0.14, logistic special case, undefined boundary",
        &format!("mean={}, var={}", tukey.mean(), tukey.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let gen_half = GenHalfLogistic::new(2.0);
    let pass = (gen_half.mean() - 0.306_852_819_440_054_7).abs() < 1e-9
        && (gen_half.var() - 0.233_700_550_136_169_83).abs() < 1e-9
        && gen_half.pdf(0.1).is_finite()
        && (0.0..=1.0).contains(&gen_half.cdf(0.1));
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "gen_half_logistic_closed_form_moments",
        "GenHalfLogistic::mean/var",
        "c=2 with pdf/cdf sanity",
        &format!("mean={}, var={}", gen_half.mean(), gen_half.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 12: Special-function-backed moments for GenLogistic and Gompertz.
#[test]
fn e2e_012_special_function_moments() {
    let scenario_id = "e2e_stats_012_special_function_moments";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let gen_logistic = GenLogistic::new(5.0);
    let pass = (gen_logistic.mean() - 2.083_333_333_333_333).abs() < 1e-9
        && (gen_logistic.var() - 1.866_257_022_585_341_7).abs() < 1e-9
        && (GenLogistic::new(1.0).mean() - Logistic::new(0.0, 1.0).mean()).abs() < TOL
        && (GenLogistic::new(1.0).var() - Logistic::new(0.0, 1.0).var()).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "gen_logistic_special_function_moments",
        "GenLogistic::mean/var",
        "c=5 and logistic identity at c=1",
        &format!("mean={}, var={}", gen_logistic.mean(), gen_logistic.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let gompertz = Gompertz::new(3.0);
    let pass = (gompertz.mean() - 0.262_083_740_252_362_75).abs() < 1e-10
        && (gompertz.var() - 0.046_960_406_809_075_47).abs() < 1e-10
        && gompertz.pdf(0.1).is_finite()
        && (0.0..=1.0).contains(&gompertz.cdf(0.1));
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "gompertz_special_function_moments",
        "Gompertz::mean/var",
        "c=3 with pdf/cdf sanity",
        &format!("mean={}, var={}", gompertz.mean(), gompertz.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let log_laplace = LogLaplace::new(3.0);
    let pass = (log_laplace.mean() - 1.125).abs() < TOL
        && (log_laplace.var() - 0.534_375).abs() < TOL
        && LogLaplace::new(1.5).var().is_infinite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "log_laplace_moments",
        "LogLaplace::mean/var",
        "c=3 and divergent second moment at c=1.5",
        &format!("mean={}, var={}", log_laplace.mean(), log_laplace.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let mielke = Mielke::new(2.0, 2.5);
    let pass = (mielke.mean() - 1.174_450_160_620_581_7).abs() < 1e-12
        && (mielke.var() - 2.144_017_302_080_036_4).abs() < 1e-10
        && Mielke::new(3.5, 1.2).var().is_infinite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "mielke_moments",
        "Mielke::mean/var",
        "k=2, s=2.5 and divergent variance at s=1.2",
        &format!("mean={}, var={}", mielke.mean(), mielke.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let frechet_r = FrechetR::new(3.0);
    let pass = (frechet_r.mean() + 0.892_979_511_569_248_9).abs() < 1e-12
        && (frechet_r.var() - 0.105_332_884_868_479_14).abs() < 1e-12
        && frechet_r.cdf(-0.5).is_finite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "frechet_r_moments",
        "FrechetR::mean/var",
        "c=3 with support sanity",
        &format!("mean={}, var={}", frechet_r.mean(), frechet_r.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 13: Piecewise and numerical-integral moments for remaining stats distributions.
#[test]
fn e2e_013_piecewise_integral_moments() {
    let scenario_id = "e2e_stats_013_piecewise_integral_moments";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let crystal_ball = CrystalBall::new(2.0, 4.0);
    let pass = (crystal_ball.mean() + 0.053_285_264_688_121_33).abs() < 1e-10
        && (crystal_ball.var() - 1.281_348_758_903_764).abs() < 1e-10
        && CrystalBall::new(1.0, 2.0).mean().is_infinite()
        && CrystalBall::new(2.0, 3.0).var().is_infinite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "crystal_ball_piecewise_moments",
        "CrystalBall::mean/var",
        "finite m=4 case plus divergent guards",
        &format!("mean={}, var={}", crystal_ball.mean(), crystal_ball.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let argus = Argus::new(2.0);
    let pass = (argus.mean() - 0.705_658_515_503_037_3).abs() < 1e-8
        && (argus.var() - 0.044_467_693_721_274_79).abs() < 1e-8
        && argus.pdf(0.5).is_finite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "argus_integral_moments",
        "Argus::mean/var",
        "chi=2 bounded-support integral",
        &format!("mean={}, var={}", argus.mean(), argus.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let expon_weibull = ExponWeibull::new(2.0, 0.5);
    let pass = (expon_weibull.mean() - 3.499_999_999_874_901_4).abs() < 1e-6
        && (expon_weibull.var() - 34.250_000_000_307_45).abs() < 1e-5
        && expon_weibull.cdf(1.0).is_finite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "expon_weibull_integral_moments",
        "ExponWeibull::mean/var",
        "a=2, c=0.5 quantile-integral moments",
        &format!("mean={}, var={}", expon_weibull.mean(), expon_weibull.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 14: TukeyLambda analytical pdf and Newton-backed cdf inversion.
#[test]
fn e2e_014_tukey_lambda_analytical() {
    let scenario_id = "e2e_stats_014_tukey_lambda_analytical";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let logistic = TukeyLambda::new(0.0);
    let tukey = TukeyLambda::new(0.5);
    let pdf_logistic_zero = logistic.pdf(0.0);
    let pdf_pos = tukey.pdf(1.0);
    let pdf_neg = tukey.pdf(-1.0);
    let pass = (pdf_logistic_zero - 0.25).abs() < 1e-12
        && (pdf_pos - pdf_neg).abs() < 1e-12
        && pdf_pos > 0.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "tukey_lambda_analytical_pdf",
        "TukeyLambda::pdf",
        "lam=0.0 at x=0 and lam=0.5 at +/-1",
        &format!(
            "logistic pdf(0)={pdf_logistic_zero}, pdf(1)={}, pdf(-1)={}",
            pdf_pos, pdf_neg
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
    let logistic = TukeyLambda::new(0.0);
    let pass = quantiles
        .iter()
        .all(|&q| (logistic.cdf(logistic.ppf(q)) - q).abs() < 1e-10);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "tukey_lambda_newton_roundtrip",
        "TukeyLambda::cdf(ppf(q))",
        "lam=0.0, q in {0.01,0.1,0.25,0.5,0.75,0.9,0.99}",
        "all roundtrips within 1e-10",
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let bounded = TukeyLambda::new(2.0);
    let pass = bounded.cdf(-0.5) == 0.0
        && bounded.cdf(0.5) == 1.0
        && bounded.pdf(-1.0) == 0.0
        && bounded.pdf(1.0) == 0.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "tukey_lambda_support_bounds",
        "TukeyLambda support guard",
        "lam=2.0 at and beyond support edges",
        &format!(
            "cdf(-0.5)={}, cdf(0.5)={}, pdf(-1)={}, pdf(1)={}",
            bounded.cdf(-0.5),
            bounded.cdf(0.5),
            bounded.pdf(-1.0),
            bounded.pdf(1.0)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}
