#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-009 (Statistical distributions).
//!
//! Acceptance criteria:
//!   Happy-path:     1-3  (distribution evaluation, CDF/PPF roundtrip, parameter estimation)
//!   Error recovery: 4-6  (invalid parameters, boundary quantiles, NaN inputs)
//!   Adversarial:    7-10 (extreme parameters, tail probabilities, identity checks)
//!
//! Each scenario emits topology-compliant artifacts to
//! `fixtures/artifacts/FSCI-P2C-012/e2e/runs/{run_id}/{scenario_id}/`
//! containing `events.jsonl` and `summary.json`.

use fsci_conformance::PacketFamily;
use std::f64::consts::PI;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    Argus,
    BetaDist,
    Burr12,
    ChiSquared,
    ContinuousDistribution,
    CrystalBall,
    ExponWeibull,
    Exponential,
    FrechetR,
    GammaDist,
    GenHalfLogistic,
    GenLogistic,
    Gompertz,
    Gumbel,
    InvWeibull,
    LogLaplace,
    Logistic,
    Lognormal,
    Maxwell,
    Mielke,
    Normal,
    Pareto,
    Rayleigh,
    StudentT,
    TukeyLambda,
    Uniform,
    Weibull,
    chi2_contingency,
    // Hypothesis tests
    chisquare,
    describe,
    f_oneway,
    fisher_exact,
    jarque_bera,
    kendalltau,
    kruskal,
    ks_1samp,
    ks_2samp,
    kurtosis,
    linregress,
    mannwhitneyu,
    normaltest,
    pearsonr,
    shapiro,
    skew,
    spearmanr,
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    wilcoxon,
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

fn packet_id() -> &'static str {
    PacketFamily::Stats.packet_id()
}

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
        .join(packet_id())
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
        packet_id: packet_id().to_string(),
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

// ======================================================================
// HYPOTHESIS TESTING SCENARIOS (15-23)
// ======================================================================

/// Scenario 15: T-tests (ttest_1samp, ttest_ind, ttest_rel).
/// Verifies t-statistics and p-values for standard test cases.
#[test]
fn e2e_015_t_tests() {
    let scenario_id = "e2e_stats_015_ttests";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // One-sample t-test: sample from N(1, 1) tested against μ=0
    let t = Instant::now();
    let sample: Vec<f64> = vec![1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4, 1.0, 0.95];
    let result = ttest_1samp(&sample, 0.0);
    // With mean ≈ 1.085 and n=10, t-statistic should be positive and significant
    let pass = result.statistic > 0.0 && result.pvalue < 0.05 && result.df == 9.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "ttest_1samp",
        "ttest_1samp(sample, 0)",
        "10 samples from N(1,1), μ0=0",
        &format!(
            "t={:.4}, p={:.4}, df={}",
            result.statistic, result.pvalue, result.df
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Independent two-sample t-test: comparing two different populations
    let t = Instant::now();
    let group_a: Vec<f64> = vec![5.1, 5.3, 5.0, 4.9, 5.2, 5.4, 5.1, 5.0];
    let group_b: Vec<f64> = vec![4.2, 4.0, 4.3, 3.9, 4.1, 4.4, 4.0, 4.2];
    let result = ttest_ind(&group_a, &group_b);
    // Group A has higher mean, so statistic should be positive
    let pass = result.statistic > 0.0 && result.pvalue < 0.001 && result.df > 0.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ttest_ind",
        "ttest_ind(A, B)",
        "8 samples each, distinct means",
        &format!(
            "t={:.4}, p={:.6}, df={}",
            result.statistic, result.pvalue, result.df
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Paired t-test: before/after measurements
    let t = Instant::now();
    let before: Vec<f64> = vec![10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 12.0, 11.0];
    let after: Vec<f64> = vec![12.0, 14.0, 13.0, 15.0, 11.0, 15.0, 13.0, 12.0];
    let result = ttest_rel(&before, &after);
    // After values are higher, so difference (before - after) is negative, t < 0
    let pass = result.statistic < 0.0 && result.pvalue < 0.01 && result.df == 7.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "ttest_rel",
        "ttest_rel(before, after)",
        "8 paired measurements",
        &format!(
            "t={:.4}, p={:.4}, df={}",
            result.statistic, result.pvalue, result.df
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Null hypothesis should not be rejected when data is from same distribution
    let t = Instant::now();
    let null_sample: Vec<f64> = vec![0.1, -0.2, 0.15, -0.05, 0.08, -0.1, 0.12, -0.08];
    let result = ttest_1samp(&null_sample, 0.0);
    // p-value should be high (not significant)
    let pass = result.pvalue > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "ttest_null_hypothesis",
        "ttest_1samp(centered, 0)",
        "8 samples centered around 0",
        &format!("t={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 16: ANOVA and variance tests (f_oneway, chisquare).
/// Verifies F-statistics and chi-square statistics.
#[test]
fn e2e_016_anova_chisquare() {
    let scenario_id = "e2e_stats_016_anova";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // One-way ANOVA: three groups with different means
    let t = Instant::now();
    let group1: [f64; 6] = [1.0, 1.2, 0.9, 1.1, 1.0, 0.8];
    let group2: [f64; 6] = [2.0, 2.1, 1.9, 2.2, 2.0, 1.8];
    let group3: [f64; 6] = [3.0, 3.2, 2.9, 3.1, 3.0, 2.8];
    let result = f_oneway(&[&group1, &group2, &group3]);
    // Groups are clearly different, F should be large and p small
    let pass = result.statistic > 50.0 && result.pvalue < 0.001;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "f_oneway",
        "f_oneway(g1, g2, g3)",
        "3 groups with means ~1, ~2, ~3",
        &format!("F={:.2}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Chi-square goodness of fit: observed vs expected uniform
    let t = Instant::now();
    let observed: [f64; 4] = [50.0, 45.0, 55.0, 50.0];
    let (chi2, pvalue) = chisquare(&observed, None);
    // Close to uniform, so chi2 should be small and p large
    let pass = chi2 < 5.0 && pvalue > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "chisquare_uniform",
        "chisquare(obs, None)",
        "observed=[50,45,55,50]",
        &format!("chi2={:.4}, p={:.4}", chi2, pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Chi-square with custom expected frequencies
    let t = Instant::now();
    let observed: [f64; 3] = [100.0, 50.0, 50.0];
    let expected: [f64; 3] = [100.0, 50.0, 50.0];
    let (chi2, pvalue) = chisquare(&observed, Some(&expected));
    // Perfect match, chi2 = 0
    let pass = chi2.abs() < TOL && pvalue > 0.999;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "chisquare_perfect",
        "chisquare(obs, exp)",
        "obs=exp=[100,50,50]",
        &format!("chi2={:.6}, p={:.4}", chi2, pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // ANOVA with no difference between groups
    let t = Instant::now();
    let g1: [f64; 5] = [5.0, 5.1, 4.9, 5.2, 4.8];
    let g2: [f64; 5] = [5.1, 4.9, 5.0, 5.2, 4.8];
    let g3: [f64; 5] = [4.9, 5.0, 5.1, 4.8, 5.2];
    let result = f_oneway(&[&g1, &g2, &g3]);
    // No real difference, p should be high
    let pass = result.pvalue > 0.5;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "f_oneway_null",
        "f_oneway(similar groups)",
        "3 groups with ~same mean",
        &format!("F={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 17: Non-parametric tests (mannwhitneyu, wilcoxon, kruskal).
/// Tests rank-based methods for comparing distributions.
#[test]
fn e2e_017_nonparametric_tests() {
    let scenario_id = "e2e_stats_017_nonparametric";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Mann-Whitney U: comparing two groups with shift
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let result = mannwhitneyu(&x, &y);
    // Clear separation, U should be 0 (all x < all y) and p very small
    let pass = result.statistic < 5.0 && result.pvalue < 0.001;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "mannwhitneyu_distinct",
        "mannwhitneyu(x, y)",
        "x=[1..8], y=[9..16]",
        &format!("U={:.2}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Wilcoxon signed-rank: paired comparison (needs n >= 10)
    let t = Instant::now();
    let before: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
    let after: Vec<f64> = vec![15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0];
    let result = wilcoxon(&before, &after);
    // All after > before, so significant (with enough pairs)
    let pass = result.pvalue < 0.2;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "wilcoxon_paired",
        "wilcoxon(before, after)",
        "10 pairs, all show increase",
        &format!("statistic={:.2}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Kruskal-Wallis: non-parametric ANOVA
    let t = Instant::now();
    let g1: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    let g2: [f64; 5] = [6.0, 7.0, 8.0, 9.0, 10.0];
    let g3: [f64; 5] = [11.0, 12.0, 13.0, 14.0, 15.0];
    let result = kruskal(&[&g1, &g2, &g3]);
    // Clear separation, H should be large
    let pass = result.statistic > 10.0 && result.pvalue < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "kruskal_distinct",
        "kruskal(g1, g2, g3)",
        "3 non-overlapping groups",
        &format!("H={:.2}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Mann-Whitney with overlapping groups
    let t = Instant::now();
    let a: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let b: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let result = mannwhitneyu(&a, &b);
    // Interleaved, no significant difference expected
    let pass = result.pvalue > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "mannwhitneyu_overlap",
        "mannwhitneyu(odd, even)",
        "interleaved sequences",
        &format!("U={:.2}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 18: Correlation tests (pearsonr, spearmanr, kendalltau).
/// Verifies correlation coefficients for known relationships.
#[test]
fn e2e_018_correlation_tests() {
    let scenario_id = "e2e_stats_018_correlation";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Perfect positive linear correlation
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
    let result = pearsonr(&x, &y);
    let pass = (result.statistic - 1.0).abs() < TOL && result.pvalue < 0.001;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "pearsonr_perfect",
        "pearsonr(x, 2x)",
        "perfect linear",
        &format!("r={:.6}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Spearman rank correlation: monotonic but non-linear
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]; // y = x^2
    let result = spearmanr(&x, &y);
    // Perfect monotonic relationship
    let pass = (result.statistic - 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "spearmanr_monotonic",
        "spearmanr(x, x^2)",
        "quadratic but monotonic",
        &format!("rho={:.6}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Kendall tau: concordance measure
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // perfectly negative
    let result = kendalltau(&x, &y);
    let pass = (result.statistic + 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "kendalltau_negative",
        "kendalltau(asc, desc)",
        "perfectly negative",
        &format!("tau={:.6}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // No correlation
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = vec![5.0, 2.0, 8.0, 1.0, 6.0, 3.0, 7.0, 4.0]; // scrambled
    let result = pearsonr(&x, &y);
    // Low correlation expected
    let pass = result.statistic.abs() < 0.5 && result.pvalue > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "pearsonr_uncorrelated",
        "pearsonr(x, scrambled)",
        "no linear relationship",
        &format!("r={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 19: Linear regression (linregress).
/// Verifies slope, intercept, and regression statistics.
#[test]
fn e2e_019_linear_regression() {
    let scenario_id = "e2e_stats_019_regression";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Perfect linear fit: y = 2x + 3
    let t = Instant::now();
    let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];
    let result = linregress(&x, &y);
    let pass = (result.slope - 2.0).abs() < TOL
        && (result.intercept - 3.0).abs() < TOL
        && (result.rvalue - 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "linregress_perfect",
        "linregress(x, 2x+3)",
        "perfect linear",
        &format!(
            "slope={:.4}, intercept={:.4}, r={:.4}",
            result.slope, result.intercept, result.rvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Regression with noise
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = vec![2.1, 3.9, 6.2, 7.8, 10.1, 11.9, 14.2, 15.8, 18.1, 19.9];
    let result = linregress(&x, &y);
    // Slope should be close to 2, intercept close to 0
    let pass = (result.slope - 2.0).abs() < 0.1
        && result.intercept.abs() < 0.5
        && result.rvalue > 0.99
        && result.pvalue < 0.001;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "linregress_noisy",
        "linregress with noise",
        "y ≈ 2x with small noise",
        &format!(
            "slope={:.4}, intercept={:.4}, r={:.4}, p={:.6}",
            result.slope, result.intercept, result.rvalue, result.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Negative slope
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    let result = linregress(&x, &y);
    let pass = (result.slope + 2.0).abs() < TOL
        && (result.intercept - 12.0).abs() < TOL
        && (result.rvalue + 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "linregress_negative",
        "linregress(x, -2x+12)",
        "negative slope",
        &format!(
            "slope={:.4}, intercept={:.4}, r={:.4}",
            result.slope, result.intercept, result.rvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Stderr should be small for good fit
    let t = Instant::now();
    let pass = result.stderr < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "linregress_stderr",
        "check stderr",
        "stderr for perfect fit",
        &format!("stderr={:.6}", result.stderr),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 20: Descriptive statistics (describe, skew, kurtosis).
/// Verifies summary statistics computation.
#[test]
fn e2e_020_descriptive_stats() {
    let scenario_id = "e2e_stats_020_descriptive";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Basic describe
    let t = Instant::now();
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let result = describe(&data);
    let expected_mean = 5.5;
    let expected_var = 9.166_666_666_666_666; // sample variance with ddof=1
    let pass = result.nobs == 10
        && (result.mean - expected_mean).abs() < TOL
        && (result.variance - expected_var).abs() < 0.001
        && result.minmax == (1.0, 10.0);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "describe_basic",
        "describe([1..10])",
        "n=10, uniform spacing",
        &format!(
            "n={}, mean={:.4}, var={:.4}, minmax={:?}",
            result.nobs, result.mean, result.variance, result.minmax
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Skewness: symmetric distribution should have skew ≈ 0
    let t = Instant::now();
    let symmetric: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let sk = skew(&symmetric);
    let pass = sk.abs() < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "skew_symmetric",
        "skew(symmetric)",
        "palindromic data",
        &format!("skew={:.6}", sk),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Skewness: right-skewed distribution
    let t = Instant::now();
    let right_skewed: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 10.0];
    let sk = skew(&right_skewed);
    let pass = sk > 1.0; // positive skew
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "skew_right",
        "skew(right-skewed)",
        "long right tail",
        &format!("skew={:.4}", sk),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Kurtosis: normal distribution should have excess kurtosis ≈ 0
    // Heavy-tailed should have positive kurtosis
    let t = Instant::now();
    let heavy_tailed: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
    let kurt = kurtosis(&heavy_tailed);
    let pass = kurt > 3.0; // heavy tails -> high kurtosis
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "kurtosis_heavy",
        "kurtosis(heavy-tailed)",
        "one extreme outlier",
        &format!("kurtosis={:.4}", kurt),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 21: Normality tests (shapiro, normaltest, jarque_bera).
/// Verifies detection of normal vs non-normal distributions.
#[test]
fn e2e_021_normality_tests() {
    let scenario_id = "e2e_stats_021_normality";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Shapiro-Wilk: uniform data should fail normality
    let t = Instant::now();
    let uniform_data: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let result = shapiro(&uniform_data);
    // Uniform should reject normality (low p-value or W < 1)
    let pass = result.statistic < 0.98 || result.pvalue < 0.05;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "shapiro_uniform",
        "shapiro(uniform)",
        "50 evenly spaced points",
        &format!("W={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // D'Agostino-Pearson normaltest: bimodal should fail
    let t = Instant::now();
    let mut bimodal: Vec<f64> = vec![];
    for _ in 0..25 {
        bimodal.push(0.0);
    }
    for _ in 0..25 {
        bimodal.push(10.0);
    }
    let result = normaltest(&bimodal);
    // Bimodal should reject normality
    let pass = result.pvalue < 0.05;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "normaltest_bimodal",
        "normaltest(bimodal)",
        "25 at 0, 25 at 10",
        &format!("statistic={:.4}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Jarque-Bera: check skewness and kurtosis
    let t = Instant::now();
    let skewed: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 100.0];
    let result = jarque_bera(&skewed);
    // Highly skewed should reject normality
    let pass = result.statistic > 1.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "jarque_bera_skewed",
        "jarque_bera(skewed)",
        "right-skewed data",
        &format!("JB={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Shapiro should give high p-value for nearly-normal data
    let t = Instant::now();
    // Approximate normal samples (actually generated, but pretend they're from N(0,1))
    let normal_ish: Vec<f64> = vec![-1.5, -1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0, 1.5];
    let result = shapiro(&normal_ish);
    // Should not reject normality
    let pass = result.pvalue > 0.05;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "shapiro_normal",
        "shapiro(normal-like)",
        "roughly bell-shaped",
        &format!("W={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 22: Kolmogorov-Smirnov tests (ks_1samp, ks_2samp).
/// Verifies goodness-of-fit testing.
#[test]
fn e2e_022_ks_tests() {
    let scenario_id = "e2e_stats_022_ks";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // KS one-sample: test against uniform CDF
    let t = Instant::now();
    let uniform_samples: Vec<f64> = (0..100).map(|i| (i as f64 + 0.5) / 100.0).collect();
    let result = ks_1samp(&uniform_samples, |x| x.clamp(0.0, 1.0)); // uniform CDF
    // Should match well (small D, high p)
    let pass = result.statistic < 0.1 && result.pvalue > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "ks_1samp_uniform",
        "ks_1samp(uniform, F_uniform)",
        "100 uniform samples",
        &format!("D={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // KS one-sample: non-uniform data against uniform CDF
    let t = Instant::now();
    let skewed_samples: Vec<f64> = (0..50).map(|i| (i as f64 / 50.0).powi(2)).collect();
    let result = ks_1samp(&skewed_samples, |x| x.clamp(0.0, 1.0));
    // Should reject (large D, small p)
    let pass = result.statistic > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ks_1samp_reject",
        "ks_1samp(x^2, F_uniform)",
        "squared samples vs uniform",
        &format!("D={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // KS two-sample: same distribution
    let t = Instant::now();
    let sample1: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let sample2: Vec<f64> = (0..30).map(|i| i as f64 + 0.5).collect();
    let result = ks_2samp(&sample1, &sample2);
    // Similar distributions, small D
    let pass = result.statistic < 0.2;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "ks_2samp_similar",
        "ks_2samp(shifted)",
        "same shape, slight offset",
        &format!("D={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // KS two-sample: different distributions
    let t = Instant::now();
    let normal_like: Vec<f64> = vec![-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    let uniform_like: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = ks_2samp(&normal_like, &uniform_like);
    // Different distributions, should have larger D
    let pass = result.statistic > 0.3;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "ks_2samp_different",
        "ks_2samp(normal, uniform)",
        "different distributions",
        &format!("D={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 23: Contingency table tests (fisher_exact, chi2_contingency).
/// Verifies tests for independence in categorical data.
#[test]
fn e2e_023_contingency_tests() {
    let scenario_id = "e2e_stats_023_contingency";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Fisher's exact test: 2x2 table with strong association
    let t = Instant::now();
    let table = [[10.0, 2.0], [1.0, 12.0]];
    let result = fisher_exact(&table);
    // Strong association, p should be small
    let pass = result.pvalue < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "fisher_exact_assoc",
        "fisher_exact([[10,2],[1,12]])",
        "strong association",
        &format!(
            "odds_ratio={:.4}, p={:.6}",
            result.odds_ratio, result.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Fisher's exact test: independent (null)
    let t = Instant::now();
    let table = [[5.0, 5.0], [5.0, 5.0]];
    let result = fisher_exact(&table);
    // No association, p should be high
    let pass = result.pvalue > 0.5 && (result.odds_ratio - 1.0).abs() < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "fisher_exact_null",
        "fisher_exact([[5,5],[5,5]])",
        "no association",
        &format!(
            "odds_ratio={:.4}, p={:.4}",
            result.odds_ratio, result.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Chi-square test of independence: larger table
    let t = Instant::now();
    let observed = vec![vec![20.0, 30.0, 50.0], vec![30.0, 40.0, 30.0]];
    let result = chi2_contingency(&observed);
    // Some deviation from independence expected
    let pass = result.statistic > 0.0 && result.dof == 2;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "chi2_contingency",
        "chi2_contingency(2x3)",
        "2x3 contingency table",
        &format!(
            "chi2={:.4}, p={:.4}, dof={}",
            result.statistic, result.pvalue, result.dof
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Chi-square: perfectly proportional (independence)
    let t = Instant::now();
    let observed = vec![vec![10.0, 20.0, 30.0], vec![10.0, 20.0, 30.0]];
    let result = chi2_contingency(&observed);
    // Perfect independence, chi2 ≈ 0
    let pass = result.statistic < 0.01 && result.pvalue > 0.99;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "chi2_independent",
        "chi2_contingency(proportional)",
        "perfectly proportional rows",
        &format!("chi2={:.6}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}
