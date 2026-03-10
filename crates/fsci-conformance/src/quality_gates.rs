#![forbid(unsafe_code)]
//! bd-3jh.22: Coverage/Flake Budgets + Reliability Gates
//!
//! Parses `quality_gates.toml`, runs quality checks, and reports SLO compliance.
//! Provides flake detection (re-run failures), runtime budget enforcement,
//! and coverage floor validation.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

// ─── Configuration Types ──────────────────────────────────────────────

/// Top-level quality gates configuration, parsed from `quality_gates.toml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QualityGatesConfig {
    pub coverage: CoverageConfig,
    pub flake: FlakeConfig,
    pub runtime: RuntimeConfig,
    pub severity: SeverityConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CoverageConfig {
    pub default_line_coverage: u32,
    pub default_branch_coverage: u32,
    #[serde(default)]
    pub per_crate: BTreeMap<String, CrateCoverageOverride>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CrateCoverageOverride {
    pub line: u32,
    pub branch: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FlakeConfig {
    pub max_flake_rate: f64,
    pub rerun_count: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeConfig {
    pub unit_suite_max_seconds: u64,
    pub e2e_suite_max_seconds: u64,
    pub benchmark_suite_max_seconds: u64,
    pub per_test_timeout_seconds: u64,
    #[serde(default)]
    pub per_crate: BTreeMap<String, CrateRuntimeOverride>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CrateRuntimeOverride {
    #[serde(default)]
    pub unit_max_seconds: Option<u64>,
    #[serde(default)]
    pub e2e_max_seconds: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SeverityConfig {
    pub coverage_violation: ViolationAction,
    pub flake_violation: ViolationAction,
    pub runtime_violation: ViolationAction,
    pub critical_coverage_floor: u32,
    pub critical_runtime_multiplier: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ViolationAction {
    Warn,
    Fail,
}

// ─── SLO Check Results ────────────────────────────────────────────────

/// Result of checking all quality gates.
#[derive(Debug, Clone, Serialize)]
pub struct QualityGateReport {
    pub coverage_checks: Vec<CoverageSloCheck>,
    pub runtime_checks: Vec<RuntimeSloCheck>,
    pub flake_checks: Vec<FlakeSloCheck>,
    pub overall_pass: bool,
    pub violations: Vec<SloViolation>,
}

/// Coverage check for a single crate.
#[derive(Debug, Clone, Serialize)]
pub struct CoverageSloCheck {
    pub crate_name: String,
    pub line_coverage: Option<f64>,
    pub branch_coverage: Option<f64>,
    pub line_target: u32,
    pub branch_target: u32,
    pub line_pass: bool,
    pub branch_pass: bool,
}

/// Runtime check for a test suite.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSloCheck {
    pub suite_name: String,
    pub elapsed_seconds: f64,
    pub budget_seconds: u64,
    pub pass: bool,
    pub critical_breach: bool,
}

/// Flake detection result for a test.
#[derive(Debug, Clone, Serialize)]
pub struct FlakeSloCheck {
    pub test_name: String,
    pub total_runs: u32,
    pub failures: u32,
    pub passes: u32,
    pub is_flaky: bool,
    pub is_real_failure: bool,
}

/// A specific SLO violation with severity.
#[derive(Debug, Clone, Serialize)]
pub struct SloViolation {
    pub category: String,
    pub target: String,
    pub message: String,
    pub action: ViolationAction,
    pub critical: bool,
}

// ─── Config Loading ───────────────────────────────────────────────────

/// Load quality gates config from a TOML file.
pub fn load_config(path: &Path) -> Result<QualityGatesConfig, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    toml::from_str(&content).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

/// Load config from the default location (workspace root).
pub fn load_default_config() -> Result<QualityGatesConfig, String> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| String::from("cannot determine workspace root"))?;
    load_config(&workspace_root.join("quality_gates.toml"))
}

// ─── Coverage Checks ──────────────────────────────────────────────────

/// Check coverage SLOs for a given crate.
pub fn check_coverage(
    config: &QualityGatesConfig,
    crate_name: &str,
    line_coverage: Option<f64>,
    branch_coverage: Option<f64>,
) -> CoverageSloCheck {
    let (line_target, branch_target) = match config.coverage.per_crate.get(crate_name) {
        Some(over) => (over.line, over.branch),
        None => (
            config.coverage.default_line_coverage,
            config.coverage.default_branch_coverage,
        ),
    };

    let line_pass = line_coverage.is_none_or(|cov| cov >= f64::from(line_target));
    let branch_pass = branch_coverage.is_none_or(|cov| cov >= f64::from(branch_target));

    CoverageSloCheck {
        crate_name: crate_name.to_string(),
        line_coverage,
        branch_coverage,
        line_target,
        branch_target,
        line_pass,
        branch_pass,
    }
}

// ─── Runtime Checks ───────────────────────────────────────────────────

/// Check runtime SLO for a suite execution.
pub fn check_runtime(
    config: &QualityGatesConfig,
    suite_name: &str,
    elapsed: Duration,
) -> RuntimeSloCheck {
    let budget = match suite_name {
        "unit" => config.runtime.unit_suite_max_seconds,
        "e2e" => config.runtime.e2e_suite_max_seconds,
        "benchmark" => config.runtime.benchmark_suite_max_seconds,
        _ => config.runtime.unit_suite_max_seconds,
    };

    let elapsed_secs = elapsed.as_secs_f64();
    let pass = elapsed_secs <= budget as f64;
    let critical_breach =
        elapsed_secs > budget as f64 * config.severity.critical_runtime_multiplier;

    RuntimeSloCheck {
        suite_name: suite_name.to_string(),
        elapsed_seconds: elapsed_secs,
        budget_seconds: budget,
        pass,
        critical_breach,
    }
}

// ─── Flake Detection ──────────────────────────────────────────────────

/// Classify a test as flaky based on rerun results.
///
/// A test that fails on all reruns is a real failure.
/// A test that passes on any rerun is flaky.
pub fn classify_flake(test_name: &str, run_results: &[bool]) -> FlakeSloCheck {
    let total = run_results.len() as u32;
    let passes = run_results.iter().filter(|&&r| r).count() as u32;
    let failures = total - passes;

    let is_flaky = failures > 0 && passes > 0;
    let is_real_failure = failures == total;

    FlakeSloCheck {
        test_name: test_name.to_string(),
        total_runs: total,
        failures,
        passes,
        is_flaky,
        is_real_failure,
    }
}

// ─── Full Gate Check ──────────────────────────────────────────────────

/// Run all quality gate checks and produce a report.
pub fn run_quality_gates(
    config: &QualityGatesConfig,
    coverage_data: &[(String, Option<f64>, Option<f64>)],
    runtime_data: &[(String, Duration)],
    flake_data: &[(String, Vec<bool>)],
) -> QualityGateReport {
    let mut violations = Vec::new();

    let coverage_checks: Vec<_> = coverage_data
        .iter()
        .map(|(name, line, branch)| check_coverage(config, name, *line, *branch))
        .collect();

    for check in &coverage_checks {
        if !check.line_pass {
            let critical = check
                .line_coverage
                .is_some_and(|c| c < f64::from(config.severity.critical_coverage_floor));
            violations.push(SloViolation {
                category: String::from("coverage"),
                target: check.crate_name.clone(),
                message: format!(
                    "line coverage {:.1}% < target {}%",
                    check.line_coverage.unwrap_or(0.0),
                    check.line_target
                ),
                action: if critical {
                    ViolationAction::Fail
                } else {
                    config.severity.coverage_violation
                },
                critical,
            });
        }
        if !check.branch_pass {
            violations.push(SloViolation {
                category: String::from("coverage"),
                target: check.crate_name.clone(),
                message: format!(
                    "branch coverage {:.1}% < target {}%",
                    check.branch_coverage.unwrap_or(0.0),
                    check.branch_target
                ),
                action: config.severity.coverage_violation,
                critical: false,
            });
        }
    }

    let runtime_checks: Vec<_> = runtime_data
        .iter()
        .map(|(name, dur)| check_runtime(config, name, *dur))
        .collect();

    for check in &runtime_checks {
        if !check.pass {
            violations.push(SloViolation {
                category: String::from("runtime"),
                target: check.suite_name.clone(),
                message: format!(
                    "{:.1}s > budget {}s",
                    check.elapsed_seconds, check.budget_seconds
                ),
                action: if check.critical_breach {
                    ViolationAction::Fail
                } else {
                    config.severity.runtime_violation
                },
                critical: check.critical_breach,
            });
        }
    }

    let flake_checks: Vec<_> = flake_data
        .iter()
        .map(|(name, results)| classify_flake(name, results))
        .collect();

    for check in &flake_checks {
        if check.is_flaky {
            violations.push(SloViolation {
                category: String::from("flake"),
                target: check.test_name.clone(),
                message: format!("flaky: {}/{} runs failed", check.failures, check.total_runs),
                action: config.severity.flake_violation,
                critical: false,
            });
        }
    }

    let overall_pass = violations.iter().all(|v| v.action != ViolationAction::Fail);

    QualityGateReport {
        coverage_checks,
        runtime_checks,
        flake_checks,
        overall_pass,
        violations,
    }
}

/// Measure the wall-clock time of a test suite execution.
pub fn timed_suite<F, T>(f: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    (result, start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> QualityGatesConfig {
        load_default_config().expect("quality_gates.toml should be loadable")
    }

    #[test]
    fn load_config_parses_correctly() {
        let config = test_config();
        assert_eq!(config.coverage.default_line_coverage, 80);
        assert_eq!(config.coverage.default_branch_coverage, 60);
        assert_eq!(config.flake.max_flake_rate, 0.001);
        assert_eq!(config.flake.rerun_count, 3);
        assert_eq!(config.runtime.unit_suite_max_seconds, 120);
        assert_eq!(config.runtime.e2e_suite_max_seconds, 600);
        assert_eq!(config.severity.coverage_violation, ViolationAction::Warn);
        assert_eq!(config.severity.runtime_violation, ViolationAction::Fail);
    }

    #[test]
    fn load_config_has_per_crate_overrides() {
        let config = test_config();
        let runtime = config.coverage.per_crate.get("fsci-runtime");
        assert!(runtime.is_some());
        let runtime = runtime.unwrap();
        assert_eq!(runtime.line, 85);
        assert_eq!(runtime.branch, 70);
    }

    #[test]
    fn check_coverage_passes_when_above_threshold() {
        let config = test_config();
        let check = check_coverage(&config, "fsci-linalg", Some(85.0), Some(65.0));
        assert!(check.line_pass);
        assert!(check.branch_pass);
    }

    #[test]
    fn check_coverage_fails_when_below_threshold() {
        let config = test_config();
        let check = check_coverage(&config, "fsci-linalg", Some(70.0), Some(50.0));
        assert!(!check.line_pass);
        assert!(!check.branch_pass);
    }

    #[test]
    fn check_coverage_uses_per_crate_overrides() {
        let config = test_config();
        // fsci-runtime has line=85, branch=70
        let check = check_coverage(&config, "fsci-runtime", Some(82.0), Some(65.0));
        assert!(!check.line_pass); // 82 < 85
        assert!(!check.branch_pass); // 65 < 70
    }

    #[test]
    fn check_coverage_passes_none_values() {
        let config = test_config();
        let check = check_coverage(&config, "fsci-linalg", None, None);
        // None means no data, passes by default (no assertion possible)
        assert!(check.line_pass);
        assert!(check.branch_pass);
    }

    #[test]
    fn check_runtime_passes_within_budget() {
        let config = test_config();
        let check = check_runtime(&config, "unit", Duration::from_secs(60));
        assert!(check.pass);
        assert!(!check.critical_breach);
    }

    #[test]
    fn check_runtime_fails_over_budget() {
        let config = test_config();
        let check = check_runtime(&config, "unit", Duration::from_secs(150));
        assert!(!check.pass);
        assert!(!check.critical_breach); // 150 < 120 * 3 = 360
    }

    #[test]
    fn check_runtime_critical_breach() {
        let config = test_config();
        let check = check_runtime(&config, "unit", Duration::from_secs(400));
        assert!(!check.pass);
        assert!(check.critical_breach); // 400 > 120 * 3 = 360
    }

    #[test]
    fn classify_flake_all_pass() {
        let check = classify_flake("test_foo", &[true, true, true]);
        assert!(!check.is_flaky);
        assert!(!check.is_real_failure);
    }

    #[test]
    fn classify_flake_all_fail() {
        let check = classify_flake("test_foo", &[false, false, false]);
        assert!(!check.is_flaky);
        assert!(check.is_real_failure);
    }

    #[test]
    fn classify_flake_intermittent() {
        let check = classify_flake("test_foo", &[false, true, true]);
        assert!(check.is_flaky);
        assert!(!check.is_real_failure);
        assert_eq!(check.failures, 1);
        assert_eq!(check.passes, 2);
    }

    #[test]
    fn run_quality_gates_all_pass() {
        let config = test_config();
        let report = run_quality_gates(
            &config,
            &[(String::from("fsci-linalg"), Some(85.0), Some(65.0))],
            &[(String::from("unit"), Duration::from_secs(60))],
            &[(String::from("test_foo"), vec![true, true, true])],
        );
        assert!(report.overall_pass);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn run_quality_gates_coverage_violation_warns() {
        let config = test_config();
        let report = run_quality_gates(
            &config,
            &[(String::from("fsci-linalg"), Some(70.0), Some(65.0))],
            &[(String::from("unit"), Duration::from_secs(60))],
            &[],
        );
        // Coverage violation is "warn", so overall still passes
        assert!(report.overall_pass);
        assert!(!report.violations.is_empty());
        assert_eq!(report.violations[0].action, ViolationAction::Warn);
    }

    #[test]
    fn run_quality_gates_runtime_violation_fails() {
        let config = test_config();
        let report = run_quality_gates(
            &config,
            &[],
            &[(String::from("unit"), Duration::from_secs(150))],
            &[],
        );
        // Runtime violation is "fail"
        assert!(!report.overall_pass);
        assert_eq!(report.violations[0].action, ViolationAction::Fail);
    }

    #[test]
    fn run_quality_gates_critical_coverage_breach_fails() {
        let config = test_config();
        let report = run_quality_gates(
            &config,
            &[(String::from("fsci-linalg"), Some(40.0), Some(65.0))],
            &[],
            &[],
        );
        // 40% < critical floor 50%, so this is always fail
        assert!(!report.overall_pass);
        assert!(report.violations[0].critical);
    }

    #[test]
    fn timed_suite_measures_duration() {
        let (result, dur) = timed_suite(|| 42);
        assert_eq!(result, 42);
        assert!(dur.as_nanos() > 0);
    }
}
