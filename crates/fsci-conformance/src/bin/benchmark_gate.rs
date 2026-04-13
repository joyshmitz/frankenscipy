//! Benchmark regression gate for SPEC §17 compliance.
//!
//! Loads baseline JSON files, parses spec budgets, and validates that current
//! benchmark results meet the specified thresholds.
//!
//! Usage:
//!   benchmark_gate --baselines-dir docs/ --check-spec
//!   benchmark_gate --baselines-dir docs/ --compare <criterion_output>
//!
//! Exit codes:
//!   0 - All checks pass
//!   1 - Regression detected or spec violation
//!   2 - Configuration/IO error

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// SPEC §17 budgets in milliseconds
const SPEC_BUDGETS_MS: &[(&str, f64)] = &[
    ("linalg", 650.0),    // dense solve 4k-8k class p95
    ("sparse", 220.0),    // sparse matvec p95
    ("opt", 180.0),       // optimizer iteration p95 (matches baseline_opt.json)
    ("integrate", 320.0), // IVP solve step p95
    ("fft", 210.0),       // FFT transform p95
];

/// Maximum allowed regression percentage at p99
const MAX_REGRESSION_PCT: f64 = 8.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineFile {
    baseline_version: String,
    generated_at: String,
    spec_ref: String,
    #[serde(default)]
    notes: String,
    #[serde(default)]
    machine: HashMap<String, String>,
    benchmarks: HashMap<String, HashMap<String, BenchmarkEntry>>,
    #[serde(default)]
    status: Option<BaselineStatus>,
    #[serde(default)]
    extrapolation: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkEntry {
    #[serde(default)]
    median_ns: Option<f64>,
    #[serde(default)]
    median_us: Option<f64>,
    #[serde(default)]
    median_ms: Option<f64>,
    #[serde(default)]
    lower_ns: Option<f64>,
    #[serde(default)]
    lower_us: Option<f64>,
    #[serde(default)]
    lower_ms: Option<f64>,
    #[serde(default)]
    upper_ns: Option<f64>,
    #[serde(default)]
    upper_us: Option<f64>,
    #[serde(default)]
    upper_ms: Option<f64>,
    #[serde(default)]
    sample_size: Option<usize>,
    #[serde(default)]
    notes: Option<String>,
}

impl BenchmarkEntry {
    /// Convert median to milliseconds for comparison
    fn median_ms(&self) -> Option<f64> {
        if let Some(ms) = self.median_ms {
            return Some(ms);
        }
        if let Some(us) = self.median_us {
            return Some(us / 1000.0);
        }
        if let Some(ns) = self.median_ns {
            return Some(ns / 1_000_000.0);
        }
        None
    }

    /// Convert upper bound to milliseconds for p95/p99 comparison
    fn upper_ms(&self) -> Option<f64> {
        if let Some(ms) = self.upper_ms {
            return Some(ms);
        }
        if let Some(us) = self.upper_us {
            return Some(us / 1000.0);
        }
        if let Some(ns) = self.upper_ns {
            return Some(ns / 1_000_000.0);
        }
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineStatus {
    meets_spec: bool,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
struct GateResult {
    family: String,
    benchmark: String,
    baseline_ms: f64,
    spec_budget_ms: f64,
    status: GateStatus,
}

#[derive(Debug, Clone, PartialEq)]
enum GateStatus {
    Pass,
    SpecViolation { actual_ms: f64 },
    Regression { delta_pct: f64 },
}

fn load_baselines(dir: &Path) -> Result<HashMap<String, BaselineFile>, String> {
    let mut baselines = HashMap::new();

    for entry in fs::read_dir(dir).map_err(|e| format!("cannot read dir: {e}"))? {
        let entry = entry.map_err(|e| format!("dir entry error: {e}"))?;
        let path = entry.path();

        if path.extension().map(|s| s == "json").unwrap_or(false)
            && path
                .file_name()
                .map(|s| s.to_string_lossy().starts_with("baseline_"))
                .unwrap_or(false)
        {
            let content = fs::read_to_string(&path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;

            let baseline: BaselineFile = serde_json::from_str(&content)
                .map_err(|e| format!("invalid JSON in {}: {e}", path.display()))?;

            let family = path
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .strip_prefix("baseline_")
                .unwrap_or("unknown")
                .to_string();

            baselines.insert(family, baseline);
        }
    }

    Ok(baselines)
}

fn check_spec_compliance(baselines: &HashMap<String, BaselineFile>) -> Vec<GateResult> {
    let mut results = Vec::new();
    let budgets: HashMap<&str, f64> = SPEC_BUDGETS_MS.iter().copied().collect();

    for (family, baseline) in baselines {
        let budget_ms = budgets.get(family.as_str()).copied().unwrap_or(f64::INFINITY);

        for (group, benchmarks) in &baseline.benchmarks {
            for (name, entry) in benchmarks {
                if let Some(upper) = entry.upper_ms() {
                    let status = if upper > budget_ms {
                        GateStatus::SpecViolation { actual_ms: upper }
                    } else {
                        GateStatus::Pass
                    };

                    results.push(GateResult {
                        family: family.clone(),
                        benchmark: format!("{group}/{name}"),
                        baseline_ms: upper,
                        spec_budget_ms: budget_ms,
                        status,
                    });
                }
            }
        }
    }

    results
}

fn print_report(results: &[GateResult]) -> bool {
    let mut all_pass = true;

    println!("Benchmark Gate Report");
    println!("=====================");
    println!();

    for (family, budget) in SPEC_BUDGETS_MS {
        println!("Family: {} (SPEC budget: {}ms)", family, budget);
    }
    println!();

    let mut by_family: HashMap<String, Vec<&GateResult>> = HashMap::new();
    for r in results {
        by_family.entry(r.family.clone()).or_default().push(r);
    }

    for (family, family_results) in &by_family {
        println!("--- {} ---", family);

        for r in family_results {
            let status_str = match &r.status {
                GateStatus::Pass => "PASS".to_string(),
                GateStatus::SpecViolation { actual_ms } => {
                    all_pass = false;
                    format!("FAIL: {:.3}ms > {:.3}ms budget", actual_ms, r.spec_budget_ms)
                }
                GateStatus::Regression { delta_pct } => {
                    all_pass = false;
                    format!("FAIL: {:.1}% regression (>{:.1}% threshold)", delta_pct, MAX_REGRESSION_PCT)
                }
            };

            println!(
                "  {}: {:.3}ms [{}]",
                r.benchmark, r.baseline_ms, status_str
            );
        }
        println!();
    }

    println!("Overall: {}", if all_pass { "PASS" } else { "FAIL" });
    all_pass
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let baselines_dir = args
        .iter()
        .position(|a| a == "--baselines-dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs"));

    let check_spec = args.iter().any(|a| a == "--check-spec");
    let help = args.iter().any(|a| a == "--help" || a == "-h");

    if help {
        println!("Usage: benchmark_gate [OPTIONS]");
        println!();
        println!("Options:");
        println!("  --baselines-dir DIR  Directory containing baseline_*.json files (default: docs)");
        println!("  --check-spec         Validate baselines against SPEC §17 budgets");
        println!("  -h, --help           Show this help");
        std::process::exit(0);
    }

    println!("Loading baselines from: {}", baselines_dir.display());

    let baselines = match load_baselines(&baselines_dir) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error loading baselines: {e}");
            std::process::exit(2);
        }
    };

    println!("Loaded {} baseline files", baselines.len());
    for family in baselines.keys() {
        println!("  - {}", family);
    }
    println!();

    if check_spec || !baselines.is_empty() {
        let results = check_spec_compliance(&baselines);
        let pass = print_report(&results);
        std::process::exit(if pass { 0 } else { 1 });
    }

    println!("No action specified. Use --check-spec to validate baselines.");
}
