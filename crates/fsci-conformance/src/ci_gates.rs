#![forbid(unsafe_code)]
//! bd-3jh.10: CI Gate Topology (G1..G8) + Failure Forensics
//!
//! Programmatic definitions of the CI gate topology with dependency wiring,
//! failure reporting, and artifact bundle generation.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Gate identifier within the CI pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GateId {
    /// G1: `cargo fmt --check` + `cargo clippy`
    G1Lint,
    /// G2: `cargo test --workspace` (unit + property)
    G2UnitTests,
    /// G3: Differential oracle conformance
    G3Conformance,
    /// G4: Adversarial/fuzz regression
    G4Adversarial,
    /// G5: E2E scenario orchestration
    G5E2e,
    /// G6: Performance regression check
    G6Performance,
    /// G7: Artifact schema validation
    G7Schema,
    /// G8: RaptorQ decode-proof verification
    G8RaptorQ,
}

impl GateId {
    /// All gate IDs in order.
    pub const ALL: [Self; 8] = [
        Self::G1Lint,
        Self::G2UnitTests,
        Self::G3Conformance,
        Self::G4Adversarial,
        Self::G5E2e,
        Self::G6Performance,
        Self::G7Schema,
        Self::G8RaptorQ,
    ];

    /// Human-readable name for the gate.
    pub fn name(self) -> &'static str {
        match self {
            Self::G1Lint => "G1: Formatting + Linting",
            Self::G2UnitTests => "G2: Unit + Property Tests",
            Self::G3Conformance => "G3: Differential Conformance",
            Self::G4Adversarial => "G4: Adversarial Regression",
            Self::G5E2e => "G5: E2E Scenarios",
            Self::G6Performance => "G6: Performance Regression",
            Self::G7Schema => "G7: Schema Validation",
            Self::G8RaptorQ => "G8: RaptorQ Proofs",
        }
    }

    /// The cargo command(s) this gate runs.
    ///
    /// Kept in sync with `.github/workflows/ci.yml`. When editing either file,
    /// update both. Drift between this manifest and the workflow was the
    /// precipitating issue in frankenscipy-m0cn.
    pub fn commands(self) -> &'static [&'static str] {
        match self {
            Self::G1Lint => &[
                "cargo fmt --check",
                "cargo clippy --workspace --all-targets -- -D warnings",
            ],
            Self::G2UnitTests => &["cargo test --workspace -- --nocapture"],
            Self::G3Conformance => &[
                "cargo test -p fsci-conformance --test golden_journeys -- --nocapture",
                "cargo test -p fsci-conformance --test diff_fft -- --nocapture",
                "cargo test -p fsci-conformance --test diff_sparse -- --nocapture",
                "cargo test -p fsci-conformance --lib tests::differential -- --nocapture",
            ],
            Self::G4Adversarial => &["cargo test -p fsci-conformance --test smoke -- --nocapture"],
            Self::G5E2e => &[
                "cargo test -p fsci-conformance --test e2e_casp -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_cluster -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_fft -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_interpolate -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_io -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_ivp -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_linalg -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_ndimage -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_optimize -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_orchestrator -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_signal -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_sparse -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_spatial -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_special -- --nocapture",
                "cargo test -p fsci-conformance --test e2e_stats -- --nocapture",
            ],
            Self::G6Performance => &[
                "cargo test -p fsci-conformance --test perf_linalg -- --nocapture",
                "cargo test -p fsci-conformance --test perf_ivp -- --nocapture",
                "cargo test -p fsci-conformance --test perf_arrayapi -- --nocapture",
            ],
            Self::G7Schema => &[
                "cargo test -p fsci-conformance --test schema_validation -- --nocapture",
                "cargo test -p fsci-conformance --test evidence_p2c001 -- --nocapture",
                "cargo test -p fsci-conformance --test evidence_p2c002 -- --nocapture",
                "cargo test -p fsci-conformance --test evidence_p2c003 -- --nocapture",
                "cargo test -p fsci-conformance --test evidence_p2c004 -- --nocapture",
                "cargo test -p fsci-conformance --test evidence_p2c005 -- --nocapture",
                "cargo test -p fsci-conformance --test evidence_p2c006 -- --nocapture",
                "cargo test -p fsci-conformance --test evidence_p2c007 -- --nocapture",
                "cargo test -p fsci-conformance --test evidence_p2c008 -- --nocapture",
                "cargo test -p fsci-conformance --test evidence_p2c012 -- --nocapture",
                "cargo test -p fsci-conformance --test evidence_p2c016 -- --nocapture",
            ],
            Self::G8RaptorQ => {
                &["cargo test -p fsci-conformance --test raptorq_proofs -- --nocapture"]
            }
        }
    }

    /// Gates that must pass before this gate can run.
    pub fn dependencies(self) -> &'static [GateId] {
        match self {
            Self::G1Lint => &[],
            Self::G2UnitTests => &[Self::G1Lint],
            Self::G3Conformance => &[Self::G2UnitTests],
            Self::G4Adversarial => &[Self::G2UnitTests],
            Self::G5E2e => &[Self::G2UnitTests],
            Self::G6Performance => &[Self::G2UnitTests],
            Self::G7Schema => &[Self::G2UnitTests],
            Self::G8RaptorQ => &[Self::G2UnitTests],
        }
    }

    /// Time budget for this gate in seconds.
    pub fn time_budget_seconds(self) -> u64 {
        match self {
            Self::G1Lint => 120,
            Self::G2UnitTests => 300,
            Self::G3Conformance => 300,
            Self::G4Adversarial => 300,
            Self::G5E2e => 900,
            Self::G6Performance => 600,
            Self::G7Schema => 300,
            Self::G8RaptorQ => 300,
        }
    }
}

impl std::fmt::Display for GateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Result of running a single CI gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub gate_id: GateId,
    pub passed: bool,
    pub failed_check: Option<String>,
    pub summary: String,
    pub artifact_bundle_path: Option<PathBuf>,
    pub replay_command: Option<String>,
    pub elapsed_seconds: f64,
    pub time_budget_seconds: u64,
    pub within_budget: bool,
}

/// Result of running all CI gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiPipelineReport {
    pub gates: BTreeMap<String, GateResult>,
    pub all_passed: bool,
    pub total_elapsed_seconds: f64,
    pub gates_passed: usize,
    pub gates_failed: usize,
}

/// Build a gate result from execution data.
pub fn build_gate_result(
    gate_id: GateId,
    passed: bool,
    failed_check: Option<String>,
    elapsed_seconds: f64,
    artifact_dir: Option<PathBuf>,
) -> GateResult {
    let summary = if passed {
        format!("{} passed in {elapsed_seconds:.1}s", gate_id.name())
    } else {
        format!(
            "{} FAILED: {}",
            gate_id.name(),
            failed_check.as_deref().unwrap_or("unknown check")
        )
    };

    let replay_command = if !passed {
        gate_id.commands().first().map(|cmd| (*cmd).to_string())
    } else {
        None
    };

    let within_budget = elapsed_seconds <= gate_id.time_budget_seconds() as f64;

    GateResult {
        gate_id,
        passed,
        failed_check,
        summary,
        artifact_bundle_path: artifact_dir,
        replay_command,
        elapsed_seconds,
        time_budget_seconds: gate_id.time_budget_seconds(),
        within_budget,
    }
}

/// Aggregate individual gate results into a pipeline report.
pub fn build_pipeline_report(results: &[GateResult]) -> CiPipelineReport {
    let mut gates = BTreeMap::new();
    let mut total = 0.0;
    let mut passed_count = 0;
    let mut failed_count = 0;

    for result in results {
        total += result.elapsed_seconds;
        if result.passed {
            passed_count += 1;
        } else {
            failed_count += 1;
        }
        gates.insert(result.gate_id.name().to_string(), result.clone());
    }

    CiPipelineReport {
        gates,
        all_passed: failed_count == 0,
        total_elapsed_seconds: total,
        gates_passed: passed_count,
        gates_failed: failed_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_id_names_are_unique() {
        let names: Vec<_> = GateId::ALL.iter().map(|g| g.name()).collect();
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(names.len(), unique.len(), "gate names must be unique");
    }

    #[test]
    fn gate_dependencies_form_dag() {
        // G1 has no deps
        assert!(GateId::G1Lint.dependencies().is_empty());
        // G2 depends on G1
        assert!(GateId::G2UnitTests.dependencies().contains(&GateId::G1Lint));
        // G3-G8 depend on G2
        for gate in &GateId::ALL[2..] {
            assert!(
                gate.dependencies().contains(&GateId::G2UnitTests),
                "{gate} should depend on G2"
            );
        }
    }

    #[test]
    fn no_circular_dependencies() {
        for gate in &GateId::ALL {
            for dep in gate.dependencies() {
                assert!(
                    !dep.dependencies().contains(gate),
                    "{gate} and {dep} have circular dependency"
                );
            }
        }
    }

    #[test]
    fn build_gate_result_pass() {
        let result = build_gate_result(GateId::G1Lint, true, None, 5.0, None);
        assert!(result.passed);
        assert!(result.summary.contains("passed"));
        assert!(result.replay_command.is_none());
        assert!(result.within_budget);
    }

    #[test]
    fn build_gate_result_fail() {
        let result = build_gate_result(
            GateId::G2UnitTests,
            false,
            Some(String::from("test_solve failed")),
            10.0,
            None,
        );
        assert!(!result.passed);
        assert!(result.summary.contains("FAILED"));
        assert!(result.replay_command.is_some());
    }

    #[test]
    fn build_pipeline_report_aggregates() {
        let results = vec![
            build_gate_result(GateId::G1Lint, true, None, 5.0, None),
            build_gate_result(GateId::G2UnitTests, true, None, 30.0, None),
            build_gate_result(
                GateId::G3Conformance,
                false,
                Some(String::from("oracle mismatch")),
                15.0,
                None,
            ),
        ];
        let report = build_pipeline_report(&results);
        assert!(!report.all_passed);
        assert_eq!(report.gates_passed, 2);
        assert_eq!(report.gates_failed, 1);
        assert!((report.total_elapsed_seconds - 50.0).abs() < 0.01);
    }

    #[test]
    fn gate_time_budgets_are_reasonable() {
        for gate in &GateId::ALL {
            let budget = gate.time_budget_seconds();
            assert!(budget >= 60, "{gate} budget too low: {budget}s");
            assert!(budget <= 900, "{gate} budget too high: {budget}s");
        }
    }

    #[test]
    fn all_gates_have_commands() {
        for gate in &GateId::ALL {
            assert!(!gate.commands().is_empty(), "{gate} has no commands");
        }
    }
}
