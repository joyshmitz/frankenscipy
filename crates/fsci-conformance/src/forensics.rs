#![forbid(unsafe_code)]
//! bd-3jh.21: Failure Forensics UX + Artifact Index
//!
//! Human-first failure diagnostics with deterministic artifact indexing.
//! Provides structured failure summaries, artifact indices, replay pointers,
//! and numerical diff views for conformance test failures.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

// ─── Failure Summary ──────────────────────────────────────────────────

/// Structured summary of a single test failure, designed for human readability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FailureSummary {
    /// Test identifier (e.g., "P2C-002/linalg_solve_well_conditioned").
    pub test_id: String,
    /// Failure category for grouping/filtering.
    pub category: FailureCategory,
    /// One-paragraph plain-English explanation of what failed.
    pub description: String,
    /// Path to the artifact bundle for this failure.
    pub artifact_path: PathBuf,
    /// Copy-paste command to reproduce this failure.
    pub replay_cmd: String,
    /// Numerical diff details, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numeric_diff: Option<NumericDiffView>,
    /// Timestamp of when the failure was recorded (ms since epoch).
    pub timestamp_ms: u128,
}

/// Categories of failure for filtering and dashboarding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FailureCategory {
    /// Numerical result exceeds tolerance.
    ToleranceExceeded,
    /// Oracle (SciPy) missing or unavailable.
    OracleMissing,
    /// Error type mismatch (expected error X, got error Y or success).
    ErrorMismatch,
    /// Shape or dimension mismatch.
    ShapeMismatch,
    /// Non-finite value detected (NaN, Inf).
    NonFiniteOutput,
    /// Convergence failure.
    ConvergenceFailure,
    /// Other/unknown failure mode.
    Other,
}

impl std::fmt::Display for FailureCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ToleranceExceeded => write!(f, "tolerance exceeded"),
            Self::OracleMissing => write!(f, "oracle missing"),
            Self::ErrorMismatch => write!(f, "error mismatch"),
            Self::ShapeMismatch => write!(f, "shape mismatch"),
            Self::NonFiniteOutput => write!(f, "non-finite output"),
            Self::ConvergenceFailure => write!(f, "convergence failure"),
            Self::Other => write!(f, "other"),
        }
    }
}

// ─── Numeric Diff View ────────────────────────────────────────────────

/// Side-by-side comparison of expected vs actual for a single numeric value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumericDiffEntry {
    /// Label for this comparison (e.g., "x[0]", "det", "residual_norm").
    pub label: String,
    /// Expected value (from oracle or fixture).
    pub expected: f64,
    /// Actual value (from Rust computation).
    pub actual: f64,
    /// Absolute difference: |actual - expected|.
    pub abs_diff: f64,
    /// Relative difference: |actual - expected| / max(|expected|, 1e-15).
    pub rel_diff: f64,
    /// Tolerance that was applied.
    pub tolerance: f64,
    /// Whether this specific comparison passed.
    pub pass: bool,
}

/// Aggregated numerical diff view for a test case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumericDiffView {
    /// Individual comparisons.
    pub entries: Vec<NumericDiffEntry>,
    /// Maximum absolute difference across all entries.
    pub max_abs_diff: f64,
    /// Maximum relative difference across all entries.
    pub max_rel_diff: f64,
    /// Tolerance used for the overall comparison.
    pub tolerance: f64,
    /// Number of entries that failed.
    pub failed_count: usize,
}

impl NumericDiffView {
    /// Create a diff view from parallel slices of expected and actual values.
    pub fn from_slices(
        expected: &[f64],
        actual: &[f64],
        tolerance: f64,
        label_prefix: &str,
    ) -> Self {
        let entries: Vec<NumericDiffEntry> = expected
            .iter()
            .zip(actual)
            .enumerate()
            .map(|(i, (&exp, &act))| {
                let abs_diff = (act - exp).abs();
                let denom = exp.abs().max(1e-15);
                let rel_diff = abs_diff / denom;
                let pass = abs_diff <= tolerance;
                NumericDiffEntry {
                    label: format!("{label_prefix}[{i}]"),
                    expected: exp,
                    actual: act,
                    abs_diff,
                    rel_diff,
                    tolerance,
                    pass,
                }
            })
            .collect();

        let max_abs_diff = entries
            .iter()
            .map(|e| e.abs_diff)
            .fold(0.0_f64, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        let max_rel_diff = entries
            .iter()
            .map(|e| e.rel_diff)
            .fold(0.0_f64, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        let failed_count = entries.iter().filter(|e| !e.pass).count();

        Self {
            entries,
            max_abs_diff,
            max_rel_diff,
            tolerance,
            failed_count,
        }
    }

    /// Generate a human-readable summary string.
    pub fn human_summary(&self, operation: &str) -> String {
        if self.failed_count == 0 {
            return format!(
                "{operation} results match expected within tolerance {:.1e} (max abs diff: {:.2e})",
                self.tolerance, self.max_abs_diff
            );
        }
        let total = self.entries.len();
        format!(
            "{operation} returned results that differ from expected by up to {:.2e}, \
             exceeding the tolerance of {:.1e} ({} of {} values failed)",
            self.max_abs_diff, self.tolerance, self.failed_count, total
        )
    }
}

// ─── Artifact Index ───────────────────────────────────────────────────

/// A single entry in the artifact index: maps a test ID to its artifact bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactIndexEntry {
    /// Test identifier.
    pub test_id: String,
    /// Packet this test belongs to.
    pub packet_id: String,
    /// Path to the artifact bundle directory.
    pub artifact_dir: PathBuf,
    /// Specific files within the bundle.
    pub files: Vec<ArtifactFile>,
    /// Copy-paste replay command.
    pub replay_cmd: String,
    /// Whether this test passed or failed.
    pub passed: bool,
    /// Failure category, if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_category: Option<FailureCategory>,
}

/// A single file within an artifact bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactFile {
    /// Relative path within the artifact directory.
    pub relative_path: PathBuf,
    /// Purpose of this file.
    pub purpose: String,
    /// blake3 hash of file contents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blake3_hash: Option<String>,
    /// File size in bytes.
    pub size_bytes: u64,
}

/// The complete artifact index for a test run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactIndex {
    /// Run identifier.
    pub run_id: String,
    /// Timestamp of the run.
    pub generated_at_ms: u128,
    /// All entries, keyed by test_id for quick lookup.
    pub entries: BTreeMap<String, ArtifactIndexEntry>,
    /// Summary statistics.
    pub summary: ArtifactIndexSummary,
}

/// Summary statistics for an artifact index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactIndexSummary {
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub total_artifacts: usize,
    pub total_bytes: u64,
    /// Breakdown by failure category.
    pub failures_by_category: BTreeMap<String, usize>,
}

// ─── Replay Pointer ───────────────────────────────────────────────────

/// Build a replay command for a specific test.
pub fn build_replay_cmd(packet_id: &str, test_name: &str) -> String {
    format!(
        "cargo test -p fsci-conformance --test {test_file} -- {test_name} --nocapture --exact",
        test_file = test_file_for_packet(packet_id),
    )
}

/// Map packet ID to the conformance test file name.
fn test_file_for_packet(packet_id: &str) -> &str {
    match packet_id {
        "P2C-001" | "FSCI-P2C-001" => "e2e_ivp",
        "P2C-002" | "FSCI-P2C-002" => "e2e_linalg",
        "P2C-003" | "FSCI-P2C-003" => "e2e_optimize",
        "P2C-004" | "FSCI-P2C-004" => "e2e_sparse",
        "P2C-005" | "FSCI-P2C-005" => "e2e_fft",
        "P2C-006" | "FSCI-P2C-006" => "e2e_special",
        "P2C-007" | "FSCI-P2C-007" => "e2e_orchestrator",
        "P2C-008" | "FSCI-P2C-008" => "e2e_casp",
        _ => "golden_journeys",
    }
}

// ─── Failure Summary Builder ──────────────────────────────────────────

/// Parameters for building a failure summary.
pub struct FailureSummaryParams<'a> {
    pub test_id: &'a str,
    pub packet_id: &'a str,
    pub operation: &'a str,
    pub passed: bool,
    pub max_diff: Option<f64>,
    pub tolerance: Option<f64>,
    pub message: &'a str,
    pub artifact_dir: &'a Path,
}

/// Build a human-friendly failure summary from a differential case result.
pub fn summarize_failure(params: &FailureSummaryParams<'_>) -> FailureSummary {
    let FailureSummaryParams {
        test_id,
        packet_id,
        operation,
        passed,
        max_diff,
        tolerance,
        message,
        artifact_dir,
    } = params;
    let category = classify_failure(message, *max_diff, *tolerance);
    let description = build_human_description(operation, &category, *max_diff, *tolerance, message);
    let replay_cmd = build_replay_cmd(packet_id, test_id);
    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    let numeric_diff = match (*max_diff, *tolerance) {
        (Some(diff), Some(tol)) if !passed => Some(NumericDiffView {
            entries: vec![NumericDiffEntry {
                label: String::from("max_diff"),
                expected: 0.0,
                actual: diff,
                abs_diff: diff,
                rel_diff: diff,
                tolerance: tol,
                pass: false,
            }],
            max_abs_diff: diff,
            max_rel_diff: diff,
            tolerance: tol,
            failed_count: 1,
        }),
        _ => None,
    };

    FailureSummary {
        test_id: test_id.to_string(),
        category,
        description,
        artifact_path: artifact_dir.to_path_buf(),
        replay_cmd,
        numeric_diff,
        timestamp_ms,
    }
}

/// Classify a failure message into a category.
fn classify_failure(
    message: &str,
    max_diff: Option<f64>,
    tolerance: Option<f64>,
) -> FailureCategory {
    let msg = message.to_lowercase();
    if msg.contains("oracle") && (msg.contains("missing") || msg.contains("unavailable")) {
        return FailureCategory::OracleMissing;
    }
    if msg.contains("nan") || msg.contains("inf") || msg.contains("non-finite") {
        return FailureCategory::NonFiniteOutput;
    }
    if msg.contains("shape") || msg.contains("dimension") {
        return FailureCategory::ShapeMismatch;
    }
    if msg.contains("converge") || msg.contains("diverge") {
        return FailureCategory::ConvergenceFailure;
    }
    if msg.contains("error") && (msg.contains("expected") || msg.contains("mismatch")) {
        return FailureCategory::ErrorMismatch;
    }
    if let (Some(diff), Some(tol)) = (max_diff, tolerance)
        && diff > tol
    {
        return FailureCategory::ToleranceExceeded;
    }
    FailureCategory::Other
}

/// Build a plain-English description avoiding jargon.
fn build_human_description(
    operation: &str,
    category: &FailureCategory,
    max_diff: Option<f64>,
    tolerance: Option<f64>,
    original_message: &str,
) -> String {
    match category {
        FailureCategory::ToleranceExceeded => {
            let diff = max_diff.unwrap_or(0.0);
            let tol = tolerance.unwrap_or(0.0);
            format!(
                "{operation}() returned a result that differs from the expected value by {diff:.2e}, \
                 exceeding the allowed tolerance of {tol:.1e}."
            )
        }
        FailureCategory::OracleMissing => {
            format!(
                "The SciPy reference oracle was not available to verify {operation}(). \
                 Install SciPy in .venv-py314 to enable oracle comparison."
            )
        }
        FailureCategory::ErrorMismatch => {
            format!("{operation}() produced a different error than expected. {original_message}")
        }
        FailureCategory::ShapeMismatch => {
            format!(
                "{operation}() returned a result with unexpected dimensions. {original_message}"
            )
        }
        FailureCategory::NonFiniteOutput => {
            format!(
                "{operation}() produced NaN or Inf values in its output, \
                 indicating a numerical instability issue."
            )
        }
        FailureCategory::ConvergenceFailure => {
            format!(
                "{operation}() failed to converge within the allowed iterations. \
                 {original_message}"
            )
        }
        FailureCategory::Other => {
            format!("{operation}(): {original_message}")
        }
    }
}

// ─── Artifact Index Builder ───────────────────────────────────────────

/// Build an artifact index by scanning the artifact root directory.
pub fn build_artifact_index(
    artifact_root: &Path,
    run_id: &str,
) -> Result<ArtifactIndex, std::io::Error> {
    let mut entries = BTreeMap::new();
    let mut total_artifacts = 0usize;
    let mut total_bytes = 0u64;

    if artifact_root.exists() {
        scan_artifact_dir(
            artifact_root,
            artifact_root,
            &mut entries,
            &mut total_artifacts,
            &mut total_bytes,
        )?;
    }

    let passed = entries.values().filter(|e| e.passed).count();
    let failed = entries.values().filter(|e| !e.passed).count();

    let mut failures_by_category: BTreeMap<String, usize> = BTreeMap::new();
    for entry in entries.values() {
        if let Some(cat) = &entry.failure_category {
            *failures_by_category.entry(cat.to_string()).or_default() += 1;
        }
    }

    let generated_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    Ok(ArtifactIndex {
        run_id: run_id.to_string(),
        generated_at_ms,
        entries,
        summary: ArtifactIndexSummary {
            total_tests: passed + failed,
            passed,
            failed,
            total_artifacts,
            total_bytes,
            failures_by_category,
        },
    })
}

/// Recursively scan an artifact directory structure.
fn scan_artifact_dir(
    root: &Path,
    dir: &Path,
    entries: &mut BTreeMap<String, ArtifactIndexEntry>,
    total_artifacts: &mut usize,
    total_bytes: &mut u64,
) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            scan_artifact_dir(root, &path, entries, total_artifacts, total_bytes)?;
        } else if path.extension().is_some_and(|ext| ext == "json") {
            let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            *total_artifacts += 1;
            *total_bytes += size;

            // Extract packet ID from path structure: artifacts/{packet}/{stage}/{file}
            if let Some(packet_id) = extract_packet_from_path(&rel) {
                let test_id = rel
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                let purpose = infer_artifact_purpose(&rel);
                let hash = fs::read(&path)
                    .ok()
                    .map(|data| blake3::hash(&data).to_hex().to_string());

                let entry_key = format!("{packet_id}/{test_id}");
                let idx_entry =
                    entries
                        .entry(entry_key.clone())
                        .or_insert_with(|| ArtifactIndexEntry {
                            test_id: test_id.clone(),
                            packet_id: packet_id.clone(),
                            artifact_dir: path.parent().unwrap_or(dir).to_path_buf(),
                            files: Vec::new(),
                            replay_cmd: build_replay_cmd(&packet_id, &test_id),
                            passed: true,
                            failure_category: None,
                        });

                idx_entry.files.push(ArtifactFile {
                    relative_path: rel,
                    purpose,
                    blake3_hash: hash,
                    size_bytes: size,
                });
            }
        }
    }
    Ok(())
}

/// Extract packet ID from a relative path like "P2C-002/e2e/scenario.json".
fn extract_packet_from_path(rel: &Path) -> Option<String> {
    rel.components().next().and_then(|c| {
        let s = c.as_os_str().to_string_lossy().to_string();
        if s.starts_with("P2C-") || s.starts_with("FSCI-P2C-") {
            Some(s)
        } else {
            None
        }
    })
}

/// Infer the purpose of an artifact file from its path.
fn infer_artifact_purpose(rel: &Path) -> String {
    let path_str = rel.to_string_lossy().to_lowercase();
    if path_str.contains("raptorq") || path_str.contains("sidecar") {
        String::from("RaptorQ durability sidecar")
    } else if path_str.contains("e2e") {
        String::from("E2E scenario result")
    } else if path_str.contains("evidence") || path_str.contains("proof") {
        String::from("Evidence/proof artifact")
    } else if path_str.contains("diff") || path_str.contains("oracle") {
        String::from("Differential oracle comparison")
    } else if path_str.contains("perf") || path_str.contains("bench") {
        String::from("Performance benchmark")
    } else if path_str.contains("golden") {
        String::from("Golden journey snapshot")
    } else {
        String::from("Test artifact")
    }
}

/// Write an artifact index to a JSON file.
pub fn write_artifact_index(index: &ArtifactIndex, output: &Path) -> Result<(), std::io::Error> {
    let json = serde_json::to_string_pretty(index)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, json)
}

/// Write a collection of failure summaries to a JSON file.
pub fn write_failure_summaries(
    summaries: &[FailureSummary],
    output: &Path,
) -> Result<(), std::io::Error> {
    let json = serde_json::to_string_pretty(summaries)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_category_classifies_tolerance_exceeded() {
        let cat = classify_failure("value too large", Some(1e-5), Some(1e-8));
        assert_eq!(cat, FailureCategory::ToleranceExceeded);
    }

    #[test]
    fn failure_category_classifies_oracle_missing() {
        let cat = classify_failure("oracle missing: SciPy not found", None, None);
        assert_eq!(cat, FailureCategory::OracleMissing);
    }

    #[test]
    fn failure_category_classifies_nan_output() {
        let cat = classify_failure("output contains NaN values", None, None);
        assert_eq!(cat, FailureCategory::NonFiniteOutput);
    }

    #[test]
    fn failure_category_classifies_convergence() {
        let cat = classify_failure("solver did not converge", None, None);
        assert_eq!(cat, FailureCategory::ConvergenceFailure);
    }

    #[test]
    fn failure_category_classifies_shape_mismatch() {
        let cat = classify_failure("shape mismatch: got [3,4], expected [4,4]", None, None);
        assert_eq!(cat, FailureCategory::ShapeMismatch);
    }

    #[test]
    fn failure_category_classifies_error_mismatch() {
        let cat = classify_failure(
            "error mismatch: expected LinalgError, got success",
            None,
            None,
        );
        assert_eq!(cat, FailureCategory::ErrorMismatch);
    }

    #[test]
    fn failure_category_defaults_to_other() {
        let cat = classify_failure("something went wrong", None, None);
        assert_eq!(cat, FailureCategory::Other);
    }

    #[test]
    fn numeric_diff_view_from_slices_computes_correctly() {
        let expected = [1.0, 2.0, 3.0];
        let actual = [1.0, 2.0001, 3.0];
        let view = NumericDiffView::from_slices(&expected, &actual, 1e-3, "x");

        assert_eq!(view.entries.len(), 3);
        assert_eq!(view.failed_count, 0); // all within 1e-3
        assert!(view.max_abs_diff < 1e-3);

        // Now with tight tolerance
        let tight = NumericDiffView::from_slices(&expected, &actual, 1e-5, "x");
        assert_eq!(tight.failed_count, 1); // x[1] fails
        assert!(!tight.entries[1].pass);
        assert!(tight.entries[0].pass);
        assert!(tight.entries[2].pass);
    }

    #[test]
    fn numeric_diff_view_human_summary_pass() {
        let view = NumericDiffView::from_slices(&[1.0], &[1.0], 1e-8, "x");
        let summary = view.human_summary("solve");
        assert!(summary.contains("match expected within tolerance"));
    }

    #[test]
    fn numeric_diff_view_human_summary_fail() {
        let view = NumericDiffView::from_slices(&[1.0], &[2.0], 1e-8, "x");
        let summary = view.human_summary("solve");
        assert!(summary.contains("differ from expected"));
        assert!(summary.contains("exceeding the tolerance"));
    }

    #[test]
    fn build_replay_cmd_maps_packets() {
        let cmd = build_replay_cmd("P2C-002", "test_solve");
        assert!(cmd.contains("e2e_linalg"));
        assert!(cmd.contains("test_solve"));
        assert!(cmd.contains("--nocapture"));

        let cmd2 = build_replay_cmd("P2C-005", "test_fft");
        assert!(cmd2.contains("e2e_fft"));
    }

    #[test]
    fn summarize_failure_creates_human_description() {
        let summary = summarize_failure(&FailureSummaryParams {
            test_id: "test_solve_3x3",
            packet_id: "P2C-002",
            operation: "solve",
            passed: false,
            max_diff: Some(1.2e-7),
            tolerance: Some(1e-8),
            message: "tolerance exceeded",
            artifact_dir: Path::new("/tmp/artifacts/P2C-002"),
        });
        assert_eq!(summary.category, FailureCategory::ToleranceExceeded);
        assert!(summary.description.contains("1.20e-7"));
        assert!(summary.description.contains("1.0e-8"));
        assert!(summary.numeric_diff.is_some());
    }

    #[test]
    fn artifact_index_builder_handles_empty_dir() {
        let tmp = std::env::temp_dir().join("fsci_forensics_test_empty");
        let _ = fs::create_dir_all(&tmp);
        let index = build_artifact_index(&tmp, "test-run").unwrap();
        assert_eq!(index.summary.total_tests, 0);
        assert_eq!(index.summary.total_artifacts, 0);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn extract_packet_from_path_works() {
        assert_eq!(
            extract_packet_from_path(Path::new("P2C-002/e2e/test.json")),
            Some(String::from("P2C-002"))
        );
        assert_eq!(
            extract_packet_from_path(Path::new("FSCI-P2C-001/evidence/proof.json")),
            Some(String::from("FSCI-P2C-001"))
        );
        assert_eq!(
            extract_packet_from_path(Path::new("golden_journeys/gj_01.json")),
            None
        );
    }

    #[test]
    fn infer_purpose_from_path() {
        assert!(infer_artifact_purpose(Path::new("P2C-002/e2e/test.json")).contains("E2E"));
        assert!(
            infer_artifact_purpose(Path::new("P2C-002/evidence/proof.json")).contains("Evidence")
        );
        assert!(
            infer_artifact_purpose(Path::new("raptorq_proofs/sidecar.json")).contains("RaptorQ")
        );
        assert!(infer_artifact_purpose(Path::new("perf/bench.json")).contains("Performance"));
        assert!(
            infer_artifact_purpose(Path::new("golden_journeys/gj.json")).contains("Golden journey")
        );
    }
}
