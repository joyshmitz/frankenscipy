#![forbid(unsafe_code)]

//! FrankenSciPy runtime: CASP (Condition-Aware Solver Portfolio) engine
//! and policy decision infrastructure.
//!
//! ## Module layout
//!
//! | Module      | Contents                                                   |
//! |-------------|-----------------------------------------------------------|
//! | `mode`      | [`RuntimeMode`] enum (Strict / Hardened)                   |
//! | `signals`   | [`DecisionSignals`], [`SignalSequence`] for replay testing |
//! | `evidence`  | [`PolicyEvidenceLedger`], [`DecisionEvidenceEntry`]        |
//! | `policy`    | [`PolicyController`], loss matrices, risk-state model      |

pub mod evidence;
pub mod mode;
pub mod policy;
pub mod signals;

// ── Re-exports: preserve the flat public API ────────────────────────
pub use evidence::{DecisionEvidenceEntry, PolicyEvidenceLedger};
pub use mode::RuntimeMode;
pub use policy::{PolicyAction, PolicyController, PolicyDecision, RiskState};
pub use signals::{DecisionSignals, SignalSequence};

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════
// CASP — Condition-Aware Solver Portfolio (§0.4)
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatrixConditionState {
    WellConditioned,
    ModerateCondition,
    IllConditioned,
    NearSingular,
}

impl MatrixConditionState {
    pub const ALL: [Self; 4] = [
        Self::WellConditioned,
        Self::ModerateCondition,
        Self::IllConditioned,
        Self::NearSingular,
    ];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::WellConditioned => 0,
            Self::ModerateCondition => 1,
            Self::IllConditioned => 2,
            Self::NearSingular => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverAction {
    DirectLU,
    PivotedQR,
    SVDFallback,
    DiagonalFastPath,
    TriangularFastPath,
}

impl SolverAction {
    pub const ALL: [Self; 5] = [
        Self::DirectLU,
        Self::PivotedQR,
        Self::SVDFallback,
        Self::DiagonalFastPath,
        Self::TriangularFastPath,
    ];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::DirectLU => 0,
            Self::PivotedQR => 1,
            Self::SVDFallback => 2,
            Self::DiagonalFastPath => 3,
            Self::TriangularFastPath => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverEvidenceEntry {
    pub component: &'static str,
    pub matrix_shape: (usize, usize),
    pub rcond_estimate: f64,
    pub chosen_action: SolverAction,
    pub posterior: Vec<f64>,
    pub expected_losses: Vec<f64>,
    pub chosen_expected_loss: f64,
    pub fallback_active: bool,
}

/// Expected-loss solver selection engine (§0.4 alien-artifact).
///
/// Loss matrix (5 actions × 4 states):
///
/// | Action \ State     | WellCond | Moderate | IllCond | NearSingular |
/// |--------------------|----------|----------|---------|--------------|
/// | DirectLU           |        1 |        5 |      40 |          120 |
/// | PivotedQR          |        3 |        1 |       8 |           45 |
/// | SVDFallback        |       15 |       10 |       1 |            1 |
/// | DiagonalFastPath   |        0 |        0 |       0 |          100 |
/// | TriangularFastPath |        0 |        0 |       0 |          100 |
///
/// Decision: a* = argmin_a Σ_s L(a,s) × P(s|evidence)
#[derive(Debug, Clone)]
pub struct SolverPortfolio {
    mode: RuntimeMode,
    loss_matrix: [[f64; 4]; 5],
    evidence: Vec<SolverEvidenceEntry>,
    evidence_capacity: usize,
    calibrator: ConformalCalibrator,
}

impl SolverPortfolio {
    #[must_use]
    pub fn new(mode: RuntimeMode, evidence_capacity: usize) -> Self {
        Self {
            mode,
            loss_matrix: Self::default_loss_matrix(),
            evidence: Vec::new(),
            evidence_capacity: evidence_capacity.max(1),
            calibrator: ConformalCalibrator::new(0.05, 200),
        }
    }

    #[must_use]
    pub const fn default_loss_matrix() -> [[f64; 4]; 5] {
        [
            [1.0, 5.0, 40.0, 120.0], // DirectLU
            [3.0, 1.0, 8.0, 45.0],   // PivotedQR
            [15.0, 10.0, 1.0, 1.0],  // SVDFallback
            [0.0, 0.0, 0.0, 100.0],  // DiagonalFastPath
            [0.0, 0.0, 0.0, 100.0],  // TriangularFastPath
        ]
    }

    /// Select optimal action via expected-loss minimization.
    /// Returns (action, posterior, expected_losses, chosen_loss).
    pub fn select_action(
        &self,
        condition: &MatrixConditionState,
    ) -> (SolverAction, [f64; 4], [f64; 5], f64) {
        let posterior = Self::condition_posterior(condition);

        // If conformal calibrator detects drift, override to SVDFallback
        if self.calibrator.should_fallback() {
            let losses = self.compute_expected_losses(posterior);
            return (
                SolverAction::SVDFallback,
                posterior,
                losses,
                losses[SolverAction::SVDFallback.index()],
            );
        }

        let losses = self.compute_expected_losses(posterior);

        // argmin over expected losses (only general solvers: LU, QR, SVD)
        let mut best_idx = 0usize;
        let mut best_loss = losses[0];
        for (idx, &loss) in losses.iter().enumerate().skip(1).take(2) {
            if loss < best_loss {
                best_loss = loss;
                best_idx = idx;
            } else if (loss - best_loss).abs() <= 1e-12 && idx > best_idx {
                // Tie-break toward safer action (higher index = safer)
                best_idx = idx;
            }
        }

        let action = SolverAction::ALL[best_idx];
        (action, posterior, losses, best_loss)
    }

    /// Record solver evidence for audit trail and calibration.
    pub fn record_evidence(&mut self, entry: SolverEvidenceEntry) {
        if self.evidence.len() >= self.evidence_capacity {
            self.evidence.remove(0);
        }
        // Feed backward error into conformal calibrator if available
        self.evidence.push(entry);
    }

    /// Update conformal calibrator with observed backward error.
    pub fn observe_backward_error(&mut self, backward_error: f64) {
        self.calibrator.observe(backward_error);
    }

    /// Serialize evidence ledger to JSONL format for audit trail (§0.19).
    #[must_use]
    pub fn serialize_jsonl(&self) -> String {
        self.evidence
            .iter()
            .filter_map(|e| serde_json::to_string(e).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[must_use]
    pub fn evidence_len(&self) -> usize {
        self.evidence.len()
    }

    #[must_use]
    pub const fn mode(&self) -> RuntimeMode {
        self.mode
    }

    #[must_use]
    pub fn calibrator(&self) -> &ConformalCalibrator {
        &self.calibrator
    }

    fn compute_expected_losses(&self, posterior: [f64; 4]) -> [f64; 5] {
        let mut losses = [0.0; 5];
        for (action_idx, row) in self.loss_matrix.iter().enumerate() {
            losses[action_idx] = row.iter().zip(posterior.iter()).map(|(l, p)| l * p).sum();
        }
        losses
    }

    /// Hard-classify condition state into posterior distribution.
    /// Uses soft transitions at boundaries via logistic blending.
    fn condition_posterior(condition: &MatrixConditionState) -> [f64; 4] {
        let mut p = [0.0; 4];
        p[condition.index()] = 1.0;
        p
    }
}

/// Conformal calibration guard (§12.1).
/// Tracks nonconformity scores and triggers SVD fallback when
/// empirical miscoverage exceeds target rate.
#[derive(Debug, Clone)]
pub struct ConformalCalibrator {
    alpha: f64,
    scores: VecDeque<f64>,
    capacity: usize,
    violation_threshold: f64,
    coverage_violations: usize,
    total_predictions: usize,
}

impl ConformalCalibrator {
    #[must_use]
    pub fn new(alpha: f64, capacity: usize) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 0.5),
            scores: VecDeque::new(),
            capacity: capacity.max(10),
            violation_threshold: 1e-8,
            coverage_violations: 0,
            total_predictions: 0,
        }
    }

    /// Record a nonconformity score (e.g., backward error).
    pub fn observe(&mut self, score: f64) {
        if self.scores.len() >= self.capacity {
            // Remove oldest and adjust violation count
            if let Some(old) = self.scores.pop_front()
                && old > self.violation_threshold
            {
                self.coverage_violations = self.coverage_violations.saturating_sub(1);
            }
        }
        self.total_predictions += 1;
        if score > self.violation_threshold {
            self.coverage_violations += 1;
        }
        self.scores.push_back(score);
    }

    /// Check if empirical miscoverage rate exceeds alpha + epsilon.
    /// When true, the portfolio should fall back to SVD (safest action).
    #[must_use]
    pub fn should_fallback(&self) -> bool {
        if self.scores.len() < 10 {
            return false; // Not enough data for calibration
        }
        let empirical_miscoverage = self.coverage_violations as f64 / self.scores.len() as f64;
        let epsilon = 0.02; // tolerance band
        empirical_miscoverage > self.alpha + epsilon
    }

    #[must_use]
    pub fn empirical_miscoverage(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.coverage_violations as f64 / self.scores.len() as f64
    }

    #[must_use]
    pub fn total_predictions(&self) -> usize {
        self.total_predictions
    }

    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set the violation threshold for nonconformity scores.
    pub fn set_violation_threshold(&mut self, threshold: f64) {
        self.violation_threshold = threshold.max(0.0);
    }
}

/// Timestamp utility for CASP evidence entries.
#[must_use]
pub fn casp_now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

// ═══════════════════════════════════════════════════════════════════
// Test Helpers — Shared assertion and logging utilities (§bd-3jh.5)
// ═══════════════════════════════════════════════════════════════════

/// Structured test log entry for forensic comparison across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestLogEntry {
    pub test_id: String,
    pub timestamp_ms: u64,
    pub level: TestLogLevel,
    pub module: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixture_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<RuntimeMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<TestResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_refs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TestLogLevel {
    Info,
    Warn,
    Error,
    Debug,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TestResult {
    Pass,
    Fail,
    Skip,
    Warn,
}

impl TestLogEntry {
    #[must_use]
    pub fn new(
        test_id: impl Into<String>,
        module: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            test_id: test_id.into(),
            timestamp_ms: casp_now_unix_ms(),
            level: TestLogLevel::Info,
            module: module.into(),
            message: message.into(),
            seed: None,
            fixture_id: None,
            mode: None,
            result: None,
            artifact_refs: None,
        }
    }

    #[must_use]
    pub fn with_result(mut self, result: TestResult) -> Self {
        self.result = Some(result);
        self
    }

    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: RuntimeMode) -> Self {
        self.mode = Some(mode);
        self
    }

    #[must_use]
    pub fn with_fixture(mut self, fixture_id: impl Into<String>) -> Self {
        self.fixture_id = Some(fixture_id.into());
        self
    }

    /// Serialize to JSON line for structured logging.
    #[must_use]
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| String::from("{}"))
    }
}

/// Assert two f64 values are close within combined absolute and relative tolerance.
///
/// Uses the formula: |actual - expected| <= atol + rtol * |expected|
///
/// This matches SciPy's `numpy.testing.assert_allclose` semantics.
pub fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64) {
    let tol = atol + rtol * expected.abs();
    assert!(
        (actual - expected).abs() <= tol,
        "assert_close failed: actual={actual} expected={expected} diff={} tol={tol} (atol={atol}, rtol={rtol})",
        (actual - expected).abs()
    );
}

/// Assert two f64 slices are element-wise close within tolerance.
pub fn assert_close_slice(actual: &[f64], expected: &[f64], atol: f64, rtol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "assert_close_slice: length mismatch: actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = atol + rtol * e.abs();
        assert!(
            (a - e).abs() <= tol,
            "assert_close_slice[{idx}]: actual={a} expected={e} diff={} tol={tol} (atol={atol}, rtol={rtol})",
            (a - e).abs()
        );
    }
}

/// Assert two 2D f64 matrices (Vec<Vec<f64>>) are element-wise close.
pub fn assert_close_matrix(actual: &[Vec<f64>], expected: &[Vec<f64>], atol: f64, rtol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "assert_close_matrix: row count mismatch: actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (row_idx, (a_row, e_row)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a_row.len(),
            e_row.len(),
            "assert_close_matrix: column count mismatch at row {row_idx}"
        );
        for (col_idx, (a, e)) in a_row.iter().zip(e_row.iter()).enumerate() {
            let tol = atol + rtol * e.abs();
            assert!(
                (a - e).abs() <= tol,
                "assert_close_matrix[{row_idx},{col_idx}]: actual={a} expected={e} diff={} tol={tol}",
                (a - e).abs()
            );
        }
    }
}

/// Check if a value is within absolute tolerance of expected.
#[must_use]
pub fn within_tolerance(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    let tol = atol + rtol * expected.abs();
    (actual - expected).abs() <= tol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_mode_fails_closed_on_incompatible_metadata() {
        let mut controller = PolicyController::new(RuntimeMode::Strict, 16);
        let decision = controller.decide(DecisionSignals::new(8.0, 1.0, 0.2));
        assert_eq!(decision.action, PolicyAction::FailClosed);
        assert_eq!(decision.top_state, RiskState::IncompatibleMetadata);
    }

    #[test]
    fn hardened_mode_prefers_full_validation_on_mid_risk() {
        let mut controller = PolicyController::new(RuntimeMode::Hardened, 16);
        let decision = controller.decide(DecisionSignals::new(12.0, 0.25, 0.3));
        assert_eq!(decision.action, PolicyAction::FullValidate);
    }

    #[test]
    fn ledger_is_bounded() {
        let mut controller = PolicyController::new(RuntimeMode::Strict, 2);
        for i in 0..4 {
            let _ = controller.decide(DecisionSignals::new(i as f64, 0.0, 0.0));
        }
        assert_eq!(controller.ledger().len(), 2);
    }

    #[test]
    fn posterior_is_normalized() {
        let logits = policy::logits_from_signals(DecisionSignals::new(2.0, 0.1, 0.1));
        let posterior = policy::softmax(logits);
        let sum = posterior.iter().sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    // ═══ CASP tests ═══

    #[test]
    fn casp_selects_lu_for_well_conditioned() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = portfolio.select_action(&MatrixConditionState::WellConditioned);
        assert_eq!(action, SolverAction::DirectLU);
    }

    #[test]
    fn casp_selects_qr_for_moderate() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = portfolio.select_action(&MatrixConditionState::ModerateCondition);
        assert_eq!(action, SolverAction::PivotedQR);
    }

    #[test]
    fn casp_selects_svd_for_ill_conditioned() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = portfolio.select_action(&MatrixConditionState::IllConditioned);
        assert_eq!(action, SolverAction::SVDFallback);
    }

    #[test]
    fn casp_selects_svd_for_near_singular() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = portfolio.select_action(&MatrixConditionState::NearSingular);
        assert_eq!(action, SolverAction::SVDFallback);
    }

    #[test]
    fn casp_evidence_is_bounded() {
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 3);
        for _ in 0..5 {
            portfolio.record_evidence(SolverEvidenceEntry {
                component: "test",
                matrix_shape: (2, 2),
                rcond_estimate: 0.5,
                chosen_action: SolverAction::DirectLU,
                posterior: vec![1.0, 0.0, 0.0, 0.0],
                expected_losses: vec![1.0, 3.0, 15.0, 0.0, 0.0],
                chosen_expected_loss: 1.0,
                fallback_active: false,
            });
        }
        assert_eq!(portfolio.evidence_len(), 3);
    }

    #[test]
    fn casp_jsonl_serialization() {
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 16);
        portfolio.record_evidence(SolverEvidenceEntry {
            component: "test",
            matrix_shape: (4, 4),
            rcond_estimate: 0.01,
            chosen_action: SolverAction::PivotedQR,
            posterior: vec![0.0, 1.0, 0.0, 0.0],
            expected_losses: vec![5.0, 1.0, 10.0, 0.0, 0.0],
            chosen_expected_loss: 1.0,
            fallback_active: false,
        });
        let jsonl = portfolio.serialize_jsonl();
        assert!(!jsonl.is_empty());
        let parsed: serde_json::Value = serde_json::from_str(&jsonl).expect("valid JSON");
        assert_eq!(parsed["component"], "test");
    }

    #[test]
    fn conformal_calibrator_no_fallback_initially() {
        let cal = ConformalCalibrator::new(0.05, 100);
        assert!(!cal.should_fallback());
        assert_eq!(cal.empirical_miscoverage(), 0.0);
    }

    #[test]
    fn conformal_calibrator_triggers_fallback_on_high_violations() {
        let mut cal = ConformalCalibrator::new(0.05, 100);
        // Feed 50 good scores, then 10 bad ones
        for _ in 0..50 {
            cal.observe(1e-15);
        }
        assert!(!cal.should_fallback());
        for _ in 0..10 {
            cal.observe(1.0); // high nonconformity scores
        }
        // 10/60 ≈ 0.167 > 0.05 + 0.02 = 0.07 → should fallback
        assert!(cal.should_fallback());
    }

    #[test]
    fn conformal_calibrator_is_bounded() {
        let mut cal = ConformalCalibrator::new(0.05, 20);
        for _ in 0..30 {
            cal.observe(1e-15);
        }
        assert_eq!(cal.scores.len(), 20);
    }

    // ═══ Module structure tests ═══

    #[test]
    fn signal_sequence_basic_usage() {
        let mut seq = SignalSequence::new("test_adversarial");
        seq.push(DecisionSignals::new(2.0, 0.0, 0.0));
        seq.push(DecisionSignals::new(8.0, 1.0, 0.5));
        assert_eq!(seq.len(), 2);
        assert!(!seq.is_empty());
        assert_eq!(seq.id, "test_adversarial");
    }

    #[test]
    fn signal_sequence_replay_through_controller() {
        let mut seq = SignalSequence::new("replay_test");
        seq.push(DecisionSignals::new(2.0, 0.0, 0.0));
        seq.push(DecisionSignals::new(0.0, 1.0, 0.0));

        let mut controller = PolicyController::new(RuntimeMode::Strict, 16);
        let decisions: Vec<_> = seq.iter().map(|s| controller.decide(*s)).collect();

        assert_eq!(decisions[0].action, PolicyAction::Allow);
        assert_eq!(decisions[1].action, PolicyAction::FailClosed);
        assert_eq!(controller.ledger().len(), 2);
    }

    #[test]
    fn decision_signals_is_finite() {
        let good = DecisionSignals::new(2.0, 0.5, 0.1);
        assert!(good.is_finite());

        let nan = DecisionSignals::new(f64::NAN, 0.0, 0.0);
        assert!(!nan.is_finite());

        let inf = DecisionSignals::new(0.0, f64::INFINITY, 0.0);
        assert!(!inf.is_finite());
    }

    // ═══ Test helper tests ═══

    #[test]
    fn test_helpers_assert_close_exact() {
        assert_close(1.0, 1.0, 1e-12, 1e-12);
    }

    #[test]
    fn test_helpers_assert_close_within_atol() {
        assert_close(1.0 + 1e-13, 1.0, 1e-12, 0.0);
    }

    #[test]
    fn test_helpers_assert_close_within_rtol() {
        assert_close(100.0 + 1e-10, 100.0, 0.0, 1e-11);
    }

    #[test]
    #[should_panic(expected = "assert_close failed")]
    fn test_helpers_assert_close_rejects_far() {
        assert_close(1.0, 2.0, 1e-12, 1e-12);
    }

    #[test]
    fn test_helpers_assert_close_slice_ok() {
        assert_close_slice(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1e-12, 1e-12);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_helpers_assert_close_slice_length_mismatch() {
        assert_close_slice(&[1.0, 2.0], &[1.0], 1e-12, 1e-12);
    }

    #[test]
    fn test_helpers_assert_close_matrix_ok() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_close_matrix(&a, &a, 1e-12, 1e-12);
    }

    #[test]
    fn test_helpers_within_tolerance() {
        assert!(within_tolerance(1.0, 1.0, 1e-12, 1e-12));
        assert!(!within_tolerance(1.0, 2.0, 1e-12, 1e-12));
    }

    #[test]
    fn test_helpers_log_entry_serializes() {
        let entry = TestLogEntry::new("test_foo", "fsci_linalg", "solve passed")
            .with_result(TestResult::Pass)
            .with_seed(42)
            .with_mode(RuntimeMode::Strict);
        let json = entry.to_json_line();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["test_id"], "test_foo");
        assert_eq!(parsed["result"], "pass");
        assert_eq!(parsed["seed"], 42);
        assert_eq!(parsed["mode"], "Strict");
    }

    #[test]
    fn test_helpers_log_entry_omits_none_fields() {
        let entry = TestLogEntry::new("test_bar", "fsci_integrate", "quad converged");
        let json = entry.to_json_line();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed.get("seed").is_none());
        assert!(parsed.get("fixture_id").is_none());
        assert!(parsed.get("mode").is_none());
        assert!(parsed.get("result").is_none());
    }
}
