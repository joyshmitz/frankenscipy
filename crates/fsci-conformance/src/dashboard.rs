use crate::{
    DecodeProofArtifact, DifferentialCaseResult, HarnessConfig, HarnessError, PacketReport,
    PacketSummary, RaptorQSidecar, packet_summary,
};
use serde_json::from_str;
use std::cmp::Ordering;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashboardPanel {
    Packets,
    Cases,
    Summary,
}

impl DashboardPanel {
    fn next(self) -> Self {
        match self {
            Self::Packets => Self::Cases,
            Self::Cases => Self::Summary,
            Self::Summary => Self::Packets,
        }
    }

    fn prev(self) -> Self {
        match self {
            Self::Packets => Self::Summary,
            Self::Cases => Self::Packets,
            Self::Summary => Self::Cases,
        }
    }

    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            Self::Packets => "PACKETS",
            Self::Cases => "CASES",
            Self::Summary => "SUMMARY",
        }
    }
}

/// How a parity_report.json was produced. Derived from the per-packet
/// artifact set so dashboard consumers can distinguish oracle-backed
/// from self-check reports without having to open each file. Per
/// frankenscipy-7h9o.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportSource {
    /// No parity_report.json present for this packet.
    None,
    /// oracle_capture.json exists alongside the report AND has no sibling
    /// oracle_capture.error.txt — report values were compared against a
    /// scipy-produced reference.
    OracleBacked,
    /// oracle_capture.error.txt is present (scipy probe failed) OR the
    /// oracle_capture.json is missing entirely — report values were
    /// compared against fixture-embedded expected values only. Equivalent
    /// to the self-check lane in ivg5's taxonomy.
    SelfCheck,
}

impl ReportSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::OracleBacked => "oracle",
            Self::SelfCheck => "self-check",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketArtifactPresence {
    pub packet_id: String,
    pub has_report: bool,
    pub has_sidecar: bool,
    pub has_decode_proof: bool,
    pub has_oracle_capture: bool,
    pub has_oracle_error: bool,
    pub sidecar_source_hash: Option<String>,
    pub sidecar_source_symbols: Option<usize>,
    pub sidecar_repair_symbols: Option<usize>,
    pub decode_proof_hash: Option<String>,
    pub decode_proof_reason: Option<String>,
    /// Classification of how the parity_report.json (if any) was produced.
    /// Per frankenscipy-7h9o: previously the dashboard showed pass/fail
    /// counts without distinguishing oracle-verified vs self-check. This
    /// field surfaces that distinction so consumers can tell at a glance.
    pub report_source: ReportSource,
}

impl PacketArtifactPresence {
    #[must_use]
    pub fn new(packet_id: String) -> Self {
        Self {
            packet_id,
            has_report: false,
            has_sidecar: false,
            has_decode_proof: false,
            has_oracle_capture: false,
            has_oracle_error: false,
            sidecar_source_hash: None,
            sidecar_source_symbols: None,
            sidecar_repair_symbols: None,
            decode_proof_hash: None,
            decode_proof_reason: None,
            report_source: ReportSource::None,
        }
    }

    /// Derive the report_source from the currently-populated flags. Must
    /// be called after all has_* fields have been set.
    pub fn classify_report_source(&mut self) {
        self.report_source = if !self.has_report {
            ReportSource::None
        } else if self.has_oracle_capture && !self.has_oracle_error {
            ReportSource::OracleBacked
        } else {
            ReportSource::SelfCheck
        };
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DashboardMetrics {
    pub total_packets: usize,
    pub failed_packets: usize,
    pub total_cases: usize,
    pub failed_cases: usize,
    pub failed_case_pct: f64,
    pub packets_with_sidecar: usize,
    pub packets_with_decode_proof: usize,
    pub packets_with_oracle_capture: usize,
    /// Packets whose parity_report.json was produced via an oracle-backed
    /// comparison path (oracle_capture.json present, no error sidecar).
    /// Per frankenscipy-7h9o: previously the dashboard reported
    /// pass/fail counts uniformly; now consumers can see that a 100%
    /// pass on a packet without an oracle backing is self-check only.
    pub packets_oracle_backed: usize,
    /// Packets whose parity_report.json was produced via self-check
    /// (no oracle, or oracle_capture.error.txt present).
    pub packets_self_check: usize,
}

type DiscoveryResult = (Vec<PacketReport>, Vec<PacketArtifactPresence>, Vec<String>);

#[derive(Debug, Clone, Copy)]
pub struct DashboardCaseListEntry<'a> {
    pub source_index: usize,
    pub case: &'a crate::CaseResult,
    pub max_diff: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct DashboardState {
    reports: Vec<PacketReport>,
    summaries: Vec<PacketSummary>,
    artifact_presence: Vec<PacketArtifactPresence>,
    warnings: Vec<String>,
    packet_filter: Option<String>,
    selected_packet: usize,
    selected_case: usize,
    active_panel: DashboardPanel,
}

impl DashboardState {
    #[must_use]
    pub fn new(
        reports: Vec<PacketReport>,
        artifact_presence: Vec<PacketArtifactPresence>,
        warnings: Vec<String>,
        packet_filter: Option<String>,
    ) -> Self {
        let summaries = reports.iter().map(packet_summary).collect();
        let mut state = Self {
            reports,
            summaries,
            artifact_presence,
            warnings,
            packet_filter,
            selected_packet: 0,
            selected_case: 0,
            active_panel: DashboardPanel::Packets,
        };
        state.clamp_indices();
        state
    }

    pub fn move_up(&mut self) {
        match self.active_panel {
            DashboardPanel::Packets => {
                if self.selected_packet > 0 {
                    self.selected_packet -= 1;
                    self.selected_case = 0;
                }
            }
            DashboardPanel::Cases => {
                if self.selected_case > 0 {
                    self.selected_case -= 1;
                }
            }
            DashboardPanel::Summary => {}
        }
    }

    pub fn move_down(&mut self) {
        match self.active_panel {
            DashboardPanel::Packets => {
                if self.selected_packet + 1 < self.reports.len() {
                    self.selected_packet += 1;
                    self.selected_case = 0;
                }
            }
            DashboardPanel::Cases => {
                if let Some(packet) = self.selected_packet()
                    && self.selected_case + 1 < packet.case_results.len()
                {
                    self.selected_case += 1;
                }
            }
            DashboardPanel::Summary => {}
        }
    }

    pub fn next_panel(&mut self) {
        self.active_panel = self.active_panel.next();
    }

    pub fn prev_panel(&mut self) {
        self.active_panel = self.active_panel.prev();
    }

    #[must_use]
    pub fn active_panel(&self) -> DashboardPanel {
        self.active_panel
    }

    #[must_use]
    pub fn reports(&self) -> &[PacketReport] {
        &self.reports
    }

    #[must_use]
    pub fn summaries(&self) -> &[PacketSummary] {
        &self.summaries
    }

    #[must_use]
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    #[must_use]
    pub fn packet_filter(&self) -> Option<&str> {
        self.packet_filter.as_deref()
    }

    #[must_use]
    pub fn selected_packet_index(&self) -> usize {
        self.selected_packet
    }

    #[must_use]
    pub fn selected_case_index(&self) -> usize {
        self.selected_case
    }

    #[must_use]
    pub fn selected_packet(&self) -> Option<&PacketReport> {
        self.reports.get(self.selected_packet)
    }

    #[must_use]
    pub fn selected_case(&self) -> Option<&crate::CaseResult> {
        self.case_list_entries()
            .get(self.selected_case)
            .map(|entry| entry.case)
    }

    #[must_use]
    pub fn case_list_entries(&self) -> Vec<DashboardCaseListEntry<'_>> {
        let Some(packet) = self.selected_packet() else {
            return Vec::new();
        };

        let mut entries = packet
            .case_results
            .iter()
            .enumerate()
            .map(|(source_index, case)| DashboardCaseListEntry {
                source_index,
                case,
                max_diff: differential_max_diff(packet, case),
            })
            .collect::<Vec<_>>();
        entries.sort_by(compare_case_list_entries);
        entries
    }

    #[must_use]
    pub fn selected_differential_case(&self) -> Option<&DifferentialCaseResult> {
        let packet = self.selected_packet()?;
        let case = self.selected_case()?;
        packet
            .differential_case_results
            .as_ref()?
            .iter()
            .find(|result| result.case_id == case.case_id)
    }

    #[must_use]
    pub fn selected_case_drilldown_lines(&self) -> Vec<String> {
        let Some(case) = self.selected_case() else {
            return vec![String::from("case=<none>")];
        };

        let mut lines = vec![
            format!("case={} passed={}", case.case_id, case.passed),
            format!("assertion={}", case.message),
        ];

        if let Some(diff) = self.selected_differential_case() {
            lines.push(match diff.max_diff {
                Some(max_diff) => format!("max_abs_diff={max_diff:.6e}"),
                None => String::from("max_abs_diff=<not numeric>"),
            });
            if let Some(tolerance) = &diff.tolerance_used {
                lines.push(format!(
                    "tolerance atol={:.6e} rtol={:.6e} mode={}",
                    tolerance.atol, tolerance.rtol, tolerance.comparison_mode
                ));
            } else {
                lines.push(String::from("tolerance=<none>"));
            }
            lines.push(format!("oracle_status={:?}", diff.oracle_status));
        } else {
            lines.push(String::from("differential=<not recorded>"));
        }

        lines
    }

    #[must_use]
    pub fn selected_packet_artifacts(&self) -> Option<&PacketArtifactPresence> {
        let packet = self.selected_packet()?;
        self.artifact_presence
            .iter()
            .find(|presence| presence.packet_id == packet.packet_id)
    }

    #[must_use]
    pub fn metrics(&self) -> DashboardMetrics {
        let total_packets = self.reports.len();
        let failed_packets = self.reports.iter().filter(|r| r.failed_cases > 0).count();
        let total_cases = self
            .reports
            .iter()
            .map(|r| r.case_results.len())
            .sum::<usize>();
        let failed_cases = self.reports.iter().map(|r| r.failed_cases).sum::<usize>();
        let failed_case_pct = if total_cases == 0 {
            0.0
        } else {
            (failed_cases as f64 * 100.0) / total_cases as f64
        };

        let packets_with_sidecar = self
            .artifact_presence
            .iter()
            .filter(|presence| presence.has_sidecar)
            .count();
        let packets_with_decode_proof = self
            .artifact_presence
            .iter()
            .filter(|presence| presence.has_decode_proof)
            .count();
        let packets_with_oracle_capture = self
            .artifact_presence
            .iter()
            .filter(|presence| presence.has_oracle_capture)
            .count();

        let packets_oracle_backed = self
            .artifact_presence
            .iter()
            .filter(|presence| presence.report_source == ReportSource::OracleBacked)
            .count();
        let packets_self_check = self
            .artifact_presence
            .iter()
            .filter(|presence| presence.report_source == ReportSource::SelfCheck)
            .count();

        DashboardMetrics {
            total_packets,
            failed_packets,
            total_cases,
            failed_cases,
            failed_case_pct,
            packets_with_sidecar,
            packets_with_decode_proof,
            packets_with_oracle_capture,
            packets_oracle_backed,
            packets_self_check,
        }
    }

    fn clamp_indices(&mut self) {
        if self.reports.is_empty() {
            self.selected_packet = 0;
            self.selected_case = 0;
            return;
        }
        self.selected_packet = self.selected_packet.min(self.reports.len() - 1);
        let case_len = self.reports[self.selected_packet].case_results.len();
        if case_len == 0 {
            self.selected_case = 0;
        } else {
            self.selected_case = self.selected_case.min(case_len - 1);
        }
    }
}

pub fn load_dashboard_state(
    config: &HarnessConfig,
    packet_filter: Option<String>,
) -> Result<DashboardState, HarnessError> {
    let artifact_root = config.fixture_root.join("artifacts");
    load_dashboard_state_from_artifact_root(&artifact_root, packet_filter)
}

pub fn load_dashboard_state_from_artifact_root(
    artifact_root: &Path,
    packet_filter: Option<String>,
) -> Result<DashboardState, HarnessError> {
    if !artifact_root.exists() {
        return Ok(DashboardState::new(
            Vec::new(),
            Vec::new(),
            vec![format!(
                "artifact root `{}` does not exist; showing empty dashboard",
                artifact_root.display()
            )],
            packet_filter,
        ));
    }

    let (reports, artifact_presence, warnings) = discover_reports_best_effort(artifact_root)?;
    let filtered = apply_filter(reports, packet_filter.as_deref());

    let filtered_ids: std::collections::HashSet<String> = filtered
        .iter()
        .map(|report| report.packet_id.clone())
        .collect();
    let filtered_presence = artifact_presence
        .into_iter()
        .filter(|presence| filtered_ids.contains(&presence.packet_id))
        .collect();

    Ok(DashboardState::new(
        filtered,
        filtered_presence,
        warnings,
        packet_filter,
    ))
}

fn differential_max_diff(packet: &PacketReport, case: &crate::CaseResult) -> Option<f64> {
    packet
        .differential_case_results
        .as_ref()?
        .iter()
        .find(|result| result.case_id == case.case_id)
        .and_then(|result| result.max_diff)
}

fn compare_case_list_entries(
    left: &DashboardCaseListEntry<'_>,
    right: &DashboardCaseListEntry<'_>,
) -> Ordering {
    left.case
        .passed
        .cmp(&right.case.passed)
        .then_with(|| compare_max_diff_desc(left.max_diff, right.max_diff))
        .then_with(|| left.case.case_id.cmp(&right.case.case_id))
        .then_with(|| left.source_index.cmp(&right.source_index))
}

fn compare_max_diff_desc(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right.total_cmp(&left),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn apply_filter(reports: Vec<PacketReport>, packet_filter: Option<&str>) -> Vec<PacketReport> {
    match packet_filter {
        None => reports,
        Some(filter) => {
            let query = filter.trim().to_ascii_lowercase();
            if query.is_empty() {
                reports
            } else {
                reports
                    .into_iter()
                    .filter(|report| {
                        report.packet_id.to_ascii_lowercase().contains(&query)
                            || report.family.to_ascii_lowercase().contains(&query)
                    })
                    .collect()
            }
        }
    }
}

fn discover_reports_best_effort(artifact_root: &Path) -> Result<DiscoveryResult, HarnessError> {
    let mut reports = Vec::new();
    let mut presence = Vec::new();
    let mut warnings = Vec::new();

    for entry in fs::read_dir(artifact_root).map_err(|source| HarnessError::ArtifactIo {
        path: artifact_root.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| HarnessError::ArtifactIo {
            path: artifact_root.to_path_buf(),
            source,
        })?;
        let packet_dir = entry.path();
        if !packet_dir.is_dir() {
            continue;
        }

        let Some(packet_name) = packet_dir
            .file_name()
            .and_then(|value| value.to_str())
            .map(ToOwned::to_owned)
        else {
            warnings.push(format!(
                "skipping packet directory with invalid UTF-8 name: {}",
                packet_dir.display()
            ));
            continue;
        };

        let mut packet_presence = PacketArtifactPresence::new(packet_name.clone());
        // Per frankenscipy-szb5: packet directories come in two flavors:
        //   FSCI-P2C-N/  -- runner output, parity_report.json at top level
        //   P2C-N/       -- evidence_p2c*.rs output, parity_report.json
        //                   under evidence/ subdirectory
        // Probe both locations so dashboards don't show the 10 P2C-N/
        // dirs as zombie packets with no report.
        let report_path = {
            let top = packet_dir.join("parity_report.json");
            if top.exists() {
                top
            } else {
                packet_dir.join("evidence").join("parity_report.json")
            }
        };
        let sidecar_path = packet_dir.join("parity_report.raptorq.json");
        let decode_proof_path = packet_dir.join("parity_report.decode_proof.json");
        let oracle_capture_path = packet_dir.join("oracle_capture.json");
        let oracle_error_path = packet_dir.join("oracle_capture.error.txt");

        packet_presence.has_report = report_path.exists();
        packet_presence.has_sidecar = sidecar_path.exists();
        packet_presence.has_decode_proof = decode_proof_path.exists();
        packet_presence.has_oracle_capture = oracle_capture_path.exists();
        packet_presence.has_oracle_error = oracle_error_path.exists();
        if packet_presence.has_sidecar {
            match fs::read_to_string(&sidecar_path) {
                Ok(raw) => match from_str::<RaptorQSidecar>(&raw) {
                    Ok(sidecar) => {
                        packet_presence.sidecar_source_hash = Some(sidecar.source_hash);
                        packet_presence.sidecar_source_symbols = Some(sidecar.source_symbols);
                        packet_presence.sidecar_repair_symbols = Some(sidecar.repair_symbols);
                    }
                    Err(error) => warnings.push(format!(
                        "failed to parse `{}`: {error}",
                        sidecar_path.display()
                    )),
                },
                Err(error) => warnings.push(format!(
                    "failed to read `{}`: {error}",
                    sidecar_path.display()
                )),
            }
        }
        if packet_presence.has_decode_proof {
            match fs::read_to_string(&decode_proof_path) {
                Ok(raw) => match from_str::<DecodeProofArtifact>(&raw) {
                    Ok(proof) => {
                        packet_presence.decode_proof_hash = Some(proof.proof_hash);
                        packet_presence.decode_proof_reason = Some(proof.reason);
                    }
                    Err(error) => warnings.push(format!(
                        "failed to parse `{}`: {error}",
                        decode_proof_path.display()
                    )),
                },
                Err(error) => warnings.push(format!(
                    "failed to read `{}`: {error}",
                    decode_proof_path.display()
                )),
            }
        }
        packet_presence.classify_report_source();

        if report_path.exists() {
            match fs::read_to_string(&report_path) {
                Ok(raw) => match from_str::<PacketReport>(&raw) {
                    Ok(report) => reports.push(report),
                    Err(error) => warnings.push(format!(
                        "failed to parse `{}`: {error}",
                        report_path.display()
                    )),
                },
                Err(error) => warnings.push(format!(
                    "failed to read `{}`: {error}",
                    report_path.display()
                )),
            }
        }

        presence.push(packet_presence);
    }

    reports.sort_by(|left, right| left.packet_id.cmp(&right.packet_id));
    presence.sort_by(|left, right| left.packet_id.cmp(&right.packet_id));
    Ok((reports, presence, warnings))
}

#[cfg(test)]
mod tests {
    use super::{
        DashboardPanel, DashboardState, PacketArtifactPresence,
        load_dashboard_state_from_artifact_root,
    };
    use crate::{CaseResult, DifferentialCaseResult, OracleStatus, PacketReport};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(tag: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let root = PathBuf::from("/tmp").join(format!("fsci-dashboard-{tag}-{ts}"));
        fs::create_dir_all(&root).expect("create temp root");
        root
    }

    fn report(packet_id: &str, failed_cases: usize) -> PacketReport {
        PacketReport {
            schema_version: 1,
            packet_id: packet_id.to_owned(),
            family: "demo".to_owned(),
            case_results: vec![
                CaseResult {
                    case_id: "c1".to_owned(),
                    passed: true,
                    message: "ok".to_owned(),
                },
                CaseResult {
                    case_id: "c2".to_owned(),
                    passed: failed_cases == 0,
                    message: "msg".to_owned(),
                },
            ],
            passed_cases: if failed_cases == 0 { 2 } else { 1 },
            failed_cases,
            fixture_path: None,
            oracle_status: None,
            differential_case_results: None,
            report_kind: crate::ReportKind::Unspecified,
            generated_unix_ms: 0,
        }
    }

    #[test]
    fn dashboard_navigation_moves_packet_and_case_indices() {
        let reports = vec![report("PKT-1", 0), report("PKT-2", 1)];
        let presence = vec![
            PacketArtifactPresence::new("PKT-1".to_owned()),
            PacketArtifactPresence::new("PKT-2".to_owned()),
        ];

        let mut state = DashboardState::new(reports, presence, Vec::new(), None);
        assert_eq!(state.active_panel(), DashboardPanel::Packets);
        assert_eq!(state.selected_packet_index(), 0);
        assert_eq!(state.selected_case_index(), 0);

        state.move_down();
        assert_eq!(state.selected_packet_index(), 1);
        assert_eq!(state.selected_case_index(), 0);

        state.next_panel();
        assert_eq!(state.active_panel(), DashboardPanel::Cases);
        state.move_down();
        assert_eq!(state.selected_case_index(), 1);

        state.next_panel();
        assert_eq!(state.active_panel(), DashboardPanel::Summary);
        state.move_down();
        assert_eq!(state.selected_case_index(), 1);

        state.prev_panel();
        assert_eq!(state.active_panel(), DashboardPanel::Cases);
    }

    #[test]
    fn dashboard_case_list_prioritizes_failures_by_numeric_delta() {
        let reports = vec![PacketReport {
            schema_version: 2,
            packet_id: "PKT-SORT".to_owned(),
            family: "demo".to_owned(),
            case_results: vec![
                CaseResult {
                    case_id: "pass-small".to_owned(),
                    passed: true,
                    message: "ok".to_owned(),
                },
                CaseResult {
                    case_id: "fail-low".to_owned(),
                    passed: false,
                    message: "low mismatch".to_owned(),
                },
                CaseResult {
                    case_id: "fail-high".to_owned(),
                    passed: false,
                    message: "high mismatch".to_owned(),
                },
            ],
            passed_cases: 1,
            failed_cases: 2,
            fixture_path: None,
            oracle_status: None,
            differential_case_results: Some(vec![
                DifferentialCaseResult {
                    case_id: "pass-small".to_owned(),
                    passed: true,
                    message: "ok".to_owned(),
                    max_diff: Some(1.0e-12),
                    tolerance_used: None,
                    oracle_status: OracleStatus::Available,
                },
                DifferentialCaseResult {
                    case_id: "fail-low".to_owned(),
                    passed: false,
                    message: "low mismatch".to_owned(),
                    max_diff: Some(2.0e-6),
                    tolerance_used: None,
                    oracle_status: OracleStatus::Available,
                },
                DifferentialCaseResult {
                    case_id: "fail-high".to_owned(),
                    passed: false,
                    message: "high mismatch".to_owned(),
                    max_diff: Some(5.0e-3),
                    tolerance_used: None,
                    oracle_status: OracleStatus::Available,
                },
            ]),
            report_kind: crate::ReportKind::Unspecified,
            generated_unix_ms: 0,
        }];
        let presence = vec![PacketArtifactPresence::new("PKT-SORT".to_owned())];
        let state = DashboardState::new(reports, presence, Vec::new(), None);

        let ordered_case_ids = state
            .case_list_entries()
            .iter()
            .map(|entry| entry.case.case_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ordered_case_ids,
            vec!["fail-high", "fail-low", "pass-small"]
        );
        assert_eq!(
            state.selected_case().expect("selected case").case_id,
            "fail-high"
        );
        assert!(
            state
                .selected_case_drilldown_lines()
                .iter()
                .any(|line| line.contains("max_abs_diff=5.000000e-3"))
        );
    }

    #[test]
    fn dashboard_loader_skips_malformed_reports() {
        let root = temp_dir("best-effort");
        let good = root.join("PKT-GOOD");
        let bad = root.join("PKT-BAD");
        fs::create_dir_all(&good).expect("create good packet dir");
        fs::create_dir_all(&bad).expect("create bad packet dir");

        let good_report = r#"{
  "packet_id": "PKT-GOOD",
  "family": "demo",
  "case_results": [{"case_id":"ok","passed":true,"message":"ok"}],
  "passed_cases": 1,
  "failed_cases": 0,
  "generated_unix_ms": 0
}"#;
        fs::write(good.join("parity_report.json"), good_report).expect("write good report");
        fs::write(
            good.join("parity_report.raptorq.json"),
            r#"{
  "schema_version": 1,
  "source_hash": "abcdef0123456789abcdef0123456789",
  "symbol_size": 1024,
  "seed": 7,
  "source_symbols": 3,
  "repair_symbols": 2,
  "repair_symbol_hashes": ["r1", "r2"]
}"#,
        )
        .expect("write sidecar");
        fs::write(
            good.join("parity_report.decode_proof.json"),
            r#"{
  "ts_unix_ms": 123,
  "reason": "test decode proof",
  "recovered_blocks": 1,
  "proof_hash": "fedcba9876543210fedcba9876543210"
}"#,
        )
        .expect("write proof");
        fs::write(good.join("oracle_capture.json"), "{}").expect("write oracle");

        fs::write(bad.join("parity_report.json"), "{not_json").expect("write bad report");

        let state = load_dashboard_state_from_artifact_root(&root, None).expect("load state");
        assert_eq!(state.reports().len(), 1);
        assert_eq!(state.reports()[0].packet_id, "PKT-GOOD");
        assert_eq!(state.warnings().len(), 1);

        let metrics = state.metrics();
        assert_eq!(metrics.total_packets, 1);
        assert_eq!(metrics.packets_with_sidecar, 1);
        assert_eq!(metrics.packets_with_decode_proof, 1);
        assert_eq!(metrics.packets_with_oracle_capture, 1);
        let artifact = state
            .selected_packet_artifacts()
            .expect("artifact presence");
        assert_eq!(artifact.sidecar_source_symbols, Some(3));
        assert_eq!(artifact.sidecar_repair_symbols, Some(2));
        assert_eq!(
            artifact.sidecar_source_hash.as_deref(),
            Some("abcdef0123456789abcdef0123456789")
        );
        assert_eq!(
            artifact.decode_proof_hash.as_deref(),
            Some("fedcba9876543210fedcba9876543210")
        );
        assert_eq!(
            artifact.decode_proof_reason.as_deref(),
            Some("test decode proof")
        );
    }

    #[test]
    fn dashboard_loader_applies_case_insensitive_packet_filter() {
        let root = temp_dir("filter");
        let p1 = root.join("FSCI-P2C-001");
        let p2 = root.join("FSCI-P2C-002");
        fs::create_dir_all(&p1).expect("create p1 dir");
        fs::create_dir_all(&p2).expect("create p2 dir");

        let report_1 = r#"{
  "packet_id": "FSCI-P2C-001",
  "family": "integrate",
  "case_results": [{"case_id":"c","passed":true,"message":"ok"}],
  "passed_cases": 1,
  "failed_cases": 0,
  "generated_unix_ms": 0
}"#;
        let report_2 = r#"{
  "packet_id": "FSCI-P2C-002",
  "family": "linalg_core",
  "case_results": [{"case_id":"c","passed":true,"message":"ok"}],
  "passed_cases": 1,
  "failed_cases": 0,
  "generated_unix_ms": 0
}"#;
        fs::write(p1.join("parity_report.json"), report_1).expect("write report 1");
        fs::write(p2.join("parity_report.json"), report_2).expect("write report 2");

        let state = load_dashboard_state_from_artifact_root(&root, Some("p2c-002".to_owned()))
            .expect("load filtered state");
        assert_eq!(state.reports().len(), 1);
        assert_eq!(state.reports()[0].packet_id, "FSCI-P2C-002");
    }

    #[test]
    fn dashboard_loader_accepts_rich_differential_reports() {
        let root = temp_dir("rich-report");
        let packet = root.join("FSCI-P2C-777");
        fs::create_dir_all(&packet).expect("create packet dir");

        let rich_report = r#"{
  "schema_version": 2,
  "packet_id": "FSCI-P2C-777",
  "family": "stats",
  "case_results": [{"case_id":"ok","passed":true,"message":"scalar match"}],
  "passed_cases": 1,
  "failed_cases": 0,
  "fixture_path": "fixtures/FSCI-P2C-777_stats.json",
  "oracle_status": {"status":"available"},
  "differential_case_results": [{
    "case_id":"ok",
    "passed":true,
    "message":"scalar match",
    "max_diff":1e-12,
    "tolerance_used":{"atol":1e-12,"rtol":1e-12,"comparison_mode":"scalar"},
    "oracle_status":{"status":"available"}
  }],
  "generated_unix_ms": 0
}"#;
        fs::write(packet.join("parity_report.json"), rich_report).expect("write report");

        let state = load_dashboard_state_from_artifact_root(&root, None).expect("load state");
        assert_eq!(state.reports().len(), 1);
        let report = &state.reports()[0];
        assert_eq!(report.schema_version, 2);
        assert_eq!(
            report.fixture_path.as_deref(),
            Some("fixtures/FSCI-P2C-777_stats.json")
        );
        assert!(report.differential_case_results.is_some());
        let drilldown = state.selected_case_drilldown_lines();
        assert!(
            drilldown
                .iter()
                .any(|line| line.contains("max_abs_diff=1.000000e-12")),
            "drilldown should surface numeric differential deltas: {drilldown:?}"
        );
        assert!(
            drilldown
                .iter()
                .any(|line| line
                    .contains("tolerance atol=1.000000e-12 rtol=1.000000e-12 mode=scalar")),
            "drilldown should surface tolerance policy: {drilldown:?}"
        );
        assert!(
            drilldown
                .iter()
                .any(|line| line.contains("oracle_status=Available")),
            "drilldown should surface oracle status: {drilldown:?}"
        );
    }

    #[test]
    fn dashboard_panel_list_is_stable() {
        assert_eq!(DashboardPanel::Packets.next(), DashboardPanel::Cases);
        assert_eq!(DashboardPanel::Cases.next(), DashboardPanel::Summary);
        assert_eq!(DashboardPanel::Summary.next(), DashboardPanel::Packets);
        assert_eq!(DashboardPanel::Packets.prev(), DashboardPanel::Summary);
        assert_eq!(DashboardPanel::Summary.prev(), DashboardPanel::Cases);
        assert_eq!(DashboardPanel::Cases.prev(), DashboardPanel::Packets);
    }
}
