#![forbid(unsafe_code)]

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_dir(suffix: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    std::env::temp_dir().join(format!(
        "fsci_conformance_e2e_{suffix}_{}_{}",
        std::process::id(),
        nonce
    ))
}

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!("failed to create {}: {error}", parent.display());
        });
    }
    fs::write(path, contents).unwrap_or_else(|error| {
        panic!("failed to write {}: {error}", path.display());
    });
}

fn collect_named_files(root: &Path, file_name: &str, out: &mut Vec<PathBuf>) {
    if !root.exists() {
        return;
    }
    let entries = fs::read_dir(root).unwrap_or_else(|error| {
        panic!("failed to read {}: {error}", root.display());
    });
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!("failed to read entry in {}: {error}", root.display());
        });
        let path = entry.path();
        if path.is_dir() {
            collect_named_files(&path, file_name, out);
        } else if path.file_name().and_then(|name| name.to_str()) == Some(file_name) {
            out.push(path);
        }
    }
}

#[test]
fn e2e_orchestrator_runs_scenario_and_emits_forensic_bundle() {
    let temp_root = unique_temp_dir("pass");
    let artifact_root = temp_root.join("artifacts");
    let packet_dir = artifact_root.join("FSCI-P2C-TEST");
    let scenario_path = packet_dir.join("e2e/scenarios/scenario_pass.json");
    let sentinel_path = packet_dir.join("sentinel.txt");
    write_file(&sentinel_path, "sentinel");
    write_file(
        &scenario_path,
        r#"{
  "scenario_id": "scenario_pass",
  "seed": 123,
  "steps": [
    {
      "step_name": "ensure_sentinel_present",
      "kind": "check_path_exists",
      "path": "FSCI-P2C-TEST/sentinel.txt"
    },
    {
      "step_name": "run_validate_tol_fixture",
      "kind": "run_differential_fixture",
      "fixture": "FSCI-P2C-001_validate_tol.json"
    }
  ]
}"#,
    );

    let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let output = Command::new(env!("CARGO_BIN_EXE_e2e_orchestrator"))
        .arg("--artifact-root")
        .arg(&artifact_root)
        .arg("--fixture-root")
        .arg(&fixture_root)
        .output()
        .expect("failed to execute e2e_orchestrator");

    assert!(
        output.status.success(),
        "expected success; stdout={}; stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let mut event_files = Vec::new();
    collect_named_files(
        &packet_dir.join("e2e/runs"),
        "events.jsonl",
        &mut event_files,
    );
    assert_eq!(
        event_files.len(),
        1,
        "expected exactly one events.jsonl file"
    );

    let events_raw = fs::read_to_string(&event_files[0]).expect("failed to read events log");
    let lines: Vec<&str> = events_raw.lines().collect();
    assert!(
        lines.len() >= 6,
        "expected lifecycle + scenario entries, got {}",
        lines.len()
    );

    let first: Value = serde_json::from_str(lines.first().expect("missing first log line"))
        .expect("failed to parse first log line as json");
    assert_eq!(first["scenario_id"], "scenario_pass");
    assert_eq!(first["step_name"], "setup");
    assert!(first["environment"]["rust_version"].is_string());
    assert_eq!(first["environment"]["seed"], 123);
    assert!(first["environment"]["input_hashes"].is_array());

    assert!(
        lines
            .iter()
            .any(|line| line.contains("ensure_sentinel_present")),
        "missing check_path_exists step log"
    );
    assert!(
        lines
            .iter()
            .any(|line| line.contains("run_validate_tol_fixture")),
        "missing run_differential_fixture step log"
    );

    let mut summary_files = Vec::new();
    collect_named_files(
        &packet_dir.join("e2e/runs"),
        "summary.json",
        &mut summary_files,
    );
    assert_eq!(
        summary_files.len(),
        1,
        "expected exactly one summary.json file"
    );
    let summary_raw = fs::read_to_string(&summary_files[0]).expect("failed to read summary file");
    let summary_json: Value =
        serde_json::from_str(&summary_raw).expect("failed to parse summary file");
    assert_eq!(summary_json["scenario_id"], "scenario_pass");
    assert_eq!(summary_json["passed"], true);
}

#[test]
fn e2e_orchestrator_emits_replay_command_as_last_log_line_on_failure() {
    let temp_root = unique_temp_dir("fail");
    let artifact_root = temp_root.join("artifacts");
    let packet_dir = artifact_root.join("FSCI-P2C-FAIL");
    let scenario_path = packet_dir.join("e2e/scenarios/scenario_fail.json");
    write_file(
        &scenario_path,
        r#"{
  "scenario_id": "scenario_fail",
  "seed": 7,
  "steps": [
    {
      "step_name": "expect_missing_path",
      "kind": "check_path_exists",
      "path": "FSCI-P2C-FAIL/does-not-exist.txt"
    }
  ]
}"#,
    );

    let output = Command::new(env!("CARGO_BIN_EXE_e2e_orchestrator"))
        .arg("--artifact-root")
        .arg(&artifact_root)
        .arg("--scenario")
        .arg("scenario_fail")
        .output()
        .expect("failed to execute e2e_orchestrator");

    assert!(
        !output.status.success(),
        "expected failure exit code for failing scenario"
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "failing scenarios should return exit code 1"
    );

    let mut event_files = Vec::new();
    collect_named_files(
        &packet_dir.join("e2e/runs"),
        "events.jsonl",
        &mut event_files,
    );
    assert_eq!(
        event_files.len(),
        1,
        "expected exactly one events.jsonl file"
    );

    let events_raw = fs::read_to_string(&event_files[0]).expect("failed to read events log");
    let lines: Vec<&str> = events_raw.lines().collect();
    let last: Value = serde_json::from_str(lines.last().expect("missing replay line"))
        .expect("failed to parse last log line");

    assert_eq!(last["step_name"], "replay_command");
    assert_eq!(last["scenario_id"], "scenario_fail");
    let replay = last["replay_command"]
        .as_str()
        .expect("replay_command should be present");
    assert!(
        replay.contains("--scenario scenario_fail"),
        "replay command should include scenario filter, got: {replay}"
    );
}
