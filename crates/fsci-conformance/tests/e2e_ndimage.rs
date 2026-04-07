#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-010 (ndimage interpolation).
//!
//! Each scenario emits topology-compliant artifacts to
//! `fixtures/artifacts/FSCI-P2C-010/e2e/runs/{run_id}/{scenario_id}/`
//! containing `events.jsonl` and `summary.json`.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{BoundaryMode, NdArray, map_coordinates, shift, zoom};
use serde::Serialize;

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

const PACKET_ID: &str = "FSCI-P2C-010";
const TOL: f64 = 1e-6;

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
        "rch exec -- cargo test -p fsci-conformance --test e2e_ndimage -- {scenario_id} --nocapture"
    )
}

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

fn write_topology_artifacts(
    scenario_id: &str,
    steps: &[ForensicStep],
    all_pass: bool,
) -> Result<(), String> {
    let rid = run_id();
    let dir = e2e_runs_dir(&rid, scenario_id);
    fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create run dir {}: {e}", dir.display()))?;

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

fn vec_close(actual: &[f64], expected: &[f64], tol: f64) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() <= tol)
}

#[test]
fn e2e_ndimage_interpolation() {
    let scenario_id = "e2e_ndimage_interpolation";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let signal = NdArray::new(vec![0.0, 10.0, 20.0, 30.0], vec![4]).unwrap();
    let shifted = shift(&signal, &[0.5], 1, BoundaryMode::Nearest, 0.0).unwrap();
    let shifted_expected = vec![0.0, 5.0, 15.0, 25.0];
    let pass = vec_close(&shifted.data, &shifted_expected, TOL);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "shift_order1_reference",
        "ndimage::shift(order=1)",
        "signal=[0,10,20,30], shift=0.5, mode=nearest",
        &format!("got={:?}", shifted.data),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let image = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let zoomed = zoom(&image, &[2.0, 2.0], 1, BoundaryMode::Nearest, 0.0).unwrap();
    let zoomed_expected = vec![
        1.0,
        1.333_333_333_333_333_3,
        1.666_666_666_666_666_5,
        2.0,
        1.666_666_666_666_666_5,
        2.0,
        2.333_333_333_333_333,
        2.666_666_666_666_666_5,
        2.333_333_333_333_333_5,
        2.666_666_666_666_666_5,
        3.0,
        3.333_333_333_333_333,
        3.0,
        3.333_333_333_333_333,
        3.666_666_666_666_666_5,
        4.0,
    ];
    let pass = vec_close(&zoomed.data, &zoomed_expected, TOL);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "zoom_order1_reference",
        "ndimage::zoom(order=1)",
        "image=[[1,2],[3,4]], zoom=(2,2), mode=nearest",
        &format!("got={:?}", zoomed.data),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let cubic_shift = shift(&signal, &[0.5], 3, BoundaryMode::Nearest, 0.0).unwrap();
    let cubic_shift_expected = vec![
        -0.807_713_659_400_537_9,
        4.264_428_414_850_133_5,
        15.0,
        25.735_571_585_149_867,
    ];
    let pass = vec_close(&cubic_shift.data, &cubic_shift_expected, 1e-9);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "shift_order3_reference",
        "ndimage::shift(order=3)",
        "signal=[0,10,20,30], shift=0.5, mode=nearest",
        &format!("got={:?}", cubic_shift.data),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let plane = NdArray::new(
        (0..10)
            .flat_map(|r| (0..10).map(move |c| (10 * r + c) as f64))
            .collect(),
        vec![10, 10],
    )
    .unwrap();
    let mapped = map_coordinates(
        &plane,
        &[vec![4.25, 5.5], vec![6.5, 3.75]],
        3,
        BoundaryMode::Nearest,
        0.0,
    )
    .unwrap();
    let mapped_expected = vec![49.003_891_481_726_93, 58.736_556_699_488_844];
    let pass = vec_close(&mapped, &mapped_expected, 5e-2);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        6,
        "map_coordinates_order3_reference",
        "ndimage::map_coordinates(order=3)",
        "10x10 linear ramp, coords=[[4.25,5.5],[6.5,3.75]], mode=nearest",
        &format!("got={mapped:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    let details = steps
        .iter()
        .map(|step| {
            format!(
                "{} => {} ({})",
                step.step_name, step.output_summary, step.outcome
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(all_pass, "scenario {scenario_id} had failures: {details}");
}
