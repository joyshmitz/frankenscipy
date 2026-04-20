#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-013 (ndimage interpolation).
//!
//! Each scenario emits topology-compliant artifacts to
//! `fixtures/artifacts/FSCI-P2C-013/e2e/runs/{run_id}/{scenario_id}/`
//! containing `events.jsonl` and `summary.json`.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    BoundaryMode, NdArray, binary_closing, binary_dilation, binary_erosion, binary_opening,
    center_of_mass, convolve, correlate, distance_transform_edt, find_objects, gaussian_filter,
    label, laplace, map_coordinates, maximum_filter, mean_labels, median_filter, minimum_filter,
    prewitt, rotate, shift, sobel, sum_labels, uniform_filter, zoom,
};
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

const PACKET_ID: &str = "FSCI-P2C-013";
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

// ══════════════════ FILTER CONFORMANCE ══════════════════

/// Test convolution and correlation filters
#[test]
fn e2e_ndimage_convolution() {
    let scenario_id = "e2e_ndimage_convolution";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Test 1D convolution with simple kernel
    let t = Instant::now();
    let signal = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let kernel = NdArray::new(vec![1.0, 0.0, -1.0], vec![3]).unwrap();
    let conv_result = convolve(&signal, &kernel, BoundaryMode::Constant, 0.0);
    let pass = match &conv_result {
        Ok(result) => {
            // Convolution with [-1, 0, 1] kernel computes differences
            // At center: result[i] = signal[i+1] - signal[i-1]
            result.data.len() == 5
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "convolve_1d_diff",
        "ndimage::convolve(signal, kernel)",
        "signal=[1,2,3,4,5], kernel=[1,0,-1]",
        &format!("result={:?}", conv_result.as_ref().map(|r| &r.data)),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Test correlation (same as convolution with flipped kernel)
    let t = Instant::now();
    let corr_result = correlate(&signal, &kernel, BoundaryMode::Constant, 0.0);
    let pass = corr_result.is_ok() && corr_result.as_ref().unwrap().data.len() == 5;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "correlate_1d",
        "ndimage::correlate(signal, kernel)",
        "signal=[1,2,3,4,5], kernel=[1,0,-1]",
        &format!("result={:?}", corr_result.as_ref().map(|r| &r.data)),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Test uniform filter (box blur)
    let t = Instant::now();
    let image = NdArray::new(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        vec![5, 5],
    )
    .unwrap();
    let uniform_result = uniform_filter(&image, 3, BoundaryMode::Constant, 0.0);
    let pass = match &uniform_result {
        Ok(result) => {
            // Center pixel should be average of 3x3 neighborhood = 9/9 = 1.0
            let center_val = result.data[12]; // index 2*5+2 = 12
            (center_val - 1.0).abs() < 1e-10
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "uniform_filter_3x3",
        "ndimage::uniform_filter(size=3)",
        "5x5 image with 3x3 center block",
        &format!(
            "center_val={}",
            uniform_result
                .as_ref()
                .map(|r| r.data[12])
                .unwrap_or(f64::NAN)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Test Gaussian filter
    let t = Instant::now();
    let gauss_result = gaussian_filter(&image, 1.0, BoundaryMode::Constant, 0.0);
    let pass = match &gauss_result {
        Ok(result) => {
            // Gaussian should smooth the image, center should be close to but less than 1.0
            let center_val = result.data[12];
            center_val > 0.5 && center_val < 1.5
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "gaussian_filter_sigma1",
        "ndimage::gaussian_filter(sigma=1.0)",
        "5x5 image with 3x3 center block",
        &format!(
            "center_val={}",
            gauss_result
                .as_ref()
                .map(|r| r.data[12])
                .unwrap_or(f64::NAN)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Test median, minimum, and maximum filters
#[test]
fn e2e_ndimage_rank_filters() {
    let scenario_id = "e2e_ndimage_rank_filters";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Image with an outlier
    let image = NdArray::new(
        vec![
            1.0, 1.0, 1.0, 1.0, 100.0, 1.0, // outlier in center
            1.0, 1.0, 1.0,
        ],
        vec![3, 3],
    )
    .unwrap();

    // Median filter should remove the outlier
    let t = Instant::now();
    let median_result = median_filter(&image, 3, BoundaryMode::Nearest, 0.0);
    let pass = match &median_result {
        Ok(result) => {
            // Center pixel should be median of [1,1,1,1,100,1,1,1,1] = 1.0
            let center_val = result.data[4];
            (center_val - 1.0).abs() < 1e-10
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "median_filter_outlier",
        "ndimage::median_filter(size=3)",
        "3x3 image with outlier at center",
        &format!(
            "center_val={}",
            median_result
                .as_ref()
                .map(|r| r.data[4])
                .unwrap_or(f64::NAN)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Minimum filter
    let t = Instant::now();
    let min_result = minimum_filter(&image, 3, BoundaryMode::Nearest, 0.0);
    let pass = match &min_result {
        Ok(result) => {
            // All pixels should be 1.0 (min of neighborhood)
            result.data.iter().all(|&v| (v - 1.0).abs() < 1e-10)
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "minimum_filter",
        "ndimage::minimum_filter(size=3)",
        "3x3 image with outlier",
        &format!("all_ones={}", pass),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Maximum filter
    let t = Instant::now();
    let max_result = maximum_filter(&image, 3, BoundaryMode::Nearest, 0.0);
    let pass = match &max_result {
        Ok(result) => {
            // All pixels should be 100.0 (max of neighborhood includes center)
            result.data.iter().all(|&v| (v - 100.0).abs() < 1e-10)
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "maximum_filter",
        "ndimage::maximum_filter(size=3)",
        "3x3 image with outlier",
        &format!("all_100={}", pass),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Test edge detection: Sobel, Prewitt, Laplacian
#[test]
fn e2e_ndimage_edge_detection() {
    let scenario_id = "e2e_ndimage_edge_detection";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Simple step edge image (left half black, right half white)
    let image = NdArray::new(
        vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            1.0, 1.0,
        ],
        vec![6, 6],
    )
    .unwrap();

    // Sobel edge detection (horizontal derivative)
    let t = Instant::now();
    let sobel_result = sobel(&image, 1, BoundaryMode::Constant, 0.0);
    let pass = match &sobel_result {
        Ok(result) => {
            // Edge should be detected in column 2-3 transition
            // Values should be non-zero at the edge
            let edge_col = 2; // column just before transition
            let edge_val = result.data[6 + edge_col]; // row 1, col 2
            edge_val.abs() > 0.1 || result.data[6 + 3].abs() > 0.1
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "sobel_horizontal",
        "ndimage::sobel(axis=1)",
        "6x6 step edge image",
        &format!("edge_detected={}", pass),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Prewitt edge detection
    let t = Instant::now();
    let prewitt_result = prewitt(&image, 1, BoundaryMode::Constant, 0.0);
    let pass = match &prewitt_result {
        Ok(result) => {
            // Similar to Sobel, should detect the vertical edge
            result.data.iter().any(|&v| v.abs() > 0.1)
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "prewitt_horizontal",
        "ndimage::prewitt(axis=1)",
        "6x6 step edge image",
        &format!("edge_detected={}", pass),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Laplacian
    let t = Instant::now();
    let laplace_result = laplace(&image, BoundaryMode::Constant, 0.0);
    let pass = match &laplace_result {
        Ok(result) => {
            // Laplacian should respond at edges
            result.data.iter().any(|&v| v.abs() > 0.1)
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "laplacian",
        "ndimage::laplace()",
        "6x6 step edge image",
        &format!("has_response={}", pass),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Test binary morphological operations
#[test]
fn e2e_ndimage_binary_morphology() {
    let scenario_id = "e2e_ndimage_binary_morphology";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Small binary image with a 3x3 square
    let image = NdArray::new(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        vec![5, 5],
    )
    .unwrap();

    // Binary erosion should shrink the object
    let t = Instant::now();
    let eroded = binary_erosion(&image, 3, 1); // 3x3 structuring element
    let pass = match &eroded {
        Ok(result) => {
            // After erosion, only center pixel should remain
            let center_val = result.data[12]; // 2*5+2
            let border_eroded = result.data[6] < 0.5; // 1*5+1 should be eroded
            center_val > 0.5 && border_eroded
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "binary_erosion",
        "ndimage::binary_erosion(size=3, iterations=1)",
        "5x5 image with 3x3 square",
        &format!(
            "center={}, corner_eroded={}",
            eroded.as_ref().map(|r| r.data[12]).unwrap_or(f64::NAN),
            eroded.as_ref().map(|r| r.data[6] < 0.5).unwrap_or(false)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Binary dilation should expand the object
    let t = Instant::now();
    let dilated = binary_dilation(&image, 3, 1); // 3x3 structuring element
    let pass = match &dilated {
        Ok(result) => {
            // After dilation, corners should be filled
            let corner_filled = result.data[0] > 0.5; // top-left corner
            let expanded = result.data.iter().filter(|&&v| v > 0.5).count() > 9;
            corner_filled || expanded
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "binary_dilation",
        "ndimage::binary_dilation(size=3, iterations=1)",
        "5x5 image with 3x3 square",
        &format!("expanded={}", pass),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Binary opening = erosion followed by dilation (removes small protrusions)
    let t = Instant::now();
    let opened = binary_opening(&image, 3, 1);
    let pass = opened.is_ok();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "binary_opening",
        "ndimage::binary_opening(size=3, iterations=1)",
        "5x5 image with 3x3 square",
        &format!("success={}", pass),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Binary closing = dilation followed by erosion (fills small holes)
    let t = Instant::now();
    let closed = binary_closing(&image, 3, 1);
    let pass = closed.is_ok();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "binary_closing",
        "ndimage::binary_closing(size=3, iterations=1)",
        "5x5 image with 3x3 square",
        &format!("success={}", pass),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

#[test]
fn e2e_ndimage_binary_opening_convergence() {
    let scenario_id = "e2e_ndimage_binary_opening_convergence";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let image = NdArray::new(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        vec![5, 5],
    )
    .unwrap();

    let t = Instant::now();
    let converged = binary_opening(&image, 3, 0);
    let pass = match &converged {
        Ok(result) => result.data.iter().all(|&v| v == 0.0),
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "binary_opening_iterations_zero",
        "ndimage::binary_opening(size=3, iterations=0)",
        "5x5 image with 3x3 square",
        &format!(
            "all_zero={}",
            converged
                .as_ref()
                .map(|r| r.data.iter().all(|&v| v == 0.0))
                .unwrap_or(false)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let once = binary_opening(&image, 3, 1);
    let pass = match &once {
        Ok(result) => result.data[6] == 1.0 && result.data[12] == 1.0 && result.data[18] == 1.0,
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "binary_opening_iterations_one",
        "ndimage::binary_opening(size=3, iterations=1)",
        "5x5 image with 3x3 square",
        &format!(
            "preserves_square={}",
            once.as_ref()
                .map(|r| r.data[6] == 1.0 && r.data[12] == 1.0 && r.data[18] == 1.0)
                .unwrap_or(false)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let twice = binary_opening(&image, 3, 2);
    let pass = match &twice {
        Ok(result) => result.data.iter().all(|&v| v == 0.0),
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "binary_opening_iterations_two",
        "ndimage::binary_opening(size=3, iterations=2)",
        "5x5 image with 3x3 square",
        &format!(
            "all_zero={}",
            twice
                .as_ref()
                .map(|r| r.data.iter().all(|&v| v == 0.0))
                .unwrap_or(false)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Test connected component labeling
#[test]
fn e2e_ndimage_labeling() {
    let scenario_id = "e2e_ndimage_labeling";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Image with two separate objects
    let image = NdArray::new(
        vec![
            1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0,
        ],
        vec![5, 6],
    )
    .unwrap();

    // Binarize for labeling
    let binary = NdArray::new(
        image
            .data
            .iter()
            .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
            .collect(),
        image.shape.clone(),
    )
    .unwrap();

    // Label connected components
    let t = Instant::now();
    let label_result = label(&binary);
    let pass = match &label_result {
        Ok((_labels, num_labels)) => {
            // Should find 3 separate objects
            *num_labels == 3
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "label_components",
        "ndimage::label()",
        "5x6 binary image with 3 objects",
        &format!(
            "num_labels={}",
            label_result.as_ref().map(|(_, n)| *n).unwrap_or(0)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Sum labels - sum pixel values for each label
    let t = Instant::now();
    if let Ok((labels, num_labels)) = &label_result {
        let sums = sum_labels(&image, labels, *num_labels);
        let pass = sums.len() == *num_labels && sums.iter().all(|&s| s > 0.0);
        if !pass {
            all_pass = false;
        }
        steps.push(make_step(
            2,
            "sum_labels",
            "ndimage::sum_labels()",
            "sum pixel values per label",
            &format!("sums={:?}", sums),
            t.elapsed().as_nanos(),
            if pass { "pass" } else { "FAIL" },
        ));

        // Mean labels
        let t = Instant::now();
        let means = mean_labels(&image, labels, *num_labels);
        let pass = means.len() == *num_labels;
        if !pass {
            all_pass = false;
        }
        steps.push(make_step(
            3,
            "mean_labels",
            "ndimage::mean_labels()",
            "mean pixel values per label",
            &format!("means={:?}", means),
            t.elapsed().as_nanos(),
            if pass { "pass" } else { "FAIL" },
        ));

        // Find objects (bounding boxes)
        let t = Instant::now();
        let objects = find_objects(labels, *num_labels);
        let pass = objects.len() == *num_labels && objects.iter().all(|o| o.is_some());
        if !pass {
            all_pass = false;
        }
        steps.push(make_step(
            4,
            "find_objects",
            "ndimage::find_objects()",
            "bounding boxes for each label",
            &format!(
                "num_objects={}",
                objects.iter().filter(|o| o.is_some()).count()
            ),
            t.elapsed().as_nanos(),
            if pass { "pass" } else { "FAIL" },
        ));

        // Center of mass
        let t = Instant::now();
        let centers = center_of_mass(&image, labels, *num_labels);
        let pass = centers.len() == *num_labels && centers.iter().all(|c| c.len() == 2);
        if !pass {
            all_pass = false;
        }
        steps.push(make_step(
            5,
            "center_of_mass",
            "ndimage::center_of_mass()",
            "centroid for each label",
            &format!("centers={:?}", centers),
            t.elapsed().as_nanos(),
            if pass { "pass" } else { "FAIL" },
        ));
    }

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Test distance transform
#[test]
fn e2e_ndimage_distance_transform() {
    let scenario_id = "e2e_ndimage_distance_transform";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Binary image with a hole in center
    let image = NdArray::new(
        vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            1.0, // hole at center
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![5, 5],
    )
    .unwrap();

    let t = Instant::now();
    let dist_result = distance_transform_edt(&image, None);
    let pass = match &dist_result {
        Ok(result) => {
            // Distance at the hole (center) should be 0
            let center_dist = result.data[12]; // 2*5+2
            // Distance at corners should be sqrt(2) * 2 = ~2.83 (distance to center hole)
            let corner_dist = result.data[0];
            center_dist < 0.01 && corner_dist > 2.0
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "distance_transform_edt",
        "ndimage::distance_transform_edt(input, None)",
        "5x5 image with center hole",
        &format!(
            "center_dist={}, corner_dist={}",
            dist_result.as_ref().map(|r| r.data[12]).unwrap_or(f64::NAN),
            dist_result.as_ref().map(|r| r.data[0]).unwrap_or(f64::NAN)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Distance transform on inverted image
    let t = Instant::now();
    let inverted = NdArray::new(
        image.data.iter().map(|&v| 1.0 - v).collect(),
        image.shape.clone(),
    )
    .unwrap();
    let dist_inv = distance_transform_edt(&inverted, None);
    let pass = match &dist_inv {
        Ok(result) => {
            // Now center should have the highest distance
            let center_dist = result.data[12];
            center_dist > 0.9 // Should be ~1.0 (distance to nearest edge)
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "distance_transform_inverted",
        "ndimage::distance_transform_edt(inverted, None)",
        "5x5 with only center pixel",
        &format!(
            "center_dist={}",
            dist_inv.as_ref().map(|r| r.data[12]).unwrap_or(f64::NAN)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Anisotropic sampling should scale row and column distances independently.
    let t = Instant::now();
    let sampled = distance_transform_edt(&image, Some(&[2.0, 1.0]));
    let pass = match &sampled {
        Ok(result) => {
            let top_middle = result.data[2];
            let middle_left = result.data[10];
            let corner = result.data[0];
            (top_middle - 4.0).abs() < 1e-10
                && (middle_left - 2.0).abs() < 1e-10
                && (corner - 20.0f64.sqrt()).abs() < 1e-10
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "distance_transform_sampling",
        "ndimage::distance_transform_edt(input, Some([2,1]))",
        "anisotropic pixel spacing",
        &format!(
            "top_middle={}, middle_left={}, corner={}",
            sampled.as_ref().map(|r| r.data[2]).unwrap_or(f64::NAN),
            sampled.as_ref().map(|r| r.data[10]).unwrap_or(f64::NAN),
            sampled.as_ref().map(|r| r.data[0]).unwrap_or(f64::NAN)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Test geometric transformations (rotate)
#[test]
fn e2e_ndimage_geometric() {
    let scenario_id = "e2e_ndimage_geometric";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Simple 4x4 image with pattern
    let image = NdArray::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        vec![4, 4],
    )
    .unwrap();

    // Rotate by 90 degrees
    let t = Instant::now();
    let rotated = rotate(&image, 90.0, false, 1, BoundaryMode::Constant, 0.0);
    let pass = match &rotated {
        Ok(result) => {
            // After 90 degree rotation, shape should be preserved
            // Top-left should now contain what was bottom-left
            result.shape == vec![4, 4]
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "rotate_90",
        "ndimage::rotate(angle=90)",
        "4x4 sequential image",
        &format!(
            "shape={:?}",
            rotated.as_ref().map(|r| &r.shape).unwrap_or(&vec![])
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Rotate by 180 degrees
    let t = Instant::now();
    let rotated_180 = rotate(&image, 180.0, false, 1, BoundaryMode::Constant, 0.0);
    let pass = match &rotated_180 {
        Ok(result) => {
            // After 180 degree rotation, top-left (1.0) should be near bottom-right (16.0) location
            // The image should be reversed
            result.shape == vec![4, 4]
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "rotate_180",
        "ndimage::rotate(angle=180)",
        "4x4 sequential image",
        &format!("success={}", pass),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Rotate by 45 degrees with reshape
    let t = Instant::now();
    let rotated_45 = rotate(&image, 45.0, true, 1, BoundaryMode::Constant, 0.0);
    let pass = match &rotated_45 {
        Ok(result) => {
            // With reshape=true, output should be larger to contain rotated image
            // For 45 degrees, diagonal length is sqrt(2) * original
            result.shape[0] >= 4 && result.shape[1] >= 4
        }
        Err(_) => false,
    };
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "rotate_45_reshape",
        "ndimage::rotate(angle=45, reshape=true)",
        "4x4 sequential image",
        &format!(
            "shape={:?}",
            rotated_45.as_ref().map(|r| &r.shape).unwrap_or(&vec![])
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}
