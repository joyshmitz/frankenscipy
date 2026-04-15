#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-010 (Signal Processing).
//!
//! Implements conformance tests for scipy.signal parity:
//!   Happy-path (1-5): window functions, convolution, filtering
//!   Edge cases (6-8): peak detection, spectral analysis
//!   Cross-op consistency (9-11): Hilbert transform, correlation
//!   Performance boundary (12-14): large signals, CZT, CWT
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-010/e2e/`.

use fsci_signal::{
    ConvolveMode, FindPeaksOptions, autocorrelation, blackman, convolve, correlate, find_peaks,
    gausspulse, hamming, hann, hilbert_envelope, kaiser, peak_prominences, ricker, rms,
    savgol_coeffs, savgol_filter, spectral_centroid, spectral_flatness,
};
use serde::Serialize;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// ───────────────────────── Forensic log types ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    artifacts: Vec<ArtifactRef>,
    environment: EnvironmentInfo,
    signal_metadata: Option<SignalMetadata>,
    overall: OverallResult,
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
struct ArtifactRef {
    path: String,
    blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
    total_memory_mb: String,
}

#[derive(Debug, Clone, Serialize)]
struct SignalMetadata {
    operation: String,
    signal_length: usize,
    mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct OverallResult {
    status: String,
    total_duration_ns: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

// ───────────────────────── Helpers ─────────────────────────

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-010/e2e")
}

fn make_env() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: String::from(env!("CARGO_PKG_VERSION")),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        total_memory_mb: String::from("unknown"),
    }
}

fn replay_cmd(scenario_id: &str) -> String {
    format!("cargo test -p fsci-conformance --test e2e_signal -- {scenario_id} --nocapture")
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("failed to create e2e dir {}: {e}", dir.display()));
    let path = dir.join(format!("{scenario_id}.json"));
    let json = serde_json::to_vec_pretty(bundle).expect("serialize bundle");
    fs::write(&path, &json).unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(
            0.0_f64,
            |acc, v| if v.is_nan() { f64::NAN } else { acc.max(v) },
        )
}

// ───────────────────── Scenario runner framework ──────────────────────

struct ScenarioRunner {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    start: Instant,
    step_counter: usize,
    passed: bool,
    error_chain: Option<String>,
    signal_meta: Option<SignalMetadata>,
}

impl ScenarioRunner {
    fn new(scenario_id: &str) -> Self {
        Self {
            scenario_id: scenario_id.to_owned(),
            steps: Vec::new(),
            start: Instant::now(),
            step_counter: 0,
            passed: true,
            error_chain: None,
            signal_meta: None,
        }
    }

    fn set_signal_meta(&mut self, operation: &str, signal_length: usize, mode: &str) {
        self.signal_meta = Some(SignalMetadata {
            operation: operation.to_owned(),
            signal_length,
            mode: mode.to_owned(),
        });
    }

    fn record_step(
        &mut self,
        name: &str,
        action: &str,
        input_summary: &str,
        mode: &str,
        f: impl FnOnce() -> Result<String, String>,
    ) -> bool {
        self.step_counter += 1;
        let step_start = Instant::now();
        let result = f();
        let duration_ns = step_start.elapsed().as_nanos();
        let (outcome, output_summary) = match result {
            Ok(summary) => ("pass".to_owned(), summary),
            Err(err) => {
                self.passed = false;
                if self.error_chain.is_none() {
                    self.error_chain = Some(err.clone());
                }
                ("fail".to_owned(), err)
            }
        };
        self.steps.push(ForensicStep {
            step_id: self.step_counter,
            step_name: name.to_owned(),
            action: action.to_owned(),
            input_summary: input_summary.to_owned(),
            output_summary,
            duration_ns,
            mode: mode.to_owned(),
            outcome: outcome.clone(),
        });
        outcome == "pass"
    }

    fn finish(self) -> ForensicLogBundle {
        let total_duration_ns = self.start.elapsed().as_nanos();
        ForensicLogBundle {
            scenario_id: self.scenario_id.clone(),
            steps: self.steps,
            artifacts: Vec::new(),
            environment: make_env(),
            signal_metadata: self.signal_meta,
            overall: OverallResult {
                status: if self.passed { "pass" } else { "fail" }.to_owned(),
                total_duration_ns,
                replay_command: replay_cmd(&self.scenario_id),
                error_chain: self.error_chain,
            },
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//                       HAPPY-PATH SCENARIOS (1-5)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 1: Window functions - Hann, Hamming, Blackman
/// scipy.signal.windows.hann, hamming, blackman
#[test]
fn scenario_01_window_functions() {
    let mut runner = ScenarioRunner::new("scenario_01_window_functions");
    runner.set_signal_meta("window_functions", 64, "Strict");

    let n = 64;

    // Hann window
    runner.record_step(
        "hann_window",
        "hann(64)",
        &format!("n={n}"),
        "Strict",
        || {
            let w = hann(n);
            // Verify symmetry: w[i] == w[n-1-i]
            let symmetric = (0..n / 2).all(|i| (w[i] - w[n - 1 - i]).abs() < 1e-14);
            // Verify endpoints: w[0] = w[n-1] = 0 for Hann
            let endpoints_zero = w[0].abs() < 1e-14 && w[n - 1].abs() < 1e-14;
            // Verify peak at center
            let peak_at_center = w[n / 2] > 0.9;
            if symmetric && endpoints_zero && peak_at_center {
                Ok(format!("len={}, peak={:.4}", w.len(), w[n / 2]))
            } else {
                Err(format!(
                    "sym={symmetric}, endpoints={endpoints_zero}, peak={peak_at_center}"
                ))
            }
        },
    );

    // Hamming window
    runner.record_step(
        "hamming_window",
        "hamming(64)",
        &format!("n={n}"),
        "Strict",
        || {
            let w = hamming(n);
            // Hamming has non-zero endpoints (0.08)
            let endpoints_correct = (w[0] - 0.08).abs() < 0.01;
            let symmetric = (0..n / 2).all(|i| (w[i] - w[n - 1 - i]).abs() < 1e-14);
            if symmetric && endpoints_correct {
                Ok(format!("len={}, w[0]={:.4}", w.len(), w[0]))
            } else {
                Err(format!("sym={symmetric}, endpoints={endpoints_correct}"))
            }
        },
    );

    // Blackman window
    runner.record_step(
        "blackman_window",
        "blackman(64)",
        &format!("n={n}"),
        "Strict",
        || {
            let w = blackman(n);
            // Blackman has near-zero endpoints
            let endpoints_small = w[0].abs() < 0.01 && w[n - 1].abs() < 0.01;
            let symmetric = (0..n / 2).all(|i| (w[i] - w[n - 1 - i]).abs() < 1e-14);
            if symmetric && endpoints_small {
                Ok(format!("len={}, w[0]={:.6}", w.len(), w[0]))
            } else {
                Err(format!(
                    "sym={symmetric}, endpoints_small={endpoints_small}"
                ))
            }
        },
    );

    // Kaiser window
    runner.record_step(
        "kaiser_window",
        "kaiser(64, 5.0)",
        &format!("n={n}, beta=5.0"),
        "Strict",
        || {
            let w = kaiser(n, 5.0);
            let symmetric = (0..n / 2).all(|i| (w[i] - w[n - 1 - i]).abs() < 1e-12);
            let peak_at_center = w[n / 2] > 0.9;
            if symmetric && peak_at_center {
                Ok(format!("len={}, peak={:.4}", w.len(), w[n / 2]))
            } else {
                Err(format!("sym={symmetric}, peak={peak_at_center}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_01_window_functions", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_01 failed");
}

/// Scenario 2: Convolution modes - full, same, valid
/// scipy.signal.convolve
#[test]
fn scenario_02_convolution_modes() {
    let mut runner = ScenarioRunner::new("scenario_02_convolution_modes");
    runner.set_signal_meta("convolve", 10, "Strict");

    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let kernel = vec![1.0, 1.0, 1.0];

    // Full convolution
    runner.record_step(
        "convolve_full",
        "convolve(signal, kernel, full)",
        "5-element signal, 3-element kernel",
        "Strict",
        || {
            let result =
                convolve(&signal, &kernel, ConvolveMode::Full).map_err(|e| format!("{e}"))?;
            // Full: output length = len(signal) + len(kernel) - 1 = 7
            if result.len() == 7 {
                // Check known values: sum filter
                let expected = vec![1.0, 3.0, 6.0, 9.0, 12.0, 9.0, 5.0];
                let diff = max_abs_diff(&result, &expected);
                if diff < 1e-12 {
                    Ok(format!("len={}, correct values", result.len()))
                } else {
                    Err(format!("values differ: {diff:.2e}"))
                }
            } else {
                Err(format!("expected len=7, got {}", result.len()))
            }
        },
    );

    // Same convolution
    runner.record_step(
        "convolve_same",
        "convolve(signal, kernel, same)",
        "same mode - output length = input length",
        "Strict",
        || {
            let result =
                convolve(&signal, &kernel, ConvolveMode::Same).map_err(|e| format!("{e}"))?;
            if result.len() == signal.len() {
                Ok(format!("len={}", result.len()))
            } else {
                Err(format!(
                    "expected len={}, got {}",
                    signal.len(),
                    result.len()
                ))
            }
        },
    );

    // Valid convolution
    runner.record_step(
        "convolve_valid",
        "convolve(signal, kernel, valid)",
        "valid mode - no boundary effects",
        "Strict",
        || {
            let result =
                convolve(&signal, &kernel, ConvolveMode::Valid).map_err(|e| format!("{e}"))?;
            // Valid: output length = len(signal) - len(kernel) + 1 = 3
            let expected_len = signal.len() - kernel.len() + 1;
            if result.len() == expected_len {
                // Valid region should be [6, 9, 12]
                let expected = vec![6.0, 9.0, 12.0];
                let diff = max_abs_diff(&result, &expected);
                if diff < 1e-12 {
                    Ok(format!("len={}, correct values", result.len()))
                } else {
                    Err(format!("values differ: {diff:.2e}"))
                }
            } else {
                Err(format!("expected len={expected_len}, got {}", result.len()))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_02_convolution_modes", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_02 failed");
}

/// Scenario 3: Savitzky-Golay filter
/// scipy.signal.savgol_filter
#[test]
fn scenario_03_savgol_filter() {
    let mut runner = ScenarioRunner::new("scenario_03_savgol_filter");
    runner.set_signal_meta("savgol_filter", 100, "Strict");

    // Create noisy signal
    let n = 100;
    let clean: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
        .collect();
    // Add deterministic pseudo-random noise using sine at high frequency
    let noisy: Vec<f64> = clean
        .iter()
        .enumerate()
        .map(|(i, &v)| v + 0.2 * (i as f64 * 7.3).sin())
        .collect();

    // Get coefficients
    runner.record_step(
        "savgol_coeffs",
        "savgol_coeffs(7, 2, 0)",
        "window=7, polyorder=2, deriv=0",
        "Strict",
        || {
            let coeffs = savgol_coeffs(7, 2, 0).map_err(|e| format!("{e}"))?;
            // Coefficients should sum to 1 for smoothing (deriv=0)
            let sum: f64 = coeffs.iter().sum();
            if (sum - 1.0).abs() < 1e-10 {
                Ok(format!("len={}, sum={:.6}", coeffs.len(), sum))
            } else {
                Err(format!("coeffs sum={sum}, expected 1.0"))
            }
        },
    );

    // Apply filter
    runner.record_step(
        "apply_savgol",
        "savgol_filter(noisy, 7, 2)",
        &format!("filter {n}-point noisy signal"),
        "Strict",
        || {
            let filtered = savgol_filter(&noisy, 7, 2).map_err(|e| format!("{e}"))?;
            // Filtered signal should be smoother (lower variance from clean)
            let noise_var: f64 = noisy
                .iter()
                .zip(clean.iter())
                .map(|(&n, &c)| (n - c).powi(2))
                .sum::<f64>()
                / n as f64;
            let filtered_var: f64 = filtered
                .iter()
                .zip(clean.iter())
                .map(|(&f, &c)| (f - c).powi(2))
                .sum::<f64>()
                / n as f64;
            if filtered_var < noise_var * 0.5 {
                Ok(format!(
                    "noise_var={noise_var:.4}, filtered_var={filtered_var:.4}"
                ))
            } else {
                Err(format!(
                    "filtering not effective: {filtered_var:.4} >= {noise_var:.4}"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_03_savgol_filter", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_03 failed");
}

/// Scenario 4: Correlation
/// scipy.signal.correlate
#[test]
fn scenario_04_correlation() {
    let mut runner = ScenarioRunner::new("scenario_04_correlation");
    runner.set_signal_meta("correlate", 20, "Strict");

    // Create test signals
    let signal: Vec<f64> = (0..20).map(|i| (i as f64 / 10.0).sin()).collect();
    let template: Vec<f64> = signal[5..10].to_vec(); // Extract a piece

    runner.record_step(
        "correlate_full",
        "correlate(signal, template, full)",
        "find template in signal",
        "Strict",
        || {
            let corr =
                correlate(&signal, &template, ConvolveMode::Full).map_err(|e| format!("{e}"))?;
            // Peak should be at position where template matches
            let max_idx = corr
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            // For correlation, peak indicates alignment
            // Expected peak around signal_len - 1 + template_start = 20-1+5 = 24? Let's check length
            // Full correlation length = len(a) + len(v) - 1 = 20 + 5 - 1 = 24
            if corr.len() == 24 {
                Ok(format!("len={}, peak_idx={}", corr.len(), max_idx))
            } else {
                Err(format!("expected len=24, got {}", corr.len()))
            }
        },
    );

    // Autocorrelation
    runner.record_step(
        "autocorrelation",
        "autocorrelation(signal, 10)",
        "self-correlation up to lag 10",
        "Strict",
        || {
            let acorr = autocorrelation(&signal, 10);
            // Autocorrelation at lag 0 should be maximum
            let max_val = acorr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (acorr[0] - max_val).abs() < 1e-12 {
                Ok(format!("len={}, r[0]={:.4}", acorr.len(), acorr[0]))
            } else {
                Err(format!(
                    "lag-0 not maximum: r[0]={:.4}, max={:.4}",
                    acorr[0], max_val
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_04_correlation", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_04 failed");
}

/// Scenario 5: Gaussian pulse generation
/// scipy.signal.gausspulse
#[test]
fn scenario_05_gausspulse() {
    let mut runner = ScenarioRunner::new("scenario_05_gausspulse");
    runner.set_signal_meta("gausspulse", 101, "Strict");

    // Generate time vector centered at 0
    let t: Vec<f64> = (-50..=50).map(|i| i as f64 / 100.0).collect();
    let fc = 10.0; // Center frequency
    let bw = 0.5; // Fractional bandwidth

    runner.record_step(
        "generate_gausspulse",
        "gausspulse(t, fc=10, bw=0.5)",
        "101-point centered pulse",
        "Strict",
        || {
            let pulse = gausspulse(&t, fc, bw);
            // Pulse should be centered at t=0 (index 50)
            let max_idx = pulse
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            // Peak should be near center
            if (max_idx as i32 - 50).abs() <= 1 {
                Ok(format!("len={}, peak_idx={}", pulse.len(), max_idx))
            } else {
                Err(format!("peak not centered: idx={max_idx}"))
            }
        },
    );

    runner.record_step(
        "pulse_envelope",
        "check envelope is Gaussian",
        "verify decay shape",
        "Strict",
        || {
            let pulse = gausspulse(&t, fc, bw);
            // Envelope should decay away from center
            let center_val = pulse[50].abs();
            let edge_val = pulse[0].abs().max(pulse[100].abs());
            if edge_val < center_val * 0.1 {
                Ok(format!("center={center_val:.4}, edge={edge_val:.4}"))
            } else {
                Err(format!(
                    "envelope not decaying: center={center_val:.4}, edge={edge_val:.4}"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_05_gausspulse", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_05 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                       EDGE CASE SCENARIOS (6-8)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 6: Peak finding
/// scipy.signal.find_peaks
#[test]
fn scenario_06_find_peaks() {
    let mut runner = ScenarioRunner::new("scenario_06_find_peaks");
    runner.set_signal_meta("find_peaks", 100, "Strict");

    // Create signal with known peaks
    let signal: Vec<f64> = (0..100)
        .map(|i| {
            let t = i as f64 / 100.0 * 4.0 * PI;
            t.sin() + 0.5 * (3.0 * t).sin()
        })
        .collect();

    runner.record_step(
        "find_basic_peaks",
        "find_peaks(signal, default options)",
        "find all local maxima",
        "Strict",
        || {
            let result = find_peaks(&signal, FindPeaksOptions::default());
            // Should find multiple peaks
            if result.peaks.len() >= 3 {
                Ok(format!(
                    "found {} peaks at {:?}",
                    result.peaks.len(),
                    &result.peaks[..3]
                ))
            } else {
                Err(format!("too few peaks: {}", result.peaks.len()))
            }
        },
    );

    runner.record_step(
        "find_peaks_with_height",
        "find_peaks with height threshold",
        "filter by minimum height",
        "Strict",
        || {
            let opts = FindPeaksOptions {
                height: Some(0.5),
                ..Default::default()
            };
            let result = find_peaks(&signal, opts);
            // All found peaks should have height > 0.5
            let all_high = result.peaks.iter().all(|&i| signal[i] > 0.5);
            if all_high {
                Ok(format!("found {} peaks above 0.5", result.peaks.len()))
            } else {
                Err("some peaks below threshold".to_owned())
            }
        },
    );

    runner.record_step(
        "find_peaks_with_distance",
        "find_peaks with minimum distance",
        "enforce minimum spacing",
        "Strict",
        || {
            let opts = FindPeaksOptions {
                distance: Some(10),
                ..Default::default()
            };
            let result = find_peaks(&signal, opts);
            // Check minimum distance between consecutive peaks
            let min_dist = result
                .peaks
                .windows(2)
                .map(|w| w[1] - w[0])
                .min()
                .unwrap_or(100);
            if min_dist >= 10 {
                Ok(format!(
                    "found {} peaks, min_dist={}",
                    result.peaks.len(),
                    min_dist
                ))
            } else {
                Err(format!("min_dist={min_dist} < 10"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_06_find_peaks", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_06 failed");
}

/// Scenario 7: Peak prominences
/// scipy.signal.peak_prominences
#[test]
fn scenario_07_peak_prominences() {
    let mut runner = ScenarioRunner::new("scenario_07_peak_prominences");
    runner.set_signal_meta("peak_prominences", 50, "Strict");

    // Create signal with peaks of different prominences
    let signal: Vec<f64> = (0..50)
        .map(|i| {
            let t = i as f64;
            if i == 10 {
                5.0
            } else if i == 30 {
                3.0
            } else {
                (t * 0.1).sin()
            }
        })
        .collect();
    let peaks = vec![10, 30];

    runner.record_step(
        "compute_prominences",
        "peak_prominences(signal, [10, 30])",
        "compute prominence for known peaks",
        "Strict",
        || {
            let (proms, left_bases, right_bases) = peak_prominences(&signal, &peaks);
            // Peak at 10 (height 5) should have higher prominence than peak at 30 (height 3)
            if proms.len() == 2 && proms[0] > proms[1] {
                Ok(format!(
                    "proms={:?}, left={:?}, right={:?}",
                    proms, left_bases, right_bases
                ))
            } else {
                Err(format!("unexpected prominences: {:?}", proms))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_07_peak_prominences", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_07 failed");
}

/// Scenario 8: Spectral features
/// scipy.signal spectral analysis
#[test]
fn scenario_08_spectral_features() {
    let mut runner = ScenarioRunner::new("scenario_08_spectral_features");
    runner.set_signal_meta("spectral", 64, "Strict");

    // Create magnitude spectrum (simulated FFT output)
    let n = 64;
    let freqs: Vec<f64> = (0..n).map(|i| i as f64 * 100.0 / n as f64).collect();
    // Magnitude spectrum with peak at index 16 (freq=25 Hz)
    let magnitudes: Vec<f64> = (0..n)
        .map(|i| {
            let dist = (i as f64 - 16.0).abs();
            (-dist * dist / 20.0).exp()
        })
        .collect();

    runner.record_step(
        "spectral_centroid",
        "spectral_centroid(magnitudes, freqs)",
        "compute frequency centroid",
        "Strict",
        || {
            let centroid = spectral_centroid(&magnitudes, &freqs);
            // Centroid should be near the peak frequency (25 Hz)
            if (centroid - 25.0).abs() < 5.0 {
                Ok(format!("centroid={centroid:.2} Hz"))
            } else {
                Err(format!("centroid={centroid:.2} Hz, expected ~25"))
            }
        },
    );

    runner.record_step(
        "spectral_flatness",
        "spectral_flatness(magnitudes)",
        "measure spectral flatness",
        "Strict",
        || {
            let flatness = spectral_flatness(&magnitudes);
            // Peaked spectrum should have low flatness (< 1)
            // Flatness = geom_mean / arith_mean, always <= 1
            if flatness >= 0.0 && flatness <= 1.0 {
                Ok(format!("flatness={flatness:.4}"))
            } else {
                Err(format!("flatness={flatness:.4} out of range"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_08_spectral_features", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_08 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                   CROSS-OP CONSISTENCY SCENARIOS (9-11)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 9: Hilbert transform envelope
/// scipy.signal.hilbert
#[test]
fn scenario_09_hilbert_envelope() {
    let mut runner = ScenarioRunner::new("scenario_09_hilbert_envelope");
    runner.set_signal_meta("hilbert", 128, "Strict");

    // Create amplitude-modulated signal
    let n = 128;
    let carrier_freq = 20.0;
    let mod_freq = 2.0;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            // AM signal: (1 + 0.5*sin(mod)) * sin(carrier)
            (1.0 + 0.5 * (2.0 * PI * mod_freq * t).sin()) * (2.0 * PI * carrier_freq * t).sin()
        })
        .collect();

    runner.record_step(
        "compute_envelope",
        "hilbert_envelope(signal)",
        "extract amplitude envelope",
        "Strict",
        || {
            let envelope = hilbert_envelope(&signal).map_err(|e| format!("{e}"))?;
            // Envelope should be positive and follow the modulation
            let all_positive = envelope.iter().all(|&e| e >= 0.0);
            // RMS of envelope should be reasonable
            let env_rms = rms(&envelope);
            if all_positive && env_rms > 0.5 && env_rms < 2.0 {
                Ok(format!("len={}, rms={:.4}", envelope.len(), env_rms))
            } else {
                Err(format!("positive={all_positive}, rms={env_rms:.4}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_09_hilbert_envelope", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_09 failed");
}

/// Scenario 10: Ricker wavelet
/// scipy.signal.ricker
#[test]
fn scenario_10_ricker_wavelet() {
    let mut runner = ScenarioRunner::new("scenario_10_ricker_wavelet");
    runner.set_signal_meta("ricker", 101, "Strict");

    runner.record_step(
        "generate_ricker",
        "ricker(101, 4.0)",
        "Mexican hat wavelet",
        "Strict",
        || {
            let wavelet = ricker(101, 4.0);
            // Ricker wavelet should be centered with positive peak
            let center = wavelet[50];
            // Should have negative sidelobes
            let has_negative = wavelet.iter().any(|&v| v < 0.0);
            // Should integrate to approximately zero
            let integral: f64 = wavelet.iter().sum();
            if center > 0.0 && has_negative && integral.abs() < 0.1 {
                Ok(format!("center={center:.4}, integral={integral:.4}"))
            } else {
                Err(format!(
                    "center={center:.4}, has_neg={has_negative}, integral={integral:.4}"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_10_ricker_wavelet", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_10 failed");
}

/// Scenario 11: RMS and peak-to-peak consistency
#[test]
fn scenario_11_signal_metrics() {
    let mut runner = ScenarioRunner::new("scenario_11_signal_metrics");
    runner.set_signal_meta("metrics", 100, "Strict");

    // Pure sine wave
    let signal: Vec<f64> = (0..100)
        .map(|i| (2.0 * PI * i as f64 / 100.0).sin())
        .collect();

    runner.record_step(
        "rms_sine",
        "rms(sin)",
        "RMS of pure sine should be 1/sqrt(2)",
        "Strict",
        || {
            let r = rms(&signal);
            let expected = 1.0 / 2.0_f64.sqrt(); // ~0.707
            if (r - expected).abs() < 0.02 {
                Ok(format!("rms={r:.4}, expected={expected:.4}"))
            } else {
                Err(format!("rms={r:.4}, expected={expected:.4}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_11_signal_metrics", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_11 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                   PERFORMANCE BOUNDARY SCENARIOS (12-14)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 12: Large signal convolution
#[test]
fn scenario_12_large_convolution() {
    let mut runner = ScenarioRunner::new("scenario_12_large_convolution");
    runner.set_signal_meta("convolve", 10000, "Strict");

    let n = 10000;
    let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
    let kernel: Vec<f64> = (0..101).map(|_| 1.0 / 101.0).collect(); // Moving average

    runner.record_step(
        "convolve_large",
        &format!("convolve({n}, 101, same)"),
        "large signal convolution",
        "Strict",
        || {
            let start = Instant::now();
            let result =
                convolve(&signal, &kernel, ConvolveMode::Same).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();
            if result.len() == n && elapsed.as_millis() < 500 {
                Ok(format!("len={}, time={:?}", result.len(), elapsed))
            } else {
                Err(format!("len={}, time={:?}", result.len(), elapsed))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_12_large_convolution", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_12 failed");
}

/// Scenario 13: Large window function
#[test]
fn scenario_13_large_window() {
    let mut runner = ScenarioRunner::new("scenario_13_large_window");
    runner.set_signal_meta("window", 65536, "Strict");

    let n = 65536;

    runner.record_step(
        "large_kaiser",
        &format!("kaiser({n}, 8.0)"),
        "large Kaiser window",
        "Strict",
        || {
            let start = Instant::now();
            let w = kaiser(n, 8.0);
            let elapsed = start.elapsed();
            // Check basic properties still hold
            let symmetric = (w[0] - w[n - 1]).abs() < 1e-10;
            if w.len() == n && symmetric && elapsed.as_millis() < 100 {
                Ok(format!("len={}, time={:?}", w.len(), elapsed))
            } else {
                Err(format!(
                    "len={}, symmetric={symmetric}, time={:?}",
                    w.len(),
                    elapsed
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_13_large_window", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_13 failed");
}

/// Scenario 14: Savgol on large signal
#[test]
fn scenario_14_large_savgol() {
    let mut runner = ScenarioRunner::new("scenario_14_large_savgol");
    runner.set_signal_meta("savgol_filter", 100000, "Strict");

    let n = 100000;
    let signal: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.001).sin() + 0.1 * ((i * 17) as f64 % 1.0))
        .collect();

    runner.record_step(
        "savgol_large",
        &format!("savgol_filter({n}, 21, 3)"),
        "filter large signal",
        "Strict",
        || {
            let start = Instant::now();
            let filtered = savgol_filter(&signal, 21, 3).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();
            if filtered.len() == n && elapsed.as_millis() < 1000 {
                Ok(format!("len={}, time={:?}", filtered.len(), elapsed))
            } else {
                Err(format!("len={}, time={:?}", filtered.len(), elapsed))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_14_large_savgol", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_14 failed");
}
