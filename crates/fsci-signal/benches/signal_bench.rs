use criterion::{Criterion, criterion_group, criterion_main};
use fsci_signal::{
    ConvolveMode, FirWindow, SosSection, coherence, csd, cwt, fftconvolve, filtfilt, firls, firwin,
    freqz, lfilter, medfilt, mfcc, order_filter, remez, ricker, sosfilt, welch,
};
use std::hint::black_box;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

fn deterministic_signal(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let t = i as f64 / len as f64;
            (37.0 * t).sin() + 0.35 * (91.0 * t).cos() + 0.1 * ((i * 17 % 29) as f64 - 14.0)
        })
        .collect()
}

fn bench_convolution(c: &mut Criterion) {
    let x = deterministic_signal(4096);
    let h = deterministic_signal(257);

    c.bench_function("convolution/fftconvolve/4096x257_same", |b| {
        b.iter(|| {
            black_box(fftconvolve(
                black_box(&x),
                black_box(&h),
                black_box(ConvolveMode::Same),
            ))
        })
    });
}

fn bench_filtering(c: &mut Criterion) {
    let x = deterministic_signal(4096);
    let b_coeffs = vec![0.067_455_27, 0.134_910_55, 0.067_455_27];
    let a_coeffs = vec![1.0, -1.142_980_5, 0.412_801_6];
    let sos: Vec<SosSection> = vec![
        [
            0.067_455_27,
            0.134_910_55,
            0.067_455_27,
            1.0,
            -1.142_980_5,
            0.412_801_6,
        ],
        [
            0.067_455_27,
            0.134_910_55,
            0.067_455_27,
            1.0,
            -1.142_980_5,
            0.412_801_6,
        ],
    ];

    c.bench_function("filtering/lfilter/4096_biquad", |b| {
        b.iter(|| {
            black_box(lfilter(
                black_box(&b_coeffs),
                black_box(&a_coeffs),
                black_box(&x),
                black_box(None),
            ))
        })
    });

    c.bench_function("filtering/filtfilt/4096_biquad", |b| {
        b.iter(|| {
            black_box(filtfilt(
                black_box(&b_coeffs),
                black_box(&a_coeffs),
                black_box(&x),
            ))
        })
    });

    c.bench_function("filtering/sosfilt/4096_two_sections", |b| {
        b.iter(|| black_box(sosfilt(black_box(&sos), black_box(&x))))
    });
}

fn bench_spectral(c: &mut Criterion) {
    let x = deterministic_signal(4096);

    c.bench_function("spectral/welch/4096_w256_o128", |b| {
        b.iter(|| {
            black_box(welch(
                black_box(&x),
                black_box(1.0),
                black_box(Some("hann")),
                black_box(Some(256)),
                black_box(Some(128)),
            ))
        })
    });

    let long_x = deterministic_signal(65_536);
    let long_y: Vec<f64> = (0..65_536)
        .map(|i| {
            let t = i as f64 / 65_536.0;
            0.8 * (41.0 * t + 0.37).sin()
                + 0.4 * (103.0 * t).cos()
                + 0.05 * ((i * 31 % 43) as f64 - 21.0)
        })
        .collect();

    c.bench_function("spectral/coherence/65536_w1024_o512", |b| {
        b.iter(|| {
            black_box(coherence(
                black_box(&long_x),
                black_box(&long_y),
                black_box(1.0),
                black_box(Some("hann")),
                black_box(Some(1024)),
                black_box(Some(512)),
            ))
        })
    });
}

fn deterministic_coherence_pair(len: usize) -> (Vec<f64>, Vec<f64>) {
    let x = deterministic_signal(len);
    let y: Vec<f64> = (0..len)
        .map(|i| {
            let t = i as f64 / len as f64;
            0.8 * (41.0 * t + 0.37).sin()
                + 0.4 * (103.0 * t).cos()
                + 0.05 * ((i * 31 % 43) as f64 - 21.0)
        })
        .collect();
    (x, y)
}

fn coherence_via_csd_composition(
    x: &[f64],
    y: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
) -> Vec<f64> {
    let pxy = csd(x, y, fs, window, nperseg, noverlap).expect("Pxy CSD");
    let pxx = csd(x, x, fs, window, nperseg, noverlap).expect("Pxx CSD");
    let pyy = csd(y, y, fs, window, nperseg, noverlap).expect("Pyy CSD");
    pxy.csd
        .iter()
        .zip(pxx.csd.iter().zip(pyy.csd.iter()))
        .map(|(&(re, im), (&(xx, _), &(yy, _)))| {
            let denom = xx * yy;
            if denom.abs() < 1e-30 {
                0.0
            } else {
                ((re * re + im * im) / denom).clamp(0.0, 1.0)
            }
        })
        .collect()
}

fn scipy_coherence_duration(
    len: usize,
    nperseg: usize,
    noverlap: usize,
    iters: u64,
) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.signal as sig

length = int(sys.argv[1])
nperseg = int(sys.argv[2])
noverlap = int(sys.argv[3])
iters = int(sys.argv[4])
i = np.arange(length, dtype=np.float64)
t = i / float(length)
x = np.sin(37.0 * t) + 0.35 * np.cos(91.0 * t) + 0.1 * ((np.mod(i * 17.0, 29.0)) - 14.0)
y = 0.8 * np.sin(41.0 * t + 0.37) + 0.4 * np.cos(103.0 * t) + 0.05 * ((np.mod(i * 31.0, 43.0)) - 21.0)
sig.coherence(x, y, fs=1.0, window="hann", nperseg=nperseg, noverlap=noverlap)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    f, cxy = sig.coherence(x, y, fs=1.0, window="hann", nperseg=nperseg, noverlap=noverlap)
    checksum += float(f[-1]) + float(cxy[len(cxy) // 2])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args([
            "-",
            &len.to_string(),
            &nperseg.to_string(),
            &noverlap.to_string(),
            &iters.to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy coherence oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy coherence oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy coherence oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy coherence oracle");
    if !output.status.success() {
        eprintln!(
            "scipy coherence oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy coherence timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy coherence timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn scipy_signal_available() -> bool {
    let script = b"import scipy.signal as sig\nassert sig.coherence is not None\n";
    let Ok(mut child) = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    let Some(mut stdin) = child.stdin.take() else {
        return false;
    };
    if stdin.write_all(script).is_err() {
        return false;
    }
    drop(stdin);
    child.wait().map(|status| status.success()).unwrap_or(false)
}

fn bench_coherence_gauntlet_scipy(c: &mut Criterion) {
    let mut group = c.benchmark_group("coherence_gauntlet_scipy");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let (x, y) = deterministic_coherence_pair(65_536);
    group.bench_function("65536_w1024_o512_rust_fused", |b| {
        b.iter(|| {
            black_box(
                coherence(
                    black_box(&x),
                    black_box(&y),
                    black_box(1.0),
                    black_box(Some("hann")),
                    black_box(Some(1024)),
                    black_box(Some(512)),
                )
                .expect("fused coherence"),
            )
        })
    });
    group.bench_function("65536_w1024_o512_rust_compositional_csd", |b| {
        b.iter(|| {
            black_box(coherence_via_csd_composition(
                black_box(&x),
                black_box(&y),
                black_box(1.0),
                black_box(Some("hann")),
                black_box(Some(1024)),
                black_box(Some(512)),
            ))
        })
    });
    if scipy_signal_available() {
        group.bench_function("65536_w1024_o512_scipy_coherence", |b| {
            b.iter_custom(|iters| {
                scipy_coherence_duration(65_536, 1024, 512, iters)
                    .expect("scipy coherence oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping 65536_w1024_o512_scipy_coherence: python3 cannot import scipy.signal");
    }

    group.finish();
}

fn bench_wavelets(c: &mut Criterion) {
    let x = deterministic_signal(2048);
    let widths: Vec<f64> = (1..=32).map(|w| w as f64).collect();

    c.bench_function("wavelets/cwt_ricker/2048x32", |b| {
        b.iter(|| black_box(cwt(black_box(&x), ricker, black_box(&widths))))
    });
}

fn bench_design(c: &mut Criterion) {
    let bands = vec![0.0, 0.2, 0.3, 0.5];
    let desired = vec![1.0, 1.0, 0.0, 0.0];
    let remez_desired = vec![1.0, 0.0];
    let weights = vec![1.0, 10.0];

    c.bench_function("design/firwin/513_hamming", |b| {
        b.iter(|| {
            black_box(firwin(
                black_box(513),
                black_box(&[0.2]),
                black_box(FirWindow::Hamming),
                black_box(true),
            ))
        })
    });

    c.bench_function("design/firls/257_two_band", |b| {
        b.iter(|| {
            black_box(firls(
                black_box(257),
                black_box(&bands),
                black_box(&desired),
                black_box(None),
            ))
        })
    });

    c.bench_function("design/remez/257_two_band", |b| {
        b.iter(|| {
            black_box(remez(
                black_box(257),
                black_box(&bands),
                black_box(&remez_desired),
                black_box(Some(&weights)),
            ))
        })
    });
    // Even numtaps exercises the WLS frequency-sampling fallback whose cos-basis is built
    // by the Chebyshev recurrence (frankenscipy-9l5oo) instead of one cos() per coefficient.
    c.bench_function("design/remez/256_two_band_wls", |b| {
        b.iter(|| {
            black_box(remez(
                black_box(256),
                black_box(&bands),
                black_box(&remez_desired),
                black_box(Some(&weights)),
            ))
        })
    });
}

fn bench_medfilt(c: &mut Criterion) {
    let signal = deterministic_signal(8192);
    let mut group = c.benchmark_group("medfilt");
    for &k in &[65usize, 257, 513] {
        group.bench_function(format!("8192_k{k}"), |b| {
            b.iter(|| medfilt(black_box(&signal), k).expect("medfilt"))
        });
    }
    group.finish();
}

fn bench_order_filter(c: &mut Criterion) {
    let signal = deterministic_signal(8192);
    let mut group = c.benchmark_group("order_filter");
    for &ws in &[65usize, 257] {
        group.bench_function(format!("8192_w{ws}_q25"), |b| {
            b.iter(|| order_filter(black_box(&signal), ws, ws / 4))
        });
    }
    group.finish();
}

/// freqz on a high-order FIR (128 taps) over 512 frequencies — exercises the per-coefficient
/// polynomial-on-unit-circle evaluation now using Horner (frankenscipy-9l5oo).
fn bench_freqz(c: &mut Criterion) {
    let b: Vec<f64> = deterministic_signal(128);
    let a = vec![1.0];
    c.bench_function("freqz/fir128_512", |bn| {
        bn.iter(|| black_box(freqz(black_box(&b), black_box(&a), black_box(Some(512)))))
    });
}

/// mfcc over a 16384-sample signal, frame_len 512 — the per-frame power spectrum now uses
/// fsci_fft (O(N log N)) instead of a naive O(N²) DFT (frankenscipy-9l5oo).
fn bench_mfcc(c: &mut Criterion) {
    let sig = deterministic_signal(16384);
    c.bench_function("mfcc/16384_frame512", |b| {
        b.iter(|| black_box(mfcc(black_box(&sig), 16000.0, 13, 26, 512, 256)))
    });
}

criterion_group!(
    benches,
    bench_mfcc,
    bench_freqz,
    bench_convolution,
    bench_filtering,
    bench_spectral,
    bench_coherence_gauntlet_scipy,
    bench_wavelets,
    bench_design,
    bench_medfilt,
    bench_order_filter
);
criterion_main!(benches);
