use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_fft::{FftOptions, cross_spectral_density, dct, fft, fft2, ifft, irfft, rfft};
use std::hint::black_box;

type Complex64 = (f64, f64);

const SIZES_1D: &[usize] = &[16, 64, 256, 1024];
const MIXED_RADIX_1D: &[usize] = &[720, 1000, 1080, 1500, 1920, 3000, 5000, 10000];
const SIZES_2D: &[(usize, usize)] = &[(8, 8), (16, 16), (32, 32)];
const CSD_SIZES: &[usize] = &[4096, 65536];

// Per SPEC §17: FFT transform p95 ≤ 210ms
// Baseline sizes for conformance testing
const BASELINE_1D: &[usize] = &[1024, 4096, 16384, 65536, 262144];
const BASELINE_2D: &[(usize, usize)] = &[(64, 64), (128, 128), (256, 256), (512, 512)];

fn make_complex_input(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            ((2.0 * std::f64::consts::PI * t).sin(), 0.0)
        })
        .collect()
}

fn make_real_input(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * t).sin()
        })
        .collect()
}

fn make_real_pair(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * t).sin() + 0.25 * (13.0 * std::f64::consts::PI * t).cos()
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (3.0 * std::f64::consts::PI * t).cos() - 0.5 * (17.0 * std::f64::consts::PI * t).sin()
        })
        .collect();
    (x, y)
}

fn default_opts() -> FftOptions {
    FftOptions::default()
}

fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_forward");

    for &n in SIZES_1D {
        let input = make_complex_input(n);
        group.bench_with_input(BenchmarkId::new("fft", n), &input, |b, input| {
            let opts = default_opts();
            b.iter(|| {
                let out = fft(black_box(input), black_box(&opts)).expect("fft");
                black_box(out.len());
            });
        });
    }

    group.finish();
}

fn bench_mixed_radix_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_mixed_radix");
    group.sample_size(50);

    for &n in MIXED_RADIX_1D {
        let input = make_complex_input(n);
        group.bench_with_input(BenchmarkId::new("fft", n), &input, |b, input| {
            let opts = default_opts();
            b.iter(|| {
                let out = fft(black_box(input), black_box(&opts)).expect("mixed-radix fft");
                black_box(out.len());
            });
        });
    }

    group.finish();
}

fn bench_ifft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_inverse");

    for &n in SIZES_1D {
        let input = make_complex_input(n);
        let spectrum = fft(&input, &default_opts()).expect("fft");
        group.bench_with_input(BenchmarkId::new("ifft", n), &spectrum, |b, spectrum| {
            let opts = default_opts();
            b.iter(|| {
                let out = ifft(black_box(spectrum), black_box(&opts)).expect("ifft");
                black_box(out.len());
            });
        });
    }

    group.finish();
}

fn bench_rfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_real");

    let sizes: &[usize] = &[256, 1024, 4096, 16384, 65536];
    for &n in sizes {
        let input = make_real_input(n);
        group.bench_with_input(BenchmarkId::new("rfft", n), &input, |b, input| {
            let opts = default_opts();
            b.iter(|| {
                let out = rfft(black_box(input), black_box(&opts)).expect("rfft");
                black_box(out.len());
            });
        });
    }

    group.finish();
}

fn bench_dct(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_dct");
    group.sample_size(50);

    let sizes: &[usize] = &[2048, 4096, 8192, 16384];
    for &n in sizes {
        let input = make_real_input(n);
        group.bench_with_input(BenchmarkId::new("dct_ii", n), &input, |b, input| {
            let opts = default_opts();
            b.iter(|| {
                let out = dct(black_box(input), black_box(&opts)).expect("dct");
                black_box(out.len());
            });
        });
    }

    group.finish();
}

fn bench_irfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_irfft");

    for &n in SIZES_1D {
        let input = make_real_input(n);
        let spectrum = rfft(&input, &default_opts()).expect("rfft");
        group.bench_with_input(
            BenchmarkId::new("irfft", n),
            &(spectrum, n),
            |b, (spectrum, n)| {
                let opts = default_opts();
                b.iter(|| {
                    let out = irfft(black_box(spectrum), black_box(Some(*n)), black_box(&opts))
                        .expect("irfft");
                    black_box(out.len());
                });
            },
        );
    }

    group.finish();
}

fn bench_fft2(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_2d");

    for &(rows, cols) in SIZES_2D {
        let n = rows * cols;
        let input = make_complex_input(n);
        let shape = (rows, cols);
        group.bench_with_input(
            BenchmarkId::new("fft2", format!("{rows}x{cols}")),
            &(input, shape),
            |b, (input, shape)| {
                let opts = default_opts();
                b.iter(|| {
                    let out =
                        fft2(black_box(input), black_box(*shape), black_box(&opts)).expect("fft2");
                    black_box(out.len());
                });
            },
        );
    }

    group.finish();
}

fn bench_cross_spectral_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_helpers");

    for &n in CSD_SIZES {
        let pair = make_real_pair(n);
        group.bench_with_input(
            BenchmarkId::new("cross_spectral_density", n),
            &pair,
            |b, (x, y)| {
                b.iter(|| {
                    let out =
                        cross_spectral_density(black_box(x), black_box(y), black_box(48_000.0))
                            .expect("cross_spectral_density");
                    black_box(out.1.len());
                });
            },
        );
    }

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════════════
// BASELINE BENCHMARKS - Per SPEC §17
// Run with: cargo bench --bench fft_bench -- baseline
// ══════════════════════════════════════════════════════════════════════════════

fn bench_baseline_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_fft");
    group.sample_size(50);

    for &n in BASELINE_1D {
        let input = make_complex_input(n);
        group.bench_with_input(BenchmarkId::new("fft", n), &input, |b, input| {
            let opts = default_opts();
            b.iter(|| {
                let out = fft(black_box(input), black_box(&opts)).expect("fft");
                black_box(out.len());
            });
        });
    }

    group.finish();
}

fn bench_baseline_rfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_rfft");
    group.sample_size(50);

    for &n in BASELINE_1D {
        let input = make_real_input(n);
        group.bench_with_input(BenchmarkId::new("rfft", n), &input, |b, input| {
            let opts = default_opts();
            b.iter(|| {
                let out = rfft(black_box(input), black_box(&opts)).expect("rfft");
                black_box(out.len());
            });
        });
    }

    group.finish();
}

fn bench_baseline_fft2(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_fft2");
    group.sample_size(50);

    for &(rows, cols) in BASELINE_2D {
        let n = rows * cols;
        let input = make_complex_input(n);
        let shape = (rows, cols);
        group.bench_with_input(
            BenchmarkId::new("fft2", format!("{rows}x{cols}")),
            &(input, shape),
            |b, (input, shape)| {
                let opts = default_opts();
                b.iter(|| {
                    let out =
                        fft2(black_box(input), black_box(*shape), black_box(&opts)).expect("fft2");
                    black_box(out.len());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fft,
    bench_mixed_radix_fft,
    bench_ifft,
    bench_rfft,
    bench_dct,
    bench_irfft,
    bench_fft2,
    bench_cross_spectral_density
);

criterion_group!(
    baseline_benches,
    bench_baseline_fft,
    bench_baseline_rfft,
    bench_baseline_fft2
);

criterion_main!(benches, baseline_benches);
