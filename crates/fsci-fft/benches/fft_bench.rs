use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_fft::plan::{
    PlanFingerprint, PlanKey, PlanMetadata, PlanningStrategy, clear_shared_plan_cache,
    lookup_shared_plan, store_shared_plan, touch_shared_plan,
};
use fsci_fft::{
    FFT_PRIME_FORCE_BLUESTEIN, FftOptions, Normalization, TransformKind, cross_spectral_density,
    dct, fft, fft2, fftshift, ifft, irfft, rfft,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;

type Complex64 = (f64, f64);

const SIZES_1D: &[usize] = &[16, 64, 256, 1024];
const MIXED_RADIX_1D: &[usize] = &[720, 1000, 1080, 1500, 1920, 3000, 5000, 10000];
const MIXED_RADIX_17_1D: &[usize] = &[1088, 2176, 4352, 8704, 17408];
const PRIME_RADER_1D: &[usize] = &[1093, 1373, 1409, 2113];
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

fn bench_mixed_radix17_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_mixed_radix17");
    group.sample_size(10);

    for &n in MIXED_RADIX_17_1D {
        let input = make_complex_input(n);
        group.bench_with_input(BenchmarkId::new("fft", n), &input, |b, input| {
            let opts = default_opts();
            b.iter(|| {
                let out = fft(black_box(input), black_box(&opts)).expect("factor-17 fft");
                black_box(out.len());
            });
        });
    }

    group.finish();
}

fn bench_prime_rader_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_prime_rader");
    group.sample_size(10);

    for &n in PRIME_RADER_1D {
        let input = make_complex_input(n);
        group.bench_with_input(BenchmarkId::new("current_rader", n), &input, |b, input| {
            let opts = default_opts();
            b.iter(|| {
                FFT_PRIME_FORCE_BLUESTEIN.store(false, Ordering::Relaxed);
                let out = fft(black_box(input), black_box(&opts)).expect("rader fft");
                black_box(out.len());
            });
        });
        group.bench_with_input(BenchmarkId::new("orig_bluestein", n), &input, |b, input| {
            let opts = default_opts();
            b.iter(|| {
                FFT_PRIME_FORCE_BLUESTEIN.store(true, Ordering::Relaxed);
                let out = fft(black_box(input), black_box(&opts)).expect("bluestein fft");
                black_box(out.len());
            });
        });
    }

    FFT_PRIME_FORCE_BLUESTEIN.store(false, Ordering::Relaxed);
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

fn fftshift_last_axis_scalar_reference(input: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let data = input.to_vec();
    let k = cols / 2;
    let mut out = data.clone();
    for (flat, slot) in out.iter_mut().enumerate() {
        let coord = flat % cols;
        let src_coord = (coord + cols - k) % cols;
        let src = flat - coord + src_coord;
        *slot = data[src];
    }
    debug_assert_eq!(out.len(), rows * cols);
    out
}

fn bench_fftshift_contiguous_axis(c: &mut Criterion) {
    const ROWS: usize = 1024;
    const COLS: usize = 1024;
    let input = make_real_input(ROWS * COLS);
    let shape = [ROWS, COLS];
    let axes = [1];
    let candidate = fftshift(&input, &shape, Some(&axes)).expect("fftshift");
    let original = fftshift_last_axis_scalar_reference(&input, ROWS, COLS);
    assert_eq!(candidate, original);

    let mut group = c.benchmark_group("fftshift_contiguous_axis");
    group.bench_function("candidate/1024x1024", |bencher| {
        bencher.iter(|| {
            black_box(
                fftshift(
                    black_box(&input),
                    black_box(shape.as_slice()),
                    Some(black_box(axes.as_slice())),
                )
                .expect("contiguous fftshift"),
            )
        })
    });
    group.bench_function("original/1024x1024", |bencher| {
        bencher.iter(|| {
            black_box(fftshift_last_axis_scalar_reference(
                black_box(&input),
                black_box(ROWS),
                black_box(COLS),
            ))
        })
    });
    group.finish();
}

fn bench_plan_cache_hit(c: &mut Criterion) {
    clear_shared_plan_cache();
    let key = PlanKey::new(
        TransformKind::Fftn,
        vec![64, 64, 64],
        vec![0, 1, 2],
        Normalization::Backward,
        false,
    );
    let metadata = PlanMetadata {
        key: key.clone(),
        fingerprint: PlanFingerprint {
            radix_path: vec![2; 18],
            estimated_flops: 23_592_960,
            scratch_bytes: 262_144 * std::mem::size_of::<Complex64>(),
        },
        generated_by: PlanningStrategy::EstimateOnly,
    };
    assert!(store_shared_plan(metadata));
    assert!(touch_shared_plan(&key));
    assert!(lookup_shared_plan(&key).is_some());

    let mut group = c.benchmark_group("fft_plan_cache_hit");
    group.bench_function("clone_free_touch", |bencher| {
        bencher.iter(|| black_box(touch_shared_plan(black_box(&key))))
    });
    group.bench_function("former_metadata_clone", |bencher| {
        bencher.iter(|| black_box(lookup_shared_plan(black_box(&key)).is_some()))
    });
    group.finish();
    clear_shared_plan_cache();
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
    bench_mixed_radix17_fft,
    bench_prime_rader_fft,
    bench_ifft,
    bench_rfft,
    bench_dct,
    bench_irfft,
    bench_fft2,
    bench_cross_spectral_density,
    bench_fftshift_contiguous_axis,
    bench_plan_cache_hit
);

criterion_group!(
    baseline_benches,
    bench_baseline_fft,
    bench_baseline_rfft,
    bench_baseline_fft2
);

criterion_main!(benches, baseline_benches);
