use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use fsci_fft::{FftOptions, fft, fft2, ifft, irfft, rfft};

type Complex64 = (f64, f64);

const SIZES_1D: &[usize] = &[16, 64, 256, 1024];
const SIZES_2D: &[(usize, usize)] = &[(8, 8), (16, 16), (32, 32)];

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

    for &n in SIZES_1D {
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
                    let out =
                        irfft(black_box(spectrum), black_box(Some(*n)), black_box(&opts)).expect("irfft");
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

criterion_group!(benches, bench_fft, bench_ifft, bench_rfft, bench_irfft, bench_fft2);
criterion_main!(benches);
