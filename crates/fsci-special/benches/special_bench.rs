use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use fsci_runtime::RuntimeMode;
use fsci_special::{
    SpecialTensor, beta, erf, erfc, erfinv, gamma, gammainc, gammaln, j0, j1, rgamma, y0,
};

fn scalar(x: f64) -> SpecialTensor {
    SpecialTensor::RealScalar(x)
}

fn real_val(t: &SpecialTensor) -> f64 {
    match t {
        SpecialTensor::RealScalar(v) => *v,
        _ => panic!("expected RealScalar"),
    }
}

const GAMMA_INPUTS: &[f64] = &[0.5, 1.0, 2.5, 5.0, 10.0, 50.0, 100.0];
const ERF_INPUTS: &[f64] = &[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
const BESSEL_INPUTS: &[f64] = &[0.1, 1.0, 5.0, 10.0, 20.0, 50.0];

fn bench_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_gamma");

    for &x in GAMMA_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("gamma", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = gamma(black_box(input), RuntimeMode::Strict).expect("gamma");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_gammaln(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_gammaln");

    for &x in GAMMA_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("gammaln", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = gammaln(black_box(input), RuntimeMode::Strict).expect("gammaln");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_rgamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_rgamma");

    for &x in GAMMA_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("rgamma", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = rgamma(black_box(input), RuntimeMode::Strict).expect("rgamma");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_gammainc(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_gammainc");

    let pairs: &[(f64, f64)] = &[(1.0, 1.0), (2.0, 3.0), (5.0, 5.0), (10.0, 10.0)];
    for &(a, x) in pairs {
        let sa = scalar(a);
        let sx = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("gammainc", format!("a{a}_x{x}")),
            &(sa, sx),
            |b, (sa, sx)| {
                b.iter(|| {
                    let out = gammainc(black_box(sa), black_box(sx), RuntimeMode::Strict)
                        .expect("gammainc");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_erf(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_erf");

    for &x in ERF_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("erf", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = erf(black_box(input), RuntimeMode::Strict).expect("erf");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_erfc(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_erfc");

    for &x in ERF_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("erfc", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = erfc(black_box(input), RuntimeMode::Strict).expect("erfc");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_erfinv(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_erfinv");

    let inputs: &[f64] = &[-0.9, -0.5, 0.0, 0.5, 0.9];
    for &x in inputs {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("erfinv", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = erfinv(black_box(input), RuntimeMode::Strict).expect("erfinv");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_beta(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_beta");

    let pairs: &[(f64, f64)] = &[(0.5, 0.5), (1.0, 1.0), (2.0, 3.0), (5.0, 5.0)];
    for &(a, b_val) in pairs {
        let sa = scalar(a);
        let sb = scalar(b_val);
        group.bench_with_input(
            BenchmarkId::new("beta", format!("a{a}_b{b_val}")),
            &(sa, sb),
            |b, (sa, sb)| {
                b.iter(|| {
                    let out =
                        beta(black_box(sa), black_box(sb), RuntimeMode::Strict).expect("beta");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_bessel_j(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_bessel_j");

    for &x in BESSEL_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("j0", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = j0(black_box(input), RuntimeMode::Strict).expect("j0");
                    black_box(real_val(&out));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("j1", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = j1(black_box(input), RuntimeMode::Strict).expect("j1");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_bessel_y0(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_bessel_y");

    for &x in &[0.1, 1.0, 5.0, 10.0, 20.0] {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("y0", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = y0(black_box(input), RuntimeMode::Strict).expect("y0");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gamma,
    bench_gammaln,
    bench_rgamma,
    bench_gammainc,
    bench_erf,
    bench_erfc,
    bench_erfinv,
    bench_beta,
    bench_bessel_j,
    bench_bessel_y0
);
criterion_main!(benches);
