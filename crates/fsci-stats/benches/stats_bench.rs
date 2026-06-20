use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use fsci_stats::{
    HaltonSampler, SobolSampler, SomersDInput, acf, argsort, centered_discrepancy, ecdf, histogram,
    binned_statistic, binned_statistic_2d, binned_statistic_dd, kendalltau, l2_star_discrepancy, mannkendall, mixture_discrepancy, pacf,
    psd_welch, rand_index, siegelslopes, somersd, theilslopes, wraparound_discrepancy,
};

fn deterministic_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            (x * 0.017).sin() + (x * 0.031).cos() * 0.25 + (i % 17) as f64 * 0.001
        })
        .collect()
}

fn qmc_halton_sample(n: usize, dimension: usize) -> Vec<f64> {
    let mut sampler = HaltonSampler::new(dimension).expect("valid Halton dimension");
    sampler.sample(n)
}

fn qmc_sobol_sample(n: usize, dimension: usize) -> Vec<f64> {
    let mut sampler = SobolSampler::new(dimension).expect("valid Sobol dimension");
    sampler.sample(n)
}

fn bench_qmc_discrepancy(c: &mut Criterion) {
    let sample = qmc_halton_sample(512, 2);
    let mut group = c.benchmark_group("qmc_discrepancy");
    group.bench_function("centered/512x2", |b| {
        b.iter(|| centered_discrepancy(&sample, 2).expect("centered discrepancy"))
    });
    group.bench_function("mixture/512x2", |b| {
        b.iter(|| mixture_discrepancy(&sample, 2).expect("mixture discrepancy"))
    });
    group.bench_function("l2_star/512x2", |b| {
        b.iter(|| l2_star_discrepancy(&sample, 2).expect("l2 star discrepancy"))
    });
    group.bench_function("wraparound/512x2", |b| {
        b.iter(|| wraparound_discrepancy(&sample, 2).expect("wraparound discrepancy"))
    });
    group.finish();
}

fn bench_qmc_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("qmc_sampling");
    for n in [1024usize, 4096] {
        group.bench_function(BenchmarkId::new("halton_4d", n), |b| {
            b.iter(|| qmc_halton_sample(n, 4))
        });
        group.bench_function(BenchmarkId::new("sobol_2d", n), |b| {
            b.iter(|| qmc_sobol_sample(n, 2))
        });
    }
    group.finish();
}

fn bench_ordering_and_bins(c: &mut Criterion) {
    let data = deterministic_data(4096);
    let x_eval = deterministic_data(512);
    let data_big = deterministic_data(65536);
    let mut group = c.benchmark_group("ordering_and_bins");
    group.bench_function("argsort/4096", |b| b.iter(|| argsort(&data)));
    group.bench_function("argsort/65536", |b| b.iter(|| argsort(&data_big)));
    group.bench_function("histogram/4096x64", |b| b.iter(|| histogram(&data, 64)));
    group.bench_function("ecdf/4096x512", |b| b.iter(|| ecdf(&data, &x_eval)));
    group.finish();
}

fn bench_time_series(c: &mut Criterion) {
    let data = deterministic_data(4096);
    let mut group = c.benchmark_group("time_series");
    group.bench_function("acf/4096x64", |b| b.iter(|| acf(&data, 64)));
    group.bench_function("pacf/4096x64", |b| b.iter(|| pacf(&data, 64)));
    group.bench_function("psd_welch/4096_w128_o64", |b| {
        b.iter(|| psd_welch(&data, 128, 64, 1.0))
    });
    group.finish();
}

fn bench_rank_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_correlation");
    for &n in &[2048usize, 4096] {
        // x roughly monotone, y a noisy/tied transform — exercises ties.
        let x = deterministic_data(n);
        let y: Vec<f64> = (0..n)
            .map(|i| ((i as f64 * 0.013).cos() + (i % 11) as f64 * 0.1).round())
            .collect();
        group.bench_function(BenchmarkId::new("kendalltau", n), |b| {
            b.iter(|| kendalltau(&x, &y))
        });
        group.bench_function(BenchmarkId::new("mannkendall", n), |b| {
            b.iter(|| mannkendall(&x))
        });
    }
    group.finish();
}

fn bench_somersd(c: &mut Criterion) {
    let mut group = c.benchmark_group("somersd");
    // Distinct ranks -> an n x n contingency table, the worst case for the
    // per-cell quadrant re-summation (O((R*C)^2)).
    for &n in &[64usize, 128] {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % n) as f64).collect();
        group.bench_function(BenchmarkId::new("rankings", n), |b| {
            b.iter(|| somersd(SomersDInput::Rankings(&x, &y), None))
        });
    }
    group.finish();
}

fn bench_rand_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand_index");
    for &n in &[2000usize, 8000] {
        let lt: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let lp: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64).collect();
        group.bench_function(BenchmarkId::new("k10", n), |b| {
            b.iter(|| rand_index(&lt, &lp))
        });
    }
    group.finish();
}

/// Batch density/mass evaluation vs mapping the scalar pdf/pmf (the "original").
/// Quantifies the constant-special-function-normalizer hoist: pdf_many/pmf_many
/// compute the expensive lgamma/ln_beta normalizer ONCE instead of per point.
fn bench_distribution_batch(c: &mut Criterion) {
    use fsci_stats::{
        BetaDist, ContinuousDistribution, DiscreteDistribution, GammaDist, Hypergeometric,
    };
    let n = 4096usize;
    let mut group = c.benchmark_group("distribution_batch");

    // GammaDist: 1 lgamma/point hoisted.
    let g = GammaDist::new(2.7, 1.5);
    let gx: Vec<f64> = (0..n).map(|i| 0.01 + i as f64 * 0.01).collect();
    group.bench_function("gamma/pdf_many", |b| b.iter(|| g.pdf_many(&gx)));
    group.bench_function("gamma/map_pdf", |b| {
        b.iter(|| gx.iter().map(|&x| g.pdf(x)).collect::<Vec<_>>())
    });

    // BetaDist: 3 lgamma/point (ln_beta) hoisted.
    let bt = BetaDist::new(2.5, 3.5);
    let bx: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    group.bench_function("beta/pdf_many", |b| b.iter(|| bt.pdf_many(&bx)));
    group.bench_function("beta/map_pdf", |b| {
        b.iter(|| bx.iter().map(|&x| bt.pdf(x)).collect::<Vec<_>>())
    });

    // Hypergeometric: 5 lgamma/point hoisted (Fisher's-exact-test full support).
    let h = Hypergeometric::new(2000, 700, 1200);
    let ks: Vec<u64> = (0..=700).collect();
    group.bench_function("hypergeom/pmf_many", |b| b.iter(|| h.pmf_many(&ks)));
    group.bench_function("hypergeom/map_pmf", |b| {
        b.iter(|| ks.iter().map(|&k| h.pmf(k)).collect::<Vec<_>>())
    });

    group.finish();
}

/// Gaussian KDE evaluation at many points — each point is an O(n_data) sum over the
/// dataset (heavy per-point work), parallelized in evaluate_many. Head-to-head vs
/// scipy.stats.gaussian_kde (docs/perf_oracle_kde.py).
fn bench_kde(c: &mut Criterion) {
    use fsci_stats::GaussianKde;
    let mut group = c.benchmark_group("gaussian_kde");
    for &n in &[1000usize, 5000] {
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (t * 0.017).sin() * 3.0 + (t * 0.0031).cos()
            })
            .collect();
        let kde = GaussianKde::new(&data);
        let pts: Vec<f64> = (0..n).map(|i| -5.0 + i as f64 * 10.0 / n as f64).collect();
        group.bench_function(BenchmarkId::new("evaluate_many", n), |b| {
            b.iter(|| kde.evaluate_many(&pts))
        });
    }
    group.finish();
}

/// Multiscale graph correlation — mgc_map is O(n²) (prefix-sum) vs the naive O(n⁴),
/// and the `reps` permutation scoring is parallelized. Head-to-head vs
/// scipy.stats.multiscale_graphcorr (docs/perf_oracle_mgc.py).
fn bench_mgc(c: &mut Criterion) {
    use fsci_stats::multiscale_graphcorr;
    let n = 80usize;
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![(i as f64 * 0.1).sin()]).collect();
    let y: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![(i as f64 * 0.1).sin() + (i as f64 * 0.37).cos() * 0.3])
        .collect();
    let mut group = c.benchmark_group("mgc");
    group.bench_function("n80_reps100", |b| {
        b.iter(|| multiscale_graphcorr(&x, &y, 100, Some(0)))
    });
    group.finish();
}

/// Theil-Sen / Siegel robust regression slopes — head-to-head vs
/// scipy.stats.theilslopes / siegelslopes (both O(n^2) in scipy's C).
fn bench_robust_slopes(c: &mut Criterion) {
    let mut group = c.benchmark_group("robust_slopes");
    for &n in &[2000usize, 4000] {
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.001 + (i % 7) as f64 * 1e-4).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 * x[i] + ((i * 2654435761usize) % 1000) as f64 * 1e-3)
            .collect();
        group.bench_function(BenchmarkId::new("theilslopes", n), |b| {
            b.iter(|| theilslopes(black_box(&x), black_box(&y), 0.95))
        });
        group.bench_function(BenchmarkId::new("siegelslopes", n), |b| {
            b.iter(|| siegelslopes(black_box(&x), black_box(&y)))
        });
    }
    group.finish();
}

fn bench_binned_statistic_1d(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let mut group = c.benchmark_group("binned_statistic_1d");
    let n = 200_000usize;
    let xs: Vec<f64> = (0..n).map(|i| ((i * 2654435761usize) % 100000) as f64 / 100000.0).collect();
    let vs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
    for &bins in &[1000usize, 5000] {
        group.bench_function(BenchmarkId::new("mean", bins), |b| {
            b.iter(|| binned_statistic(black_box(&xs), black_box(&vs), bins, "mean"))
        });
    }
    group.finish();
}

fn bench_binned_statistic_dd(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let mut group = c.benchmark_group("binned_statistic_dd");
    let n = 200_000usize;
    let sample: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            vec![
                ((i * 2654435761usize) % 100000) as f64 / 100000.0,
                ((i * 40503usize + 7) % 100000) as f64 / 100000.0,
                ((i * 92821usize + 3) % 100000) as f64 / 100000.0,
            ]
        })
        .collect();
    let vs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
    for &bins in &[20usize, 30] {
        group.bench_function(BenchmarkId::new("d3_mean", bins), |b| {
            b.iter(|| binned_statistic_dd(black_box(&sample), black_box(&vs), bins, "mean"))
        });
    }
    group.finish();
}

fn bench_binned_statistic_2d(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let mut group = c.benchmark_group("binned_statistic_2d");
    let n = 200_000usize;
    let xs: Vec<f64> = (0..n).map(|i| ((i * 2654435761usize) % 100000) as f64 / 100000.0).collect();
    let ys: Vec<f64> = (0..n).map(|i| ((i * 40503usize + 7) % 100000) as f64 / 100000.0).collect();
    let vs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
    for stat in &["mean", "sum", "count"] {
        group.bench_function(BenchmarkId::new(*stat, n), |b| {
            b.iter(|| binned_statistic_2d(black_box(&xs), black_box(&ys), black_box(&vs), 50, stat))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_robust_slopes,
    bench_binned_statistic_2d,
    bench_binned_statistic_1d,
    bench_binned_statistic_dd,
    bench_mgc,
    bench_qmc_discrepancy,
    bench_qmc_sampling,
    bench_ordering_and_bins,
    bench_time_series,
    bench_rank_correlation,
    bench_somersd,
    bench_rand_index,
    bench_distribution_batch,
    bench_kde
);
criterion_main!(benches);
