use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_stats::{
    HaltonSampler, SobolSampler, SomersDInput, acf, argsort, binned_statistic, binned_statistic_2d,
    binned_statistic_dd, centered_discrepancy, ecdf, energy_distance, histogram, kendalltau,
    kruskal, ks_2samp, l2_star_discrepancy, mannkendall, mannwhitneyu, mixture_discrepancy, pacf,
    psd_welch, rand_index, siegelslopes, somersd, theilslopes, wasserstein_distance,
    wraparound_discrepancy,
};
use std::hint::black_box;

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
    group.bench_function(BenchmarkId::new("sobol_8d", 65_536usize), |b| {
        b.iter(|| qmc_sobol_sample(65_536, 8))
    });
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
        BetaBinomial, BetaDist, Binomial, Chi, ChiSquared, ContinuousDistribution,
        DiscreteDistribution, FDistribution, GammaDist, GenGamma, GenNorm, Hypergeometric,
        InverseGamma, Nakagami, NegBinomial, StudentT, VonMises,
    };
    let n = 4096usize;
    let mut group = c.benchmark_group("distribution_batch");
    let positive_x: Vec<f64> = (0..n).map(|i| 0.001 + (i as f64 + 0.5) * 0.005).collect();
    let symmetric_x: Vec<f64> = (0..n)
        .map(|i| -8.0 + 16.0 * (i as f64 + 0.5) / n as f64)
        .collect();
    let angles: Vec<f64> = (0..n)
        .map(|i| -std::f64::consts::PI + std::f64::consts::TAU * (i as f64 + 0.5) / n as f64)
        .collect();

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

    // StudentT: gamma-ratio coefficient hoisted over a symmetric grid.
    let st = StudentT::new(7.5);
    group.bench_function("student_t/pdf_many", |b| {
        b.iter(|| st.pdf_many(&symmetric_x))
    });
    group.bench_function("student_t/map_pdf", |b| {
        b.iter(|| symmetric_x.iter().map(|&x| st.pdf(x)).collect::<Vec<_>>())
    });

    // Chi/ChiSquared: lgamma normalizers hoisted over positive support.
    let chi = Chi::new(5.5);
    group.bench_function("chi/pdf_many", |b| b.iter(|| chi.pdf_many(&positive_x)));
    group.bench_function("chi/map_pdf", |b| {
        b.iter(|| positive_x.iter().map(|&x| chi.pdf(x)).collect::<Vec<_>>())
    });
    let chi2 = ChiSquared::new(7.5);
    group.bench_function("chi2/pdf_many", |b| b.iter(|| chi2.pdf_many(&positive_x)));
    group.bench_function("chi2/map_pdf", |b| {
        b.iter(|| positive_x.iter().map(|&x| chi2.pdf(x)).collect::<Vec<_>>())
    });

    // FDistribution and generalized gamma: beta/gamma normalizers hoisted.
    let fdist = FDistribution::new(9.0, 17.0);
    group.bench_function("f/pdf_many", |b| b.iter(|| fdist.pdf_many(&positive_x)));
    group.bench_function("f/map_pdf", |b| {
        b.iter(|| positive_x.iter().map(|&x| fdist.pdf(x)).collect::<Vec<_>>())
    });
    let gengamma = GenGamma::new(2.5, 1.7);
    group.bench_function("gengamma/pdf_many", |b| {
        b.iter(|| gengamma.pdf_many(&positive_x))
    });
    group.bench_function("gengamma/map_pdf", |b| {
        b.iter(|| {
            positive_x
                .iter()
                .map(|&x| gengamma.pdf(x))
                .collect::<Vec<_>>()
        })
    });

    // Inverse-gamma, Nakagami, generalized normal, and Von Mises hoist
    // distribution-wide special-function constants out of the timed map.
    let invgamma = InverseGamma::new(3.5);
    group.bench_function("invgamma/pdf_many", |b| {
        b.iter(|| invgamma.pdf_many(&positive_x))
    });
    group.bench_function("invgamma/map_pdf", |b| {
        b.iter(|| {
            positive_x
                .iter()
                .map(|&x| invgamma.pdf(x))
                .collect::<Vec<_>>()
        })
    });
    let nakagami = Nakagami::new(2.25);
    group.bench_function("nakagami/pdf_many", |b| {
        b.iter(|| nakagami.pdf_many(&positive_x))
    });
    group.bench_function("nakagami/map_pdf", |b| {
        b.iter(|| {
            positive_x
                .iter()
                .map(|&x| nakagami.pdf(x))
                .collect::<Vec<_>>()
        })
    });
    let gennorm = GenNorm::new(1.5);
    group.bench_function("gennorm/pdf_many", |b| {
        b.iter(|| gennorm.pdf_many(&symmetric_x))
    });
    group.bench_function("gennorm/map_pdf", |b| {
        b.iter(|| {
            symmetric_x
                .iter()
                .map(|&x| gennorm.pdf(x))
                .collect::<Vec<_>>()
        })
    });
    let vonmises = VonMises::new(3.0, 0.25);
    group.bench_function("vonmises/pdf_many", |b| {
        b.iter(|| vonmises.pdf_many(&angles))
    });
    group.bench_function("vonmises/map_pdf", |b| {
        b.iter(|| angles.iter().map(|&x| vonmises.pdf(x)).collect::<Vec<_>>())
    });

    // Binomial: 3 parameter-only terms hoisted over a full support sweep.
    let bin = Binomial::new(2000, 0.37);
    let bin_ks: Vec<u64> = (0..=2000).collect();
    group.bench_function("binomial/pmf_many", |b| b.iter(|| bin.pmf_many(&bin_ks)));
    group.bench_function("binomial/map_pmf", |b| {
        b.iter(|| bin_ks.iter().map(|&k| bin.pmf(k)).collect::<Vec<_>>())
    });

    // Negative binomial: parameter-only lnGamma/log terms hoisted across a long tail.
    let nbin = NegBinomial::new(20.0, 0.42);
    let tail_ks: Vec<u64> = (0..n as u64).collect();
    group.bench_function("negbinom/pmf_many", |b| b.iter(|| nbin.pmf_many(&tail_ks)));
    group.bench_function("negbinom/map_pmf", |b| {
        b.iter(|| tail_ks.iter().map(|&k| nbin.pmf(k)).collect::<Vec<_>>())
    });

    // Beta-binomial: 5 lgamma/point hoisted over a bounded support.
    let bb = BetaBinomial::new(2000, 2.5, 7.0);
    let bb_ks: Vec<u64> = (0..=2000).collect();
    group.bench_function("betabinom/pmf_many", |b| b.iter(|| bb.pmf_many(&bb_ks)));
    group.bench_function("betabinom/map_pmf", |b| {
        b.iter(|| bb_ks.iter().map(|&k| bb.pmf(k)).collect::<Vec<_>>())
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
fn bench_rank_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_tests");
    let n = 200_000usize;
    let x: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 100003) as f64 * 1e-4)
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| ((i * 40503usize + 7) % 100003) as f64 * 1e-4 + 0.1)
        .collect();
    group.bench_function("ks_2samp_200k", |b| {
        b.iter(|| ks_2samp(black_box(&x), black_box(&y)))
    });
    group.bench_function("mannwhitneyu_200k", |b| {
        b.iter(|| mannwhitneyu(black_box(&x), black_box(&y)))
    });
    group.bench_function("kruskal_200k", |b| {
        b.iter(|| kruskal(&[black_box(&x), black_box(&y)]))
    });
    group.finish();
}

fn bench_mvt_pdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::MultivariateT;
    let mut group = c.benchmark_group("multivariate_t_pdf");
    let m = 100_000usize;
    for &d in &[3usize, 10] {
        let loc: Vec<f64> = (0..d).map(|i| (i as f64) * 0.1).collect();
        let a: Vec<Vec<f64>> = (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 7 + j * 3) % 5) as f64 * 0.2 - 0.4)
                    .collect()
            })
            .collect();
        let mut shape = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..d {
                let mut sv = 0.0;
                for k in 0..d {
                    sv += a[i][k] * a[j][k];
                }
                shape[i][j] = sv + if i == j { d as f64 } else { 0.0 };
            }
        }
        let mvt = MultivariateT::new(&loc, &shape, 5.0).expect("mvt");
        let q: Vec<Vec<f64>> = (0..m)
            .map(|t| {
                (0..d)
                    .map(|j| ((t * 13 + j * 17) % 1000) as f64 * 0.01 - 5.0)
                    .collect()
            })
            .collect();
        group.bench_function(BenchmarkId::new("pdf_many", d), |b| {
            b.iter(|| mvt.pdf_many(black_box(&q)).expect("pdf"))
        });
    }
    group.finish();
}

fn bench_mvn_pdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::MultivariateNormal;
    let mut group = c.benchmark_group("multivariate_normal_pdf");
    let m = 100_000usize;
    for &d in &[3usize, 5, 8, 10] {
        let mean: Vec<f64> = (0..d).map(|i| (i as f64) * 0.1).collect();
        // SPD covariance: A A^T + d I
        let a: Vec<Vec<f64>> = (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 7 + j * 3) % 5) as f64 * 0.2 - 0.4)
                    .collect()
            })
            .collect();
        let mut cov = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0;
                for k in 0..d {
                    s += a[i][k] * a[j][k];
                }
                cov[i][j] = s + if i == j { d as f64 } else { 0.0 };
            }
        }
        let mvn = MultivariateNormal::new(&mean, &cov).expect("mvn");
        let q: Vec<Vec<f64>> = (0..m)
            .map(|t| {
                (0..d)
                    .map(|j| ((t * 13 + j * 17) % 1000) as f64 * 0.01 - 5.0)
                    .collect()
            })
            .collect();
        group.bench_function(BenchmarkId::new("pdf_many", d), |b| {
            b.iter(|| mvn.pdf_many(black_box(&q)).expect("pdf"))
        });
    }
    group.finish();
}

fn bench_kde_nd(c: &mut Criterion) {
    use fsci_stats::GaussianKdeNd;
    let mut group = c.benchmark_group("gaussian_kde_nd");
    let n = 2000usize;
    let m = 5000usize;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let t = i as f64;
            vec![
                (t * 0.017).sin(),
                (t * 0.0031).cos() * 2.0,
                (t * 0.011).sin() * 0.5,
            ]
        })
        .collect();
    let kde = GaussianKdeNd::new(&data).expect("kde");
    let q: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let t = i as f64;
            vec![
                (t * 0.02).cos(),
                (t * 0.005).sin() * 2.0,
                (t * 0.013).cos() * 0.5,
            ]
        })
        .collect();
    group.bench_function("d3_eval5k", |b| b.iter(|| kde.evaluate_many(black_box(&q))));
    group.finish();
}

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
        let x: Vec<f64> = (0..n)
            .map(|i| i as f64 * 0.001 + (i % 7) as f64 * 1e-4)
            .collect();
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

fn bench_distribution_distances(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let mut group = c.benchmark_group("distribution_distances");
    for &n in &[50_000usize, 200_000] {
        let u: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.0007).sin() * 2.0 + (i % 31) as f64 * 0.03)
            .collect();
        let v: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.0011).cos() * 2.0 + 0.3 + (i % 17) as f64 * 0.05)
            .collect();
        group.bench_function(BenchmarkId::new("wasserstein", n), |b| {
            b.iter(|| wasserstein_distance(black_box(&u), black_box(&v)))
        });
        group.bench_function(BenchmarkId::new("energy", n), |b| {
            b.iter(|| energy_distance(black_box(&u), black_box(&v)))
        });
    }
    group.finish();
}

fn bench_binned_statistic_1d(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let mut group = c.benchmark_group("binned_statistic_1d");
    let n = 200_000usize;
    let xs: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 100000) as f64 / 100000.0)
        .collect();
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
    let xs: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 100000) as f64 / 100000.0)
        .collect();
    let ys: Vec<f64> = (0..n)
        .map(|i| ((i * 40503usize + 7) % 100000) as f64 / 100000.0)
        .collect();
    let vs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
    for stat in &["mean", "sum", "count"] {
        group.bench_function(BenchmarkId::new(*stat, n), |b| {
            b.iter(|| binned_statistic_2d(black_box(&xs), black_box(&ys), black_box(&vs), 50, stat))
        });
    }
    group.finish();
}

fn bench_skellam_cdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{DiscreteDistribution, Skellam};
    let mut group = c.benchmark_group("skellam_cdf");
    // cdf/sf cost scales with μ: the prior Bessel-`ive` window sum spanned ~24σ+80
    // points, so large μ exercised hundreds of Bessel evals per call. Evaluate a
    // batch of k across the central body for each μ.
    for &(mu1, mu2) in &[(5.0_f64, 3.0_f64), (100.0, 100.0), (1000.0, 1000.0)] {
        let sk = Skellam::new(mu1, mu2);
        let mean = mu1 - mu2;
        let std = (mu1 + mu2).sqrt();
        let ks: Vec<u64> = (0..256)
            .map(|i| (mean + (i as f64 - 128.0) / 256.0 * 6.0 * std).max(0.0) as u64)
            .collect();
        group.bench_function(BenchmarkId::new("cdf", format!("mu{mu1}_{mu2}")), |b| {
            b.iter(|| {
                ks.iter()
                    .map(|&k| sk.cdf(black_box(k)))
                    .sum::<f64>()
            })
        });
        group.bench_function(BenchmarkId::new("sf", format!("mu{mu1}_{mu2}")), |b| {
            b.iter(|| ks.iter().map(|&k| sk.sf(black_box(k))).sum::<f64>())
        });
    }
    group.finish();
}

fn bench_betabinom_cdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{BetaBinomial, DiscreteDistribution};
    let mut group = c.benchmark_group("betabinom_cdf");
    // Default cdf sums pmf(0..=k) at ~6 ln_gamma/pmf — O(k) lgamma; the pmf-ratio
    // recurrence makes each term one multiply. Cost scales with n.
    for &n in &[50_u64, 200, 1000] {
        let d = BetaBinomial::new(n, 2.0, 3.0);
        let ks: Vec<u64> = (0..=n).collect();
        group.bench_function(BenchmarkId::new("cdf", n), |b| {
            b.iter(|| ks.iter().map(|&k| d.cdf(black_box(k))).sum::<f64>())
        });
    }
    group.finish();
}

fn bench_discrete_moments(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{BetaBinomial, DiscreteDistribution, NegHypergeometric};
    let mut group = c.benchmark_group("discrete_moments");
    // skewness/kurtosis summed central moments over the support at ~6 ln_gamma/pmf;
    // the pmf-ratio recurrence (mode-anchored) makes each term pure mults. Scales w/ n.
    for &n in &[200_u64, 1000] {
        let d = BetaBinomial::new(n, 2.0, 3.0);
        group.bench_function(BenchmarkId::new("betabinom_kurt", n), |b| {
            b.iter(|| black_box(d.kurtosis()))
        });
    }
    for &n in &[200_u64, 400] {
        let d = NegHypergeometric::new(2 * n + 100, n, n + 50);
        group.bench_function(BenchmarkId::new("neghypergeom_kurt", n), |b| {
            b.iter(|| black_box(d.kurtosis()))
        });
    }
    group.finish();
}

fn bench_discrete_entropy(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{
        BetaBinomial, BetaNegativeBinomial, Binomial, Boltzmann, DiscreteDistribution,
        NegHypergeometric, Poisson,
    };
    let mut group = c.benchmark_group("discrete_entropy");
    // entropy summed −Σ pmf·ln(pmf) over the support at ~6 ln_gamma/pmf; the
    // pmf-ratio recurrence (mode-anchored) makes each term one ln. Scales with n.
    for &n in &[200_u64, 1000] {
        let d = BetaBinomial::new(n, 2.0, 3.0);
        group.bench_function(BenchmarkId::new("betabinom", n), |b| b.iter(|| black_box(d.entropy())));
    }
    let bnb = BetaNegativeBinomial::new(10, 3.0, 3.0);
    group.bench_function(BenchmarkId::new("betanbinom", 10_u64), |b| {
        b.iter(|| black_box(bnb.entropy()))
    });
    for &n in &[1000_u64, 10000] {
        let d = Binomial::new(n, 0.5);
        group.bench_function(BenchmarkId::new("binomial", n), |b| b.iter(|| black_box(d.entropy())));
    }
    for &mu in &[100.0_f64, 900.0] {
        let d = Poisson::new(mu);
        group.bench_function(BenchmarkId::new("poisson", mu as u64), |b| {
            b.iter(|| black_box(d.entropy()))
        });
    }
    for &n in &[10_000_u32, 100_000] {
        let d = Boltzmann::new(0.2, n);
        group.bench_function(BenchmarkId::new("boltzmann", n), |b| {
            b.iter(|| black_box(d.entropy()))
        });
    }
    for &n in &[200_u64, 400] {
        let d = NegHypergeometric::new(2 * n + 100, n, n + 50);
        group.bench_function(BenchmarkId::new("neghypergeom", n), |b| {
            b.iter(|| black_box(d.entropy()))
        });
    }
    group.finish();
}

fn bench_neghypergeom_cdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{DiscreteDistribution, NegHypergeometric};
    let mut group = c.benchmark_group("neghypergeom_cdf");
    // cdf summed pmf(0..=k) at 3 ln_comb (~6 ln_gamma)/pmf; the pmf-ratio recurrence
    // (mode-anchored) makes each term one multiply. Cost scales with n (support [0,n]).
    for &n in &[50_u64, 200, 400] {
        let d = NegHypergeometric::new(2 * n + 100, n, n + 50);
        let ks: Vec<u64> = (0..=n).collect();
        group.bench_function(BenchmarkId::new("cdf", n), |b| {
            b.iter(|| ks.iter().map(|&k| d.cdf(black_box(k))).sum::<f64>())
        });
    }
    group.finish();
}

fn bench_betanbinom_cdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{BetaNegativeBinomial, DiscreteDistribution};
    let mut group = c.benchmark_group("betanbinom_cdf");
    // Default cdf sums pmf(0..=k) at ~6 ln_gamma/pmf; the pmf-ratio recurrence
    // (mode=0, anchor at pmf(0)) makes each term one multiply. Cost scales with k.
    let d = BetaNegativeBinomial::new(10, 3.0, 3.0);
    for &kmax in &[100_u64, 500, 2000] {
        let ks: Vec<u64> = (0..=kmax).collect();
        group.bench_function(BenchmarkId::new("cdf", kmax), |b| {
            b.iter(|| ks.iter().map(|&k| d.cdf(black_box(k))).sum::<f64>())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_discrete_moments,
    bench_discrete_entropy,
    bench_neghypergeom_cdf,
    bench_betanbinom_cdf,
    bench_betabinom_cdf,
    bench_skellam_cdf,
    bench_robust_slopes,
    bench_binned_statistic_2d,
    bench_distribution_distances,
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
    bench_kde,
    bench_kde_nd,
    bench_mvn_pdf,
    bench_mvt_pdf,
    bench_rank_tests
);
criterion_main!(benches);
