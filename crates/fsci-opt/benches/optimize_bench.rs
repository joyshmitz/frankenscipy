#![forbid(unsafe_code)]
//! Criterion benchmarks for fsci-opt (P2C-003-H).
//!
//! Groups: bfgs, lbfgsb, cg, powell, brentq, brenth, bisect, ridder

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_opt::DifferentialEvolutionOptions;
use fsci_opt::{
    LeastSquaresOptions, MinimizeOptions, OptimizeMethod, RootMethod, RootOptions, bfgs, bisect,
    brenth, brentq, cg_pr_plus, differential_evolution, lbfgsb, least_squares,
    linear_sum_assignment, numerical_gradient, numerical_jacobian, powell, ridder,
};
use fsci_runtime::RuntimeMode;
use rand::{Rng, RngExt, SeedableRng};

// ── Test functions ────────────────────────────────────────────────────

fn rosenbrock(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        s += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    s
}

fn rosenbrock_gradient(x: &[f64]) -> Vec<f64> {
    let mut grad = vec![0.0; x.len()];
    for i in 0..x.len() - 1 {
        let residual = x[i + 1] - x[i] * x[i];
        grad[i] += -400.0 * x[i] * residual - 2.0 * (1.0 - x[i]);
        grad[i + 1] += 200.0 * residual;
    }
    grad
}

fn quadratic(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn cubic_root(x: f64) -> f64 {
    x * x * x - 2.0 * x - 5.0
}

fn sin_root(x: f64) -> f64 {
    x.sin()
}

fn opts(method: OptimizeMethod) -> MinimizeOptions {
    MinimizeOptions {
        method: Some(method),
        mode: RuntimeMode::Strict,
        ..Default::default()
    }
}

fn lbfgsb_opts() -> MinimizeOptions {
    MinimizeOptions {
        method: Some(OptimizeMethod::LBfgsB),
        mode: RuntimeMode::Strict,
        tol: Some(1.0e-8),
        maxiter: Some(2000),
        ..Default::default()
    }
}

fn root_opts(method: RootMethod) -> RootOptions {
    RootOptions {
        method: Some(method),
        mode: RuntimeMode::Strict,
        ..Default::default()
    }
}

// ── Minimize benchmarks ──────────────────────────────────────────────

fn bench_bfgs(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfgs");
    for &dim in &[2usize, 5, 10] {
        let x0: Vec<f64> = vec![0.0; dim];
        group.bench_with_input(
            BenchmarkId::new("rosenbrock", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = bfgs(&rosenbrock, x0, opts(OptimizeMethod::Bfgs));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("rosenbrock_exact_gradient", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let mut options = opts(OptimizeMethod::Bfgs);
                    options.gradient = Some(rosenbrock_gradient);
                    let _ = bfgs(&rosenbrock, x0, options);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("quadratic", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = bfgs(&quadratic, x0, opts(OptimizeMethod::Bfgs));
                });
            },
        );
    }
    group.finish();
}

fn bench_lbfgsb(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbfgsb");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    let x0 = vec![-1.2, 1.0];
    group.bench_with_input(
        BenchmarkId::new("rosenbrock_unconstrained_fd", 2usize),
        &x0,
        |b, x0| {
            b.iter(|| {
                let _ = lbfgsb(&rosenbrock, x0, lbfgsb_opts(), None);
            });
        },
    );

    let x0: Vec<f64> = (0..10)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect();
    group.bench_with_input(
        BenchmarkId::new("rosenbrock_unconstrained_fd", 10usize),
        &x0,
        |b, x0| {
            b.iter(|| {
                let _ = lbfgsb(&rosenbrock, x0, lbfgsb_opts(), None);
            });
        },
    );

    let x0: Vec<f64> = (0..32).map(|i| (i as f64 % 7.0) - 3.0).collect();
    group.bench_with_input(
        BenchmarkId::new("quadratic_unconstrained_fd", 32usize),
        &x0,
        |b, x0| {
            b.iter(|| {
                let _ = lbfgsb(&quadratic, x0, lbfgsb_opts(), None);
            });
        },
    );
    group.finish();
}

fn bench_cg(c: &mut Criterion) {
    let mut group = c.benchmark_group("cg");
    for &dim in &[2usize, 5, 10] {
        let x0: Vec<f64> = vec![0.0; dim];
        group.bench_with_input(
            BenchmarkId::new("rosenbrock", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = cg_pr_plus(&rosenbrock, x0, opts(OptimizeMethod::ConjugateGradient));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("rosenbrock_exact_gradient", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let mut options = opts(OptimizeMethod::ConjugateGradient);
                    options.gradient = Some(rosenbrock_gradient);
                    let _ = cg_pr_plus(&rosenbrock, x0, options);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("quadratic", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = cg_pr_plus(&quadratic, x0, opts(OptimizeMethod::ConjugateGradient));
                });
            },
        );
    }
    group.finish();
}

fn bench_powell(c: &mut Criterion) {
    let mut group = c.benchmark_group("powell");
    for &dim in &[2usize, 5, 10] {
        let x0: Vec<f64> = vec![0.0; dim];
        group.bench_with_input(
            BenchmarkId::new("rosenbrock", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = powell(&rosenbrock, x0, opts(OptimizeMethod::Powell));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("quadratic", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = powell(&quadratic, x0, opts(OptimizeMethod::Powell));
                });
            },
        );
    }
    group.finish();
}

fn bench_brentq(c: &mut Criterion) {
    let mut group = c.benchmark_group("brentq");
    group.bench_function("cubic", |b| {
        b.iter(|| {
            let _ = brentq(cubic_root, (1.0, 3.0), root_opts(RootMethod::Brentq));
        });
    });
    group.bench_function("sin", |b| {
        b.iter(|| {
            let _ = brentq(sin_root, (3.0, 4.0), root_opts(RootMethod::Brentq));
        });
    });
    group.finish();
}

fn bench_brenth(c: &mut Criterion) {
    let mut group = c.benchmark_group("brenth");
    group.bench_function("cubic", |b| {
        b.iter(|| {
            let _ = brenth(cubic_root, (1.0, 3.0), root_opts(RootMethod::Brenth));
        });
    });
    group.bench_function("sin", |b| {
        b.iter(|| {
            let _ = brenth(sin_root, (3.0, 4.0), root_opts(RootMethod::Brenth));
        });
    });
    group.finish();
}

fn bench_bisect(c: &mut Criterion) {
    let mut group = c.benchmark_group("bisect");
    group.bench_function("cubic", |b| {
        b.iter(|| {
            let _ = bisect(cubic_root, (1.0, 3.0), root_opts(RootMethod::Bisect));
        });
    });
    group.bench_function("sin", |b| {
        b.iter(|| {
            let _ = bisect(sin_root, (3.0, 4.0), root_opts(RootMethod::Bisect));
        });
    });
    group.finish();
}

fn bench_ridder(c: &mut Criterion) {
    let mut group = c.benchmark_group("ridder");
    group.bench_function("cubic", |b| {
        b.iter(|| {
            let _ = ridder(cubic_root, (1.0, 3.0), root_opts(RootMethod::Ridder));
        });
    });
    group.bench_function("sin", |b| {
        b.iter(|| {
            let _ = ridder(sin_root, (3.0, 4.0), root_opts(RootMethod::Ridder));
        });
    });
    group.finish();
}

fn bench_least_squares(c: &mut Criterion) {
    let mut group = c.benchmark_group("least_squares");
    group.bench_function("rosenbrock_residual", |b| {
        b.iter(|| {
            let residuals = |x: &[f64]| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]];
            let _ = least_squares(residuals, &[-1.2, 1.0], LeastSquaresOptions::default());
        });
    });

    let xs: Vec<f64> = (0..64).map(|i| i as f64 * 0.125).collect();
    let truth = [2.5_f64, 1.3, 0.5];
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| truth[0] * (-truth[1] * x).exp() + truth[2])
        .collect();
    group.bench_function("exp_curve_64", |b| {
        b.iter(|| {
            let residuals = |p: &[f64]| {
                xs.iter()
                    .zip(ys.iter())
                    .map(|(&x, &y)| p[0] * (-p[1] * x).exp() + p[2] - y)
                    .collect::<Vec<_>>()
            };
            let _ = least_squares(residuals, &[1.0, 1.0, 1.0], LeastSquaresOptions::default());
        });
    });

    let xs: Vec<f64> = (0..128).map(|i| i as f64 * 0.05).collect();
    let truth = [2.0_f64, 0.7, 0.25, -0.03];
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| truth[0] * (-truth[1] * x).exp() + truth[2] + truth[3] * x)
        .collect();
    group.bench_function("exp_linear_curve_128", |b| {
        b.iter(|| {
            let residuals = |p: &[f64]| {
                xs.iter()
                    .zip(ys.iter())
                    .map(|(&x, &y)| p[0] * (-p[1] * x).exp() + p[2] + p[3] * x - y)
                    .collect::<Vec<_>>()
            };
            let _ = least_squares(
                residuals,
                &[1.0, 0.5, 0.0, 0.0],
                LeastSquaresOptions::default(),
            );
        });
    });
    group.finish();
}

/// Hungarian assignment — head-to-head vs scipy.optimize.linear_sum_assignment.
fn bench_assignment(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let mut group = c.benchmark_group("linear_sum_assignment");
    for &n in &[500usize, 1000] {
        let cost: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        // LCG-based continuous values (few ties), matching scipy's uniform.
                        let s = (i
                            .wrapping_mul(1103515245)
                            .wrapping_add(j)
                            .wrapping_mul(12345)
                            ^ (i.wrapping_mul(2654435761) >> 7))
                            as u64;
                        (s as f64 / u64::MAX as f64) * (i % 9 + 1) as f64
                    })
                    .collect()
            })
            .collect();
        group.bench_function(BenchmarkId::new("dense", n), |b| {
            b.iter(|| linear_sum_assignment(std::hint::black_box(&cost)).expect("lsa"))
        });
    }
    group.finish();
}

fn bench_differential_evolution(c: &mut Criterion) {
    // Global optimizer over a user objective evaluated INLINE in Rust (vs scipy's
    // Python callback per nfev). Rosenbrock-5d, matched config to the scipy run
    // (maxiter=100, popsize=15, tol=1e-8, seed=1). scipy ~271 ms (nfev=7689).
    let rosen = |x: &[f64]| -> f64 {
        let mut s = 0.0;
        for i in 0..x.len() - 1 {
            s += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        s
    };
    let bounds = vec![(-5.0, 5.0); 5];
    let mut group = c.benchmark_group("differential_evolution");
    group.sample_size(20);
    group.bench_function("rosen_5d", |b| {
        b.iter(|| {
            let opts = DifferentialEvolutionOptions {
                maxiter: 100,
                popsize: 15,
                tol: 1e-8,
                seed: Some(1),
                ..Default::default()
            };
            differential_evolution(rosen, &bounds, opts).expect("DE")
        })
    });
    group.finish();
}

fn select_three_fixed(rng: &mut impl Rng, n: usize, exclude: usize) -> (usize, usize, usize) {
    let mut indices = [0; 3];
    let mut len = 0;
    let mut attempts = 0;
    while len < indices.len() {
        let idx = rng.random_range(0..n);
        if idx != exclude && !indices[..len].contains(&idx) {
            indices[len] = idx;
            len += 1;
        }
        attempts += 1;
        if attempts > 1000 {
            for k in 0..n {
                if len >= indices.len() {
                    break;
                }
                if !indices[..len].contains(&k) {
                    indices[len] = k;
                    len += 1;
                }
            }
            break;
        }
    }
    (indices[0], indices[1], indices[2])
}

fn select_three_allocating(rng: &mut impl Rng, n: usize, exclude: usize) -> (usize, usize, usize) {
    let mut indices = Vec::with_capacity(3);
    let mut attempts = 0;
    while indices.len() < 3 {
        let idx = rng.random_range(0..n);
        if idx != exclude && !indices.contains(&idx) {
            indices.push(idx);
        }
        attempts += 1;
        if attempts > 1000 {
            for k in 0..n {
                if indices.len() >= 3 {
                    break;
                }
                if !indices.contains(&k) {
                    indices.push(k);
                }
            }
            break;
        }
    }
    (indices[0], indices[1], indices[2])
}

fn select_three_checksum(
    mut rng: rand::rngs::StdRng,
    select: fn(&mut rand::rngs::StdRng, usize, usize) -> (usize, usize, usize),
) -> usize {
    let mut checksum = 0usize;
    for trial in 0..7_500 {
        let (r0, r1, r2) = select(&mut rng, 75, trial % 75);
        checksum = checksum
            .wrapping_mul(31)
            .wrapping_add(r0)
            .wrapping_add(r1.wrapping_mul(3))
            .wrapping_add(r2.wrapping_mul(7));
    }
    checksum
}

fn bench_select_three_ab(c: &mut Criterion) {
    let mut fixed_rng = rand::rngs::StdRng::seed_from_u64(1);
    let mut allocating_rng = rand::rngs::StdRng::seed_from_u64(1);
    for trial in 0..7_500 {
        assert_eq!(
            select_three_fixed(&mut fixed_rng, 75, trial % 75),
            select_three_allocating(&mut allocating_rng, 75, trial % 75),
        );
    }

    let mut group = c.benchmark_group("select_three_ab");
    group.bench_function("fixed_array/7500", |b| {
        b.iter(|| {
            black_box(select_three_checksum(
                rand::rngs::StdRng::seed_from_u64(1),
                select_three_fixed,
            ))
        });
    });
    group.bench_function("vec_capacity_3/7500", |b| {
        b.iter(|| {
            black_box(select_three_checksum(
                rand::rngs::StdRng::seed_from_u64(1),
                select_three_allocating,
            ))
        });
    });
    group.finish();
}

fn reference_clone_gradient<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let f0 = f(x);
    let mut grad = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        xp[i] += eps;
        grad.push((f(&xp) - f0) / eps);
    }
    grad
}

fn reference_clone_jacobian<F>(f: F, x: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let f0 = f(x);
    let mut jac = vec![vec![0.0; x.len()]; f0.len()];
    for j in 0..x.len() {
        let mut xp = x.to_vec();
        xp[j] += eps;
        let fp = f(&xp);
        for i in 0..f0.len() {
            jac[i][j] = (fp[i] - f0[i]) / eps;
        }
    }
    jac
}

fn finite_difference_scalar(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, &value)| {
            let weight = (i % 7 + 1) as f64;
            weight * value * value + 0.125 * value
        })
        .sum()
}

fn finite_difference_vector(x: &[f64]) -> Vec<f64> {
    let mut sums = [0.0; 4];
    for (i, &value) in x.iter().enumerate() {
        sums[i & 3] += value * ((i % 11 + 1) as f64);
    }
    sums.to_vec()
}

fn bench_finite_difference_helpers(c: &mut Criterion) {
    let mut group = c.benchmark_group("finite_difference_helpers");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(1));

    for &dim in &[256usize, 512] {
        let x: Vec<f64> = (0..dim)
            .map(|i| ((i % 29) as f64 - 14.0) * 0.03125)
            .collect();
        group.bench_with_input(
            BenchmarkId::new("gradient_clone_reference", dim),
            &x,
            |b, x| {
                b.iter(|| {
                    black_box(reference_clone_gradient(
                        finite_difference_scalar,
                        black_box(x),
                        1.0e-6,
                    ))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gradient_scratch_reuse", dim),
            &x,
            |b, x| {
                b.iter(|| {
                    black_box(numerical_gradient(
                        finite_difference_scalar,
                        black_box(x),
                        1.0e-6,
                    ))
                });
            },
        );
    }

    for &dim in &[128usize, 256] {
        let x: Vec<f64> = (0..dim)
            .map(|i| ((i % 31) as f64 - 15.0) * 0.015625)
            .collect();
        group.bench_with_input(
            BenchmarkId::new("jacobian_clone_reference", dim),
            &x,
            |b, x| {
                b.iter(|| {
                    black_box(reference_clone_jacobian(
                        finite_difference_vector,
                        black_box(x),
                        1.0e-6,
                    ))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("jacobian_scratch_reuse", dim),
            &x,
            |b, x| {
                b.iter(|| {
                    black_box(numerical_jacobian(
                        finite_difference_vector,
                        black_box(x),
                        1.0e-6,
                    ))
                });
            },
        );
    }
    group.finish();
}

// ── lm_root J^T J + J^T F build: strided-original vs symmetric-row-outer A/B ──
//
// Same-process A/B isolating the per-iteration normal-equation build inside
// `lm_root` (root.rs). The `original` arm is the shipped-before naive triple-nest
// (row-reduction innermost = n² cache-missing strided passes over `jac`); the
// `candidate` arm is the current lib code (single row-outer pass, J^T J symmetry).
// Both produce bit-identical output (asserted), so this measures pure cache/flop win.

fn jtj_jtf_original(jac: &[Vec<f64>], fx: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = jac.len();
    let mut jtj = vec![vec![0.0; n]; n];
    let mut jtf = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            for row in jac.iter().take(n) {
                jtj[i][j] += row[i] * row[j];
            }
        }
        for (row, fx_value) in jac.iter().zip(fx.iter()).take(n) {
            jtf[i] += row[i] * *fx_value;
        }
    }
    (jtj, jtf)
}

fn jtj_jtf_candidate(jac: &[Vec<f64>], fx: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = jac.len();
    let mut jtj = vec![vec![0.0; n]; n];
    let mut jtf = vec![0.0; n];
    for (row, &fx_value) in jac.iter().zip(fx.iter()) {
        for i in 0..n {
            let ri = row[i];
            for j in i..n {
                let v = ri * row[j];
                jtj[i][j] += v;
                if i != j {
                    jtj[j][i] += v;
                }
            }
            jtf[i] += ri * fx_value;
        }
    }
    (jtj, jtf)
}

fn bench_lm_jtj_build_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("lm_root_jtj_build_ab");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for &n in &[64usize, 128, 256, 512] {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xB1AC_7 ^ n as u64);
        let jac: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect())
            .collect();
        let fx: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();

        // Byte-identical proof: candidate output matches original bit-for-bit.
        let (o_jtj, o_jtf) = jtj_jtf_original(&jac, &fx);
        let (c_jtj, c_jtf) = jtj_jtf_candidate(&jac, &fx);
        for (ro, rc) in o_jtj.iter().zip(c_jtj.iter()) {
            for (a, b) in ro.iter().zip(rc.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "jtj mismatch at n={n}");
            }
        }
        for (a, b) in o_jtf.iter().zip(c_jtf.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "jtf mismatch at n={n}");
        }

        group.bench_function(BenchmarkId::new("original_strided", n), |b| {
            b.iter(|| black_box(jtj_jtf_original(black_box(&jac), black_box(&fx))));
        });
        group.bench_function(BenchmarkId::new("candidate_symmetric", n), |b| {
            b.iter(|| black_box(jtj_jtf_candidate(black_box(&jac), black_box(&fx))));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_lm_jtj_build_ab,
    bench_select_three_ab,
    bench_finite_difference_helpers,
    bench_assignment,
    bench_differential_evolution,
    bench_bfgs,
    bench_lbfgsb,
    bench_cg,
    bench_powell,
    bench_brentq,
    bench_brenth,
    bench_bisect,
    bench_ridder,
    bench_least_squares,
);
criterion_main!(benches);
