use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_interpolate::{
    Akima1DInterpolator, BarycentricInterpolator, CloughTocher2DInterpolator, CubicHermiteSpline,
    CubicSplineStandalone, GriddataMethod, INTERP_CUBIC_CURSOR_DISABLE, Interp1d, Interp1dOptions,
    InterpKind, LinearNDInterpolator, PchipInterpolator, RbfInterpolator, RbfKernel,
    RectBivariateSpline, RegularGridInterpolator, RegularGridMethod, SmoothBivariateSpline,
    SmoothBivariateSplineOptions, SplineBc, barycentric_eval, bisplrep, griddata, interp1d_linear,
    lagrange, make_interp_spline, make_smoothing_spline, polyadd, polyder, polyint_definite,
    polymul, polyroots, polysub, polyval_der, ratval,
};
use fsci_runtime::RuntimeMode;
use std::hint::black_box;
use std::sync::atomic::Ordering;

fn grid_1d(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

fn values_1d(xs: &[f64]) -> Vec<f64> {
    xs.iter()
        .map(|&x| (x * 11.0).sin() + 0.25 * (x * 7.0).cos())
        .collect()
}

fn query_1d(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            0.001 + 0.998 * t
        })
        .collect()
}

fn barycentric_eval_two_pass(nodes: &[f64], values: &[f64], weights: &[f64], x: f64) -> f64 {
    let n = nodes.len();
    if n == 0 || values.len() != n || weights.len() != n || !x.is_finite() {
        return f64::NAN;
    }
    for (i, &xi) in nodes.iter().enumerate() {
        if (x - xi).abs() < 1e-15 {
            return values[i];
        }
    }

    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..nodes.len() {
        let t = weights[i] / (x - nodes[i]);
        num += t * values[i];
        den += t;
    }
    if den == 0.0 {
        return f64::NAN;
    }
    num / den
}

fn ratval_power_sum(p: &[f64], q: &[f64], x: f64) -> f64 {
    let num: f64 = p
        .iter()
        .enumerate()
        .map(|(i, &coefficient)| coefficient * x.powi(i as i32))
        .sum();
    let den: f64 = q
        .iter()
        .enumerate()
        .map(|(i, &coefficient)| coefficient * x.powi(i as i32))
        .sum();
    if den.abs() < 1e-30 {
        return f64::NAN;
    }
    num / den
}

fn query_1d_unsorted(n: usize) -> Vec<f64> {
    let mut xs = query_1d(n);
    xs.reverse();
    xs
}

fn points_2d(side: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut points = Vec::with_capacity(side * side);
    let mut values = Vec::with_capacity(side * side);
    for iy in 0..side {
        for ix in 0..side {
            let x = (ix as f64 + 0.3 * (iy % 3) as f64) / side as f64;
            let y = (iy as f64 + 0.2 * (ix % 5) as f64) / side as f64;
            points.push(vec![x, y]);
            values.push((x * 5.0).sin() + (y * 3.0).cos());
        }
    }
    (points, values)
}

fn queries_2d(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let x = ((i * 37) % 997) as f64 / 997.0;
            let y = ((i * 53 + 17) % 991) as f64 / 991.0;
            vec![x, y]
        })
        .collect()
}

fn rect_grid(side: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let x = grid_1d(side);
    let y = grid_1d(side);
    let z = x
        .iter()
        .map(|&xi| {
            y.iter()
                .map(|&yi| (xi * 6.0).sin() * (yi * 4.0).cos())
                .collect()
        })
        .collect();
    (x, y, z)
}

fn regular_grid_values(points: &[Vec<f64>]) -> Vec<f64> {
    let nx = points[0].len();
    let ny = points[1].len();
    let nz = points[2].len();
    let mut values = Vec::with_capacity(nx * ny * nz);
    for &x in &points[0] {
        for &y in &points[1] {
            for &z in &points[2] {
                values.push((x * 5.0).sin() + (y * 3.0).cos() + z * z);
            }
        }
    }
    values
}

fn bench_interp1d(c: &mut Criterion) {
    let x = grid_1d(4096);
    let y = values_1d(&x);
    let sorted = query_1d(8192);
    let unsorted = query_1d_unsorted(8192);
    let interp = Interp1d::new(
        &x,
        &y,
        Interp1dOptions {
            kind: InterpKind::Linear,
            mode: RuntimeMode::Strict,
            fill_value: None,
            bounds_error: false,
            spline_bc: SplineBc::Natural,
        },
    )
    .expect("linear interpolator");

    let mut group = c.benchmark_group("interp1d");
    group.bench_function("linear_sorted_eval_many/4096x8192", |b| {
        b.iter(|| interp.eval_many(&sorted).expect("sorted eval"))
    });
    group.bench_function("linear_unsorted_eval_many/4096x8192", |b| {
        b.iter(|| interp.eval_many(&unsorted).expect("unsorted eval"))
    });
    group.bench_function("linear_one_shot/4096x8192", |b| {
        b.iter(|| interp1d_linear(&x, &y, &sorted).expect("one-shot linear"))
    });
    group.finish();
}

fn bench_splines(c: &mut Criterion) {
    let x = grid_1d(1024);
    let y = values_1d(&x);
    let x_new = query_1d(4096);
    let cubic = CubicSplineStandalone::new(&x, &y, SplineBc::Natural).expect("cubic spline");
    let pchip = PchipInterpolator::new(&x, &y).expect("pchip");

    let x_big = query_1d(100_000);
    let mut group = c.benchmark_group("splines");
    group.bench_function("cubic_eval_many/1024x4096", |b| {
        b.iter(|| cubic.eval_many(&x_new))
    });
    group.bench_function("cubic_eval_many/1024x100000", |b| {
        b.iter(|| cubic.eval_many(&x_big))
    });
    group.bench_function("pchip_eval_many/1024x4096", |b| {
        b.iter(|| pchip.eval_many(&x_new))
    });
    group.bench_function("cubic_construct/1024", |b| {
        b.iter(|| CubicSplineStandalone::new(&x, &y, SplineBc::Natural).expect("cubic construct"))
    });
    group.finish();
}

/// Same-binary A/B for the sorted-batch interval cursor extended from PchipInterpolator
/// to the sibling piecewise-cubic interpolators (frankenscipy-b75mf pattern). The cursor
/// advances monotonically over a sorted finite batch (O(N+M)) instead of binary-searching
/// each query (O(M·log N)). Both arms asserted byte-identical before timing.
fn bench_cubic_cursor_eval_many_ab(c: &mut Criterion) {
    let x = grid_1d(1024);
    let y = values_1d(&x);
    let dydx = vec![0.0f64; x.len()];
    let cubic = CubicSplineStandalone::new(&x, &y, SplineBc::Natural).expect("cubic");
    let akima = Akima1DInterpolator::new(&x, &y).expect("akima");
    let hermite = CubicHermiteSpline::new(&x, &y, &dydx).expect("hermite");

    let mut group = c.benchmark_group("cubic_cursor_eval_many_ab");
    for &m in &[4096usize, 100_000usize] {
        let x_new = query_1d(m);

        // Byte-identity: cursor arm vs par_query_map arm, for each interpolator.
        for (name, cursor, orig) in [
            (
                "cubic",
                {
                    INTERP_CUBIC_CURSOR_DISABLE.store(false, Ordering::Relaxed);
                    cubic.eval_many(&x_new)
                },
                {
                    INTERP_CUBIC_CURSOR_DISABLE.store(true, Ordering::Relaxed);
                    cubic.eval_many(&x_new)
                },
            ),
            (
                "akima",
                {
                    INTERP_CUBIC_CURSOR_DISABLE.store(false, Ordering::Relaxed);
                    akima.eval_many(&x_new)
                },
                {
                    INTERP_CUBIC_CURSOR_DISABLE.store(true, Ordering::Relaxed);
                    akima.eval_many(&x_new)
                },
            ),
            (
                "hermite",
                {
                    INTERP_CUBIC_CURSOR_DISABLE.store(false, Ordering::Relaxed);
                    hermite.eval_many(&x_new)
                },
                {
                    INTERP_CUBIC_CURSOR_DISABLE.store(true, Ordering::Relaxed);
                    hermite.eval_many(&x_new)
                },
            ),
        ] {
            assert!(
                cursor
                    .iter()
                    .zip(&orig)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "cursor eval_many must be byte-identical to par_query_map for {name} m={m}"
            );
        }

        group.bench_function(format!("cubic_current_cursor/{m}"), |b| {
            b.iter(|| {
                INTERP_CUBIC_CURSOR_DISABLE.store(false, Ordering::Relaxed);
                black_box(cubic.eval_many(black_box(&x_new)))
            })
        });
        group.bench_function(format!("cubic_orig_binsearch/{m}"), |b| {
            b.iter(|| {
                INTERP_CUBIC_CURSOR_DISABLE.store(true, Ordering::Relaxed);
                black_box(cubic.eval_many(black_box(&x_new)))
            })
        });
        group.bench_function(format!("akima_current_cursor/{m}"), |b| {
            b.iter(|| {
                INTERP_CUBIC_CURSOR_DISABLE.store(false, Ordering::Relaxed);
                black_box(akima.eval_many(black_box(&x_new)))
            })
        });
        group.bench_function(format!("akima_orig_binsearch/{m}"), |b| {
            b.iter(|| {
                INTERP_CUBIC_CURSOR_DISABLE.store(true, Ordering::Relaxed);
                black_box(akima.eval_many(black_box(&x_new)))
            })
        });
    }
    INTERP_CUBIC_CURSOR_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_polynomial(c: &mut Criterion) {
    let x = grid_1d(128);
    let y = values_1d(&x);
    let bary = BarycentricInterpolator::new(&x, &y).expect("barycentric");
    let x_new = query_1d(2048);
    let a = values_1d(&grid_1d(512));
    let b = values_1d(&grid_1d(512));
    let sub_a = values_1d(&grid_1d(1_000_000));
    let sub_b = values_1d(&grid_1d(1_000_000));
    let roots_coeffs = vec![1.0, -2.5, 1.1, -0.25, 0.03];
    let bary_n = 2_097_152;
    let bary_nodes: Vec<f64> = (0..bary_n).map(|i| i as f64 / bary_n as f64).collect();
    let bary_values: Vec<f64> = (0..bary_n).map(|i| (i as f64 * 0.001).sin()).collect();
    let bary_weights: Vec<f64> = (0..bary_n).map(|i| 1.0 + (i % 17) as f64 * 0.001).collect();
    let bary_x = 1.125;
    let rat_p: Vec<f64> = (0..4096).map(|i| 1.0 / (i + 1) as f64).collect();
    let mut rat_q: Vec<f64> = (0..4096).map(|i| 0.75 / (i + 2) as f64).collect();
    rat_q[0] += 1.0;
    let rat_x = 0.999;
    assert_eq!(
        barycentric_eval(&bary_nodes, &bary_values, &bary_weights, bary_x).to_bits(),
        barycentric_eval_two_pass(&bary_nodes, &bary_values, &bary_weights, bary_x).to_bits()
    );
    let rat_expected = ratval_power_sum(&rat_p, &rat_q, rat_x);
    let rat_actual = ratval(&rat_p, &rat_q, rat_x);
    assert!((rat_actual - rat_expected).abs() <= rat_expected.abs().max(1.0) * 2e-11);

    let mut group = c.benchmark_group("polynomial");
    group.bench_function("barycentric_eval_many/128x2048", |b| {
        b.iter(|| bary.eval_many(&x_new))
    });
    group.bench_function("barycentric_eval_fused/2097152/candidate", |bench| {
        bench.iter(|| {
            black_box(barycentric_eval(
                black_box(&bary_nodes),
                black_box(&bary_values),
                black_box(&bary_weights),
                black_box(bary_x),
            ))
        })
    });
    group.bench_function("barycentric_eval_fused/2097152/original", |bench| {
        bench.iter(|| {
            black_box(barycentric_eval_two_pass(
                black_box(&bary_nodes),
                black_box(&bary_values),
                black_box(&bary_weights),
                black_box(bary_x),
            ))
        })
    });
    group.bench_function("lagrange_construct/128", |b| {
        b.iter(|| lagrange(&x, &y).expect("lagrange"))
    });
    group.bench_function("polymul/512x512", |bench| bench.iter(|| polymul(&a, &b)));
    group.bench_function("polyadd/1000000x1000000", |bench| {
        bench.iter(|| polyadd(black_box(&sub_a), black_box(&sub_b)))
    });
    group.bench_function("polysub/1000000x1000000", |bench| {
        bench.iter(|| polysub(black_box(&sub_a), black_box(&sub_b)))
    });
    group.bench_function("polyder/1000000/m8", |bench| {
        bench.iter(|| polyder(black_box(&sub_a), black_box(8)))
    });
    group.bench_function("polyval_der/1000000/d8", |bench| {
        bench.iter(|| {
            black_box(polyval_der(
                black_box(&sub_a),
                black_box(0.875),
                black_box(8),
            ))
        })
    });
    group.bench_function("polyint_definite/1000000", |bench| {
        bench.iter(|| {
            black_box(polyint_definite(
                black_box(&sub_a),
                black_box(-0.75),
                black_box(0.875),
            ))
        })
    });
    group.bench_function("ratval_horner_ab/4096/candidate", |bench| {
        bench.iter(|| {
            black_box(ratval(
                black_box(&rat_p),
                black_box(&rat_q),
                black_box(rat_x),
            ))
        })
    });
    group.bench_function("ratval_horner_ab/4096/original", |bench| {
        bench.iter(|| {
            black_box(ratval_power_sum(
                black_box(&rat_p),
                black_box(&rat_q),
                black_box(rat_x),
            ))
        })
    });
    group.bench_function("polyroots/degree4", |b| b.iter(|| polyroots(&roots_coeffs)));
    group.finish();
}

fn bench_regular_grid(c: &mut Criterion) {
    let points = vec![grid_1d(32), grid_1d(32), grid_1d(16)];
    let values = regular_grid_values(&points);
    let queries = queries_2d(4096)
        .into_iter()
        .map(|q| vec![q[0], q[1], (q[0] * 0.7 + q[1] * 0.3).fract()])
        .collect::<Vec<_>>();
    let linear = RegularGridInterpolator::new(
        points.clone(),
        values.clone(),
        RegularGridMethod::Linear,
        false,
        None,
    )
    .expect("regular linear");
    let nearest =
        RegularGridInterpolator::new(points, values, RegularGridMethod::Nearest, false, None)
            .expect("regular nearest");

    let mut group = c.benchmark_group("regular_grid");
    group.bench_function("linear_eval_many/32x32x16_4096", |b| {
        b.iter(|| linear.eval_many(&queries).expect("linear grid"))
    });
    group.bench_function("nearest_eval_many/32x32x16_4096", |b| {
        b.iter(|| nearest.eval_many(&queries).expect("nearest grid"))
    });
    group.finish();
}

fn bench_scattered(c: &mut Criterion) {
    let (points, values) = points_2d(24);
    let queries = queries_2d(1024);
    let linear = LinearNDInterpolator::new(&points, &values).expect("linear nd");
    let clough = CloughTocher2DInterpolator::new(&points, &values).expect("clough-tocher");

    let mut group = c.benchmark_group("scattered_2d");
    group.bench_function("linear_nd_eval_many/576x1024", |b| {
        b.iter(|| linear.eval_many(&queries).expect("linear nd eval"))
    });
    group.bench_function("clough_tocher_eval_many/576x1024", |b| {
        b.iter(|| clough.eval_many(&queries).expect("clough eval"))
    });
    group.bench_function("griddata_linear/576x1024", |b| {
        b.iter(|| griddata(&points, &values, &queries, GriddataMethod::Linear).expect("griddata"))
    });
    group.finish();
}

fn bench_make_interp_spline(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_interp_spline");
    group.sample_size(10);
    for &n in &[1000usize, 3000] {
        let x: Vec<f64> = (0..n)
            .map(|i| i as f64 + ((i * 2654435761usize) % 97) as f64 * 0.001)
            .collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        group.bench_function(criterion::BenchmarkId::new("k3", n), |b| {
            b.iter(|| make_interp_spline(black_box(&x), black_box(&y), 3).expect("spline"))
        });
    }
    group.finish();
}

fn bench_rbf_scattered(c: &mut Criterion) {
    // Scattered RBF (thin-plate-spline) build (O(N^3) dense solve) + eval, matching
    // scipy.interpolate.RBFInterpolator (default kernel) at n=2000 -> 20000 queries
    // (scipy ~1205 ms).
    let n = 2000usize;
    let pts: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            vec![
                ((i * 2654435761usize) % 10007) as f64 / 10007.0,
                ((i * 40503usize + 7) % 10007) as f64 / 10007.0,
            ]
        })
        .collect();
    let vals: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
    let q: Vec<Vec<f64>> = (0..20000)
        .map(|i| {
            vec![
                ((i * 92821usize) % 9973) as f64 / 9973.0,
                ((i * 13usize + 3) % 9973) as f64 / 9973.0,
            ]
        })
        .collect();
    let mut group = c.benchmark_group("rbf_scattered");
    group.sample_size(10);
    group.bench_function("tps_build_eval_2k_to_20k", |b| {
        b.iter(|| {
            let rbf =
                RbfInterpolator::new(&pts, &vals, RbfKernel::ThinPlateSpline, 1.0).expect("rbf");
            rbf.eval_many(&q)
        })
    });
    group.finish();
}

fn bench_rbf_and_rect(c: &mut Criterion) {
    let (x, y, z) = rect_grid(32);
    let rect = RectBivariateSpline::new(&x, &y, &z, 3, 3).expect("rect bivariate");
    let xi = query_1d(64);
    let yi = query_1d(64);

    let mut group = c.benchmark_group("rbf_rect");
    group.bench_function("rect_eval_grid/32x32_to_64x64", |b| {
        b.iter(|| rect.eval_grid(&xi, &yi))
    });
    group.bench_function(BenchmarkId::new("rect_integral", "32x32"), |b| {
        b.iter(|| rect.integral(0.05, 0.95, 0.05, 0.95))
    });
    group.finish();
}

/// Batch evaluation vs mapping the scalar evaluate (the "original"). Quantifies the
/// loop-invariant hoist: evaluate_many computes the point-independent setup
/// (Bernstein binomials / tensor strides + per-dim orders + scratch) ONCE.
fn bench_batch_eval(c: &mut Criterion) {
    use fsci_interpolate::{BPoly, NdPPoly, PPoly};
    let m = 4096usize;
    let mut group = c.benchmark_group("batch_eval");

    // PPoly: interval lookup is binary search over sorted breakpoints.
    let n_pieces = 200usize;
    let px: Vec<f64> = (0..=n_pieces).map(|i| i as f64).collect();
    let pc: Vec<Vec<f64>> = (0..n_pieces)
        .map(|i| vec![0.125, -0.5, i as f64, 1.0])
        .collect();
    let pp = PPoly::new(pc, px).expect("ppoly");
    let qs: Vec<f64> = (0..m)
        .map(|i| (i as f64) * n_pieces as f64 / m as f64)
        .collect();
    group.bench_function("ppoly/evaluate_many", |b| b.iter(|| pp.evaluate_many(&qs)));
    group.bench_function("ppoly/map_evaluate", |b| {
        b.iter(|| qs.iter().map(|&x| pp.evaluate(x)).collect::<Vec<_>>())
    });

    // BPoly: per-segment Bernstein binomials hoisted in evaluate_many.
    let bx: Vec<f64> = (0..=n_pieces).map(|i| i as f64).collect();
    let bc: Vec<Vec<f64>> = (0..n_pieces)
        .map(|i| vec![i as f64, (i + 1) as f64, 0.5, 1.5])
        .collect();
    let bp = BPoly::new(bc, bx).expect("bpoly");
    group.bench_function("bpoly/evaluate_many", |b| b.iter(|| bp.evaluate_many(&qs)));
    group.bench_function("bpoly/map_evaluate", |b| {
        b.iter(|| qs.iter().map(|&x| bp.evaluate(x)).collect::<Vec<_>>())
    });

    // NdPPoly: tensor strides + per-dim orders + powers/idx scratch hoisted.
    let c_tensor: Vec<f64> = (1..=36).map(|v| v as f64).collect();
    let c_shape = vec![3usize, 2, 2, 3];
    let x = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0, 3.0]];
    let np = NdPPoly::new(c_tensor, c_shape, x).expect("ndppoly");
    let pts: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let t = i as f64 / m as f64;
            vec![t * 2.0, t * 3.0]
        })
        .collect();
    group.bench_function("ndppoly/evaluate_many", |b| {
        b.iter(|| np.evaluate_many(&pts))
    });
    group.bench_function("ndppoly/map_evaluate", |b| {
        b.iter(|| pts.iter().map(|p| np.evaluate(p)).collect::<Vec<_>>())
    });

    group.finish();
}

/// Large-m batch eval to exercise the parallel path (BPoly par_query_map gate
/// m·k≥2¹⁸; NdPPoly gate m·total≥2¹⁶) and check it doesn't regress on the CHEAP
/// per-point work typical of low-degree/low-dim splines. frankenscipy-yw7ts A/B.
fn bench_batch_eval_large(c: &mut Criterion) {
    use fsci_interpolate::{BPoly, NdPPoly};
    let m = 200_000usize;
    let mut group = c.benchmark_group("batch_eval_large");

    let n_pieces = 200usize;
    let bx: Vec<f64> = (0..=n_pieces).map(|i| i as f64).collect();
    let bc: Vec<Vec<f64>> = (0..n_pieces)
        .map(|i| vec![i as f64, (i + 1) as f64, 0.5, 1.5])
        .collect();
    let bp = BPoly::new(bc, bx).expect("bpoly");
    let qs: Vec<f64> = (0..m)
        .map(|i| (i as f64) * n_pieces as f64 / m as f64)
        .collect();
    group.bench_function("bpoly/200k", |b| b.iter(|| bp.evaluate_many(&qs)));

    let c_tensor: Vec<f64> = (1..=36).map(|v| v as f64).collect();
    let c_shape = vec![3usize, 2, 2, 3];
    let x = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0, 3.0]];
    let np = NdPPoly::new(c_tensor, c_shape, x).expect("ndppoly");
    let pts: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let t = i as f64 / m as f64;
            vec![t * 2.0, t * 3.0]
        })
        .collect();
    group.bench_function("ndppoly/200k", |b| b.iter(|| np.evaluate_many(&pts)));

    group.finish();
}

fn bench_smoothing_spline(c: &mut Criterion) {
    let mut group = c.benchmark_group("smoothing_spline_gcv");
    for &n in &[200usize, 500, 1000, 2000, 5000] {
        // deterministic noisy data; lam=None => GCV path (factor-once banded-Cholesky trace)
        let x: Vec<f64> = (0..n).map(|i| 10.0 * i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi.sin() + 0.1 * ((i as f64 * 12.9898).sin() * 43758.5453).fract())
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| make_smoothing_spline(black_box(&x), black_box(&y), None, None).unwrap())
        });
    }
    group.finish();
}

fn bench_smooth_bivariate(c: &mut Criterion) {
    let mut group = c.benchmark_group("smooth_bivariate_spline");
    for &m in &[400usize, 1000, 2500] {
        let x: Vec<f64> = (0..m)
            .map(|i| ((i as f64 * 12.9898).sin() * 43758.5).fract())
            .collect();
        let y: Vec<f64> = (0..m)
            .map(|i| ((i as f64 * 78.233).sin() * 12345.6).fract())
            .collect();
        let z: Vec<f64> = x
            .iter()
            .zip(&y)
            .map(|(&xi, &yi)| (6.0 * xi).sin() * (6.0 * yi).cos())
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(m), &m, |b, _| {
            b.iter(|| {
                SmoothBivariateSpline::new(
                    black_box(&x),
                    black_box(&y),
                    black_box(&z),
                    SmoothBivariateSplineOptions::default(),
                )
                .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_bisplrep(c: &mut Criterion) {
    let mut group = c.benchmark_group("bisplrep");
    for &m in &[400usize, 1000, 2500] {
        let x: Vec<f64> = (0..m)
            .map(|i| ((i as f64 * 12.9898).sin() * 43758.5).fract())
            .collect();
        let y: Vec<f64> = (0..m)
            .map(|i| ((i as f64 * 78.233).sin() * 12345.6).fract())
            .collect();
        let z: Vec<f64> = x
            .iter()
            .zip(&y)
            .map(|(&xi, &yi)| (6.0 * xi).sin() * (6.0 * yi).cos())
            .collect();
        let s = m as f64;
        group.bench_with_input(BenchmarkId::from_parameter(m), &m, |b, _| {
            b.iter(|| {
                bisplrep(
                    black_box(&x),
                    black_box(&y),
                    black_box(&z),
                    3,
                    3,
                    black_box(s),
                )
                .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_make_lsq(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_lsq_spline");
    let k = 3usize;
    for &nk in &[200usize, 1000, 3000] {
        let m = nk * 4;
        let mut x: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        x.sort_by(|a, b| a.total_cmp(b));
        let y: Vec<f64> = x.iter().map(|&xi| (10.0 * xi).sin()).collect();
        // interior knots from quantiles + clamped boundary
        let n_int = nk - 2;
        let mut t: Vec<f64> = vec![0.0; k + 1];
        for j in 1..=n_int {
            t.push(j as f64 / (n_int + 1) as f64);
        }
        t.extend(std::iter::repeat(1.0).take(k + 1));
        group.bench_with_input(BenchmarkId::from_parameter(nk), &nk, |b, _| {
            b.iter(|| make_interp_spline_lsq_probe(black_box(&x), black_box(&y), black_box(&t), k))
        });
    }
    group.finish();
}
fn make_interp_spline_lsq_probe(x: &[f64], y: &[f64], t: &[f64], k: usize) -> usize {
    fsci_interpolate::make_lsq_spline(x, y, t, k)
        .map(|_| 1usize)
        .unwrap_or(0)
}

criterion_group!(
    benches,
    bench_interp1d,
    bench_splines,
    bench_cubic_cursor_eval_many_ab,
    bench_polynomial,
    bench_regular_grid,
    bench_scattered,
    bench_rbf_and_rect,
    bench_make_interp_spline,
    bench_smoothing_spline,
    bench_smooth_bivariate,
    bench_bisplrep,
    bench_make_lsq,
    bench_rbf_scattered,
    bench_batch_eval,
    bench_batch_eval_large
);
criterion_main!(benches);
