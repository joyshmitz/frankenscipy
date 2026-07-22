use criterion::{Criterion, criterion_group, criterion_main};
use fsci_integrate::bdf::BDF_FORCE_PER_ITER_ALLOC;
use fsci_integrate::radau::RADAU_FORCE_PER_ITER_ALLOC;
use fsci_integrate::{
    IntegrateValidationError, MIN_RTOL, SolveIvpOptions, SolverKind, ToleranceValue,
    ToleranceWarning, ValidatedTolerance, solve_ivp, trapezoid_irregular, trapezoid_richardson,
    validate_tol,
};
use fsci_runtime::RuntimeMode;
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// Exponential decay: y' = -y, y(0) = 1.
fn exponential_decay(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![-y[0]]
}

/// Lorenz system (3D ODE).
fn lorenz(_t: f64, y: &[f64]) -> Vec<f64> {
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![
        sigma * (y[1] - y[0]),
        y[0] * (rho - y[2]) - y[1],
        y[0] * y[1] - beta * y[2],
    ]
}

fn bench_solve_ivp_exponential(c: &mut Criterion) {
    let y0 = [1.0];
    let opts = SolveIvpOptions {
        t_span: (0.0, 10.0),
        y0: &y0,
        method: SolverKind::Rk45,
        rtol: 1e-6,
        atol: ToleranceValue::Scalar(1e-9),
        max_step: f64::INFINITY,
        mode: RuntimeMode::Strict,
        ..Default::default()
    };
    c.bench_function("solve_ivp_exponential_rk45", |b| {
        b.iter(|| {
            let mut rhs = exponential_decay;
            solve_ivp(&mut rhs, &opts)
        });
    });
}

fn bench_solve_ivp_lorenz(c: &mut Criterion) {
    let y0 = [1.0, 1.0, 1.0];
    let opts = SolveIvpOptions {
        t_span: (0.0, 1.0),
        y0: &y0,
        method: SolverKind::Rk45,
        rtol: 1e-6,
        atol: ToleranceValue::Scalar(1e-9),
        max_step: f64::INFINITY,
        mode: RuntimeMode::Strict,
        ..Default::default()
    };
    c.bench_function("solve_ivp_lorenz_rk45", |b| {
        b.iter(|| {
            let mut rhs = lorenz;
            solve_ivp(&mut rhs, &opts)
        });
    });
}

fn bench_solve_ivp_t_eval_validation(c: &mut Criterion) {
    let y0 = [1.0];
    let mut t_eval = (0..16_384)
        .map(|idx| idx as f64 / 16_384.0)
        .collect::<Vec<_>>();
    *t_eval.last_mut().expect("t_eval should be nonempty") = 2.0;
    let opts = SolveIvpOptions {
        t_span: (0.0, 1.0),
        y0: &y0,
        method: SolverKind::Rk45,
        t_eval: Some(&t_eval),
        ..Default::default()
    };

    c.bench_function("solve_ivp_t_eval_out_of_span_16384", |b| {
        b.iter(|| {
            let mut rhs = exponential_decay;
            black_box(solve_ivp(&mut rhs, black_box(&opts)))
        });
    });
}

fn solve_ivp_non_finite_y0_eager_original(
    options: &SolveIvpOptions<'_>,
) -> Result<(), IntegrateValidationError> {
    let (t0, tf) = options.t_span;
    if options.y0.is_empty() {
        return Err(IntegrateValidationError::EmptyY0);
    }
    if !t0.is_finite() || !tf.is_finite() {
        return Err(IntegrateValidationError::NonFiniteSpan);
    }
    if options.y0.iter().any(|value| !value.is_finite()) {
        let fingerprint = format!(
            "solve_ivp:reason=non_finite_y0;y0={:?};mode={:?}",
            options.y0, options.mode
        )
        .into_bytes();
        black_box(&fingerprint);
        return Err(IntegrateValidationError::NonFiniteY0);
    }
    Ok(())
}

fn bench_solve_ivp_non_finite_y0_validation(c: &mut Criterion) {
    let mut y0 = vec![1.0; 16_384];
    *y0.last_mut().expect("y0 should be nonempty") = f64::NAN;
    let opts = SolveIvpOptions {
        t_span: (0.0, 1.0),
        y0: &y0,
        method: SolverKind::Rk45,
        ..Default::default()
    };

    let mut group = c.benchmark_group("solve_ivp_non_finite_y0_audit_fingerprint_ab");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("original_eager/16384", |b| {
        b.iter(|| solve_ivp_non_finite_y0_eager_original(black_box(&opts)));
    });
    group.bench_function("current_lazy/16384", |b| {
        b.iter(|| {
            let mut rhs = exponential_decay;
            black_box(solve_ivp(&mut rhs, black_box(&opts)))
        });
    });
    group.finish();
}

fn bench_validate_tol(c: &mut Criterion) {
    c.bench_function("validate_tol_scalar", |b| {
        b.iter(|| {
            validate_tol(
                ToleranceValue::Scalar(1e-6),
                ToleranceValue::Scalar(1e-9),
                100,
                RuntimeMode::Strict,
            )
        });
    });
    let atol_vec = vec![1e-8; 100];
    c.bench_function("validate_tol_vector_100", |b| {
        b.iter(|| {
            validate_tol(
                ToleranceValue::Scalar(1e-6),
                ToleranceValue::Vector(atol_vec.clone()),
                100,
                RuntimeMode::Strict,
            )
        });
    });
}

fn tolerance_any_owned(value: ToleranceValue, mut predicate: impl FnMut(f64) -> bool) -> bool {
    match value {
        ToleranceValue::Scalar(value) => predicate(value),
        ToleranceValue::Vector(values) => values.into_iter().any(predicate),
    }
}

fn tolerance_map_owned(value: ToleranceValue, mut f: impl FnMut(f64) -> f64) -> ToleranceValue {
    match value {
        ToleranceValue::Scalar(value) => ToleranceValue::Scalar(f(value)),
        ToleranceValue::Vector(values) => {
            ToleranceValue::Vector(values.into_iter().map(f).collect())
        }
    }
}

fn validate_tol_clone_scans_original(
    rtol: ToleranceValue,
    atol: ToleranceValue,
    n: usize,
    mode: RuntimeMode,
) -> Result<ValidatedTolerance, IntegrateValidationError> {
    let mut warnings = Vec::new();
    if tolerance_any_owned(rtol.clone(), f64::is_nan) {
        return Err(IntegrateValidationError::NonFiniteRtol);
    }
    if tolerance_any_owned(atol.clone(), f64::is_nan) {
        return Err(IntegrateValidationError::NonFiniteAtol);
    }
    let needs_clamp = tolerance_any_owned(rtol.clone(), |x| x < MIN_RTOL);
    let rtol = if needs_clamp {
        warnings.push(ToleranceWarning::RtolClamped { minimum: MIN_RTOL });
        tolerance_map_owned(rtol, |x| x.max(MIN_RTOL))
    } else {
        rtol
    };

    if let ToleranceValue::Vector(values) = &atol
        && values.len() != n
    {
        return Err(IntegrateValidationError::AtolWrongShape {
            expected: n,
            actual: values.len(),
        });
    }
    if tolerance_any_owned(atol.clone(), |x| x < 0.0) {
        return Err(IntegrateValidationError::AtolMustBePositive);
    }

    Ok(ValidatedTolerance {
        rtol,
        atol,
        mode,
        warnings,
    })
}

fn validate_tol_eager_fingerprint_original(
    rtol: ToleranceValue,
    atol: ToleranceValue,
    n: usize,
    mode: RuntimeMode,
) -> Result<ValidatedTolerance, IntegrateValidationError> {
    let fingerprint =
        format!("validate_tol:rtol={rtol:?};atol={atol:?};n={n};mode={mode:?}").into_bytes();
    black_box(&fingerprint);
    validate_tol_clone_scans_original(rtol, atol, n, mode)
}

fn bench_validate_tol_audit_fingerprint_ab(c: &mut Criterion) {
    const N: usize = 16_384;
    let atol = vec![1e-8; N];
    let mut group = c.benchmark_group("validate_tol_audit_fingerprint_ab");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("original", |b| {
        b.iter(|| {
            validate_tol_eager_fingerprint_original(
                ToleranceValue::Scalar(black_box(1e-6)),
                ToleranceValue::Vector(black_box(atol.clone())),
                black_box(N),
                RuntimeMode::Strict,
            )
        });
    });
    group.bench_function("current", |b| {
        b.iter(|| {
            validate_tol(
                ToleranceValue::Scalar(black_box(1e-6)),
                ToleranceValue::Vector(black_box(atol.clone())),
                black_box(N),
                RuntimeMode::Strict,
            )
        });
    });
    group.finish();
}

fn bench_validate_tol_clone_scan_ab(c: &mut Criterion) {
    const N: usize = 16_384;
    let atol = vec![1e-8; N];
    let mut group = c.benchmark_group("validate_tol_clone_scan_ab");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    // This bench-local source model is a calibration witness only. Cross-crate
    // optimization makes the public `current` row the acceptance comparator.
    group.bench_function("owned_scan_source_model", |b| {
        b.iter(|| {
            black_box(validate_tol_clone_scans_original(
                ToleranceValue::Scalar(black_box(1e-6)),
                ToleranceValue::Vector(black_box(atol.clone())),
                black_box(N),
                RuntimeMode::Strict,
            ))
        });
    });
    group.bench_function("current", |b| {
        b.iter(|| {
            black_box(validate_tol(
                ToleranceValue::Scalar(black_box(1e-6)),
                ToleranceValue::Vector(black_box(atol.clone())),
                black_box(N),
                RuntimeMode::Strict,
            ))
        });
    });
    group.finish();
}

fn trapezoid_richardson_collected_original(y: &[f64], x: &[f64]) -> f64 {
    if y.len() < 2 || x.len() != y.len() {
        return 0.0;
    }
    let t1 = trapezoid_irregular(y, x);
    if y.len() < 5 {
        return t1;
    }
    let y2: Vec<f64> = y.iter().step_by(2).copied().collect();
    let x2: Vec<f64> = x.iter().step_by(2).copied().collect();
    let t2 = trapezoid_irregular(&y2, &x2);
    (4.0 * t1 - t2) / 3.0
}

fn bench_trapezoid_richardson_allocation_ab(c: &mut Criterion) {
    const N: usize = 65_537;
    let x = (0..N).map(|i| i as f64 * 1e-4).collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|value| (17.0 * value).sin() + value * 0.25)
        .collect::<Vec<_>>();
    assert_eq!(
        trapezoid_richardson(&y, &x).to_bits(),
        trapezoid_richardson_collected_original(&y, &x).to_bits()
    );

    let mut group = c.benchmark_group("trapezoid_richardson_allocation_ab");
    group.bench_function("original_collected/65537", |b| {
        b.iter(|| {
            black_box(trapezoid_richardson_collected_original(
                black_box(&y),
                black_box(&x),
            ))
        });
    });
    group.bench_function("candidate_strided/65537", |b| {
        b.iter(|| black_box(trapezoid_richardson(black_box(&y), black_box(&x))));
    });
    group.finish();
}

/// Stiff Van der Pol oscillator (n=2), μ large ⇒ the Radau simplified-Newton
/// corrector fires many times per step. Small n makes each Newton iteration
/// malloc-bound, so this maximizes the visible per-iteration-allocation cost.
fn van_der_pol_stiff(_t: f64, y: &[f64]) -> Vec<f64> {
    const MU: f64 = 1000.0;
    vec![y[1], MU * (1.0 - y[0] * y[0]) * y[1] - y[0]]
}

fn bench_radau_newton_alloc_ab(c: &mut Criterion) {
    let y0 = [2.0, 0.0];
    let opts = SolveIvpOptions {
        t_span: (0.0, 30.0),
        y0: &y0,
        method: SolverKind::Radau,
        rtol: 1e-6,
        atol: ToleranceValue::Scalar(1e-9),
        max_step: f64::INFINITY,
        mode: RuntimeMode::Strict,
        ..Default::default()
    };

    // Byte-identical proof: hoisted-scratch (default) vs per-iteration-alloc trajectories
    // must match bit-for-bit.
    RADAU_FORCE_PER_ITER_ALLOC.store(true, Ordering::Relaxed);
    let base = {
        let mut rhs = van_der_pol_stiff;
        solve_ivp(&mut rhs, &opts).expect("radau baseline")
    };
    RADAU_FORCE_PER_ITER_ALLOC.store(false, Ordering::Relaxed);
    let cand = {
        let mut rhs = van_der_pol_stiff;
        solve_ivp(&mut rhs, &opts).expect("radau candidate")
    };
    assert_eq!(base.y.len(), cand.y.len(), "step count differs");
    for (rb, rc) in base.y.iter().zip(cand.y.iter()) {
        for (a, b) in rb.iter().zip(rc.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "radau trajectory mismatch");
        }
    }

    let mut group = c.benchmark_group("radau_newton_alloc_ab");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("original_per_iter_alloc", |b| {
        RADAU_FORCE_PER_ITER_ALLOC.store(true, Ordering::Relaxed);
        b.iter(|| {
            let mut rhs = van_der_pol_stiff;
            black_box(solve_ivp(&mut rhs, black_box(&opts)))
        });
    });
    group.bench_function("current_hoisted", |b| {
        RADAU_FORCE_PER_ITER_ALLOC.store(false, Ordering::Relaxed);
        b.iter(|| {
            let mut rhs = van_der_pol_stiff;
            black_box(solve_ivp(&mut rhs, black_box(&opts)))
        });
    });
    RADAU_FORCE_PER_ITER_ALLOC.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_bdf_newton_alloc_ab(c: &mut Criterion) {
    let y0 = [2.0, 0.0];
    let opts = SolveIvpOptions {
        t_span: (0.0, 30.0),
        y0: &y0,
        method: SolverKind::Bdf,
        rtol: 1e-6,
        atol: ToleranceValue::Scalar(1e-9),
        max_step: f64::INFINITY,
        mode: RuntimeMode::Strict,
        ..Default::default()
    };

    // Byte-identical proof: hoisted-scratch (default) vs per-iteration-alloc trajectories
    // must match bit-for-bit.
    BDF_FORCE_PER_ITER_ALLOC.store(true, Ordering::Relaxed);
    let base = {
        let mut rhs = van_der_pol_stiff;
        solve_ivp(&mut rhs, &opts).expect("bdf baseline")
    };
    BDF_FORCE_PER_ITER_ALLOC.store(false, Ordering::Relaxed);
    let cand = {
        let mut rhs = van_der_pol_stiff;
        solve_ivp(&mut rhs, &opts).expect("bdf candidate")
    };
    assert_eq!(base.y.len(), cand.y.len(), "bdf step count differs");
    for (rb, rc) in base.y.iter().zip(cand.y.iter()) {
        for (a, b) in rb.iter().zip(rc.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "bdf trajectory mismatch");
        }
    }

    let mut group = c.benchmark_group("bdf_newton_alloc_ab");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("original_per_iter_alloc", |b| {
        BDF_FORCE_PER_ITER_ALLOC.store(true, Ordering::Relaxed);
        b.iter(|| {
            let mut rhs = van_der_pol_stiff;
            black_box(solve_ivp(&mut rhs, black_box(&opts)))
        });
    });
    group.bench_function("current_hoisted", |b| {
        BDF_FORCE_PER_ITER_ALLOC.store(false, Ordering::Relaxed);
        b.iter(|| {
            let mut rhs = van_der_pol_stiff;
            black_box(solve_ivp(&mut rhs, black_box(&opts)))
        });
    });
    BDF_FORCE_PER_ITER_ALLOC.store(false, Ordering::Relaxed);
    group.finish();
}

criterion_group!(
    benches,
    bench_radau_newton_alloc_ab,
    bench_bdf_newton_alloc_ab,
    bench_solve_ivp_exponential,
    bench_solve_ivp_lorenz,
    bench_solve_ivp_t_eval_validation,
    bench_solve_ivp_non_finite_y0_validation,
    bench_validate_tol,
    bench_validate_tol_audit_fingerprint_ab,
    bench_validate_tol_clone_scan_ab,
    bench_trapezoid_richardson_allocation_ab
);
criterion_main!(benches);
