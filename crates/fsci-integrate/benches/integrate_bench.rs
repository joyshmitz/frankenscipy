use criterion::{Criterion, criterion_group, criterion_main};
use fsci_integrate::{SolveIvpOptions, SolverKind, ToleranceValue, solve_ivp, validate_tol};
use fsci_runtime::RuntimeMode;

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

criterion_group!(
    benches,
    bench_solve_ivp_exponential,
    bench_solve_ivp_lorenz,
    bench_validate_tol
);
criterion_main!(benches);
