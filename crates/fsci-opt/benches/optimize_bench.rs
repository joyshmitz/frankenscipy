#![forbid(unsafe_code)]
//! Criterion benchmarks for fsci-opt (P2C-003-H).
//!
//! Groups: bfgs, cg, powell, brentq, brenth, bisect, ridder

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_opt::{
    MinimizeOptions, OptimizeMethod, RootOptions, RootMethod,
    bfgs, bisect, brentq, brenth, cg_pr_plus, powell, ridder,
};
use fsci_runtime::RuntimeMode;

// ── Test functions ────────────────────────────────────────────────────

fn rosenbrock(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        s += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    s
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
        group.bench_with_input(BenchmarkId::new("rosenbrock", dim), &(x0.clone()), |b, x0| {
            b.iter(|| {
                let _ = bfgs(&rosenbrock, x0, opts(OptimizeMethod::Bfgs));
            });
        });
        group.bench_with_input(BenchmarkId::new("quadratic", dim), &(x0.clone()), |b, x0| {
            b.iter(|| {
                let _ = bfgs(&quadratic, x0, opts(OptimizeMethod::Bfgs));
            });
        });
    }
    group.finish();
}

fn bench_cg(c: &mut Criterion) {
    let mut group = c.benchmark_group("cg");
    for &dim in &[2usize, 5, 10] {
        let x0: Vec<f64> = vec![0.0; dim];
        group.bench_with_input(BenchmarkId::new("rosenbrock", dim), &(x0.clone()), |b, x0| {
            b.iter(|| {
                let _ = cg_pr_plus(&rosenbrock, x0, opts(OptimizeMethod::ConjugateGradient));
            });
        });
        group.bench_with_input(BenchmarkId::new("quadratic", dim), &(x0.clone()), |b, x0| {
            b.iter(|| {
                let _ = cg_pr_plus(&quadratic, x0, opts(OptimizeMethod::ConjugateGradient));
            });
        });
    }
    group.finish();
}

fn bench_powell(c: &mut Criterion) {
    let mut group = c.benchmark_group("powell");
    for &dim in &[2usize, 5, 10] {
        let x0: Vec<f64> = vec![0.0; dim];
        group.bench_with_input(BenchmarkId::new("rosenbrock", dim), &(x0.clone()), |b, x0| {
            b.iter(|| {
                let _ = powell(&rosenbrock, x0, opts(OptimizeMethod::Powell));
            });
        });
        group.bench_with_input(BenchmarkId::new("quadratic", dim), &(x0.clone()), |b, x0| {
            b.iter(|| {
                let _ = powell(&quadratic, x0, opts(OptimizeMethod::Powell));
            });
        });
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

criterion_group!(
    benches,
    bench_bfgs,
    bench_cg,
    bench_powell,
    bench_brentq,
    bench_brenth,
    bench_bisect,
    bench_ridder,
);
criterion_main!(benches);
