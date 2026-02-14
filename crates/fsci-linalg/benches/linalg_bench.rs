use criterion::{Criterion, criterion_group, criterion_main};
use fsci_linalg::{InvOptions, SolveOptions, inv, solve};

fn make_diag_dominant(n: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i == j {
                (n as f64) * 2.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }
    a
}

fn make_rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i + 1) as f64).collect()
}

fn bench_solve(c: &mut Criterion) {
    for &n in &[4, 16, 64] {
        let a = make_diag_dominant(n);
        let b = make_rhs(n);
        c.bench_function(&format!("solve_{n}x{n}"), |bencher| {
            bencher.iter(|| solve(&a, &b, SolveOptions::default()).unwrap());
        });
    }
}

fn bench_inv(c: &mut Criterion) {
    for &n in &[4, 16, 64] {
        let a = make_diag_dominant(n);
        c.bench_function(&format!("inv_{n}x{n}"), |bencher| {
            bencher.iter(|| inv(&a, InvOptions::default()).unwrap());
        });
    }
}

criterion_group!(benches, bench_solve, bench_inv);
criterion_main!(benches);
