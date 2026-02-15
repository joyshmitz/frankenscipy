use criterion::{Criterion, criterion_group, criterion_main};
use fsci_linalg::{
    InvOptions, LstsqOptions, PinvOptions, SolveOptions, TriangularSolveOptions, det, inv, lstsq,
    pinv, solve, solve_banded, solve_triangular,
};
use fsci_runtime::RuntimeMode;

const SIZES: &[usize] = &[4, 16, 64, 256];

/// Diagonally-dominant matrix: guaranteed non-singular, well-conditioned.
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

/// Upper-triangular matrix extracted from diag-dominant.
fn make_upper_triangular(n: usize) -> Vec<Vec<f64>> {
    let mut a = make_diag_dominant(n);
    for (i, row) in a.iter_mut().enumerate() {
        for cell in row.iter_mut().take(i) {
            *cell = 0.0;
        }
    }
    a
}

/// Tall rectangular matrix (rows > cols) for lstsq/pinv.
fn make_overdetermined(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; cols]; rows];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i == j {
                (cols as f64) * 2.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }
    a
}

/// Tridiagonal banded matrix in LAPACK band storage format.
/// Returns (l_and_u, ab) where ab has shape (3, n).
fn make_tridiag_banded(n: usize) -> ((usize, usize), Vec<Vec<f64>>) {
    // LAPACK band storage: ab[nupper + i - j][j] = A[i][j]
    // For tridiag (l=1, u=1):
    //   ab[0][j] = A[j-1][j]  (superdiag, j >= 1)
    //   ab[1][j] = A[j][j]    (main diagonal)
    //   ab[2][j] = A[j+1][j]  (subdiag, j <= n-2)
    let mut ab = vec![vec![0.0; n]; 3];
    for col in &mut ab[1] {
        *col = 4.0;
    }
    for col in ab[0].iter_mut().skip(1) {
        *col = -1.0;
    }
    for col in ab[2].iter_mut().take(n.saturating_sub(1)) {
        *col = -1.0;
    }
    ((1, 1), ab)
}

fn make_rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i + 1) as f64).collect()
}

// ── solve ──────────────────────────────────────────────────────────────────────

fn bench_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve");
    for &n in SIZES {
        let a = make_diag_dominant(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| solve(&a, &b, SolveOptions::default()).unwrap());
        });
    }
    group.finish();
}

// ── solve_triangular ───────────────────────────────────────────────────────────

fn bench_solve_triangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_triangular");
    for &n in SIZES {
        let a = make_upper_triangular(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| solve_triangular(&a, &b, TriangularSolveOptions::default()).unwrap());
        });
    }
    group.finish();
}

// ── solve_banded ───────────────────────────────────────────────────────────────

fn bench_solve_banded(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_banded");
    for &n in SIZES {
        let (l_u, ab) = make_tridiag_banded(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| solve_banded(l_u, &ab, &b, SolveOptions::default()).unwrap());
        });
    }
    group.finish();
}

// ── inv ────────────────────────────────────────────────────────────────────────

fn bench_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("inv");
    for &n in SIZES {
        let a = make_diag_dominant(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| inv(&a, InvOptions::default()).unwrap());
        });
    }
    group.finish();
}

// ── det ────────────────────────────────────────────────────────────────────────

fn bench_det(c: &mut Criterion) {
    let mut group = c.benchmark_group("det");
    for &n in SIZES {
        let a = make_diag_dominant(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| det(&a, RuntimeMode::Strict, true).unwrap());
        });
    }
    group.finish();
}

// ── lstsq ──────────────────────────────────────────────────────────────────────

fn bench_lstsq(c: &mut Criterion) {
    let mut group = c.benchmark_group("lstsq");
    for &n in SIZES {
        // Overdetermined: 2n rows x n cols
        let rows = n * 2;
        let a = make_overdetermined(rows, n);
        let b = make_rhs(rows);
        group.bench_function(format!("{rows}x{n}"), |bencher| {
            bencher.iter(|| lstsq(&a, &b, LstsqOptions::default()).unwrap());
        });
    }
    group.finish();
}

// ── pinv ───────────────────────────────────────────────────────────────────────

fn bench_pinv(c: &mut Criterion) {
    let mut group = c.benchmark_group("pinv");
    for &n in SIZES {
        let rows = n * 2;
        let a = make_overdetermined(rows, n);
        group.bench_function(format!("{rows}x{n}"), |bencher| {
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_solve,
    bench_solve_triangular,
    bench_solve_banded,
    bench_inv,
    bench_det,
    bench_lstsq,
    bench_pinv
);
criterion_main!(benches);
