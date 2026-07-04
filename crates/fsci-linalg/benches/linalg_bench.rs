use std::{
    hint::black_box,
    io::Write,
    process::{Command, Stdio},
    time::Duration,
};

use criterion::{Criterion, criterion_group, criterion_main};
use fsci_linalg::{
    DecompOptions, InvOptions, LstsqOptions, MatrixAssumption, PinvOptions, SolveOptions,
    TriangularSolveOptions, cho_factor, cho_solve, det, dft, eigh, inv, lstsq, lu_factor, lu_solve,
    matmul, orthogonal_procrustes, pinv, randomized_eigh, solve, solve_banded, solve_triangular,
    svd,
};
use fsci_runtime::RuntimeMode;
use nalgebra::{DMatrix, DVector, Dyn, LU};

// Per SPEC §17, baseline sizes for dense solve family.
// SIZES: quick smoke tests; BASELINE_SIZES: full p50/p95/p99 capture
const SIZES: &[usize] = &[4, 16, 64, 256];
const BASELINE_SIZES: &[usize] = &[100, 500, 1000, 2000, 4000];
const MATMUL_SIZES: &[usize] = &[256, 512, 768, 1024];
const EIGH_SIZES: &[usize] = &[256, 512];
const RANDOMIZED_EIGH_CASES: &[(usize, usize)] = &[(256, 16), (512, 24)];
const DFT_GAUNTLET_SIZES: &[usize] = &[256, 512, 1024];
const CHO_FACTOR_GAUNTLET_SIZES: &[usize] = &[500, 1000];
const LU_FACTOR_GAUNTLET_SIZES: &[usize] = &[1000];
const ORTHOGONAL_PROCRUSTES_CASES: &[(usize, usize)] = &[(3000, 150), (5000, 200)];
const LU_FACTOR_SOLVE_CASES: &[usize] = &[1000, 1500];

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

/// Wide full-row-rank matrix for minimum-norm lstsq/pinv routes.
fn make_underdetermined(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; cols]; rows];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i == j {
                (rows as f64) * 2.0
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

fn make_matmul_matrix(rows: usize, cols: usize, seed: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| ((i * 31 + j * 17 + seed) % 97) as f64 * 0.01)
                .collect()
        })
        .collect()
}

fn bench_dmatrix_from_rows(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let m = rows.len();
    let n = rows.first().map_or(0, Vec::len);
    let mut data = Vec::with_capacity(m * n);
    for col in 0..n {
        for row in rows {
            data.push(row[col]);
        }
    }
    DMatrix::from_vec(m, n, data)
}

fn bench_matrix_norm1(matrix: &DMatrix<f64>) -> f64 {
    let mut max_col_sum = 0.0_f64;
    for col in 0..matrix.ncols() {
        let mut sum = 0.0_f64;
        for row in 0..matrix.nrows() {
            let value = matrix[(row, col)].abs();
            if !value.is_finite() {
                return f64::NAN;
            }
            sum += value;
        }
        max_col_sum = max_col_sum.max(sum);
    }
    max_col_sum
}

fn bench_solve_lu_transpose(
    lu: &LU<f64, Dyn, Dyn>,
    u_t: &DMatrix<f64>,
    l_t: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Option<DVector<f64>> {
    let y = u_t.solve_lower_triangular(b)?;
    let z = l_t.solve_upper_triangular(&y)?;
    let mut x = z;
    lu.p().inv_permute_rows(&mut x);
    Some(x)
}

fn bench_fast_rcond_from_lu(lu: &LU<f64, Dyn, Dyn>, a_norm: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if a_norm == 0.0 || !a_norm.is_finite() {
        return 0.0;
    }

    let mut x = DVector::from_element(n, 1.0 / (n as f64));
    let mut inv_a_norm = 0.0;
    let u_t = lu.u().transpose();
    let l_t = lu.l().transpose();

    for _ in 0..5 {
        let sign_x = x.map(|value| if value >= 0.0 { 1.0 } else { -1.0 });
        let Some(w) = bench_solve_lu_transpose(lu, &u_t, &l_t, &sign_x) else {
            return 0.0;
        };
        let sign_w = w.map(|value| if value >= 0.0 { 1.0 } else { -1.0 });
        let Some(x_new) = lu.solve(&sign_w) else {
            return 0.0;
        };

        let new_norm = x_new.lp_norm(1);
        let direction_delta = x_new
            .iter()
            .zip(x.iter())
            .map(|(&new, &old)| (new - old).abs())
            .sum::<f64>();
        if (new_norm - inv_a_norm).abs() <= 1e-10 * new_norm {
            inv_a_norm = new_norm;
            break;
        }
        inv_a_norm = new_norm;
        x = x_new;

        if direction_delta <= f64::EPSILON * new_norm {
            break;
        }
    }

    if inv_a_norm <= 0.0 {
        return 0.0;
    }
    let rcond = 1.0 / (a_norm * inv_a_norm);
    if rcond.is_nan() { 0.0 } else { rcond.min(1.0) }
}

fn lu_factor_solve_original_nalgebra(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let matrix = bench_dmatrix_from_rows(a);
    let a_norm = bench_matrix_norm1(&matrix);
    let lu = matrix.lu();
    black_box(bench_fast_rcond_from_lu(&lu, a_norm, a.len()));
    let rhs = DVector::from_column_slice(b);
    lu.solve(&rhs)
        .expect("original nalgebra lu solve")
        .iter()
        .copied()
        .collect()
}

#[allow(clippy::needless_range_loop)]
fn make_symmetric_eigh_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let value = if i == j {
                (n as f64) * 3.0 + (i as f64) * 0.01
            } else {
                1.0 / ((i - j + 1) as f64)
            };
            a[i][j] = value;
            a[j][i] = value;
        }
    }
    a
}

#[allow(clippy::needless_range_loop)]
fn make_low_rank_symmetric_eigh_matrix(n: usize, rank: usize) -> Vec<Vec<f64>> {
    let mut factors = vec![vec![0.0; n]; rank];
    for (r, row) in factors.iter_mut().enumerate() {
        for (i, value) in row.iter_mut().enumerate() {
            let x = ((r + 3) * (i + 5)) as f64;
            *value = (x.sin() * 0.5) + (x.cos() * 0.25);
        }
    }

    let mut a = vec![vec![0.0; n]; n];
    for r in 0..rank {
        let weight = (rank - r) as f64;
        for i in 0..n {
            let scaled = weight * factors[r][i];
            for j in 0..=i {
                a[i][j] += scaled * factors[r][j];
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            a[j][i] = a[i][j];
        }
    }
    a
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

// ── matmul ────────────────────────────────────────────────────────────────────

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    for &n in MATMUL_SIZES {
        let a = make_matmul_matrix(n, n, 0);
        let b = make_matmul_matrix(n, n, 11);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| matmul(std::hint::black_box(&a), std::hint::black_box(&b)).unwrap());
        });
    }
    group.finish();
}

// ── eigh ──────────────────────────────────────────────────────────────────────

fn bench_eigh_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigh_dense");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    for &n in EIGH_SIZES {
        let a = make_symmetric_eigh_matrix(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| eigh(std::hint::black_box(&a), DecompOptions::default()).unwrap());
        });
    }
    group.finish();
}

fn bench_randomized_eigh(c: &mut Criterion) {
    let mut group = c.benchmark_group("randomized_eigh");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    for &(n, k) in RANDOMIZED_EIGH_CASES {
        let a = make_low_rank_symmetric_eigh_matrix(n, k + 8);
        group.bench_function(format!("{n}x{n}_k{k}"), |bencher| {
            bencher.iter(|| {
                randomized_eigh(
                    std::hint::black_box(&a),
                    std::hint::black_box(k),
                    8,
                    2,
                    0x5eed_1234,
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

// ══════════════════════════════════════════════════════════════════════════════
// BASELINE BENCHMARKS - Per SPEC §17, capture p50/p95/p99 at standard sizes
// Run with: cargo bench --bench linalg_bench -- baseline
// ══════════════════════════════════════════════════════════════════════════════

fn bench_baseline_solve(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut group = c.benchmark_group("baseline_solve");
    group.sample_size(100);
    for &n in BASELINE_SIZES {
        let a = make_diag_dominant(n);
        let b = make_rhs(n);
        // Mixed-precision (default) arm.
        group.bench_function(format!("{n}x{n}"), |bencher| {
            fsci_linalg::DISABLE_MIXED_LU.store(false, Relaxed);
            bencher.iter(|| solve(&a, &b, SolveOptions::default()).unwrap());
        });
        // Exact-f64 arm on the SAME worker/binary — the A/B baseline for the mixed route.
        group.bench_function(format!("{n}x{n}_f64"), |bencher| {
            fsci_linalg::DISABLE_MIXED_LU.store(true, Relaxed);
            bencher.iter(|| solve(&a, &b, SolveOptions::default()).unwrap());
            fsci_linalg::DISABLE_MIXED_LU.store(false, Relaxed);
        });
    }
    group.finish();
}

fn bench_lu_factor_gauntlet(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut group = c.benchmark_group("lu_factor_gauntlet");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for &n in LU_FACTOR_GAUNTLET_SIZES {
        let a = make_diag_dominant(n);
        group.bench_function(format!("{n}x{n}_flat_factor"), |bencher| {
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(false, Relaxed);
            bencher.iter(|| {
                black_box(lu_factor(black_box(&a), black_box(DecompOptions::default())).unwrap())
            });
        });
        group.bench_function(format!("{n}x{n}_orig_nalgebra_factor"), |bencher| {
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(true, Relaxed);
            bencher.iter(|| {
                black_box(lu_factor(black_box(&a), black_box(DecompOptions::default())).unwrap())
            });
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(false, Relaxed);
        });
    }
    group.finish();
}

/// Well-conditioned, finite-determinant matrix (diagonal 1.5): its `∏ pivots` stays
/// under `f64::MAX`, so `det`'s flat blocked-LU path is exercised rather than falling
/// back to nalgebra on the overflow guard (which `make_diag_dominant`'s huge diagonal
/// would trip).
fn make_det_finite(n: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i == j {
                1.5
            } else {
                0.4 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }
    a
}

/// Same-binary A/B for `det`: the flat blocked-LU determinant vs the original nalgebra
/// `.lu().determinant()`, toggled through `DISABLE_FLAT_LU_FACTOR`.
fn bench_det_gauntlet(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut group = c.benchmark_group("det_gauntlet");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for &n in LU_FACTOR_GAUNTLET_SIZES {
        let a = make_det_finite(n);
        group.bench_function(format!("{n}x{n}_flat_det"), |bencher| {
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(false, Relaxed);
            bencher.iter(|| black_box(det(black_box(&a), RuntimeMode::Strict, black_box(true)).unwrap()));
        });
        group.bench_function(format!("{n}x{n}_orig_nalgebra_det"), |bencher| {
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(true, Relaxed);
            bencher.iter(|| black_box(det(black_box(&a), RuntimeMode::Strict, black_box(true)).unwrap()));
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(false, Relaxed);
        });
    }
    group.finish();
}

fn bench_baseline_solve_pos(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_solve_pos");
    group.sample_size(100);
    for &n in BASELINE_SIZES {
        let a = make_symmetric_eigh_matrix(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| {
                solve(
                    std::hint::black_box(&a),
                    std::hint::black_box(&b),
                    SolveOptions {
                        assume_a: Some(MatrixAssumption::PositiveDefinite),
                        ..SolveOptions::default()
                    },
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_baseline_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_inv");
    group.sample_size(100);
    for &n in BASELINE_SIZES {
        let a = make_diag_dominant(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| inv(&a, InvOptions::default()).unwrap());
        });
    }
    group.finish();
}

fn bench_lu_factor_solve_gauntlet(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_factor_solve_gauntlet");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    group.warm_up_time(Duration::from_secs(1));
    for &n in LU_FACTOR_SOLVE_CASES {
        let a = make_diag_dominant(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}_original_nalgebra"), |bencher| {
            bencher.iter(|| lu_factor_solve_original_nalgebra(black_box(&a), black_box(&b)));
        });
        group.bench_function(format!("{n}x{n}_current_public"), |bencher| {
            bencher.iter(|| {
                let factor = lu_factor(black_box(&a), DecompOptions::default()).expect("lu_factor");
                lu_solve(black_box(&factor), black_box(&b))
                    .expect("lu_solve")
                    .x
            });
        });
    }
    group.finish();
}

fn bench_baseline_lstsq(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_lstsq");
    group.sample_size(100);
    for &n in BASELINE_SIZES {
        let rows = n * 2;
        let a = make_overdetermined(rows, n);
        let b = make_rhs(rows);
        group.bench_function(format!("{rows}x{n}"), |bencher| {
            bencher.iter(|| lstsq(&a, &b, LstsqOptions::default()).unwrap());
        });
    }
    group.finish();
}

fn bench_u0ucw_wide_lstsq(c: &mut Criterion) {
    let mut group = c.benchmark_group("u0ucw_wide_lstsq");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for &(rows, cols) in &[(500usize, 1000usize), (1000, 2000)] {
        let a = make_underdetermined(rows, cols);
        let b = make_rhs(rows);
        group.bench_function(format!("{rows}x{cols}_materialized_transpose"), |bencher| {
            bencher.iter(|| lstsq(&a, &b, LstsqOptions::default()).unwrap());
        });
    }
    group.finish();
}

fn bench_u0ucw_wide_pinv(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;

    let mut group = c.benchmark_group("u0ucw_wide_pinv");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for &(rows, cols) in &[(500usize, 1000usize), (1000, 2000)] {
        let a = make_underdetermined(rows, cols);
        group.bench_function(
            format!("{rows}x{cols}_normal_equation_cholesky"),
            |bencher| {
                fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
                fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
                bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            },
        );
        group.bench_function(format!("{rows}x{cols}_eigen_rcond_gate"), |bencher| {
            fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
            fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(true, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{cols}_svd_fallback"), |bencher| {
            fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(true, Relaxed);
            fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
        });
    }
    group.finish();
}

fn scipy_pinv_duration(rows: usize, cols: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

rows = int(sys.argv[1])
cols = int(sys.argv[2])
iters = int(sys.argv[3])
i = np.arange(rows, dtype=np.float64)[:, None]
j = np.arange(cols, dtype=np.float64)[None, :]
a = 1.0 / (np.abs(i - j) + 1.0)
d = np.arange(rows)
a[d, d] = rows * 2.0
la.pinv(a, check_finite=False)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    result = la.pinv(a, check_finite=False)
    checksum += float(result[0, 0])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args([
            "-",
            &rows.to_string(),
            &cols.to_string(),
            &iters.to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy pinv oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy oracle script");
    let output = child.wait_with_output().expect("wait for scipy oracle");
    if !output.status.success() {
        eprintln!(
            "scipy pinv oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy timing");
    let seconds: f64 = stdout.trim().parse().expect("parse scipy timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn scipy_pinv_available() -> bool {
    let mut child = match Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return false,
    };
    let Some(stdin) = child.stdin.as_mut() else {
        return false;
    };
    if stdin.write_all(b"import scipy.linalg\n").is_err() {
        return false;
    }
    child.wait().is_ok_and(|status| status.success())
}

fn scipy_lstsq_duration(rows: usize, cols: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

rows = int(sys.argv[1])
cols = int(sys.argv[2])
iters = int(sys.argv[3])
i = np.arange(rows, dtype=np.float64)[:, None]
j = np.arange(cols, dtype=np.float64)[None, :]
a = 1.0 / (np.abs(i - j) + 1.0)
d = np.arange(rows)
a[d, d] = rows * 2.0
b = np.arange(1, rows + 1, dtype=np.float64)
la.lstsq(a, b, check_finite=False)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    x, residuals, rank, s = la.lstsq(a, b, check_finite=False)
    checksum += float(x[0]) + float(rank)
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args([
            "-",
            &rows.to_string(),
            &cols.to_string(),
            &iters.to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy lstsq oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy lstsq oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy lstsq oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy lstsq oracle");
    if !output.status.success() {
        eprintln!(
            "scipy lstsq oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy lstsq timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy lstsq timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn scipy_eigh_subset_duration(n: usize, k: usize, rank: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

n = int(sys.argv[1])
k = int(sys.argv[2])
rank = int(sys.argv[3])
iters = int(sys.argv[4])
i = np.arange(n, dtype=np.float64)
factors = []
for r in range(rank):
    x = float(r + 3) * (i + 5.0)
    factors.append(np.sin(x) * 0.5 + np.cos(x) * 0.25)
a = np.zeros((n, n), dtype=np.float64)
for r, f in enumerate(factors):
    weight = float(rank - r)
    a += weight * np.outer(f, f)
lo = max(0, n - k)
la.eigh(a, subset_by_index=(lo, n - 1), check_finite=False)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    w, v = la.eigh(a, subset_by_index=(lo, n - 1), check_finite=False)
    checksum += float(w[-1]) + float(v[0, -1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args([
            "-",
            &n.to_string(),
            &k.to_string(),
            &rank.to_string(),
            &iters.to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy eigh oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy eigh oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy eigh oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy eigh oracle");
    if !output.status.success() {
        eprintln!(
            "scipy eigh oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy eigh timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy eigh timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn scipy_dft_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

n = int(sys.argv[1])
iters = int(sys.argv[2])
la.dft(n)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    m = la.dft(n)
    checksum += float(np.real(m[1, 1])) + float(np.imag(m[1, 1]))
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy dft oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy dft oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy dft oracle script");
    let output = child.wait_with_output().expect("wait for scipy dft oracle");
    if !output.status.success() {
        eprintln!(
            "scipy dft oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy dft timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy dft timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn scipy_cho_factor_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

n = int(sys.argv[1])
iters = int(sys.argv[2])
i = np.arange(n, dtype=np.float64)
a = 1.0 / (np.abs(i[:, None] - i[None, :]) + 1.0)
a[np.arange(n), np.arange(n)] = n * 3.0 + i * 0.01
la.cho_factor(a, lower=True, check_finite=False)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    c, lower = la.cho_factor(a, lower=True, check_finite=False)
    checksum += float(c[0, 0]) + float(c[-1, -1]) + float(lower)
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy cho_factor oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy cho_factor oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy cho_factor oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy cho_factor oracle");
    if !output.status.success() {
        eprintln!(
            "scipy cho_factor oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy cho_factor timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy cho_factor timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn scipy_cho_factor_solve_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

n = int(sys.argv[1])
iters = int(sys.argv[2])
i = np.arange(n, dtype=np.float64)
a = 1.0 / (np.abs(i[:, None] - i[None, :]) + 1.0)
a[np.arange(n), np.arange(n)] = n * 3.0 + i * 0.01
b = np.arange(1, n + 1, dtype=np.float64)
factor = la.cho_factor(a, lower=True, check_finite=False)
la.cho_solve(factor, b, check_finite=False)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    factor = la.cho_factor(a, lower=True, check_finite=False)
    x = la.cho_solve(factor, b, check_finite=False)
    checksum += float(x[0]) + float(x[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy cho_factor+cho_solve oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy cho_factor+cho_solve oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy cho_factor+cho_solve oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy cho_factor+cho_solve oracle");
    if !output.status.success() {
        eprintln!(
            "scipy cho_factor+cho_solve oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy cho_factor+cho_solve timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy cho_factor+cho_solve timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn bench_u0ucw_gauntlet_scipy_pinv(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;

    const ROWS: usize = 500;
    const COLS: usize = 1000;
    let a = make_underdetermined(ROWS, COLS);
    let mut group = c.benchmark_group("u0ucw_gauntlet_scipy_pinv");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("500x1000_rust_current_diag_gate", |bencher| {
        fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
        fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
        bencher.iter(|| black_box(pinv(black_box(&a), PinvOptions::default()).unwrap()));
    });
    group.bench_function("500x1000_rust_eigen_rcond_gate", |bencher| {
        fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
        fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(true, Relaxed);
        bencher.iter(|| black_box(pinv(black_box(&a), PinvOptions::default()).unwrap()));
        fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
    });
    group.bench_function("500x1000_rust_svd_fallback", |bencher| {
        fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(true, Relaxed);
        fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
        bencher.iter(|| black_box(pinv(black_box(&a), PinvOptions::default()).unwrap()));
        fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
    });
    if scipy_pinv_available() {
        group.bench_function("500x1000_scipy_pinv", |bencher| {
            bencher.iter_custom(|iters| {
                scipy_pinv_duration(ROWS, COLS, iters)
                    .expect("scipy pinv oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping 500x1000_scipy_pinv: python3 cannot import scipy.linalg");
    }
    group.finish();
}

fn bench_u0ucw_gauntlet_scipy_lstsq(c: &mut Criterion) {
    const ROWS: usize = 500;
    const COLS: usize = 1000;
    let a = make_underdetermined(ROWS, COLS);
    let b = make_rhs(ROWS);
    let mut group = c.benchmark_group("u0ucw_gauntlet_scipy_lstsq");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("500x1000_rust_current_materialized_transpose", |bencher| {
        bencher.iter(|| {
            black_box(lstsq(black_box(&a), black_box(&b), LstsqOptions::default())).unwrap()
        });
    });
    if scipy_pinv_available() {
        group.bench_function("500x1000_scipy_lstsq", |bencher| {
            bencher.iter_custom(|iters| {
                scipy_lstsq_duration(ROWS, COLS, iters)
                    .expect("scipy lstsq oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping 500x1000_scipy_lstsq: python3 cannot import scipy.linalg");
    }
    group.finish();
}

fn bench_randomized_eigh_gauntlet_scipy(c: &mut Criterion) {
    let mut group = c.benchmark_group("randomized_eigh_gauntlet_scipy");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    for &(n, k) in RANDOMIZED_EIGH_CASES {
        let rank = k + 8;
        let a = make_low_rank_symmetric_eigh_matrix(n, rank);
        group.bench_function(format!("{n}x{n}_k{k}_rust_randomized"), |bencher| {
            bencher.iter(|| {
                black_box(
                    randomized_eigh(
                        black_box(&a),
                        black_box(k),
                        black_box(8),
                        black_box(2),
                        black_box(0x5eed_1234),
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_function(format!("{n}x{n}_k{k}_rust_full_eigh"), |bencher| {
            bencher.iter(|| black_box(eigh(black_box(&a), DecompOptions::default()).unwrap()));
        });
        if scipy_pinv_available() {
            group.bench_function(format!("{n}x{n}_k{k}_scipy_subset_eigh"), |bencher| {
                bencher.iter_custom(|iters| {
                    scipy_eigh_subset_duration(n, k, rank, iters)
                        .expect("scipy eigh oracle should run after availability check")
                });
            });
        } else {
            eprintln!(
                "skipping {n}x{n}_k{k}_scipy_subset_eigh: python3 cannot import scipy.linalg"
            );
        }
    }
    group.finish();
}

fn bench_dft_gauntlet_scipy(c: &mut Criterion) {
    let mut group = c.benchmark_group("dft_gauntlet_scipy");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for &n in DFT_GAUNTLET_SIZES {
        group.bench_function(format!("{n}_rust_dft"), |bencher| {
            bencher.iter(|| black_box(dft(black_box(n), black_box(Option::<&str>::None)).unwrap()));
        });
        if scipy_pinv_available() {
            group.bench_function(format!("{n}_scipy_dft"), |bencher| {
                bencher.iter_custom(|iters| {
                    scipy_dft_duration(n, iters)
                        .expect("scipy dft oracle should run after availability check")
                });
            });
        } else {
            eprintln!("skipping {n}_scipy_dft: python3 cannot import scipy.linalg");
        }
    }
    group.finish();
}

fn bench_cho_factor_gauntlet_scipy(c: &mut Criterion) {
    let mut group = c.benchmark_group("cho_factor_gauntlet_scipy");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for &n in CHO_FACTOR_GAUNTLET_SIZES {
        let a = make_symmetric_eigh_matrix(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}_rust_cho_factor"), |bencher| {
            bencher
                .iter(|| black_box(cho_factor(black_box(&a), DecompOptions::default()).unwrap()));
        });
        group.bench_function(format!("{n}x{n}_rust_cho_factor_solve"), |bencher| {
            bencher.iter(|| {
                let factor = cho_factor(black_box(&a), DecompOptions::default()).unwrap();
                black_box(cho_solve(black_box(&factor), black_box(&b)).unwrap())
            });
        });
        if scipy_pinv_available() {
            group.bench_function(format!("{n}x{n}_scipy_cho_factor"), |bencher| {
                bencher.iter_custom(|iters| {
                    scipy_cho_factor_duration(n, iters)
                        .expect("scipy cho_factor oracle should run after availability check")
                });
            });
            group.bench_function(format!("{n}x{n}_scipy_cho_factor_solve"), |bencher| {
                bencher.iter_custom(|iters| {
                    scipy_cho_factor_solve_duration(n, iters).expect(
                        "scipy cho_factor+cho_solve oracle should run after availability check",
                    )
                });
            });
        } else {
            eprintln!("skipping {n}x{n}_scipy_cho_factor: python3 cannot import scipy.linalg");
        }
    }
    group.finish();
}

fn bench_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cols = a.first().map_or(0, Vec::len);
    (0..cols)
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect()
}

fn orthogonal_procrustes_original_bench(a: &[Vec<f64>], b: &[Vec<f64>]) -> (Vec<Vec<f64>>, f64) {
    let at = bench_transpose(a);
    let m_mat = matmul(&at, b).unwrap();
    let svd_result = svd(&m_mat, DecompOptions::default()).unwrap();
    let r = matmul(&svd_result.u, &svd_result.vt).unwrap();
    let scale = svd_result.s.iter().sum();
    (r, scale)
}

fn bench_orthogonal_procrustes_gauntlet(c: &mut Criterion) {
    let mut group = c.benchmark_group("orthogonal_procrustes_gauntlet");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for &(rows, cols) in ORTHOGONAL_PROCRUSTES_CASES {
        let a = make_matmul_matrix(rows, cols, 0x33aa);
        let b = make_matmul_matrix(rows, cols, 0x5eed);
        group.bench_function(format!("{rows}x{cols}_orig"), |bencher| {
            bencher.iter(|| {
                black_box(orthogonal_procrustes_original_bench(
                    black_box(&a),
                    black_box(&b),
                ))
            });
        });
        group.bench_function(format!("{rows}x{cols}_candidate"), |bencher| {
            bencher
                .iter(|| black_box(orthogonal_procrustes(black_box(&a), black_box(&b)).unwrap()));
        });
    }
    group.finish();
}

fn bench_baseline_pinv(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut group = c.benchmark_group("baseline_pinv");
    group.sample_size(100);
    for &n in BASELINE_SIZES {
        let rows = n * 2;
        let a = make_overdetermined(rows, n);
        group.bench_function(format!("{rows}x{n}"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
        });
        group.bench_function(format!("{rows}x{n}_trsm_serial"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{n}_materialized_rhs"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(true, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{n}_dmatrix_solve"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{n}_dmatrix_cert"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{n}_materialized_cert"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{n}_chol_solve"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
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
    bench_pinv,
    bench_matmul,
    bench_eigh_dense,
    bench_randomized_eigh
);

criterion_group!(
    baseline_benches,
    bench_baseline_solve,
    bench_baseline_solve_pos,
    bench_baseline_inv,
    bench_baseline_lstsq,
    bench_u0ucw_wide_lstsq,
    bench_u0ucw_wide_pinv,
    bench_u0ucw_gauntlet_scipy_pinv,
    bench_u0ucw_gauntlet_scipy_lstsq,
    bench_randomized_eigh_gauntlet_scipy,
    bench_dft_gauntlet_scipy,
    bench_cho_factor_gauntlet_scipy,
    bench_lu_factor_gauntlet,
    bench_det_gauntlet,
    bench_orthogonal_procrustes_gauntlet,
    bench_lu_factor_solve_gauntlet,
    bench_baseline_pinv
);

criterion_main!(benches, baseline_benches);
