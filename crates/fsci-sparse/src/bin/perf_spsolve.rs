//! Same-process A/B + parity harness for spsolve routing.
//!
//! Before: spsolve densified any sparse A (n<=32768) into an n×n dense matrix and
//! ran O(n³) nalgebra dense LU. After: genuinely-sparse A routes to the native
//! sparse LU (~O(n·fill)). On a diagonally-dominant pentadiagonal system the fill
//! is O(n), so the sparse path is orders of magnitude cheaper. The solution is
//! unique, so x matches the dense path to rounding (PARITY block prints max|Δx|).
//! Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_spsolve`.

use std::hint::black_box;
use std::time::Instant;

use fsci_sparse::{
    CooMatrix, CscMatrix, CsrMatrix, FormatConvertible, LuOptions, Shape2D, SolveOptions, splu,
    splu_solve, spsolve,
};
use nalgebra::{DMatrix, DVector};

// Diagonally-dominant pentadiagonal A (bandwidth 2): diag 6, off-diagonals -1 at
// ±1, ±2. nnz/row ~= 5, so a.nnz() <= 16n -> routes to the native sparse LU.
fn pentadiagonal(n: usize) -> CsrMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        for off in [-2i64, -1, 0, 1, 2] {
            let j = i as i64 + off;
            if j >= 0 && (j as usize) < n {
                rows.push(i);
                cols.push(j as usize);
                data.push(if off == 0 { 6.0 } else { -1.0 });
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Verbatim of the OLD dense path: densify the CSR and solve with nalgebra LU.
fn dense_solve_baseline(a: &CsrMatrix, b: &[f64]) -> Vec<f64> {
    let n = a.shape().rows;
    let mut dense = vec![0.0f64; n * n];
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    for i in 0..n {
        for idx in indptr[i]..indptr[i + 1] {
            dense[i * n + indices[idx]] = data[idx];
        }
    }
    let matrix = DMatrix::from_row_slice(n, n, &dense);
    let rhs = DVector::from_column_slice(b);
    let x = matrix.lu().solve(&rhs).expect("dense lu");
    x.iter().copied().collect()
}

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn main() {
    println!("===PARITY+AB===");
    for &n in &[512usize, 1024, 2048] {
        let a = pentadiagonal(n);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();

        // correctness: sparse-routed result vs old dense result
        let x_sparse = spsolve(&a, &b, SolveOptions::default()).expect("spsolve").solution;
        let x_dense = dense_solve_baseline(&a, &b);
        let max_dx = x_sparse
            .iter()
            .zip(x_dense.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps_sparse = (50_000_000 / (n + 1)).clamp(20, 5000);
        let t_after = time(reps_sparse, || {
            black_box(spsolve(black_box(&a), black_box(&b), SolveOptions::default()).unwrap());
        });
        let reps_dense = if n >= 2048 { 1 } else { 3 };
        let t_before = time(reps_dense, || {
            black_box(dense_solve_baseline(black_box(&a), black_box(&b)));
        });

        println!(
            "spsolve n={n:>5}: dense={t_before:>10.4}ms  sparse={t_after:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            t_before / t_after
        );

        // splu factorization: same routing. Time factorize-only (the dominant cost).
        let a_csc: CscMatrix = a.to_csc().unwrap();
        let fac = splu(&a_csc, LuOptions::default()).expect("splu");
        let x_splu = splu_solve(&fac, &b).expect("splu_solve");
        let max_dx2 = x_splu
            .iter()
            .zip(x_dense.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);
        let t_splu = time(reps_sparse, || {
            black_box(splu(black_box(&a_csc), LuOptions::default()).unwrap());
        });
        let t_dense_fac = time(reps_dense, || {
            let n = a.shape().rows;
            let mut dense = vec![0.0f64; n * n];
            let indptr = a.indptr();
            let indices = a.indices();
            let data = a.data();
            for i in 0..n {
                for idx in indptr[i]..indptr[i + 1] {
                    dense[i * n + indices[idx]] = data[idx];
                }
            }
            black_box(DMatrix::from_row_slice(n, n, &dense).lu());
        });
        println!(
            "splu    n={n:>5}: dense={t_dense_fac:>10.4}ms  sparse={t_splu:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx2:.2e}",
            t_dense_fac / t_splu
        );
    }
}
