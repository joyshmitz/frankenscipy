//! Reproduction for frankenscipy-me4pf: `eigh_tridiagonal` returns CORRECT
//! eigenvalues but INCORRECT eigenvectors when the spectrum has ± pairs
//! (e.g. a zero-diagonal symmetric tridiagonal — the Gauss-Legendre/Hermite
//! Golub-Welsch Jacobi matrix). Generic matrices are fine. The conformance test
//! `diff_linalg_eigh_tridiagonal` only exercises `eigvals_only=True`, so the
//! eigenvector path is untested and this defect was masked.
//!
//! Run: `cargo run -p fsci-linalg --bin check_eigh_tridiag_vectors`.
use fsci_linalg::{DecompOptions, eigh_tridiagonal};

// Worst per-column residual ||T v_k - lambda_k v_k||_inf over the returned basis.
fn worst_residual(d: &[f64], e: &[f64]) -> f64 {
    let n = d.len();
    let (vals, vecs) = eigh_tridiagonal(d, e, false, DecompOptions::default()).unwrap();
    let vecs = vecs.unwrap();
    let mut worst = 0.0f64;
    for k in 0..n {
        let v: Vec<f64> = (0..n).map(|r| vecs[r][k]).collect();
        for i in 0..n {
            let mut tv = d[i] * v[i];
            if i > 0 {
                tv += e[i - 1] * v[i - 1];
            }
            if i + 1 < n {
                tv += e[i] * v[i + 1];
            }
            worst = worst.max((tv - vals[k] * v[i]).abs());
        }
    }
    worst
}

fn main() {
    let generic = worst_residual(&[2.0, 3.0, 4.0, 5.0, 6.0], &[1.0, 1.0, 1.0, 1.0]);
    let zero_diag = worst_residual(&[0.0; 6], &[0.5, 0.6, 0.7, 0.8, 0.9]);
    println!("generic-spectrum   worst eigenvector residual = {generic:.3e} (expect ~1e-13)");
    println!("zero-diag (±pairs) worst eigenvector residual = {zero_diag:.3e} (expect ~1e-13)");
    println!(
        "STATUS: {}",
        if zero_diag < 1e-9 {
            "OK (frankenscipy-me4pf fixed)"
        } else {
            "BUG PRESENT (frankenscipy-me4pf) — eigenvectors wrong for ±-pair spectra"
        }
    );
}
