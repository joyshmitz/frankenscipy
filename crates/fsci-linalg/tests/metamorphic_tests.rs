//! Metamorphic tests for `fsci-linalg`.
//!
//! Each test asserts a relation between inputs and outputs that holds for
//! the analytical operation, regardless of the specific numerical values.
//! This is oracle-free: instead of "for input X expect output Y", the
//! relations check forward-residuals, factor reconstructions, and
//! algebraic identities like det(AB) = det(A) det(B).
//!
//! Run with: `cargo test -p fsci-linalg --test metamorphic_tests`

use fsci_linalg::{
    DecompOptions, InvOptions, NormKind, SolveOptions, cholesky, det, inv, lu, norm, qr, solve,
    svd,
};

const RTOL: f64 = 1e-9;
const ATOL: f64 = 1e-12;

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= ATOL + RTOL * expected.abs().max(1.0)
}

fn assert_close(actual: f64, expected: f64, msg: &str) {
    assert!(
        close(actual, expected),
        "{msg}: actual={actual:.16e} expected={expected:.16e} diff={:.3e}",
        (actual - expected).abs()
    );
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let inner = a[0].len();
    let n = b[0].len();
    assert_eq!(b.len(), inner, "matmul shape mismatch");
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for k in 0..inner {
            let aik = a[i][k];
            for j in 0..n {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = a[0].len();
    let mut t = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn frobenius_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut s = 0.0;
    for (ra, rb) in a.iter().zip(b) {
        for (x, y) in ra.iter().zip(rb) {
            let d = x - y;
            s += d * d;
        }
    }
    s.sqrt()
}

fn frobenius(a: &[Vec<f64>]) -> f64 {
    let mut s = 0.0;
    for row in a {
        for x in row {
            s += x * x;
        }
    }
    s.sqrt()
}

fn vec_l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut id = vec![vec![0.0; n]; n];
    for i in 0..n {
        id[i][i] = 1.0;
    }
    id
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — solve residual: ||A x − b|| / ||b|| ≪ 1
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_solve_residual_small_dense() {
    let a = vec![
        vec![4.0, -1.0, 0.0, 0.0],
        vec![-1.0, 4.0, -1.0, 0.0],
        vec![0.0, -1.0, 4.0, -1.0],
        vec![0.0, 0.0, -1.0, 4.0],
    ];
    let b = vec![1.0, 0.0, 0.0, 1.0];
    let result = solve(&a, &b, SolveOptions::default()).unwrap();
    let r = matvec(&a, &result.x);
    let resid: f64 = r
        .iter()
        .zip(&b)
        .map(|(rv, bv)| (rv - bv).powi(2))
        .sum::<f64>()
        .sqrt();
    let bnorm = vec_l2(&b);
    assert!(
        resid <= 1e-10 * (1.0 + bnorm),
        "MR1 dense residual too large: ||Ax-b||={resid:e}, ||b||={bnorm:e}"
    );
}

#[test]
fn mr_solve_residual_ill_conditioned() {
    // Hilbert-like matrix is famously ill-conditioned but well-defined.
    let n = 5;
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = 1.0 / (i as f64 + j as f64 + 1.0);
        }
    }
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let result = solve(&a, &b, SolveOptions::default()).unwrap();
    let r = matvec(&a, &result.x);
    let resid: f64 = r
        .iter()
        .zip(&b)
        .map(|(rv, bv)| (rv - bv).powi(2))
        .sum::<f64>()
        .sqrt();
    // For the 5x5 Hilbert matrix, condition number is ≈ 4.8e5; relative
    // residual should still be much smaller than that times machine eps.
    assert!(
        resid <= 1e-7 * vec_l2(&b),
        "MR1 ill-cond residual={resid:e} too large"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — LU reconstruction: P · L · U = A
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_lu_reconstructs_matrix() {
    let a = vec![
        vec![4.0, 3.0, 2.0],
        vec![6.0, 3.0, 1.0],
        vec![2.0, 5.0, 4.0],
    ];
    let result = lu(&a, DecompOptions::default()).unwrap();
    let lu_product = matmul(&result.l, &result.u);
    let plu = matmul(&result.p, &lu_product);
    let diff = frobenius_diff(&plu, &a);
    let scale = frobenius(&a).max(1.0);
    assert!(
        diff <= 1e-10 * scale,
        "MR2 LU reconstruct: ||PLU - A||_F = {diff:e}, scale={scale:e}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — QR reconstruction: Q · R = A and Q^T Q = I (orthogonality)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_qr_reconstruct_and_orthogonal() {
    let a = vec![
        vec![1.0, 2.0, 0.0],
        vec![3.0, 4.0, 1.0],
        vec![0.0, -1.0, 2.0],
        vec![5.0, 1.0, -1.0],
    ];
    let result = qr(&a, DecompOptions::default()).unwrap();
    let qr_product = matmul(&result.q, &result.r);
    let scale = frobenius(&a).max(1.0);
    assert!(
        frobenius_diff(&qr_product, &a) <= 1e-10 * scale,
        "MR3 QR reconstruct: ||QR - A||_F too large"
    );

    // Q^T Q = I (m x m identity for full Q)
    let q_t = transpose(&result.q);
    let q_t_q = matmul(&q_t, &result.q);
    let id = identity(q_t_q.len());
    assert!(
        frobenius_diff(&q_t_q, &id) <= 1e-10,
        "MR3 Q is not orthogonal: ||Q^T Q - I||_F too large"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — Cholesky reconstruction: L · L^T = A for SPD A
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cholesky_reconstructs_spd() {
    // Build A = M^T M + I to guarantee strict positive definiteness.
    let m = vec![
        vec![1.0, 2.0, 0.0],
        vec![3.0, 1.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let m_t = transpose(&m);
    let mtm = matmul(&m_t, &m);
    let n = mtm.len();
    let mut a = mtm;
    for i in 0..n {
        a[i][i] += 1.0;
    }

    let result = cholesky(&a, true, DecompOptions::default()).unwrap();
    let l = result.factor;
    let l_t = transpose(&l);
    let llt = matmul(&l, &l_t);
    let scale = frobenius(&a).max(1.0);
    assert!(
        frobenius_diff(&llt, &a) <= 1e-10 * scale,
        "MR4 Cholesky reconstruct: ||LL^T - A||_F too large"
    );

    // L should be lower triangular: L[i][j] = 0 for j > i.
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(
                l[i][j].abs() < 1e-14,
                "L is not lower triangular at ({i},{j}): {}",
                l[i][j]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — determinant multiplicativity: det(A B) = det(A) det(B)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_det_multiplicativity() {
    let a = vec![
        vec![2.0, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 4.0],
    ];
    let b = vec![
        vec![1.0, 0.5, 0.0],
        vec![0.0, 2.0, 1.0],
        vec![1.0, 1.0, 3.0],
    ];
    use fsci_runtime::RuntimeMode;

    let det_a = det(&a, RuntimeMode::Strict, true).unwrap();
    let det_b = det(&b, RuntimeMode::Strict, true).unwrap();
    let ab = matmul(&a, &b);
    let det_ab = det(&ab, RuntimeMode::Strict, true).unwrap();

    assert_close(det_ab, det_a * det_b, "MR5 det(AB) = det(A) det(B)");
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — inverse identity: A · A^{-1} = I (and A^{-1} · A = I)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_inverse_identity() {
    let a = vec![
        vec![4.0, 3.0, 2.0],
        vec![6.0, 3.0, 1.0],
        vec![2.0, 5.0, 4.0],
    ];
    let result = inv(&a, InvOptions::default()).unwrap();
    let prod = matmul(&a, &result.inverse);
    let id = identity(a.len());
    let diff = frobenius_diff(&prod, &id);
    assert!(diff <= 1e-10, "MR6 A * A^-1 != I: ||AA^-1 - I||_F = {diff:e}");

    let prod2 = matmul(&result.inverse, &a);
    let diff2 = frobenius_diff(&prod2, &id);
    assert!(
        diff2 <= 1e-10,
        "MR6 A^-1 * A != I: ||A^-1 A - I||_F = {diff2:e}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — SVD reconstruction: U · diag(s) · V^T = A
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_svd_reconstruct() {
    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ];
    let result = svd(&a, DecompOptions::default()).unwrap();
    let k = result.s.len();

    // U is m × k; build U · diag(s).
    let m = result.u.len();
    let mut us = vec![vec![0.0; k]; m];
    for i in 0..m {
        for j in 0..k {
            us[i][j] = result.u[i][j] * result.s[j];
        }
    }
    let recon = matmul(&us, &result.vt);
    let scale = frobenius(&a).max(1.0);
    assert!(
        frobenius_diff(&recon, &a) <= 1e-10 * scale,
        "MR7 SVD reconstruct: ||U Σ V^T - A||_F too large"
    );

    // Singular values must be non-negative and non-increasing.
    for i in 0..k {
        assert!(result.s[i] >= 0.0, "negative singular value at {i}");
    }
    for i in 1..k {
        assert!(
            result.s[i] <= result.s[i - 1] + 1e-12,
            "singular values not in descending order at {i}: {} > {}",
            result.s[i],
            result.s[i - 1]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — solve linearity: A·(α x + β y) = α b1 + β b2 where Ax = b1, Ay = b2
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_solve_linearity() {
    let a = vec![
        vec![3.0, 1.0, 0.0],
        vec![1.0, 4.0, 1.0],
        vec![0.0, 1.0, 5.0],
    ];
    let b1 = vec![1.0, 2.0, 3.0];
    let b2 = vec![-1.0, 0.5, 1.5];
    let alpha = 2.5;
    let beta = -0.7;

    let x1 = solve(&a, &b1, SolveOptions::default()).unwrap().x;
    let x2 = solve(&a, &b2, SolveOptions::default()).unwrap().x;

    let combined_b: Vec<f64> = b1
        .iter()
        .zip(&b2)
        .map(|(p, q)| alpha * p + beta * q)
        .collect();
    let x_combined = solve(&a, &combined_b, SolveOptions::default()).unwrap().x;

    let expected: Vec<f64> = x1
        .iter()
        .zip(&x2)
        .map(|(p, q)| alpha * p + beta * q)
        .collect();

    for (i, (got, want)) in x_combined.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= 1e-10 * (1.0 + want.abs()),
            "MR8 linearity at i={i}: got={got} expected={want}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — Frobenius norm equals sum of singular values squared (root of)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_frobenius_norm_matches_singular_values() {
    let a = vec![
        vec![1.5, -2.0, 0.5],
        vec![0.7, 1.0, 2.5],
        vec![-1.0, 0.0, 3.0],
        vec![2.0, -0.5, 1.5],
    ];
    let svd_result = svd(&a, DecompOptions::default()).unwrap();
    let from_svd: f64 = svd_result.s.iter().map(|s| s * s).sum::<f64>().sqrt();
    let from_norm = norm(&a, NormKind::Fro, DecompOptions::default()).unwrap();
    let direct = frobenius(&a);

    assert_close(from_norm, direct, "MR9 norm Frobenius matches direct");
    assert_close(from_svd, direct, "MR9 ||A||_F = sqrt(sum sigma_i^2)");
}
