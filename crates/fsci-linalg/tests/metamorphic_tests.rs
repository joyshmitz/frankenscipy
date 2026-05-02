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
    DecompOptions, InvOptions, LstsqOptions, NormKind, PinvOptions, SolveOptions, cholesky, det,
    diag, diagm, eigh, eigvalsh, expm, hadamard_product, hessenberg, inv, logm, lstsq, lu, norm,
    outer, pinv, qr, schur, solve, sqrtm, svd, trace, vdot, vnorm,
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

// ─────────────────────────────────────────────────────────────────────
// MR10 — expm(0) = I (matrix exponential of the zero matrix is identity)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_expm_of_zero_is_identity() {
    let n = 4;
    let zero = vec![vec![0.0; n]; n];
    let e = expm(&zero, DecompOptions::default()).unwrap();
    let id = identity(n);
    let diff = frobenius_diff(&e, &id);
    assert!(diff <= 1e-10, "MR10 expm(0) != I: ||expm(0) - I||_F = {diff:e}");
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — sqrtm(A)² = A for SPD A.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sqrtm_squared_equals_input_spd() {
    // SPD: A = M^T M + I.
    let m = vec![
        vec![1.0, 2.0, 0.0],
        vec![3.0, 1.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let mt = transpose(&m);
    let mtm = matmul(&mt, &m);
    let n = mtm.len();
    let mut a = mtm;
    for i in 0..n {
        a[i][i] += 1.0;
    }

    let s = sqrtm(&a, DecompOptions::default()).unwrap();
    let s2 = matmul(&s, &s);
    let scale = frobenius(&a).max(1.0);
    assert!(
        frobenius_diff(&s2, &a) <= 1e-8 * scale,
        "MR11 sqrtm(A)² = A: ||S² - A||_F too large"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — logm(expm(A)) = A for symmetric A.
//
// The symmetric path uses spectral decomposition for both expm and
// logm (V·diag(f(λ))·V^T), so the round-trip is well-conditioned and
// accurate to spectral-decomposition round-off. For non-symmetric A,
// the Schur-based general path has substantially weaker precision —
// see project bead `frankenscipy-cgzb`.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_logm_inverts_expm_symmetric() {
    let a = vec![
        vec![0.30, 0.05, -0.02],
        vec![0.05, 0.20, 0.07],
        vec![-0.02, 0.07, 0.40],
    ];
    let exp_a = expm(&a, DecompOptions::default()).unwrap();
    let log_exp_a = logm(&exp_a, DecompOptions::default()).unwrap();
    let diff = frobenius_diff(&log_exp_a, &a);
    let scale = frobenius(&a).max(1.0);
    assert!(
        diff <= 1e-9 * scale,
        "MR12 logm(expm(A)) != A (symmetric): ||logm(expm(A)) - A||_F = {diff:e}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — Schur decomposition: A = Z·T·Z^T with Z orthogonal,
//        T upper quasi-triangular (real Schur form).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_schur_reconstructs_matrix() {
    let a = vec![
        vec![4.0, 1.0, 2.0, 0.5],
        vec![0.0, 3.0, 1.0, 0.7],
        vec![1.0, -1.0, 5.0, 0.2],
        vec![0.5, 0.0, 2.0, 6.0],
    ];
    let result = schur(&a, DecompOptions::default()).unwrap();

    // Reconstruction: Z · T · Z^T = A.
    let zt = matmul(&result.z, &result.t);
    let z_t_t = transpose(&result.z);
    let recon = matmul(&zt, &z_t_t);
    let scale = frobenius(&a).max(1.0);
    assert!(
        frobenius_diff(&recon, &a) <= 1e-10 * scale,
        "MR13 Schur reconstruct: ||Z T Z^T - A||_F too large"
    );

    // Z orthogonality: Z^T Z = I.
    let zt_z = matmul(&z_t_t, &result.z);
    let id = identity(zt_z.len());
    assert!(
        frobenius_diff(&zt_z, &id) <= 1e-10,
        "MR13 Schur Z not orthogonal"
    );

    // T is upper quasi-triangular: only the immediate sub-diagonal can
    // be non-zero (for 2x2 complex-eigenvalue blocks).
    let n = result.t.len();
    for i in 2..n {
        for j in 0..(i - 1) {
            assert!(
                result.t[i][j].abs() < 1e-10,
                "MR13 Schur T not quasi-triangular at ({i},{j}): {}",
                result.t[i][j]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — Hessenberg decomposition: A = Q·H·Q^T with Q orthogonal and
//        H upper-Hessenberg (only the first sub-diagonal is non-zero).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hessenberg_reconstructs_matrix() {
    let a = vec![
        vec![4.0, 1.0, 2.0, 0.5, 1.1],
        vec![0.0, 3.0, 1.0, 0.7, -0.3],
        vec![1.0, -1.0, 5.0, 0.2, 0.6],
        vec![0.5, 0.0, 2.0, 6.0, 1.0],
        vec![0.7, 0.4, -0.2, 1.5, 2.0],
    ];
    let result = hessenberg(&a, DecompOptions::default()).unwrap();

    // Reconstruction: Q · H · Q^T = A.
    let qh = matmul(&result.q, &result.h);
    let q_t = transpose(&result.q);
    let recon = matmul(&qh, &q_t);
    let scale = frobenius(&a).max(1.0);
    assert!(
        frobenius_diff(&recon, &a) <= 1e-10 * scale,
        "MR14 Hessenberg reconstruct: ||Q H Q^T - A||_F too large"
    );

    // Q orthogonality: Q^T Q = I.
    let qt_q = matmul(&q_t, &result.q);
    let id = identity(qt_q.len());
    assert!(
        frobenius_diff(&qt_q, &id) <= 1e-10,
        "MR14 Hessenberg Q not orthogonal"
    );

    // H is upper-Hessenberg: H[i][j] = 0 for i > j + 1.
    let n = result.h.len();
    for i in 0..n {
        for j in 0..n {
            if i > j + 1 {
                assert!(
                    result.h[i][j].abs() < 1e-10,
                    "MR14 H not upper-Hessenberg at ({i},{j}): {}",
                    result.h[i][j]
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — eigh on a symmetric matrix returns (λ_i, v_i) with the
// eigenequation A v = λ v satisfied to f64-eps tolerance.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_eigh_satisfies_eigenequation() {
    let a = vec![
        vec![4.0_f64, 1.0, 0.5, 0.0],
        vec![1.0, 3.0, 0.7, 0.2],
        vec![0.5, 0.7, 5.0, 0.1],
        vec![0.0, 0.2, 0.1, 6.0],
    ];
    let result = eigh(&a, DecompOptions::default()).unwrap();
    let n = result.eigenvalues.len();
    assert_eq!(n, a.len());
    for (k, lambda) in result.eigenvalues.iter().enumerate() {
        // eigenvectors are stored as columns: v_k = result.eigenvectors[i][k] for i in 0..n
        let v: Vec<f64> = (0..n).map(|i| result.eigenvectors[i][k]).collect();
        let mut av = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += a[i][j] * v[j];
            }
        }
        let mut residual_sq = 0.0_f64;
        for i in 0..n {
            let r = av[i] - lambda * v[i];
            residual_sq += r * r;
        }
        assert!(
            residual_sq.sqrt() < 1e-10,
            "MR15 eigh eigenequation k={k}: ||A v - λ v|| = {}",
            residual_sq.sqrt()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — eigvalsh returns eigenvalues in ascending order.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_eigvalsh_ascending() {
    let a = vec![
        vec![4.0_f64, 1.0, 0.5, 0.0],
        vec![1.0, 3.0, 0.7, 0.2],
        vec![0.5, 0.7, 5.0, 0.1],
        vec![0.0, 0.2, 0.1, 6.0],
    ];
    let evals = eigvalsh(&a, DecompOptions::default()).unwrap();
    for w in evals.windows(2) {
        assert!(
            w[0] <= w[1] + 1e-12,
            "MR16 eigvalsh not ascending: {} > {}",
            w[0],
            w[1]
        );
    }
    // Sum of eigenvalues equals trace.
    let trace: f64 = (0..a.len()).map(|i| a[i][i]).sum();
    let sum: f64 = evals.iter().sum();
    assert!(
        (sum - trace).abs() < 1e-10,
        "MR16 sum of eigenvalues {} != trace {}",
        sum,
        trace
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR17 — Moore-Penrose condition 1: A · A⁺ · A = A.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pinv_satisfies_penrose_one() {
    // Tall, full column-rank matrix.
    let a = vec![
        vec![1.0_f64, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 0.5],
        vec![-1.0, 1.5],
    ];
    let pinv_a = pinv(&a, PinvOptions::default()).unwrap().pseudo_inverse;
    let aap = matmul(&a, &pinv_a);
    let aapa = matmul(&aap, &a);
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            assert!(
                close(aapa[i][j], a[i][j]),
                "MR17 (A·A⁺·A)[{i}, {j}] = {} vs A[{i}, {j}] = {}",
                aapa[i][j],
                a[i][j]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — lstsq returns the least-squares minimiser: AᵀA·x = Aᵀb
// (the normal equations). Holds when A has full column rank.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_lstsq_satisfies_normal_equations() {
    let a = vec![
        vec![1.0_f64, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 0.5],
        vec![-1.0, 1.5],
    ];
    let b = vec![1.0_f64, 2.0, 3.0, 4.0];
    let res = lstsq(&a, &b, LstsqOptions::default()).unwrap();
    // Build Aᵀ.
    let m = a.len();
    let n = a[0].len();
    let mut at = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            at[j][i] = a[i][j];
        }
    }
    let ata = matmul(&at, &a);
    let lhs = matvec(&ata, &res.x);
    let atb = matvec(&at, &b);
    for k in 0..n {
        assert!(
            (lhs[k] - atb[k]).abs() < 1e-7,
            "MR18 normal eq at {k}: AᵀA·x = {} vs Aᵀb = {}",
            lhs[k],
            atb[k]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — det(I_n) = 1 for any size n.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_det_of_identity_is_one() {
    use fsci_runtime::RuntimeMode;
    for n in [1usize, 2, 3, 5, 8] {
        let mut id = vec![vec![0.0; n]; n];
        for i in 0..n {
            id[i][i] = 1.0;
        }
        let d = det(&id, RuntimeMode::Strict, true).unwrap();
        assert!(
            (d - 1.0).abs() < 1e-12,
            "MR19 det(I_{n}) = {d}, expected 1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — det((Aᵀ)ᵀ) = det(A) — verified via determinant of A and Aᵀᵀ
// computed by transposing twice. (Numerical stability of two transposes.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_det_double_transpose() {
    use fsci_runtime::RuntimeMode;
    let a = vec![
        vec![2.0_f64, 0.5, -1.0],
        vec![1.0, 3.0, 0.25],
        vec![0.0, -0.7, 2.5],
    ];
    let n = a.len();
    let mut at = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            at[j][i] = a[i][j];
        }
    }
    let mut att = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            att[j][i] = at[i][j];
        }
    }
    let da = det(&a, RuntimeMode::Strict, true).unwrap();
    let datt = det(&att, RuntimeMode::Strict, true).unwrap();
    assert!(
        (da - datt).abs() < 1e-12,
        "MR20 det(A) = {da}, det((Aᵀ)ᵀ) = {datt}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — Frobenius norm of A equals Frobenius norm of Aᵀ.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_frobenius_norm_transpose_invariant() {
    let a = vec![
        vec![1.0_f64, -2.0, 3.0],
        vec![0.5, 1.5, -0.25],
        vec![-1.0, 0.0, 2.0],
        vec![3.5, -0.5, 1.0],
    ];
    let m = a.len();
    let n = a[0].len();
    let mut at = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            at[j][i] = a[i][j];
        }
    }
    let na = norm(&a, NormKind::Fro, DecompOptions::default()).unwrap();
    let nat = norm(&at, NormKind::Fro, DecompOptions::default()).unwrap();
    assert!(
        (na - nat).abs() < 1e-12,
        "MR21 ‖A‖_F = {na}, ‖Aᵀ‖_F = {nat}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — solve(I, b) = b: identity matrix solve returns the rhs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_solve_with_identity_returns_rhs() {
    for n in [1usize, 3, 5, 7] {
        let mut id = vec![vec![0.0; n]; n];
        for i in 0..n {
            id[i][i] = 1.0;
        }
        let b: Vec<f64> = (0..n).map(|i| (i as f64) * 1.5 - 2.0).collect();
        let res = solve(&id, &b, SolveOptions::default()).unwrap();
        for i in 0..n {
            assert!(
                (res.x[i] - b[i]).abs() < 1e-12,
                "MR22 solve(I, b)[{i}] = {} vs b = {}",
                res.x[i],
                b[i]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — trace of I_n is n.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_trace_of_identity_is_n() {
    for n in [1usize, 3, 5, 8, 16] {
        let mut id = vec![vec![0.0; n]; n];
        for i in 0..n {
            id[i][i] = 1.0;
        }
        let t = trace(&id);
        assert!(
            (t - n as f64).abs() < 1e-12,
            "MR23 trace(I_{n}) = {t}, expected {n}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — diag(diagm(v)) = v: lifting a vector to a diagonal matrix and
// projecting back recovers the input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_diag_diagm_roundtrip() {
    let vs: &[Vec<f64>] = &[
        vec![1.0, 2.0, 3.0, 4.0],
        vec![-1.5, 0.0, 7.5, -2.5, 1.0],
        vec![100.0],
    ];
    for v in vs {
        let m = diagm(v);
        let v2 = diag(&m);
        assert_eq!(v.len(), v2.len(), "MR24 length mismatch");
        for (i, (&a, &b)) in v.iter().zip(&v2).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "MR24 diag∘diagm at {i}: {a} vs {b}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — hadamard_product with the zero matrix is zero.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hadamard_with_zero_is_zero() {
    let a = vec![
        vec![1.0_f64, -2.0, 3.0],
        vec![4.0, 5.0, -6.0],
        vec![-7.0, 8.0, 9.0],
    ];
    let zeros = vec![vec![0.0; 3]; 3];
    let h = hadamard_product(&a, &zeros);
    for row in &h {
        for &v in row {
            assert!(v.abs() < 1e-15, "MR25 hadamard with 0 entry = {v}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — outer(a, b) has shape (len(a), len(b)).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_outer_product_shape() {
    let a = vec![1.0_f64, 2.0, 3.0];
    let b = vec![4.0_f64, 5.0];
    let o = outer(&a, &b);
    assert_eq!(o.len(), a.len(), "MR26 outer rows");
    for row in &o {
        assert_eq!(row.len(), b.len(), "MR26 outer cols");
    }
    // Verify outer[i][j] = a[i] * b[j].
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            assert!(
                (o[i][j] - ai * bj).abs() < 1e-12,
                "MR26 outer[{i}, {j}] = {} vs {ai}*{bj}",
                o[i][j]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — vnorm(v) = sqrt(vdot(v, v)).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_vnorm_matches_sqrt_vdot() {
    let vs: &[Vec<f64>] = &[
        vec![3.0_f64, 4.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![-1.0, 0.0, 1.0],
    ];
    for v in vs {
        let n = vnorm(v);
        let d = vdot(v, v);
        assert!(
            (n - d.sqrt()).abs() < 1e-12,
            "MR27 vnorm = {n} vs sqrt(vdot) = {}",
            d.sqrt()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — trace(A) equals sum of A's diagonal entries (definition check).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_trace_equals_sum_of_diagonal() {
    let a = vec![
        vec![2.0_f64, 1.0, 0.5],
        vec![0.5, 3.5, -1.0],
        vec![-0.7, 2.0, 4.0],
    ];
    let n = a.len();
    let t = trace(&a);
    let manual: f64 = (0..n).map(|i| a[i][i]).sum();
    assert!(
        (t - manual).abs() < 1e-12,
        "MR28 trace = {t} vs Σdiag = {manual}"
    );
}


