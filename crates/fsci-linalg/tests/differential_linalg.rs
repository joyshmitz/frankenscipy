//! Differential oracle, metamorphic relation, and adversarial tests
//! for fsci-linalg operations (bd-3jh.13.6).
//!
//! Oracle values are hand-computed or derived from matrix algebra identities.

use fsci_linalg::{
    InvOptions, LstsqOptions, PinvOptions, SolveOptions, TriangularSolveOptions, det, inv, lstsq,
    pinv, solve, solve_banded, solve_triangular,
};
use fsci_runtime::RuntimeMode;

const ATOL: f64 = 1e-10;
const RTOL: f64 = 1e-10;

fn assert_close(actual: f64, expected: f64) {
    let tol = ATOL + RTOL * expected.abs();
    assert!(
        (actual - expected).abs() <= tol,
        "assert_close: actual={actual} expected={expected} diff={} tol={tol}",
        (actual - expected).abs()
    );
}

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = ATOL + RTOL * e.abs();
        assert!(
            (a - e).abs() <= tol,
            "[{i}]: actual={a} expected={e} diff={} tol={tol}",
            (a - e).abs()
        );
    }
}

fn assert_mat_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (ar, er)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(ar.len(), er.len(), "row {i} length mismatch");
        for (j, (a, e)) in ar.iter().zip(er.iter()).enumerate() {
            let tol = ATOL + RTOL * e.abs();
            assert!(
                (a - e).abs() <= tol,
                "[{i},{j}]: actual={a} expected={e} diff={}",
                (a - e).abs()
            );
        }
    }
}

/// Matrix-vector multiply: A @ x.
fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Matrix-matrix multiply: A @ B.
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let k = b.len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            for p in 0..k {
                c[i][j] += a[i][p] * b[p][j];
            }
        }
    }
    c
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

// ═══════════════════════════════════════════════════════════════════
// §1  Differential Oracle Tests (>=15)
// ═══════════════════════════════════════════════════════════════════

// D1: solve 2x2 general
#[test]
fn diff_solve_2x2() {
    // A = [[2, 1], [1, 3]], b = [5, 7]
    // Solution: x = [8/5, 9/5] = [1.6, 1.8]
    let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
    let b = vec![5.0, 7.0];
    let result = solve(&a, &b, SolveOptions::default()).unwrap();
    assert_vec_close(&result.x, &[1.6, 1.8]);
}

// D2: solve 3x3 general
#[test]
fn diff_solve_3x3() {
    // A = [[1,0,0],[0,2,0],[0,0,3]], b = [1,4,9] → x = [1,2,3]
    let a = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ];
    let b = vec![1.0, 4.0, 9.0];
    let result = solve(&a, &b, SolveOptions::default()).unwrap();
    assert_vec_close(&result.x, &[1.0, 2.0, 3.0]);
}

// D3: solve_triangular lower 3x3
#[test]
fn diff_solve_triangular_lower_3x3() {
    // L = [[2,0,0],[1,3,0],[4,2,5]], x = [1,2,2]
    // b = L @ x = [2, 1+6, 4+4+10] = [2, 7, 18]
    let a = vec![
        vec![2.0, 0.0, 0.0],
        vec![1.0, 3.0, 0.0],
        vec![4.0, 2.0, 5.0],
    ];
    let b = vec![2.0, 7.0, 18.0];
    let opts = TriangularSolveOptions {
        lower: true,
        ..Default::default()
    };
    let result = solve_triangular(&a, &b, opts).unwrap();
    assert_vec_close(&result.x, &[1.0, 2.0, 2.0]);
}

// D4: solve_triangular upper 3x3
#[test]
fn diff_solve_triangular_upper_3x3() {
    // U = [[2,1,3],[0,4,2],[0,0,5]], b = [17,14,10] → x = [1,1,2]
    let a = vec![
        vec![2.0, 1.0, 3.0],
        vec![0.0, 4.0, 2.0],
        vec![0.0, 0.0, 5.0],
    ];
    let _b = [17.0, 14.0, 10.0];
    // Verify: U @ [1,1,2] = [2+1+6, 0+4+4, 0+0+10] = [9,8,10]... let me recompute
    // Actually: U @ [1,1,2] = [2*1+1*1+3*2, 0*1+4*1+2*2, 0*1+0*1+5*2] = [9,8,10]
    // So b should be [9,8,10]
    let b = vec![9.0, 8.0, 10.0];
    let opts = TriangularSolveOptions {
        lower: false,
        ..Default::default()
    };
    let result = solve_triangular(&a, &b, opts).unwrap();
    assert_vec_close(&result.x, &[1.0, 1.0, 2.0]);
}

// D5: solve_banded tridiagonal 4x4
#[test]
fn diff_solve_banded_tridiag() {
    // Tridiagonal: diag=[4,4,4,4], off-diag=[-1,-1,-1]
    // Dense: [[4,-1,0,0],[-1,4,-1,0],[0,-1,4,-1],[0,0,-1,4]]
    // b = [3,2,2,3] → x = [1,1,1,1] (verify: [4-1,  -1+4-1,  -1+4-1,  -1+4] = [3,2,2,3])
    let ab = vec![
        vec![0.0, -1.0, -1.0, -1.0], // upper diagonal
        vec![4.0, 4.0, 4.0, 4.0],    // main diagonal
        vec![-1.0, -1.0, -1.0, 0.0], // lower diagonal
    ];
    let b = vec![3.0, 2.0, 2.0, 3.0];
    let result = solve_banded((1, 1), &ab, &b, SolveOptions::default()).unwrap();
    assert_vec_close(&result.x, &[1.0, 1.0, 1.0, 1.0]);
}

// D6: inv 2x2
#[test]
fn diff_inv_2x2() {
    // A = [[4, 7], [2, 6]], det=10, inv = 1/10 * [[6,-7],[-2,4]]
    let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
    let result = inv(&a, InvOptions::default()).unwrap();
    assert_mat_close(&result.inverse, &[vec![0.6, -0.7], vec![-0.2, 0.4]]);
}

// D7: inv 3x3 diagonal
#[test]
fn diff_inv_3x3_diag() {
    let a = vec![
        vec![2.0, 0.0, 0.0],
        vec![0.0, 4.0, 0.0],
        vec![0.0, 0.0, 5.0],
    ];
    let result = inv(&a, InvOptions::default()).unwrap();
    assert_mat_close(
        &result.inverse,
        &[
            vec![0.5, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 0.2],
        ],
    );
}

// D8: det 2x2
#[test]
fn diff_det_2x2() {
    // det([[3, 8], [4, 6]]) = 3*6 - 8*4 = 18 - 32 = -14
    let a = vec![vec![3.0, 8.0], vec![4.0, 6.0]];
    let d = det(&a, RuntimeMode::Strict, true).unwrap();
    assert_close(d, -14.0);
}

// D9: det 3x3
#[test]
fn diff_det_3x3() {
    // det([[1,2,3],[4,5,6],[7,8,10]]) = 1*(50-48) - 2*(40-42) + 3*(32-35) = 2+4-9 = -3
    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ];
    let d = det(&a, RuntimeMode::Strict, true).unwrap();
    assert_close(d, -3.0);
}

// D10: lstsq overdetermined 3x2
#[test]
fn diff_lstsq_overdetermined() {
    // A = [[1,0],[0,1],[1,1]], b = [1,2,3]
    // Normal eq: A^T A x = A^T b
    // A^T A = [[2,1],[1,2]], A^T b = [4,5]
    // x = [1, 2] from (2x-1)(2-1) system... let me solve:
    // 2x1 + x2 = 4, x1 + 2x2 = 5 → x1 = 1, x2 = 2
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let b = vec![1.0, 2.0, 3.0];
    let result = lstsq(&a, &b, LstsqOptions::default()).unwrap();
    assert_vec_close(&result.x, &[1.0, 2.0]);
}

// D11: lstsq exact system (full rank square)
#[test]
fn diff_lstsq_exact() {
    // A = [[1,0],[0,1]], b = [3,4] → x = [3,4]
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let b = vec![3.0, 4.0];
    let result = lstsq(&a, &b, LstsqOptions::default()).unwrap();
    assert_vec_close(&result.x, &[3.0, 4.0]);
}

// D12: pinv of identity
#[test]
fn diff_pinv_identity() {
    let a = identity(3);
    let result = pinv(&a, PinvOptions::default()).unwrap();
    assert_mat_close(&result.pseudo_inverse, &identity(3));
}

// D13: pinv of rectangular (wide)
#[test]
fn diff_pinv_wide() {
    // A = [[1,0,0],[0,1,0]] (2x3)
    // pinv(A) = A^T (A A^T)^{-1} = A^T for orthogonal rows
    let a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let result = pinv(&a, PinvOptions::default()).unwrap();
    // Expected pinv: 3x2 = [[1,0],[0,1],[0,0]]
    assert_eq!(result.pseudo_inverse.len(), 3);
    assert_eq!(result.pseudo_inverse[0].len(), 2);
    assert_close(result.pseudo_inverse[0][0], 1.0);
    assert_close(result.pseudo_inverse[1][1], 1.0);
    assert_close(result.pseudo_inverse[2][0], 0.0);
    assert_close(result.pseudo_inverse[2][1], 0.0);
}

// D14: solve + inv consistency: solve(A,b) == inv(A) @ b
#[test]
fn diff_solve_inv_consistency() {
    let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
    let b = vec![5.0, 7.0];
    let x_solve = solve(&a, &b, SolveOptions::default()).unwrap().x;
    let a_inv = inv(&a, InvOptions::default()).unwrap().inverse;
    let x_inv = matvec(&a_inv, &b);
    assert_vec_close(&x_solve, &x_inv);
}

// D15: det identity = 1
#[test]
fn diff_det_identity() {
    for n in 1..=4 {
        let a = identity(n);
        let d = det(&a, RuntimeMode::Strict, true).unwrap();
        assert_close(d, 1.0);
    }
}

// D16: det diagonal = product
#[test]
fn diff_det_diagonal() {
    let a = vec![
        vec![2.0, 0.0, 0.0],
        vec![0.0, 3.0, 0.0],
        vec![0.0, 0.0, 5.0],
    ];
    let d = det(&a, RuntimeMode::Strict, true).unwrap();
    assert_close(d, 30.0);
}

// ═══════════════════════════════════════════════════════════════════
// §2  Metamorphic Relation Tests (>=6)
// ═══════════════════════════════════════════════════════════════════

// M1: solve linearity: solve(A, b1+b2) == solve(A, b1) + solve(A, b2)
#[test]
fn meta_solve_linearity() {
    let a = vec![vec![3.0, 1.0], vec![1.0, 2.0]];
    let b1 = vec![4.0, 3.0];
    let b2 = vec![1.0, 2.0];
    let b_sum = vec_add(&b1, &b2);
    let x1 = solve(&a, &b1, SolveOptions::default()).unwrap().x;
    let x2 = solve(&a, &b2, SolveOptions::default()).unwrap().x;
    let x_sum = solve(&a, &b_sum, SolveOptions::default()).unwrap().x;
    let x12_sum = vec_add(&x1, &x2);
    assert_vec_close(&x_sum, &x12_sum);
}

// M2: inv(A) @ A == I
#[test]
fn meta_inv_identity() {
    let a = vec![
        vec![2.0, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 4.0],
    ];
    let a_inv = inv(&a, InvOptions::default()).unwrap().inverse;
    let product = matmul(&a_inv, &a);
    assert_mat_close(&product, &identity(3));
}

// M3: det(A^T) == det(A) — transpose via index swap
#[test]
fn meta_det_transpose_invariance() {
    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ];
    let at = vec![
        vec![1.0, 4.0, 7.0],
        vec![2.0, 5.0, 8.0],
        vec![3.0, 6.0, 10.0],
    ];
    let da = det(&a, RuntimeMode::Strict, true).unwrap();
    let dat = det(&at, RuntimeMode::Strict, true).unwrap();
    assert_close(da, dat);
}

// M4: pinv(A) @ A @ pinv(A) == pinv(A) (Moore-Penrose condition 2)
#[test]
fn meta_pinv_moore_penrose() {
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let pi = pinv(&a, PinvOptions::default()).unwrap().pseudo_inverse;
    // pi @ A @ pi should equal pi
    let pi_a = matmul(&pi, &a);
    let pi_a_pi = matmul(&pi_a, &pi);
    // pi is 2x3, pi_a is 2x2, pi_a_pi is 2x3
    assert_mat_close(&pi_a_pi, &pi);
}

// M5: lstsq(A, A @ x) recovers x for full-rank A
#[test]
fn meta_lstsq_recovery() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let x_true = vec![1.0, 2.0];
    let b = matvec(&a, &x_true);
    let result = lstsq(&a, &b, LstsqOptions::default()).unwrap();
    assert_vec_close(&result.x, &x_true);
}

// M6: solve(A, b) roundtrip: A @ solve(A, b) == b
#[test]
fn meta_solve_roundtrip() {
    let a = vec![
        vec![4.0, 1.0, 2.0],
        vec![1.0, 5.0, 1.0],
        vec![2.0, 1.0, 6.0],
    ];
    let b = vec![7.0, 8.0, 9.0];
    let x = solve(&a, &b, SolveOptions::default()).unwrap().x;
    let b_recon = matvec(&a, &x);
    assert_vec_close(&b_recon, &b);
}

// M7: det(c*A) = c^n * det(A) for scalar c
#[test]
fn meta_det_scaling() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let c = 3.0;
    let ca: Vec<Vec<f64>> = a
        .iter()
        .map(|row| row.iter().map(|v| v * c).collect())
        .collect();
    let da = det(&a, RuntimeMode::Strict, true).unwrap();
    let dca = det(&ca, RuntimeMode::Strict, true).unwrap();
    let n = a.len() as f64;
    assert_close(dca, c.powf(n) * da);
}

// ═══════════════════════════════════════════════════════════════════
// §3  Adversarial Tests (>=8)
// ═══════════════════════════════════════════════════════════════════

// A1: Singular matrix (det=0) → solve should fail
#[test]
fn adv_singular_solve() {
    let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]]; // rank 1
    let b = vec![1.0, 2.0];
    let result = solve(&a, &b, SolveOptions::default());
    assert!(result.is_err() || result.unwrap().warning.is_some());
}

// A2: NaN in matrix
#[test]
fn adv_nan_in_matrix() {
    let a = vec![vec![1.0, f64::NAN], vec![0.0, 1.0]];
    let b = vec![1.0, 1.0];
    let opts = SolveOptions {
        check_finite: true,
        ..Default::default()
    };
    let result = solve(&a, &b, opts);
    assert!(result.is_err());
}

// A3: Infinity in matrix
#[test]
fn adv_inf_in_matrix() {
    let a = vec![vec![1.0, f64::INFINITY], vec![0.0, 1.0]];
    let b = vec![1.0, 1.0];
    let opts = SolveOptions {
        check_finite: true,
        ..Default::default()
    };
    let result = solve(&a, &b, opts);
    assert!(result.is_err());
}

// A4: Ill-conditioned Hilbert matrix
#[test]
fn adv_hilbert_matrix() {
    // Hilbert 4x4: H[i,j] = 1/(i+j+1)
    let n = 4;
    let a: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| 1.0 / (i as f64 + j as f64 + 1.0)).collect())
        .collect();
    let b = vec![1.0; n];
    // Should not panic; may have warning about conditioning
    let result = solve(&a, &b, SolveOptions::default());
    assert!(result.is_ok());
}

// A5: Empty matrix — nalgebra panics on empty, so this tests that path
#[test]
#[should_panic]
fn adv_empty_matrix() {
    let a: Vec<Vec<f64>> = vec![];
    let b: Vec<f64> = vec![];
    let _ = solve(&a, &b, SolveOptions::default());
}

// A6: Mismatched dimensions
#[test]
fn adv_dimension_mismatch() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b = vec![1.0, 2.0, 3.0]; // 3 elements vs 2x2 matrix
    let result = solve(&a, &b, SolveOptions::default());
    assert!(result.is_err());
}

// A7: Non-square matrix for solve
#[test]
fn adv_non_square_solve() {
    let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let b = vec![1.0, 2.0];
    let result = solve(&a, &b, SolveOptions::default());
    assert!(result.is_err());
}

// A8: Very large entries near overflow
#[test]
fn adv_large_entries() {
    let a = vec![vec![1e150, 0.0], vec![0.0, 1e150]];
    let b = vec![1e150, 1e150];
    let result = solve(&a, &b, SolveOptions::default());
    if let Ok(r) = result {
        assert_vec_close(&r.x, &[1.0, 1.0]);
    }
    // Either succeeds with correct answer or errors gracefully
}

// A9: Very small entries near underflow
#[test]
fn adv_small_entries() {
    let a = vec![vec![1e-150, 0.0], vec![0.0, 1e-150]];
    let b = vec![1e-150, 1e-150];
    let result = solve(&a, &b, SolveOptions::default());
    if let Ok(r) = result {
        assert_vec_close(&r.x, &[1.0, 1.0]);
    }
}

// A10: Non-square for inv
#[test]
fn adv_inv_non_square() {
    let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let result = inv(&a, InvOptions::default());
    assert!(result.is_err());
}

// A11: Non-square for det
#[test]
fn adv_det_non_square() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let result = det(&a, RuntimeMode::Strict, true);
    assert!(result.is_err());
}
