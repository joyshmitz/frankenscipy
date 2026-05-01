//! Metamorphic tests for `fsci-sparse`.
//!
//! Format conversion round-trips, matvec/spmv consistency, transpose
//! involution, sparse-dense agreement, spsolve correctness.
//!
//! Run with: `cargo test -p fsci-sparse --test metamorphic_tests`

use fsci_runtime::RuntimeMode;
use fsci_sparse::{
    CooMatrix, CsrMatrix, EigsOptions, IterativeSolveOptions, Shape2D, SolveOptions, add_csr, bicg,
    bicgstab, cg, coo_to_csr_with_mode, csr_to_csc_with_mode, eigsh, gmres, sparse_transpose,
    spmv, spsolve, svds,
};

const ATOL: f64 = 1e-9;
const RTOL: f64 = 1e-7;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

/// Convert CSR to a dense row-major Vec<Vec<f64>>.
fn csr_to_dense(a: &CsrMatrix) -> Vec<Vec<f64>> {
    let s = a.shape();
    let mut m = vec![vec![0.0; s.cols]; s.rows];
    for i in 0..s.rows {
        let start = a.indptr()[i];
        let end = a.indptr()[i + 1];
        for k in start..end {
            let j = a.indices()[k];
            m[i][j] = a.data()[k];
        }
    }
    m
}

/// Build a 5×5 SPD CSR test matrix (diagonally dominant).
fn build_spd_csr() -> CsrMatrix {
    // Tridiagonal SPD: 2 on diagonal, -1 on off-diagonals.
    let n = 5;
    let mut data = Vec::new();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for i in 0..n {
        if i > 0 {
            data.push(-1.0);
            rows.push(i);
            cols.push(i - 1);
        }
        data.push(2.0);
        rows.push(i);
        cols.push(i);
        if i + 1 < n {
            data.push(-1.0);
            rows.push(i);
            cols.push(i + 1);
        }
    }
    let coo =
        CooMatrix::from_triplets(Shape2D { rows: n, cols: n }, data, rows, cols, true).unwrap();
    coo_to_csr_with_mode(&coo, RuntimeMode::Strict, "test_build_spd").unwrap().0
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — CSR → CSC → CSR round-trip preserves the dense matrix.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_csr_csc_roundtrip() {
    let a = build_spd_csr();
    let dense_orig = csr_to_dense(&a);

    let (csc, _) = csr_to_csc_with_mode(&a, RuntimeMode::Strict, "mr1_csc").unwrap();
    // Convert back via COO triplets reconstructed from CSC.
    let mut data = Vec::new();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for j in 0..csc.shape().cols {
        let start = csc.indptr()[j];
        let end = csc.indptr()[j + 1];
        for k in start..end {
            data.push(csc.data()[k]);
            rows.push(csc.indices()[k]);
            cols.push(j);
        }
    }
    let coo = CooMatrix::from_triplets(csc.shape(), data, rows, cols, true).unwrap();
    let back = coo_to_csr_with_mode(&coo, RuntimeMode::Strict, "mr1_back").unwrap().0;
    let dense_back = csr_to_dense(&back);
    assert_eq!(dense_orig.len(), dense_back.len(), "MR1 row count");
    for (r1, r2) in dense_orig.iter().zip(&dense_back) {
        for (a, b) in r1.iter().zip(r2) {
            assert!(close(*a, *b), "MR1 element {a} vs {b}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — sparse_transpose is involutive: T(T(A)) = A.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_transpose_involution() {
    let a = build_spd_csr();
    let at = sparse_transpose(&a);
    let att = sparse_transpose(&at);
    let dense_a = csr_to_dense(&a);
    let dense_att = csr_to_dense(&att);
    for (r1, r2) in dense_a.iter().zip(&dense_att) {
        for (x, y) in r1.iter().zip(r2) {
            assert!(close(*x, *y), "MR2 transpose involution broke");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — spmv linearity: A · (αx + βy) = α (A · x) + β (A · y).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_spmv_linearity() {
    let a = build_spd_csr();
    let n = a.shape().cols;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.7).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * (-0.3)).collect();
    let alpha = 1.7_f64;
    let beta = -0.4_f64;
    let combined: Vec<f64> = x.iter().zip(&y).map(|(a, b)| alpha * a + beta * b).collect();

    let r_combined = spmv(&a, &combined);
    let r_x = spmv(&a, &x);
    let r_y = spmv(&a, &y);

    for i in 0..n {
        let expected = alpha * r_x[i] + beta * r_y[i];
        assert!(
            close(r_combined[i], expected),
            "MR3 linearity at i={i}: got {} expected {expected}",
            r_combined[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — spmv against the zero vector is zero.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_spmv_zero_vector_is_zero() {
    let a = build_spd_csr();
    let n = a.shape().cols;
    let zero = vec![0.0; n];
    let result = spmv(&a, &zero);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.abs() < 1e-15, "MR4 spmv(A, 0) at i={i}: {v}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — spsolve gives the correct solution: A · x = b.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_spsolve_residual_small() {
    let a = build_spd_csr();
    let n = a.shape().cols;
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let res = spsolve(&a, &b, SolveOptions::default()).unwrap();
    let ax = spmv(&a, &res.solution);
    for i in 0..n {
        let r = ax[i] - b[i];
        assert!(
            r.abs() < 1e-9,
            "MR5 spsolve residual at i={i}: r={r}, ax={}, b={}",
            ax[i],
            b[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — cg (conjugate gradient) solves an SPD system to small residual.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cg_residual_small_on_spd() {
    let a = build_spd_csr();
    let n = a.shape().cols;
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let opts = IterativeSolveOptions {
        mode: RuntimeMode::Strict,
        check_finite: false,
        tol: 1e-12,
        max_iter: Some(200),
    };
    let res = cg(&a, &b, None, opts).unwrap();
    assert!(
        res.converged,
        "MR6 cg did not converge in {} iters",
        res.iterations
    );
    let ax = spmv(&a, &res.solution);
    for i in 0..n {
        let r = ax[i] - b[i];
        assert!(r.abs() < 1e-8, "MR6 cg residual at i={i}: {r}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — spsolve and cg agree on the solution for an SPD system.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_spsolve_cg_agree_on_spd() {
    let a = build_spd_csr();
    let n = a.shape().cols;
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let direct = spsolve(&a, &b, SolveOptions::default()).unwrap().solution;
    let opts = IterativeSolveOptions {
        mode: RuntimeMode::Strict,
        check_finite: false,
        tol: 1e-12,
        max_iter: Some(200),
    };
    let iterative = cg(&a, &b, None, opts).unwrap().solution;
    for i in 0..n {
        assert!(
            close(direct[i], iterative[i]),
            "MR7 spsolve vs cg at i={i}: {} vs {}",
            direct[i],
            iterative[i]
        );
    }
}

/// Build a 5x5 non-symmetric diagonally-dominant CSR for iterative
/// solver tests.
fn build_dd_nonsym_csr() -> CsrMatrix {
    let n = 5;
    let mut data = Vec::new();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for i in 0..n {
        // Asymmetric off-diagonals so the matrix is not symmetric but
        // remains strictly diagonally dominant (CG would fail; bicgstab
        // and gmres still converge).
        if i > 0 {
            data.push(-0.4);
            rows.push(i);
            cols.push(i - 1);
        }
        data.push(3.0);
        rows.push(i);
        cols.push(i);
        if i + 1 < n {
            data.push(-0.7);
            rows.push(i);
            cols.push(i + 1);
        }
    }
    let coo =
        CooMatrix::from_triplets(Shape2D { rows: n, cols: n }, data, rows, cols, true).unwrap();
    coo_to_csr_with_mode(&coo, RuntimeMode::Strict, "build_dd_nonsym")
        .unwrap()
        .0
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — bicgstab on a non-symmetric DD matrix produces small residual.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bicgstab_residual_small_nonsym() {
    let a = build_dd_nonsym_csr();
    let n = a.shape().cols;
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let opts = IterativeSolveOptions {
        mode: RuntimeMode::Strict,
        check_finite: false,
        tol: 1e-12,
        max_iter: Some(500),
    };
    let res = bicgstab(&a, &b, None, opts).unwrap();
    let ax = spmv(&a, &res.solution);
    for i in 0..n {
        let r = ax[i] - b[i];
        assert!(r.abs() < 1e-8, "MR8 bicgstab residual at i={i}: {r}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — gmres on a non-symmetric DD matrix produces small residual.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gmres_residual_small_nonsym() {
    let a = build_dd_nonsym_csr();
    let n = a.shape().cols;
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let opts = IterativeSolveOptions {
        mode: RuntimeMode::Strict,
        check_finite: false,
        tol: 1e-12,
        max_iter: Some(500),
    };
    let res = gmres(&a, &b, None, opts).unwrap();
    let ax = spmv(&a, &res.solution);
    for i in 0..n {
        let r = ax[i] - b[i];
        assert!(r.abs() < 1e-8, "MR9 gmres residual at i={i}: {r}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — bicgstab and gmres agree element-wise on the non-symmetric
// DD problem (different methods but same converged solution).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bicgstab_gmres_agree_nonsym() {
    let a = build_dd_nonsym_csr();
    let n = a.shape().cols;
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let opts = IterativeSolveOptions {
        mode: RuntimeMode::Strict,
        check_finite: false,
        tol: 1e-12,
        max_iter: Some(500),
    };
    let r1 = bicgstab(&a, &b, None, opts).unwrap().solution;
    let r2 = gmres(&a, &b, None, opts).unwrap().solution;
    for i in 0..n {
        assert!(
            close(r1[i], r2[i]),
            "MR10 bicgstab vs gmres at i={i}: {} vs {}",
            r1[i],
            r2[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — bicg on the non-symmetric DD matrix produces small residual.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bicg_residual_small_nonsym() {
    let a = build_dd_nonsym_csr();
    let n = a.shape().cols;
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let opts = IterativeSolveOptions {
        mode: RuntimeMode::Strict,
        check_finite: false,
        tol: 1e-12,
        max_iter: Some(500),
    };
    let res = bicg(&a, &b, None, opts).unwrap();
    let ax = spmv(&a, &res.solution);
    for i in 0..n {
        let r = ax[i] - b[i];
        assert!(r.abs() < 1e-8, "MR11 bicg residual at i={i}: {r}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — eigsh on a SPD matrix returns real eigenvalues with small
// residual ||A v − λ v|| / ||v||.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_eigsh_residual_small_on_spd() {
    let a = build_spd_csr();
    let n = a.shape().rows;
    let k = 2.min(n);
    let res = eigsh(&a, k, EigsOptions::default()).unwrap();
    assert_eq!(res.eigenvalues.len(), k);
    assert_eq!(res.eigenvectors.len(), k);
    for (idx, lambda) in res.eigenvalues.iter().enumerate() {
        let v = &res.eigenvectors[idx];
        let av = spmv(&a, v);
        let mut residual = 0.0_f64;
        let mut norm = 0.0_f64;
        for (i, av_i) in av.iter().enumerate() {
            let r = av_i - lambda * v[i];
            residual += r * r;
            norm += v[i] * v[i];
        }
        let rel = residual.sqrt() / norm.sqrt().max(1e-12);
        // Default eigsh tol ≈ 1e-6 on the residual — allow a small
        // multiplier for the test bound.
        assert!(
            rel < 1e-4,
            "MR12 eigsh residual at idx={idx}: λ={lambda}, ||Av-λv||/||v||={rel}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — svds singular values are non-negative and sorted descending.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_svds_singular_values_sorted_nonneg() {
    let a = build_spd_csr();
    let n = a.shape().rows;
    let k = 3.min(n - 1);
    let res = svds(&a, k, EigsOptions::default()).unwrap();
    let s = &res.singular_values;
    assert_eq!(s.len(), k);
    for (i, &v) in s.iter().enumerate() {
        assert!(v >= 0.0, "MR13 negative singular value at i={i}: {v}");
    }
    for w in s.windows(2) {
        assert!(
            w[0] >= w[1] - 1e-10,
            "MR13 singular values not descending: {} < {}",
            w[0],
            w[1]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — add_csr is commutative: A + B = B + A as a dense matrix.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_add_csr_commutative() {
    let a = build_spd_csr();
    let b = build_spd_csr(); // same structure but a and b are independently constructed
    let ab = add_csr(&a, &b).unwrap();
    let ba = add_csr(&b, &a).unwrap();
    let dense_ab = csr_to_dense(&ab);
    let dense_ba = csr_to_dense(&ba);
    assert_eq!(dense_ab.len(), dense_ba.len(), "MR14 row count");
    for (r1, r2) in dense_ab.iter().zip(&dense_ba) {
        for (a, b) in r1.iter().zip(r2) {
            assert!(close(*a, *b), "MR14 add_csr commutative element diff");
        }
    }
}
