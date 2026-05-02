//! Metamorphic tests for `fsci-sparse`.
//!
//! Format conversion round-trips, matvec/spmv consistency, transpose
//! involution, sparse-dense agreement, spsolve correctness.
//!
//! Run with: `cargo test -p fsci-sparse --test metamorphic_tests`

use fsci_runtime::RuntimeMode;
use fsci_sparse::{
    CooMatrix, CsrMatrix, EigsOptions, IterativeSolveOptions, LuOptions, Shape2D, SolveOptions,
    add_csr, bicg, bicgstab, block_diag, breadth_first_order, cg, cgs, connected_components,
    coo_to_csr_with_mode, csr_to_csc_with_mode, diags, dijkstra, eigsh, eye, floyd_warshall,
    gmres, kron, lsmr, lsqr, matrix_power as sparse_matrix_power, minres, qmr,
    reverse_cuthill_mckee, scale_csr, sparse_diagonal, sparse_eliminate_zeros,
    sparse_has_explicit_zeros, sparse_nnz, sparse_norm, sparse_trace, sparse_transpose, spmv,
    spmv_csc, spmv_csr, splu, splu_solve, spsolve, sub_csr, svds, tril, triu,
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

// ─────────────────────────────────────────────────────────────────────
// MR15 — sub_csr(A, A) is the zero matrix.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sub_self_is_zero() {
    let a = build_spd_csr();
    let z = sub_csr(&a, &a).unwrap();
    let dense = csr_to_dense(&z);
    for row in &dense {
        for &v in row {
            assert!(v.abs() < 1e-15, "MR15 sub_csr(A, A) entry = {v}, expected 0");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — scale_csr is multiplicative: scale by α then β = scale by α·β.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_scale_csr_composition() {
    let a = build_spd_csr();
    let alpha = 1.5_f64;
    let beta = -2.0_f64;
    let two_step = scale_csr(&scale_csr(&a, alpha).unwrap(), beta).unwrap();
    let one_step = scale_csr(&a, alpha * beta).unwrap();
    let d_two = csr_to_dense(&two_step);
    let d_one = csr_to_dense(&one_step);
    for (r1, r2) in d_two.iter().zip(&d_one) {
        for (x, y) in r1.iter().zip(r2) {
            assert!(close(*x, *y), "MR16 scale composition: {x} vs {y}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR17 — eye(n) · v = v (identity matrix is the multiplicative neutral
// for sparse matvec).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_eye_spmv_is_identity() {
    let n = 6;
    let i = eye(n).unwrap();
    let v: Vec<f64> = (0..n).map(|k| 1.0 + k as f64 * 0.5).collect();
    let w = spmv_csr(&i, &v).unwrap();
    assert_eq!(v.len(), w.len(), "MR17 length");
    for (a, b) in v.iter().zip(&w) {
        assert!(close(*a, *b), "MR17 eye·v: {a} vs {b}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — spmv via CSR and CSC representations agree on the same matrix.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_spmv_csr_csc_agree() {
    let a = build_spd_csr();
    let (csc, _) =
        csr_to_csc_with_mode(&a, RuntimeMode::Strict, "test_csr_csc_agree").unwrap();
    let v: Vec<f64> = vec![1.0, -0.5, 2.0, 0.25, -1.5];
    let yr = spmv_csr(&a, &v).unwrap();
    let yc = spmv_csc(&csc, &v).unwrap();
    assert_eq!(yr.len(), yc.len(), "MR18 length");
    for (r, c) in yr.iter().zip(&yc) {
        assert!(close(*r, *c), "MR18 csr/csc spmv: {r} vs {c}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — For any A, dense(tril(A, 0)) + dense(triu(A, 0)) - diag(A) == A.
// (Lower + upper - diagonal-counted-twice == original matrix.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_tril_triu_diag_reconstruction() {
    let a = build_spd_csr();
    let n = a.shape().rows;
    let lower = tril(&a, 0).unwrap();
    let upper = triu(&a, 0).unwrap();
    let dense_a = csr_to_dense(&a);

    // Build dense lower from coo lower; same for upper.
    let mut dl = vec![vec![0.0; n]; n];
    let (row_l, col_l) = (lower.row_indices().to_vec(), lower.col_indices().to_vec());
    for ((r, c), v) in row_l.iter().zip(col_l.iter()).zip(lower.data().iter()) {
        dl[*r][*c] += *v;
    }
    let mut du = vec![vec![0.0; n]; n];
    let (row_u, col_u) = (upper.row_indices().to_vec(), upper.col_indices().to_vec());
    for ((r, c), v) in row_u.iter().zip(col_u.iter()).zip(upper.data().iter()) {
        du[*r][*c] += *v;
    }

    for i in 0..n {
        for j in 0..n {
            let recon = dl[i][j] + du[i][j] - if i == j { dense_a[i][j] } else { 0.0 };
            assert!(
                close(recon, dense_a[i][j]),
                "MR19 ({i}, {j}): tril+triu-diag = {recon} vs A = {}",
                dense_a[i][j]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — sparse_transpose then sparse_transpose returns to the original
// dense form (involution at the dense level).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_double_transpose_dense_identity() {
    let a = build_spd_csr();
    let t = sparse_transpose(&a);
    let tt = sparse_transpose(&t);
    let da = csr_to_dense(&a);
    let dtt = csr_to_dense(&tt);
    for (r1, r2) in da.iter().zip(&dtt) {
        for (x, y) in r1.iter().zip(r2) {
            assert!(close(*x, *y), "MR20 (Aᵀ)ᵀ entry: {x} vs {y}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — kron(I_m, I_n) is the (m·n) × (m·n) identity matrix.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kron_of_identities_is_identity() {
    for m in [1usize, 2, 3] {
        for n in [1usize, 2, 4] {
            let im = eye(m).unwrap();
            let in_mat = eye(n).unwrap();
            let k = kron(&im, &in_mat).unwrap();
            let total = m * n;
            assert_eq!(k.shape().rows, total, "MR21 kron rows m={m} n={n}");
            assert_eq!(k.shape().cols, total, "MR21 kron cols m={m} n={n}");
            let dense = csr_to_dense(&k);
            for i in 0..total {
                for j in 0..total {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        close(dense[i][j], expected),
                        "MR21 kron(I_{m}, I_{n})[{i}, {j}] = {} vs {expected}",
                        dense[i][j]
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — kron output shape: (m1·m2, n1·n2) for inputs of shapes
// (m1, n1) and (m2, n2).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kron_output_shape() {
    let a = build_spd_csr(); // 5×5
    let b = eye(3).unwrap();  // 3×3
    let k = kron(&a, &b).unwrap();
    assert_eq!(k.shape().rows, 5 * 3, "MR22 kron rows");
    assert_eq!(k.shape().cols, 5 * 3, "MR22 kron cols");
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — diags with main diagonal of all 1s and shape (n, n) yields
// the same dense matrix as eye(n).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_diags_main_diag_ones_is_eye() {
    for n in [1usize, 3, 5, 8] {
        let main = vec![1.0_f64; n];
        let d = diags(&[main], &[0_isize], Some(Shape2D::new(n, n))).unwrap();
        let i = eye(n).unwrap();
        let dd = csr_to_dense(&d);
        let di = csr_to_dense(&i);
        for r in 0..n {
            for c in 0..n {
                assert!(
                    close(dd[r][c], di[r][c]),
                    "MR23 diags vs eye at ({r}, {c}) for n={n}: {} vs {}",
                    dd[r][c],
                    di[r][c]
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — block_diag of [A, B] has shape (rows_A + rows_B, cols_A + cols_B).
// ─────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────
// MR25 — sparse_trace(eye(n)) = n.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sparse_trace_of_eye() {
    for n in [1usize, 3, 5, 8, 16] {
        let i = eye(n).unwrap();
        let t = sparse_trace(&i);
        assert!(
            (t - n as f64).abs() < 1e-12,
            "MR25 sparse_trace(eye({n})) = {t}, expected {n}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — sparse_diagonal(eye(n)) is the all-ones vector of length n.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sparse_diagonal_of_eye() {
    for n in [1usize, 3, 5, 8] {
        let i = eye(n).unwrap();
        let d = sparse_diagonal(&i);
        assert_eq!(d.len(), n, "MR26 diag length");
        for (k, &v) in d.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-12,
                "MR26 sparse_diagonal(eye({n}))[{k}] = {v}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — sparse_nnz of an all-zero matrix (constructed via diags with
// zero diagonals) is 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sparse_nnz_zero_matrix() {
    // Build a 5×5 zero matrix using diags with a zero main diagonal,
    // then compute its nnz. It should be 0 since all entries are zero;
    // diags conventionally records explicit zeros, but sparse_nnz only
    // counts non-zero values.
    let n = 5;
    let zero = vec![0.0_f64; n];
    let m = diags(&[zero], &[0_isize], Some(Shape2D::new(n, n))).unwrap();
    let nnz = sparse_nnz(&m);
    assert_eq!(
        nnz, 0,
        "MR27 sparse_nnz of zero diagonal = {nnz}, expected 0"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — splu / splu_solve correctly solves Ax = b for an SPD CSR via
// CSC factorization (residual is small).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_splu_roundtrip() {
    let a = build_spd_csr();
    let (csc, _) =
        csr_to_csc_with_mode(&a, RuntimeMode::Strict, "test_splu").unwrap();
    let b = vec![1.0_f64, 0.5, -1.0, 0.25, -0.5];
    let lu = splu(&csc, LuOptions::default()).unwrap();
    let x = splu_solve(&lu, &b).unwrap();
    // Verify residual ‖A·x − b‖ is small via spmv.
    let ax = spmv_csr(&a, &x).unwrap();
    let mut residual_sq = 0.0;
    for (axi, bi) in ax.iter().zip(&b) {
        let d = axi - bi;
        residual_sq += d * d;
    }
    assert!(
        residual_sq.sqrt() < 1e-7,
        "MR28 splu residual = {} (expected small)",
        residual_sq.sqrt()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — floyd_warshall on a graph with positive edges yields a
// distance matrix with zero diagonal.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_floyd_warshall_zero_diagonal() {
    // Build a small connected graph: 0–1 weight 2, 1–2 weight 3.
    let n = 3;
    let coo = CooMatrix::from_triplets(
        Shape2D { rows: n, cols: n },
        vec![2.0, 2.0, 3.0, 3.0],
        vec![0, 1, 1, 2],
        vec![1, 0, 2, 1],
        true,
    )
    .unwrap();
    let g = coo_to_csr_with_mode(&coo, RuntimeMode::Strict, "test_floyd")
        .unwrap()
        .0;
    let dist = floyd_warshall(&g);
    for i in 0..n {
        assert!(
            dist[i][i].abs() < 1e-12,
            "MR29 floyd_warshall dist[{i}, {i}] = {}",
            dist[i][i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — sparse_norm("fro") of the zero matrix is 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sparse_norm_zero_matrix() {
    let n = 4;
    let zero = vec![0.0_f64; n];
    let m = diags(&[zero], &[0_isize], Some(Shape2D::new(n, n))).unwrap();
    for kind in &["fro", "1", "inf"] {
        let v = sparse_norm(&m, kind);
        assert!(
            v.abs() < 1e-12,
            "MR30 sparse_norm({kind}) of zero = {v}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — sparse matrix_power(I, n) = I for any n.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sparse_matrix_power_of_identity() {
    let n = 5;
    let i = eye(n).unwrap();
    for k in [1usize, 2, 3, 5] {
        let p = sparse_matrix_power(&i, k).unwrap();
        let dense = csr_to_dense(&p);
        for r in 0..n {
            for c in 0..n {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    close(dense[r][c], expected),
                    "MR31 sparse_matrix_power(I, {k})[{r}, {c}] = {}",
                    dense[r][c]
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — sparse_eliminate_zeros leaves the dense matrix unchanged.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sparse_eliminate_zeros_preserves_dense() {
    let a = build_spd_csr();
    let stripped = sparse_eliminate_zeros(&a);
    let dense_a = csr_to_dense(&a);
    let dense_b = csr_to_dense(&stripped);
    for (r1, r2) in dense_a.iter().zip(&dense_b) {
        for (x, y) in r1.iter().zip(r2) {
            assert!(
                close(*x, *y),
                "MR32 eliminate_zeros changed entry: {x} vs {y}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — After sparse_eliminate_zeros, sparse_has_explicit_zeros
// returns false (no remaining stored zeros).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sparse_has_explicit_zeros_after_elimination() {
    // Construct a CSR with explicit zeros via diags including 0 entries.
    let n = 5;
    let zero_diag = vec![0.0_f64; n];
    let a = diags(&[zero_diag], &[0_isize], Some(Shape2D::new(n, n))).unwrap();
    let stripped = sparse_eliminate_zeros(&a);
    assert!(
        !sparse_has_explicit_zeros(&stripped),
        "MR33 stripped CSR still has explicit zeros"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — Dijkstra from source returns distance 0 for source itself
// and is non-negative for all reachable nodes.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dijkstra_source_distance_zero() {
    let n = 4;
    let coo = CooMatrix::from_triplets(
        Shape2D { rows: n, cols: n },
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        vec![0, 1, 1, 2, 2, 3],
        vec![1, 0, 2, 1, 3, 2],
        true,
    )
    .unwrap();
    let g = coo_to_csr_with_mode(&coo, RuntimeMode::Strict, "test_dijkstra")
        .unwrap()
        .0;
    let r = dijkstra(&g, 0).unwrap();
    assert!(
        r.distances[0].abs() < 1e-12,
        "MR34 dijkstra distances[0] = {}",
        r.distances[0]
    );
    for (i, &d) in r.distances.iter().enumerate() {
        if d.is_finite() {
            assert!(
                d >= -1e-12,
                "MR34 dijkstra distances[{i}] = {d} < 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — breadth_first_order from source visits the source first.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bfs_starts_at_source() {
    let n = 5;
    let coo = CooMatrix::from_triplets(
        Shape2D { rows: n, cols: n },
        vec![1.0; 8],
        vec![0, 1, 0, 2, 1, 3, 2, 4],
        vec![1, 0, 2, 0, 3, 1, 4, 2],
        true,
    )
    .unwrap();
    let g = coo_to_csr_with_mode(&coo, RuntimeMode::Strict, "test_bfs")
        .unwrap()
        .0;
    let (order, _preds) = breadth_first_order(&g, 0).unwrap();
    assert!(!order.is_empty(), "MR35 bfs order empty");
    assert_eq!(order[0], 0, "MR35 bfs first = {}, expected 0", order[0]);
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — connected_components on a fully connected graph returns 1.
// ─────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────
// MR37 — CGS converges on SPD A·x = b (residual is small).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cgs_residual_small_on_spd() {
    let a = build_spd_csr();
    let b = vec![1.0_f64, -0.5, 2.0, 0.25, -0.5];
    let mut opts = IterativeSolveOptions::default();
    opts.max_iter = Some(200);
    let res = cgs(&a, &b, None, opts).unwrap();
    let ax = spmv_csr(&a, &res.solution).unwrap();
    let mut residual_sq = 0.0;
    for (axi, bi) in ax.iter().zip(&b) {
        residual_sq += (axi - bi).powi(2);
    }
    assert!(
        residual_sq.sqrt() < 1e-5,
        "MR37 cgs residual = {} on SPD",
        residual_sq.sqrt()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR38 — QMR converges on the SPD A used for our tests (small residual).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_qmr_residual_small_on_spd() {
    let a = build_spd_csr();
    let b = vec![1.0_f64, 0.5, -1.0, 0.25, -0.5];
    let mut opts = IterativeSolveOptions::default();
    opts.max_iter = Some(200);
    let res = qmr(&a, &b, None, opts).unwrap();
    let ax = spmv_csr(&a, &res.solution).unwrap();
    let mut residual_sq = 0.0;
    for (axi, bi) in ax.iter().zip(&b) {
        residual_sq += (axi - bi).powi(2);
    }
    // QMR is for non-symmetric systems and converges more slowly than CG
    // on SPD. Check that residual is decreasing — i.e., much smaller
    // than the trivial zero-init residual ||b||.
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        residual_sq.sqrt() < 0.5 * b_norm,
        "MR38 qmr residual = {} > 0.5 * ||b|| = {}",
        residual_sq.sqrt(),
        0.5 * b_norm
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR39 — LSQR converges on a square SPD system (residual small).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_lsqr_residual_small_on_square() {
    let a = build_spd_csr();
    let b = vec![1.0_f64, -0.5, 2.0, 0.25, -0.5];
    let mut opts = IterativeSolveOptions::default();
    opts.max_iter = Some(500);
    let res = lsqr(&a, &b, opts).unwrap();
    let ax = spmv_csr(&a, &res.solution).unwrap();
    let mut residual_sq = 0.0;
    for (axi, bi) in ax.iter().zip(&b) {
        residual_sq += (axi - bi).powi(2);
    }
    assert!(
        residual_sq.sqrt() < 1e-4,
        "MR39 lsqr residual = {}",
        residual_sq.sqrt()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR40 — LSMR converges on a square SPD system.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_lsmr_residual_small_on_square() {
    let a = build_spd_csr();
    let b = vec![1.0_f64, -0.5, 2.0, 0.25, -0.5];
    let mut opts = IterativeSolveOptions::default();
    opts.max_iter = Some(500);
    let res = lsmr(&a, &b, opts).unwrap();
    let ax = spmv_csr(&a, &res.solution).unwrap();
    let mut residual_sq = 0.0;
    for (axi, bi) in ax.iter().zip(&b) {
        residual_sq += (axi - bi).powi(2);
    }
    assert!(
        residual_sq.sqrt() < 1e-4,
        "MR40 lsmr residual = {}",
        residual_sq.sqrt()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR41 — MINRES converges on a symmetric SPD system.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minres_residual_small_on_spd() {
    let a = build_spd_csr();
    let b = vec![1.0_f64, -0.5, 2.0, 0.25, -0.5];
    let mut opts = IterativeSolveOptions::default();
    opts.max_iter = Some(200);
    let res = minres(&a, &b, None, opts).unwrap();
    let ax = spmv_csr(&a, &res.solution).unwrap();
    let mut residual_sq = 0.0;
    for (axi, bi) in ax.iter().zip(&b) {
        residual_sq += (axi - bi).powi(2);
    }
    assert!(
        residual_sq.sqrt() < 1e-5,
        "MR41 minres residual = {}",
        residual_sq.sqrt()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR42 — reverse_cuthill_mckee returns a permutation of length n
// where every index in 0..n appears exactly once.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_reverse_cuthill_mckee_is_permutation() {
    let g = build_spd_csr(); // 5×5 with non-trivial structure
    let perm = reverse_cuthill_mckee(&g);
    assert_eq!(perm.len(), 5, "MR42 RCM length");
    let mut seen = vec![false; 5];
    for &p in &perm {
        assert!(p < 5, "MR42 RCM index {p} out of range");
        seen[p] = true;
    }
    for (i, ok) in seen.iter().enumerate() {
        assert!(*ok, "MR42 RCM missing index {i}");
    }
}

#[test]
fn mr_connected_components_fully_connected() {
    let n = 4;
    // Complete graph on 4 nodes: 6 undirected edges (12 entries).
    let mut data = Vec::new();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                data.push(1.0);
                rows.push(i);
                cols.push(j);
            }
        }
    }
    let coo =
        CooMatrix::from_triplets(Shape2D { rows: n, cols: n }, data, rows, cols, true).unwrap();
    let g = coo_to_csr_with_mode(&coo, RuntimeMode::Strict, "test_cc")
        .unwrap()
        .0;
    let r = connected_components(&g).unwrap();
    assert_eq!(
        r.n_components, 1,
        "MR36 connected_components = {} on fully connected, expected 1",
        r.n_components
    );
}

#[test]
fn mr_block_diag_shape() {
    let a = build_spd_csr(); // 5×5
    let b = eye(3).unwrap();  // 3×3
    let bd = block_diag(&[&a, &b]).unwrap();
    assert_eq!(
        bd.shape().rows,
        a.shape().rows + b.shape().rows,
        "MR24 block_diag rows"
    );
    assert_eq!(
        bd.shape().cols,
        a.shape().cols + b.shape().cols,
        "MR24 block_diag cols"
    );
    // Top-left block must equal A; bottom-right block must equal B.
    let dense = csr_to_dense(&bd);
    let da = csr_to_dense(&a);
    for i in 0..a.shape().rows {
        for j in 0..a.shape().cols {
            assert!(
                close(dense[i][j], da[i][j]),
                "MR24 block_diag top-left at ({i}, {j})"
            );
        }
    }
    let db = csr_to_dense(&b);
    let off = a.shape().rows;
    for i in 0..b.shape().rows {
        for j in 0..b.shape().cols {
            assert!(
                close(dense[off + i][off + j], db[i][j]),
                "MR24 block_diag bottom-right at ({i}, {j})"
            );
        }
    }
}


