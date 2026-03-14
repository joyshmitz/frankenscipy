use fsci_runtime::RuntimeMode;
use nalgebra::{DMatrix, DVector, Dyn, LU};

use crate::formats::{CscMatrix, CsrMatrix, SparseError, SparseResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseBackend {
    Auto,
    Umfpack,
    Superlu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermutationOrdering {
    Colamd,
    Natural,
    MmdAta,
    MmdAtPlusA,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveOptions {
    pub mode: RuntimeMode,
    pub backend: SparseBackend,
    pub ordering: PermutationOrdering,
    pub check_finite: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            backend: SparseBackend::Auto,
            ordering: PermutationOrdering::Colamd,
            check_finite: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LuOptions {
    pub mode: RuntimeMode,
    pub ordering: PermutationOrdering,
    pub diag_pivot_thresh: f64,
}

impl Default for LuOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            ordering: PermutationOrdering::Colamd,
            diag_pivot_thresh: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IluOptions {
    pub mode: RuntimeMode,
    pub ordering: PermutationOrdering,
    pub drop_tol: f64,
    pub fill_factor: f64,
}

impl Default for IluOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            ordering: PermutationOrdering::Colamd,
            drop_tol: 1e-4,
            fill_factor: 10.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveResult {
    pub solution: Vec<f64>,
    pub backend_used: SparseBackend,
    pub ordering_used: PermutationOrdering,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SparseLuFactorization {
    pub shape: (usize, usize),
    pub backend_used: SparseBackend,
    pub ordering_used: PermutationOrdering,
    lu_internal: LU<f64, Dyn, Dyn>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseIluFactorization {
    pub shape: (usize, usize),
    pub backend_used: SparseBackend,
    pub ordering_used: PermutationOrdering,
}

pub fn spsolve(a: &CsrMatrix, b: &[f64], options: SolveOptions) -> SparseResult<SolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "spsolve requires a square matrix".to_string(),
        });
    }
    if b.len() != shape.rows {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }
    if options.check_finite
        && (a.data().iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()))
    {
        return Err(SparseError::NonFiniteInput {
            message: "matrix/rhs contains NaN or Inf".to_string(),
        });
    }

    if options.mode == RuntimeMode::Hardened && has_empty_structural_row(a) {
        return Err(SparseError::SingularMatrix {
            message: "detected empty structural row in hardened mode".to_string(),
        });
    }

    // Convert sparse CSR to dense matrix for LU solve.
    // This is correct for V1; a native sparse direct solver can replace this later.
    let n = shape.rows;
    let dense = csr_to_dense(a);
    let matrix = DMatrix::from_row_slice(n, n, &dense);
    let rhs = DVector::from_column_slice(b);
    let lu: LU<f64, Dyn, Dyn> = matrix.lu();
    let x = lu.solve(&rhs).ok_or(SparseError::SingularMatrix {
        message: "LU factorization detected singular matrix".to_string(),
    })?;

    Ok(SolveResult {
        solution: x.iter().copied().collect(),
        backend_used: SparseBackend::Auto,
        ordering_used: options.ordering,
        warnings: Vec::new(),
    })
}

pub fn splu(a: &CscMatrix, options: LuOptions) -> SparseResult<SparseLuFactorization> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "splu requires a square matrix".to_string(),
        });
    }
    if !(0.0..=1.0).contains(&options.diag_pivot_thresh) {
        return Err(SparseError::InvalidArgument {
            message: "diag_pivot_thresh must be in [0, 1]".to_string(),
        });
    }

    let n = shape.rows;
    let dense = csc_to_dense(a);
    let matrix = DMatrix::from_row_slice(n, n, &dense);
    let lu: LU<f64, Dyn, Dyn> = matrix.lu();

    Ok(SparseLuFactorization {
        shape: (n, n),
        backend_used: SparseBackend::Auto,
        ordering_used: options.ordering,
        lu_internal: lu,
    })
}

/// Solve a linear system using a precomputed sparse LU factorization.
pub fn splu_solve(factorization: &SparseLuFactorization, b: &[f64]) -> SparseResult<Vec<f64>> {
    let n = factorization.shape.0;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: format!("rhs length {} must match matrix size {}", b.len(), n),
        });
    }
    let rhs = DVector::from_column_slice(b);
    let x = factorization
        .lu_internal
        .solve(&rhs)
        .ok_or(SparseError::SingularMatrix {
            message: "LU factorization detected singular matrix".to_string(),
        })?;
    Ok(x.iter().copied().collect())
}

pub fn spilu(a: &CscMatrix, options: IluOptions) -> SparseResult<SparseIluFactorization> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "spilu requires a square matrix".to_string(),
        });
    }
    if options.drop_tol < 0.0 || options.fill_factor < 1.0 {
        return Err(SparseError::InvalidArgument {
            message: "drop_tol must be >= 0 and fill_factor must be >= 1".to_string(),
        });
    }

    Err(SparseError::Unsupported {
        feature: "spilu backend wiring skeleton landed; numeric factorization pending".to_string(),
    })
}

/// Options for iterative solvers (CG, GMRES, etc.).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterativeSolveOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub tol: f64,
    pub max_iter: Option<usize>,
}

impl Default for IterativeSolveOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            tol: 1e-5,
            max_iter: None,
        }
    }
}

/// Result from an iterative solver.
#[derive(Debug, Clone, PartialEq)]
pub struct IterativeSolveResult {
    /// Solution vector.
    pub solution: Vec<f64>,
    /// Whether the solver converged within the tolerance.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm ||b - Ax|| / ||b||.
    pub residual_norm: f64,
}

/// Conjugate Gradient solver for symmetric positive-definite sparse systems.
///
/// Solves Ax = b where A is SPD. If A is not SPD, the solver may diverge.
/// Matches `scipy.sparse.linalg.cg(A, b)`.
pub fn cg(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "CG requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }
    if options.check_finite
        && (a.data().iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()))
    {
        return Err(SparseError::NonFiniteInput {
            message: "matrix/rhs contains NaN or Inf".to_string(),
        });
    }

    let max_iter = options.max_iter.unwrap_or(n * 10);

    // Initialize x
    let mut x: Vec<f64> = match x0 {
        Some(initial) => {
            if initial.len() != n {
                return Err(SparseError::IncompatibleShape {
                    message: "initial guess length must match matrix rows".to_string(),
                });
            }
            initial.to_vec()
        }
        None => vec![0.0; n],
    };

    // Compute b_norm for relative tolerance
    let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if b_norm == 0.0 {
        // b is zero, solution is zero
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    // r = b - A*x
    let ax = csr_matvec(a, &x);
    let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
    let mut p = r.clone();
    let mut rs_old: f64 = r.iter().map(|v| v * v).sum();

    for iteration in 0..max_iter {
        let r_norm = rs_old.sqrt();
        if r_norm / b_norm < options.tol {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: true,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        let ap = csr_matvec(a, &p);
        let p_ap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();

        if p_ap.abs() < f64::EPSILON * 100.0 {
            // Near-zero denominator; matrix may not be SPD
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        let alpha = rs_old / p_ap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rs_new: f64 = r.iter().map(|v| v * v).sum();
        let beta = rs_new / rs_old;

        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        rs_old = rs_new;
    }

    let final_norm = rs_old.sqrt() / b_norm;
    Ok(IterativeSolveResult {
        solution: x,
        converged: false,
        iterations: max_iter,
        residual_norm: final_norm,
    })
}

/// Sparse CSR matrix-vector product (internal helper for iterative solvers).
fn csr_matvec(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let n = a.shape().rows;
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for idx in indptr[i]..indptr[i + 1] {
            sum += data[idx] * x[indices[idx]];
        }
        result[i] = sum;
    }
    result
}

/// GMRES (Generalized Minimal Residual) solver for general (non-symmetric) sparse systems.
///
/// Solves Ax = b for general square A using restarted GMRES with Arnoldi iteration.
/// Matches `scipy.sparse.linalg.gmres(A, b)`.
pub fn gmres(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "GMRES requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }
    if options.check_finite
        && (a.data().iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()))
    {
        return Err(SparseError::NonFiniteInput {
            message: "matrix/rhs contains NaN or Inf".to_string(),
        });
    }

    let max_iter = options.max_iter.unwrap_or(n * 10);
    let restart = n.min(30); // Krylov subspace dimension before restart

    let mut x: Vec<f64> = match x0 {
        Some(initial) => {
            if initial.len() != n {
                return Err(SparseError::IncompatibleShape {
                    message: "initial guess length must match matrix rows".to_string(),
                });
            }
            initial.to_vec()
        }
        None => vec![0.0; n],
    };

    let b_norm = vec_norm(b);
    if b_norm == 0.0 {
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    let mut total_iter = 0;

    // Outer restart loop
    for _ in 0..(max_iter / restart.max(1) + 1) {
        let (converged, iters) = gmres_inner(
            a,
            b,
            &mut x,
            b_norm,
            restart,
            options.tol,
            max_iter - total_iter,
        )?;
        total_iter += iters;

        if converged || total_iter >= max_iter {
            let ax = csr_matvec(a, &x);
            let r_norm = vec_norm_diff(&ax, b) / b_norm;
            return Ok(IterativeSolveResult {
                solution: x,
                converged,
                iterations: total_iter,
                residual_norm: r_norm,
            });
        }
    }

    let ax = csr_matvec(a, &x);
    let r_norm = vec_norm_diff(&ax, b) / b_norm;
    Ok(IterativeSolveResult {
        solution: x,
        converged: false,
        iterations: total_iter,
        residual_norm: r_norm,
    })
}

/// Inner GMRES iteration (one restart cycle).
/// Returns (converged, iterations_used).
fn gmres_inner(
    a: &CsrMatrix,
    b: &[f64],
    x: &mut [f64],
    b_norm: f64,
    restart: usize,
    tol: f64,
    max_iter: usize,
) -> SparseResult<(bool, usize)> {
    let n = x.len();
    let m = restart.min(max_iter);

    // r = b - A*x
    let ax = csr_matvec(a, x);
    let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
    let r_norm = vec_norm(&r);

    if r_norm / b_norm < tol {
        return Ok((true, 0));
    }

    // Arnoldi process with modified Gram-Schmidt
    let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    v.push(r.iter().map(|&ri| ri / r_norm).collect());

    // Upper Hessenberg matrix H (stored as (m+1) x m)
    let mut h = vec![vec![0.0; m]; m + 1];

    // Givens rotation components
    let mut cs = vec![0.0; m];
    let mut sn = vec![0.0; m];
    let mut g = vec![0.0; m + 1];
    g[0] = r_norm;

    let mut iters = 0;

    for j in 0..m {
        iters = j + 1;

        // w = A * v_j
        let w = csr_matvec(a, &v[j]);
        let mut wj = w;

        // Modified Gram-Schmidt orthogonalization
        for i in 0..=j {
            h[i][j] = dot_product(&wj, &v[i]);
            for k in 0..n {
                wj[k] -= h[i][j] * v[i][k];
            }
        }

        h[j + 1][j] = vec_norm(&wj);

        if h[j + 1][j].abs() < f64::EPSILON * 100.0 {
            // Lucky breakdown — solution is in the current Krylov subspace
            // Apply previous Givens rotations to column j
            apply_givens_to_column(&mut h, &cs, &sn, j);
            // Solve the triangular system and update x
            update_solution(x, &v, &h, &g, j + 1);
            return Ok((true, iters));
        }

        // Normalize
        let inv_h = 1.0 / h[j + 1][j];
        v.push(wj.iter().map(|&wi| wi * inv_h).collect());

        // Apply previous Givens rotations to column j of H
        apply_givens_to_column(&mut h, &cs, &sn, j);

        // Compute new Givens rotation for row j
        let (c, s) = givens_rotation(h[j][j], h[j + 1][j]);
        cs[j] = c;
        sn[j] = s;

        // Apply new rotation to H and g
        h[j][j] = c * h[j][j] + s * h[j + 1][j];
        h[j + 1][j] = 0.0;

        let g_j = g[j];
        g[j] = c * g_j;
        g[j + 1] = -s * g_j;

        let residual = g[j + 1].abs() / b_norm;
        if residual < tol {
            update_solution(x, &v, &h, &g, j + 1);
            return Ok((true, iters));
        }
    }

    // Update solution with current approximation
    update_solution(x, &v, &h, &g, m);
    Ok((false, iters))
}

/// Apply previous Givens rotations to column j of H.
fn apply_givens_to_column(h: &mut [Vec<f64>], cs: &[f64], sn: &[f64], j: usize) {
    for i in 0..j {
        let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
        h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
        h[i][j] = temp;
    }
}

/// Compute Givens rotation coefficients.
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let s = 1.0 / (1.0 + tau * tau).sqrt();
        (s * tau, s)
    } else {
        let tau = b / a;
        let c = 1.0 / (1.0 + tau * tau).sqrt();
        (c, c * tau)
    }
}

/// Solve the upper triangular system H*y = g, then update x += V*y.
fn update_solution(x: &mut [f64], v: &[Vec<f64>], h: &[Vec<f64>], g: &[f64], k: usize) {
    // Back-substitution: solve H[0..k, 0..k] y = g[0..k]
    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
        y[i] = g[i];
        for j in (i + 1)..k {
            y[i] -= h[i][j] * y[j];
        }
        if h[i][i].abs() > f64::EPSILON * 100.0 {
            y[i] /= h[i][i];
        }
    }

    // x += V * y
    for (j, &yj) in y.iter().enumerate() {
        for (i, xi) in x.iter_mut().enumerate() {
            *xi += yj * v[j][i];
        }
    }
}

/// Euclidean norm of a vector.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Euclidean norm of (a - b).
fn vec_norm_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Dot product of two vectors.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Convert a CSR matrix to dense row-major storage.
fn csr_to_dense(a: &CsrMatrix) -> Vec<f64> {
    let shape = a.shape();
    let n = shape.rows;
    let m = shape.cols;
    let mut dense = vec![0.0; n * m];
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    for i in 0..n {
        for idx in indptr[i]..indptr[i + 1] {
            dense[i * m + indices[idx]] = data[idx];
        }
    }
    dense
}

/// Convert a CSC matrix to dense row-major storage.
fn csc_to_dense(a: &CscMatrix) -> Vec<f64> {
    let shape = a.shape();
    let n = shape.rows;
    let m = shape.cols;
    let mut dense = vec![0.0; n * m];
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    for j in 0..m {
        for idx in indptr[j]..indptr[j + 1] {
            dense[indices[idx] * m + j] = data[idx];
        }
    }
    dense
}

fn has_empty_structural_row(a: &CsrMatrix) -> bool {
    let indptr = a.indptr();
    indptr.windows(2).any(|w| w[0] == w[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::{CooMatrix, Shape2D};
    use crate::ops::FormatConvertible;

    #[test]
    fn solve_options_default_matches_contract() {
        let options = SolveOptions::default();
        assert_eq!(options.mode, RuntimeMode::Strict);
        assert_eq!(options.backend, SparseBackend::Auto);
        assert_eq!(options.ordering, PermutationOrdering::Colamd);
        assert!(options.check_finite);
    }

    #[test]
    fn lu_options_default_matches_contract() {
        let options = LuOptions::default();
        assert_eq!(options.mode, RuntimeMode::Strict);
        assert_eq!(options.ordering, PermutationOrdering::Colamd);
        assert!((options.diag_pivot_thresh - 1.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn ilu_options_default_matches_contract() {
        let options = IluOptions::default();
        assert_eq!(options.mode, RuntimeMode::Strict);
        assert_eq!(options.ordering, PermutationOrdering::Colamd);
        assert!((options.drop_tol - 1e-4).abs() <= f64::EPSILON);
        assert!((options.fill_factor - 10.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn spsolve_rejects_non_square_matrix() {
        let a = non_square_csr();
        let err = spsolve(&a, &[1.0, 2.0], SolveOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn spsolve_rejects_rhs_length_mismatch() {
        let a = square_csr();
        let err = spsolve(&a, &[1.0], SolveOptions::default()).expect_err("rhs mismatch");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn spsolve_rejects_non_finite_when_enabled() {
        let a = square_csr();
        let err = spsolve(&a, &[f64::NAN, 1.0], SolveOptions::default()).expect_err("non-finite");
        assert!(matches!(err, SparseError::NonFiniteInput { .. }));
    }

    #[test]
    fn spsolve_skips_non_finite_check_when_disabled() {
        let a = square_csr();
        let options = SolveOptions {
            check_finite: false,
            ..SolveOptions::default()
        };
        // With check_finite=false, NaN is passed through to the solver
        // (the result may be NaN but we don't reject the input)
        let result = spsolve(&a, &[f64::NAN, 1.0], options);
        assert!(
            result.is_ok(),
            "NaN should not be rejected when check_finite=false"
        );
    }

    #[test]
    fn spsolve_hardened_rejects_empty_structural_row() {
        let a = csr_with_empty_row();
        let options = SolveOptions {
            mode: RuntimeMode::Hardened,
            ..SolveOptions::default()
        };
        let err = spsolve(&a, &[1.0, 0.0], options).expect_err("empty row singular");
        assert!(matches!(err, SparseError::SingularMatrix { .. }));
    }

    #[test]
    fn spsolve_strict_empty_structural_row_reaches_solver() {
        let a = csr_with_empty_row();
        let options = SolveOptions {
            mode: RuntimeMode::Strict,
            ..SolveOptions::default()
        };
        // In strict mode, empty structural row is not pre-rejected.
        // The LU solver will detect singularity.
        let err = spsolve(&a, &[1.0, 0.0], options).expect_err("singular");
        assert!(matches!(err, SparseError::SingularMatrix { .. }));
    }

    #[test]
    fn splu_rejects_non_square_matrix() {
        let a = non_square_csc();
        let err = splu(&a, LuOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn splu_rejects_invalid_diag_pivot_threshold_low() {
        let a = square_csc();
        let options = LuOptions {
            diag_pivot_thresh: -0.1,
            ..LuOptions::default()
        };
        let err = splu(&a, options).expect_err("invalid threshold");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn splu_rejects_invalid_diag_pivot_threshold_high() {
        let a = square_csc();
        let options = LuOptions {
            diag_pivot_thresh: 1.1,
            ..LuOptions::default()
        };
        let err = splu(&a, options).expect_err("invalid threshold");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn splu_valid_input_succeeds() {
        let a = square_csc();
        let result = splu(&a, LuOptions::default()).expect("splu should succeed");
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn spilu_rejects_non_square_matrix() {
        let a = non_square_csc();
        let err = spilu(&a, IluOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn spilu_rejects_negative_drop_tol() {
        let a = square_csc();
        let options = IluOptions {
            drop_tol: -1e-6,
            ..IluOptions::default()
        };
        let err = spilu(&a, options).expect_err("negative drop_tol");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn spilu_rejects_fill_factor_below_one() {
        let a = square_csc();
        let options = IluOptions {
            fill_factor: 0.9,
            ..IluOptions::default()
        };
        let err = spilu(&a, options).expect_err("fill factor");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn spilu_valid_input_still_unsupported() {
        // ILU is still a skeleton - incomplete factorization needs dedicated algorithm
        let a = square_csc();
        let err = spilu(&a, IluOptions::default()).expect_err("backend pending");
        assert!(matches!(err, SparseError::Unsupported { .. }));
    }

    #[test]
    fn has_empty_structural_row_detects_gaps() {
        let with_gap = csr_with_empty_row();
        assert!(has_empty_structural_row(&with_gap));
        let dense = square_csr();
        assert!(!has_empty_structural_row(&dense));
    }

    // ── spsolve correctness tests ─────────────────────────────────

    #[test]
    fn spsolve_identity_system() {
        let a = identity_csr(3);
        let b = vec![1.0, 2.0, 3.0];
        let result = spsolve(&a, &b, SolveOptions::default()).expect("spsolve works");
        assert_close_slice(&result.solution, &b, 1e-14);
    }

    #[test]
    fn spsolve_diagonal_system() {
        // [[2, 0], [0, 3]] x = [4, 9] => x = [2, 3]
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 3.0],
            vec![0, 1],
            vec![0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![4.0, 9.0];
        let result = spsolve(&a, &b, SolveOptions::default()).expect("spsolve works");
        assert_close_slice(&result.solution, &[2.0, 3.0], 1e-14);
    }

    #[test]
    fn spsolve_general_system() {
        // [[3, 2], [1, 2]] x = [5, 5] => x = [0, 2.5]
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![3.0, 2.0, 1.0, 2.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![5.0, 5.0];
        let result = spsolve(&a, &b, SolveOptions::default()).expect("spsolve works");
        assert_close_slice(&result.solution, &[0.0, 2.5], 1e-12);
    }

    #[test]
    fn spsolve_singular_system() {
        // [[1, 2], [2, 4]] is singular
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 2.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0];
        let err = spsolve(&a, &b, SolveOptions::default()).expect_err("singular");
        assert!(matches!(err, SparseError::SingularMatrix { .. }));
    }

    #[test]
    fn splu_solve_roundtrip() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![3.0, 2.0, 1.0, 2.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csc()
        .expect("csc");
        let factorization = splu(&a, LuOptions::default()).expect("splu works");
        let x = splu_solve(&factorization, &[5.0, 5.0]).expect("splu_solve works");
        assert_close_slice(&x, &[0.0, 2.5], 1e-12);
    }

    #[test]
    fn splu_solve_rhs_mismatch() {
        let a = square_csc();
        let factorization = splu(&a, LuOptions::default()).expect("splu works");
        let err = splu_solve(&factorization, &[1.0, 2.0, 3.0]).expect_err("mismatch");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    fn assert_close_slice(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "slice lengths differ");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index={i} actual={a} expected={e} diff={}",
                (a - e).abs()
            );
        }
    }

    fn identity_csr(n: usize) -> CsrMatrix {
        let data: Vec<f64> = vec![1.0; n];
        let indices: Vec<usize> = (0..n).collect();
        let indptr: Vec<usize> = (0..=n).collect();
        CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("identity csr")
    }

    fn square_csr() -> CsrMatrix {
        CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 3.0, 4.0],
            vec![0, 1, 1],
            vec![0, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr")
    }

    fn square_csc() -> CscMatrix {
        square_csr().to_csc().expect("csc")
    }

    fn non_square_csr() -> CsrMatrix {
        CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr")
    }

    fn non_square_csc() -> CscMatrix {
        non_square_csr().to_csc().expect("csc")
    }

    fn csr_with_empty_row() -> CsrMatrix {
        CsrMatrix::from_components(Shape2D::new(2, 2), vec![1.0], vec![0], vec![0, 1, 1], true)
            .expect("csr with empty row")
    }

    // ── CG iterative solver tests ───────────────────────────────────

    fn spd_csr_3x3() -> CsrMatrix {
        // Symmetric positive definite: [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
        CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
            vec![0, 0, 1, 1, 1, 2, 2],
            vec![0, 1, 0, 1, 2, 1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr")
    }

    #[test]
    fn cg_spd_system_converges() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let result = cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg works");
        assert!(result.converged, "CG should converge for SPD system");
        // Verify A*x ≈ b
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn cg_identity_system() {
        let a = identity_csr(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &b, 1e-10);
        assert!(
            result.iterations <= 1,
            "identity should converge in <= 1 iteration"
        );
    }

    #[test]
    fn cg_with_initial_guess() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        // Start from a good guess
        let x0 = vec![1.0, 1.0, 1.0];
        let result = cg(&a, &b, Some(&x0), IterativeSolveOptions::default()).expect("cg works");
        assert!(result.converged);
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn cg_zero_rhs() {
        let a = spd_csr_3x3();
        let b = vec![0.0, 0.0, 0.0];
        let result = cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg works");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_close_slice(&result.solution, &[0.0, 0.0, 0.0], 1e-14);
    }

    #[test]
    fn cg_rejects_non_square() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let err =
            cg(&a, &[1.0, 2.0], None, IterativeSolveOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn cg_rejects_rhs_mismatch() {
        let a = spd_csr_3x3();
        let err =
            cg(&a, &[1.0, 2.0], None, IterativeSolveOptions::default()).expect_err("mismatch");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn cg_max_iter_limits_iterations() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let options = IterativeSolveOptions {
            max_iter: Some(1),
            tol: 1e-15, // extremely tight tolerance
            ..IterativeSolveOptions::default()
        };
        let result = cg(&a, &b, None, options).expect("cg works");
        assert!(result.iterations <= 1, "should be limited to max_iter");
    }

    #[test]
    fn cg_diagonal_system() {
        // [[2, 0], [0, 5]] x = [4, 10] => x = [2, 2]
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 5.0],
            vec![0, 1],
            vec![0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![4.0, 10.0];
        let result = cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &[2.0, 2.0], 1e-10);
    }

    // ── GMRES iterative solver tests ────────────────────────────────

    fn nonsymmetric_csr_3x3() -> CsrMatrix {
        // Non-symmetric: [[4, 1, 0], [0, 3, 1], [0, 0, 2]]
        CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![4.0, 1.0, 3.0, 1.0, 2.0],
            vec![0, 0, 1, 1, 2],
            vec![0, 1, 1, 2, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr")
    }

    #[test]
    fn gmres_identity_system() {
        let a = identity_csr(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &b, 1e-10);
    }

    #[test]
    fn gmres_nonsymmetric_system() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![5.0, 7.0, 4.0];
        let result = gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged, "GMRES should converge");
        // Verify A*x ≈ b
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn gmres_diagonal_system() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![3.0, 7.0],
            vec![0, 1],
            vec![0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![9.0, 14.0];
        let result = gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &[3.0, 2.0], 1e-10);
    }

    #[test]
    fn gmres_zero_rhs() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![0.0, 0.0, 0.0];
        let result = gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_close_slice(&result.solution, &[0.0, 0.0, 0.0], 1e-14);
    }

    #[test]
    fn gmres_with_initial_guess() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![5.0, 7.0, 4.0];
        let x0 = vec![1.0, 1.0, 1.0];
        let result =
            gmres(&a, &b, Some(&x0), IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged);
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn gmres_rejects_non_square() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let err =
            gmres(&a, &[1.0, 2.0], None, IterativeSolveOptions::default()).expect_err("non-sq");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn gmres_rejects_rhs_mismatch() {
        let a = nonsymmetric_csr_3x3();
        let err =
            gmres(&a, &[1.0, 2.0], None, IterativeSolveOptions::default()).expect_err("mismatch");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn gmres_spd_system_matches_cg() {
        // GMRES should work on SPD systems too
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let cg_result = cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg works");
        let gmres_result =
            gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(gmres_result.converged);
        assert_close_slice(&gmres_result.solution, &cg_result.solution, 1e-6);
    }

    #[test]
    fn gmres_general_dense_system() {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 10]] x = [14, 32, 53] => x = [1, 2, 3]
        let a = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![14.0, 32.0, 53.0];
        let result = gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &[1.0, 2.0, 3.0], 1e-6);
    }
}
