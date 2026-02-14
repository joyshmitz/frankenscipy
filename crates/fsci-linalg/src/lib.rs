#![forbid(unsafe_code)]

use fsci_runtime::{
    MatrixConditionState, RuntimeMode, SolverAction, SolverEvidenceEntry, SolverPortfolio,
};
use nalgebra::{DMatrix, DVector, Dyn, LU, linalg::SVD};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixAssumption {
    General,
    Diagonal,
    UpperTriangular,
    LowerTriangular,
    Symmetric,
    Hermitian,
    PositiveDefinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularTranspose {
    NoTranspose,
    Transpose,
    ConjugateTranspose,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub assume_a: Option<MatrixAssumption>,
    pub lower: bool,
    pub transposed: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            assume_a: None,
            lower: false,
            transposed: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangularSolveOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub trans: TriangularTranspose,
    pub lower: bool,
    pub unit_diagonal: bool,
}

impl Default for TriangularSolveOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            trans: TriangularTranspose::NoTranspose,
            lower: false,
            unit_diagonal: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub assume_a: Option<MatrixAssumption>,
    pub lower: bool,
}

impl Default for InvOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            assume_a: None,
            lower: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LstsqOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub cond: Option<f64>,
}

impl Default for LstsqOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            cond: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinvOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub atol: Option<f64>,
    pub rtol: Option<f64>,
}

impl Default for PinvOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            atol: None,
            rtol: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinalgWarning {
    IllConditioned { reciprocal_condition: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveResult {
    pub x: Vec<f64>,
    pub warning: Option<LinalgWarning>,
    pub backward_error: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InvResult {
    pub inverse: Vec<Vec<f64>>,
    pub warning: Option<LinalgWarning>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LstsqResult {
    pub x: Vec<f64>,
    pub residuals: Vec<f64>,
    pub rank: usize,
    pub singular_values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PinvResult {
    pub pseudo_inverse: Vec<Vec<f64>>,
    pub rank: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LinalgError {
    RaggedMatrix,
    ExpectedSquareMatrix,
    IncompatibleShapes {
        a_shape: (usize, usize),
        b_len: usize,
    },
    NonFiniteInput,
    SingularMatrix,
    UnsupportedAssumption,
    InvalidBandShape {
        expected_rows: usize,
        actual_rows: usize,
    },
    InvalidPinvThreshold,
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RaggedMatrix => write!(f, "matrix rows must all have equal length"),
            Self::ExpectedSquareMatrix => write!(f, "expected square matrix"),
            Self::IncompatibleShapes { a_shape, b_len } => {
                write!(f, "incompatible shapes: a_shape={a_shape:?}, b_len={b_len}")
            }
            Self::NonFiniteInput => write!(f, "array must not contain infs or NaNs"),
            Self::SingularMatrix => write!(f, "singular matrix"),
            Self::UnsupportedAssumption => write!(f, "unsupported matrix assumption"),
            Self::InvalidBandShape {
                expected_rows,
                actual_rows,
            } => write!(
                f,
                "invalid values for the number of lower and upper diagonals: expected {expected_rows} rows in ab, got {actual_rows}"
            ),
            Self::InvalidPinvThreshold => write!(f, "atol and rtol values must be positive."),
        }
    }
}

impl std::error::Error for LinalgError {}

pub fn solve(a: &[Vec<f64>], b: &[f64], options: SolveOptions) -> Result<SolveResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if b.len() != rows {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (rows, cols),
            b_len: b.len(),
        });
    }
    validate_finite_matrix_and_vector(a, b, options.mode, options.check_finite)?;

    let matrix = if options.transposed {
        transpose(a)
    } else {
        a.to_vec()
    };

    match options.assume_a.unwrap_or(MatrixAssumption::General) {
        MatrixAssumption::General
        | MatrixAssumption::Symmetric
        | MatrixAssumption::Hermitian
        | MatrixAssumption::PositiveDefinite => solve_general(&matrix, b),
        MatrixAssumption::Diagonal => solve_diagonal(&matrix, b),
        MatrixAssumption::UpperTriangular => {
            solve_triangular_internal(&matrix, b, TriangularTranspose::NoTranspose, false, false)
        }
        MatrixAssumption::LowerTriangular => {
            solve_triangular_internal(&matrix, b, TriangularTranspose::NoTranspose, true, false)
        }
    }
}

pub fn solve_triangular(
    a: &[Vec<f64>],
    b: &[f64],
    options: TriangularSolveOptions,
) -> Result<SolveResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if b.len() != rows {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (rows, cols),
            b_len: b.len(),
        });
    }
    validate_finite_matrix_and_vector(a, b, options.mode, options.check_finite)?;
    solve_triangular_internal(a, b, options.trans, options.lower, options.unit_diagonal)
}

pub fn solve_banded(
    l_and_u: (usize, usize),
    ab: &[Vec<f64>],
    b: &[f64],
    options: SolveOptions,
) -> Result<SolveResult, LinalgError> {
    let (nlower, nupper) = l_and_u;
    let (rows, cols) = matrix_shape(ab)?;
    let expected_rows = nlower + nupper + 1;
    if rows != expected_rows {
        return Err(LinalgError::InvalidBandShape {
            expected_rows,
            actual_rows: rows,
        });
    }
    if b.len() != cols {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (cols, cols),
            b_len: b.len(),
        });
    }
    validate_finite_matrix_and_vector(ab, b, options.mode, options.check_finite)?;

    let dense = dense_from_banded(nlower, nupper, ab, cols);
    solve(
        &dense,
        b,
        SolveOptions {
            assume_a: Some(MatrixAssumption::General),
            ..options
        },
    )
}

pub fn inv(a: &[Vec<f64>], options: InvOptions) -> Result<InvResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(InvResult {
            inverse: Vec::new(),
            warning: None,
        });
    }

    match options.assume_a.unwrap_or(MatrixAssumption::General) {
        MatrixAssumption::General
        | MatrixAssumption::Symmetric
        | MatrixAssumption::Hermitian
        | MatrixAssumption::PositiveDefinite => inv_general(a, rows),
        _ => inv_column_by_column(a, rows, cols, options),
    }
}

/// Single LU factorization + solve against identity. O(n³) instead of O(n⁴).
fn inv_general(a: &[Vec<f64>], n: usize) -> Result<InvResult, LinalgError> {
    let matrix = dmatrix_from_rows(a)?;
    let lu: LU<f64, Dyn, Dyn> = matrix.lu();
    let rcond = fast_rcond_from_lu(&lu, n);

    // Reject near-singular matrices that LU may not catch exactly
    if rcond < f64::EPSILON {
        return Err(LinalgError::SingularMatrix);
    }

    let identity = DMatrix::identity(n, n);
    let inv_matrix = lu.solve(&identity).ok_or(LinalgError::SingularMatrix)?;

    Ok(InvResult {
        inverse: rows_from_dmatrix(&inv_matrix),
        warning: rcond_warning(rcond),
    })
}

/// Fallback for diagonal/triangular assumptions: solve column by column.
fn inv_column_by_column(
    a: &[Vec<f64>],
    rows: usize,
    cols: usize,
    options: InvOptions,
) -> Result<InvResult, LinalgError> {
    let mut warning = None;
    let mut inverse = vec![vec![0.0; cols]; rows];
    for col in 0..cols {
        let mut e = vec![0.0; rows];
        e[col] = 1.0;
        let solve_result = solve(
            a,
            &e,
            SolveOptions {
                mode: options.mode,
                check_finite: options.check_finite,
                assume_a: options.assume_a,
                lower: options.lower,
                transposed: false,
            },
        )?;
        if warning.is_none() {
            warning = solve_result.warning;
        }
        for (row_idx, value) in solve_result.x.iter().enumerate() {
            inverse[row_idx][col] = *value;
        }
    }
    Ok(InvResult { inverse, warning })
}

pub fn det(a: &[Vec<f64>], mode: RuntimeMode, check_finite: bool) -> Result<f64, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    validate_finite_matrix(a, mode, check_finite)?;

    if rows == 0 {
        return Ok(1.0);
    }
    let matrix = dmatrix_from_rows(a)?;
    Ok(matrix.lu().determinant())
}

pub fn lstsq(a: &[Vec<f64>], b: &[f64], options: LstsqOptions) -> Result<LstsqResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if b.len() != rows {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (rows, cols),
            b_len: b.len(),
        });
    }
    validate_finite_matrix_and_vector(a, b, options.mode, options.check_finite)?;

    let matrix = dmatrix_from_rows(a)?;
    let rhs = DVector::from_column_slice(b);
    let svd = SVD::new(matrix.clone(), true, true);
    let singular_values: Vec<f64> = svd.singular_values.iter().copied().collect();
    let max_s = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let cond = options.cond.unwrap_or(f64::EPSILON);
    let threshold = cond * max_s;
    let rank = singular_values.iter().filter(|s| **s > threshold).count();

    let pinv = pseudo_inverse_from_svd(&svd, threshold)?;
    let x = pinv * rhs.clone();
    // Only compute residuals when rows > cols AND full column rank (matches scipy behavior)
    let residuals = if rows > cols && rank == cols {
        let residual = rhs - matrix * x.clone();
        vec![residual.dot(&residual)]
    } else {
        Vec::new()
    };

    Ok(LstsqResult {
        x: x.iter().copied().collect(),
        residuals,
        rank,
        singular_values,
    })
}

pub fn pinv(a: &[Vec<f64>], options: PinvOptions) -> Result<PinvResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    let atol = options.atol.unwrap_or(0.0);
    let rtol = options
        .rtol
        .unwrap_or((rows.max(cols) as f64) * f64::EPSILON);
    if atol < 0.0 || rtol < 0.0 {
        return Err(LinalgError::InvalidPinvThreshold);
    }

    let matrix = dmatrix_from_rows(a)?;
    let svd = SVD::new(matrix, true, true);
    let singular_values = &svd.singular_values;
    let max_s = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let threshold = atol + rtol * max_s;
    let rank = singular_values.iter().filter(|s| **s > threshold).count();
    let pinv_matrix = pseudo_inverse_from_svd(&svd, threshold)?;

    Ok(PinvResult {
        pseudo_inverse: rows_from_dmatrix(&pinv_matrix),
        rank,
    })
}

/// O(n) reciprocal condition estimate from LU diagonal — conservative lower bound
/// on the true rcond. Avoids the O(n³) SVD that was previously used.
fn fast_rcond_from_lu(lu: &LU<f64, Dyn, Dyn>, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let u = lu.u();
    let mut max_diag: f64 = 0.0;
    let mut min_diag = f64::INFINITY;
    for i in 0..n {
        let d = u[(i, i)].abs();
        max_diag = max_diag.max(d);
        if d > 0.0 {
            min_diag = min_diag.min(d);
        }
    }
    if max_diag == 0.0 {
        return 0.0;
    }
    min_diag / max_diag
}

/// Classify rcond into a MatrixConditionState for CASP portfolio decisions.
fn classify_condition(rcond: f64) -> MatrixConditionState {
    if rcond > 1e-4 {
        MatrixConditionState::WellConditioned
    } else if rcond > 1e-8 {
        MatrixConditionState::ModerateCondition
    } else if rcond > 1e-14 {
        MatrixConditionState::IllConditioned
    } else {
        MatrixConditionState::NearSingular
    }
}

/// Compute backward error: ||Ax - b|| / (||A|| × ||x|| + ||b||).
/// Returns 0.0 when the denominator is zero.
fn compute_backward_error(matrix: &DMatrix<f64>, x: &DVector<f64>, rhs: &DVector<f64>) -> f64 {
    let residual = matrix * x - rhs;
    let denom = matrix.norm() * x.norm() + rhs.norm();
    if denom > 0.0 {
        residual.norm() / denom
    } else {
        0.0
    }
}

fn rcond_warning(rcond: f64) -> Option<LinalgWarning> {
    if rcond < 1e-12 {
        Some(LinalgWarning::IllConditioned {
            reciprocal_condition: rcond,
        })
    } else {
        None
    }
}

fn solve_general(a: &[Vec<f64>], b: &[f64]) -> Result<SolveResult, LinalgError> {
    let matrix = dmatrix_from_rows(a)?;
    let rhs = DVector::from_column_slice(b);
    let lu: LU<f64, Dyn, Dyn> = matrix.clone().lu();
    let x = lu.solve(&rhs).ok_or(LinalgError::SingularMatrix)?;
    let n = a.len();
    let rcond = fast_rcond_from_lu(&lu, n);
    let backward_err = compute_backward_error(&matrix, &x, &rhs);

    Ok(SolveResult {
        x: x.iter().copied().collect(),
        warning: rcond_warning(rcond),
        backward_error: Some(backward_err),
    })
}

fn solve_qr(a: &[Vec<f64>], b: &[f64]) -> Result<SolveResult, LinalgError> {
    let matrix = dmatrix_from_rows(a)?;
    let rhs = DVector::from_column_slice(b);
    let qr = matrix.clone().qr();
    let x = qr.solve(&rhs).ok_or(LinalgError::SingularMatrix)?;
    let backward_err = compute_backward_error(&matrix, &x, &rhs);

    Ok(SolveResult {
        x: x.iter().copied().collect(),
        warning: None,
        backward_error: Some(backward_err),
    })
}

fn solve_svd_fallback(a: &[Vec<f64>], b: &[f64]) -> Result<SolveResult, LinalgError> {
    let matrix = dmatrix_from_rows(a)?;
    let rhs = DVector::from_column_slice(b);
    let svd = SVD::new(matrix.clone(), true, true);
    let x = svd
        .solve(&rhs, f64::EPSILON)
        .map_err(|_| LinalgError::SingularMatrix)?;
    let max_s = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let min_s = svd
        .singular_values
        .iter()
        .copied()
        .filter(|s| *s > 0.0)
        .fold(f64::INFINITY, f64::min);
    let rcond = if min_s.is_finite() && max_s > 0.0 {
        min_s / max_s
    } else {
        0.0
    };
    let backward_err = compute_backward_error(&matrix, &x, &rhs);

    Ok(SolveResult {
        x: x.iter().copied().collect(),
        warning: rcond_warning(rcond),
        backward_error: Some(backward_err),
    })
}

/// Solve with CASP: condition-aware solver portfolio selects optimal solver.
/// The portfolio is updated with evidence from this solve call.
pub fn solve_with_casp(
    a: &[Vec<f64>],
    b: &[f64],
    options: SolveOptions,
    portfolio: &mut SolverPortfolio,
) -> Result<SolveResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if b.len() != rows {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (rows, cols),
            b_len: b.len(),
        });
    }
    validate_finite_matrix_and_vector(a, b, options.mode, options.check_finite)?;

    let effective_a = if options.transposed {
        transpose(a)
    } else {
        a.to_vec()
    };

    // Quick LU for condition estimation (O(n³) but needed anyway for LU path)
    let matrix = dmatrix_from_rows(&effective_a)?;
    let lu: LU<f64, Dyn, Dyn> = matrix.clone().lu();
    let rcond = fast_rcond_from_lu(&lu, rows);
    let condition_state = classify_condition(rcond);

    // Query portfolio for optimal action via expected-loss minimization
    let (action, posterior, expected_losses, chosen_loss) =
        portfolio.select_action(&condition_state);

    // Dispatch to chosen solver
    let result = match action {
        SolverAction::DirectLU => {
            let rhs = DVector::from_column_slice(b);
            let x = lu.solve(&rhs).ok_or(LinalgError::SingularMatrix)?;
            let backward_err = compute_backward_error(&matrix, &x, &rhs);
            Ok(SolveResult {
                x: x.iter().copied().collect(),
                warning: rcond_warning(rcond),
                backward_error: Some(backward_err),
            })
        }
        SolverAction::PivotedQR => solve_qr(&effective_a, b),
        SolverAction::SVDFallback => solve_svd_fallback(&effective_a, b),
        SolverAction::DiagonalFastPath => solve_diagonal(&effective_a, b),
        SolverAction::TriangularFastPath => solve_triangular_internal(
            &effective_a,
            b,
            TriangularTranspose::NoTranspose,
            false,
            false,
        ),
    };

    // Record evidence regardless of outcome
    let fallback_active = !matches!(action, SolverAction::DirectLU);
    portfolio.record_evidence(SolverEvidenceEntry {
        component: "solver_portfolio",
        matrix_shape: (rows, cols),
        rcond_estimate: rcond,
        chosen_action: action,
        posterior: posterior.to_vec(),
        expected_losses: expected_losses.to_vec(),
        chosen_expected_loss: chosen_loss,
        fallback_active,
    });

    result
}

/// Estimate spectral condition number via power iteration.
/// Cost: O(n² × iterations) vs O(n³) for full SVD.
/// Returns the reciprocal condition number (σ_min / σ_max).
pub fn randomized_rcond_estimate(
    matrix: &DMatrix<f64>,
    lu: &LU<f64, Dyn, Dyn>,
    iterations: usize,
) -> f64 {
    let n = matrix.nrows();
    if n == 0 {
        return 1.0;
    }

    // Power iteration for largest singular value: converge on σ_max
    let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
    for _ in 0..iterations {
        let av = matrix * &v;
        v = matrix.transpose() * av;
        let norm = v.norm();
        if norm == 0.0 {
            return 0.0;
        }
        v /= norm;
    }
    let sigma_max = (matrix * &v).norm();
    if sigma_max == 0.0 {
        return 0.0;
    }

    // Inverse power iteration for smallest singular value using LU
    let mut w = DVector::from_element(n, 1.0 / (n as f64).sqrt());
    for _ in 0..iterations {
        match lu.solve(&w) {
            Some(solved) => {
                let norm = solved.norm();
                if norm == 0.0 {
                    return 0.0;
                }
                w = solved / norm;
            }
            None => return 0.0,
        }
    }
    let sigma_min_inv = lu.solve(&w).map_or(f64::INFINITY, |s| s.norm());
    if sigma_min_inv == 0.0 || !sigma_min_inv.is_finite() {
        return 0.0;
    }
    let sigma_min = 1.0 / sigma_min_inv;

    sigma_min / sigma_max
}

fn solve_diagonal(a: &[Vec<f64>], b: &[f64]) -> Result<SolveResult, LinalgError> {
    let n = a.len();
    let mut x = vec![0.0; n];
    let mut max_diag: f64 = 0.0;
    let mut min_diag = f64::INFINITY;
    for i in 0..n {
        let diag = a[i][i];
        if diag == 0.0 {
            return Err(LinalgError::SingularMatrix);
        }
        let abs_diag = diag.abs();
        max_diag = max_diag.max(abs_diag);
        min_diag = min_diag.min(abs_diag);
        x[i] = b[i] / diag;
    }
    let warning = if max_diag > 0.0 {
        let rcond = min_diag / max_diag;
        if rcond < 1e-12 {
            Some(LinalgWarning::IllConditioned {
                reciprocal_condition: rcond,
            })
        } else {
            None
        }
    } else {
        None
    };

    Ok(SolveResult {
        x,
        warning,
        backward_error: None,
    })
}

fn solve_triangular_internal(
    a: &[Vec<f64>],
    b: &[f64],
    trans: TriangularTranspose,
    lower: bool,
    unit_diagonal: bool,
) -> Result<SolveResult, LinalgError> {
    let n = a.len();
    let (mat, is_lower) = match trans {
        TriangularTranspose::NoTranspose => (a.to_vec(), lower),
        TriangularTranspose::Transpose | TriangularTranspose::ConjugateTranspose => {
            (transpose(a), !lower)
        }
    };

    let mut x = vec![0.0; n];
    if is_lower {
        for i in 0..n {
            let mut sum = 0.0;
            for (j, xj) in x.iter().enumerate().take(i) {
                sum += mat[i][j] * *xj;
            }
            let diag = if unit_diagonal { 1.0 } else { mat[i][i] };
            if diag == 0.0 {
                return Err(LinalgError::SingularMatrix);
            }
            x[i] = (b[i] - sum) / diag;
        }
    } else {
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for (j, xj) in x.iter().enumerate().skip(i + 1) {
                sum += mat[i][j] * *xj;
            }
            let diag = if unit_diagonal { 1.0 } else { mat[i][i] };
            if diag == 0.0 {
                return Err(LinalgError::SingularMatrix);
            }
            x[i] = (b[i] - sum) / diag;
        }
    }

    Ok(SolveResult {
        x,
        warning: None,
        backward_error: None,
    })
}

fn dense_from_banded(nlower: usize, nupper: usize, ab: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0; n]; n];
    for j in 0..n {
        let i_start = j.saturating_sub(nupper);
        let i_end = (j + nlower).min(n.saturating_sub(1));
        for (i, row) in dense.iter_mut().enumerate().take(i_end + 1).skip(i_start) {
            let band_row = nupper + i - j;
            row[j] = ab[band_row][j];
        }
    }
    dense
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let rows = a.len();
    let cols = a[0].len();
    let mut out = vec![vec![0.0; rows]; cols];
    for (r, row) in a.iter().enumerate().take(rows) {
        for (c, value) in row.iter().enumerate().take(cols) {
            out[c][r] = *value;
        }
    }
    out
}

fn matrix_shape(a: &[Vec<f64>]) -> Result<(usize, usize), LinalgError> {
    if a.is_empty() {
        return Ok((0, 0));
    }
    let cols = a[0].len();
    if a.iter().any(|row| row.len() != cols) {
        return Err(LinalgError::RaggedMatrix);
    }
    Ok((a.len(), cols))
}

fn validate_finite_matrix(
    a: &[Vec<f64>],
    mode: RuntimeMode,
    check_finite: bool,
) -> Result<(), LinalgError> {
    let must_check = check_finite || mode == RuntimeMode::Hardened;
    if must_check && a.iter().flatten().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFiniteInput);
    }
    Ok(())
}

fn validate_finite_matrix_and_vector(
    a: &[Vec<f64>],
    b: &[f64],
    mode: RuntimeMode,
    check_finite: bool,
) -> Result<(), LinalgError> {
    validate_finite_matrix(a, mode, check_finite)?;
    let must_check = check_finite || mode == RuntimeMode::Hardened;
    if must_check && b.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFiniteInput);
    }
    Ok(())
}

fn dmatrix_from_rows(rows: &[Vec<f64>]) -> Result<DMatrix<f64>, LinalgError> {
    let (m, n) = matrix_shape(rows)?;
    let mut data = Vec::with_capacity(m * n);
    for row in rows {
        data.extend_from_slice(row);
    }
    Ok(DMatrix::from_row_slice(m, n, &data))
}

fn rows_from_dmatrix(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; m.ncols()]; m.nrows()];
    for r in 0..m.nrows() {
        for c in 0..m.ncols() {
            out[r][c] = m[(r, c)];
        }
    }
    out
}

fn pseudo_inverse_from_svd(
    svd: &SVD<f64, Dyn, Dyn>,
    threshold: f64,
) -> Result<DMatrix<f64>, LinalgError> {
    let u = svd.u.as_ref().ok_or(LinalgError::UnsupportedAssumption)?;
    let v_t = svd.v_t.as_ref().ok_or(LinalgError::UnsupportedAssumption)?;
    let p = svd.singular_values.len();
    if u.ncols() != p || v_t.nrows() != p {
        return Err(LinalgError::UnsupportedAssumption);
    }

    let mut sigma_pinv = DMatrix::zeros(p, p);
    for (i, s) in svd.singular_values.iter().enumerate() {
        if *s > threshold {
            sigma_pinv[(i, i)] = 1.0 / *s;
        }
    }
    Ok(v_t.transpose() * sigma_pinv * u.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close_slice(actual: &[f64], expected: &[f64], atol: f64, rtol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let tol = atol + rtol * e.abs();
            assert!(
                (a - e).abs() <= tol,
                "index={idx} expected={e} actual={a} tol={tol}"
            );
        }
    }

    fn assert_close_matrix(actual: &[Vec<f64>], expected: &[Vec<f64>], atol: f64, rtol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (row_idx, (a_row, e_row)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a_row.len(), e_row.len());
            for (col_idx, (a, e)) in a_row.iter().zip(e_row.iter()).enumerate() {
                let tol = atol + rtol * e.abs();
                assert!(
                    (a - e).abs() <= tol,
                    "row={row_idx} col={col_idx} expected={e} actual={a} tol={tol}"
                );
            }
        }
    }

    #[test]
    fn solve_general_happy_path() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let result = solve(&a, &b, SolveOptions::default()).expect("solve must work");
        assert_close_slice(&result.x, &[0.0, 2.5], 1e-12, 1e-12);
    }

    #[test]
    fn solve_triangular_lower_path() {
        let a = vec![vec![2.0, 0.0], vec![3.0, 4.0]];
        let b = vec![4.0, 11.0];
        let result = solve_triangular(
            &a,
            &b,
            TriangularSolveOptions {
                lower: true,
                ..TriangularSolveOptions::default()
            },
        )
        .expect("triangular solve must work");
        assert_close_slice(&result.x, &[2.0, 1.25], 1e-12, 1e-12);
    }

    #[test]
    fn solve_banded_matches_dense() {
        let ab = vec![
            vec![0.0, 0.0, -1.0, -1.0, -1.0],
            vec![0.0, 2.0, 2.0, 2.0, 2.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 0.0],
        ];
        let b = vec![0.0, 1.0, 2.0, 2.0, 3.0];
        let result =
            solve_banded((1, 2), &ab, &b, SolveOptions::default()).expect("banded solve must work");
        assert_close_slice(
            &result.x,
            &[-2.37288136, 3.93220339, -4.0, 4.3559322, -1.3559322],
            1e-8,
            1e-8,
        );
    }

    #[test]
    fn inverse_matches_known_result() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = inv(&a, InvOptions::default()).expect("inverse must work");
        assert_close_matrix(
            &result.inverse,
            &[vec![-2.0, 1.0], vec![1.5, -0.5]],
            1e-12,
            1e-12,
        );
    }

    #[test]
    fn determinant_matches_known_result() {
        let a = vec![
            vec![0.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let value = det(&a, RuntimeMode::Strict, true).expect("det should work");
        assert!((value - 3.0).abs() < 1e-12);
    }

    #[test]
    fn lstsq_overdetermined_system() {
        let a = vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
        ];
        let b = vec![1.0, 2.0, 2.0, 4.0];
        let result = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq must work");
        assert_eq!(result.rank, 2);
        assert_close_slice(&result.x, &[0.9, 0.9], 1e-10, 1e-10);
        assert_eq!(result.residuals.len(), 1);
    }

    #[test]
    fn pinv_rectangular_and_rank() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
        let result = pinv(&a, PinvOptions::default()).expect("pinv should work");
        assert_eq!(result.rank, 2);
        assert_close_matrix(
            &result.pseudo_inverse,
            &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            1e-12,
            1e-12,
        );
    }

    #[test]
    fn pinv_negative_thresholds_are_rejected() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let err = pinv(
            &a,
            PinvOptions {
                atol: Some(-1.0),
                ..PinvOptions::default()
            },
        )
        .expect_err("negative threshold must fail");
        assert_eq!(err, LinalgError::InvalidPinvThreshold);
    }

    #[test]
    fn hardened_mode_forces_finite_checks() {
        let a = vec![vec![1.0, f64::NAN], vec![0.0, 1.0]];
        let b = vec![1.0, 2.0];
        let strict = solve(
            &a,
            &b,
            SolveOptions {
                mode: RuntimeMode::Strict,
                check_finite: false,
                ..SolveOptions::default()
            },
        );
        assert!(
            strict.is_ok(),
            "strict mode with disabled finite checks should proceed"
        );

        let hardened = solve(
            &a,
            &b,
            SolveOptions {
                mode: RuntimeMode::Hardened,
                check_finite: false,
                ..SolveOptions::default()
            },
        );
        assert_eq!(
            hardened.expect_err("hardened mode should fail-closed"),
            LinalgError::NonFiniteInput
        );
    }

    #[test]
    fn solve_general_backward_error_is_small() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let result = solve(&a, &b, SolveOptions::default()).expect("solve must work");
        assert!(
            result.backward_error.unwrap() < 1e-14,
            "backward error should be near machine epsilon"
        );
    }

    #[test]
    fn inv_general_single_lu_matches_column_by_column() {
        let a = vec![
            vec![4.0, 7.0, 2.0],
            vec![3.0, 6.0, 1.0],
            vec![2.0, 5.0, 3.0],
        ];
        let result = inv(&a, InvOptions::default()).expect("inv must work");
        // Verify A * A^{-1} ≈ I
        for (i, a_row) in a.iter().enumerate() {
            for j in 0..a.len() {
                let sum: f64 = a_row
                    .iter()
                    .enumerate()
                    .map(|(k, &a_ik)| a_ik * result.inverse[k][j])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "A * inv(A) should be identity: [{i}][{j}] = {sum}"
                );
            }
        }
    }

    #[test]
    fn fast_rcond_from_lu_well_conditioned() {
        let matrix = DMatrix::from_row_slice(2, 2, &[3.0, 2.0, 1.0, 2.0]);
        let lu = matrix.lu();
        let rcond = fast_rcond_from_lu(&lu, 2);
        assert!(
            rcond > 0.1,
            "well-conditioned matrix should have high rcond, got {rcond}"
        );
    }

    #[test]
    fn fast_rcond_from_lu_ill_conditioned() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1e-15]);
        let lu = matrix.lu();
        let rcond = fast_rcond_from_lu(&lu, 2);
        assert!(
            rcond < 1e-12,
            "ill-conditioned matrix should have low rcond, got {rcond}"
        );
    }

    #[test]
    fn solve_with_casp_well_conditioned_uses_lu() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let result = solve_with_casp(&a, &b, SolveOptions::default(), &mut portfolio)
            .expect("solve must work");
        assert_close_slice(&result.x, &[0.0, 2.5], 1e-12, 1e-12);
        assert_eq!(portfolio.evidence_len(), 1);
    }

    #[test]
    fn solve_qr_matches_lu() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let lu_result = solve_general(&a, &b).expect("LU solve works");
        let qr_result = solve_qr(&a, &b).expect("QR solve works");
        assert_close_slice(&qr_result.x, &lu_result.x, 1e-12, 1e-12);
    }

    #[test]
    fn solve_svd_fallback_matches_lu() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let lu_result = solve_general(&a, &b).expect("LU solve works");
        let svd_result = solve_svd_fallback(&a, &b).expect("SVD solve works");
        assert_close_slice(&svd_result.x, &lu_result.x, 1e-12, 1e-12);
    }

    #[test]
    fn randomized_rcond_estimate_well_conditioned() {
        let matrix = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let lu = matrix.clone().lu();
        let rcond = randomized_rcond_estimate(&matrix, &lu, 5);
        assert!(
            rcond > 0.1,
            "well-conditioned matrix rcond estimate should be high, got {rcond}"
        );
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_invertible_2x2() -> impl Strategy<Value = Vec<Vec<f64>>> {
        (
            -10.0..10.0_f64,
            -10.0..10.0_f64,
            -10.0..10.0_f64,
            -10.0..10.0_f64,
        )
            .prop_filter("non-singular", |(a, b, c, d)| (a * d - b * c).abs() > 0.01)
            .prop_map(|(a, b, c, d)| vec![vec![a, b], vec![c, d]])
    }

    fn arb_vec2() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(-10.0..10.0_f64, 2..=2)
    }

    proptest! {
        #[test]
        fn roundtrip_solve(a in arb_invertible_2x2(), b in arb_vec2()) {
            let result = solve(&a, &b, SolveOptions::default()).expect("solve works");
            // Verify A * x ≈ b
            for (i, (a_row, &b_i)) in a.iter().zip(b.iter()).enumerate() {
                let row_sum: f64 = a_row.iter().zip(result.x.iter()).map(|(a_ij, x_j)| a_ij * x_j).sum();
                prop_assert!((row_sum - b_i).abs() < 1e-8,
                    "A*x should equal b: row {i}: {row_sum} vs {b_i}");
            }
        }

        #[test]
        fn inverse_roundtrip(a in arb_invertible_2x2()) {
            let inv_result = inv(&a, InvOptions::default()).expect("inv works");
            // Verify A * inv(A) ≈ I
            for (i, a_row) in a.iter().enumerate() {
                for j in 0..a.len() {
                    let sum: f64 = a_row.iter().enumerate()
                        .map(|(k, &a_ik)| a_ik * inv_result.inverse[k][j])
                        .sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    prop_assert!((sum - expected).abs() < 1e-8,
                        "A * inv(A) should be I: [{i}][{j}] = {sum}");
                }
            }
        }

        #[test]
        fn det_inverse_consistency(a in arb_invertible_2x2()) {
            let d = det(&a, RuntimeMode::Strict, true).expect("det works");
            let inv_result = inv(&a, InvOptions::default()).expect("inv works");
            let d_inv = det(&inv_result.inverse, RuntimeMode::Strict, true).expect("det(inv) works");
            // det(A) * det(inv(A)) ≈ 1
            prop_assert!((d * d_inv - 1.0).abs() < 1e-6,
                "det(A) * det(inv(A)) should be 1: {} * {} = {}", d, d_inv, d * d_inv);
        }
    }
}
