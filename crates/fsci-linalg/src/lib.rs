#![forbid(unsafe_code)]

use fsci_runtime::{
    RuntimeMode, SolverAction, SolverEvidenceEntry, SolverPortfolio, StructuralEvidence,
};
use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, Dyn, LU, linalg::SVD};
use serde::Serialize;

/// Hardened-mode reciprocal condition threshold: matrices with rcond below this
/// are rejected as too ill-conditioned for reliable computation.
const HARDENED_RCOND_THRESHOLD: f64 = 1e-14;

/// Hardened-mode maximum matrix dimension. Prevents resource exhaustion.
const HARDENED_MAX_DIM: usize = 10_000;

/// Structured audit log entry emitted by every linalg operation.
#[derive(Debug, Clone, Serialize)]
pub struct LinalgTrace {
    pub operation: &'static str,
    pub matrix_size: (usize, usize),
    pub mode: RuntimeMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rcond: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixAssumption {
    General,
    Diagonal,
    UpperTriangular,
    LowerTriangular,
    Symmetric,
    Hermitian,
    PositiveDefinite,
    Banded,
    TriDiagonal,
}

/// LAPACK driver selection for least-squares problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LstsqDriver {
    /// SVD with divide-and-conquer (default, most robust).
    #[default]
    Gelsd,
    /// QR with column pivoting (fastest for well-conditioned).
    Gelsy,
    /// SVD (older, slower than Gelsd).
    Gelss,
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
    pub driver: LstsqDriver,
}

impl Default for LstsqOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            cond: None,
            driver: LstsqDriver::default(),
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

/// Result of LU decomposition with partial pivoting.
#[derive(Debug, Clone, PartialEq)]
pub struct LuResult {
    /// Permutation matrix P (row-major).
    pub p: Vec<Vec<f64>>,
    /// Lower triangular factor L (unit diagonal).
    pub l: Vec<Vec<f64>>,
    /// Upper triangular factor U.
    pub u: Vec<Vec<f64>>,
}

/// Compact LU factorization for use with `lu_solve`.
#[derive(Debug, Clone)]
pub struct LuFactorResult {
    /// The internal nalgebra LU object.
    lu_internal: LU<f64, Dyn, Dyn>,
    /// Matrix dimension.
    n: usize,
    /// 1-norm of the original matrix.
    a_norm_1: f64,
}

/// Result of QR decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct QrResult {
    /// Orthogonal factor Q (m×m or m×min(m,n) for economy mode).
    pub q: Vec<Vec<f64>>,
    /// Upper triangular factor R.
    pub r: Vec<Vec<f64>>,
}

/// Result of SVD decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct SvdResult {
    /// Left singular vectors U (m×k where k=min(m,n)).
    pub u: Vec<Vec<f64>>,
    /// Singular values σ in descending order.
    pub s: Vec<f64>,
    /// Right singular vectors Vᵀ (k×n where k=min(m,n)).
    pub vt: Vec<Vec<f64>>,
}

/// Result of Cholesky decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct CholeskyResult {
    /// Lower triangular factor L such that A = LLᵀ (or upper if requested).
    pub factor: Vec<Vec<f64>>,
}

/// Result of LDL decomposition: A = L * D * Lᵀ.
#[derive(Debug, Clone, PartialEq)]
pub struct LdlResult {
    /// Unit lower triangular factor L.
    pub l: Vec<Vec<f64>>,
    /// Diagonal entries of D.
    pub d: Vec<f64>,
}

/// Compact Cholesky factorization for use with `cho_solve`.
#[derive(Debug, Clone)]
pub struct ChoFactorResult {
    /// The internal nalgebra Cholesky object.
    chol_internal: Cholesky<f64, Dyn>,
    /// Matrix dimension.
    n: usize,
}

impl ChoFactorResult {
    /// Matrix dimension of the factored matrix.
    pub fn dimension(&self) -> usize {
        self.n
    }
}

/// Result of eigenvalue decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct EigResult {
    /// Eigenvalues as (real, imaginary) pairs.
    pub eigenvalues_re: Vec<f64>,
    pub eigenvalues_im: Vec<f64>,
    /// Right eigenvectors as columns (row-major storage).
    pub eigenvectors: Vec<Vec<f64>>,
}

/// Result of symmetric eigenvalue decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct EighResult {
    /// Real eigenvalues in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Orthogonal eigenvectors as columns (row-major storage).
    pub eigenvectors: Vec<Vec<f64>>,
}

/// Result of Schur decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct SchurResult {
    /// Unitary/orthogonal matrix Z (Schur vectors).
    pub z: Vec<Vec<f64>>,
    /// Upper quasi-triangular Schur form T (real Schur form).
    pub t: Vec<Vec<f64>>,
}

/// Result of Hessenberg decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct HessenbergResult {
    /// Orthogonal matrix Q.
    pub q: Vec<Vec<f64>>,
    /// Upper Hessenberg matrix H.
    pub h: Vec<Vec<f64>>,
}

/// Result of polar decomposition: A = U * P.
#[derive(Debug, Clone, PartialEq)]
pub struct PolarResult {
    /// Unitary (or semi-unitary) factor U.
    pub u: Vec<Vec<f64>>,
    /// Positive semi-definite factor P.
    pub p: Vec<Vec<f64>>,
}

/// Options for decomposition operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecompOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
}

impl Default for DecompOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
        }
    }
}

/// Norm type selection matching SciPy conventions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormKind {
    /// Frobenius norm (matrix) or L2 norm (vector).
    Fro,
    /// Spectral norm (largest singular value for matrix, max abs for vector).
    Spectral,
    /// 1-norm (max column sum for matrix, sum of abs for vector).
    One,
    /// Infinity norm (max row sum for matrix, max abs for vector).
    Inf,
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
    /// Complex transposed solve not supported (matches SciPy NotImplementedError).
    NotSupported {
        detail: String,
    },
    /// SVD or eigenvalue convergence failure.
    ConvergenceFailure {
        detail: String,
    },
    /// Hardened mode: condition number exceeds threshold.
    ConditionTooHigh {
        rcond: f64,
        threshold: f64,
    },
    /// Hardened mode: matrix dimension exceeds resource limit.
    ResourceExhausted {
        detail: String,
    },
    /// Generic invalid argument.
    InvalidArgument {
        detail: String,
    },
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
            Self::NotSupported { detail } => write!(f, "{detail}"),
            Self::ConvergenceFailure { detail } => write!(f, "{detail}"),
            Self::ConditionTooHigh { rcond, threshold } => {
                write!(
                    f,
                    "matrix condition number 1/{rcond:.2e} exceeds hardened threshold 1/{threshold:.2e}"
                )
            }
            Self::ResourceExhausted { detail } => write!(f, "resource exhausted: {detail}"),
            Self::InvalidArgument { detail } => write!(f, "{detail}"),
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
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix_and_vector(a, b, options.mode, options.check_finite)?;

    let matrix = if options.transposed {
        transpose(a)
    } else {
        a.to_vec()
    };

    let result = match options.assume_a.unwrap_or(MatrixAssumption::General) {
        MatrixAssumption::General
        | MatrixAssumption::Symmetric
        | MatrixAssumption::Hermitian
        | MatrixAssumption::PositiveDefinite => {
            solve_general_with_hardening(&matrix, b, options.mode)
        }
        MatrixAssumption::Diagonal => solve_diagonal(&matrix, b),
        MatrixAssumption::UpperTriangular => {
            let lower = options.transposed; // Upper^T is Lower
            solve_triangular_internal(&matrix, b, TriangularTranspose::NoTranspose, lower, false)
        }
        MatrixAssumption::LowerTriangular => {
            let lower = !options.transposed; // Lower^T is Upper
            solve_triangular_internal(&matrix, b, TriangularTranspose::NoTranspose, lower, false)
        }
        MatrixAssumption::Banded | MatrixAssumption::TriDiagonal => {
            // Banded/tridiagonal structure in solve() falls back to general LU.
            // Use solve_banded() for explicit banded storage format.
            solve_general_with_hardening(&matrix, b, options.mode)
        }
    };

    emit_trace(LinalgTrace {
        operation: "solve",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: result.as_ref().ok().and_then(|r| {
            r.warning.as_ref().map(|w| match w {
                LinalgWarning::IllConditioned {
                    reciprocal_condition,
                } => *reciprocal_condition,
            })
        }),
        warning: result
            .as_ref()
            .ok()
            .and_then(|r| r.warning.as_ref().map(|w| format!("{w:?}"))),
        error: result.as_ref().err().map(|e| e.to_string()),
    });

    result
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
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix_and_vector(a, b, options.mode, options.check_finite)?;
    let result =
        solve_triangular_internal(a, b, options.trans, options.lower, options.unit_diagonal);
    emit_trace(LinalgTrace {
        operation: "solve_triangular",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: result.as_ref().err().map(|e| e.to_string()),
    });
    result
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
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(InvResult {
            inverse: Vec::new(),
            warning: None,
        });
    }

    let result = match options.assume_a.unwrap_or(MatrixAssumption::General) {
        MatrixAssumption::General
        | MatrixAssumption::Symmetric
        | MatrixAssumption::Hermitian
        | MatrixAssumption::PositiveDefinite => inv_general(a, rows),
        _ => inv_column_by_column(a, rows, cols, options),
    };

    emit_trace(LinalgTrace {
        operation: "inv",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: result.as_ref().ok().and_then(|r| {
            r.warning.as_ref().map(|w| match w {
                LinalgWarning::IllConditioned {
                    reciprocal_condition,
                } => *reciprocal_condition,
            })
        }),
        warning: result
            .as_ref()
            .ok()
            .and_then(|r| r.warning.as_ref().map(|w| format!("{w:?}"))),
        error: result.as_ref().err().map(|e| e.to_string()),
    });

    result
}

/// Single LU factorization + solve against identity. O(n³) instead of O(n⁴).
fn inv_general(a: &[Vec<f64>], n: usize) -> Result<InvResult, LinalgError> {
    let matrix = dmatrix_from_rows(a)?;
    let a_norm_1 = matrix.lp_norm(1);
    let lu: LU<f64, Dyn, Dyn> = matrix.lu();
    let rcond = fast_rcond_from_lu(&lu, a_norm_1, n);

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
    hardened_dimension_check(mode, rows, cols)?;
    validate_finite_matrix(a, mode, check_finite)?;

    if rows == 0 {
        return Ok(1.0);
    }
    let matrix = dmatrix_from_rows(a)?;
    let result = matrix.lu().determinant();
    emit_trace(LinalgTrace {
        operation: "det",
        matrix_size: (rows, cols),
        mode,
        rcond: None,
        warning: None,
        error: None,
    });
    Ok(result)
}

pub fn lstsq(a: &[Vec<f64>], b: &[f64], options: LstsqOptions) -> Result<LstsqResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if b.len() != rows {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (rows, cols),
            b_len: b.len(),
        });
    }
    hardened_dimension_check(options.mode, rows, cols)?;
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

    emit_trace(LinalgTrace {
        operation: "lstsq",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: Some(if max_s > 0.0 {
            singular_values.last().copied().unwrap_or(0.0) / max_s
        } else {
            0.0
        }),
        warning: None,
        error: None,
    });

    Ok(LstsqResult {
        x: x.iter().copied().collect(),
        residuals,
        rank,
        singular_values,
    })
}

pub fn pinv(a: &[Vec<f64>], options: PinvOptions) -> Result<PinvResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    hardened_dimension_check(options.mode, rows, cols)?;
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

    emit_trace(LinalgTrace {
        operation: "pinv",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(PinvResult {
        pseudo_inverse: rows_from_dmatrix(&pinv_matrix),
        rank,
    })
}

/// Solve Aᵀ x = b using LU factorization PA = LU => Aᵀ = Uᵀ Lᵀ P.
fn solve_lu_transpose(lu: &LU<f64, Dyn, Dyn>, b: &DVector<f64>) -> Option<DVector<f64>> {
    // Aᵀ = Uᵀ Lᵀ P
    // Uᵀ Lᵀ P x = b
    // 1. Solve Uᵀ y = b (lower triangular)
    let u_t = lu.u().transpose();
    let y = u_t.solve_lower_triangular(b)?;
    // 2. Solve Lᵀ z = y (upper triangular)
    let l_t = lu.l().transpose();
    let z = l_t.solve_upper_triangular(&y)?;
    // 3. P x = z => x = Pᵀ z
    let mut x = z;
    lu.p().inv_permute_rows(&mut x);
    Some(x)
}

/// O(n²) reciprocal condition estimate from LU — 1-norm Higham estimator.
/// Cost: 2 solves (O(n²)) vs O(n³) for full SVD.
fn fast_rcond_from_lu(lu: &LU<f64, Dyn, Dyn>, a_norm: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if a_norm == 0.0 {
        return 0.0;
    }

    // Estimate ||A⁻¹||₁ using Higham's iterative algorithm (up to 5 iterations)
    let mut x = DVector::from_element(n, 1.0 / (n as f64));
    let mut inv_a_norm = 0.0;

    for _ in 0..5 {
        let x_old = x.clone();
        // 1. Solve Aᵀ w = sign(x)
        let sign_x = x.map(|val| val.signum());
        let w = match solve_lu_transpose(lu, &sign_x) {
            Some(w) => w,
            None => return 0.0,
        };

        // 2. Solve A x_new = sign(w)
        let sign_w = w.map(|val| val.signum());
        let x_new = match lu.solve(&sign_w) {
            Some(x) => x,
            None => return 0.0,
        };

        let new_norm = x_new.lp_norm(1);
        if (new_norm - inv_a_norm).abs() <= 1e-10 * new_norm {
            inv_a_norm = new_norm;
            break;
        }
        inv_a_norm = new_norm;
        x = x_new;

        // Check if we are oscillating or converged in direction
        if (x.clone() - x_old).lp_norm(1) <= f64::EPSILON * x.lp_norm(1) {
            break;
        }
    }

    if inv_a_norm <= 0.0 {
        return 0.0;
    }

    let rcond = 1.0 / (a_norm * inv_a_norm);
    rcond.min(1.0)
}

/// Map linalg assumption to runtime structural evidence for CASP.
fn assumption_to_evidence(a: MatrixAssumption) -> fsci_runtime::StructuralEvidence {
    match a {
        MatrixAssumption::Diagonal => fsci_runtime::StructuralEvidence::Diagonal,
        MatrixAssumption::UpperTriangular | MatrixAssumption::LowerTriangular => {
            fsci_runtime::StructuralEvidence::Triangular
        }
        _ => fsci_runtime::StructuralEvidence::General,
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

/// Emit a structured JSON log entry to stderr for audit trail compatibility.
fn emit_trace(trace: LinalgTrace) {
    if let Ok(json) = serde_json::to_string(&trace) {
        eprintln!("{json}");
    }
}

/// Hardened-mode dimension guard: reject matrices exceeding resource limits.
fn hardened_dimension_check(
    mode: RuntimeMode,
    rows: usize,
    cols: usize,
) -> Result<(), LinalgError> {
    if mode == RuntimeMode::Hardened && (rows > HARDENED_MAX_DIM || cols > HARDENED_MAX_DIM) {
        return Err(LinalgError::ResourceExhausted {
            detail: format!(
                "matrix dimension ({rows}x{cols}) exceeds hardened limit ({HARDENED_MAX_DIM})"
            ),
        });
    }
    Ok(())
}

/// General solver with hardened-mode condition number rejection.
fn solve_general_with_hardening(
    a: &[Vec<f64>],
    b: &[f64],
    mode: RuntimeMode,
) -> Result<SolveResult, LinalgError> {
    let matrix = dmatrix_from_rows(a)?;
    let rhs = DVector::from_column_slice(b);
    let lu: LU<f64, Dyn, Dyn> = matrix.clone().lu();
    let n = a.len();
    let rcond = fast_rcond_from_lu(&lu, matrix.lp_norm(1), n);

    // Hardened mode: reject if condition number is too high
    if mode == RuntimeMode::Hardened && rcond < HARDENED_RCOND_THRESHOLD && rcond > 0.0 {
        return Err(LinalgError::ConditionTooHigh {
            rcond,
            threshold: HARDENED_RCOND_THRESHOLD,
        });
    }

    let x = lu.solve(&rhs).ok_or(LinalgError::SingularMatrix)?;
    let backward_err = compute_backward_error(&matrix, &x, &rhs);

    Ok(SolveResult {
        x: x.iter().copied().collect(),
        warning: rcond_warning(rcond),
        backward_error: Some(backward_err),
    })
}

#[cfg(test)]
fn solve_general(a: &[Vec<f64>], b: &[f64]) -> Result<SolveResult, LinalgError> {
    solve_general_with_hardening(a, b, RuntimeMode::Strict)
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
    let max_s = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let threshold = (matrix.nrows().max(matrix.ncols()) as f64) * f64::EPSILON * max_s;
    let rank = svd
        .singular_values
        .iter()
        .filter(|&&s| s > threshold)
        .count();
    if matrix.nrows() == matrix.ncols() && rank < matrix.nrows() {
        return Err(LinalgError::SingularMatrix);
    }

    let x = svd
        .solve(&rhs, threshold)
        .map_err(|_| LinalgError::SingularMatrix)?;
    let min_s = svd
        .singular_values
        .iter()
        .copied()
        .filter(|s| *s > threshold)
        .fold(None, |acc, s| match acc {
            None => Some(s),
            Some(m) => Some(m.min(s)),
        })
        .unwrap_or(0.0);
    let rcond = if max_s > 0.0 { min_s / max_s } else { 0.0 };
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

    let mut matrix_cache: Option<DMatrix<f64>> = None;
    let mut lu_cache: Option<LU<f64, Dyn, Dyn>> = None;

    // Determine rcond and action, deferring expensive LU if possible.
    let (action, rcond, posterior, expected_losses, chosen_loss) =
        if options.assume_a == Some(MatrixAssumption::Diagonal) {
            let rcond = fast_rcond_diagonal(&effective_a);
            let (act, post, losses, choice) =
                portfolio.select_action(rcond, Some(StructuralEvidence::Diagonal));
            (act, rcond, post, losses, choice)
        } else if options.assume_a == Some(MatrixAssumption::LowerTriangular)
            || options.assume_a == Some(MatrixAssumption::UpperTriangular)
        {
            let is_upper = options.assume_a == Some(MatrixAssumption::UpperTriangular);
            let lower = if options.transposed {
                is_upper
            } else {
                !is_upper
            };
            let rcond = fast_rcond_triangular(&effective_a, lower);
            let (act, post, losses, choice) =
                portfolio.select_action(rcond, Some(StructuralEvidence::Triangular));
            (act, rcond, post, losses, choice)
        } else {
            let matrix = dmatrix_from_rows(&effective_a)?;
            let lu: LU<f64, Dyn, Dyn> = matrix.clone().lu();
            let rcond = fast_rcond_from_lu(&lu, matrix.lp_norm(1), rows);
            let evidence = options.assume_a.map(assumption_to_evidence);
            let (act, post, losses, choice) = portfolio.select_action(rcond, evidence);
            matrix_cache = Some(matrix);
            lu_cache = Some(lu);
            (act, rcond, post, losses, choice)
        };

    // Dispatch to chosen solver
    let result = match action {
        SolverAction::DirectLU => {
            let matrix = if let Some(m) = matrix_cache {
                m
            } else {
                dmatrix_from_rows(&effective_a)?
            };
            let lu = if let Some(l) = lu_cache {
                l
            } else {
                matrix.clone().lu()
            };
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
        SolverAction::TriangularFastPath => {
            let is_upper = options.assume_a == Some(MatrixAssumption::UpperTriangular);
            let lower = if options.transposed {
                is_upper
            } else {
                !is_upper
            };
            solve_triangular_internal(
                &effective_a,
                b,
                TriangularTranspose::NoTranspose,
                lower,
                false,
            )
        }
    };

    emit_trace(LinalgTrace {
        operation: "solve_with_casp",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: Some(rcond),
        warning: rcond_warning(rcond).map(|w| format!("{w:?}")),
        error: result.as_ref().err().map(|e| e.to_string()),
    });

    // Record evidence regardless of outcome
    let fallback_active = !matches!(action, SolverAction::DirectLU);
    let backward_error = result.as_ref().ok().and_then(|r| r.backward_error);
    portfolio.record_evidence(SolverEvidenceEntry {
        component: "solver_portfolio",
        matrix_shape: (rows, cols),
        rcond_estimate: rcond,
        chosen_action: action,
        posterior: posterior.to_vec(),
        expected_losses: expected_losses.to_vec(),
        chosen_expected_loss: chosen_loss,
        fallback_active,
        backward_error,
    });

    result
}

fn fast_rcond_triangular(a: &[Vec<f64>], lower: bool) -> f64 {
    let n = a.len();
    if n == 0 {
        return 1.0;
    }

    // 1. Compute ||A||_1 in O(n^2)
    let mut a_norm = 0.0;
    for j in 0..n {
        let mut col_sum = 0.0;
        if lower {
            for row in a.iter().take(n).skip(j) {
                col_sum += row[j].abs();
            }
        } else {
            for row in a.iter().take(j + 1) {
                col_sum += row[j].abs();
            }
        }
        if col_sum > a_norm {
            a_norm = col_sum;
        }
    }

    if a_norm == 0.0 {
        return 0.0;
    }

    // 2. Estimate ||A^-1||_1 via a greedy 1-norm estimator.
    // We estimate ||A^-1||_1 = ||(A^T)^-1||_\infty by solving A^T x = b
    // where b_i in {-1, 1} is chosen greedily to maximize x_i.
    let mut x = vec![0.0; n];
    if lower {
        // A^T is upper triangular. Solve A^T x = b backward.
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += a[j][i] * x[j]; // A^T_ij = A_ji
            }
            if a[i][i].abs() < 1e-18 {
                return 0.0;
            }
            let b_i = if sum < 0.0 { 1.0 } else { -1.0 };
            x[i] = (b_i - sum) / a[i][i];
        }
    } else {
        // A^T is lower triangular. Solve A^T x = b forward.
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += a[j][i] * x[j]; // A^T_ij = A_ji
            }
            if a[i][i].abs() < 1e-18 {
                return 0.0;
            }
            let b_i = if sum < 0.0 { 1.0 } else { -1.0 };
            x[i] = (b_i - sum) / a[i][i];
        }
    }
    let a_inv_norm: f64 = x.iter().map(|v| v.abs()).fold(0.0, f64::max);

    let rcond = 1.0 / (a_norm * a_inv_norm);
    if rcond.is_nan() {
        0.0
    } else {
        rcond.min(1.0)
    }
}

fn fast_rcond_diagonal(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 1.0;
    }
    let mut max_abs = 0.0;
    let mut min_abs = f64::INFINITY;
    for (i, row) in a.iter().enumerate() {
        if i >= row.len() {
            return 0.0; // Ragged matrix treated as singular
        }
        let val = row[i].abs();
        if val > max_abs {
            max_abs = val;
        }
        if val < min_abs {
            min_abs = val;
        }
    }
    if max_abs == 0.0 {
        0.0
    } else {
        min_abs / max_abs
    }
}

/// Estimate spectral condition number via power iteration.
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

    // Inverse power iteration for smallest singular value using LU on (AᵀA)⁻¹
    // w_{k+1} = (AᵀA)⁻¹ w_k = A⁻¹ (Aᵀ)⁻¹ w_k
    let mut w = DVector::from_element(n, 1.0 / (n as f64).sqrt());
    for _ in 0..iterations {
        // 1. Solve Aᵀ y = w
        match solve_lu_transpose(lu, &w) {
            Some(y) => {
                // 2. Solve A w_new = y
                match lu.solve(&y) {
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
            None => return 0.0,
        }
    }

    // Final estimate: ||A⁻¹ w||₂ = 1/σ_min
    // (AᵀA)⁻¹ w = (1/σ_min²) w
    let y = match solve_lu_transpose(lu, &w) {
        Some(y) => y,
        None => return 0.0,
    };
    let w_final = match lu.solve(&y) {
        Some(s) => s,
        None => return 0.0,
    };

    let sigma_min_sq_inv = w_final.norm();
    if sigma_min_sq_inv <= 0.0 || !sigma_min_sq_inv.is_finite() {
        return 0.0;
    }
    let sigma_min = (1.0 / sigma_min_sq_inv).sqrt();

    sigma_min / sigma_max
}

// ══════════════════════════════════════════════════════════════════════
// Matrix Decompositions — Public API
// ══════════════════════════════════════════════════════════════════════

/// LU decomposition with partial pivoting: PA = LU.
///
/// Returns permutation matrix P, unit lower triangular L, and upper triangular U.
/// Matches `scipy.linalg.lu(a)`.
pub fn lu(a: &[Vec<f64>], options: DecompOptions) -> Result<LuResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(LuResult {
            p: Vec::new(),
            l: Vec::new(),
            u: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let lu_decomp: LU<f64, Dyn, Dyn> = matrix.lu();

    // Extract L, U directly; build P from the permutation sequence
    let l_mat = lu_decomp.l();
    let u_mat = lu_decomp.u();
    // Build permutation matrix by applying the permutation to identity
    let mut p_mat = DMatrix::<f64>::identity(rows, rows);
    lu_decomp.p().permute_rows(&mut p_mat);

    emit_trace(LinalgTrace {
        operation: "lu",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(LuResult {
        p: rows_from_dmatrix(&p_mat),
        l: rows_from_dmatrix(&l_mat),
        u: rows_from_dmatrix(&u_mat),
    })
}

/// Compact LU factorization for subsequent solves via `lu_solve`.
///
/// Matches `scipy.linalg.lu_factor(a)`.
pub fn lu_factor(a: &[Vec<f64>], options: DecompOptions) -> Result<LuFactorResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    let matrix = dmatrix_from_rows(a)?;
    let a_norm_1 = matrix.lp_norm(1);
    let lu_decomp: LU<f64, Dyn, Dyn> = matrix.lu();

    emit_trace(LinalgTrace {
        operation: "lu_factor",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(LuFactorResult {
        lu_internal: lu_decomp,
        n: rows,
        a_norm_1,
    })
}

/// Solve a linear system using a precomputed LU factorization.
///
/// Matches `scipy.linalg.lu_solve(lu_and_piv, b)`.
pub fn lu_solve(lu_factor: &LuFactorResult, b: &[f64]) -> Result<SolveResult, LinalgError> {
    if b.len() != lu_factor.n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (lu_factor.n, lu_factor.n),
            b_len: b.len(),
        });
    }

    let rhs = DVector::from_column_slice(b);
    let x = lu_factor
        .lu_internal
        .solve(&rhs)
        .ok_or(LinalgError::SingularMatrix)?;

    let rcond = fast_rcond_from_lu(&lu_factor.lu_internal, lu_factor.a_norm_1, lu_factor.n);

    emit_trace(LinalgTrace {
        operation: "lu_solve",
        matrix_size: (lu_factor.n, lu_factor.n),
        mode: RuntimeMode::Strict,
        rcond: Some(rcond),
        warning: None,
        error: None,
    });

    Ok(SolveResult {
        x: x.iter().copied().collect(),
        warning: rcond_warning(rcond),
        backward_error: None,
    })
}

/// QR decomposition: A = QR.
///
/// Returns orthogonal Q and upper triangular R.
/// Matches `scipy.linalg.qr(a)`.
pub fn qr(a: &[Vec<f64>], options: DecompOptions) -> Result<QrResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 || cols == 0 {
        return Ok(QrResult {
            q: Vec::new(),
            r: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let qr_decomp = matrix.qr();
    let q_mat = qr_decomp.q();
    let r_mat = qr_decomp.r();

    emit_trace(LinalgTrace {
        operation: "qr",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(QrResult {
        q: rows_from_dmatrix(&q_mat),
        r: rows_from_dmatrix(&r_mat),
    })
}

/// Singular Value Decomposition: A = U Σ Vᵀ.
///
/// Returns left singular vectors U, singular values σ, and right singular vectors Vᵀ.
/// Matches `scipy.linalg.svd(a)`.
pub fn svd(a: &[Vec<f64>], options: DecompOptions) -> Result<SvdResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 || cols == 0 {
        return Ok(SvdResult {
            u: Vec::new(),
            s: Vec::new(),
            vt: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let svd_decomp = SVD::new(matrix, true, true);

    let u_mat = svd_decomp
        .u
        .as_ref()
        .ok_or(LinalgError::ConvergenceFailure {
            detail: "SVD failed to compute U".into(),
        })?;
    let vt_mat = svd_decomp
        .v_t
        .as_ref()
        .ok_or(LinalgError::ConvergenceFailure {
            detail: "SVD failed to compute Vt".into(),
        })?;
    let s: Vec<f64> = svd_decomp.singular_values.iter().copied().collect();

    emit_trace(LinalgTrace {
        operation: "svd",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(SvdResult {
        u: rows_from_dmatrix(u_mat),
        s,
        vt: rows_from_dmatrix(vt_mat),
    })
}

/// Compute singular values only (without U and Vᵀ).
///
/// Matches `scipy.linalg.svdvals(a)`.
pub fn svdvals(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<f64>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }

    let matrix = dmatrix_from_rows(a)?;
    let svd_decomp = SVD::new(matrix, false, false);
    let s: Vec<f64> = svd_decomp.singular_values.iter().copied().collect();

    emit_trace(LinalgTrace {
        operation: "svdvals",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(s)
}

/// Cholesky decomposition for symmetric positive-definite matrices: A = LLᵀ.
///
/// If `lower` is true (default), returns L such that A = LLᵀ.
/// If `lower` is false, returns U such that A = UᵀU.
/// Matches `scipy.linalg.cholesky(a, lower=True)`.
pub fn cholesky(
    a: &[Vec<f64>],
    lower: bool,
    options: DecompOptions,
) -> Result<CholeskyResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(CholeskyResult { factor: Vec::new() });
    }

    let matrix = dmatrix_from_rows(a)?;
    let chol = Cholesky::new(matrix).ok_or(LinalgError::InvalidArgument {
        detail: "matrix is not positive definite".into(),
    })?;

    let factor = if lower {
        chol.l()
    } else {
        chol.l().transpose()
    };

    emit_trace(LinalgTrace {
        operation: "cholesky",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(CholeskyResult {
        factor: rows_from_dmatrix(&factor),
    })
}

/// Compact Cholesky factorization for subsequent solves via `cho_solve`.
///
/// Always uses the lower triangular factor internally.
/// Matches `scipy.linalg.cho_factor(a)`.
pub fn cho_factor(a: &[Vec<f64>], options: DecompOptions) -> Result<ChoFactorResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    let matrix = dmatrix_from_rows(a)?;
    let chol = Cholesky::new(matrix).ok_or(LinalgError::InvalidArgument {
        detail: "matrix is not positive definite".into(),
    })?;

    emit_trace(LinalgTrace {
        operation: "cho_factor",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(ChoFactorResult {
        chol_internal: chol,
        n: rows,
    })
}

/// Solve a linear system using a precomputed Cholesky factorization.
///
/// Matches `scipy.linalg.cho_solve((c, lower), b)`.
pub fn cho_solve(cho: &ChoFactorResult, b: &[f64]) -> Result<SolveResult, LinalgError> {
    if b.len() != cho.n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (cho.n, cho.n),
            b_len: b.len(),
        });
    }

    let rhs = DVector::from_column_slice(b);
    let x = cho.chol_internal.solve(&rhs);

    emit_trace(LinalgTrace {
        operation: "cho_solve",
        matrix_size: (cho.n, cho.n),
        mode: RuntimeMode::Strict,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(SolveResult {
        x: x.iter().copied().collect(),
        warning: None,
        backward_error: None,
    })
}

/// LDL decomposition for symmetric indefinite matrices.
///
/// Factors A = L * D * Lᵀ where L is unit lower triangular and D is diagonal.
/// Unlike Cholesky, this works for symmetric matrices that are not positive definite.
/// Matches `scipy.linalg.ldl(a)` for the real symmetric case.
pub fn ldl(a: &[Vec<f64>], options: DecompOptions) -> Result<LdlResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    let n = rows;
    if n == 0 {
        return Ok(LdlResult {
            l: Vec::new(),
            d: Vec::new(),
        });
    }

    // Symmetric check in hardened mode
    if matches!(options.mode, RuntimeMode::Hardened) {
        for (i, row_i) in a.iter().enumerate() {
            for (j, &aij) in row_i.iter().enumerate().skip(i + 1) {
                let aji = a[j][i];
                if (aij - aji).abs() > 1e-12 * (aij.abs() + aji.abs() + 1.0) {
                    return Err(LinalgError::InvalidArgument {
                        detail: format!(
                            "matrix is not symmetric: a[{i}][{j}]={aij} != a[{j}][{i}]={aji}"
                        ),
                    });
                }
            }
        }
    }

    // Aasen / Bunch-Kaufman-style 1×1 pivoting LDL factorization
    // (no permutation for simplicity — matches basic scipy.linalg.ldl behavior)
    let mut l_mat = vec![vec![0.0; n]; n];
    let mut d_vec = vec![0.0; n];

    // Copy lower triangle of A
    let mut work = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            work[i][j] = a[i][j];
        }
    }

    for j in 0..n {
        // Compute d[j]
        let mut sum = work[j][j];
        for k in 0..j {
            sum -= l_mat[j][k] * l_mat[j][k] * d_vec[k];
        }
        d_vec[j] = sum;

        // Set diagonal of L to 1
        l_mat[j][j] = 1.0;

        if d_vec[j].abs() < f64::EPSILON * 1e3 {
            // Near-zero diagonal: skip to avoid division by zero
            // The matrix has a near-zero eigenvalue
            for row in l_mat.iter_mut().skip(j + 1) {
                row[j] = 0.0;
            }
            continue;
        }

        // Compute column j of L below diagonal
        for i in (j + 1)..n {
            let mut sum = work[i][j];
            for k in 0..j {
                sum -= l_mat[i][k] * l_mat[j][k] * d_vec[k];
            }
            l_mat[i][j] = sum / d_vec[j];
        }
    }

    emit_trace(LinalgTrace {
        operation: "ldl",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(LdlResult { l: l_mat, d: d_vec })
}

// ══════════════════════════════════════════════════════════════════════
// Eigenvalue Decompositions — Public API
// ══════════════════════════════════════════════════════════════════════

/// Eigenvalue decomposition for general (possibly non-symmetric) matrices.
///
/// Returns eigenvalues as (real, imaginary) pairs and right eigenvectors.
/// For real matrices with complex eigenvalues, complex conjugate pairs appear
/// consecutively. Matches `scipy.linalg.eig(a)`.
pub fn eig(a: &[Vec<f64>], options: DecompOptions) -> Result<EigResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(EigResult {
            eigenvalues_re: Vec::new(),
            eigenvalues_im: Vec::new(),
            eigenvectors: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;

    // Use Schur decomposition: A = Q T Q^T, eigenvalues on diagonal of T
    // For real matrices, Schur form has 1×1 blocks (real eigenvalues) and
    // 2×2 blocks (complex conjugate pairs) on the diagonal.
    let schur = matrix.clone().schur();
    let (q_mat, t_mat) = schur.unpack();

    let mut eigenvalues_re = Vec::with_capacity(rows);
    let mut eigenvalues_im = Vec::with_capacity(rows);

    let mut i = 0;
    while i < rows {
        if i + 1 < rows && t_mat[(i + 1, i)].abs() > f64::EPSILON * 100.0 {
            // 2×2 block: complex conjugate pair
            let a11 = t_mat[(i, i)];
            let a12 = t_mat[(i, i + 1)];
            let a21 = t_mat[(i + 1, i)];
            let a22 = t_mat[(i + 1, i + 1)];
            let trace = a11 + a22;
            let det_block = a11 * a22 - a12 * a21;
            let disc = trace * trace - 4.0 * det_block;
            let re = trace / 2.0;
            let im = (-disc).sqrt() / 2.0;
            eigenvalues_re.push(re);
            eigenvalues_im.push(im);
            eigenvalues_re.push(re);
            eigenvalues_im.push(-im);
            i += 2;
        } else {
            eigenvalues_re.push(t_mat[(i, i)]);
            eigenvalues_im.push(0.0);
            i += 1;
        }
    }

    emit_trace(LinalgTrace {
        operation: "eig",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(EigResult {
        eigenvalues_re,
        eigenvalues_im,
        eigenvectors: rows_from_dmatrix(&q_mat),
    })
}

/// Compute eigenvalues only for a general matrix.
///
/// Returns eigenvalues as (real_parts, imaginary_parts).
/// Matches `scipy.linalg.eigvals(a)`.
pub fn eigvals(
    a: &[Vec<f64>],
    options: DecompOptions,
) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
    let result = eig(a, options)?;
    Ok((result.eigenvalues_re, result.eigenvalues_im))
}

/// Eigenvalue decomposition for symmetric/Hermitian matrices.
///
/// Eigenvalues are returned in ascending order. Eigenvectors form an
/// orthogonal matrix. Matches `scipy.linalg.eigh(a)`.
pub fn eigh(a: &[Vec<f64>], options: DecompOptions) -> Result<EighResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(EighResult {
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let sym = matrix.symmetric_eigen();

    // nalgebra returns eigenvalues in arbitrary order; sort ascending
    let mut pairs: Vec<(f64, Vec<f64>)> = sym
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(j, &val)| {
            let col: Vec<f64> = (0..rows).map(|i| sym.eigenvectors[(i, j)]).collect();
            (val, col)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = pairs.iter().map(|(v, _)| *v).collect();
    // Reconstruct eigenvector matrix from sorted columns
    let mut eigenvectors = vec![vec![0.0; cols]; rows];
    for (col_idx, (_, col_data)) in pairs.iter().enumerate() {
        for (row_idx, &val) in col_data.iter().enumerate() {
            eigenvectors[row_idx][col_idx] = val;
        }
    }

    emit_trace(LinalgTrace {
        operation: "eigh",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(EighResult {
        eigenvalues,
        eigenvectors,
    })
}

/// Compute eigenvalues only for a symmetric/Hermitian matrix.
///
/// Returns real eigenvalues in ascending order.
/// Matches `scipy.linalg.eigvalsh(a)`.
pub fn eigvalsh(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<f64>, LinalgError> {
    let result = eigh(a, options)?;
    Ok(result.eigenvalues)
}

// ══════════════════════════════════════════════════════════════════════
// Schur and Hessenberg Decompositions — Public API
// ══════════════════════════════════════════════════════════════════════

/// Schur decomposition: A = Z T Zᴴ.
///
/// Returns the orthogonal Schur vectors Z and the upper quasi-triangular
/// Schur form T (real Schur form for real matrices).
/// Matches `scipy.linalg.schur(a)`.
pub fn schur(a: &[Vec<f64>], options: DecompOptions) -> Result<SchurResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(SchurResult {
            z: Vec::new(),
            t: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let schur_decomp = matrix.schur();
    let (q_mat, t_mat) = schur_decomp.unpack();

    emit_trace(LinalgTrace {
        operation: "schur",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(SchurResult {
        z: rows_from_dmatrix(&q_mat),
        t: rows_from_dmatrix(&t_mat),
    })
}

/// Hessenberg decomposition: A = Q H Qᵀ.
///
/// Returns the orthogonal matrix Q and the upper Hessenberg matrix H.
/// Matches `scipy.linalg.hessenberg(a, calc_q=True)`.
pub fn hessenberg(a: &[Vec<f64>], options: DecompOptions) -> Result<HessenbergResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(HessenbergResult {
            q: Vec::new(),
            h: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let hess = matrix.hessenberg();
    let q_mat = hess.q();
    let h_mat = hess.unpack_h();

    emit_trace(LinalgTrace {
        operation: "hessenberg",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(HessenbergResult {
        q: rows_from_dmatrix(&q_mat),
        h: rows_from_dmatrix(&h_mat),
    })
}

// ══════════════════════════════════════════════════════════════════════
// Matrix Functions — Public API
// ══════════════════════════════════════════════════════════════════════

/// Matrix exponential using scaling and squaring with Padé approximation.
///
/// Computes exp(A) for a square matrix A.
/// Matches `scipy.linalg.expm(A)`.
pub fn expm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(Vec::new());
    }

    let matrix = dmatrix_from_rows(a)?;
    let result = expm_pade_scaling_squaring(&matrix);

    emit_trace(LinalgTrace {
        operation: "expm",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&result))
}

/// Scaling and squaring method with truncated Taylor series.
///
/// Algorithm: scale A by 2^(-s) so ||A/2^s|| is small, compute exp via
/// Taylor series (accurate for small norm), then square s times.
fn expm_pade_scaling_squaring(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    let identity = DMatrix::<f64>::identity(n, n);

    // Compute 1-norm for scaling
    let norm1 = matrix_one_norm(a);

    if norm1 == 0.0 {
        return identity;
    }

    // Scale until ||A/2^s||_1 < 0.5 to ensure Taylor convergence is fast
    let s = if norm1 <= 0.5 {
        0u32
    } else {
        ((norm1 * 2.0).log2().ceil()) as u32
    };
    let scaled = a / (2.0_f64.powi(s as i32));

    // Taylor series: exp(A) = I + A + A²/2! + A³/3! + ... + A^k/k!
    // For ||A|| <= 0.5, 20 terms gives ~10^-16 accuracy
    let result = taylor_exp(&scaled, &identity, 20);

    // Squaring phase: exp(A) = exp(A/2^s)^(2^s)
    let mut exp_a = result;
    for _ in 0..s {
        exp_a = &exp_a * &exp_a;
    }

    exp_a
}

/// Taylor series approximation of exp(A) = I + A + A²/2! + ... + A^k/k!
fn taylor_exp(a: &DMatrix<f64>, identity: &DMatrix<f64>, terms: usize) -> DMatrix<f64> {
    let mut result = identity.clone();
    let mut term = identity.clone(); // A^k / k!

    for k in 1..=terms {
        term = &term * a / (k as f64);
        result += &term;
    }

    result
}

/// Compute the 1-norm (max column sum of absolute values) of a matrix.
fn matrix_one_norm(m: &DMatrix<f64>) -> f64 {
    let (rows, cols) = (m.nrows(), m.ncols());
    (0..cols)
        .map(|j| (0..rows).map(|i| m[(i, j)].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

// ══════════════════════════════════════════════════════════════════════
// Matrix Functions — logm, sqrtm, matrix_power
// ══════════════════════════════════════════════════════════════════════

/// Matrix logarithm.
///
/// Matches `scipy.linalg.logm(A)`.
///
/// Computes logm via eigendecomposition: if A = V D V^{-1}, then
/// logm(A) = V diag(log(d_i)) V^{-1}.
///
/// Requires all eigenvalues to be positive real.
pub fn logm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(Vec::new());
    }

    let matrix = dmatrix_from_rows(a)?;
    let n = rows;

    // Check if the matrix is symmetric
    let is_symmetric = (0..n).all(|i| {
        (0..n).all(|j| {
            (matrix[(i, j)] - matrix[(j, i)]).abs() < 1e-12 * matrix[(i, j)].abs().max(1.0)
        })
    });

    if !is_symmetric {
        // Non-symmetric: use general Schur-based path
        return logm_general(&matrix, n, options);
    }

    // Symmetric matrix: use eigendecomposition (symmetric_eigen is safe here)
    let eig = matrix.clone().symmetric_eigen();
    let eigenvalues = &eig.eigenvalues;

    // Check all eigenvalues are positive (required for real logarithm)
    for (i, &ev) in eigenvalues.iter().enumerate() {
        if ev <= 0.0 {
            emit_trace(LinalgTrace {
                operation: "logm",
                matrix_size: (n, n),
                mode: options.mode,
                rcond: None,
                warning: Some(format!("non-positive eigenvalue {ev} at index {i}")),
                error: None,
            });
            return logm_general(&matrix, n, options);
        }
    }

    // logm = V * diag(log(eigenvalues)) * V^T
    let v = &eig.eigenvectors;
    let mut log_diag = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        log_diag[(i, i)] = eigenvalues[i].ln();
    }

    let result = v * log_diag * v.transpose();

    emit_trace(LinalgTrace {
        operation: "logm",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&result))
}

/// General logm via eigendecomposition (for non-symmetric matrices).
fn logm_general(
    matrix: &DMatrix<f64>,
    n: usize,
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    // Use Schur decomposition: A = Q T Q^T where T is quasi-upper-triangular
    let schur = matrix.clone().schur();
    let (q, t) = schur.unpack();

    // Compute log of the quasi-triangular matrix
    let mut log_t = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        if t[(i, i)] > 0.0 {
            log_t[(i, i)] = t[(i, i)].ln();
        } else {
            // Non-positive diagonal: return NaN for this element
            log_t[(i, i)] = f64::NAN;
        }
    }

    // Compute off-diagonal of log(T) using Parlett's recurrence:
    //   (t[j,j] - t[i,i]) f[i,j]
    //     = t[i,j] (f[j,j] - f[i,i])
    //       + Σ_{k=i+1}^{j-1} (t[i,k] f[k,j] - f[i,k] t[k,j])
    //
    // For repeated diagonal entries we fall back to the first-derivative limit
    // used by the logarithm of a Jordan block: f[i,j] ~= n[i,j] / λ.
    for j in 1..n {
        for i in (0..j).rev() {
            let tii = t[(i, i)];
            let tjj = t[(j, j)];
            let di = log_t[(i, i)];
            let dj = log_t[(j, j)];
            let mut correction = 0.0;
            for k in (i + 1)..j {
                correction += t[(i, k)] * log_t[(k, j)] - log_t[(i, k)] * t[(k, j)];
            }
            let eigen_gap = tjj - tii;
            if eigen_gap.abs() > 1e-15 {
                let numerator = t[(i, j)] * (dj - di) + correction;
                log_t[(i, j)] = numerator / eigen_gap;
            } else {
                // Near-degenerate case: T[i,i] ≈ T[j,j]
                // Use the first-derivative limit for log(λI + N).
                let mut jordan_sum = t[(i, j)];
                for k in (i + 1)..j {
                    jordan_sum += t[(i, k)] * log_t[(k, j)] - log_t[(i, k)] * t[(k, j)];
                }
                if tii.abs() > 1e-15 {
                    log_t[(i, j)] = jordan_sum / tii;
                } else {
                    log_t[(i, j)] = 0.0;
                }
            }
        }
    }

    let result = &q * log_t * q.transpose();

    emit_trace(LinalgTrace {
        operation: "logm",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: Some("used general (non-symmetric) path".to_string()),
        error: None,
    });

    Ok(rows_from_dmatrix(&result))
}

/// Matrix square root.
///
/// Matches `scipy.linalg.sqrtm(A)`.
///
/// Computes sqrtm via eigendecomposition: if A = V D V^{-1}, then
/// sqrtm(A) = V diag(sqrt(d_i)) V^{-1}.
pub fn sqrtm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(Vec::new());
    }

    let matrix = dmatrix_from_rows(a)?;
    let n = rows;

    // Check symmetry before using symmetric eigendecomposition
    let is_symmetric = (0..n).all(|i| {
        (0..n).all(|j| {
            (matrix[(i, j)] - matrix[(j, i)]).abs() < 1e-12 * matrix[(i, j)].abs().max(1.0)
        })
    });

    if !is_symmetric {
        // Non-symmetric: sqrtm via Schur decomposition
        // A = Q T Q^T, sqrt(A) = Q sqrt(T) Q^T
        let schur = matrix.clone().schur();
        let (q, t) = schur.unpack();
        let mut sqrt_t = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            if t[(i, i)] >= 0.0 {
                sqrt_t[(i, i)] = t[(i, i)].sqrt();
            } else {
                sqrt_t[(i, i)] = f64::NAN;
            }
        }
        // Off-diagonal: Parlett recurrence for sqrt
        for j in 1..n {
            for i in (0..j).rev() {
                let si = sqrt_t[(i, i)];
                let sj = sqrt_t[(j, j)];
                let mut sum = t[(i, j)];
                for k in (i + 1)..j {
                    sum -= sqrt_t[(i, k)] * sqrt_t[(k, j)];
                }
                let denom = si + sj;
                sqrt_t[(i, j)] = if denom.abs() > 1e-15 {
                    sum / denom
                } else {
                    0.0
                };
            }
        }
        let result = &q * sqrt_t * q.transpose();
        emit_trace(LinalgTrace {
            operation: "sqrtm",
            matrix_size: (n, n),
            mode: options.mode,
            rcond: None,
            warning: Some("used general (non-symmetric) Schur path".to_string()),
            error: None,
        });
        return Ok(rows_from_dmatrix(&result));
    }

    // Symmetric path: eigendecomposition is safe here
    let eig = matrix.clone().symmetric_eigen();
    let eigenvalues = &eig.eigenvalues;

    // Check eigenvalues are non-negative
    let all_nonneg = eigenvalues.iter().all(|&ev| ev >= -1e-14);

    if all_nonneg {
        let v = &eig.eigenvectors;
        let mut sqrt_diag = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            sqrt_diag[(i, i)] = eigenvalues[i].max(0.0).sqrt();
        }
        let result = v * sqrt_diag * v.transpose();

        emit_trace(LinalgTrace {
            operation: "sqrtm",
            matrix_size: (n, n),
            mode: options.mode,
            rcond: None,
            warning: None,
            error: None,
        });

        Ok(rows_from_dmatrix(&result))
    } else {
        // For matrices with negative eigenvalues, sqrtm produces complex results.
        // Return NaN to indicate this.
        emit_trace(LinalgTrace {
            operation: "sqrtm",
            matrix_size: (n, n),
            mode: options.mode,
            rcond: None,
            warning: Some("matrix has negative eigenvalues; real sqrtm undefined".to_string()),
            error: None,
        });
        let nan_matrix = vec![vec![f64::NAN; n]; n];
        Ok(nan_matrix)
    }
}

/// Integer matrix power: A^n.
///
/// Matches `numpy.linalg.matrix_power(A, n)`.
///
/// - n > 0: repeated multiplication (binary exponentiation)
/// - n == 0: identity matrix
/// - n < 0: (A^{-1})^{|n|}
pub fn matrix_power(
    a: &[Vec<f64>],
    power: i32,
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(Vec::new());
    }

    let n = rows;

    if power == 0 {
        // A^0 = I
        let mut identity = vec![vec![0.0; n]; n];
        for (i, row) in identity.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        return Ok(identity);
    }

    let matrix = dmatrix_from_rows(a)?;

    let base = if power < 0 {
        // Need inverse
        let lu = matrix.clone().lu();
        match lu.try_inverse() {
            Some(inv) => inv,
            None => {
                return Err(LinalgError::SingularMatrix);
            }
        }
    } else {
        matrix
    };

    let abs_power = power.unsigned_abs();

    // Binary exponentiation
    let mut result = DMatrix::<f64>::identity(n, n);
    let mut current = base;
    let mut p = abs_power;

    while p > 0 {
        if p & 1 == 1 {
            result = &result * &current;
        }
        p >>= 1;
        if p > 0 {
            current = &current * &current;
        }
    }

    emit_trace(LinalgTrace {
        operation: "matrix_power",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&result))
}

/// Matrix sign function.
///
/// For a matrix with no purely imaginary eigenvalues, sign(A) is defined as
/// A * (A²)^{-1/2}, or equivalently via eigendecomposition:
/// sign(A) = V * diag(sign(λ_i)) * V^{-1}.
///
/// Matches `scipy.linalg.signm(A)`.
pub fn signm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    funm(a, |x| if x >= 0.0 { 1.0 } else { -1.0 }, options)
}

/// General matrix function via eigendecomposition.
///
/// Computes f(A) = V * diag(f(λ_i)) * V^{-1} where A = V * diag(λ) * V^{-1}.
///
/// For symmetric matrices, uses orthogonal eigendecomposition (more stable).
/// For general matrices, uses Schur decomposition with Parlett's recurrence.
///
/// Matches `scipy.linalg.funm(A, func)`.
pub fn funm(
    a: &[Vec<f64>],
    func: impl Fn(f64) -> f64,
    _options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if rows == 0 {
        return Ok(Vec::new());
    }

    let matrix = dmatrix_from_rows(a)?;
    let n = rows;

    // Check symmetry
    let is_symmetric = (0..n).all(|i| {
        (0..n).all(|j| {
            (matrix[(i, j)] - matrix[(j, i)]).abs() < 1e-12 * matrix[(i, j)].abs().max(1.0)
        })
    });

    if is_symmetric {
        // Symmetric: A = V D V^T, f(A) = V f(D) V^T
        let eig = matrix.clone().symmetric_eigen();
        let v = &eig.eigenvectors;
        let mut fd = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            fd[(i, i)] = func(eig.eigenvalues[i]);
        }
        let result = v * fd * v.transpose();
        return Ok(rows_from_dmatrix(&result));
    }

    // General: use Schur decomposition A = Q T Q^T
    // Then f(A) = Q f(T) Q^T via Parlett's recurrence
    let schur = matrix.clone().schur();
    let (q, t) = schur.unpack();

    let mut ft = DMatrix::<f64>::zeros(n, n);
    // Diagonal: f(T[i,i])
    for i in 0..n {
        ft[(i, i)] = func(t[(i, i)]);
    }

    // Off-diagonal via Parlett's recurrence
    for j in 1..n {
        for i in (0..j).rev() {
            let mut sum = t[(i, j)] * (ft[(j, j)] - ft[(i, i)]);
            for k in (i + 1)..j {
                sum += t[(i, k)] * ft[(k, j)] - ft[(i, k)] * t[(k, j)];
            }
            let denom = t[(j, j)] - t[(i, i)];
            ft[(i, j)] = if denom.abs() > 1e-15 {
                sum / denom
            } else if t[(i, i)].abs() > 1e-15 {
                sum / t[(i, i)]
            } else {
                0.0
            };
        }
    }

    let result = &q * ft * q.transpose();
    Ok(rows_from_dmatrix(&result))
}

/// Fractional matrix power A^p for non-integer p.
///
/// Computes A^p via eigendecomposition: A^p = V * diag(λ_i^p) * V^{-1}.
///
/// Matches `scipy.linalg.fractional_matrix_power(A, p)`.
pub fn fractional_matrix_power(
    a: &[Vec<f64>],
    p: f64,
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    funm(a, |x| x.powf(p), options)
}

// ══════════════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════════════
// Specialized Solvers and Products
// ══════════════════════════════════════════════════════════════════════

/// Khatri-Rao product (column-wise Kronecker product).
///
/// For matrices A (m×n) and B (p×n), produces an (m*p × n) matrix where
/// column j is the Kronecker product of A[:,j] and B[:,j].
///
/// Matches `scipy.linalg.khatri_rao(A, B)`.
pub fn khatri_rao(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (m, na) = matrix_shape(a)?;
    let (p, nb) = matrix_shape(b)?;
    if na != nb {
        return Err(LinalgError::NotSupported {
            detail: format!("A has {na} columns but B has {nb} columns; must match"),
        });
    }
    let ncols = na;
    let nrows = m * p;

    let mut result = vec![vec![0.0; ncols]; nrows];
    for j in 0..ncols {
        for i in 0..m {
            for k in 0..p {
                result[i * p + k][j] = a[i][j] * b[k][j];
            }
        }
    }
    Ok(result)
}

/// Solve a circulant linear system C x = b using FFT.
///
/// A circulant matrix is defined by its first column c. The solution
/// is computed in O(n log n) via FFT: x = ifft(fft(b) / fft(c)).
///
/// Matches `scipy.linalg.solve_circulant(c, b)`.
pub fn solve_circulant(c: &[f64], b: &[f64]) -> Result<Vec<f64>, LinalgError> {
    let n = c.len();
    if b.len() != n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (n, n),
            b_len: b.len(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let opts = fsci_fft::FftOptions::default();
    let c_complex: Vec<(f64, f64)> = c.iter().map(|&value| (value, 0.0)).collect();
    let b_complex: Vec<(f64, f64)> = b.iter().map(|&value| (value, 0.0)).collect();
    let fft_c = fsci_fft::fft(&c_complex, &opts).map_err(|err| LinalgError::InvalidArgument {
        detail: format!("circulant FFT failed: {err}"),
    })?;
    let fft_b = fsci_fft::fft(&b_complex, &opts).map_err(|err| LinalgError::InvalidArgument {
        detail: format!("rhs FFT failed: {err}"),
    })?;

    // Element-wise division: fft_b / fft_c
    let fft_x: Vec<(f64, f64)> = fft_b
        .iter()
        .zip(fft_c.iter())
        .map(|(&(br, bi), &(cr, ci))| {
            let denom = cr * cr + ci * ci;
            if denom < 1e-30 {
                (0.0, 0.0)
            } else {
                ((br * cr + bi * ci) / denom, (bi * cr - br * ci) / denom)
            }
        })
        .collect();

    let x = fsci_fft::ifft(&fft_x, &opts).map_err(|err| LinalgError::InvalidArgument {
        detail: format!("circulant inverse FFT failed: {err}"),
    })?;
    Ok(x.into_iter().map(|(re, _)| re).collect())
}

/// Solve a Toeplitz linear system T x = b.
///
/// The Toeplitz matrix is defined by its first column `c` and optional first
/// row `r`. If `r` is omitted, a symmetric Toeplitz matrix is assumed.
///
/// Matches `scipy.linalg.solve_toeplitz((c, r), b)`.
pub fn solve_toeplitz(
    c: &[f64],
    r: Option<&[f64]>,
    b: &[f64],
) -> Result<Vec<f64>, LinalgError> {
    let n = c.len();
    if b.len() != n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (n, n),
            b_len: b.len(),
        });
    }
    let row = r.unwrap_or(c);
    if row.len() != n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (n, row.len()),
            b_len: b.len(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let matrix = toeplitz(c, r);
    Ok(solve(&matrix, b, SolveOptions::default())?.x)
}

// ══════════════════════════════════════════════════════════════════════
// Norms and Rank — Public API
// ══════════════════════════════════════════════════════════════════════

/// Compute matrix or vector norm.
///
/// Matches `scipy.linalg.norm(a, ord)`.
pub fn norm(a: &[Vec<f64>], kind: NormKind, options: DecompOptions) -> Result<f64, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }

    let matrix = dmatrix_from_rows(a)?;

    let result = match kind {
        NormKind::Fro => {
            // Frobenius norm: sqrt(sum of squares)
            matrix.norm()
        }
        NormKind::Spectral => {
            // Spectral norm: largest singular value
            let svd_decomp = SVD::new(matrix, false, false);
            svd_decomp
                .singular_values
                .iter()
                .copied()
                .fold(0.0_f64, f64::max)
        }
        NormKind::One => {
            // 1-norm: max column sum of absolute values
            (0..cols)
                .map(|j| (0..rows).map(|i| matrix[(i, j)].abs()).sum::<f64>())
                .fold(0.0_f64, f64::max)
        }
        NormKind::Inf => {
            // Infinity norm: max row sum of absolute values
            (0..rows)
                .map(|i| (0..cols).map(|j| matrix[(i, j)].abs()).sum::<f64>())
                .fold(0.0_f64, f64::max)
        }
    };

    emit_trace(LinalgTrace {
        operation: "norm",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(result)
}

/// Compute the numerical rank of a matrix using SVD.
///
/// A singular value is considered zero if it is less than `tol`.
/// If `tol` is None, uses `max(m,n) * eps * max(singular_values)`.
pub fn matrix_rank(
    a: &[Vec<f64>],
    tol: Option<f64>,
    options: DecompOptions,
) -> Result<usize, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 || cols == 0 {
        return Ok(0);
    }

    let matrix = dmatrix_from_rows(a)?;
    let svd_decomp = SVD::new(matrix, false, false);
    let singular_values = &svd_decomp.singular_values;
    let max_s = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let threshold = tol.unwrap_or_else(|| (rows.max(cols) as f64) * f64::EPSILON * max_s);
    let rank = singular_values.iter().filter(|s| **s > threshold).count();

    emit_trace(LinalgTrace {
        operation: "matrix_rank",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rank)
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
    let rcond = if max_diag > 0.0 {
        min_diag / max_diag
    } else {
        0.0
    };
    let warning = if rcond < 1e-12 {
        Some(LinalgWarning::IllConditioned {
            reciprocal_condition: rcond,
        })
    } else {
        None
    };

    let backward_error = compute_backward_error_dense(a, &x, b);

    Ok(SolveResult {
        x,
        warning,
        backward_error: Some(backward_error),
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

    let backward_error = compute_backward_error_dense(&mat, &x, b);

    Ok(SolveResult {
        x,
        warning: None,
        backward_error: Some(backward_error),
    })
}

/// Compute backward error using dense iteration: ||Ax - b|| / (||A|| × ||x|| + ||b||).
fn compute_backward_error_dense(a: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let m = if a.is_empty() { 0 } else { a[0].len() };

    let mut residual_sum_sq = 0.0;
    for i in 0..n {
        let mut ax_i = 0.0;
        for (j, &xj) in x.iter().enumerate().take(m) {
            ax_i += a[i][j] * xj;
        }
        let res_i = ax_i - b[i];
        residual_sum_sq += res_i * res_i;
    }
    let residual_norm = residual_sum_sq.sqrt();

    let mut a_sum_sq = 0.0;
    for row in a {
        for &val in row {
            a_sum_sq += val * val;
        }
    }
    let a_norm = a_sum_sq.sqrt();

    let mut x_sum_sq = 0.0;
    for &val in x {
        x_sum_sq += val * val;
    }
    let x_norm = x_sum_sq.sqrt();

    let mut b_sum_sq = 0.0;
    for &val in b {
        b_sum_sq += val * val;
    }
    let b_norm = b_sum_sq.sqrt();

    let denom = a_norm * x_norm + b_norm;
    if denom > 0.0 {
        residual_norm / denom
    } else {
        0.0
    }
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

// ══════════════════════════════════════════════════════════════════════
// Matrix Equation Solvers
// ══════════════════════════════════════════════════════════════════════

/// Solve the Sylvester equation AX + XB = Q.
///
/// Uses the Bartels-Stewart algorithm: reduce A and B to Schur form,
/// then solve the resulting triangular Sylvester equation column by column.
///
/// Matches `scipy.linalg.solve_sylvester(a, b, q)`.
pub fn solve_sylvester(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    q: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (m, ma) = matrix_shape(a)?;
    let (n, nb) = matrix_shape(b)?;
    let (qr, qc) = matrix_shape(q)?;

    if m != ma {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if n != nb {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if qr != m || qc != n {
        return Err(LinalgError::NotSupported {
            detail: format!("Q shape ({qr}x{qc}) must match A rows ({m}) x B cols ({n})"),
        });
    }
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }

    let a_mat = dmatrix_from_rows(a)?;
    let b_mat = dmatrix_from_rows(b)?;
    let q_mat = dmatrix_from_rows(q)?;

    // Schur decompositions: A = U T_A U^T, B = V T_B V^T
    let schur_a = a_mat.clone().schur();
    let (u, ta) = schur_a.unpack();
    let schur_b = b_mat.clone().schur();
    let (v, tb) = schur_b.unpack();

    // Transform Q: F = U^T Q V
    let f = u.transpose() * &q_mat * &v;

    // Solve T_A Y + Y T_B = F by vectorization:
    // vec(Y) satisfies (I_n ⊗ T_A + T_B^T ⊗ I_m) vec(Y) = vec(F)
    // For small problems this is direct; for larger ones we use the column-by-column
    // approach with the Schur structure.
    //
    // Direct approach: build the (mn × mn) system and solve via LU.
    let mn = m * n;
    let mut system = DMatrix::<f64>::zeros(mn, mn);

    // I_n ⊗ T_A: for each block (j,j), place T_A
    for j in 0..n {
        for r in 0..m {
            for c in 0..m {
                system[(j * m + r, j * m + c)] += ta[(r, c)];
            }
        }
    }

    // T_B^T ⊗ I_m: for each (j,k) in T_B^T, place T_B[k,j] * I_m
    for j in 0..n {
        for k in 0..n {
            let tbkj = tb[(k, j)]; // T_B^T[j,k] = T_B[k,j]
            if tbkj.abs() > 0.0 {
                for i in 0..m {
                    system[(j * m + i, k * m + i)] += tbkj;
                }
            }
        }
    }

    // RHS: vec(F)
    let mut rhs_vec = nalgebra::DVector::<f64>::zeros(mn);
    for j in 0..n {
        for i in 0..m {
            rhs_vec[j * m + i] = f[(i, j)];
        }
    }

    // Solve via LU
    let lu = system.full_piv_lu();
    let sol = lu
        .solve(&rhs_vec)
        .unwrap_or_else(|| nalgebra::DVector::zeros(mn));

    // Unvectorize Y
    let mut y = DMatrix::<f64>::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            y[(i, j)] = sol[j * m + i];
        }
    }

    // Transform back: X = U Y V^T
    let x = &u * y * v.transpose();

    emit_trace(LinalgTrace {
        operation: "solve_sylvester",
        matrix_size: (m, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&x))
}

/// Solve the continuous Lyapunov equation AX + XA^T = Q.
///
/// Special case of Sylvester equation with B = A^T.
///
/// Matches `scipy.linalg.solve_continuous_lyapunov(a, q)`.
pub fn solve_continuous_lyapunov(
    a: &[Vec<f64>],
    q: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (n, _) = matrix_shape(a)?;
    // B = A^T
    let a_t: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| a[j][i]).collect()).collect();
    solve_sylvester(a, &a_t, q, options)
}

/// Solve the discrete Lyapunov equation A X A^T - X + Q = 0.
///
/// Uses the bilinear transformation to convert to a continuous Lyapunov equation:
///   B = (A - I)(A + I)^{-1}
///   C = 2(A + I)^{-T} Q (A + I)^{-1}
/// Then solves B C B^T - C = -R for the continuous form.
///
/// Matches `scipy.linalg.solve_discrete_lyapunov(a, q)`.
pub fn solve_discrete_lyapunov(
    a: &[Vec<f64>],
    q: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (n, na) = matrix_shape(a)?;
    if n != na {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    let (qr, qc) = matrix_shape(q)?;
    if qr != n || qc != n {
        return Err(LinalgError::NotSupported {
            detail: format!("Q shape ({qr}x{qc}) must be {n}x{n}"),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // Direct approach: vectorize the equation
    // A X A^T - X = -Q
    // (A ⊗ A) vec(X) - vec(X) = -vec(Q)
    // (A ⊗ A - I) vec(X) = -vec(Q)
    let a_mat = dmatrix_from_rows(a)?;
    let q_mat = dmatrix_from_rows(q)?;
    let nn = n * n;

    let mut system = DMatrix::<f64>::zeros(nn, nn);

    // Build A ⊗ A
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    // (A ⊗ A)[i*n+k, j*n+l] = A[i,j] * A[k,l]
                    system[(i * n + k, j * n + l)] += a_mat[(i, j)] * a_mat[(k, l)];
                }
            }
        }
    }

    // Subtract identity
    for i in 0..nn {
        system[(i, i)] -= 1.0;
    }

    // RHS = -vec(Q)
    let mut rhs = nalgebra::DVector::<f64>::zeros(nn);
    for j in 0..n {
        for i in 0..n {
            rhs[j * n + i] = -q_mat[(i, j)];
        }
    }

    // Solve
    let lu = system.full_piv_lu();
    let sol = lu
        .solve(&rhs)
        .unwrap_or_else(|| nalgebra::DVector::zeros(nn));

    // Unvectorize
    let mut x = DMatrix::<f64>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            x[(i, j)] = sol[j * n + i];
        }
    }

    emit_trace(LinalgTrace {
        operation: "solve_discrete_lyapunov",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&x))
}

// ══════════════════════════════════════════════════════════════════════
// Subspace Operations: orth, null_space, subspace_angles, polar
// ══════════════════════════════════════════════════════════════════════

/// Compute an orthonormal basis for the range (column space) of a matrix.
///
/// Matches `scipy.linalg.orth(A, rcond)`.
///
/// Returns a matrix whose columns form an orthonormal basis for the column
/// space of A, determined by keeping singular vectors with singular values
/// above `rcond * max(s)`.
pub fn orth(
    a: &[Vec<f64>],
    rcond: Option<f64>,
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (m, n) = matrix_shape(a)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if m == 0 || n == 0 {
        return Ok(vec![vec![]; m]);
    }

    let matrix = dmatrix_from_rows(a)?;
    let svd_decomp = SVD::new(matrix, true, false);
    let u = svd_decomp
        .u
        .as_ref()
        .ok_or(LinalgError::UnsupportedAssumption)?;
    let singular_values = &svd_decomp.singular_values;

    let max_s = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tol = rcond.unwrap_or_else(|| (m.max(n) as f64) * f64::EPSILON) * max_s;

    let rank = singular_values.iter().filter(|&&s| s > tol).count();

    // Extract first `rank` columns of U.
    let mut result = vec![vec![0.0; rank]; m];
    for i in 0..m {
        for j in 0..rank {
            result[i][j] = u[(i, j)];
        }
    }

    Ok(result)
}

/// Compute an orthonormal basis for the null space of a matrix.
///
/// Matches `scipy.linalg.null_space(A, rcond)`.
///
/// Returns a matrix whose columns form an orthonormal basis for the null
/// space of A, determined by keeping right singular vectors with singular
/// values below `rcond * max(s)`.
pub fn null_space(
    a: &[Vec<f64>],
    rcond: Option<f64>,
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (m, n) = matrix_shape(a)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if m == 0 || n == 0 {
        // If no columns, null space is empty; if no rows, null space is all of R^n.
        if n == 0 {
            return Ok(vec![]);
        }
        // Return identity for n-dimensional null space.
        let mut ident = vec![vec![0.0; n]; n];
        for (i, row) in ident.iter_mut().enumerate().take(n) {
            row[i] = 1.0;
        }

        return Ok(ident);
    }

    let matrix = dmatrix_from_rows(a)?;

    // To get the full V matrix (n×n), compute SVD of A^T (n×m).
    // The U of A^T equals V of A, giving us all n right singular vectors.
    let at = matrix.transpose();
    let svd_decomp = SVD::new(at, true, false);
    let u_of_at = svd_decomp
        .u
        .as_ref()
        .ok_or(LinalgError::UnsupportedAssumption)?;
    let singular_values = &svd_decomp.singular_values;

    let max_s = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tol = rcond.unwrap_or_else(|| (m.max(n) as f64) * f64::EPSILON) * max_s;

    let rank = singular_values.iter().filter(|&&s| s > tol).count();
    let null_dim = n - rank;

    if null_dim == 0 {
        return Ok(vec![vec![]; n]);
    }

    // Null space = columns rank..n of U(A^T) = columns rank..n of V(A).
    // U(A^T) is n × min(n,m), so for m < n we may not have enough columns.
    // However, since rank <= min(m,n), and null_dim = n - rank, we need columns
    // up to index n-1 in U(A^T). When n > m, U(A^T) has n rows but only min(n,m)=m columns.
    // Columns beyond m correspond to singular value 0, but nalgebra's thin SVD
    // doesn't compute them. In this case, we need to find an orthonormal basis
    // for the remaining dimensions via QR of the existing columns' complement.
    let u_cols = u_of_at.ncols(); // = min(n, m)
    if rank < u_cols {
        // We have enough columns in U(A^T) to extract the null space.
        let mut result = vec![vec![0.0; null_dim]; n];
        for i in 0..n {
            for j in 0..null_dim.min(u_cols - rank) {
                result[i][j] = u_of_at[(i, rank + j)];
            }
        }
        // If null_dim > u_cols - rank, the remaining null-space vectors are
        // orthogonal to all columns of U(A^T). This happens when m < n and rank = m.
        // These extra vectors span the complement and must be computed separately.
        if null_dim > u_cols - rank {
            // Use Gram-Schmidt to find remaining orthonormal vectors.
            let _existing = u_cols; // number of columns in U(A^T)
            let needed = null_dim - (u_cols - rank);
            let extra = gram_schmidt_complement(u_of_at, n, needed);
            for j in 0..needed {
                for i in 0..n {
                    result[i][(u_cols - rank) + j] = extra[j][i];
                }
            }
        }
        Ok(result)
    } else {
        // rank == u_cols: all computed singular vectors are in the range.
        // The entire null space is orthogonal to all of them.
        // This happens when rank = min(n,m) = m < n.
        let mut result = vec![vec![0.0; null_dim]; n];
        let extra = gram_schmidt_complement(u_of_at, n, null_dim);
        for j in 0..null_dim {
            for i in 0..n {
                result[i][j] = extra[j][i];
            }
        }
        Ok(result)
    }
}

/// Find `needed` orthonormal vectors that are orthogonal to all columns of `basis`.
/// `basis` is an n×k matrix (DMatrix). Returns `needed` vectors of length `n`.
fn gram_schmidt_complement(basis: &DMatrix<f64>, n: usize, needed: usize) -> Vec<Vec<f64>> {
    let k = basis.ncols();
    let mut result: Vec<Vec<f64>> = Vec::with_capacity(needed);

    // Try standard basis vectors e_i and orthogonalize against existing basis
    // and previously found complement vectors.
    for candidate_idx in 0..n {
        if result.len() >= needed {
            break;
        }
        // Start with e_{candidate_idx}
        let mut v = vec![0.0; n];
        v[candidate_idx] = 1.0;

        // Orthogonalize against all basis columns.
        for col in 0..k {
            let mut dot = 0.0;
            for r in 0..n {
                dot += v[r] * basis[(r, col)];
            }
            for r in 0..n {
                v[r] -= dot * basis[(r, col)];
            }
        }

        // Orthogonalize against previously found complement vectors.
        for prev in &result {
            let mut dot = 0.0;
            for r in 0..n {
                dot += v[r] * prev[r];
            }
            for r in 0..n {
                v[r] -= dot * prev[r];
            }
        }

        // Normalize.
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
            result.push(v);
        }
    }

    result
}

/// Compute the angles between two subspaces.
///
/// Matches `scipy.linalg.subspace_angles(A, B)`.
///
/// Returns the principal angles (in radians, sorted in decreasing order)
/// between the column spaces of A and B, computed via SVD of Q_A^T Q_B.
pub fn subspace_angles(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<f64>, LinalgError> {
    // Compute orthonormal bases.
    let qa = orth(a, None, options)?;
    let qb = orth(b, None, options)?;

    if qa.is_empty() || qa[0].is_empty() || qb.is_empty() || qb[0].is_empty() {
        return Ok(vec![]);
    }

    let m = qa.len();
    let ka = qa[0].len();
    let kb = qb[0].len();

    // Compute QA^T * QB.
    let mut product = vec![vec![0.0; kb]; ka];
    for i in 0..ka {
        for j in 0..kb {
            let mut sum = 0.0;
            for r in 0..m {
                sum += qa[r][i] * qb[r][j];
            }
            product[i][j] = sum;
        }
    }

    // SVD of the product to get cosines of principal angles.
    let sv = svdvals(&product, options)?;
    let mut angles: Vec<f64> = sv.iter().map(|&s| s.clamp(0.0, 1.0).acos()).collect();
    angles.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)); // descending
    Ok(angles)
}

/// Compute the polar decomposition of a matrix: A = U * P.
///
/// Matches `scipy.linalg.polar(a)`.
///
/// Returns `(U, P)` where U is unitary (or semi-unitary) and P is
/// positive semi-definite Hermitian.
///
/// Computed via SVD: `A = U_svd * S * Vt`, then `U = U_svd * Vt` and `P = V * S * Vt`.
pub fn polar(a: &[Vec<f64>], options: DecompOptions) -> Result<PolarResult, LinalgError> {
    let (m, n) = matrix_shape(a)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if m == 0 || n == 0 {
        return Ok(PolarResult {
            u: vec![vec![]; m],
            p: vec![vec![]; n],
        });
    }

    let svd_result = svd(a, options)?;
    let u_svd = dmatrix_from_rows(&svd_result.u)?;
    let vt = dmatrix_from_rows(&svd_result.vt)?;
    let s = &svd_result.s;
    let k = s.len();

    // U = U_svd * Vt (m × n).
    let u_polar = u_svd * vt.clone();

    // P = V * S * Vt = Vt^T * diag(S) * Vt
    let mut sigma = DMatrix::zeros(k, k);
    for (i, &val) in s.iter().enumerate() {
        sigma[(i, i)] = val;
    }
    let p_polar = vt.transpose() * sigma * vt;

    Ok(PolarResult {
        u: rows_from_dmatrix(&u_polar),
        p: rows_from_dmatrix(&p_polar),
    })
}

// ══════════════════════════════════════════════════════════════════════
// Special Matrix Constructors
// ══════════════════════════════════════════════════════════════════════

/// Construct a Toeplitz matrix.
///
/// Matches `scipy.linalg.toeplitz(c, r)`.
///
/// A Toeplitz matrix has constant diagonals. The first column is `c` and
/// the first row is `r`. If `r` is None, the matrix is symmetric.
pub fn toeplitz(c: &[f64], r: Option<&[f64]>) -> Vec<Vec<f64>> {
    let n = c.len();
    let row = r.unwrap_or(c);
    let m = row.len();

    let mut result = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            result[i][j] = if j >= i {
                if i == 0 && j == 0 { c[0] } else { row[j - i] }
            } else {
                c[i - j]
            };
        }
    }
    result
}

/// Construct a circulant matrix.
///
/// Matches `scipy.linalg.circulant(c)`.
///
/// Each row is a cyclic permutation of the first row.
pub fn circulant(c: &[f64]) -> Vec<Vec<f64>> {
    let n = c.len();
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = c[(j + n - i) % n];
        }
    }
    result
}

/// Construct a Hilbert matrix.
///
/// Matches `scipy.linalg.hilbert(n)`.
///
/// H_{ij} = 1 / (i + j + 1) for i,j starting at 0. The Hilbert matrix
/// is a classic example of an ill-conditioned matrix.
pub fn hilbert(n: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; n]; n];
    for (i, row) in result.iter_mut().enumerate().take(n) {
        for (j, entry) in row.iter_mut().enumerate().take(n) {
            *entry = 1.0 / (i + j + 1) as f64;
        }
    }
    result
}

/// Construct the inverse of a Hilbert matrix.
///
/// Matches `scipy.linalg.invhilbert(n)`.
///
/// Returns the exact inverse (integer entries) computed via the closed-form
/// formula, avoiding numerical inversion of the ill-conditioned Hilbert matrix.
pub fn invhilbert(n: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; n]; n];
    for (i, row) in result.iter_mut().enumerate().take(n) {
        for (j, entry) in row.iter_mut().enumerate().take(n) {
            let mut val = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            val *= ((i + j + 1) as f64)
                * binom(n + i, n - j - 1)
                * binom(n + j, n - i - 1)
                * binom(i + j, i).powi(2);
            *entry = val;
        }
    }
    result
}

/// Binomial coefficient C(n, k) as f64.
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

/// Construct a Hadamard matrix of order n.
///
/// Matches `scipy.linalg.hadamard(n)`.
///
/// n must be a power of 2. The matrix has entries ±1 and satisfies H^T H = n I.
/// Constructed via the Sylvester (recursive) construction.
pub fn hadamard(n: usize) -> Result<Vec<Vec<f64>>, LinalgError> {
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(LinalgError::InvalidArgument {
            detail: format!("n must be a power of 2, got {n}"),
        });
    }

    let mut h = vec![vec![1.0]];
    let mut size = 1;
    while size < n {
        let mut new_h = vec![vec![0.0; 2 * size]; 2 * size];
        for i in 0..size {
            for j in 0..size {
                new_h[i][j] = h[i][j];
                new_h[i][j + size] = h[i][j];
                new_h[i + size][j] = h[i][j];
                new_h[i + size][j + size] = -h[i][j];
            }
        }
        h = new_h;
        size *= 2;
    }
    Ok(h)
}

/// Construct a companion matrix.
///
/// Matches `scipy.linalg.companion(a)`.
///
/// The companion matrix for polynomial `a[0]*x^n + a[1]*x^{n-1} + ... + a[n]`
/// has the coefficients `-a[1:]/a[0]` in the first row and ones on the sub-diagonal.
pub fn companion(a: &[f64]) -> Result<Vec<Vec<f64>>, LinalgError> {
    if a.len() < 2 {
        return Err(LinalgError::InvalidArgument {
            detail: "companion requires at least 2 coefficients".to_string(),
        });
    }
    if a[0].abs() < f64::EPSILON {
        return Err(LinalgError::InvalidArgument {
            detail: "leading coefficient must be nonzero".to_string(),
        });
    }

    let n = a.len() - 1;
    let mut result = vec![vec![0.0; n]; n];

    // First row: -a[1..]/a[0].
    for j in 0..n {
        result[0][j] = -a[j + 1] / a[0];
    }

    // Sub-diagonal: ones.
    for i in 1..n {
        result[i][i - 1] = 1.0;
    }

    Ok(result)
}

/// Construct a block diagonal matrix from a list of matrices.
///
/// Matches `scipy.linalg.block_diag(*arrs)`.
///
/// Arranges the input matrices along the diagonal of a larger matrix,
/// with zeros elsewhere.
pub fn block_diag(blocks: &[Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
    if blocks.is_empty() {
        return vec![];
    }

    let total_rows: usize = blocks.iter().map(|b| b.len()).sum();
    let total_cols: usize = blocks
        .iter()
        .map(|b| if b.is_empty() { 0 } else { b[0].len() })
        .sum();

    let mut result = vec![vec![0.0; total_cols]; total_rows];
    let mut row_offset = 0;
    let mut col_offset = 0;

    for block in blocks {
        let br = block.len();
        let bc = if br > 0 { block[0].len() } else { 0 };
        for i in 0..br {
            for j in 0..bc {
                result[row_offset + i][col_offset + j] = block[i][j];
            }
        }
        row_offset += br;
        col_offset += bc;
    }

    result
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
        let lu = matrix.clone().lu();
        let rcond = fast_rcond_from_lu(&lu, matrix.lp_norm(1), 2);
        assert!(
            rcond > 0.1,
            "well-conditioned matrix should have high rcond, got {rcond}"
        );
    }

    #[test]
    fn fast_rcond_from_lu_ill_conditioned() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1e-15]);
        let lu = matrix.clone().lu();
        let rcond = fast_rcond_from_lu(&lu, matrix.lp_norm(1), 2);
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
    fn solve_svd_fallback_rejects_rank_deficient_square_systems() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        let err = solve_svd_fallback(&a, &b).expect_err("rank-deficient system");
        assert_eq!(err, LinalgError::SingularMatrix);
    }

    #[test]
    fn solve_svd_fallback_warns_on_ill_conditioned_but_full_rank_systems() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1e-14]];
        let b = vec![1.0, 1.0];
        let result = solve_svd_fallback(&a, &b).expect("full-rank system");
        assert!(matches!(result.warning, Some(LinalgWarning::IllConditioned { .. })));
    }

    #[test]
    fn fast_rcond_triangular_matches_exact_upper_2x2_inverse_norm() {
        let a = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
        let rcond = fast_rcond_triangular(&a, false);
        assert!(
            (rcond - 0.25).abs() < 1e-12,
            "expected exact reciprocal condition 0.25, got {rcond}"
        );
    }

    #[test]
    fn fast_rcond_triangular_matches_exact_lower_2x2_inverse_norm() {
        let a = vec![vec![1.0, 0.0], vec![1.0, 1.0]];
        let rcond = fast_rcond_triangular(&a, true);
        assert!(
            (rcond - 0.25).abs() < 1e-12,
            "expected exact reciprocal condition 0.25, got {rcond}"
        );
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

    // ── Error variant coverage tests ──────────────────────────────

    #[test]
    fn error_ragged_matrix() {
        let a = vec![vec![1.0, 2.0], vec![3.0]];
        let b = vec![1.0, 2.0];
        let err = solve(&a, &b, SolveOptions::default()).unwrap_err();
        assert_eq!(err, LinalgError::RaggedMatrix);
    }

    #[test]
    fn error_expected_square_matrix() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![1.0, 2.0];
        let err = solve(&a, &b, SolveOptions::default()).unwrap_err();
        assert_eq!(err, LinalgError::ExpectedSquareMatrix);
    }

    #[test]
    fn error_incompatible_shapes() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0, 2.0, 3.0];
        let err = solve(&a, &b, SolveOptions::default()).unwrap_err();
        assert!(matches!(err, LinalgError::IncompatibleShapes { .. }));
    }

    #[test]
    fn error_singular_matrix() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        let err = solve(&a, &b, SolveOptions::default()).unwrap_err();
        assert_eq!(err, LinalgError::SingularMatrix);
    }

    #[test]
    fn error_invalid_band_shape() {
        let ab = vec![vec![1.0, 2.0]]; // 1 row but l+u+1=3 expected
        let b = vec![1.0, 2.0];
        let err = solve_banded((1, 1), &ab, &b, SolveOptions::default()).unwrap_err();
        assert!(matches!(err, LinalgError::InvalidBandShape { .. }));
    }

    #[test]
    fn error_condition_too_high_hardened() {
        // Near-singular matrix: rcond ≈ 1e-15
        let a = vec![vec![1.0, 0.0], vec![0.0, 1e-15]];
        let b = vec![1.0, 1.0];
        let err = solve(
            &a,
            &b,
            SolveOptions {
                mode: RuntimeMode::Hardened,
                ..SolveOptions::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, LinalgError::ConditionTooHigh { .. }),
            "expected ConditionTooHigh, got {err:?}"
        );
    }

    #[test]
    fn error_not_supported_display() {
        let err = LinalgError::NotSupported {
            detail: "complex transpose not available".into(),
        };
        assert_eq!(err.to_string(), "complex transpose not available");
    }

    #[test]
    fn error_convergence_failure_display() {
        let err = LinalgError::ConvergenceFailure {
            detail: "SVD did not converge".into(),
        };
        assert_eq!(err.to_string(), "SVD did not converge");
    }

    #[test]
    fn error_resource_exhausted_display() {
        let err = LinalgError::ResourceExhausted {
            detail: "dimension too large".into(),
        };
        assert_eq!(err.to_string(), "resource exhausted: dimension too large");
    }

    #[test]
    fn error_invalid_argument_display() {
        let err = LinalgError::InvalidArgument {
            detail: "bad param".into(),
        };
        assert_eq!(err.to_string(), "bad param");
    }

    #[test]
    fn linalg_trace_serializes_to_json() {
        let trace = LinalgTrace {
            operation: "solve",
            matrix_size: (3, 3),
            mode: RuntimeMode::Strict,
            rcond: Some(0.5),
            warning: None,
            error: None,
        };
        let json = serde_json::to_string(&trace).expect("serialize");
        assert!(json.contains("\"operation\":\"solve\""));
        assert!(json.contains("\"matrix_size\":[3,3]"));
    }

    // ── Comprehensive unit tests for bd-3jh.13.5 ─────────────────

    #[test]
    fn solve_3x3_general_system() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
        ];
        let b = vec![14.0, 32.0, 53.0];
        let result = solve(&a, &b, SolveOptions::default()).expect("solve works");
        assert_close_slice(&result.x, &[1.0, 2.0, 3.0], 1e-10, 1e-10);
    }

    #[test]
    fn solve_diagonal_fast_path() {
        let a = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 4.0],
        ];
        let b = vec![4.0, 9.0, 16.0];
        let result = solve(
            &a,
            &b,
            SolveOptions {
                assume_a: Some(MatrixAssumption::Diagonal),
                ..SolveOptions::default()
            },
        )
        .expect("diagonal solve works");
        assert_close_slice(&result.x, &[2.0, 3.0, 4.0], 1e-14, 1e-14);
    }

    #[test]
    fn solve_upper_triangular_fast_path() {
        let a = vec![vec![2.0, 3.0], vec![0.0, 4.0]];
        let b = vec![14.0, 8.0];
        let result = solve(
            &a,
            &b,
            SolveOptions {
                assume_a: Some(MatrixAssumption::UpperTriangular),
                ..SolveOptions::default()
            },
        )
        .expect("upper triangular solve works");
        assert_close_slice(&result.x, &[4.0, 2.0], 1e-14, 1e-14);
    }

    #[test]
    fn solve_triangular_upper_non_transposed() {
        let a = vec![vec![2.0, 3.0], vec![0.0, 4.0]];
        let b = vec![14.0, 8.0];
        let result = solve_triangular(
            &a,
            &b,
            TriangularSolveOptions {
                lower: false,
                ..TriangularSolveOptions::default()
            },
        )
        .expect("upper non-transposed works");
        assert_close_slice(&result.x, &[4.0, 2.0], 1e-14, 1e-14);
    }

    #[test]
    fn solve_triangular_lower_transposed() {
        // L = [[2, 0], [3, 4]], L^T = [[2, 3], [0, 4]]
        // L^T x = b is same as solving upper triangular with L^T
        let a = vec![vec![2.0, 0.0], vec![3.0, 4.0]];
        let b = vec![14.0, 8.0];
        let result = solve_triangular(
            &a,
            &b,
            TriangularSolveOptions {
                lower: true,
                trans: TriangularTranspose::Transpose,
                ..TriangularSolveOptions::default()
            },
        )
        .expect("lower transposed works");
        // L^T x = b => [[2,3],[0,4]] x = [14,8] => x = [4, 2]
        assert_close_slice(&result.x, &[4.0, 2.0], 1e-14, 1e-14);
    }

    #[test]
    fn solve_triangular_unit_diagonal() {
        let a = vec![vec![99.0, 0.0], vec![3.0, 99.0]]; // diagonals ignored
        let b = vec![5.0, 14.0];
        let result = solve_triangular(
            &a,
            &b,
            TriangularSolveOptions {
                lower: true,
                unit_diagonal: true,
                ..TriangularSolveOptions::default()
            },
        )
        .expect("unit diagonal works");
        // [1, 0; 3, 1] x = [5, 14] => x = [5, -1]
        assert_close_slice(&result.x, &[5.0, -1.0], 1e-14, 1e-14);
    }

    #[test]
    fn inv_identity_matrix() {
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = inv(&a, InvOptions::default()).expect("inv works");
        assert_close_matrix(&result.inverse, &a, 1e-14, 1e-14);
    }

    #[test]
    fn inv_singular_matrix_error() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let err = inv(&a, InvOptions::default()).unwrap_err();
        assert_eq!(err, LinalgError::SingularMatrix);
    }

    #[test]
    fn inv_ill_conditioned_emits_warning() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1e-13]];
        let result = inv(&a, InvOptions::default()).expect("inv should succeed");
        assert!(
            result.warning.is_some(),
            "ill-conditioned matrix should emit warning"
        );
    }

    #[test]
    fn det_identity_is_one() {
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let d = det(&a, RuntimeMode::Strict, true).expect("det works");
        assert!((d - 1.0).abs() < 1e-14);
    }

    #[test]
    fn det_singular_is_zero() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let d = det(&a, RuntimeMode::Strict, true).expect("det works");
        assert!(d.abs() < 1e-14, "det of singular should be 0, got {d}");
    }

    #[test]
    fn lstsq_underdetermined_system() {
        // 2x3: more unknowns than equations -> minimum norm solution
        let a = vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
        let b = vec![2.0, 3.0];
        let result = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq works");
        assert_eq!(result.x.len(), 3);
        // Verify residual: A @ x == b
        for (i, row) in a.iter().enumerate() {
            let dot: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
            assert!((dot - b[i]).abs() < 1e-10, "residual too large at row {i}");
        }
    }

    #[test]
    fn lstsq_exact_system() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 7.0];
        let result = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq works");
        assert_close_slice(&result.x, &[3.0, 7.0], 1e-14, 1e-14);
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn lstsq_rank_deficient() {
        let a = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
        let b = vec![2.0, 2.0, 2.0];
        let result = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq works");
        assert_eq!(result.rank, 1, "rank should be 1 for rank-deficient system");
    }

    #[test]
    fn pinv_square_nonsingular_matches_inv() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let pinv_result = pinv(&a, PinvOptions::default()).expect("pinv works");
        let inv_result = inv(&a, InvOptions::default()).expect("inv works");
        assert_close_matrix(
            &pinv_result.pseudo_inverse,
            &inv_result.inverse,
            1e-10,
            1e-10,
        );
    }

    #[test]
    fn pinv_rank_deficient() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let result = pinv(&a, PinvOptions::default()).expect("pinv works");
        assert_eq!(result.rank, 1, "rank should be 1");
    }

    #[test]
    fn pinv_zero_matrix() {
        let a = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let result = pinv(&a, PinvOptions::default()).expect("pinv works");
        assert_eq!(result.rank, 0);
        for row in &result.pseudo_inverse {
            for &v in row {
                assert!(v.abs() < 1e-14, "pinv of zero should be zero");
            }
        }
    }

    #[test]
    fn rcond_threshold_boundary() {
        // Well-conditioned: rcond > 1e-12 -> no warning
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0, 1.0];
        let result = solve(&a, &b, SolveOptions::default()).expect("solve works");
        assert!(result.warning.is_none(), "identity should have no warning");

        // Ill-conditioned: rcond < 1e-12 -> warning
        let a2 = vec![vec![1.0, 0.0], vec![0.0, 1e-13]];
        let b2 = vec![1.0, 1.0];
        let result2 = solve(&a2, &b2, SolveOptions::default()).expect("solve works");
        assert!(
            result2.warning.is_some(),
            "ill-conditioned matrix should emit warning"
        );
    }

    #[test]
    fn solve_large_100x100_system() {
        let n = 100;
        // Diagonally dominant matrix for guaranteed non-singularity
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { n as f64 + 1.0 } else { 1.0 })
                    .collect()
            })
            .collect();
        let x_expected: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b: Vec<f64> = a
            .iter()
            .map(|row| row.iter().zip(&x_expected).map(|(a, x)| a * x).sum())
            .collect();
        let result = solve(&a, &b, SolveOptions::default()).expect("large solve works");
        assert_close_slice(&result.x, &x_expected, 1e-8, 1e-8);
    }

    #[test]
    fn solve_symmetric_positive_definite_assumption() {
        let a = vec![
            vec![4.0, 2.0, 0.0],
            vec![2.0, 5.0, 2.0],
            vec![0.0, 2.0, 6.0],
        ];
        let b = vec![8.0, 13.0, 16.0];
        let result = solve(
            &a,
            &b,
            SolveOptions {
                assume_a: Some(MatrixAssumption::PositiveDefinite),
                ..SolveOptions::default()
            },
        )
        .expect("SPD solve works");
        // Verify A*x = b
        for (i, row) in a.iter().enumerate() {
            let dot: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
            assert!(
                (dot - b[i]).abs() < 1e-10,
                "residual too large at row {i}: {dot} vs {}",
                b[i]
            );
        }
    }

    #[test]
    fn solve_banded_pentadiagonal() {
        // 5-diagonal system (l=2, u=2)
        let ab = vec![
            vec![0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0, 1.0, 1.0],
            vec![4.0, 4.0, 4.0, 4.0, 4.0],
            vec![1.0, 1.0, 1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0, 0.0],
        ];
        let b = vec![7.0, 8.0, 8.0, 8.0, 7.0];
        let result =
            solve_banded((2, 2), &ab, &b, SolveOptions::default()).expect("pentadiagonal works");
        assert_eq!(result.x.len(), 5);
        // Verify via dense reconstruction
        let dense = dense_from_banded(2, 2, &ab, 5);
        for (i, row) in dense.iter().enumerate() {
            let dot: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
            assert!(
                (dot - b[i]).abs() < 1e-10,
                "pentadiagonal residual too large at row {i}"
            );
        }
    }

    #[test]
    fn det_empty_matrix_is_one() {
        let a: Vec<Vec<f64>> = vec![];
        let d = det(&a, RuntimeMode::Strict, true).expect("det of empty");
        assert!((d - 1.0).abs() < 1e-14);
    }

    #[test]
    fn inv_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = inv(&a, InvOptions::default()).expect("inv of empty");
        assert!(result.inverse.is_empty());
    }

    #[test]
    fn pinv_moore_penrose_condition() {
        // Verify: A @ pinv(A) @ A == A (Moore-Penrose condition 1)
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = pinv(&a, PinvOptions::default()).expect("pinv works");
        let a_pinv = &result.pseudo_inverse;
        // Compute A @ A_pinv @ A
        let n_rows = a.len();
        let n_cols = a[0].len();
        let pinv_rows = a_pinv.len();
        // A_pinv is (n_cols x n_rows), so A @ A_pinv is (n_rows x n_rows)
        let mut a_ap = vec![vec![0.0; n_rows]; n_rows];
        for i in 0..n_rows {
            for j in 0..n_rows {
                for k in 0..pinv_rows {
                    a_ap[i][j] += a[i][k] * a_pinv[k][j];
                }
            }
        }
        // (A @ A_pinv) @ A -> (n_rows x n_cols)
        let mut a_ap_a = vec![vec![0.0; n_cols]; n_rows];
        for i in 0..n_rows {
            for j in 0..n_cols {
                for k in 0..n_rows {
                    a_ap_a[i][j] += a_ap[i][k] * a[k][j];
                }
            }
        }
        assert_close_matrix(&a_ap_a, &a, 1e-10, 1e-10);
    }

    // ── LU decomposition tests ──────────────────────────────────────

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn lu_decomposition_pa_equals_lu() {
        let a = vec![
            vec![2.0, 1.0, 1.0],
            vec![4.0, 3.0, 3.0],
            vec![8.0, 7.0, 9.0],
        ];
        let result = lu(&a, DecompOptions::default()).expect("lu works");
        // Verify P*A = L*U
        let n = a.len();
        for (i, (p_row, l_row)) in result.p.iter().zip(result.l.iter()).enumerate() {
            for j in 0..n {
                let pa: f64 = (0..n).map(|k| p_row[k] * a[k][j]).sum();
                let lu_val: f64 = (0..n).map(|k| l_row[k] * result.u[k][j]).sum();
                assert!(
                    (pa - lu_val).abs() < 1e-10,
                    "PA != LU at [{i}][{j}]: {pa} vs {lu_val}"
                );
            }
        }
    }

    #[test]
    fn lu_l_is_unit_lower_triangular() {
        let a = vec![vec![4.0, 3.0], vec![6.0, 3.0]];
        let result = lu(&a, DecompOptions::default()).expect("lu works");
        // L diagonal should be 1.0
        for i in 0..2 {
            assert!(
                (result.l[i][i] - 1.0).abs() < 1e-14,
                "L diagonal should be 1"
            );
        }
        // L upper triangle should be 0.0
        assert!(result.l[0][1].abs() < 1e-14, "L should be lower triangular");
    }

    #[test]
    fn lu_u_is_upper_triangular() {
        let a = vec![vec![4.0, 3.0], vec![6.0, 3.0]];
        let result = lu(&a, DecompOptions::default()).expect("lu works");
        assert!(result.u[1][0].abs() < 1e-14, "U should be upper triangular");
    }

    #[test]
    fn lu_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = lu(&a, DecompOptions::default()).expect("lu of empty");
        assert!(result.p.is_empty());
        assert!(result.l.is_empty());
        assert!(result.u.is_empty());
    }

    #[test]
    fn lu_factor_and_solve_roundtrip() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let factor = lu_factor(&a, DecompOptions::default()).expect("lu_factor works");
        let result = lu_solve(&factor, &b).expect("lu_solve works");
        assert_close_slice(&result.x, &[0.0, 2.5], 1e-12, 1e-12);
    }

    #[test]
    fn lu_solve_incompatible_shapes() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let factor = lu_factor(&a, DecompOptions::default()).expect("lu_factor works");
        let err = lu_solve(&factor, &[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, LinalgError::IncompatibleShapes { .. }));
    }

    // ── QR decomposition tests ──────────────────────────────────────

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn qr_decomposition_a_equals_qr() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = qr(&a, DecompOptions::default()).expect("qr works");
        let cols = a[0].len();
        // Verify A = Q*R
        for (i, (a_row, q_row)) in a.iter().zip(result.q.iter()).enumerate() {
            for j in 0..cols {
                let qr_val: f64 = (0..result.r.len()).map(|k| q_row[k] * result.r[k][j]).sum();
                assert!(
                    (a_row[j] - qr_val).abs() < 1e-10,
                    "A != QR at [{i}][{j}]: {} vs {qr_val}",
                    a_row[j]
                );
            }
        }
    }

    #[test]
    fn qr_q_is_orthogonal() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = qr(&a, DecompOptions::default()).expect("qr works");
        let n = result.q.len();
        // Q^T * Q should be identity
        for i in 0..n {
            for j in i..n {
                let dot: f64 = result.q.iter().map(|row| row[i] * row[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q not orthogonal at [{i}][{j}]: {dot} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn qr_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = qr(&a, DecompOptions::default()).expect("qr of empty");
        assert!(result.q.is_empty());
        assert!(result.r.is_empty());
    }

    // ── SVD tests ───────────────────────────────────────────────────

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn svd_decomposition_a_equals_usv() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = svd(&a, DecompOptions::default()).expect("svd works");
        let cols = a[0].len();
        let k = result.s.len();
        // Reconstruct A from U * diag(S) * Vt
        for (i, (a_row, u_row)) in a.iter().zip(result.u.iter()).enumerate() {
            for j in 0..cols {
                let val: f64 = (0..k)
                    .map(|l| u_row[l] * result.s[l] * result.vt[l][j])
                    .sum();
                assert!(
                    (a_row[j] - val).abs() < 1e-10,
                    "A != U*S*Vt at [{i}][{j}]: {} vs {val}",
                    a_row[j]
                );
            }
        }
    }

    #[test]
    fn svd_singular_values_non_negative_and_descending() {
        let a = vec![vec![3.0, 2.0, 2.0], vec![2.0, 3.0, -2.0]];
        let result = svd(&a, DecompOptions::default()).expect("svd works");
        for s in &result.s {
            assert!(*s >= 0.0, "singular value should be non-negative: {s}");
        }
        for w in result.s.windows(2) {
            assert!(w[0] >= w[1], "singular values should be descending");
        }
    }

    #[test]
    fn svdvals_matches_svd() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let full_result = svd(&a, DecompOptions::default()).expect("svd works");
        let vals = svdvals(&a, DecompOptions::default()).expect("svdvals works");
        assert_close_slice(&vals, &full_result.s, 1e-14, 1e-14);
    }

    #[test]
    fn svd_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = svd(&a, DecompOptions::default()).expect("svd of empty");
        assert!(result.u.is_empty());
        assert!(result.s.is_empty());
        assert!(result.vt.is_empty());
    }

    // ── Cholesky tests ──────────────────────────────────────────────

    #[test]
    fn cholesky_lower_factor_reconstructs_a() {
        let a = vec![
            vec![4.0, 2.0, 0.0],
            vec![2.0, 5.0, 2.0],
            vec![0.0, 2.0, 6.0],
        ];
        let result = cholesky(&a, true, DecompOptions::default()).expect("cholesky works");
        let n = a.len();
        // Verify L * L^T = A
        for (i, (a_row, l_row)) in a.iter().zip(result.factor.iter()).enumerate() {
            for (j, &a_ij) in a_row.iter().enumerate() {
                let val: f64 = (0..n).map(|k| l_row[k] * result.factor[j][k]).sum();
                assert!(
                    (a_ij - val).abs() < 1e-10,
                    "L*L^T != A at [{i}][{j}]: {a_ij} vs {val}"
                );
            }
        }
    }

    #[test]
    fn cholesky_upper_factor() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 5.0]];
        let result = cholesky(&a, false, DecompOptions::default()).expect("cholesky upper works");
        // Verify U^T * U = A
        for (i, a_row) in a.iter().enumerate() {
            for (j, &a_ij) in a_row.iter().enumerate() {
                let val: f64 = result.factor.iter().map(|row| row[i] * row[j]).sum();
                assert!(
                    (a_ij - val).abs() < 1e-10,
                    "U^T*U != A at [{i}][{j}]: {a_ij} vs {val}"
                );
            }
        }
    }

    #[test]
    fn cholesky_not_positive_definite() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]]; // not PD
        let err = cholesky(&a, true, DecompOptions::default()).unwrap_err();
        assert!(matches!(err, LinalgError::InvalidArgument { .. }));
    }

    #[test]
    fn cho_factor_and_solve() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 5.0]];
        let b = vec![8.0, 7.0];
        let factor = cho_factor(&a, DecompOptions::default()).expect("cho_factor works");
        let result = cho_solve(&factor, &b).expect("cho_solve works");
        // Verify A * x = b
        for (i, row) in a.iter().enumerate() {
            let dot: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
            assert!(
                (dot - b[i]).abs() < 1e-10,
                "A*x != b at row {i}: {dot} vs {}",
                b[i]
            );
        }
    }

    #[test]
    fn cholesky_identity() {
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = cholesky(&a, true, DecompOptions::default()).expect("cholesky of I");
        assert_close_matrix(&result.factor, &a, 1e-14, 1e-14);
    }

    // ── Eigenvalue decomposition tests ──────────────────────────────

    #[test]
    fn eigh_symmetric_eigenvalues_ascending() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let result = eigh(&a, DecompOptions::default()).expect("eigh works");
        // Eigenvalues should be ascending
        for w in result.eigenvalues.windows(2) {
            assert!(w[0] <= w[1], "eigenvalues should be ascending");
        }
        // Known eigenvalues for this matrix: (5 ± sqrt(5)) / 2
        let expected_1 = (5.0 - 5.0_f64.sqrt()) / 2.0;
        let expected_2 = (5.0 + 5.0_f64.sqrt()) / 2.0;
        assert!(
            (result.eigenvalues[0] - expected_1).abs() < 1e-10,
            "eigenvalue 0: {} vs {expected_1}",
            result.eigenvalues[0]
        );
        assert!(
            (result.eigenvalues[1] - expected_2).abs() < 1e-10,
            "eigenvalue 1: {} vs {expected_2}",
            result.eigenvalues[1]
        );
    }

    #[test]
    fn eigh_eigenvectors_orthogonal() {
        let a = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let result = eigh(&a, DecompOptions::default()).expect("eigh works");
        let n = a.len();
        // V^T * V should be identity
        for i in 0..n {
            for j in i..n {
                let dot: f64 = result.eigenvectors.iter().map(|row| row[i] * row[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "eigenvectors not orthogonal at [{i}][{j}]: {dot} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn eigh_av_equals_v_lambda() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let result = eigh(&a, DecompOptions::default()).expect("eigh works");
        let n = a.len();
        // For each eigenvalue/eigenvector pair: A*v = lambda*v
        for col in 0..n {
            let lambda = result.eigenvalues[col];
            for (i, a_row) in a.iter().enumerate() {
                let av: f64 = (0..n).map(|k| a_row[k] * result.eigenvectors[k][col]).sum();
                let lv = lambda * result.eigenvectors[i][col];
                assert!(
                    (av - lv).abs() < 1e-10,
                    "A*v != lambda*v at eigenvector {col}, row {i}: {av} vs {lv}"
                );
            }
        }
    }

    #[test]
    fn eigvalsh_matches_eigh() {
        let a = vec![vec![5.0, 2.0], vec![2.0, 3.0]];
        let full_result = eigh(&a, DecompOptions::default()).expect("eigh works");
        let vals = eigvalsh(&a, DecompOptions::default()).expect("eigvalsh works");
        assert_close_slice(&vals, &full_result.eigenvalues, 1e-14, 1e-14);
    }

    #[test]
    fn eig_general_eigenvalues() {
        let a = vec![vec![0.0, -1.0], vec![1.0, 0.0]]; // rotation 90 deg
        let result = eig(&a, DecompOptions::default()).expect("eig works");
        // Eigenvalues should be ±i (purely imaginary)
        assert_eq!(result.eigenvalues_re.len(), 2);
        for re in &result.eigenvalues_re {
            assert!(re.abs() < 1e-10, "real part should be ~0, got {re}");
        }
        let mut im_sorted: Vec<f64> = result.eigenvalues_im.clone();
        im_sorted.sort_by(|a, b| a.total_cmp(b));
        assert!(
            (im_sorted[0] - (-1.0)).abs() < 1e-10,
            "imaginary part should be -1"
        );
        assert!(
            (im_sorted[1] - 1.0).abs() < 1e-10,
            "imaginary part should be +1"
        );
    }

    #[test]
    fn eig_real_eigenvalues_symmetric() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let result = eig(&a, DecompOptions::default()).expect("eig works");
        // Symmetric -> all eigenvalues real
        for im in &result.eigenvalues_im {
            assert!(im.abs() < 1e-10, "symmetric eigenvalues should be real");
        }
        let mut re_sorted: Vec<f64> = result.eigenvalues_re.clone();
        re_sorted.sort_by(|a, b| a.total_cmp(b));
        assert!((re_sorted[0] - 1.0).abs() < 1e-10);
        assert!((re_sorted[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eigvals_matches_eig() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let full_result = eig(&a, DecompOptions::default()).expect("eig works");
        let (re, im) = eigvals(&a, DecompOptions::default()).expect("eigvals works");
        assert_close_slice(&re, &full_result.eigenvalues_re, 1e-14, 1e-14);
        assert_close_slice(&im, &full_result.eigenvalues_im, 1e-14, 1e-14);
    }

    #[test]
    fn eig_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = eig(&a, DecompOptions::default()).expect("eig of empty");
        assert!(result.eigenvalues_re.is_empty());
    }

    #[test]
    fn eigh_identity_eigenvalues() {
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = eigh(&a, DecompOptions::default()).expect("eigh of I");
        for &val in &result.eigenvalues {
            assert!((val - 1.0).abs() < 1e-14, "eigenvalue of I should be 1");
        }
    }

    // ── Schur decomposition tests ────────────────────────────────────

    #[test]
    fn schur_a_equals_ztzt() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 0.0],
        ];
        let result = schur(&a, DecompOptions::default()).expect("schur works");
        let n = a.len();
        // Verify A = Z * T * Z^T
        for (i, a_row) in a.iter().enumerate() {
            for (j, &a_ij) in a_row.iter().enumerate() {
                let mut val = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        val += result.z[i][k] * result.t[k][l] * result.z[j][l];
                    }
                }
                assert!(
                    (a_ij - val).abs() < 1e-10,
                    "A != Z*T*Z^T at [{i}][{j}]: {a_ij} vs {val}"
                );
            }
        }
    }

    #[test]
    fn schur_z_is_orthogonal() {
        let a = vec![vec![4.0, 1.0], vec![2.0, 3.0]];
        let result = schur(&a, DecompOptions::default()).expect("schur works");
        let n = result.z.len();
        for i in 0..n {
            for j in i..n {
                let dot: f64 = result.z.iter().map(|row| row[i] * row[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Z not orthogonal at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn schur_t_is_quasi_upper_triangular() {
        let a = vec![vec![4.0, 1.0], vec![2.0, 3.0]];
        let result = schur(&a, DecompOptions::default()).expect("schur works");
        // For 2x2, T should be upper triangular or have a 2x2 block
        // In general, elements below the first sub-diagonal should be zero
        let n = result.t.len();
        for i in 2..n {
            for j in 0..i.saturating_sub(1) {
                assert!(
                    result.t[i][j].abs() < 1e-10,
                    "T[{i}][{j}] should be zero: {}",
                    result.t[i][j]
                );
            }
        }
    }

    #[test]
    fn schur_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = schur(&a, DecompOptions::default()).expect("schur of empty");
        assert!(result.z.is_empty());
        assert!(result.t.is_empty());
    }

    // ── Hessenberg decomposition tests ──────────────────────────────

    #[test]
    fn hessenberg_a_equals_qhqt() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 0.0],
        ];
        let result = hessenberg(&a, DecompOptions::default()).expect("hessenberg works");
        let n = a.len();
        // Verify A = Q * H * Q^T
        for (i, a_row) in a.iter().enumerate() {
            for (j, &a_ij) in a_row.iter().enumerate() {
                let mut val = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        val += result.q[i][k] * result.h[k][l] * result.q[j][l];
                    }
                }
                assert!(
                    (a_ij - val).abs() < 1e-10,
                    "A != Q*H*Q^T at [{i}][{j}]: {a_ij} vs {val}"
                );
            }
        }
    }

    #[test]
    fn hessenberg_h_is_upper_hessenberg() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 0.0],
        ];
        let result = hessenberg(&a, DecompOptions::default()).expect("hessenberg works");
        let n = result.h.len();
        // Elements below the first sub-diagonal should be zero
        for i in 2..n {
            for j in 0..i.saturating_sub(1) {
                assert!(
                    result.h[i][j].abs() < 1e-10,
                    "H[{i}][{j}] should be zero: {}",
                    result.h[i][j]
                );
            }
        }
    }

    #[test]
    fn hessenberg_q_is_orthogonal() {
        let a = vec![
            vec![4.0, 1.0, 0.0],
            vec![2.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let result = hessenberg(&a, DecompOptions::default()).expect("hessenberg works");
        let n = result.q.len();
        for i in 0..n {
            for j in i..n {
                let dot: f64 = result.q.iter().map(|row| row[i] * row[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q not orthogonal at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn hessenberg_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = hessenberg(&a, DecompOptions::default()).expect("hessenberg of empty");
        assert!(result.q.is_empty());
        assert!(result.h.is_empty());
    }

    // ── Matrix exponential tests ──────────────────────────────────────

    #[test]
    fn expm_identity() {
        // exp(0) = I
        let a = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let result = expm(&a, DecompOptions::default()).expect("expm works");
        assert_close_matrix(&result, &[vec![1.0, 0.0], vec![0.0, 1.0]], 1e-12, 1e-12);
    }

    #[test]
    fn expm_diagonal() {
        // exp(diag(1, 2)) = diag(e, e²)
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let result = expm(&a, DecompOptions::default()).expect("expm works");
        assert!(
            (result[0][0] - std::f64::consts::E).abs() < 1e-10,
            "expm[0][0] should be e, got {}",
            result[0][0]
        );
        assert!(
            (result[1][1] - std::f64::consts::E.powi(2)).abs() < 1e-10,
            "expm[1][1] should be e², got {}",
            result[1][1]
        );
        assert!(result[0][1].abs() < 1e-12);
        assert!(result[1][0].abs() < 1e-12);
    }

    #[test]
    fn expm_nilpotent() {
        // A = [[0, 1], [0, 0]], exp(A) = [[1, 1], [0, 1]]
        let a = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
        let result = expm(&a, DecompOptions::default()).expect("expm works");
        assert_close_matrix(&result, &[vec![1.0, 1.0], vec![0.0, 1.0]], 1e-12, 1e-12);
    }

    #[test]
    fn expm_rotation() {
        // A = [[0, -pi/2], [pi/2, 0]], exp(A) = [[0, -1], [1, 0]] (90° rotation)
        let half_pi = std::f64::consts::FRAC_PI_2;
        let a = vec![vec![0.0, -half_pi], vec![half_pi, 0.0]];
        let result = expm(&a, DecompOptions::default()).expect("expm works");
        assert!((result[0][0] - 0.0).abs() < 1e-10, "cos(pi/2) should be ~0");
        assert!(
            (result[0][1] - (-1.0)).abs() < 1e-10,
            "-sin(pi/2) should be -1"
        );
        assert!((result[1][0] - 1.0).abs() < 1e-10, "sin(pi/2) should be 1");
        assert!((result[1][1] - 0.0).abs() < 1e-10, "cos(pi/2) should be ~0");
    }

    #[test]
    fn expm_empty() {
        let a: Vec<Vec<f64>> = vec![];
        let result = expm(&a, DecompOptions::default()).expect("expm of empty");
        assert!(result.is_empty());
    }

    #[test]
    fn expm_scalar() {
        // exp([[3]]) = [[e³]]
        let a = vec![vec![3.0]];
        let result = expm(&a, DecompOptions::default()).expect("expm of scalar");
        assert!(
            (result[0][0] - 3.0_f64.exp()).abs() < 1e-10,
            "expm of scalar should be e³"
        );
    }

    #[test]
    fn expm_non_square_error() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let err = expm(&a, DecompOptions::default()).unwrap_err();
        assert_eq!(err, LinalgError::ExpectedSquareMatrix);
    }

    #[test]
    fn logm_upper_triangular_distinct_diagonal_matches_closed_form() {
        let a = vec![vec![2.0, 1.0], vec![0.0, 3.0]];
        let result = logm(&a, DecompOptions::default()).expect("logm works");
        let expected = vec![
            vec![2.0_f64.ln(), 3.0_f64.ln() - 2.0_f64.ln()],
            vec![0.0, 3.0_f64.ln()],
        ];
        assert_close_matrix(&result, &expected, 1e-12, 1e-12);
    }

    #[test]
    fn logm_upper_triangular_repeated_diagonal_uses_jordan_limit() {
        let a = vec![vec![2.0, 1.0], vec![0.0, 2.0]];
        let result = logm(&a, DecompOptions::default()).expect("logm works");
        let expected = vec![vec![2.0_f64.ln(), 0.5], vec![0.0, 2.0_f64.ln()]];
        assert_close_matrix(&result, &expected, 1e-12, 1e-12);
    }

    // ── Norm and rank tests ─────────────────────────────────────────

    #[test]
    fn norm_frobenius_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = norm(&a, NormKind::Fro, DecompOptions::default()).expect("norm works");
        assert!((n - 2.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn norm_spectral_diagonal() {
        let a = vec![vec![3.0, 0.0], vec![0.0, 5.0]];
        let n = norm(&a, NormKind::Spectral, DecompOptions::default()).expect("norm works");
        assert!(
            (n - 5.0).abs() < 1e-10,
            "spectral norm should be 5, got {n}"
        );
    }

    #[test]
    fn norm_one_norm() {
        let a = vec![vec![1.0, -7.0], vec![-2.0, 3.0]];
        let n = norm(&a, NormKind::One, DecompOptions::default()).expect("norm works");
        // Max column sum: col0 = 1+2=3, col1 = 7+3=10
        assert!((n - 10.0).abs() < 1e-14, "1-norm should be 10, got {n}");
    }

    #[test]
    fn norm_inf_norm() {
        let a = vec![vec![1.0, -7.0], vec![-2.0, 3.0]];
        let n = norm(&a, NormKind::Inf, DecompOptions::default()).expect("norm works");
        // Max row sum: row0 = 1+7=8, row1 = 2+3=5
        assert!((n - 8.0).abs() < 1e-14, "inf-norm should be 8, got {n}");
    }

    #[test]
    fn matrix_rank_full_rank() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let r = matrix_rank(&a, None, DecompOptions::default()).expect("rank works");
        assert_eq!(r, 2);
    }

    #[test]
    fn matrix_rank_rank_deficient() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let r = matrix_rank(&a, None, DecompOptions::default()).expect("rank works");
        assert_eq!(r, 1);
    }

    #[test]
    fn matrix_rank_zero_matrix() {
        let a = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let r = matrix_rank(&a, None, DecompOptions::default()).expect("rank works");
        assert_eq!(r, 0);
    }

    #[test]
    fn matrix_rank_rectangular() {
        let a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let r = matrix_rank(&a, None, DecompOptions::default()).expect("rank works");
        assert_eq!(r, 2);
    }

    // ── Subspace operation tests ───────────────────────────────────

    #[test]
    fn orth_full_rank() {
        // 3×3 identity → orth should return 3 orthonormal columns
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = orth(&a, None, DecompOptions::default()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn orth_rank_deficient() {
        // Rank-1 matrix: all rows are multiples
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let result = orth(&a, None, DecompOptions::default()).unwrap();
        // Should return 1 orthonormal column
        assert_eq!(result[0].len(), 1);
        // Column should be unit norm
        let norm: f64 = result.iter().map(|r| r[0] * r[0]).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "orth column not unit norm: {norm}"
        );
    }

    #[test]
    fn null_space_full_rank() {
        // Full-rank 3×3 → null space should be empty
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = null_space(&a, None, DecompOptions::default()).unwrap();
        // Each row should have 0 null-space columns
        assert!(
            result.is_empty() || result[0].is_empty(),
            "full-rank matrix should have empty null space"
        );
    }

    #[test]
    fn null_space_rank_deficient() {
        // [[1, 2], [2, 4]] has rank 1, null space dim = 1
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let result = null_space(&a, None, DecompOptions::default()).unwrap();
        assert_eq!(result.len(), 2); // n=2 rows
        assert_eq!(result[0].len(), 1); // 1 null-space vector
        // Verify A * ns ≈ 0
        let ns = [result[0][0], result[1][0]];
        let ax0 = a[0][0] * ns[0] + a[0][1] * ns[1];
        let ax1 = a[1][0] * ns[0] + a[1][1] * ns[1];
        assert!(ax0.abs() < 1e-10, "A*ns[0] = {ax0}");
        assert!(ax1.abs() < 1e-10, "A*ns[1] = {ax1}");
    }

    #[test]
    fn subspace_angles_identical() {
        // Same subspace → angle should be 0
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
        let angles = subspace_angles(&a, &a, DecompOptions::default()).unwrap();
        for &angle in &angles {
            assert!(angle.abs() < 1e-10, "identical subspace angle: {angle}");
        }
    }

    #[test]
    fn subspace_angles_orthogonal() {
        // Two orthogonal 1-D subspaces in R^2
        let a = vec![vec![1.0], vec![0.0]]; // x-axis
        let b = vec![vec![0.0], vec![1.0]]; // y-axis
        let angles = subspace_angles(&a, &b, DecompOptions::default()).unwrap();
        assert!(!angles.is_empty());
        let half_pi = std::f64::consts::FRAC_PI_2;
        assert!(
            (angles[0] - half_pi).abs() < 1e-10,
            "orthogonal angle: {}, expected π/2",
            angles[0]
        );
    }

    #[test]
    fn polar_identity() {
        // Polar decomposition of identity: U = I, P = I
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = polar(&a, DecompOptions::default()).unwrap();
        let u = result.u;
        let p = result.p;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (u[i][j] - expected).abs() < 1e-10,
                    "U[{i}][{j}] = {}, expected {expected}",
                    u[i][j]
                );
                assert!(
                    (p[i][j] - expected).abs() < 1e-10,
                    "P[{i}][{j}] = {}, expected {expected}",
                    p[i][j]
                );
            }
        }
    }

    #[test]
    fn polar_reconstruction() {
        // A = U * P, verify the reconstruction
        let a = vec![vec![3.0, 1.0], vec![1.0, 4.0]];
        let result = polar(&a, DecompOptions::default()).unwrap();
        let u = result.u;
        let p = result.p;

        // Reconstruct: U * P
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u[i][k] * p[k][j];
                }
                assert!(
                    (sum - a[i][j]).abs() < 1e-8,
                    "UP[{i}][{j}] = {sum}, A[{i}][{j}] = {}",
                    a[i][j]
                );
            }
        }
    }

    #[test]
    fn polar_p_is_symmetric_positive() {
        let a = vec![vec![2.0, 1.0], vec![-1.0, 3.0]];
        let result = polar(&a, DecompOptions::default()).unwrap();
        let p = result.p;
        // P should be symmetric
        assert!(
            (p[0][1] - p[1][0]).abs() < 1e-10,
            "P not symmetric: P[0][1]={}, P[1][0]={}",
            p[0][1],
            p[1][0]
        );
        // P should be positive semi-definite (eigenvalues >= 0)
        // For 2×2: trace > 0 and det >= 0
        let trace = p[0][0] + p[1][1];
        let det_val = p[0][0] * p[1][1] - p[0][1] * p[1][0];
        assert!(trace > -1e-10, "P trace: {trace}");
        assert!(det_val > -1e-10, "P determinant: {det_val}");
    }

    // ── Special matrix constructor tests ───────────────────────────

    #[test]
    fn toeplitz_symmetric() {
        let c = vec![1.0, 2.0, 3.0];
        let t = toeplitz(&c, None);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(t[1], vec![2.0, 1.0, 2.0]);
        assert_eq!(t[2], vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn toeplitz_asymmetric() {
        let c = vec![1.0, 2.0, 3.0];
        let r = vec![1.0, 4.0, 5.0];
        let t = toeplitz(&c, Some(&r));
        assert_eq!(t[0], vec![1.0, 4.0, 5.0]);
        assert_eq!(t[1], vec![2.0, 1.0, 4.0]);
        assert_eq!(t[2], vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn circulant_basic() {
        let c = vec![1.0, 2.0, 3.0];
        let m = circulant(&c);
        assert_eq!(m[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(m[1], vec![3.0, 1.0, 2.0]);
        assert_eq!(m[2], vec![2.0, 3.0, 1.0]);
    }

    #[test]
    fn hilbert_3x3() {
        let h = hilbert(3);
        assert!((h[0][0] - 1.0).abs() < 1e-12);
        assert!((h[0][1] - 0.5).abs() < 1e-12);
        assert!((h[0][2] - 1.0 / 3.0).abs() < 1e-12);
        assert!((h[1][0] - 0.5).abs() < 1e-12);
        assert!((h[1][1] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn invhilbert_times_hilbert_is_identity() {
        let n = 4;
        let h = hilbert(n);
        let ih = invhilbert(n);
        // Product should be approximately identity.
        for (i, ih_row) in ih.iter().enumerate().take(n) {
            for (j, _) in h.iter().enumerate().take(n) {
                let mut sum = 0.0;
                for (k, &ih_val) in ih_row.iter().enumerate().take(n) {
                    sum += ih_val * h[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-6,
                    "invhilbert*hilbert[{i}][{j}] = {sum}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn hadamard_4() {
        let h = hadamard(4).unwrap();
        assert_eq!(h.len(), 4);
        // H^T H = 4 I
        for i in 0..4 {
            for j in 0..4 {
                let mut dot = 0.0;
                for row in h.iter().take(4) {
                    dot += row[i] * row[j];
                }
                let expected = if i == j { 4.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "H^T H[{i}][{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn hadamard_non_power_of_2_rejected() {
        assert!(hadamard(3).is_err());
        assert!(hadamard(6).is_err());
    }

    #[test]
    fn companion_polynomial() {
        // p(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        // Roots are eigenvalues of companion matrix
        let a = vec![1.0, -6.0, 11.0, -6.0];
        let c = companion(&a).unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c[0], vec![6.0, -11.0, 6.0]); // -a[1..]/a[0]
        assert_eq!(c[1], vec![1.0, 0.0, 0.0]);
        assert_eq!(c[2], vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn companion_too_few_coeffs() {
        assert!(companion(&[1.0]).is_err());
    }

    #[test]
    fn block_diag_basic() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0]];
        let bd = block_diag(&[a, b]);
        assert_eq!(bd.len(), 3);
        assert_eq!(bd[0].len(), 3);
        assert_eq!(bd[0], vec![1.0, 2.0, 0.0]);
        assert_eq!(bd[1], vec![3.0, 4.0, 0.0]);
        assert_eq!(bd[2], vec![0.0, 0.0, 5.0]);
    }

    #[test]
    fn block_diag_empty() {
        let bd = block_diag(&[]);
        assert!(bd.is_empty());
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

        #[test]
        fn det_product_property(
            a in arb_invertible_2x2(),
            b_mat in arb_invertible_2x2()
        ) {
            let det_a = det(&a, RuntimeMode::Strict, true).expect("det(A)");
            let det_b = det(&b_mat, RuntimeMode::Strict, true).expect("det(B)");
            // Compute A @ B
            let ab = vec![
                vec![
                    a[0][0] * b_mat[0][0] + a[0][1] * b_mat[1][0],
                    a[0][0] * b_mat[0][1] + a[0][1] * b_mat[1][1],
                ],
                vec![
                    a[1][0] * b_mat[0][0] + a[1][1] * b_mat[1][0],
                    a[1][0] * b_mat[0][1] + a[1][1] * b_mat[1][1],
                ],
            ];
            let det_ab = det(&ab, RuntimeMode::Strict, true).expect("det(AB)");
            prop_assert!((det_ab - det_a * det_b).abs() < 1e-6,
                "det(AB) should equal det(A)*det(B): {det_ab} vs {}", det_a * det_b);
        }

        #[test]
        fn lstsq_residual_minimality(a in arb_invertible_2x2(), b in arb_vec2()) {
            // For square non-singular: lstsq should give exact solution
            let result = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq works");
            for (i, row) in a.iter().enumerate() {
                let dot: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
                prop_assert!((dot - b[i]).abs() < 1e-8,
                    "A*x should equal b for exact system: row {i}");
            }
        }

        #[test]
        fn solve_triangular_matches_solve_for_triangular_input(
            a11 in 1.0..10.0_f64,
            a12 in -10.0..10.0_f64,
            a22 in 1.0..10.0_f64,
            b in arb_vec2()
        ) {
            let a = vec![vec![a11, a12], vec![0.0, a22]]; // upper triangular
            let solve_result = solve(&a, &b, SolveOptions {
                assume_a: Some(MatrixAssumption::UpperTriangular),
                ..SolveOptions::default()
            }).expect("solve with assumption works");
            let tri_result = solve_triangular(&a, &b, TriangularSolveOptions {
                lower: false,
                ..TriangularSolveOptions::default()
            }).expect("solve_triangular works");
            for (i, (s, t)) in solve_result.x.iter().zip(tri_result.x.iter()).enumerate() {
                prop_assert!((s - t).abs() < 1e-10,
                    "solve and solve_triangular should match: [{i}] {s} vs {t}");
            }
        }

        #[test]
        fn pinv_rank_bounded(
            a in arb_invertible_2x2()
        ) {
            let result = pinv(&a, PinvOptions::default()).expect("pinv works");
            prop_assert!(result.rank <= 2,
                "rank should be <= min(m,n)=2, got {}", result.rank);
        }

        #[test]
        fn rcond_is_bounded(a in arb_invertible_2x2()) {
            let matrix = dmatrix_from_rows(&a).expect("valid matrix");
            let lu = matrix.clone().lu();
            let rcond = fast_rcond_from_lu(&lu, matrix.lp_norm(1), 2);
            prop_assert!(rcond >= 0.0, "rcond should be >= 0, got {rcond}");
            prop_assert!(rcond <= 1.0, "rcond should be <= 1, got {rcond}");
        }
    }

    #[test]
    fn test_solve_transposed_upper_triangular() {
        // Upper triangular matrix A
        // [ 2, 1 ]
        // [ 0, 2 ]
        let a = vec![vec![2.0, 1.0], vec![0.0, 2.0]];

        // b = [ 5, 4 ]
        let b = vec![5.0, 4.0];

        // A^T x = b
        // [ 2, 0 ] [ x1 ] = [ 5 ]
        // [ 1, 2 ] [ x2 ] = [ 4 ]
        // 2*x1 = 5 => x1 = 2.5
        // 1*x1 + 2*x2 = 4 => 2.5 + 2*x2 = 4 => 2*x2 = 1.5 => x2 = 0.75
        // Expected x = [ 2.5, 0.75 ]

        let options = SolveOptions {
            assume_a: Some(MatrixAssumption::UpperTriangular),
            transposed: true,
            ..SolveOptions::default()
        };

        let result = solve(&a, &b, options).expect("solve works");
        let expected = [2.5, 0.75];

        for (i, (&r, &e)) in result.x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-10,
                "FAILURE at index {i}: got {r}, expected {e}"
            );
        }
    }

    #[test]
    fn test_solve_with_casp_transposed_upper_triangular() {
        use fsci_runtime::{RuntimeMode, SolverPortfolio};

        let a = vec![vec![2.0, 1.0], vec![0.0, 2.0]];
        let b = vec![5.0, 4.0];
        let options = SolveOptions {
            assume_a: Some(MatrixAssumption::UpperTriangular),
            transposed: true,
            ..SolveOptions::default()
        };

        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let result =
            solve_with_casp(&a, &b, options, &mut portfolio).expect("solve_with_casp works");
        let expected = [2.5, 0.75];

        for (i, (&r, &e)) in result.x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-10,
                "FAILURE at index {i}: got {r}, expected {e}"
            );
        }

        // Also verify the action was TriangularFastPath
        let evidence = portfolio.serialize_jsonl();
        assert!(evidence.contains("TriangularFastPath"));
    }

    // ── LDL decomposition tests ─────────────────────────────────────

    fn verify_ldl_reconstruction(a: &[Vec<f64>], result: &LdlResult) {
        let n = result.d.len();
        for (i, row_a) in a.iter().enumerate().take(n) {
            for (j, &aij) in row_a.iter().enumerate().take(n) {
                let sum: f64 = (0..n)
                    .map(|k| result.l[i][k] * result.d[k] * result.l[j][k])
                    .sum();
                assert!(
                    (sum - aij).abs() < 1e-10,
                    "LDLᵀ[{i}][{j}]={sum} != A[{i}][{j}]={aij}",
                );
            }
        }
    }

    #[test]
    fn ldl_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = ldl(&a, DecompOptions::default()).expect("ldl works");
        // L should be identity, D should be [1, 1]
        assert_eq!(result.l, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(result.d, vec![1.0, 1.0]);
    }

    #[test]
    fn ldl_positive_definite() {
        // A = [[4, 2], [2, 3]]
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let result = ldl(&a, DecompOptions::default()).expect("ldl works");
        verify_ldl_reconstruction(&a, &result);
    }

    #[test]
    fn ldl_symmetric_indefinite() {
        // A = [[1, 2], [2, -3]] — symmetric but not positive definite
        let a = vec![vec![1.0, 2.0], vec![2.0, -3.0]];
        let result = ldl(&a, DecompOptions::default()).expect("ldl works");
        verify_ldl_reconstruction(&a, &result);
        // D should have a negative element (indefinite)
        assert!(
            result.d.iter().any(|&v| v < 0.0),
            "D should have negative entry for indefinite matrix"
        );
    }

    #[test]
    fn ldl_3x3() {
        let a = vec![
            vec![4.0, 1.0, 2.0],
            vec![1.0, 5.0, 3.0],
            vec![2.0, 3.0, 6.0],
        ];
        let result = ldl(&a, DecompOptions::default()).expect("ldl works");
        verify_ldl_reconstruction(&a, &result);
        // L should be unit lower triangular
        for (i, row) in result.l.iter().enumerate() {
            assert_eq!(row[i], 1.0, "L diagonal should be 1");
            for &val in row.iter().skip(i + 1) {
                assert_eq!(val, 0.0, "L upper triangle should be 0");
            }
        }
    }

    #[test]
    fn ldl_empty_matrix() {
        let a: Vec<Vec<f64>> = Vec::new();
        let result = ldl(&a, DecompOptions::default()).expect("ldl empty");
        assert!(result.l.is_empty());
        assert!(result.d.is_empty());
    }

    #[test]
    fn ldl_1x1() {
        let a = vec![vec![7.0]];
        let result = ldl(&a, DecompOptions::default()).expect("ldl 1x1");
        assert_eq!(result.l, vec![vec![1.0]]);
        assert_eq!(result.d, vec![7.0]);
    }

    #[test]
    fn ldl_non_square_rejected() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let err = ldl(&a, DecompOptions::default()).expect_err("non-square");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    #[test]
    fn ldl_hardened_rejects_asymmetric() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // not symmetric
        let options = DecompOptions {
            mode: RuntimeMode::Hardened,
            check_finite: true,
        };
        let err = ldl(&a, options).expect_err("asymmetric in hardened");
        assert!(matches!(err, LinalgError::InvalidArgument { .. }));
    }

    // ── Sylvester / Lyapunov tests ──────────────────────────────────

    #[test]
    fn solve_sylvester_diagonal() {
        // A = diag(1,2), B = diag(3,4), known X = [[1,1],[1,1]]
        // Q = AX + XB = [[1+3, 1+4], [2+3, 2+4]] = [[4,5],[5,6]]
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let b = vec![vec![3.0, 0.0], vec![0.0, 4.0]];
        let q = vec![vec![4.0, 5.0], vec![5.0, 6.0]];
        let x = solve_sylvester(&a, &b, &q, DecompOptions::default()).expect("sylvester");
        // Verify AX + XB = Q
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut ax_xb = 0.0;
                for k in 0..n {
                    ax_xb += a[i][k] * x[k][j] + x[i][k] * b[k][j];
                }
                assert!(
                    (ax_xb - q[i][j]).abs() < 1e-8,
                    "AX+XB[{i},{j}] = {ax_xb}, expected {}",
                    q[i][j]
                );
            }
        }
    }

    #[test]
    fn solve_sylvester_general() {
        // Non-diagonal A, B
        let a = vec![vec![1.0, 2.0], vec![0.0, 3.0]];
        let b = vec![vec![4.0, 1.0], vec![0.0, 5.0]];
        // Choose X = [[1,0],[0,1]] → Q = AX + XB = A + B = [[5,3],[0,8]]
        let q = vec![vec![5.0, 3.0], vec![0.0, 8.0]];
        let x = solve_sylvester(&a, &b, &q, DecompOptions::default()).expect("sylvester");
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut ax_xb = 0.0;
                for k in 0..n {
                    ax_xb += a[i][k] * x[k][j] + x[i][k] * b[k][j];
                }
                assert!(
                    (ax_xb - q[i][j]).abs() < 1e-6,
                    "AX+XB[{i},{j}] = {ax_xb}, expected {}",
                    q[i][j]
                );
            }
        }
    }

    #[test]
    fn solve_sylvester_3x3() {
        let a = vec![
            vec![2.0, 1.0, 0.0],
            vec![0.0, 3.0, 1.0],
            vec![0.0, 0.0, 4.0],
        ];
        let b = vec![
            vec![5.0, 0.0, 0.0],
            vec![1.0, 6.0, 0.0],
            vec![0.0, 1.0, 7.0],
        ];
        // X = I → Q = A + B
        let q = vec![
            vec![7.0, 1.0, 0.0],
            vec![1.0, 9.0, 1.0],
            vec![0.0, 1.0, 11.0],
        ];
        let x = solve_sylvester(&a, &b, &q, DecompOptions::default()).expect("sylvester 3x3");
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let mut ax_xb = 0.0;
                for k in 0..n {
                    ax_xb += a[i][k] * x[k][j] + x[i][k] * b[k][j];
                }
                assert!(
                    (ax_xb - q[i][j]).abs() < 1e-6,
                    "AX+XB[{i},{j}] = {ax_xb}, expected {}",
                    q[i][j]
                );
            }
        }
    }

    #[test]
    fn solve_sylvester_rejects_non_square() {
        let a = vec![vec![1.0, 2.0]]; // 1x2
        let b = vec![vec![1.0]];
        let q = vec![vec![1.0]];
        let err = solve_sylvester(&a, &b, &q, DecompOptions::default()).expect_err("non-square A");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    #[test]
    fn solve_continuous_lyapunov_basic() {
        // A = [[1,0],[0,2]], X = I → Q = AX + XA^T = A + A^T = 2*A (since A is symmetric)
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let q = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let x = solve_continuous_lyapunov(&a, &q, DecompOptions::default()).expect("lyapunov");
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut ax_xat = 0.0;
                for k in 0..n {
                    ax_xat += a[i][k] * x[k][j] + x[i][k] * a[j][k]; // A^T[k,j] = A[j,k]
                }
                assert!(
                    (ax_xat - q[i][j]).abs() < 1e-6,
                    "AX+XA^T[{i},{j}] = {ax_xat}, expected {}",
                    q[i][j]
                );
            }
        }
    }

    #[test]
    fn solve_continuous_lyapunov_non_symmetric_a() {
        // Non-symmetric stable A: A = [[-1,2],[0,-3]]
        // X = I → Q = AX + XA^T = A + A^T = [[-2,2],[2,-6]]
        let a = vec![vec![-1.0, 2.0], vec![0.0, -3.0]];
        let q = vec![vec![-2.0, 2.0], vec![2.0, -6.0]];
        let x =
            solve_continuous_lyapunov(&a, &q, DecompOptions::default()).expect("lyapunov non-sym");
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut ax_xat = 0.0;
                for k in 0..n {
                    ax_xat += a[i][k] * x[k][j] + x[i][k] * a[j][k];
                }
                assert!(
                    (ax_xat - q[i][j]).abs() < 1e-6,
                    "AX+XA^T[{i},{j}] = {ax_xat}, expected {}",
                    q[i][j]
                );
            }
        }
    }

    // ── Discrete Lyapunov tests ──────────────────────────────────────

    #[test]
    fn solve_discrete_lyapunov_diagonal() {
        // A = diag(0.5, 0.3), Q = I
        // A X A^T - X + Q = 0 → X[i,i] = Q[i,i] / (1 - A[i,i]²)
        let a = vec![vec![0.5, 0.0], vec![0.0, 0.3]];
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let x =
            solve_discrete_lyapunov(&a, &q, DecompOptions::default()).expect("discrete lyapunov");
        // Verify: A X A^T - X + Q = 0
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut axa_t = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        axa_t += a[i][k] * x[k][l] * a[j][l];
                    }
                }
                let residual = axa_t - x[i][j] + q[i][j];
                assert!(residual.abs() < 1e-6, "AXA^T-X+Q[{i},{j}] = {residual}");
            }
        }
    }

    #[test]
    fn solve_discrete_lyapunov_stable() {
        // Stable A (all eigenvalues < 1)
        let a = vec![vec![0.8, 0.1], vec![-0.1, 0.7]];
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let x = solve_discrete_lyapunov(&a, &q, DecompOptions::default())
            .expect("discrete lyapunov stable");
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut axa_t = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        axa_t += a[i][k] * x[k][l] * a[j][l];
                    }
                }
                let residual = axa_t - x[i][j] + q[i][j];
                assert!(residual.abs() < 1e-6, "AXA^T-X+Q[{i},{j}] = {residual}");
            }
        }
    }

    #[test]
    fn solve_discrete_lyapunov_rejects_non_square() {
        let a = vec![vec![1.0, 2.0]];
        let q = vec![vec![1.0]];
        let err =
            solve_discrete_lyapunov(&a, &q, DecompOptions::default()).expect_err("non-square");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    // ── signm / funm / fractional_matrix_power tests ─────────────────

    #[test]
    fn signm_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = signm(&a, DecompOptions::default()).expect("signm");
        assert!((result[0][0] - 1.0).abs() < 1e-10);
        assert!((result[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn signm_negative_identity() {
        let a = vec![vec![-1.0, 0.0], vec![0.0, -1.0]];
        let result = signm(&a, DecompOptions::default()).expect("signm");
        assert!((result[0][0] - (-1.0)).abs() < 1e-10);
        assert!((result[1][1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn funm_exp_matches_expm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let funm_result = funm(&a, f64::exp, DecompOptions::default()).expect("funm exp");
        let expm_result = expm(&a, DecompOptions::default()).expect("expm");
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (funm_result[i][j] - expm_result[i][j]).abs() < 1e-8,
                    "funm exp vs expm at [{i},{j}]: {} vs {}",
                    funm_result[i][j],
                    expm_result[i][j]
                );
            }
        }
    }

    #[test]
    fn fractional_matrix_power_sqrt() {
        // A^0.5 squared should give A (for positive definite A)
        let a = vec![vec![4.0, 0.0], vec![0.0, 9.0]];
        let sqrt_a =
            fractional_matrix_power(&a, 0.5, DecompOptions::default()).expect("fractional power");
        // sqrt_a should be [[2,0],[0,3]]
        assert!(
            (sqrt_a[0][0] - 2.0).abs() < 1e-8,
            "sqrt(4) = {}",
            sqrt_a[0][0]
        );
        assert!(
            (sqrt_a[1][1] - 3.0).abs() < 1e-8,
            "sqrt(9) = {}",
            sqrt_a[1][1]
        );
    }

    #[test]
    fn fractional_matrix_power_integer() {
        // A^2 should match A * A
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let a2 = fractional_matrix_power(&a, 2.0, DecompOptions::default()).expect("A^2");
        // A² = [[7,10],[15,22]]
        assert!((a2[0][0] - 7.0).abs() < 1e-6, "A²[0,0] = {}", a2[0][0]);
        assert!((a2[1][1] - 22.0).abs() < 1e-6, "A²[1,1] = {}", a2[1][1]);
    }

    #[test]
    fn funm_rejects_non_square() {
        let a = vec![vec![1.0, 2.0]];
        let err = funm(&a, |x| x, DecompOptions::default()).expect_err("non-square");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    // ── khatri_rao tests ─────────────────────────────────────────────

    #[test]
    fn khatri_rao_basic() {
        // A = [[1,2],[3,4]], B = [[5,6]]
        // Column 0: kron([1,3], [5]) = [5, 15]
        // Column 1: kron([2,4], [6]) = [12, 24]
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0]];
        let result = khatri_rao(&a, &b).expect("khatri_rao");
        assert_eq!(result.len(), 2); // 2*1 rows
        assert_eq!(result[0].len(), 2); // 2 cols
        assert!((result[0][0] - 5.0).abs() < 1e-12);
        assert!((result[1][0] - 15.0).abs() < 1e-12);
        assert!((result[0][1] - 12.0).abs() < 1e-12);
        assert!((result[1][1] - 24.0).abs() < 1e-12);
    }

    #[test]
    fn khatri_rao_mismatched_cols() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![vec![1.0, 2.0, 3.0]];
        let err = khatri_rao(&a, &b).expect_err("mismatched cols");
        assert!(matches!(err, LinalgError::NotSupported { .. }));
    }

    // ── solve_circulant tests ────────────────────────────────────────

    #[test]
    fn solve_circulant_identity() {
        // Circulant of [1, 0, 0] is the identity matrix
        let c = [1.0, 0.0, 0.0];
        let b = [2.0, 3.0, 4.0];
        let x = solve_circulant(&c, &b).expect("solve_circulant");
        for (i, (&xi, &bi)) in x.iter().zip(b.iter()).enumerate() {
            assert!(
                (xi - bi).abs() < 1e-10,
                "x[{i}] = {xi}, expected {bi}"
            );
        }
    }

    #[test]
    fn solve_circulant_shift() {
        // Circulant of [0, 1, 0] is a shift matrix
        // [0 0 1]   [x0]   [b0]
        // [1 0 0] * [x1] = [b1]
        // [0 1 0]   [x2]   [b2]
        // So x0=b1, x1=b2, x2=b0
        let c = [0.0, 1.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        let x = solve_circulant(&c, &b).expect("solve_circulant shift");
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
        assert!((x[2] - 1.0).abs() < 1e-10);
    }

    // ── solve_toeplitz tests ─────────────────────────────────────────

    #[test]
    fn solve_toeplitz_symmetric_identity() {
        let c = [1.0, 0.0, 0.0];
        let b = [2.0, 3.0, 4.0];
        let x = solve_toeplitz(&c, None, &b).expect("solve_toeplitz");
        for (i, (&xi, &bi)) in x.iter().zip(b.iter()).enumerate() {
            assert!(
                (xi - bi).abs() < 1e-10,
                "x[{i}] = {xi}, expected {bi}"
            );
        }
    }

    #[test]
    fn solve_toeplitz_matches_scipy_doc_example() {
        let c = [1.0, 3.0, 6.0, 10.0];
        let r = [1.0, -1.0, -2.0, -3.0];
        let b = [1.0, 2.0, 2.0, 5.0];
        let x = solve_toeplitz(&c, Some(&r), &b).expect("solve_toeplitz");
        let expected = [
            1.6666666666666667,
            -1.0,
            -2.6666666666666665,
            2.3333333333333335,
        ];
        for (i, (&xi, &want)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (xi - want).abs() < 1e-10,
                "x[{i}] = {xi}, expected {want}"
            );
        }
    }

    #[test]
    fn solve_toeplitz_ignores_row_zero_entry() {
        let c = [2.0, 1.0, 0.0];
        let r = [99.0, -1.0, 0.5];
        let matrix = toeplitz(&c, Some(&r));
        assert!((matrix[0][0] - 2.0).abs() < 1e-12, "top-left must come from c[0]");

        let x_expected = [1.0, -2.0, 0.5];
        let mut b = vec![0.0; 3];
        for i in 0..3 {
            for (j, value) in x_expected.iter().enumerate() {
                b[i] += matrix[i][j] * *value;
            }
        }

        let x = solve_toeplitz(&c, Some(&r), &b).expect("solve_toeplitz");
        for (i, (&xi, &want)) in x.iter().zip(x_expected.iter()).enumerate() {
            assert!(
                (xi - want).abs() < 1e-10,
                "x[{i}] = {xi}, expected {want}"
            );
        }
    }

    #[test]
    fn solve_toeplitz_rejects_shape_mismatch() {
        let err = solve_toeplitz(&[1.0, 2.0, 3.0], Some(&[1.0, 2.0]), &[1.0, 2.0, 3.0])
            .expect_err("row length mismatch");
        assert!(matches!(err, LinalgError::IncompatibleShapes { .. }));
    }
}
