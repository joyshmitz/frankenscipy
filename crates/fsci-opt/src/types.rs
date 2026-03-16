#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;
use serde::{Deserialize, Serialize};

pub type MinimizeCallback = fn(&[f64]) -> bool;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizeMethod {
    Bfgs,
    ConjugateGradient,
    Powell,
    NelderMead,
    LBfgsB,
    NewtonCg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RootMethod {
    Brentq,
    Brenth,
    Bisect,
    Ridder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    Success,
    MaxIterations,
    MaxEvaluations,
    PrecisionLoss,
    NanEncountered,
    OutOfBounds,
    CallbackStop,
    NotImplemented,
    InvalidInput,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizeTraceEntry {
    pub ts_unix_ms: u64,
    pub event: String,
    pub method: OptimizeMethod,
    pub iter_num: usize,
    pub f_val: Option<f64>,
    pub grad_norm: Option<f64>,
    pub step_size: Option<f64>,
    pub mode: RuntimeMode,
    pub reason: Option<String>,
    pub final_x: Option<Vec<f64>>,
    pub final_f: Option<f64>,
    pub total_nfev: usize,
    pub fixture_id: Option<String>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizeResult {
    pub x: Vec<f64>,
    pub fun: Option<f64>,
    pub success: bool,
    pub status: ConvergenceStatus,
    pub message: String,
    pub nfev: usize,
    pub njev: usize,
    pub nhev: usize,
    pub nit: usize,
    pub jac: Option<Vec<f64>>,
    pub hess_inv: Option<Vec<Vec<f64>>>,
    pub maxcv: Option<f64>,
}

impl OptimizeResult {
    #[must_use]
    pub fn not_implemented(seed: &[f64], message: impl Into<String>) -> Self {
        Self {
            x: seed.to_vec(),
            fun: None,
            success: false,
            status: ConvergenceStatus::NotImplemented,
            message: message.into(),
            nfev: 0,
            njev: 0,
            nhev: 0,
            nit: 0,
            jac: None,
            hess_inv: None,
            maxcv: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MinimizeOptions {
    pub method: Option<OptimizeMethod>,
    pub tol: Option<f64>,
    pub maxiter: Option<usize>,
    pub maxfev: Option<usize>,
    pub gradient_eps: f64,
    pub callback: Option<MinimizeCallback>,
    pub fixture_id: Option<&'static str>,
    pub seed: Option<u64>,
    pub mode: RuntimeMode,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        Self {
            method: None,
            tol: None,
            maxiter: None,
            maxfev: None,
            gradient_eps: 1.0e-8,
            callback: None,
            fixture_id: None,
            seed: None,
            mode: RuntimeMode::Strict,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootOptions {
    pub method: Option<RootMethod>,
    pub xtol: f64,
    pub rtol: f64,
    pub maxiter: usize,
    pub fixture_id: Option<&'static str>,
    pub seed: Option<u64>,
    pub mode: RuntimeMode,
}

impl Default for RootOptions {
    fn default() -> Self {
        Self {
            method: None,
            xtol: 2.0e-12,
            rtol: 8.881_784_197_001_252e-16,
            maxiter: 100,
            fixture_id: None,
            seed: None,
            mode: RuntimeMode::Strict,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptError {
    InvalidArgument { detail: String },
    InvalidBounds { detail: String },
    SignChangeRequired { detail: String },
    NonFiniteInput { detail: String },
    EvaluationBudgetExceeded { detail: String },
    NotImplemented { detail: String },
}

impl std::fmt::Display for OptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument { detail } => write!(f, "{detail}"),
            Self::InvalidBounds { detail } => write!(f, "{detail}"),
            Self::SignChangeRequired { detail } => write!(f, "{detail}"),
            Self::NonFiniteInput { detail } => write!(f, "{detail}"),
            Self::EvaluationBudgetExceeded { detail } => write!(f, "{detail}"),
            Self::NotImplemented { detail } => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for OptError {}

// ══════════════════════════════════════════════════════════════════════
// Constraint Types
// ══════════════════════════════════════════════════════════════════════

/// Box constraints on optimization variables.
///
/// Matches `scipy.optimize.Bounds(lb, ub)`.
///
/// Each element constrains one variable: lb[i] <= x[i] <= ub[i].
/// Use `f64::NEG_INFINITY` / `f64::INFINITY` for unbounded.
#[derive(Debug, Clone, PartialEq)]
pub struct Bounds {
    /// Lower bounds per variable. Length must match x0.
    pub lb: Vec<f64>,
    /// Upper bounds per variable. Length must match x0.
    pub ub: Vec<f64>,
}

impl Bounds {
    /// Create bounds from lower and upper bound vectors.
    /// Validates that lb[i] <= ub[i] for all i.
    pub fn new(lb: Vec<f64>, ub: Vec<f64>) -> Result<Self, OptError> {
        if lb.len() != ub.len() {
            return Err(OptError::InvalidBounds {
                detail: format!(
                    "lb and ub must have same length (got {} and {})",
                    lb.len(),
                    ub.len()
                ),
            });
        }
        for (i, (&lo, &hi)) in lb.iter().zip(ub.iter()).enumerate() {
            if lo > hi {
                return Err(OptError::InvalidBounds {
                    detail: format!("lb[{i}]={lo} > ub[{i}]={hi}"),
                });
            }
        }
        Ok(Self { lb, ub })
    }

    /// Create unbounded constraints for n variables.
    #[must_use]
    pub fn unbounded(n: usize) -> Self {
        Self {
            lb: vec![f64::NEG_INFINITY; n],
            ub: vec![f64::INFINITY; n],
        }
    }

    /// Number of constrained variables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lb.len()
    }

    /// Whether there are no constraints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lb.is_empty()
    }

    /// Check if a point is feasible (within all bounds).
    #[must_use]
    pub fn is_feasible(&self, x: &[f64]) -> bool {
        x.iter()
            .zip(self.lb.iter().zip(self.ub.iter()))
            .all(|(&xi, (&lo, &hi))| xi >= lo && xi <= hi)
    }

    /// Project a point onto the feasible set (clip to bounds).
    #[must_use]
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(self.lb.iter().zip(self.ub.iter()))
            .map(|(&xi, (&lo, &hi))| xi.clamp(lo, hi))
            .collect()
    }

    /// Convert to the legacy `Bound` tuple format used by lbfgsb.
    #[must_use]
    pub fn to_bound_tuples(&self) -> Vec<(Option<f64>, Option<f64>)> {
        self.lb
            .iter()
            .zip(self.ub.iter())
            .map(|(&lo, &hi)| {
                let lb = if lo == f64::NEG_INFINITY {
                    None
                } else {
                    Some(lo)
                };
                let ub = if hi == f64::INFINITY { None } else { Some(hi) };
                (lb, ub)
            })
            .collect()
    }
}

/// Linear constraint: lb <= A @ x <= ub.
///
/// Matches `scipy.optimize.LinearConstraint(A, lb, ub)`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearConstraint {
    /// Constraint matrix A (m rows × n cols, row-major).
    /// m = number of constraints, n = number of variables.
    pub a: Vec<Vec<f64>>,
    /// Lower bounds on A @ x. Length = m.
    pub lb: Vec<f64>,
    /// Upper bounds on A @ x. Length = m.
    pub ub: Vec<f64>,
}

impl LinearConstraint {
    /// Create a linear constraint. Validates dimensions.
    pub fn new(a: Vec<Vec<f64>>, lb: Vec<f64>, ub: Vec<f64>) -> Result<Self, OptError> {
        let m = a.len();
        if m == 0 {
            return Err(OptError::InvalidArgument {
                detail: "constraint matrix A must have at least one row".to_string(),
            });
        }
        if lb.len() != m || ub.len() != m {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "lb/ub length ({}/{}) must match number of constraint rows ({m})",
                    lb.len(),
                    ub.len()
                ),
            });
        }
        for (i, (&lo, &hi)) in lb.iter().zip(ub.iter()).enumerate() {
            if lo > hi {
                return Err(OptError::InvalidBounds {
                    detail: format!("constraint {i}: lb={lo} > ub={hi}"),
                });
            }
        }
        Ok(Self { a, lb, ub })
    }

    /// Evaluate A @ x and check feasibility.
    pub fn evaluate(&self, x: &[f64]) -> Vec<f64> {
        self.a
            .iter()
            .map(|row| row.iter().zip(x.iter()).map(|(&ai, &xi)| ai * xi).sum())
            .collect()
    }

    /// Check if x satisfies all constraints.
    #[must_use]
    pub fn is_feasible(&self, x: &[f64]) -> bool {
        let ax = self.evaluate(x);
        ax.iter()
            .zip(self.lb.iter().zip(self.ub.iter()))
            .all(|(&v, (&lo, &hi))| v >= lo - 1e-10 && v <= hi + 1e-10)
    }
}

/// Nonlinear constraint: lb <= fun(x) <= ub.
///
/// Matches `scipy.optimize.NonlinearConstraint(fun, lb, ub)`.
#[derive(Clone)]
pub struct NonlinearConstraint {
    /// Constraint function. Returns vector of constraint values.
    pub fun: fn(&[f64]) -> Vec<f64>,
    /// Lower bounds on fun(x).
    pub lb: Vec<f64>,
    /// Upper bounds on fun(x).
    pub ub: Vec<f64>,
}

impl NonlinearConstraint {
    /// Create a nonlinear constraint.
    pub fn new(fun: fn(&[f64]) -> Vec<f64>, lb: Vec<f64>, ub: Vec<f64>) -> Result<Self, OptError> {
        if lb.len() != ub.len() {
            return Err(OptError::InvalidArgument {
                detail: "lb and ub must have same length".to_string(),
            });
        }
        for (i, (&lo, &hi)) in lb.iter().zip(ub.iter()).enumerate() {
            if lo > hi {
                return Err(OptError::InvalidBounds {
                    detail: format!("constraint {i}: lb={lo} > ub={hi}"),
                });
            }
        }
        Ok(Self { fun, lb, ub })
    }

    /// Evaluate fun(x) and check feasibility.
    #[must_use]
    pub fn is_feasible(&self, x: &[f64]) -> bool {
        let values = (self.fun)(x);
        values
            .iter()
            .zip(self.lb.iter().zip(self.ub.iter()))
            .all(|(&v, (&lo, &hi))| v >= lo - 1e-10 && v <= hi + 1e-10)
    }
}

impl std::fmt::Debug for NonlinearConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NonlinearConstraint")
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .field("fun", &"<function>")
            .finish()
    }
}
