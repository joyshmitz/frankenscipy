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
