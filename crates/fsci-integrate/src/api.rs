#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::{
    IntegrateValidationError, ToleranceValue, validate_first_step, validate_max_step, validate_tol,
};

pub type EventFn = fn(f64, &[f64]) -> f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    Rk23,
    Rk45,
    Dop853,
    Radau,
    Bdf,
    Lsoda,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OdeSolution {
    pub knots: Vec<f64>,
    pub values: Vec<Vec<f64>>,
    pub alt_segment: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveIvpOptions<'a> {
    pub t_span: (f64, f64),
    pub y0: &'a [f64],
    pub method: SolverKind,
    pub t_eval: Option<&'a [f64]>,
    pub dense_output: bool,
    pub events: Option<Vec<EventFn>>,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub first_step: Option<f64>,
    pub max_step: f64,
    pub mode: RuntimeMode,
}

impl Default for SolveIvpOptions<'_> {
    fn default() -> Self {
        Self {
            t_span: (0.0, 0.0),
            y0: &[],
            method: SolverKind::Rk45,
            t_eval: None,
            dense_output: false,
            events: None,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            first_step: None,
            max_step: f64::INFINITY,
            mode: RuntimeMode::Strict,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveIvpResult {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    pub sol: Option<OdeSolution>,
    pub t_events: Option<Vec<Vec<f64>>>,
    pub y_events: Option<Vec<Vec<Vec<f64>>>>,
    pub nfev: usize,
    pub njev: usize,
    pub nlu: usize,
    pub status: i32,
    pub message: String,
    pub success: bool,
}

pub fn solve_ivp<F>(
    fun: &mut F,
    options: &SolveIvpOptions<'_>,
) -> Result<SolveIvpResult, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let _ = fun;

    validate_max_step(options.max_step)?;
    if let Some(first_step) = options.first_step {
        validate_first_step(first_step, options.t_span.0, options.t_span.1)?;
    }
    let _ = validate_tol(
        ToleranceValue::Scalar(options.rtol),
        options.atol.clone(),
        options.y0.len(),
        options.mode,
    )?;

    Err(IntegrateValidationError::NotYetImplemented {
        function: "solve_ivp",
    })
}
