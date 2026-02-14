#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::{IntegrateValidationError, ToleranceValue};

pub type StepRhsFn = dyn FnMut(f64, &[f64]) -> Vec<f64>;

#[derive(Debug, Clone, PartialEq)]
pub struct InitialStepRequest<'a> {
    pub t0: f64,
    pub y0: &'a [f64],
    pub t_bound: f64,
    pub max_step: f64,
    pub f0: &'a [f64],
    pub direction: f64,
    pub order: f64,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub mode: RuntimeMode,
}

pub fn select_initial_step<F>(
    fun: &mut F,
    request: &InitialStepRequest<'_>,
) -> Result<f64, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let _ = fun;
    let _ = request;
    Err(IntegrateValidationError::NotYetImplemented {
        function: "select_initial_step",
    })
}
