#![forbid(unsafe_code)]

use std::fmt;

use fsci_runtime::RuntimeMode;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum SpecialTensor {
    #[default]
    Empty,
    RealScalar(f64),
    ComplexScalar(Complex64),
    RealVec(Vec<f64>),
    ComplexVec(Vec<Complex64>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelRegime {
    Series,
    Asymptotic,
    Recurrence,
    ContinuedFraction,
    Reflection,
    BackendDelegate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchStep {
    pub regime: KernelRegime,
    pub when: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchPlan {
    pub function: &'static str,
    pub steps: &'static [DispatchStep],
    pub notes: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialErrorKind {
    DomainError,
    PoleInput,
    NonFiniteInput,
    CancellationRisk,
    OverflowRisk,
    SingularityRisk,
    NotYetImplemented,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialError {
    pub function: &'static str,
    pub kind: SpecialErrorKind,
    pub mode: RuntimeMode,
    pub detail: &'static str,
}

impl fmt::Display for SpecialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "special function `{}` failed in {:?} mode: {} ({:?})",
            self.function, self.mode, self.detail, self.kind
        )
    }
}

impl std::error::Error for SpecialError {}

pub type SpecialResult = Result<SpecialTensor, SpecialError>;

pub fn not_yet_implemented(
    function: &'static str,
    mode: RuntimeMode,
    detail: &'static str,
) -> SpecialResult {
    Err(SpecialError {
        function,
        kind: SpecialErrorKind::NotYetImplemented,
        mode,
        detail,
    })
}
