#![forbid(unsafe_code)]

use std::fmt;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTraceEntry {
    pub timestamp_ms: u128,
    pub function: &'static str,
    pub mode: RuntimeMode,
    pub category: &'static str,
    pub input_summary: String,
    pub action_taken: &'static str,
    pub result_summary: String,
    pub clamped: bool,
}

impl SpecialTraceEntry {
    #[must_use]
    pub fn to_json_line(&self) -> String {
        format!(
            "{{\"timestamp_ms\":{},\"function\":\"{}\",\"mode\":\"{:?}\",\"category\":\"{}\",\"input_summary\":\"{}\",\"action_taken\":\"{}\",\"result_summary\":\"{}\",\"clamped\":{}}}",
            self.timestamp_ms,
            self.function,
            self.mode,
            self.category,
            escape_json(&self.input_summary),
            self.action_taken,
            escape_json(&self.result_summary),
            self.clamped
        )
    }
}

static TRACE_LOG: OnceLock<Mutex<Vec<SpecialTraceEntry>>> = OnceLock::new();

fn trace_log() -> &'static Mutex<Vec<SpecialTraceEntry>> {
    TRACE_LOG.get_or_init(|| Mutex::new(Vec::new()))
}

#[must_use]
pub fn take_special_traces() -> Vec<SpecialTraceEntry> {
    if let Ok(mut log) = trace_log().lock() {
        let mut out = Vec::with_capacity(log.len());
        std::mem::swap(&mut *log, &mut out);
        return out;
    }
    Vec::new()
}

pub fn record_special_trace(
    function: &'static str,
    mode: RuntimeMode,
    category: &'static str,
    input_summary: impl Into<String>,
    action_taken: &'static str,
    result_summary: impl Into<String>,
    clamped: bool,
) {
    if let Ok(mut log) = trace_log().lock() {
        log.push(SpecialTraceEntry {
            timestamp_ms: now_unix_ms(),
            function,
            mode,
            category,
            input_summary: input_summary.into(),
            action_taken,
            result_summary: result_summary.into(),
            clamped,
        });
    }
}

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

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn escape_json(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
