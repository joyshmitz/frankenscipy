use std::f64::consts::PI;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use fsci_runtime::RuntimeMode;

use crate::plan::{
    PlanFingerprint, PlanKey, PlanMetadata, PlanningStrategy, lookup_shared_plan, store_shared_plan,
};
use crate::{Normalization, TransformKind};

/// Internal complex representation used by the initial FFT surface.
pub type Complex64 = (f64, f64);

/// Backends that can serve FFT requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    #[default]
    NaiveDft,
}

pub trait FftBackend {
    fn kind(&self) -> BackendKind;
    fn transform_1d_unscaled(&self, input: &[Complex64], inverse: bool) -> Vec<Complex64>;
}

#[derive(Debug, Default)]
pub struct NaiveDftBackend;

impl FftBackend for NaiveDftBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::NaiveDft
    }

    fn transform_1d_unscaled(&self, input: &[Complex64], inverse: bool) -> Vec<Complex64> {
        let n = input.len();
        if n == 0 {
            return Vec::new();
        }

        let sign = if inverse { 1.0 } else { -1.0 };
        let mut output = vec![(0.0, 0.0); n];
        for (k, out) in output.iter_mut().enumerate() {
            let mut acc = (0.0, 0.0);
            for (t, &value) in input.iter().enumerate() {
                let angle = sign * 2.0 * PI * (k as f64) * (t as f64) / (n as f64);
                let twiddle = (angle.cos(), angle.sin());
                acc = complex_add(acc, complex_mul(value, twiddle));
            }
            *out = acc;
        }
        output
    }
}

static NAIVE_BACKEND: NaiveDftBackend = NaiveDftBackend;

/// Worker control policy for transform execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkerPolicy {
    /// Let backend/runtime pick an execution width.
    #[default]
    Auto,
    /// Require an exact worker count.
    Exact(usize),
    /// Upper-bound worker count.
    Max(usize),
}

/// Common options shared by FFT transform entrypoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftOptions {
    pub mode: RuntimeMode,
    pub normalization: Normalization,
    pub workers: WorkerPolicy,
    pub backend: BackendKind,
    pub check_finite: bool,
    pub overwrite_input: bool,
}

impl Default for FftOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            normalization: Normalization::Backward,
            workers: WorkerPolicy::Auto,
            backend: BackendKind::NaiveDft,
            check_finite: false,
            overwrite_input: false,
        }
    }
}

impl FftOptions {
    #[must_use]
    pub fn with_mode(mut self, mode: RuntimeMode) -> Self {
        self.mode = mode;
        self
    }

    #[must_use]
    pub fn with_normalization(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    #[must_use]
    pub fn with_workers(mut self, workers: WorkerPolicy) -> Self {
        self.workers = workers;
        self
    }

    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    pub fn with_check_finite(mut self, check_finite: bool) -> Self {
        self.check_finite = check_finite;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FftError {
    InvalidShape { detail: &'static str },
    InvalidAxes { detail: &'static str },
    InvalidWorkers { requested: usize },
    LengthMismatch { expected: usize, actual: usize },
    NonPositiveSampleSpacing,
    NonFiniteInput,
}

impl Display for FftError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape { detail } => write!(f, "invalid shape: {detail}"),
            Self::InvalidAxes { detail } => write!(f, "invalid axes: {detail}"),
            Self::InvalidWorkers { requested } => write!(f, "invalid worker count: {requested}"),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "length mismatch: expected {expected}, got {actual}")
            }
            Self::NonPositiveSampleSpacing => {
                write!(f, "sample spacing must be finite and greater than zero")
            }
            Self::NonFiniteInput => write!(f, "non-finite input rejected by policy"),
        }
    }
}

impl std::error::Error for FftError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformTrace {
    pub operation_id: String,
    pub kind: TransformKind,
    pub direction: &'static str,
    pub n: usize,
    pub backend: BackendKind,
    pub plan_cache_hit: bool,
    pub mode: RuntimeMode,
    pub timing_ns: u128,
}

impl TransformTrace {
    #[must_use]
    pub fn to_json_line(&self) -> String {
        format!(
            "{{\"operation_id\":\"{}\",\"operation\":\"{}\",\"direction\":\"{}\",\"n\":{},\"backend\":\"{}\",\"plan_cache_hit\":{},\"mode\":\"{}\",\"timing_ns\":{}}}",
            self.operation_id,
            transform_kind_name(self.kind),
            self.direction,
            self.n,
            backend_kind_name(self.backend),
            self.plan_cache_hit,
            runtime_mode_name(self.mode),
            self.timing_ns,
        )
    }
}

static TRACE_LOG: OnceLock<Mutex<Vec<TransformTrace>>> = OnceLock::new();
static OPERATION_COUNTER: AtomicU64 = AtomicU64::new(1);

fn trace_log() -> &'static Mutex<Vec<TransformTrace>> {
    TRACE_LOG.get_or_init(|| Mutex::new(Vec::new()))
}

fn next_operation_id() -> String {
    let next = OPERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("fft-op-{next:016x}")
}

fn record_trace(trace: TransformTrace) {
    if let Ok(mut log) = trace_log().lock() {
        log.push(trace);
    }
}

#[must_use]
pub fn take_transform_traces() -> Vec<TransformTrace> {
    if let Ok(mut log) = trace_log().lock() {
        let mut out = Vec::with_capacity(log.len());
        std::mem::swap(&mut *log, &mut out);
        return out;
    }
    Vec::new()
}

/// 1D forward complex FFT.
pub fn fft(input: &[Complex64], options: &FftOptions) -> Result<Vec<Complex64>, FftError> {
    run_complex_1d(TransformKind::Fft, input, options, false)
}

/// 1D inverse complex FFT.
pub fn ifft(input: &[Complex64], options: &FftOptions) -> Result<Vec<Complex64>, FftError> {
    run_complex_1d(TransformKind::Ifft, input, options, true)
}

/// 1D real-input FFT.
pub fn rfft(input: &[f64], options: &FftOptions) -> Result<Vec<Complex64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_workers(options.workers)?;
    validate_finite_real(input, options)?;

    let backend = resolve_backend(options.backend);
    let key = PlanKey::new(
        TransformKind::Rfft,
        vec![input.len()],
        vec![0],
        options.normalization,
        true,
    );
    let plan_cache_hit = touch_plan_cache(&key, input.len());

    let complex_input: Vec<Complex64> = input.iter().map(|&x| (x, 0.0)).collect();
    let started = Instant::now();
    let mut full = backend.transform_1d_unscaled(&complex_input, false);
    apply_normalization(&mut full, options.normalization, input.len(), false);
    let output = full
        .into_iter()
        .take(input.len() / 2 + 1)
        .collect::<Vec<_>>();

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind: TransformKind::Rfft,
        direction: "forward",
        n: input.len(),
        backend: backend.kind(),
        plan_cache_hit,
        mode: options.mode,
        timing_ns: started.elapsed().as_nanos(),
    });

    Ok(output)
}

/// 1D inverse real FFT.
pub fn irfft(
    input: &[Complex64],
    output_len: Option<usize>,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_workers(options.workers)?;
    validate_finite_complex(input, options)?;

    let n = output_len.unwrap_or_else(|| input.len().saturating_sub(1).saturating_mul(2));
    if n == 0 {
        return Err(FftError::InvalidShape {
            detail: "output_len cannot be zero",
        });
    }

    let expected_len = n / 2 + 1;
    if input.len() != expected_len {
        return Err(FftError::LengthMismatch {
            expected: expected_len,
            actual: input.len(),
        });
    }

    let backend = resolve_backend(options.backend);
    let key = PlanKey::new(
        TransformKind::Irfft,
        vec![n],
        vec![0],
        options.normalization,
        true,
    );
    let plan_cache_hit = touch_plan_cache(&key, n);

    let reconstructed = rebuild_hermitian(input, n);
    let started = Instant::now();
    let mut signal = backend.transform_1d_unscaled(&reconstructed, true);
    apply_normalization(&mut signal, options.normalization, n, true);
    let output = signal.into_iter().map(|(re, _)| re).collect::<Vec<_>>();

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind: TransformKind::Irfft,
        direction: "inverse",
        n,
        backend: backend.kind(),
        plan_cache_hit,
        mode: options.mode,
        timing_ns: started.elapsed().as_nanos(),
    });

    Ok(output)
}

/// 2D forward complex FFT via row/column decomposition.
pub fn fft2(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    run_complex_nd(TransformKind::Fft2, input, &dims, options, false)
}

/// 2D inverse complex FFT via row/column decomposition.
pub fn ifft2(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    run_complex_nd(TransformKind::Ifft2, input, &dims, options, true)
}

/// N-dimensional forward complex FFT.
pub fn fftn(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_nd(TransformKind::Fftn, input, shape, options, false)
}

fn run_complex_1d(
    kind: TransformKind,
    input: &[Complex64],
    options: &FftOptions,
    inverse: bool,
) -> Result<Vec<Complex64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_workers(options.workers)?;
    validate_finite_complex(input, options)?;

    let backend = resolve_backend(options.backend);
    let key = PlanKey::new(
        kind,
        vec![input.len()],
        vec![0],
        options.normalization,
        false,
    );
    let plan_cache_hit = touch_plan_cache(&key, input.len());

    let started = Instant::now();
    let mut output = backend.transform_1d_unscaled(input, inverse);
    apply_normalization(&mut output, options.normalization, input.len(), inverse);

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind,
        direction: if inverse { "inverse" } else { "forward" },
        n: input.len(),
        backend: backend.kind(),
        plan_cache_hit,
        mode: options.mode,
        timing_ns: started.elapsed().as_nanos(),
    });

    Ok(output)
}

fn run_complex_nd(
    kind: TransformKind,
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    inverse: bool,
) -> Result<Vec<Complex64>, FftError> {
    validate_shape(shape)?;
    let expected_len = checked_product(shape).ok_or(FftError::InvalidShape {
        detail: "nd shape product overflow",
    })?;
    if input.len() != expected_len {
        return Err(FftError::LengthMismatch {
            expected: expected_len,
            actual: input.len(),
        });
    }

    validate_workers(options.workers)?;
    validate_finite_complex(input, options)?;

    let backend = resolve_backend(options.backend);
    let axes = (0..shape.len()).collect::<Vec<_>>();
    let key = PlanKey::new(kind, shape.to_vec(), axes, options.normalization, false);
    let plan_cache_hit = touch_plan_cache(&key, expected_len);

    let started = Instant::now();
    let mut output = transform_nd_unscaled(backend, input, shape, inverse);
    apply_normalization(&mut output, options.normalization, expected_len, inverse);

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind,
        direction: if inverse { "inverse" } else { "forward" },
        n: expected_len,
        backend: backend.kind(),
        plan_cache_hit,
        mode: options.mode,
        timing_ns: started.elapsed().as_nanos(),
    });

    Ok(output)
}

fn transform_nd_unscaled(
    backend: &dyn FftBackend,
    input: &[Complex64],
    shape: &[usize],
    inverse: bool,
) -> Vec<Complex64> {
    let mut data = input.to_vec();
    for axis in 0..shape.len() {
        apply_axis_transform(backend, &mut data, shape, axis, inverse);
    }
    data
}

fn apply_axis_transform(
    backend: &dyn FftBackend,
    data: &mut [Complex64],
    shape: &[usize],
    axis: usize,
    inverse: bool,
) {
    let axis_len = shape[axis];
    let stride = shape[axis + 1..].iter().product::<usize>().max(1);
    let repeats = shape[..axis].iter().product::<usize>().max(1);
    let block = axis_len * stride;

    let mut scratch = vec![(0.0, 0.0); axis_len];
    for outer in 0..repeats {
        let outer_base = outer * block;
        for offset in 0..stride {
            for (index, slot) in scratch.iter_mut().enumerate() {
                *slot = data[outer_base + index * stride + offset];
            }
            let transformed = backend.transform_1d_unscaled(&scratch, inverse);
            for (index, &value) in transformed.iter().enumerate() {
                data[outer_base + index * stride + offset] = value;
            }
        }
    }
}

fn rebuild_hermitian(half: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut full = vec![(0.0, 0.0); n];
    full[0] = half[0];

    for (k, &value) in half.iter().enumerate().skip(1) {
        full[k] = value;
    }
    for k in half.len()..n {
        let mirror = n - k;
        full[k] = complex_conj(full[mirror]);
    }

    if n.is_multiple_of(2) && n / 2 < full.len() {
        full[n / 2].1 = 0.0;
    }
    full
}

fn resolve_backend(_kind: BackendKind) -> &'static dyn FftBackend {
    &NAIVE_BACKEND
}

fn touch_plan_cache(key: &PlanKey, n: usize) -> bool {
    if lookup_shared_plan(key).is_some() {
        return true;
    }

    store_shared_plan(PlanMetadata {
        key: key.clone(),
        fingerprint: PlanFingerprint {
            radix_path: factorize_radix_path(n),
            estimated_flops: (n as u64).saturating_mul(n as u64),
            scratch_bytes: n.saturating_mul(std::mem::size_of::<Complex64>()),
        },
        generated_by: PlanningStrategy::EstimateOnly,
    });
    false
}

fn factorize_radix_path(mut n: usize) -> Vec<usize> {
    if n <= 1 {
        return vec![1];
    }

    let mut factors = Vec::new();
    let mut p = 2usize;
    while p * p <= n {
        while n.is_multiple_of(p) {
            factors.push(p);
            n /= p;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

fn validate_shape(shape: &[usize]) -> Result<(), FftError> {
    if shape.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "nd shape cannot be empty",
        });
    }
    if shape.contains(&0) {
        return Err(FftError::InvalidShape {
            detail: "nd shape dimensions must be greater than zero",
        });
    }
    Ok(())
}

fn checked_product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &next| acc.checked_mul(next))
}

fn validate_workers(policy: WorkerPolicy) -> Result<(), FftError> {
    match policy {
        WorkerPolicy::Auto => Ok(()),
        WorkerPolicy::Exact(0) | WorkerPolicy::Max(0) => {
            Err(FftError::InvalidWorkers { requested: 0 })
        }
        WorkerPolicy::Exact(_) | WorkerPolicy::Max(_) => Ok(()),
    }
}

fn validate_finite_complex(input: &[Complex64], options: &FftOptions) -> Result<(), FftError> {
    let should_check = options.check_finite || options.mode == RuntimeMode::Hardened;
    if should_check
        && input
            .iter()
            .any(|&(re, im)| !re.is_finite() || !im.is_finite())
    {
        return Err(FftError::NonFiniteInput);
    }
    Ok(())
}

fn validate_finite_real(input: &[f64], options: &FftOptions) -> Result<(), FftError> {
    let should_check = options.check_finite || options.mode == RuntimeMode::Hardened;
    if should_check && input.iter().any(|value| !value.is_finite()) {
        return Err(FftError::NonFiniteInput);
    }
    Ok(())
}

fn ensure_non_empty(len: usize) -> Result<(), FftError> {
    if len == 0 {
        return Err(FftError::InvalidShape {
            detail: "input length must be greater than zero",
        });
    }
    Ok(())
}

fn normalization_scale(normalization: Normalization, n: usize, inverse: bool) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let n_as_f64 = n as f64;
    match normalization {
        Normalization::Backward => {
            if inverse {
                1.0 / n_as_f64
            } else {
                1.0
            }
        }
        Normalization::Forward => {
            if inverse {
                1.0
            } else {
                1.0 / n_as_f64
            }
        }
        Normalization::Ortho => 1.0 / n_as_f64.sqrt(),
    }
}

fn apply_normalization(
    data: &mut [Complex64],
    normalization: Normalization,
    n: usize,
    inverse: bool,
) {
    let scale = normalization_scale(normalization, n, inverse);
    if (scale - 1.0).abs() <= f64::EPSILON {
        return;
    }
    for value in data.iter_mut() {
        *value = complex_scale(*value, scale);
    }
}

fn complex_add(lhs: Complex64, rhs: Complex64) -> Complex64 {
    (lhs.0 + rhs.0, lhs.1 + rhs.1)
}

fn complex_mul(lhs: Complex64, rhs: Complex64) -> Complex64 {
    (lhs.0 * rhs.0 - lhs.1 * rhs.1, lhs.0 * rhs.1 + lhs.1 * rhs.0)
}

fn complex_scale(value: Complex64, scale: f64) -> Complex64 {
    (value.0 * scale, value.1 * scale)
}

fn complex_conj(value: Complex64) -> Complex64 {
    (value.0, -value.1)
}

fn transform_kind_name(kind: TransformKind) -> &'static str {
    match kind {
        TransformKind::Fft => "fft",
        TransformKind::Ifft => "ifft",
        TransformKind::Rfft => "rfft",
        TransformKind::Irfft => "irfft",
        TransformKind::Fft2 => "fft2",
        TransformKind::Ifft2 => "ifft2",
        TransformKind::Fftn => "fftn",
    }
}

fn backend_kind_name(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::NaiveDft => "naive_dft",
    }
}

fn runtime_mode_name(mode: RuntimeMode) -> &'static str {
    match mode {
        RuntimeMode::Strict => "Strict",
        RuntimeMode::Hardened => "Hardened",
    }
}

#[cfg(test)]
mod tests {
    use fsci_runtime::RuntimeMode;

    use super::{
        FftError, FftOptions, TransformKind, WorkerPolicy, fft, fft2, fftn, ifft, ifft2, irfft,
        rfft, take_transform_traces,
    };
    use crate::Normalization;
    use crate::plan::clear_shared_plan_cache;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!((actual - expected).abs() <= tol, "{actual} !~= {expected}");
    }

    fn assert_close_complex(actual: (f64, f64), expected: (f64, f64), tol: f64) {
        assert_close(actual.0, expected.0, tol);
        assert_close(actual.1, expected.1, tol);
    }

    #[test]
    fn options_default_to_strict_backward_mode() {
        let opts = FftOptions::default();
        assert_eq!(opts.mode, RuntimeMode::Strict);
        assert_eq!(opts.normalization, Normalization::Backward);
    }

    #[test]
    fn exact_zero_workers_is_rejected() {
        let opts = FftOptions::default().with_workers(WorkerPolicy::Exact(0));
        let err = fft(&[(0.0, 0.0)], &opts).expect_err("zero workers should be rejected");
        assert_eq!(err, FftError::InvalidWorkers { requested: 0 });
    }

    #[test]
    fn fft_ifft_roundtrip_identity() {
        let input = vec![(1.0, 0.0), (2.0, -1.0), (0.5, 0.25), (-3.0, 2.0)];
        let opts = FftOptions::default();
        let spectrum = fft(&input, &opts).expect("fft should succeed");
        let recovered = ifft(&spectrum, &opts).expect("ifft should succeed");
        for (&lhs, &rhs) in recovered.iter().zip(&input) {
            assert_close_complex(lhs, rhs, 1e-9);
        }
    }

    #[test]
    fn rfft_length_matches_scipy_convention() {
        let output =
            rfft(&[1.0, 2.0, 3.0, 4.0, 5.0], &FftOptions::default()).expect("rfft should succeed");
        assert_eq!(output.len(), 5 / 2 + 1);
    }

    #[test]
    fn fft2_ifft2_roundtrip_identity() {
        let input = vec![(1.0, 0.0), (2.0, 1.0), (3.0, -1.0), (4.0, 0.5)];
        let spectrum = fft2(&input, (2, 2), &FftOptions::default()).expect("fft2 should succeed");
        let recovered =
            ifft2(&spectrum, (2, 2), &FftOptions::default()).expect("ifft2 should succeed");
        for (&lhs, &rhs) in recovered.iter().zip(&input) {
            assert_close_complex(lhs, rhs, 1e-9);
        }
    }

    #[test]
    fn irfft_respects_requested_length() {
        let input = vec![(15.0, 0.0), (-2.5, 3.4409548), (-2.5, 0.81229924)];
        let signal = irfft(&input, Some(5), &FftOptions::default()).expect("irfft should succeed");
        assert_eq!(signal.len(), 5);
    }

    #[test]
    fn hardened_mode_rejects_non_finite_input() {
        let opts = FftOptions::default().with_mode(RuntimeMode::Hardened);
        let err = rfft(&[1.0, f64::NAN], &opts).expect_err("hardened mode should reject NaN");
        assert_eq!(err, FftError::NonFiniteInput);
    }

    #[test]
    fn fftn_validates_shape_product_against_input() {
        let err = fftn(&[(0.0, 0.0); 5], &[2, 3], &FftOptions::default())
            .expect_err("shape/input mismatch should fail");
        assert_eq!(
            err,
            FftError::LengthMismatch {
                expected: 6,
                actual: 5,
            }
        );
    }

    #[test]
    fn fft_impulse_produces_constant_spectrum() {
        // DFT of impulse [1,0,0,0] = [1,1,1,1]
        let input = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        let spectrum = fft(&input, &FftOptions::default()).expect("fft impulse");
        for &val in &spectrum {
            assert_close_complex(val, (1.0, 0.0), 1e-9);
        }
    }

    #[test]
    fn fft_constant_produces_impulse_spectrum() {
        // DFT of constant [1,1,1,1] = [4,0,0,0]
        let input = vec![(1.0, 0.0); 4];
        let spectrum = fft(&input, &FftOptions::default()).expect("fft constant");
        assert_close_complex(spectrum[0], (4.0, 0.0), 1e-9);
        for &val in &spectrum[1..] {
            assert_close_complex(val, (0.0, 0.0), 1e-9);
        }
    }

    #[test]
    fn fft_length_1_is_identity() {
        let input = vec![(3.5, -1.2)];
        let result = fft(&input, &FftOptions::default()).expect("fft len 1");
        assert_close_complex(result[0], input[0], 1e-12);
    }

    #[test]
    fn fft_length_2_manual_check() {
        // DFT of [a, b] = [a+b, a-b]
        let input = vec![(3.0, 0.0), (1.0, 0.0)];
        let result = fft(&input, &FftOptions::default()).expect("fft len 2");
        assert_close_complex(result[0], (4.0, 0.0), 1e-9);
        assert_close_complex(result[1], (2.0, 0.0), 1e-9);
    }

    #[test]
    fn fft_non_power_of_2_roundtrip() {
        let input: Vec<_> = (0..7).map(|i| (i as f64, 0.0)).collect();
        let spectrum = fft(&input, &FftOptions::default()).expect("fft n=7");
        let recovered = ifft(&spectrum, &FftOptions::default()).expect("ifft n=7");
        for (&a, &b) in recovered.iter().zip(&input) {
            assert_close_complex(a, b, 1e-9);
        }
    }

    #[test]
    fn fft_power_of_2_roundtrip() {
        let input: Vec<_> = (0..16).map(|i| ((i as f64).sin(), (i as f64).cos())).collect();
        let spectrum = fft(&input, &FftOptions::default()).expect("fft n=16");
        let recovered = ifft(&spectrum, &FftOptions::default()).expect("ifft n=16");
        for (&a, &b) in recovered.iter().zip(&input) {
            assert_close_complex(a, b, 1e-9);
        }
    }

    #[test]
    fn fft_known_sine_wave() {
        // sin(2π·k/N) for k=1 has DFT peak at bin 1 and N-1
        let n = 8;
        let input: Vec<_> = (0..n)
            .map(|k| ((2.0 * std::f64::consts::PI * k as f64 / n as f64).sin(), 0.0))
            .collect();
        let spectrum = fft(&input, &FftOptions::default()).expect("fft sine");
        // bin 0 should be ~0, bin 1 should have large imaginary part
        assert_close(spectrum[0].0, 0.0, 1e-9);
        assert!(spectrum[1].0.abs() < 1e-9);
        assert!(spectrum[1].1.abs() > 1.0); // non-trivial imaginary at bin 1
    }

    #[test]
    fn rfft_roundtrip_even_length() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let opts = FftOptions::default();
        let spectrum = rfft(&input, &opts).expect("rfft");
        let recovered = irfft(&spectrum, Some(6), &opts).expect("irfft");
        for (a, b) in recovered.iter().zip(&input) {
            assert_close(*a, *b, 1e-9);
        }
    }

    #[test]
    fn rfft_roundtrip_odd_length() {
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let opts = FftOptions::default();
        let spectrum = rfft(&input, &opts).expect("rfft");
        let recovered = irfft(&spectrum, Some(5), &opts).expect("irfft");
        for (a, b) in recovered.iter().zip(&input) {
            assert_close(*a, *b, 1e-9);
        }
    }

    #[test]
    fn rfft_hermitian_symmetry() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let spectrum = rfft(&input, &FftOptions::default()).expect("rfft");
        // For real input, X[0] is real-valued
        assert_close(spectrum[0].1, 0.0, 1e-9);
        // X[N/2] (last element for even N) is real-valued
        assert_close(spectrum[spectrum.len() - 1].1, 0.0, 1e-9);
    }

    #[test]
    fn rfft_real_sine_wave() {
        let n = 16;
        let input: Vec<f64> = (0..n)
            .map(|k| (2.0 * std::f64::consts::PI * k as f64 / n as f64).sin())
            .collect();
        let spectrum = rfft(&input, &FftOptions::default()).expect("rfft sine");
        assert_eq!(spectrum.len(), n / 2 + 1);
        // DC component should be ~0
        assert_close(spectrum[0].0, 0.0, 1e-9);
    }

    #[test]
    fn fft2_impulse_2d() {
        // 2D impulse at (0,0): all spectral bins should be 1
        let input = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        let spectrum = fft2(&input, (2, 2), &FftOptions::default()).expect("fft2 impulse");
        for &val in &spectrum {
            assert_close_complex(val, (1.0, 0.0), 1e-9);
        }
    }

    #[test]
    fn fft2_non_square() {
        let input: Vec<_> = (0..6).map(|i| (i as f64, 0.0)).collect();
        let opts = FftOptions::default();
        let spectrum = fft2(&input, (2, 3), &opts).expect("fft2 2x3");
        let recovered = ifft2(&spectrum, (2, 3), &opts).expect("ifft2 2x3");
        for (&a, &b) in recovered.iter().zip(&input) {
            assert_close_complex(a, b, 1e-9);
        }
    }

    #[test]
    fn fft2_separable_product() {
        // f(x,y) = a(x) * b(y) => F(u,v) = A(u) * B(v)
        let a = vec![(1.0, 0.0), (2.0, 0.0)];
        let b = vec![(3.0, 0.0), (-1.0, 0.0)];
        let opts = FftOptions::default();
        let mut input = Vec::with_capacity(4);
        for ax in &a {
            for bx in &b {
                input.push((ax.0 * bx.0, 0.0));
            }
        }
        let result_2d = fft2(&input, (2, 2), &opts).expect("fft2");
        let fa = fft(&a, &opts).expect("fft a");
        let fb = fft(&b, &opts).expect("fft b");
        for i in 0..2 {
            for j in 0..2 {
                let expected = complex_mul_test(fa[i], fb[j]);
                assert_close_complex(result_2d[i * 2 + j], expected, 1e-9);
            }
        }
    }

    #[test]
    fn normalization_forward_scales_forward_pass() {
        let input = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let opts_backward = FftOptions::default();
        let opts_forward = FftOptions::default().with_normalization(Normalization::Forward);
        let backward = fft(&input, &opts_backward).expect("backward");
        let forward = fft(&input, &opts_forward).expect("forward");
        let n = input.len() as f64;
        for (&bv, &fv) in backward.iter().zip(&forward) {
            assert_close_complex(fv, (bv.0 / n, bv.1 / n), 1e-9);
        }
    }

    #[test]
    fn normalization_ortho_is_symmetric() {
        let input = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let opts = FftOptions::default().with_normalization(Normalization::Ortho);
        let spectrum = fft(&input, &opts).expect("fft ortho");
        let recovered = ifft(&spectrum, &opts).expect("ifft ortho");
        for (&a, &b) in recovered.iter().zip(&input) {
            assert_close_complex(a, b, 1e-9);
        }
    }

    #[test]
    fn check_finite_rejects_nan_complex() {
        let opts = FftOptions::default().with_check_finite(true);
        let err = fft(&[(1.0, f64::NAN)], &opts).expect_err("NaN should be rejected");
        assert_eq!(err, FftError::NonFiniteInput);
    }

    #[test]
    fn check_finite_rejects_inf_real() {
        let opts = FftOptions::default().with_check_finite(true);
        let err = rfft(&[1.0, f64::INFINITY], &opts).expect_err("Inf should be rejected");
        assert_eq!(err, FftError::NonFiniteInput);
    }

    fn complex_mul_test(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    #[test]
    fn repeated_calls_emit_plan_cache_hits_in_trace() {
        clear_shared_plan_cache();
        let _ = take_transform_traces();

        // Use n=37 (prime, outside proptest ranges 1..=32) to avoid
        // parallel test cache collisions.
        let input = (0..37usize)
            .map(|i| (i as f64, (i % 3) as f64 - 1.0))
            .collect::<Vec<_>>();
        let opts = FftOptions::default();
        let _ = fft(&input, &opts).expect("first fft should succeed");
        let _ = fft(&input, &opts).expect("second fft should succeed");

        let mut fft_traces = take_transform_traces()
            .into_iter()
            .filter(|trace| trace.kind == TransformKind::Fft)
            .collect::<Vec<_>>();
        fft_traces.sort_by(|lhs, rhs| lhs.operation_id.cmp(&rhs.operation_id));

        assert!(fft_traces.len() >= 2);
        let last_two = &fft_traces[fft_traces.len() - 2..];
        assert!(!last_two[0].plan_cache_hit);
        assert!(last_two[1].plan_cache_hit);
        assert!(last_two[0].to_json_line().contains("\"operation_id\""));
    }
}
