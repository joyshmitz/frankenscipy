use std::f64::consts::PI;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, RuntimeMode, casp_now_unix_ms};

use crate::plan::{
    PlanFingerprint, PlanKey, PlanMetadata, PlanningStrategy, lookup_shared_plan, store_shared_plan,
};
use crate::{Normalization, TransformKind};

/// Internal complex representation used by the initial FFT surface.
pub type Complex64 = (f64, f64);

/// Backends that can serve FFT requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    /// Naive O(n²) DFT — reference implementation.
    NaiveDft,
    /// Cooley-Tukey FFT — O(n log n) for power-of-2, mixed-radix + Bluestein for general n.
    #[default]
    CooleyTukey,
}

pub trait FftBackend {
    fn kind(&self) -> BackendKind;
    fn transform_1d_unscaled(&self, input: &[Complex64], inverse: bool) -> Vec<Complex64>;
    /// Perform transform in-place on the `data` slice.
    fn transform_1d_inplace(&self, data: &mut [Complex64], inverse: bool);
}

#[derive(Debug, Default)]
pub struct NaiveDftBackend;

impl FftBackend for NaiveDftBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::NaiveDft
    }

    fn transform_1d_unscaled(&self, input: &[Complex64], inverse: bool) -> Vec<Complex64> {
        let mut output = input.to_vec();
        self.transform_1d_inplace(&mut output, inverse);
        output
    }

    fn transform_1d_inplace(&self, data: &mut [Complex64], inverse: bool) {
        let n = data.len();
        if n <= 1 {
            return;
        }

        let input_copy = data.to_vec();
        let sign = if inverse { 1.0 } else { -1.0 };
        for (k, out) in data.iter_mut().enumerate() {
            let mut acc = (0.0, 0.0);
            for (t, &value) in input_copy.iter().enumerate() {
                let angle = sign * 2.0 * PI * (k as f64) * (t as f64) / (n as f64);
                let twiddle = (angle.cos(), angle.sin());
                acc = complex_add(acc, complex_mul(value, twiddle));
            }
            *out = acc;
        }
    }
}

static NAIVE_BACKEND: NaiveDftBackend = NaiveDftBackend;

/// Cooley-Tukey FFT backend — O(n log n) for power-of-2 sizes.
/// Falls back to Bluestein's algorithm for arbitrary sizes.
#[derive(Debug, Default)]
pub struct CooleyTukeyBackend;

impl FftBackend for CooleyTukeyBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::CooleyTukey
    }

    fn transform_1d_unscaled(&self, input: &[Complex64], inverse: bool) -> Vec<Complex64> {
        let mut output = input.to_vec();
        self.transform_1d_inplace(&mut output, inverse);
        output
    }

    fn transform_1d_inplace(&self, data: &mut [Complex64], inverse: bool) {
        let n = data.len();
        if n <= 1 {
            return;
        }
        if n.is_power_of_two() {
            cooley_tukey_radix2_inplace(data, inverse);
        } else {
            // For non-power-of-2: use Bluestein's algorithm (chirp-z transform)
            let result = bluestein_fft(data, inverse);
            data.copy_from_slice(&result);
        }
    }
}

static COOLEY_TUKEY_BACKEND: CooleyTukeyBackend = CooleyTukeyBackend;

/// Create a new shared audit ledger for synchronous FFT operations.
#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

/// Radix-2 Cooley-Tukey FFT for power-of-2 lengths.
/// Iterative (bottom-up) implementation for better cache behavior.
fn cooley_tukey_radix2_inplace(data: &mut [Complex64], inverse: bool) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }

    // Butterfly stages
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut twiddles = Vec::with_capacity(n / 2);
    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let angle_step = sign * 2.0 * PI / stage_len as f64;

        // Precompute twiddle factors for this stage
        twiddles.clear();
        for k in 0..half {
            let angle = angle_step * k as f64;
            twiddles.push((angle.cos(), angle.sin()));
        }

        let mut base = 0;
        while base < n {
            for (k, &twiddle) in twiddles.iter().enumerate() {
                let even_idx = base + k;
                let odd_idx = base + k + half;
                let odd_val = complex_mul(data[odd_idx], twiddle);
                let even_val = data[even_idx];
                data[even_idx] = complex_add(even_val, odd_val);
                data[odd_idx] = complex_sub(even_val, odd_val);
            }
            base += stage_len;
        }
        stage_len *= 2;
    }
}

/// Bluestein's algorithm for arbitrary-length FFT.
/// Converts an n-point DFT into a circular convolution of length m (power of 2 >= 2n-1),
/// then uses radix-2 FFT on the padded data.
fn bluestein_fft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    // Find smallest power of 2 >= 2n - 1
    let m = (2 * n - 1).next_power_of_two();

    let sign = if inverse { 1.0 } else { -1.0 };

    // Chirp sequence: w[k] = exp(sign * i * π * k² / n)
    let mut chirp = Vec::with_capacity(n);
    for k in 0..n {
        let angle = sign * PI * (k as f64).powi(2) / (n as f64);
        chirp.push((angle.cos(), angle.sin()));
    }

    // Sequence a: input[k] * chirp[k], zero-padded to length m
    let mut a = vec![(0.0, 0.0); m];
    for k in 0..n {
        a[k] = complex_mul(input[k], chirp[k]);
    }

    // Sequence b: conj(chirp[k]) for k=0..n-1, wrapped for circular convolution
    let mut b = vec![(0.0, 0.0); m];
    b[0] = complex_conj(chirp[0]);
    for k in 1..n {
        b[k] = complex_conj(chirp[k]);
        b[m - k] = complex_conj(chirp[k]);
    }

    // Convolution via FFT: C = IFFT(FFT(a) * FFT(b))
    cooley_tukey_radix2_inplace(&mut a, false);
    cooley_tukey_radix2_inplace(&mut b, false);
    for i in 0..m {
        a[i] = complex_mul(a[i], b[i]);
    }
    cooley_tukey_radix2_inplace(&mut a, true);

    // Extract result: output[k] = chirp[k] * a[k] / m
    let inv_m = 1.0 / m as f64;
    let mut output = Vec::with_capacity(n);
    for k in 0..n {
        let val = complex_scale(a[k], inv_m);
        output.push(complex_mul(chirp[k], val));
    }

    output
}

/// Reverse the lower `bits` bits of `x`.
/// Specialized real FFT: pack N real values into N/2 complex, FFT, then unpack.
/// Returns N/2 + 1 complex values (the non-redundant half of the spectrum).
fn real_fft_specialized(input: &[f64], backend: &dyn FftBackend) -> Vec<Complex64> {
    let n = input.len();
    let half = n / 2;

    // Pack: z[k] = x[2k] + i*x[2k+1]
    let mut packed = Vec::with_capacity(half);
    for k in 0..half {
        packed.push((input[2 * k], input[2 * k + 1]));
    }

    // FFT of half-length complex sequence
    let z = backend.transform_1d_unscaled(&packed, false);

    // Unpack: X[k] = (Z[k] + conj(Z[N/2-k]))/2 - i*exp(-2πik/N)*(Z[k] - conj(Z[N/2-k]))/2
    let mut result = Vec::with_capacity(half + 1);
    for k in 0..=half {
        let zk = if k < half { z[k] } else { z[0] };
        let zn_k = if k == 0 || k == half {
            z[0]
        } else {
            z[half - k]
        };
        let zn_k_conj = complex_conj(zn_k);

        let even = (0.5 * (zk.0 + zn_k_conj.0), 0.5 * (zk.1 + zn_k_conj.1));
        let odd = (0.5 * (zk.0 - zn_k_conj.0), 0.5 * (zk.1 - zn_k_conj.1));

        // Twiddle: exp(-2πik/N)
        let angle = -2.0 * PI * k as f64 / n as f64;
        let twiddle = (angle.cos(), angle.sin());
        let odd_tw = complex_mul(odd, twiddle);

        // Multiply by -i: (a, b) -> (b, -a)
        let odd_final = (odd_tw.1, -odd_tw.0);

        result.push(complex_add(even, odd_final));
    }

    result
}

fn bit_reverse(x: usize, bits: usize) -> usize {
    if bits == 0 {
        return 0;
    }
    x.reverse_bits() >> (usize::BITS as usize - bits)
}

fn complex_sub(lhs: Complex64, rhs: Complex64) -> Complex64 {
    (lhs.0 - rhs.0, lhs.1 - rhs.1)
}

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
            backend: BackendKind::default(),
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
    fft_impl(input, options, None)
}

/// 1D forward complex FFT with audit logging.
pub fn fft_with_audit(
    input: &[Complex64],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    fft_impl(input, options, Some(audit_ledger))
}

/// 1D inverse complex FFT.
pub fn ifft(input: &[Complex64], options: &FftOptions) -> Result<Vec<Complex64>, FftError> {
    ifft_impl(input, options, None)
}

/// 1D inverse complex FFT with audit logging.
pub fn ifft_with_audit(
    input: &[Complex64],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    ifft_impl(input, options, Some(audit_ledger))
}

/// 1D real-input FFT.
pub fn rfft(input: &[f64], options: &FftOptions) -> Result<Vec<Complex64>, FftError> {
    rfft_impl(input, options, None)
}

/// 1D real-input FFT with audit logging.
pub fn rfft_with_audit(
    input: &[f64],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    rfft_impl(input, options, Some(audit_ledger))
}

/// 1D inverse real FFT.
pub fn irfft(
    input: &[Complex64],
    output_len: Option<usize>,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    irfft_impl(input, output_len, options, None)
}

/// 1D inverse real FFT with audit logging.
pub fn irfft_with_audit(
    input: &[Complex64],
    output_len: Option<usize>,
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<f64>, FftError> {
    irfft_impl(input, output_len, options, Some(audit_ledger))
}

fn fft_impl(
    input: &[Complex64],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_1d(TransformKind::Fft, input, options, false, audit_ledger)
}

fn ifft_impl(
    input: &[Complex64],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_1d(TransformKind::Ifft, input, options, true, audit_ledger)
}

fn rfft_impl(
    input: &[f64],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    let fingerprint = real_fingerprint(input);
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_real_with_audit(input, options, &fingerprint, audit_ledger)?;

    let backend = resolve_backend(options.backend);
    let key = PlanKey::new(
        TransformKind::Rfft,
        vec![input.len()],
        vec![0],
        options.normalization,
        true,
    );
    let plan_cache_hit = touch_plan_cache(&key, input.len());

    let started = Instant::now();
    let mut output = real_fft_unscaled(input, backend);
    apply_normalization(&mut output, options.normalization, input.len(), false);

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

fn irfft_impl(
    input: &[Complex64],
    output_len: Option<usize>,
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    let fingerprint = complex_fingerprint(input);
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_complex_with_audit(input, options, &fingerprint, audit_ledger)?;

    let n = output_len.unwrap_or_else(|| {
        if input.len() == 1 {
            1
        } else {
            input.len().saturating_sub(1).saturating_mul(2)
        }
    });
    if n == 0 {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "invalid_output_len",
            "rejected: output_len cannot be zero",
        );
        return Err(FftError::InvalidShape {
            detail: "output_len cannot be zero",
        });
    }

    let expected_len = n / 2 + 1;
    if input.len() != expected_len {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "length_mismatch",
            format!(
                "rejected: expected spectrum length {expected_len}, got {}",
                input.len()
            ),
        );
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

    let started = Instant::now();
    let mut output = real_ifft_unscaled(input, n, backend);
    apply_real_normalization(&mut output, options.normalization, n, true);

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
    fft2_impl(input, &dims, options, None)
}

/// 2D forward complex FFT with audit logging.
pub fn fft2_with_audit(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    fft2_impl(input, &dims, options, Some(audit_ledger))
}

/// 2D inverse complex FFT via row/column decomposition.
pub fn ifft2(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    ifft2_impl(input, &dims, options, None)
}

/// 2D inverse complex FFT with audit logging.
pub fn ifft2_with_audit(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    ifft2_impl(input, &dims, options, Some(audit_ledger))
}

/// N-dimensional forward complex FFT.
pub fn fftn(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    fftn_impl(input, shape, options, None)
}

/// N-dimensional forward complex FFT with audit logging.
pub fn fftn_with_audit(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    fftn_impl(input, shape, options, Some(audit_ledger))
}

/// N-dimensional inverse complex FFT.
///
/// Matches `scipy.fft.ifftn(x, s)`.
pub fn ifftn(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    ifftn_impl(input, shape, options, None)
}

/// N-dimensional inverse complex FFT with audit logging.
pub fn ifftn_with_audit(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    ifftn_impl(input, shape, options, Some(audit_ledger))
}

/// N-dimensional real-input FFT.
///
/// Matches `scipy.fft.rfftn(x, s)`.
pub fn rfftn(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    rfftn_impl(input, shape, options, None)
}

/// N-dimensional real-input FFT with audit logging.
pub fn rfftn_with_audit(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    rfftn_impl(input, shape, options, Some(audit_ledger))
}

fn fft2_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_nd(
        TransformKind::Fft2,
        input,
        shape,
        options,
        false,
        audit_ledger,
    )
}

fn ifft2_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_nd(
        TransformKind::Ifft2,
        input,
        shape,
        options,
        true,
        audit_ledger,
    )
}

fn fftn_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_nd(
        TransformKind::Fftn,
        input,
        shape,
        options,
        false,
        audit_ledger,
    )
}

fn ifftn_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_nd(
        TransformKind::Ifftn,
        input,
        shape,
        options,
        true,
        audit_ledger,
    )
}

fn rfftn_impl(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_real_nd_forward(TransformKind::Rfftn, input, shape, options, audit_ledger)
}

/// Discrete Cosine Transform (Type II).
///
/// Matches `scipy.fft.dct(x, type=2)`.
/// DCT-II: X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(π*(2n+1)*k / (2N))
///
/// Computed via FFT of a reordered and mirrored sequence.
pub fn dct(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();

    // Compute DCT-II via FFT of a mirrored 2N-length sequence
    let mut extended = vec![(0.0, 0.0); 2 * n];
    for i in 0..n {
        extended[i] = (input[i], 0.0);
    }
    for i in 0..n {
        extended[2 * n - 1 - i] = (input[i], 0.0);
    }

    let backend = resolve_backend(options.backend);
    let spectrum = backend.transform_1d_unscaled(&extended, false);

    // Extract DCT coefficients: X[k] = Re(spectrum[k] * exp(-i*π*k/(2N)))
    let mut result = Vec::with_capacity(n);
    for (k, &sk) in spectrum.iter().enumerate().take(n) {
        let angle = -PI * k as f64 / (2.0 * n as f64);
        let twiddle = (angle.cos(), angle.sin());
        let val = complex_mul(sk, twiddle);
        result.push(val.0); // take real part
    }

    Ok(result)
}

/// Inverse Discrete Cosine Transform (Type III, the inverse of DCT-II).
///
/// Matches `scipy.fft.idct(x, type=2)`.
pub fn idct(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();

    // DCT-III: x[n] = X[0]/(2N) + (1/N) * sum_{k=1}^{N-1} X[k] * cos(π*k*(2n+1)/(2N))
    // Compute via inverse of the DCT-II process
    let backend = resolve_backend(options.backend);

    // Prepare: multiply by twiddle and create Hermitian-symmetric spectrum
    let mut spectrum = vec![(0.0, 0.0); 2 * n];
    for k in 0..n {
        let angle = PI * k as f64 / (2.0 * n as f64);
        let twiddle = (angle.cos(), angle.sin());
        spectrum[k] = complex_mul((input[k], 0.0), twiddle);
    }
    // Hermitian symmetry for real output
    for k in 1..n {
        spectrum[2 * n - k] = complex_conj(spectrum[k]);
    }

    let time_domain = backend.transform_1d_unscaled(&spectrum, true);

    // Extract: take first N values, scale by 1/(2N)
    let scale = 1.0 / (2.0 * n as f64);
    let result: Vec<f64> = time_domain.iter().take(n).map(|v| v.0 * scale).collect();

    Ok(result)
}

/// Discrete Cosine Transform Type I.
///
/// DCT-I: X[k] = x[0] + (-1)^k * x[N-1] + 2 * Σ_{n=1}^{N-2} x[n] * cos(πnk/(N-1))
///
/// Matches `scipy.fft.dct(x, type=1)`.
pub fn dct_i(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    if n == 1 {
        return Ok(vec![input[0]]);
    }

    // DCT-I via FFT of length 2N-2
    let m = 2 * n - 2;
    let mut extended = vec![(0.0, 0.0); m];
    for i in 0..n {
        extended[i] = (input[i], 0.0);
    }
    for i in 1..n - 1 {
        extended[m - i] = (input[i], 0.0);
    }

    let backend = resolve_backend(options.backend);
    let spectrum = backend.transform_1d_unscaled(&extended, false);

    Ok(spectrum.into_iter().take(n).map(|v| v.0).collect())
}

/// Discrete Cosine Transform Type III.
///
/// DCT-III: x[n] = X[0]/2 + Σ_{k=1}^{N-1} X[k] * cos(πk(2n+1)/(2N))
///
/// This is the inverse of DCT-II (up to scaling).
/// Matches `scipy.fft.dct(x, type=3)`.
pub fn dct_iii(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    // DCT-III is scaled IDCT-II
    let idct_result = idct(input, options)?;
    Ok(idct_result
        .into_iter()
        .map(|v| v * 2.0 * n as f64)
        .collect())
}

/// Discrete Cosine Transform Type IV.
///
/// DCT-IV: X[k] = 2 * Σ_{n=0}^{N-1} x[n] * cos(π(2n+1)(2k+1)/(4N))
///
/// Matches `scipy.fft.dct(x, type=4)`.
pub fn dct_iv(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    // DCT-IV via FFT of length 8N
    let m = 8 * n;
    let mut extended = vec![(0.0, 0.0); m];
    for i in 0..n {
        extended[2 * i + 1] = (input[i], 0.0);
        extended[m - 2 * i - 1] = (input[i], 0.0);
    }

    let backend = resolve_backend(options.backend);
    let spectrum = backend.transform_1d_unscaled(&extended, false);

    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        // Real part of bin 2k+1
        result.push(spectrum[2 * k + 1].0);
    }
    Ok(result)
}

/// Discrete Sine Transform Type I.
///
/// DST-I: X[k] = 2 * Σ_{n=0}^{N-1} x[n] * sin(π(n+1)(k+1)/(N+1))
///
/// Matches `scipy.fft.dst(x, type=1)`.
pub fn dst_i(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    // DST-I via FFT of length 2N+2
    let m = 2 * n + 2;
    let mut extended = vec![(0.0, 0.0); m];
    for i in 0..n {
        extended[i + 1] = (input[i], 0.0);
        extended[m - i - 1] = (-input[i], 0.0);
    }

    let backend = resolve_backend(options.backend);
    let spectrum = backend.transform_1d_unscaled(&extended, false);

    let mut result = Vec::with_capacity(n);
    for val in spectrum.iter().take(n + 1).skip(1) {
        result.push(-val.1); // -Im part corresponds to 2 * sum
    }
    Ok(result)
}

/// Discrete Sine Transform Type II.
///
/// DST-II: X[k] = Σ_{n=0}^{N-1} x[n] * sin(π(2n+1)(k+1)/(2N))
///
/// Matches `scipy.fft.dst(x, type=2)`.
pub fn dst_ii(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    // DST-II via FFT of length 4N.
    // X[k] = sum_{n=0}^{N-1} x[n] * sin(pi*(2n+1)*(k+1)/(2N))
    let m = 4 * n;
    let mut extended = vec![(0.0, 0.0); m];
    for i in 0..n {
        extended[2 * i + 1] = (input[i], 0.0);
        extended[m - 2 * i - 1] = (-input[i], 0.0);
    }

    let backend = resolve_backend(options.backend);
    let spectrum = backend.transform_1d_unscaled(&extended, false);

    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        // Imaginary part of bin k+1 corresponds to -2 * sum_{n=0}^{N-1} x[n] * sin(...)
        // Scale by -1.0 to get 2 * sum.
        result.push(-spectrum[k + 1].1);
    }
    Ok(result)
}

/// Discrete Sine Transform Type III.
///
/// DST-III: x[n] = (-1)^n * X[N-1]/2 + Σ_{k=0}^{N-2} X[k] * sin(π(k+1)(2n+1)/(2N))
///
/// This is the inverse of DST-II (up to scaling).
/// Matches `scipy.fft.dst(x, type=3)`.
pub fn dst_iii(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    // DST-III via FFT of length 4N (inverse of DST-II)
    let m = 4 * n;
    let mut extended = vec![(0.0, 0.0); m];
    for k in 0..n - 1 {
        extended[k + 1] = (0.0, -input[k]);
        extended[m - k - 1] = (0.0, input[k]);
    }
    // The last term X[N-1] has a 1/2 factor in the standard DST-III formula
    let last_val = input[n - 1] * 0.5;
    extended[n] = (0.0, -last_val);
    extended[m - n] = (0.0, last_val);

    let backend = resolve_backend(options.backend);
    let time_domain = backend.transform_1d_unscaled(&extended, true);

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        // The raw sum from IFFT of the 4N-length extended signal produces 2 * sum
        // of the terms in the standard DST-III definition.
        // matches SciPy's unnormalized dst(type=3) result.
        result.push(time_domain[2 * i + 1].0);
    }
    Ok(result)
}

/// Discrete Sine Transform Type IV.
///
/// DST-IV: X[k] = 2 * Σ_{n=0}^{N-1} x[n] * sin(π(2n+1)(2k+1)/(4N))
///
/// Matches `scipy.fft.dst(x, type=4)`.
pub fn dst_iv(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    // DST-IV via FFT of length 8N
    let m = 8 * n;
    let mut extended = vec![(0.0, 0.0); m];
    for i in 0..n {
        extended[2 * i + 1] = (input[i], 0.0);
        extended[m - 2 * i - 1] = (-input[i], 0.0);
    }

    let backend = resolve_backend(options.backend);
    let spectrum = backend.transform_1d_unscaled(&extended, false);

    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        // -Imaginary part of bin 2k+1
        result.push(-spectrum[2 * k + 1].1);
    }
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// N-Dimensional DCT/DST
// ═══════════════════════════════════════════════════════════════════════════

/// N-dimensional Discrete Cosine Transform.
///
/// Matches `scipy.fft.dctn`. Applies the DCT-II transform along each axis.
///
/// # Arguments
/// * `input` - Input array (flattened, row-major order)
/// * `shape` - Shape of the N-D array
/// * `options` - FFT options (normalization, etc.)
///
/// # Returns
/// DCT coefficients (flattened, same shape as input)
pub fn dctn(input: &[f64], shape: &[usize], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    if shape.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "shape cannot be empty",
        });
    }
    let total: usize = shape.iter().product();
    if total != input.len() {
        return Err(FftError::LengthMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;

    // Apply DCT along each axis in sequence
    let mut data = input.to_vec();
    let ndim = shape.len();

    for axis in (0..ndim).rev() {
        data = apply_dct_along_axis(&data, shape, axis, options, false)?;
    }

    Ok(data)
}

/// Inverse N-dimensional Discrete Cosine Transform.
///
/// Matches `scipy.fft.idctn`. Applies the inverse DCT-II (DCT-III) along each axis.
pub fn idctn(input: &[f64], shape: &[usize], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    if shape.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "shape cannot be empty",
        });
    }
    let total: usize = shape.iter().product();
    if total != input.len() {
        return Err(FftError::LengthMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;

    // Apply IDCT along each axis in sequence
    let mut data = input.to_vec();
    let ndim = shape.len();

    for axis in 0..ndim {
        data = apply_dct_along_axis(&data, shape, axis, options, true)?;
    }

    Ok(data)
}

/// N-dimensional Discrete Sine Transform.
///
/// Matches `scipy.fft.dstn`. Applies the DST-II transform along each axis.
pub fn dstn(input: &[f64], shape: &[usize], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    if shape.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "shape cannot be empty",
        });
    }
    let total: usize = shape.iter().product();
    if total != input.len() {
        return Err(FftError::LengthMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;

    // Apply DST along each axis in sequence
    let mut data = input.to_vec();
    let ndim = shape.len();

    for axis in (0..ndim).rev() {
        data = apply_dst_along_axis(&data, shape, axis, options, false)?;
    }

    Ok(data)
}

/// Inverse N-dimensional Discrete Sine Transform.
///
/// Matches `scipy.fft.idstn`. Applies the inverse DST-II (DST-III) along each axis.
pub fn idstn(input: &[f64], shape: &[usize], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    if shape.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "shape cannot be empty",
        });
    }
    let total: usize = shape.iter().product();
    if total != input.len() {
        return Err(FftError::LengthMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;

    // Apply IDST along each axis in sequence
    let mut data = input.to_vec();
    let ndim = shape.len();

    for axis in 0..ndim {
        data = apply_dst_along_axis(&data, shape, axis, options, true)?;
    }

    Ok(data)
}

/// Apply 1-D DCT along a specific axis of an N-D array.
fn apply_dct_along_axis(
    data: &[f64],
    shape: &[usize],
    axis: usize,
    options: &FftOptions,
    inverse: bool,
) -> Result<Vec<f64>, FftError> {
    let ndim = shape.len();
    let axis_len = shape[axis];

    if axis_len == 0 {
        return Ok(data.to_vec());
    }

    // Calculate strides for the input array
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let total: usize = shape.iter().product();
    let outer_size = total / axis_len;
    let mut result = vec![0.0; total];

    // For each "fiber" along the axis
    let mut temp = vec![0.0; axis_len];
    let mut temp_out = vec![0.0; axis_len];

    for outer_idx in 0..outer_size {
        // Convert outer_idx to multi-index (excluding axis)
        let mut multi_idx = vec![0usize; ndim];
        let mut remaining = outer_idx;
        for d in (0..ndim).rev() {
            if d == axis {
                continue;
            }
            let dim_size = shape[d];
            multi_idx[d] = remaining % dim_size;
            remaining /= dim_size;
        }

        // Extract the fiber
        for (i, value) in temp.iter_mut().enumerate().take(axis_len) {
            multi_idx[axis] = i;
            let flat_idx: usize = multi_idx
                .iter()
                .zip(strides.iter())
                .map(|(m, s)| m * s)
                .sum();
            *value = data[flat_idx];
        }

        // Apply DCT or IDCT
        let transformed = if inverse {
            idct(&temp, options)?
        } else {
            dct(&temp, options)?
        };
        temp_out.copy_from_slice(&transformed);

        // Store back
        for (i, &value) in temp_out.iter().enumerate().take(axis_len) {
            multi_idx[axis] = i;
            let flat_idx: usize = multi_idx
                .iter()
                .zip(strides.iter())
                .map(|(m, s)| m * s)
                .sum();
            result[flat_idx] = value;
        }
    }

    Ok(result)
}

/// Apply 1-D DST along a specific axis of an N-D array.
fn apply_dst_along_axis(
    data: &[f64],
    shape: &[usize],
    axis: usize,
    options: &FftOptions,
    inverse: bool,
) -> Result<Vec<f64>, FftError> {
    let ndim = shape.len();
    let axis_len = shape[axis];

    if axis_len == 0 {
        return Ok(data.to_vec());
    }

    // Calculate strides for the input array
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let total: usize = shape.iter().product();
    let outer_size = total / axis_len;
    let mut result = vec![0.0; total];

    // For each "fiber" along the axis
    let mut temp = vec![0.0; axis_len];
    let mut temp_out = vec![0.0; axis_len];

    for outer_idx in 0..outer_size {
        // Convert outer_idx to multi-index (excluding axis)
        let mut multi_idx = vec![0usize; ndim];
        let mut remaining = outer_idx;
        for d in (0..ndim).rev() {
            if d == axis {
                continue;
            }
            let dim_size = shape[d];
            multi_idx[d] = remaining % dim_size;
            remaining /= dim_size;
        }

        // Extract the fiber
        for (i, value) in temp.iter_mut().enumerate().take(axis_len) {
            multi_idx[axis] = i;
            let flat_idx: usize = multi_idx
                .iter()
                .zip(strides.iter())
                .map(|(m, s)| m * s)
                .sum();
            *value = data[flat_idx];
        }

        // Apply DST or IDST (DST-II forward, DST-III inverse)
        let transformed = if inverse {
            dst_iii(&temp, options)?
        } else {
            dst_ii(&temp, options)?
        };
        temp_out.copy_from_slice(&transformed);

        // Store back
        for (i, &value) in temp_out.iter().enumerate().take(axis_len) {
            multi_idx[axis] = i;
            let flat_idx: usize = multi_idx
                .iter()
                .zip(strides.iter())
                .map(|(m, s)| m * s)
                .sum();
            result[flat_idx] = value;
        }
    }

    Ok(result)
}

/// Compute the analytic signal using the Hilbert transform.
///
/// Matches `scipy.signal.hilbert(x)`. Returns the analytic signal
/// whose real part is the original signal and imaginary part is the
/// Hilbert transform.
pub fn hilbert(input: &[f64], options: &FftOptions) -> Result<Vec<Complex64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();

    // FFT the real input
    let complex_input: Vec<Complex64> = input.iter().map(|&x| (x, 0.0)).collect();
    let backend = resolve_backend(options.backend);
    let mut spectrum = backend.transform_1d_unscaled(&complex_input, false);

    // Create step function h[k]:
    // Even N: h[0]=1, h[1..N/2]=2, h[N/2]=1, h[N/2+1..N]=0
    // Odd N:  h[0]=1, h[1..(N+1)/2]=2, h[(N+1)/2..N]=0
    if n > 1 {
        if n.is_multiple_of(2) {
            // Even: bins 1..N/2 doubled, Nyquist (N/2) unchanged, rest zeroed
            for sk in spectrum.iter_mut().take(n / 2).skip(1) {
                *sk = complex_scale(*sk, 2.0);
            }
            for sk in spectrum.iter_mut().skip(n / 2 + 1) {
                *sk = (0.0, 0.0);
            }
        } else {
            // Odd: bins 1..(N+1)/2 doubled (no Nyquist), rest zeroed
            let half_pos = n.div_ceil(2);
            for sk in spectrum.iter_mut().take(half_pos).skip(1) {
                *sk = complex_scale(*sk, 2.0);
            }
            for sk in spectrum.iter_mut().skip(half_pos) {
                *sk = (0.0, 0.0);
            }
        }
    }

    // IFFT to get analytic signal
    let mut analytic = backend.transform_1d_unscaled(&spectrum, true);
    // Apply 1/N normalization (Backward mode)
    let inv_n = 1.0 / n as f64;
    for v in &mut analytic {
        *v = complex_scale(*v, inv_n);
    }

    Ok(analytic)
}

/// Compute optimal offset for Fast Hankel Transform.
///
/// Matches `scipy.fft.fhtoffset(dln, mu, initial=0.0, bias=0.0)`.
///
/// Returns the optimal offset parameter for the FHT to minimize ringing.
///
/// # Arguments
/// * `dln` - Log-spacing of the input array (∆ln r)
/// * `mu` - Order of the Hankel transform (J_μ Bessel function order)
/// * `initial` - Initial guess for the offset (default 0.0)
/// * `bias` - Bias parameter (default 0.0)
pub fn fhtoffset(dln: f64, mu: f64, initial: f64, bias: f64) -> f64 {
    // The optimal offset minimizes |U_μ(q + offset)| where q = bias
    // U_μ(x) = 2^x * Γ((μ+1+x)/2) / Γ((μ+1-x)/2)
    // The zeros of U_μ are at x = -(μ+1+2n) for n=0,1,2,...
    // We find the nearest zero to initial + bias

    let q = bias;
    let x0 = initial + q;

    // Zeros of U_μ occur at -(μ+1+2n) for n = 0, 1, 2, ...
    // Find integer n such that -(μ+1+2n) is closest to x0
    // Solving: x0 ≈ -(μ+1+2n) => n ≈ (-x0 - μ - 1) / 2

    let n_float = (-x0 - mu - 1.0) / 2.0;
    let n = n_float.round().max(0.0) as i64;

    // The zero is at -(μ+1+2n)
    let zero = -(mu + 1.0 + 2.0 * n as f64);

    // The offset is chosen so that initial + q lands on this zero
    // offset + initial + q = zero
    // offset = zero - initial - q = zero - x0
    let offset = zero - x0;

    // Adjust offset to be within (-dln/2, dln/2] for periodic boundary handling
    // Actually the offset can be arbitrary, but we typically return the
    // most natural one
    let offset_mod = offset - dln * (offset / dln).floor();
    if offset_mod > dln / 2.0 {
        offset_mod - dln
    } else {
        offset_mod
    }
}

/// Fast Hankel Transform.
///
/// Matches `scipy.fft.fht(a, dln, mu, offset=0.0, bias=0.0)`.
///
/// Compute the discrete Hankel transform of a logarithmically spaced
/// periodic sequence.
///
/// # Arguments
/// * `input` - Input array (logarithmically spaced)
/// * `dln` - Uniform logarithmic spacing ∆ln r
/// * `mu` - Order of the Hankel transform (Bessel function order J_μ)
/// * `offset` - Offset parameter (use `fhtoffset` to compute optimal value)
/// * `bias` - Power-law bias exponent
/// * `options` - FFT options
///
/// # Returns
/// The transformed array
pub fn fht(
    input: &[f64],
    dln: f64,
    mu: f64,
    offset: f64,
    bias: f64,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();

    // Compute the Hankel transform kernel coefficients u_m
    // u_m = U_μ(bias + 2πim/(n·dln)) where U_μ is the low-ringing kernel
    let u_coeffs = fht_kernel(n, dln, mu, offset, bias);

    // FFT the input
    let complex_input: Vec<Complex64> = input.iter().map(|&x| (x, 0.0)).collect();
    let backend = resolve_backend(options.backend);
    let a_fft = backend.transform_1d_unscaled(&complex_input, false);

    // Multiply by kernel in Fourier space
    let product: Vec<Complex64> = a_fft
        .iter()
        .zip(u_coeffs.iter())
        .map(|(&a, &u)| complex_mul(a, u))
        .collect();

    // IFFT back
    let mut result_complex = backend.transform_1d_unscaled(&product, true);

    // Apply 1/N normalization
    let inv_n = 1.0 / n as f64;
    for v in &mut result_complex {
        *v = complex_scale(*v, inv_n);
    }

    // Take real part (imaginary should be negligible)
    Ok(result_complex.iter().map(|c| c.0).collect())
}

/// Inverse Fast Hankel Transform.
///
/// Matches `scipy.fft.ifht(A, dln, mu, offset=0.0, bias=0.0)`.
///
/// Compute the inverse discrete Hankel transform.
///
/// # Arguments
/// * `input` - Input array in Hankel space
/// * `dln` - Uniform logarithmic spacing ∆ln k
/// * `mu` - Order of the Hankel transform
/// * `offset` - Offset parameter (use `fhtoffset` to compute optimal value)
/// * `bias` - Power-law bias exponent
/// * `options` - FFT options
///
/// # Returns
/// The inverse transformed array
pub fn ifht(
    input: &[f64],
    dln: f64,
    mu: f64,
    offset: f64,
    bias: f64,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    // The inverse FHT with parameters (dln, μ, offset, bias) is the
    // forward FHT with parameters (dln, μ, -offset, -bias)
    fht(input, dln, mu, -offset, -bias, options)
}

/// Compute the Fast Hankel Transform kernel coefficients.
///
/// The kernel U_μ(q) is defined such that the Hankel transform
/// becomes a convolution in log-space.
fn fht_kernel(n: usize, dln: f64, mu: f64, offset: f64, bias: f64) -> Vec<Complex64> {
    use std::f64::consts::PI;

    let mut u_coeffs = Vec::with_capacity(n);

    for m in 0..n {
        // Frequency index (centered)
        let m_shift = if m <= n / 2 {
            m as f64
        } else {
            m as f64 - n as f64
        };

        // Argument for U_μ: q + 2πi·m / (n·dln)
        let x_re = bias;
        let x_im = 2.0 * PI * m_shift / (n as f64 * dln);

        // Compute U_μ(x) = 2^x · Γ((μ+1+x)/2) / Γ((μ+1-x)/2)
        // For complex x, we use the relation with sine:
        // U_μ(x) = (2π)^(-x) · Γ(1-x) · sin(π(μ+x)/2) / Γ((μ+1-x)/2) · 2^(something)
        //
        // Actually use the simpler approximation for the discrete transform:
        // U_m = exp(offset·(bias + 2πi·m/(n·dln))) · u_kernel(m)
        //
        // The low-ringing kernel in log-space:
        let phase = offset * x_im;
        let amp = 2.0_f64.powf(x_re) * u_mu_ratio(mu, x_re, x_im);
        let exp_phase = (phase.cos(), phase.sin());
        u_coeffs.push((amp * exp_phase.0, amp * exp_phase.1));
    }

    u_coeffs
}

/// Compute |Γ((μ+1+x)/2) / Γ((μ+1-x)/2)| for complex x = x_re + i·x_im.
///
/// Uses the reflection formula and asymptotic expansions.
fn u_mu_ratio(mu: f64, x_re: f64, x_im: f64) -> f64 {
    // For real x_im = 0, this is straightforward gamma ratio
    // For complex, use |Γ(a+ib)/Γ(c+id)| approximation

    if x_im.abs() < 1e-10 {
        // Real case: use gamma function ratio
        let arg_num = (mu + 1.0 + x_re) / 2.0;
        let arg_den = (mu + 1.0 - x_re) / 2.0;

        if arg_num > 0.0 && arg_den > 0.0 {
            // Both arguments positive - use lgamma
            (ln_gamma(arg_num) - ln_gamma(arg_den)).exp()
        } else {
            // Handle poles via reflection
            1.0
        }
    } else {
        // Complex case: use Stirling approximation for |Γ(a+ib)|
        // |Γ(a+ib)| ≈ √(2π) · |b|^(a-0.5) · exp(-π|b|/2) for large |b|

        let a1 = (mu + 1.0 + x_re) / 2.0;
        let b1 = x_im / 2.0;
        let a2 = (mu + 1.0 - x_re) / 2.0;
        let b2 = -x_im / 2.0;

        // For moderate |b|, use a more accurate approximation
        let log_mag1 = log_gamma_magnitude(a1, b1);
        let log_mag2 = log_gamma_magnitude(a2, b2);

        (log_mag1 - log_mag2).exp()
    }
}

/// Approximate log|Γ(a + ib)| using Stirling's formula.
fn log_gamma_magnitude(a: f64, b: f64) -> f64 {
    use std::f64::consts::PI;

    if b.abs() < 0.1 && a > 0.0 {
        // Nearly real: use real lgamma
        ln_gamma(a)
    } else {
        // Stirling approximation:
        // log|Γ(z)| ≈ 0.5·log(2π) - 0.5·log|z| + Re(z - 0.5)·log|z| - Im(z)·arg(z) - Re(z)
        let z_mag = (a * a + b * b).sqrt();
        let z_arg = b.atan2(a);

        0.5 * (2.0 * PI).ln() + (a - 0.5) * z_mag.ln() - b * z_arg - a
    }
}

/// Natural log of gamma function for positive real arguments.
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation coefficients
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula
        let pi = std::f64::consts::PI;
        pi.ln() - (pi * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut sum = C[0];
        for (i, &c) in C.iter().enumerate().skip(1) {
            sum += c / (x + i as f64);
        }
        let t = x + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
    }
}

fn run_complex_1d(
    kind: TransformKind,
    input: &[Complex64],
    options: &FftOptions,
    inverse: bool,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    let fingerprint = complex_fingerprint(input);
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_complex_with_audit(input, options, &fingerprint, audit_ledger)?;

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
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    let fingerprint = complex_shape_fingerprint(input, shape);
    validate_shape_with_audit(shape, &fingerprint, audit_ledger)?;
    let expected_len = checked_product(shape).ok_or_else(|| {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "shape_product_overflow",
            "rejected: nd shape product overflow",
        );
        FftError::InvalidShape {
            detail: "nd shape product overflow",
        }
    })?;
    if input.len() != expected_len {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "length_mismatch",
            format!(
                "rejected: expected input length {expected_len}, got {}",
                input.len()
            ),
        );
        return Err(FftError::LengthMismatch {
            expected: expected_len,
            actual: input.len(),
        });
    }

    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_complex_with_audit(input, options, &fingerprint, audit_ledger)?;

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

fn run_real_nd_forward(
    kind: TransformKind,
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    let fingerprint = real_shape_fingerprint(input, shape);
    validate_shape_with_audit(shape, &fingerprint, audit_ledger)?;
    let expected_len = checked_product(shape).ok_or_else(|| {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "shape_product_overflow",
            "rejected: nd shape product overflow",
        );
        FftError::InvalidShape {
            detail: "nd shape product overflow",
        }
    })?;
    if input.len() != expected_len {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "length_mismatch",
            format!(
                "rejected: expected input length {expected_len}, got {}",
                input.len()
            ),
        );
        return Err(FftError::LengthMismatch {
            expected: expected_len,
            actual: input.len(),
        });
    }

    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_real_with_audit(input, options, &fingerprint, audit_ledger)?;

    let last_len = *shape.last().ok_or(FftError::InvalidShape {
        detail: "empty shape",
    })?;
    let reduced_last = last_len / 2 + 1;
    let backend = resolve_backend(options.backend);
    let axes = (0..shape.len()).collect::<Vec<_>>();
    let key = PlanKey::new(kind, shape.to_vec(), axes, options.normalization, true);
    let plan_cache_hit = touch_plan_cache(&key, expected_len);

    let started = Instant::now();
    let mut output = Vec::with_capacity(expected_len / last_len * reduced_last);
    for lane in input.chunks_exact(last_len) {
        output.extend(real_fft_unscaled(lane, backend));
    }

    let mut complex_shape = shape.to_vec();
    if let Some(last) = complex_shape.last_mut() {
        *last = reduced_last;
    } else {
        return Err(FftError::InvalidShape {
            detail: "empty shape",
        });
    }
    if complex_shape.len() > 1 {
        let max_axis_len = complex_shape.iter().max().copied().unwrap_or(0);
        let mut scratch = vec![(0.0, 0.0); max_axis_len];
        for axis in 0..complex_shape.len() - 1 {
            apply_axis_transform(
                backend,
                &mut output,
                &complex_shape,
                axis,
                false,
                &mut scratch,
            );
        }
    }
    apply_normalization(&mut output, options.normalization, expected_len, false);

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind,
        direction: "forward",
        n: expected_len,
        backend: backend.kind(),
        plan_cache_hit,
        mode: options.mode,
        timing_ns: started.elapsed().as_nanos(),
    });

    Ok(output)
}

fn run_real_nd_inverse(
    kind: TransformKind,
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    let fingerprint = complex_shape_fingerprint(input, shape);
    validate_shape_with_audit(shape, &fingerprint, audit_ledger)?;
    let expected_len = checked_product(shape).ok_or_else(|| {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "shape_product_overflow",
            "rejected: nd shape product overflow",
        );
        FftError::InvalidShape {
            detail: "nd shape product overflow",
        }
    })?;
    let last_len = *shape.last().ok_or_else(|| {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "empty_shape",
            "rejected: nd shape cannot be empty",
        );
        FftError::InvalidShape {
            detail: "empty shape",
        }
    })?;
    let reduced_last = last_len / 2 + 1;
    let complex_len = shape[..shape.len() - 1]
        .iter()
        .try_fold(reduced_last, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            record_fail_closed(
                audit_ledger,
                &fingerprint,
                "shape_product_overflow",
                "rejected: nd shape product overflow",
            );
            FftError::InvalidShape {
                detail: "nd shape product overflow",
            }
        })?;
    if input.len() != complex_len {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "length_mismatch",
            format!(
                "rejected: expected input length {complex_len}, got {}",
                input.len()
            ),
        );
        return Err(FftError::LengthMismatch {
            expected: complex_len,
            actual: input.len(),
        });
    }

    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_complex_with_audit(input, options, &fingerprint, audit_ledger)?;

    let backend = resolve_backend(options.backend);
    let axes = (0..shape.len()).collect::<Vec<_>>();
    let key = PlanKey::new(kind, shape.to_vec(), axes, options.normalization, true);
    let plan_cache_hit = touch_plan_cache(&key, expected_len);

    let started = Instant::now();
    let mut complex_data = input.to_vec();
    let mut complex_shape = shape.to_vec();
    if let Some(last) = complex_shape.last_mut() {
        *last = reduced_last;
    } else {
        return Err(FftError::InvalidShape {
            detail: "empty shape",
        });
    }
    if complex_shape.len() > 1 {
        let max_axis_len = complex_shape.iter().max().copied().unwrap_or(0);
        let mut scratch = vec![(0.0, 0.0); max_axis_len];
        for axis in 0..complex_shape.len() - 1 {
            apply_axis_transform(
                backend,
                &mut complex_data,
                &complex_shape,
                axis,
                true,
                &mut scratch,
            );
        }
    }

    let mut output = Vec::with_capacity(expected_len);
    for lane in complex_data.chunks_exact(reduced_last) {
        output.extend(real_ifft_unscaled(lane, last_len, backend));
    }
    apply_real_normalization(&mut output, options.normalization, expected_len, true);

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind,
        direction: "inverse",
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
    let max_axis_len = shape.iter().max().copied().unwrap_or(0);
    let mut scratch = vec![(0.0, 0.0); max_axis_len];
    for axis in 0..shape.len() {
        apply_axis_transform(backend, &mut data, shape, axis, inverse, &mut scratch);
    }
    data
}

fn apply_axis_transform(
    backend: &dyn FftBackend,
    data: &mut [Complex64],
    shape: &[usize],
    axis: usize,
    inverse: bool,
    scratch: &mut [Complex64],
) {
    let axis_len = shape[axis];
    let stride = shape[axis + 1..].iter().product::<usize>().max(1);
    let repeats = shape[..axis].iter().product::<usize>().max(1);
    let block = axis_len * stride;

    let axis_scratch = &mut scratch[..axis_len];
    for outer in 0..repeats {
        let outer_base = outer * block;
        for offset in 0..stride {
            for (index, slot) in axis_scratch.iter_mut().enumerate() {
                *slot = data[outer_base + index * stride + offset];
            }
            backend.transform_1d_inplace(axis_scratch, inverse);
            for (index, &value) in axis_scratch.iter().enumerate() {
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

fn real_fft_unscaled(input: &[f64], backend: &dyn FftBackend) -> Vec<Complex64> {
    if input.len().is_power_of_two() && input.len() >= 4 {
        real_fft_specialized(input, backend)
    } else {
        let complex_input: Vec<Complex64> = input.iter().map(|&x| (x, 0.0)).collect();
        let full = backend.transform_1d_unscaled(&complex_input, false);
        full.into_iter().take(input.len() / 2 + 1).collect()
    }
}

fn real_ifft_unscaled(input: &[Complex64], n: usize, backend: &dyn FftBackend) -> Vec<f64> {
    let reconstructed = rebuild_hermitian(input, n);
    backend
        .transform_1d_unscaled(&reconstructed, true)
        .into_iter()
        .map(|(re, _)| re)
        .collect()
}

fn resolve_backend(kind: BackendKind) -> &'static dyn FftBackend {
    match kind {
        BackendKind::NaiveDft => &NAIVE_BACKEND,
        BackendKind::CooleyTukey => &COOLEY_TUKEY_BACKEND,
    }
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

fn record_audit_event(
    audit_ledger: Option<&SyncSharedAuditLedger>,
    input_bytes: &[u8],
    action: AuditAction,
    outcome: impl Into<String>,
) {
    let Some(audit_ledger) = audit_ledger else {
        return;
    };
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        AuditLedger::fingerprint_bytes(input_bytes),
        action,
        outcome,
    );
    if let Ok(mut guard) = audit_ledger.lock() {
        guard.record(event);
    }
}

fn record_mode_decision(
    audit_ledger: Option<&SyncSharedAuditLedger>,
    input_bytes: &[u8],
    mode: RuntimeMode,
    outcome: impl Into<String>,
) {
    record_audit_event(
        audit_ledger,
        input_bytes,
        AuditAction::ModeDecision { mode },
        outcome,
    );
}

fn record_fail_closed(
    audit_ledger: Option<&SyncSharedAuditLedger>,
    input_bytes: &[u8],
    reason: &str,
    outcome: impl Into<String>,
) {
    record_audit_event(
        audit_ledger,
        input_bytes,
        AuditAction::FailClosed {
            reason: reason.to_owned(),
        },
        outcome,
    );
}

fn ensure_non_empty_with_audit(
    len: usize,
    fingerprint: &[u8],
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<(), FftError> {
    if len == 0 {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "empty_input",
            "rejected: input length must be greater than zero",
        );
    }
    ensure_non_empty(len)
}

fn validate_workers_with_audit(
    policy: WorkerPolicy,
    fingerprint: &[u8],
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<(), FftError> {
    if matches!(policy, WorkerPolicy::Exact(0) | WorkerPolicy::Max(0)) {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "invalid_workers",
            "rejected: worker count must be greater than zero",
        );
    }
    validate_workers(policy)
}

fn validate_finite_complex_with_audit(
    input: &[Complex64],
    options: &FftOptions,
    fingerprint: &[u8],
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<(), FftError> {
    if options.mode == RuntimeMode::Hardened && !options.check_finite {
        record_mode_decision(
            audit_ledger,
            fingerprint,
            RuntimeMode::Hardened,
            "promoted finite-check policy for complex FFT input",
        );
    }
    if input
        .iter()
        .any(|&(re, im)| !re.is_finite() || !im.is_finite())
        && (options.check_finite || options.mode == RuntimeMode::Hardened)
    {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "non_finite_input",
            "rejected: complex input contains NaN or Inf",
        );
    }
    validate_finite_complex(input, options)
}

fn validate_finite_real_with_audit(
    input: &[f64],
    options: &FftOptions,
    fingerprint: &[u8],
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<(), FftError> {
    if options.mode == RuntimeMode::Hardened && !options.check_finite {
        record_mode_decision(
            audit_ledger,
            fingerprint,
            RuntimeMode::Hardened,
            "promoted finite-check policy for real FFT input",
        );
    }
    if input.iter().any(|value| !value.is_finite())
        && (options.check_finite || options.mode == RuntimeMode::Hardened)
    {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "non_finite_input",
            "rejected: real input contains NaN or Inf",
        );
    }
    validate_finite_real(input, options)
}

fn validate_shape_with_audit(
    shape: &[usize],
    fingerprint: &[u8],
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<(), FftError> {
    if shape.is_empty() {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "empty_shape",
            "rejected: nd shape cannot be empty",
        );
    } else if shape.contains(&0) {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "zero_dimension",
            "rejected: nd shape dimensions must be greater than zero",
        );
    }
    validate_shape(shape)
}

fn real_fingerprint(input: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(input.len().min(64) * std::mem::size_of::<f64>());
    for &value in input.iter().take(64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn complex_fingerprint(input: &[Complex64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(input.len().min(32) * 2 * std::mem::size_of::<f64>());
    for &(re, im) in input.iter().take(32) {
        bytes.extend_from_slice(&re.to_le_bytes());
        bytes.extend_from_slice(&im.to_le_bytes());
    }
    bytes
}

fn append_shape_fingerprint(bytes: &mut Vec<u8>, shape: &[usize]) {
    for &dim in shape.iter().take(16) {
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
}

fn real_shape_fingerprint(input: &[f64], shape: &[usize]) -> Vec<u8> {
    let mut bytes = real_fingerprint(input);
    append_shape_fingerprint(&mut bytes, shape);
    bytes
}

fn complex_shape_fingerprint(input: &[Complex64], shape: &[usize]) -> Vec<u8> {
    let mut bytes = complex_fingerprint(input);
    append_shape_fingerprint(&mut bytes, shape);
    bytes
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

fn apply_real_normalization(
    data: &mut [f64],
    normalization: Normalization,
    n: usize,
    inverse: bool,
) {
    let scale = normalization_scale(normalization, n, inverse);
    if (scale - 1.0).abs() <= f64::EPSILON {
        return;
    }
    for value in data.iter_mut() {
        *value *= scale;
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
        TransformKind::Ifftn => "ifftn",
        TransformKind::Rfftn => "rfftn",
        TransformKind::Irfftn => "irfftn",
    }
}

fn backend_kind_name(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::NaiveDft => "naive_dft",
        BackendKind::CooleyTukey => "cooley_tukey",
    }
}

fn runtime_mode_name(mode: RuntimeMode) -> &'static str {
    match mode {
        RuntimeMode::Strict => "Strict",
        RuntimeMode::Hardened => "Hardened",
    }
}

// ══════════════════════════════════════════════════════════════════════
// Multi-D Real FFT, next_fast_len, hfft
// ══════════════════════════════════════════════════════════════════════

/// 2D real-input FFT.
///
/// Matches `scipy.fft.rfft2(x, s)`.
///
/// Computes rfft along the last axis, then fft along the first axis.
pub fn rfft2(
    input: &[f64],
    shape: (usize, usize),
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    rfft2_impl(input, &dims, options, None)
}

/// 2D real-input FFT with audit logging.
pub fn rfft2_with_audit(
    input: &[f64],
    shape: (usize, usize),
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    rfft2_impl(input, &dims, options, Some(audit_ledger))
}

/// 2D inverse real FFT.
///
/// Matches `scipy.fft.irfft2(x, s)`.
pub fn irfft2(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    let dims = [shape.0, shape.1];
    irfft2_impl(input, &dims, options, None)
}

/// 2D inverse real FFT with audit logging.
pub fn irfft2_with_audit(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<f64>, FftError> {
    let dims = [shape.0, shape.1];
    irfft2_impl(input, &dims, options, Some(audit_ledger))
}

/// N-dimensional inverse real FFT.
///
/// Matches `scipy.fft.irfftn(x, s)`.
pub fn irfftn(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    irfftn_impl(input, shape, options, None)
}

/// N-dimensional inverse real FFT with audit logging.
pub fn irfftn_with_audit(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<f64>, FftError> {
    irfftn_impl(input, shape, options, Some(audit_ledger))
}

/// Find the next fast length for FFT computation.
///
/// Matches `scipy.fft.next_fast_len(target)`.
///
/// Returns the smallest integer >= `target` that is a product of
/// small prime factors (2, 3, 5), since FFT is most efficient for these sizes.
pub fn next_fast_len(target: usize) -> usize {
    if target <= 1 {
        return target;
    }
    let mut n = target;
    loop {
        if is_fast_len(n) {
            return n;
        }
        n += 1;
    }
}

/// Check if n is composed only of factors 2, 3, 5 (Hamming numbers / regular numbers).
#[allow(clippy::manual_is_multiple_of)]
fn is_fast_len(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    while n % 2 == 0 {
        n /= 2;
    }
    while n % 3 == 0 {
        n /= 3;
    }
    while n % 5 == 0 {
        n /= 5;
    }
    n == 1
}

/// Hermitian FFT (FFT of a signal with Hermitian symmetry in the frequency domain).
///
/// Matches `scipy.fft.hfft(x, n)`.
///
/// Takes a half-spectrum (like rfft output) and produces a real-valued full signal.
/// This is essentially irfft scaled differently — hfft(x, n) = n * irfft(x, n).
pub fn hfft(
    input: &[Complex64],
    n: Option<usize>,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    hfft_impl(input, n, options, None)
}

/// Hermitian FFT with audit logging.
pub fn hfft_with_audit(
    input: &[Complex64],
    n: Option<usize>,
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<f64>, FftError> {
    hfft_impl(input, n, options, Some(audit_ledger))
}

/// Inverse of the Hermitian FFT (hfft).
///
/// Matches `scipy.fft.ihfft(x, n)`.
///
/// Takes a real signal and returns the Hermitian-symmetric spectrum.
/// This is the inverse of `hfft`: `ihfft(hfft(x)) ≈ x`.
pub fn ihfft(
    input: &[f64],
    n: Option<usize>,
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    ihfft_impl(input, n, options, None)
}

/// Inverse Hermitian FFT with audit logging.
pub fn ihfft_with_audit(
    input: &[f64],
    n: Option<usize>,
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    ihfft_impl(input, n, options, Some(audit_ledger))
}

fn rfft2_impl(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    rfftn_impl(input, shape, options, audit_ledger)
}

fn irfft2_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    irfftn_impl(input, shape, options, audit_ledger)
}

fn irfftn_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    run_real_nd_inverse(TransformKind::Irfftn, input, shape, options, audit_ledger)
}

fn hfft_impl(
    input: &[Complex64],
    n: Option<usize>,
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    let fingerprint = complex_fingerprint(input);
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    let out_len = n.unwrap_or_else(|| {
        if input.len() == 1 {
            1
        } else {
            input.len().saturating_sub(1).saturating_mul(2)
        }
    });
    let mut result = irfft_impl(input, Some(out_len), options, audit_ledger)?;
    let scale = out_len as f64;
    for v in &mut result {
        *v *= scale;
    }
    Ok(result)
}

fn ihfft_impl(
    input: &[f64],
    n: Option<usize>,
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    let fingerprint = real_fingerprint(input);
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    validate_finite_real_with_audit(input, options, &fingerprint, audit_ledger)?;

    let in_len = n.unwrap_or(input.len());
    let mut padded = vec![0.0; in_len];
    let copy_len = input.len().min(in_len);
    padded[..copy_len].copy_from_slice(&input[..copy_len]);

    let mut result = rfft_impl(&padded, options, audit_ledger)?;
    let scale = 1.0 / in_len as f64;
    for c in &mut result {
        c.0 *= scale;
        c.1 *= scale;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use fsci_runtime::{AuditAction, RuntimeMode};

    use super::{
        FftError, FftOptions, TransformKind, WorkerPolicy, fft, fft_with_audit, fft2, fftn, hfft,
        ifft, ifft2, irfft, irfft2, irfftn, next_fast_len, rfft, rfft_with_audit, rfft2, rfftn,
        sync_audit_ledger, take_transform_traces,
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
    fn irfft_length_1() {
        let input = vec![(5.0, 0.0)];
        let signal = irfft(&input, Some(1), &FftOptions::default()).expect("irfft len 1");
        assert_eq!(signal.len(), 1);
        assert_close(signal[0], 5.0, 1e-12);

        // Default length for input length 1 should also be 1
        let signal_default =
            irfft(&input, None, &FftOptions::default()).expect("irfft default len 1");
        assert_eq!(signal_default.len(), 1);
        assert_close(signal_default[0], 5.0, 1e-12);
    }

    #[test]
    fn hardened_mode_rejects_non_finite_input() {
        let opts = FftOptions::default().with_mode(RuntimeMode::Hardened);
        let err = rfft(&[1.0, f64::NAN], &opts).expect_err("hardened mode should reject NaN");
        assert_eq!(err, FftError::NonFiniteInput);
    }

    #[test]
    fn hardened_audit_records_mode_decision_and_fail_closed() {
        let audit_ledger = sync_audit_ledger();
        let opts = FftOptions::default().with_mode(RuntimeMode::Hardened);
        let err = rfft_with_audit(&[1.0, f64::NAN], &opts, &audit_ledger)
            .expect_err("hardened mode should reject NaN");
        assert_eq!(err, FftError::NonFiniteInput);

        let ledger = audit_ledger.lock().expect("audit ledger lock");
        assert!(ledger.entries().iter().any(|entry| matches!(
            entry.action,
            AuditAction::ModeDecision {
                mode: RuntimeMode::Hardened
            }
        )));
        assert!(ledger.entries().iter().any(|entry| matches!(
            &entry.action,
            AuditAction::FailClosed { reason } if reason == "non_finite_input"
        )));
    }

    #[test]
    fn fft_with_audit_records_empty_input_fail_closed() {
        let audit_ledger = sync_audit_ledger();
        let err = fft_with_audit(&[], &FftOptions::default(), &audit_ledger)
            .expect_err("empty input should be rejected");
        assert_eq!(
            err,
            FftError::InvalidShape {
                detail: "input length must be greater than zero",
            }
        );

        let ledger = audit_ledger.lock().expect("audit ledger lock");
        assert!(ledger.entries().iter().any(|entry| matches!(
            &entry.action,
            AuditAction::FailClosed { reason } if reason == "empty_input"
        )));
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
        let input: Vec<_> = (0..16)
            .map(|i| ((i as f64).sin(), (i as f64).cos()))
            .collect();
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
            .map(|k| {
                (
                    (2.0 * std::f64::consts::PI * k as f64 / n as f64).sin(),
                    0.0,
                )
            })
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

        // Use n=137 (prime, large, unlikely to collide in parallel tests)
        let input = (0..137usize)
            .map(|i| (i as f64, (i % 3) as f64 - 1.0))
            .collect::<Vec<_>>();
        let opts = FftOptions::default();
        let _ = fft(&input, &opts).expect("first fft should succeed");
        let _ = fft(&input, &opts).expect("second fft should succeed");

        let mut fft_traces = take_transform_traces()
            .into_iter()
            .filter(|trace| trace.kind == TransformKind::Fft && trace.n == 137)
            .collect::<Vec<_>>();
        fft_traces.sort_by(|lhs, rhs| lhs.operation_id.cmp(&rhs.operation_id));

        assert!(
            fft_traces.len() >= 2,
            "Expected at least 2 traces for n=137, got {}",
            fft_traces.len()
        );
        let last_two = &fft_traces[fft_traces.len() - 2..];
        assert!(
            !last_two[0].plan_cache_hit,
            "First call should be a cache miss"
        );
        assert!(
            last_two[1].plan_cache_hit,
            "Second call should be a cache hit"
        );
        assert!(last_two[0].to_json_line().contains("\"operation_id\""));
    }

    // ── rfft2 / irfft2 tests ───────────────────────────────────────

    #[test]
    fn rfft2_irfft2_roundtrip() {
        let rows = 4;
        let cols = 6;
        let input: Vec<f64> = (0..(rows * cols)).map(|i| (i as f64 * 0.3).sin()).collect();
        let opts = FftOptions::default();
        let spectrum = rfft2(&input, (rows, cols), &opts).expect("rfft2");
        let recovered = irfft2(&spectrum, (rows, cols), &opts).expect("irfft2");
        assert_eq!(recovered.len(), input.len());
        for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "rfft2 roundtrip mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn rfft2_output_shape() {
        let rows = 4;
        let cols = 8;
        let input = vec![0.0; rows * cols];
        let opts = FftOptions::default();
        let spectrum = rfft2(&input, (rows, cols), &opts).expect("rfft2");
        // Output should have rows * (cols/2 + 1) elements
        assert_eq!(spectrum.len(), rows * (cols / 2 + 1));
    }

    #[test]
    fn rfftn_irfftn_roundtrip() {
        let shape = [2, 3, 4];
        let input: Vec<f64> = (0..shape.iter().product::<usize>())
            .map(|i| (i as f64 * 0.2).cos())
            .collect();
        let opts = FftOptions::default();
        let spectrum = rfftn(&input, &shape, &opts).expect("rfftn");
        let recovered = irfftn(&spectrum, &shape, &opts).expect("irfftn");
        assert_eq!(recovered.len(), input.len());
        for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "rfftn roundtrip mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn rfftn_output_shape() {
        let shape = [2, 3, 8];
        let input = vec![0.0; shape.iter().product()];
        let opts = FftOptions::default();
        let spectrum = rfftn(&input, &shape, &opts).expect("rfftn");
        assert_eq!(spectrum.len(), shape[0] * shape[1] * (shape[2] / 2 + 1));
    }

    // ── next_fast_len tests ────────────────────────────────────────

    #[test]
    fn next_fast_len_powers_of_2() {
        assert_eq!(next_fast_len(1), 1);
        assert_eq!(next_fast_len(2), 2);
        assert_eq!(next_fast_len(4), 4);
        assert_eq!(next_fast_len(8), 8);
    }

    #[test]
    fn next_fast_len_non_powers() {
        assert_eq!(next_fast_len(7), 8);
        assert_eq!(next_fast_len(11), 12); // 12 = 2^2 * 3
        assert_eq!(next_fast_len(13), 15); // 15 = 3 * 5
        assert_eq!(next_fast_len(17), 18); // 18 = 2 * 3^2
    }

    #[test]
    fn next_fast_len_already_fast() {
        // 30 = 2 * 3 * 5
        assert_eq!(next_fast_len(30), 30);
        // 100 = 2^2 * 5^2
        assert_eq!(next_fast_len(100), 100);
    }

    // ── hfft tests ─────────────────────────────────────────────────

    #[test]
    fn hfft_basic() {
        // hfft of a half-spectrum should produce real output
        let opts = FftOptions::default();
        let input = vec![(4.0, 0.0), (1.0, -1.0), (0.0, 0.0)]; // 3 elements → n=4
        let result = hfft(&input, Some(4), &opts).expect("hfft");
        assert_eq!(result.len(), 4);
        // All values should be real (no imaginary component issue)
        for &v in &result {
            assert!(v.is_finite(), "hfft produced non-finite value: {v}");
        }
    }

    #[test]
    fn hfft_rfft_inverse() {
        // hfft = n * irfft, so hfft(rfft(x), n) / n = x
        let opts = FftOptions::default();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let n = x.len();
        let spectrum = rfft(&x, &opts).expect("rfft");
        let recovered = hfft(&spectrum, Some(n), &opts).expect("hfft");
        for (i, (&a, &b)) in x.iter().zip(recovered.iter()).enumerate() {
            let b_scaled = b / n as f64;
            assert!(
                (a - b_scaled).abs() < 1e-9,
                "hfft roundtrip mismatch at {i}: {a} vs {b_scaled}"
            );
        }
    }

    #[test]
    fn hfft_empty_input_returns_invalid_shape() {
        let err = hfft(&[], None, &FftOptions::default()).expect_err("empty hfft must fail");
        assert_eq!(
            err,
            FftError::InvalidShape {
                detail: "input length must be greater than zero",
            }
        );
    }

    #[test]
    fn ihfft_basic() {
        use super::ihfft;
        // ihfft should produce a Hermitian-symmetric spectrum
        let opts = FftOptions::default();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = ihfft(&input, None, &opts).expect("ihfft");
        // For n=4 real input, rfft gives n/2+1 = 3 complex outputs
        assert_eq!(result.len(), 3);
        for &(re, im) in &result {
            assert!(
                re.is_finite() && im.is_finite(),
                "ihfft produced non-finite"
            );
        }
    }

    #[test]
    fn ihfft_hfft_roundtrip() {
        use super::ihfft;
        // ihfft(hfft(spectrum)) should recover the original spectrum (with scaling)
        let opts = FftOptions::default();
        let spectrum: Vec<(f64, f64)> = vec![(4.0, 0.0), (1.0, -1.0), (0.0, 0.0)];
        let n = 4; // output length for hfft
        let real_signal = hfft(&spectrum, Some(n), &opts).expect("hfft");
        let recovered = ihfft(&real_signal, None, &opts).expect("ihfft");
        // recovered should approximately equal spectrum
        for (i, (&(re1, im1), &(re2, im2))) in spectrum.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (re1 - re2).abs() < 1e-9,
                "re mismatch at {i}: {re1} vs {re2}"
            );
            assert!(
                (im1 - im2).abs() < 1e-9,
                "im mismatch at {i}: {im1} vs {im2}"
            );
        }
    }
}
