#![forbid(unsafe_code)]

//! Signal processing routines for FrankenSciPy.
//!
//! Implements ~158 public functions matching `scipy.signal` (module
//! docstring previously listed only 2 of those — per bead
//! frankenscipy-s9tt). High-level index by category:
//!
//! - **Windows**: `boxcar`, `hann`, `hamming`, `general_hamming`,
//!   `blackman`, `kaiser`, `tukey_window`, `blackmanharris`, `barthann`,
//!   `nuttall_window`, `bohman_window`.
//! - **Convolution / correlation**: `convolve`, `deconvolve`,
//!   `fftconvolve`, `correlate`, `correlate2d`.
//! - **Analytic signal**: `hilbert`, `hilbert_envelope`.
//! - **Spectral**: `lombscargle`, `czt`, `zoom_fft`, `chirp`,
//!   `gausspulse`, `sweep_poly`, `max_len_seq`, `matched_filter`.
//! - **Wavelets**: `ricker`, `morlet`, `cwt`.
//! - **Peaks / extrema**: `find_peaks`, `peak_prominences`,
//!   `peak_widths`, `argrelmax`, `argrelmin`, `argrelextrema`,
//!   `vectorstrength`, `order_filter`, `unwrap_phase`,
//!   `instantaneous_frequency`.
//! - **Savitzky-Golay**: `savgol_coeffs`, `savgol_filter`.
//! - **FIR/IIR design**: `firwin`, `firwin2`, `kaiserord`, `butter`,
//!   `cheby1`, `cheby2`, `ellip`, `bessel`, `iirdesign`, `iirfilter`.
//! - **Filter application**: `lfilter`, `filtfilt`, `sosfilt`,
//!   `sosfiltfilt`, `resample`, `resample_poly`, `decimate`.
//! - **Transfer / LTI**: `DLTI`, `tf2zpk`, `zpk2tf`, `bilinear`,
//!   `freqz`, `dlsim`.
//!
//! Full enumeration: `grep -E '^pub fn' crates/fsci-signal/src/lib.rs`.
//!
//! # Error taxonomy
//!
//! `SignalError` exposes structured failure classes for input length/shape,
//! non-finite values, unsupported modes, FFT failures, convolution-mode
//! failures, frequency-domain validation, numerical failures, and generic
//! parameter validation. Legacy string-form validation sites are normalized
//! through the crate-local `InvalidArgument` constructor so downstream callers
//! can match on the resulting enum variants instead of substring-parsing
//! `Display` text.

/// Error type for signal processing operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignalError {
    InvalidWindowLength(String),
    InvalidPolyOrder(String),
    InvalidInputLength { expected: usize, actual: usize },
    InvalidInputShape { detail: String },
    NonFiniteInput { detail: String },
    UnsupportedMode { detail: String },
    FftFailure(String),
    ConvolutionModeError(String),
    FrequencyOutOfBand { detail: String },
    NumericalFailure(String),
    InvalidParameter { detail: String },
    UnclassifiedArgument(String),
}

impl SignalError {
    #[allow(non_snake_case)]
    fn InvalidArgument(detail: String) -> Self {
        Self::classify_invalid_argument(detail)
    }

    fn classify_invalid_argument(detail: String) -> Self {
        let lower = detail.to_ascii_lowercase();
        if lower.contains("fft") || lower.contains("ifft") {
            return Self::FftFailure(detail);
        }
        if lower.contains("convolution") || lower.contains("valid mode") {
            return Self::ConvolutionModeError(detail);
        }
        if lower.contains("padtype")
            || lower.contains("window type")
            || lower.contains("unknown window")
            || lower.contains("unsupported")
            || lower.contains("mode")
        {
            return Self::UnsupportedMode { detail };
        }
        if lower.contains("shape")
            || lower.contains("same length")
            || lower.contains("must match")
            || lower.contains("match signal length")
            || lower.contains("match t length")
            || lower.contains("match system order")
            || lower.contains("length (")
        {
            return Self::InvalidInputShape { detail };
        }
        if lower.contains("non-empty")
            || lower.contains("not be empty")
            || lower.contains("cannot be empty")
            || lower.contains("empty")
        {
            return Self::InvalidInputLength {
                expected: 1,
                actual: 0,
            };
        }
        if lower.starts_with("dpss requires") {
            return Self::UnclassifiedArgument(detail);
        }
        if lower.contains("non-finite") || lower.contains("finite") {
            return Self::NonFiniteInput { detail };
        }
        if lower.contains("freq")
            || lower.contains("frequency")
            || lower.contains("cutoff")
            || lower.contains("wn")
            || lower.contains("w0")
            || lower.contains("ripple")
            || lower.contains("width")
            || lower.contains("nyquist")
        {
            return Self::FrequencyOutOfBand { detail };
        }
        if lower.contains("solve")
            || lower.contains("eigen")
            || lower.contains("singular")
            || lower.contains("degenerate")
            || lower.contains("overflow")
        {
            return Self::NumericalFailure(detail);
        }
        if lower.contains("sym==true") {
            return Self::UnclassifiedArgument(detail);
        }
        if lower.contains("must be")
            || lower.contains("out of range")
            || lower.contains("too large")
            || lower.contains("less than")
            || lower.contains("greater than")
        {
            return Self::InvalidParameter { detail };
        }
        Self::UnclassifiedArgument(detail)
    }

    pub fn is_argument_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidWindowLength(_)
                | Self::InvalidPolyOrder(_)
                | Self::InvalidInputLength { .. }
                | Self::InvalidInputShape { .. }
                | Self::NonFiniteInput { .. }
                | Self::UnsupportedMode { .. }
                | Self::FftFailure(_)
                | Self::ConvolutionModeError(_)
                | Self::FrequencyOutOfBand { .. }
                | Self::NumericalFailure(_)
                | Self::InvalidParameter { .. }
                | Self::UnclassifiedArgument(_)
        )
    }
}

impl std::fmt::Display for SignalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWindowLength(msg) => write!(f, "invalid window length: {msg}"),
            Self::InvalidPolyOrder(msg) => write!(f, "invalid polynomial order: {msg}"),
            Self::InvalidInputLength { expected, actual } => {
                write!(
                    f,
                    "invalid input length: expected at least {expected}, got {actual}"
                )
            }
            Self::InvalidInputShape { detail } => write!(f, "invalid input shape: {detail}"),
            Self::NonFiniteInput { detail } => write!(f, "non-finite input: {detail}"),
            Self::UnsupportedMode { detail } => write!(f, "unsupported mode: {detail}"),
            Self::FftFailure(msg) => write!(f, "FFT failure: {msg}"),
            Self::ConvolutionModeError(msg) => {
                write!(f, "convolution mode error: {msg}")
            }
            Self::FrequencyOutOfBand { detail } => {
                write!(f, "frequency out of band: {detail}")
            }
            Self::NumericalFailure(msg) => write!(f, "numerical failure: {msg}"),
            Self::InvalidParameter { detail } => write!(f, "invalid parameter: {detail}"),
            Self::UnclassifiedArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for SignalError {}

/// Compute Savitzky-Golay filter coefficients.
///
/// Matches `scipy.signal.savgol_coeffs(window_length, polyorder, deriv=0)`.
///
/// Returns a vector of `window_length` coefficients that, when convolved with
/// the signal, produce the smoothed (or differentiated) output.
///
/// # Arguments
/// * `window_length` — Must be odd and >= 1
/// * `polyorder` — Polynomial order, must be < window_length
/// * `deriv` — Derivative order (0 = smoothing, 1 = first derivative, etc.)
pub fn savgol_coeffs(
    window_length: usize,
    polyorder: usize,
    deriv: usize,
) -> Result<Vec<f64>, SignalError> {
    if window_length == 0 || window_length.is_multiple_of(2) {
        return Err(SignalError::InvalidWindowLength(
            "window_length must be a positive odd integer".to_string(),
        ));
    }
    if polyorder >= window_length {
        return Err(SignalError::InvalidPolyOrder(
            "polyorder must be less than window_length".to_string(),
        ));
    }
    if deriv > polyorder {
        return Err(SignalError::InvalidPolyOrder(
            "deriv must be <= polyorder".to_string(),
        ));
    }

    let half = (window_length / 2) as i64;
    let order = polyorder + 1;

    // Build the normal equations matrix A^T A.
    // (A^T A)_{r,c} = sum_{i=-half}^{half} i^{r+c}
    // We only need to compute sums of powers up to 2 * polyorder.
    let mut sums = vec![0.0; 2 * polyorder + 1];
    for i in 0..window_length {
        let xi = (i as i64 - half) as f64;
        let mut p = 1.0;
        for value in sums.iter_mut().take(2 * polyorder + 1) {
            *value += p;
            p *= xi;
        }
    }

    let mut ata = vec![vec![0.0; order]; order];
    for r in 0..order {
        ata[r][..order].copy_from_slice(&sums[r..(order + r)]);
    }

    // For each data point, the filter coefficient is the value of the
    // least-squares polynomial evaluated at x=0 (for deriv=0), or
    // the deriv-th derivative at x=0.
    // Filter weight for point i: w_i = P_i(x=0) where P_i is the polynomial
    // fit to a unit impulse at position i.
    // Equivalently: w = (A (A^T A)^{-1} e_d * d!) where e_d is the (deriv)-th unit vector.

    // Solve (A^T A) c = e_d where e_d[deriv] = 1
    let mut rhs = vec![0.0; order];
    rhs[deriv] = 1.0;

    let c = solve_symmetric_positive(&ata, &rhs)?;

    // Filter coefficients: w[i] = sum_j c[j] * x_i^j
    let deriv_factorial = factorial_small(deriv);
    let mut coeffs = Vec::with_capacity(window_length);
    for i in 0..window_length {
        let xi = (i as i64 - half) as f64;
        let mut val = 0.0;
        let mut power = 1.0;
        for &cj in &c {
            val += cj * power;
            power *= xi;
        }
        coeffs.push(val * deriv_factorial);
    }

    Ok(coeffs)
}

/// Apply a Savitzky-Golay filter to a signal.
///
/// Matches `scipy.signal.savgol_filter(x, window_length, polyorder)`.
///
/// Uses 'nearest' mode for boundary handling (repeats edge values).
pub fn savgol_filter(
    x: &[f64],
    window_length: usize,
    polyorder: usize,
) -> Result<Vec<f64>, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input signal must be non-empty".to_string(),
        ));
    }
    if window_length > x.len() {
        return Err(SignalError::InvalidWindowLength(
            "window_length must not exceed signal length".to_string(),
        ));
    }

    let coeffs = savgol_coeffs(window_length, polyorder, 0)?;
    let half = window_length / 2;
    let n = x.len();

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mut val = 0.0;
        for (j, &c) in coeffs.iter().enumerate() {
            let idx = i as i64 + j as i64 - half as i64;
            // Nearest-mode boundary: clamp to [0, n-1]
            let clamped = idx.clamp(0, n as i64 - 1) as usize;
            val += c * x[clamped];
        }
        result.push(val);
    }

    Ok(result)
}

/// Solve a small SPD system via Gaussian elimination with partial pivoting.
fn solve_symmetric_positive(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, SignalError> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivoting
        let mut pivot_row = col;
        for i in col + 1..n {
            if aug[i][col].abs() > aug[pivot_row][col].abs() {
                pivot_row = i;
            }
        }
        aug.swap(col, pivot_row);

        if aug[col][col].abs() < 1e-18 {
            return Err(SignalError::InvalidArgument(
                "linear system is singular or poorly conditioned".to_string(),
            ));
        }

        let pivot_val = aug[col][col];
        for i in col + 1..n {
            let factor = aug[i][col] / pivot_val;
            let (head, tail) = aug.split_at_mut(i);
            let pivot_row = &head[col];
            let target_row = &mut tail[0];
            for (target, pivot) in target_row
                .iter_mut()
                .zip(pivot_row.iter())
                .skip(col)
                .take(n + 1 - col)
            {
                *target -= factor * *pivot;
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in i + 1..n {
            sum -= aug[i][j] * x[j];
        }
        let diag = aug[i][i];
        if diag.abs() < 1e-18 {
            return Err(SignalError::InvalidArgument(
                "linear system is singular during back-substitution".to_string(),
            ));
        }
        x[i] = sum / diag;
    }
    Ok(x)
}

fn factorial_small(n: usize) -> f64 {
    match n {
        0 | 1 => 1.0,
        2 => 2.0,
        3 => 6.0,
        4 => 24.0,
        5 => 120.0,
        _ => (1..=n).map(|i| i as f64).product(),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Window Functions
// ══════════════════════════════════════════════════════════════════════

/// Generate a rectangular (boxcar) window of length `n`.
///
/// A boxcar window is simply a vector of ones. It provides no tapering
/// and is equivalent to not applying any window function.
///
/// Matches `scipy.signal.windows.boxcar(n, sym)`.
///
/// # Arguments
/// * `n` - Number of points in the window
/// * `sym` - If true (default), generates a symmetric window for filter design.
///   If false, generates a periodic window for spectral analysis.
pub fn boxcar(n: usize, sym: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if sym {
        vec![1.0; n]
    } else {
        // For periodic window, we would generate n+1 points and truncate
        // but since all values are 1.0, the result is the same
        vec![1.0; n]
    }
}

/// Generate a Hann (Hanning) window of length `n`.
///
/// Matches `scipy.signal.windows.hann(n)`.
pub fn hann(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = (n - 1) as f64;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / m).cos()))
        .collect()
}

/// Generate a Hamming window of length `n`.
///
/// Matches `scipy.signal.windows.hamming(n)`.
pub fn hamming(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = (n - 1) as f64;
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / m).cos())
        .collect()
}

/// Generate a generalized Hamming window of length `n`.
///
/// Matches `scipy.signal.windows.general_hamming(n, alpha)`.
pub fn general_hamming(n: usize, alpha: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = (n - 1) as f64;
    let beta = 1.0 - alpha;
    (0..n)
        .map(|i| alpha - beta * (2.0 * std::f64::consts::PI * i as f64 / m).cos())
        .collect()
}

/// Generate a Blackman window of length `n`.
///
/// Matches `scipy.signal.windows.blackman(n)`.
pub fn blackman(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * i as f64 / m;
            0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
        })
        .collect()
}

/// Generate a Kaiser window of length `n` with shape parameter `beta`.
///
/// Matches `scipy.signal.windows.kaiser(n, beta)`.
/// Uses a polynomial approximation for I0 (modified Bessel function of the first kind).
pub fn kaiser(n: usize, beta: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = (n - 1) as f64;
    let denom = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let alpha = 2.0 * i as f64 / m - 1.0;
            let arg = beta * (1.0 - alpha * alpha).max(0.0).sqrt();
            bessel_i0(arg) / denom
        })
        .collect()
}

/// Kaiser-Bessel derived window.
///
/// Matches `scipy.signal.windows.kaiser_bessel_derived(M, beta, sym=True)`.
/// The KBD construction is only defined for symmetric, even-length windows.
pub fn kaiser_bessel_derived(n: usize, beta: f64, sym: bool) -> Result<Vec<f64>, SignalError> {
    if !sym {
        return Err(SignalError::InvalidArgument(
            "Kaiser-Bessel Derived windows are only defined for symmetric shapes".to_string(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if !n.is_multiple_of(2) {
        return Err(SignalError::InvalidArgument(
            "Kaiser-Bessel Derived windows are only defined for even number of points".to_string(),
        ));
    }
    if !beta.is_finite() {
        return Err(SignalError::InvalidArgument(format!(
            "invalid kaiser_bessel_derived beta: {beta}"
        )));
    }

    let kaiser_window = kaiser(n / 2 + 1, beta);
    let total: f64 = kaiser_window.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "kaiser_bessel_derived normalization became singular".to_string(),
        ));
    }

    let mut cumulative = 0.0;
    let mut half = Vec::with_capacity(n / 2);
    for value in kaiser_window.iter().take(n / 2) {
        cumulative += *value;
        half.push((cumulative / total).sqrt());
    }

    let mut window = Vec::with_capacity(n);
    window.extend_from_slice(&half);
    window.extend(half.iter().rev().copied());
    Ok(window)
}

/// Modified Bessel function I0(x) via polynomial approximation.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75).powi(2);
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + t * (0.01328592
                    + t * (0.00225319
                        + t * (-0.00157565
                            + t * (0.00916281
                                + t * (-0.02057706
                                    + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

// ══════════════════════════════════════════════════════════════════════
// Convolution
// ══════════════════════════════════════════════════════════════════════

/// Convolution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvolveMode {
    /// Full convolution output (length = len(a) + len(b) - 1).
    #[default]
    Full,
    /// Output same length as the first input.
    Same,
    /// Only parts where signals fully overlap.
    Valid,
}

/// Direct (time-domain) convolution.
///
/// Matches `scipy.signal.convolve(a, b, mode)`.
pub fn convolve(a: &[f64], b: &[f64], mode: ConvolveMode) -> Result<Vec<f64>, SignalError> {
    if a.is_empty() || b.is_empty() {
        return Err(SignalError::InvalidArgument(
            "inputs must be non-empty".to_string(),
        ));
    }

    let na = a.len();
    let nb = b.len();

    // Automatic dispatch to FFT for large inputs (SciPy 'auto' method)
    if na.saturating_mul(nb) > 1000 {
        return fftconvolve(a, b, mode);
    }

    let full_len = na + nb - 1;
    // Compute full convolution
    let mut full = vec![0.0; full_len];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            full[i + j] += ai * bj;
        }
    }

    match mode {
        ConvolveMode::Full => Ok(full),
        ConvolveMode::Same => {
            let start = (nb - 1) / 2;
            Ok(full[start..start + na].to_vec())
        }
        ConvolveMode::Valid => {
            let valid_len = na.max(nb) - na.min(nb) + 1;
            let start = na.min(nb) - 1;
            Ok(full[start..start + valid_len].to_vec())
        }
    }
}

/// Deconvolve `divisor` out of `signal` using polynomial long division.
///
/// Returns `(quotient, remainder)` such that
/// `signal = convolve(divisor, quotient, Full) + remainder`.
///
/// Matches `scipy.signal.deconvolve(signal, divisor)`.
pub fn deconvolve(signal: &[f64], divisor: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    if signal.is_empty() {
        return Err(SignalError::InvalidArgument(
            "signal must be non-empty".to_string(),
        ));
    }
    if divisor.is_empty() {
        return Err(SignalError::InvalidArgument(
            "divisor must be non-empty".to_string(),
        ));
    }
    if divisor[0] == 0.0 {
        return Err(SignalError::InvalidArgument(
            "divisor[0] must not be zero".to_string(),
        ));
    }
    if divisor.len() > signal.len() {
        return Ok((Vec::new(), signal.to_vec()));
    }

    let quotient_len = signal.len() - divisor.len() + 1;
    let mut quotient = vec![0.0; quotient_len];
    let mut remainder = signal.to_vec();

    for i in 0..quotient_len {
        let coeff = remainder[i] / divisor[0];
        quotient[i] = coeff;
        for (j, &div) in divisor.iter().enumerate() {
            remainder[i + j] -= coeff * div;
        }
    }

    Ok((quotient, remainder))
}

/// FFT-based convolution (faster for large inputs).
///
/// Matches `scipy.signal.fftconvolve(a, b, mode)`.
pub fn fftconvolve(a: &[f64], b: &[f64], mode: ConvolveMode) -> Result<Vec<f64>, SignalError> {
    if a.is_empty() || b.is_empty() {
        return Err(SignalError::InvalidArgument(
            "inputs must be non-empty".to_string(),
        ));
    }

    let na = a.len();
    let nb = b.len();
    let full_len = na + nb - 1;

    // Pad to power of 2 for efficient FFT
    let fft_len = full_len.next_power_of_two();
    let opts = fsci_fft::FftOptions::default();

    // Zero-pad inputs to fft_len
    let mut a_padded: Vec<fsci_fft::Complex64> = a.iter().map(|&v| (v, 0.0)).collect();
    a_padded.resize(fft_len, (0.0, 0.0));

    let mut b_padded: Vec<fsci_fft::Complex64> = b.iter().map(|&v| (v, 0.0)).collect();
    b_padded.resize(fft_len, (0.0, 0.0));

    // FFT both
    let fa = fsci_fft::fft(&a_padded, &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;
    let fb = fsci_fft::fft(&b_padded, &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;

    // Pointwise multiply
    let fc: Vec<fsci_fft::Complex64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&(ar, ai), &(br, bi))| (ar * br - ai * bi, ar * bi + ai * br))
        .collect();

    // Inverse FFT
    let conv_full =
        fsci_fft::ifft(&fc, &opts).map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;

    // Extract real part, trimmed to full_len
    let full: Vec<f64> = conv_full.iter().take(full_len).map(|&(re, _)| re).collect();

    match mode {
        ConvolveMode::Full => Ok(full),
        ConvolveMode::Same => {
            let start = (nb - 1) / 2;
            Ok(full[start..start + na].to_vec())
        }
        ConvolveMode::Valid => {
            let valid_len = na.max(nb) - na.min(nb) + 1;
            let start = na.min(nb) - 1;
            Ok(full[start..start + valid_len].to_vec())
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Cross-correlation
// ══════════════════════════════════════════════════════════════════════

/// Cross-correlation of two 1D arrays.
///
/// Equivalent to `convolve(a, v[::-1], mode)` — reverse v and convolve.
/// Matches `scipy.signal.correlate(a, v, mode)`.
///
/// # Arguments
/// * `a` — First input array.
/// * `v` — Second input array (will be time-reversed internally).
/// * `mode` — Output mode: Full, Same, or Valid.
pub fn correlate(a: &[f64], v: &[f64], mode: ConvolveMode) -> Result<Vec<f64>, SignalError> {
    if a.is_empty() || v.is_empty() {
        return Err(SignalError::InvalidArgument(
            "inputs must be non-empty".to_string(),
        ));
    }
    // Correlate = convolve with reversed kernel
    let v_rev: Vec<f64> = v.iter().rev().copied().collect();
    convolve(a, &v_rev, mode)
}

/// 2D cross-correlation of two 2D arrays.
///
/// Matches `scipy.signal.correlate2d(in1, in2, mode)`.
///
/// # Arguments
/// * `a` — First 2D input (rows x cols, row-major flat array).
/// * `a_shape` — (rows, cols) of first input.
/// * `v` — Second 2D input (kernel, row-major flat array).
/// * `v_shape` — (rows, cols) of kernel.
/// * `mode` — Output mode: Full, Same, or Valid.
pub fn correlate2d(
    a: &[f64],
    a_shape: (usize, usize),
    v: &[f64],
    v_shape: (usize, usize),
    mode: ConvolveMode,
) -> Result<Vec<f64>, SignalError> {
    let (ar, ac) = a_shape;
    let (vr, vc) = v_shape;
    if a.len() != ar * ac || v.len() != vr * vc {
        return Err(SignalError::InvalidArgument(
            "array length must match shape".to_string(),
        ));
    }
    if ar == 0 || ac == 0 || vr == 0 || vc == 0 {
        return Err(SignalError::InvalidArgument(
            "inputs must be non-empty".to_string(),
        ));
    }

    // Reverse v in both dimensions for correlation
    let mut v_rev = vec![0.0; vr * vc];
    for i in 0..vr {
        for j in 0..vc {
            v_rev[i * vc + j] = v[(vr - 1 - i) * vc + (vc - 1 - j)];
        }
    }

    // 2D convolution (direct method)
    let (out_r, out_c, start_r, start_c) = match mode {
        ConvolveMode::Full => (ar + vr - 1, ac + vc - 1, 0, 0),
        ConvolveMode::Same => (ar, ac, vr / 2, vc / 2),
        ConvolveMode::Valid => {
            if ar < vr || ac < vc {
                return Err(SignalError::InvalidArgument(
                    "in valid mode, a must be at least as large as v".to_string(),
                ));
            }
            (ar - vr + 1, ac - vc + 1, vr - 1, vc - 1)
        }
    };

    let full_r = ar + vr - 1;
    let full_c = ac + vc - 1;

    // Compute full 2D convolution
    let mut full = vec![0.0; full_r * full_c];
    for i in 0..ar {
        for j in 0..ac {
            let aval = a[i * ac + j];
            for ki in 0..vr {
                for kj in 0..vc {
                    full[(i + ki) * full_c + (j + kj)] += aval * v_rev[ki * vc + kj];
                }
            }
        }
    }

    // Extract requested region
    let mut result = vec![0.0; out_r * out_c];
    for i in 0..out_r {
        for j in 0..out_c {
            result[i * out_c + j] = full[(i + start_r) * full_c + (j + start_c)];
        }
    }

    Ok(result)
}

// ══════════════════════════════════════════════════════════════════════
// Hilbert Transform / Analytic Signal
// ══════════════════════════════════════════════════════════════════════

/// Compute the analytic signal using the Hilbert transform.
///
/// Returns a complex-valued signal where:
/// - Real part = original signal
/// - Imaginary part = Hilbert transform of the signal
///
/// The instantaneous amplitude (envelope) is `|analytic|` and the
/// instantaneous phase is `angle(analytic)`.
///
/// Matches `scipy.signal.hilbert(x)`.
///
/// # Algorithm
/// 1. Compute FFT of the input
/// 2. Create a filter: h[0] = 1, h[1..N/2] = 2, h[N/2] = 1 (if N even), h[N/2+1..] = 0
/// 3. Multiply FFT by h
/// 4. Inverse FFT gives the analytic signal
pub fn hilbert(x: &[f64]) -> Result<Vec<(f64, f64)>, SignalError> {
    let n = x.len();
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "input must be non-empty".to_string(),
        ));
    }

    let fft_opts = fsci_fft::FftOptions::default();

    // Convert real input to complex for full FFT
    let complex_input: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();

    // Compute full complex FFT
    let spectrum = fsci_fft::fft(&complex_input, &fft_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;

    // Build the analytic signal filter h:
    // h[0] = 1, h[1..N/2] = 2, h[N/2] = 1 (even N), h[N/2+1..] = 0
    let mut h = vec![0.0; n];
    h[0] = 1.0;
    if n.is_multiple_of(2) {
        for item in h.iter_mut().take(n / 2).skip(1) {
            *item = 2.0;
        }
        h[n / 2] = 1.0;
    } else {
        for item in h.iter_mut().take((n - 1) / 2 + 1).skip(1) {
            *item = 2.0;
        }
    }

    // Multiply spectrum by h (zero negative frequencies, double positive)
    let filtered: Vec<(f64, f64)> = spectrum
        .iter()
        .zip(h.iter())
        .map(|(&(re, im), &hi)| (re * hi, im * hi))
        .collect();

    // Inverse FFT gives the analytic signal
    let analytic = fsci_fft::ifft(&filtered, &fft_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("IFFT failed: {e}")))?;

    Ok(analytic)
}

/// Compute the envelope (instantaneous amplitude) of a signal.
///
/// This is `|hilbert(x)|` — the magnitude of the analytic signal.
pub fn hilbert_envelope(x: &[f64]) -> Result<Vec<f64>, SignalError> {
    let analytic = hilbert(x)?;
    Ok(analytic
        .iter()
        .map(|&(re, im)| (re * re + im * im).sqrt())
        .collect())
}

// ══════════════════════════════════════════════════════════════════════
// Spectral Analysis (Lomb-Scargle) and Waveform Generation
// ══════════════════════════════════════════════════════════════════════

/// Lomb-Scargle periodogram for unevenly-spaced data.
///
/// Computes the Lomb-Scargle periodogram power at specified angular frequencies.
///
/// Matches `scipy.signal.lombscargle(x, y, freqs, normalize=...)`.
///
/// # Arguments
/// * `x` — Sample times (must be non-decreasing).
/// * `y` — Measurements at each sample time.
/// * `freqs` — Angular frequencies at which to evaluate the periodogram.
/// * `normalize` — Whether to divide by the signal energy as SciPy does.
pub fn lombscargle(
    x: &[f64],
    y: &[f64],
    freqs: &[f64],
    normalize: bool,
) -> Result<Vec<f64>, SignalError> {
    if x.len() != y.len() {
        return Err(SignalError::InvalidArgument(
            "x and y must have the same length".to_string(),
        ));
    }
    if x.len() < 2 {
        return Err(SignalError::InvalidArgument(
            "need at least 2 data points".to_string(),
        ));
    }

    let mut power = Vec::with_capacity(freqs.len());
    let signal_energy = y.iter().map(|value| value * value).sum::<f64>();

    for &omega in freqs {
        if !omega.is_finite() {
            power.push(f64::NAN);
            continue;
        }
        if omega == 0.0 {
            let y_sum = y.iter().sum::<f64>();
            let p = 0.5 * y_sum.powi(2) / x.len() as f64;
            power.push(if normalize {
                if signal_energy == 0.0 {
                    f64::NAN
                } else {
                    p * (2.0 / signal_energy)
                }
            } else {
                p
            });
            continue;
        }
        // Compute tau: phase offset for orthogonality
        let mut s2 = 0.0;
        let mut c2 = 0.0;
        for &xi in x {
            s2 += (2.0 * omega * xi).sin();
            c2 += (2.0 * omega * xi).cos();
        }
        let tau = (s2 / c2).atan() / (2.0 * omega);

        // Compute power at this frequency
        let mut y_cos_sum = 0.0;
        let mut cos2_sum = 0.0;
        let mut y_sin_sum = 0.0;
        let mut sin2_sum = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let phase = omega * (xi - tau);
            let c = phase.cos();
            let s = phase.sin();
            y_cos_sum += yi * c;
            cos2_sum += c * c;
            y_sin_sum += yi * s;
            sin2_sum += s * s;
        }

        let p = if cos2_sum > 0.0 && sin2_sum > 0.0 {
            0.5 * (y_cos_sum.powi(2) / cos2_sum + y_sin_sum.powi(2) / sin2_sum)
        } else {
            0.0
        };
        power.push(if normalize {
            if signal_energy == 0.0 {
                f64::NAN
            } else {
                p * (2.0 / signal_energy)
            }
        } else {
            p
        });
    }

    Ok(power)
}

/// Gaussian-modulated sinusoidal pulse.
///
/// Returns the samples of a Gaussian-modulated sinusoid:
///   y(t) = exp(-a t²) cos(2π fc t)
/// where a = (π fc bw)² / (-2 ln(bwr)) and bwr is the reference level (-6dB default).
///
/// Matches `scipy.signal.gausspulse(t, fc, bw)`.
pub fn gausspulse(t: &[f64], fc: f64, bw: f64) -> Vec<f64> {
    if fc < 0.0 || bw <= 0.0 {
        return vec![0.0; t.len()];
    }
    let bwr = -6.0; // reference level in dB
    let ref_level = 10.0_f64.powf(bwr / 20.0);
    let a = -(std::f64::consts::PI * fc * bw).powi(2) / (2.0 * ref_level.ln());

    t.iter()
        .map(|&ti| {
            let envelope = (-a * ti * ti).exp();
            let carrier = (2.0 * std::f64::consts::PI * fc * ti).cos();
            envelope * carrier
        })
        .collect()
}

/// Frequency-swept cosine with polynomial instantaneous frequency.
///
/// Generates cos(2π ∫₀ᵗ f(τ) dτ) where f(τ) is a polynomial.
///
/// Matches `scipy.signal.sweep_poly(t, poly)`.
///
/// # Arguments
/// * `t` — Time points.
/// * `poly` — Polynomial coefficients [p₀, p₁, ..., pₙ] for f(t) = p₀tⁿ + ... + pₙ.
pub fn sweep_poly(t: &[f64], poly: &[f64]) -> Vec<f64> {
    t.iter()
        .map(|&ti| {
            // Integrate polynomial: ∫₀ᵗ (p₀τⁿ + ... + pₙ) dτ
            // = p₀t^{n+1}/(n+1) + ... + pₙt
            let n = poly.len();
            let mut phase_integral = 0.0;
            for (k, &coeff) in poly.iter().enumerate() {
                let power = n - 1 - k; // degree of this term
                phase_integral += coeff * ti.powi(power as i32 + 1) / (power as f64 + 1.0);
            }
            (2.0 * std::f64::consts::PI * phase_integral).cos()
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════
// Chirp Z-Transform
// ══════════════════════════════════════════════════════════════════════

/// Chirp Z-Transform: evaluate the Z-transform at M points along a spiral
/// in the complex plane starting at `a`, stepping by `w`.
///
/// Given x[n] for n=0..N-1, computes:
///   X[k] = Σ_{n=0}^{N-1} x[n] · a^{-n} · w^{nk},  k = 0..M-1
///
/// # Arguments
/// * `x` — Input signal (real-valued).
/// * `m` — Number of output points.
/// * `w` — Step factor as (magnitude, angle) in polar form.
///   Default: `(1.0, -2π/M)` (unit circle, same as DFT).
/// * `a` — Starting point as (magnitude, angle) in polar form.
///   Default: `(1.0, 0.0)` (z = 1).
///
/// Matches `scipy.signal.czt`.
pub fn czt(
    x: &[f64],
    m: usize,
    w: Option<(f64, f64)>,
    a: Option<(f64, f64)>,
) -> Result<Vec<(f64, f64)>, SignalError> {
    let n = x.len();
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "Input signal must be non-empty".to_string(),
        ));
    }
    if m == 0 {
        return Ok(vec![]);
    }

    let two_pi = 2.0 * std::f64::consts::PI;

    // Default w: unit circle step = exp(-j2π/M)
    let (w_mag, w_ang) = w.unwrap_or((1.0, -two_pi / m as f64));
    // Default a: z = 1
    let (a_mag, a_ang) = a.unwrap_or((1.0, 0.0));

    // Bluestein's algorithm for CZT via convolution:
    // 1) Form yn[n] = x[n] * a^{-n} * w^{n²/2}
    // 2) Form h[n] = w^{-n²/2}
    // 3) Convolve yn with h, then multiply by w^{k²/2}

    let l = (n + m - 1).next_power_of_two(); // FFT length

    // Precompute w^{k²/2} chirp factors
    // w^{k²/2} = (w_mag)^{k²/2} * exp(j * w_ang * k²/2)
    let chirp = |k: i64| -> (f64, f64) {
        let half_k2 = (k * k) as f64 / 2.0;
        let mag = w_mag.powf(half_k2);
        let ang = w_ang * half_k2;
        (mag * ang.cos(), mag * ang.sin())
    };

    // a^{-n} = (a_mag)^{-n} * exp(-j * a_ang * n)
    let a_neg = |nn: usize| -> (f64, f64) {
        let nf = nn as f64;
        let mag = a_mag.powf(-nf);
        let ang = -a_ang * nf;
        (mag * ang.cos(), mag * ang.sin())
    };

    // Complex multiply helper
    let cmul = |a: (f64, f64), b: (f64, f64)| -> (f64, f64) {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    };

    // Build yn: length L, zero-padded
    let mut yn = vec![(0.0f64, 0.0f64); l];
    for i in 0..n {
        let an = a_neg(i);
        let wn = chirp(i as i64);
        let xi = (x[i], 0.0);
        yn[i] = cmul(cmul(xi, an), wn);
    }

    // Build h: w^{-k²/2} for k = -(N-1)..M-1, stored in wrap-around order
    let mut h = vec![(0.0f64, 0.0f64); l];
    // h[k] for k = 0..M-1
    for (k, hk) in h.iter_mut().enumerate().take(m) {
        let c = chirp(k as i64);
        // w^{-k²/2} = conjugate of w^{k²/2} when w is on unit circle,
        // but in general we need to invert
        let mag_sq = c.0 * c.0 + c.1 * c.1;
        if mag_sq > 0.0 {
            *hk = (c.0 / mag_sq, -c.1 / mag_sq);
        }
    }
    // h[L-k] for k = 1..N-1 (negative indices wrapped)
    for k in 1..n {
        let c = chirp(k as i64);
        let mag_sq = c.0 * c.0 + c.1 * c.1;
        if mag_sq > 0.0 {
            h[l - k] = (c.0 / mag_sq, -c.1 / mag_sq);
        }
    }

    // FFT of yn and h, multiply, IFFT
    let fft_opts = fsci_fft::FftOptions::default();
    let yn_fft = fsci_fft::fft(&yn, &fft_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;
    let h_fft = fsci_fft::fft(&h, &fft_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;

    let product_fft: Vec<(f64, f64)> = yn_fft
        .iter()
        .zip(h_fft.iter())
        .map(|(&a, &b)| cmul(a, b))
        .collect();

    let product = fsci_fft::ifft(&product_fft, &fft_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("IFFT failed: {e}")))?;

    // Extract result: X[k] = product[k] * w^{k²/2}
    let mut result = Vec::with_capacity(m);
    for (k, value) in product.iter().enumerate().take(m) {
        let wk = chirp(k as i64);
        result.push(cmul(*value, wk));
    }

    Ok(result)
}

/// Zoom FFT: compute the DFT over a frequency sub-range [f1, f2] with M points.
///
/// This uses the Chirp Z-Transform to evaluate the Z-transform at M equally
/// spaced points on the unit circle arc from f1 to f2 (normalized frequencies,
/// where 1.0 = sampling rate).
///
/// # Arguments
/// * `x` — Input signal.
/// * `f_range` — (f1, f2) normalized frequency range.
/// * `m` — Number of output points.
///
/// Matches `scipy.signal.zoom_fft`.
pub fn zoom_fft(x: &[f64], f_range: (f64, f64), m: usize) -> Result<Vec<(f64, f64)>, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "Input signal must be non-empty".to_string(),
        ));
    }
    if m == 0 {
        return Ok(vec![]);
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let (f1, f2) = f_range;

    // Starting point on unit circle: a = exp(j * 2π * f1)
    let a = (1.0, two_pi * f1);

    // We want z_k = exp(j*2π*f_k) where f_k = f1 + k*(f2-f1)/M.
    // In CZT: z_k = a * w^{-k}, so w = exp(-j*2π*(f2-f1)/M).

    let w = (1.0, -two_pi * (f2 - f1) / m as f64);

    czt(x, m, Some(w), Some(a))
}

// ══════════════════════════════════════════════════════════════════════
// Sequences and Matched Filtering
// ══════════════════════════════════════════════════════════════════════

/// Generate a maximum-length sequence (m-sequence) using a linear feedback shift register.
///
/// Returns a binary sequence of length 2^nbits - 1 with values in {-1, 1}.
/// M-sequences have ideal autocorrelation properties.
///
/// Matches `scipy.signal.max_len_seq(nbits)`.
pub fn max_len_seq(nbits: usize) -> Result<Vec<f64>, SignalError> {
    if !(2..=31).contains(&nbits) {
        return Err(SignalError::InvalidArgument(
            "nbits must be in [2, 31]".to_string(),
        ));
    }

    let seq_len = (1usize << nbits) - 1;

    // Primitive polynomial taps for each nbits (from tables)
    let taps: &[usize] = match nbits {
        2 => &[2, 1],
        3 => &[3, 1],
        4 => &[4, 1],
        5 => &[5, 2],
        6 => &[6, 1],
        7 => &[7, 1],
        8 => &[8, 4, 3, 2],
        9 => &[9, 4],
        10 => &[10, 3],
        11 => &[11, 2],
        12 => &[12, 6, 4, 1],
        13 => &[13, 4, 3, 1],
        14 => &[14, 5, 3, 1],
        15 => &[15, 1],
        16 => &[16, 5, 3, 2],
        _ => &[nbits, 1], // fallback (may not be primitive for all n)
    };

    let mut state = 1u32; // Initial state (nonzero)
    let mut seq = Vec::with_capacity(seq_len);

    for _ in 0..seq_len {
        // Output is the last bit
        let output = state & 1;
        seq.push(if output == 1 { 1.0 } else { -1.0 });

        // Feedback: XOR of tapped bits
        let mut feedback = 0u32;
        for &tap in taps {
            feedback ^= (state >> (tap - 1)) & 1;
        }

        // Shift register
        state = (state >> 1) | (feedback << (nbits - 1));
    }

    Ok(seq)
}

/// Matched filter: correlate signal with template, normalized by template energy.
///
/// Returns the normalized cross-correlation, which peaks at the location
/// where the template best matches the signal.
pub fn matched_filter(template: &[f64], signal: &[f64]) -> Result<Vec<f64>, SignalError> {
    if template.is_empty() || signal.is_empty() {
        return Err(SignalError::InvalidArgument(
            "template and signal must be non-empty".to_string(),
        ));
    }

    // Template energy
    let energy: f64 = template.iter().map(|&t| t * t).sum();
    if energy == 0.0 {
        return Err(SignalError::InvalidArgument(
            "template must have nonzero energy".to_string(),
        ));
    }

    // Cross-correlate and normalize
    let corr = correlate(signal, template, ConvolveMode::Full)?;
    Ok(corr.iter().map(|&c| c / energy).collect())
}

// ══════════════════════════════════════════════════════════════════════
// Wavelets
// ══════════════════════════════════════════════════════════════════════

/// Ricker wavelet (Mexican hat wavelet).
///
/// The second derivative of a Gaussian: ψ(t) = (2/(√3σ π^{1/4})) (1 - (t/σ)²) exp(-t²/(2σ²))
///
/// Matches `scipy.signal.ricker(points, a)`.
///
/// # Arguments
/// * `points` — Number of points in the output vector.
/// * `a` — Width parameter (standard deviation of the Gaussian).
pub fn ricker(points: usize, a: f64) -> Vec<f64> {
    if points == 0 || a <= 0.0 || !a.is_finite() {
        return vec![];
    }
    let mut output = Vec::with_capacity(points);
    let center = (points as f64 - 1.0) / 2.0;
    let a2 = a * a;
    let norm = 2.0 / (3.0_f64.sqrt() * std::f64::consts::PI.powf(0.25));

    for i in 0..points {
        let t = i as f64 - center;
        let t2 = t * t;
        let gauss = (-t2 / (2.0 * a2)).exp();
        output.push(norm / a.sqrt() * (1.0 - t2 / a2) * gauss);
    }
    output
}

/// Complex Morlet wavelet.
///
/// ψ(t) = exp(2πi w₀ t) exp(-t²/2) (with optional correction for non-zero mean)
///
/// Matches `scipy.signal.morlet(M, w, s, complete)`.
///
/// # Arguments
/// * `m` — Number of points in the output.
/// * `w` — Omega0 (center frequency), default 5.0.
/// * `s` — Scaling factor, default 1.0.
/// * `complete` — If true, apply correction term for non-zero mean.
pub fn morlet(m: usize, w: f64, s: f64, complete: bool) -> Vec<(f64, f64)> {
    if m == 0 || s <= 0.0 || !s.is_finite() || !w.is_finite() {
        return vec![];
    }
    let mut output = Vec::with_capacity(m);
    let center = (m as f64 - 1.0) / 2.0;

    for i in 0..m {
        let t = (i as f64 - center) / s;
        let gauss = (-t * t / 2.0).exp();
        let phase = 2.0 * std::f64::consts::PI * w * t;
        let mut re = gauss * phase.cos();
        let im = gauss * phase.sin();

        if complete {
            // Correction: subtract the DC component to ensure zero mean
            let correction =
                (-2.0 * std::f64::consts::PI * std::f64::consts::PI * w * w / 2.0).exp();
            re -= gauss * correction;
        }

        output.push((re, im));
    }
    output
}

/// Continuous Wavelet Transform.
///
/// Computes CWT using the specified wavelet function at different scales.
///
/// Matches `scipy.signal.cwt(data, wavelet, widths)`.
///
/// # Arguments
/// * `data` — Input signal.
/// * `wavelet_fn` — Function that generates a wavelet for a given width: `fn(points, width) -> Vec<f64>`.
/// * `widths` — Array of wavelet widths (scales) to use.
///
/// Returns a 2D array (widths.len() × data.len()), where each row is the CWT
/// at the corresponding scale.
pub fn cwt<F>(data: &[f64], wavelet_fn: F, widths: &[f64]) -> Result<Vec<Vec<f64>>, SignalError>
where
    F: Fn(usize, f64) -> Vec<f64>,
{
    if data.is_empty() {
        return Err(SignalError::InvalidArgument(
            "data must be non-empty".to_string(),
        ));
    }
    if widths.is_empty() {
        return Err(SignalError::InvalidArgument(
            "widths must be non-empty".to_string(),
        ));
    }

    let _n = data.len();
    let mut result = Vec::with_capacity(widths.len());

    for &width in widths {
        if width <= 0.0 || !width.is_finite() {
            return Err(SignalError::InvalidArgument(
                "all widths must be positive and finite".to_string(),
            ));
        }
        // Generate wavelet at this scale
        let wavelet_len = (10.0 * width).ceil() as usize;
        let wavelet_len = wavelet_len.max(1);
        let wavelet = wavelet_fn(wavelet_len, width);

        // Convolve data with wavelet (mode = same)
        let conv = convolve(data, &wavelet, ConvolveMode::Same)?;
        result.push(conv);
    }

    Ok(result)
}

// ── Additional window functions ──────────────────────────────────────

/// Tukey (tapered cosine) window.
///
/// Matches `scipy.signal.windows.tukey(M, alpha)`.
///
/// alpha=0 → rectangular, alpha=1 → Hann.
pub fn tukey_window(m: usize, alpha: f64) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let mut w = vec![1.0; m];
    let n = m - 1;

    if alpha > 0.0 {
        let taper_len = (alpha * n as f64 / 2.0) as usize;
        for i in 0..=taper_len.min(n) {
            let val =
                0.5 * (1.0 - (std::f64::consts::PI * i as f64 / (alpha * n as f64 / 2.0)).cos());
            w[i] = val;
            w[n - i] = val;
        }
    }
    w
}

/// Blackman-Harris window.
///
/// Matches `scipy.signal.windows.blackmanharris(M)`.
pub fn blackmanharris(m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let n = m - 1;
    (0..m)
        .map(|i| {
            let x = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            0.358_75 - 0.488_29 * x.cos() + 0.141_28 * (2.0 * x).cos() - 0.011_68 * (3.0 * x).cos()
        })
        .collect()
}

/// Bartlett-Hann window.
///
/// Matches `scipy.signal.windows.barthann(M)`.
pub fn barthann(m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let denom = (m - 1) as f64;
    (0..m)
        .map(|i| {
            let fac = (i as f64 / denom - 0.5).abs();
            0.62 - 0.48 * fac + 0.38 * (2.0 * std::f64::consts::PI * fac).cos()
        })
        .collect()
}

/// Nuttall window (minimum 4-term Blackman-Harris).
///
/// Matches `scipy.signal.windows.nuttall(M)`.
///
/// br-z3ni: scipy's "nuttall" is the "minimum 4-term Blackman-Harris"
/// variant (Nuttall 1981, table II, "Min 4-term"). fsci previously
/// used the standard continuous-flattop Nuttall coefficients
/// [0.355768, 0.487396, 0.144232, 0.012604] which are a different
/// point on the same coefficient table. The scipy-compatible
/// coefficients [0.3635819, 0.4891775, 0.1365995, 0.0106411] are now
/// used so parity fixtures pass.
pub fn nuttall_window(m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let n = m - 1;
    (0..m)
        .map(|i| {
            let x = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            0.363_581_9 - 0.489_177_5 * x.cos() + 0.136_599_5 * (2.0 * x).cos()
                - 0.010_641_1 * (3.0 * x).cos()
        })
        .collect()
}

/// Bohman window.
///
/// Matches `scipy.signal.windows.bohman(M)`.
pub fn bohman_window(m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let n = m - 1;
    (0..m)
        .map(|i| {
            let x = (2.0 * i as f64 / n as f64 - 1.0).abs();
            if x >= 1.0 {
                0.0
            } else {
                (1.0 - x) * (std::f64::consts::PI * x).cos()
                    + (1.0 / std::f64::consts::PI) * (std::f64::consts::PI * x).sin()
            }
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════
// Peak Detection
// ══════════════════════════════════════════════════════════════════════

/// Options for peak detection.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FindPeaksOptions {
    /// Minimum height of peaks.
    pub height: Option<f64>,
    /// Minimum horizontal distance between peaks (in samples).
    pub distance: Option<usize>,
    /// Minimum prominence of peaks.
    pub prominence: Option<f64>,
    /// Minimum width of peaks (in samples).
    pub width: Option<f64>,
}

/// Result of peak detection.
#[derive(Debug, Clone, PartialEq)]
pub struct FindPeaksResult {
    /// Indices of detected peaks.
    pub peaks: Vec<usize>,
    /// Heights of detected peaks (if height filter was used).
    pub peak_heights: Vec<f64>,
    /// Prominences of detected peaks (if prominence filter was used).
    pub prominences: Vec<f64>,
}

/// Find peaks in a 1-D signal.
///
/// Matches `scipy.signal.find_peaks(x, height, distance, prominence, width)`.
///
/// A sample is a peak if it is strictly greater than its immediate neighbors.
pub fn find_peaks(x: &[f64], options: FindPeaksOptions) -> FindPeaksResult {
    if x.len() < 3 {
        return FindPeaksResult {
            peaks: Vec::new(),
            peak_heights: Vec::new(),
            prominences: Vec::new(),
        };
    }

    // Step 1: find all local maxima (including plateaus)
    let mut peaks: Vec<usize> = Vec::new();
    let mut i = 1;
    while i < x.len() - 1 {
        if x[i] > x[i - 1] {
            let mut j = i + 1;
            while j < x.len() && x[j] == x[i] {
                j += 1;
            }
            // If the plateau is followed by a smaller value or the end of the signal
            if j < x.len() && x[i] > x[j] {
                // Peak is the middle of the plateau
                peaks.push((i + j - 1) / 2);
            }
            i = j;
        } else {
            i += 1;
        }
    }

    // Step 2: filter by height
    if let Some(min_height) = options.height {
        peaks.retain(|&i| x[i] >= min_height);
    }

    // Step 3: compute prominences (needed for prominence and width filters)
    let prominences: Vec<f64> = peaks.iter().map(|&pk| compute_prominence(x, pk)).collect();

    // Step 4: filter by prominence
    if let Some(min_prom) = options.prominence {
        let mut keep = Vec::new();
        for (idx, &pk) in peaks.iter().enumerate() {
            if prominences[idx] >= min_prom {
                keep.push((pk, prominences[idx]));
            }
        }
        peaks = keep.iter().map(|&(pk, _)| pk).collect();
        let proms: Vec<f64> = keep.iter().map(|&(_, p)| p).collect();
        // Rebuild prominences for kept peaks
        let peak_heights: Vec<f64> = peaks.iter().map(|&i| x[i]).collect();

        // Step 5: filter by distance (keep highest peak within each distance window)
        if let Some(min_dist) = options.distance {
            let filtered = filter_by_distance(&peaks, &peak_heights, min_dist);
            let final_proms: Vec<f64> = filtered
                .iter()
                .map(|&pk| proms[peaks.binary_search(&pk).unwrap_or(0)])
                .collect();
            let final_heights: Vec<f64> = filtered.iter().map(|&i| x[i]).collect();
            return FindPeaksResult {
                peaks: filtered,
                peak_heights: final_heights,
                prominences: final_proms,
            };
        }

        return FindPeaksResult {
            peaks,
            peak_heights,
            prominences: proms,
        };
    }

    let peak_heights: Vec<f64> = peaks.iter().map(|&i| x[i]).collect();

    // Step 5: filter by distance
    if let Some(min_dist) = options.distance {
        let filtered = filter_by_distance(&peaks, &peak_heights, min_dist);
        let final_heights: Vec<f64> = filtered.iter().map(|&i| x[i]).collect();
        let final_proms: Vec<f64> = filtered
            .iter()
            .map(|&pk| compute_prominence(x, pk))
            .collect();
        return FindPeaksResult {
            peaks: filtered,
            peak_heights: final_heights,
            prominences: final_proms,
        };
    }

    FindPeaksResult {
        peaks,
        peak_heights,
        prominences,
    }
}

/// Compute the prominence of a peak.
/// Prominence = peak height - max(left base, right base).
fn compute_prominence(x: &[f64], peak_idx: usize) -> f64 {
    let peak_val = x[peak_idx];

    // Search left for the lowest point before reaching a higher peak or the edge
    let mut left_min = peak_val;
    for &xi in x[..peak_idx].iter().rev() {
        left_min = left_min.min(xi);
        if xi > peak_val {
            break;
        }
    }

    // Search right
    let mut right_min = peak_val;
    for &xi in &x[peak_idx + 1..] {
        right_min = right_min.min(xi);
        if xi > peak_val {
            break;
        }
    }

    peak_val - left_min.max(right_min)
}

/// Filter peaks by minimum distance, keeping the highest peak in each window.
fn filter_by_distance(peaks: &[usize], heights: &[f64], min_dist: usize) -> Vec<usize> {
    if peaks.is_empty() {
        return Vec::new();
    }

    // Sort peaks by height (descending) for greedy selection.
    // Store original index to quickly access 'excluded' status.
    let mut indexed: Vec<(usize, usize, f64)> = peaks
        .iter()
        .zip(heights.iter())
        .enumerate()
        .map(|(i, (&p, &h))| (i, p, h))
        .collect();
    indexed.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| b.1.cmp(&a.1)));

    let mut selected = Vec::new();
    let mut excluded = vec![false; peaks.len()];

    for &(orig_idx, pk, _) in &indexed {
        if excluded[orig_idx] {
            continue;
        }
        selected.push(pk);

        // Exclude all peaks within min_dist of pk.
        // Since 'peaks' is sorted by position, use binary search to find the range.
        let start_val = pk.saturating_sub(min_dist.saturating_sub(1));
        let end_val = pk.saturating_add(min_dist);

        let left = peaks.binary_search(&start_val).unwrap_or_else(|x| x);
        let right = peaks.binary_search(&end_val).unwrap_or_else(|x| x);

        for slot in excluded.iter_mut().take(right).skip(left) {
            *slot = true;
        }
    }

    selected.sort_unstable();
    selected
}

/// Compute peak prominences for given peak indices.
///
/// Matches `scipy.signal.peak_prominences`.
///
/// Returns (prominences, left_bases, right_bases).
pub fn peak_prominences(x: &[f64], peaks: &[usize]) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let mut prominences = Vec::with_capacity(peaks.len());
    let mut left_bases = Vec::with_capacity(peaks.len());
    let mut right_bases = Vec::with_capacity(peaks.len());

    for &pk in peaks {
        if pk >= x.len() {
            prominences.push(0.0);
            left_bases.push(pk);
            right_bases.push(pk);
            continue;
        }

        let peak_val = x[pk];

        // Search left for minimum
        let mut left_min = peak_val;
        let mut left_base = pk;
        for i in (0..pk).rev() {
            if x[i] < left_min {
                left_min = x[i];
                left_base = i;
            }
            if x[i] > peak_val {
                break;
            }
        }

        // Search right for minimum
        let mut right_min = peak_val;
        let mut right_base = pk;
        for (i, &value) in x.iter().enumerate().skip(pk + 1) {
            if value < right_min {
                right_min = value;
                right_base = i;
            }
            if value > peak_val {
                break;
            }
        }

        prominences.push(peak_val - left_min.max(right_min));
        left_bases.push(left_base);
        right_bases.push(right_base);
    }

    (prominences, left_bases, right_bases)
}

/// Compute peak widths at a given relative height.
///
/// Matches `scipy.signal.peak_widths`.
///
/// `rel_height` is the fraction of prominence at which to measure width (0.5 = half-prominence).
///
/// Returns (widths, width_heights, left_ips, right_ips).
pub fn peak_widths(
    x: &[f64],
    peaks: &[usize],
    rel_height: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (prominences, left_bases, right_bases) = peak_prominences(x, peaks);

    let mut widths = Vec::with_capacity(peaks.len());
    let mut width_heights = Vec::with_capacity(peaks.len());
    let mut left_ips = Vec::with_capacity(peaks.len());
    let mut right_ips = Vec::with_capacity(peaks.len());

    for (idx, &pk) in peaks.iter().enumerate() {
        if pk >= x.len() {
            let ip = pk as f64;
            widths.push(0.0);
            width_heights.push(0.0);
            left_ips.push(ip);
            right_ips.push(ip);
            continue;
        }

        let height = x[pk] - prominences[idx] * rel_height;
        width_heights.push(height);

        // Find left intersection point (interpolated)
        let mut left_ip = left_bases[idx] as f64;
        for i in (left_bases[idx]..pk).rev() {
            if x[i] <= height {
                // Linear interpolation between i and i+1
                if i + 1 < x.len() && (x[i + 1] - x[i]).abs() > 1e-15 {
                    left_ip = i as f64 + (height - x[i]) / (x[i + 1] - x[i]);
                } else {
                    left_ip = i as f64;
                }
                break;
            }
        }

        // Find right intersection point
        let mut right_ip = right_bases[idx] as f64;
        for i in pk + 1..=right_bases[idx] {
            if x[i] <= height {
                if i > 0 && (x[i] - x[i - 1]).abs() > 1e-15 {
                    right_ip = i as f64 - (x[i] - height) / (x[i] - x[i - 1]);
                } else {
                    right_ip = i as f64;
                }
                break;
            }
        }

        left_ips.push(left_ip);
        right_ips.push(right_ip);
        widths.push(right_ip - left_ip);
    }

    (widths, width_heights, left_ips, right_ips)
}

/// Compute the analytic signal's instantaneous frequency.
///
/// Matches `scipy.signal.instantaneous_frequency` (via Hilbert transform).
pub fn instantaneous_frequency(x: &[f64], fs: f64) -> Result<Vec<f64>, SignalError> {
    if fs <= 0.0 || !fs.is_finite() {
        return Err(SignalError::InvalidArgument(
            "sampling frequency fs must be positive and finite".to_string(),
        ));
    }
    let analytic = hilbert(x)?;

    // Instantaneous phase
    let phase: Vec<f64> = analytic.iter().map(|&(re, im)| im.atan2(re)).collect();

    // Unwrap phase
    let unwrapped = unwrap_phase(&phase);

    // Instantaneous frequency = d(phase)/dt / (2π)
    let mut freq = Vec::with_capacity(x.len());
    freq.push(0.0);
    for i in 1..unwrapped.len() {
        freq.push((unwrapped[i] - unwrapped[i - 1]) * fs / (2.0 * std::f64::consts::PI));
    }

    Ok(freq)
}

/// Find indices of relative maxima.
///
/// Matches `scipy.signal.argrelmax`.
pub fn argrelmax(x: &[f64], order: usize) -> Vec<usize> {
    let order = order.max(1);
    let mut maxima = Vec::new();
    for i in order..x.len().saturating_sub(order) {
        let mut is_max = true;
        for j in 1..=order {
            if x[i] <= x[i - j] || x[i] <= x[i + j] {
                is_max = false;
                break;
            }
        }
        if is_max {
            maxima.push(i);
        }
    }
    maxima
}

/// Find indices of relative minima.
///
/// Matches `scipy.signal.argrelmin`.
pub fn argrelmin(x: &[f64], order: usize) -> Vec<usize> {
    let order = order.max(1);
    let mut minima = Vec::new();
    for i in order..x.len().saturating_sub(order) {
        let mut is_min = true;
        for j in 1..=order {
            if x[i] >= x[i - j] || x[i] >= x[i + j] {
                is_min = false;
                break;
            }
        }
        if is_min {
            minima.push(i);
        }
    }
    minima
}

/// Find indices of relative extrema (both maxima and minima).
///
/// Matches `scipy.signal.argrelextrema`.
pub fn argrelextrema(x: &[f64], order: usize, greater: bool) -> Vec<usize> {
    if greater {
        argrelmax(x, order)
    } else {
        argrelmin(x, order)
    }
}

/// Compute the vector strength of a set of events at a given period.
///
/// Returns (strength, pvalue).
/// Matches `scipy.signal.vectorstrength`.
pub fn vectorstrength(events: &[f64], period: f64) -> (f64, f64) {
    if events.is_empty() || period <= 0.0 || !period.is_finite() {
        return (0.0, 1.0);
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let n = events.len() as f64;

    // Phase of each event in the cycle
    let sin_sum: f64 = events.iter().map(|&t| (two_pi * t / period).sin()).sum();
    let cos_sum: f64 = events.iter().map(|&t| (two_pi * t / period).cos()).sum();

    let r = (sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / n;

    // Rayleigh test p-value
    let pvalue = (-n * r * r).exp();

    (r, pvalue.clamp(0.0, 1.0))
}

/// 1D order filter: select the k-th smallest value in each window.
///
/// Matches `scipy.signal.order_filter`.
pub fn order_filter(x: &[f64], window_size: usize, rank: usize) -> Vec<f64> {
    if x.is_empty() || window_size == 0 {
        return vec![];
    }
    let half = window_size / 2;
    let mut result = Vec::with_capacity(x.len());

    for i in 0..x.len() {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(x.len());
        let mut window: Vec<f64> = x[start..end].to_vec();
        window.sort_by(|a, b| a.total_cmp(b));
        let idx = rank.min(window.len() - 1);
        result.push(window[idx]);
    }

    result
}

/// Unwrap phase angles to avoid discontinuities.
///
/// Matches `numpy.unwrap`.
pub fn unwrap_phase(phase: &[f64]) -> Vec<f64> {
    if phase.is_empty() {
        return vec![];
    }
    let mut unwrapped = vec![phase[0]];
    for i in 1..phase.len() {
        let mut diff = phase[i] - phase[i - 1];
        if diff.is_finite() {
            while diff > std::f64::consts::PI {
                diff -= 2.0 * std::f64::consts::PI;
            }
            while diff < -std::f64::consts::PI {
                diff += 2.0 * std::f64::consts::PI;
            }
        }
        unwrapped.push(unwrapped[i - 1] + diff);
    }
    unwrapped
}

/// Compute the envelope of a signal via Hilbert transform.
///
/// Returns the magnitude of the analytic signal.
/// Matches `scipy.signal.hilbert` → `np.abs(...)`.
pub fn envelope(x: &[f64]) -> Result<Vec<f64>, SignalError> {
    hilbert_envelope(x)
}

/// Zero-crossing rate: fraction of consecutive samples with sign change.
///
/// Useful for audio/speech analysis.
pub fn zero_crossing_rate(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let crossings = x
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f64 / (x.len() - 1) as f64
}

/// Compute the short-time energy of a signal.
///
/// Returns the energy in each frame of length `frame_len` with `hop_len` stride.
pub fn short_time_energy(x: &[f64], frame_len: usize, hop_len: usize) -> Vec<f64> {
    if frame_len == 0 || hop_len == 0 || x.is_empty() {
        return vec![];
    }
    let mut energies = Vec::new();
    let mut start = 0;
    while start + frame_len <= x.len() {
        let energy: f64 = x[start..start + frame_len].iter().map(|&v| v * v).sum();
        energies.push(energy);
        start += hop_len;
    }
    energies
}

/// Compute the autocorrelation of a signal.
///
/// Returns the normalized autocorrelation for lags 0 to max_lag.
pub fn autocorrelation(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum();

    if var == 0.0 {
        return vec![1.0; max_lag + 1];
    }

    (0..=max_lag.min(n - 1))
        .map(|lag| {
            let sum: f64 = (0..n - lag)
                .map(|i| (x[i] - mean) * (x[i + lag] - mean))
                .sum();
            sum / var
        })
        .collect()
}

/// Compute the spectral centroid of a magnitude spectrum.
///
/// Returns the weighted mean frequency.
pub fn spectral_centroid(magnitudes: &[f64], freqs: &[f64]) -> f64 {
    if magnitudes.is_empty() || freqs.is_empty() {
        return 0.0;
    }
    let total: f64 = magnitudes.iter().zip(freqs.iter()).map(|(&m, _)| m).sum();
    if total == 0.0 {
        return 0.0;
    }
    magnitudes
        .iter()
        .zip(freqs.iter())
        .map(|(&m, &f)| m * f)
        .sum::<f64>()
        / total
}

/// Compute the spectral rolloff frequency.
///
/// Returns the frequency below which `rolloff_percent` of the total energy is contained.
pub fn spectral_rolloff(magnitudes: &[f64], freqs: &[f64], rolloff_percent: f64) -> f64 {
    if magnitudes.is_empty() || freqs.is_empty() {
        return 0.0;
    }
    let total: f64 = magnitudes.iter().zip(freqs.iter()).map(|(&m, _)| m).sum();
    let threshold = total * rolloff_percent / 100.0;
    let mut cumsum = 0.0;
    for (&m, &f) in magnitudes.iter().zip(freqs.iter()) {
        cumsum += m;
        if cumsum >= threshold {
            return f;
        }
    }
    freqs.last().copied().unwrap_or(0.0)
}

/// Spectral bandwidth: weighted standard deviation of frequencies.
pub fn spectral_bandwidth(magnitudes: &[f64], freqs: &[f64]) -> f64 {
    if magnitudes.is_empty() || freqs.is_empty() {
        return 0.0;
    }
    let centroid = spectral_centroid(magnitudes, freqs);
    let total: f64 = magnitudes.iter().zip(freqs.iter()).map(|(&m, _)| m).sum();
    if total == 0.0 {
        return 0.0;
    }
    let var: f64 = magnitudes
        .iter()
        .zip(freqs.iter())
        .map(|(&m, &f)| m * (f - centroid).powi(2))
        .sum::<f64>()
        / total;
    var.sqrt()
}

/// Spectral flatness: geometric mean / arithmetic mean of power spectrum.
///
/// 1.0 = white noise, 0.0 = pure tone.
pub fn spectral_flatness(magnitudes: &[f64]) -> f64 {
    if magnitudes.is_empty() {
        return 0.0;
    }
    let n = magnitudes.len() as f64;
    let arith_mean: f64 = magnitudes.iter().sum::<f64>() / n;
    if arith_mean == 0.0 {
        return 0.0;
    }
    let log_sum: f64 = magnitudes
        .iter()
        .map(|&m| if m > 0.0 { m.ln() } else { -700.0 })
        .sum();
    let geom_mean = (log_sum / n).exp();
    (geom_mean / arith_mean).clamp(0.0, 1.0)
}

/// Root mean square (RMS) of a signal.
pub fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

/// Peak-to-peak amplitude of a signal.
pub fn peak_to_peak(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let min = x.iter().cloned().fold(f64::INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    });
    max - min
}

/// Crest factor: peak / RMS ratio.
pub fn crest_factor(x: &[f64]) -> f64 {
    let r = rms(x);
    if r == 0.0 {
        return 0.0;
    }
    let peak = x.iter().map(|&v| v.abs()).fold(0.0f64, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    peak / r
}

/// Compute the power of a signal in decibels relative to reference.
pub fn power_db(x: &[f64], ref_power: f64) -> f64 {
    let power: f64 = x.iter().map(|&v| v * v).sum::<f64>() / x.len().max(1) as f64;
    if power <= 0.0 || ref_power <= 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * (power / ref_power).log10()
}

/// Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1].
///
/// Common in speech processing with coeff ≈ 0.97.
///
/// # Arguments
/// * `x` — Input signal.
/// * `coeff` — Pre-emphasis coefficient. Non-finite values treated as 0.
pub fn preemphasis(x: &[f64], coeff: f64) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let coeff = if coeff.is_finite() { coeff } else { 0.0 };
    let mut y = Vec::with_capacity(x.len());
    y.push(x[0]);
    for i in 1..x.len() {
        y.push(x[i] - coeff * x[i - 1]);
    }
    y
}

/// Apply de-emphasis filter (inverse of pre-emphasis).
///
/// # Arguments
/// * `x` — Input signal.
/// * `coeff` — De-emphasis coefficient. Non-finite values treated as 0.
pub fn deemphasis(x: &[f64], coeff: f64) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let coeff = if coeff.is_finite() { coeff } else { 0.0 };
    let mut y = Vec::with_capacity(x.len());
    y.push(x[0]);
    for i in 1..x.len() {
        y.push(x[i] + coeff * y[i - 1]);
    }
    y
}

/// Frame a signal into overlapping windows.
///
/// Returns frames of length `frame_len` with `hop_len` stride.
pub fn frame_signal(x: &[f64], frame_len: usize, hop_len: usize) -> Vec<Vec<f64>> {
    if frame_len == 0 || hop_len == 0 || x.len() < frame_len {
        return vec![];
    }
    let mut frames = Vec::new();
    let mut start = 0;
    while start + frame_len <= x.len() {
        frames.push(x[start..start + frame_len].to_vec());
        start += hop_len;
    }
    frames
}

/// Bilinear transform: convert analog (s-domain) to digital (z-domain) filter.
///
/// Transforms analog numerator/denominator to digital using the bilinear transform
/// s = 2*fs*(z-1)/(z+1).
///
/// Returns (b_digital, a_digital).
/// Matches `scipy.signal.bilinear`.
pub fn bilinear(b_analog: &[f64], a_analog: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    let n = b_analog.len().max(a_analog.len());
    let d = n - 1;

    // Pad to same length
    let mut b = vec![0.0; n];
    let mut a = vec![0.0; n];
    for (i, &v) in b_analog.iter().enumerate() {
        b[n - b_analog.len() + i] = v;
    }
    for (i, &v) in a_analog.iter().enumerate() {
        a[n - a_analog.len() + i] = v;
    }

    let fs2 = 2.0 * fs;

    // Apply the bilinear transform via polynomial manipulation
    let mut b_dig = vec![0.0; n];
    let mut a_dig = vec![0.0; n];

    for j in 0..n {
        let mut val_b = 0.0;
        let mut val_a = 0.0;
        for i in 0..n {
            let k = d - i;
            // Binomial coefficient C(k, j') and C(d-k, j-j') patterns
            let mut coeff = 0.0;
            for jp in 0..=j.min(k) {
                let rem = j - jp;
                if rem <= d - k {
                    let c1 = binom_coeff(k, jp);
                    let c2 = binom_coeff(d - k, rem);
                    let sign = if (d - k - rem).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    coeff += c1 * c2 * sign;
                }
            }
            val_b += b[i] * coeff * fs2.powi(k as i32);
            val_a += a[i] * coeff * fs2.powi(k as i32);
        }
        b_dig[j] = val_b;
        a_dig[j] = val_a;
    }

    // Normalize by a[0]
    if a_dig[0].abs() > 1e-30 {
        let norm = a_dig[0];
        for v in &mut b_dig {
            *v /= norm;
        }
        for v in &mut a_dig {
            *v /= norm;
        }
    }

    (b_dig, a_dig)
}

fn binom_coeff(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// Compute the impulse response of a digital filter.
///
/// Returns n_samples of the response to a unit impulse.
/// Matches `scipy.signal.dimpulse` (simplified).
pub fn impulse_response(b: &[f64], a: &[f64], n_samples: usize) -> Result<Vec<f64>, SignalError> {
    let tf_num = normalize_discrete_tf_numerator(b, a)?;
    let mut impulse = vec![0.0; n_samples];
    if !impulse.is_empty() {
        impulse[0] = 1.0;
    }
    lfilter(&tf_num, a, &impulse, None)
}

/// Compute the step response of a digital filter.
///
/// Returns n_samples of the response to a unit step.
/// Matches `scipy.signal.dstep` (simplified).
pub fn step_response(b: &[f64], a: &[f64], n_samples: usize) -> Result<Vec<f64>, SignalError> {
    let tf_num = normalize_discrete_tf_numerator(b, a)?;
    let step = vec![1.0; n_samples];
    lfilter(&tf_num, a, &step, None)
}

fn normalize_discrete_tf_numerator(b: &[f64], a: &[f64]) -> Result<Vec<f64>, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }
    if b.len() > a.len() {
        return Err(SignalError::InvalidArgument(
            "improper transfer function: numerator order must not exceed denominator order"
                .to_string(),
        ));
    }

    let mut padded = vec![0.0; a.len() - b.len()];
    padded.extend_from_slice(b);
    Ok(padded)
}

/// Compute the group delay of a digital filter at specified frequencies.
///
/// Returns the group delay τ(ω) = -d(phase)/dω.
/// This is a convenience wrapper that returns (frequencies, delays).
pub fn group_delay_from_ba(b: &[f64], a: &[f64], n_freqs: usize) -> (Vec<f64>, Vec<f64>) {
    let mut freqs = Vec::with_capacity(n_freqs);
    let mut delays = Vec::with_capacity(n_freqs);

    for k in 0..n_freqs {
        let w = std::f64::consts::PI * k as f64 / n_freqs as f64;
        freqs.push(w);

        // Evaluate B(e^{jω}) and A(e^{jω})
        let mut br = 0.0;
        let mut bi = 0.0;
        for (i, &coeff) in b.iter().enumerate() {
            br += coeff * (w * i as f64).cos();
            bi -= coeff * (w * i as f64).sin();
        }

        let mut ar = 0.0;
        let mut ai = 0.0;
        for (i, &coeff) in a.iter().enumerate() {
            ar += coeff * (w * i as f64).cos();
            ai -= coeff * (w * i as f64).sin();
        }

        // Derivative: d/dω B(e^{jω})
        let mut dbr = 0.0;
        let mut dbi = 0.0;
        for (i, &coeff) in b.iter().enumerate() {
            let n = i as f64;
            dbr -= n * coeff * (w * n).sin();
            dbi -= n * coeff * (w * n).cos();
        }

        let mut dar = 0.0;
        let mut dai = 0.0;
        for (i, &coeff) in a.iter().enumerate() {
            let n = i as f64;
            dar -= n * coeff * (w * n).sin();
            dai -= n * coeff * (w * n).cos();
        }

        // Group delay: -d/dω[arg(H)] = Re[(dB*A - B*dA) / (B*A)]
        // Using: d/dω[arg(H)] = Im[H'/H] where H = B/A
        let h_mag2 = br * br + bi * bi;
        if h_mag2 > 1e-30 {
            let num_r = dbr * br + dbi * bi;
            let gd_b = -num_r / h_mag2;

            let a_mag2 = ar * ar + ai * ai;
            let gd_a = if a_mag2 > 1e-30 {
                let num_a = dar * ar + dai * ai;
                -num_a / a_mag2
            } else {
                0.0
            };

            delays.push(gd_b - gd_a);
        } else {
            delays.push(0.0);
        }
    }

    (freqs, delays)
}

/// Compute the magnitude response of a digital filter.
///
/// Returns (frequencies, magnitudes_db).
pub fn magnitude_response(b: &[f64], a: &[f64], n_freqs: usize) -> (Vec<f64>, Vec<f64>) {
    let mut freqs = Vec::with_capacity(n_freqs);
    let mut mags = Vec::with_capacity(n_freqs);

    for k in 0..n_freqs {
        let w = std::f64::consts::PI * k as f64 / n_freqs as f64;
        freqs.push(w);

        let mut br = 0.0;
        let mut bi = 0.0;
        for (i, &coeff) in b.iter().enumerate() {
            br += coeff * (w * i as f64).cos();
            bi -= coeff * (w * i as f64).sin();
        }

        let mut ar = 0.0;
        let mut ai = 0.0;
        for (i, &coeff) in a.iter().enumerate() {
            ar += coeff * (w * i as f64).cos();
            ai -= coeff * (w * i as f64).sin();
        }

        let h_mag2 = br * br + bi * bi;
        let a_mag2 = ar * ar + ai * ai;
        let mag = if a_mag2 > 1e-30 {
            (h_mag2 / a_mag2).sqrt()
        } else {
            0.0
        };

        mags.push(if mag > 0.0 {
            20.0 * mag.log10()
        } else {
            f64::NEG_INFINITY
        });
    }

    (freqs, mags)
}

/// Compute the phase response of a digital filter.
///
/// Returns (frequencies, phase_radians).
pub fn phase_response(b: &[f64], a: &[f64], n_freqs: usize) -> (Vec<f64>, Vec<f64>) {
    let mut freqs = Vec::with_capacity(n_freqs);
    let mut phases = Vec::with_capacity(n_freqs);

    for k in 0..n_freqs {
        let w = std::f64::consts::PI * k as f64 / n_freqs as f64;
        freqs.push(w);

        let mut br = 0.0;
        let mut bi = 0.0;
        for (i, &coeff) in b.iter().enumerate() {
            br += coeff * (w * i as f64).cos();
            bi -= coeff * (w * i as f64).sin();
        }

        let mut ar = 0.0;
        let mut ai = 0.0;
        for (i, &coeff) in a.iter().enumerate() {
            ar += coeff * (w * i as f64).cos();
            ai -= coeff * (w * i as f64).sin();
        }

        // H = B/A, phase = arg(B) - arg(A)
        let phase_b = bi.atan2(br);
        let phase_a = ai.atan2(ar);
        phases.push(phase_b - phase_a);
    }

    (freqs, phases)
}

/// Normalize a signal to have zero mean and unit variance.
pub fn normalize_signal(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let n = x.len() as f64;
    let mean: f64 = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std == 0.0 {
        return vec![0.0; x.len()];
    }
    x.iter().map(|&v| (v - mean) / std).collect()
}

/// Normalize a signal to range [0, 1].
pub fn normalize_minmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let min = x.iter().cloned().fold(f64::INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    });
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let range = max - min;
    if range == 0.0 {
        return vec![0.5; x.len()];
    }
    x.iter().map(|&v| (v - min) / range).collect()
}

/// Downsample a signal by taking every n-th sample.
pub fn downsample(x: &[f64], factor: usize) -> Vec<f64> {
    if factor == 0 {
        return x.to_vec();
    }
    x.iter().step_by(factor).cloned().collect()
}

/// Upsample a signal by inserting zeros between samples.
pub fn upsample(x: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 {
        return x.to_vec();
    }
    let mut result = vec![0.0; x.len() * factor];
    for (i, &v) in x.iter().enumerate() {
        result[i * factor] = v;
    }
    result
}

/// Compute the signal-to-noise ratio in dB.
pub fn snr(signal: &[f64], noise: &[f64]) -> f64 {
    let sig_power: f64 = signal.iter().map(|&v| v * v).sum::<f64>() / signal.len().max(1) as f64;
    let noise_power: f64 = noise.iter().map(|&v| v * v).sum::<f64>() / noise.len().max(1) as f64;
    if noise_power == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig_power / noise_power).log10()
}

/// Compute the total harmonic distortion (THD).
///
/// Given a signal and its fundamental frequency bin index, computes
/// THD = sqrt(sum of harmonic powers / fundamental power).
pub fn thd(magnitudes: &[f64], fundamental_bin: usize) -> f64 {
    if fundamental_bin >= magnitudes.len() || magnitudes[fundamental_bin] == 0.0 {
        return f64::NAN;
    }

    let fund_power = magnitudes[fundamental_bin].powi(2);
    let mut harmonic_power = 0.0;

    // Sum power at harmonics (2f, 3f, 4f, ...)
    let mut bin = 2 * fundamental_bin;
    while bin < magnitudes.len() {
        harmonic_power += magnitudes[bin].powi(2);
        bin += fundamental_bin;
    }

    (harmonic_power / fund_power).sqrt()
}

/// Add white Gaussian noise to a signal at a specified SNR (dB).
pub fn add_noise(signal: &[f64], snr_db: f64, seed: u64) -> Vec<f64> {
    let sig_power: f64 = signal.iter().map(|&v| v * v).sum::<f64>() / signal.len().max(1) as f64;
    let noise_power = sig_power / 10.0f64.powf(snr_db / 10.0);
    let noise_std = noise_power.sqrt();

    let mut rng = seed;
    signal
        .iter()
        .map(|&s| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((rng >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;
            let noise =
                noise_std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            s + noise
        })
        .collect()
}

/// Compute the energy of a signal.
pub fn signal_energy(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum()
}

/// Compute the power of a signal (energy per sample).
pub fn signal_power(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    signal_energy(x) / x.len() as f64
}

/// Compute the cross-correlation coefficient between two signals.
///
/// Returns a value between -1 and 1.
pub fn xcorr_coefficient(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let n = x.len() as f64;
    let mx: f64 = x.iter().sum::<f64>() / n;
    let my: f64 = y.iter().sum::<f64>() / n;
    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - mx) * (b - my))
        .sum();
    let sx: f64 = x.iter().map(|&a| (a - mx).powi(2)).sum::<f64>().sqrt();
    let sy: f64 = y.iter().map(|&b| (b - my).powi(2)).sum::<f64>().sqrt();
    if sx * sy == 0.0 {
        return 0.0;
    }
    cov / (sx * sy)
}

/// Compute the delay between two signals via cross-correlation.
///
/// Returns the lag (in samples) that maximizes cross-correlation.
pub fn find_delay(x: &[f64], y: &[f64]) -> i64 {
    let n = x.len();
    let m = y.len();
    if n == 0 || m == 0 {
        return 0;
    }

    let max_lag = n.max(m) - 1;
    let mut best_lag: i64 = 0;
    let mut best_corr = f64::NEG_INFINITY;

    for lag in -(max_lag as i64)..=(max_lag as i64) {
        let mut sum = 0.0;
        for (i, &x_i) in x.iter().enumerate().take(n) {
            let j = i as i64 - lag;
            if j >= 0 && (j as usize) < m {
                sum += x_i * y[j as usize];
            }
        }
        if sum > best_corr {
            best_corr = sum;
            best_lag = lag;
        }
    }

    best_lag
}

/// Apply a moving median filter.
pub fn medfilt1(x: &[f64], kernel_size: usize) -> Vec<f64> {
    if x.is_empty() || kernel_size == 0 {
        return x.to_vec();
    }
    let half = kernel_size / 2;
    let n = x.len();

    (0..n)
        .map(|i| {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let mut window: Vec<f64> = x[start..end].to_vec();
            window.sort_by(|a, b| a.total_cmp(b));
            window[window.len() / 2]
        })
        .collect()
}

/// Apply exponential smoothing to a signal.
///
/// # Arguments
/// * `x` — Input signal.
/// * `alpha` — Smoothing factor in [0, 1]. Values near 0 give more smoothing.
pub fn exponential_smooth(x: &[f64], alpha: f64) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let alpha = if !alpha.is_finite() {
        0.5
    } else {
        alpha.clamp(0.0, 1.0)
    };
    let mut result = Vec::with_capacity(x.len());
    result.push(x[0]);
    for i in 1..x.len() {
        result.push(alpha * x[i] + (1.0 - alpha) * result[i - 1]);
    }
    result
}

/// Compute the analytic signal magnitude (envelope) via absolute value of Hilbert.
pub fn analytic_envelope(x: &[f64]) -> Result<Vec<f64>, SignalError> {
    hilbert_envelope(x)
}

/// Compute the frequency of maximum power in a spectrum.
pub fn dominant_frequency(magnitudes: &[f64], freqs: &[f64]) -> f64 {
    if magnitudes.is_empty() || freqs.is_empty() {
        return 0.0;
    }
    let max_idx = magnitudes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    freqs.get(max_idx).copied().unwrap_or(0.0)
}

/// Compute the spectral entropy.
///
/// Normalized Shannon entropy of the power spectrum.
pub fn spectral_entropy(magnitudes: &[f64]) -> f64 {
    if magnitudes.is_empty() {
        return 0.0;
    }
    let total: f64 = magnitudes.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &m in magnitudes {
        if m > 0.0 {
            let p = m / total;
            entropy -= p * p.ln();
        }
    }

    // Normalize by log(N)
    let n = magnitudes.len() as f64;
    if n > 1.0 { entropy / n.ln() } else { entropy }
}

/// Convert frequency in Hz to Mel scale.
pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).ln() / std::f64::consts::LN_10
}

/// Convert Mel scale to frequency in Hz.
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

/// Generate Mel filterbank matrix.
///
/// Returns a matrix of shape (n_mels, n_fft/2+1) for applying to a magnitude spectrum.
pub fn mel_filterbank(n_mels: usize, n_fft: usize, sr: f64, fmin: f64, fmax: f64) -> Vec<Vec<f64>> {
    let n_freq = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax.min(sr / 2.0));

    // Mel-spaced center frequencies
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let bin_points: Vec<f64> = hz_points.iter().map(|&hz| hz * n_fft as f64 / sr).collect();

    let mut filterbank = vec![vec![0.0; n_freq]; n_mels];

    for m in 0..n_mels {
        let start = bin_points[m];
        let center = bin_points[m + 1];
        let end = bin_points[m + 2];

        for (k, val) in filterbank[m].iter_mut().enumerate().take(n_freq) {
            let kf = k as f64;
            if kf >= start && kf <= center && center > start {
                *val = (kf - start) / (center - start);
            } else if kf > center && kf <= end && end > center {
                *val = (end - kf) / (end - center);
            }
        }
    }

    filterbank
}

/// Compute Mel-frequency cepstral coefficients (MFCCs).
///
/// Returns a matrix of shape (n_frames, n_mfcc).
pub fn mfcc(
    signal: &[f64],
    sr: f64,
    n_mfcc: usize,
    n_mels: usize,
    frame_len: usize,
    hop_len: usize,
) -> Vec<Vec<f64>> {
    if signal.is_empty() || frame_len == 0 || hop_len == 0 {
        return vec![];
    }

    let n_fft = frame_len;
    let n_freq = n_fft / 2 + 1;
    let fb = mel_filterbank(n_mels, n_fft, sr, 0.0, sr / 2.0);

    // Window
    let window: Vec<f64> = (0..frame_len)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (frame_len - 1) as f64).cos())
        })
        .collect();

    let mut mfccs = Vec::new();
    let mut start = 0;

    while start + frame_len <= signal.len() {
        // Windowed frame
        let frame: Vec<f64> = signal[start..start + frame_len]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        // Power spectrum via DFT
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut power_spec = vec![0.0; n_freq];
        for (k, spec_val) in power_spec.iter_mut().enumerate().take(n_freq) {
            let mut re = 0.0;
            let mut im = 0.0;
            for (n, &s) in frame.iter().enumerate() {
                let angle = two_pi * k as f64 * n as f64 / n_fft as f64;
                re += s * angle.cos();
                im -= s * angle.sin();
            }
            *spec_val = (re * re + im * im) / n_fft as f64;
        }

        // Apply Mel filterbank
        let mel_energies: Vec<f64> = fb
            .iter()
            .map(|filter| {
                let energy: f64 = filter
                    .iter()
                    .zip(power_spec.iter())
                    .map(|(&f, &p)| f * p)
                    .sum();
                energy.max(1e-10).ln() // log Mel energies
            })
            .collect();

        // DCT (type-II) to get MFCCs
        let mut coeffs = Vec::with_capacity(n_mfcc);
        for i in 0..n_mfcc {
            let mut sum = 0.0;
            for (j, &e) in mel_energies.iter().enumerate() {
                sum +=
                    e * (std::f64::consts::PI * i as f64 * (j as f64 + 0.5) / n_mels as f64).cos();
            }
            coeffs.push(sum * (2.0 / n_mels as f64).sqrt());
        }

        mfccs.push(coeffs);
        start += hop_len;
    }

    mfccs
}

/// Compute the chroma feature (pitch class profile) from a magnitude spectrum.
///
/// Maps frequency bins to 12 pitch classes (C, C#, D, ..., B).
pub fn chroma(magnitudes: &[f64], sr: f64, n_fft: usize) -> [f64; 12] {
    let n_freq = magnitudes.len();
    let mut chroma_vec = [0.0f64; 12];

    for (k, magnitude) in magnitudes.iter().enumerate().take(n_freq).skip(1) {
        let freq = k as f64 * sr / n_fft as f64;
        if !(20.0..=5000.0).contains(&freq) {
            continue;
        }
        // Map frequency to pitch class: pitch = 12 * log2(f / 440) + 69
        let midi = 12.0 * (freq / 440.0).log2() + 69.0;
        let pitch_class = ((midi.round() as i64 % 12 + 12) % 12) as usize;
        chroma_vec[pitch_class] += *magnitude;
    }

    // Normalize
    let max_val = chroma_vec
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .fold(0.0_f64, f64::max);
    if max_val > 0.0 {
        for v in &mut chroma_vec {
            *v /= max_val;
        }
    }

    chroma_vec
}

/// Compute the spectral contrast between peaks and valleys.
pub fn spectral_contrast(magnitudes: &[f64], n_bands: usize) -> Vec<f64> {
    if magnitudes.is_empty() || n_bands == 0 {
        return vec![];
    }

    let n = magnitudes.len();
    let band_size = n / n_bands;
    if band_size == 0 {
        return vec![0.0; n_bands];
    }

    (0..n_bands)
        .map(|b| {
            let start = b * band_size;
            let end = ((b + 1) * band_size).min(n);
            let band = &magnitudes[start..end];
            if band.is_empty() {
                return 0.0;
            }
            let mut sorted = band.to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let peak = sorted[sorted.len() - 1];
            let valley = sorted[0];
            if valley > 0.0 {
                (peak / valley).log10() * 20.0
            } else {
                0.0
            }
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════
// IIR Filter Design
// ══════════════════════════════════════════════════════════════════════

/// Filter type for IIR design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Low-pass filter.
    Lowpass,
    /// High-pass filter.
    Highpass,
    /// Band-pass filter.
    Bandpass,
    /// Band-stop filter.
    Bandstop,
}

/// Analog prototype family for IIR design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IirFamily {
    /// Butterworth: maximally flat passband.
    Butterworth,
    /// Chebyshev Type I: equiripple passband.
    Chebyshev1,
    /// Chebyshev Type II: equiripple stopband.
    Chebyshev2,
    /// Bessel/Thomson: maximally flat group delay.
    Bessel,
    /// Elliptic/Cauer: equiripple both passband and stopband. Sharpest transition.
    Elliptic,
}

/// IIR filter coefficients in transfer function form (b, a).
///
/// H(z) = B(z)/A(z) where B(z) = b[0] + b[1]*z^-1 + ... and A(z) = 1 + a[1]*z^-1 + ...
#[derive(Debug, Clone, PartialEq)]
pub struct BaCoeffs {
    /// Numerator coefficients.
    pub b: Vec<f64>,
    /// Denominator coefficients (a[0] is always 1.0).
    pub a: Vec<f64>,
}

/// Butterworth output form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButterOutput {
    /// Transfer-function coefficients `(b, a)`.
    Ba,
    /// Second-order sections.
    Sos,
}

/// Design a Butterworth IIR filter.
///
/// Matches `scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba')`.
///
/// # Arguments
/// * `order` — Filter order (1, 2, 3, ...)
/// * `wn` — Critical frequencies (normalized, 0 < wn < 1 for digital).
///   Use one value for lowpass/highpass and two values for bandpass/bandstop.
/// * `btype` — Filter type.
///
/// Returns (b, a) transfer function coefficients.
pub fn butter(order: usize, wn: &[f64], btype: FilterType) -> Result<BaCoeffs, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidArgument(
            "order must be >= 1".to_string(),
        ));
    }
    validate_iir_wn(wn, btype)?;

    let (proto_poles_re, proto_poles_im) = butterworth_analog_poles(order);
    let analog_zpk = ZpkCoeffs {
        zeros_re: Vec::new(),
        zeros_im: Vec::new(),
        poles_re: proto_poles_re,
        poles_im: proto_poles_im,
        gain: 1.0,
    };

    design_digital_iir(analog_zpk, wn, btype)
}

/// Design a Butterworth IIR filter with an explicit output form.
///
/// Matches the `output` kwarg of `scipy.signal.butter(..., output=...)`
/// for the supported digital forms.
pub fn butter_with_output(
    order: usize,
    wn: &[f64],
    btype: FilterType,
    output: ButterOutput,
) -> Result<BaCoeffsOrSos, SignalError> {
    let ba = butter(order, wn, btype)?;
    match output {
        ButterOutput::Ba => Ok(BaCoeffsOrSos::Ba(ba)),
        ButterOutput::Sos => Ok(BaCoeffsOrSos::Sos(tf2sos(&ba.b, &ba.a)?)),
    }
}

/// Butterworth design result for an explicit output form.
#[derive(Debug, Clone, PartialEq)]
pub enum BaCoeffsOrSos {
    /// Transfer-function coefficients `(b, a)`.
    Ba(BaCoeffs),
    /// Second-order sections.
    Sos(Vec<SosSection>),
}

/// Design a Butterworth IIR filter directly in second-order-section form.
///
/// Matches `scipy.signal.butter(..., output='sos')`.
pub fn butter_sos(
    order: usize,
    wn: &[f64],
    btype: FilterType,
) -> Result<Vec<SosSection>, SignalError> {
    match butter_with_output(order, wn, btype, ButterOutput::Sos)? {
        BaCoeffsOrSos::Sos(sos) => Ok(sos),
        BaCoeffsOrSos::Ba(_) => unreachable!("ButterOutput::Sos must return SOS"),
    }
}

/// Design a Chebyshev Type I IIR filter.
pub fn cheby1(
    order: usize,
    rp: f64,
    wn: &[f64],
    btype: FilterType,
) -> Result<BaCoeffs, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidArgument(
            "order must be >= 1".to_string(),
        ));
    }
    if !rp.is_finite() || rp <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "rp must be finite and > 0 dB".to_string(),
        ));
    }
    validate_iir_wn(wn, btype)?;
    let epsilon = (10_f64.powf(rp / 10.0) - 1.0).sqrt();
    let mu = (1.0 / epsilon).asinh() / order as f64;
    let mut poles_re = Vec::with_capacity(order);
    let mut poles_im = Vec::with_capacity(order);
    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        poles_re.push(-mu.sinh() * theta.sin());
        poles_im.push(mu.cosh() * theta.cos());
    }
    let analog_zpk = ZpkCoeffs {
        zeros_re: Vec::new(),
        zeros_im: Vec::new(),
        poles_re,
        poles_im,
        gain: 1.0,
    };
    let mut coeffs = design_digital_iir(analog_zpk, wn, btype)?;
    let passband_gain = cheby1_passband_reference_gain(order, rp);
    coeffs
        .b
        .iter_mut()
        .for_each(|coeff| *coeff *= passband_gain);
    Ok(coeffs)
}

/// Design a Chebyshev Type II IIR filter.
pub fn cheby2(
    order: usize,
    rs: f64,
    wn: &[f64],
    btype: FilterType,
) -> Result<BaCoeffs, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidArgument(
            "order must be >= 1".to_string(),
        ));
    }
    if !rs.is_finite() || rs <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "rs must be finite and > 0 dB".to_string(),
        ));
    }
    validate_iir_wn(wn, btype)?;
    let epsilon = 1.0 / (10_f64.powf(rs / 10.0) - 1.0).sqrt();
    let mu = (1.0 / epsilon).asinh() / order as f64;
    let mut zeros_re = Vec::with_capacity(order);
    let mut zeros_im = Vec::with_capacity(order);
    let mut poles_re = Vec::with_capacity(order);
    let mut poles_im = Vec::with_capacity(order);
    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        let cos_theta = theta.cos();
        // Odd-order Chebyshev-II prototypes have one zero at infinity.
        // Preserve that degree difference instead of materializing a huge finite zero.
        if cos_theta.abs() > 1.0e-12 {
            zeros_re.push(0.0);
            zeros_im.push(1.0 / cos_theta);
        }

        let proto_re = -mu.sinh() * theta.sin();
        let proto_im = mu.cosh() * theta.cos();
        let (pole_re, pole_im) = complex_div(1.0, 0.0, proto_re, proto_im);
        poles_re.push(pole_re);
        poles_im.push(pole_im);
    }
    let analog_zpk = ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain: 1.0,
    };
    design_digital_iir(analog_zpk, wn, btype)
}

/// Design a Bessel/Thomson IIR filter.
pub fn bessel(order: usize, wn: &[f64], btype: FilterType) -> Result<BaCoeffs, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidArgument(
            "order must be >= 1".to_string(),
        ));
    }
    validate_iir_wn(wn, btype)?;

    let coeffs = reverse_bessel_polynomial(order)?;
    let (mut poles_re, mut poles_im) = poly_roots(&coeffs)?;
    let phase_scale = reverse_bessel_phase_scale(order)?;
    for pole in &mut poles_re {
        *pole *= phase_scale;
    }
    for pole in &mut poles_im {
        *pole *= phase_scale;
    }
    let analog_zpk = ZpkCoeffs {
        zeros_re: Vec::new(),
        zeros_im: Vec::new(),
        poles_re,
        poles_im,
        gain: 1.0,
    };
    design_digital_iir(analog_zpk, wn, btype)
}

/// Design an Elliptic (Cauer) IIR filter.
///
/// The elliptic filter achieves the sharpest transition band for a given order,
/// at the cost of equiripple in both passband and stopband.
///
/// # Arguments
/// * `order` — Filter order
/// * `rp` — Maximum passband ripple (dB)
/// * `rs` — Minimum stopband attenuation (dB)
/// * `wn` — Critical frequencies (normalized, 0 < wn < 1)
/// * `btype` — Filter type
pub fn ellip(
    order: usize,
    rp: f64,
    rs: f64,
    wn: &[f64],
    btype: FilterType,
) -> Result<BaCoeffs, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidArgument(
            "order must be >= 1".to_string(),
        ));
    }
    if !rp.is_finite() || rp <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "rp must be finite and > 0 dB".to_string(),
        ));
    }
    if !rs.is_finite() || rs <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "rs must be finite and > 0 dB".to_string(),
        ));
    }
    validate_iir_wn(wn, btype)?;

    let analog_zpk = elliptic_analog_zpk(order, rp, rs)?;
    let mut coeffs = design_digital_iir(analog_zpk, wn, btype)?;
    let passband_gain = cheby1_passband_reference_gain(order, rp);
    coeffs
        .b
        .iter_mut()
        .for_each(|coeff| *coeff *= passband_gain);
    Ok(coeffs)
}

/// Design a second-order IIR notch (band-reject) filter.
///
/// Removes a narrow band of frequencies centered at `w0`.
///
/// Matches `scipy.signal.iirnotch(w0, Q)`.
///
/// # Arguments
/// * `w0` — Frequency to reject (normalized, 0 < w0 < 1).
/// * `q` — Quality factor. Higher Q = narrower notch.
///
/// Returns `BaCoeffs` (b, a) for the second-order notch filter.
pub fn iirnotch(w0: f64, q: f64) -> Result<BaCoeffs, SignalError> {
    if !w0.is_finite() || w0 <= 0.0 || w0 >= 1.0 {
        return Err(SignalError::InvalidArgument(
            "w0 must be in (0, 1)".to_string(),
        ));
    }
    if !q.is_finite() || q <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "Q must be positive".to_string(),
        ));
    }

    // Digital frequency: ω₀ = π·w0 (w0 is normalized to [0,1] where 1 = Nyquist)
    let omega0 = std::f64::consts::PI * w0;
    let bw = omega0 / q;
    let r = (1.0 - bw / 2.0).max(0.0);
    let cos_w0 = omega0.cos();

    // Numerator: zeros on unit circle at e^(±jω₀)
    // b = [1, -2cos(ω₀), 1] (normalized for unity gain at DC/Nyquist)
    let b0 = 1.0;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0;

    // Denominator: poles inside unit circle at r·e^(±jω₀)
    // a = [1, -2r·cos(ω₀), r²]
    let a0 = 1.0;
    let a1 = -2.0 * r * cos_w0;
    let a2 = r * r;

    // Normalize for unity passband gain
    let gain = (a0 + a1 + a2) / (b0 + b1 + b2);
    Ok(BaCoeffs {
        b: vec![b0 * gain, b1 * gain, b2 * gain],
        a: vec![a0, a1, a2],
    })
}

/// Design a second-order IIR peak (band-pass) filter.
///
/// Boosts a narrow band of frequencies centered at `w0`.
///
/// Matches `scipy.signal.iirpeak(w0, Q)`.
///
/// # Arguments
/// * `w0` — Center frequency (normalized, 0 < w0 < 1).
/// * `q` — Quality factor. Higher Q = narrower peak.
///
/// Returns `BaCoeffs` (b, a) for the second-order peak filter.
pub fn iirpeak(w0: f64, q: f64) -> Result<BaCoeffs, SignalError> {
    if !w0.is_finite() || w0 <= 0.0 || w0 >= 1.0 {
        return Err(SignalError::InvalidArgument(
            "w0 must be in (0, 1)".to_string(),
        ));
    }
    if !q.is_finite() || q <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "Q must be positive".to_string(),
        ));
    }

    let omega0 = std::f64::consts::PI * w0;
    let bw = omega0 / q;
    let r = (1.0 - bw / 2.0).max(0.0);
    let cos_w0 = omega0.cos();

    // Peak filter: poles on unit circle, zeros inside
    // b = [1-r², 0, -(1-r²)] / 2 (normalized to unity gain at w0)
    // a = [1, -2r·cos(ω₀), r²]
    let gain_factor = (1.0 - r * r) / 2.0;
    let b = vec![gain_factor, 0.0, -gain_factor];
    let a = vec![1.0, -2.0 * r * cos_w0, r * r];

    Ok(BaCoeffs { b, a })
}

/// General IIR filter design dispatcher.
pub fn iirfilter(
    order: usize,
    wn: &[f64],
    btype: FilterType,
    family: IirFamily,
    rp: Option<f64>,
    rs: Option<f64>,
) -> Result<BaCoeffs, SignalError> {
    match family {
        IirFamily::Butterworth => butter(order, wn, btype),
        IirFamily::Chebyshev1 => cheby1(
            order,
            rp.ok_or_else(|| {
                SignalError::InvalidArgument("cheby1 requires passband ripple rp".to_string())
            })?,
            wn,
            btype,
        ),
        IirFamily::Chebyshev2 => cheby2(
            order,
            rs.ok_or_else(|| {
                SignalError::InvalidArgument("cheby2 requires stopband attenuation rs".to_string())
            })?,
            wn,
            btype,
        ),
        IirFamily::Bessel => bessel(order, wn, btype),
        IirFamily::Elliptic => {
            let rp_val = rp.ok_or_else(|| {
                SignalError::InvalidArgument("elliptic requires passband ripple rp".to_string())
            })?;
            let rs_val = rs.ok_or_else(|| {
                SignalError::InvalidArgument(
                    "elliptic requires stopband attenuation rs".to_string(),
                )
            })?;
            ellip(order, rp_val, rs_val, wn, btype)
        }
    }
}

fn cheby1_passband_reference_gain(order: usize, rp: f64) -> f64 {
    if order.is_multiple_of(2) {
        10_f64.powf(-rp / 20.0)
    } else {
        1.0
    }
}

fn design_digital_iir(
    analog_zpk: ZpkCoeffs,
    wn: &[f64],
    btype: FilterType,
) -> Result<BaCoeffs, SignalError> {
    let warped: Vec<f64> = wn.iter().map(|&w| prewarp_digital_frequency(w)).collect();
    let analog_zpk = match btype {
        FilterType::Lowpass => lowpass_zpk(&analog_zpk, warped[0]),
        FilterType::Highpass => highpass_zpk(&analog_zpk, warped[0]),
        FilterType::Bandpass => bandpass_zpk(&analog_zpk, warped[0], warped[1]),
        FilterType::Bandstop => bandstop_zpk(&analog_zpk, warped[0], warped[1]),
    };

    let digital_zpk = bilinear_zpk(&analog_zpk);
    let mut ba = zpk2tf(&digital_zpk);
    normalize_digital_ba(&mut ba, &warped, btype)?;
    Ok(ba)
}

fn normalize_digital_ba(
    ba: &mut BaCoeffs,
    warped: &[f64],
    btype: FilterType,
) -> Result<(), SignalError> {
    let normalize_at = match btype {
        FilterType::Lowpass => 0.0,
        FilterType::Highpass => std::f64::consts::PI,
        FilterType::Bandpass => 2.0 * ((warped[0] * warped[1]).sqrt() / 2.0).atan(),
        FilterType::Bandstop => 0.0,
    };
    let (b_re, b_im) = eval_poly_on_unit_circle(&ba.b, normalize_at);
    let (a_re, a_im) = eval_poly_on_unit_circle(&ba.a, normalize_at);
    let b_mag = (b_re * b_re + b_im * b_im).sqrt();
    let a_mag = (a_re * a_re + a_im * a_im).sqrt();
    if b_mag < 1.0e-30 || a_mag < 1.0e-30 {
        return Err(SignalError::InvalidArgument(
            "degenerate filter: could not normalize digital response".to_string(),
        ));
    }
    let gain = a_mag / b_mag;
    ba.b.iter_mut().for_each(|coeff| *coeff *= gain);
    Ok(())
}

fn elliptic_analog_zpk(order: usize, rp: f64, rs: f64) -> Result<ZpkCoeffs, SignalError> {
    if order == 1 {
        let pole = -(1.0 / pow10m1(0.1 * rp)).sqrt();
        return Ok(ZpkCoeffs {
            zeros_re: Vec::new(),
            zeros_im: Vec::new(),
            poles_re: vec![pole],
            poles_im: vec![0.0],
            gain: -pole,
        });
    }

    let eps_sq = pow10m1(0.1 * rp);
    let ck1_sq = eps_sq / pow10m1(0.1 * rs);
    if ck1_sq <= 0.0 || !ck1_sq.is_finite() {
        return Err(SignalError::InvalidArgument(
            "Cannot design a filter with given rp and rs specifications.".to_string(),
        ));
    }

    let eps = eps_sq.sqrt();
    let m = ellipdeg(order, ck1_sq)?;
    let capk = complete_ellipk(m)?;
    let start = if order.is_multiple_of(2) { 1 } else { 0 };

    let mut zeros_re = Vec::new();
    let mut zeros_im = Vec::new();
    let mut base_poles = Vec::new();
    for j in (start..order).step_by(2) {
        let u = j as f64 * capk / order as f64;
        let (s, c, d) = jacobi_ellipj_real(u, m)?;
        if s.abs() > 1.0e-14 {
            zeros_re.push(0.0);
            zeros_im.push(1.0 / (m.sqrt() * s));
        }

        base_poles.push((s, c, d));
    }

    let r = arc_jac_sc1_real(1.0 / eps, ck1_sq)?;
    let v0 = capk * r / (order as f64 * complete_ellipk(ck1_sq)?);
    let (sv, cv, dv) = jacobi_ellipj_real(v0, 1.0 - m)?;

    let mut poles_re = Vec::with_capacity(order);
    let mut poles_im = Vec::with_capacity(order);
    for &(s, c, d) in &base_poles {
        let denom = 1.0 - (d * sv).powi(2);
        let pole_re = -(c * d * sv * cv) / denom;
        let pole_im = -(s * dv) / denom;
        poles_re.push(pole_re);
        poles_im.push(pole_im);
    }
    if order.is_multiple_of(2) {
        for &(s, c, d) in &base_poles {
            let denom = 1.0 - (d * sv).powi(2);
            poles_re.push(-(c * d * sv * cv) / denom);
            poles_im.push((s * dv) / denom);
        }
    } else {
        for &(s, c, d) in &base_poles {
            let denom = 1.0 - (d * sv).powi(2);
            let pole_re = -(c * d * sv * cv) / denom;
            let pole_im = -(s * dv) / denom;
            if pole_im.abs() > 1.0e-14 * pole_re.hypot(pole_im) {
                poles_re.push(pole_re);
                poles_im.push(-pole_im);
            }
        }
    }

    let zero_count = zeros_re.len();
    zeros_re.extend_from_within(..);
    let zero_conjugates: Vec<f64> = zeros_im[..zero_count].iter().map(|&value| -value).collect();
    zeros_im.extend(zero_conjugates);

    let (pole_prod_re, pole_prod_im) = complex_product_neg_roots(&poles_re, &poles_im);
    let (zero_prod_re, zero_prod_im) = complex_product_neg_roots(&zeros_re, &zeros_im);
    let (mut gain, _) = complex_div_complex(pole_prod_re, pole_prod_im, zero_prod_re, zero_prod_im);
    if order.is_multiple_of(2) {
        gain /= (1.0 + eps_sq).sqrt();
    }

    Ok(ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain,
    })
}

fn pow10m1(x: f64) -> f64 {
    (std::f64::consts::LN_10 * x).exp_m1()
}

fn ellipdeg(order: usize, m1: f64) -> Result<f64, SignalError> {
    let k1 = complete_ellipk(m1)?;
    let k1p = complete_ellipkm1(m1)?;
    let q1 = (-std::f64::consts::PI * k1p / k1).exp();
    let q = q1.powf(1.0 / order as f64);

    let num = (0..=7).map(|m| q.powi(m * (m + 1))).sum::<f64>();
    let den = 1.0 + 2.0 * (1..=8).map(|m| q.powi(m * m)).sum::<f64>();
    Ok(16.0 * q * (num / den).powi(4))
}

fn complete_ellipk(m: f64) -> Result<f64, SignalError> {
    if !(0.0..=1.0).contains(&m) || !m.is_finite() {
        return Err(SignalError::InvalidArgument(
            "elliptic modulus must be finite and in [0, 1]".to_string(),
        ));
    }
    if m == 1.0 {
        return Ok(f64::INFINITY);
    }
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    for _ in 0..64 {
        let next_a = (a + b) / 2.0;
        let next_b = (a * b).sqrt();
        if (next_a - next_b).abs() <= 1.0e-15 * next_a.abs().max(1.0) {
            return Ok(std::f64::consts::PI / (2.0 * next_a));
        }
        a = next_a;
        b = next_b;
    }
    Ok(std::f64::consts::PI / (2.0 * a))
}

fn complete_ellipkm1(m: f64) -> Result<f64, SignalError> {
    if !(0.0..=1.0).contains(&m) || !m.is_finite() {
        return Err(SignalError::InvalidArgument(
            "elliptic complementary modulus must be finite and in [0, 1]".to_string(),
        ));
    }
    let mut a = 1.0;
    let mut b = m.sqrt();
    for _ in 0..64 {
        let next_a = (a + b) / 2.0;
        let next_b = (a * b).sqrt();
        if (next_a - next_b).abs() <= 1.0e-15 * next_a.abs().max(1.0) {
            return Ok(std::f64::consts::PI / (2.0 * next_a));
        }
        a = next_a;
        b = next_b;
    }
    Ok(std::f64::consts::PI / (2.0 * a))
}

fn arc_jac_sc1_real(w: f64, m: f64) -> Result<f64, SignalError> {
    let mut low = 0.0;
    let mut high = complete_ellipkm1(m)? * (1.0 - 1.0e-14);
    for _ in 0..96 {
        let mid = (low + high) / 2.0;
        let (sn, cn, _) = jacobi_ellipj_real(mid, 1.0 - m)?;
        let sc = sn / cn;
        if sc < w {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok((low + high) / 2.0)
}

fn jacobi_ellipj_real(u: f64, m: f64) -> Result<(f64, f64, f64), SignalError> {
    if !(0.0..=1.0).contains(&m) || !m.is_finite() || !u.is_finite() {
        return Err(SignalError::InvalidArgument(
            "Jacobi elliptic arguments must be finite with m in [0, 1]".to_string(),
        ));
    }
    if m <= 1.0e-15 {
        return Ok((u.sin(), u.cos(), 1.0));
    }
    if (1.0 - m).abs() <= 1.0e-15 {
        let cn = 1.0 / u.cosh();
        return Ok((u.tanh(), cn, cn));
    }

    let mut a_values = vec![1.0];
    let mut c_values = Vec::new();
    let mut b = (1.0 - m).sqrt();
    let mut twon = 1.0;
    for _ in 0..32 {
        let a = a_values.last().copied().unwrap_or(1.0);
        let c = (a - b) / 2.0;
        let next_a = (a + b) / 2.0;
        c_values.push(c);
        a_values.push(next_a);
        b = (a * b).sqrt();
        twon *= 2.0;
        if c.abs() <= 1.0e-15 * next_a.abs().max(1.0) {
            break;
        }
    }

    let mut phi = twon * a_values.last().copied().unwrap_or(1.0) * u;
    for index in (0..c_values.len()).rev() {
        let ratio = c_values[index] * phi.sin() / a_values[index + 1];
        phi = (ratio.clamp(-1.0, 1.0).asin() + phi) / 2.0;
    }
    let sn = phi.sin();
    let cn = phi.cos();
    let dn = (1.0 - m * sn * sn).max(0.0).sqrt();
    Ok((sn, cn, dn))
}

fn complex_product_neg_roots(roots_re: &[f64], roots_im: &[f64]) -> (f64, f64) {
    let mut prod_re = 1.0;
    let mut prod_im = 0.0;
    for (&root_re, &root_im) in roots_re.iter().zip(roots_im.iter()) {
        let (next_re, next_im) = complex_mul(prod_re, prod_im, -root_re, -root_im);
        prod_re = next_re;
        prod_im = next_im;
    }
    (prod_re, prod_im)
}

fn reverse_bessel_phase_scale(order: usize) -> Result<f64, SignalError> {
    let Some(double_order) = order.checked_mul(2) else {
        return Err(SignalError::InvalidArgument(
            "order is too large for Bessel polynomial construction".to_string(),
        ));
    };
    let a_last = factorial_f64_checked(double_order)?
        / (factorial_f64_checked(order)? * 2.0_f64.powi(order as i32));
    Ok(a_last.powf(-1.0 / order as f64))
}

fn reverse_bessel_polynomial(order: usize) -> Result<Vec<f64>, SignalError> {
    let Some(double_order) = order.checked_mul(2) else {
        return Err(SignalError::InvalidArgument(
            "order is too large for Bessel polynomial construction".to_string(),
        ));
    };
    let mut ascending = Vec::with_capacity(order + 1);
    for k in 0..=order {
        let numerator = factorial_f64_checked(double_order - k)?;
        let denominator = 2.0_f64.powi((order - k) as i32)
            * factorial_f64_checked(order - k)?
            * factorial_f64_checked(k)?;
        ascending.push(numerator / denominator);
    }
    ascending.reverse();
    Ok(ascending)
}

fn factorial_f64_checked(n: usize) -> Result<f64, SignalError> {
    let mut acc = 1.0;
    for value in 1..=n {
        acc *= value as f64;
        if !acc.is_finite() {
            return Err(SignalError::InvalidArgument(
                "order is too large for stable Bessel polynomial construction".to_string(),
            ));
        }
    }
    Ok(acc)
}

fn validate_iir_wn(wn: &[f64], btype: FilterType) -> Result<(), SignalError> {
    let expected_len = match btype {
        FilterType::Lowpass | FilterType::Highpass => 1,
        FilterType::Bandpass | FilterType::Bandstop => 2,
    };
    if wn.len() != expected_len {
        return Err(SignalError::InvalidArgument(format!(
            "Wn must contain {expected_len} value(s) for {btype:?}"
        )));
    }
    if wn
        .iter()
        .any(|&value| !value.is_finite() || value <= 0.0 || value >= 1.0)
    {
        return Err(SignalError::InvalidArgument(
            "Wn values must be finite and in (0, 1) for digital filters".to_string(),
        ));
    }
    if matches!(btype, FilterType::Bandpass | FilterType::Bandstop) && wn[0] >= wn[1] {
        return Err(SignalError::InvalidArgument(
            "band filters require Wn[0] < Wn[1]".to_string(),
        ));
    }
    Ok(())
}

fn prewarp_digital_frequency(wn: f64) -> f64 {
    2.0 * (std::f64::consts::PI * wn / 2.0).tan()
}

fn butterworth_analog_poles(order: usize) -> (Vec<f64>, Vec<f64>) {
    let mut poles_re = Vec::with_capacity(order);
    let mut poles_im = Vec::with_capacity(order);
    for k in 0..order {
        let theta =
            std::f64::consts::PI * (2.0 * k as f64 + order as f64 + 1.0) / (2.0 * order as f64);
        poles_re.push(theta.cos());
        poles_im.push(theta.sin());
    }
    (poles_re, poles_im)
}

fn lowpass_zpk(zpk: &ZpkCoeffs, wo: f64) -> ZpkCoeffs {
    let degree = zpk.poles_re.len().saturating_sub(zpk.zeros_re.len()) as i32;
    ZpkCoeffs {
        zeros_re: zpk.zeros_re.iter().map(|&zr| zr * wo).collect(),
        zeros_im: zpk.zeros_im.iter().map(|&zi| zi * wo).collect(),
        poles_re: zpk.poles_re.iter().map(|&pr| pr * wo).collect(),
        poles_im: zpk.poles_im.iter().map(|&pi| pi * wo).collect(),
        gain: zpk.gain * wo.powi(degree),
    }
}

fn highpass_zpk(zpk: &ZpkCoeffs, wo: f64) -> ZpkCoeffs {
    let degree = zpk.poles_re.len().saturating_sub(zpk.zeros_re.len());
    let mut zeros_re = Vec::with_capacity(zpk.zeros_re.len() + degree);
    let mut zeros_im = Vec::with_capacity(zpk.zeros_im.len() + degree);
    for (&zr, &zi) in zpk.zeros_re.iter().zip(zpk.zeros_im.iter()) {
        let (re, im) = complex_div(wo, 0.0, zr, zi);
        zeros_re.push(re);
        zeros_im.push(im);
    }
    zeros_re.extend(std::iter::repeat_n(0.0, degree));
    zeros_im.extend(std::iter::repeat_n(0.0, degree));

    let mut poles_re = Vec::with_capacity(zpk.poles_re.len());
    let mut poles_im = Vec::with_capacity(zpk.poles_im.len());
    for (&pr, &pi) in zpk.poles_re.iter().zip(zpk.poles_im.iter()) {
        let (re, im) = complex_div(wo, 0.0, pr, pi);
        poles_re.push(re);
        poles_im.push(im);
    }

    ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain: zpk.gain,
    }
}

fn bandpass_zpk(zpk: &ZpkCoeffs, w1: f64, w2: f64) -> ZpkCoeffs {
    let bw = w2 - w1;
    let wo = (w1 * w2).sqrt();
    let degree = zpk.poles_re.len().saturating_sub(zpk.zeros_re.len());
    let mut zeros_re = Vec::with_capacity(zpk.zeros_re.len() * 2 + degree);
    let mut zeros_im = Vec::with_capacity(zpk.zeros_im.len() * 2 + degree);
    let mut poles_re = Vec::with_capacity(zpk.poles_re.len() * 2);
    let mut poles_im = Vec::with_capacity(zpk.poles_im.len() * 2);

    for (&zr, &zi) in zpk.zeros_re.iter().zip(zpk.zeros_im.iter()) {
        let (base_re, base_im) = complex_scale(zr, zi, bw / 2.0);
        let (disc_re, disc_im) = complex_sub(
            base_re * base_re - base_im * base_im,
            2.0 * base_re * base_im,
            wo * wo,
            0.0,
        );
        let (root_re, root_im) = complex_sqrt(disc_re, disc_im);
        zeros_re.push(base_re + root_re);
        zeros_im.push(base_im + root_im);
        zeros_re.push(base_re - root_re);
        zeros_im.push(base_im - root_im);
    }
    zeros_re.extend(std::iter::repeat_n(0.0, degree));
    zeros_im.extend(std::iter::repeat_n(0.0, degree));

    for (&pr, &pi) in zpk.poles_re.iter().zip(zpk.poles_im.iter()) {
        let (base_re, base_im) = complex_scale(pr, pi, bw / 2.0);
        let (disc_re, disc_im) = complex_sub(
            base_re * base_re - base_im * base_im,
            2.0 * base_re * base_im,
            wo * wo,
            0.0,
        );
        let (root_re, root_im) = complex_sqrt(disc_re, disc_im);
        poles_re.push(base_re + root_re);
        poles_im.push(base_im + root_im);
        poles_re.push(base_re - root_re);
        poles_im.push(base_im - root_im);
    }

    ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain: zpk.gain * bw.powi(degree as i32),
    }
}

fn bandstop_zpk(zpk: &ZpkCoeffs, w1: f64, w2: f64) -> ZpkCoeffs {
    let bw = w2 - w1;
    let wo = (w1 * w2).sqrt();
    let degree = zpk.poles_re.len().saturating_sub(zpk.zeros_re.len());
    let mut zeros_re = Vec::with_capacity(zpk.zeros_re.len() * 2 + 2 * degree);
    let mut zeros_im = Vec::with_capacity(zpk.zeros_im.len() * 2 + 2 * degree);
    let mut poles_re = Vec::with_capacity(zpk.poles_re.len() * 2);
    let mut poles_im = Vec::with_capacity(zpk.poles_im.len() * 2);

    for (&zr, &zi) in zpk.zeros_re.iter().zip(zpk.zeros_im.iter()) {
        let (base_re, base_im) = complex_div(bw / 2.0, 0.0, zr, zi);
        let (disc_re, disc_im) = complex_sub(
            base_re * base_re - base_im * base_im,
            2.0 * base_re * base_im,
            wo * wo,
            0.0,
        );
        let (root_re, root_im) = complex_sqrt(disc_re, disc_im);
        zeros_re.push(base_re + root_re);
        zeros_im.push(base_im + root_im);
        zeros_re.push(base_re - root_re);
        zeros_im.push(base_im - root_im);
    }
    for _ in 0..degree {
        zeros_re.push(0.0);
        zeros_im.push(wo);
        zeros_re.push(0.0);
        zeros_im.push(-wo);
    }

    for (&pr, &pi) in zpk.poles_re.iter().zip(zpk.poles_im.iter()) {
        let (base_re, base_im) = complex_div(bw / 2.0, 0.0, pr, pi);
        let (disc_re, disc_im) = complex_sub(
            base_re * base_re - base_im * base_im,
            2.0 * base_re * base_im,
            wo * wo,
            0.0,
        );
        let (root_re, root_im) = complex_sqrt(disc_re, disc_im);
        poles_re.push(base_re + root_re);
        poles_im.push(base_im + root_im);
        poles_re.push(base_re - root_re);
        poles_im.push(base_im - root_im);
    }

    ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain: zpk.gain,
    }
}

fn bilinear_zpk(zpk: &ZpkCoeffs) -> ZpkCoeffs {
    let fs2 = 2.0;
    let degree = zpk.poles_re.len().saturating_sub(zpk.zeros_re.len());
    let mut zeros_re = Vec::with_capacity(zpk.zeros_re.len() + degree);
    let mut zeros_im = Vec::with_capacity(zpk.zeros_im.len() + degree);
    let mut poles_re = Vec::with_capacity(zpk.poles_re.len());
    let mut poles_im = Vec::with_capacity(zpk.poles_im.len());

    for (&zr, &zi) in zpk.zeros_re.iter().zip(zpk.zeros_im.iter()) {
        let (num_re, num_im) = complex_add(fs2, 0.0, zr, zi);
        let (den_re, den_im) = complex_sub(fs2, 0.0, zr, zi);
        let (re, im) = complex_div_complex(num_re, num_im, den_re, den_im);
        zeros_re.push(re);
        zeros_im.push(im);
    }
    zeros_re.extend(std::iter::repeat_n(-1.0, degree));
    zeros_im.extend(std::iter::repeat_n(0.0, degree));

    for (&pr, &pi) in zpk.poles_re.iter().zip(zpk.poles_im.iter()) {
        let (num_re, num_im) = complex_add(fs2, 0.0, pr, pi);
        let (den_re, den_im) = complex_sub(fs2, 0.0, pr, pi);
        let (re, im) = complex_div_complex(num_re, num_im, den_re, den_im);
        poles_re.push(re);
        poles_im.push(im);
    }

    let mut gain_num_re = zpk.gain;
    let mut gain_num_im = 0.0;
    for (&zr, &zi) in zpk.zeros_re.iter().zip(zpk.zeros_im.iter()) {
        let (re, im) = complex_sub(fs2, 0.0, zr, zi);
        let (new_re, new_im) = complex_mul(gain_num_re, gain_num_im, re, im);
        gain_num_re = new_re;
        gain_num_im = new_im;
    }
    let mut gain_den_re = 1.0;
    let mut gain_den_im = 0.0;
    for (&pr, &pi) in zpk.poles_re.iter().zip(zpk.poles_im.iter()) {
        let (re, im) = complex_sub(fs2, 0.0, pr, pi);
        let (new_re, new_im) = complex_mul(gain_den_re, gain_den_im, re, im);
        gain_den_re = new_re;
        gain_den_im = new_im;
    }
    let (gain_re, _) = complex_div_complex(gain_num_re, gain_num_im, gain_den_re, gain_den_im);

    ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain: gain_re,
    }
}

fn complex_add(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar + br, ai + bi)
}

fn complex_sub(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar - br, ai - bi)
}

fn complex_mul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

fn complex_scale(re: f64, im: f64, scalar: f64) -> (f64, f64) {
    (re * scalar, im * scalar)
}

fn complex_div(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    complex_div_complex(ar, ai, br, bi)
}

fn complex_div_complex(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    let denom = br * br + bi * bi;
    ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
}

fn complex_sqrt(re: f64, im: f64) -> (f64, f64) {
    if im == 0.0 {
        if re >= 0.0 {
            return (re.sqrt(), 0.0);
        }
        return (0.0, (-re).sqrt());
    }
    let r = (re * re + im * im).sqrt();
    let real = ((r + re) / 2.0).sqrt();
    let imag = ((r - re) / 2.0).sqrt().copysign(im);
    (real, imag)
}

/// Compute initial conditions for lfilter for step response steady-state.
///
/// Matches `scipy.signal.lfilter_zi(b, a)`.
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Result<Vec<f64>, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }

    let mut a_start = 0;
    while a_start + 1 < a.len() && a[a_start] == 0.0 {
        a_start += 1;
    }
    let a_trimmed = &a[a_start..];
    let a0 = a_trimmed[0];

    let nb = b.len();
    let na = a_trimmed.len();
    let n = nb.max(na);

    if n <= 1 {
        return Err(SignalError::InvalidArgument(
            "The length of `a` along the last axis must be at least 2.".to_string(),
        ));
    }

    // Pad and normalize to the direct-form-II transposed state-space system.
    let mut b_norm = vec![0.0; n];
    let mut a_norm = vec![0.0; n];
    for (i, &val) in b.iter().enumerate() {
        b_norm[i] = val / a0;
    }
    for (i, &val) in a_trimmed.iter().enumerate() {
        a_norm[i] = val / a0;
    }

    let m = n - 1;
    // Build the system (I - A)z = b[1:] - a[1:]*b[0]
    let mut matrix = vec![vec![0.0; m]; m];
    let mut rhs = vec![0.0; m];

    let b0 = b_norm[0];
    for i in 0..m {
        rhs[i] = b_norm[i + 1] - a_norm[i + 1] * b0;

        matrix[i][0] += a_norm[i + 1];
        matrix[i][i] += 1.0;
        if i + 1 < m {
            matrix[i][i + 1] -= 1.0;
        }
    }

    let solve_opts = fsci_linalg::SolveOptions {
        mode: fsci_runtime::RuntimeMode::Strict,
        ..Default::default()
    };

    let result = fsci_linalg::solve(&matrix, &rhs, solve_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("lfilter_zi solve failed: {e}")))?;

    Ok(result.x)
}

/// Construct initial conditions for `lfilter` from prior input/output history.
///
/// Matches `scipy.signal.lfiltic(b, a, y, x=None)`.
pub fn lfiltic(
    b: &[f64],
    a: &[f64],
    y: &[f64],
    x: Option<&[f64]>,
) -> Result<Vec<f64>, SignalError> {
    if a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "There must be at least one `a` coefficient.".to_string(),
        ));
    }
    if a[0] == 0.0 {
        return Err(SignalError::InvalidArgument(
            "First `a` filter coefficient must be non-zero.".to_string(),
        ));
    }

    let n = a.len() - 1;
    let m = b.len().saturating_sub(1);
    let k = m.max(n);
    if k == 0 {
        return Ok(Vec::new());
    }

    let mut x_hist = x.map_or_else(|| vec![0.0; m], ToOwned::to_owned);
    if x_hist.len() < m {
        x_hist.resize(m, 0.0);
    }

    let mut y_hist = y.to_vec();
    if y_hist.len() < n {
        y_hist.resize(n, 0.0);
    }

    let mut zi = vec![0.0; k];
    for state in 0..m {
        let span = m - state;
        zi[state] = b[state + 1..]
            .iter()
            .take(span)
            .zip(x_hist.iter())
            .map(|(&coeff, &sample)| coeff * sample)
            .sum();
    }
    for state in 0..n {
        let span = n - state;
        zi[state] -= a[state + 1..]
            .iter()
            .take(span)
            .zip(y_hist.iter())
            .map(|(&coeff, &sample)| coeff * sample)
            .sum::<f64>();
    }

    if a[0] != 1.0 {
        for value in &mut zi {
            *value /= a[0];
        }
    }

    Ok(zi)
}

/// Apply a digital IIR filter using the Direct Form II transposed structure.
///
/// Matches `scipy.signal.lfilter(b, a, x, zi=zi)`.
pub fn lfilter(
    b: &[f64],
    a: &[f64],
    x: &[f64],
    zi: Option<&[f64]>,
) -> Result<Vec<f64>, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }
    if a[0] == 0.0 {
        return Err(SignalError::InvalidArgument(
            "a[0] must not be zero".to_string(),
        ));
    }

    let a0 = a[0];
    let nb = b.len();
    let na = a.len();
    let nfilt = nb.max(na);

    // Normalize coefficients
    let b_norm: Vec<f64> = b.iter().map(|&v| v / a0).collect();
    let a_norm: Vec<f64> = a.iter().map(|&v| v / a0).collect();

    // Direct Form II transposed
    let mut y = Vec::with_capacity(x.len());
    let mut d = if let Some(initial) = zi {
        if initial.len() != nfilt - 1 {
            return Err(SignalError::InvalidArgument(format!(
                "zi length ({}) must be nfilt-1 ({})",
                initial.len(),
                nfilt - 1
            )));
        }
        let mut d_init = vec![0.0; nfilt];
        d_init[..nfilt - 1].copy_from_slice(initial);
        d_init
    } else {
        vec![0.0; nfilt]
    };

    for &xi in x {
        let yi = b_norm.first().copied().unwrap_or(0.0) * xi + d[0];
        y.push(yi);

        // Update delay line
        for j in 0..nfilt - 1 {
            let bj = if j + 1 < nb { b_norm[j + 1] } else { 0.0 };
            let aj = if j + 1 < na { a_norm[j + 1] } else { 0.0 };
            d[j] = bj * xi - aj * yi + if j + 1 < nfilt - 1 { d[j + 1] } else { 0.0 };
        }
    }

    Ok(y)
}

/// Apply a digital filter forward and backward (zero-phase filtering).
///
/// Matches `scipy.signal.filtfilt(b, a, x)`.
///
/// This results in zero phase distortion (linear phase) and doubled filter order.
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Result<Vec<f64>, SignalError> {
    filtfilt_with_padtype(b, a, x, None)
}

/// Apply a digital filter forward and backward with explicit SciPy padtype control.
///
/// Matches `scipy.signal.filtfilt(b, a, x, padtype=...)` for
/// `padtype in {"odd", "even", "constant"}`.
pub fn filtfilt_with_padtype(
    b: &[f64],
    a: &[f64],
    x: &[f64],
    padtype: Option<&str>,
) -> Result<Vec<f64>, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }
    let n = x.len();
    if n < 3 {
        return Err(SignalError::InvalidArgument(
            "input must have length >= 3 for filtfilt".to_string(),
        ));
    }

    let nfilt = b.len().max(a.len());
    let padlen = (3 * (nfilt - 1)).min(n - 1);
    let padtype = padtype.unwrap_or("odd");

    // 1. Pad signal with mirrored values
    let mut padded = Vec::with_capacity(n + 2 * padlen);
    let x0 = x[0];
    match padtype {
        "odd" => {
            for i in (1..=padlen).rev() {
                padded.push(2.0 * x0 - x[i]);
            }
        }
        "even" => {
            for i in (1..=padlen).rev() {
                padded.push(x[i]);
            }
        }
        "constant" => {
            for _ in 0..padlen {
                padded.push(x0);
            }
        }
        _ => {
            return Err(SignalError::InvalidArgument(
                "padtype must be one of {'odd', 'even', 'constant'}".to_string(),
            ));
        }
    }
    padded.extend_from_slice(x);
    let xn = x[n - 1];
    match padtype {
        "odd" => {
            for i in 1..=padlen {
                padded.push(2.0 * xn - x[n - 1 - i]);
            }
        }
        "even" => {
            for i in 1..=padlen {
                padded.push(x[n - 1 - i]);
            }
        }
        "constant" => {
            for _ in 0..padlen {
                padded.push(xn);
            }
        }
        _ => unreachable!("padtype validated above"),
    }

    // Forward pass with initial conditions
    let zi = lfilter_zi(b, a)?;
    let zi_f: Vec<f64> = zi.iter().map(|&v| v * padded[0]).collect();
    let mut forward = lfilter(b, a, &padded, Some(&zi_f))?;

    // Reverse
    forward.reverse();

    // Backward pass with initial conditions
    let zi_b: Vec<f64> = zi.iter().map(|&v| v * forward[0]).collect();
    let mut reversed = lfilter(b, a, &forward, Some(&zi_b))?;

    // Reverse again and remove padding
    reversed.reverse();
    let result = reversed[padlen..padlen + n].to_vec();
    Ok(result)
}

/// Apply a digital filter in SOS (second-order sections) form.
///
/// Matches `scipy.signal.sosfilt(sos, x)`.
///
/// Each section is a biquad [b0, b1, b2, a0, a1, a2] applied in cascade
/// using Direct Form II transposed. This is numerically superior to
/// lfilter for high-order filters.
pub fn sosfilt(sos: &[SosSection], x: &[f64]) -> Result<Vec<f64>, SignalError> {
    if sos.is_empty() {
        return Err(SignalError::InvalidArgument(
            "sos must not be empty".to_string(),
        ));
    }

    let mut signal = x.to_vec();

    for section in sos {
        let b = [section[0], section[1], section[2]];
        let a0 = section[3];
        let a1 = section[4];
        let a2 = section[5];

        if a0.abs() < 1e-30 {
            return Err(SignalError::InvalidArgument(
                "SOS section a[0] must not be zero".to_string(),
            ));
        }

        // Normalize by a0
        let b0 = b[0] / a0;
        let b1 = b[1] / a0;
        let b2 = b[2] / a0;
        let a1n = a1 / a0;
        let a2n = a2 / a0;

        // Direct Form II transposed for this biquad section
        let mut d1 = 0.0;
        let mut d2 = 0.0;

        for sample in &mut signal {
            let xi = *sample;
            let yi = b0 * xi + d1;
            d1 = b1 * xi - a1n * yi + d2;
            d2 = b2 * xi - a2n * yi;
            *sample = yi;
        }
    }

    Ok(signal)
}

/// Apply SOS filter forward and backward for zero-phase filtering.
///
/// Matches `scipy.signal.sosfiltfilt(sos, x)`.
///
/// Results in zero phase distortion and doubled filter order.
/// More numerically stable than filtfilt for high-order filters.
pub fn sosfiltfilt(sos: &[SosSection], x: &[f64]) -> Result<Vec<f64>, SignalError> {
    let n = x.len();
    if n < 3 {
        return Err(SignalError::InvalidArgument(
            "input must have length >= 3 for sosfiltfilt".to_string(),
        ));
    }

    // Determine padding length (matches SciPy's default: 3 * (ntaps - 1))
    // For SOS, ntaps = 2 * n_sections + 1. So 3 * (2 * n_sections) = 6 * n_sections.
    let padlen = (6 * sos.len()).min(n - 1);

    // 1. Pad signal with mirrored values:
    // [2*x[0]-x[padlen], ..., 2*x[0]-x[1], x[0], ..., x[n-1], 2*x[n-1]-x[n-2], ..., 2*x[n-1]-x[n-padlen-1]]
    let mut padded = Vec::with_capacity(n + 2 * padlen);
    let x0 = x[0];
    for i in (1..=padlen).rev() {
        padded.push(2.0 * x0 - x[i]);
    }
    padded.extend_from_slice(x);
    let xn = x[n - 1];
    for i in 1..=padlen {
        padded.push(2.0 * xn - x[n - 1 - i]);
    }

    // 2. Initialize filter with steady-state for the first padded sample
    let zi = sosfilt_zi(sos)?;
    let mut z_forward = zi.clone();
    let p0 = padded[0];
    for z in &mut z_forward {
        z[0] *= p0;
        z[1] *= p0;
    }

    // Forward pass with initial conditions
    let mut signal = padded;
    sosfilt_in_place(sos, &mut signal, &z_forward)?;

    // 3. Reverse and repeat for backward pass
    signal.reverse();
    let mut z_backward = zi;
    let s0 = signal[0];
    for z in &mut z_backward {
        z[0] *= s0;
        z[1] *= s0;
    }
    sosfilt_in_place(sos, &mut signal, &z_backward)?;

    // 4. Reverse again and remove padding
    signal.reverse();
    let result = signal[padlen..padlen + n].to_vec();
    Ok(result)
}

/// Internal helper for in-place SOS filtering with initial conditions.
fn sosfilt_in_place(sos: &[SosSection], x: &mut [f64], zi: &[[f64; 2]]) -> Result<(), SignalError> {
    for (i, section) in sos.iter().enumerate() {
        let b0 = section[0] / section[3];
        let b1 = section[1] / section[3];
        let b2 = section[2] / section[3];
        let a1 = section[4] / section[3];
        let a2 = section[5] / section[3];

        let mut d1 = zi[i][0];
        let mut d2 = zi[i][1];

        for sample in x.iter_mut() {
            let xi = *sample;
            let yi = b0 * xi + d1;
            d1 = b1 * xi - a1 * yi + d2;
            d2 = b2 * xi - a2 * yi;
            *sample = yi;
        }
    }
    Ok(())
}

/// Compute initial conditions for SOS filter for step response steady-state.
///
/// Matches `scipy.signal.sosfilt_zi(sos)`.
///
/// Returns initial delay values such that filtering a constant signal
/// produces a constant output (no transient). Each section gets 2 initial
/// conditions [d1, d2].
pub fn sosfilt_zi(sos: &[SosSection]) -> Result<Vec<[f64; 2]>, SignalError> {
    if sos.is_empty() {
        return Err(SignalError::InvalidArgument(
            "sos must not be empty".to_string(),
        ));
    }

    let mut zi = Vec::with_capacity(sos.len());

    for section in sos {
        let a0 = section[3];
        if a0.abs() < 1e-30 {
            return Err(SignalError::InvalidArgument(
                "SOS section a[0] must not be zero".to_string(),
            ));
        }
        let b0 = section[0] / a0;
        let b1 = section[1] / a0;
        let b2 = section[2] / a0;
        let a1 = section[4] / a0;
        let a2 = section[5] / a0;

        // For steady-state with input=1, output=gain of section at DC:
        // gain = (b0+b1+b2) / (1+a1+a2)
        // Initial conditions satisfy: d1 = b1 - a1*gain + d2, d2 = b2 - a2*gain
        // Solving: d2 = b2 - a2*gain, d1 = (b1 - a1*gain) + d2
        let dc_denom = 1.0 + a1 + a2;
        if dc_denom.abs() < 1e-30 {
            zi.push([0.0, 0.0]);
            continue;
        }
        let gain = (b0 + b1 + b2) / dc_denom;
        let d2 = b2 - a2 * gain;
        let d1 = b1 - a1 * gain + d2;
        zi.push([d1, d2]);
    }

    Ok(zi)
}

// ── IIR helper functions ────────────────────────────────────────

/// Multiply polynomial by (1 + c*z^-1).
fn poly_mul_binomial(poly: &[f64], c: f64) -> Vec<f64> {
    let mut result = vec![0.0; poly.len() + 1];
    for (i, &p) in poly.iter().enumerate() {
        result[i] += p;
        result[i + 1] += p * c;
    }
    result
}

/// Multiply polynomial by (1 + b*z^-1 + c*z^-2).
fn poly_mul_quadratic(poly: &[f64], b: f64, c: f64) -> Vec<f64> {
    let mut result = vec![0.0; poly.len() + 2];
    for (i, &p) in poly.iter().enumerate() {
        result[i] += p;
        result[i + 1] += p * b;
        result[i + 2] += p * c;
    }
    result
}

// ══════════════════════════════════════════════════════════════════════
// Filter Representation Conversions
// ══════════════════════════════════════════════════════════════════════

/// Zero-pole-gain representation of a filter.
#[derive(Debug, Clone, PartialEq)]
pub struct ZpkCoeffs {
    /// Zeros (complex: pairs of (re, im)).
    pub zeros_re: Vec<f64>,
    pub zeros_im: Vec<f64>,
    /// Poles (complex: pairs of (re, im)).
    pub poles_re: Vec<f64>,
    pub poles_im: Vec<f64>,
    /// Scalar gain.
    pub gain: f64,
}

/// A single second-order section: [b0, b1, b2, a0, a1, a2].
/// Represents H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2).
pub type SosSection = [f64; 6];

/// Convert transfer function (b, a) to zero-pole-gain form.
///
/// Matches `scipy.signal.tf2zpk(b, a)`.
///
/// Finds zeros (roots of b) and poles (roots of a) via companion matrix eigenvalues.
pub fn tf2zpk(b: &[f64], a: &[f64]) -> Result<ZpkCoeffs, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }
    if a[0] == 0.0 {
        return Err(SignalError::InvalidArgument(
            "a[0] must not be zero".to_string(),
        ));
    }

    // Find leading non-zero coefficient of b (effective degree may differ from length)
    let b_lead_idx = match b.iter().position(|&v| v.abs() > 1e-30) {
        Some(idx) => idx,
        None => {
            // All coefficients are essentially zero — zero polynomial
            // Find poles from a but no zeros, gain = 0
            let a_norm: Vec<f64> = a.iter().map(|&v| v / a[0]).collect();
            let (poles_re, poles_im) = if a_norm.len() > 1 {
                poly_roots(&a_norm)?
            } else {
                (vec![], vec![])
            };
            return Ok(ZpkCoeffs {
                zeros_re: vec![],
                zeros_im: vec![],
                poles_re,
                poles_im,
                gain: 0.0,
            });
        }
    };
    let b_lead = b[b_lead_idx];

    let gain = b_lead / a[0];

    // Normalize: divide by leading non-zero coefficient
    let b_effective = &b[b_lead_idx..];
    let b_norm: Vec<f64> = b_effective.iter().map(|&v| v / b_lead).collect();
    let a_norm: Vec<f64> = a.iter().map(|&v| v / a[0]).collect();

    // Find roots via companion matrix eigenvalues
    let (zeros_re, zeros_im) = if b_norm.len() > 1 {
        poly_roots(&b_norm)?
    } else {
        (vec![], vec![])
    };

    let (poles_re, poles_im) = if a_norm.len() > 1 {
        poly_roots(&a_norm)?
    } else {
        (vec![], vec![])
    };

    Ok(ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain,
    })
}

/// Convert zero-pole-gain to transfer function (b, a) form.
///
/// Matches `scipy.signal.zpk2tf(z, p, k)`.
pub fn zpk2tf(zpk: &ZpkCoeffs) -> BaCoeffs {
    let b = poly_from_complex_roots_full(&zpk.zeros_re, &zpk.zeros_im);
    let a = poly_from_complex_roots_full(&zpk.poles_re, &zpk.poles_im);

    // Scale numerator by gain
    let b: Vec<f64> = b.iter().map(|&v| v * zpk.gain).collect();

    BaCoeffs { b, a }
}

/// Convert transfer function to second-order sections.
///
/// Matches `scipy.signal.tf2sos(b, a)`.
///
/// The SOS form is numerically superior for high-order filters.
pub fn tf2sos(b: &[f64], a: &[f64]) -> Result<Vec<SosSection>, SignalError> {
    let zpk = tf2zpk(b, a)?;
    Ok(zpk2sos(&zpk))
}

/// Convert second-order sections to transfer function.
///
/// Matches `scipy.signal.sos2tf(sos)`.
pub fn sos2tf(sos: &[SosSection]) -> BaCoeffs {
    if sos.is_empty() {
        return BaCoeffs {
            b: vec![1.0],
            a: vec![1.0],
        };
    }

    let mut b = vec![1.0];
    let mut a = vec![1.0];

    for section in sos {
        // Numerator: [b0, b1, b2]
        let sec_b = [section[0], section[1], section[2]];
        // Denominator: [a0, a1, a2]
        let sec_a = [section[3], section[4], section[5]];

        b = poly_multiply(&b, &sec_b);
        a = poly_multiply(&a, &sec_a);
    }

    BaCoeffs { b, a }
}

/// Convert zero-pole-gain to second-order sections.
///
/// Matches `scipy.signal.zpk2sos(z, p, k)`.
///
/// Pairs poles and zeros into biquad sections, sorted by proximity to
/// the unit circle (outermost poles last for numerical stability).
pub fn zpk2sos(zpk: &ZpkCoeffs) -> Vec<SosSection> {
    let n_poles = zpk.poles_re.len();
    let n_zeros = zpk.zeros_re.len();

    if n_poles == 0 && n_zeros == 0 {
        return vec![[zpk.gain, 0.0, 0.0, 1.0, 0.0, 0.0]];
    }

    // Build denominator sections from poles
    let denom_sections = roots_to_sos_factors(&zpk.poles_re, &zpk.poles_im);
    // Build numerator sections from zeros
    let numer_sections = roots_to_sos_factors(&zpk.zeros_re, &zpk.zeros_im);

    let n_sections = denom_sections.len().max(numer_sections.len()).max(1);
    let mut sections: Vec<SosSection> = Vec::with_capacity(n_sections);

    for i in 0..n_sections {
        let (b0, b1, b2) = if i < numer_sections.len() {
            numer_sections[i]
        } else {
            (1.0, 0.0, 0.0)
        };
        let (a0, a1, a2) = if i < denom_sections.len() {
            denom_sections[i]
        } else {
            (1.0, 0.0, 0.0)
        };
        sections.push([b0, b1, b2, a0, a1, a2]);
    }

    if sections.is_empty() {
        sections.push([zpk.gain, 0.0, 0.0, 1.0, 0.0, 0.0]);
    } else {
        // Apply gain to first section
        sections[0][0] *= zpk.gain;
        sections[0][1] *= zpk.gain;
        sections[0][2] *= zpk.gain;
    }

    sections
}

/// Convert roots to second-order section factors.
/// Groups conjugate pairs and pairs of real roots into (1, c1, c2) tuples.
fn roots_to_sos_factors(re: &[f64], im: &[f64]) -> Vec<(f64, f64, f64)> {
    let n = re.len();
    let mut used = vec![false; n];
    let mut factors = Vec::new();
    let mut real_roots: Vec<f64> = Vec::new();

    // First: pair complex conjugates
    for i in 0..n {
        if used[i] {
            continue;
        }
        if im[i].abs() < 1e-14 {
            real_roots.push(re[i]);
            used[i] = true;
        } else {
            // Find conjugate
            let mut found = false;
            for j in (i + 1)..n {
                if !used[j] && (re[j] - re[i]).abs() < 1e-10 && (im[j] + im[i]).abs() < 1e-10 {
                    // Conjugate pair: (1 - 2*re*z^-1 + |r|²*z^-2)
                    let mag2 = re[i] * re[i] + im[i] * im[i];
                    factors.push((1.0, -2.0 * re[i], mag2));
                    used[j] = true;
                    found = true;
                    break;
                }
            }
            if !found {
                let mag2 = re[i] * re[i] + im[i] * im[i];
                factors.push((1.0, -2.0 * re[i], mag2));
            }
            used[i] = true;
        }
    }

    // Pair real roots into biquad sections
    real_roots.sort_by(|a, b| a.total_cmp(b));
    let mut i = 0;
    while i + 1 < real_roots.len() {
        let r1 = real_roots[i];
        let r2 = real_roots[i + 1];
        // (1 - r1*z^-1)(1 - r2*z^-1) = 1 - (r1+r2)*z^-1 + r1*r2*z^-2
        factors.push((1.0, -(r1 + r2), r1 * r2));
        i += 2;
    }
    if i < real_roots.len() {
        // Odd real root out: single first-order section
        factors.push((1.0, -real_roots[i], 0.0));
    }

    factors
}

/// Convert second-order sections to zero-pole-gain form.
///
/// Matches `scipy.signal.sos2zpk(sos)`.
pub fn sos2zpk(sos: &[SosSection]) -> ZpkCoeffs {
    let mut all_zeros_re = Vec::new();
    let mut all_zeros_im = Vec::new();
    let mut all_poles_re = Vec::new();
    let mut all_poles_im = Vec::new();
    let mut gain = 1.0;

    for section in sos {
        let sec_b = [section[0], section[1], section[2]];
        let sec_a = [section[3], section[4], section[5]];

        if sec_a[0].abs() > 1e-30 {
            gain *= if sec_b[0].abs() > 1e-30 {
                sec_b[0] / sec_a[0]
            } else {
                0.0
            };
        }

        // Normalize and find roots (skip if leading coeff is zero)
        let b_norm: Vec<f64> = if sec_b[0].abs() > 1e-30 {
            sec_b.iter().map(|&v| v / sec_b[0]).collect()
        } else {
            continue; // degenerate section, skip
        };
        let a_norm: Vec<f64> = if sec_a[0].abs() > 1e-30 {
            sec_a.iter().map(|&v| v / sec_a[0]).collect()
        } else {
            continue;
        };

        // Trim trailing zeros for root finding
        let b_trim = trim_trailing_zeros(&b_norm);
        let a_trim = trim_trailing_zeros(&a_norm);

        if b_trim.len() > 1
            && let Ok((zr, zi)) = poly_roots(&b_trim)
        {
            all_zeros_re.extend(zr);
            all_zeros_im.extend(zi);
        }
        if a_trim.len() > 1
            && let Ok((pr, pi)) = poly_roots(&a_trim)
        {
            all_poles_re.extend(pr);
            all_poles_im.extend(pi);
        }
    }

    ZpkCoeffs {
        zeros_re: all_zeros_re,
        zeros_im: all_zeros_im,
        poles_re: all_poles_re,
        poles_im: all_poles_im,
        gain,
    }
}

// ── Filter conversion helpers ───────────────────────────────

/// Find roots of a monic polynomial using companion matrix eigenvalues.
/// Input: coefficients [1, c1, c2, ...] of p(x) = x^n + c1*x^{n-1} + ... + cn.
fn poly_roots(coeffs: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    let n = coeffs.len() - 1; // degree
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if n == 1 {
        // Linear: x + c1 = 0 → x = -c1
        return Ok((vec![-coeffs[1] / coeffs[0]], vec![0.0]));
    }
    if n == 2 {
        // Quadratic: use formula
        let a = coeffs[0];
        let b = coeffs[1];
        let c = coeffs[2];
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            return Ok((
                vec![(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)],
                vec![0.0, 0.0],
            ));
        } else {
            let sq = (-disc).sqrt();
            return Ok((
                vec![-b / (2.0 * a), -b / (2.0 * a)],
                vec![sq / (2.0 * a), -sq / (2.0 * a)],
            ));
        }
    }

    // General case: companion matrix eigenvalues
    // Companion matrix for p(x) = x^n + c1*x^{n-1} + ... + cn
    // has -c_i/c_0 in the last column and 1's on the subdiagonal
    let mut companion = vec![vec![0.0; n]; n];
    for i in 1..n {
        companion[i][i - 1] = 1.0;
    }
    for i in 0..n {
        companion[i][n - 1] = -coeffs[n - i] / coeffs[0];
    }

    // Use fsci-linalg eig to get eigenvalues
    let opts = fsci_linalg::DecompOptions::default();
    let eig_result = fsci_linalg::eig(&companion, opts)
        .map_err(|e| SignalError::InvalidArgument(format!("eigenvalue computation failed: {e}")))?;

    Ok((eig_result.eigenvalues_re, eig_result.eigenvalues_im))
}

/// Build polynomial coefficients in z^-1 convention from complex roots.
/// Returns [1, c1, c2, ...] where P(z) = (1 - r1*z^-1)(1 - r2*z^-1)...
/// This is the transfer function convention: b[0] + b[1]*z^-1 + b[2]*z^-2 + ...
fn poly_from_complex_roots_full(roots_re: &[f64], roots_im: &[f64]) -> Vec<f64> {
    if roots_re.is_empty() {
        return vec![1.0];
    }

    let n = roots_re.len();
    let mut used = vec![false; n];
    let mut poly = vec![1.0];

    for i in 0..n {
        if used[i] {
            continue;
        }

        if roots_im[i].abs() < 1e-14 {
            // Real root: multiply by (1 - r*z^-1)
            poly = poly_mul_binomial(&poly, -roots_re[i]);
            used[i] = true;
        } else {
            // Find conjugate pair
            let mut found = false;
            for j in (i + 1)..n {
                if !used[j]
                    && (roots_re[j] - roots_re[i]).abs() < 1e-10
                    && (roots_im[j] + roots_im[i]).abs() < 1e-10
                {
                    let two_re = 2.0 * roots_re[i];
                    let mag2 = roots_re[i] * roots_re[i] + roots_im[i] * roots_im[i];
                    poly = poly_mul_quadratic(&poly, -two_re, mag2);
                    used[j] = true;
                    found = true;
                    break;
                }
            }
            if !found {
                // Unpaired: treat as conjugate pair with itself
                let two_re = 2.0 * roots_re[i];
                let mag2 = roots_re[i] * roots_re[i] + roots_im[i] * roots_im[i];
                poly = poly_mul_quadratic(&poly, -two_re, mag2);
            }
            used[i] = true;
        }
    }

    poly
}

/// Multiply two polynomials (convolution of coefficient arrays).
fn poly_multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut result = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Remove trailing near-zero coefficients from a polynomial.
fn trim_trailing_zeros(p: &[f64]) -> Vec<f64> {
    let mut end = p.len();
    while end > 1 && p[end - 1].abs() < 1e-15 {
        end -= 1;
    }
    p[..end].to_vec()
}

// ══════════════════════════════════════════════════════════════════════
// Frequency Response Analysis
// ══════════════════════════════════════════════════════════════════════

/// Complex frequency response at a set of frequencies.
#[derive(Debug, Clone, PartialEq)]
pub struct FreqzResult {
    /// Angular frequencies (radians/sample for digital, rad/s for analog).
    pub w: Vec<f64>,
    /// Complex frequency response: magnitude at each frequency.
    pub h_mag: Vec<f64>,
    /// Phase response (radians) at each frequency.
    pub h_phase: Vec<f64>,
}

/// Compute the frequency response of a digital filter.
///
/// Matches `scipy.signal.freqz(b, a, worN)`.
///
/// Evaluates H(e^{jω}) = B(e^{jω}) / A(e^{jω}) at `n_freqs` equally
/// spaced frequencies from 0 to π.
///
/// # Arguments
/// * `b` — Numerator coefficients
/// * `a` — Denominator coefficients
/// * `n_freqs` — Number of frequency points (default 512)
pub fn freqz(b: &[f64], a: &[f64], n_freqs: Option<usize>) -> Result<FreqzResult, SignalError> {
    freqz_with_whole(b, a, n_freqs, false)
}

/// Compute the frequency response of a digital filter over half or the whole unit circle.
///
/// Matches `scipy.signal.freqz(b, a, worN, whole=...)`.
///
/// When `whole` is false, evaluates on `[0, π)`. When `whole` is true,
/// evaluates on `[0, 2π)` with SciPy's `endpoint=False` spacing.
pub fn freqz_with_whole(
    b: &[f64],
    a: &[f64],
    n_freqs: Option<usize>,
    whole: bool,
) -> Result<FreqzResult, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }

    let n = n_freqs.unwrap_or(512);
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "n_freqs must be > 0".to_string(),
        ));
    }

    let mut w = Vec::with_capacity(n);
    let mut h_mag = Vec::with_capacity(n);
    let mut h_phase = Vec::with_capacity(n);

    for i in 0..n {
        // scipy uses endpoint=False:
        // half spectrum: w = np.linspace(0, pi, n, endpoint=False)
        // whole spectrum: w = np.linspace(0, 2*pi, n, endpoint=False)
        let omega = if whole {
            std::f64::consts::TAU * i as f64 / n as f64
        } else {
            std::f64::consts::PI * i as f64 / n as f64
        };
        w.push(omega);

        // Evaluate B(e^{jω}) and A(e^{jω}) using Horner's method
        // B(z) = b[0] + b[1]*z^{-1} + b[2]*z^{-2} + ...
        // At z = e^{jω}: z^{-k} = e^{-jkω} = cos(kω) - j*sin(kω)
        let (b_re, b_im) = eval_poly_on_unit_circle(b, omega);
        let (a_re, a_im) = eval_poly_on_unit_circle(a, omega);

        // H = B / A (complex division)
        let denom = a_re * a_re + a_im * a_im;
        if denom < 1e-30 {
            h_mag.push(f64::INFINITY);
            h_phase.push(0.0);
        } else {
            let h_re = (b_re * a_re + b_im * a_im) / denom;
            let h_im = (b_im * a_re - b_re * a_im) / denom;
            h_mag.push((h_re * h_re + h_im * h_im).sqrt());
            h_phase.push(h_im.atan2(h_re));
        }
    }

    Ok(FreqzResult { w, h_mag, h_phase })
}

/// Compute the frequency response of a digital filter from SOS form.
///
/// Matches `scipy.signal.sosfreqz(sos, worN)`.
///
/// More numerically stable than freqz for high-order filters.
pub fn freqz_sos(sos: &[SosSection], n_freqs: Option<usize>) -> Result<FreqzResult, SignalError> {
    if sos.is_empty() {
        return Err(SignalError::InvalidArgument(
            "sos must not be empty".to_string(),
        ));
    }

    let n = n_freqs.unwrap_or(512);
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "n_freqs must be > 0".to_string(),
        ));
    }

    let mut w = Vec::with_capacity(n);
    let mut h_mag = Vec::with_capacity(n);
    let mut h_phase = Vec::with_capacity(n);

    for i in 0..n {
        // Match scipy endpoint=False: w[i] = i * pi / n
        let omega = std::f64::consts::PI * i as f64 / n as f64;
        w.push(omega);

        // Multiply frequency responses of each section
        let mut total_re = 1.0;
        let mut total_im = 0.0;

        for section in sos {
            let sec_b = &[section[0], section[1], section[2]];
            let sec_a = &[section[3], section[4], section[5]];

            let (b_re, b_im) = eval_poly_on_unit_circle(sec_b, omega);
            let (a_re, a_im) = eval_poly_on_unit_circle(sec_a, omega);

            let denom = a_re * a_re + a_im * a_im;
            if denom < 1e-30 {
                total_re = f64::INFINITY;
                total_im = 0.0;
                break;
            }

            let h_re = (b_re * a_re + b_im * a_im) / denom;
            let h_im = (b_im * a_re - b_re * a_im) / denom;

            // Complex multiply: total *= h
            let new_re = total_re * h_re - total_im * h_im;
            let new_im = total_re * h_im + total_im * h_re;
            total_re = new_re;
            total_im = new_im;
        }

        h_mag.push((total_re * total_re + total_im * total_im).sqrt());
        h_phase.push(total_im.atan2(total_re));
    }

    Ok(FreqzResult { w, h_mag, h_phase })
}

/// Compute the frequency response of a digital filter in SOS format.
///
/// Alias for `freqz_sos`. Matches `scipy.signal.sosfreqz(sos, worN)`.
pub fn sosfreqz(sos: &[SosSection], n_freqs: Option<usize>) -> Result<FreqzResult, SignalError> {
    freqz_sos(sos, n_freqs)
}

/// Compute the frequency response of a digital filter in SOS format over
/// half or the whole unit circle.
///
/// Matches `scipy.signal.sosfreqz(sos, worN, whole=...)`.
///
/// When `whole` is false, evaluates on `[0, π)`. When `whole` is true,
/// evaluates on `[0, 2π)` with scipy's `endpoint=False` spacing.
pub fn sosfreqz_with_whole(
    sos: &[SosSection],
    n_freqs: Option<usize>,
    whole: bool,
) -> Result<FreqzResult, SignalError> {
    if sos.is_empty() {
        return Err(SignalError::InvalidArgument(
            "sos must not be empty".to_string(),
        ));
    }

    let n = n_freqs.unwrap_or(512);
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "n_freqs must be > 0".to_string(),
        ));
    }

    let mut w = Vec::with_capacity(n);
    let mut h_mag = Vec::with_capacity(n);
    let mut h_phase = Vec::with_capacity(n);

    for i in 0..n {
        let omega = if whole {
            std::f64::consts::TAU * i as f64 / n as f64
        } else {
            std::f64::consts::PI * i as f64 / n as f64
        };
        w.push(omega);

        let mut total_re = 1.0;
        let mut total_im = 0.0;

        for section in sos {
            let sec_b = &[section[0], section[1], section[2]];
            let sec_a = &[section[3], section[4], section[5]];

            let (b_re, b_im) = eval_poly_on_unit_circle(sec_b, omega);
            let (a_re, a_im) = eval_poly_on_unit_circle(sec_a, omega);

            let denom = a_re * a_re + a_im * a_im;
            if denom < 1e-30 {
                total_re = f64::INFINITY;
                total_im = 0.0;
                break;
            }

            let h_re = (b_re * a_re + b_im * a_im) / denom;
            let h_im = (b_im * a_re - b_re * a_im) / denom;

            let new_re = total_re * h_re - total_im * h_im;
            let new_im = total_re * h_im + total_im * h_re;
            total_re = new_re;
            total_im = new_im;
        }

        h_mag.push((total_re * total_re + total_im * total_im).sqrt());
        h_phase.push(total_im.atan2(total_re));
    }

    Ok(FreqzResult { w, h_mag, h_phase })
}

/// Compute the frequency response of an analog filter.
///
/// Matches `scipy.signal.freqs(b, a, worN)`.
///
/// Evaluates H(jω) = B(jω) / A(jω) at specified angular frequencies.
pub fn freqs(b: &[f64], a: &[f64], w: &[f64]) -> Result<FreqzResult, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }
    if w.is_empty() {
        return Err(SignalError::InvalidArgument(
            "w must not be empty".to_string(),
        ));
    }

    let mut h_mag = Vec::with_capacity(w.len());
    let mut h_phase = Vec::with_capacity(w.len());

    for &omega in w {
        // Evaluate B(jω) and A(jω)
        // B(s) = b[0]*s^n + b[1]*s^{n-1} + ... + b[n]
        // At s = jω: (jω)^k = (jω)^k computed via complex arithmetic
        let (b_re, b_im) = eval_analog_poly(b, omega);
        let (a_re, a_im) = eval_analog_poly(a, omega);

        let denom = a_re * a_re + a_im * a_im;
        if denom < 1e-30 {
            h_mag.push(f64::INFINITY);
            h_phase.push(0.0);
        } else {
            let h_re = (b_re * a_re + b_im * a_im) / denom;
            let h_im = (b_im * a_re - b_re * a_im) / denom;
            h_mag.push((h_re * h_re + h_im * h_im).sqrt());
            h_phase.push(h_im.atan2(h_re));
        }
    }

    Ok(FreqzResult {
        w: w.to_vec(),
        h_mag,
        h_phase,
    })
}

/// Compute group delay of a digital filter.
///
/// Matches `scipy.signal.group_delay((b, a), w)`.
///
/// Group delay: τ_g(ω) = -dφ/dω where φ is the phase response.
///
/// Uses the analytic formula based on polynomial evaluation:
///   τ_g(ω) = Re{ [B'(z)/B(z) - A'(z)/A(z)] } where z = e^{jω}
/// and B'(z) = d/dω B(e^{jω}) (the derivative polynomial evaluated on the unit circle).
pub fn group_delay(
    b: &[f64],
    a: &[f64],
    n_freqs: Option<usize>,
) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }

    let n = n_freqs.unwrap_or(512);
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "n_freqs must be > 0".to_string(),
        ));
    }

    // Derivative polynomial: if B(z) = Σ b[k] z^{-k}, then
    // dB/dω = Σ (-jk) b[k] z^{-k}, so the "derivative coefficients" are
    // b_deriv[k] = -k * b[k] and we multiply by j.
    //
    // Group delay = Re{ (B'·conj(B))/(|B|²) - (A'·conj(A))/(|A|²) }
    // where B' = Σ (-jk) b[k] e^{-jkω}

    let mut w_out = Vec::with_capacity(n);
    let mut gd_out = Vec::with_capacity(n);

    for i in 0..n {
        // Match scipy endpoint=False: w[i] = i * pi / n
        let omega = std::f64::consts::PI * i as f64 / n as f64;
        w_out.push(omega);
        gd_out.push(group_delay_at_frequency(b, a, omega));
    }

    Ok((w_out, gd_out))
}

/// Compute phase delay of a digital filter.
///
/// Matches the core SciPy surface of `scipy.signal.phase_delay((b, a), w)`.
///
/// Phase delay: τ_p(ω) = -φ(ω) / ω, where φ is the unwrapped phase response.
/// At ω = 0 the limit is taken from the analytic group delay.
pub fn phase_delay(
    b: &[f64],
    a: &[f64],
    n_freqs: Option<usize>,
) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    let response = freqz(b, a, n_freqs)?;
    let unwrapped_phase = unwrap_phase(&response.h_phase);
    let mut pd_out = Vec::with_capacity(response.w.len());

    for ((&omega, &mag), &phase) in response
        .w
        .iter()
        .zip(response.h_mag.iter())
        .zip(unwrapped_phase.iter())
    {
        let delay = if omega.abs() < 1e-14 {
            group_delay_at_frequency(b, a, omega)
        } else if mag <= 1e-30 {
            0.0
        } else {
            -phase / omega
        };
        pd_out.push(delay);
    }

    Ok((response.w, pd_out))
}

fn group_delay_at_frequency(b: &[f64], a: &[f64], omega: f64) -> f64 {
    // Evaluate B(e^{jω})
    let (b_re, b_im) = eval_poly_on_unit_circle(b, omega);
    let b_mag2 = b_re * b_re + b_im * b_im;

    // Standard formula: τ_g(ω) = Re{D_B / B} - Re{D_A / A}
    // where D_B(e^{jω}) = Σ_k k * b[k] * e^{-jkω}
    // and Re{D/B} = (D_re*B_re + D_im*B_im) / |B|²
    let (db_re, db_im) = eval_weighted_poly_on_unit_circle(b, omega);
    let gd_b = if b_mag2 > 1e-30 {
        (db_re * b_re + db_im * b_im) / b_mag2
    } else {
        0.0
    };

    // For A: τ_A = Re{Σ k*a[k]*e^{-jkω} / A(e^{jω})}
    let (a_re, a_im) = eval_poly_on_unit_circle(a, omega);
    let a_mag2 = a_re * a_re + a_im * a_im;
    let (da_re, da_im) = eval_weighted_poly_on_unit_circle(a, omega);
    let gd_a = if a_mag2 > 1e-30 {
        (da_re * a_re + da_im * a_im) / a_mag2
    } else {
        0.0
    };

    gd_b - gd_a
}

/// Evaluate Σ k*c[k]*e^{-jkω} (weighted polynomial for group delay).
fn eval_weighted_poly_on_unit_circle(coeffs: &[f64], omega: f64) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for (k, &c) in coeffs.iter().enumerate() {
        let angle = -(k as f64) * omega;
        let weight = k as f64;
        re += weight * c * angle.cos();
        im += weight * c * angle.sin();
    }
    (re, im)
}

/// Evaluate polynomial in z^{-1} on the unit circle at angle ω.
/// Returns (real_part, imag_part) of Σ c[k] * e^{-jkω}.
fn eval_poly_on_unit_circle(coeffs: &[f64], omega: f64) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for (k, &c) in coeffs.iter().enumerate() {
        let angle = -(k as f64) * omega;
        re += c * angle.cos();
        im += c * angle.sin();
    }
    (re, im)
}

/// Evaluate analog polynomial B(jω) = Σ b[k] * (jω)^{n-k}.
/// Note: analog polynomials are in descending power order.
fn eval_analog_poly(coeffs: &[f64], omega: f64) -> (f64, f64) {
    let n = coeffs.len();
    let mut re = 0.0;
    let mut im = 0.0;
    for (k, &c) in coeffs.iter().enumerate() {
        let power = (n - 1 - k) as u32;
        // (jω)^power: j^0=1, j^1=j, j^2=-1, j^3=-j, j^4=1, ...
        let omega_pow = omega.powi(power as i32);
        match power % 4 {
            0 => re += c * omega_pow, // j^0 = 1
            1 => im += c * omega_pow, // j^1 = j
            2 => re -= c * omega_pow, // j^2 = -1
            3 => im -= c * omega_pow, // j^3 = -j
            _ => re += 0.0,           // Mathematically impossible
        }
    }
    (re, im)
}

// ══════════════════════════════════════════════════════════════════════
// Spectral Analysis
// ══════════════════════════════════════════════════════════════════════

/// Result of a spectral density estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralResult {
    /// Frequency bins.
    pub frequencies: Vec<f64>,
    /// Power spectral density estimates.
    pub psd: Vec<f64>,
}

/// Scaling mode for spectral-density style estimators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralScaling {
    /// Power spectral density with units per Hz.
    Density,
    /// Power spectrum with squared-amplitude units.
    Spectrum,
}

/// Estimate power spectral density using a periodogram.
///
/// Matches `scipy.signal.periodogram(x, fs, window, scaling='density')`.
///
/// # Arguments
/// * `x` — Input signal
/// * `fs` — Sampling frequency (Hz)
/// * `window` — Window function applied to the signal (None = boxcar/rectangular)
pub fn periodogram(
    x: &[f64],
    fs: f64,
    window: Option<&[f64]>,
) -> Result<SpectralResult, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }
    let n = x.len();

    // Apply window
    let windowed: Vec<f64> = match window {
        Some(w) => {
            if w.len() != n {
                return Err(SignalError::InvalidArgument(format!(
                    "window length ({}) must match signal length ({n})",
                    w.len()
                )));
            }
            x.iter().zip(w.iter()).map(|(&xi, &wi)| xi * wi).collect()
        }
        None => x.to_vec(),
    };

    // Window power for normalization
    let win_power: f64 = match window {
        Some(w) => w.iter().map(|&wi| wi * wi).sum::<f64>() / n as f64,
        None => 1.0,
    };

    // Compute FFT
    let opts = fsci_fft::FftOptions::default();
    let spectrum = fsci_fft::rfft(&windowed, &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;

    // One-sided PSD: |X[k]|² / (fs * N * win_power)
    let scale = 1.0 / (fs * n as f64 * win_power);
    let n_freqs = spectrum.len();
    let mut psd = Vec::with_capacity(n_freqs);
    for (k, &(re, im)) in spectrum.iter().enumerate() {
        let mag2 = re * re + im * im;
        // Double all bins except DC and Nyquist for one-sided spectrum
        let factor = if k == 0 || (n.is_multiple_of(2) && k == n_freqs - 1) {
            1.0
        } else {
            2.0
        };
        psd.push(mag2 * scale * factor);
    }

    // Frequency bins
    let freq_step = fs / n as f64;
    let frequencies: Vec<f64> = (0..n_freqs).map(|k| k as f64 * freq_step).collect();

    Ok(SpectralResult { frequencies, psd })
}

/// Estimate power spectral density using Welch's method.
///
/// Matches `scipy.signal.welch(x, fs, nperseg, noverlap)`.
///
/// Divides the signal into overlapping segments, windows each, computes
/// periodograms, and averages them.
///
/// # Arguments
/// * `x` — Input signal
/// * `fs` — Sampling frequency (Hz)
/// * `window` — Window type to use (default: "hann")
/// * `nperseg` — Length of each segment (default: 256)
/// * `noverlap` — Number of overlapping samples (default: nperseg/2)
pub fn welch(
    x: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
) -> Result<SpectralResult, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }

    let nperseg = nperseg.unwrap_or_else(|| x.len().min(256));
    if nperseg == 0 {
        return Err(SignalError::InvalidArgument(
            "nperseg must be > 0".to_string(),
        ));
    }
    let nperseg = nperseg.min(x.len());

    let noverlap = noverlap.unwrap_or(nperseg / 2);
    if noverlap >= nperseg {
        return Err(SignalError::InvalidArgument(
            "noverlap must be < nperseg".to_string(),
        ));
    }

    let step = nperseg - noverlap;
    let win_coeffs = get_window(window.unwrap_or("hann"), nperseg)?;

    // Segment the signal and compute periodograms
    let n_freqs = nperseg / 2 + 1;
    let mut avg_psd = vec![0.0; n_freqs];
    let mut n_segments = 0usize;

    let mut start = 0;
    while start + nperseg <= x.len() {
        let segment = &x[start..start + nperseg];
        let seg_result = periodogram(segment, fs, Some(&win_coeffs))?;

        for (avg, &val) in avg_psd.iter_mut().zip(seg_result.psd.iter()) {
            *avg += val;
        }
        n_segments += 1;
        start += step;
    }

    if n_segments == 0 {
        return Err(SignalError::InvalidArgument(
            "signal too short for any segment".to_string(),
        ));
    }

    // Average
    for val in &mut avg_psd {
        *val /= n_segments as f64;
    }

    // Frequency bins
    let freq_step = fs / nperseg as f64;
    let frequencies: Vec<f64> = (0..n_freqs).map(|k| k as f64 * freq_step).collect();

    Ok(SpectralResult {
        frequencies,
        psd: avg_psd,
    })
}

// ══════════════════════════════════════════════════════════════════════
// FIR Filter Design
// ══════════════════════════════════════════════════════════════════════

/// Window type for FIR filter design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FirWindow {
    Hamming,
    Hann,
    Blackman,
    /// Kaiser window with shape parameter beta.
    Kaiser(f64),
    /// Rectangular window (no windowing).
    Rectangular,
}

/// Design a FIR filter using the window method.
///
/// Matches `scipy.signal.firwin(numtaps, cutoff, window, pass_zero, fs)`.
///
/// Creates a linear-phase FIR filter by windowing the ideal sinc response.
///
/// # Arguments
/// * `numtaps` — Filter length (must be odd for Type I, even for Type II).
/// * `cutoff` — Cutoff frequency/frequencies (normalized 0 to 1, where 1 = Nyquist).
///   One value for lowpass/highpass, two for bandpass/bandstop.
/// * `window` — Window function to apply.
/// * `pass_zero` — If true, DC gain is 1 (lowpass/bandstop). If false, DC gain is 0 (highpass/bandpass).
pub fn firwin(
    numtaps: usize,
    cutoff: &[f64],
    window: FirWindow,
    pass_zero: bool,
) -> Result<Vec<f64>, SignalError> {
    if numtaps == 0 {
        return Err(SignalError::InvalidArgument(
            "numtaps must be > 0".to_string(),
        ));
    }
    if cutoff.is_empty() {
        return Err(SignalError::InvalidArgument(
            "cutoff must be non-empty".to_string(),
        ));
    }
    for &c in cutoff {
        if !c.is_finite() || c <= 0.0 || c >= 1.0 {
            return Err(SignalError::InvalidArgument(format!(
                "cutoff {c} out of range (0, 1) or non-finite"
            )));
        }
    }
    for pair in cutoff.windows(2) {
        if pair[0] >= pair[1] {
            return Err(SignalError::InvalidArgument(
                "cutoff frequencies must be strictly increasing".to_string(),
            ));
        }
    }

    let m = numtaps;
    let alpha = (m - 1) as f64 / 2.0;

    // Build the ideal bandpass/bandstop impulse response
    // Start from all the bands implied by cutoff
    let mut bands: Vec<f64> = Vec::new();
    if pass_zero {
        // Starts at 0: lowpass or bandstop
        bands.push(0.0);
    }
    bands.extend_from_slice(cutoff);
    if pass_zero == cutoff.len().is_multiple_of(2) {
        // End at 1 (Nyquist)
        bands.push(1.0);
    }

    // Sum of sinc responses for each passband
    let mut h = vec![0.0; m];
    for pair in bands.chunks(2) {
        if pair.len() < 2 {
            break;
        }
        let (left, right) = (pair[0], pair[1]);
        for (n, hi) in h.iter_mut().enumerate() {
            let x = n as f64 - alpha;
            if x.abs() < 1e-15 {
                *hi += right - left;
            } else {
                *hi += (std::f64::consts::PI * right * x).sin() / (std::f64::consts::PI * x)
                    - (std::f64::consts::PI * left * x).sin() / (std::f64::consts::PI * x);
            }
        }
    }

    // Apply window
    let win = make_window(m, window);
    for (hi, wi) in h.iter_mut().zip(win.iter()) {
        *hi *= wi;
    }

    // Normalize for unity gain at the appropriate frequency:
    // - Lowpass/bandstop (pass_zero=true): normalize at DC (omega=0)
    // - Highpass (pass_zero=false, 1 cutoff): normalize at Nyquist (omega=PI)
    // - Bandpass (pass_zero=false, 2+ cutoffs): normalize at passband center
    let norm_freq = if pass_zero {
        0.0
    } else if cutoff.len() >= 2 {
        std::f64::consts::PI * (cutoff[0] + cutoff[1]) / 2.0
    } else {
        std::f64::consts::PI
    };
    let mut gain = 0.0;
    for (n, &hi) in h.iter().enumerate() {
        gain += hi * (norm_freq * (n as f64 - alpha)).cos();
    }
    if gain.abs() > 1e-15 {
        for hi in &mut h {
            *hi /= gain;
        }
    }

    Ok(h)
}

/// Design a FIR filter with arbitrary frequency response using frequency-sampling.
///
/// Matches `scipy.signal.firwin2(numtaps, freq, gain, window)`.
///
/// # Arguments
/// * `numtaps` — Filter length (should be odd for Type I filter).
/// * `freq` — Frequency points (0 to 1, normalized), must start at 0 and end at 1.
/// * `gain` — Desired gain at each frequency point.
/// * `window` — Window to apply.
pub fn firwin2(
    numtaps: usize,
    freq: &[f64],
    gain: &[f64],
    window: FirWindow,
) -> Result<Vec<f64>, SignalError> {
    if numtaps == 0 {
        return Err(SignalError::InvalidArgument(
            "numtaps must be > 0".to_string(),
        ));
    }
    if freq.len() != gain.len() {
        return Err(SignalError::InvalidArgument(
            "freq and gain must have same length".to_string(),
        ));
    }
    if freq.len() < 2 {
        return Err(SignalError::InvalidArgument(
            "need at least 2 frequency points".to_string(),
        ));
    }
    if (freq[0] - 0.0).abs() > 1e-10 || (freq[freq.len() - 1] - 1.0).abs() > 1e-10 {
        return Err(SignalError::InvalidArgument(
            "freq must start at 0 and end at 1".to_string(),
        ));
    }

    // Interpolate the desired frequency response to n_fft points
    let n_fft = if numtaps % 2 == 1 {
        numtaps
    } else {
        numtaps + 1
    };
    let n_half = n_fft / 2 + 1;

    // Linearly interpolate gain at uniform frequency grid
    let mut h_desired = vec![0.0; n_half];
    for (k, h_val) in h_desired.iter_mut().enumerate() {
        let f_k = k as f64 / (n_half - 1) as f64;
        // Find the interval in freq containing f_k
        let mut seg = 0;
        while seg + 1 < freq.len() - 1 && freq[seg + 1] < f_k {
            seg += 1;
        }
        let t = if (freq[seg + 1] - freq[seg]).abs() > 1e-15 {
            (f_k - freq[seg]) / (freq[seg + 1] - freq[seg])
        } else {
            0.0
        };
        *h_val = gain[seg] + t * (gain[seg + 1] - gain[seg]);
    }

    // Construct symmetric spectrum and do inverse FFT
    let mut spectrum = vec![0.0; n_fft];
    for (k, &val) in h_desired.iter().enumerate() {
        spectrum[k] = val;
    }
    // Mirror for negative frequencies
    for k in 1..n_half - 1 {
        if n_fft - k < n_fft {
            spectrum[n_fft - k] = spectrum[k];
        }
    }

    // Inverse DFT to get impulse response
    let mut h = vec![0.0; n_fft];
    let nf = n_fft as f64;
    for (n, hn) in h.iter_mut().enumerate() {
        for (k, &sk) in spectrum.iter().enumerate() {
            *hn += sk * (2.0 * std::f64::consts::PI * n as f64 * k as f64 / nf).cos();
        }
        *hn /= nf;
    }

    // Shift to center (circular shift by n_fft/2)
    let shift = n_fft / 2;
    let mut h_shifted = vec![0.0; n_fft];
    for (i, val) in h.iter().enumerate() {
        h_shifted[(i + shift) % n_fft] = *val;
    }

    // Truncate to numtaps
    let h_out: Vec<f64> = h_shifted[..numtaps].to_vec();

    // Apply window
    let win = make_window(numtaps, window);
    let mut result: Vec<f64> = h_out.iter().zip(win.iter()).map(|(&h, &w)| h * w).collect();

    // Normalize
    let sum: f64 = result.iter().sum();
    if sum.abs() > 1e-15 && gain[0].abs() > 1e-15 {
        let target_gain = gain[0];
        for r in &mut result {
            *r *= target_gain / sum;
        }
    }

    Ok(result)
}

/// Estimate Kaiser window parameters for FIR filter design.
///
/// Matches `scipy.signal.kaiserord(ripple, width)`.
///
/// # Arguments
/// * `ripple` — Maximum ripple/attenuation in dB (positive value, e.g. 60 for 60dB).
/// * `width` — Transition width (normalized frequency, 0 to 1).
///
/// # Returns
/// `(numtaps, beta)` — Number of taps and Kaiser beta parameter.
pub fn kaiserord(ripple: f64, width: f64) -> Result<(usize, f64), SignalError> {
    if !ripple.is_finite() || ripple <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "ripple must be positive (in dB)".to_string(),
        ));
    }
    if !width.is_finite() || width <= 0.0 || width >= 1.0 {
        return Err(SignalError::InvalidArgument(
            "width must be in (0, 1)".to_string(),
        ));
    }

    let a = ripple; // attenuation in dB

    // Kaiser's empirical formula for beta
    let beta = if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a > 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    };

    // Kaiser's formula for number of taps
    let numtaps = ((a - 7.95) / (2.285 * std::f64::consts::PI * width)).ceil() as usize + 1;
    let numtaps = numtaps.max(1);

    Ok((numtaps, beta))
}

/// Generate a window of given type and length.
fn make_window(n: usize, window: FirWindow) -> Vec<f64> {
    match window {
        FirWindow::Hamming => hamming(n),
        FirWindow::Hann => hann(n),
        FirWindow::Blackman => blackman(n),
        FirWindow::Kaiser(beta) => kaiser(n, beta),
        FirWindow::Rectangular => vec![1.0; n],
    }
}

/// Design a FIR filter using the Parks-McClellan (Remez exchange) algorithm.
///
/// Matches `scipy.signal.remez(numtaps, bands, desired)`.
///
/// # Arguments
/// * `numtaps` — Number of filter coefficients (must be odd for type I).
/// * `bands` — Frequency band edges, normalized [0, 0.5]. Must come in pairs.
///   e.g., `[0.0, 0.1, 0.2, 0.5]` defines passband [0, 0.1] and stopband [0.2, 0.5].
/// * `desired` — Desired gain in each band (one per band pair).
/// * `weight` — Optional weights per band (one per band pair). Default: all 1.0.
pub fn remez(
    numtaps: usize,
    bands: &[f64],
    desired: &[f64],
    weight: Option<&[f64]>,
) -> Result<Vec<f64>, SignalError> {
    if numtaps < 1 {
        return Err(SignalError::InvalidArgument(
            "numtaps must be >= 1".to_string(),
        ));
    }
    if !bands.len().is_multiple_of(2) || bands.is_empty() {
        return Err(SignalError::InvalidArgument(
            "bands must have even number of elements".to_string(),
        ));
    }
    let nbands = bands.len() / 2;
    if desired.len() != nbands {
        return Err(SignalError::InvalidArgument(format!(
            "desired length {} must equal number of bands {}",
            desired.len(),
            nbands
        )));
    }

    let weights: Vec<f64> = weight.map_or_else(|| vec![1.0; nbands], |w| w.to_vec());
    if weights.len() != nbands {
        return Err(SignalError::InvalidArgument(
            "weight length must equal number of bands".to_string(),
        ));
    }

    // Simplified Remez: use frequency-sampling method as approximation
    // (Full Parks-McClellan is very complex; this provides a reasonable approximation)
    let n = numtaps;
    let m = n / 2; // number of cosine coefficients for type I (odd length)

    // Create dense frequency grid
    let grid_size = 512.max(16 * n);
    let mut freq_grid = Vec::with_capacity(grid_size);
    let mut desired_grid = Vec::with_capacity(grid_size);
    let mut weight_grid = Vec::with_capacity(grid_size);

    for band_idx in 0..nbands {
        let f_start = bands[2 * band_idx];
        let f_end = bands[2 * band_idx + 1];
        let npts = (grid_size as f64 * (f_end - f_start) * 2.0).ceil() as usize;
        let npts = npts.max(4);

        for k in 0..npts {
            let freq = f_start + (f_end - f_start) * k as f64 / (npts - 1).max(1) as f64;
            freq_grid.push(freq);
            desired_grid.push(desired[band_idx]);
            weight_grid.push(weights[band_idx]);
        }
    }

    // Least-squares design on the cosine basis
    // H(f) = sum_{k=0}^{m} a_k * cos(2π * k * f)
    // Minimize weighted error: sum w_i * (H(f_i) - D(f_i))^2
    let n_coeffs = m + 1;
    let ng = freq_grid.len();

    // Normal equations: A^T W A x = A^T W d
    let mut ata = vec![vec![0.0; n_coeffs]; n_coeffs];
    let mut atd = vec![0.0; n_coeffs];
    let two_pi = 2.0 * std::f64::consts::PI;

    for i in 0..ng {
        let f = freq_grid[i];
        let w = weight_grid[i]; // WLS weight: minimize Σ w_i * (H(f_i) - D(f_i))²
        let d = desired_grid[i];

        for j in 0..n_coeffs {
            let cj = (two_pi * j as f64 * f).cos();
            atd[j] += w * cj * d;
            let (head, tail) = ata.split_at_mut(j + 1);
            let ata_j = &mut head[j];
            ata_j[j] += w * cj * cj;
            for (offset, ata_row) in tail.iter_mut().enumerate() {
                let k = j + 1 + offset;
                let ck = (two_pi * k as f64 * f).cos();
                let value = w * cj * ck;
                ata_row[j] += value;
                ata_j[k] += value;
            }
        }
    }

    // Solve normal equations (Cholesky or simple Gaussian elimination)
    let a_coeffs = solve_symmetric(&ata, &atd)?;

    // Convert cosine coefficients to symmetric FIR filter taps.
    // H(f) = a_0 + 2 * Σ_{k=1}^{m} a_k cos(2πkf)
    // corresponds to h[m] = a_0, h[m±k] = a_k/2 for k=1..m
    let mut h = vec![0.0; n];
    h[m] = a_coeffs[0];
    for k in 1..n_coeffs {
        if m >= k {
            h[m - k] = a_coeffs[k] / 2.0;
        }
        if m + k < n {
            h[m + k] = a_coeffs[k] / 2.0;
        }
    }

    Ok(h)
}

/// Design a linear-phase FIR filter using least-squares.
///
/// Matches `scipy.signal.firls(numtaps, bands, desired)`.
///
/// # Arguments
/// * `numtaps` — Number of taps (must be odd for type I).
/// * `bands` — Frequency band edges in pairs, normalized [0, 1] where 1 = Nyquist.
/// * `desired` — Desired gain at each band edge (length must equal bands length).
/// * `weight` — Optional weight per band pair.
pub fn firls(
    numtaps: usize,
    bands: &[f64],
    desired: &[f64],
    weight: Option<&[f64]>,
) -> Result<Vec<f64>, SignalError> {
    if numtaps < 1 || numtaps.is_multiple_of(2) {
        return Err(SignalError::InvalidArgument(
            "numtaps must be odd and >= 1".to_string(),
        ));
    }
    if !bands.len().is_multiple_of(2) || bands.is_empty() {
        return Err(SignalError::InvalidArgument(
            "bands must have even number of elements".to_string(),
        ));
    }
    if desired.len() != bands.len() {
        return Err(SignalError::InvalidArgument(format!(
            "desired length {} must equal bands length {}",
            desired.len(),
            bands.len()
        )));
    }

    let nbands = bands.len() / 2;
    let weights: Vec<f64> = weight.map_or_else(|| vec![1.0; nbands], |w| w.to_vec());

    let m = (numtaps - 1) / 2;
    let n_coeffs = m + 1;

    // Build the Q matrix and b vector for the least-squares problem
    // Q[i,j] = Σ_bands w_k ∫_{f_lo}^{f_hi} cos(πif) cos(πjf) df
    // b[i] = Σ_bands w_k ∫_{f_lo}^{f_hi} D(f) cos(πif) df
    // where D(f) is linearly interpolated desired response

    let mut q = vec![vec![0.0; n_coeffs]; n_coeffs];
    let mut b_vec = vec![0.0; n_coeffs];

    for band in 0..nbands {
        let f_lo = bands[2 * band] / 2.0; // Convert from [0,1] Nyquist to [0,0.5] normalized
        let f_hi = bands[2 * band + 1] / 2.0;
        let d_lo = desired[2 * band];
        let d_hi = desired[2 * band + 1];
        let w = weights[band];

        let df = f_hi - f_lo;
        if df <= 0.0 {
            continue;
        }

        // D(f) = d_lo + (d_hi - d_lo) * (f - f_lo) / (f_hi - f_lo)
        //      = d_lo + slope * (f - f_lo)
        let slope = (d_hi - d_lo) / df;

        let pi = std::f64::consts::PI;

        for (i, row_i) in q.iter_mut().enumerate().take(n_coeffs) {
            // b[i] = w * ∫ D(f) cos(2πif) df over [f_lo, f_hi]
            let bi = if i == 0 {
                // ∫ D(f) df = d_lo * df + slope * df^2 / 2
                w * (d_lo * df + slope * df * df / 2.0)
            } else {
                let pi_i = pi * i as f64;
                // ∫ (d_lo + slope*(f-f_lo)) * cos(2πif) df
                // = d_lo * sin(2πif)/(2πi) + slope * [(f-f_lo)*sin(2πif)/(2πi) + cos(2πif)/(2πi)^2]
                let sin_hi = (2.0 * pi_i * f_hi).sin();
                let sin_lo = (2.0 * pi_i * f_lo).sin();
                let cos_hi = (2.0 * pi_i * f_hi).cos();
                let cos_lo = (2.0 * pi_i * f_lo).cos();
                let inv_pi_i = 1.0 / (2.0 * pi_i);

                let part1 = d_lo * (sin_hi - sin_lo) * inv_pi_i;
                let part2 =
                    slope * (df * sin_hi * inv_pi_i + (cos_hi - cos_lo) * inv_pi_i * inv_pi_i);
                w * (part1 + part2)
            };
            b_vec[i] += bi;

            for (j, cell) in row_i.iter_mut().enumerate().skip(i) {
                // Q[i,j] = w * ∫ cos(2πif) cos(2πjf) df
                // = w/2 * ∫ [cos(2π(i-j)f) + cos(2π(i+j)f)] df
                let qij = if i == 0 && j == 0 {
                    w * df
                } else {
                    let integrate_cos = |freq: f64| -> f64 {
                        if freq.abs() < 1e-15 {
                            df
                        } else {
                            let pf = 2.0 * pi * freq;
                            ((pf * f_hi).sin() - (pf * f_lo).sin()) / pf
                        }
                    };
                    w * 0.5
                        * (integrate_cos(i as f64 - j as f64) + integrate_cos(i as f64 + j as f64))
                };
                *cell += qij;
            }
        }
    }

    let upper_triangle = q.clone();
    for (i, row_i) in q.iter_mut().enumerate().take(n_coeffs) {
        for (j, cell) in row_i.iter_mut().enumerate().take(i) {
            *cell = upper_triangle[j][i];
        }
    }

    // Solve Q a = b
    let a_coeffs = solve_symmetric(&q, &b_vec)?;

    // Convert to FIR filter taps
    let n = numtaps;
    let mut h = vec![0.0; n];
    h[m] = a_coeffs[0];
    for k in 1..n_coeffs {
        h[m - k] = a_coeffs[k] / 2.0;
        h[m + k] = a_coeffs[k] / 2.0;
    }

    Ok(h)
}

/// Solve symmetric positive definite system via Cholesky.
fn solve_symmetric(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, SignalError> {
    let n = a.len();
    // Cholesky factorization: A = L L^T
    let mut l = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut sum = a[j][j];
        for &value in l[j].iter().take(j) {
            sum -= value * value;
        }
        if sum <= 0.0 {
            // Fall back to regularization
            l[j][j] = 1e-10;
        } else {
            l[j][j] = sum.sqrt();
        }
        for i in j + 1..n {
            let mut sum = a[i][j];
            for (&li_k, &lj_k) in l[i].iter().zip(l[j].iter()).take(j) {
                sum -= li_k * lj_k;
            }
            l[i][j] = sum / l[j][j];
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        y[i] = sum / l[i][i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in i + 1..n {
            sum -= l[j][i] * x[j];
        }
        x[i] = sum / l[i][i];
    }

    Ok(x)
}

// ── Chirp method ────────────────────────────────────────────────────

/// Method for frequency sweep in `chirp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChirpMethod {
    /// Linear frequency sweep (instantaneous frequency is linear in time).
    #[default]
    Linear,
    /// Quadratic frequency sweep.
    Quadratic,
    /// Logarithmic frequency sweep (f0 and f1 must be > 0).
    Logarithmic,
}

/// Generate a swept-frequency cosine (chirp) signal.
///
/// Matches `scipy.signal.chirp(t, f0, t1, f1, method)`.
///
/// # Arguments
/// * `t` — Time array at which to evaluate the signal.
/// * `f0` — Frequency at time 0.
/// * `t1` — Time at which `f1` is specified.
/// * `f1` — Frequency at time `t1`.
/// * `method` — Chirp type: linear, quadratic, or logarithmic.
pub fn chirp(
    t: &[f64],
    f0: f64,
    t1: f64,
    f1: f64,
    method: ChirpMethod,
) -> Result<Vec<f64>, SignalError> {
    if !t1.is_finite() || t1 <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "t1 must be positive".to_string(),
        ));
    }
    if !f0.is_finite() || !f1.is_finite() {
        return Err(SignalError::InvalidArgument(
            "f0 and f1 must be finite".to_string(),
        ));
    }
    if method == ChirpMethod::Logarithmic && (f0 <= 0.0 || f1 <= 0.0) {
        return Err(SignalError::InvalidArgument(
            "logarithmic chirp requires f0 > 0 and f1 > 0".to_string(),
        ));
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let result = match method {
        ChirpMethod::Linear => {
            let k = (f1 - f0) / t1;
            t.iter()
                .map(|&ti| (two_pi * (f0 * ti + 0.5 * k * ti * ti)).cos())
                .collect()
        }
        ChirpMethod::Quadratic => {
            let k = (f1 - f0) / (t1 * t1);
            t.iter()
                .map(|&ti| (two_pi * (f0 * ti + k * ti * ti * ti / 3.0)).cos())
                .collect()
        }
        ChirpMethod::Logarithmic => {
            let ratio = f1 / f0;
            if (ratio - 1.0).abs() < f64::EPSILON {
                // f0 == f1: constant frequency, just a cosine
                t.iter().map(|&ti| (two_pi * f0 * ti).cos()).collect()
            } else {
                let log_ratio = ratio.ln();
                t.iter()
                    .map(|&ti| {
                        let phase = two_pi * f0 * t1 / log_ratio * (ratio.powf(ti / t1) - 1.0);
                        phase.cos()
                    })
                    .collect()
            }
        }
    };
    Ok(result)
}

/// Generate a sawtooth or triangle wave.
///
/// Matches `scipy.signal.sawtooth(t, width)`.
///
/// # Arguments
/// * `t` — Time array (phase in radians; period is 2π).
/// * `width` — Width of the rising ramp as a proportion of the period (0 to 1).
///   `width=1` gives a rising sawtooth, `width=0` a falling sawtooth,
///   `width=0.5` a triangle wave.
pub fn sawtooth(t: &[f64], width: f64) -> Result<Vec<f64>, SignalError> {
    if !width.is_finite() || !(0.0..=1.0).contains(&width) {
        return Err(SignalError::InvalidArgument(
            "width must be finite and in [0, 1]".to_string(),
        ));
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let result = t
        .iter()
        .map(|&ti| {
            // Normalize to [0, 1) within one period.
            let phase = ((ti / two_pi) % 1.0 + 1.0) % 1.0;
            if width == 0.0 {
                // Pure falling ramp.
                1.0 - 2.0 * phase
            } else if phase < width {
                -1.0 + 2.0 * phase / width
            } else {
                1.0 - 2.0 * (phase - width) / (1.0 - width)
            }
        })
        .collect();
    Ok(result)
}

/// Generate a square wave.
///
/// Matches `scipy.signal.square(t, duty)`.
///
/// # Arguments
/// * `t` — Time array (phase in radians; period is 2π).
/// * `duty` — Duty cycle (fraction of period at +1). Default 0.5.
pub fn square(t: &[f64], duty: f64) -> Result<Vec<f64>, SignalError> {
    if !duty.is_finite() || !(0.0..=1.0).contains(&duty) {
        return Err(SignalError::InvalidArgument(
            "duty must be finite and in [0, 1]".to_string(),
        ));
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let result = t
        .iter()
        .map(|&ti| {
            let phase = ((ti / two_pi) % 1.0 + 1.0) % 1.0;
            if phase < duty { 1.0 } else { -1.0 }
        })
        .collect();
    Ok(result)
}

/// Generate a discrete unit impulse (Kronecker delta).
///
/// Matches `scipy.signal.unit_impulse(shape, idx)`.
///
/// # Arguments
/// * `shape` — Length of the output array.
/// * `idx` — Index at which the impulse is placed. If `None`, places at index 0.
pub fn unit_impulse(shape: usize, idx: Option<usize>) -> Result<Vec<f64>, SignalError> {
    let i = idx.unwrap_or(0);
    if i >= shape {
        return Err(SignalError::InvalidArgument(format!(
            "idx {i} out of range for shape {shape}"
        )));
    }
    let mut out = vec![0.0; shape];
    out[i] = 1.0;
    Ok(out)
}

/// Detrend type for `detrend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetrendType {
    /// Remove the mean (constant trend).
    Constant,
    /// Remove a linear trend (least-squares fit of y = a + b*x).
    #[default]
    Linear,
}

/// Remove a trend from data.
///
/// Matches `scipy.signal.detrend(data, type)`.
///
/// # Arguments
/// * `data` — Input signal.
/// * `dtype` — Type of detrending: constant (remove mean) or linear (remove linear fit).
pub fn detrend(data: &[f64], dtype: DetrendType) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::InvalidArgument(
            "data must not be empty".to_string(),
        ));
    }
    match dtype {
        DetrendType::Constant => {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            Ok(data.iter().map(|&v| v - mean).collect())
        }
        DetrendType::Linear => {
            let n = data.len();
            let n_f = n as f64;
            if n < 2 {
                return Ok(vec![0.0; n]);
            }

            // Center x-values to [-(n-1)/2, (n-1)/2] for numerical stability.
            // This makes sum(x) = 0, simplifying the normal equations.
            let x_mean = (n_f - 1.0) / 2.0;

            let mut s_xx = 0.0;
            let mut s_xy = 0.0;
            let mut y_sum = 0.0;

            for (i, &y) in data.iter().enumerate() {
                let x = i as f64 - x_mean;
                s_xx += x * x;
                s_xy += x * y;
                y_sum += y;
            }

            let y_mean = y_sum / n_f;
            let slope = s_xy / s_xx;

            Ok(data
                .iter()
                .enumerate()
                .map(|(i, &y)| {
                    let x = i as f64 - x_mean;
                    y - (y_mean + slope * x)
                })
                .collect())
        }
    }
}

/// Apply a median filter to a 1-D signal.
///
/// Matches `scipy.signal.medfilt(volume, kernel_size)`.
///
/// # Arguments
/// * `data` — Input signal.
/// * `kernel_size` — Size of the median filter window (must be odd and >= 1).
pub fn medfilt(data: &[f64], kernel_size: usize) -> Result<Vec<f64>, SignalError> {
    if kernel_size == 0 || kernel_size.is_multiple_of(2) {
        return Err(SignalError::InvalidArgument(
            "kernel_size must be odd and >= 1".to_string(),
        ));
    }
    if data.is_empty() {
        return Ok(vec![]);
    }

    let half = kernel_size / 2;
    let n = data.len();
    let mut result = Vec::with_capacity(n);
    let mut window = vec![0.0; kernel_size];

    for i in 0..n {
        // Fill window with zero-padding at boundaries (matches SciPy)
        for (j, val) in window.iter_mut().enumerate() {
            let idx = i as i64 + j as i64 - half as i64;
            *val = if idx >= 0 && idx < n as i64 {
                data[idx as usize]
            } else {
                0.0
            };
        }

        // Linear-time median selection
        let mid = kernel_size / 2;
        let (_, &mut m, _) = window.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        result.push(m);
    }

    Ok(result)
}

/// Apply a 1-D adaptive Wiener filter.
///
/// Matches the high-level behavior of `scipy.signal.wiener(im, mysize, noise)`
/// for 1-D inputs with zero-padded local windows.
pub fn wiener(data: &[f64], mysize: usize, noise: Option<f64>) -> Result<Vec<f64>, SignalError> {
    if mysize == 0 || mysize.is_multiple_of(2) {
        return Err(SignalError::InvalidArgument(
            "mysize must be odd and >= 1".to_string(),
        ));
    }
    if let Some(value) = noise
        && (!value.is_finite() || value < 0.0)
    {
        return Err(SignalError::InvalidArgument(
            "noise must be a finite non-negative value when provided".to_string(),
        ));
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let half = mysize / 2;
    let n = data.len();
    let mut local_mean = vec![0.0; n];
    let mut local_var = vec![0.0; n];
    let window_area = mysize as f64;

    for i in 0..n {
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for offset in 0..mysize {
            let idx = i as i64 + offset as i64 - half as i64;
            let value = if idx >= 0 && idx < n as i64 {
                data[idx as usize]
            } else {
                0.0
            };
            sum += value;
            sumsq += value * value;
        }
        let mean = sum / window_area;
        local_mean[i] = mean;
        local_var[i] = (sumsq / window_area - mean * mean).max(0.0);
    }

    let noise_power =
        noise.unwrap_or_else(|| local_var.iter().sum::<f64>() / local_var.len() as f64);

    Ok(data
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            if local_var[i] <= noise_power {
                local_mean[i]
            } else {
                let gain = 1.0 - noise_power / local_var[i];
                local_mean[i] + gain * (value - local_mean[i])
            }
        })
        .collect())
}

/// Generate a Bartlett window.
pub fn bartlett(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nf = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let x = i as f64 - nf / 2.0;
            1.0 - (2.0 * x / nf).abs()
        })
        .collect()
}

/// Generate a flat top window.
pub fn flattop(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let nf = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let arg = two_pi * i as f64 / nf;
            0.21557895 - 0.41663158 * arg.cos() + 0.277263158 * (2.0 * arg).cos()
                - 0.083578947 * (3.0 * arg).cos()
                + 0.006947368 * (4.0 * arg).cos()
        })
        .collect()
}

/// Generate a cosine window.
///
/// Matches `scipy.signal.windows.cosine(M)`.
pub fn cosine(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nf = n as f64;
    (0..n)
        .map(|i| (std::f64::consts::PI * (i as f64 + 0.5) / nf).sin())
        .collect()
}

/// Generic weighted sum of cosine terms window.
///
/// Matches `scipy.signal.windows.general_cosine(M, a, sym)`.
pub fn general_cosine(n: usize, coeffs: &[f64], sym: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    let m = if sym { n } else { n + 1 };
    let denom = (m - 1) as f64;
    let mut window: Vec<f64> = (0..m)
        .map(|i| {
            let angle = -std::f64::consts::PI + 2.0 * std::f64::consts::PI * i as f64 / denom;
            coeffs
                .iter()
                .enumerate()
                .map(|(k, &coefficient)| coefficient * (k as f64 * angle).cos())
                .sum()
        })
        .collect();
    if !sym {
        window.truncate(n);
    }
    window
}

/// Gaussian window.
///
/// Matches `scipy.signal.windows.gaussian(n, std, sym)`.
pub fn gaussian(n: usize, std: f64, sym: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = if sym { n } else { n + 1 };
    let center = (m as f64 - 1.0) / 2.0;
    let mut window: Vec<f64> = (0..m)
        .map(|i| {
            let x = (i as f64 - center) / std;
            (-0.5 * x * x).exp()
        })
        .collect();
    if !sym {
        window.truncate(n);
    }
    window
}

/// General Gaussian window: exp(-0.5 * |x/σ|^(2p)).
///
/// Matches `scipy.signal.windows.general_gaussian(n, p, sig, sym)`.
pub fn general_gaussian(n: usize, p: f64, sig: f64, sym: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = if sym { n } else { n + 1 };
    let center = (m as f64 - 1.0) / 2.0;
    let mut window: Vec<f64> = (0..m)
        .map(|i| {
            let x = (i as f64 - center) / sig;
            (-0.5 * x.abs().powf(2.0 * p)).exp()
        })
        .collect();
    if !sym {
        window.truncate(n);
    }
    window
}

/// Parzen (de la Vallée Poussin) window.
///
/// Matches `scipy.signal.windows.parzen(n)`.
pub fn parzen(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nf = n as f64;
    let half = nf / 2.0;
    (0..n)
        .map(|i| {
            let k = (i as f64 - (nf - 1.0) / 2.0).abs();
            if k <= nf / 4.0 {
                1.0 - 6.0 * (k / half).powi(2) + 6.0 * (k / half).powi(3)
            } else {
                2.0 * (1.0 - k / half).powi(3)
            }
        })
        .collect()
}

/// Exponential (Poisson) window.
///
/// Matches `scipy.signal.windows.exponential(M, center=None, tau=1.0, sym=True)`.
pub fn exponential(
    n: usize,
    center: Option<f64>,
    tau: f64,
    sym: bool,
) -> Result<Vec<f64>, SignalError> {
    if sym && center.is_some() {
        return Err(SignalError::InvalidArgument(
            "If sym==True, center must be None.".into(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let m = if sym { n } else { n + 1 };
    let center = center.unwrap_or((m as f64 - 1.0) / 2.0);
    let mut window: Vec<f64> = (0..m)
        .map(|i| (-(i as f64 - center).abs() / tau).exp())
        .collect();
    if !sym {
        window.truncate(n);
    }
    Ok(window)
}

/// Lanczos window (sinc window).
///
/// The Lanczos window is the central lobe of a sinc function. It has the form:
/// w(n) = sinc(2n/(N-1) - 1) for n = 0, ..., N-1
///
/// where sinc(x) = sin(πx)/(πx).
///
/// Matches `scipy.signal.windows.lanczos(n)`.
pub fn lanczos(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nf = n as f64;
    (0..n)
        .map(|i| {
            let x = 2.0 * (i as f64) / (nf - 1.0) - 1.0;
            if x.abs() < 1e-10 {
                1.0
            } else {
                let pi_x = std::f64::consts::PI * x;
                pi_x.sin() / pi_x
            }
        })
        .collect()
}

fn fft_real_parts(input: &[fsci_fft::Complex64]) -> Vec<f64> {
    let opts = fsci_fft::FftOptions::default();
    match fsci_fft::fft(input, &opts) {
        Ok(values) => values.into_iter().map(|(re, _)| re).collect(),
        Err(_) => {
            let n = input.len() as f64;
            (0..input.len())
                .map(|k| {
                    input.iter().enumerate().fold(0.0, |acc, (m, &(re, im))| {
                        let angle = 2.0 * std::f64::consts::PI * k as f64 * m as f64 / n;
                        acc + re * angle.cos() + im * angle.sin()
                    })
                })
                .collect()
        }
    }
}

/// Dolph-Chebyshev window.
///
/// Matches `scipy.signal.windows.chebwin(n, at)`.
pub fn chebwin(n: usize, at: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    let order = n as f64 - 1.0;
    let beta = (10.0_f64.powf(at.abs() / 20.0).acosh() / order).cosh();
    let negative_sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
    let p: Vec<f64> = (0..n)
        .map(|k| {
            let x = beta * (std::f64::consts::PI * k as f64 / n as f64).cos();
            if x > 1.0 {
                (order * x.acosh()).cosh()
            } else if x < -1.0 {
                negative_sign * (order * (-x).acosh()).cosh()
            } else {
                (order * x.acos()).cos()
            }
        })
        .collect();

    let fft_input: Vec<fsci_fft::Complex64> = if n.is_multiple_of(2) {
        p.iter()
            .enumerate()
            .map(|(k, &value)| {
                let angle = std::f64::consts::PI * k as f64 / n as f64;
                (value * angle.cos(), value * angle.sin())
            })
            .collect()
    } else {
        p.into_iter().map(|value| (value, 0.0)).collect()
    };

    let fft_values = fft_real_parts(&fft_input);
    let mut window = Vec::with_capacity(n);
    if n.is_multiple_of(2) {
        let split = n / 2 + 1;
        window.extend(fft_values[1..split].iter().rev().copied());
        window.extend_from_slice(&fft_values[1..split]);
    } else {
        let split = n.div_ceil(2);
        window.extend(fft_values[1..split].iter().rev().copied());
        window.extend_from_slice(&fft_values[..split]);
    }

    let max_value = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_value.is_finite() && max_value > 0.0 {
        for value in &mut window {
            *value /= max_value;
        }
    }
    window
}

/// Taylor window.
///
/// The Taylor window is designed for radar and antenna applications. It provides
/// nearly constant sidelobe levels with a specified peak sidelobe attenuation.
///
/// Matches `scipy.signal.windows.taylor(M, nbar, sll, norm, sym)`.
///
/// # Arguments
/// * `n` - Window length
/// * `nbar` - Number of nearly constant-level sidelobes adjacent to the mainlobe
/// * `sll` - Desired peak sidelobe level in dB (negative, e.g., -30.0)
/// * `norm` - If true, normalize the window to have unit peak
/// * `sym` - If true, generate symmetric window (default for filter design)
pub fn taylor(n: usize, nbar: usize, sll: f64, norm: bool, sym: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    let nbar = nbar.max(1);
    let n_use = if sym { n } else { n + 1 };

    // br-y6wj: match scipy.signal.windows.taylor exactly. Two prior
    // bugs: (a) Fm denominator was missing the factor of 2, (b) sample
    // positions used i/(N-1) instead of scipy's centered (i - (N-1)/2)/N.
    let b = 10.0_f64.powf(sll.abs() / 20.0);
    let a = (b + (b * b - 1.0).max(0.0).sqrt()).ln() / std::f64::consts::PI;
    let a_sq = a * a;

    let nbar_f = nbar as f64;
    let sigma_sq = nbar_f * nbar_f / (a_sq + (nbar_f - 0.5) * (nbar_f - 0.5));

    let mut fm = vec![0.0; nbar];
    for (m, fm_m) in fm.iter_mut().enumerate().take(nbar).skip(1) {
        let mf = m as f64;
        let m_sq = mf * mf;

        let mut num = 1.0;
        for nn in 1..nbar {
            let nf = nn as f64;
            let term = a_sq + (nf - 0.5) * (nf - 0.5);
            num *= 1.0 - m_sq / (sigma_sq * term);
        }

        let mut den = 1.0;
        for nn in 1..nbar {
            if nn != m {
                let nf = nn as f64;
                den *= 1.0 - m_sq / (nf * nf);
            }
        }
        // br-y6wj fix (a): scipy multiplies the denominator by 2.
        den *= 2.0;

        let sign = if m % 2 == 1 { 1.0 } else { -1.0 };
        *fm_m = if den.abs() > 1e-15 {
            sign * num / den
        } else {
            0.0
        };
    }

    // br-y6wj fix (b): scipy's sample positions are (i - (N-1)/2) / N.
    let nf = n_use as f64;
    let center = (nf - 1.0) / 2.0;
    let mut w: Vec<f64> = (0..n_use)
        .map(|i| {
            let pos = (i as f64) - center;
            let x = pos / nf;
            let mut val = 1.0;
            for (m, &fm_m) in fm.iter().enumerate().take(nbar).skip(1) {
                val += 2.0 * fm_m * (2.0 * std::f64::consts::PI * m as f64 * x).cos();
            }
            val
        })
        .collect();

    // If asymmetric, truncate to n samples
    if !sym {
        w.truncate(n);
    }

    // br-y6wj fix (c): scipy normalizes by W((M-1)/2), the value at the
    // exact center (fractional sample for even M), not the max of sampled
    // values. For even M the center falls between samples, so max(w) <
    // W(center) and using max would underscale the window.
    if norm {
        let center_x = ((nf - 1.0) / 2.0 - center) / nf;
        let mut center_val = 1.0;
        for (m, &fm_m) in fm.iter().enumerate().take(nbar).skip(1) {
            center_val += 2.0 * fm_m * (2.0 * std::f64::consts::PI * m as f64 * center_x).cos();
        }
        if center_val.abs() > 1e-15 {
            for v in &mut w {
                *v /= center_val;
            }
        }
    }

    w
}

/// Normalization modes for DPSS windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpssNorm {
    /// Preserve the L2-normalized eigenvectors from the tridiagonal solver.
    L2,
    /// Match SciPy's default single-window correction for even-length windows.
    Approximate,
    /// Apply the FFT-based subsample correction used by SciPy.
    Subsample,
}

/// Result bundle for `dpss`.
#[derive(Debug, Clone, PartialEq)]
pub struct DpssResult {
    /// DPSS tapers. `kmax=None` still returns a single taper at `windows[0]`.
    pub windows: Vec<Vec<f64>>,
    /// Concentration ratios when requested.
    pub ratios: Option<Vec<f64>>,
}

fn normalized_sinc(x: f64) -> f64 {
    if x.abs() < 1.0e-14 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

fn dpss_concentration_ratio(window: &[f64], bandwidth: f64) -> f64 {
    let mut ratio = 0.0;
    for lag in 0..window.len() {
        let autocorr = (0..window.len() - lag)
            .map(|idx| window[idx] * window[idx + lag])
            .sum::<f64>();
        let kernel = if lag == 0 {
            2.0 * bandwidth
        } else {
            4.0 * bandwidth * normalized_sinc(2.0 * bandwidth * lag as f64)
        };
        ratio += autocorr * kernel;
    }
    ratio
}

fn dpss_subsample_correction(window: &[f64]) -> Result<f64, SignalError> {
    let spectrum = fsci_fft::rfft(window, &fsci_fft::FftOptions::default())
        .map_err(|err| SignalError::InvalidArgument(format!("dpss subsample FFT failed: {err}")))?;
    let n = window.len() as f64;
    let mut real_sum = spectrum[0].0;
    for (k, &(re, im)) in spectrum.iter().enumerate().skip(1) {
        let shift = -(1.0 - 1.0 / n) * k as f64;
        let angle = -std::f64::consts::PI * shift;
        let scale_re = 2.0 * angle.cos();
        let scale_im = 2.0 * angle.sin();
        real_sum += re * scale_re - im * scale_im;
    }
    if real_sum.abs() < 1.0e-15 {
        return Err(SignalError::InvalidArgument(
            "dpss subsample correction became singular".to_string(),
        ));
    }
    Ok(n / real_sum)
}

/// Discrete prolate spheroidal sequences (Slepian tapers).
///
/// Matches `scipy.signal.windows.dpss(M, NW, Kmax=None, sym=True, norm=None, return_ratios=False)`.
pub fn dpss(
    n: usize,
    nw: f64,
    kmax: Option<usize>,
    sym: bool,
    norm: Option<DpssNorm>,
    return_ratios: bool,
) -> Result<DpssResult, SignalError> {
    let norm = norm.unwrap_or(if kmax.is_none() {
        DpssNorm::Approximate
    } else {
        DpssNorm::L2
    });
    let k = kmax.unwrap_or(1);

    if n <= 1 {
        let window = if n == 0 { Vec::new() } else { vec![1.0] };
        return Ok(DpssResult {
            windows: vec![window],
            ratios: return_ratios.then(|| vec![1.0]),
        });
    }
    if k == 0 || k > n {
        return Err(SignalError::InvalidArgument(
            "kmax must be greater than 0 and less than or equal to M".to_string(),
        ));
    }
    if !nw.is_finite() || nw <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "NW must be positive".to_string(),
        ));
    }
    if nw >= n as f64 / 2.0 {
        return Err(SignalError::InvalidArgument(
            "NW must be less than M/2.".to_string(),
        ));
    }

    let work_n = if sym {
        n
    } else {
        n.checked_add(1).ok_or_else(|| {
            SignalError::InvalidArgument(format!(
                "window length overflow while building periodic DPSS for n={n}"
            ))
        })?
    };
    let bandwidth = nw / work_n as f64;
    let cosine = (2.0 * std::f64::consts::PI * bandwidth).cos();
    let diagonal: Vec<f64> = (0..work_n)
        .map(|idx| {
            let centered = (work_n as f64 - 1.0 - 2.0 * idx as f64) / 2.0;
            centered * centered * cosine
        })
        .collect();
    let off_diagonal: Vec<f64> = (1..work_n)
        .map(|idx| idx as f64 * (work_n - idx) as f64 / 2.0)
        .collect();

    // Materialize the symmetric tridiagonal matrix and use the dense symmetric
    // eigensolver. The dedicated tridiagonal path is not yet numerically
    // compatible enough for SciPy DPSS parity.
    let mut tridiagonal = vec![vec![0.0; work_n]; work_n];
    for idx in 0..work_n {
        tridiagonal[idx][idx] = diagonal[idx];
        if idx + 1 < work_n {
            tridiagonal[idx][idx + 1] = off_diagonal[idx];
            tridiagonal[idx + 1][idx] = off_diagonal[idx];
        }
    }
    let eigh = fsci_linalg::eigh(&tridiagonal, fsci_linalg::DecompOptions::default())
        .map_err(|err| SignalError::InvalidArgument(format!("dpss eigensolve failed: {err}")))?;

    let mut windows: Vec<Vec<f64>> = (work_n - k..work_n)
        .rev()
        .map(|col| {
            eigh.eigenvectors
                .iter()
                .map(|row| row[col])
                .collect::<Vec<_>>()
        })
        .collect();

    for (idx, window) in windows.iter_mut().enumerate() {
        if idx.is_multiple_of(2) {
            if window.iter().sum::<f64>() < 0.0 {
                for value in window {
                    *value = -*value;
                }
            }
        } else {
            let threshold = (1.0 / work_n as f64).max(1.0e-7);
            if window
                .iter()
                .find(|value| **value * **value > threshold)
                .is_some_and(|value| *value < 0.0)
            {
                for value in window {
                    *value = -*value;
                }
            }
        }
    }

    let ratios = if return_ratios {
        Some(
            windows
                .iter()
                .map(|window| dpss_concentration_ratio(window, bandwidth))
                .collect(),
        )
    } else {
        None
    };

    if norm != DpssNorm::L2 {
        let max_value = windows
            .iter()
            .flat_map(|window| window.iter().copied())
            .fold(f64::NEG_INFINITY, f64::max);
        if max_value.is_finite() && max_value > 0.0 {
            for window in &mut windows {
                for value in window {
                    *value /= max_value;
                }
            }
        }
        if work_n.is_multiple_of(2) {
            let correction = match norm {
                DpssNorm::Approximate => (work_n * work_n) as f64 / ((work_n * work_n) as f64 + nw),
                DpssNorm::Subsample => dpss_subsample_correction(&windows[0])?,
                DpssNorm::L2 => 1.0,
            };
            for window in &mut windows {
                for value in window {
                    *value *= correction;
                }
            }
        }
    }

    if !sym {
        for window in &mut windows {
            window.truncate(n);
        }
    }
    Ok(DpssResult { windows, ratios })
}

/// Triangular window.
///
/// Matches `scipy.signal.windows.triang(n)`.
pub fn triang(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nf = n as f64;
    (0..n)
        .map(|i| {
            if n.is_multiple_of(2) {
                // Even: peak between two center samples
                let k = i as f64;
                1.0 - ((k - (nf - 1.0) / 2.0) / (nf / 2.0)).abs()
            } else {
                // Odd: peak at center
                let k = i as f64;
                1.0 - ((k - (nf - 1.0) / 2.0) / ((nf + 1.0) / 2.0)).abs()
            }
        })
        .collect()
}

fn periodic_from_symmetric<F>(nx: usize, build: F) -> Result<Vec<f64>, SignalError>
where
    F: FnOnce(usize) -> Vec<f64>,
{
    if nx <= 1 {
        return Ok(build(nx));
    }
    let expanded = nx.checked_add(1).ok_or_else(|| {
        SignalError::InvalidArgument(format!(
            "window length overflow while building periodic form for nx={nx}"
        ))
    })?;
    let mut window = build(expanded);
    window.truncate(nx);
    Ok(window)
}

fn symmetric_or_periodic_window<F>(nx: usize, sym: bool, build: F) -> Result<Vec<f64>, SignalError>
where
    F: FnOnce(usize) -> Vec<f64>,
{
    if sym {
        Ok(build(nx))
    } else {
        periodic_from_symmetric(nx, build)
    }
}

/// Dispatch a window function by name string using SciPy's default periodic policy.
///
/// Matches `scipy.signal.get_window(window, Nx)` with `fftbins=True`.
pub fn get_window(window: &str, nx: usize) -> Result<Vec<f64>, SignalError> {
    get_window_with_fftbins(window, nx, true)
}

fn parse_window_bool(value: &str, label: &str) -> Result<bool, SignalError> {
    match value.trim() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        invalid => Err(SignalError::InvalidArgument(format!(
            "invalid {label}: {invalid}"
        ))),
    }
}

fn parse_finite_window_param(value: &str, label: &str) -> Result<f64, SignalError> {
    let parsed = value
        .trim()
        .parse::<f64>()
        .map_err(|_| SignalError::InvalidArgument(format!("invalid {label}: {value}")))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(SignalError::InvalidArgument(format!(
            "invalid {label}: {value}"
        )))
    }
}

/// Dispatch a window function by name string with explicit SciPy `fftbins` control.
///
/// `fftbins=true` returns the periodic form used for FFT analysis. `fftbins=false`
/// returns the symmetric form used for filter design. A `_periodic` or
/// `_symmetric` suffix on the window name overrides the flag.
///
/// # Supported windows
/// `"hann"`, `"hamming"`, `"blackman"`, `"blackmanharris"`, `"barthann"`,
/// `"bartlett"`, `"flattop"`, `"cosine"`, `"rectangular"` / `"boxcar"`,
/// `"lanczos"` / `"sinc"`,
/// `"kaiser,<beta>"` (e.g. `"kaiser,8.6"`), `"chebwin,<at>"` (e.g. `"chebwin,100"`),
/// `"dpss,<nw>"` (e.g. `"dpss,2.5"`),
/// `"general_hamming,<alpha>"` (e.g. `"general_hamming,0.75"`),
/// `"general_cosine,<a0>,<a1>,..."` (e.g. `"general_cosine,0.5,0.5"`),
/// `"gaussian,<std>"` (e.g. `"gaussian,2.0"`),
/// `"general_gaussian,<p>,<sig>"` (e.g. `"general_gaussian,1.5,2.0"`),
/// `"exponential,<tau>"` (e.g. `"exponential,2.0"`),
/// `"taylor"` / `"taylorwin"` with optional `"taylor,<nbar>[,<sll>[,<norm>]]"`
/// parameters (e.g. `"taylor,6,45,false"`).
pub fn get_window_with_fftbins(
    window: &str,
    nx: usize,
    fftbins: bool,
) -> Result<Vec<f64>, SignalError> {
    let lower = window.trim().to_lowercase();
    let (raw_name, raw_params) = lower
        .split_once(',')
        .map_or((lower.as_str(), None), |(name, params)| {
            (name.trim(), Some(params.trim()))
        });

    let mut sym = !fftbins;
    let win_name = if let Some(name) = raw_name.strip_suffix("_symmetric") {
        sym = true;
        name
    } else if let Some(name) = raw_name.strip_suffix("_periodic") {
        sym = false;
        name
    } else {
        raw_name
    };

    if let Some(rest) = raw_params.filter(|rest| !rest.is_empty()) {
        match win_name {
            "kaiser" => {
                let beta = parse_finite_window_param(rest, "kaiser beta")?;
                return symmetric_or_periodic_window(nx, sym, |n| kaiser(n, beta));
            }
            "kaiser_bessel_derived" | "kbd" => {
                let beta = parse_finite_window_param(rest, "kaiser_bessel_derived beta")?;
                return kaiser_bessel_derived(nx, beta, sym);
            }
            "dpss" => {
                let nw = parse_finite_window_param(rest, "dpss NW")?;
                return dpss(nx, nw, None, sym, None, false)
                    .map(|result| result.windows.into_iter().next().unwrap_or_default());
            }
            "chebwin" => {
                let attenuation = parse_finite_window_param(rest, "chebwin attenuation")?;
                return symmetric_or_periodic_window(nx, sym, |n| chebwin(n, attenuation));
            }
            "general_hamming" => {
                let alpha = parse_finite_window_param(rest, "general_hamming alpha")?;
                return symmetric_or_periodic_window(nx, sym, |n| general_hamming(n, alpha));
            }
            "general_cosine" => {
                let coeffs: Vec<f64> = rest
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(|value| parse_finite_window_param(value, "general_cosine coefficient"))
                    .collect::<Result<_, _>>()?;
                return Ok(general_cosine(nx, &coeffs, sym));
            }
            "gaussian" => {
                let std = parse_finite_window_param(rest, "gaussian std")?;
                return Ok(gaussian(nx, std, sym));
            }
            "general_gaussian" => {
                let params: Vec<&str> = rest
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .collect();
                if params.len() != 2 {
                    return Err(SignalError::InvalidArgument(format!(
                        "general_gaussian expects p,sig parameters: {rest}"
                    )));
                }
                let p = parse_finite_window_param(params[0], "general_gaussian shape parameter")?;
                let sig = parse_finite_window_param(params[1], "general_gaussian sigma parameter")?;
                return Ok(general_gaussian(nx, p, sig, sym));
            }
            "exponential" => {
                let tau = parse_finite_window_param(rest, "exponential tau")?;
                return exponential(nx, None, tau, sym);
            }
            "taylor" | "taylorwin" => {
                let params: Vec<&str> = rest.split(',').map(str::trim).collect();
                if params.is_empty()
                    || params.len() > 3
                    || params.iter().any(|value| value.is_empty())
                {
                    return Err(SignalError::InvalidArgument(format!(
                        "taylor expects nbar[,sll[,norm]] parameters: {rest}"
                    )));
                }
                let nbar = params[0].parse::<usize>().map_err(|_| {
                    SignalError::InvalidArgument(format!("invalid taylor nbar: {}", params[0]))
                })?;
                let sll = params.get(1).map_or(Ok(30.0), |value| {
                    parse_finite_window_param(value, "taylor sll")
                })?;
                let norm = params
                    .get(2)
                    .map_or(Ok(true), |value| parse_window_bool(value, "taylor norm"))?;
                return Ok(taylor(nx, nbar, sll, norm, sym));
            }
            _ => {}
        }
    }

    match win_name {
        "hann" | "hanning" => symmetric_or_periodic_window(nx, sym, hann),
        "hamming" => symmetric_or_periodic_window(nx, sym, hamming),
        "blackman" => symmetric_or_periodic_window(nx, sym, blackman),
        "blackmanharris" => symmetric_or_periodic_window(nx, sym, blackmanharris),
        "barthann" => symmetric_or_periodic_window(nx, sym, barthann),
        "bartlett" | "triangle" => symmetric_or_periodic_window(nx, sym, bartlett),
        "flattop" => symmetric_or_periodic_window(nx, sym, flattop),
        "cosine" => symmetric_or_periodic_window(nx, sym, cosine),
        "lanczos" | "sinc" => symmetric_or_periodic_window(nx, sym, lanczos),
        "rectangular" | "boxcar" | "rect" => Ok(boxcar(nx, sym)),
        "chebwin" => symmetric_or_periodic_window(nx, sym, |n| chebwin(n, 100.0)),
        "parzen" => symmetric_or_periodic_window(nx, sym, parzen),
        "triang" => symmetric_or_periodic_window(nx, sym, triang),
        "tukey" => symmetric_or_periodic_window(nx, sym, |n| tukey_window(n, 0.5)),
        "nuttall" => symmetric_or_periodic_window(nx, sym, nuttall_window),
        "bohman" => symmetric_or_periodic_window(nx, sym, bohman_window),
        "exponential" => exponential(nx, None, 1.0, sym),
        "taylor" | "taylorwin" => Ok(taylor(nx, 4, 30.0, true, sym)),
        "kaiser_bessel_derived" | "kbd" => kaiser_bessel_derived(nx, 14.0, sym),
        "dpss" => Err(SignalError::InvalidArgument(
            "dpss requires a normalized half-bandwidth parameter".to_string(),
        )),
        _ => Err(SignalError::InvalidArgument(format!(
            "unknown window type: {window}"
        ))),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Time-Frequency Analysis: STFT, ISTFT, spectrogram, CSD, coherence
// ══════════════════════════════════════════════════════════════════════

/// Result of the Short-Time Fourier Transform.
#[derive(Debug, Clone)]
pub struct StftResult {
    /// Frequency bins (length = nperseg/2 + 1).
    pub frequencies: Vec<f64>,
    /// Time centers for each segment.
    pub times: Vec<f64>,
    /// Complex STFT matrix: `zxx[t][f] = (re, im)`.
    /// Outer index is time segment, inner is frequency bin.
    pub zxx: Vec<Vec<(f64, f64)>>,
}

/// Compute the Short-Time Fourier Transform.
///
/// Matches `scipy.signal.stft(x, fs, window, nperseg, noverlap)`.
///
/// # Arguments
/// * `x` — Input signal.
/// * `fs` — Sampling frequency (Hz).
/// * `window` — Window type to use (default: "hann").
/// * `nperseg` — Length of each segment (default: 256).
/// * `noverlap` — Overlap between segments (default: nperseg/2).
pub fn stft(
    x: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
) -> Result<StftResult, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }

    let nperseg = nperseg.unwrap_or_else(|| x.len().min(256));
    if nperseg == 0 {
        return Err(SignalError::InvalidArgument(
            "nperseg must be > 0".to_string(),
        ));
    }
    if nperseg > x.len() {
        return Err(SignalError::InvalidArgument(format!(
            "nperseg ({nperseg}) must be <= signal length ({})",
            x.len()
        )));
    }

    let noverlap = noverlap.unwrap_or(nperseg / 2);
    if noverlap >= nperseg {
        return Err(SignalError::InvalidArgument(
            "noverlap must be < nperseg".to_string(),
        ));
    }

    let step = nperseg - noverlap;
    let win_coeffs = get_window(window.unwrap_or("hann"), nperseg)?;
    let n_freqs = nperseg / 2 + 1;
    let opts = fsci_fft::FftOptions::default();

    let mut zxx = Vec::new();
    let mut times = Vec::new();
    let mut start = 0;

    while start + nperseg <= x.len() {
        // Window the segment.
        let windowed: Vec<f64> = x[start..start + nperseg]
            .iter()
            .zip(&win_coeffs)
            .map(|(&xi, &wi)| xi * wi)
            .collect();

        // Compute rfft.
        let spectrum = fsci_fft::rfft(&windowed, &opts)
            .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;

        zxx.push(spectrum[..n_freqs].to_vec());
        times.push((start as f64 + (nperseg - 1) as f64 / 2.0) / fs);
        start += step;
    }

    let freq_step = fs / nperseg as f64;
    let frequencies: Vec<f64> = (0..n_freqs).map(|k| k as f64 * freq_step).collect();

    Ok(StftResult {
        frequencies,
        times,
        zxx,
    })
}

/// Compute the inverse Short-Time Fourier Transform (overlap-add reconstruction).
///
/// Matches `scipy.signal.istft(Zxx, fs, nperseg, noverlap)`.
///
/// # Arguments
/// * `stft_result` — STFT result from `stft()`.
/// * `nperseg` — Segment length used in the forward STFT.
/// * `noverlap` — Overlap used in the forward STFT (default: nperseg/2).
pub fn istft(
    stft_result: &StftResult,
    nperseg: usize,
    noverlap: Option<usize>,
) -> Result<Vec<f64>, SignalError> {
    if stft_result.zxx.is_empty() {
        return Err(SignalError::InvalidArgument(
            "empty STFT result".to_string(),
        ));
    }

    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let step = nperseg - noverlap;
    let n_segments = stft_result.zxx.len();
    let output_len = nperseg + (n_segments - 1) * step;

    let window = hann(nperseg);
    let opts = fsci_fft::FftOptions::default();

    let mut output = vec![0.0; output_len];
    let mut window_sum = vec![0.0; output_len];

    for (seg_idx, spectrum) in stft_result.zxx.iter().enumerate() {
        // Inverse rfft.
        let segment = fsci_fft::irfft(spectrum, Some(nperseg), &opts)
            .map_err(|e| SignalError::InvalidArgument(format!("IFFT failed: {e}")))?;

        let start = seg_idx * step;
        for (j, (&s, &w)) in segment.iter().zip(&window).enumerate() {
            output[start + j] += s * w;
            window_sum[start + j] += w * w;
        }
    }

    // Normalize by window sum (COLA condition).
    for (o, &ws) in output.iter_mut().zip(&window_sum) {
        if ws > 1e-15 {
            *o /= ws;
        }
    }

    Ok(output)
}

/// Result of the spectrogram computation.
#[derive(Debug, Clone)]
pub struct SpectrogramResult {
    /// Frequency bins.
    pub frequencies: Vec<f64>,
    /// Time centers for each segment.
    pub times: Vec<f64>,
    /// Power spectral density: `sxx[t][f]`.
    pub sxx: Vec<Vec<f64>>,
}

/// Compute a spectrogram (time-frequency power representation).
///
/// Matches `scipy.signal.spectrogram(x, fs, nperseg, noverlap)`.
///
/// # Arguments
/// * `x` — Input signal.
/// * `fs` — Sampling frequency (Hz).
/// * `window` — Window type to use (default: "hann").
/// * `nperseg` — Length of each segment (default: 256).
/// * `noverlap` — Overlap between segments (default: nperseg/8).
pub fn spectrogram(
    x: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
) -> Result<SpectrogramResult, SignalError> {
    let nperseg_val = nperseg.unwrap_or_else(|| x.len().min(256));
    let noverlap_val = noverlap.unwrap_or(nperseg_val / 8);

    let stft_res = stft(x, fs, window, Some(nperseg_val), Some(noverlap_val))?;

    let n_freqs = stft_res.frequencies.len();
    let win_coeffs = get_window(window.unwrap_or("hann"), nperseg_val)?;
    let win_power: f64 = win_coeffs.iter().map(|&w| w * w).sum::<f64>() / nperseg_val as f64;

    // Convert complex STFT to PSD.
    let scale = 1.0 / (fs * nperseg_val as f64 * win_power);
    let sxx: Vec<Vec<f64>> = stft_res
        .zxx
        .iter()
        .map(|seg| {
            seg.iter()
                .enumerate()
                .map(|(k, &(re, im))| {
                    let mag2 = re * re + im * im;
                    let factor = if k == 0 || (nperseg_val.is_multiple_of(2) && k == n_freqs - 1) {
                        1.0
                    } else {
                        2.0
                    };
                    mag2 * scale * factor
                })
                .collect()
        })
        .collect();

    Ok(SpectrogramResult {
        frequencies: stft_res.frequencies,
        times: stft_res.times,
        sxx,
    })
}

/// Cross-spectral density estimation using Welch's method.
///
/// Matches `scipy.signal.csd(x, y, fs, nperseg, noverlap)`.
///
/// Returns complex cross-spectral density `Pxy[k] = conj(X[k]) * Y[k]` averaged
/// over segments.
///
/// # Arguments
/// * `x` — First input signal.
/// * `y` — Second input signal (same length as x).
/// * `fs` — Sampling frequency (Hz).
/// * `window` — Window type to use (default: "hann").
/// * `nperseg` — Segment length (default: 256).
/// * `noverlap` — Overlap (default: nperseg/2).
pub fn csd(
    x: &[f64],
    y: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
) -> Result<CsdResult, SignalError> {
    csd_with_scaling(
        x,
        y,
        fs,
        window,
        nperseg,
        noverlap,
        SpectralScaling::Density,
    )
}

/// Cross-spectral density / spectrum estimation using Welch's method.
///
/// Matches `scipy.signal.csd(..., scaling='density'|'spectrum')`.
pub fn csd_with_scaling(
    x: &[f64],
    y: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    scaling: SpectralScaling,
) -> Result<CsdResult, SignalError> {
    if x.len() != y.len() {
        return Err(SignalError::InvalidArgument(format!(
            "x and y must have same length ({} vs {})",
            x.len(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }

    let nperseg = nperseg.unwrap_or_else(|| x.len().min(256));
    if nperseg == 0 || nperseg > x.len() {
        return Err(SignalError::InvalidArgument(
            "nperseg must be > 0 and <= signal length".to_string(),
        ));
    }
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    if noverlap >= nperseg {
        return Err(SignalError::InvalidArgument(
            "noverlap must be < nperseg".to_string(),
        ));
    }

    let step = nperseg - noverlap;
    let win_coeffs = get_window(window.unwrap_or("hann"), nperseg)?;
    let win_power: f64 = win_coeffs.iter().map(|&w| w * w).sum::<f64>() / nperseg as f64;
    let n_freqs = nperseg / 2 + 1;
    let opts = fsci_fft::FftOptions::default();

    let mut avg_csd = vec![(0.0, 0.0); n_freqs];
    let mut n_segments = 0usize;
    let mut start = 0;

    while start + nperseg <= x.len() {
        let wx: Vec<f64> = x[start..start + nperseg]
            .iter()
            .zip(&win_coeffs)
            .map(|(&xi, &wi)| xi * wi)
            .collect();
        let wy: Vec<f64> = y[start..start + nperseg]
            .iter()
            .zip(&win_coeffs)
            .map(|(&yi, &wi)| yi * wi)
            .collect();

        let sx = fsci_fft::rfft(&wx, &opts)
            .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;
        let sy = fsci_fft::rfft(&wy, &opts)
            .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;

        // Pxy = conj(X) * Y
        for (k, ((avg_re, avg_im), (&(xr, xi), &(yr, yi)))) in
            avg_csd.iter_mut().zip(sx.iter().zip(sy.iter())).enumerate()
        {
            // conj(X) * Y = (xr - j*xi) * (yr + j*yi) = (xr*yr + xi*yi) + j*(xr*yi - xi*yr)
            let re = xr * yr + xi * yi;
            let im = xr * yi - xi * yr;
            let factor = if k == 0 || (nperseg.is_multiple_of(2) && k == n_freqs - 1) {
                1.0
            } else {
                2.0
            };
            *avg_re += re * factor;
            *avg_im += im * factor;
        }
        n_segments += 1;
        start += step;
    }

    if n_segments == 0 {
        return Err(SignalError::InvalidArgument(
            "signal too short for any segment".to_string(),
        ));
    }

    let scale = match scaling {
        SpectralScaling::Density => 1.0 / (fs * nperseg as f64 * win_power * n_segments as f64),
        SpectralScaling::Spectrum => {
            let win_sum = win_coeffs.iter().sum::<f64>();
            1.0 / (win_sum * win_sum * n_segments as f64)
        }
    };
    for (re, im) in &mut avg_csd {
        *re *= scale;
        *im *= scale;
    }

    let freq_step = fs / nperseg as f64;
    let frequencies: Vec<f64> = (0..n_freqs).map(|k| k as f64 * freq_step).collect();

    Ok(CsdResult {
        frequencies,
        csd: avg_csd,
    })
}

/// Result of cross-spectral density computation.
#[derive(Debug, Clone)]
pub struct CsdResult {
    /// Frequency bins.
    pub frequencies: Vec<f64>,
    /// Complex cross-spectral density: `(re, im)` per frequency bin.
    pub csd: Vec<(f64, f64)>,
}

/// Result of coherence computation.
#[derive(Debug, Clone)]
pub struct CoherenceResult {
    /// Frequency bins.
    pub frequencies: Vec<f64>,
    /// Magnitude-squared coherence: `|Pxy|² / (Pxx * Pyy)` per frequency bin.
    pub coherence: Vec<f64>,
}

/// Compute magnitude-squared coherence between two signals.
///
/// Matches `scipy.signal.coherence(x, y, fs, nperseg, noverlap)`.
///
/// `Cxy[f] = |Pxy[f]|² / (Pxx[f] * Pyy[f])` where Pxy is the cross-spectral
/// density and Pxx, Pyy are the auto-spectral densities.
///
/// # Arguments
/// * `x` — First input signal.
/// * `y` — Second input signal (same length as x).
/// * `fs` — Sampling frequency (Hz).
/// * `window` — Window type to use (default: "hann").
/// * `nperseg` — Segment length (default: 256).
/// * `noverlap` — Overlap (default: nperseg/2).
pub fn coherence(
    x: &[f64],
    y: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
) -> Result<CoherenceResult, SignalError> {
    let pxy = csd(x, y, fs, window, nperseg, noverlap)?;
    let pxx = csd(x, x, fs, window, nperseg, noverlap)?;
    let pyy = csd(y, y, fs, window, nperseg, noverlap)?;

    let coh: Vec<f64> = pxy
        .csd
        .iter()
        .zip(pxx.csd.iter().zip(pyy.csd.iter()))
        .map(|(&(pxy_re, pxy_im), (&(pxx_re, _), &(pyy_re, _)))| {
            // |Pxy|² / (Pxx * Pyy). Auto-spectra are real (imaginary ≈ 0).
            let pxy_mag2 = pxy_re * pxy_re + pxy_im * pxy_im;
            let denom = pxx_re * pyy_re;
            if denom.abs() < 1e-30 {
                0.0
            } else {
                (pxy_mag2 / denom).clamp(0.0, 1.0)
            }
        })
        .collect();

    Ok(CoherenceResult {
        frequencies: pxy.frequencies,
        coherence: coh,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Signal Resampling: resample, resample_poly, decimate
// ══════════════════════════════════════════════════════════════════════

/// Resample a signal using the FFT method.
///
/// Matches `scipy.signal.resample(x, num)`.
///
/// Changes the number of samples in a signal by zero-padding or truncating
/// in the frequency domain.
///
/// # Arguments
/// * `x` — Input signal.
/// * `num` — Desired number of output samples.
pub fn resample(x: &[f64], num: usize) -> Result<Vec<f64>, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }
    if num == 0 {
        return Err(SignalError::InvalidArgument("num must be > 0".to_string()));
    }

    let n = x.len();
    if num == n {
        return Ok(x.to_vec());
    }

    let opts = fsci_fft::FftOptions::default();
    let m = n.min(num);
    let m2 = m / 2 + 1;

    // Forward FFT.
    let spectrum = fsci_fft::rfft(x, &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;

    // Compute target spectrum length for the new sample count.
    let target_nfreqs = num / 2 + 1;

    // Zero-pad or truncate the spectrum, keeping only the bins that are
    // representable in both the input and output grids.
    let mut new_spectrum = vec![(0.0, 0.0); target_nfreqs];
    new_spectrum[..m2].copy_from_slice(&spectrum[..m2]);

    // Match SciPy's treatment of the unpaired Nyquist bin when the smaller
    // of the input/output lengths is even and the lengths differ.
    if m.is_multiple_of(2) {
        let nyquist = m / 2;
        if num < n {
            new_spectrum[nyquist].0 *= 2.0;
            new_spectrum[nyquist].1 *= 2.0;
        } else {
            new_spectrum[nyquist].0 *= 0.5;
            new_spectrum[nyquist].1 *= 0.5;
        }
    }

    // Inverse FFT with target length.
    let result = fsci_fft::irfft(&new_spectrum, Some(num), &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("IFFT failed: {e}")))?;

    // Scale by num/n to preserve amplitude.
    let scale = num as f64 / n as f64;
    Ok(result.into_iter().map(|v| v * scale).collect())
}

/// Resample a signal using polyphase filtering (rational rate change).
///
/// Matches `scipy.signal.resample_poly(x, up, down)`.
///
/// Resamples by factor `up/down` using an anti-aliasing FIR filter.
///
/// # Arguments
/// * `x` — Input signal.
/// * `up` — Upsampling factor.
/// * `down` — Downsampling factor.
pub fn resample_poly(x: &[f64], up: usize, down: usize) -> Result<Vec<f64>, SignalError> {
    resample_poly_with_padtype(x, up, down, None, None)
}

#[derive(Clone, Copy, Debug)]
enum ResamplePolyPadMode {
    Constant(f64),
    Background(f64),
    Edge,
    Reflect,
    Symmetric,
    Wrap,
    Line { start: f64, slope: f64 },
}

fn resample_poly_reflected_index(idx: i64, len: usize, symmetric: bool) -> usize {
    if len <= 1 {
        return 0;
    }

    let len = len as i64;
    let period = if symmetric { 2 * len } else { 2 * len - 2 };
    let idx = idx.rem_euclid(period);
    if symmetric {
        if idx < len {
            idx as usize
        } else {
            (period - idx - 1) as usize
        }
    } else if idx < len {
        idx as usize
    } else {
        (period - idx) as usize
    }
}

fn resample_poly_median(x: &[f64]) -> f64 {
    let mut values = x.to_vec();
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn parse_resample_poly_pad_mode(
    x: &[f64],
    padtype: Option<&str>,
    cval: Option<f64>,
) -> Result<ResamplePolyPadMode, SignalError> {
    let padtype = padtype.unwrap_or("constant");
    if cval.is_some() && padtype != "constant" {
        return Err(SignalError::InvalidArgument(format!(
            "cval has no effect when padtype is {padtype}"
        )));
    }

    let constant = |value: f64| Ok(ResamplePolyPadMode::Constant(value));
    match padtype {
        "constant" => constant(cval.unwrap_or(0.0)),
        "mean" => Ok(ResamplePolyPadMode::Background(
            x.iter().sum::<f64>() / x.len() as f64,
        )),
        "median" => Ok(ResamplePolyPadMode::Background(resample_poly_median(x))),
        "maximum" => Ok(ResamplePolyPadMode::Background(
            x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )),
        "minimum" => Ok(ResamplePolyPadMode::Background(
            x.iter().copied().fold(f64::INFINITY, f64::min),
        )),
        "edge" => Ok(ResamplePolyPadMode::Edge),
        "reflect" => Ok(ResamplePolyPadMode::Reflect),
        "symmetric" => Ok(ResamplePolyPadMode::Symmetric),
        "wrap" => Ok(ResamplePolyPadMode::Wrap),
        "line" => {
            let slope = if x.len() > 1 {
                (x[x.len() - 1] - x[0]) / (x.len() - 1) as f64
            } else {
                0.0
            };
            Ok(ResamplePolyPadMode::Line { start: x[0], slope })
        }
        _ => Err(SignalError::InvalidArgument(
            "padtype must be one of {'constant', 'line', 'mean', 'median', \
             'maximum', 'minimum', 'edge', 'reflect', 'symmetric', 'wrap'}"
                .to_string(),
        )),
    }
}

fn resample_poly_sample(x: &[f64], idx: i64, mode: ResamplePolyPadMode) -> f64 {
    if idx >= 0 && idx < x.len() as i64 {
        return x[idx as usize];
    }

    match mode {
        ResamplePolyPadMode::Constant(value) => value,
        ResamplePolyPadMode::Background(_) => 0.0,
        ResamplePolyPadMode::Edge => {
            if idx < 0 {
                x[0]
            } else {
                x[x.len() - 1]
            }
        }
        ResamplePolyPadMode::Reflect => x[resample_poly_reflected_index(idx, x.len(), false)],
        ResamplePolyPadMode::Symmetric => x[resample_poly_reflected_index(idx, x.len(), true)],
        ResamplePolyPadMode::Wrap => x[idx.rem_euclid(x.len() as i64) as usize],
        ResamplePolyPadMode::Line { start, slope } => start + slope * idx as f64,
    }
}

/// Resample a signal using polyphase filtering with SciPy-style boundary modes.
///
/// Matches `scipy.signal.resample_poly(..., padtype=..., cval=...)` for the
/// implemented 1-D modes.
pub fn resample_poly_with_padtype(
    x: &[f64],
    up: usize,
    down: usize,
    padtype: Option<&str>,
    cval: Option<f64>,
) -> Result<Vec<f64>, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }
    if up == 0 || down == 0 {
        return Err(SignalError::InvalidArgument(
            "up and down must be > 0".to_string(),
        ));
    }
    let parsed_mode = parse_resample_poly_pad_mode(x, padtype, cval)?;
    let (background, pad_mode, input) = match parsed_mode {
        ResamplePolyPadMode::Background(value) => (
            value,
            ResamplePolyPadMode::Constant(0.0),
            x.iter().map(|sample| sample - value).collect::<Vec<_>>(),
        ),
        mode => (0.0, mode, x.to_vec()),
    };

    // Simplify the ratio using GCD.
    let g = gcd(up, down);
    let up = up / g;
    let down = down / g;

    if up == 1 && down == 1 {
        return Ok(input.iter().map(|sample| sample + background).collect());
    }

    // Design anti-aliasing FIR filter.
    let cutoff = 1.0 / (up.max(down) as f64);
    let n_taps = 2 * 10 * up.max(down) + 1; // ~10 periods per side
    let h = firwin(n_taps, &[cutoff], FirWindow::Kaiser(5.0), true)?;
    // Scale filter by up to compensate for upsampling gain.
    let h_scaled: Vec<f64> = h.iter().map(|&v| v * up as f64).collect();

    // Polyphase implementation to avoid materializing the full upsampled signal.
    // Equivalent to: upsample -> convolve(mode=Same) -> downsample.
    let upsampled_len = input.len() * up;
    let half_taps = (n_taps - 1) as i64 / 2;
    let mut output = Vec::with_capacity(upsampled_len.div_ceil(down));

    let mut i = 0usize;
    while i < upsampled_len {
        let mut val = 0.0;
        let target = i as i64 + half_taps;

        // k must satisfy: 0 <= k < n_taps AND (target - k) is a multiple of 'up'.
        let k_start = (target % up as i64 + up as i64) % up as i64;
        for k in (k_start as usize..n_taps).step_by(up) {
            let x_idx = (target - k as i64) / up as i64;
            val += h_scaled[k] * resample_poly_sample(&input, x_idx, pad_mode);
        }
        output.push(val + background);
        i += down;
    }

    Ok(output)
}

/// Upsample, FIR filter, and downsample a 1-D signal.
///
/// Matches `scipy.signal.upfirdn(h, x, up, down)` for the default 1-D case.
pub fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Result<Vec<f64>, SignalError> {
    if h.is_empty() {
        return Err(SignalError::InvalidArgument(
            "h must be 1-D with non-zero length".to_string(),
        ));
    }
    if up == 0 || down == 0 {
        return Err(SignalError::InvalidArgument(
            "up and down must be >= 1".to_string(),
        ));
    }
    if x.is_empty() {
        return Ok(Vec::new());
    }

    let upsampled_len = (x.len() - 1) * up + 1;
    let full_len = upsampled_len + h.len() - 1;
    let mut output = vec![0.0; full_len];

    for (i, &sample) in x.iter().enumerate() {
        let base = i * up;
        for (tap_idx, &tap) in h.iter().enumerate() {
            output[base + tap_idx] += sample * tap;
        }
    }

    Ok(output.into_iter().step_by(down).collect())
}

/// Convert a linear-phase FIR filter to minimum phase.
///
/// Matches `scipy.signal.minimum_phase(h)` for the default homomorphic method
/// with `half=True`.
pub fn minimum_phase(h: &[f64]) -> Result<Vec<f64>, SignalError> {
    if h.len() <= 2 {
        return Err(SignalError::InvalidArgument(
            "h must be 1-D and at least 2 samples long".to_string(),
        ));
    }

    let suggested_fft = ((2 * (h.len() - 1)) as f64 / 0.01).ceil() as usize;
    let n_fft = suggested_fft
        .max(h.len())
        .checked_next_power_of_two()
        .unwrap_or(suggested_fft.max(h.len()));
    let opts = fsci_fft::FftOptions::default();

    let mut padded = vec![(0.0, 0.0); n_fft];
    for (dst, &coeff) in padded.iter_mut().zip(h.iter()) {
        dst.0 = coeff;
    }

    let spectrum = fsci_fft::fft(&padded, &opts)
        .map_err(|err| SignalError::InvalidArgument(format!("minimum_phase FFT failed: {err}")))?;
    let magnitudes: Vec<f64> = spectrum
        .iter()
        .map(|&(re, im)| (re * re + im * im).sqrt())
        .collect();
    let min_positive = magnitudes
        .iter()
        .copied()
        .filter(|&value| value > 0.0)
        .fold(f64::INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        });
    let epsilon = if min_positive.is_finite() {
        1e-7 * min_positive
    } else {
        1e-7
    };

    let log_spectrum: Vec<(f64, f64)> = magnitudes
        .into_iter()
        .map(|value| ((value + epsilon).ln() * 0.5, 0.0))
        .collect();
    let cepstrum = fsci_fft::ifft(&log_spectrum, &opts).map_err(|err| {
        SignalError::InvalidArgument(format!("minimum_phase inverse FFT failed: {err}"))
    })?;

    let mut homomorphic_window = vec![0.0; n_fft];
    homomorphic_window[0] = 1.0;
    let stop = n_fft / 2;
    for value in homomorphic_window.iter_mut().take(stop).skip(1) {
        *value = 2.0;
    }
    if n_fft % 2 == 1 {
        homomorphic_window[stop] = 1.0;
    }

    let weighted_cepstrum: Vec<(f64, f64)> = cepstrum
        .iter()
        .zip(homomorphic_window.iter())
        .map(|(&(re, _), &weight)| (re * weight, 0.0))
        .collect();
    let minimum_log_spectrum = fsci_fft::fft(&weighted_cepstrum, &opts).map_err(|err| {
        SignalError::InvalidArgument(format!("minimum_phase cepstrum FFT failed: {err}"))
    })?;
    let minimum_spectrum: Vec<(f64, f64)> = minimum_log_spectrum
        .into_iter()
        .map(|(re, im)| {
            let exp_re = re.exp();
            (exp_re * im.cos(), exp_re * im.sin())
        })
        .collect();
    let minimum_impulse = fsci_fft::ifft(&minimum_spectrum, &opts).map_err(|err| {
        SignalError::InvalidArgument(format!("minimum_phase reconstruction failed: {err}"))
    })?;

    let n_out = h.len().div_ceil(2);
    Ok(minimum_impulse
        .into_iter()
        .take(n_out)
        .map(|(re, _)| re)
        .collect())
}

/// Downsample a signal after applying an anti-aliasing filter.
///
/// Matches `scipy.signal.decimate(x, q)`.
///
/// Applies a lowpass Butterworth filter at Nyquist/q frequency, then
/// takes every q-th sample.
///
/// # Arguments
/// * `x` — Input signal.
/// * `q` — Downsampling factor (integer >= 2).
pub fn decimate(x: &[f64], q: usize) -> Result<Vec<f64>, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }
    if q < 2 {
        return Err(SignalError::InvalidArgument("q must be >= 2".to_string()));
    }

    // Design lowpass filter at cutoff = 1/q (normalized to Nyquist = 1).
    let cutoff = 1.0 / q as f64;
    let order = 8.min(q); // order 8 max, reduce for small q
    let ba = butter(order, &[cutoff], FilterType::Lowpass)?;

    // Zero-phase filtering for no distortion.
    let filtered = filtfilt(&ba.b, &ba.a, x)?;

    // Downsample.
    let output: Vec<f64> = filtered.iter().step_by(q).copied().collect();
    Ok(output)
}

/// Compute GCD of two positive integers.
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ═══════════════════════════════════════════════════════════════════
// Linear Time-Invariant (LTI) System Classes
// ═══════════════════════════════════════════════════════════════════

/// Continuous-time Linear Time-Invariant system representation.
///
/// Matches `scipy.signal.lti`. Represents a continuous-time LTI system
/// in transfer function form (numerator/denominator polynomials).
///
/// The transfer function is: H(s) = num(s) / den(s)
#[derive(Debug, Clone, PartialEq)]
pub struct Lti {
    /// Numerator polynomial coefficients (highest power first).
    pub num: Vec<f64>,
    /// Denominator polynomial coefficients (highest power first).
    pub den: Vec<f64>,
}

impl Lti {
    /// Create a new continuous-time LTI system from transfer function coefficients.
    ///
    /// # Arguments
    /// * `num` - Numerator polynomial coefficients (highest power first)
    /// * `den` - Denominator polynomial coefficients (highest power first)
    pub fn new(num: Vec<f64>, den: Vec<f64>) -> Result<Self, SignalError> {
        if num.is_empty() {
            return Err(SignalError::InvalidArgument(
                "numerator cannot be empty".to_string(),
            ));
        }
        if den.is_empty() {
            return Err(SignalError::InvalidArgument(
                "denominator cannot be empty".to_string(),
            ));
        }
        if den.iter().all(|&x| x == 0.0) {
            return Err(SignalError::InvalidArgument(
                "denominator cannot be all zeros".to_string(),
            ));
        }
        Ok(Self { num, den })
    }

    /// Create an LTI system from zeros, poles, and gain.
    ///
    /// H(s) = k * prod(s - z_i) / prod(s - p_j)
    pub fn from_zpk(zeros: &[f64], poles: &[f64], gain: f64) -> Result<Self, SignalError> {
        let num = zpk_to_poly(zeros, gain);
        let den = zpk_to_poly(poles, 1.0);
        Self::new(num, den)
    }

    /// Evaluate the transfer function at a complex frequency s.
    pub fn eval_at(&self, s_re: f64, s_im: f64) -> (f64, f64) {
        let num_val = poly_eval_complex(&self.num, s_re, s_im);
        let den_val = poly_eval_complex(&self.den, s_re, s_im);
        complex_div(num_val.0, num_val.1, den_val.0, den_val.1)
    }

    /// Compute the frequency response H(jω) at angular frequencies.
    ///
    /// Returns (magnitude, phase) arrays.
    pub fn freqresp(&self, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut mag = Vec::with_capacity(w.len());
        let mut phase = Vec::with_capacity(w.len());

        for &omega in w {
            let (re, im) = self.eval_at(0.0, omega);
            mag.push((re * re + im * im).sqrt());
            phase.push(im.atan2(re));
        }

        (mag, phase)
    }

    /// Compute the step response of the system.
    ///
    /// Uses numerical simulation with RK4 integration.
    pub fn step(&self, t: &[f64]) -> Result<Vec<f64>, SignalError> {
        if t.is_empty() {
            return Ok(Vec::new());
        }
        // Convert to state-space and simulate
        let (a, b, c, d) = tf2ss(&self.num, &self.den)?;
        simulate_lti_step(&a, &b, &c, d, t)
    }

    /// Compute the impulse response of the system.
    pub fn impulse(&self, t: &[f64]) -> Result<Vec<f64>, SignalError> {
        if t.is_empty() {
            return Ok(Vec::new());
        }
        let (a, b, c, d) = tf2ss(&self.num, &self.den)?;
        simulate_lti_impulse(&a, &b, &c, d, t)
    }

    /// Get the system's poles (roots of denominator).
    pub fn poles(&self) -> Result<Vec<(f64, f64)>, SignalError> {
        let (re, im) = poly_roots(&self.den)?;
        Ok(re.into_iter().zip(im).collect())
    }

    /// Get the system's zeros (roots of numerator).
    pub fn zeros(&self) -> Result<Vec<(f64, f64)>, SignalError> {
        let (re, im) = poly_roots(&self.num)?;
        Ok(re.into_iter().zip(im).collect())
    }

    /// Simulate the response of the LTI system to arbitrary input.
    ///
    /// Matches `scipy.signal.lsim`. Uses state-space representation with RK4.
    ///
    /// # Arguments
    /// * `u` - Input signal values at each time point
    /// * `t` - Time points (must be uniformly spaced)
    /// * `x0` - Initial state (None for zero initial conditions)
    ///
    /// # Returns
    /// Output signal values at each time point
    pub fn lsim(&self, u: &[f64], t: &[f64], x0: Option<&[f64]>) -> Result<Vec<f64>, SignalError> {
        if t.is_empty() {
            return Ok(Vec::new());
        }
        if u.len() != t.len() {
            return Err(SignalError::InvalidArgument(format!(
                "input u length {} must match t length {}",
                u.len(),
                t.len()
            )));
        }

        let (a, b, c, d) = tf2ss(&self.num, &self.den)?;
        let n = b.len();

        // Initialize state
        let mut x = if let Some(x0_init) = x0 {
            if x0_init.len() != n {
                return Err(SignalError::InvalidArgument(format!(
                    "initial state length {} must match system order {}",
                    x0_init.len(),
                    n
                )));
            }
            x0_init.to_vec()
        } else {
            vec![0.0; n]
        };

        if n == 0 {
            // Static system: y = d*u
            return Ok(u.iter().map(|&ui| d * ui).collect());
        }

        let mut y = Vec::with_capacity(t.len());

        for i in 0..t.len() {
            // Output: y = c'x + d*u
            let output: f64 = c
                .iter()
                .zip(x.iter())
                .map(|(&ci, &xi)| ci * xi)
                .sum::<f64>()
                + d * u[i];
            y.push(output);

            if i + 1 < t.len() {
                let dt = t[i + 1] - t[i];
                // RK4 step with current input
                x = rk4_step(&a, &b, &x, u[i], dt);
            }
        }

        Ok(y)
    }
}

/// Discrete-time Linear Time-Invariant system representation.
///
/// Matches `scipy.signal.dlti`. Represents a discrete-time LTI system
/// in transfer function form.
///
/// The transfer function is: H(z) = num(z) / den(z)
#[derive(Debug, Clone, PartialEq)]
pub struct Dlti {
    /// Numerator polynomial coefficients (highest power first).
    pub num: Vec<f64>,
    /// Denominator polynomial coefficients (highest power first).
    pub den: Vec<f64>,
    /// Sampling period (dt > 0).
    pub dt: f64,
}

impl Dlti {
    /// Create a new discrete-time LTI system.
    ///
    /// # Arguments
    /// * `num` - Numerator polynomial coefficients
    /// * `den` - Denominator polynomial coefficients
    /// * `dt` - Sampling period (must be > 0)
    pub fn new(num: Vec<f64>, den: Vec<f64>, dt: f64) -> Result<Self, SignalError> {
        if num.is_empty() {
            return Err(SignalError::InvalidArgument(
                "numerator cannot be empty".to_string(),
            ));
        }
        if den.is_empty() {
            return Err(SignalError::InvalidArgument(
                "denominator cannot be empty".to_string(),
            ));
        }
        if den.iter().all(|&x| x == 0.0) {
            return Err(SignalError::InvalidArgument(
                "denominator cannot be all zeros".to_string(),
            ));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(SignalError::InvalidArgument(
                "sampling period dt must be positive".to_string(),
            ));
        }
        Ok(Self { num, den, dt })
    }

    /// Create a DLTI system from zeros, poles, gain, and sampling period.
    pub fn from_zpk(zeros: &[f64], poles: &[f64], gain: f64, dt: f64) -> Result<Self, SignalError> {
        let num = zpk_to_poly(zeros, gain);
        let den = zpk_to_poly(poles, 1.0);
        Self::new(num, den, dt)
    }

    /// Evaluate the transfer function at z = exp(jω*dt).
    pub fn eval_at_freq(&self, omega: f64) -> (f64, f64) {
        let angle = omega * self.dt;
        let z_re = angle.cos();
        let z_im = angle.sin();
        let num_val = poly_eval_complex(&self.num, z_re, z_im);
        let den_val = poly_eval_complex(&self.den, z_re, z_im);
        complex_div(num_val.0, num_val.1, den_val.0, den_val.1)
    }

    /// Compute the frequency response H(e^{jω}) at angular frequencies.
    pub fn freqresp(&self, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut mag = Vec::with_capacity(w.len());
        let mut phase = Vec::with_capacity(w.len());

        for &omega in w {
            let (re, im) = self.eval_at_freq(omega);
            mag.push((re * re + im * im).sqrt());
            phase.push(im.atan2(re));
        }

        (mag, phase)
    }

    /// Compute the step response of the discrete system.
    pub fn step(&self, n_samples: usize) -> Result<Vec<f64>, SignalError> {
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        // Unit step input
        let x: Vec<f64> = vec![1.0; n_samples];
        self.lfilter(&x)
    }

    /// Compute the impulse response of the discrete system.
    pub fn impulse(&self, n_samples: usize) -> Result<Vec<f64>, SignalError> {
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        // Unit impulse input
        let mut x = vec![0.0; n_samples];
        x[0] = 1.0;
        self.lfilter(&x)
    }

    /// Filter a signal using this discrete system (direct form II transposed).
    pub fn lfilter(&self, x: &[f64]) -> Result<Vec<f64>, SignalError> {
        lfilter(&self.num, &self.den, x, None)
    }

    /// Get the system's poles (roots of denominator).
    pub fn poles(&self) -> Result<Vec<(f64, f64)>, SignalError> {
        let (re, im) = poly_roots(&self.den)?;
        Ok(re.into_iter().zip(im).collect())
    }

    /// Get the system's zeros (roots of numerator).
    pub fn zeros(&self) -> Result<Vec<(f64, f64)>, SignalError> {
        let (re, im) = poly_roots(&self.num)?;
        Ok(re.into_iter().zip(im).collect())
    }

    /// Check if the system is stable (all poles inside unit circle).
    pub fn is_stable(&self) -> Result<bool, SignalError> {
        let poles = self.poles()?;
        Ok(poles.iter().all(|&(re, im)| re * re + im * im < 1.0))
    }

    /// Simulate the response of the discrete LTI system to arbitrary input.
    ///
    /// Matches `scipy.signal.dlsim`. Uses lfilter with optional initial state.
    ///
    /// # Arguments
    /// * `u` - Input signal values
    /// * `zi` - Initial filter state (None for zero initial conditions)
    ///
    /// # Returns
    /// Output signal values
    pub fn dlsim(&self, u: &[f64], zi: Option<&[f64]>) -> Result<Vec<f64>, SignalError> {
        lfilter(&self.num, &self.den, u, zi)
    }
}

/// Simulate the response of a continuous-time LTI system.
///
/// Standalone function matching `scipy.signal.lsim`.
///
/// # Arguments
/// * `system` - The LTI system
/// * `u` - Input signal values
/// * `t` - Time points
/// * `x0` - Initial state (None for zero)
pub fn lsim(
    system: &Lti,
    u: &[f64],
    t: &[f64],
    x0: Option<&[f64]>,
) -> Result<Vec<f64>, SignalError> {
    system.lsim(u, t, x0)
}

/// Simulate the response of a discrete-time LTI system.
///
/// Standalone function matching `scipy.signal.dlsim`.
///
/// # Arguments
/// * `system` - The DLTI system
/// * `u` - Input signal values
/// * `zi` - Initial filter state (None for zero)
pub fn dlsim(system: &Dlti, u: &[f64], zi: Option<&[f64]>) -> Result<Vec<f64>, SignalError> {
    system.dlsim(u, zi)
}

// ═══════════════════════════════════════════════════════════════════
// LTI Helper Functions
// ═══════════════════════════════════════════════════════════════════

/// Convert zeros/poles to polynomial coefficients.
fn zpk_to_poly(roots: &[f64], gain: f64) -> Vec<f64> {
    if roots.is_empty() {
        return vec![gain];
    }
    // Start with [1]
    let mut poly = vec![1.0];
    for &root in roots {
        // Multiply by (x - root)
        let mut new_poly = vec![0.0; poly.len() + 1];
        for (i, &c) in poly.iter().enumerate() {
            new_poly[i] += c;
            new_poly[i + 1] -= c * root;
        }
        poly = new_poly;
    }
    // Apply gain
    for c in &mut poly {
        *c *= gain;
    }
    poly
}

/// Evaluate polynomial with complex argument.
fn poly_eval_complex(coeffs: &[f64], z_re: f64, z_im: f64) -> (f64, f64) {
    if coeffs.is_empty() {
        return (0.0, 0.0);
    }
    // Horner's method for complex evaluation
    let mut re = coeffs[0];
    let mut im = 0.0;
    for &c in &coeffs[1..] {
        // (re, im) = (re, im) * (z_re, z_im) + (c, 0)
        let new_re = re * z_re - im * z_im + c;
        let new_im = re * z_im + im * z_re;
        re = new_re;
        im = new_im;
    }
    (re, im)
}

/// Convert transfer function to state-space (controllable canonical form).
type StateSpace = (Vec<Vec<f64>>, Vec<f64>, Vec<f64>, f64);

fn tf2ss(num: &[f64], den: &[f64]) -> Result<StateSpace, SignalError> {
    if den.is_empty() || den[0] == 0.0 {
        return Err(SignalError::InvalidArgument(
            "denominator leading coefficient cannot be zero".to_string(),
        ));
    }

    let n = den.len() - 1; // System order
    if n == 0 {
        // Static gain
        let d = if num.is_empty() { 0.0 } else { num[0] / den[0] };
        return Ok((Vec::new(), Vec::new(), Vec::new(), d));
    }

    // Normalize denominator
    let a0 = den[0];
    let den_norm: Vec<f64> = den.iter().map(|&x| x / a0).collect();

    // Pad numerator to match denominator length
    let mut num_padded = vec![0.0; den.len()];
    let offset = den.len().saturating_sub(num.len());
    for (i, &c) in num.iter().enumerate() {
        if offset + i < num_padded.len() {
            num_padded[offset + i] = c / a0;
        }
    }

    // Controllable canonical form
    // A matrix (n x n)
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n - 1 {
        a[i][i + 1] = 1.0;
    }
    for i in 0..n {
        a[n - 1][i] = -den_norm[n - i];
    }

    // B vector (n x 1)
    let mut b = vec![0.0; n];
    b[n - 1] = 1.0;

    // C vector (1 x n)
    let mut c = vec![0.0; n];
    for i in 0..n {
        c[i] = num_padded[n - i] - num_padded[0] * den_norm[n - i];
    }

    // D scalar
    let d = num_padded[0];

    Ok((a, b, c, d))
}

/// Simulate step response using RK4 integration.
fn simulate_lti_step(
    a: &[Vec<f64>],
    b: &[f64],
    c: &[f64],
    d: f64,
    t: &[f64],
) -> Result<Vec<f64>, SignalError> {
    let n = b.len();
    if n == 0 {
        // Static system
        return Ok(vec![d; t.len()]);
    }

    let mut x = vec![0.0; n];
    let mut y = Vec::with_capacity(t.len());

    for i in 0..t.len() {
        // Output: y = c'x + d*u (u = 1 for step)
        let output: f64 = c
            .iter()
            .zip(x.iter())
            .map(|(&ci, &xi)| ci * xi)
            .sum::<f64>()
            + d;
        y.push(output);

        if i + 1 < t.len() {
            let dt = t[i + 1] - t[i];
            // RK4 step with u = 1
            x = rk4_step(a, b, &x, 1.0, dt);
        }
    }

    Ok(y)
}

/// Simulate impulse response.
fn simulate_lti_impulse(
    a: &[Vec<f64>],
    b: &[f64],
    c: &[f64],
    d: f64,
    t: &[f64],
) -> Result<Vec<f64>, SignalError> {
    let n = b.len();
    if n == 0 {
        // Static system - impulse response is delta * d
        let mut y = vec![0.0; t.len()];
        if !y.is_empty() {
            y[0] = d;
        }
        return Ok(y);
    }

    // Initial condition: x(0+) = b (from impulse)
    let mut x = b.to_vec();
    let mut y = Vec::with_capacity(t.len());

    for i in 0..t.len() {
        // Output: y = c'x (no feedthrough for impulse after t=0)
        let output: f64 = c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
        y.push(output);

        if i + 1 < t.len() {
            let dt = t[i + 1] - t[i];
            // RK4 step with u = 0 (no input after initial impulse)
            x = rk4_step(a, b, &x, 0.0, dt);
        }
    }

    // Add direct feedthrough at t=0
    if !y.is_empty() && d != 0.0 {
        // The impulse contribution at t=0 should include d*delta(0)
        // For discrete representation, we add d to y[0]
        y[0] += d;
    }

    Ok(y)
}

/// Single RK4 integration step for x' = Ax + Bu.
fn rk4_step(a: &[Vec<f64>], b: &[f64], x: &[f64], u: f64, dt: f64) -> Vec<f64> {
    let n = x.len();

    let f = |x: &[f64]| -> Vec<f64> {
        let mut dx = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                dx[i] += a[i][j] * x[j];
            }
            dx[i] += b[i] * u;
        }
        dx
    };

    let k1 = f(x);
    let x1: Vec<f64> = x
        .iter()
        .zip(&k1)
        .map(|(&xi, &ki)| xi + 0.5 * dt * ki)
        .collect();
    let k2 = f(&x1);
    let x2: Vec<f64> = x
        .iter()
        .zip(&k2)
        .map(|(&xi, &ki)| xi + 0.5 * dt * ki)
        .collect();
    let k3 = f(&x2);
    let x3: Vec<f64> = x.iter().zip(&k3).map(|(&xi, &ki)| xi + dt * ki).collect();
    let k4 = f(&x3);

    x.iter()
        .enumerate()
        .map(|(i, &xi)| xi + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_error_classifies_string_constructor() {
        assert!(matches!(
            SignalError::InvalidArgument("FFT failed: backend error".to_string()),
            SignalError::FftFailure(_)
        ));
        assert!(matches!(
            SignalError::InvalidArgument("input must not be empty".to_string()),
            SignalError::InvalidInputLength {
                expected: 1,
                actual: 0
            }
        ));
        assert!(matches!(
            SignalError::InvalidArgument("Wn values must be finite and in (0, 1)".to_string()),
            SignalError::NonFiniteInput { .. }
        ));
        assert!(matches!(
            SignalError::InvalidArgument("unknown window type: nope".to_string()),
            SignalError::UnsupportedMode { .. }
        ));
    }

    #[test]
    fn savgol_coeffs_smoothing_sum_to_one() {
        // Smoothing coefficients should sum to 1
        let c = savgol_coeffs(5, 2, 0).expect("coeffs");
        let sum: f64 = c.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "coeffs sum = {sum}, expected 1.0"
        );
    }

    #[test]
    fn savgol_coeffs_symmetric() {
        // Smoothing coefficients for deriv=0 should be symmetric
        let c = savgol_coeffs(7, 3, 0).expect("coeffs");
        let n = c.len();
        for i in 0..n / 2 {
            assert!(
                (c[i] - c[n - 1 - i]).abs() < 1e-10,
                "c[{i}]={} != c[{}]={}",
                c[i],
                n - 1 - i,
                c[n - 1 - i]
            );
        }
    }

    #[test]
    fn savgol_filter_preserves_constant() {
        // Constant signal should be unchanged
        let x = vec![5.0; 20];
        let filtered = savgol_filter(&x, 5, 2).expect("filter");
        for (i, &v) in filtered.iter().enumerate() {
            assert!((v - 5.0).abs() < 1e-10, "filtered[{i}] = {v}, expected 5.0");
        }
    }

    #[test]
    fn savgol_filter_preserves_linear() {
        // Linear signal should be preserved (polyorder >= 1)
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let filtered = savgol_filter(&x, 5, 2).expect("filter");
        // Interior points should match exactly
        for i in 2..18 {
            assert!(
                (filtered[i] - x[i]).abs() < 1e-8,
                "filtered[{i}] = {}, expected {}",
                filtered[i],
                x[i]
            );
        }
    }

    #[test]
    fn savgol_filter_smooths_noise() {
        // Noisy signal should be smoothed
        let mut x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        // Add "noise"
        for (i, xi) in x.iter_mut().enumerate() {
            *xi += if i % 2 == 0 { 0.1 } else { -0.1 };
        }
        let filtered = savgol_filter(&x, 7, 3).expect("filter");
        // Filtered should have less variation than original
        let original_var: f64 = x.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        let filtered_var: f64 = filtered.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        assert!(
            filtered_var < original_var,
            "filtered should be smoother: {filtered_var} < {original_var}"
        );
    }

    #[test]
    fn savgol_coeffs_even_window_rejected() {
        let err = savgol_coeffs(4, 2, 0).expect_err("even window");
        assert!(matches!(err, SignalError::InvalidWindowLength(_)));
    }

    #[test]
    fn savgol_coeffs_polyorder_too_high() {
        let err = savgol_coeffs(5, 5, 0).expect_err("polyorder >= window");
        assert!(matches!(err, SignalError::InvalidPolyOrder(_)));
    }

    #[test]
    fn savgol_filter_window_too_large() {
        let err = savgol_filter(&[1.0, 2.0], 5, 2).expect_err("window > signal");
        assert!(matches!(err, SignalError::InvalidWindowLength(_)));
    }

    #[test]
    fn savgol_coeffs_first_derivative() {
        // First derivative coefficients for window=5, polyorder=2
        let c = savgol_coeffs(5, 2, 1).expect("deriv coeffs");
        assert_eq!(c.len(), 5);
        // For first derivative, sum should be 0 (since differentiating a constant gives 0)
        let sum: f64 = c.iter().sum();
        assert!(
            sum.abs() < 1e-10,
            "derivative coeffs sum = {sum}, expected 0"
        );
        // Should be antisymmetric: c[i] = -c[n-1-i]
        let n = c.len();
        for i in 0..n / 2 {
            assert!(
                (c[i] + c[n - 1 - i]).abs() < 1e-10,
                "c[{i}]={} should be -c[{}]={}",
                c[i],
                n - 1 - i,
                c[n - 1 - i]
            );
        }
    }

    // ── Window function tests ───────────────────────────────────────

    #[test]
    fn hann_window_endpoints_are_zero() {
        let w = hann(8);
        assert!((w[0]).abs() < 1e-12, "hann[0] = {}", w[0]);
        assert!((w[7]).abs() < 1e-12, "hann[7] = {}", w[7]);
    }

    #[test]
    fn hann_window_symmetric() {
        let w = hann(9);
        for i in 0..w.len() / 2 {
            assert!(
                (w[i] - w[w.len() - 1 - i]).abs() < 1e-12,
                "hann not symmetric at {i}"
            );
        }
    }

    #[test]
    fn hamming_window_nonzero_endpoints() {
        let w = hamming(8);
        assert!(w[0] > 0.07, "hamming[0] should be ~0.08");
    }

    #[test]
    fn blackman_window_endpoints_near_zero() {
        let w = blackman(16);
        assert!(w[0].abs() < 0.01, "blackman[0] = {}", w[0]);
    }

    #[test]
    fn blackmanharris_window_matches_scipy_reference() {
        let w = blackmanharris(5);
        let expected = [0.00006, 0.21747, 1.0, 0.21747, 0.00006];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn barthann_window_matches_scipy_reference() {
        let w = barthann(8);
        let expected = [
            0.0,
            0.21164530386510985,
            0.6017008120462566,
            0.9280824555172049,
            0.9280824555172051,
            0.6017008120462566,
            0.21164530386511002,
            0.0,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn general_hamming_window_matches_scipy_reference() {
        let w = general_hamming(8, 0.75);
        let expected = [
            0.5,
            0.5941275495353167,
            0.8056302334890786,
            0.9752422169756048,
            0.9752422169756048,
            0.8056302334890786,
            0.5941275495353167,
            0.5,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn general_cosine_window_matches_scipy_reference() {
        let w = general_cosine(8, &[0.5, 0.3, 0.2], true);
        let expected = [
            0.4,
            0.26844887265111705,
            0.38656250660641056,
            0.8949886207424725,
            0.8949886207424725,
            0.38656250660641056,
            0.26844887265111705,
            0.4,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn general_cosine_periodic_window_matches_scipy_reference() {
        let w = general_cosine(8, &[0.5, 0.5], false);
        let expected = [
            0.0,
            0.14644660940672627,
            0.5,
            0.8535533905932737,
            1.0,
            0.8535533905932737,
            0.5,
            0.14644660940672627,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn boxcar_window_all_ones() {
        let w = boxcar(5, true);
        assert_eq!(w, vec![1.0; 5]);

        let w_periodic = boxcar(5, false);
        assert_eq!(w_periodic, vec![1.0; 5]);

        assert!(boxcar(0, true).is_empty());
    }

    fn assert_slice_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label} length mismatch: got {}, expected {}",
            actual.len(),
            expected.len()
        );
        for (idx, (&actual_value, &expected_value)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert!(
                (actual_value - expected_value).abs() < tol,
                "{label}[{idx}] = {actual_value}, expected {expected_value}, diff={}",
                (actual_value - expected_value).abs()
            );
        }
    }

    #[test]
    fn gaussian_window_matches_scipy_reference() {
        let w = gaussian(5, 1.0, true);
        let expected = [
            0.1353352832366127,
            0.6065306597126334,
            1.0,
            0.6065306597126334,
            0.1353352832366127,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn gaussian_periodic_window_matches_scipy_reference() {
        let w = gaussian(8, 2.0, false);
        let expected = [
            0.1353352832366127,
            0.32465246735834974,
            0.6065306597126334,
            0.8824969025845955,
            1.0,
            0.8824969025845955,
            0.6065306597126334,
            0.32465246735834974,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn general_gaussian_window_matches_scipy_reference() {
        let w = general_gaussian(8, 1.5, 2.0, true);
        let expected = [
            0.06858458348811015,
            0.3766034507108804,
            0.8098246793420792,
            0.9922179382602435,
            0.9922179382602435,
            0.8098246793420792,
            0.3766034507108804,
            0.06858458348811015,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn general_gaussian_periodic_window_matches_scipy_reference() {
        let w = general_gaussian(8, 1.5, 2.0, false);
        let expected = [
            0.01831563888873418,
            0.18498139990730428,
            0.6065306597126334,
            0.9394130628134758,
            1.0,
            0.9394130628134758,
            0.6065306597126334,
            0.18498139990730428,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn exponential_window_matches_scipy_reference() {
        let w = exponential(8, None, 2.0, true).unwrap();
        let expected = [
            0.17377394345044514,
            0.2865047968601901,
            0.4723665527410147,
            0.7788007830714049,
            0.7788007830714049,
            0.4723665527410147,
            0.2865047968601901,
            0.17377394345044514,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn exponential_periodic_window_matches_scipy_reference() {
        let w = exponential(8, None, 2.0, false).unwrap();
        let expected = [
            0.1353352832366127,
            0.22313016014842982,
            0.36787944117144233,
            0.6065306597126334,
            1.0,
            0.6065306597126334,
            0.36787944117144233,
            0.22313016014842982,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn exponential_window_supports_non_symmetric_center() {
        let w = exponential(5, Some(1.0), 2.0, false).unwrap();
        let expected = [
            0.6065306597126334,
            1.0,
            0.6065306597126334,
            0.36787944117144233,
            0.22313016014842982,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn exponential_window_rejects_center_for_symmetric_mode() {
        let err = exponential(5, Some(1.0), 2.0, true).unwrap_err();
        assert!(err.is_argument_error());
        assert_eq!(
            err.to_string(),
            "invalid argument: If sym==True, center must be None."
        );
    }

    #[test]
    fn chebwin_window_matches_scipy_reference() {
        let w = chebwin(5, 100.0);
        let expected = [
            0.16866468734562678,
            0.6686513141856113,
            1.0,
            0.6686513141856113,
            0.16866468734562678,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn chebwin_even_window_matches_scipy_reference() {
        let w = chebwin(8, 60.0);
        let expected = [
            0.06847555416399577,
            0.303219161655201,
            0.686846620773932,
            1.0,
            1.0,
            0.686846620773932,
            0.303219161655201,
            0.06847555416399577,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn kaiser_window_beta_zero_is_rectangular() {
        let w = kaiser(8, 0.0);
        for &v in &w {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "kaiser(beta=0) should be rectangular"
            );
        }
    }

    #[test]
    fn dpss_symmetric_window_matches_scipy_reference() {
        let result = dpss(8, 2.5, None, true, None, false).expect("dpss symmetric");
        let expected = [
            0.052404360853994365,
            0.2631860434811115,
            0.639750596958442,
            0.9624060150375939,
            0.9624060150375939,
            0.639750596958442,
            0.2631860434811115,
            0.052404360853994365,
        ];
        assert_eq!(result.windows.len(), 1);
        assert_slice_close(&result.windows[0], &expected, 1.0e-12, "dpss symmetric");
        assert!(result.ratios.is_none());
    }

    #[test]
    fn dpss_periodic_window_matches_scipy_reference() {
        let result = dpss(8, 2.5, None, false, None, false).expect("dpss periodic");
        let expected = [
            0.04080887106381651,
            0.2018541771042083,
            0.5121647719505525,
            0.8503476239828903,
            1.0,
            0.8503476239828905,
            0.5121647719505525,
            0.20185417710420828,
        ];
        assert_eq!(result.windows.len(), 1);
        assert_slice_close(&result.windows[0], &expected, 1.0e-12, "dpss periodic");
    }

    #[test]
    fn dpss_multiple_tapers_and_ratios_match_scipy_reference() {
        let result = dpss(8, 2.5, Some(3), true, Some(DpssNorm::L2), true).expect("dpss kmax");
        let expected_windows = [
            [
                0.03123383094162537,
                0.15686305975922865,
                0.3813015112592644,
                0.5736092623023827,
                0.5736092623023827,
                0.3813015112592644,
                0.15686305975922865,
                0.03123383094162537,
            ],
            [
                0.11133547272330951,
                0.3749691021790795,
                0.5282319109750825,
                0.2607175351833948,
                -0.2607175351833948,
                -0.5282319109750825,
                -0.37496910217907947,
                -0.11133547272330951,
            ],
            [
                0.26478310487439105,
                0.5199646064881195,
                0.24218795419331154,
                -0.31760307022507,
                -0.31760307022507,
                0.24218795419331157,
                0.5199646064881195,
                0.26478310487439105,
            ],
        ];
        let expected_ratios = [0.9999998455973096, 0.9999787214591116, 0.9988738410781977];

        assert_eq!(result.windows.len(), 3);
        for (actual_window, expected_window) in result.windows.iter().zip(expected_windows.iter()) {
            assert_slice_close(actual_window, expected_window, 1.0e-12, "dpss taper");
        }

        let ratios = result.ratios.expect("ratios requested");
        for (actual, want) in ratios.iter().zip(expected_ratios.iter()) {
            assert!((*actual - *want).abs() < 1.0e-9);
        }
    }

    #[test]
    fn dpss_subsample_norm_matches_scipy_reference() {
        let result =
            dpss(5, 1.5, None, true, Some(DpssNorm::Subsample), false).expect("dpss subsample");
        let expected = [
            0.2608050703712001,
            0.7407980987672104,
            1.0,
            0.7407980987672104,
            0.2608050703712001,
        ];
        assert_slice_close(&result.windows[0], &expected, 1.0e-12, "dpss subsample");
    }

    #[test]
    fn dpss_rejects_invalid_parameters() {
        assert!(dpss(8, 0.0, None, true, None, false).is_err());
        assert!(dpss(8, 4.0, None, true, None, false).is_err());
        assert!(dpss(8, 2.5, Some(0), true, None, false).is_err());
        assert!(dpss(8, 2.5, Some(9), true, None, false).is_err());
    }

    #[test]
    fn taylor_window_symmetric() {
        // Taylor window should be symmetric
        let w = taylor(11, 4, -30.0, true, true);
        assert_eq!(w.len(), 11);
        for i in 0..5 {
            assert!(
                (w[i] - w[10 - i]).abs() < 1e-12,
                "Taylor asymmetric at {i}: {} vs {}",
                w[i],
                w[10 - i]
            );
        }
    }

    #[test]
    fn taylor_window_normalized_peak() {
        // With norm=true, peak should be 1.0
        let w = taylor(21, 4, -30.0, true, true);
        let max_val = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (max_val - 1.0).abs() < 1e-10,
            "Taylor norm peak = {}, expected 1.0",
            max_val
        );
    }

    #[test]
    fn taylor_window_empty_and_single() {
        assert!(taylor(0, 4, -30.0, true, true).is_empty());
        assert_eq!(taylor(1, 4, -30.0, true, true), [1.0]);
    }

    #[test]
    fn window_empty_returns_empty() {
        assert!(hann(0).is_empty());
        assert!(hamming(0).is_empty());
        assert!(general_hamming(0, 0.75).is_empty());
        assert!(general_cosine(0, &[0.5, 0.5], true).is_empty());
        assert!(gaussian(0, 2.0, true).is_empty());
        assert!(general_gaussian(0, 1.5, 2.0, true).is_empty());
        assert!(exponential(0, None, 2.0, true).unwrap().is_empty());
        assert!(blackman(0).is_empty());
        assert!(blackmanharris(0).is_empty());
        assert!(barthann(0).is_empty());
        assert!(chebwin(0, 100.0).is_empty());
        assert!(kaiser(0, 5.0).is_empty());
        assert!(kaiser_bessel_derived(0, 5.0, true).unwrap().is_empty());
        assert!(lanczos(0).is_empty());
        assert!(taylor(0, 4, -30.0, true, true).is_empty());
    }

    #[test]
    fn window_single_returns_one() {
        assert_eq!(hann(1), [1.0]);
        assert_eq!(hamming(1), [1.0]);
        assert_eq!(general_hamming(1, 0.75), [1.0]);
        assert_eq!(general_cosine(1, &[0.5, 0.5], true), [1.0]);
        assert_eq!(gaussian(1, 2.0, true), [1.0]);
        assert_eq!(general_gaussian(1, 1.5, 2.0, true), [1.0]);
        assert_eq!(exponential(1, None, 2.0, true).unwrap(), [1.0]);
        assert_eq!(blackman(1), [1.0]);
        assert_eq!(blackmanharris(1), [1.0]);
        assert_eq!(barthann(1), [1.0]);
        assert_eq!(chebwin(1, 100.0), [1.0]);
        assert_eq!(kaiser(1, 5.0), [1.0]);
        assert_eq!(lanczos(1), [1.0]);
    }

    #[test]
    fn kaiser_bessel_derived_matches_scipy_reference() {
        let w = kaiser_bessel_derived(8, 5.0, true).unwrap();
        let expected = [
            0.1297945228712448,
            0.5201443455712672,
            0.8540783686350089,
            0.9915409128385101,
            0.9915409128385101,
            0.8540783686350089,
            0.5201443455712672,
            0.1297945228712448,
        ];
        assert_slice_close(&w, &expected, 1.0e-8, "kaiser_bessel_derived");
    }

    #[test]
    fn kaiser_bessel_derived_rejects_non_symmetric_and_odd_lengths() {
        let periodic = kaiser_bessel_derived(8, 5.0, false).expect_err("periodic KBD");
        assert_eq!(
            periodic.to_string(),
            "invalid input shape: Kaiser-Bessel Derived windows are only defined for symmetric shapes"
        );

        let odd = kaiser_bessel_derived(7, 5.0, true).expect_err("odd KBD");
        assert_eq!(
            odd.to_string(),
            "invalid argument: Kaiser-Bessel Derived windows are only defined for even number of points"
        );
    }

    #[test]
    fn lanczos_window_symmetric() {
        // Lanczos window should be symmetric
        let w = lanczos(7);
        assert_eq!(w.len(), 7);
        for i in 0..3 {
            assert!((w[i] - w[6 - i]).abs() < 1e-12, "asymmetry at {i}");
        }
        // Center should be 1.0 (sinc(0) = 1)
        assert!(
            (w[3] - 1.0).abs() < 1e-12,
            "center = {}, expected 1.0",
            w[3]
        );
    }

    #[test]
    fn lanczos_window_endpoints() {
        // At endpoints, x = ±1, so sinc(±1) = sin(±π)/(±π) = 0
        let w = lanczos(11);
        assert!((w[0]).abs() < 1e-12, "w[0] = {}, expected 0", w[0]);
        assert!((w[10]).abs() < 1e-12, "w[10] = {}, expected 0", w[10]);
    }

    // ── Convolution tests ───────────────────────────────────────────

    #[test]
    fn convolve_impulse() {
        // Convolving with [1] should return the original signal
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0];
        let result = convolve(&a, &b, ConvolveMode::Full).expect("convolve");
        assert_eq!(result, a);
    }

    #[test]
    fn convolve_known_result() {
        // [1, 2, 3] * [0, 1, 0.5] = [0, 1, 2.5, 4, 1.5]
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.5];
        let result = convolve(&a, &b, ConvolveMode::Full).expect("convolve");
        assert_eq!(result.len(), 5);
        let expected = [0.0, 1.0, 2.5, 4.0, 1.5];
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-12, "conv[{i}] = {r}, expected {e}");
        }
    }

    #[test]
    fn convolve_same_mode() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0];
        let result = convolve(&a, &b, ConvolveMode::Same).expect("same");
        assert_eq!(result.len(), a.len());
    }

    #[test]
    fn convolve_valid_mode() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0];
        let result = convolve(&a, &b, ConvolveMode::Valid).expect("valid");
        // Valid: length = max(5,3) - min(5,3) + 1 = 3
        assert_eq!(result.len(), 3);
        assert!((result[0] - 6.0).abs() < 1e-12); // 1+2+3
        assert!((result[1] - 9.0).abs() < 1e-12); // 2+3+4
        assert!((result[2] - 12.0).abs() < 1e-12); // 3+4+5
    }

    #[test]
    fn fftconvolve_matches_direct() {
        let a = vec![1.0, -1.0, 2.0, 3.0, -0.5];
        let b = vec![0.5, 1.0, -0.5, 0.25];
        let direct = convolve(&a, &b, ConvolveMode::Full).expect("direct");
        let fft_conv = fftconvolve(&a, &b, ConvolveMode::Full).expect("fft");
        assert_eq!(direct.len(), fft_conv.len());
        for (i, (&d, &f)) in direct.iter().zip(fft_conv.iter()).enumerate() {
            assert!(
                (d - f).abs() < 1e-10,
                "fftconvolve[{i}] = {f}, direct = {d}"
            );
        }
    }

    #[test]
    fn fftconvolve_same_mode() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, -1.0];
        let result = fftconvolve(&a, &b, ConvolveMode::Same).expect("same");
        assert_eq!(result.len(), a.len());
    }

    #[test]
    fn convolve_empty_rejected() {
        let err = convolve(&[], &[1.0], ConvolveMode::Full).expect_err("empty");
        assert!(err.is_argument_error());
    }

    #[test]
    fn deconvolve_recovers_original_signal() {
        let original = vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let impulse_response = vec![2.0, 1.0];
        let recorded =
            convolve(&impulse_response, &original, ConvolveMode::Full).expect("convolve");

        let (quotient, remainder) = deconvolve(&recorded, &impulse_response).expect("deconvolve");

        assert_eq!(quotient.len(), original.len());
        for (i, (&got, &expected)) in quotient.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-12,
                "quotient[{i}] = {got}, expected {expected}"
            );
        }
        assert_eq!(remainder.len(), recorded.len());
        for (i, &value) in remainder.iter().enumerate() {
            assert!(value.abs() < 1e-12, "remainder[{i}] = {value}");
        }
    }

    #[test]
    fn deconvolve_known_polynomial_division() {
        let signal = vec![0.0, 0.0, 1.0, 2.0, 3.0];
        let divisor = vec![1.0, 2.0];

        let (quotient, remainder) = deconvolve(&signal, &divisor).expect("deconvolve");

        assert_eq!(quotient, vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(remainder, vec![0.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn deconvolve_divisor_longer_than_signal_returns_empty_quotient() {
        let signal = vec![1.0, 2.0];
        let divisor = vec![1.0, 2.0, 3.0];

        let (quotient, remainder) = deconvolve(&signal, &divisor).expect("deconvolve");

        assert!(quotient.is_empty());
        assert_eq!(remainder, signal);
    }

    #[test]
    fn deconvolve_zero_leading_divisor_rejected() {
        let err = deconvolve(&[1.0, 2.0, 3.0], &[0.0, 1.0]).expect_err("zero leading divisor");
        assert!(err.is_argument_error());
    }

    // ── find_peaks tests ────────────────────────────────────────────

    #[test]
    fn find_peaks_simple() {
        let x = [0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
        let result = find_peaks(&x, FindPeaksOptions::default());
        assert_eq!(result.peaks, vec![1, 3, 5]);
    }

    #[test]
    fn find_peaks_with_height() {
        let x = [0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0];
        let result = find_peaks(
            &x,
            FindPeaksOptions {
                height: Some(1.5),
                ..FindPeaksOptions::default()
            },
        );
        assert_eq!(result.peaks, vec![3]); // only the peak at 2.0
    }

    #[test]
    fn find_peaks_with_distance() {
        let x = [0.0, 1.0, 0.0, 0.8, 0.0, 2.0, 0.0];
        let result = find_peaks(
            &x,
            FindPeaksOptions {
                distance: Some(3),
                ..FindPeaksOptions::default()
            },
        );
        // Distance=3: peaks at 1 and 3 are too close (dist=2), keep highest (1.0 at idx 1)
        // Peak at 5 (2.0) is far enough from both
        assert!(result.peaks.contains(&5));
    }

    #[test]
    fn find_peaks_distance_equal_height_tie_prefers_rightmost() {
        let x = [0.0, 1.0, 0.0, 1.0, 0.0];
        let result = find_peaks(
            &x,
            FindPeaksOptions {
                distance: Some(3),
                ..FindPeaksOptions::default()
            },
        );
        assert_eq!(result.peaks, vec![3]);
    }

    #[test]
    fn find_peaks_distance_saturates_large_window() {
        let x = [0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
        let result = find_peaks(
            &x,
            FindPeaksOptions {
                distance: Some(usize::MAX),
                ..FindPeaksOptions::default()
            },
        );
        assert_eq!(result.peaks, vec![3]);
    }

    #[test]
    fn find_peaks_with_prominence() {
        let x = [0.0, 3.0, 2.5, 2.8, 0.0]; // peak at 1 (prom=3), peak at 3 (prom=0.3)
        let result = find_peaks(
            &x,
            FindPeaksOptions {
                prominence: Some(1.0),
                ..FindPeaksOptions::default()
            },
        );
        assert_eq!(result.peaks, vec![1]); // only the prominent peak
    }

    #[test]
    fn find_peaks_sine_wave() {
        let n = 100;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let result = find_peaks(&x, FindPeaksOptions::default());
        // 3 full cycles should have ~3 peaks
        assert!(
            result.peaks.len() >= 2 && result.peaks.len() <= 4,
            "expected ~3 peaks, got {}",
            result.peaks.len()
        );
    }

    #[test]
    fn find_peaks_too_short() {
        assert!(
            find_peaks(&[1.0, 2.0], FindPeaksOptions::default())
                .peaks
                .is_empty()
        );
        assert!(
            find_peaks(&[1.0], FindPeaksOptions::default())
                .peaks
                .is_empty()
        );
        assert!(
            find_peaks(&[], FindPeaksOptions::default())
                .peaks
                .is_empty()
        );
    }

    #[test]
    fn peak_widths_handles_out_of_range_peaks_without_panicking() {
        let (widths, width_heights, left_ips, right_ips) = peak_widths(&[0.0, 1.0, 0.0], &[9], 0.5);
        assert_eq!(widths, vec![0.0]);
        assert_eq!(width_heights, vec![0.0]);
        assert_eq!(left_ips, vec![9.0]);
        assert_eq!(right_ips, vec![9.0]);
    }

    #[test]
    fn find_peaks_flat_signal() {
        let x = [5.0; 10];
        let result = find_peaks(&x, FindPeaksOptions::default());
        assert!(result.peaks.is_empty(), "flat signal has no peaks");
    }

    // ── Butterworth filter design ──────────────────────────────────

    #[test]
    fn butter_lowpass_order1() {
        let coeffs = butter(1, &[0.5], FilterType::Lowpass).expect("butter");
        assert_eq!(coeffs.b.len(), 2);
        assert_eq!(coeffs.a.len(), 2);
        assert!((coeffs.a[0] - 1.0).abs() < 1e-12, "a[0] should be 1.0");
        // DC gain should be 1: sum(b) / sum(a) = 1
        let b_sum: f64 = coeffs.b.iter().sum();
        let a_sum: f64 = coeffs.a.iter().sum();
        assert!(
            (b_sum / a_sum - 1.0).abs() < 1e-10,
            "DC gain should be 1, got {}",
            b_sum / a_sum
        );
    }

    #[test]
    fn butter_lowpass_order2() {
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        assert_eq!(coeffs.b.len(), 3);
        assert_eq!(coeffs.a.len(), 3);
        // DC gain = 1
        let b_sum: f64 = coeffs.b.iter().sum();
        let a_sum: f64 = coeffs.a.iter().sum();
        assert!((b_sum / a_sum - 1.0).abs() < 1e-10, "DC gain should be 1");
    }

    #[test]
    fn butter_highpass_nyquist_gain() {
        let coeffs = butter(2, &[0.3], FilterType::Highpass).expect("butter hp");
        // Nyquist gain should be 1: evaluate at z = -1
        let b_nyq: f64 = coeffs
            .b
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (-1.0_f64).powi(-(i as i32)))
            .sum();
        let a_nyq: f64 = coeffs
            .a
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (-1.0_f64).powi(-(i as i32)))
            .sum();
        assert!(
            (b_nyq / a_nyq - 1.0).abs() < 1e-10,
            "Nyquist gain should be 1"
        );
    }

    #[test]
    fn butter_invalid_order() {
        assert!(butter(0, &[0.5], FilterType::Lowpass).is_err());
    }

    #[test]
    fn butter_invalid_wn() {
        assert!(butter(2, &[0.0], FilterType::Lowpass).is_err());
        assert!(butter(2, &[1.0], FilterType::Lowpass).is_err());
        assert!(butter(2, &[0.2, 0.4], FilterType::Lowpass).is_err());
        assert!(butter(2, &[0.4, 0.2], FilterType::Bandpass).is_err());
    }

    #[test]
    fn butter_bandpass_order_doubles() {
        let coeffs = butter(2, &[0.2, 0.4], FilterType::Bandpass).expect("butter bp");
        assert_eq!(coeffs.b.len(), 5);
        assert_eq!(coeffs.a.len(), 5);
    }

    #[test]
    fn butter_bandpass_passes_midband_and_rejects_outside() {
        let coeffs = butter(2, &[0.2, 0.4], FilterType::Bandpass).expect("butter bp");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let omega_lo = 0.1 * std::f64::consts::PI;
        let omega_mid = 0.3 * std::f64::consts::PI;
        let omega_hi = 0.8 * std::f64::consts::PI;
        let lo_idx = nearest_freq_index(&result.w, omega_lo);
        let mid_idx = nearest_freq_index(&result.w, omega_mid);
        let hi_idx = nearest_freq_index(&result.w, omega_hi);
        assert!(
            result.h_mag[mid_idx] > result.h_mag[lo_idx],
            "bandpass should pass midband better than low stopband"
        );
        assert!(
            result.h_mag[mid_idx] > result.h_mag[hi_idx],
            "bandpass should pass midband better than high stopband"
        );
    }

    #[test]
    fn butter_bandstop_rejects_midband_and_passes_outside() {
        let coeffs = butter(2, &[0.2, 0.4], FilterType::Bandstop).expect("butter bs");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let omega_lo = 0.05 * std::f64::consts::PI;
        let omega_mid = 0.3 * std::f64::consts::PI;
        let omega_hi = 0.8 * std::f64::consts::PI;
        let lo_idx = nearest_freq_index(&result.w, omega_lo);
        let mid_idx = nearest_freq_index(&result.w, omega_mid);
        let hi_idx = nearest_freq_index(&result.w, omega_hi);
        assert!(
            result.h_mag[mid_idx] < result.h_mag[lo_idx],
            "bandstop should attenuate the stop band"
        );
        assert!(
            result.h_mag[mid_idx] < result.h_mag[hi_idx],
            "bandstop should attenuate the stop band"
        );
    }

    #[test]
    fn cheby1_lowpass_shows_passband_ripple() {
        let coeffs = cheby1(4, 1.0, &[0.3], FilterType::Lowpass).expect("cheby1");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let passband_end = nearest_freq_index(&result.w, 0.3 * std::f64::consts::PI);
        let passband = &result.h_mag[..=passband_end];
        let max_mag = passband.iter().copied().fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        let min_mag = passband
            .iter()
            .copied()
            .fold(f64::INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            });
        assert!(
            max_mag - min_mag > 0.01,
            "Chebyshev-I passband should ripple, got range {}",
            max_mag - min_mag
        );
        assert!(
            max_mag <= 1.25 && min_mag >= 0.75,
            "passband ripple should stay bounded, got min={min_mag}, max={max_mag}"
        );
    }

    #[test]
    fn cheby1_bandpass_passes_midband_and_rejects_outside() {
        let coeffs = cheby1(3, 1.0, &[0.2, 0.45], FilterType::Bandpass).expect("cheby1 bp");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let omega_lo = 0.08 * std::f64::consts::PI;
        let omega_mid = 0.32 * std::f64::consts::PI;
        let omega_hi = 0.8 * std::f64::consts::PI;
        let lo_idx = nearest_freq_index(&result.w, omega_lo);
        let mid_idx = nearest_freq_index(&result.w, omega_mid);
        let hi_idx = nearest_freq_index(&result.w, omega_hi);
        assert!(
            result.h_mag[mid_idx] > result.h_mag[lo_idx],
            "Chebyshev-I bandpass should pass midband better than low stopband"
        );
        assert!(
            result.h_mag[mid_idx] > result.h_mag[hi_idx],
            "Chebyshev-I bandpass should pass midband better than high stopband"
        );
    }

    #[test]
    fn cheby1_even_order_lowpass_matches_scipy_gain() {
        let coeffs = cheby1(2, 1.0, &[0.25], FilterType::Lowpass).expect("cheby1 lp");
        let expected_b = [
            0.10255744141814213,
            0.20511488283628426,
            0.10255744141814213,
        ];
        let expected_a = [1.0, -0.9865079240565594, 0.44679329164515147];
        assert_eq!(coeffs.b.len(), expected_b.len());
        assert_eq!(coeffs.a.len(), expected_a.len());
        for (idx, (actual, expected)) in coeffs.b.iter().zip(expected_b.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("cheby1 lowpass b[{idx}]"),
            );
        }
        for (idx, (actual, expected)) in coeffs.a.iter().zip(expected_a.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("cheby1 lowpass a[{idx}]"),
            );
        }
    }

    #[test]
    fn cheby1_even_order_bandpass_matches_scipy_gain() {
        let coeffs = cheby1(4, 1.0, &[0.2, 0.45], FilterType::Bandpass).expect("cheby1 bp");
        let expected_b = [
            0.0042412377794045445,
            0.0,
            -0.016964951117618178,
            0.0,
            0.025447426676427267,
            0.0,
            -0.016964951117618178,
            0.0,
            0.0042412377794045445,
        ];
        let expected_a = [
            1.0,
            -3.805027975445876,
            8.305833307350976,
            -12.0401045837622,
            12.844987905921329,
            -10.009315088065474,
            5.72688219445523,
            -2.1640816756583447,
            0.4751428601925499,
        ];
        assert_eq!(coeffs.b.len(), expected_b.len());
        assert_eq!(coeffs.a.len(), expected_a.len());
        for (idx, (actual, expected)) in coeffs.b.iter().zip(expected_b.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("cheby1 bandpass b[{idx}]"),
            );
        }
        for (idx, (actual, expected)) in coeffs.a.iter().zip(expected_a.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("cheby1 bandpass a[{idx}]"),
            );
        }
    }

    #[test]
    fn bessel_phase_normalized_lowpass_matches_scipy_coefficients() {
        let coeffs = bessel(2, &[0.25], FilterType::Lowpass).expect("bessel lp");
        let expected_b = [
            0.09082678800790182,
            0.18165357601580365,
            0.09082678800790182,
        ];
        let expected_a = [1.0, -0.8771010537418505, 0.24040820577345767];
        assert_eq!(coeffs.b.len(), expected_b.len());
        assert_eq!(coeffs.a.len(), expected_a.len());
        for (idx, (actual, expected)) in coeffs.b.iter().zip(expected_b.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("bessel lowpass b[{idx}]"),
            );
        }
        for (idx, (actual, expected)) in coeffs.a.iter().zip(expected_a.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("bessel lowpass a[{idx}]"),
            );
        }
    }

    #[test]
    fn ellip_lowpass_matches_scipy_cauer_prototype() {
        let coeffs = ellip(2, 1.0, 40.0, &[0.25], FilterType::Lowpass).expect("ellip lp");
        let expected_b = [
            0.10926548486600407,
            0.19417386968773206,
            0.10926548486600406,
        ];
        let expected_a = [1.0, -0.9863237792094665, 0.4493862252181434];
        assert_eq!(coeffs.b.len(), expected_b.len());
        assert_eq!(coeffs.a.len(), expected_a.len());
        for (idx, (actual, expected)) in coeffs.b.iter().zip(expected_b.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("ellip lowpass b[{idx}]"),
            );
        }
        for (idx, (actual, expected)) in coeffs.a.iter().zip(expected_a.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("ellip lowpass a[{idx}]"),
            );
        }
    }

    #[test]
    fn cheby2_lowpass_rejects_stopband_more_than_passband() {
        let coeffs = cheby2(4, 20.0, &[0.3], FilterType::Lowpass).expect("cheby2");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let pass_idx = nearest_freq_index(&result.w, 0.1 * std::f64::consts::PI);
        let stop_idx = nearest_freq_index(&result.w, 0.8 * std::f64::consts::PI);
        assert!(
            result.h_mag[stop_idx] < result.h_mag[pass_idx] * 0.2,
            "Chebyshev-II should strongly attenuate the stopband"
        );
    }

    #[test]
    fn cheby2_odd_order_remains_finite_and_attenuates_stopband() {
        let coeffs = cheby2(3, 20.0, &[0.3], FilterType::Lowpass).expect("cheby2 odd");
        assert!(coeffs.b.iter().all(|value| value.is_finite()));
        assert!(coeffs.a.iter().all(|value| value.is_finite()));

        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let pass_idx = nearest_freq_index(&result.w, 0.1 * std::f64::consts::PI);
        let stop_idx = nearest_freq_index(&result.w, 0.8 * std::f64::consts::PI);
        assert!(
            result.h_mag[stop_idx] < result.h_mag[pass_idx] * 0.3,
            "odd-order Chebyshev-II should still attenuate the stopband"
        );
    }

    #[test]
    fn bessel_group_delay_is_flatter_than_butter() {
        let bessel_coeffs = bessel(3, &[0.25], FilterType::Lowpass).expect("bessel");
        let butter_coeffs = butter(3, &[0.25], FilterType::Lowpass).expect("butter");
        let (_, gd_bessel) =
            group_delay(&bessel_coeffs.b, &bessel_coeffs.a, Some(256)).expect("gd bessel");
        let (_, gd_butter) =
            group_delay(&butter_coeffs.b, &butter_coeffs.a, Some(256)).expect("gd butter");
        let pass_end = 64;
        let bessel_span = gd_span(&gd_bessel[..pass_end]);
        let butter_span = gd_span(&gd_butter[..pass_end]);
        assert!(
            bessel_span < butter_span,
            "Bessel passband delay should be flatter: {bessel_span} < {butter_span}"
        );
    }

    #[test]
    fn bessel_rejects_excessive_order() {
        let err = bessel(90, &[0.25], FilterType::Lowpass).expect_err("large order should fail");
        assert!(err.is_argument_error());
    }

    #[test]
    fn iirfilter_dispatch_matches_butter() {
        let direct = butter(3, &[0.25], FilterType::Lowpass).expect("butter");
        let dispatch = iirfilter(
            3,
            &[0.25],
            FilterType::Lowpass,
            IirFamily::Butterworth,
            None,
            None,
        )
        .expect("iirfilter butter");
        assert_ba_close(&direct, &dispatch, 1e-10, "butter");
    }

    #[test]
    fn iirfilter_dispatch_matches_cheby1() {
        let direct = cheby1(4, 0.8, &[0.2, 0.5], FilterType::Bandpass).expect("cheby1");
        let dispatch = iirfilter(
            4,
            &[0.2, 0.5],
            FilterType::Bandpass,
            IirFamily::Chebyshev1,
            Some(0.8),
            None,
        )
        .expect("iirfilter cheby1");
        assert_ba_close(&direct, &dispatch, 1e-10, "cheby1");
    }

    #[test]
    fn iirfilter_dispatch_matches_cheby2() {
        let direct = cheby2(4, 20.0, &[0.25], FilterType::Lowpass).expect("cheby2");
        let dispatch = iirfilter(
            4,
            &[0.25],
            FilterType::Lowpass,
            IirFamily::Chebyshev2,
            None,
            Some(20.0),
        )
        .expect("iirfilter cheby2");
        assert_ba_close(&direct, &dispatch, 1e-10, "cheby2");
    }

    #[test]
    fn iirfilter_dispatch_matches_bessel() {
        let direct = bessel(3, &[0.35], FilterType::Highpass).expect("bessel");
        let dispatch = iirfilter(
            3,
            &[0.35],
            FilterType::Highpass,
            IirFamily::Bessel,
            None,
            None,
        )
        .expect("iirfilter bessel");
        assert_ba_close(&direct, &dispatch, 1e-10, "bessel");
    }

    #[test]
    fn iirfilter_dispatch_requires_family_params() {
        let err = iirfilter(
            3,
            &[0.25],
            FilterType::Lowpass,
            IirFamily::Chebyshev1,
            None,
            None,
        )
        .expect_err("missing rp should fail");
        assert!(err.is_argument_error());
    }

    // ── Elliptic filter tests ──────────────────────────────────────

    #[test]
    fn ellip_lowpass_produces_finite_coefficients() {
        let coeffs = ellip(4, 1.0, 40.0, &[0.3], FilterType::Lowpass).expect("ellip");
        assert!(
            coeffs.b.iter().all(|v| v.is_finite()),
            "b coeffs not finite"
        );
        assert!(
            coeffs.a.iter().all(|v| v.is_finite()),
            "a coeffs not finite"
        );
    }

    #[test]
    fn ellip_lowpass_passes_low_frequencies_and_rejects_high() {
        let coeffs = ellip(4, 1.0, 40.0, &[0.3], FilterType::Lowpass).expect("ellip");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let pass_idx = nearest_freq_index(&result.w, 0.1 * std::f64::consts::PI);
        let stop_idx = nearest_freq_index(&result.w, 0.8 * std::f64::consts::PI);
        assert!(
            result.h_mag[stop_idx] < result.h_mag[pass_idx] * 0.5,
            "elliptic should attenuate stopband: pass={}, stop={}",
            result.h_mag[pass_idx],
            result.h_mag[stop_idx]
        );
    }

    #[test]
    fn ellip_bandstop_rejects_midband_and_passes_outside() {
        let coeffs = ellip(4, 1.0, 40.0, &[0.2, 0.45], FilterType::Bandstop).expect("ellip bs");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let omega_lo = 0.08 * std::f64::consts::PI;
        let omega_mid = 0.32 * std::f64::consts::PI;
        let omega_hi = 0.8 * std::f64::consts::PI;
        let lo_idx = nearest_freq_index(&result.w, omega_lo);
        let mid_idx = nearest_freq_index(&result.w, omega_mid);
        let hi_idx = nearest_freq_index(&result.w, omega_hi);
        assert!(
            result.h_mag[mid_idx] < result.h_mag[lo_idx],
            "elliptic bandstop should reject the middle stopband"
        );
        assert!(
            result.h_mag[mid_idx] < result.h_mag[hi_idx],
            "elliptic bandstop should reject the middle stopband"
        );
    }

    #[test]
    fn ellip_rejects_invalid_ripple() {
        let err = ellip(4, -1.0, 40.0, &[0.3], FilterType::Lowpass).expect_err("negative rp");
        assert!(err.is_argument_error());
        let err = ellip(4, 1.0, -40.0, &[0.3], FilterType::Lowpass).expect_err("negative rs");
        assert!(err.is_argument_error());
    }

    #[test]
    fn ellip_via_iirfilter_dispatch() {
        let direct = ellip(3, 1.0, 40.0, &[0.25], FilterType::Lowpass).expect("ellip direct");
        let dispatch = iirfilter(
            3,
            &[0.25],
            FilterType::Lowpass,
            IirFamily::Elliptic,
            Some(1.0),
            Some(40.0),
        )
        .expect("iirfilter elliptic");
        assert_ba_close(&direct, &dispatch, 1e-10, "elliptic");
    }

    #[test]
    fn chroma_normalizes_finite_bins_even_with_nan_input() {
        let mut magnitudes = vec![0.0; 12];
        magnitudes[1] = 1.0;
        magnitudes[2] = f64::NAN;
        magnitudes[3] = 2.0;

        let chroma_vec = chroma(&magnitudes, 12_000.0, 12);
        let finite_max = chroma_vec
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(0.0_f64, f64::max);

        assert!(
            (finite_max - 1.0).abs() < 1e-12,
            "finite bins should remain normalized, got max {finite_max}"
        );
        assert!(
            chroma_vec.iter().any(|value| value.is_nan()),
            "NaN input should remain visible in at least one chroma bin"
        );
    }

    #[test]
    fn spectral_centroid_ignores_unpaired_tail_magnitudes() {
        let magnitudes = [1.0, 3.0, 100.0];
        let freqs = [10.0, 30.0];
        let centroid = spectral_centroid(&magnitudes, &freqs);

        assert!(
            (centroid - 25.0).abs() < 1e-12,
            "expected paired-sample centroid, got {centroid}"
        );
    }

    #[test]
    fn spectral_rolloff_ignores_unpaired_tail_magnitudes() {
        let magnitudes = [1.0, 3.0, 100.0];
        let freqs = [10.0, 30.0];
        let rolloff = spectral_rolloff(&magnitudes, &freqs, 75.0);

        assert!(
            (rolloff - 30.0).abs() < 1e-12,
            "expected paired-sample rolloff, got {rolloff}"
        );
    }

    #[test]
    fn spectral_bandwidth_ignores_unpaired_tail_magnitudes() {
        let magnitudes = [1.0, 3.0, 100.0];
        let freqs = [10.0, 30.0];
        let bandwidth = spectral_bandwidth(&magnitudes, &freqs);

        assert!(
            (bandwidth - 8.660_254_037_844_387).abs() < 1e-12,
            "expected paired-sample bandwidth, got {bandwidth}"
        );
    }

    // ── lfilter tests ──────────────────────────────────────────────

    #[test]
    fn lfilter_identity() {
        // b=[1], a=[1] should pass signal through
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = lfilter(&[1.0], &[1.0], &x, None).expect("lfilter");
        assert_eq!(y, x);
    }

    #[test]
    fn lfilter_delay() {
        // b=[0, 1], a=[1] = one-sample delay
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = lfilter(&[0.0, 1.0], &[1.0], &x, None).expect("lfilter");
        assert!((y[0]).abs() < 1e-12);
        assert!((y[1] - 1.0).abs() < 1e-12);
        assert!((y[2] - 2.0).abs() < 1e-12);
        assert!((y[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn lfilter_first_order_iir() {
        // y[n] = x[n] + 0.5*y[n-1] (b=[1], a=[1, -0.5])
        let x = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let y = lfilter(&[1.0], &[1.0, -0.5], &x, None).expect("lfilter");
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 0.5).abs() < 1e-12);
        assert!((y[2] - 0.25).abs() < 1e-12);
        assert!((y[3] - 0.125).abs() < 1e-12);
    }

    #[test]
    fn lfilter_with_butter() {
        let coeffs = butter(2, &[0.2], FilterType::Lowpass).expect("butter");
        let x: Vec<f64> = (0..100)
            .map(|i| (2.0 * std::f64::consts::PI * 0.05 * i as f64).sin())
            .collect();
        let y = lfilter(&coeffs.b, &coeffs.a, &x, None).expect("lfilter");
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lfilter_zi_matches_scipy_reference_vectors() {
        let zi = lfilter_zi(&[0.5, 0.5], &[1.0, -0.5]).expect("first-order reference");
        assert_eq!(zi.len(), 1);
        assert!((zi[0] - 1.5).abs() <= 1.0e-12, "got {}", zi[0]);

        let zi = lfilter_zi(&[1.0, -0.25], &[2.0, -0.3]).expect("normalized reference");
        assert_eq!(zi.len(), 1);
        assert!(
            (zi[0] - (-0.058_823_529_411_764_71)).abs() <= 1.0e-12,
            "got {}",
            zi[0]
        );

        let zi = lfilter_zi(&[0.2, 0.3, 0.5], &[1.0, -0.1, 0.2]).expect("second-order reference");
        let expected = [0.709_090_909_090_909_1, 0.318_181_818_181_818_1];
        assert_eq!(zi.len(), expected.len());
        for (got, want) in zi.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }

        let zi = lfilter_zi(&[0.2; 5], &[1.0]).expect("fir reference");
        let expected = [0.8, 0.6, 0.4, 0.2];
        assert_eq!(zi.len(), expected.len());
        for (got, want) in zi.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn lfilter_zi_trims_leading_zero_denominator() {
        let zi = lfilter_zi(&[1.0, 2.0], &[0.0, 2.0, -0.6]).expect("trimmed denominator");
        assert_eq!(zi.len(), 1);
        assert!((zi[0] - 1.642_857_142_857_142_8).abs() <= 1.0e-12);
    }

    #[test]
    fn lfilter_zi_sets_step_response_steady_state() {
        let b = [0.2, 0.3, 0.5];
        let a = [1.0, -0.1, 0.2];
        let zi = lfilter_zi(&b, &a).expect("lfilter_zi");
        let y = lfilter(&b, &a, &[1.0, 1.0, 1.0], Some(&zi)).expect("lfilter");
        for value in y {
            assert!(
                (value - 0.909_090_909_090_909_2).abs() <= 1.0e-12,
                "got {value}"
            );
        }
    }

    #[test]
    fn lfilter_zi_rejects_scalar_transfer_function() {
        let err = lfilter_zi(&[0.1], &[1.0]).expect_err("scalar transfer function must fail");
        assert!(
            err.to_string()
                .contains("The length of `a` along the last axis must be at least 2."),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn lfiltic_matches_scipy_reference_vector() {
        let zi = lfiltic(&[1.0, 2.0, 3.0], &[1.0, 0.5], &[0.5], None).expect("lfiltic");
        let expected = [-0.25, 0.0];
        assert_eq!(zi.len(), expected.len());
        for (got, want) in zi.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn lfiltic_matches_non_unit_a0_reference() {
        let zi = lfiltic(
            &[0.5, 1.0, 0.2],
            &[2.0, 1.0, 3.0],
            &[1.2, -0.7, 0.4],
            Some(&[0.25, -1.5]),
        )
        .expect("lfiltic");
        let expected = [0.425, -1.775];
        assert_eq!(zi.len(), expected.len());
        for (got, want) in zi.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn lfiltic_restores_lfilter_continuation() {
        let b = [0.5, 1.0, 0.2];
        let a = [2.0, 1.0, 3.0];
        let zi = lfiltic(
            &b,
            &a,
            &[-2.72515625, 0.7971875, 1.499375, -0.70125, -0.3625, 0.725],
            Some(&[0.25, -0.5, 1.1, 0.7, -0.2, 1.3]),
        )
        .expect("lfiltic");
        let future = [0.9, -0.1, 0.3, -0.8];
        let y_future = lfilter(&b, &a, &future, Some(&zi)).expect("lfilter");
        let expected = [0.466796875, 4.3043359375, -2.73736328125, -5.147822265625];
        for (got, want) in y_future.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn lfiltic_rejects_empty_a_coefficients() {
        let err = lfiltic(&[1.0], &[], &[0.0], Some(&[0.0])).expect_err("empty a must fail");
        assert!(
            err.to_string()
                .contains("There must be at least one `a` coefficient."),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn lfiltic_rejects_zero_a0() {
        let err = lfiltic(&[1.0, 2.0], &[0.0, 2.0], &[0.0, 0.0], Some(&[0.0, 1.0]))
            .expect_err("zero a0 must fail");
        assert!(
            err.to_string()
                .contains("First `a` filter coefficient must be non-zero."),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn impulse_response_matches_scipy_transfer_function_convention() {
        let response = impulse_response(&[1.0], &[1.0, -0.5], 6).expect("impulse response");
        let expected = [0.0, 1.0, 0.5, 0.25, 0.125, 0.0625];
        for (got, want) in response.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn step_response_matches_scipy_transfer_function_convention() {
        let response = step_response(&[1.0], &[1.0, -0.5], 6).expect("step response");
        let expected = [0.0, 1.0, 1.5, 1.75, 1.875, 1.9375];
        for (got, want) in response.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn impulse_response_rejects_improper_transfer_function() {
        let err = impulse_response(&[1.0, 2.0, 3.0], &[1.0, -0.5], 6)
            .expect_err("improper transfer functions should be rejected");
        assert!(
            err.to_string().contains("improper transfer function"),
            "unexpected error: {err}"
        );
    }

    // ── filtfilt tests ─────────────────────────────────────────────

    #[test]
    fn filtfilt_zero_phase() {
        // filtfilt should produce zero phase shift
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let n = 100;
        let freq = 0.05;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).sin())
            .collect();
        let y = filtfilt(&coeffs.b, &coeffs.a, &x).expect("filtfilt");
        assert_eq!(y.len(), n);

        // filtfilt produces zero phase: verify output is in phase with input
        // by checking correlation between interior points (skip transients at edges)
        let start = 30;
        let end = 70;
        let mut corr_same = 0.0;
        let mut corr_shift = 0.0;
        for i in start..end {
            corr_same += x[i] * y[i];
            if i + 1 < n {
                corr_shift += x[i] * y[i + 1];
            }
        }
        // Zero phase means correlation at zero lag should be highest
        assert!(
            corr_same.abs() >= corr_shift.abs() * 0.9,
            "zero-phase: same-lag corr {corr_same} should be >= shifted corr {corr_shift}"
        );
    }

    #[test]
    fn filtfilt_attenuates_high_freq() {
        let coeffs = butter(3, &[0.1], FilterType::Lowpass).expect("butter");
        // Low frequency signal + high frequency noise
        let n = 200;
        let low: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 0.02 * i as f64).sin())
            .collect();
        let high: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * 0.45 * i as f64).sin())
            .collect();
        let x: Vec<f64> = low.iter().zip(high.iter()).map(|(&l, &h)| l + h).collect();

        let y = filtfilt(&coeffs.b, &coeffs.a, &x).expect("filtfilt");

        // Output should be closer to low-frequency component
        let low_err: f64 = y
            .iter()
            .zip(low.iter())
            .skip(20)
            .take(n - 40)
            .map(|(&yi, &li)| (yi - li).powi(2))
            .sum();
        let total_err: f64 = y
            .iter()
            .zip(x.iter())
            .skip(20)
            .take(n - 40)
            .map(|(&yi, &xi)| (yi - xi).powi(2))
            .sum();
        assert!(
            low_err < total_err,
            "filtered should be closer to low-freq component"
        );
    }

    #[test]
    fn filtfilt_short_input_rejected() {
        assert!(filtfilt(&[1.0], &[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn filtfilt_empty_coefficients_rejected() {
        assert!(filtfilt(&[], &[1.0], &[1.0, 2.0, 3.0]).is_err());
        assert!(filtfilt(&[1.0], &[], &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn filtfilt_padtypes_match_scipy_reference() {
        let x = [
            0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0,
            0.0, 1.0, 2.0, 1.0,
        ];
        let odd = filtfilt_with_padtype(&[0.2; 5], &[1.0], &x, Some("odd")).expect("odd");
        let even = filtfilt_with_padtype(&[0.2; 5], &[1.0], &x, Some("even")).expect("even");
        let constant =
            filtfilt_with_padtype(&[0.2; 5], &[1.0], &x, Some("constant")).expect("constant");
        let expected_odd = [
            0.0, 0.28, 0.4, 0.28, 0.0, -0.28, -0.4, -0.28, 0.0, 0.28, 0.4, 0.28, 0.0, -0.28, -0.4,
            -0.28, 0.0, 0.36, 0.72, 1.0,
        ];
        let expected_even = [
            0.96, 0.92, 0.72, 0.36, 0.0, -0.28, -0.4, -0.28, 0.0, 0.28, 0.4, 0.28, 0.0, -0.28,
            -0.4, -0.28, 0.08, 0.52, 0.88, 1.0,
        ];
        let expected_constant = [
            0.48, 0.6, 0.56, 0.32, 0.0, -0.28, -0.4, -0.28, 0.0, 0.28, 0.4, 0.28, 0.0, -0.28, -0.4,
            -0.28, 0.04, 0.44, 0.8, 1.0,
        ];

        for (actual, want) in odd.iter().zip(expected_odd.iter()) {
            assert!((actual - want).abs() < 1e-12, "{actual} vs {want}");
        }
        for (actual, want) in even.iter().zip(expected_even.iter()) {
            assert!((actual - want).abs() < 1e-12, "{actual} vs {want}");
        }
        for (actual, want) in constant.iter().zip(expected_constant.iter()) {
            assert!((actual - want).abs() < 1e-12, "{actual} vs {want}");
        }
    }

    #[test]
    fn filtfilt_rejects_unknown_padtype() {
        let err = filtfilt_with_padtype(&[0.2; 5], &[1.0], &[0.0, 1.0, 2.0, 1.0], Some("wrap"))
            .expect_err("invalid padtype");
        assert!(err.is_argument_error());
    }

    // ── Periodogram tests ──────────────────────────────────────────

    #[test]
    fn periodogram_single_frequency() {
        // Pure sine wave at 10 Hz, sampled at 100 Hz
        let fs = 100.0;
        let freq = 10.0;
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
            .collect();

        let result = periodogram(&x, fs, None).expect("periodogram");
        assert_eq!(result.frequencies.len(), result.psd.len());
        assert!(!result.psd.is_empty());

        // Peak should be near 10 Hz
        let peak_idx = result
            .psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        let peak_freq = result.frequencies[peak_idx];
        assert!(
            (peak_freq - freq).abs() < 1.0,
            "peak at {peak_freq} Hz, expected ~{freq} Hz"
        );
    }

    #[test]
    fn periodogram_with_hann_window() {
        let fs = 100.0;
        let n = 128;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / fs).sin())
            .collect();
        let window = hann(n);
        let result = periodogram(&x, fs, Some(&window)).expect("periodogram");
        assert_eq!(result.psd.len(), n / 2 + 1);
        assert!(result.psd.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn periodogram_empty_rejected() {
        assert!(periodogram(&[], 1.0, None).is_err());
    }

    // ── Welch tests ────────────────────────────────────────────────

    #[test]
    fn welch_basic() {
        let fs = 1000.0;
        let n = 1024;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 50.0 * i as f64 / fs).sin())
            .collect();

        let result = welch(&x, fs, None, Some(256), None).expect("welch");
        assert!(!result.frequencies.is_empty());
        assert_eq!(result.frequencies.len(), result.psd.len());

        // Peak should be near 50 Hz
        let peak_idx = result
            .psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        let peak_freq = result.frequencies[peak_idx];
        assert!(
            (peak_freq - 50.0).abs() < 5.0,
            "peak at {peak_freq} Hz, expected ~50 Hz"
        );
    }

    #[test]
    fn welch_reduces_variance() {
        // Welch with more averaging should have smoother PSD than periodogram
        let fs = 100.0;
        let n = 1024;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin()
                    + if i % 3 == 0 { 0.5 } else { -0.5 } // noise
            })
            .collect();

        let psd_raw = periodogram(&x, fs, None).expect("periodogram");
        let psd_welch = welch(&x, fs, None, Some(128), None).expect("welch");

        // Welch PSD should be smoother (lower variance)
        let raw_var = variance_of_psd(&psd_raw.psd);
        let welch_var = variance_of_psd(&psd_welch.psd);
        assert!(
            welch_var < raw_var,
            "Welch should be smoother: var={welch_var} < {raw_var}"
        );
    }

    #[test]
    fn welch_empty_rejected() {
        assert!(welch(&[], 1.0, None, None, None).is_err());
    }

    #[test]
    fn welch_nperseg_clamps_to_signal_length() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let oversized = welch(&x, 10.0, None, Some(16), None).expect("welch oversized");
        let clamped = welch(&x, 10.0, None, Some(x.len()), None).expect("welch clamped");
        let expected_frequencies = [0.0, 2.0, 4.0];

        assert_eq!(oversized.frequencies.len(), expected_frequencies.len());
        assert_eq!(oversized.frequencies.len(), clamped.frequencies.len());
        assert_eq!(oversized.psd.len(), clamped.psd.len());

        for (index, ((actual, expected), clamped_frequency)) in oversized
            .frequencies
            .iter()
            .zip(expected_frequencies.iter())
            .zip(clamped.frequencies.iter())
            .enumerate()
        {
            assert_close(
                *actual,
                *expected,
                1e-12,
                &format!("frequency[{index}] should match SciPy clamp"),
            );
            assert_close(
                *actual,
                *clamped_frequency,
                1e-12,
                &format!("frequency[{index}] should match explicit clamp"),
            );
        }

        for (index, (actual, clamped_psd)) in
            oversized.psd.iter().zip(clamped.psd.iter()).enumerate()
        {
            assert_close(
                *actual,
                *clamped_psd,
                1e-12,
                &format!("psd[{index}] should match explicit clamp"),
            );
        }
    }

    fn variance_of_psd(psd: &[f64]) -> f64 {
        let n = psd.len() as f64;
        let mean = psd.iter().sum::<f64>() / n;
        psd.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
    }

    fn gd_span(values: &[f64]) -> f64 {
        let min = values
            .iter()
            .copied()
            .fold(f64::INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            });
        let max = values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        max - min
    }

    // ── Filter conversion tests ────────────────────────────────────

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{msg}: {a} vs {b} (diff={})",
            (a - b).abs()
        );
    }

    fn assert_ba_close(direct: &BaCoeffs, dispatch: &BaCoeffs, tol: f64, family: &str) {
        assert_eq!(direct.b.len(), dispatch.b.len());
        assert_eq!(direct.a.len(), dispatch.a.len());
        for (lhs, rhs) in direct.b.iter().zip(dispatch.b.iter()) {
            assert!(
                (lhs - rhs).abs() < tol,
                "{family} dispatcher mismatch in b: {lhs} vs {rhs}"
            );
        }
        for (lhs, rhs) in direct.a.iter().zip(dispatch.a.iter()) {
            assert!(
                (lhs - rhs).abs() < tol,
                "{family} dispatcher mismatch in a: {lhs} vs {rhs}"
            );
        }
    }

    fn nearest_freq_index(w: &[f64], omega: f64) -> usize {
        w.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| ((**a) - omega).abs().total_cmp(&((**b) - omega).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[test]
    fn tf2zpk_first_order() {
        // b=[1, -0.5], a=[1, -0.9] → zero at 0.5, pole at 0.9
        let zpk = tf2zpk(&[1.0, -0.5], &[1.0, -0.9]).expect("tf2zpk");
        assert_eq!(zpk.zeros_re.len(), 1);
        assert_eq!(zpk.poles_re.len(), 1);
        assert_close(zpk.zeros_re[0], 0.5, 1e-10, "zero");
        assert_close(zpk.poles_re[0], 0.9, 1e-10, "pole");
        assert_close(zpk.gain, 1.0, 1e-10, "gain");
    }

    #[test]
    fn tf2zpk_second_order_complex() {
        // butter(2, 0.3) should give complex conjugate poles
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let zpk = tf2zpk(&coeffs.b, &coeffs.a).expect("tf2zpk");
        assert_eq!(zpk.poles_re.len(), 2);
        // Poles should be complex conjugates
        assert_close(zpk.poles_re[0], zpk.poles_re[1], 1e-10, "pole re match");
        assert!(
            (zpk.poles_im[0] + zpk.poles_im[1]).abs() < 1e-10,
            "poles should be conjugate"
        );
        // Poles should be inside unit circle (stable)
        let mag = (zpk.poles_re[0].powi(2) + zpk.poles_im[0].powi(2)).sqrt();
        assert!(mag < 1.0, "poles inside unit circle, got mag={mag}");
    }

    #[test]
    fn zpk2tf_roundtrip() {
        let b_orig = [0.1, 0.2, 0.1];
        let a_orig = [1.0, -0.5, 0.2];
        let zpk = tf2zpk(&b_orig, &a_orig).expect("tf2zpk");
        let ba = zpk2tf(&zpk);

        // Normalize and compare
        let scale_b = b_orig[0] / ba.b[0];
        let scale_a = a_orig[0] / ba.a[0];
        for (i, (&orig, &recovered)) in b_orig.iter().zip(ba.b.iter()).enumerate() {
            assert_close(
                recovered * scale_b,
                orig,
                1e-8,
                &format!("b[{i}] roundtrip"),
            );
        }
        for (i, (&orig, &recovered)) in a_orig.iter().zip(ba.a.iter()).enumerate() {
            assert_close(
                recovered * scale_a,
                orig,
                1e-8,
                &format!("a[{i}] roundtrip"),
            );
        }
    }

    #[test]
    fn tf2sos_butter2() {
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        // Order 2 → 1 SOS section
        assert_eq!(sos.len(), 1, "butter(2) should give 1 SOS section");
        // a0 should be 1.0
        assert_close(sos[0][3], 1.0, 1e-10, "a0 = 1");
    }

    #[test]
    fn tf2sos_butter4() {
        let coeffs = butter(4, &[0.3], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        // Order 4 → 2 SOS sections
        assert_eq!(sos.len(), 2, "butter(4) should give 2 SOS sections");
    }

    #[test]
    fn butter_with_output_ba_matches_legacy_butter() {
        let direct = butter(4, &[0.25], FilterType::Lowpass).expect("butter");
        let via_output = butter_with_output(4, &[0.25], FilterType::Lowpass, ButterOutput::Ba)
            .expect("butter_with_output");
        assert!(matches!(via_output, BaCoeffsOrSos::Ba(_)));
        let via_output = match via_output {
            BaCoeffsOrSos::Ba(ba) => ba,
            BaCoeffsOrSos::Sos(_) => return,
        };

        assert_eq!(direct.b.len(), via_output.b.len());
        assert_eq!(direct.a.len(), via_output.a.len());
        for (i, (&got, &expected)) in via_output.b.iter().zip(direct.b.iter()).enumerate() {
            assert_close(got, expected, 1e-12, &format!("b[{i}]"));
        }
        for (i, (&got, &expected)) in via_output.a.iter().zip(direct.a.iter()).enumerate() {
            assert_close(got, expected, 1e-12, &format!("a[{i}]"));
        }
    }

    #[test]
    fn butter_sos_matches_scipy_reference_sections() {
        let sos = butter_sos(4, &[0.25], FilterType::Lowpass).expect("butter_sos");
        let expected = [
            [
                0.010209480791203138,
                0.020418961582406275,
                0.010209480791203138,
                1.0,
                -0.8553979327751704,
                0.20971535775655478,
            ],
            [1.0, 2.0, 1.0, 1.0, -1.1130298541633479, 0.5740619150839545],
        ];
        assert_eq!(sos.len(), expected.len(), "section count");
        for (section_idx, (got, expected_section)) in sos.iter().zip(expected.iter()).enumerate() {
            for (coeff_idx, (&got_coeff, &expected_coeff)) in
                got.iter().zip(expected_section.iter()).enumerate()
            {
                assert_close(
                    got_coeff,
                    expected_coeff,
                    1e-7,
                    &format!("section {section_idx} coeff {coeff_idx}"),
                );
            }
        }
    }

    #[test]
    fn sos2tf_roundtrip() {
        let coeffs = butter(3, &[0.4], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        let recovered = sos2tf(&sos);

        // DC gain should match: sum(b)/sum(a) for both
        let orig_dc: f64 = coeffs.b.iter().sum::<f64>() / coeffs.a.iter().sum::<f64>();
        let rec_dc: f64 = recovered.b.iter().sum::<f64>() / recovered.a.iter().sum::<f64>();
        assert_close(orig_dc, rec_dc, 1e-8, "DC gain roundtrip");
    }

    #[test]
    fn zpk2sos_simple_real_poles() {
        let zpk = ZpkCoeffs {
            zeros_re: vec![-1.0],
            zeros_im: vec![0.0],
            poles_re: vec![0.5],
            poles_im: vec![0.0],
            gain: 2.0,
        };
        let sos = zpk2sos(&zpk);
        assert!(!sos.is_empty());
        // Single real pole/zero → 1 section
        // Denominator: (1 - 0.5*z^-1)
        assert_close(sos[0][4], -0.5, 1e-10, "a1 = -pole");
    }

    #[test]
    fn sos2zpk_roundtrip() {
        let coeffs = butter(4, &[0.25], FilterType::Lowpass).expect("butter");
        let zpk_orig = tf2zpk(&coeffs.b, &coeffs.a).expect("tf2zpk");
        let sos = zpk2sos(&zpk_orig);
        let zpk_rec = sos2zpk(&sos);

        // Same number of poles and zeros
        assert_eq!(
            zpk_orig.poles_re.len(),
            zpk_rec.poles_re.len(),
            "pole count"
        );
        assert_eq!(
            zpk_orig.zeros_re.len(),
            zpk_rec.zeros_re.len(),
            "zero count"
        );
    }

    #[test]
    fn tf2sos_high_order_stability() {
        // Order 8 filter — SOS should be more stable than tf form
        let coeffs = butter(8, &[0.2], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        assert_eq!(sos.len(), 4, "order 8 → 4 sections");

        // Each section's poles should be inside unit circle
        for (i, section) in sos.iter().enumerate() {
            let a2 = section[5];
            // For stability: |a2| < 1
            assert!(a2.abs() < 1.0 + 1e-10, "section {i} unstable: a2={a2}");
        }
    }

    #[test]
    fn poly_multiply_basic() {
        // (1 + 2z^-1) * (1 + 3z^-1) = 1 + 5z^-1 + 6z^-2
        let result = poly_multiply(&[1.0, 2.0], &[1.0, 3.0]);
        assert_eq!(result.len(), 3);
        assert_close(result[0], 1.0, 1e-12, "c0");
        assert_close(result[1], 5.0, 1e-12, "c1");
        assert_close(result[2], 6.0, 1e-12, "c2");
    }

    #[test]
    fn tf2zpk_empty_rejected() {
        assert!(tf2zpk(&[], &[1.0]).is_err());
        assert!(tf2zpk(&[1.0], &[]).is_err());
    }

    // ── Frequency response tests ───────────────────────────────────

    #[test]
    fn freqz_unity_gain() {
        // H(z) = 1 (b=[1], a=[1]) → magnitude 1 everywhere
        let result = freqz(&[1.0], &[1.0], Some(64)).expect("freqz");
        assert_eq!(result.w.len(), 64);
        assert_eq!(result.h_mag.len(), 64);
        for (i, &mag) in result.h_mag.iter().enumerate() {
            assert_close(mag, 1.0, 1e-12, &format!("unity mag at bin {i}"));
        }
    }

    #[test]
    fn freqz_whole_matches_scipy_reference() {
        let tol = 1e-8;
        let result = freqz_with_whole(&[1.0, 0.5, 0.25], &[1.0, -0.3, 0.2], Some(8), true)
            .expect("freqz whole");
        let expected_mag = [
            1.944_444_44,
            1.880_828_33,
            1.054_994_64,
            0.511_363_23,
            0.5,
            0.511_363_23,
            1.054_994_64,
            1.880_828_33,
        ];
        let expected_phase = [
            0.0,
            -0.434_838_91,
            -0.946_773_27,
            -0.486_582_96,
            0.0,
            0.486_582_96,
            0.946_773_27,
            0.434_838_91,
        ];

        assert_eq!(result.w.len(), 8);
        assert_close(result.w[1], std::f64::consts::FRAC_PI_4, 1e-12, "pi/4 bin");
        assert_close(result.w[4], std::f64::consts::PI, 1e-12, "pi bin");
        assert_close(
            result.w[7],
            7.0 * std::f64::consts::PI / 4.0,
            1e-12,
            "7pi/4 bin",
        );

        for (i, (&actual, &want)) in result.h_mag.iter().zip(expected_mag.iter()).enumerate() {
            assert_close(actual, want, tol, &format!("whole magnitude bin {i}"));
        }

        for (i, (&actual, &want)) in result.h_phase.iter().zip(expected_phase.iter()).enumerate() {
            assert_close(actual, want, tol, &format!("whole phase bin {i}"));
        }
    }

    #[test]
    fn freqz_butter_at_cutoff() {
        // Butter(2, 0.3): at cutoff frequency, magnitude should be ≈ 1/√2 (-3dB)
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");

        // Cutoff at ω = 0.3π
        let cutoff_omega = 0.3 * std::f64::consts::PI;
        // Find nearest frequency bin
        let cutoff_idx = result
            .w
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a) - cutoff_omega)
                    .abs()
                    .total_cmp(&((**b) - cutoff_omega).abs())
            })
            .map(|(i, _)| i)
            .unwrap();

        let mag_at_cutoff = result.h_mag[cutoff_idx];
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!(
            (mag_at_cutoff - inv_sqrt2).abs() < 0.05,
            "mag at cutoff should be ~{inv_sqrt2}, got {mag_at_cutoff}"
        );
    }

    #[test]
    fn freqz_lowpass_dc_unity() {
        // Lowpass filter: DC gain (ω=0) should be 1
        let coeffs = butter(3, &[0.4], FilterType::Lowpass).expect("butter");
        let result = freqz(&coeffs.b, &coeffs.a, Some(256)).expect("freqz");
        assert_close(result.h_mag[0], 1.0, 1e-6, "DC gain = 1");
    }

    #[test]
    fn freqz_highpass_nyquist_unity() {
        // Highpass filter: Nyquist gain (ω=π) should be 1
        let coeffs = butter(2, &[0.3], FilterType::Highpass).expect("butter");
        let result = freqz(&coeffs.b, &coeffs.a, Some(256)).expect("freqz");
        let last_idx = result.h_mag.len() - 1;
        assert_close(result.h_mag[last_idx], 1.0, 1e-6, "Nyquist gain = 1");
    }

    #[test]
    fn freqz_sos_matches_ba() {
        // SOS and BA forms should give same frequency response
        let coeffs = butter(4, &[0.25], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");

        let ba_result = freqz(&coeffs.b, &coeffs.a, Some(128)).expect("freqz ba");
        let sos_result = freqz_sos(&sos, Some(128)).expect("freqz sos");

        for i in 0..128 {
            assert_close(
                ba_result.h_mag[i],
                sos_result.h_mag[i],
                1e-4,
                &format!("mag match at bin {i}"),
            );
        }
    }

    #[test]
    fn freqz_monotone_lowpass() {
        // Butterworth lowpass: magnitude should be monotonically decreasing
        let coeffs = butter(4, &[0.3], FilterType::Lowpass).expect("butter");
        let result = freqz(&coeffs.b, &coeffs.a, Some(256)).expect("freqz");
        for i in 1..result.h_mag.len() {
            assert!(
                result.h_mag[i] <= result.h_mag[i - 1] + 1e-10,
                "butter lowpass should be monotone decreasing: mag[{i}]={} > mag[{}]={}",
                result.h_mag[i],
                i - 1,
                result.h_mag[i - 1]
            );
        }
    }

    #[test]
    fn group_delay_fir_linear_phase() {
        // A symmetric FIR filter has constant group delay = (N-1)/2
        let b = vec![1.0, 2.0, 3.0, 2.0, 1.0]; // symmetric, N=5
        let a = vec![1.0];
        let result = freqz(&b, &a, Some(64)).expect("freqz for gd check");
        let (w, gd) = group_delay(&b, &a, Some(64)).expect("group_delay");
        assert_eq!(w.len(), 64);

        // Expected group delay = (5-1)/2 = 2.0 samples
        // Only check at frequencies where the filter has significant magnitude
        // (group delay is undefined at filter nulls where |H|≈0)
        for (i, (&gd_val, &mag)) in gd.iter().zip(result.h_mag.iter()).enumerate() {
            if mag > 0.1 {
                assert!(
                    (gd_val - 2.0).abs() < 0.15,
                    "group delay at bin {i} (mag={mag:.3}) should be ~2.0, got {gd_val}",
                );
            }
        }
    }

    #[test]
    fn phase_delay_fir_linear_phase() {
        let b = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let a = vec![1.0];
        let result = freqz(&b, &a, Some(64)).expect("freqz for pd check");
        let (w, pd) = phase_delay(&b, &a, Some(64)).expect("phase_delay");
        assert_eq!(w.len(), 64);

        for (i, (&pd_val, &mag)) in pd.iter().zip(result.h_mag.iter()).enumerate() {
            if mag > 0.1 {
                assert!(
                    (pd_val - 2.0).abs() < 0.15,
                    "phase delay at bin {i} (mag={mag:.3}) should be ~2.0, got {pd_val}",
                );
            }
        }
    }

    #[test]
    fn phase_delay_matches_group_delay_for_pure_delay() {
        let b = vec![0.0, 0.0, 1.0];
        let a = vec![1.0];
        let (_, gd) = group_delay(&b, &a, Some(64)).expect("group_delay");
        let (_, pd) = phase_delay(&b, &a, Some(64)).expect("phase_delay");

        for (i, (&gd_val, &pd_val)) in gd.iter().zip(pd.iter()).enumerate().skip(1) {
            assert!(
                (gd_val - pd_val).abs() < 1e-10,
                "pure delay mismatch at bin {i}: gd={gd_val}, pd={pd_val}",
            );
        }
    }

    #[test]
    fn phase_delay_dc_uses_group_delay_limit() {
        let b = vec![1.0, 0.0, -1.0];
        let a = vec![1.0];
        let (_, gd) = group_delay(&b, &a, Some(32)).expect("group_delay");
        let (_, pd) = phase_delay(&b, &a, Some(32)).expect("phase_delay");
        assert!(
            (pd[0] - gd[0]).abs() < 1e-12,
            "phase delay DC limit should match group delay: {} vs {}",
            pd[0],
            gd[0]
        );
    }

    #[test]
    fn phase_delay_rejects_empty_coefficients() {
        assert!(phase_delay(&[], &[1.0], Some(16)).is_err());
        assert!(phase_delay(&[1.0], &[], Some(16)).is_err());
    }

    #[test]
    fn group_delay_rejects_zero_frequency_count() {
        let err = group_delay(&[1.0], &[1.0], Some(0)).expect_err("n_freqs=0 should fail");
        assert_eq!(
            err,
            SignalError::InvalidArgument("n_freqs must be > 0".to_string())
        );
    }

    #[test]
    fn freqz_empty_rejected() {
        assert!(freqz(&[], &[1.0], None).is_err());
        assert!(freqz(&[1.0], &[], None).is_err());
    }

    // ── SOS filtering tests ────────────────────────────────────────

    #[test]
    fn sosfilt_matches_lfilter_low_order() {
        // For low-order filter, sosfilt and lfilter should give same results
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");

        let x: Vec<f64> = (0..100)
            .map(|i| (2.0 * std::f64::consts::PI * 0.05 * i as f64).sin())
            .collect();

        let y_lfilter = lfilter(&coeffs.b, &coeffs.a, &x, None).expect("lfilter");
        let y_sosfilt = sosfilt(&sos, &x).expect("sosfilt");

        for (i, (&yl, &ys)) in y_lfilter.iter().zip(y_sosfilt.iter()).enumerate() {
            assert!(
                (yl - ys).abs() < 1e-8,
                "sosfilt vs lfilter at sample {i}: {ys} vs {yl}",
            );
        }
    }

    #[test]
    fn sosfilt_identity_passthrough() {
        // SOS with unity gain sections: output = input
        let sos = vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = sosfilt(&sos, &x).expect("sosfilt identity");
        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            assert!((xi - yi).abs() < 1e-12, "passthrough at {i}: {yi} != {xi}",);
        }
    }

    #[test]
    fn sosfilt_high_order_stable() {
        // Order-8 Butterworth: sosfilt should remain stable
        let coeffs = butter(8, &[0.2], FilterType::Lowpass).expect("butter 8");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");

        let x: Vec<f64> = (0..200)
            .map(|i| (2.0 * std::f64::consts::PI * 0.05 * i as f64).sin())
            .collect();

        let y = sosfilt(&sos, &x).expect("sosfilt");
        // Output should be finite (stable) and bounded
        assert!(
            y.iter().all(|v| v.is_finite()),
            "sosfilt output should be finite for stable filter"
        );
        let max_abs = y.iter().map(|v| v.abs()).fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        assert!(max_abs < 2.0, "output should be bounded (max={max_abs})");
    }

    #[test]
    fn sosfiltfilt_zero_phase() {
        // sosfiltfilt should produce zero phase shift
        let coeffs = butter(3, &[0.3], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");

        let n = 200;
        let freq = 0.05;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).sin())
            .collect();

        let y = sosfiltfilt(&sos, &x).expect("sosfiltfilt");
        assert_eq!(y.len(), n);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sosfilt_zi_steady_state() {
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        let zi = sosfilt_zi(&sos).expect("sosfilt_zi");
        assert_eq!(zi.len(), sos.len());
        // Each section should have finite initial conditions
        for (i, zi_sec) in zi.iter().enumerate() {
            assert!(
                zi_sec[0].is_finite() && zi_sec[1].is_finite(),
                "section {i} zi should be finite",
            );
        }
    }

    #[test]
    fn sosfilt_empty_rejected() {
        assert!(sosfilt(&[], &[1.0]).is_err());
    }

    #[test]
    fn sosfiltfilt_short_rejected() {
        let sos = vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        assert!(sosfiltfilt(&sos, &[1.0, 2.0]).is_err());
    }

    // ── FIR filter design tests ──────────────────────────────────────

    #[test]
    fn firwin_lowpass_symmetric() {
        // FIR lowpass filter should have symmetric (linear phase) coefficients
        let h = firwin(21, &[0.3], FirWindow::Hamming, true).expect("firwin lowpass");
        assert_eq!(h.len(), 21);
        for i in 0..10 {
            assert!(
                (h[i] - h[20 - i]).abs() < 1e-12,
                "symmetry broken at {i}: h[{i}]={}, h[{}]={}",
                h[i],
                20 - i,
                h[20 - i]
            );
        }
    }

    #[test]
    fn firwin_lowpass_dc_gain() {
        // Lowpass with pass_zero=true should have unity DC gain
        let h = firwin(31, &[0.4], FirWindow::Hamming, true).expect("firwin lowpass");
        let dc_gain: f64 = h.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < 0.05,
            "DC gain should be ~1.0, got {dc_gain}"
        );
    }

    #[test]
    fn firwin_highpass() {
        // Highpass: pass_zero=false, DC gain should be ~0
        let h = firwin(31, &[0.4], FirWindow::Hamming, false).expect("firwin highpass");
        let dc_gain: f64 = h.iter().sum();
        assert!(
            dc_gain.abs() < 0.05,
            "highpass DC gain should be ~0, got {dc_gain}"
        );
    }

    #[test]
    fn firwin_bandpass() {
        // Bandpass: two cutoff frequencies, pass_zero=false
        let h = firwin(51, &[0.2, 0.5], FirWindow::Hamming, false).expect("firwin bandpass");
        assert_eq!(h.len(), 51);
        let dc_gain: f64 = h.iter().sum();
        assert!(
            dc_gain.abs() < 0.1,
            "bandpass DC gain should be ~0, got {dc_gain}"
        );
    }

    #[test]
    fn firwin_bandstop() {
        // Bandstop: two cutoff frequencies, pass_zero=true
        let h = firwin(51, &[0.2, 0.5], FirWindow::Hamming, true).expect("firwin bandstop");
        let dc_gain: f64 = h.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < 0.1,
            "bandstop DC gain should be ~1.0, got {dc_gain}"
        );
    }

    #[test]
    fn firwin_cutoff_attenuation() {
        // At the cutoff frequency, magnitude should be approximately -6dB (≈0.5)
        let numtaps = 51;
        let cutoff = 0.3;
        let h = firwin(numtaps, &[cutoff], FirWindow::Hamming, true).expect("firwin");
        // Compute magnitude at cutoff
        let alpha = (numtaps - 1) as f64 / 2.0;
        let omega = std::f64::consts::PI * cutoff;
        let mut mag_real = 0.0;
        let mut mag_imag = 0.0;
        for (n, &hn) in h.iter().enumerate() {
            let phase = omega * (n as f64 - alpha);
            mag_real += hn * phase.cos();
            mag_imag += hn * phase.sin();
        }
        let mag = (mag_real * mag_real + mag_imag * mag_imag).sqrt();
        assert!(
            (mag - 0.5).abs() < 0.15,
            "magnitude at cutoff should be ~0.5 (-6dB), got {mag}"
        );
    }

    #[test]
    fn firwin2_lowpass_basic() {
        // Flat passband from 0 to 0.3, stopband from 0.5 to 1.0
        let freq = vec![0.0, 0.3, 0.5, 1.0];
        let gain = vec![1.0, 1.0, 0.0, 0.0];
        let h = firwin2(51, &freq, &gain, FirWindow::Hamming).expect("firwin2");
        assert_eq!(h.len(), 51);
        // DC gain should be ~1
        let dc_gain: f64 = h.iter().sum();
        assert!((dc_gain - 1.0).abs() < 0.15, "firwin2 DC gain: {dc_gain}");
    }

    #[test]
    fn kaiserord_typical_specs() {
        // 60dB attenuation, transition width 0.1
        let (numtaps, beta) = kaiserord(60.0, 0.1).expect("kaiserord");
        assert!(
            numtaps > 20 && numtaps < 100,
            "numtaps={numtaps} should be 20-100 for 60dB/0.1"
        );
        assert!(
            beta > 4.0 && beta < 7.0,
            "beta={beta} should be 4-7 for 60dB"
        );
    }

    #[test]
    fn kaiserord_mild_specs() {
        // 20dB attenuation, wide transition
        let (numtaps, beta) = kaiserord(20.0, 0.2).expect("kaiserord mild");
        assert!(
            numtaps < 30,
            "numtaps={numtaps} should be small for mild specs"
        );
        // beta should be 0 for attenuation <= 21
        assert!(beta < 0.5, "beta={beta} should be ~0 for A<=21");
    }

    #[test]
    fn kaiserord_rejects_non_finite_args() {
        assert!(kaiserord(f64::NAN, 0.1).is_err());
        assert!(kaiserord(60.0, f64::NAN).is_err());
        assert!(kaiserord(f64::INFINITY, 0.1).is_err());
        assert!(kaiserord(60.0, f64::INFINITY).is_err());
    }

    #[test]
    fn firwin_invalid_args() {
        assert!(firwin(0, &[0.3], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[0.0], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[1.0], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[1.5], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[0.4, 0.2], FirWindow::Hamming, false).is_err());
        assert!(firwin(21, &[0.2, 0.2], FirWindow::Hamming, false).is_err());
    }

    #[test]
    fn firwin_kaiser_window() {
        // Test that Kaiser window variant works
        let h = firwin(31, &[0.3], FirWindow::Kaiser(5.0), true).expect("firwin kaiser");
        assert_eq!(h.len(), 31);
        let dc_gain: f64 = h.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < 0.05,
            "Kaiser lowpass DC gain: {dc_gain}"
        );
    }

    // ── Chirp tests ────────────────────────────────────────────────

    #[test]
    fn chirp_linear_endpoints() {
        // At t=0, instantaneous frequency = f0, so signal ≈ cos(0) = 1.0
        let t: Vec<f64> = vec![0.0];
        let sig = chirp(&t, 10.0, 1.0, 50.0, ChirpMethod::Linear).unwrap();
        assert!(
            (sig[0] - 1.0).abs() < 1e-12,
            "chirp at t=0 should be cos(0)=1, got {}",
            sig[0]
        );
    }

    #[test]
    fn chirp_linear_frequency_sweep() {
        // Generate chirp from 1 Hz to 10 Hz over 1 second; verify it oscillates
        let n = 1000;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let sig = chirp(&t, 1.0, 1.0, 10.0, ChirpMethod::Linear).unwrap();
        assert_eq!(sig.len(), n);
        // Count zero crossings — should increase with frequency
        let crossings: usize = sig
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        // With sweep from 1-10 Hz over 1s, average freq ~5.5 Hz → ~11 crossings
        assert!(crossings > 5, "too few zero crossings: {crossings}");
    }

    #[test]
    fn chirp_logarithmic() {
        let t: Vec<f64> = vec![0.0];
        let sig = chirp(&t, 10.0, 1.0, 100.0, ChirpMethod::Logarithmic).unwrap();
        assert!((sig[0] - 1.0).abs() < 1e-10, "log chirp at t=0: {}", sig[0]);
    }

    #[test]
    fn chirp_logarithmic_rejects_zero_freq() {
        let t = vec![0.0];
        assert!(chirp(&t, 0.0, 1.0, 10.0, ChirpMethod::Logarithmic).is_err());
    }

    #[test]
    fn chirp_rejects_non_finite_parameters() {
        let t = vec![0.0, 0.5, 1.0];
        assert!(chirp(&t, 1.0, f64::NAN, 2.0, ChirpMethod::Linear).is_err());
        assert!(chirp(&t, 1.0, f64::INFINITY, 2.0, ChirpMethod::Linear).is_err());
        assert!(chirp(&t, f64::NAN, 1.0, 2.0, ChirpMethod::Linear).is_err());
        assert!(chirp(&t, 1.0, 1.0, f64::INFINITY, ChirpMethod::Linear).is_err());
        assert!(chirp(&t, f64::NAN, 1.0, 2.0, ChirpMethod::Logarithmic).is_err());
    }

    // ── Sawtooth / Square tests ────────────────────────────────────

    #[test]
    fn sawtooth_period_and_amplitude() {
        let two_pi = 2.0 * std::f64::consts::PI;
        // Rising sawtooth (width=1): goes from -1 to +1 over one period
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 100.0 * two_pi).collect();
        let sig = sawtooth(&t, 1.0).unwrap();
        // First sample at phase ≈ 0 should be near -1
        assert!(sig[0] > -1.1 && sig[0] < -0.9, "start: {}", sig[0]);
        // Mid-period should be near 0
        assert!(sig[50].abs() < 0.1, "mid: {}", sig[50]);
    }

    #[test]
    fn sawtooth_triangle_wave() {
        let two_pi = 2.0 * std::f64::consts::PI;
        // Triangle wave (width=0.5): symmetric
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 100.0 * two_pi).collect();
        let sig = sawtooth(&t, 0.5).unwrap();
        // Peak should be at phase = 0.5*2π = π (index 50)
        // All values should be in [-1, 1]
        for (i, &v) in sig.iter().enumerate() {
            assert!(
                (-1.0 - 1e-10..=1.0 + 1e-10).contains(&v),
                "triangle out of range at {i}: {v}"
            );
        }
    }

    #[test]
    fn square_wave_duty_cycle() {
        let two_pi = 2.0 * std::f64::consts::PI;
        let t: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0 * two_pi).collect();
        let sig = square(&t, 0.5).unwrap();
        // Count +1 samples — should be ~50%
        let pos_count = sig.iter().filter(|&&v| v > 0.0).count();
        assert!(
            (pos_count as f64 / 1000.0 - 0.5).abs() < 0.02,
            "duty cycle: {}",
            pos_count as f64 / 1000.0
        );
    }

    #[test]
    fn square_wave_values() {
        let two_pi = 2.0 * std::f64::consts::PI;
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 100.0 * two_pi).collect();
        let sig = square(&t, 0.5).unwrap();
        // All values should be exactly +1 or -1
        for &v in &sig {
            assert!(v == 1.0 || v == -1.0, "unexpected value: {v}");
        }
    }

    // ── Unit impulse tests ─────────────────────────────────────────

    #[test]
    fn unit_impulse_default_at_zero() {
        let imp = unit_impulse(10, None).unwrap();
        assert_eq!(imp.len(), 10);
        assert!((imp[0] - 1.0).abs() < 1e-12);
        for &v in &imp[1..] {
            assert!((v).abs() < 1e-12);
        }
    }

    #[test]
    fn unit_impulse_at_index() {
        let imp = unit_impulse(5, Some(3)).unwrap();
        assert!((imp[3] - 1.0).abs() < 1e-12);
        assert!((imp[0]).abs() < 1e-12);
        assert!((imp[4]).abs() < 1e-12);
    }

    #[test]
    fn unit_impulse_out_of_range() {
        assert!(unit_impulse(5, Some(5)).is_err());
    }

    // ── Detrend tests ──────────────────────────────────────────────

    #[test]
    fn detrend_linear_removes_slope() {
        // y = 2x + 3 → detrended should be ~0 everywhere
        let data: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let result = detrend(&data, DetrendType::Linear).unwrap();
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-10, "detrend linear residual at {i}: {v}");
        }
    }

    #[test]
    fn detrend_constant_removes_mean() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let result = detrend(&data, DetrendType::Constant).unwrap();
        for &v in &result {
            assert!(v.abs() < 1e-12, "detrend constant: {v}");
        }
    }

    #[test]
    fn detrend_preserves_residual() {
        // y = 2x + noise → detrend should remove trend, preserve noise structure
        let noise = [0.1, -0.2, 0.05, -0.15, 0.3];
        let data: Vec<f64> = noise
            .iter()
            .enumerate()
            .map(|(i, &n)| 3.0 * i as f64 + 10.0 + n)
            .collect();
        let result = detrend(&data, DetrendType::Linear).unwrap();
        // Residuals should be close to the noise (within tolerance of linear fit error)
        let max_residual = result
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        assert!(max_residual < 0.5, "max residual: {max_residual}");
    }

    // ── Medfilt tests ──────────────────────────────────────────────

    #[test]
    fn medfilt_removes_impulse_noise() {
        // Clean signal with spike impulses
        let mut data = vec![1.0; 20];
        data[5] = 100.0; // spike
        data[15] = -100.0; // spike
        let filtered = medfilt(&data, 3).unwrap();
        // Spikes should be removed
        assert!(
            (filtered[5] - 1.0).abs() < 1e-12,
            "spike at 5 not removed: {}",
            filtered[5]
        );
        assert!(
            (filtered[15] - 1.0).abs() < 1e-12,
            "spike at 15 not removed: {}",
            filtered[15]
        );
    }

    #[test]
    fn medfilt_preserves_edges() {
        // Step function: medfilt should preserve the step
        let mut data = vec![0.0; 10];
        for v in &mut data[5..] {
            *v = 1.0;
        }
        let filtered = medfilt(&data, 3).unwrap();
        // Away from transition, values should be preserved
        assert!((filtered[0]).abs() < 1e-12);
        assert!((filtered[3]).abs() < 1e-12);
        assert!((filtered[7] - 1.0).abs() < 1e-12);
        assert!((filtered[9] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn medfilt_even_kernel_rejected() {
        assert!(medfilt(&[1.0, 2.0, 3.0], 4).is_err());
    }

    #[test]
    fn wiener_constant_signal_is_stable() {
        let data = vec![2.0; 9];
        let filtered = wiener(&data, 3, None).unwrap();
        for value in filtered.iter().skip(1).take(filtered.len() - 2) {
            assert!(
                (*value - 2.0).abs() < 1e-12,
                "interior constant signal drifted: {value}"
            );
        }
    }

    #[test]
    fn wiener_smooths_impulse_with_explicit_noise() {
        let data = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let filtered = wiener(&data, 3, Some(1.0)).unwrap();
        assert!(
            filtered[2] < 10.0,
            "center should be attenuated: {}",
            filtered[2]
        );
        assert!(filtered[2] > filtered[1], "center should remain dominant");
        assert!(filtered[1] > 0.0, "neighbors should pick up local energy");
    }

    #[test]
    fn wiener_estimated_noise_falls_back_to_local_mean() {
        let data = vec![0.0, 1.0, 0.0];
        let filtered = wiener(&data, 3, Some(10.0)).unwrap();
        assert!(
            (filtered[1] - (1.0 / 3.0)).abs() < 1e-12,
            "expected local mean at center"
        );
    }

    #[test]
    fn wiener_rejects_invalid_window() {
        assert!(wiener(&[1.0, 2.0, 3.0], 0, None).is_err());
        assert!(wiener(&[1.0, 2.0, 3.0], 4, None).is_err());
    }

    // ── get_window tests ───────────────────────────────────────────

    #[test]
    fn get_window_dispatches_hann() {
        let w = get_window_with_fftbins("hann", 10, false).unwrap();
        let expected = hann(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_hamming() {
        let w = get_window_with_fftbins("hamming", 10, false).unwrap();
        let expected = hamming(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_blackman() {
        let w = get_window_with_fftbins("blackman", 10, false).unwrap();
        let expected = blackman(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_blackmanharris() {
        let w = get_window_with_fftbins("blackmanharris", 10, false).unwrap();
        let expected = blackmanharris(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_barthann() {
        let w = get_window_with_fftbins("barthann", 10, false).unwrap();
        let expected = barthann(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_chebwin() {
        let w = get_window_with_fftbins("chebwin,100", 5, false).unwrap();
        let expected = chebwin(5, 100.0);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_general_hamming() {
        let w = get_window_with_fftbins("general_hamming,0.75", 8, false).unwrap();
        let expected = general_hamming(8, 0.75);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_general_cosine() {
        let w = get_window_with_fftbins("general_cosine,0.5,0.3,0.2", 8, false).unwrap();
        let expected = general_cosine(8, &[0.5, 0.3, 0.2], true);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_gaussian() {
        let w = get_window_with_fftbins("gaussian,2.0", 8, false).unwrap();
        let expected = gaussian(8, 2.0, true);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_general_gaussian() {
        let w = get_window_with_fftbins("general_gaussian,1.5,2.0", 8, false).unwrap();
        let expected = general_gaussian(8, 1.5, 2.0, true);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_exponential() {
        let w = get_window_with_fftbins("exponential,2.0", 8, false).unwrap();
        let expected = exponential(8, None, 2.0, true).unwrap();
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_kaiser() {
        let w = get_window_with_fftbins("kaiser,8.6", 10, false).unwrap();
        let expected = kaiser(10, 8.6);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_kaiser_bessel_derived() {
        let w = get_window_with_fftbins("kaiser_bessel_derived,5.0", 8, false).unwrap();
        let alias = get_window_with_fftbins("kbd,5.0", 8, false).unwrap();
        let expected = kaiser_bessel_derived(8, 5.0, true).unwrap();
        assert_eq!(w, expected);
        assert_eq!(alias, expected);
    }

    #[test]
    fn get_window_dispatches_dpss() {
        let symmetric = get_window_with_fftbins("dpss,2.5", 8, false).unwrap();
        let periodic = get_window("dpss,2.5", 8).unwrap();
        let expected_symmetric = dpss(8, 2.5, None, true, None, false)
            .expect("direct symmetric")
            .windows
            .remove(0);
        let expected_periodic = dpss(8, 2.5, None, false, None, false)
            .expect("direct periodic")
            .windows
            .remove(0);
        assert_eq!(symmetric, expected_symmetric);
        assert_eq!(periodic, expected_periodic);
    }

    #[test]
    fn get_window_dispatches_lanczos_and_sinc_alias() {
        let symmetric = get_window_with_fftbins("lanczos", 9, false).unwrap();
        let periodic = get_window("lanczos", 9).unwrap();
        let alias = get_window_with_fftbins("sinc", 9, false).unwrap();
        let expected_symmetric = lanczos(9);
        let expected_periodic = lanczos(10).into_iter().take(9).collect::<Vec<_>>();

        assert_eq!(symmetric, expected_symmetric);
        assert_eq!(periodic, expected_periodic);
        assert_eq!(alias, expected_symmetric);
    }

    #[test]
    fn get_window_dispatches_taylor_defaults_and_parameters() {
        let symmetric_default = get_window_with_fftbins("taylor", 9, false).unwrap();
        let periodic_param = get_window("taylor,6,45,false", 9).unwrap();
        let symmetric_alias = get_window_with_fftbins("taylorwin,5,35,true", 9, false).unwrap();
        let forced_periodic =
            get_window_with_fftbins("taylor_periodic,5,35,true", 9, false).unwrap();

        assert_eq!(symmetric_default, taylor(9, 4, 30.0, true, true));
        assert_eq!(periodic_param, taylor(9, 6, 45.0, false, false));
        assert_eq!(symmetric_alias, taylor(9, 5, 35.0, true, true));
        assert_eq!(forced_periodic, taylor(9, 5, 35.0, true, false));
    }

    #[test]
    fn get_window_rejects_invalid_taylor_parameters() {
        let too_many = get_window("taylor,4,30,true,false", 8).expect_err("too many taylor params");
        assert_eq!(
            too_many.to_string(),
            "invalid argument: taylor expects nbar[,sll[,norm]] parameters: 4,30,true,false"
        );

        let invalid_nbar = get_window("taylor,foo", 8).expect_err("invalid taylor nbar");
        assert_eq!(
            invalid_nbar.to_string(),
            "invalid argument: invalid taylor nbar: foo"
        );

        let invalid_norm = get_window("taylor,4,30,maybe", 8).expect_err("invalid taylor norm");
        assert_eq!(
            invalid_norm.to_string(),
            "invalid argument: invalid taylor norm: maybe"
        );
    }

    #[test]
    fn get_window_rejects_missing_dpss_parameter() {
        let err = get_window("dpss", 8).expect_err("missing dpss parameter");
        assert_eq!(
            err.to_string(),
            "invalid argument: dpss requires a normalized half-bandwidth parameter"
        );
    }

    #[test]
    fn get_window_rejects_non_finite_window_parameters() {
        let kaiser_err = get_window("kaiser,nan", 8).expect_err("NaN kaiser beta");
        assert_eq!(
            kaiser_err.to_string(),
            "invalid argument: invalid kaiser beta: nan"
        );

        let gaussian_err = get_window("gaussian,nan", 8).expect_err("NaN gaussian std");
        assert_eq!(
            gaussian_err.to_string(),
            "invalid argument: invalid gaussian std: nan"
        );

        let exponential_err = get_window("exponential,nan", 8).expect_err("NaN exponential tau");
        assert_eq!(
            exponential_err.to_string(),
            "invalid argument: invalid exponential tau: nan"
        );
    }

    #[test]
    fn get_window_rectangular() {
        let w = get_window("boxcar", 5).unwrap();
        assert_eq!(w, vec![1.0; 5]);
    }

    #[test]
    fn get_window_dispatches_bartlett() {
        let w = get_window_with_fftbins("bartlett", 11, false).unwrap();
        assert_eq!(w.len(), 11);
        assert!((w[5] - 1.0).abs() < 1e-12, "peak at center");
        assert!((w[0]).abs() < 1e-12, "zero at endpoints");
        assert!((w[10]).abs() < 1e-12, "zero at endpoints");
    }

    #[test]
    fn get_window_dispatches_flattop() {
        let w = get_window_with_fftbins("flattop", 11, false).unwrap();
        assert_eq!(w.len(), 11);
        // Flattop peak is ~1.0 but slightly different due to coeffs
        let peak = w.iter().copied().fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        assert!((peak - 1.0).abs() < 0.01);
    }

    #[test]
    fn get_window_dispatches_cosine() {
        let w = get_window_with_fftbins("cosine", 5, false).unwrap();
        let expected = [
            0.3090169943749474,
            0.8090169943749475,
            1.0,
            0.8090169943749475,
            0.3090169943749474,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn get_window_unknown_rejected() {
        assert!(get_window("foobar", 10).is_err());
    }

    #[test]
    fn get_window_defaults_to_periodic_hann() {
        let w = get_window("hann", 8).unwrap();
        let expected = [
            0.0,
            0.1464466094067262,
            0.5,
            0.8535533905932737,
            1.0,
            0.8535533905932737,
            0.5,
            0.1464466094067262,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn get_window_fftbins_false_returns_symmetric_hann() {
        let w = get_window_with_fftbins("hann", 8, false).unwrap();
        let expected = [
            0.0,
            0.1882550990706332,
            0.6112604669781572,
            0.9504844339512095,
            0.9504844339512095,
            0.6112604669781572,
            0.1882550990706332,
            0.0,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn get_window_defaults_to_periodic_bartlett() {
        let w = get_window("bartlett", 5).unwrap();
        let expected = [0.0, 0.4, 0.8, 0.8, 0.4];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn get_window_suffixes_override_fftbins() {
        let periodic = get_window_with_fftbins("hann_periodic", 8, false).unwrap();
        let symmetric = get_window_with_fftbins("hann_symmetric", 8, true).unwrap();

        assert!((periodic[1] - 0.1464466094067262).abs() < 1e-12);
        assert!((periodic[7] - 0.1464466094067262).abs() < 1e-12);
        assert!((symmetric[1] - 0.1882550990706332).abs() < 1e-12);
        assert!(symmetric[7].abs() < 1e-12);
    }

    #[test]
    fn get_window_parameterized_window_respects_fftbins() {
        let periodic = get_window("gaussian,2.0", 8).unwrap();
        let symmetric = get_window_with_fftbins("gaussian,2.0", 8, false).unwrap();

        let expected_periodic = [
            0.1353352832366127,
            0.32465246735834974,
            0.6065306597126334,
            0.8824969025845955,
            1.0,
            0.8824969025845955,
            0.6065306597126334,
            0.32465246735834974,
        ];
        let expected_symmetric = [
            0.2162651668298873,
            0.45783336177161427,
            0.7548396019890073,
            0.9692332344763441,
            0.9692332344763441,
            0.7548396019890073,
            0.45783336177161427,
            0.2162651668298873,
        ];

        for (actual, want) in periodic.iter().zip(expected_periodic.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
        for (actual, want) in symmetric.iter().zip(expected_symmetric.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    // ── STFT / ISTFT tests ─────────────────────────────────────────

    #[test]
    fn stft_istft_roundtrip() {
        // Generate a simple sinusoidal signal, STFT then ISTFT, compare.
        let fs = 100.0;
        let n = 400;
        let freq = 10.0;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
            .collect();

        let nperseg = 64;
        let noverlap = 48;
        let stft_res = stft(&x, fs, None, Some(nperseg), Some(noverlap)).unwrap();
        let reconstructed = istft(&stft_res, nperseg, Some(noverlap)).unwrap();

        // Compare in the well-reconstructed interior (avoiding edge effects).
        let margin = nperseg;
        let end = reconstructed.len().min(n) - margin;
        for i in margin..end {
            assert!(
                (reconstructed[i] - x[i]).abs() < 0.15,
                "STFT-ISTFT mismatch at {i}: got {}, expected {}",
                reconstructed[i],
                x[i]
            );
        }
    }

    #[test]
    fn stft_frequency_bins() {
        let fs = 1000.0;
        let x = vec![0.0; 256];
        let res = stft(&x, fs, None, Some(256), Some(128)).unwrap();
        assert_eq!(res.frequencies.len(), 129); // 256/2 + 1
        assert!((res.frequencies[0]).abs() < 1e-12);
        // Last frequency should be Nyquist
        assert!((res.frequencies[128] - 500.0).abs() < 1.0);
    }

    // ── Spectrogram tests ──────────────────────────────────────────

    #[test]
    fn spectrogram_chirp_energy_tracks_frequency() {
        // Generate a chirp from 5 Hz to 45 Hz over 1 second
        let fs = 200.0;
        let n = 200;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let sig = chirp(&t, 5.0, 1.0, 45.0, ChirpMethod::Linear).unwrap();

        let nperseg = 64;
        let noverlap = 56;
        let res = spectrogram(&sig, fs, None, Some(nperseg), Some(noverlap)).unwrap();

        // Should have multiple time segments and frequency bins
        assert!(!res.sxx.is_empty(), "spectrogram produced no segments");
        assert!(
            res.frequencies.len() > 1,
            "spectrogram produced no frequencies"
        );
        assert!(res.times.len() > 1, "spectrogram produced no time bins");
    }

    #[test]
    fn spectrogram_dimensions() {
        let fs = 100.0;
        let x: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
        let res = spectrogram(&x, fs, None, Some(64), Some(32)).unwrap();

        // Each time segment should have the same number of frequency bins
        let n_freqs = res.frequencies.len();
        for (t, seg) in res.sxx.iter().enumerate() {
            assert_eq!(
                seg.len(),
                n_freqs,
                "segment {t} has {} freq bins, expected {n_freqs}",
                seg.len()
            );
        }
        assert_eq!(res.times.len(), res.sxx.len());
    }

    // ── CSD tests ──────────────────────────────────────────────────

    #[test]
    fn csd_auto_spectrum_matches_welch() {
        // csd(x, x) should match welch(x) in magnitude
        let fs = 100.0;
        let n = 512;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin())
            .collect();

        let nperseg = 128;
        let noverlap = 64;
        let welch_res = welch(&x, fs, None, Some(nperseg), Some(noverlap)).unwrap();
        let csd_res = csd(&x, &x, fs, None, Some(nperseg), Some(noverlap)).unwrap();

        // Auto-CSD should be real and match welch PSD
        assert_eq!(welch_res.psd.len(), csd_res.csd.len());
        for (k, (&welch_val, &(csd_re, csd_im))) in
            welch_res.psd.iter().zip(csd_res.csd.iter()).enumerate()
        {
            // Imaginary part of auto-spectrum should be ~0
            assert!(
                csd_im.abs() < welch_val.abs() * 0.01 + 1e-20,
                "auto-CSD imaginary at bin {k}: {csd_im}"
            );
            // Real part should match welch PSD
            if welch_val.abs() > 1e-15 {
                let ratio = csd_re / welch_val;
                assert!(
                    (ratio - 1.0).abs() < 0.1,
                    "auto-CSD/welch mismatch at bin {k}: csd={csd_re}, welch={welch_val}"
                );
            }
        }
    }

    #[test]
    fn csd_scaling_matches_scipy_density_and_spectrum() {
        let fs = 16.0;
        let n = 32;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let x: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * 2.0 * ti).sin())
            .collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * 2.0 * ti).cos())
            .collect();
        let expected_frequencies = [0.0, 2.0, 4.0, 6.0, 8.0];
        let expected_density = [
            (-3.962744417486146e-17, 0.0),
            (1.3346567516981065e-17, 0.16666666666666666),
            (-7.103421885274257e-19, 0.04166666666666669),
            (-2.5133161553135813e-33, 2.363886302605754e-32),
            (-2.1146214215064937e-33, 0.0),
        ];
        let expected_spectrum = [
            (-1.127784089578213e-16, 0.0),
            (2.9738116731031224e-17, 0.5),
            (-9.686270956913913e-18, 0.12500000000000003),
            (-2.9579909131093405e-34, 6.980810313780984e-32),
            (-9.199850601359331e-33, 0.0),
        ];

        let density =
            csd_with_scaling(&x, &y, fs, None, Some(8), Some(4), SpectralScaling::Density)
                .expect("density");
        let spectrum = csd_with_scaling(
            &x,
            &y,
            fs,
            None,
            Some(8),
            Some(4),
            SpectralScaling::Spectrum,
        )
        .expect("spectrum");

        for (index, ((actual, expected), spectrum_frequency)) in density
            .frequencies
            .iter()
            .zip(expected_frequencies.iter())
            .zip(spectrum.frequencies.iter())
            .enumerate()
        {
            assert_close(
                *actual,
                *expected,
                1e-12,
                &format!("frequency[{index}] should match SciPy"),
            );
            assert_close(
                *actual,
                *spectrum_frequency,
                1e-12,
                &format!("frequency[{index}] should match across scaling modes"),
            );
        }

        for (index, (((density_re, density_im), expected_density), expected_spectrum)) in density
            .csd
            .iter()
            .zip(expected_density.iter())
            .zip(expected_spectrum.iter())
            .enumerate()
        {
            let (expected_density_re, expected_density_im) = *expected_density;
            let (expected_spectrum_re, expected_spectrum_im) = *expected_spectrum;
            let (spectrum_re, spectrum_im) = spectrum.csd[index];

            assert_close(
                *density_re,
                expected_density_re,
                1e-12,
                &format!("density.re[{index}] should match SciPy"),
            );
            assert_close(
                *density_im,
                expected_density_im,
                1e-12,
                &format!("density.im[{index}] should match SciPy"),
            );
            assert_close(
                spectrum_re,
                expected_spectrum_re,
                1e-12,
                &format!("spectrum.re[{index}] should match SciPy"),
            );
            assert_close(
                spectrum_im,
                expected_spectrum_im,
                1e-12,
                &format!("spectrum.im[{index}] should match SciPy"),
            );
        }
    }

    // ── Coherence tests ────────────────────────────────────────────

    #[test]
    fn coherence_identical_signals() {
        // Coherence of a signal with itself should be ~1 everywhere
        let fs = 100.0;
        let n = 512;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin())
            .collect();

        let res = coherence(&x, &x, fs, None, Some(128), Some(64)).unwrap();
        for (k, &c) in res.coherence.iter().enumerate() {
            assert!(
                c >= 0.99 || res.frequencies[k].abs() < 1e-10,
                "coherence of identical signals at bin {k} (f={}): {c}",
                res.frequencies[k]
            );
        }
    }

    #[test]
    fn coherence_uncorrelated_signals() {
        // Coherence of uncorrelated signals should be low
        // Use two different frequency sinusoids as "uncorrelated" approximation
        let fs = 1000.0;
        let n = 4096;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 50.0 * i as f64 / fs).sin())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / fs).sin())
            .collect();

        let res = coherence(&x, &y, fs, None, Some(256), Some(128)).unwrap();
        // Average coherence should be much less than 1
        let avg_coh: f64 = res.coherence.iter().sum::<f64>() / res.coherence.len() as f64;
        assert!(
            avg_coh < 0.5,
            "average coherence of uncorrelated signals: {avg_coh}"
        );
    }

    #[test]
    fn coherence_range_zero_to_one() {
        let fs = 100.0;
        let n = 512;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();

        let res = coherence(&x, &y, fs, None, Some(128), Some(64)).unwrap();
        for (k, &c) in res.coherence.iter().enumerate() {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&c),
                "coherence at bin {k} out of range: {c}"
            );
        }
    }

    // ── Resample tests ─────────────────────────────────────────────

    #[test]
    fn resample_upsample_preserves_frequency() {
        // A sine wave at 10 Hz sampled at 100 Hz, upsampled to 200 Hz
        let fs = 100.0;
        let n = 100;
        let freq = 10.0;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
            .collect();

        let result = resample(&x, 200).unwrap();
        assert_eq!(result.len(), 200);

        // Count zero crossings — should approximately double
        let x_crossings: usize = x
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        let r_crossings: usize = result
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        // Upsampled signal should have roughly double the zero crossings
        assert!(
            r_crossings >= x_crossings,
            "upsampled crossings {} < original {}",
            r_crossings,
            x_crossings
        );
    }

    #[test]
    fn resample_downsample() {
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin()).collect();
        let result = resample(&x, 100).unwrap();
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn resample_same_length() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = resample(&x, 4).unwrap();
        assert_eq!(result.len(), 4);
        for (i, (&a, &b)) in x.iter().zip(result.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "same-length resample mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn resample_upsample_even_nyquist_matches_scipy() {
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let result = resample(&x, 8).unwrap();
        let expected = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];

        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "upsampled Nyquist mismatch at {i}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn resample_downsample_even_nyquist_matches_scipy() {
        let x = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let result = resample(&x, 6).unwrap();
        let expected = [
            0.0,
            0.8660254037844386,
            -0.8660254037844386,
            0.0,
            0.8660254037844386,
            -0.8660254037844386,
        ];

        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "downsampled Nyquist mismatch at {i}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn resample_poly_rational_rate() {
        // Upsample by 3/2: 100 samples → ~150 samples
        let n = 100;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = resample_poly(&x, 3, 2).unwrap();
        // Output length should be approximately n * up / down
        let expected_len = n * 3 / 2;
        assert!(
            (result.len() as i64 - expected_len as i64).unsigned_abs() <= n / 5,
            "resample_poly length: got {}, expected ~{expected_len}",
            result.len()
        );
    }

    #[test]
    fn resample_poly_identity() {
        // up=1, down=1 should return the same signal
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = resample_poly(&x, 1, 1).unwrap();
        assert_eq!(result.len(), x.len());
        for (i, (&a, &b)) in x.iter().zip(result.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "identity resample_poly mismatch at {i}"
            );
        }
    }

    #[test]
    fn resample_poly_padtype_line_matches_scipy_reference() {
        let x = vec![1.0, 2.0, 4.0, 8.0];
        let result = resample_poly_with_padtype(&x, 3, 2, Some("line"), None).unwrap();
        let expected = [
            1.000606173553777,
            1.780628511492595,
            2.331390913393649,
            4.002424694215109,
            6.739706761743146,
            8.971945980209963,
        ];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-8,
                "line padtype mismatch at {i}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn resample_poly_padtype_mean_matches_scipy_reference() {
        let x = vec![1.0, 2.0, 4.0, 8.0];
        let result = resample_poly_with_padtype(&x, 3, 2, Some("mean"), None).unwrap();
        let expected = [
            0.998333022727113,
            1.573448321613185,
            2.293752166053205,
            4.000151543388444,
            7.390581540874371,
            7.201952100936264,
        ];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-8,
                "mean padtype mismatch at {i}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn resample_poly_padtype_constant_cval_matches_scipy_reference() {
        let x = vec![1.0, 2.0, 4.0, 8.0];
        let result = resample_poly_with_padtype(&x, 3, 2, Some("constant"), Some(5.0)).unwrap();
        let expected = [
            1.000606173553777,
            1.39589024581026,
            2.44120445519666,
            4.002424694215109,
            7.179517790579072,
            7.616661693078465,
        ];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-8,
                "constant cval mismatch at {i}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn resample_poly_padtype_rejects_nonconstant_cval() {
        let x = vec![1.0, 2.0, 4.0, 8.0];
        let err =
            resample_poly_with_padtype(&x, 3, 2, Some("mean"), Some(5.0)).expect_err("invalid");
        assert_eq!(
            err,
            SignalError::InvalidArgument("cval has no effect when padtype is mean".to_string())
        );
    }

    // ── Decimate tests ─────────────────────────────────────────────

    #[test]
    fn decimate_basic() {
        // Decimate by 4: 400 samples → 100 samples
        let n = 400;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let result = decimate(&x, 4).unwrap();
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn decimate_prevents_aliasing() {
        // Signal with low-freq (5 Hz) and high-freq (45 Hz) at fs=100
        // After decimate by 4 (new fs=25), 45 Hz should be removed
        let fs = 100.0;
        let n = 400;
        let two_pi = 2.0 * std::f64::consts::PI;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (two_pi * 5.0 * t).sin() + (two_pi * 45.0 * t).sin()
            })
            .collect();

        let result = decimate(&x, 4).unwrap();
        // The decimated signal should have the low-freq component preserved
        // but the high-freq component should be attenuated by the anti-alias filter
        let energy: f64 = result.iter().map(|v| v * v).sum::<f64>() / result.len() as f64;
        // If alias wasn't prevented, energy would be higher
        assert!(
            energy < 1.0,
            "decimated signal energy too high (aliasing?): {energy}"
        );
    }

    #[test]
    fn decimate_rejects_q_less_than_2() {
        assert!(decimate(&[1.0, 2.0], 1).is_err());
    }

    // ── Cross-correlation tests ──────────────────────────────────────

    #[test]
    fn correlate_autocorrelation_peak_at_zero_lag() {
        // Auto-correlation peak should be at center (zero lag)
        let x = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let result = correlate(&x, &x, ConvolveMode::Full).expect("correlate");
        let mid = result.len() / 2;
        for (i, &val) in result.iter().enumerate() {
            assert!(
                val <= result[mid] + 1e-10,
                "peak should be at center lag: r[{i}]={val} > r[{mid}]={}",
                result[mid]
            );
        }
    }

    #[test]
    fn correlate_shifted_signal() {
        // Cross-correlate with a shifted copy — peak indicates shift
        let a = vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0];
        let v = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0];
        let result = correlate(&a, &v, ConvolveMode::Full).expect("correlate");
        // Peak should indicate the shift
        let peak_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        // v is shifted right by 1 relative to a, so peak should be at center-1
        let center = result.len() / 2;
        assert!(
            (peak_idx as i64 - center as i64).abs() <= 2,
            "peak at {peak_idx}, center at {center}"
        );
    }

    #[test]
    fn correlate_same_mode() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = vec![1.0, 0.0, -1.0];
        let result = correlate(&a, &v, ConvolveMode::Same).expect("correlate same");
        assert_eq!(
            result.len(),
            a.len(),
            "same mode output length = input length"
        );
    }

    #[test]
    fn correlate_valid_mode() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = vec![1.0, 0.0, -1.0];
        let result = correlate(&a, &v, ConvolveMode::Valid).expect("correlate valid");
        assert_eq!(result.len(), 3, "valid mode output length");
    }

    #[test]
    fn correlate_symmetry() {
        // correlate(a, a) should be symmetric
        let a = vec![1.0, 3.0, 2.0, 5.0];
        let result = correlate(&a, &a, ConvolveMode::Full).expect("correlate");
        let n = result.len();
        for i in 0..n / 2 {
            assert!(
                (result[i] - result[n - 1 - i]).abs() < 1e-10,
                "autocorrelation should be symmetric: r[{i}]={} vs r[{}]={}",
                result[i],
                n - 1 - i,
                result[n - 1 - i]
            );
        }
    }

    #[test]
    fn correlate2d_identity_kernel() {
        // Correlation with delta function should return the input
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let v = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // center = 1
        let result = correlate2d(&a, (3, 3), &v, (3, 3), ConvolveMode::Same).expect("correlate2d");
        assert_eq!(result.len(), 9);
        // With a centered delta, same-mode correlation should approximate input
        assert!(
            (result[4] - a[4]).abs() < 1e-10,
            "center should match: {} vs {}",
            result[4],
            a[4]
        );
    }

    #[test]
    fn correlate2d_same_even_kernel_matches_scipy_centering() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let v = vec![1.0, 1.0, 1.0, 1.0];
        let result = correlate2d(&a, (3, 3), &v, (2, 2), ConvolveMode::Same).expect("correlate2d");
        assert_eq!(
            result,
            vec![12.0, 16.0, 9.0, 24.0, 28.0, 15.0, 15.0, 17.0, 9.0]
        );
    }

    // ── Hilbert transform tests ──────────────────────────────────────

    #[test]
    fn hilbert_sinusoid_envelope() {
        // Pure sinusoid should have ~constant envelope of 1.0
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / n as f64).sin())
            .collect();
        let envelope = hilbert_envelope(&x).expect("hilbert envelope");
        assert_eq!(envelope.len(), n);
        // Exclude edges (transient effects) and check middle
        for (i, &env_val) in envelope.iter().enumerate().take(n - 20).skip(20) {
            assert!(
                (env_val - 1.0).abs() < 0.1,
                "envelope[{i}] = {}, expected ~1.0",
                env_val
            );
        }
    }

    #[test]
    fn hilbert_real_part_preserved() {
        // Real part of analytic signal should equal original signal
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let analytic = hilbert(&x).expect("hilbert");
        for (i, (&xi, &(re, _))) in x.iter().zip(analytic.iter()).enumerate() {
            assert!(
                (re - xi).abs() < 1e-10,
                "real part[{i}] = {re}, expected {xi}"
            );
        }
    }

    #[test]
    fn hilbert_empty_input_rejected() {
        let err = hilbert(&[]).expect_err("empty");
        assert!(err.is_argument_error());
    }

    #[test]
    fn hilbert_dc_signal() {
        // DC signal: Hilbert transform of constant is zero
        let x = vec![5.0; 64];
        let analytic = hilbert(&x).expect("hilbert dc");
        // Imaginary parts should be ~0 for constant input
        for (i, &(_, im)) in analytic.iter().enumerate() {
            assert!(im.abs() < 1e-10, "DC imaginary[{i}] = {im}, expected ~0");
        }
    }

    // ── Notch/peak filter tests ──────────────────────────────────────

    #[test]
    fn iirnotch_rejects_target_frequency() {
        let coeffs = iirnotch(0.3, 30.0).expect("iirnotch");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        // Find response at notch frequency (0.3π)
        let notch_idx = nearest_freq_index(&result.w, 0.3 * std::f64::consts::PI);
        // Response at notch should be near zero
        assert!(
            result.h_mag[notch_idx] < 0.1,
            "notch should attenuate target freq: mag = {}",
            result.h_mag[notch_idx]
        );
        // Response away from notch should be near 1
        let pass_idx = nearest_freq_index(&result.w, 0.1 * std::f64::consts::PI);
        assert!(
            result.h_mag[pass_idx] > 0.9,
            "passband should be near unity: mag = {}",
            result.h_mag[pass_idx]
        );
    }

    #[test]
    fn iirpeak_boosts_target_frequency() {
        let coeffs = iirpeak(0.3, 30.0).expect("iirpeak");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let peak_idx = nearest_freq_index(&result.w, 0.3 * std::f64::consts::PI);
        let pass_idx = nearest_freq_index(&result.w, 0.1 * std::f64::consts::PI);
        assert!(
            result.h_mag[peak_idx] > result.h_mag[pass_idx] * 2.0,
            "peak should boost target: peak={}, pass={}",
            result.h_mag[peak_idx],
            result.h_mag[pass_idx]
        );
    }

    #[test]
    fn iirnotch_invalid_params() {
        assert!(iirnotch(0.0, 10.0).is_err());
        assert!(iirnotch(1.0, 10.0).is_err());
        assert!(iirnotch(0.5, -1.0).is_err());
    }

    #[test]
    fn iirpeak_invalid_params() {
        assert!(iirpeak(0.0, 10.0).is_err());
        assert!(iirpeak(0.5, 0.0).is_err());
    }

    // ── Wavelet tests ────────────────────────────────────────────────

    #[test]
    fn ricker_symmetric() {
        let w = ricker(101, 5.0);
        assert_eq!(w.len(), 101);
        // Symmetric around center
        for i in 0..50 {
            assert!(
                (w[i] - w[100 - i]).abs() < 1e-12,
                "ricker not symmetric at {i}"
            );
        }
    }

    #[test]
    fn ricker_peak_at_center() {
        let w = ricker(101, 5.0);
        let center = 50;
        // Peak should be at center
        for (i, &val) in w.iter().enumerate() {
            assert!(
                val <= w[center] + 1e-12,
                "ricker: w[{i}]={val} > w[center]={}",
                w[center]
            );
        }
    }

    #[test]
    fn ricker_zero_integral() {
        // Ricker wavelet has zero mean (wavelet admissibility)
        let w = ricker(1001, 10.0);
        let integral: f64 = w.iter().sum();
        assert!(
            integral.abs() < 0.1,
            "ricker integral should be ~0: {integral}"
        );
    }

    #[test]
    fn morlet_complex_oscillatory() {
        let w = morlet(200, 5.5, 10.0, false);
        assert_eq!(w.len(), 200);
        // With wider s, the Gaussian envelope is broader and oscillations are visible
        let max_im = w
            .iter()
            .map(|&(_, im)| im.abs())
            .fold(0.0_f64, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        assert!(
            max_im > 0.01,
            "morlet should have nonzero imaginary part, max_im = {max_im}"
        );
    }

    #[test]
    fn cwt_detects_frequency() {
        // Create signal with known frequency component
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / n as f64).sin())
            .collect();
        let widths: Vec<f64> = (1..=20).map(|w| w as f64).collect();
        let result = cwt(&x, ricker, &widths).expect("cwt");
        assert_eq!(result.len(), widths.len());
        assert_eq!(result[0].len(), n);
    }

    #[test]
    fn cwt_empty_data_rejected() {
        let err = cwt(&[], ricker, &[1.0]).expect_err("empty");
        assert!(err.is_argument_error());
    }

    #[test]
    fn tukey_window_endpoints() {
        let w = tukey_window(100, 0.5);
        assert_eq!(w.len(), 100);
        // Endpoints should be near zero
        assert!(w[0] < 0.01, "tukey start should be ~0: {}", w[0]);
        // Center should be 1.0
        assert!(
            (w[50] - 1.0).abs() < 0.01,
            "tukey center should be ~1: {}",
            w[50]
        );
    }

    #[test]
    fn nuttall_window_symmetric() {
        let w = nuttall_window(64);
        assert_eq!(w.len(), 64);
        for i in 0..32 {
            assert!(
                (w[i] - w[63 - i]).abs() < 1e-12,
                "nuttall not symmetric at {i}"
            );
        }
    }

    #[test]
    fn bohman_window_endpoints_zero() {
        let w = bohman_window(64);
        assert!(w[0].abs() < 1e-12, "bohman start should be 0");
        assert!(w[63].abs() < 1e-12, "bohman end should be 0");
    }

    #[test]
    fn cosine_window_matches_scipy_reference() {
        let w = cosine(5);
        let expected = [
            0.3090169943749474,
            0.8090169943749475,
            1.0,
            0.8090169943749475,
            0.3090169943749474,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn upfirdn_fir_only_matches_scipy_example() {
        let y = upfirdn(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0], 1, 1).expect("upfirdn");
        assert_eq!(y, vec![1.0, 2.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn upfirdn_upsampling_matches_scipy_example() {
        let y = upfirdn(&[1.0], &[1.0, 2.0, 3.0], 3, 1).expect("upfirdn");
        assert_eq!(y, vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn upfirdn_interpolate_and_decimate_matches_scipy_example() {
        let y = upfirdn(&[0.5, 1.0, 0.5], &[0.0, 1.0, 2.0, 3.0, 4.0], 2, 3).expect("upfirdn");
        let expected = [0.0, 1.0, 2.5, 4.0];
        for (actual, want) in y.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn upfirdn_empty_input_returns_empty() {
        let y = upfirdn(&[1.0, 2.0], &[], 2, 1).expect("empty input");
        assert!(y.is_empty());
    }

    #[test]
    fn upfirdn_rejects_invalid_arguments() {
        assert!(upfirdn(&[], &[1.0, 2.0], 1, 1).is_err());
        assert!(upfirdn(&[1.0], &[1.0, 2.0], 0, 1).is_err());
        assert!(upfirdn(&[1.0], &[1.0, 2.0], 1, 0).is_err());
    }

    // ── Max-length sequence + matched filter tests ───────────────────

    #[test]
    fn max_len_seq_length() {
        let seq = max_len_seq(5).expect("mls");
        assert_eq!(seq.len(), 31);
        assert!(seq.iter().all(|&v| v == 1.0 || v == -1.0));
    }

    #[test]
    fn max_len_seq_balanced() {
        let seq = max_len_seq(6).expect("mls");
        let ones = seq.iter().filter(|&&v| v == 1.0).count();
        let neg_ones = seq.iter().filter(|&&v| v == -1.0).count();
        assert_eq!(ones + neg_ones, 63);
        assert_eq!((ones as i64 - neg_ones as i64).unsigned_abs(), 1);
    }

    #[test]
    fn max_len_seq_invalid_nbits() {
        assert!(max_len_seq(0).is_err());
        assert!(max_len_seq(1).is_err());
    }

    #[test]
    fn matched_filter_peak_at_template_location() {
        let template = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let mut signal = vec![0.0; 20];
        for (i, &t) in template.iter().enumerate() {
            signal[8 + i] = t;
        }
        let result = matched_filter(&template, &signal).expect("matched_filter");
        let peak_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert!(
            (peak_idx as i64 - 12).abs() <= 1,
            "peak at {peak_idx}, expected ~12"
        );
    }

    #[test]
    fn minimum_phase_matches_scipy_reference() {
        let h = minimum_phase(&[1.0, 2.0, 3.0, 2.0, 1.0]).expect("minimum_phase");
        let expected = [1.00107477, 0.99999638, 0.99892826];
        for (actual, want) in h.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-6, "{actual} vs {want}");
        }
    }

    #[test]
    fn minimum_phase_rejects_short_filters() {
        assert!(minimum_phase(&[]).is_err());
        assert!(minimum_phase(&[1.0, 2.0]).is_err());
    }

    // ── Lomb-Scargle tests ───────────────────────────────────────────

    #[test]
    fn lombscargle_detects_known_frequency() {
        // Signal with known frequency at 5 rad/s
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|&t| (5.0 * t).sin()).collect();
        let freqs: Vec<f64> = (1..20).map(|f| f as f64).collect();
        let power = lombscargle(&x, &y, &freqs, false).expect("lombscargle");
        // Peak should be near freq=5
        let peak_idx = power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        assert!(
            (freqs[peak_idx] - 5.0).abs() <= 1.0,
            "peak at freq={}, expected ~5",
            freqs[peak_idx]
        );
    }

    #[test]
    fn lombscargle_rejects_mismatched() {
        let err = lombscargle(&[1.0, 2.0], &[1.0], &[1.0], false).expect_err("mismatch");
        assert!(err.is_argument_error());
    }

    #[test]
    fn lombscargle_normalize_matches_scipy_reference() {
        let tol = 5e-9;
        let x = [0.0, 0.5, 1.1, 1.7, 2.4];
        let y = [1.0, -0.5, 0.75, 0.25, -1.25];
        let freqs = [0.5, 1.0, 1.5, 2.0];
        let power = lombscargle(&x, &y, &freqs, false).expect("lombscargle power");
        let normalized = lombscargle(&x, &y, &freqs, true).expect("lombscargle normalized");
        let expected_power = [
            0.707_498_041_923_665,
            0.651_738_901_352_701,
            0.587_708_048_398_399,
            0.390_030_087_171_934,
        ];
        let expected_normalized = [
            0.411_635_223_256_532,
            0.379_193_542_532_48,
            0.341_939_227_868_014,
            0.226_926_597_192_203,
        ];

        for (actual, want) in power.iter().zip(expected_power.iter()) {
            assert!((actual - want).abs() < tol, "{actual} vs {want}");
        }

        for (actual, want) in normalized.iter().zip(expected_normalized.iter()) {
            assert!((actual - want).abs() < tol, "{actual} vs {want}");
        }
    }

    #[test]
    fn lombscargle_zero_frequency_matches_scipy_reference() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 1.0, 0.0];
        let freqs = [0.0, 1.0];
        let power = lombscargle(&x, &y, &freqs, false).expect("lombscargle power");
        let normalized = lombscargle(&x, &y, &freqs, true).expect("lombscargle normalized");

        assert!((power[0] - 2.0).abs() < 1e-12);
        assert!((normalized[0] - (2.0 / 3.0)).abs() < 1e-12);
        assert!(power[1].is_finite());
        assert!(normalized[1].is_finite());
    }

    #[test]
    fn lombscargle_non_finite_frequency_returns_nan() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 1.0, 0.0];
        let freqs = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let power = lombscargle(&x, &y, &freqs, false).expect("lombscargle power");
        assert!(power.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn lombscargle_normalize_returns_nan_for_zero_energy_signal() {
        let x = [0.1, 0.4, 1.1, 2.0];
        let y = [0.0, 0.0, 0.0, 0.0];
        let freqs = [1.0, 2.0];
        let normalized = lombscargle(&x, &y, &freqs, true).expect("lombscargle normalized");
        assert!(normalized.iter().all(|value| value.is_nan()));
    }

    // ── gausspulse tests ─────────────────────────────────────────────

    #[test]
    fn gausspulse_peak_at_zero() {
        let t: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.001).collect();
        let y = gausspulse(&t, 1000.0, 0.5);
        // Peak should be at t=0 (index 50)
        assert!(
            y[50] > y[0].abs() && y[50] > y[100].abs(),
            "gausspulse should peak at t=0"
        );
    }

    #[test]
    fn gausspulse_zero_center_frequency_matches_scipy_constant_pulse() {
        let t = [-1.0, 0.0, 1.0];
        let y = gausspulse(&t, 0.0, 0.5);
        assert_eq!(y, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn gausspulse_envelope_decay() {
        // Envelope should decay away from center
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.001).collect();
        let y = gausspulse(&t, 100.0, 0.5);
        let max_abs = y.iter().map(|v| v.abs()).fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        assert!(y[0].abs() >= max_abs * 0.99, "peak should be at t=0");
    }

    // ── sweep_poly tests ─────────────────────────────────────────────

    #[test]
    fn sweep_poly_constant_freq() {
        // Constant frequency f(t) = 10 → cos(2π·10·t)
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let y = sweep_poly(&t, &[10.0]); // poly = [10] → f(t) = 10
        assert_eq!(y.len(), 100);
        // At t=0, should be cos(0) = 1
        assert!((y[0] - 1.0).abs() < 1e-10, "sweep_poly(0) = {}", y[0]);
    }

    #[test]
    fn sweep_poly_linear_chirp() {
        // Linear chirp: f(t) = 10 + 100*t → poly = [100, 10]
        let t: Vec<f64> = (0..200).map(|i| i as f64 * 0.001).collect();
        let y = sweep_poly(&t, &[100.0, 10.0]);
        assert_eq!(y.len(), 200);
        // Should oscillate
        let crossings = y
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        assert!(
            crossings > 2,
            "linear chirp should oscillate: {crossings} crossings"
        );
    }

    #[test]
    fn czt_matches_dft_for_unit_circle() {
        // With default parameters, CZT should match the DFT
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let n = x.len();
        let result = czt(&x, n, None, None).unwrap();

        // Compare with direct DFT
        let two_pi = 2.0 * std::f64::consts::PI;
        for (k, item) in result.iter().enumerate().take(n) {
            let mut re = 0.0;
            let mut im = 0.0;
            for (j, &xj) in x.iter().enumerate() {
                let angle = -two_pi * (k as f64) * (j as f64) / (n as f64);
                re += xj * angle.cos();
                im += xj * angle.sin();
            }
            assert!(
                (item.0 - re).abs() < 1e-10,
                "CZT real[{k}] = {}, expected {re}",
                item.0
            );
            assert!(
                (item.1 - im).abs() < 1e-10,
                "CZT imag[{k}] = {}, expected {im}",
                item.1
            );
        }
    }

    #[test]
    fn czt_single_tone_detection() {
        // A pure sinusoid should produce a peak at its frequency
        let n = 64;
        let freq = 5.0; // bin 5
        let two_pi = 2.0 * std::f64::consts::PI;
        let x: Vec<f64> = (0..n)
            .map(|i| (two_pi * freq * i as f64 / n as f64).cos())
            .collect();

        let result = czt(&x, n, None, None).unwrap();

        // Find magnitude peak
        let mags: Vec<f64> = result
            .iter()
            .map(|&(r, i)| (r * r + i * i).sqrt())
            .collect();
        // Cosine has energy at both bin f and bin N-f (mirror)
        let peak_bin = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            peak_bin == freq as usize || peak_bin == n - freq as usize,
            "peak should be at bin {freq} or {}, got {peak_bin}",
            n - freq as usize
        );
        // Check that both bins have significant energy
        assert!(
            mags[freq as usize] > mags[0] * 10.0,
            "bin {freq} should have energy"
        );
    }

    #[test]
    fn zoom_fft_resolves_narrow_band() {
        // Two close frequencies that zoom FFT should resolve
        let n = 128;
        let two_pi = 2.0 * std::f64::consts::PI;
        let f1 = 10.0 / n as f64; // normalized freq
        let f2 = 12.0 / n as f64;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                (two_pi * f1 * n as f64 * i as f64 / n as f64).cos()
                    + (two_pi * f2 * n as f64 * i as f64 / n as f64).cos()
            })
            .collect();

        // Zoom into the frequency range containing both tones
        let m = 64;
        let result = zoom_fft(&x, (8.0 / n as f64, 14.0 / n as f64), m).unwrap();

        // Should have two peaks in the zoomed spectrum
        let mags: Vec<f64> = result
            .iter()
            .map(|&(r, i)| (r * r + i * i).sqrt())
            .collect();
        let max_mag = mags.iter().cloned().fold(0.0f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        let peaks: Vec<usize> = mags
            .iter()
            .enumerate()
            .filter(|&(_, m)| *m > max_mag * 0.5)
            .map(|(i, _)| i)
            .collect();
        assert!(
            peaks.len() >= 2,
            "zoom FFT should resolve 2 peaks, got {} peaks",
            peaks.len()
        );
    }

    #[test]
    fn czt_empty_output() {
        let x = vec![1.0, 2.0];
        let result = czt(&x, 0, None, None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn czt_empty_input_errors() {
        let result = czt(&[], 4, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn remez_lowpass_filter() {
        // Design a 31-tap lowpass filter with cutoff at 0.2
        let h = remez(31, &[0.0, 0.15, 0.25, 0.5], &[1.0, 0.0], None).unwrap();
        assert_eq!(h.len(), 31);

        // Filter should be symmetric (linear phase)
        for i in 0..15 {
            assert!(
                (h[i] - h[30 - i]).abs() < 1e-10,
                "h[{i}] = {} != h[{}] = {}",
                h[i],
                30 - i,
                h[30 - i]
            );
        }

        // Sum of coefficients should approximate passband gain (1.0)
        let sum: f64 = h.iter().sum();
        assert!(sum > 0.5 && sum < 1.5, "filter sum = {sum}, expected ~1.0");
    }

    // ── LTI/DLTI system tests ───────────────────────────────────────

    #[test]
    fn lti_new_valid() {
        let sys = Lti::new(vec![1.0], vec![1.0, 2.0]).expect("valid system");
        assert_eq!(sys.num, vec![1.0]);
        assert_eq!(sys.den, vec![1.0, 2.0]);
    }

    #[test]
    fn lti_new_empty_num_rejected() {
        let err = Lti::new(vec![], vec![1.0]).expect_err("empty num");
        assert!(err.is_argument_error());
    }

    #[test]
    fn lti_new_empty_den_rejected() {
        let err = Lti::new(vec![1.0], vec![]).expect_err("empty den");
        assert!(err.is_argument_error());
    }

    #[test]
    fn lti_new_zero_den_rejected() {
        let err = Lti::new(vec![1.0], vec![0.0, 0.0]).expect_err("zero den");
        assert!(err.is_argument_error());
    }

    #[test]
    fn lti_from_zpk_no_roots() {
        // H(s) = 2 / 1 = 2
        let sys = Lti::from_zpk(&[], &[], 2.0).expect("valid");
        assert_eq!(sys.num, vec![2.0]);
        assert_eq!(sys.den, vec![1.0]);
    }

    #[test]
    fn lti_from_zpk_simple_pole() {
        // H(s) = 1 / (s + 1)
        let sys = Lti::from_zpk(&[], &[-1.0], 1.0).expect("valid");
        assert_eq!(sys.num, vec![1.0]);
        assert!((sys.den[0] - 1.0).abs() < 1e-10);
        assert!((sys.den[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn lti_freqresp_dc_gain() {
        // H(s) = 2 / (s + 1), DC gain = H(0) = 2
        let sys = Lti::new(vec![2.0], vec![1.0, 1.0]).expect("valid");
        let (mag, _phase) = sys.freqresp(&[0.0]);
        assert!((mag[0] - 2.0).abs() < 1e-10, "DC gain = {}", mag[0]);
    }

    #[test]
    fn lti_step_response_converges() {
        // H(s) = 1 / (s + 1), step response converges to 1
        let sys = Lti::new(vec![1.0], vec![1.0, 1.0]).expect("valid");
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y = sys.step(&t).expect("step");
        // Final value should be close to 1 (DC gain)
        let final_val = y.last().unwrap();
        assert!(
            (final_val - 1.0).abs() < 0.1,
            "final step value = {final_val}"
        );
    }

    #[test]
    fn lti_poles_first_order() {
        // H(s) = 1 / (s + 2), pole at s = -2
        let sys = Lti::new(vec![1.0], vec![1.0, 2.0]).expect("valid");
        let poles = sys.poles().expect("poles");
        assert_eq!(poles.len(), 1);
        assert!(
            (poles[0].0 + 2.0).abs() < 1e-10,
            "pole real = {}",
            poles[0].0
        );
        assert!(poles[0].1.abs() < 1e-10, "pole imag = {}", poles[0].1);
    }

    #[test]
    fn dlti_new_valid() {
        let sys = Dlti::new(vec![1.0], vec![1.0, 0.5], 0.1).expect("valid");
        assert_eq!(sys.dt, 0.1);
    }

    #[test]
    fn dlti_new_negative_dt_rejected() {
        let err = Dlti::new(vec![1.0], vec![1.0], -0.1).expect_err("negative dt");
        assert!(err.is_argument_error());
    }

    #[test]
    fn dlti_new_zero_dt_rejected() {
        let err = Dlti::new(vec![1.0], vec![1.0], 0.0).expect_err("zero dt");
        assert!(err.is_argument_error());
    }

    #[test]
    fn dlti_new_non_finite_dt_rejected() {
        let err = Dlti::new(vec![1.0], vec![1.0], f64::NAN).expect_err("nan dt");
        assert!(err.is_argument_error());

        let err = Dlti::new(vec![1.0], vec![1.0], f64::INFINITY).expect_err("inf dt");
        assert!(err.is_argument_error());
    }

    #[test]
    fn dlti_step_response() {
        // H(z) = 1 / (1 - 0.5*z^-1), step response grows to 2
        let sys = Dlti::new(vec![1.0], vec![1.0, -0.5], 0.01).expect("valid");
        let y = sys.step(50).expect("step");
        // Should converge toward 1/(1-0.5) = 2
        let final_val = y.last().unwrap();
        assert!(
            (final_val - 2.0).abs() < 0.1,
            "final step value = {final_val}"
        );
    }

    #[test]
    fn dlti_impulse_response_first_sample() {
        // H(z) = 1 / (1 - 0.5*z^-1), impulse h[0] = 1
        let sys = Dlti::new(vec![1.0], vec![1.0, -0.5], 0.01).expect("valid");
        let h = sys.impulse(10).expect("impulse");
        assert!((h[0] - 1.0).abs() < 1e-10, "h[0] = {}", h[0]);
        // h[n] = 0.5^n for this system
        assert!((h[1] - 0.5).abs() < 1e-10, "h[1] = {}", h[1]);
    }

    #[test]
    fn dlti_is_stable_inside_unit_circle() {
        // Pole at z = 0.5 (inside unit circle) -> stable
        let sys = Dlti::new(vec![1.0], vec![1.0, -0.5], 0.01).expect("valid");
        assert!(sys.is_stable().expect("stable check"));
    }

    #[test]
    fn dlti_is_unstable_outside_unit_circle() {
        // Pole at z = 1.5 (outside unit circle) -> unstable
        let sys = Dlti::new(vec![1.0], vec![1.0, -1.5], 0.01).expect("valid");
        assert!(!sys.is_stable().expect("stable check"));
    }

    #[test]
    fn dlti_lfilter_matches_direct() {
        // Verify DLTI.lfilter matches direct lfilter call
        let sys = Dlti::new(vec![1.0, 0.5], vec![1.0, -0.3], 0.01).expect("valid");
        let x: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5).sin()).collect();
        let y1 = sys.lfilter(&x).expect("dlti lfilter");
        let y2 = lfilter(&sys.num, &sys.den, &x, None).expect("direct lfilter");
        for (i, (a, b)) in y1.iter().zip(y2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "y[{i}]: {} vs {}", a, b);
        }
    }

    #[test]
    fn lti_lsim_step_input() {
        // lsim with step input should match step response
        let sys = Lti::new(vec![1.0], vec![1.0, 1.0]).expect("valid");
        let t: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let u: Vec<f64> = vec![1.0; t.len()]; // step input
        let y_lsim = sys.lsim(&u, &t, None).expect("lsim");
        let y_step = sys.step(&t).expect("step");
        for (i, (a, b)) in y_lsim.iter().zip(y_step.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "lsim[{i}] = {} vs step[{i}] = {}",
                a,
                b
            );
        }
    }

    #[test]
    fn lti_lsim_zero_input() {
        // lsim with zero input should give zero output (zero IC)
        let sys = Lti::new(vec![1.0], vec![1.0, 1.0]).expect("valid");
        let t: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let u: Vec<f64> = vec![0.0; t.len()];
        let y = sys.lsim(&u, &t, None).expect("lsim");
        for (i, &yi) in y.iter().enumerate() {
            assert!(yi.abs() < 1e-12, "y[{i}] = {yi}, expected 0");
        }
    }

    #[test]
    fn lti_lsim_mismatched_lengths_rejected() {
        let sys = Lti::new(vec![1.0], vec![1.0, 1.0]).expect("valid");
        let t: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let u: Vec<f64> = vec![1.0; 5]; // shorter than t
        let err = sys.lsim(&u, &t, None).expect_err("mismatched");
        assert!(err.is_argument_error());
    }

    #[test]
    fn dlti_dlsim_matches_lfilter() {
        // dlsim should match lfilter with no initial state
        let sys = Dlti::new(vec![1.0, 0.5], vec![1.0, -0.3], 0.01).expect("valid");
        let u: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5).sin()).collect();
        let y_dlsim = sys.dlsim(&u, None).expect("dlsim");
        let y_lfilter = sys.lfilter(&u).expect("lfilter");
        for (i, (a, b)) in y_dlsim.iter().zip(y_lfilter.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "y[{i}]: {} vs {}", a, b);
        }
    }

    #[test]
    fn lsim_standalone_matches_method() {
        let sys = Lti::new(vec![2.0], vec![1.0, 0.5]).expect("valid");
        let t: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
        let u: Vec<f64> = t.iter().map(|&ti| ti.sin()).collect();
        let y_method = sys.lsim(&u, &t, None).expect("method");
        let y_standalone = lsim(&sys, &u, &t, None).expect("standalone");
        for (i, (a, b)) in y_method.iter().zip(y_standalone.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "y[{i}]: {} vs {}", a, b);
        }
    }

    #[test]
    fn dlsim_standalone_matches_method() {
        let sys = Dlti::new(vec![1.0], vec![1.0, -0.5], 0.01).expect("valid");
        let u: Vec<f64> = (0..15).map(|i| i as f64 * 0.1).collect();
        let y_method = sys.dlsim(&u, None).expect("method");
        let y_standalone = dlsim(&sys, &u, None).expect("standalone");
        for (i, (a, b)) in y_method.iter().zip(y_standalone.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "y[{i}]: {} vs {}", a, b);
        }
    }

    #[test]
    fn sosfreqz_matches_freqz_sos() {
        let sos: Vec<SosSection> = vec![[1.0, 0.5, 0.0, 1.0, -0.3, 0.1]];
        let r1 = freqz_sos(&sos, Some(64)).expect("freqz_sos");
        let r2 = sosfreqz(&sos, Some(64)).expect("sosfreqz");
        assert_eq!(r1.w.len(), r2.w.len());
        for (i, (a, b)) in r1.h_mag.iter().zip(r2.h_mag.iter()).enumerate() {
            assert!((a - b).abs() < 1e-14, "h_mag[{i}]: {} vs {}", a, b);
        }
    }

    #[test]
    fn sosfreqz_with_whole_unit_circle() {
        let sos: Vec<SosSection> = vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let half = sosfreqz(&sos, Some(128)).expect("half");
        let whole = sosfreqz_with_whole(&sos, Some(256), true).expect("whole");
        assert_eq!(half.w.len(), 128);
        assert_eq!(whole.w.len(), 256);
        assert!(whole.w.last().unwrap() > &std::f64::consts::PI);
    }
}
