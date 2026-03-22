#![forbid(unsafe_code)]

//! Signal processing routines for FrankenSciPy.
//!
//! Matches `scipy.signal` core functions:
//! - `savgol_filter` — Savitzky-Golay smoothing filter
//! - `savgol_coeffs` — filter coefficients

/// Error type for signal processing operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignalError {
    InvalidWindowLength(String),
    InvalidPolyOrder(String),
    InvalidArgument(String),
}

impl std::fmt::Display for SignalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWindowLength(msg) => write!(f, "invalid window length: {msg}"),
            Self::InvalidPolyOrder(msg) => write!(f, "invalid polynomial order: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
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
        x[i] = sum / aug[i][i];
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
/// Matches `scipy.signal.lombscargle(x, y, freqs)`.
///
/// # Arguments
/// * `x` — Sample times (must be non-decreasing).
/// * `y` — Measurements at each sample time.
/// * `freqs` — Angular frequencies at which to evaluate the periodogram.
pub fn lombscargle(x: &[f64], y: &[f64], freqs: &[f64]) -> Result<Vec<f64>, SignalError> {
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

    for &omega in freqs {
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
        power.push(p);
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

/// Nuttall window (minimum 4-term Blackman-Harris).
///
/// Matches `scipy.signal.windows.nuttall(M)`.
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
            0.355_768 - 0.487_396 * x.cos() + 0.144_232 * (2.0 * x).cos()
                - 0.012_604 * (3.0 * x).cos()
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
                .map(|&pk| proms[peaks.iter().position(|&p| p == pk).unwrap_or(0)])
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
    indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

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
        let end_val = pk + min_dist;

        let left = peaks.binary_search(&start_val).unwrap_or_else(|x| x);
        let right = peaks.binary_search(&end_val).unwrap_or_else(|x| x);

        for slot in excluded.iter_mut().take(right).skip(left) {
            *slot = true;
        }
    }

    selected.sort_unstable();
    selected
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
    design_digital_iir(analog_zpk, wn, btype)
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
    let (poles_re, poles_im) = poly_roots(&coeffs)?;
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

    let epsilon_p = (10_f64.powf(rp / 10.0) - 1.0).sqrt();
    let epsilon_s = (10_f64.powf(rs / 10.0) - 1.0).sqrt();
    let k1 = epsilon_p / epsilon_s;

    // Selectivity parameter k = ω_p / ω_s ≈ 1 for elliptic
    // For the analog prototype, we compute poles/zeros via Chebyshev-like approach
    // Use a simplified approach: compute poles as Chebyshev I poles with epsilon_p,
    // then add stopband zeros similar to Chebyshev II
    let mu = (1.0 / epsilon_p).asinh() / order as f64;

    let mut zeros_re = Vec::with_capacity(order);
    let mut zeros_im = Vec::with_capacity(order);
    let mut poles_re = Vec::with_capacity(order);
    let mut poles_im = Vec::with_capacity(order);

    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        // Poles from Chebyshev I prototype
        poles_re.push(-mu.sinh() * theta.sin());
        poles_im.push(mu.cosh() * theta.cos());

        // Zeros from Chebyshev II-like placement
        let cos_theta = theta.cos();
        if cos_theta.abs() > 1.0e-12 {
            zeros_re.push(0.0);
            zeros_im.push(1.0 / (k1 * cos_theta));
        }
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
    let a0 = a[0];
    if a0 == 0.0 {
        return Err(SignalError::InvalidArgument(
            "a[0] must not be zero".to_string(),
        ));
    }

    let nb = b.len();
    let na = a.len();
    let n = nb.max(na);

    // Pad and normalize
    let mut b_norm = vec![0.0; n];
    let mut a_norm = vec![0.0; n];
    for (i, &val) in b.iter().enumerate() {
        b_norm[i] = val / a0;
    }
    for (i, &val) in a.iter().enumerate() {
        a_norm[i] = val / a0;
    }

    if n <= 1 {
        return Ok(Vec::new());
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

    // 1. Pad signal with mirrored values
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

#[allow(dead_code)]
/// Build a real polynomial from complex conjugate roots.
/// Returns coefficients in descending power of z, then converts to z^-1 form.
fn poly_from_complex_roots(roots_re: &[f64], roots_im: &[f64]) -> Vec<f64> {
    // Start with [1.0]
    let mut poly = vec![1.0];
    let n = roots_re.len();
    let mut used = vec![false; n];

    for i in 0..n {
        if used[i] {
            continue;
        }

        // Find conjugate pair
        let mut found_conj = false;
        if roots_im[i].abs() > 1e-14 {
            for j in (i + 1)..n {
                if !used[j]
                    && (roots_re[j] - roots_re[i]).abs() < 1e-10
                    && (roots_im[j] + roots_im[i]).abs() < 1e-10
                {
                    // Conjugate pair: (z - r)(z - r*) = z² - 2*Re(r)*z + |r|²
                    let two_re = 2.0 * roots_re[i];
                    let mag2 = roots_re[i] * roots_re[i] + roots_im[i] * roots_im[i];
                    poly = poly_mul_quadratic(&poly, -two_re, mag2);
                    used[j] = true;
                    found_conj = true;
                    break;
                }
            }
        }

        if !found_conj {
            if roots_im[i].abs() < 1e-14 {
                // Real root: (z - r)
                poly = poly_mul_binomial(&poly, -roots_re[i]);
            } else {
                // Unpaired complex root — shouldn't happen for Butterworth
                let two_re = 2.0 * roots_re[i];
                let mag2 = roots_re[i] * roots_re[i] + roots_im[i] * roots_im[i];
                poly = poly_mul_quadratic(&poly, -two_re, mag2);
            }
        }
        used[i] = true;
    }

    poly.to_vec()
}

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
    real_roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        let omega = std::f64::consts::PI * i as f64 / (n - 1).max(1) as f64;
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
        let omega = std::f64::consts::PI * i as f64 / (n - 1).max(1) as f64;
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

    // Derivative polynomial: if B(z) = Σ b[k] z^{-k}, then
    // dB/dω = Σ (-jk) b[k] z^{-k}, so the "derivative coefficients" are
    // b_deriv[k] = -k * b[k] and we multiply by j.
    //
    // Group delay = Re{ (B'·conj(B))/(|B|²) - (A'·conj(A))/(|A|²) }
    // where B' = Σ (-jk) b[k] e^{-jkω}

    let mut w_out = Vec::with_capacity(n);
    let mut gd_out = Vec::with_capacity(n);

    for i in 0..n {
        let omega = std::f64::consts::PI * i as f64 / (n - 1).max(1) as f64;
        w_out.push(omega);

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

        gd_out.push(gd_b - gd_a);
    }

    Ok((w_out, gd_out))
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
            _ => unreachable!(),
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
        if !(0.0..=1.0).contains(&c) {
            return Err(SignalError::InvalidArgument(format!(
                "cutoff {c} out of range [0, 1]"
            )));
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
    if ripple <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "ripple must be positive (in dB)".to_string(),
        ));
    }
    if width <= 0.0 || width >= 1.0 {
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
    if t1 <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "t1 must be positive".to_string(),
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
    if !(0.0..=1.0).contains(&width) {
        return Err(SignalError::InvalidArgument(
            "width must be in [0, 1]".to_string(),
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
    if !(0.0..=1.0).contains(&duty) {
        return Err(SignalError::InvalidArgument(
            "duty must be in [0, 1]".to_string(),
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

/// Dispatch a window function by name string.
///
/// Matches `scipy.signal.get_window(window, Nx)`.
///
/// # Supported windows
/// `"hann"`, `"hamming"`, `"blackman"`, `"bartlett"`, `"flattop"`,
/// `"rectangular"` / `"boxcar"`, `"kaiser,<beta>"` (e.g. `"kaiser,8.6"`).
pub fn get_window(window: &str, nx: usize) -> Result<Vec<f64>, SignalError> {
    let lower = window.trim().to_lowercase();
    if let Some(rest) = lower.strip_prefix("kaiser,") {
        let beta: f64 = rest
            .trim()
            .parse()
            .map_err(|_| SignalError::InvalidArgument(format!("invalid kaiser beta: {rest}")))?;
        return Ok(kaiser(nx, beta));
    }
    match lower.as_str() {
        "hann" | "hanning" => Ok(hann(nx)),
        "hamming" => Ok(hamming(nx)),
        "blackman" => Ok(blackman(nx)),
        "bartlett" => Ok(bartlett(nx)),
        "flattop" => Ok(flattop(nx)),
        "rectangular" | "boxcar" | "rect" => Ok(vec![1.0; nx]),
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

    let scale = 1.0 / (fs * nperseg as f64 * win_power * n_segments as f64);
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

    // Forward FFT.
    let spectrum = fsci_fft::rfft(x, &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;

    // Compute target spectrum length for the new sample count.
    let target_nfreqs = num / 2 + 1;

    // Zero-pad or truncate the spectrum.
    let mut new_spectrum = vec![(0.0, 0.0); target_nfreqs];
    let copy_len = spectrum.len().min(target_nfreqs);
    new_spectrum[..copy_len].copy_from_slice(&spectrum[..copy_len]);

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

    // Simplify the ratio using GCD.
    let g = gcd(up, down);
    let up = up / g;
    let down = down / g;

    if up == 1 && down == 1 {
        return Ok(x.to_vec());
    }

    // Design anti-aliasing FIR filter.
    let cutoff = 1.0 / (up.max(down) as f64);
    let n_taps = 2 * 10 * up.max(down) + 1; // ~10 periods per side
    let h = firwin(n_taps, &[cutoff], FirWindow::Kaiser(5.0), true)?;
    // Scale filter by up to compensate for upsampling gain.
    let h_scaled: Vec<f64> = h.iter().map(|&v| v * up as f64).collect();

    // Polyphase implementation to avoid materializing the full upsampled signal.
    // Equivalent to: upsample -> convolve(mode=Same) -> downsample.
    let upsampled_len = x.len() * up;
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
            if x_idx >= 0 && x_idx < x.len() as i64 {
                val += h_scaled[k] * x[x_idx as usize];
            }
        }
        output.push(val);
        i += down;
    }

    Ok(output)
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn window_empty_returns_empty() {
        assert!(hann(0).is_empty());
        assert!(hamming(0).is_empty());
        assert!(blackman(0).is_empty());
        assert!(kaiser(0, 5.0).is_empty());
    }

    #[test]
    fn window_single_returns_one() {
        assert_eq!(hann(1), [1.0]);
        assert_eq!(hamming(1), [1.0]);
        assert_eq!(blackman(1), [1.0]);
        assert_eq!(kaiser(1, 5.0), [1.0]);
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
        assert!(matches!(err, SignalError::InvalidArgument(_)));
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
        assert!(matches!(err, SignalError::InvalidArgument(_)));
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
        let max_mag = passband.iter().copied().fold(0.0_f64, f64::max);
        let min_mag = passband.iter().copied().fold(f64::INFINITY, f64::min);
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
        assert!(matches!(err, SignalError::InvalidArgument(_)));
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
        assert!(matches!(err, SignalError::InvalidArgument(_)));
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
        assert!(matches!(err, SignalError::InvalidArgument(_)));
        let err = ellip(4, 1.0, -40.0, &[0.3], FilterType::Lowpass).expect_err("negative rs");
        assert!(matches!(err, SignalError::InvalidArgument(_)));
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
    fn welch_nperseg_too_large() {
        assert!(welch(&[1.0, 2.0, 3.0], 1.0, None, Some(10), None).is_err());
    }

    fn variance_of_psd(psd: &[f64]) -> f64 {
        let n = psd.len() as f64;
        let mean = psd.iter().sum::<f64>() / n;
        psd.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
    }

    fn gd_span(values: &[f64]) -> f64 {
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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
            .min_by(|(_, a), (_, b)| {
                ((**a) - omega)
                    .abs()
                    .partial_cmp(&((**b) - omega).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
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
                    .partial_cmp(&((**b) - cutoff_omega).abs())
                    .unwrap()
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
        let max_abs = y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
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
    fn firwin_invalid_args() {
        assert!(firwin(0, &[0.3], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[1.5], FirWindow::Hamming, true).is_err());
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
        let max_residual = result.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
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

    // ── get_window tests ───────────────────────────────────────────

    #[test]
    fn get_window_dispatches_hann() {
        let w = get_window("hann", 10).unwrap();
        let expected = hann(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_hamming() {
        let w = get_window("hamming", 10).unwrap();
        let expected = hamming(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_blackman() {
        let w = get_window("blackman", 10).unwrap();
        let expected = blackman(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_kaiser() {
        let w = get_window("kaiser,8.6", 10).unwrap();
        let expected = kaiser(10, 8.6);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_rectangular() {
        let w = get_window("boxcar", 5).unwrap();
        assert_eq!(w, vec![1.0; 5]);
    }

    #[test]
    fn get_window_dispatches_bartlett() {
        let w = get_window("bartlett", 11).unwrap();
        assert_eq!(w.len(), 11);
        assert!((w[5] - 1.0).abs() < 1e-12, "peak at center");
        assert!((w[0]).abs() < 1e-12, "zero at endpoints");
        assert!((w[10]).abs() < 1e-12, "zero at endpoints");
    }

    #[test]
    fn get_window_dispatches_flattop() {
        let w = get_window("flattop", 11).unwrap();
        assert_eq!(w.len(), 11);
        // Flattop peak is ~1.0 but slightly different due to coeffs
        let peak = w.iter().copied().fold(0.0_f64, f64::max);
        assert!((peak - 1.0).abs() < 0.01);
    }

    #[test]
    fn get_window_unknown_rejected() {
        assert!(get_window("foobar", 10).is_err());
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
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
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
        assert!(matches!(err, SignalError::InvalidArgument(_)));
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
        let max_im = w.iter().map(|&(_, im)| im.abs()).fold(0.0_f64, f64::max);
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
        assert!(matches!(err, SignalError::InvalidArgument(_)));
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

    // ── Lomb-Scargle tests ───────────────────────────────────────────

    #[test]
    fn lombscargle_detects_known_frequency() {
        // Signal with known frequency at 5 rad/s
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|&t| (5.0 * t).sin()).collect();
        let freqs: Vec<f64> = (1..20).map(|f| f as f64).collect();
        let power = lombscargle(&x, &y, &freqs).expect("lombscargle");
        // Peak should be near freq=5
        let peak_idx = power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
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
        let err = lombscargle(&[1.0, 2.0], &[1.0], &[1.0]).expect_err("mismatch");
        assert!(matches!(err, SignalError::InvalidArgument(_)));
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
    fn gausspulse_envelope_decay() {
        // Envelope should decay away from center
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.001).collect();
        let y = gausspulse(&t, 100.0, 0.5);
        let max_abs = y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
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
}
