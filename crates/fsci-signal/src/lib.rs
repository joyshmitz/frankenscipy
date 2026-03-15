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

    // Build Vandermonde-like matrix: A[i, j] = x_i^j where x_i = i - half
    // Then solve A^T A c = A^T e_deriv for the coefficients
    // where e_deriv selects the deriv-th polynomial coefficient

    // Construct the normal equations: (A^T A) and (A^T b)
    // A is window_length × order, b = e_{deriv} (unit vector)
    let mut ata = vec![vec![0.0; order]; order];

    for i in 0..window_length {
        let xi = (i as i64 - half) as f64;
        let mut powers = vec![1.0; order];
        for j in 1..order {
            powers[j] = powers[j - 1] * xi;
        }
        for r in 0..order {
            for c in 0..order {
                ata[r][c] += powers[r] * powers[c];
            }
        }
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

    let c = solve_symmetric_positive(&ata, &rhs);

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

/// Solve a small SPD system via Cholesky-like decomposition (no pivoting).
fn solve_symmetric_positive(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    // Simple Gaussian elimination (A is small, typically < 10×10)
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
        if aug[col][col].abs() < f64::EPSILON * 1e6 {
            continue;
        }
        let pivot_row = aug[col].clone();
        for aug_row in aug.iter_mut().skip(col + 1) {
            let factor = aug_row[col] / pivot_row[col];
            for (j, pv) in pivot_row.iter().enumerate().skip(col) {
                aug_row[j] -= factor * pv;
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < f64::EPSILON * 1e6 {
            continue;
        }
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    x
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

    // Step 1: find all local maxima (strictly greater than both neighbors)
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..x.len() - 1 {
        if x[i] > x[i - 1] && x[i] > x[i + 1] {
            peaks.push(i);
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

    // Sort peaks by height (descending) for greedy selection
    let mut indexed: Vec<(usize, f64)> = peaks
        .iter()
        .zip(heights.iter())
        .map(|(&p, &h)| (p, h))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected = Vec::new();
    let mut excluded = vec![false; peaks.len()];

    for &(pk, _) in &indexed {
        let orig_idx = peaks.iter().position(|&p| p == pk).unwrap();
        if excluded[orig_idx] {
            continue;
        }
        selected.push(pk);
        // Exclude all peaks within min_dist
        for (j, &other_pk) in peaks.iter().enumerate() {
            if !excluded[j] && other_pk != pk && other_pk.abs_diff(pk) < min_dist {
                excluded[j] = true;
            }
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
/// * `wn` — Critical frequency (normalized, 0 < wn < 1 for digital)
/// * `btype` — Filter type (lowpass or highpass)
///
/// Returns (b, a) transfer function coefficients.
pub fn butter(order: usize, wn: f64, btype: FilterType) -> Result<BaCoeffs, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidArgument(
            "order must be >= 1".to_string(),
        ));
    }
    if wn <= 0.0 || wn >= 1.0 {
        return Err(SignalError::InvalidArgument(
            "Wn must be in (0, 1) for digital filters".to_string(),
        ));
    }

    // Prewarp: convert digital cutoff to analog cutoff
    let warped = 2.0 * (std::f64::consts::PI * wn / 2.0).tan();

    // Compute analog Butterworth prototype poles on unit circle in s-domain
    let n = order;
    let mut s_poles_re = Vec::with_capacity(n);
    let mut s_poles_im = Vec::with_capacity(n);
    for k in 0..n {
        let theta =
            std::f64::consts::PI * (2.0 * k as f64 + n as f64 + 1.0) / (2.0 * n as f64);
        // Scale poles by warped frequency
        s_poles_re.push(warped * theta.cos());
        s_poles_im.push(warped * theta.sin());
    }

    // For highpass: transform s -> warped^2 / s
    if btype == FilterType::Highpass {
        let w2 = warped * warped;
        for k in 0..n {
            let sr = s_poles_re[k];
            let si = s_poles_im[k];
            let mag2 = sr * sr + si * si;
            s_poles_re[k] = w2 * sr / mag2;
            s_poles_im[k] = -w2 * si / mag2;
        }
    }

    // Bilinear transform: z = (1 + s/2) / (1 - s/2)
    let mut z_poles_re = Vec::with_capacity(n);
    let mut z_poles_im = Vec::with_capacity(n);
    for k in 0..n {
        let sr = s_poles_re[k] / 2.0;
        let si = s_poles_im[k] / 2.0;
        let num_re = 1.0 + sr;
        let num_im = si;
        let den_re = 1.0 - sr;
        let den_im = -si;
        let den_mag2 = den_re * den_re + den_im * den_im;
        z_poles_re.push((num_re * den_re + num_im * den_im) / den_mag2);
        z_poles_im.push((num_im * den_re - num_re * den_im) / den_mag2);
    }

    // Build denominator polynomial from poles
    let a = poly_from_complex_roots(&z_poles_re, &z_poles_im);

    // Build numerator: for lowpass, zeros at z = -1; for highpass, zeros at z = 1
    let zero_loc = match btype {
        FilterType::Lowpass => -1.0,
        FilterType::Highpass => 1.0,
    };
    let mut b = vec![1.0];
    for _ in 0..n {
        b = poly_mul_binomial(&b, zero_loc);
    }

    // Normalize: gain at DC (lowpass) or Nyquist (highpass) = 1
    let eval_z: f64 = match btype {
        FilterType::Lowpass => 1.0,
        FilterType::Highpass => -1.0,
    };
    let b_at_z: f64 = b
        .iter()
        .enumerate()
        .map(|(i, &c)| c * eval_z.powi(i as i32))
        .sum();
    let a_at_z: f64 = a
        .iter()
        .enumerate()
        .map(|(i, &c)| c * eval_z.powi(i as i32))
        .sum();

    if b_at_z.abs() < 1.0e-30 {
        return Err(SignalError::InvalidArgument(
            "degenerate filter: numerator evaluates to zero at normalization frequency"
                .to_string(),
        ));
    }
    let gain = a_at_z / b_at_z;
    let b: Vec<f64> = b.iter().map(|&v| v * gain).collect();

    Ok(BaCoeffs { b, a })
}

/// Apply a digital IIR filter using the Direct Form II transposed structure.
///
/// Matches `scipy.signal.lfilter(b, a, x)`.
///
/// # Arguments
/// * `b` — Numerator coefficients
/// * `a` — Denominator coefficients (a[0] should be 1.0 or will be normalized)
/// * `x` — Input signal
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Result<Vec<f64>, SignalError> {
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
    let mut d = vec![0.0; nfilt]; // delay line

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
    if x.len() < 3 {
        return Err(SignalError::InvalidArgument(
            "input must have length >= 3 for filtfilt".to_string(),
        ));
    }

    // Forward pass
    let forward = lfilter(b, a, x)?;

    // Reverse
    let mut reversed: Vec<f64> = forward.into_iter().rev().collect();

    // Backward pass
    reversed = lfilter(b, a, &reversed)?;

    // Reverse again
    reversed.reverse();
    Ok(reversed)
}

// ── IIR helper functions ────────────────────────────────────────

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
        let coeffs = butter(1, 0.5, FilterType::Lowpass).expect("butter");
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
        let coeffs = butter(2, 0.3, FilterType::Lowpass).expect("butter");
        assert_eq!(coeffs.b.len(), 3);
        assert_eq!(coeffs.a.len(), 3);
        // DC gain = 1
        let b_sum: f64 = coeffs.b.iter().sum();
        let a_sum: f64 = coeffs.a.iter().sum();
        assert!(
            (b_sum / a_sum - 1.0).abs() < 1e-10,
            "DC gain should be 1"
        );
    }

    #[test]
    fn butter_highpass_nyquist_gain() {
        let coeffs = butter(2, 0.3, FilterType::Highpass).expect("butter hp");
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
        assert!(butter(0, 0.5, FilterType::Lowpass).is_err());
    }

    #[test]
    fn butter_invalid_wn() {
        assert!(butter(2, 0.0, FilterType::Lowpass).is_err());
        assert!(butter(2, 1.0, FilterType::Lowpass).is_err());
    }

    // ── lfilter tests ──────────────────────────────────────────────

    #[test]
    fn lfilter_identity() {
        // b=[1], a=[1] should pass signal through
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = lfilter(&[1.0], &[1.0], &x).expect("lfilter");
        assert_eq!(y, x);
    }

    #[test]
    fn lfilter_delay() {
        // b=[0, 1], a=[1] = one-sample delay
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = lfilter(&[0.0, 1.0], &[1.0], &x).expect("lfilter");
        assert!((y[0]).abs() < 1e-12);
        assert!((y[1] - 1.0).abs() < 1e-12);
        assert!((y[2] - 2.0).abs() < 1e-12);
        assert!((y[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn lfilter_first_order_iir() {
        // y[n] = x[n] + 0.5*y[n-1] (b=[1], a=[1, -0.5])
        let x = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let y = lfilter(&[1.0], &[1.0, -0.5], &x).expect("lfilter");
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 0.5).abs() < 1e-12);
        assert!((y[2] - 0.25).abs() < 1e-12);
        assert!((y[3] - 0.125).abs() < 1e-12);
    }

    #[test]
    fn lfilter_with_butter() {
        let coeffs = butter(2, 0.2, FilterType::Lowpass).expect("butter");
        let x: Vec<f64> = (0..100)
            .map(|i| (2.0 * std::f64::consts::PI * 0.05 * i as f64).sin())
            .collect();
        let y = lfilter(&coeffs.b, &coeffs.a, &x).expect("lfilter");
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // ── filtfilt tests ─────────────────────────────────────────────

    #[test]
    fn filtfilt_zero_phase() {
        // filtfilt should produce zero phase shift
        let coeffs = butter(2, 0.3, FilterType::Lowpass).expect("butter");
        let n = 100;
        let freq = 0.05;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).sin())
            .collect();
        let y = filtfilt(&coeffs.b, &coeffs.a, &x).expect("filtfilt");
        assert_eq!(y.len(), n);

        // Peak of input and output should be at approximately the same position
        // (zero phase means no shift)
        let input_peak = x
            .iter()
            .enumerate()
            .skip(10)
            .take(40)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let output_peak = y
            .iter()
            .enumerate()
            .skip(10)
            .take(40)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            input_peak.abs_diff(output_peak) <= 1,
            "filtfilt should have zero phase: input peak at {input_peak}, output peak at {output_peak}"
        );
    }

    #[test]
    fn filtfilt_attenuates_high_freq() {
        let coeffs = butter(3, 0.1, FilterType::Lowpass).expect("butter");
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
}
