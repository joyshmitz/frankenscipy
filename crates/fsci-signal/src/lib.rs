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
        let theta = std::f64::consts::PI * (2.0 * k as f64 + n as f64 + 1.0) / (2.0 * n as f64);
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

    // Build numerator: bilinear transform maps s=∞ zeros to z=-1 (lowpass)
    // In z^{-1} notation: (1 + z^{-1}) for each lowpass zero, (1 - z^{-1}) for highpass
    let zero_coeff = match btype {
        FilterType::Lowpass => 1.0,   // factor (1 + z^{-1}), zero at z=-1
        FilterType::Highpass => -1.0, // factor (1 - z^{-1}), zero at z=1
    };
    let mut b = vec![1.0];
    for _ in 0..n {
        b = poly_mul_binomial(&b, zero_coeff);
    }

    // Normalize: gain at DC (z=1) for lowpass, gain at Nyquist (z=-1) for highpass = 1
    // In z^{-1} notation: H(z) = Σ b[k]*z^{-k} / Σ a[k]*z^{-k}
    // At z=1: z^{-k} = 1 for all k, so H(1) = sum(b) / sum(a)
    // At z=-1: z^{-k} = (-1)^k
    let (b_at_z, a_at_z) = match btype {
        FilterType::Lowpass => {
            let bz: f64 = b.iter().sum();
            let az: f64 = a.iter().sum();
            (bz, az)
        }
        FilterType::Highpass => {
            let bz: f64 = b
                .iter()
                .enumerate()
                .map(|(i, &c)| c * if i % 2 == 0 { 1.0 } else { -1.0 })
                .sum();
            let az: f64 = a
                .iter()
                .enumerate()
                .map(|(i, &c)| c * if i % 2 == 0 { 1.0 } else { -1.0 })
                .sum();
            (bz, az)
        }
    };

    if b_at_z.abs() < 1.0e-30 {
        return Err(SignalError::InvalidArgument(
            "degenerate filter: numerator evaluates to zero at normalization frequency".to_string(),
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
    let b_lead_idx = b.iter().position(|&v| v.abs() > 1e-30).unwrap_or(0);
    let b_lead = if b_lead_idx < b.len() {
        b[b_lead_idx]
    } else {
        return Ok(ZpkCoeffs {
            zeros_re: vec![],
            zeros_im: vec![],
            poles_re: vec![],
            poles_im: vec![],
            gain: 0.0,
        });
    };

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
                if !used[j]
                    && (re[j] - re[i]).abs() < 1e-10
                    && (im[j] + im[i]).abs() < 1e-10
                {
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

        if b_trim.len() > 1 && let Ok((zr, zi)) = poly_roots(&b_trim) {
            all_zeros_re.extend(zr);
            all_zeros_im.extend(zi);
        }
        if a_trim.len() > 1 && let Ok((pr, pi)) = poly_roots(&a_trim) {
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
    let eig_result = fsci_linalg::eig(&companion, opts).map_err(|e| {
        SignalError::InvalidArgument(format!("eigenvalue computation failed: {e}"))
    })?;

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
    let spectrum = fsci_fft::rfft(&windowed, &opts).map_err(|e| {
        SignalError::InvalidArgument(format!("FFT failed: {e}"))
    })?;

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
/// * `nperseg` — Length of each segment (default: 256)
/// * `noverlap` — Number of overlapping samples (default: nperseg/2)
pub fn welch(
    x: &[f64],
    fs: f64,
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
    let window = hann(nperseg);

    // Segment the signal and compute periodograms
    let n_freqs = nperseg / 2 + 1;
    let mut avg_psd = vec![0.0; n_freqs];
    let mut n_segments = 0usize;

    let mut start = 0;
    while start + nperseg <= x.len() {
        let segment = &x[start..start + nperseg];
        let seg_result = periodogram(segment, fs, Some(&window))?;

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

    Ok(SpectralResult { frequencies, psd: avg_psd })
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
        assert!((b_sum / a_sum - 1.0).abs() < 1e-10, "DC gain should be 1");
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
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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

        let result = welch(&x, fs, Some(256), None).expect("welch");
        assert!(!result.frequencies.is_empty());
        assert_eq!(result.frequencies.len(), result.psd.len());

        // Peak should be near 50 Hz
        let peak_idx = result
            .psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
        let psd_welch = welch(&x, fs, Some(128), None).expect("welch");

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
        assert!(welch(&[], 1.0, None, None).is_err());
    }

    #[test]
    fn welch_nperseg_too_large() {
        assert!(welch(&[1.0, 2.0, 3.0], 1.0, Some(10), None).is_err());
    }

    fn variance_of_psd(psd: &[f64]) -> f64 {
        let n = psd.len() as f64;
        let mean = psd.iter().sum::<f64>() / n;
        psd.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
    }

    // ── Filter conversion tests ────────────────────────────────────

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{msg}: {a} vs {b} (diff={})",
            (a - b).abs()
        );
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
        let coeffs = butter(2, 0.3, FilterType::Lowpass).expect("butter");
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
        let coeffs = butter(2, 0.3, FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        // Order 2 → 1 SOS section
        assert_eq!(sos.len(), 1, "butter(2) should give 1 SOS section");
        // a0 should be 1.0
        assert_close(sos[0][3], 1.0, 1e-10, "a0 = 1");
    }

    #[test]
    fn tf2sos_butter4() {
        let coeffs = butter(4, 0.3, FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        // Order 4 → 2 SOS sections
        assert_eq!(sos.len(), 2, "butter(4) should give 2 SOS sections");
    }

    #[test]
    fn sos2tf_roundtrip() {
        let coeffs = butter(3, 0.4, FilterType::Lowpass).expect("butter");
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
        let coeffs = butter(4, 0.25, FilterType::Lowpass).expect("butter");
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
        let coeffs = butter(8, 0.2, FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        assert_eq!(sos.len(), 4, "order 8 → 4 sections");

        // Each section's poles should be inside unit circle
        for (i, section) in sos.iter().enumerate() {
            let a2 = section[5];
            // For stability: |a2| < 1
            assert!(
                a2.abs() < 1.0 + 1e-10,
                "section {i} unstable: a2={a2}"
            );
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
}
