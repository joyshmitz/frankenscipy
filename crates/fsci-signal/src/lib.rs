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
}
