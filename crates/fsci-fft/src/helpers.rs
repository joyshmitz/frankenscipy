use crate::transforms::FftError;

const DIRECT_POLYMUL_CUTOFF: usize = 16;

/// Sample frequencies for the length-`n` complex FFT.
pub fn fftfreq(n: usize, sample_spacing: f64) -> Result<Vec<f64>, FftError> {
    validate_frequency_args(n, sample_spacing)?;
    let scale = 1.0 / (n as f64 * sample_spacing);
    let split = n.div_ceil(2);

    let mut freqs = Vec::with_capacity(n);
    for idx in 0..n {
        if idx < split {
            freqs.push(idx as f64 * scale);
        } else {
            freqs.push(-((n - idx) as f64) * scale);
        }
    }
    Ok(freqs)
}

/// Sample frequencies for the length-`n` real FFT.
pub fn rfftfreq(n: usize, sample_spacing: f64) -> Result<Vec<f64>, FftError> {
    validate_frequency_args(n, sample_spacing)?;
    let scale = 1.0 / (n as f64 * sample_spacing);
    let upper = n / 2;
    Ok((0..=upper).map(|idx| idx as f64 * scale).collect())
}

/// Shift zero-frequency component to the center for 1D input.
#[must_use]
pub fn fftshift_1d<T: Clone>(input: &[T]) -> Vec<T> {
    rotate_left_owned(input, input.len() / 2)
}

/// Inverse shift for [`fftshift_1d`] over 1D input.
#[must_use]
pub fn ifftshift_1d<T: Clone>(input: &[T]) -> Vec<T> {
    rotate_left_owned(input, input.len().div_ceil(2))
}

fn validate_frequency_args(n: usize, sample_spacing: f64) -> Result<(), FftError> {
    if n == 0 {
        return Err(FftError::InvalidShape {
            detail: "n must be greater than zero",
        });
    }
    if !(sample_spacing.is_finite() && sample_spacing > 0.0) {
        return Err(FftError::NonPositiveSampleSpacing);
    }
    Ok(())
}

fn rotate_left_owned<T: Clone>(input: &[T], shift: usize) -> Vec<T> {
    if input.is_empty() {
        return Vec::new();
    }
    let split = shift % input.len();
    input[split..]
        .iter()
        .cloned()
        .chain(input[..split].iter().cloned())
        .collect()
}

/// Multiply two real-coefficient polynomials using FFT convolution.
///
/// Coefficients are ordered by ascending power:
/// `a[0] + a[1] x + ...`. Empty inputs return an empty product.
/// Small products use the exact direct kernel so tiny scientific kernels do
/// not pay FFT setup costs; larger products use the convolution theorem.
pub fn polynomial_multiply_fft(
    a: &[f64],
    b: &[f64],
    options: &crate::FftOptions,
) -> Result<Vec<f64>, FftError> {
    if a.is_empty() || b.is_empty() {
        return Ok(Vec::new());
    }
    validate_polynomial_coefficients(a, options)?;
    validate_polynomial_coefficients(b, options)?;

    let output_len = a
        .len()
        .checked_add(b.len())
        .and_then(|len| len.checked_sub(1))
        .ok_or(FftError::InvalidShape {
            detail: "polynomial product length overflow",
        })?;
    if output_len <= DIRECT_POLYMUL_CUTOFF {
        return Ok(polynomial_multiply_direct(a, b));
    }
    let fft_len = output_len
        .checked_next_power_of_two()
        .ok_or(FftError::InvalidShape {
            detail: "polynomial FFT length overflow",
        })?;

    let mut transform_options = options.clone();
    transform_options.normalization = crate::Normalization::Backward;

    let mut lhs: Vec<(f64, f64)> = a.iter().map(|&value| (value, 0.0)).collect();
    lhs.resize(fft_len, (0.0, 0.0));
    let mut rhs: Vec<(f64, f64)> = b.iter().map(|&value| (value, 0.0)).collect();
    rhs.resize(fft_len, (0.0, 0.0));

    let lhs_spectrum = crate::fft(&lhs, &transform_options)?;
    let rhs_spectrum = crate::fft(&rhs, &transform_options)?;
    let product_spectrum: Vec<(f64, f64)> = lhs_spectrum
        .iter()
        .zip(rhs_spectrum.iter())
        .map(|(&(lhs_re, lhs_im), &(rhs_re, rhs_im))| {
            (
                lhs_re * rhs_re - lhs_im * rhs_im,
                lhs_re * rhs_im + lhs_im * rhs_re,
            )
        })
        .collect();

    let product = crate::ifft(&product_spectrum, &transform_options)?;
    Ok(product
        .iter()
        .take(output_len)
        .map(|&(real, _)| real)
        .collect())
}

fn validate_polynomial_coefficients(
    coeffs: &[f64],
    options: &crate::FftOptions,
) -> Result<(), FftError> {
    let should_check = options.check_finite || options.mode == fsci_runtime::RuntimeMode::Hardened;
    if should_check && coeffs.iter().any(|value| !value.is_finite()) {
        return Err(FftError::NonFiniteInput);
    }
    Ok(())
}

fn polynomial_multiply_direct(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut product = vec![0.0; a.len() + b.len() - 1];
    for (i, &lhs) in a.iter().enumerate() {
        for (j, &rhs) in b.iter().enumerate() {
            product[i + j] += lhs * rhs;
        }
    }
    product
}

/// FFT-based convolution of two 1D real signals.
///
/// Matches `scipy.signal.fftconvolve(a, b, mode)`.
///
/// `mode`: "full" (default), "same", or "valid".
pub fn fftconvolve(a: &[f64], b: &[f64], mode: &str) -> Result<Vec<f64>, FftError> {
    if a.is_empty() || b.is_empty() {
        return Ok(vec![]);
    }

    let n = a.len() + b.len() - 1;
    let fft_len = n.next_power_of_two();
    let opts = crate::FftOptions::default();

    // Zero-pad and convert to complex
    let mut ca: Vec<(f64, f64)> = a.iter().map(|&x| (x, 0.0)).collect();
    ca.resize(fft_len, (0.0, 0.0));
    let mut cb: Vec<(f64, f64)> = b.iter().map(|&x| (x, 0.0)).collect();
    cb.resize(fft_len, (0.0, 0.0));

    // FFT both
    let fa = crate::fft(&ca, &opts)?;
    let fb = crate::fft(&cb, &opts)?;

    // Pointwise multiply
    let fc: Vec<(f64, f64)> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&(ar, ai), &(br, bi))| (ar * br - ai * bi, ar * bi + ai * br))
        .collect();

    // IFFT
    let result_full = crate::ifft(&fc, &opts)?;

    // Extract real parts, truncated to length n
    let full: Vec<f64> = result_full[..n].iter().map(|&(r, _)| r).collect();

    // [frankenscipy-yjp4n] The 'same' and 'valid' arms below subtract
    // 1 from a.len() / b.len() and would underflow on empty inputs.
    // The early-return at the top of fftconvolve already guards
    // against that; pin the precondition here so any future
    // refactor that moves the guard fails under cargo test, not in
    // production.
    debug_assert!(
        !a.is_empty() && !b.is_empty(),
        "fftconvolve mode-arm precondition: both inputs must be non-empty here"
    );

    // Apply mode
    match mode {
        "same" => {
            let start = (b.len() - 1) / 2;
            let end = start + a.len();
            Ok(full[start..end.min(full.len())].to_vec())
        }
        "valid" => {
            if a.len() >= b.len() {
                let start = b.len() - 1;
                let end = a.len();
                Ok(full[start..end.min(full.len())].to_vec())
            } else {
                let start = a.len() - 1;
                let end = b.len();
                Ok(full[start..end.min(full.len())].to_vec())
            }
        }
        _ => Ok(full), // "full"
    }
}

/// FFT-based cross-correlation of two 1D real signals.
///
/// Equivalent to convolving a with reversed b.
pub fn fftcorrelate(a: &[f64], b: &[f64], mode: &str) -> Result<Vec<f64>, FftError> {
    // Correlation = convolution with time-reversed b
    let b_rev: Vec<f64> = b.iter().rev().cloned().collect();
    fftconvolve(a, &b_rev, mode)
}

/// Compute the power spectral density of a real signal.
///
/// Returns (frequencies, power) using Welch-like periodogram.
pub fn periodogram_simple(x: &[f64], fs: f64) -> Result<(Vec<f64>, Vec<f64>), FftError> {
    if x.is_empty() {
        return Ok((vec![], vec![]));
    }
    let n = x.len();
    let opts = crate::FftOptions::default();

    let complex_input: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();
    let spectrum = crate::fft(&complex_input, &opts)?;

    let n_freq = n / 2 + 1;
    let mut power = Vec::with_capacity(n_freq);
    let mut freqs = Vec::with_capacity(n_freq);

    for (k, item) in spectrum.iter().enumerate().take(n_freq) {
        let (re, im) = *item;
        let mut p = (re * re + im * im) / (n as f64 * fs);
        // Double non-DC, non-Nyquist bins for one-sided spectrum
        if k > 0 && k < n / 2 {
            p *= 2.0;
        }
        power.push(p);
        freqs.push(k as f64 * fs / n as f64);
    }

    Ok((freqs, power))
}

pub type CrossSpectralResult = Result<(Vec<f64>, Vec<(f64, f64)>), FftError>;

/// Compute the cross-spectral density of two real signals.
pub fn cross_spectral_density(x: &[f64], y: &[f64], fs: f64) -> CrossSpectralResult {
    if x.len() != y.len() || x.is_empty() {
        return Ok((vec![], vec![]));
    }
    let n = x.len();
    let opts = crate::FftOptions::default();

    let cx: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();
    let cy: Vec<(f64, f64)> = y.iter().map(|&v| (v, 0.0)).collect();

    let fx = crate::fft(&cx, &opts)?;
    let fy = crate::fft(&cy, &opts)?;

    let n_freq = n / 2 + 1;
    let mut csd = Vec::with_capacity(n_freq);
    let mut freqs = Vec::with_capacity(n_freq);

    for k in 0..n_freq {
        let (xr, xi) = fx[k];
        let (yr, yi) = fy[k];
        // Cross-spectrum: X * conj(Y)
        let cr = xr * yr + xi * yi;
        let ci = xi * yr - xr * yi;
        let scale = 1.0 / (n as f64 * fs);
        csd.push((cr * scale, ci * scale));
        freqs.push(k as f64 * fs / n as f64);
    }

    Ok((freqs, csd))
}

/// Apply a window function to a signal.
pub fn apply_window(x: &[f64], window: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(window.iter())
        .map(|(&xi, &wi)| xi * wi)
        .collect()
}

/// Generate a Hann window of length n.
pub fn hann_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let nf = (n - 1) as f64;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / nf).cos()))
        .collect()
}

/// Generate a Hamming window of length n.
pub fn hamming_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let nf = (n - 1) as f64;
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / nf).cos())
        .collect()
}

/// Generate a Blackman window of length n.
pub fn blackman_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let nf = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * i as f64 / nf;
            0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
        })
        .collect()
}

/// Compute the magnitude spectrum of a real signal.
///
/// Returns magnitudes (one-sided) for frequencies 0 to fs/2.
pub fn magnitude_spectrum(x: &[f64]) -> Result<Vec<f64>, FftError> {
    if x.is_empty() {
        return Ok(vec![]);
    }
    let n = x.len();
    let opts = crate::FftOptions::default();
    let complex_input: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();
    let spectrum = crate::fft(&complex_input, &opts)?;
    let n_freq = n / 2 + 1;
    Ok((0..n_freq)
        .map(|k| {
            let (re, im) = spectrum[k];
            (re * re + im * im).sqrt() / n as f64
        })
        .collect())
}

/// Compute the phase spectrum of a real signal.
///
/// Returns phase angles (one-sided).
pub fn phase_spectrum_signal(x: &[f64]) -> Result<Vec<f64>, FftError> {
    if x.is_empty() {
        return Ok(vec![]);
    }
    let n = x.len();
    let opts = crate::FftOptions::default();
    let complex_input: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();
    let spectrum = crate::fft(&complex_input, &opts)?;
    let n_freq = n / 2 + 1;
    Ok((0..n_freq)
        .map(|k| {
            let (re, im) = spectrum[k];
            im.atan2(re)
        })
        .collect())
}

/// Zero-pad a signal to the next power of 2.
pub fn zero_pad_pow2(x: &[f64]) -> Vec<f64> {
    let n = x.len().next_power_of_two();
    let mut padded = x.to_vec();
    padded.resize(n, 0.0);
    padded
}

/// Compute the analytic signal via FFT (Hilbert-like).
///
/// Returns complex analytic signal as Vec<(real, imag)>.
pub fn analytic_signal(x: &[f64]) -> Result<Vec<(f64, f64)>, FftError> {
    let n = x.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let opts = crate::FftOptions::default();
    let complex_input: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();
    let mut spectrum = crate::fft(&complex_input, &opts)?;

    // Double positive frequencies, zero negative frequencies
    if n > 1 {
        let half = n / 2;
        for item in &mut spectrum[1..half] {
            *item = (item.0 * 2.0, item.1 * 2.0);
        }
        for item in &mut spectrum[half + 1..n] {
            *item = (0.0, 0.0);
        }
    }

    crate::ifft(&spectrum, &opts)
}

#[cfg(test)]
mod tests {
    use fsci_runtime::RuntimeMode;

    use super::{
        fftconvolve, fftfreq, fftshift_1d, ifftshift_1d, polynomial_multiply_fft, rfftfreq,
    };
    use crate::FftOptions;

    #[test]
    fn fftfreq_even_length_matches_expected_ordering() {
        let freqs = fftfreq(8, 1.0).expect("fftfreq should succeed");
        assert_eq!(
            freqs,
            vec![0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
        );
    }

    #[test]
    fn rfftfreq_returns_non_negative_half_spectrum() {
        let freqs = rfftfreq(7, 0.5).expect("rfftfreq should succeed");
        assert_eq!(freqs, vec![0.0, 2.0 / 7.0, 4.0 / 7.0, 6.0 / 7.0]);
    }

    #[test]
    fn fftshift_and_ifftshift_roundtrip() {
        let data = vec![0, 1, 2, 3, 4];
        let shifted = fftshift_1d(&data);
        assert_eq!(shifted, vec![2, 3, 4, 0, 1]);
        assert_eq!(ifftshift_1d(&shifted), data);
    }

    #[test]
    fn fftfreq_odd_length_matches_expected() {
        let freqs = fftfreq(5, 1.0).expect("fftfreq odd");
        let expected = vec![0.0, 0.2, 0.4, -0.4, -0.2];
        assert_eq!(freqs.len(), expected.len());
        for (a, b) in freqs.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn rfftfreq_even_length_matches_expected() {
        let freqs = rfftfreq(8, 1.0).expect("rfftfreq even");
        let expected = vec![0.0, 0.125, 0.25, 0.375, 0.5];
        assert_eq!(freqs.len(), expected.len());
        for (a, b) in freqs.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn fftfreq_rejects_zero_n() {
        assert!(fftfreq(0, 1.0).is_err());
    }

    #[test]
    fn fftfreq_rejects_non_positive_spacing() {
        assert!(fftfreq(4, 0.0).is_err());
        assert!(fftfreq(4, -1.0).is_err());
        assert!(fftfreq(4, f64::NAN).is_err());
    }

    #[test]
    fn fftshift_even_length() {
        let data = vec![0, 1, 2, 3, 4, 5];
        let shifted = fftshift_1d(&data);
        assert_eq!(shifted, vec![3, 4, 5, 0, 1, 2]);
    }

    #[test]
    fn fftshift_length_1() {
        assert_eq!(fftshift_1d(&[42]), vec![42]);
    }

    #[test]
    fn fftshift_empty() {
        let empty: Vec<i32> = vec![];
        assert_eq!(fftshift_1d(&empty), empty);
    }

    #[test]
    fn fftconvolve_impulse() {
        // Convolving with [1] should return the signal unchanged
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0];
        let result = fftconvolve(&a, &b, "full").unwrap();
        assert_eq!(result.len(), 4);
        for (i, (&r, &e)) in result.iter().zip(a.iter()).enumerate() {
            assert!((r - e).abs() < 1e-10, "idx {i}: {r} != {e}");
        }
    }

    #[test]
    fn fftconvolve_known() {
        // [1, 2, 3] * [0, 1, 0.5] = [0, 1, 2.5, 4, 1.5]
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.5];
        let result = fftconvolve(&a, &b, "full").unwrap();
        let expected = [0.0, 1.0, 2.5, 4.0, 1.5];
        assert_eq!(result.len(), 5);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-10, "idx {i}: {r} != {e}");
        }
    }

    #[test]
    fn polynomial_multiply_fft_matches_direct_known_product() {
        let lhs = vec![3.0, -2.0, 5.0, 1.0];
        let rhs = vec![4.0, 0.5, -1.0];
        let got = polynomial_multiply_fft(&lhs, &rhs, &FftOptions::default()).unwrap();
        let expected = [12.0, -6.5, 16.0, 8.5, -4.5, -1.0];
        assert_eq!(got.len(), expected.len());
        for (idx, (&actual, &expected)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-10,
                "coefficient {idx}: {actual} != {expected}",
            );
        }
    }

    #[test]
    fn polynomial_multiply_fft_handles_empty_input() {
        let got = polynomial_multiply_fft(&[], &[1.0, 2.0], &FftOptions::default()).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn polynomial_multiply_fft_large_product_uses_convolution_theorem() {
        let lhs: Vec<f64> = (0..24).map(|idx| (idx % 7) as f64 - 3.0).collect();
        let rhs: Vec<f64> = (0..19).map(|idx| (idx % 5) as f64 * 0.25 - 0.5).collect();
        let got = polynomial_multiply_fft(&lhs, &rhs, &FftOptions::default()).unwrap();

        let mut expected = vec![0.0; lhs.len() + rhs.len() - 1];
        for (i, &left) in lhs.iter().enumerate() {
            for (j, &right) in rhs.iter().enumerate() {
                expected[i + j] += left * right;
            }
        }

        assert_eq!(got.len(), expected.len());
        for (idx, (&actual, &expected)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-8,
                "coefficient {idx}: {actual} != {expected}",
            );
        }
    }

    #[test]
    fn polynomial_multiply_fft_hardened_mode_rejects_non_finite_coefficients() {
        let options = FftOptions::default().with_mode(RuntimeMode::Hardened);
        let err = polynomial_multiply_fft(&[1.0, f64::NAN], &[2.0], &options).unwrap_err();
        assert_eq!(err, crate::FftError::NonFiniteInput);
    }

    #[test]
    fn fftconvolve_same_mode() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0];
        let result = fftconvolve(&a, &b, "same").unwrap();
        assert_eq!(result.len(), 5); // same length as a
    }

    #[test]
    fn fftconvolve_metamorphic_commutativity() {
        // /testing-metamorphic: fftconvolve(a, b, 'full') = fftconvolve(b, a, 'full').
        // The convolution operator is commutative, so swapping inputs
        // must produce identical output (up to floating-point noise).
        // Pin across multiple length pairs.
        let cases: &[(Vec<f64>, Vec<f64>)] = &[
            (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
            (vec![1.0, 0.5, -0.25], vec![2.0, -1.0]),
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![0.5, 0.5]),
            (vec![1.0], vec![3.0, 1.0, 4.0, 1.0, 5.0]),
        ];
        for (a, b) in cases {
            let ab = fftconvolve(a, b, "full").unwrap();
            let ba = fftconvolve(b, a, "full").unwrap();
            assert_eq!(ab.len(), ba.len(), "length mismatch");
            for (i, (&v1, &v2)) in ab.iter().zip(ba.iter()).enumerate() {
                assert!(
                    (v1 - v2).abs() < 1e-10,
                    "fftconvolve commutativity broken at i={i}: {v1} vs {v2}"
                );
            }
        }
    }
}
