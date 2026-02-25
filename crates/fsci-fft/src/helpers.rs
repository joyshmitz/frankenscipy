use crate::transforms::FftError;

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

#[cfg(test)]
mod tests {
    use super::{fftfreq, fftshift_1d, ifftshift_1d, rfftfreq};

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
}
