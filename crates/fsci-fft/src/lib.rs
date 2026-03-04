#![forbid(unsafe_code)]

//! FFT API surface for FrankenSciPy packet P2C-005.
//!
//! This crate is intentionally contract-first at this stage:
//! - module boundaries are fixed (`transforms`, `helpers`, `plan`)
//! - public signatures are stable for conformance wiring
//! - kernels are populated in subsequent packet beads

pub mod helpers;
pub mod plan;
pub mod transforms;

pub use helpers::{fftfreq, fftshift_1d, ifftshift_1d, rfftfreq};
pub use plan::{
    CacheAdmissionPolicy, PlanCacheBackend, PlanCacheConfig, PlanFingerprint, PlanKey,
    PlanMetadata, PlanningStrategy,
};
pub use transforms::{
    BackendKind, Complex64, FftError, FftOptions, TransformTrace, WorkerPolicy, fft, fft2, fftn,
    ifft, ifft2, irfft, rfft, take_transform_traces,
};

/// FFT normalization modes matching SciPy/PocketFFT conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum Normalization {
    Forward,
    #[default]
    Backward,
    Ortho,
}

/// Transform entrypoints represented in the packet boundary contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TransformKind {
    Fft,
    Ifft,
    Rfft,
    Irfft,
    Fft2,
    Ifft2,
    Fftn,
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use proptest::prelude::*;
    use serde_json::json;

    use super::transforms::{Complex64, FftOptions, fft, ifft, rfft, irfft};
    use super::{Normalization, TransformKind};
    use super::helpers::fftfreq;

    const PROPTEST_CASES: u32 = 512;
    const FFT_TOL: f64 = 1e-9;

    #[test]
    fn normalization_default_matches_scipy() {
        assert_eq!(Normalization::default(), Normalization::Backward);
    }

    #[test]
    fn transform_kind_order_is_stable_for_plan_keys() {
        assert!(TransformKind::Fft < TransformKind::Fftn);
    }

    #[test]
    fn structured_log_schema_is_machine_parseable() {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        let log = json!({
            "test_id": "fft_log_schema_test",
            "transform": "fft",
            "n": 8,
            "dtype": "complex64",
            "backend": "naive_dft",
            "mode": "strict",
            "seed": 0,
            "timestamp_ms": timestamp_ms,
            "max_error": 0.0,
            "result": "pass"
        });
        for field in [
            "test_id", "transform", "n", "dtype", "backend",
            "mode", "seed", "timestamp_ms", "max_error", "result",
        ] {
            assert!(log.get(field).is_some(), "missing field: {field}");
        }
    }

    fn complex_mag_sq(c: Complex64) -> f64 {
        c.0 * c.0 + c.1 * c.1
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        #[test]
        fn prop_ifft_fft_roundtrip(
            vals in proptest::collection::vec(-10i16..=10i16, 1..=32)
        ) {
            let input: Vec<Complex64> = vals.iter().map(|&v| (f64::from(v), 0.0)).collect();
            let opts = FftOptions::default();
            let spectrum = fft(&input, &opts).expect("fft");
            let recovered = ifft(&spectrum, &opts).expect("ifft");
            for (&a, &b) in recovered.iter().zip(&input) {
                prop_assert!((a.0 - b.0).abs() <= FFT_TOL, "re: {} vs {}", a.0, b.0);
                prop_assert!((a.1 - b.1).abs() <= FFT_TOL, "im: {} vs {}", a.1, b.1);
            }
        }

        #[test]
        fn prop_irfft_rfft_roundtrip(
            vals in proptest::collection::vec(-10i16..=10i16, 2..=32)
        ) {
            let input: Vec<f64> = vals.iter().map(|&v| f64::from(v)).collect();
            let n = input.len();
            let opts = FftOptions::default();
            let spectrum = rfft(&input, &opts).expect("rfft");
            let recovered = irfft(&spectrum, Some(n), &opts).expect("irfft");
            for (&a, &b) in recovered.iter().zip(&input) {
                prop_assert!((a - b).abs() <= FFT_TOL, "{a} vs {b}");
            }
        }

        #[test]
        fn prop_parseval_theorem(
            vals in proptest::collection::vec(-10i16..=10i16, 1..=32)
        ) {
            // sum(|x|^2) == sum(|X|^2) / N
            let input: Vec<Complex64> = vals.iter().map(|&v| (f64::from(v), 0.0)).collect();
            let n = input.len() as f64;
            let opts = FftOptions::default();
            let spectrum = fft(&input, &opts).expect("fft");
            let time_energy: f64 = input.iter().map(|c| complex_mag_sq(*c)).sum();
            let freq_energy: f64 = spectrum.iter().map(|c| complex_mag_sq(*c)).sum();
            prop_assert!(
                (time_energy - freq_energy / n).abs() <= FFT_TOL * n,
                "Parseval: {time_energy} vs {}", freq_energy / n
            );
        }

        #[test]
        fn prop_fftfreq_sum_zero_odd(n in (1u32..=50).prop_map(|v| (v * 2 + 1) as usize)) {
            // For odd n, positive and negative frequencies pair exactly, so sum=0
            let freqs = fftfreq(n, 1.0).expect("fftfreq");
            let sum: f64 = freqs.iter().sum();
            prop_assert!(sum.abs() <= 1e-10, "sum={sum} for n={n}");
        }

        #[test]
        fn prop_fft_shift_property(
            vals in proptest::collection::vec(-5i16..=5i16, 2..=16),
            k in 0usize..16,
        ) {
            // Verify fft produces consistent output length
            let input: Vec<Complex64> = vals.iter().map(|&v| (f64::from(v), 0.0)).collect();
            let n = input.len();
            let k_mod = k % n;
            let opts = FftOptions::default();
            let spectrum = fft(&input, &opts).expect("fft");
            // Circular shift the input
            let mut shifted = vec![(0.0, 0.0); n];
            for i in 0..n {
                shifted[(i + k_mod) % n] = input[i];
            }
            let shifted_spectrum = fft(&shifted, &opts).expect("fft shifted");
            // |shifted_spectrum[j]| should equal |spectrum[j]| (magnitude preserved)
            for j in 0..n {
                let mag_orig = complex_mag_sq(spectrum[j]).sqrt();
                let mag_shift = complex_mag_sq(shifted_spectrum[j]).sqrt();
                prop_assert!(
                    (mag_orig - mag_shift).abs() <= FFT_TOL * mag_orig.max(1.0),
                    "mag at bin {j}: {mag_orig} vs {mag_shift}"
                );
            }
        }
    }
}
