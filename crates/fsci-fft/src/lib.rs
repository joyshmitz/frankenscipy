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
    BackendKind, Complex64, FftError, FftOptions, TransformTrace, WorkerPolicy, dct, dct_i,
    dct_iii, dct_iv, dst_i, dst_ii, dst_iii, dst_iv, fft, fft2, fftn, hilbert, idct, ifft, ifft2,
    irfft, rfft, take_transform_traces,
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

    use std::f64::consts::PI;

    use super::helpers::fftfreq;
    use super::transforms::{
        Complex64, FftOptions, dct, dct_i, dct_iii, dct_iv, dst_i, dst_ii, dst_iii, dst_iv, fft,
        hilbert, idct, ifft, irfft, rfft,
    };
    use super::{Normalization, TransformKind};

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
            "test_id",
            "transform",
            "n",
            "dtype",
            "backend",
            "mode",
            "seed",
            "timestamp_ms",
            "max_error",
            "result",
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

    // ── DCT tests ───────────────────────────────────────────────────

    #[test]
    fn dct_constant_input() {
        // DCT of constant [1,1,1,1]: DC component should be N*2 = 8, others should be ~0
        let input = vec![1.0; 4];
        let result = dct(&input, &FftOptions::default()).expect("dct");
        assert_eq!(result.len(), 4);
        assert!((result[0] - 8.0).abs() < 1e-9, "DC = {}", result[0]);
        for (k, &val) in result.iter().enumerate().skip(1) {
            assert!(val.abs() < 1e-9, "AC[{k}] = {val}");
        }
    }

    #[test]
    fn dct_impulse() {
        // DCT of [1, 0, 0, 0] should be [2*cos(0), 2*cos(π/8), 2*cos(2π/8), 2*cos(3π/8)]
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let result = dct(&input, &FftOptions::default()).expect("dct");
        assert_eq!(result.len(), 4);
        for (k, &val) in result.iter().enumerate() {
            let expected = 2.0 * (std::f64::consts::PI * k as f64 / 8.0).cos();
            assert!(
                (val - expected).abs() < 1e-9,
                "DCT[{k}] = {val}, expected {expected}",
            );
        }
    }

    #[test]
    fn dct_idct_roundtrip() {
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let opts = FftOptions::default();
        let spectrum = dct(&input, &opts).expect("dct");
        let recovered = idct(&spectrum, &opts).expect("idct");
        for (a, b) in recovered.iter().zip(&input) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn dct_length_1() {
        let result = dct(&[5.0], &FftOptions::default()).expect("dct len 1");
        assert!((result[0] - 10.0).abs() < 1e-9, "DCT[0] = {}", result[0]);
    }

    // ── DCT type variants ──────────────────────────────────────────

    #[test]
    fn dct_i_preserves_length() {
        let input = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let result = dct_i(&input, &FftOptions::default()).expect("dct_i");
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn dct_i_known_values() {
        // DCT-I of [1, 0, 0] with N=3:
        // X[k] = Σ x[n]*cos(πnk/2) = cos(0) = 1 for all k
        let input = vec![1.0, 0.0, 0.0];
        let result = dct_i(&input, &FftOptions::default()).expect("dct_i");
        assert_eq!(result.len(), 3);
        for (k, &val) in result.iter().enumerate() {
            let mut expected = 0.0;
            for (n, &x) in input.iter().enumerate() {
                expected += x * (PI * n as f64 * k as f64 / 2.0).cos();
            }
            assert!(
                (val - expected).abs() < 1e-9,
                "DCT-I[{k}] = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn dct_iii_inverts_dct_ii() {
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let opts = FftOptions::default();
        let forward = dct(&input, &opts).expect("dct");
        let inverse = dct_iii(&forward, &opts).expect("dct_iii");
        // Find scale factor
        let scale = inverse[0] / input[0];
        assert!(scale > 0.0, "scale should be positive");
        for (a, b) in inverse.iter().zip(&input) {
            assert!(
                (a / scale - b).abs() < 1e-8,
                "DCT-III(DCT-II(x)): {a}/{scale} vs {b}"
            );
        }
    }

    #[test]
    fn dct_iv_self_inverse() {
        // DCT-IV is its own inverse (up to scaling by N/2)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let opts = FftOptions::default();
        let forward = dct_iv(&input, &opts).expect("dct_iv");
        let roundtrip = dct_iv(&forward, &opts).expect("dct_iv roundtrip");
        let scale = input.len() as f64 / 2.0;
        for (a, b) in roundtrip.iter().zip(&input) {
            assert!(
                (a / scale - b).abs() < 1e-9,
                "DCT-IV roundtrip: {} vs {b}",
                a / scale
            );
        }
    }

    // ── DST variants ──────────────────────────────────────────────

    #[test]
    fn dst_i_roundtrip_proportional() {
        // DST-I applied twice gives back a scaled version of input
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let opts = FftOptions::default();
        let forward = dst_i(&input, &opts).expect("dst_i");
        let roundtrip = dst_i(&forward, &opts).expect("dst_i roundtrip");
        let scale = roundtrip[0] / input[0];
        assert!(scale > 0.0, "scale should be positive");
        for (a, b) in roundtrip.iter().zip(&input) {
            assert!(
                (a / scale - b).abs() < 1e-9,
                "DST-I roundtrip: {a}/{scale} vs {b}"
            );
        }
    }

    #[test]
    fn dst_ii_constant_input() {
        // DST-II of constant should give specific pattern
        let input = vec![1.0; 4];
        let result = dst_ii(&input, &FftOptions::default()).expect("dst_ii");
        assert_eq!(result.len(), 4);
        // All values should be finite
        assert!(result.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn dst_iii_inverts_dst_ii() {
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let opts = FftOptions::default();
        let forward = dst_ii(&input, &opts).expect("dst_ii");
        let inverse = dst_iii(&forward, &opts).expect("dst_iii");
        // Find scale factor
        let scale = inverse[0] / input[0];
        assert!(scale > 0.0, "scale should be positive");
        for (a, b) in inverse.iter().zip(&input) {
            assert!(
                (a / scale - b).abs() < 1e-8,
                "DST-III(DST-II(x)): {a}/{scale} vs {b}"
            );
        }
    }

    #[test]
    fn dst_iv_self_inverse() {
        // DST-IV is its own inverse (up to scaling by N/2)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let opts = FftOptions::default();
        let forward = dst_iv(&input, &opts).expect("dst_iv");
        let roundtrip = dst_iv(&forward, &opts).expect("dst_iv roundtrip");
        let scale = input.len() as f64 / 2.0;
        for (a, b) in roundtrip.iter().zip(&input) {
            assert!(
                (a / scale - b).abs() < 1e-9,
                "DST-IV roundtrip: {} vs {b}",
                a / scale
            );
        }
    }

    #[test]
    fn dst_i_known_values() {
        // DST-I of [1, 1, 1, 1] with N=4, N+1=5
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let result = dst_i(&input, &FftOptions::default()).expect("dst_i");
        // X[k] = 2 * Σ x[n] * sin(π(n+1)(k+1)/5)
        for (k, &val) in result.iter().enumerate() {
            let mut expected = 0.0;
            for (n, &x) in input.iter().enumerate() {
                expected += x * (PI * (n as f64 + 1.0) * (k as f64 + 1.0) / 5.0).sin();
            }
            assert!(
                (val - expected).abs() < 1e-9,
                "DST-I[{k}] = {val}, expected {expected}"
            );
        }
    }

    // ── Hilbert transform tests ─────────────────────────────────────

    #[test]
    fn hilbert_real_part_preserved() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let analytic = hilbert(&x, &FftOptions::default()).expect("hilbert");
        assert_eq!(analytic.len(), x.len());
        for (i, (&(re, _), &xi)) in analytic.iter().zip(x.iter()).enumerate() {
            assert!(
                (re - xi).abs() < 1e-9,
                "real part[{i}] = {re}, expected {xi}"
            );
        }
    }

    #[test]
    fn hilbert_cosine_gives_sine() {
        // Hilbert transform of cos(t) = sin(t) (for interior points, away from edges)
        let n = 64;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 2.0 * i as f64 / n as f64).cos())
            .collect();
        let analytic = hilbert(&x, &FftOptions::default()).expect("hilbert");
        // The imaginary part should approximate sin(2*2π*i/n) for interior points
        for (i, &(_, im)) in analytic.iter().enumerate().skip(4).take(n - 8) {
            let expected_sin = (2.0 * std::f64::consts::PI * 2.0 * i as f64 / n as f64).sin();
            assert!(
                (im - expected_sin).abs() < 0.1,
                "hilbert imag[{i}] = {im}, expected ~{expected_sin}",
            );
        }
    }

    #[test]
    fn hilbert_envelope() {
        // Envelope = |analytic signal| should be ~1 for a pure cosine
        let n = 64;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 3.0 * i as f64 / n as f64).cos())
            .collect();
        let analytic = hilbert(&x, &FftOptions::default()).expect("hilbert");
        // Interior envelope should be close to 1
        for (i, &(re, im)) in analytic.iter().enumerate().skip(4).take(n - 8) {
            let envelope = (re * re + im * im).sqrt();
            assert!(
                (envelope - 1.0).abs() < 0.15,
                "envelope[{i}] = {envelope}, expected ~1.0"
            );
        }
    }
}
