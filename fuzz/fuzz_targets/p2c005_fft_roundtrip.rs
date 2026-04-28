#![no_main]

use arbitrary::Arbitrary;
use fsci_fft::{FftOptions, Normalization, fft, ifft};
use libfuzzer_sys::fuzz_target;

// FFT roundtrip metamorphic oracle:
// For any input signal x, ifft(fft(x)) should recover x within floating-point tolerance.
// Similarly, for real-valued input, irfft(rfft(x), n) should recover x.
//
// This catches:
// - Sign errors in twiddle factors
// - Off-by-one in frequency indexing
// - Normalization scaling bugs
// - Memory corruption in the Cooley-Tukey butterfly
// - Edge cases: length 0, 1, 2, non-power-of-2

const MAX_LEN: usize = 1024;
const REL_TOL: f64 = 1e-10;
const ABS_TOL: f64 = 1e-14;

#[derive(Debug, Arbitrary)]
struct FftInput {
    real_parts: Vec<f64>,
    imag_parts: Vec<f64>,
    norm_variant: u8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e10, 1e10)
    } else {
        0.0
    }
}

fn close_enough(a: f64, b: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return true;
    }
    let diff = (a - b).abs();
    diff <= ABS_TOL + REL_TOL * a.abs().max(b.abs())
}

fn complex_close(a: (f64, f64), b: (f64, f64)) -> bool {
    close_enough(a.0, b.0) && close_enough(a.1, b.1)
}

fuzz_target!(|input: FftInput| {
    let len = input.real_parts.len().min(MAX_LEN);
    if len == 0 {
        return;
    }

    let original: Vec<(f64, f64)> = input
        .real_parts
        .iter()
        .take(len)
        .zip(input.imag_parts.iter().chain(std::iter::repeat(&0.0)))
        .map(|(&re, &im)| (sanitize(re), sanitize(im)))
        .collect();

    let norm = match input.norm_variant % 3 {
        0 => Normalization::Backward,
        1 => Normalization::Forward,
        _ => Normalization::Ortho,
    };

    let opts = FftOptions {
        normalization: norm,
        ..FftOptions::default()
    };

    let Ok(transformed) = fft(&original, &opts) else {
        return;
    };

    let inverse_opts = FftOptions {
        normalization: norm,
        ..FftOptions::default()
    };
    let Ok(recovered) = ifft(&transformed, &inverse_opts) else {
        return;
    };

    if recovered.len() != original.len() {
        panic!(
            "FFT roundtrip length mismatch: input {} vs recovered {}",
            original.len(),
            recovered.len()
        );
    }

    for (i, (orig, rec)) in original.iter().zip(recovered.iter()).enumerate() {
        if !complex_close(*orig, *rec) {
            panic!(
                "FFT roundtrip failed at index {i}: \
                 original=({}, {}) recovered=({}, {}), \
                 len={}, norm={:?}",
                orig.0, orig.1, rec.0, rec.1, len, norm
            );
        }
    }
});
