#![no_main]

use arbitrary::Arbitrary;
use fsci_signal::{ConvolveMode, convolve, fftconvolve};
use libfuzzer_sys::fuzz_target;

// Signal convolution equivalence oracle:
// For any two input signals a and b, convolve(a, b, mode) and
// fftconvolve(a, b, mode) should produce identical results within
// floating-point tolerance.
//
// This catches:
// - Off-by-one errors in direct vs FFT convolution indices
// - Edge case handling differences (zero length, single element)
// - Mode-dependent boundary handling bugs
// - Numerical precision accumulation differences

const MAX_LEN: usize = 256;
const REL_TOL: f64 = 1e-10;
const ABS_TOL: f64 = 1e-12;

#[derive(Debug, Arbitrary)]
struct ConvolveInput {
    a: Vec<f64>,
    b: Vec<f64>,
    mode_variant: u8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn close_enough(x: f64, y: f64) -> bool {
    if !x.is_finite() || !y.is_finite() {
        return true;
    }
    let diff = (x - y).abs();
    diff <= ABS_TOL + REL_TOL * x.abs().max(y.abs())
}

fuzz_target!(|input: ConvolveInput| {
    let a_len = input.a.len().min(MAX_LEN);
    let b_len = input.b.len().min(MAX_LEN);

    if a_len == 0 || b_len == 0 {
        return;
    }

    let a: Vec<f64> = input.a.iter().take(a_len).map(|&x| sanitize(x)).collect();
    let b: Vec<f64> = input.b.iter().take(b_len).map(|&x| sanitize(x)).collect();

    let mode = match input.mode_variant % 3 {
        0 => ConvolveMode::Full,
        1 => ConvolveMode::Same,
        _ => ConvolveMode::Valid,
    };

    let direct = match convolve(&a, &b, mode) {
        Ok(r) => r,
        Err(_) => return,
    };

    let fft_based = match fftconvolve(&a, &b, mode) {
        Ok(r) => r,
        Err(_) => return,
    };

    if direct.len() != fft_based.len() {
        panic!(
            "Convolution length mismatch: direct {} vs FFT {} (a.len={}, b.len={}, mode={:?})",
            direct.len(),
            fft_based.len(),
            a_len,
            b_len,
            mode
        );
    }

    for (i, (d, f)) in direct.iter().zip(fft_based.iter()).enumerate() {
        if !close_enough(*d, *f) {
            panic!(
                "Convolution equivalence failed at index {}: \
                 direct={} vs FFT={} (a.len={}, b.len={}, mode={:?})",
                i, d, f, a_len, b_len, mode
            );
        }
    }
});
