#![no_main]

use arbitrary::Arbitrary;
use fsci_signal::{ZpkCoeffs, freqz_zpk, freqz_zpk_with_whole};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-rva8k]:
// freqz_zpk has scipy diff coverage and unit anchors but no
// libfuzzer harness. Drive sanitized random ZpkCoeffs (zeros,
// poles, gain) + n_freqs and assert the call:
//   1. Never panics.
//   2. Returns Ok with vectors of length exactly n_freqs.
//   3. h_mag is non-negative whenever finite.
//   4. h_phase is in [-π, π] whenever finite (atan2 contract).
//   5. Half-circle and whole-circle calls both succeed for the
//      same input.

const BOUND: f64 = 1.0e6;
const MAX_ROOTS: usize = 16;
const MAX_N_FREQS: usize = 256;
const MIN_N_FREQS: usize = 1;

#[derive(Debug, Arbitrary)]
struct ZpkInput {
    zeros_re: Vec<f64>,
    zeros_im: Vec<f64>,
    poles_re: Vec<f64>,
    poles_im: Vec<f64>,
    gain: f64,
    n_freqs: u16,
}

fn sanitize(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-BOUND, BOUND)
    } else {
        0.0
    }
}

fn pad_or_truncate(re: Vec<f64>, im: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = re.len().min(im.len()).min(MAX_ROOTS);
    let r: Vec<f64> = re.into_iter().take(n).map(sanitize).collect();
    let i: Vec<f64> = im.into_iter().take(n).map(sanitize).collect();
    (r, i)
}

fuzz_target!(|input: ZpkInput| {
    let (zeros_re, zeros_im) = pad_or_truncate(input.zeros_re, input.zeros_im);
    let (poles_re, poles_im) = pad_or_truncate(input.poles_re, input.poles_im);
    let gain = sanitize(input.gain);

    let n_freqs = (input.n_freqs as usize).clamp(MIN_N_FREQS, MAX_N_FREQS);

    let zpk = ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain,
    };

    let result = match freqz_zpk(&zpk, Some(n_freqs)) {
        Ok(r) => r,
        Err(e) => panic!("freqz_zpk rejected sanitized input: {e}"),
    };

    assert_eq!(
        result.w.len(),
        n_freqs,
        "freqz_zpk: w must have length n_freqs"
    );
    assert_eq!(
        result.h_mag.len(),
        n_freqs,
        "freqz_zpk: h_mag must have length n_freqs"
    );
    assert_eq!(
        result.h_phase.len(),
        n_freqs,
        "freqz_zpk: h_phase must have length n_freqs"
    );

    for &m in &result.h_mag {
        if m.is_finite() {
            assert!(
                m >= 0.0,
                "freqz_zpk: finite h_mag must be non-negative, got {m}"
            );
        }
    }

    for &p in &result.h_phase {
        if p.is_finite() {
            assert!(
                p.abs() <= std::f64::consts::PI + 1e-12,
                "freqz_zpk: h_phase must lie in [-π, π], got {p}"
            );
        }
    }

    // Whole-circle path must succeed on the same input.
    let _whole = freqz_zpk_with_whole(&zpk, Some(n_freqs), true)
        .expect("freqz_zpk_with_whole must accept same sanitized input");
});
