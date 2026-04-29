#![no_main]

//! Robustness fuzzing for `fsci_signal::filtfilt`. Drives random
//! small filter coefficients (b, a) and arbitrary signals; checks
//! shape preservation and finite output.
//!
//! Bead: `frankenscipy-u7z5`.

use arbitrary::Arbitrary;
use fsci_signal::filtfilt;
use libfuzzer_sys::fuzz_target;

const MIN_LEN: usize = 8;
const MAX_LEN: usize = 256;
const MAX_COEFF: usize = 6;

#[derive(Debug, Arbitrary)]
struct FiltfiltInput {
    sig_len: u8,
    n_b: u8,
    n_a: u8,
    b_raw: Vec<f64>,
    a_raw: Vec<f64>,
    signal_raw: Vec<f64>,
}

fn sanitize(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-10.0, 10.0)
    } else {
        0.0
    }
}

fuzz_target!(|input: FiltfiltInput| {
    let sig_len = MIN_LEN + (input.sig_len as usize % (MAX_LEN - MIN_LEN + 1));
    let n_b = 1 + (input.n_b as usize % MAX_COEFF);
    let n_a = 1 + (input.n_a as usize % MAX_COEFF);

    // Build coefficients; ensure a[0] is non-zero so the filter is well-defined.
    let mut b: Vec<f64> = (0..n_b)
        .map(|i| sanitize(input.b_raw.get(i).copied().unwrap_or(0.0)))
        .collect();
    let mut a: Vec<f64> = (0..n_a)
        .map(|i| sanitize(input.a_raw.get(i).copied().unwrap_or(0.0)))
        .collect();

    if b.iter().all(|v| v.abs() < 1e-12) {
        b[0] = 1.0;
    }
    if a[0].abs() < 1e-9 {
        a[0] = 1.0;
    }
    if a.iter().all(|v| v.abs() < 1e-12) {
        a[0] = 1.0;
    }

    let signal: Vec<f64> = (0..sig_len)
        .map(|i| sanitize(input.signal_raw.get(i).copied().unwrap_or(0.0)))
        .collect();

    if let Ok(out) = filtfilt(&b, &a, &signal) {
        assert_eq!(
            out.len(),
            signal.len(),
            "filtfilt length mismatch: out={} signal={}",
            out.len(),
            signal.len()
        );
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "filtfilt produced non-finite at i={i}: {v} (sig_len={sig_len}, n_b={n_b}, n_a={n_a})"
            );
        }
    }
});
