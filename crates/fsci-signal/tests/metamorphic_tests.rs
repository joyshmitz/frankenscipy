//! Metamorphic tests for `fsci-signal`.
//!
//! Convolution commutativity, filtfilt zero-phase, lfilter impulse
//! identity, hilbert real-part recovery, window normalisation.
//!
//! Run with: `cargo test -p fsci-signal --test metamorphic_tests`

use fsci_signal::{
    ConvolveMode, FindPeaksOptions, blackman, convolve, correlate, fftconvolve, filtfilt,
    find_peaks, hamming, hann, hilbert, kaiser, lfilter, sosfilt, tf2sos,
};

const ATOL: f64 = 1e-9;
const RTOL: f64 = 1e-7;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

fn assert_close(a: f64, b: f64, msg: &str) {
    assert!(
        close(a, b),
        "{msg}: actual={a:.16e} expected={b:.16e} diff={:.3e}",
        (a - b).abs()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — convolve(a, b) = convolve(b, a) (commutativity).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_commutative() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![0.5, -1.0, 0.25, 2.0];
    let ab = convolve(&a, &b, ConvolveMode::Full).unwrap();
    let ba = convolve(&b, &a, ConvolveMode::Full).unwrap();
    assert_eq!(ab.len(), ba.len(), "MR1 length mismatch");
    for (i, (l, r)) in ab.iter().zip(&ba).enumerate() {
        assert_close(*l, *r, &format!("MR1 commutativity at i={i}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — convolve and fftconvolve (Full mode) agree element-wise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_matches_fftconvolve() {
    let a = vec![1.0, -0.5, 0.25, 0.7, 1.4, -0.3, 0.0, 1.1];
    let b = vec![0.3, 0.5, -0.2, 0.1];
    let direct = convolve(&a, &b, ConvolveMode::Full).unwrap();
    let fft = fftconvolve(&a, &b, ConvolveMode::Full).unwrap();
    assert_eq!(direct.len(), fft.len(), "MR2 length mismatch");
    for (i, (l, r)) in direct.iter().zip(&fft).enumerate() {
        assert!(
            (l - r).abs() <= 1e-8 * l.abs().max(r.abs()).max(1.0) + 1e-10,
            "MR2 direct vs fft at i={i}: {l} vs {r}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — convolve with the unit impulse [1] returns the input unchanged.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_with_unit_impulse_is_identity() {
    let x = vec![3.5, -1.2, 0.8, 4.1, -2.3];
    let out = convolve(&x, &[1.0], ConvolveMode::Full).unwrap();
    assert_eq!(out.len(), x.len(), "MR3 length");
    for (i, (got, want)) in out.iter().zip(&x).enumerate() {
        assert_close(*got, *want, &format!("MR3 impulse at i={i}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — correlate(a, b) = convolve(a, reverse(b)) (Full mode).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_correlate_equals_convolve_reversed() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![0.5, -1.0, 0.25];
    let corr = correlate(&a, &b, ConvolveMode::Full).unwrap();
    let mut b_rev = b.clone();
    b_rev.reverse();
    let conv = convolve(&a, &b_rev, ConvolveMode::Full).unwrap();
    assert_eq!(corr.len(), conv.len(), "MR4 length mismatch");
    for (i, (l, r)) in corr.iter().zip(&conv).enumerate() {
        assert_close(*l, *r, &format!("MR4 at i={i}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — lfilter with b=[1.0], a=[1.0] is identity.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_lfilter_passthrough_is_identity() {
    let x = vec![1.0, 2.0, -3.0, 0.7, 4.2, -1.1, 5.5];
    let out = lfilter(&[1.0], &[1.0], &x, None).unwrap();
    assert_eq!(out.len(), x.len(), "MR5 length");
    for (i, (got, want)) in out.iter().zip(&x).enumerate() {
        assert_close(*got, *want, &format!("MR5 at i={i}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — filtfilt with the same coefficients applied twice converges to
// a zero-phase result. As a structural check, filtfilt of a constant
// signal returns the same constant.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_filtfilt_preserves_constant() {
    // A simple low-pass: y[n] = 0.5 x[n] + 0.5 x[n-1].
    let b = vec![0.5, 0.5];
    let a = vec![1.0];
    let n = 32;
    let x = vec![3.7_f64; n];
    let out = filtfilt(&b, &a, &x).unwrap();
    assert_eq!(out.len(), n);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            close(v, 3.7),
            "MR6 filtfilt changed constant at i={i}: {v}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — hilbert real part recovers the input.
//
// scipy.signal.hilbert returns the analytic signal x_a = x + i·H[x].
// The real component must equal the input (up to round-off).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hilbert_real_part_recovers_input() {
    // Pick lengths where the harness exercises typical FFT sizes.
    for &n in &[16_usize, 32, 64, 100] {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64).sin()).collect();
        let analytic = hilbert(&x).unwrap();
        assert_eq!(analytic.len(), n, "MR7 length");
        for (i, ((re, _im), &xi)) in analytic.iter().zip(&x).enumerate() {
            assert!(
                close(*re, xi),
                "MR7 hilbert real part at n={n} i={i}: {re} vs {xi}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — window functions are non-negative on the standard [0, 1]
// support and symmetric.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_windows_nonneg_and_symmetric() {
    for n in [8usize, 16, 33, 64] {
        let windows: Vec<(&str, Vec<f64>)> = vec![
            ("hann", hann(n)),
            ("hamming", hamming(n)),
            ("blackman", blackman(n)),
            ("kaiser-6", kaiser(n, 6.0)),
        ];
        for (name, w) in windows {
            assert_eq!(w.len(), n, "{name} length n={n}");
            // Non-negative: hann/hamming/blackman are ≥0; kaiser(β>0) too.
            for (i, &v) in w.iter().enumerate() {
                assert!(v >= -1e-12, "MR8 {name} negative at i={i}: {v}");
                assert!(v <= 1.0 + 1e-12, "MR8 {name} > 1 at i={i}: {v}");
            }
            // Symmetric: w[k] == w[n-1-k].
            for k in 0..n / 2 {
                let mirror = w[n - 1 - k];
                assert!(
                    (w[k] - mirror).abs() < 1e-12,
                    "MR8 {name} not symmetric at k={k}: {} vs {mirror}",
                    w[k]
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — convolve linearity in the first argument:
//   convolve(α x + β y, h) = α convolve(x, h) + β convolve(y, h)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_linear_in_first_arg() {
    let x = vec![1.0, 0.5, -0.3, 0.8, 1.2];
    let y = vec![0.5, 1.5, -0.2, 0.9, 0.4];
    let h = vec![0.3, -0.1, 0.4];
    let alpha = 1.7_f64;
    let beta = -0.5_f64;
    let combined: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(xi, yi)| alpha * xi + beta * yi)
        .collect();
    let combined_c = convolve(&combined, &h, ConvolveMode::Full).unwrap();
    let xc = convolve(&x, &h, ConvolveMode::Full).unwrap();
    let yc = convolve(&y, &h, ConvolveMode::Full).unwrap();
    assert_eq!(combined_c.len(), xc.len());
    for (i, ((cv, xv), yv)) in combined_c.iter().zip(&xc).zip(&yc).enumerate() {
        let expected = alpha * xv + beta * yv;
        assert_close(*cv, expected, &format!("MR9 linearity at i={i}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — Hilbert envelope identity: |x_a|² = x² + H[x]², the
// definition of analytic-signal magnitude squared.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hilbert_envelope_squared_identity() {
    for &n in &[16_usize, 32, 64, 100] {
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 / n as f64 * std::f64::consts::TAU).sin())
            .collect();
        let analytic = hilbert(&x).unwrap();
        for (i, ((re, im), &xi)) in analytic.iter().zip(&x).enumerate() {
            // Real part should equal input (already covered by MR7).
            // Envelope: |x_a|² = re² + im² should equal xi² + im².
            let env_sq = re * re + im * im;
            let expected = xi * xi + im * im;
            assert!(
                (env_sq - expected).abs() < 1e-9 * env_sq.max(1.0),
                "MR10 hilbert envelope at n={n} i={i}: got {env_sq}, expected {expected}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — Hilbert applied to a constant signal returns x_a.real = const
// and x_a.imag ≈ 0 (since the Hilbert transform of a constant is 0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hilbert_of_constant() {
    let n = 64_usize;
    let value = 3.5_f64;
    let x = vec![value; n];
    let analytic = hilbert(&x).unwrap();
    for (i, (re, im)) in analytic.iter().enumerate() {
        assert!(
            (re - value).abs() < 1e-9,
            "MR11 hilbert const real part at i={i}: {re} vs {value}"
        );
        assert!(
            im.abs() < 1e-9,
            "MR11 hilbert const imag part at i={i}: {im}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — find_peaks on a strictly monotone signal returns no peaks.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_find_peaks_monotone_has_no_peaks() {
    // Strictly increasing: no peak at any interior position.
    let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let result = find_peaks(&x, FindPeaksOptions::default());
    assert!(
        result.peaks.is_empty(),
        "MR12 monotone signal should have no peaks, got {:?}",
        result.peaks
    );
    // Strictly decreasing: also no peaks.
    let y: Vec<f64> = (0..50).rev().map(|i| i as f64 * 0.1).collect();
    let result = find_peaks(&y, FindPeaksOptions::default());
    assert!(
        result.peaks.is_empty(),
        "MR12 monotone-decreasing signal should have no peaks, got {:?}",
        result.peaks
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — find_peaks on a discrete tent [0, 1, 0] returns index 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_find_peaks_tent() {
    let x = vec![0.0, 1.0, 0.0];
    let result = find_peaks(&x, FindPeaksOptions::default());
    assert_eq!(result.peaks, vec![1], "MR13 expected single peak at idx 1");
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — sosfilt and lfilter produce the same output for the same
// transfer function (when the SOS is constructed via tf2sos).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sosfilt_matches_lfilter() {
    // Simple low-pass: y[n] = (x[n] + 0.5 x[n-1]) / 1.5
    let b = vec![1.0_f64, 0.5];
    let a = vec![1.0_f64, 0.0];
    let x: Vec<f64> = (0..32).map(|i| (i as f64 / 4.0).sin()).collect();

    let y_lf = lfilter(&b, &a, &x, None).unwrap();
    let sos = tf2sos(&b, &a).unwrap();
    let y_sos = sosfilt(&sos, &x).unwrap();
    assert_eq!(y_lf.len(), y_sos.len(), "MR14 length mismatch");
    for (i, (l, s)) in y_lf.iter().zip(&y_sos).enumerate() {
        assert!(
            close(*l, *s),
            "MR14 lfilter vs sosfilt at i={i}: {l} vs {s}"
        );
    }
}
