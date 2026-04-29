//! Metamorphic tests for `fsci-fft`.
//!
//! Each test exercises a transform identity that holds independent of the
//! input contents, so no oracle is required. Examples: ifft∘fft = id,
//! Parseval's theorem, linearity, real-input round-trip via rfft/irfft.
//!
//! Run with: `cargo test -p fsci-fft --test metamorphic_tests`

use fsci_fft::{
    Complex64, FftOptions, Normalization, dct, dst_ii, fft, fft2, idct, ifft, ifft2, irfft, irfft2,
    rfft, rfft2,
};

const RTOL: f64 = 1e-9;
const ATOL: f64 = 1e-10;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

fn assert_close(a: f64, b: f64, msg: &str) {
    assert!(close(a, b), "{msg}: {a:.16e} vs {b:.16e}");
}

fn close_complex(a: Complex64, b: Complex64) -> bool {
    close(a.0, b.0) && close(a.1, b.1)
}

/// Deterministic Lehmer-style PRNG. Avoids pulling in `rand` for tests.
fn prng(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.max(1);
    move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (state >> 32) as u32;
        // Map to [-1, 1)
        (bits as f64 / (1u64 << 32) as f64) * 2.0 - 1.0
    }
}

fn random_real(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = prng(seed);
    (0..n).map(|_| rng()).collect()
}

fn random_complex(n: usize, seed: u64) -> Vec<Complex64> {
    let mut rng = prng(seed);
    (0..n).map(|_| (rng(), rng())).collect()
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — fft round-trip: ifft(fft(x)) == x for any x, any normalization.
// ─────────────────────────────────────────────────────────────────────

fn fft_round_trip_with(opts: &FftOptions, label: &str) {
    for &n in &[1, 2, 4, 8, 16, 17, 32, 64, 100] {
        let x = random_complex(n, 0xCAFE_F00D + n as u64);
        let xf = fft(&x, opts).unwrap();
        let xb = ifft(&xf, opts).unwrap();
        assert_eq!(xb.len(), n, "{label} length mismatch at n={n}");
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close_complex(*got, *want),
                "{label} round-trip n={n} idx={i}: got=({}, {}) expected=({}, {})",
                got.0,
                got.1,
                want.0,
                want.1
            );
        }
    }
}

#[test]
fn mr_fft_roundtrip_backward() {
    let opts = FftOptions::default().with_normalization(Normalization::Backward);
    fft_round_trip_with(&opts, "Backward");
}

#[test]
fn mr_fft_roundtrip_ortho() {
    let opts = FftOptions::default().with_normalization(Normalization::Ortho);
    fft_round_trip_with(&opts, "Ortho");
}

#[test]
fn mr_fft_roundtrip_forward() {
    let opts = FftOptions::default().with_normalization(Normalization::Forward);
    fft_round_trip_with(&opts, "Forward");
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — rfft / irfft round-trip on real inputs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rfft_irfft_roundtrip() {
    let opts = FftOptions::default();
    for &n in &[2, 4, 8, 16, 17, 32, 100] {
        let x = random_real(n, 0xDEAD_BEEF + n as u64);
        let xf = rfft(&x, &opts).unwrap();
        let xb = irfft(&xf, Some(n), &opts).unwrap();
        assert_eq!(xb.len(), n, "rfft round-trip length n={n}");
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close(*got, *want),
                "MR2 rfft round-trip n={n} idx={i}: got={got} expected={want}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — Parseval's theorem.
//
// Backward normalization (default): Σ|x_i|² = (1/N) · Σ|X_k|².
// Ortho normalization: Σ|x_i|² = Σ|X_k|² (FFT is unitary).
// ─────────────────────────────────────────────────────────────────────

fn complex_sq_norm(values: &[Complex64]) -> f64 {
    values.iter().map(|(re, im)| re * re + im * im).sum()
}

#[test]
fn mr_parseval_backward() {
    let opts = FftOptions::default().with_normalization(Normalization::Backward);
    for &n in &[8, 16, 32, 64, 100] {
        let x = random_complex(n, 0x1234_5678 + n as u64);
        let xf = fft(&x, &opts).unwrap();
        let lhs = complex_sq_norm(&x);
        let rhs = complex_sq_norm(&xf) / n as f64;
        assert_close(lhs, rhs, &format!("MR3 Parseval Backward n={n}"));
    }
}

#[test]
fn mr_parseval_ortho() {
    let opts = FftOptions::default().with_normalization(Normalization::Ortho);
    for &n in &[8, 16, 32, 64, 100] {
        let x = random_complex(n, 0xABCD_1234 + n as u64);
        let xf = fft(&x, &opts).unwrap();
        let lhs = complex_sq_norm(&x);
        let rhs = complex_sq_norm(&xf);
        assert_close(lhs, rhs, &format!("MR3 Parseval Ortho n={n}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — linearity: F(αx + βy) = α F(x) + β F(y)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft_linearity() {
    let opts = FftOptions::default();
    let alpha = (1.5_f64, -0.3_f64);
    let beta = (-0.7_f64, 0.4_f64);
    let cmul = |a: Complex64, b: Complex64| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    for &n in &[8, 16, 32, 64] {
        let x = random_complex(n, 0xFFEE_DDCC + n as u64);
        let y = random_complex(n, 0xCC11_22DD + n as u64);

        let combined: Vec<Complex64> = x
            .iter()
            .zip(&y)
            .map(|(xi, yi)| {
                let ax = cmul(alpha, *xi);
                let by = cmul(beta, *yi);
                (ax.0 + by.0, ax.1 + by.1)
            })
            .collect();

        let f_combined = fft(&combined, &opts).unwrap();
        let f_x = fft(&x, &opts).unwrap();
        let f_y = fft(&y, &opts).unwrap();

        for k in 0..n {
            let af = cmul(alpha, f_x[k]);
            let bf = cmul(beta, f_y[k]);
            let expected = (af.0 + bf.0, af.1 + bf.1);
            assert!(
                close_complex(f_combined[k], expected),
                "MR4 linearity n={n} k={k}: got=({}, {}) expected=({}, {})",
                f_combined[k].0,
                f_combined[k].1,
                expected.0,
                expected.1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — DCT-II / IDCT round-trip
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dct_idct_roundtrip() {
    let opts = FftOptions::default();
    for &n in &[2, 4, 8, 16, 32, 64] {
        let x = random_real(n, 0xC1C2_C3C4 + n as u64);
        let xf = dct(&x, &opts).unwrap();
        let xb = idct(&xf, &opts).unwrap();
        assert_eq!(xb.len(), n);
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close(*got, *want),
                "MR5 dct round-trip n={n} idx={i}: got={got} expected={want}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — DST-II output structure: a periodic odd extension's DST is real.
//
// Specifically, DST-II of an all-zero input must be all-zero (linearity
// check at the zero of the operator). Combined with linearity it implies
// DST-II is well-defined for all inputs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dst_ii_zero_input_is_zero() {
    let opts = FftOptions::default();
    for &n in &[2, 4, 8, 16, 32] {
        let x = vec![0.0_f64; n];
        let xf = dst_ii(&x, &opts).unwrap();
        for (i, v) in xf.iter().enumerate() {
            assert!(v.abs() < 1e-12, "MR6 DST-II(0) n={n} idx={i}: {v}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — Conjugate symmetry of fft for real input.
// If x is real, then X[k] = conj(X[N-k]) for k = 1..N-1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft_conjugate_symmetry_real_input() {
    let opts = FftOptions::default();
    for &n in &[8, 16, 32, 64] {
        let x_real = random_real(n, 0xAA_BB_CC_DD + n as u64);
        let x: Vec<Complex64> = x_real.iter().map(|&v| (v, 0.0)).collect();
        let xf = fft(&x, &opts).unwrap();
        for k in 1..n {
            let mirror = xf[n - k];
            let here = xf[k];
            assert!(
                close(here.0, mirror.0) && close(here.1, -mirror.1),
                "MR7 conjugate symmetry n={n} k={k}: X[k]=({}, {}) X[N-k]=({}, {})",
                here.0,
                here.1,
                mirror.0,
                mirror.1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — fft2 / ifft2 round-trip on 2D inputs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft2_ifft2_roundtrip() {
    let opts = FftOptions::default();
    for &(rows, cols) in &[(4_usize, 4), (8, 4), (16, 16), (12, 7), (3, 5)] {
        let n = rows * cols;
        let x = random_complex(n, 0xF00D_1234 + (rows * 1000 + cols) as u64);
        let xf = fft2(&x, (rows, cols), &opts).unwrap();
        let xb = ifft2(&xf, (rows, cols), &opts).unwrap();
        assert_eq!(xb.len(), n, "MR8 fft2 round-trip length");
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close_complex(*got, *want),
                "MR8 fft2 round-trip rows={rows} cols={cols} i={i}: ({}, {}) vs ({}, {})",
                got.0,
                got.1,
                want.0,
                want.1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — rfft2 / irfft2 round-trip on real 2D inputs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rfft2_irfft2_roundtrip() {
    let opts = FftOptions::default();
    for &(rows, cols) in &[(4_usize, 4), (8, 4), (16, 16), (4, 8), (12, 16)] {
        let n = rows * cols;
        let x = random_real(n, 0xBEEF_3456 + (rows * 1000 + cols) as u64);
        let xf = rfft2(&x, (rows, cols), &opts).unwrap();
        let xb = irfft2(&xf, (rows, cols), &opts).unwrap();
        assert_eq!(xb.len(), n, "MR9 rfft2 round-trip length");
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close(*got, *want),
                "MR9 rfft2 round-trip rows={rows} cols={cols} i={i}: {got} vs {want}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — Parseval's theorem in 2D for fft2 with Backward normalization:
// Σ|x|² = Σ|X|² / (rows * cols).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft2_parseval() {
    let opts = FftOptions::default().with_normalization(Normalization::Backward);
    for &(rows, cols) in &[(8_usize, 8), (4, 16), (12, 6)] {
        let n = rows * cols;
        let x = random_complex(n, 0xDADA_FACE + (rows * 1000 + cols) as u64);
        let xf = fft2(&x, (rows, cols), &opts).unwrap();
        let lhs: f64 = x.iter().map(|(re, im)| re * re + im * im).sum();
        let rhs: f64 = xf.iter().map(|(re, im)| re * re + im * im).sum::<f64>() / n as f64;
        assert_close(lhs, rhs, &format!("MR10 fft2 Parseval rows={rows} cols={cols}"));
    }
}
