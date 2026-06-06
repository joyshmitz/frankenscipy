//! Correctness + timing harness for DCT-II (`dct`), which now uses a single
//! N-point FFT (Makhoul reorder) instead of a 2N-point complex FFT of the
//! mirror-symmetric extension.
//!
//! Correctness is checked against a naive O(n²) DCT-II ground truth (max abs
//! error must be ~machine eps·n, well under the 1e-9 scipy-parity tolerance).
//! The same `dct` public API is timed, so this harness is build-agnostic: run
//! it, `git stash` the transforms.rs edit, rebuild (the old 2N path), and run
//! again to read the speedup.
//! Run: `cargo run --release -p fsci-fft --bin perf_dct`.

use std::hint::black_box;
use std::time::Instant;

use fsci_fft::{FftOptions, dct};

fn naive_dct2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    for (k, ok) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (nn, &xn) in x.iter().enumerate() {
            acc += xn
                * (std::f64::consts::PI * k as f64 * (2.0 * nn as f64 + 1.0) / (2.0 * n as f64))
                    .cos();
        }
        *ok = 2.0 * acc;
    }
    out
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn signal(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s)).collect()
}

fn main() {
    let opts = FftOptions::default();

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    let mut worst = 0.0f64;
    for &n in &[2usize, 3, 4, 5, 8, 12, 16, 17, 24, 32, 60, 64, 100, 128, 360] {
        let x = signal(n, n as u64 * 1231 + 7);
        let got = dct(&x, &opts).expect("dct");
        let want = naive_dct2(&x);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        worst = worst.max(err);
        let sum: f64 = got.iter().sum();
        println!("n={n:>4} maxerr={err:.3e} sum={sum:+.9e}");
    }
    println!("===GOLDEN_PAYLOAD_END===");
    println!("worst maxerr vs naive DCT-II = {worst:.3e} (parity tol 1e-9)");
    assert!(worst < 1e-9, "dct exceeds parity tolerance");

    for &n in &[512usize, 1000, 2048, 4096, 10000, 16384] {
        let x = signal(n, 99);
        let reps = 500;
        let _ = dct(&x, &opts).unwrap();
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let r = dct(black_box(&x), &opts).unwrap();
            acc += r[r.len() / 2];
        }
        let dt = t0.elapsed() / reps;
        println!("n={n:>6} {dt:>10.3?}/call (acc={acc:.3})");
    }
}
