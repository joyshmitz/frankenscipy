//! Correctness + timing harness for DST-II (`dst_ii`) and DST-III (`dst_iii`),
//! which now route through the fast N/2-point real-FFT dct/idct via the identity
//!   DST-II(x)[k]  = DCT-II((-1)^n·x)[N-1-k],
//!   DST-III(X)[n] = 2N·(-1)^n·idct(reverse(X))[n],
//! instead of the old 4N-point complex (inverse) FFT.
//!
//! Correctness: dst_iii(dst_ii(x)) must reproduce 2N·x (roundtrip) to ~machine
//! eps, well under the 1e-9 parity tolerance. The same public APIs are timed, so
//! this harness is build-agnostic: run it, `git stash` the transforms.rs edit,
//! rebuild (old 4N path), and run again for the speedup.
//! Run: `cargo run --release -p fsci-fft --bin perf_dst`.

use std::hint::black_box;
use std::time::Instant;

use fsci_fft::{FftOptions, dst_ii, dst_iii};

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
    for &n in &[2usize, 3, 4, 5, 8, 12, 16, 17, 24, 32, 60, 64, 128, 360, 1000] {
        let x = signal(n, n as u64 * 3331 + 11);
        let fwd = dst_ii(&x, &opts).expect("dst_ii");
        let inv = dst_iii(&fwd, &opts).expect("dst_iii");
        let scale = 2.0 * n as f64;
        let err = x
            .iter()
            .zip(&inv)
            .map(|(a, b)| (a - b / scale).abs())
            .fold(0.0f64, f64::max);
        worst = worst.max(err);
        let sum: f64 = fwd.iter().sum();
        println!("n={n:>4} roundtrip_err={err:.3e} dst2_sum={sum:+.9e}");
    }
    println!("===GOLDEN_PAYLOAD_END===");
    println!("worst dst_iii(dst_ii(x))/2N roundtrip err = {worst:.3e} (parity tol 1e-9)");
    assert!(worst < 1e-9, "dst roundtrip exceeds parity tolerance");

    for &n in &[512usize, 1000, 2048, 4096, 10000, 16384] {
        let x = signal(n, 99);
        let reps = 500;
        let _ = dst_ii(&x, &opts).unwrap();
        macro_rules! time {
            ($name:expr, $f:path) => {{
                let t0 = Instant::now();
                let mut acc = 0.0;
                for _ in 0..reps {
                    let r = $f(black_box(&x), &opts).unwrap();
                    acc += r[r.len() / 2];
                }
                println!("{:<8} n={n:>6} {:>10.3?}/call (acc={acc:.3})", $name, t0.elapsed() / reps);
            }};
        }
        time!("dst_ii", dst_ii);
        time!("dst_iii", dst_iii);
    }
}
