//! Timing + bit-identity harness for `correlate_1d` (convolve_1d inherits it).
//!
//! Each output index is an independent multiply-accumulate over the kernel; the
//! outputs fan out across threads in contiguous chunks (one spawn-set), each
//! keeping its own summation order — so values are bit-identical to the serial
//! loop. This dumps an FNV digest of the output (compare across the stashed
//! serial build) and times the large convolution win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_correlate_1d`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::{convolve_1d, correlate_1d};

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn vec_of(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s)).collect()
}

fn digest(v: &[f64]) -> u64 {
    v.iter()
        .fold(1469598103934665603u64, |h, &x| (h ^ x.to_bits()).wrapping_mul(1099511628211))
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, m) in &[(100usize, 1usize), (1000, 8), (5000, 64), (20000, 300)] {
        let a = vec_of(n, 7);
        let v = vec_of(m, 99);
        println!(
            "n={n} m={m} corr={:016x} conv={:016x}",
            digest(&correlate_1d(&a, &v)),
            digest(&convolve_1d(&a, &v))
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, m) in &[(50000usize, 256usize), (100000, 512), (200000, 1024)] {
        let a = vec_of(n, 7);
        let v = vec_of(m, 99);
        let reps = 5;
        let _ = correlate_1d(&a, &v);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let r = correlate_1d(black_box(&a), black_box(&v));
            acc += r[r.len() / 2];
        }
        println!("n={n} m={m}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
