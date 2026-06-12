//! Timing + bit-identity harness for `peak_prominences`.
//!
//! The naive form re-scans the whole array for each peak (two full min-folds),
//! O(n·k). Precomputing prefix-min + suffix-min once makes each peak an O(1)
//! lookup (O(n+k)); the minimum is order-independent so values are bit-identical.
//! This dumps an FNV digest of the prominences (compare across the stashed naive
//! build) and times the large-(n,k) win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_peak_prominences`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::peak_prominences;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn signal(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s)).collect()
}

// Every other index is a "peak" (just indices for the prominence formula).
fn peak_idx(n: usize, k: usize) -> Vec<usize> {
    (0..k).map(|i| (i * (n.max(1))) / k.max(1)).collect()
}

fn digest(v: &[f64]) -> u64 {
    v.iter().fold(1469598103934665603u64, |h, &x| {
        (h ^ x.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, k) in &[(100usize, 10usize), (1000, 200), (5000, 5000), (20000, 1)] {
        let d = signal(n, 7);
        let pk = peak_idx(n, k);
        println!(
            "n={n} k={k} digest={:016x}",
            digest(&peak_prominences(&d, &pk))
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, k) in &[(50000usize, 2000usize), (100000, 5000), (200000, 10000)] {
        let d = signal(n, 7);
        let pk = peak_idx(n, k);
        let reps = 10;
        let _ = peak_prominences(&d, &pk);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let pr = peak_prominences(black_box(&d), black_box(&pk));
            acc += pr[pr.len() / 2];
        }
        println!(
            "n={n} k={k}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
