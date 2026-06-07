//! Timing + tolerance-parity harness for `medcouple`.
//!
//! The naive medcouple materialized ≈n²/4 kernel values and median-sorted them
//! (O(n² log n)). The kernel matrix is sorted along both axes and bounded in
//! [-1,1], so the median is now found by bisection + O(n) saddleback counting
//! with nothing materialized — O(n log n). This dumps the value (compare vs the
//! stashed naive build: must agree to ~1e-12) and times the large-n win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_medcouple`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::medcouple;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

// Skewed, all-distinct data so a large fraction of pairs straddle the median.
fn arr(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| -(1.0 - lcg(&mut s)).ln()).collect()
}

fn main() {
    println!("===PARITY_PAYLOAD_BEGIN===");
    for &n in &[31usize, 255, 256, 1000, 2049] {
        let data = arr(n, 7);
        println!("n={n} medcouple={:.17e}", medcouple(&data));
    }
    println!("===PARITY_PAYLOAD_END===");

    for &n in &[1500usize, 3000, 6000] {
        let data = arr(n, 7);
        let reps = 5;
        let _ = medcouple(&data);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += medcouple(black_box(&data));
        }
        println!("n={n}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
