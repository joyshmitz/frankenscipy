//! Same-process timing + tolerance harness for `weightedtau`.
//!
//! weightedtau averages two one-side weighted Kendall taus; each was an O(n^2) all-pairs
//! sum and is now an O(n log n) Fenwick-tree computation. Dumps the statistic (compare
//! across the stashed O(n^2) build for tolerance-parity) and times the call.
//! Run: `cargo run -p fsci-stats --bin perf_weightedtau`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::weightedtau;

fn data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut s = 0x9e3779b97f4a7c15u64;
    let mut rng = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let x: Vec<f64> = (0..n).map(|_| rng() * 100.0).collect();
    let y: Vec<f64> = (0..n).map(|i| 0.7 * x[i] + rng() * 30.0).collect();
    (x, y)
}

fn main() {
    let sizes = [2000usize, 5000, 10000];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &sizes {
        let (x, y) = data(n);
        println!("n={n} tau={:.15e}", weightedtau(&x, &y));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &sizes {
        let (x, y) = data(n);
        let reps = 5;
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += weightedtau(black_box(&x), black_box(&y));
        }
        let dt = t0.elapsed();
        println!("n={n:>6}  {:>10.3?}/call  (acc={acc:.6})", dt / reps);
    }
}
