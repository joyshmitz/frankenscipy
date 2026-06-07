//! Timing + bit-identity harness for `acf` (autocorrelation function).
//!
//! Each lag's autocovariance is an independent O(n) dot of the centered series
//! against its shifted self; the lags fan out across threads in contiguous
//! chunks (one spawn-set), each lag's inner sum kept sequential — so values are
//! bit-identical to the serial map. This dumps an FNV digest of the acf vector
//! (compare across the stashed serial build) and times the large-(n,lag) win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_acf`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::acf;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn series(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    // mild AR(1) so autocorrelation is nontrivial
    let mut prev = 0.0;
    (0..n)
        .map(|_| {
            prev = 0.6 * prev + lcg(&mut s);
            prev
        })
        .collect()
}

fn digest(v: &[f64]) -> u64 {
    v.iter()
        .fold(1469598103934665603u64, |h, &x| (h ^ x.to_bits()).wrapping_mul(1099511628211))
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, lag) in &[(1000usize, 3usize), (1000, 40), (5000, 200), (20000, 500)] {
        let d = series(n, 7);
        println!("n={n} lag={lag} digest={:016x}", digest(&acf(&d, lag)));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, lag) in &[(50000usize, 200usize), (100000, 500), (200000, 1000)] {
        let d = series(n, 7);
        let reps = 5;
        let _ = acf(&d, lag);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let a = acf(black_box(&d), lag);
            acc += a[a.len() / 2];
        }
        println!("n={n} lag={lag}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
