//! Timing + bit-identity harness for `hdquantiles` / `hdquantiles_sd`.
//!
//! Each Harrell-Davis quantile sums incomplete-beta CDF weights over the n data
//! breakpoints. hdquantiles previously evaluated beta.cdf 2n times per prob
//! (adjacent k share a breakpoint); it now evaluates the n+1 distinct
//! breakpoints once (halving the calls) and maps them in parallel (par_beta_cdf),
//! and hdquantiles_sd's cdf map is parallelised too. Same cdf values + same
//! weighted sums => bit-identical. This dumps an FNV digest (compare across the
//! stashed serial build) and times the large-n win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_hdquantiles`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::{hdquantiles, hdquantiles_sd};

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s) * 100.0).collect()
}
fn digest(v: &[f64]) -> u64 {
    v.iter()
        .fold(1469598103934665603u64, |h, &x| (h ^ x.to_bits()).wrapping_mul(1099511628211))
}

fn main() {
    let probs = [0.1, 0.25, 0.5, 0.75, 0.9];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &[50usize, 2047, 2048, 8000] {
        let d = data(n, 7);
        println!(
            "n={n} hd={:016x} sd={:016x}",
            digest(&hdquantiles(&d, &probs)),
            digest(&hdquantiles_sd(&d, &probs))
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &[5000usize, 20000, 60000] {
        let d = data(n, 7);
        let reps = 5;
        let _ = hdquantiles(&d, &probs);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += hdquantiles(black_box(&d), &probs)[2];
        }
        println!("hdq   n={n}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
        let t1 = Instant::now();
        let mut acc2 = 0.0;
        for _ in 0..reps {
            acc2 += hdquantiles_sd(black_box(&d), &probs)[2];
        }
        println!("hdqsd n={n}  {:>10.3?}/call (acc={acc2:.6})", t1.elapsed() / reps);
    }
}
