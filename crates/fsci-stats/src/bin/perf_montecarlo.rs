//! Byte-identity + timing harness for `monte_carlo_test`, whose null
//! distribution is now evaluated in parallel across the independent resamples
//! (seeds generated sequentially first, so it stays byte-identical).
//!
//! Proof: the FNV digest of the null distribution and the p-value must be
//! IDENTICAL across the stashed serial build. Run it, `git stash` the lib.rs
//! edit, rebuild (serial), and run again: digest+pvalue match, parallel faster.
//! Run: `cargo run --release -p fsci-stats --bin perf_montecarlo`.

use std::time::Instant;

use fsci_stats::monte_carlo_test;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn main() {
    // Resample: generate `m` standard-uniform-ish values from the seed. Statistic:
    // a non-trivial O(m) reduction (mean of sin-warped values) so the parallel
    // win is visible. Both are deterministic in their seed/input.
    let m = 800usize;
    let rvs = |seed: u64| -> Vec<f64> {
        let mut s = seed;
        (0..m).map(|_| lcg(&mut s)).collect()
    };
    let statistic = |sample: &[f64]| -> f64 {
        sample.iter().map(|&v| (3.0 * v).sin() * (2.0 * v).cos()).sum::<f64>() / sample.len() as f64
    };
    let data: Vec<f64> = (0..m).map(|i| (i as f64 * 0.01).tanh()).collect();

    fn digest(v: &[f64]) -> u64 {
        v.iter().fold(1469598103934665603u64, |h, x| (h ^ x.to_bits()).wrapping_mul(1099511628211))
    }

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &nr in &[100usize, 1000, 5000] {
        let r = monte_carlo_test(&data, rvs, statistic, nr, 12345, "two-sided");
        println!(
            "n_resamples={nr:>6} stat={:.12e} pvalue={:.12e} null_digest={:016x}",
            r.statistic,
            r.pvalue,
            digest(&r.null_distribution)
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &nr in &[2000usize, 10000, 40000] {
        let reps = 5;
        let _ = monte_carlo_test(&data, rvs, statistic, nr, 7, "two-sided");
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let r = monte_carlo_test(&data, rvs, statistic, nr, 7, "two-sided");
            acc += r.pvalue;
        }
        println!("n_resamples={nr:>6} {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
