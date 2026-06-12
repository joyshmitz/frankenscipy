//! Byte-identity + timing harness for `multivariate_normal_rvs`, whose sample
//! loop is now evaluated in parallel. Each sample consumes exactly 2*d LCG draws
//! (Box-Muller) + an O(d^2) Cholesky transform, so chunk-boundary RNG states
//! come from a (chunk*2*d)-step LCG jump — byte-identical to the serial loop.
//!
//! Proof: the FNV digest over all sample values must be IDENTICAL across the
//! stashed serial build. Run it, `git stash` lib.rs, rebuild (serial), run again.
//! Run: `cargo run --release -p fsci-stats --bin perf_mvn_rvs`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::multivariate_normal_rvs;

fn cov_spd(d: usize) -> Vec<Vec<f64>> {
    // Diagonally-dominant SPD matrix.
    (0..d)
        .map(|i| {
            (0..d)
                .map(|j| {
                    if i == j {
                        d as f64 + 1.0
                    } else {
                        1.0 / (1.0 + (i + j) as f64)
                    }
                })
                .collect()
        })
        .collect()
}

fn digest(s: &[Vec<f64>]) -> u64 {
    s.iter().flatten().fold(1469598103934665603u64, |h, &v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(d, ns) in &[(2usize, 100usize), (5, 1000), (16, 2000), (32, 5000)] {
        let mean: Vec<f64> = (0..d).map(|i| i as f64 * 0.5).collect();
        let cov = cov_spd(d);
        for seed in [42u64, 12345] {
            let s = multivariate_normal_rvs(&mean, &cov, ns, seed);
            println!(
                "d={d:>3} ns={ns:>5} seed={seed:<6} digest={:016x}",
                digest(&s)
            );
        }
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(d, ns) in &[(16usize, 5000usize), (32, 10000), (64, 20000)] {
        let mean: Vec<f64> = (0..d).map(|i| i as f64 * 0.5).collect();
        let cov = cov_spd(d);
        let reps = 5;
        let _ = multivariate_normal_rvs(&mean, &cov, ns, 1);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let s = multivariate_normal_rvs(black_box(&mean), &cov, ns, 1);
            acc += s[ns / 2][0];
        }
        println!(
            "d={d:>3} ns={ns:>6} {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
