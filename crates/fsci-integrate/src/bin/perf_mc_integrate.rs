//! Byte-identity + timing harness for `monte_carlo_integrate`, whose per-sample
//! integrand evaluation is now parallel across samples (each sample consumes a
//! fixed `d` LCG draws, so chunk-boundary RNG states come from an LCG jump; the
//! sum/sum_sq reduction stays sequential in sample order).
//!
//! Proof: (estimate, std_err) must be bit-identical across the stashed serial
//! build. Run it, `git stash` quad.rs, rebuild (serial), run again.
//! Run: `cargo run --release -p fsci-integrate --bin perf_mc_integrate`.

use std::hint::black_box;
use std::time::Instant;

use fsci_integrate::monte_carlo_integrate;

fn main() {
    // A non-trivial integrand so the parallel win is visible.
    let f = |x: &[f64]| -> f64 {
        x.iter()
            .map(|&v| (3.0 * v).sin().powi(2) * (1.7 * v).cos())
            .sum::<f64>()
    };
    let bounds3 = [(0.0, 1.0), (-1.0, 2.0), (0.5, 1.5)];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &ns in &[100usize, 1000, 10000] {
        for seed in [42u64, 12345] {
            let (est, se) = monte_carlo_integrate(f, &bounds3, ns, seed);
            println!(
                "ns={ns:>6} seed={seed:<6} est_bits={:016x} se_bits={:016x}",
                est.to_bits(),
                se.to_bits()
            );
        }
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &ns in &[20000usize, 100000, 400000] {
        let reps = 5;
        let _ = monte_carlo_integrate(f, &bounds3, ns, 1);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let (est, _) = monte_carlo_integrate(f, black_box(&bounds3), ns, 1);
            acc += est;
        }
        println!(
            "ns={ns:>7} {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
