//! Byte-identity + timing harness for `bootstrap_mean` / `bootstrap_std`, whose
//! resample loops are now evaluated in parallel. Each resample consumes exactly
//! `n` LCG draws, so chunk-boundary RNG states are reached by an LCG jump (no
//! sequential pre-pass) — byte-identical to the serial loop.
//!
//! Proof: the (lo, hi) bounds must be bit-identical across the stashed serial
//! build. Run it, `git stash` the lib.rs edit, rebuild (serial), run again.
//! Run: `cargo run --release -p fsci-stats --bin perf_bootstrap`.

use std::time::Instant;

use fsci_stats::{bootstrap_mean, bootstrap_std};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn main() {
    let mut s = 42u64;
    let n = 600usize;
    let data: Vec<f64> = (0..n).map(|_| lcg(&mut s) * 5.0 + 2.0).collect();

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &nb in &[100usize, 1000, 5000] {
        for seed in [12345u64, 99] {
            let (lo, hi) = bootstrap_mean(&data, nb, 0.95, seed);
            let (slo, shi) = bootstrap_std(&data, nb, 0.95, seed);
            println!(
                "nb={nb:>6} seed={seed:<6} mean_bits={:016x},{:016x} std_bits={:016x},{:016x}",
                lo.to_bits(),
                hi.to_bits(),
                slo.to_bits(),
                shi.to_bits()
            );
        }
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &nb in &[2000usize, 10000, 40000] {
        let reps = 5;
        let _ = bootstrap_mean(&data, nb, 0.95, 1);
        macro_rules! time {
            ($name:expr, $f:path) => {{
                let t0 = Instant::now();
                let mut acc = 0.0;
                for _ in 0..reps {
                    let (lo, _) = $f(&data, nb, 0.95, 1);
                    acc += lo;
                }
                println!(
                    "{:<14} nb={nb:>6} {:>10.3?}/call (acc={acc:.6})",
                    $name,
                    t0.elapsed() / reps
                );
            }};
        }
        time!("bootstrap_mean", bootstrap_mean);
        time!("bootstrap_std", bootstrap_std);
    }
}
