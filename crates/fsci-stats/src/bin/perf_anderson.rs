//! Timing + bit-identity harness for `anderson` (Anderson-Darling, dist="norm").
//!
//! The serial form evaluates the (erf-based) normal CDF twice per sorted point
//! (forward F(Y_i) and reversed F(Y_{n+1-i})), computing the same n CDF values
//! twice. It now evaluates each point's clamped CDF logs once into a table (in
//! parallel for large n) and does the weighted sum serially in i order — same
//! values + order => bit-identical statistic. This dumps the statistic bits
//! (compare across the stashed serial build) and times the large-n win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_anderson`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::anderson;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
// roughly-normal data via central-limit of 6 uniforms
fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| (0..6).map(|_| lcg(&mut s)).sum::<f64>() - 3.0)
        .collect()
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &[50usize, 2047, 2048, 8000] {
        let d = data(n, 7);
        let r = anderson(&d, "norm");
        println!("n={n} stat={:.17e}", r.statistic);
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &[20000usize, 200000, 2000000] {
        let d = data(n, 7);
        let reps = 10;
        let _ = anderson(&d, "norm");
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += anderson(black_box(&d), "norm").statistic;
        }
        println!("n={n}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
