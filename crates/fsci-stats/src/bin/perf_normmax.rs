//! Timing + bit-identity harness for `boxcox_normmax` / `yeojohnson_normmax`.
//!
//! Both fit lambda via a 401-point grid search, each point an independent O(n)
//! transform + variance + log-likelihood. The points fan out across threads via
//! grid_argmax, then an exact first-max argmax (matching the serial strict-`>`
//! scan) picks lambda — bit-identical. This dumps the chosen lambda bits (compare
//! across the stashed serial build) and times the large-n win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_normmax`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::{boxcox_normmax, yeojohnson_normmax};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

// positive for boxcox; signed for yeojohnson
fn pos(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| -(1.0 - lcg(&mut s)).ln() + 0.1).collect()
}
fn signed(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s) * 6.0 - 3.0).collect()
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &[20usize, 600, 5000, 50000] {
        let bc = boxcox_normmax(&pos(n, 7), (-2.0, 2.0));
        let yj = yeojohnson_normmax(&signed(n, 7), (-2.0, 2.0));
        println!("n={n} bc={bc:.17e} yj={yj:.17e}");
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &[2000usize, 20000, 100000] {
        let dp = pos(n, 7);
        let ds = signed(n, 7);
        let reps = 5;
        let _ = boxcox_normmax(&dp, (-2.0, 2.0));
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += boxcox_normmax(black_box(&dp), (-2.0, 2.0));
        }
        println!(
            "boxcox n={n}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
        let t1 = Instant::now();
        let mut acc2 = 0.0;
        for _ in 0..reps {
            acc2 += yeojohnson_normmax(black_box(&ds), (-2.0, 2.0));
        }
        println!(
            "yeojohnson n={n}  {:>10.3?}/call (acc={acc2:.6})",
            t1.elapsed() / reps
        );
    }
}
