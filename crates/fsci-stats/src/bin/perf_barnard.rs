//! Byte-identity + timing harness for barnard_exact / boschloo_exact, whose
//! nuisance-parameter grid search is now parallel over the independent grid
//! points (max-reduce is order-independent => byte-identical p-value bits).
//! Compare across the stashed serial build.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_barnard`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::{barnard_exact, boschloo_exact};

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for t in [
        [[8usize, 2], [1, 5]],
        [[40, 10], [15, 35]],
        [[120, 30], [45, 105]],
    ] {
        let b = barnard_exact(&t);
        println!(
            "barnard {:?} pvalue_bits={:016x} stat_bits={:016x}",
            t,
            b.pvalue.to_bits(),
            b.statistic.to_bits()
        );
    }
    for t in [[[8usize, 2], [1, 5]], [[40, 10], [15, 35]]] {
        let b = boschloo_exact(&t);
        println!("boschloo {:?} pvalue_bits={:016x}", t, b.pvalue.to_bits());
    }
    println!("===GOLDEN_PAYLOAD_END===");

    // Timing: large margins make each grid point's O(n1*n2) table sweep heavy.
    for t in [[[200usize, 80], [120, 160]], [[400, 100], [150, 350]]] {
        let reps = 5;
        let _ = barnard_exact(&t);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += barnard_exact(black_box(&t)).pvalue;
        }
        println!(
            "barnard n1n2~{:?}  {:>10.3?}/call (acc={acc:.6})",
            t,
            t0.elapsed() / reps
        );
    }
    for t in [[[200usize, 80], [120, 160]], [[400, 100], [150, 350]]] {
        let reps = 5;
        let _ = boschloo_exact(&t);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += boschloo_exact(black_box(&t)).pvalue;
        }
        println!(
            "boschloo n1n2~{:?}  {:>10.3?}/call (acc={acc:.6})",
            t,
            t0.elapsed() / reps
        );
    }
}
