//! Byte-identity + timing harness for permutation_test.
//! The per-resample stat_fn eval is now parallel over the (sequentially
//! materialized) permutation states. Byte-identical (observed/pvalue bits) to
//! the serial build because the extreme count is an order-independent integer
//! sum over the same arrays. Compare across the stashed serial build.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_permutation`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::permutation_test;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
fn data(nx: usize, ny: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut s = seed;
    let x: Vec<f64> = (0..nx).map(|_| lcg(&mut s)).collect();
    let y: Vec<f64> = (0..ny).map(|_| lcg(&mut s) + 0.3).collect();
    (x, y)
}

// A non-trivial statistic: difference of trimmed means (sorts each side).
fn trimmed_mean_diff(a: &[f64], b: &[f64]) -> f64 {
    let tm = |v: &[f64]| -> f64 {
        let mut s: Vec<f64> = v.to_vec();
        s.sort_by(|p, q| p.total_cmp(q));
        let k = s.len() / 10;
        let core = &s[k..s.len() - k];
        core.iter().sum::<f64>() / core.len() as f64
    };
    tm(a) - tm(b)
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(nx, ny, np) in &[(15usize, 12usize, 200usize), (40, 35, 999), (80, 90, 500)] {
        let (x, y) = data(nx, ny, 1);
        let (stat, pval) = permutation_test(&x, &y, trimmed_mean_diff, np, 12345);
        println!(
            "nx={nx} ny={ny} np={np} stat_bits={:016x} pval_bits={:016x}",
            stat.to_bits(),
            pval.to_bits()
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(nx, ny, np) in &[(300usize, 300usize, 5000usize), (1000, 1000, 5000)] {
        let (x, y) = data(nx, ny, 7);
        let reps = 5;
        let _ = permutation_test(&x, &y, trimmed_mean_diff, np, 99);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += permutation_test(black_box(&x), black_box(&y), trimmed_mean_diff, np, 99).1;
        }
        println!(
            "nx={nx} ny={ny} np={np}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
