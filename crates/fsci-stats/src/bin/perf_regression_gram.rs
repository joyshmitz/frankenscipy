//! Timing + bit-identity harness for the regression normal-equation build
//! (multiple_regression / ridge_regression).
//!
//! Both build the intercept-augmented XtX by per-sample rank-1 scatter into the
//! whole (p+1)² matrix (memory-bound for large p). The shared
//! augmented_normal_equations helper instead forms each entry as an in-order dot
//! of transposed contiguous columns (same products, same sample order =>
//! bit-identical) and fans the output rows across threads. This digests the
//! fitted coefficients (compare across the stashed scatter build) and times the
//! large-p win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_regression_gram`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::{multiple_regression, ridge_regression};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn dataset(n: usize, p: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut s = seed;
    let x: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..p).map(|_| lcg(&mut s)).collect())
        .collect();
    let y: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
    (x, y)
}

fn digest(v: &[f64]) -> u64 {
    v.iter().fold(1469598103934665603u64, |h, &x| {
        (h ^ x.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, p) in &[
        (200usize, 16usize),
        (200, 47),
        (300, 48),
        (400, 120),
        (500, 256),
    ] {
        let (x, y) = dataset(n, p, 7);
        let (beta, _res, r2, se) = multiple_regression(&x, &y);
        let rb = ridge_regression(&x, &y, 0.5);
        println!(
            "n={n} p={p} mreg={:016x} r2={:.6e} se={:016x} ridge={:016x}",
            digest(&beta),
            r2,
            digest(&se),
            digest(&rb)
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, p) in &[(1000usize, 256usize), (2000, 512), (3000, 768)] {
        let (x, y) = dataset(n, p, 7);
        let reps = 5;
        let _ = multiple_regression(&x, &y);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let (beta, _, _, _) = multiple_regression(black_box(&x), black_box(&y));
            acc += beta[p / 2];
        }
        println!(
            "mreg  n={n} p={p}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
        let t1 = Instant::now();
        let mut acc2 = 0.0;
        for _ in 0..reps {
            let rb = ridge_regression(black_box(&x), black_box(&y), 0.5);
            acc2 += rb[p / 2];
        }
        println!(
            "ridge n={n} p={p}  {:>10.3?}/call (acc={acc2:.6})",
            t1.elapsed() / reps
        );
    }
}
