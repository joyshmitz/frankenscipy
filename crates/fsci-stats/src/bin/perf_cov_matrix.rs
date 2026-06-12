//! Timing + bit-identity harness for `cov_matrix` (numpy.cov; corr_matrix
//! inherits it).
//!
//! The naive form scatter-updates the whole d×d matrix once per observation
//! (memory-bound at large d). cov[i][j] is a dot of two centered variable-series
//! over observations, so transposing into contiguous columns + parallelising the
//! independent output rows keeps the exact same products/summation order — the
//! result is bit-identical. This dumps an FNV digest of the matrix bits (compare
//! across the stashed naive build) and times the large-d win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_cov_matrix`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::cov_matrix;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

// n observations × d variables.
fn data(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    (0..n)
        .map(|_| (0..d).map(|_| lcg(&mut s) * 4.0 - 2.0).collect())
        .collect()
}

fn digest(c: &[Vec<f64>]) -> u64 {
    let mut h = 1469598103934665603u64;
    for row in c {
        for &v in row {
            h = (h ^ v.to_bits()).wrapping_mul(1099511628211);
        }
    }
    h
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, d) in &[
        (200usize, 16usize),
        (200, 47),
        (200, 48),
        (300, 128),
        (256, 300),
    ] {
        let x = data(n, d, 7);
        println!("n={n} d={d} digest={:016x}", digest(&cov_matrix(&x)));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, d) in &[(512usize, 256usize), (1024, 512), (2048, 768)] {
        let x = data(n, d, 7);
        let reps = 3;
        let _ = cov_matrix(&x);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let c = cov_matrix(black_box(&x));
            acc += c[d / 2][d / 3];
        }
        println!(
            "n={n} d={d}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
