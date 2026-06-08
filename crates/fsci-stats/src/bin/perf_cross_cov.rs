//! Byte-identity + timing harness for cross_cov. The per-observation scatter that built
//! the dx×dy cross-covariance is now, for large problems, a transposed contiguous dot per
//! output row, fanned out across threads. The summation order (over observations) and the
//! terms are unchanged, so every entry is bit-identical to the serial scatter.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_cross_cov`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::cross_cov;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn dataset(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    (0..n)
        .map(|_| (0..d).map(|_| (lcg(&mut s) - 0.5) * 20.0).collect())
        .collect()
}

fn golden_bits(cov: &[Vec<f64>]) -> u64 {
    let mut acc = 0u64;
    for (i, row) in cov.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            acc ^= v.to_bits().rotate_left(((i * 31 + j) % 64) as u32);
        }
    }
    acc
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, dx, dy) in &[(50usize, 6usize, 4usize), (200, 80, 64), (500, 120, 90)] {
        let x = dataset(n, dx, 1);
        let y = dataset(n, dy, 2);
        let cov = cross_cov(&x, &y);
        println!(
            "n={n} dx={dx} dy={dy} shape=({},{}) xor_bits={:016x}",
            cov.len(),
            cov.first().map_or(0, Vec::len),
            golden_bits(&cov)
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, dx, dy) in &[(2000usize, 160usize, 160usize), (4000, 200, 200)] {
        let x = dataset(n, dx, 7);
        let y = dataset(n, dy, 8);
        let reps = 5;
        let _ = cross_cov(&x, &y);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += cross_cov(black_box(&x), black_box(&y))[0][0];
        }
        println!("n={n} dx={dx} dy={dy}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
