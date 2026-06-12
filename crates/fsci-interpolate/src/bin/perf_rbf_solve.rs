//! Byte-identity + timing harness for the dense O(n^3) Gaussian-elimination solve
//! behind RbfInterpolator::new (solve_dense_system). The trailing-row eliminations
//! are now parallel for a large dense block; the per-row arithmetic and column order
//! are unchanged, so the recovered weights — and hence eval() at any query — are
//! bit-identical to the serial build. Compare across the stashed serial build.
//! Run: `cargo run --profile release-perf -p fsci-interpolate --bin perf_rbf_solve`.

use std::hint::black_box;
use std::time::Instant;

use fsci_interpolate::{RbfInterpolator, RbfKernel};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
fn dataset(n: usize, dim: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut s = seed;
    let points: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..dim).map(|_| lcg(&mut s) * 10.0).collect())
        .collect();
    let values: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
    (points, values)
}

fn main() {
    // Multiquadric is strictly positive for all distances -> the system matrix is
    // genuinely dense (no entry is exactly 0.0), so the dense O(n^3) path is exercised.
    let kernel = RbfKernel::Multiquadric;

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, dim) in &[(200usize, 2usize), (800, 3), (1500, 3)] {
        let (pts, vals) = dataset(n, dim, 1);
        let rbf = RbfInterpolator::new(&pts, &vals, kernel, 1.0).expect("rbf new");
        // eval at fixed deterministic queries -> bits depend on every solved weight.
        let mut acc = 0u64;
        for q in 0..8 {
            let query: Vec<f64> = (0..dim)
                .map(|d| (q * 7 + d * 3) as f64 % 10.0 + 0.5)
                .collect();
            acc ^= rbf.eval(&query).to_bits().rotate_left(q as u32);
        }
        println!("n={n} dim={dim} eval_xor_bits={acc:016x}");
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, dim) in &[(1500usize, 3usize), (2500, 3), (3500, 3)] {
        let (pts, vals) = dataset(n, dim, 7);
        let reps = 3;
        let _ = RbfInterpolator::new(&pts, &vals, kernel, 1.0);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let rbf = RbfInterpolator::new(black_box(&pts), black_box(&vals), kernel, 1.0)
                .expect("rbf new");
            acc += rbf.eval(&vec![1.5; dim]);
        }
        println!(
            "n={n} dim={dim}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
