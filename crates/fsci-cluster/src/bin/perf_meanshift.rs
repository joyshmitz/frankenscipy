//! Byte-identity + timing harness for `mean_shift`, whose per-center Gaussian
//! mean updates are now computed in parallel within each iteration (each center
//! reads only its own old position + the immutable data) — byte-identical.
//!
//! Proof: the center coordinates + labels must be IDENTICAL across the stashed
//! serial build. Run it, `git stash` lib.rs, rebuild (serial), run again.
//! Run: `cargo run --release -p fsci-cluster --bin perf_meanshift`.

use std::hint::black_box;
use std::time::Instant;

use fsci_cluster::mean_shift;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn dataset(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let c = (lcg(&mut s) * 3.0).floor();
            (0..d).map(|_| c * 4.0 + lcg(&mut s)).collect()
        })
        .collect()
}

fn digest(centers: &[Vec<f64>], labels: &[usize]) -> u64 {
    let mut h = 1469598103934665603u64;
    for v in centers.iter().flatten() {
        h = (h ^ v.to_bits()).wrapping_mul(1099511628211);
    }
    for &l in labels {
        h = (h ^ l as u64).wrapping_mul(1099511628211);
    }
    h
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, d) in &[(60usize, 2usize), (200, 3), (500, 4)] {
        let data = dataset(n, d, 7);
        let (c, lbl) = mean_shift(&data, 2.0, 100).expect("mean_shift");
        println!("n={n} d={d} ncenters={} digest={:016x}", c.len(), digest(&c, &lbl));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, d) in &[(500usize, 4usize), (1200, 6), (2000, 8)] {
        let data = dataset(n, d, 7);
        let reps = 3;
        let _ = mean_shift(&data, 2.0, 100);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let (c, _) = mean_shift(black_box(&data), 2.0, 100).unwrap();
            acc += c[0][0];
        }
        println!("n={n} d={d}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
