//! Byte-identity + timing harness for `gap_statistic`, whose per-reference
//! kmeans dispersions are now computed in parallel (each reference seed derives
//! directly from (r, k) — fully independent — and results are summed
//! sequentially in r order, so the output is byte-identical).
//!
//! Proof: the gap vector bits must be IDENTICAL across the stashed serial build.
//! Run it, `git stash` lib.rs, rebuild (serial), run again.
//! Run: `cargo run --release -p fsci-cluster --bin perf_gap`.

use std::hint::black_box;
use std::time::Instant;

use fsci_cluster::gap_statistic;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn dataset(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            // three loose blobs so kmeans has structure
            let c = (lcg(&mut s) * 3.0).floor();
            (0..d).map(|_| c * 5.0 + lcg(&mut s)).collect()
        })
        .collect()
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, d, max_k, n_ref) in &[(150usize, 3usize, 4usize, 10usize), (300, 4, 5, 20)] {
        let data = dataset(n, d, 7);
        for seed in [42u64, 12345] {
            let g = gap_statistic(&data, max_k, n_ref, seed);
            let bits: Vec<String> = g.iter().map(|v| format!("{:016x}", v.to_bits())).collect();
            println!("n={n} d={d} max_k={max_k} n_ref={n_ref} seed={seed} gaps={}", bits.join(","));
        }
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, d, max_k, n_ref) in &[(300usize, 4usize, 5usize, 20usize), (500, 6, 6, 40)] {
        let data = dataset(n, d, 7);
        let reps = 3;
        let _ = gap_statistic(&data, max_k, n_ref, 1);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let g = gap_statistic(black_box(&data), max_k, n_ref, 1);
            acc += g.iter().sum::<f64>();
        }
        println!("n={n} d={d} max_k={max_k} n_ref={n_ref}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
