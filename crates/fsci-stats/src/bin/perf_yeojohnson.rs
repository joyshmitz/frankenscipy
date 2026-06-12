//! Timing + bit-identity harness for `yeojohnson` / `yeojohnson_inv`.
//!
//! Each element is an independent powf/ln (transcendental, compute-bound), so the
//! elements fan out across threads in contiguous chunks, concatenated in order
//! with each element unchanged — bit-identical to the serial map. This dumps an
//! FNV digest of the outputs (compare across the stashed serial build) and times
//! the large-array win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_yeojohnson`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::{yeojohnson, yeojohnson_inv};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64 * 6.0 - 3.0
}

fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s)).collect()
}

fn digest(v: &[f64]) -> u64 {
    v.iter().fold(1469598103934665603u64, |h, &x| {
        (h ^ x.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &[5usize, 1000, 100000] {
        let d = data(n, 7);
        for &lam in &[0.5f64, 0.0, 2.0, 1.3] {
            let f = yeojohnson(&d, lam);
            let inv = yeojohnson_inv(&f, lam);
            println!(
                "n={n} lam={lam} fwd={:016x} inv={:016x}",
                digest(&f),
                digest(&inv)
            );
        }
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &[200000usize, 1000000, 4000000] {
        let d = data(n, 7);
        let reps = 5;
        let _ = yeojohnson(&d, 1.3);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let f = yeojohnson(black_box(&d), 1.3);
            acc += f[f.len() / 2];
        }
        println!("n={n}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
