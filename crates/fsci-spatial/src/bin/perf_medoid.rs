//! Same-process timing + bit-identity harness for `medoid` and `diameter`.
//!
//! Both are O(n^2*d) all-pairs reductions: medoid = argmin of per-point distance sums,
//! diameter = scalar max pairwise distance. Both are now parallel across the outer index
//! (independent per-point sums for medoid; NaN-propagating commutative max for diameter),
//! so results are bit-identical to the serial loops. Dumps the medoid index and diameter
//! bits (compare across the stashed serial build) and times the calls.
//! Run: `cargo run -p fsci-spatial --bin perf_medoid`.

use std::hint::black_box;
use std::time::Instant;

use fsci_spatial::{diameter, medoid};

fn points(n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..d)
                .map(|j| {
                    let t = (i * 31 + j * 17) as f64;
                    (0.013 * t).sin() + 0.5 * (0.07 * t + j as f64).cos()
                })
                .collect()
        })
        .collect()
}

fn main() {
    let cases = [(2000usize, 8usize), (4000, 12), (6000, 16)];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, d) in &cases {
        let p = points(n, d);
        let m = medoid(&p).unwrap();
        let diam = diameter(&p);
        println!("n={n} d={d} medoid={m} diameter={:016x}", diam.to_bits());
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, d) in &cases {
        let p = points(n, d);
        let reps = 5;
        let t0 = Instant::now();
        let mut acc = 0usize;
        for _ in 0..reps {
            acc += medoid(black_box(&p)).unwrap();
        }
        let med_dt = t0.elapsed();
        let t1 = Instant::now();
        let mut dacc = 0.0;
        for _ in 0..reps {
            dacc += diameter(black_box(&p));
        }
        let dia_dt = t1.elapsed();
        println!(
            "n={n:>5} d={d:>3}  medoid={:>9.3?}/call  diameter={:>9.3?}/call  (acc={acc} dacc={dacc:.3})",
            med_dt / reps,
            dia_dt / reps
        );
    }
}
