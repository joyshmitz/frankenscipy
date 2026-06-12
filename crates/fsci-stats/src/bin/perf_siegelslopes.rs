//! Byte-identity + timing harness for siegelslopes (repeated-median regression).
//! The O(n²) per-anchor median loop is now parallel over the independent anchors.
//! Byte-identical (slope/intercept bits) to the serial build; compare across stash.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_siegelslopes`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::siegelslopes;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
fn data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut s = seed;
    let x: Vec<f64> = (0..n).map(|i| i as f64 + lcg(&mut s)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 2.5 * xi - 1.3 + (lcg(&mut s) - 0.5) * 50.0)
        .collect();
    (x, y)
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &[50usize, 500, 2000] {
        let (x, y) = data(n, 1);
        let r = siegelslopes(&x, &y);
        println!(
            "n={n} slope_bits={:016x} intercept_bits={:016x}",
            r.slope.to_bits(),
            r.intercept.to_bits()
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &[2000usize, 4000, 8000] {
        let (x, y) = data(n, 7);
        let reps = 5;
        let _ = siegelslopes(&x, &y);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += siegelslopes(black_box(&x), black_box(&y)).slope;
        }
        println!("n={n}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
