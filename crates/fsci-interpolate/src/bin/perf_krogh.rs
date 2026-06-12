//! Timing + bit-identity harness for `krogh_interpolate`.
//!
//! Each query is an independent O(degree) Horner evaluation of the Newton form
//! over the reused coefficients; the queries fan out across threads via
//! par_query_map, concatenated in order with each eval unchanged — bit-identical
//! to the serial map. This dumps an FNV digest of the outputs (compare across the
//! stashed serial build) and times the many-query win.
//! Run: `cargo run --profile release-perf -p fsci-interpolate --bin perf_krogh`.

use std::hint::black_box;
use std::time::Instant;

use fsci_interpolate::krogh_interpolate;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn nodes(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut s = seed;
    let xi: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let yi: Vec<f64> = (0..n).map(|_| lcg(&mut s) * 2.0 - 1.0).collect();
    (xi, yi)
}

fn queries(m: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..m).map(|_| lcg(&mut s) * 0.9 + 0.05).collect()
}

fn digest(v: &[f64]) -> u64 {
    v.iter().fold(1469598103934665603u64, |h, &x| {
        (h ^ x.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, m) in &[(15usize, 3usize), (30, 100), (60, 5000), (40, 20000)] {
        let (xi, yi) = nodes(n, 7);
        let xq = queries(m, 99);
        let out = krogh_interpolate(&xi, &yi, &xq).unwrap();
        println!("n={n} m={m} digest={:016x}", digest(&out));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, m) in &[(40usize, 200000usize), (80, 200000), (60, 500000)] {
        let (xi, yi) = nodes(n, 7);
        let xq = queries(m, 99);
        let reps = 5;
        let _ = krogh_interpolate(&xi, &yi, &xq).unwrap();
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let out = krogh_interpolate(&xi, &yi, black_box(&xq)).unwrap();
            acc += out[out.len() / 2];
        }
        println!(
            "n={n} m={m}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
