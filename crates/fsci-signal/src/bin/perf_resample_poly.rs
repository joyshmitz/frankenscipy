//! Same-process timing + bit-identity harness for `resample_poly`.
//!
//! Each output sample is an independent polyphase FIR dot product at upsampled position
//! j*down; the output samples are now filled in parallel (each into its own index), so the
//! result is bit-identical to the serial loop. Dumps an FNV digest (compare across the
//! stashed serial build) and times the call. Run:
//! `cargo run -p fsci-signal --bin perf_resample_poly`.

use std::hint::black_box;
use std::time::Instant;

use fsci_signal::resample_poly;

fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            (0.02 * t).sin() + 0.4 * (0.005 * t + 0.3).cos() + 0.05 * (0.13 * t).sin()
        })
        .collect()
}

fn digest(values: &[f64]) -> u64 {
    values.iter().fold(1469598103934665603u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    // (signal length, up, down)
    let cases = [
        (200_000usize, 3usize, 2usize),
        (400_000, 5, 4),
        (300_000, 7, 3),
    ];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, up, down) in &cases {
        let x = signal(n);
        let y = resample_poly(&x, up, down).unwrap();
        println!(
            "n={n} up={up} down={down} len={} digest={:016x}",
            y.len(),
            digest(&y)
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, up, down) in &cases {
        let x = signal(n);
        let reps = 5;
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let y = resample_poly(black_box(&x), up, down).unwrap();
            acc += y[y.len() / 2];
        }
        let dt = t0.elapsed();
        println!(
            "n={n:>7} up={up} down={down}  {:>10.3?}/call  (acc={acc:.6})",
            dt / reps
        );
    }
}
