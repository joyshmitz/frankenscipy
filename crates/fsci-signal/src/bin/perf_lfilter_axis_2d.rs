//! Timing + bit-identity harness for `lfilter_axis_2d`.
//!
//! Each row (axis=-1) / column (axis=0) of the 2-D input is filtered by an
//! independent IIR `lfilter`, so the lines fan out across threads in contiguous
//! chunks (one spawn-set), each worker filtering whole lines it owns — the
//! per-line outputs and their layout are unchanged, so the result is
//! bit-identical. This dumps an FNV digest of the output bits (compare across the
//! stashed serial build) and times the large-matrix win.
//! Run: `cargo run --profile release-perf -p fsci-signal --bin perf_lfilter_axis_2d`.

use std::hint::black_box;
use std::time::Instant;

use fsci_signal::lfilter_axis_2d;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    (0..rows)
        .map(|_| (0..cols).map(|_| lcg(&mut s)).collect())
        .collect()
}

fn digest(y: &[Vec<f64>]) -> u64 {
    let mut h = 1469598103934665603u64;
    for row in y {
        for &v in row {
            h = (h ^ v.to_bits()).wrapping_mul(1099511628211);
        }
    }
    h
}

fn main() {
    // 4th-order IIR (b,a length 5).
    let b = [0.1, 0.15, 0.2, 0.15, 0.1];
    let a = [1.0, -0.5, 0.3, -0.1, 0.05];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(r, c) in &[(4usize, 50usize), (7, 200), (8, 200), (64, 300)] {
        let x = matrix(r, c, 7);
        for ax in [-1isize, 0] {
            let y = lfilter_axis_2d(&b, &a, &x, ax).expect("lfilter");
            println!("r={r} c={c} ax={ax} digest={:016x}", digest(&y));
        }
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(r, c) in &[(512usize, 2048usize), (2048, 2048)] {
        let x = matrix(r, c, 7);
        for ax in [-1isize, 0] {
            let reps = 3;
            let _ = lfilter_axis_2d(&b, &a, &x, ax).unwrap();
            let t0 = Instant::now();
            let mut acc = 0.0;
            for _ in 0..reps {
                let y = lfilter_axis_2d(&b, &a, black_box(&x), ax).unwrap();
                acc += y[r / 2][c / 2];
            }
            println!(
                "r={r} c={c} ax={ax}  {:>10.3?}/call (acc={acc:.6})",
                t0.elapsed() / reps
            );
        }
    }
}
