//! Byte-identity + timing harness for `generic_filter`, whose per-output
//! neighbourhood gather + user-function call is now distributed across threads
//! via `fill_pixels_parallel` (F: Fn + Sync) — byte-identical to the serial loop.
//! Run: `cargo run --release -p fsci-ndimage --bin perf_generic`.

use std::hint::black_box;
use std::time::Instant;

use fsci_ndimage::{BoundaryMode, NdArray, generic_filter};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
fn arr(shape: &[usize], seed: u64) -> NdArray {
    let mut s = seed;
    let n: usize = shape.iter().product();
    NdArray::new((0..n).map(|_| lcg(&mut s)).collect(), shape.to_vec()).unwrap()
}
fn digest(a: &NdArray) -> u64 {
    a.data.iter().fold(1469598103934665603u64, |h, &v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

// Non-trivial neighbourhood statistic: variance (deterministic, order-fixed).
fn nvar(w: &[f64]) -> f64 {
    let n = w.len() as f64;
    let mean = w.iter().sum::<f64>() / n;
    w.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(img, size) in &[
        ([64usize, 64usize], 3usize),
        ([128, 128], 5),
        ([200, 200], 7),
    ] {
        let input = arr(&img, 7);
        let out = generic_filter(&input, nvar, size, BoundaryMode::Reflect, 0.0).unwrap();
        println!("img={img:?} size={size} digest={:016x}", digest(&out));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(img, size) in &[([256usize, 256usize], 7usize), ([400, 400], 9)] {
        let input = arr(&img, 7);
        let reps = 5;
        let _ = generic_filter(&input, nvar, size, BoundaryMode::Reflect, 0.0).unwrap();
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let out =
                generic_filter(black_box(&input), nvar, size, BoundaryMode::Reflect, 0.0).unwrap();
            acc += out.data[out.data.len() / 2];
        }
        println!(
            "img={img:?} size={size}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
