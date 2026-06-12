//! Byte-identity + timing harness for `correlate` (the nd correlate/convolve
//! workhorse), whose per-output-pixel weighted sum is now distributed across
//! threads via `fill_pixels_parallel` — byte-identical to the serial loop.
//!
//! Proof: the output bits must be IDENTICAL across the stashed serial build.
//! Run it, `git stash` lib.rs, rebuild (serial), run again.
//! Run: `cargo run --release -p fsci-ndimage --bin perf_correlate`.

use std::hint::black_box;
use std::time::Instant;

use fsci_ndimage::{BoundaryMode, NdArray, correlate};

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

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(img, ker) in &[
        ([64usize, 64usize], [3usize, 3usize]),
        ([128, 128], [5, 5]),
        ([256, 256], [7, 7]),
    ] {
        let input = arr(&img, 7);
        let weights = arr(&ker, 99);
        let out = correlate(&input, &weights, BoundaryMode::Reflect, 0.0).unwrap();
        println!("img={img:?} ker={ker:?} digest={:016x}", digest(&out));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(img, ker) in &[
        ([256usize, 256usize], [7usize, 7usize]),
        ([512, 512], [9, 9]),
        ([700, 700], [11, 11]),
    ] {
        let input = arr(&img, 7);
        let weights = arr(&ker, 99);
        let reps = 5;
        let _ = correlate(&input, &weights, BoundaryMode::Reflect, 0.0).unwrap();
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let out = correlate(black_box(&input), &weights, BoundaryMode::Reflect, 0.0).unwrap();
            acc += out.data[out.data.len() / 2];
        }
        println!(
            "img={img:?} ker={ker:?}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
