//! Byte-identity + timing harness for RegularGridInterpolator::eval_many, whose
//! per-query evaluations are now parallel over the independent query batch (via
//! par_query_try_map). Bit-identical to the sequential map (same pure eval per query,
//! order preserved, first error in query order). Compare across the stashed serial build.
//! Run: `cargo run --profile release-perf -p fsci-interpolate --bin perf_reggrid_eval`.

use std::hint::black_box;
use std::time::Instant;

use fsci_interpolate::{RegularGridInterpolator, RegularGridMethod};

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn build(ndim: usize, len: usize) -> RegularGridInterpolator {
    let axes: Vec<Vec<f64>> = (0..ndim)
        .map(|_| (0..len).map(|i| i as f64).collect())
        .collect();
    let total: usize = axes.iter().map(|a| a.len()).product();
    let mut s = 12345u64;
    let values: Vec<f64> = (0..total).map(|_| lcg(&mut s) * 10.0).collect();
    RegularGridInterpolator::new(axes, values, RegularGridMethod::Cubic, false, None)
        .expect("build reggrid")
}

fn queries(m: usize, ndim: usize, len: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    (0..m)
        .map(|_| (0..ndim).map(|_| lcg(&mut s) * (len - 1) as f64).collect())
        .collect()
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(ndim, len, m) in &[(2usize, 40usize, 1000usize), (3, 24, 5000)] {
        let interp = build(ndim, len);
        let qs = queries(m, ndim, len, 7);
        let out = interp.eval_many(&qs).expect("eval_many");
        let mut acc = 0u64;
        for (i, &v) in out.iter().enumerate() {
            acc ^= v.to_bits().rotate_left((i % 64) as u32);
        }
        println!("ndim={ndim} len={len} m={m} out_xor_bits={acc:016x} n_out={}", out.len());
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(ndim, len, m) in &[(3usize, 30usize, 200_000usize), (4, 16, 200_000)] {
        let interp = build(ndim, len);
        let qs = queries(m, ndim, len, 11);
        let reps = 5;
        let _ = interp.eval_many(&qs);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += interp.eval_many(black_box(&qs)).expect("eval_many")[0];
        }
        println!("ndim={ndim} len={len} m={m}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
