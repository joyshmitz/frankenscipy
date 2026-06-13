//! Same-process A/B + byte-identity for the N-D correlate/convolve workhorse: the new
//! interior/flat-gather path (library) vs the old per-pixel get_boundary reference. Byte-
//! identical (bit_identical=true) — the per-tap weights.unravel and per-pixel input.unravel
//! are gone and interior pixels skip boundary handling.
//! Run: `cargo run --release -p fsci-ndimage --bin perf_ndcorrelate`.

use std::hint::black_box;
use std::time::Instant;

use fsci_ndimage::{BoundaryMode, NdArray, convolve, correlate, nd_filter_perpixel_ref};

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn digest(v: &[f64]) -> u64 {
    v.iter().fold(1469598103934665603u64, |h, &x| {
        (h ^ x.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    let (rows, cols) = (1400usize, 1400usize);
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i * 2654435761usize) as f64 / u32::MAX as f64).fract() - 0.5)
        .collect();
    let arr = NdArray::new(data, vec![rows, cols]).unwrap();
    let origins = [0i64, 0];

    for &k in &[3usize, 5, 7, 11] {
        let w = NdArray::new(
            (0..k * k).map(|i| (i as f64 + 1.0).sin()).collect(),
            vec![k, k],
        )
        .unwrap();
        let new = correlate(&arr, &w, BoundaryMode::Reflect, 0.0).unwrap();
        let old = nd_filter_perpixel_ref(&arr, &w, &origins, BoundaryMode::Reflect, 0.0, false).unwrap();
        let bit = digest(&new.data) == digest(&old.data);
        let reps = 20usize;
        let t_new = time(reps, || {
            black_box(correlate(black_box(&arr), &w, BoundaryMode::Reflect, 0.0).unwrap());
        });
        let t_old = time(reps, || {
            black_box(
                nd_filter_perpixel_ref(black_box(&arr), &w, &origins, BoundaryMode::Reflect, 0.0, false)
                    .unwrap(),
            );
        });
        println!(
            "correlate {rows}x{cols} k={k}x{k}: old={t_old:>9.4}ms  new={t_new:>8.4}ms  speedup={:>6.2}x  bit_identical={bit}",
            t_old / t_new
        );

        let cnew = convolve(&arr, &w, BoundaryMode::Reflect, 0.0).unwrap();
        let cold = nd_filter_perpixel_ref(&arr, &w, &origins, BoundaryMode::Reflect, 0.0, true).unwrap();
        let cbit = digest(&cnew.data) == digest(&cold.data);
        let tc_new = time(reps, || {
            black_box(convolve(black_box(&arr), &w, BoundaryMode::Reflect, 0.0).unwrap());
        });
        let tc_old = time(reps, || {
            black_box(
                nd_filter_perpixel_ref(black_box(&arr), &w, &origins, BoundaryMode::Reflect, 0.0, true)
                    .unwrap(),
            );
        });
        println!(
            "convolve  {rows}x{cols} k={k}x{k}: old={tc_old:>9.4}ms  new={tc_new:>8.4}ms  speedup={:>6.2}x  bit_identical={cbit}",
            tc_old / tc_new
        );
    }
}
