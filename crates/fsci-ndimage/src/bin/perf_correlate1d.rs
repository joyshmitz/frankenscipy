//! Same-process A/B + byte-identity proof for correlate1d: the new slab line-walk (library)
//! vs the old per-pixel unravel path (retained reference). The per-output dot is summed in
//! the SAME order with the SAME boundary values, so the result is byte-identical (max|dx|=0).
//! Run: `cargo run --release -p fsci-ndimage --bin perf_correlate1d`.

use std::hint::black_box;
use std::time::Instant;

use fsci_ndimage::{BoundaryMode, NdArray, correlate1d, correlate1d_perwindow_ref};

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
    let (rows, cols) = (2000usize, 2000usize);
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i * 2654435761usize) as f64 / u32::MAX as f64).fract() - 0.5)
        .collect();
    let arr = NdArray::new(data, vec![rows, cols]).unwrap();

    for axis in [1usize, 0usize] {
        println!("correlate1d {rows}x{cols}, axis={axis}, Reflect (vs old per-pixel path):");
        for &klen in &[3usize, 7, 15, 31, 51] {
            let weights: Vec<f64> = (0..klen).map(|k| (k as f64 + 1.0).sin()).collect();
            let new = correlate1d(&arr, &weights, axis, BoundaryMode::Reflect, 0.0).unwrap();
            let old =
                correlate1d_perwindow_ref(&arr, &weights, axis, BoundaryMode::Reflect, 0.0, 0)
                    .unwrap();
            let max_dx = new
                .data
                .iter()
                .zip(&old.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let bitident = digest(&new.data) == digest(&old.data);
            let reps = 40usize;
            let t_new = time(reps, || {
                black_box(
                    correlate1d(black_box(&arr), &weights, axis, BoundaryMode::Reflect, 0.0)
                        .unwrap(),
                );
            });
            let t_old = time(reps, || {
                black_box(
                    correlate1d_perwindow_ref(
                        black_box(&arr),
                        &weights,
                        axis,
                        BoundaryMode::Reflect,
                        0.0,
                        0,
                    )
                    .unwrap(),
                );
            });
            println!(
                "  k={klen:>3}: old={t_old:>8.4}ms  new={t_new:>8.4}ms  speedup={:>6.2}x  max|dx|={max_dx:.1e}  bit_identical={bitident}",
                t_old / t_new
            );
        }
    }
}
