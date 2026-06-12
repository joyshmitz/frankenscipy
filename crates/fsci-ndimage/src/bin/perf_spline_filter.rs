//! Same-process timing + bit-identity harness for `spline_filter` (the B-spline
//! prefilter) and, via `affine_transform`, the geometric-transform prefilter step.
//!
//! The prefilter solves one independent 1D IIR/de Boor coefficient run per line along
//! each axis; those lines are now computed in parallel (scatter in line order), so the
//! result is bit-identical to the serial loop. Run:
//! `cargo run -p fsci-ndimage --bin perf_spline_filter`.

use std::hint::black_box;
use std::time::Instant;

use fsci_ndimage::{BoundaryMode, NdArray, affine_transform, spline_filter};

fn image(n: usize) -> NdArray {
    let data: Vec<f64> = (0..n * n)
        .map(|k| {
            let r = (k / n) as f64;
            let c = (k % n) as f64;
            (0.05 * r).sin() + 0.5 * (0.03 * c + 0.4).cos() + 0.01 * (r * c).sqrt()
        })
        .collect();
    NdArray::new(data, vec![n, n]).unwrap()
}

fn digest(values: &[f64]) -> u64 {
    values.iter().fold(1469598103934665603u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    let order = 3usize;
    let mode = BoundaryMode::Reflect;
    let sizes = [800usize, 1500, 2500];
    let matrix = [[0.92, -0.35, 12.0], [0.35, 0.92, -8.0]];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &sizes {
        let img = image(n);
        let sf = spline_filter(&img, order, mode).unwrap();
        let aff = affine_transform(&img, &matrix, order, mode, 0.0).unwrap();
        println!(
            "n={n} spline_filter={:016x} affine={:016x}",
            digest(&sf.data),
            digest(&aff.data)
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &sizes {
        let img = image(n);
        let reps = 5;
        macro_rules! time {
            ($name:expr, $body:expr) => {{
                let t0 = Instant::now();
                let mut acc = 0.0;
                for _ in 0..reps {
                    acc += $body;
                }
                println!(
                    "n={n:>5} {:<14} {:>9.3?}/call (acc={acc:.3})",
                    $name,
                    t0.elapsed() / reps
                );
            }};
        }
        time!(
            "spline_filter",
            spline_filter(black_box(&img), order, mode).unwrap().data[0]
        );
        time!(
            "affine",
            affine_transform(black_box(&img), &matrix, order, mode, 0.0)
                .unwrap()
                .data[0]
        );
    }
}
