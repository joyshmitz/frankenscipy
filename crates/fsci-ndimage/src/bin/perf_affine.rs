//! Same-process timing + bit-identity harness for the ndimage geometric transforms
//! (affine_transform, rotate, zoom, map_coordinates).
//!
//! Each output pixel is an independent spline interpolation of the read-only prefiltered
//! coefficients, so the per-pixel loops are now parallel across the output index. This
//! dumps FNV digests of each transform's output (compare across the stashed serial build
//! to prove byte-identity) and times them. Run: `cargo run -p fsci-ndimage --bin perf_affine`.

use std::hint::black_box;
use std::time::Instant;

use fsci_ndimage::{BoundaryMode, NdArray, affine_transform, map_coordinates, rotate, zoom};

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
    let sizes = [400usize, 700, 1000];
    // 30-degree-ish rotation + scale affine (output->input map).
    let matrix = [[0.92, -0.35, 12.0], [0.35, 0.92, -8.0]];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &sizes {
        let img = image(n);
        let aff = affine_transform(&img, &matrix, order, mode, 0.0).unwrap();
        let rot = rotate(&img, 27.0, false, order, mode, 0.0).unwrap();
        let zm = zoom(&img, &[1.7, 1.7], order, mode, 0.0).unwrap();
        let coords: Vec<Vec<f64>> = vec![
            (0..n * n).map(|k| (k / n) as f64 * 0.7 + 1.3).collect(),
            (0..n * n).map(|k| (k % n) as f64 * 0.7 + 0.9).collect(),
        ];
        let mc = map_coordinates(&img, &coords, order, mode, 0.0).unwrap();
        println!(
            "n={n} affine={:016x} rotate={:016x} zoom={:016x} mapcoord={:016x}",
            digest(&aff.data),
            digest(&rot.data),
            digest(&zm.data),
            digest(&mc)
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &sizes {
        let img = image(n);
        let coords: Vec<Vec<f64>> = vec![
            (0..n * n).map(|k| (k / n) as f64 * 0.7 + 1.3).collect(),
            (0..n * n).map(|k| (k % n) as f64 * 0.7 + 0.9).collect(),
        ];
        let reps = 5;
        macro_rules! time {
            ($name:expr, $body:expr) => {{
                let t0 = Instant::now();
                let mut acc = 0.0;
                for _ in 0..reps {
                    acc += $body;
                }
                println!("n={n:>4} {:<10} {:>9.3?}/call (acc={acc:.3})", $name, t0.elapsed() / reps);
            }};
        }
        time!("affine", affine_transform(black_box(&img), &matrix, order, mode, 0.0).unwrap().data[0]);
        time!("rotate", rotate(black_box(&img), 27.0, false, order, mode, 0.0).unwrap().data[0]);
        time!("zoom", zoom(black_box(&img), &[1.7, 1.7], order, mode, 0.0).unwrap().data[0]);
        time!("mapcoord", map_coordinates(black_box(&img), &coords, order, mode, 0.0).unwrap()[0]);
    }
}
