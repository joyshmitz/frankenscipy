//! Measurement harness: time RbfInterpolator::new() across n to find the
//! build (O(n^2) kernel-matrix fill) vs solve (parallel GEPP) regime.
//! Run: `cargo run --profile release-perf -p fsci-interpolate --bin perf_rbf_build`.

use std::time::Instant;

use fsci_interpolate::{RbfInterpolator, RbfKernel};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn main() {
    let dim = 3;
    for &n in &[256usize, 512, 1024, 2048] {
        let mut s = 1u64;
        let points: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dim).map(|_| lcg(&mut s) * 10.0).collect())
            .collect();
        let values: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        // warm
        let _ = RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, 1.0);
        let reps = 3;
        let t0 = Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..reps {
            let rbf =
                RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, 1.0).expect("rbf new");
            acc += rbf.eval(&points[0]);
        }
        let per = t0.elapsed() / reps;
        println!("n={n:>5} new()={per:>12.3?}/call (acc={acc:.6})");
    }
}
