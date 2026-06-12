//! Timing + bit-identity harness for `brute` (grid global search).
//!
//! Every grid point's objective is independent, so the points fan out across
//! threads into a flat score array; the argmin then keeps the FIRST index at the
//! minimum, bit-identical to the serial strict-`<` scan. This dumps the result
//! bits (x*, f*) compared across the stashed serial build and times the win on a
//! representative (Rosenbrock-like) objective.
//! Run: `cargo run --profile release-perf -p fsci-opt --bin perf_brute`.

use std::hint::black_box;
use std::time::Instant;

use fsci_opt::{brute, rosen};

fn obj(x: &[f64]) -> f64 {
    // Rosenbrock + a couple of transcendentals — representative of the kind of
    // rough objective brute is used for.
    rosen(x) + (x[0] * 3.0).sin() * (x[1] * 2.0).cos()
}

fn digest(r: &fsci_opt::OptimizeResult) -> u64 {
    let mut h = 1469598103934665603u64;
    for &v in &r.x {
        h = (h ^ v.to_bits()).wrapping_mul(1099511628211);
    }
    if let Some(f) = r.fun {
        h = (h ^ f.to_bits()).wrapping_mul(1099511628211);
    }
    h
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    // 2D below threshold, 2D/3D above threshold.
    for &(ndim, ns) in &[(2usize, 20usize), (2, 256), (3, 64), (2, 400)] {
        let ranges: Vec<(f64, f64)> = (0..ndim).map(|_| (-2.0, 2.0)).collect();
        let r = brute(obj, &ranges, ns).unwrap();
        println!("ndim={ndim} ns={ns} digest={:016x}", digest(&r));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(ndim, ns) in &[(2usize, 400usize), (3, 80), (2, 1000)] {
        let ranges: Vec<(f64, f64)> = (0..ndim).map(|_| (-2.0, 2.0)).collect();
        let reps = 5;
        let _ = brute(obj, &ranges, ns).unwrap();
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let r = brute(black_box(obj), &ranges, ns).unwrap();
            acc += r.fun.unwrap();
        }
        let pts = ns.pow(ndim as u32);
        println!(
            "ndim={ndim} ns={ns} pts={pts}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
