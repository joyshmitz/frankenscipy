//! Timing + bit-identity harness for `BarycentricInterpolator::new` construction.
//!
//! Building the interpolator computes the barycentric weights w_i = 1/∏_{j≠i}(x_i-x_j),
//! an O(n²) pass over all node pairs (plus an O(n²) duplicate-node check). Each row i is an
//! independent product/scan into its own slot, so the construction fans out across cores;
//! the weights (and the first reported duplicate/denominator error in index order) are
//! bit-identical to the serial double loop. Chebyshev nodes keep large-n barycentric stable
//! (spectral methods), so this path is real-value at n in the thousands.
//! Golden: FNV digest of eval outputs (a bit-exact function of the weights).
//! Run: `cargo run --profile release-perf -p fsci-interpolate --bin perf_barycentric_build`.

use std::hint::black_box;
use std::time::Instant;

use fsci_interpolate::BarycentricInterpolator;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

// n distinct Chebyshev nodes on [-1,1] (strictly increasing in index) + random values.
fn nodes(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut s = seed;
    let xi: Vec<f64> = (0..n)
        .map(|i| -(std::f64::consts::PI * (i as f64 + 0.5) / n as f64).cos())
        .collect();
    let yi: Vec<f64> = (0..n).map(|_| lcg(&mut s) * 2.0 - 1.0).collect();
    (xi, yi)
}

fn digest(v: &[f64]) -> u64 {
    v.iter().fold(1469598103934665603u64, |h, &x| {
        (h ^ x.to_bits()).wrapping_mul(1099511628211)
    })
}

fn eval_digest(interp: &BarycentricInterpolator, m: usize) -> u64 {
    let mut s = 99u64;
    let xq: Vec<f64> = (0..m).map(|_| lcg(&mut s) * 1.8 - 0.9).collect();
    digest(&interp.eval_many(&xq))
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &[20usize, 200, 1000, 3000] {
        let (xi, yi) = nodes(n, 7);
        let interp = BarycentricInterpolator::new(&xi, &yi).unwrap();
        println!("n={n} eval_digest={:016x}", eval_digest(&interp, 256));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &[2000usize, 4000, 8000] {
        let (xi, yi) = nodes(n, 7);
        let reps = 5;
        let _ = BarycentricInterpolator::new(&xi, &yi).unwrap();
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let interp = BarycentricInterpolator::new(black_box(&xi), black_box(&yi)).unwrap();
            acc += interp.eval(0.123);
        }
        println!("build n={n}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
