//! Same-process timing + bit-identity harness for the N-D interpolator eval_many
//! (LinearNDInterpolator, CloughTocher2DInterpolator, NearestNDInterpolator).
//!
//! Each query independently locates its simplex / nearest neighbour and evaluates the
//! interpolant, so eval_many is now parallel across queries (results in query order). This
//! dumps FNV digests (compare across the stashed serial build) and times the calls.
//! Run: `cargo run -p fsci-interpolate --bin perf_ndinterp`.

use std::hint::black_box;
use std::time::Instant;

use fsci_interpolate::{CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn dataset(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut s = seed;
    let pts: Vec<Vec<f64>> = (0..n)
        .map(|_| vec![lcg(&mut s) * 10.0, lcg(&mut s) * 10.0])
        .collect();
    let vals: Vec<f64> = pts
        .iter()
        .map(|p| (0.3 * p[0]).sin() + (0.2 * p[1]).cos())
        .collect();
    (pts, vals)
}

fn queries(m: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    (0..m)
        .map(|_| vec![lcg(&mut s) * 10.0, lcg(&mut s) * 10.0])
        .collect()
}

fn digest(values: &[f64]) -> u64 {
    values.iter().fold(1469598103934665603u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    let (pts, vals) = dataset(2000, 1);
    let lin = LinearNDInterpolator::new(&pts, &vals).unwrap();
    let clo = CloughTocher2DInterpolator::new(&pts, &vals).unwrap();
    let near = NearestNDInterpolator::new(&pts, &vals).unwrap();
    let sizes = [20_000usize, 80_000, 200_000];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &m in &sizes {
        let q = queries(m, 99);
        let l = lin.eval_many(&q).unwrap();
        let c = clo.eval_many(&q).unwrap();
        let nr = near.eval_many(&q).unwrap();
        println!(
            "m={m} linear={:016x} clough={:016x} nearest={:016x}",
            digest(&l),
            digest(&c),
            digest(&nr)
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &m in &sizes {
        let q = queries(m, 99);
        let reps = 5;
        macro_rules! time {
            ($name:expr, $obj:expr) => {{
                let t0 = Instant::now();
                let mut acc = 0.0;
                for _ in 0..reps {
                    let r = $obj.eval_many(black_box(&q)).unwrap();
                    acc += r[r.len() / 2];
                }
                println!(
                    "m={m:>7} {:<8} {:>9.3?}/call (acc={acc:.3})",
                    $name,
                    t0.elapsed() / reps
                );
            }};
        }
        time!("linear", lin);
        time!("clough", clo);
        time!("nearest", near);
    }
}
