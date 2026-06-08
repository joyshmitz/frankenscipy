//! Byte-identity + timing harness for opt::hessian, whose O(n^2) independent
//! upper-triangle finite-difference components are now computed in parallel and
//! folded back in the original (row, column) order. Bit-identical to the serial
//! double loop (disjoint cell writes; integer nfev sum; max/AND/precedence-merge
//! reductions are all commutative). Compare across the stashed serial build.
//! Run: `cargo run --profile release-perf -p fsci-opt --bin perf_hessian`.

use std::hint::black_box;
use std::time::Instant;

use fsci_opt::{hessian, DifferentiateOptions};

// A scalar objective whose every evaluation is O(n^2) (pairwise coupling), so each
// finite-difference component is non-trivial and the per-component parallelism is
// visible. Deterministic -> stable golden output.
fn objective(v: &[f64]) -> f64 {
    let n = v.len();
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..n {
            s += ((v[i] - v[j]).cos()) * (-(v[i] * v[i]) / 100.0).exp();
        }
        s += (v[i]).sin() * (i as f64 + 1.0);
    }
    s
}

fn point(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn main() {
    let opts = DifferentiateOptions::default();

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &[4usize, 12, 30] {
        let x = point(n, 1);
        let r = hessian(objective, &x, opts).expect("hessian");
        let mut acc = 0u64;
        for (i, row) in r.ddf.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                acc ^= v.to_bits().rotate_left(((i * 31 + j) % 64) as u32);
            }
        }
        println!(
            "n={n} ddf_xor_bits={acc:016x} nfev={} nit={} success={} status={:?}",
            r.nfev, r.nit, r.success, r.status
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &[40usize, 70, 100] {
        let x = point(n, 7);
        let reps = 3;
        let _ = hessian(objective, &x, opts);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += hessian(black_box(objective), black_box(&x), opts).expect("hessian").ddf[0][0];
        }
        println!("n={n}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
