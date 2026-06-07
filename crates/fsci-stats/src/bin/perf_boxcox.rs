//! Timing + bit-identity harness for `boxcox` auto-lambda (find_optimal_boxcox_lambda).
//!
//! The auto-lambda path is a 401-point grid search, each point an independent
//! O(n) powf map + variance. The points fan out across threads, then an exact
//! first-max argmax (matching the serial strict-`>` scan) picks lambda — so the
//! chosen lambda and transformed data are bit-identical. This dumps the chosen
//! lambda bits + an FNV digest of the transform (compare across the stashed
//! serial build) and times the large-n win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_boxcox`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::boxcox;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

// strictly-positive, mildly skewed data (Box-Cox requires positive)
fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| -(1.0 - lcg(&mut s)).ln() + 0.1).collect()
}

fn digest(v: &[f64]) -> u64 {
    v.iter()
        .fold(1469598103934665603u64, |h, &x| (h ^ x.to_bits()).wrapping_mul(1099511628211))
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &[20usize, 600, 5000, 50000] {
        let d = data(n, 7);
        let r = boxcox(&d, None).unwrap();
        println!("n={n} lmbda={:.17e} digest={:016x}", r.lmbda, digest(&r.data));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &[2000usize, 20000, 100000] {
        let d = data(n, 7);
        let reps = 5;
        let _ = boxcox(&d, None).unwrap();
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let r = boxcox(black_box(&d), None).unwrap();
            acc += r.lmbda;
        }
        println!("n={n}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
