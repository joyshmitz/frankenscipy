//! Same-process A/B + byte-identity for cdist_metric Cosine/Correlation: the new precompute-
//! once path vs an inline old-style per-pair reference (recomputes norms/means each pair,
//! parallel over rows). Byte-identical (digest). Run:
//! `cargo run --release -p fsci-spatial --bin perf_cdist_corr`.

use std::hint::black_box;
use std::time::Instant;

use fsci_spatial::{DistanceMetric, cdist_metric, correlation, cosine};

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn digest(m: &[Vec<f64>]) -> u64 {
    m.iter().flatten().fold(1469598103934665603u64, |h, &x| {
        (h ^ x.to_bits()).wrapping_mul(1099511628211)
    })
}

// Old-style cdist: per-pair metric call, parallel over xa rows.
fn old_cdist(xa: &[Vec<f64>], xb: &[Vec<f64>], f: fn(&[f64], &[f64]) -> f64) -> Vec<Vec<f64>> {
    let na = xa.len();
    let nth = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1).min(na.max(1));
    let chunk = na.div_ceil(nth);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..nth)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= na {
                    return None;
                }
                let i1 = (i0 + chunk).min(na);
                Some(s.spawn(move || {
                    xa[i0..i1]
                        .iter()
                        .map(|a| xb.iter().map(|b| f(a, b)).collect::<Vec<f64>>())
                        .collect::<Vec<Vec<f64>>>()
                }))
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

fn main() {
    let mut s = 0xBEEFu64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    for &(na, nb, d) in &[(600usize, 600usize, 64usize), (1000, 1000, 128), (1500, 1500, 256)] {
        let xa: Vec<Vec<f64>> = (0..na).map(|_| (0..d).map(|_| next()).collect()).collect();
        let xb: Vec<Vec<f64>> = (0..nb).map(|_| (0..d).map(|_| next()).collect()).collect();
        for (name, metric, f) in [
            ("cosine", DistanceMetric::Cosine, cosine as fn(&[f64], &[f64]) -> f64),
            ("correl", DistanceMetric::Correlation, correlation as fn(&[f64], &[f64]) -> f64),
        ] {
            let new = cdist_metric(&xa, &xb, metric).unwrap();
            let old = old_cdist(&xa, &xb, f);
            let bit = digest(&new) == digest(&old);
            let reps = 5usize;
            let t_new = time(reps, || {
                black_box(cdist_metric(black_box(&xa), black_box(&xb), metric).unwrap());
            });
            let t_old = time(reps, || {
                black_box(old_cdist(black_box(&xa), black_box(&xb), f));
            });
            println!(
                "cdist {name} {na}x{nb} d={d:>4}: old={t_old:>8.4}ms  new={t_new:>8.4}ms  speedup={:>6.2}x  bit_identical={bit}",
                t_old / t_new
            );
        }
    }
}
