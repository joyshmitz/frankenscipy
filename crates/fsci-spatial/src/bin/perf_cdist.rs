//! Same-process A/B for the parallel cdist lever: proves the multithreaded
//! `cdist_metric` is bit-identical to a verbatim sequential reference and times both.

use std::hint::black_box;
use std::time::Instant;

use fsci_spatial::{
    DistanceMetric, canberra, cdist_metric, mahalanobis, metric_distance, pdist, seuclidean,
};

fn grid(n: usize, dim: usize, seed: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| (i as f64 * 0.013 + j as f64 * 0.07 + seed).sin() + 0.5)
                .collect()
        })
        .collect()
}

fn sequential(xa: &[Vec<f64>], xb: &[Vec<f64>], metric: DistanceMetric) -> Vec<Vec<f64>> {
    xa.iter()
        .map(|a| xb.iter().map(|b| metric_distance(a, b, metric)).collect())
        .collect()
}

fn time_it(iters: usize, mut f: impl FnMut() -> Vec<Vec<f64>>) -> f64 {
    black_box(f());
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    let metric = DistanceMetric::Euclidean;
    for &(na, nb, dim) in &[
        (2000usize, 2000usize, 3usize),
        (4000, 1000, 8),
        (3000, 3000, 16),
    ] {
        let xa = grid(na, dim, 0.3);
        let xb = grid(nb, dim, 1.1);

        let got = cdist_metric(&xa, &xb, metric).expect("cdist");
        let want = sequential(&xa, &xb, metric);
        let mut exact = got.len() == want.len();
        for (gr, wr) in got.iter().zip(&want) {
            for (&g, &w) in gr.iter().zip(wr) {
                if g.to_bits() != w.to_bits() {
                    exact = false;
                }
            }
        }

        let iters = (200_000_000 / (na * nb + 1)).clamp(2, 50);
        let par = time_it(iters, || cdist_metric(&xa, &xb, metric).expect("cdist"));
        let seq = time_it(iters, || sequential(&xa, &xb, metric));
        println!(
            "cdist na={na} nb={nb} dim={dim}: seq={seq:>8.3}ms  par={par:>8.3}ms  speedup={:>6.2}x  bit_identical={exact}",
            seq / par
        );
    }

    // pdist (condensed). Sequential reference matches the shipped i<j push order.
    for &(n, dim) in &[(3000usize, 3usize), (4000, 16)] {
        let x = grid(n, dim, 0.7);
        let seq_ref = |x: &[Vec<f64>]| -> Vec<f64> {
            let mut r = Vec::with_capacity(n * (n - 1) / 2);
            for i in 0..n {
                for j in (i + 1)..n {
                    r.push(metric_distance(&x[i], &x[j], metric));
                }
            }
            r
        };
        let got = pdist(&x, metric).expect("pdist");
        let want = seq_ref(&x);
        let exact = got.len() == want.len()
            && got
                .iter()
                .zip(&want)
                .all(|(&g, &w)| g.to_bits() == w.to_bits());
        let iters = (400_000_000 / (n * n + 1)).clamp(2, 50);
        let par = time_it2(iters, || pdist(&x, metric).expect("pdist"));
        let seq = time_it2(iters, || seq_ref(&x));
        println!(
            "pdist n={n} dim={dim}: seq={seq:>8.3}ms  par={par:>8.3}ms  speedup={:>6.2}x  bit_identical={exact}",
            seq / par
        );
    }
    maha_bench();
}

fn maha_bench() {
    for &n in &[64usize, 128, 256] {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.41 + 1.0).cos()).collect();
        let vi: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| 1.0 / ((i + j + 1) as f64)).collect())
            .collect();
        let reps = (40_000_000 / (n * n + 1)).clamp(50, 5000);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += mahalanobis(black_box(&x), black_box(&y), black_box(&vi));
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("mahalanobis n={n}: per_call={ms:>8.5}ms checksum={acc:.6e}");
    }
    for &n in &[256usize, 1024, 4096] {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.41 + 1.0).cos()).collect();
        let v: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64).collect();
        let reps = (200_000_000 / (n + 1)).clamp(1000, 200_000);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += seuclidean(black_box(&x), black_box(&y), black_box(&v));
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("seuclidean n={n}: per_call={ms:>8.6}ms checksum={acc:.6e}");
    }
    for &n in &[256usize, 1024, 4096] {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.41 + 1.0).cos()).collect();
        let reps = (200_000_000 / (n + 1)).clamp(1000, 200_000);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += canberra(black_box(&x), black_box(&y));
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("canberra n={n}: per_call={ms:>8.6}ms checksum={acc:.6e}");
    }
}

fn time_it2(iters: usize, mut f: impl FnMut() -> Vec<f64>) -> f64 {
    black_box(f());
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}
