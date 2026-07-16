use fsci_spatial::{SPATIAL_SEUCLIDEAN_PRESCALE_DISABLE, pdist_seuclidean};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn main() {
    let mut s = 6u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    for &(n, d) in &[(600usize, 64usize), (600, 256), (600, 1024)] {
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| r() * 2.0 - 1.0).collect())
            .collect();
        let vv: Vec<f64> = (0..d).map(|k| 0.4 + (k as f64) * 0.13 + r()).collect();
        let run = |dis: bool| {
            SPATIAL_SEUCLIDEAN_PRESCALE_DISABLE.store(dis, Ordering::Relaxed);
            pdist_seuclidean(&x, &vv).unwrap()
        };
        let pre = run(false);
        let per = run(true);
        let mr = pre
            .iter()
            .zip(&per)
            .map(|(a, b)| {
                if b.abs() > 1e-9 {
                    (a - b).abs() / b.abs()
                } else {
                    0.0
                }
            })
            .fold(0.0f64, f64::max);
        let bench = |dis: bool| {
            SPATIAL_SEUCLIDEAN_PRESCALE_DISABLE.store(dis, Ordering::Relaxed);
            let _ = black_box(pdist_seuclidean(&x, &vv).unwrap());
            let reps = 5;
            let t = Instant::now();
            for _ in 0..reps {
                let _ = black_box(pdist_seuclidean(black_box(&x), black_box(&vv)).unwrap());
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        let (mut pret, mut pert) = (f64::MAX, f64::MAX);
        for _ in 0..4 {
            pert = pert.min(bench(true));
            pret = pret.min(bench(false));
        }
        println!(
            "pdist n={n} d={d}: perpair {pert:.2}ms  prescale {pret:.2}ms  {:.2}x  maxrel={mr:.1e}",
            pert / pret
        );
    }
}
