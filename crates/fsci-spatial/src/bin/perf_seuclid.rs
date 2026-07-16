use fsci_spatial::{SPATIAL_SEUCLIDEAN_PRESCALE_DISABLE, cdist_seuclidean};
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
    for &(n, d) in &[
        (400usize, 16usize),
        (400, 64),
        (400, 256),
        (400, 1024),
        (800, 128),
    ] {
        let xa: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| r() * 2.0 - 1.0).collect())
            .collect();
        let xb: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| r() * 2.0 - 1.0).collect())
            .collect();
        let vv: Vec<f64> = (0..d).map(|k| 0.4 + (k as f64) * 0.13 + r()).collect();
        let run = |dis: bool| {
            SPATIAL_SEUCLIDEAN_PRESCALE_DISABLE.store(dis, Ordering::Relaxed);
            cdist_seuclidean(&xa, &xb, &vv).unwrap()
        };
        let pre = run(false);
        let per = run(true);
        let (mut md, mut mr) = (0.0f64, 0.0f64);
        for (a, b) in pre.iter().flatten().zip(per.iter().flatten()) {
            let e = (a - b).abs();
            md = md.max(e);
            if b.abs() > 1e-9 {
                mr = mr.max(e / b.abs());
            }
        }
        let bench = |dis: bool| {
            SPATIAL_SEUCLIDEAN_PRESCALE_DISABLE.store(dis, Ordering::Relaxed);
            let _ = black_box(cdist_seuclidean(&xa, &xb, &vv).unwrap());
            let reps = 5;
            let t = Instant::now();
            for _ in 0..reps {
                let _ = black_box(
                    cdist_seuclidean(black_box(&xa), black_box(&xb), black_box(&vv)).unwrap(),
                );
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        let (mut pret, mut pert) = (f64::MAX, f64::MAX);
        for _ in 0..4 {
            pert = pert.min(bench(true));
            pret = pret.min(bench(false));
        }
        println!(
            "n={n} d={d}: perpair {pert:.2}ms  prescale {pret:.2}ms  {:.2}x  maxdiff={md:.1e} maxrel={mr:.1e}",
            pert / pret
        );
    }
}
