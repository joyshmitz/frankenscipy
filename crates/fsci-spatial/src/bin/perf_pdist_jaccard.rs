use fsci_spatial::{DistanceMetric, SPATIAL_BOOL_POPCOUNT_DISABLE, pdist};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn main() {
    let mut s = 8u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = 600usize;
    for &d in &[128usize, 256, 512, 1024, 2048] {
        let xb: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| if r() < 0.5 { 1.0 } else { 0.0 }).collect())
            .collect();
        let xnb: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..d)
                    .map(|_| if r() < 0.5 { 2.9 * r() } else { 0.0 })
                    .collect()
            })
            .collect();
        for (name, x) in [("bool", &xb), ("nonbool", &xnb)] {
            let m = DistanceMetric::Jaccard;
            let run = |dis: bool| {
                SPATIAL_BOOL_POPCOUNT_DISABLE.store(dis, Ordering::Relaxed);
                pdist(x, m).unwrap()
            };
            let pk = run(false);
            let sc = run(true);
            let md = pk
                .iter()
                .zip(&sc)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            let bench = |dis: bool| {
                SPATIAL_BOOL_POPCOUNT_DISABLE.store(dis, Ordering::Relaxed);
                let _ = black_box(pdist(x, m).unwrap());
                let reps = 5;
                let t = Instant::now();
                for _ in 0..reps {
                    let _ = black_box(pdist(black_box(x), m).unwrap());
                }
                t.elapsed().as_secs_f64() / reps as f64 * 1000.0
            };
            let (mut pkt, mut sct) = (f64::MAX, f64::MAX);
            for _ in 0..4 {
                sct = sct.min(bench(true));
                pkt = pkt.min(bench(false));
            }
            println!(
                "d={d} Jaccard-{name}: scalar {sct:.2}ms  popcount {pkt:.2}ms  {:.2}x  maxdiff={md:.1e}",
                sct / pkt
            );
        }
    }
}
