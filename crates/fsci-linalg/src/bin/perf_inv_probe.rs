// A/B: inv parallel-blocked-LU fast path vs portfolio (nalgebra try_inverse) at
// medium n. The blocked path's 256 gate regressed per ledger — check 256..768.
use fsci_linalg::{inv, InvOptions, INV_FLAT_MIN_OVERRIDE};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn main() {
    let mut seed = 11u64;
    let mut r = || { seed^=seed<<13; seed^=seed>>7; seed^=seed<<17; (seed>>11) as f64/(1u64<<53) as f64 - 0.5 };
    for &n in &[256usize, 384, 512, 640, 768] {
        let a: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        INV_FLAT_MIN_OVERRIDE.store(4096, Ordering::Relaxed); // force portfolio
        let ip = inv(&a, InvOptions::default()).unwrap().inverse;
        INV_FLAT_MIN_OVERRIDE.store(1, Ordering::Relaxed); // force blocked
        let ib = inv(&a, InvOptions::default()).unwrap().inverse;
        let maxdiff = ip.iter().zip(&ib).flat_map(|(pr,br)| pr.iter().zip(br)).map(|(a,b)| (a-b).abs()).fold(0.0f64,f64::max);
        let reps = 4;
        let bench = |ov: usize| { INV_FLAT_MIN_OVERRIDE.store(ov, Ordering::Relaxed);
            let _=black_box(inv(&a,InvOptions::default()).unwrap());
            let t=Instant::now(); for _ in 0..reps { let _=black_box(inv(black_box(&a),InvOptions::default()).unwrap()); }
            t.elapsed().as_secs_f64()/reps as f64*1000.0 };
        let portfolio=bench(4096).min(bench(4096)); let blocked=bench(1).min(bench(1));
        println!("inv n={n:4}: portfolio {portfolio:8.2}ms -> blocked {blocked:8.2}ms = {:.2}x  maxdiff={maxdiff:.1e}", portfolio/blocked);
    }
    INV_FLAT_MIN_OVERRIDE.store(0, Ordering::Relaxed);
}
