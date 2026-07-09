// A/B: public lu() (P/L/U) blocked vs nalgebra at medium n (shares lu_factor gate).
use fsci_linalg::{lu, DecompOptions, LU_FACTOR_FLAT_MIN_OVERRIDE};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn main() {
    let mut seed = 11u64;
    let mut r = || { seed^=seed<<13; seed^=seed>>7; seed^=seed<<17; (seed>>11) as f64/(1u64<<53) as f64 - 0.5 };
    for &n in &[128usize, 256, 384, 512, 768] {
        let a: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let reps = 6;
        let bench = |ov: usize| { LU_FACTOR_FLAT_MIN_OVERRIDE.store(ov, Ordering::Relaxed);
            let _=black_box(lu(&a,DecompOptions::default()).unwrap());
            let t=Instant::now(); for _ in 0..reps { let _=black_box(lu(black_box(&a),DecompOptions::default()).unwrap()); }
            t.elapsed().as_secs_f64()/reps as f64*1000.0 };
        let nalg=bench(4096).min(bench(4096)); let blocked=bench(1).min(bench(1));
        println!("lu n={n:4}: nalgebra {nalg:8.2}ms -> blocked {blocked:8.2}ms = {:.2}x", nalg/blocked);
    }
    LU_FACTOR_FLAT_MIN_OVERRIDE.store(0, Ordering::Relaxed);
}
