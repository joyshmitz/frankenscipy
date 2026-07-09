// A/B: cholesky blocked (parallel SYRK) vs unblocked left-looking SIMD at medium n.
use fsci_linalg::{cholesky, DecompOptions, CHOL_FACTOR_FLAT_MIN_OVERRIDE};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn main() {
    let mut seed = 7u64;
    let mut r = || { seed^=seed<<13; seed^=seed>>7; seed^=seed<<17; (seed>>11) as f64/(1u64<<53) as f64 };
    for &n in &[128usize, 256, 384, 512, 768] {
        let m: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()-0.5).collect()).collect();
        let mut a = vec![vec![0.0;n];n];
        for i in 0..n { for j in 0..n { let mut s=0.0; for k in 0..n { s+=m[i][k]*m[j][k]; } a[i][j]=s/n as f64 + if i==j {n as f64} else {0.0}; } }
        CHOL_FACTOR_FLAT_MIN_OVERRIDE.store(4096, Ordering::Relaxed);
        let cs = cholesky(&a, true, DecompOptions::default()).unwrap().factor;
        CHOL_FACTOR_FLAT_MIN_OVERRIDE.store(1, Ordering::Relaxed);
        let cb = cholesky(&a, true, DecompOptions::default()).unwrap().factor;
        let maxdiff = cs.iter().zip(&cb).flat_map(|(pr,br)| pr.iter().zip(br)).map(|(a,b)| (a-b).abs()).fold(0.0f64,f64::max);
        let reps = 6;
        let bench = |ov: usize| { CHOL_FACTOR_FLAT_MIN_OVERRIDE.store(ov, Ordering::Relaxed);
            let _=black_box(cholesky(&a,true,DecompOptions::default()).unwrap());
            let t=Instant::now(); for _ in 0..reps { let _=black_box(cholesky(black_box(&a),true,DecompOptions::default()).unwrap()); }
            t.elapsed().as_secs_f64()/reps as f64*1000.0 };
        let simd=bench(4096).min(bench(4096)); let blocked=bench(1).min(bench(1));
        println!("chol n={n:4}: simd {simd:8.2}ms -> blocked {blocked:8.2}ms = {:.2}x  maxdiff={maxdiff:.1e}", simd/blocked);
    }
    CHOL_FACTOR_FLAT_MIN_OVERRIDE.store(0, Ordering::Relaxed);
}
