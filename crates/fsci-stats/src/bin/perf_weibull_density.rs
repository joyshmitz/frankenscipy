use fsci_stats::{Weibull, ContinuousDistribution, WEIBULL_DENSITY_REUSE_DISABLE};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn main() {
    let n = 500_000usize;
    let mut s = 7u64;
    let mut r = || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (s>>11) as f64 / (1u64<<53) as f64 };
    let xs: Vec<f64> = (0..n).map(|_| 0.01 + 8.0*r()).collect();
    for &(c, scale) in &[(1.5f64, 2.0f64), (0.8, 1.0), (3.0, 1.5)] {
        let w = Weibull::new(c, scale);
        for which in ["pdf", "logpdf"] {
            let eval = |disable: bool| -> Vec<f64> {
                WEIBULL_DENSITY_REUSE_DISABLE.store(disable, Ordering::Relaxed);
                if which=="pdf" { xs.iter().map(|&x| w.pdf(x)).collect() } else { xs.iter().map(|&x| w.logpdf(x)).collect() }
            };
            let re = eval(false); let og = eval(true);
            let maxrel = re.iter().zip(&og).map(|(a,b)| if b.abs()>1e-300 {(a-b).abs()/b.abs()} else {0.0}).fold(0.0f64,f64::max);
            let bench = |disable: bool| {
                WEIBULL_DENSITY_REUSE_DISABLE.store(disable, Ordering::Relaxed);
                let f = || -> f64 { if which=="pdf" { xs.iter().map(|&x| w.pdf(x)).sum() } else { xs.iter().map(|&x| w.logpdf(x)).sum() } };
                let _ = black_box(f());
                let reps=6; let t=Instant::now(); for _ in 0..reps { let _=black_box(f()); } t.elapsed().as_secs_f64()/reps as f64*1000.0
            };
            let (mut ret, mut ogt) = (f64::MAX, f64::MAX);
            for _ in 0..4 { ogt=ogt.min(bench(true)); ret=ret.min(bench(false)); }
            println!("c={c} scale={scale} {which}: orig {ogt:.2}ms  reuse {ret:.2}ms  speedup {:.2}x  maxrel={maxrel:.1e}", ogt/ret);
        }
    }
}
