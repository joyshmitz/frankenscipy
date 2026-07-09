use fsci_stats::{GenGamma, GENGAMMA_LN_REUSE_DISABLE};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn main() {
    let n = 500_000usize;
    let mut s = 3u64;
    let mut r = || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (s>>11) as f64 / (1u64<<53) as f64 };
    let xs: Vec<f64> = (0..n).map(|_| 0.01 + 12.0*r()).collect();
    for &(a, c) in &[(2.5f64, 1.7f64), (1.3, 0.8), (3.1, 2.4)] {
        let g = GenGamma::new(a, c);
        let run = |disable: bool| { GENGAMMA_LN_REUSE_DISABLE.store(disable, Ordering::Relaxed); g.pdf_many(&xs) };
        let reuse = run(false); let orig = run(true);
        let maxrel = reuse.iter().zip(&orig).map(|(a,b)| if b.abs()>1e-300 {(a-b).abs()/b.abs()} else {0.0}).fold(0.0f64,f64::max);
        let bench = |disable: bool| {
            GENGAMMA_LN_REUSE_DISABLE.store(disable, Ordering::Relaxed);
            let _ = black_box(g.pdf_many(&xs));
            let reps = 6; let t = Instant::now();
            for _ in 0..reps { let _ = black_box(g.pdf_many(black_box(&xs))); }
            t.elapsed().as_secs_f64()/reps as f64*1000.0
        };
        let (mut re, mut og) = (f64::MAX, f64::MAX);
        for _ in 0..4 { og = og.min(bench(true)); re = re.min(bench(false)); }
        println!("a={a} c={c}: orig(powf) {og:.2}ms  reuse(ln) {re:.2}ms  speedup {:.2}x  maxrel={maxrel:.1e}", og/re);
    }
}
