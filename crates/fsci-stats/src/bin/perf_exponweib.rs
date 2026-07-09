use fsci_stats::{ContinuousDistribution, ExponWeibull, WEIBULL_DENSITY_REUSE_DISABLE};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn main() {
    let n = 500_000usize;
    let mut s = 5u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let xs: Vec<f64> = (0..n).map(|_| 0.05 + 5.0 * r()).collect();
    for &(a, c) in &[(2.0f64, 1.5f64), (1.3, 0.9), (2.5, 2.2)] {
        let d = ExponWeibull::new(a, c);
        for which in ["pdf", "logpdf"] {
            let eval = |disable: bool| -> Vec<f64> {
                WEIBULL_DENSITY_REUSE_DISABLE.store(disable, Ordering::Relaxed);
                if which == "pdf" {
                    xs.iter().map(|&x| d.pdf(x)).collect()
                } else {
                    xs.iter().map(|&x| d.logpdf(x)).collect()
                }
            };
            let re = eval(false);
            let og = eval(true);
            let mr = re
                .iter()
                .zip(&og)
                .map(|(a, b)| {
                    if b.is_finite() && b.abs() > 1e-300 {
                        (a - b).abs() / b.abs()
                    } else {
                        0.0
                    }
                })
                .fold(0.0f64, f64::max);
            let bench = |disable: bool| {
                WEIBULL_DENSITY_REUSE_DISABLE.store(disable, Ordering::Relaxed);
                let f = || -> f64 {
                    if which == "pdf" {
                        xs.iter().map(|&x| d.pdf(x)).filter(|v| v.is_finite()).sum()
                    } else {
                        xs.iter()
                            .map(|&x| d.logpdf(x))
                            .filter(|v| v.is_finite())
                            .sum()
                    }
                };
                let _ = black_box(f());
                let reps = 6;
                let t = Instant::now();
                for _ in 0..reps {
                    let _ = black_box(f());
                }
                t.elapsed().as_secs_f64() / reps as f64 * 1000.0
            };
            let (mut ret, mut ogt) = (f64::MAX, f64::MAX);
            for _ in 0..4 {
                ogt = ogt.min(bench(true));
                ret = ret.min(bench(false));
            }
            println!(
                "a={a} c={c} {which}: orig {ogt:.2}ms reuse {ret:.2}ms  {:.2}x  maxrel={mr:.1e}",
                ogt / ret
            );
        }
    }
}
