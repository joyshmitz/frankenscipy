//! Median-null-gated A/B for `stats::logrank`: the ORIG per-time linear at-risk/death counts
//! (O(|times|*n)) vs pre-sorted binary search (O(n log n)). Toggled by `LOGRANK_FORCE_QUADRATIC`,
//! alternated per iteration. BYTE-IDENTICAL (same integer counts, same accumulation order over
//! `times`). Continuous data => all-distinct times => the quadratic path. Args: n [iters].
use fsci_stats::{LOGRANK_FORCE_QUADRATIC, logrank};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn med(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn cv(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64;
    if m > 0.0 { var.sqrt() / m * 100.0 } else { 0.0 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(11);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 1000.0
    };
    let x: Vec<f64> = (0..n).map(|_| r()).collect();
    let y: Vec<f64> = (0..n).map(|_| r()).collect();

    LOGRANK_FORCE_QUADRATIC.store(true, Ordering::Relaxed);
    let a = logrank(&x, &y, "two-sided");
    LOGRANK_FORCE_QUADRATIC.store(false, Ordering::Relaxed);
    let b = logrank(&x, &y, "two-sided");
    let bitmism = usize::from(a.statistic.to_bits() != b.statistic.to_bits())
        + usize::from(a.pvalue.to_bits() != b.pvalue.to_bits());
    println!("# stats::logrank n={n} (stat quad={} sorted={}) bitmism={bitmism}", a.statistic, b.statistic);

    let run = || logrank(black_box(&x), black_box(&y), "two-sided");
    let bench = |quad: bool| -> f64 {
        LOGRANK_FORCE_QUADRATIC.store(quad, Ordering::Relaxed);
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };

    let (mut ov, mut fv, mut nr, mut cr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true);
        let f = bench(false);
        let o2 = bench(true);
        nr.push(o / o2);
        cr.push(o / f);
        ov.push(o);
        fv.push(f);
    }
    LOGRANK_FORCE_QUADRATIC.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} logrank quadratic {ob:.2}ms (cv {:.1}%) sorted {fb:.2}ms (cv {:.1}%) | \
         CAND(quad/sorted) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
