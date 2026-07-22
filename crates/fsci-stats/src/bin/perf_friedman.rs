//! Median-null-gated A/B for `stats::friedmanchisquare` per-block scratch allocation:
//! the ORIG fresh per-block `Vec` alloc (n allocs) vs one hoisted+reused buffer (1 alloc).
//! Toggled by `FRIEDMAN_ALLOC_IN_LOOP`, alternated per iteration. BYTE-IDENTICAL (same
//! per-block contents/sort/rank accumulation). frankenscipy-26zjo. Args: n_blocks [iters] [k].
use fsci_stats::{FRIEDMAN_ALLOC_IN_LOOP, friedmanchisquare};
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
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let cols: Vec<Vec<f64>> = (0..k).map(|_| (0..n).map(|_| r()).collect()).collect();
    let groups: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();

    // Parity: statistic + pvalue + df must be bit-identical across arms.
    FRIEDMAN_ALLOC_IN_LOOP.store(true, Ordering::Relaxed);
    let a = friedmanchisquare(&groups);
    FRIEDMAN_ALLOC_IN_LOOP.store(false, Ordering::Relaxed);
    let b = friedmanchisquare(&groups);
    let bitmism = usize::from(a.statistic.to_bits() != b.statistic.to_bits())
        + usize::from(a.pvalue.to_bits() != b.pvalue.to_bits())
        + usize::from(a.df.to_bits() != b.df.to_bits());
    println!(
        "# stats::friedmanchisquare n_blocks={n} k={k} (stat serial={} hoisted={}) bitmism={bitmism}",
        a.statistic, b.statistic
    );

    let run = || friedmanchisquare(black_box(&groups));
    // alloc_in_loop=true is the ORIG (per-block alloc) baseline; false is the hoisted candidate.
    let bench = |alloc_in_loop: bool| -> f64 {
        FRIEDMAN_ALLOC_IN_LOOP.store(alloc_in_loop, Ordering::Relaxed);
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };

    let (mut ov, mut fv, mut nr, mut cr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true); // baseline: per-block alloc
        let f = bench(false); // candidate: hoisted
        let o2 = bench(true);
        nr.push(o / o2);
        cr.push(o / f);
        ov.push(o);
        fv.push(f);
    }
    FRIEDMAN_ALLOC_IN_LOOP.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} friedman per-block-alloc {ob:.2}ms (cv {:.1}%) hoisted {fb:.2}ms (cv {:.1}%) | \
         CAND(alloc/hoisted) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
