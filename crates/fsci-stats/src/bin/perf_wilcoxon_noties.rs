//! Median-null-gated A/B for `wilcoxon`: ORIG eager `no_ties` clone+sort of abs_diffs vs the
//! lazy form gated behind `nr <= 1000` (skipped on the large-n normal-approx path). Toggled by
//! `WILCOXON_FORCE_EAGER_NOTIES`, alternated per iteration. BYTE-IDENTICAL. Args: n [iters].
use fsci_stats::{WILCOXON_FORCE_EAGER_NOTIES, wilcoxon};
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
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(400_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);

    // Distinct non-zero diffs (no dropped zeros, no ties) so `no_zeros` is true and the eager
    // path pays the full sort; nr > 1000 forces the normal-approximation fallback either way.
    let mut s = 0x1234_5678u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let x: Vec<f64> = (0..n).map(|i| i as f64 + r()).collect();
    let y: Vec<f64> = (0..n).map(|_| r() * 0.5).collect();

    WILCOXON_FORCE_EAGER_NOTIES.store(true, Ordering::Relaxed);
    let a = wilcoxon(&x, &y);
    WILCOXON_FORCE_EAGER_NOTIES.store(false, Ordering::Relaxed);
    let b = wilcoxon(&x, &y);
    let bitmism = usize::from(a.statistic.to_bits() != b.statistic.to_bits())
        + usize::from(a.pvalue.to_bits() != b.pvalue.to_bits());
    println!("# wilcoxon n={n} orig(stat={},p={}) lazy(stat={},p={}) bitmism={bitmism}",
        a.statistic, a.pvalue, b.statistic, b.pvalue);

    let bench = |eager: bool| -> f64 {
        WILCOXON_FORCE_EAGER_NOTIES.store(eager, Ordering::Relaxed);
        let _ = black_box(wilcoxon(black_box(&x), black_box(&y)));
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(wilcoxon(black_box(&x), black_box(&y)));
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };

    let (mut ov, mut fv, mut nr_, mut cr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true);
        let f = bench(false);
        let o2 = bench(true);
        nr_.push(o / o2);
        cr.push(o / f);
        ov.push(o);
        fv.push(f);
    }
    WILCOXON_FORCE_EAGER_NOTIES.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr_.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr_.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} wilcoxon eager {ob:.2}ms (cv {:.1}%) lazy {fb:.2}ms (cv {:.1}%) | \
         CAND(eager/lazy) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
