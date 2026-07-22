//! Median-null-gated A/B for `stats::zmap_weighted`: ORIG separate finite-check + Σw + weighted-mean
//! passes over (compare,weights) vs one fused pass. Toggled by `ZSCORE_W_FUSE_DISABLE` (shared with
//! zscore_weighted — same fold), alternated per iteration. BYTE-IDENTICAL. Args: n [iters].
use fsci_stats::{ZSCORE_W_FUSE_DISABLE, zmap_weighted};
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
        .unwrap_or(16_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);

    let mut s = 0xb7e1_5163u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let compare: Vec<f64> = (0..n).map(|_| r() * 20.0 - 10.0).collect();
    let w: Vec<f64> = (0..n).map(|_| r() + 0.1).collect();
    let scores: Vec<f64> = (0..4096).map(|_| r() * 20.0 - 10.0).collect();

    ZSCORE_W_FUSE_DISABLE.store(true, Ordering::Relaxed);
    let a = zmap_weighted(&scores, &compare, &w);
    ZSCORE_W_FUSE_DISABLE.store(false, Ordering::Relaxed);
    let b = zmap_weighted(&scores, &compare, &w);
    let bitmism = a
        .iter()
        .zip(&b)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();
    println!(
        "# stats::zmap_weighted compare_n={n} scores={} orig[1]={} fused[1]={} bitmism={bitmism}",
        scores.len(),
        a[1],
        b[1]
    );

    let bench = |disable: bool| -> f64 {
        ZSCORE_W_FUSE_DISABLE.store(disable, Ordering::Relaxed);
        let _ = black_box(zmap_weighted(
            black_box(&scores),
            black_box(&compare),
            black_box(&w),
        ));
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(zmap_weighted(
                black_box(&scores),
                black_box(&compare),
                black_box(&w),
            ));
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
    ZSCORE_W_FUSE_DISABLE.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} zmap_weighted orig {ob:.2}ms (cv {:.1}%) fused {fb:.2}ms (cv {:.1}%) | \
         CAND(orig/fused) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
