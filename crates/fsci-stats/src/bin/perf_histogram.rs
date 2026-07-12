//! Median-null-gated A/B for `stats::histogram`: the ORIG three separate finite/min/max
//! passes vs the single fused traversal (the binning pass is unchanged). Toggled by
//! `HISTOGRAM_FUSE_DISABLE`, alternated per iteration. BYTE-IDENTICAL (same NaN-aware
//! min/max, same empty-on-non-finite). Args: n [iters] [bins].
use fsci_stats::{HISTOGRAM_FUSE_DISABLE, histogram};
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
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let bins: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 100.0 - 50.0
    };
    let data: Vec<f64> = (0..n).map(|_| r()).collect();

    // Parity: counts + bin edges must be bit-identical across arms.
    HISTOGRAM_FUSE_DISABLE.store(true, Ordering::Relaxed);
    let (ca, ea) = histogram(&data, bins);
    HISTOGRAM_FUSE_DISABLE.store(false, Ordering::Relaxed);
    let (cb, eb) = histogram(&data, bins);
    let bitmism = usize::from(ca != cb)
        + ea.iter().zip(&eb).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    println!("# stats::histogram n={n} bins={bins} (count[0] {}/{}) bitmism={bitmism}", ca[0], cb[0]);

    let run = || histogram(black_box(&data), black_box(bins));
    // disable=true is the ORIG (three-pass) baseline; false is the fused candidate.
    let bench = |disable: bool| -> f64 {
        HISTOGRAM_FUSE_DISABLE.store(disable, Ordering::Relaxed);
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
    HISTOGRAM_FUSE_DISABLE.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} histogram 3-pass {ob:.2}ms (cv {:.1}%) fused {fb:.2}ms (cv {:.1}%) | \
         CAND(3pass/fused) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
