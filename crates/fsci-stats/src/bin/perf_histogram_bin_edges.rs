//! Median-null-gated A/B for `stats::histogram_bin_edges`: ORIG three separate passes (finite-check,
//! min fold, max fold) vs one fused pass. Method "sqrt" (nbins from n only, so those 3 are the only
//! data passes). Toggled by `HIST_EDGES_FUSE_DISABLE`. BYTE-IDENTICAL. Args: n [iters].
use fsci_stats::{HIST_EDGES_FUSE_DISABLE, histogram_bin_edges};
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

    let mut s = 0x9e37_79b9u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 100.0
    };
    let data: Vec<f64> = (0..n).map(|_| r()).collect();

    HIST_EDGES_FUSE_DISABLE.store(true, Ordering::Relaxed);
    let a = histogram_bin_edges(&data, "sqrt");
    HIST_EDGES_FUSE_DISABLE.store(false, Ordering::Relaxed);
    let b = histogram_bin_edges(&data, "sqrt");
    let bitmism = a
        .iter()
        .zip(&b)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
        + usize::from(a.len() != b.len());
    println!(
        "# stats::histogram_bin_edges n={n} edges={} first={} last={} bitmism={bitmism}",
        a.len(),
        a.first().copied().unwrap_or(0.0),
        a.last().copied().unwrap_or(0.0)
    );

    let bench = |disable: bool| -> f64 {
        HIST_EDGES_FUSE_DISABLE.store(disable, Ordering::Relaxed);
        let _ = black_box(histogram_bin_edges(black_box(&data), "sqrt"));
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(histogram_bin_edges(black_box(&data), "sqrt"));
        }
        t.elapsed().as_secs_f64() / 5.0 * 1e3
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
    HIST_EDGES_FUSE_DISABLE.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} histogram_bin_edges orig {ob:.2}ms (cv {:.1}%) fused {fb:.2}ms (cv {:.1}%) | \
         CAND(orig/fused) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
