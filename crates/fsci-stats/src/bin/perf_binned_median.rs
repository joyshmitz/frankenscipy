//! Median-null-gated A/B for `stats::binned_statistic` with statistic="median": the ORIG per-bin
//! full O(bin log bin) sort vs the O(bin) quickselect (`median_in_place`). Toggled by
//! `BINNED_MEDIAN_FORCE_SORT`, alternated per iteration. BYTE-IDENTICAL (same central order
//! statistics). Args: n [iters] [bins].
use fsci_stats::{BINNED_MEDIAN_FORCE_SORT, binned_statistic};
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
    let bins: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let x: Vec<f64> = (0..n).map(|_| r()).collect(); // bin coordinate in [0,1)
    let values: Vec<f64> = (0..n).map(|_| r() * 100.0 - 50.0).collect();

    BINNED_MEDIAN_FORCE_SORT.store(true, Ordering::Relaxed);
    let a = binned_statistic(&x, &values, bins, "median").0;
    BINNED_MEDIAN_FORCE_SORT.store(false, Ordering::Relaxed);
    let b = binned_statistic(&x, &values, bins, "median").0;
    let bitmism = a
        .iter()
        .zip(&b)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();
    println!(
        "# stats::binned_statistic median n={n} bins={bins} (m[0] {}/{}) bitmism={bitmism}",
        a[0], b[0]
    );

    let run = || binned_statistic(black_box(&x), black_box(&values), bins, "median").0;
    let bench = |force_sort: bool| -> f64 {
        BINNED_MEDIAN_FORCE_SORT.store(force_sort, Ordering::Relaxed);
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
    BINNED_MEDIAN_FORCE_SORT.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} binned_statistic-median full-sort {ob:.2}ms (cv {:.1}%) quickselect {fb:.2}ms (cv {:.1}%) | \
         CAND(sort/select) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
