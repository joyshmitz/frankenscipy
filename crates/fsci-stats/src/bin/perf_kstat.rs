//! Median-null-gated A/B for `stats::kstat`: the ORIG collect-filtered-Vec + four separate
//! power-sum passes vs one inline pass folding all four sums (no allocation). Toggled by
//! `KSTAT_FORCE_COLLECT`, alternated per iteration. BYTE-IDENTICAL (independent accumulators,
//! same left folds over the same finite elements in data order). Args: n [iters] [order].
use fsci_stats::{KSTAT_FORCE_COLLECT, kstat};
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
        .unwrap_or(32_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let order: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 10.0 - 5.0
    };
    let data: Vec<f64> = (0..n)
        .map(|i| if i % 991 == 0 { f64::INFINITY } else { r() })
        .collect();

    KSTAT_FORCE_COLLECT.store(true, Ordering::Relaxed);
    let a = kstat(&data, order);
    KSTAT_FORCE_COLLECT.store(false, Ordering::Relaxed);
    let b = kstat(&data, order);
    let bitmism = usize::from(a.to_bits() != b.to_bits());
    println!("# stats::kstat n={n} order={order} (collect={a} inline={b}) bitmism={bitmism}");

    let run = || kstat(black_box(&data), black_box(order));
    let bench = |collect: bool| -> f64 {
        KSTAT_FORCE_COLLECT.store(collect, Ordering::Relaxed);
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
    KSTAT_FORCE_COLLECT.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} kstat collect+4pass {ob:.2}ms (cv {:.1}%) inline-1pass {fb:.2}ms (cv {:.1}%) | \
         CAND(collect/inline) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
