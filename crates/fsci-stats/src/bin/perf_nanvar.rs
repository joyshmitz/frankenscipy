//! Median-null-gated A/B for `stats::nanvar` (backs `nanstd`): the ORIG collect-filtered-Vec +
//! two Vec passes vs two inline filter passes with no allocation. Toggled by
//! `NANVAR_FORCE_COLLECT`, alternated per iteration. BYTE-IDENTICAL (each pass reduces the same
//! non-NaN elements in data order; mean and SS are the same left folds). Args: n [iters].
use fsci_stats::{NANVAR_FORCE_COLLECT, nanvar};
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

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 100.0 - 50.0
    };
    let data: Vec<f64> = (0..n)
        .map(|i| if i % 991 == 0 { f64::NAN } else { r() })
        .collect();

    NANVAR_FORCE_COLLECT.store(true, Ordering::Relaxed);
    let a = nanvar(&data);
    NANVAR_FORCE_COLLECT.store(false, Ordering::Relaxed);
    let b = nanvar(&data);
    let bitmism = usize::from(a.to_bits() != b.to_bits());
    println!("# stats::nanvar n={n} (collect={a} inline={b}) bitmism={bitmism}");

    let run = || nanvar(black_box(&data));
    let bench = |collect: bool| -> f64 {
        NANVAR_FORCE_COLLECT.store(collect, Ordering::Relaxed);
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
    NANVAR_FORCE_COLLECT.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} nanvar collect+2reads {ob:.2}ms (cv {:.1}%) inline-2pass {fb:.2}ms (cv {:.1}%) | \
         CAND(collect/inline) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
