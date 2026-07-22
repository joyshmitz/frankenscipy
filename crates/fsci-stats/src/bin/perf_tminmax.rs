//! Median-null-gated A/B for `stats::tmin`/`tmax`: the ORIG collect-filtered-Vec + serial fold
//! vs the no-alloc parallel `par_filter_fold`. Toggled by `TMINMAX_FORCE_SERIAL`, alternated
//! per iteration. BYTE-IDENTICAL (same kept set/order; f64::min/max associative + signed-zero
//! total order; empty ⇒ NaN). Captures both the alloc-elimination and the parallelization.
//! Args: n [iters].
use fsci_stats::{TMINMAX_FORCE_SERIAL, tmax, tmin};
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

fn parity(name: &str, f: &dyn Fn() -> f64) -> usize {
    TMINMAX_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = f();
    TMINMAX_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = f();
    let m = usize::from(a.to_bits() != b.to_bits());
    println!("# {name}: serial={a} parallel={b} bitmism={m}");
    m
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
    // Mix in NaNs and out-of-range values so the finite+limit filter is exercised.
    let data: Vec<f64> = (0..n)
        .map(|i| if i % 991 == 0 { f64::NAN } else { r() })
        .collect();

    let lo = -10.0_f64;
    let hi = 10.0_f64;
    let mn: &dyn Fn() -> f64 = &|| tmin(&data, lo, true);
    let mx: &dyn Fn() -> f64 = &|| tmax(&data, hi, true);
    println!("# stats::tmin/tmax n={n} lo={lo} hi={hi} iters={iters}");
    let bitmism = parity("tmin", mn) + parity("tmax", mx);

    // Gate tmax (tmin shares par_filter_fold). Serial arm allocs; parallel arm is alloc-free.
    let run = || tmax(black_box(&data), black_box(hi), true);
    let bench = |force_serial: bool| -> f64 {
        TMINMAX_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
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
    TMINMAX_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} tmax orig(collect+fold) {ob:.2}ms (cv {:.1}%) noalloc-parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(orig/new) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
