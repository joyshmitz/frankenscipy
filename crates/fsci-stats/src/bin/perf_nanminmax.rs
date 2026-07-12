//! Median-null-gated A/B for `stats::nanmin`/`nanmax`: the ORIG serial NaN-filtered fold vs
//! the parallel chunk+merge (`par_nan_fold`). Toggled by `NANMINMAX_FORCE_SERIAL`, alternated
//! per iteration. BYTE-IDENTICAL (f64::min/max associative + signed-zero total order; NaN
//! filtered per chunk). Args: n [iters].
use fsci_stats::{NANMINMAX_FORCE_SERIAL, nanmax, nanmin};
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
    NANMINMAX_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = f();
    NANMINMAX_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = f();
    let m = usize::from(a.to_bits() != b.to_bits());
    println!("# {name}: serial={a} parallel={b} bitmism={m}");
    m
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(32_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 100.0 - 50.0
    };
    // Sprinkle NaNs so the filter branch is exercised.
    let data: Vec<f64> = (0..n)
        .map(|i| if i % 997 == 0 { f64::NAN } else { r() })
        .collect();

    let mn: &dyn Fn() -> f64 = &|| nanmin(&data);
    let mx: &dyn Fn() -> f64 = &|| nanmax(&data);
    println!("# stats::nanmin/nanmax n={n} iters={iters}");
    let bitmism = parity("nanmin", mn) + parity("nanmax", mx);

    // Gate nanmax (nanmin shares par_nan_fold).
    let run = || nanmax(black_box(&data));
    let bench = |force_serial: bool| -> f64 {
        NANMINMAX_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
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
    NANMINMAX_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} nanmax serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/parallel) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
