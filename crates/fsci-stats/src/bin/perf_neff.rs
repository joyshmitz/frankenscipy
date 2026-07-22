//! Median-null-gated A/B for `stats::neff`: ORIG three separate passes (finite-check, Σw, Σw²) vs one
//! fused pass. Toggled by `NEFF_FUSE_DISABLE`, alternated per iteration. BYTE-IDENTICAL. Args: n [iters].
use fsci_stats::{NEFF_FUSE_DISABLE, neff};
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

    let mut s = 0x62a9_d9e5u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 + 0.05
    };
    let w: Vec<f64> = (0..n).map(|_| r()).collect();

    NEFF_FUSE_DISABLE.store(true, Ordering::Relaxed);
    let a = neff(&w);
    NEFF_FUSE_DISABLE.store(false, Ordering::Relaxed);
    let b = neff(&w);
    let bitmism = usize::from(a.to_bits() != b.to_bits());
    println!("# stats::neff n={n} orig={a} fused={b} bitmism={bitmism}");

    let bench = |disable: bool| -> f64 {
        NEFF_FUSE_DISABLE.store(disable, Ordering::Relaxed);
        let _ = black_box(neff(black_box(&w)));
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(neff(black_box(&w)));
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
    NEFF_FUSE_DISABLE.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} neff orig {ob:.2}ms (cv {:.1}%) fused {fb:.2}ms (cv {:.1}%) | \
         CAND(orig/fused) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
