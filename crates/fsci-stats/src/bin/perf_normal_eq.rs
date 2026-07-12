//! Median-null-gated A/B for `multiple_regression`'s normal-equation build (small-p/large-n now
//! routed through the transposed XtX path): ORIG serial row fill vs row-fan parallel. BYTE-IDENTICAL
//! (streamed dots in sample order, symmetric mirror; verified via coefficients). Toggled by
//! `NORMAL_EQ_FORCE_SERIAL`. Args: npts [p] [iters].
use fsci_stats::{NORMAL_EQ_FORCE_SERIAL, multiple_regression};
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
    let npts: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_500_000);
    let p: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);

    let mut s = 0x2ad9_c15bu64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0
    };
    let x: Vec<Vec<f64>> = (0..npts).map(|_| (0..p).map(|_| r()).collect()).collect();
    let y: Vec<f64> = (0..npts).map(|_| r()).collect();

    NORMAL_EQ_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let (ba, _, _, _) = multiple_regression(&x, &y);
    NORMAL_EQ_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (bb, _, _, _) = multiple_regression(&x, &y);
    let bitmism = ba.iter().zip(&bb).filter(|(u, v)| u.to_bits() != v.to_bits()).count();
    println!("# stats::multiple_regression npts={npts} p={p} beta[0]={} bitmism={bitmism}", ba[0]);

    let bench = |serial: bool| -> f64 {
        NORMAL_EQ_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(multiple_regression(black_box(&x), black_box(&y)));
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(multiple_regression(black_box(&x), black_box(&y)));
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
    NORMAL_EQ_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} normal_eq serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
