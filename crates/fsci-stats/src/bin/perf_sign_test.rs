//! Median-null-gated A/B for `stats::sign_test`: the ORIG collect-diffs-Vec + two count passes
//! vs one alloc-free zip pass counting positives/negatives. Toggled by `SIGN_TEST_FUSE_DISABLE`,
//! alternated per iteration. BYTE-IDENTICAL (exact integer counts, order-independent). Args: n [iters].
use fsci_stats::{SIGN_TEST_FUSE_DISABLE, sign_test};
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
    let x: Vec<f64> = (0..n).map(|_| r()).collect();
    let y: Vec<f64> = (0..n).map(|_| r()).collect();

    SIGN_TEST_FUSE_DISABLE.store(true, Ordering::Relaxed);
    let a = sign_test(&x, &y).unwrap();
    SIGN_TEST_FUSE_DISABLE.store(false, Ordering::Relaxed);
    let b = sign_test(&x, &y).unwrap();
    let bitmism = usize::from(a.statistic.to_bits() != b.statistic.to_bits())
        + usize::from(a.pvalue.to_bits() != b.pvalue.to_bits());
    println!(
        "# stats::sign_test n={n} (stat 2pass={} fused={}) bitmism={bitmism}",
        a.statistic, b.statistic
    );

    let run = || sign_test(black_box(&x), black_box(&y)).unwrap();
    let bench = |disable: bool| -> f64 {
        SIGN_TEST_FUSE_DISABLE.store(disable, Ordering::Relaxed);
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
    SIGN_TEST_FUSE_DISABLE.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} sign_test collect+2count {ob:.2}ms (cv {:.1}%) fused-1pass {fb:.2}ms (cv {:.1}%) | \
         CAND(orig/fused) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
