//! Median-null-gated A/B for `stats::min_max_scale`: ORIG serial min/max folds + serial scale map vs
//! chunked-parallel min/max reduce + parallel output map. Toggled by `MIN_MAX_SCALE_FORCE_SERIAL`,
//! alternated per iteration. BYTE-IDENTICAL. Args: n [iters].
use fsci_stats::{MIN_MAX_SCALE_FORCE_SERIAL, min_max_scale};
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

    let mut s = 0x1a2b_3c4du64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 100.0 - 50.0
    };
    let data: Vec<f64> = (0..n).map(|_| r()).collect();

    MIN_MAX_SCALE_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = min_max_scale(&data, None);
    MIN_MAX_SCALE_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = min_max_scale(&data, None);
    let bitmism = a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count()
        + usize::from(a.len() != b.len());
    println!("# stats::min_max_scale n={n} a[1]={} b[1]={} bitmism={bitmism}", a[1], b[1]);

    let bench = |serial: bool| -> f64 {
        MIN_MAX_SCALE_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(min_max_scale(black_box(&data), None));
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(min_max_scale(black_box(&data), None));
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
    MIN_MAX_SCALE_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} min_max_scale serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
