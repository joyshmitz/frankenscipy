//! Median-null-gated A/B for `RectBivariateSpline::eval_many` (scattered points): the ORIG per-query
//! path (rebuilds the ny x-direction BSplines on every query) vs the hoisted path (builds them once,
//! reuses across the batch). Both arms live in ONE binary, toggled by
//! `RECTBISPLINE_EVAL_MANY_FORCE_SCALAR` and ALTERNATED per iteration inside one measured routine, so
//! a single `rch exec` invocation measures both on the same worker.
use fsci_interpolate::{RECTBISPLINE_EVAL_MANY_FORCE_SCALAR, RectBivariateSpline};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn median(v: &mut [f64]) -> f64 {
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
    let side: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let nq: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20000);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(13);

    let x: Vec<f64> = (0..side).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..side).map(|j| j as f64).collect();
    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let z: Vec<Vec<f64>> = (0..side)
        .map(|_| (0..side).map(|_| r() * 10.0 - 5.0).collect())
        .collect();
    let spline = RectBivariateSpline::new(&x, &y, &z, 3, 3).unwrap();
    let hi = (side - 1) as f64;
    let qx: Vec<f64> = (0..nq).map(|_| r() * hi).collect();
    let qy: Vec<f64> = (0..nq).map(|_| r() * hi).collect();

    // Parity: hoisted must be byte-identical to the per-query path.
    RECTBISPLINE_EVAL_MANY_FORCE_SCALAR.store(true, Ordering::Relaxed);
    let a = spline.eval_many(&qx, &qy).unwrap();
    RECTBISPLINE_EVAL_MANY_FORCE_SCALAR.store(false, Ordering::Relaxed);
    let b = spline.eval_many(&qx, &qy).unwrap();
    let bitmism = a
        .iter()
        .zip(&b)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();

    let bench = |force_scalar: bool| -> f64 {
        RECTBISPLINE_EVAL_MANY_FORCE_SCALAR.store(force_scalar, Ordering::Relaxed);
        let run = || spline.eval_many(black_box(&qx), black_box(&qy)).unwrap();
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };

    let (mut ov, mut hv) = (Vec::new(), Vec::new());
    let (mut null_r, mut cand_r) = (Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true);
        let h = bench(false);
        let o2 = bench(true);
        null_r.push(o / o2);
        cand_r.push(o / h);
        ov.push(o);
        hv.push(h);
    }
    RECTBISPLINE_EVAL_MANY_FORCE_SCALAR.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = median(&mut nr);
    let cand_med = median(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let hb = hv.iter().copied().fold(f64::MAX, f64::min);
    println!("# RectBivariateSpline eval_many {side}x{side} grid, {nq} scattered queries");
    println!(
        "{} scalar {ob:.2}ms (cv {:.1}%) hoisted {hb:.2}ms (cv {:.1}%) | CAND(scalar/hoisted) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&hv),
    );
}
