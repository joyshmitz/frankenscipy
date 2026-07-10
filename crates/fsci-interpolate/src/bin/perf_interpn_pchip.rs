//! Median-null-gated A/B for `RegularGridInterpolator::eval_many` on PCHIP: the ORIG per-query path
//! (re-fits the last-axis fibers on every query) vs the hoisted-fit batch path (fits them once).
//! Both arms live in ONE binary, toggled by `INTERPN_PCHIP_BATCH_FORCE_SCALAR` and ALTERNATED per
//! iteration in one measured routine, so a single `rch exec` invocation measures both on the same
//! worker. Prints the CAND median (scalar/hoisted) against an A/A null on the scalar arm.
use fsci_interpolate::{
    INTERPN_PCHIP_BATCH_FORCE_SCALAR, RegularGridInterpolator, RegularGridMethod,
};
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
    let side: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let nq: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(13);

    let ax: Vec<f64> = (0..side).map(|i| i as f64).collect();
    let points = vec![ax.clone(), ax.clone()];
    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let values: Vec<f64> = (0..side * side).map(|_| r() * 10.0 - 5.0).collect();
    let interp =
        RegularGridInterpolator::new(points, values, RegularGridMethod::Pchip, false, Some(0.0))
            .unwrap();
    let hi = (side - 1) as f64;
    let queries: Vec<Vec<f64>> = (0..nq).map(|_| vec![r() * hi, r() * hi]).collect();

    // Parity: hoisted must be byte-identical to the per-query path.
    INTERPN_PCHIP_BATCH_FORCE_SCALAR.store(true, Ordering::Relaxed);
    let a = interp.eval_many(&queries).unwrap();
    INTERPN_PCHIP_BATCH_FORCE_SCALAR.store(false, Ordering::Relaxed);
    let b = interp.eval_many(&queries).unwrap();
    let bitmism = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();

    let bench = |force_scalar: bool| -> f64 {
        INTERPN_PCHIP_BATCH_FORCE_SCALAR.store(force_scalar, Ordering::Relaxed);
        let run = || interp.eval_many(black_box(&queries)).unwrap();
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };

    // Interleave scalar / hoisted / scalar-again (A/A null on the scalar arm).
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
    INTERPN_PCHIP_BATCH_FORCE_SCALAR.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = median(&mut nr);
    let cand_med = median(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let hb = hv.iter().copied().fold(f64::MAX, f64::min);
    println!("# interpn pchip eval_many {side}x{side} grid, {nq} queries");
    println!(
        "{} scalar {ob:.2}ms (cv {:.1}%) hoisted {hb:.2}ms (cv {:.1}%) | CAND(scalar/hoisted) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&hv),
    );
}
