//! Median-null-gated A/B for `RegularGridInterpolator::eval_many` 3-D nearest fast path: ORIG serial
//! per-query loop vs par_query_try_map fan-out. BYTE-IDENTICAL (each query is an independent
//! read-only nearest lookup). Toggled by `INTERPN_NEAREST3D_FORCE_SERIAL`. Args: nq [axis_len] [iters].
use fsci_interpolate::{
    INTERPN_NEAREST3D_FORCE_SERIAL, RegularGridInterpolator, RegularGridMethod,
};
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
    let nq: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    let axis_len: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);

    // Three NON-uniform axes (quadratic spacing) so nearest_index takes the O(log n) binary-search
    // path per axis -> real per-query work.
    let axis: Vec<f64> = (0..axis_len)
        .map(|i| {
            let t = i as f64 / (axis_len - 1) as f64;
            t * t * 10.0
        })
        .collect();
    let points = vec![axis.clone(), axis.clone(), axis.clone()];
    let nvals = axis_len * axis_len * axis_len;
    let values: Vec<f64> = (0..nvals).map(|i| (i as f64).sin()).collect();
    let interp =
        RegularGridInterpolator::new(points, values, RegularGridMethod::Nearest, false, Some(0.0))
            .expect("interp");

    // Random in-bounds 3-D queries.
    let mut s = 0x2f6e_1122u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 10.0
    };
    let queries: Vec<Vec<f64>> = (0..nq).map(|_| vec![r(), r(), r()]).collect();

    INTERPN_NEAREST3D_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = interp.eval_many(&queries).expect("eval");
    INTERPN_NEAREST3D_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = interp.eval_many(&queries).expect("eval");
    let bitmism = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    println!(
        "# interp::eval_many(nearest3d) nq={nq} axis_len={axis_len} a[1]={} bitmism={bitmism}",
        a[1]
    );

    let bench = |serial: bool| -> f64 {
        INTERPN_NEAREST3D_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(interp.eval_many(black_box(&queries)).unwrap());
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(interp.eval_many(black_box(&queries)).unwrap());
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
    INTERPN_NEAREST3D_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} nearest3d serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
