//! Median-null-gated A/B for the non-reflect spline prefilter (Constant/Wrap/Mirror + general banded
//! kernel) inside `prefilter_spline_coefficients`, driven through `zoom` (order>=2 prefilters): the
//! ORIG serial per-line walk vs the parallel-across-outer-slabs path. Both arms live in ONE binary,
//! toggled by `NDIMAGE_SPLINE_PREFILTER_FORCE_SERIAL` and ALTERNATED per iteration inside one measured
//! routine, so a single `rch exec` invocation measures both on the same worker.
use fsci_ndimage::{BoundaryMode, NDIMAGE_SPLINE_PREFILTER_FORCE_SERIAL, NdArray, zoom};
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
    let d0: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(600);
    let d1: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(600);
    let d2: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(600);
    let order: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);
    let iters: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(13);

    let total = d0 * d1 * d2;
    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 10.0 - 5.0
    };
    let data: Vec<f64> = (0..total).map(|_| r()).collect();
    let input = NdArray::new(data, vec![d0, d1, d2]).unwrap();
    let mode = BoundaryMode::Mirror;
    let factors = [1.0, 1.0, 1.0];

    // Parity: parallel prefilter must be byte-identical to serial.
    NDIMAGE_SPLINE_PREFILTER_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = zoom(&input, &factors, order, mode, 0.0).unwrap();
    NDIMAGE_SPLINE_PREFILTER_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = zoom(&input, &factors, order, mode, 0.0).unwrap();
    let bitmism = a
        .data
        .iter()
        .zip(&b.data)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();

    let bench = |force_serial: bool| -> f64 {
        NDIMAGE_SPLINE_PREFILTER_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || zoom(black_box(&input), &factors, order, mode, 0.0).unwrap();
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };

    let (mut ov, mut fv) = (Vec::new(), Vec::new());
    let (mut null_r, mut cand_r) = (Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true);
        let f = bench(false);
        let o2 = bench(true);
        null_r.push(o / o2);
        cand_r.push(o / f);
        ov.push(o);
        fv.push(f);
    }
    NDIMAGE_SPLINE_PREFILTER_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = median(&mut nr);
    let cand_med = median(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# zoom spline prefilter {d0}x{d1}x{d2} order={order} mode=Mirror (prefilter A/B)");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
