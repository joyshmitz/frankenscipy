//! Median-null-gated A/B for `ndimage::array_max`: ORIG serial NaN-aware fold vs chunked parallel
//! reduction. Toggled by `NDIMAGE_ARRAY_MINMAX_FORCE_SERIAL`, alternated per iteration.
//! BYTE-IDENTICAL. Args: n [iters].
use fsci_ndimage::{NDIMAGE_ARRAY_MINMAX_FORCE_SERIAL, NdArray, array_max, array_min};
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

    let mut s = 0x243f_6a88u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let arr = NdArray::new((0..n).map(|_| r()).collect(), vec![n]).unwrap();

    NDIMAGE_ARRAY_MINMAX_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let (a_max, a_min) = (array_max(&arr), array_min(&arr));
    NDIMAGE_ARRAY_MINMAX_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (b_max, b_min) = (array_max(&arr), array_min(&arr));
    let bitmism = usize::from(a_max.to_bits() != b_max.to_bits())
        + usize::from(a_min.to_bits() != b_min.to_bits());
    println!("# ndimage::array_max/min n={n} max({a_max}=={b_max}) min({a_min}=={b_min}) bitmism={bitmism}");

    let bench = |serial: bool| -> f64 {
        NDIMAGE_ARRAY_MINMAX_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(array_max(black_box(&arr)));
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(array_max(black_box(&arr)));
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
    NDIMAGE_ARRAY_MINMAX_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} array_max serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
