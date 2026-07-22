//! Median-null-gated A/B for `linalg::transpose_rows`: ORIG serial strided-gather transpose vs
//! row-parallel (via linalg_par_matrix_rows). Toggled by `LINALG_MAT_ELEMENTWISE_FORCE_SERIAL`.
//! BYTE-IDENTICAL. Args: side [iters].
use fsci_linalg::{LINALG_MAT_ELEMENTWISE_FORCE_SERIAL, transpose_rows};
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
    let side: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);

    let mut s = 0xff51_afd7u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 10.0 - 5.0
    };
    let m: Vec<Vec<f64>> = (0..side)
        .map(|_| (0..side).map(|_| r()).collect())
        .collect();

    LINALG_MAT_ELEMENTWISE_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = transpose_rows(&m);
    LINALG_MAT_ELEMENTWISE_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = transpose_rows(&m);
    let bitmism: usize = a
        .iter()
        .zip(&b)
        .map(|(ra, rb)| {
            ra.iter()
                .zip(rb)
                .filter(|(u, v)| u.to_bits() != v.to_bits())
                .count()
        })
        .sum();
    println!(
        "# linalg::transpose_rows {side}x{side} a[1][0]={} b[1][0]={} bitmism={bitmism}",
        a[1][0], b[1][0]
    );

    let bench = |serial: bool| -> f64 {
        LINALG_MAT_ELEMENTWISE_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(transpose_rows(black_box(&m)));
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(transpose_rows(black_box(&m)));
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
    LINALG_MAT_ELEMENTWISE_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} transpose_rows serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
