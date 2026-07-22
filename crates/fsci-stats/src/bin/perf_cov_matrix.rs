//! Median-null-gated A/B for `cov_matrix` on small-d/large-n (now routed through the transposed
//! cross-covariance path): ORIG serial row fill vs row-fan parallel. BYTE-IDENTICAL (streamed dots
//! in obs order, symmetric mirror). Toggled by `COV_MATRIX_FORCE_SERIAL`. Args: npts [dims] [iters].
use fsci_stats::{COV_MATRIX_FORCE_SERIAL, cov_matrix};
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
    let npts: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500_000);
    let dims: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);

    let mut s = 0x4c1f_ab77u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0
    };
    let data: Vec<Vec<f64>> = (0..npts)
        .map(|_| (0..dims).map(|_| r()).collect())
        .collect();

    COV_MATRIX_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = cov_matrix(&data);
    COV_MATRIX_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = cov_matrix(&data);
    let bitmism: usize = a
        .iter()
        .zip(&b)
        .map(|(ra, rb)| {
            ra.iter()
                .zip(rb)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count()
        })
        .sum();
    println!(
        "# stats::cov_matrix npts={npts} dims={dims} cov[0][1]={} bitmism={bitmism}",
        a[0][1]
    );

    let bench = |serial: bool| -> f64 {
        COV_MATRIX_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(cov_matrix(black_box(&data)));
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(cov_matrix(black_box(&data)));
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
    COV_MATRIX_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} cov_matrix serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
