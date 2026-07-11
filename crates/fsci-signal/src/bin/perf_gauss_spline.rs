//! Median-null-gated A/B for `signal::gauss_spline`: the ORIG serial `exp` map vs the parallel
//! `par_index_fill` path. Both arms live in ONE binary, toggled by `GAUSS_SPLINE_FORCE_SERIAL` and
//! ALTERNATED per iteration. BYTE-IDENTICAL (bitmism=0): order-preserving elementwise `exp` map.
//! Peer: scipy.signal.gauss_spline.
use fsci_signal::{GAUSS_SPLINE_FORCE_SERIAL, gauss_spline};
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
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(21);
    let order: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0
    };
    let x: Vec<f64> = (0..n).map(|_| r()).collect();

    // Parity: every element of the output must be bit-identical across arms.
    GAUSS_SPLINE_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = gauss_spline(&x, order);
    GAUSS_SPLINE_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = gauss_spline(&x, order);
    let bitmism = a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count()
        + usize::from(a.len() != b.len());

    let bench = |force_serial: bool| -> f64 {
        GAUSS_SPLINE_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || gauss_spline(black_box(&x), black_box(order));
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
    GAUSS_SPLINE_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# signal::gauss_spline {n} elements, order={order}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
