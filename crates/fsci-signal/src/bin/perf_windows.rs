//! Median-null-gated A/B for the two remaining serial window generators `nuttall_window` (3 `cos`/
//! sample) and `bohman_window` (`cos`+`sin`/sample): the ORIG serial map vs the parallel `par_index_fill`
//! path, toggled by `WINDOW_FORCE_SERIAL` and ALTERNATED per iteration. BYTE-IDENTICAL (bitmism=0):
//! order-preserving elementwise map. Peer: scipy.signal.windows.nuttall / .bohman (single-threaded numpy).
use fsci_signal::{WINDOW_FORCE_SERIAL, bohman_window, nuttall_window};
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

fn measure(name: &str, m: usize, iters: usize, genfn: impl Fn(usize) -> Vec<f64>) {
    WINDOW_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = genfn(m);
    WINDOW_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = genfn(m);
    let bitmism = a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count()
        + usize::from(a.len() != b.len());

    let bench = |force_serial: bool| -> f64 {
        WINDOW_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || genfn(black_box(m));
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
    WINDOW_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} {name} {m}: serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND median \
         {cand_med:.3}x | NULL(A/A) [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(21);
    measure("nuttall_window", m, iters, nuttall_window);
    measure("bohman_window", m, iters, bohman_window);
}
