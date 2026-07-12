//! Median-null-gated A/B for `firwin_2d_circular`'s radial grid fill: ORIG serial per-row loop vs
//! row-chunk parallel. BYTE-IDENTICAL (each cell = interp(sqrt(f1²+f2²)), per-row independent).
//! Toggled by `FIRWIN2D_CIRCULAR_FORCE_SERIAL`. Args: hsize [iters].
use fsci_signal::{FIRWIN2D_CIRCULAR_FORCE_SERIAL, FirWindow, firwin_2d_circular};
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
    let hs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let hsize = (hs, hs);
    let cutoff = [0.3_f64];
    let win = FirWindow::Hamming;

    FIRWIN2D_CIRCULAR_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = firwin_2d_circular(hsize, &cutoff, win, true).expect("firwin2d");
    FIRWIN2D_CIRCULAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = firwin_2d_circular(hsize, &cutoff, win, true).expect("firwin2d");
    let bitmism: usize = a
        .iter()
        .zip(&b)
        .map(|(ra, rb)| ra.iter().zip(rb).filter(|(x, y)| x.to_bits() != y.to_bits()).count())
        .sum();
    println!("# signal::firwin_2d_circular hs={hs} a[1][1]={} bitmism={bitmism}", a[1][1]);

    let bench = |serial: bool| -> f64 {
        FIRWIN2D_CIRCULAR_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(firwin_2d_circular(black_box(hsize), &cutoff, win, true).unwrap());
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(firwin_2d_circular(black_box(hsize), &cutoff, win, true).unwrap());
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
    FIRWIN2D_CIRCULAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} firwin_2d_circular serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
