//! Median-null-gated A/B for `GaussianKdeNd::new`'s per-point whitening: ORIG serial map vs
//! point-chunk parallel triangular solves. BYTE-IDENTICAL (each point's L^-1 x is independent;
//! verified via evaluate). Toggled by `GAUSSIAN_KDE_ND_WHITEN_FORCE_SERIAL`. Args: npts [dims] [iters].
use fsci_stats::{GAUSSIAN_KDE_ND_WHITEN_FORCE_SERIAL, GaussianKdeNd};
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
    let npts: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500_000);
    let dims: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);

    let mut s = 0x71c3_a90du64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0
    };
    let dataset: Vec<Vec<f64>> = (0..npts)
        .map(|_| (0..dims).map(|_| r()).collect())
        .collect();
    let queries: Vec<Vec<f64>> = (0..64).map(|_| (0..dims).map(|_| r()).collect()).collect();

    GAUSSIAN_KDE_ND_WHITEN_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let ka = GaussianKdeNd::new(&dataset).expect("kde a");
    GAUSSIAN_KDE_ND_WHITEN_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let kb = GaussianKdeNd::new(&dataset).expect("kde b");
    let ea = ka.evaluate_many(&queries);
    let eb = kb.evaluate_many(&queries);
    let bitmism = ea
        .iter()
        .zip(&eb)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    println!(
        "# stats::GaussianKdeNd::new npts={npts} dims={dims} eval[0]={} bitmism={bitmism}",
        ea[0]
    );

    let bench = |serial: bool| -> f64 {
        GAUSSIAN_KDE_ND_WHITEN_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(GaussianKdeNd::new(black_box(&dataset)));
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(GaussianKdeNd::new(black_box(&dataset)));
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
    GAUSSIAN_KDE_ND_WHITEN_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} kde_nd_new serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
