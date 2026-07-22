//! Median-null-gated A/B for `signal::morlet2` (the default cwt wavelet): the ORIG serial per-sample
//! `exp`+`sin_cos` fill vs the parallel `par_index_fill_pairs` path. Both arms live in ONE binary,
//! toggled by `MORLET_FORCE_SERIAL` and ALTERNATED per iteration. BYTE-IDENTICAL (bitmism=0 across the
//! (re, im) pairs): order-preserving elementwise fill. Peer: scipy.signal.morlet2 (single-threaded numpy).
use fsci_signal::{MORLET_FORCE_SERIAL, morlet2};
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
    let m: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(21);
    let w: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let s = (m as f64 / 16.0).max(1.0); // scale spanning the window (realistic wavelet)

    // Parity: every (re, im) pair must be bit-identical across arms.
    MORLET_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = morlet2(m, s, w);
    MORLET_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = morlet2(m, s, w);
    let bitmism = a
        .iter()
        .zip(&b)
        .filter(|((ar, ai), (br, bi))| ar.to_bits() != br.to_bits() || ai.to_bits() != bi.to_bits())
        .count()
        + usize::from(a.len() != b.len());

    let bench = |force_serial: bool| -> f64 {
        MORLET_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || morlet2(black_box(m), black_box(s), black_box(w));
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
    MORLET_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# signal::morlet2 {m} samples, w={w}, s={s:.1}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
