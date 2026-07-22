//! Median-null-gated A/B for `bode`'s magnitude/phase post-processing (`bode_from_complex`): the ORIG
//! serial `hypot`+`log10` (mag) and `atan2` (phase) maps vs the parallel `freqz_par_collect` path.
//! Both arms live in ONE binary, toggled by `BODE_POST_FORCE_SERIAL` and ALTERNATED per iteration.
//! The complex response `h` is computed in parallel in BOTH arms; with a LOW-ORDER filter and MANY
//! frequencies (dense Bode plot) the post-processing dominates. BYTE-IDENTICAL (mag + phase). Also
//! lifts `dbode`. Peer: scipy.signal.bode.
use fsci_signal::{BODE_POST_FORCE_SERIAL, bode};
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
    let n_freqs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(21);

    // Low-order analog filter H(jω) = 1 / (1 + 0.5·jω): cheap complex response, so the mag/phase
    // post-processing dominates. Log-spaced frequencies (dense Bode plot).
    let num = vec![1.0];
    let den = vec![0.5, 1.0];
    let w: Vec<f64> = (0..n_freqs)
        .map(|k| 10.0_f64.powf(-3.0 + 6.0 * k as f64 / n_freqs as f64))
        .collect();

    // Parity: mag and phase must be bit-identical across arms.
    BODE_POST_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let (_wa, ma, pa) = bode(&num, &den, &w).unwrap();
    BODE_POST_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (_wb, mb, pb) = bode(&num, &den, &w).unwrap();
    let bitmism = ma
        .iter()
        .zip(&mb)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
        + pa.iter()
            .zip(&pb)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count()
        + usize::from(ma.len() != mb.len() || pa.len() != pb.len());

    let bench = |force_serial: bool| -> f64 {
        BODE_POST_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || bode(black_box(&num), black_box(&den), black_box(&w)).unwrap();
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
    BODE_POST_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# signal::bode post-processing {n_freqs} freqs (low-order filter)");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
