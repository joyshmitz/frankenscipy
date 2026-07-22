//! Median-null-gated A/B for the circular-statistics `Σsin`/`Σcos` reduction (via `circmean`): the
//! ORIG fused serial `map(sin).sum()` / `map(cos).sum()` vs parallel-transcendental-maps + serial
//! index-ordered sums (parallelize ONLY the compute-bound `sin`/`cos` maps via `par_continuous_map`).
//! Both arms live in ONE binary, toggled by `CIRC_FORCE_SERIAL` and ALTERNATED per iteration.
//! BYTE-IDENTICAL (bitmism=0): independent sin/cos values, preserved left-fold. Also lifts
//! `circvar`/`circstd` (shared helper). Peer: scipy.stats.circmean.
use fsci_stats::{CIRC_FORCE_SERIAL, circmean};
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
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(21);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * (2.0 * std::f64::consts::PI) // angles [0, 2π)
    };
    let data: Vec<f64> = (0..n).map(|_| r()).collect();

    // Parity: the single f64 circular mean must be bit-identical across arms.
    CIRC_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = circmean(&data);
    CIRC_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = circmean(&data);
    let bitmism = usize::from(a.to_bits() != b.to_bits());

    let bench = |force_serial: bool| -> f64 {
        CIRC_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || circmean(black_box(&data));
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
    CIRC_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# stats::circmean {n} elements (result serial={a} parallel={b})");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
