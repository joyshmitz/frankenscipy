//! Median-null-gated A/B for `stats::gzscore` (geometric z-score): the ORIG serial
//! `data.iter().map(ln).collect()` vs the parallel `par_continuous_map` build of the log vector that
//! the downstream `zscore` reduces (mean, std, per-element output). Both arms live in ONE binary,
//! toggled by `GZSCORE_FORCE_SERIAL` and ALTERNATED per iteration. BYTE-IDENTICAL over the full output
//! Vec (bitmism=0). Also lifts gzscore_ddof/gzscore_weighted (shared helper). Peer: scipy.stats.gzscore.
use fsci_stats::{GZSCORE_FORCE_SERIAL, gzscore};
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

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 4.0 + 0.25 // positive (0.25, 4.25) for ln
    };
    let data: Vec<f64> = (0..n).map(|_| r()).collect();

    // Parity: every element of the output z-score vector must be bit-identical across arms.
    GZSCORE_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = gzscore(&data);
    GZSCORE_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = gzscore(&data);
    let bitmism = a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count()
        + usize::from(a.len() != b.len());

    let bench = |force_serial: bool| -> f64 {
        GZSCORE_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || gzscore(black_box(&data));
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
    GZSCORE_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# stats::gzscore {n} elements");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
