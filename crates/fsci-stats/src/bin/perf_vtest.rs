//! Median-null-gated A/B for `stats::vtest` (V-test of circular uniformity with a known mean
//! direction): the ORIG serial `map(cos).sum()`/`map(sin).sum()` over `samples` vs the parallel
//! circular reduction (`par_continuous_map` on the shifted sin/cos maps, sums index-ordered). Toggled
//! by the shared `CIRC_FORCE_SERIAL` (circmean/rayleightest family) and ALTERNATED per iteration.
//! BYTE-IDENTICAL (both the statistic and pvalue). Peer: astropy.stats.vtest (scipy has no vtest);
//! byte-identical self-speedup.
use fsci_stats::{CIRC_FORCE_SERIAL, vtest};
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
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(21);
    let mu: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * (2.0 * std::f64::consts::PI) // angles [0, 2π)
    };
    let samples: Vec<f64> = (0..n).map(|_| r()).collect();

    // Parity: BOTH the statistic and pvalue must be bit-identical across arms.
    CIRC_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let (va, pa) = vtest(&samples, mu);
    CIRC_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (vb, pb) = vtest(&samples, mu);
    let bitmism = usize::from(va.to_bits() != vb.to_bits()) + usize::from(pa.to_bits() != pb.to_bits());

    let bench = |force_serial: bool| -> f64 {
        CIRC_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || vtest(black_box(&samples), black_box(mu));
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
    println!("# stats::vtest {n} elements, mu={mu} (v serial={va} parallel={vb})");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
