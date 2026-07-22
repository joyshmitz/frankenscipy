//! Median-null-gated A/B for `signal::group_delay_from_ba`: the ORIG serial per-frequency loop vs
//! the parallel-across-frequencies path (via the shared `freqz_par_collect` helper). Both arms live
//! in ONE binary, toggled by `GROUP_DELAY_FROM_BA_FORCE_SERIAL` and ALTERNATED per iteration.
//!
//! Each frequency's group delay is a pure function of its index: `group_delay_at_frequency` does two
//! `eval_weighted_poly_on_unit_circle` sweeps (a `cos`+`sin` PER coefficient), so the per-frequency
//! work is O(len(b)+len(a)) transcendentals — compute-bound and independent. A high-order FIR filter
//! (`nb` in the hundreds) over a large `n_freqs` sweep is the realistic filter-analysis workload; the
//! scipy-named sibling `group_delay` is already parallel, this convenience helper was the straggler.
use fsci_signal::{GROUP_DELAY_FROM_BA_FORCE_SERIAL, group_delay_from_ba};
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
    let order: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n_freqs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16384);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(25);

    // Realistic high-order FIR-ish numerator + short IIR denominator (deterministic pseudo-random).
    let mut s = 0x9e3779b97f4a7c15u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let b: Vec<f64> = (0..=order).map(|_| r()).collect();
    let a: Vec<f64> = {
        let mut a = vec![1.0];
        a.extend((0..4).map(|_| 0.1 * r()));
        a
    };

    // Parity: both output vectors must be bit-identical across the two arms.
    GROUP_DELAY_FROM_BA_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let (fa, da) = group_delay_from_ba(&b, &a, n_freqs);
    GROUP_DELAY_FROM_BA_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (fb, db) = group_delay_from_ba(&b, &a, n_freqs);
    let bit = |x: &[f64], y: &[f64]| -> usize {
        x.iter()
            .zip(y)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count()
            + usize::from(x.len() != y.len())
    };
    let bitmism = bit(&fa, &fb) + bit(&da, &db);

    let bench = |force_serial: bool| -> f64 {
        GROUP_DELAY_FROM_BA_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || group_delay_from_ba(black_box(&b), black_box(&a), n_freqs);
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 5.0 * 1e3
    };

    let (mut ov, mut fv) = (Vec::new(), Vec::new());
    let (mut null_r, mut cand_r) = (Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true);
        let f = bench(false);
        let o2 = bench(true);
        null_r.push(o / o2);
        cand_r.push(o / f);
        ov.push(o);
        fv.push(f);
    }
    GROUP_DELAY_FROM_BA_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = med(&mut nr);
    let cand_med = med(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb2 = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# signal::group_delay_from_ba order={order} n_freqs={n_freqs}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb2:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
