//! Median-null-gated A/B for `signal::dfreqresp`: the ORIG serial per-frequency `map().collect()` vs
//! the parallel-across-frequencies path (via the shared `freqz_par_collect` helper). Both arms live in
//! ONE binary, toggled by `DFREQRESP_FORCE_SERIAL` and ALTERNATED per iteration.
//!
//! Each frequency's complex response H(e^{jω})=num/den is a pure function of its index: cos/sin + two
//! Horner `eval_poly_complex` sweeps (O(len(num)+len(den)) complex MACs) + a complex divide —
//! compute-bound at high filter order, independent per frequency. The analog sibling `bode` was
//! already parallel; `dfreqresp` (and `dbode`, which calls it) were the serial digital stragglers.
use fsci_signal::{DFREQRESP_FORCE_SERIAL, dfreqresp};
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
    let order: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3072);
    let n_freqs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16384);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(25);

    // Realistic high-order digital transfer function (deterministic pseudo-random).
    let mut s = 0x9e3779b97f4a7c15u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let num: Vec<f64> = (0..=order).map(|_| r()).collect();
    let den: Vec<f64> = {
        let mut d = vec![1.0];
        d.extend((0..4).map(|_| 0.1 * r()));
        d
    };
    // Explicit frequency grid over [0, π) — the scipy dfreqresp(system, w) case.
    let w: Vec<f64> = (0..n_freqs)
        .map(|k| std::f64::consts::PI * k as f64 / n_freqs as f64)
        .collect();

    // Parity: the frequency vector and every complex sample must be bit-identical across arms.
    DFREQRESP_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let (wa, ha) = dfreqresp(&num, &den, 1.0, &w).unwrap();
    DFREQRESP_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (wb, hb) = dfreqresp(&num, &den, 1.0, &w).unwrap();
    let bitmism = wa
        .iter()
        .zip(&wb)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
        + usize::from(wa.len() != wb.len())
        + ha.iter()
            .zip(&hb)
            .filter(|((ar, ai), (br, bi))| {
                ar.to_bits() != br.to_bits() || ai.to_bits() != bi.to_bits()
            })
            .count()
        + usize::from(ha.len() != hb.len());

    let bench = |force_serial: bool| -> f64 {
        DFREQRESP_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || dfreqresp(black_box(&num), black_box(&den), 1.0, black_box(&w)).unwrap();
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
    DFREQRESP_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = med(&mut nr);
    let cand_med = med(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb2 = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# signal::dfreqresp order={order} n_freqs={n_freqs}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb2:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
