//! Median-null-gated A/B for `signal::Lti::freqresp`: the ORIG serial per-frequency loop vs the
//! parallel-across-frequencies path (via the shared `freqz_par_collect` helper). Both arms live in
//! ONE binary, toggled by `FREQRESP_METHOD_FORCE_SERIAL` and ALTERNATED per iteration.
//!
//! Each frequency's response H(jω)=num(jω)/den(jω) is a pure function of its index: `eval_at` does
//! two Horner `poly_eval_complex` sweeps (O(len(num)+len(den)) complex MACs) + a complex divide, then
//! a `sqrt`/`atan2` — compute-bound at high system order, independent per frequency. The free-fn
//! `bode`/`dfreqresp` sweeps are already parallel; the `Lti`/`Dlti` `freqresp` methods were stragglers.
use fsci_signal::{FREQRESP_METHOD_FORCE_SERIAL, Lti};
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

    // Realistic high-order continuous-time transfer function (deterministic pseudo-random).
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
    let sys = Lti { num, den };
    // Explicit angular-frequency grid — the scipy lti.freqresp(w=...) case.
    let w: Vec<f64> = (0..n_freqs)
        .map(|k| 0.01 + 10.0 * k as f64 / n_freqs as f64)
        .collect();

    // Parity: magnitude and phase must be bit-identical across the two arms.
    FREQRESP_METHOD_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let (ma, pa) = sys.freqresp(&w);
    FREQRESP_METHOD_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mb, pb) = sys.freqresp(&w);
    let bit = |x: &[f64], y: &[f64]| -> usize {
        x.iter()
            .zip(y)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count()
            + usize::from(x.len() != y.len())
    };
    let bitmism = bit(&ma, &mb) + bit(&pa, &pb);

    let bench = |force_serial: bool| -> f64 {
        FREQRESP_METHOD_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || sys.freqresp(black_box(&w));
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
    FREQRESP_METHOD_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = med(&mut nr);
    let cand_med = med(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb2 = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# signal::Lti::freqresp order={order} n_freqs={n_freqs}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb2:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
