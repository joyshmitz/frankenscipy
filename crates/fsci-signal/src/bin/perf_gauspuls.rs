//! Median-null-gated A/B for `signal::gauspuls`: the ORIG serial per-sample `exp`/`cos`/`sin` kernel
//! vs the work-gated parallel 3-output fill. Both arms live in ONE binary, toggled by
//! `GAUSPULS_FORCE_SERIAL` and ALTERNATED per iteration. BYTE-IDENTICAL (bitmism=0 across all three
//! output vectors i/q/envelope): each element is a pure function of `t[i]`. Peer: scipy.signal.gauspuls.
use fsci_signal::{GAUSPULS_FORCE_SERIAL, gauspuls};
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
    let fc: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000.0);
    let (bw, bwr) = (0.5, -6.0);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 6.0e-3 - 3.0e-3 // t in [-3ms, 3ms)
    };
    let t: Vec<f64> = (0..n).map(|_| r()).collect();

    // Parity: every element of ALL THREE output vectors must be bit-identical across arms.
    GAUSPULS_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = gauspuls(&t, fc, bw, bwr).unwrap();
    GAUSPULS_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = gauspuls(&t, fc, bw, bwr).unwrap();
    let bit = |x: &[f64], y: &[f64]| {
        x.iter()
            .zip(y)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count()
            + usize::from(x.len() != y.len())
    };
    let bitmism = bit(&a.i, &b.i) + bit(&a.q, &b.q) + bit(&a.envelope, &b.envelope);

    let bench = |force_serial: bool| -> f64 {
        GAUSPULS_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || gauspuls(black_box(&t), black_box(fc), black_box(bw), black_box(bwr)).unwrap();
        let _ = black_box(run());
        let tm = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        tm.elapsed().as_secs_f64() / 3.0 * 1e3
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
    GAUSPULS_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# signal::gauspuls {n} elements, fc={fc}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
