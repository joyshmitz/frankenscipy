//! Median-null-gated A/B for `stats::cross_entropy`: the ORIG exact scalar term loop vs the parallel
//! chunked (4-way-unrolled) sum mirroring `entropy`/`kl_divergence`. Toggled by
//! `CROSS_ENTROPY_FORCE_SERIAL`, ALTERNATED per iteration. WITHIN per-op ULP tolerance (parallel
//! reorder) — reports the exact ULP distance alongside the speedup. Peer: scipy.special-style cross-entropy.
use fsci_stats::{CROSS_ENTROPY_FORCE_SERIAL, cross_entropy};
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
        (s >> 11) as f64 / (1u64 << 53) as f64 + 0.01
    };
    let pk: Vec<f64> = (0..n).map(|_| r()).collect();
    let qk: Vec<f64> = (0..n).map(|_| r()).collect();

    CROSS_ENTROPY_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = cross_entropy(&pk, &qk, None);
    CROSS_ENTROPY_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = cross_entropy(&pk, &qk, None);
    let ulp = (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs();
    let rel = if a != 0.0 { ((a - b) / a).abs() } else { (a - b).abs() };

    let bench = |force_serial: bool| -> f64 {
        CROSS_ENTROPY_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || cross_entropy(black_box(&pk), black_box(&qk), None);
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
    CROSS_ENTROPY_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# stats::cross_entropy {n} elements | serial={a:.17e} parallel={b:.17e} ULP_dist={ulp} rel={rel:.3e}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}]",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
