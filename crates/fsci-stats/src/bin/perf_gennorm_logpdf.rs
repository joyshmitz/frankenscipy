//! Median-null-gated A/B for `GenNorm::logpdf_many`: the ORIG serial `powf` map vs the parallel
//! `par_continuous_map_min` path (the SAME helper the sibling `pdf_many` already uses). Both arms live
//! in ONE binary, toggled by `GENNORM_LOGPDF_FORCE_SERIAL` and ALTERNATED per iteration. BYTE-IDENTICAL
//! (bitmism=0): order-preserving elementwise map. Peer: scipy.stats.gennorm.logpdf.
use fsci_stats::{GENNORM_LOGPDF_FORCE_SERIAL, GenNorm};
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
        .unwrap_or(8_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(21);
    let beta: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.5);

    let dist = GenNorm::new(beta);
    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0 // points in [-4, 4)
    };
    let xs: Vec<f64> = (0..n).map(|_| r()).collect();

    // Parity: every element of the logpdf vector must be bit-identical across arms.
    GENNORM_LOGPDF_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = dist.logpdf_many(&xs);
    GENNORM_LOGPDF_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = dist.logpdf_many(&xs);
    let bitmism = a
        .iter()
        .zip(&b)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
        + usize::from(a.len() != b.len());

    let bench = |force_serial: bool| -> f64 {
        GENNORM_LOGPDF_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || dist.logpdf_many(black_box(&xs));
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
    GENNORM_LOGPDF_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# stats::GenNorm::logpdf_many {n} elements, beta={beta}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
