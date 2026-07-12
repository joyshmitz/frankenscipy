//! Median-null-gated A/B for `stats::yeojohnson_llf`: the ORIG serial `Σ signum·ln`
//! log-term vs the parallel `par_continuous_map` + index-ordered sum (BYTE-IDENTICAL).
//! Toggled by `YEOJOHNSON_LLF_FORCE_SERIAL`, alternated per iteration. The transform
//! already fans across cores; this parallelizes the last serial `ln` pass (mirrors the
//! shipped boxcox_llf). SciPy vectorizes it, so the serial pass was the loss.
//! Args: n [iters] [lambda].
use fsci_stats::{YEOJOHNSON_LLF_FORCE_SERIAL, yeojohnson_llf};
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
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let lmb: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.7);

    // Mixed-sign data (Yeo-Johnson accepts negatives), values in (-5, 5).
    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 10.0 - 5.0
    };
    let data: Vec<f64> = (0..n).map(|_| r()).collect();

    // Parity: the scalar log-likelihood must be bit-identical across arms.
    YEOJOHNSON_LLF_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = yeojohnson_llf(lmb, &data);
    YEOJOHNSON_LLF_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = yeojohnson_llf(lmb, &data);
    let bitmism = usize::from(a.to_bits() != b.to_bits());
    println!("# stats::yeojohnson_llf n={n} lambda={lmb} (serial={a} parallel={b}) bitmism={bitmism}");

    let run = || yeojohnson_llf(black_box(lmb), black_box(&data));
    let bench = |force_serial: bool| -> f64 {
        YEOJOHNSON_LLF_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
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
    YEOJOHNSON_LLF_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} yeojohnson_llf serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/parallel) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
