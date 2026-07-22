//! Median-null-gated A/B for `odr`'s dense LM `solve_lm_step` JᵀJ Gram + Jᵀr build: ORIG serial
//! row-outer accumulation vs per-output-row parallel. BYTE-IDENTICAL (each normal row / rhs slot is
//! a fixed-row-order reduction). Toggled by `ODR_LMSTEP_FORCE_SERIAL`. Args: ndata [iters].
use fsci_odr::{ODR_LMSTEP_FORCE_SERIAL, odr};
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

// Non-separable custom model y = b0*exp(b1*x) + b2 -> Model::new default (scalar_separable=false)
// forces odr onto the dense solve_least_squares / solve_lm_step path.
fn model(beta: &[f64], x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&xi| beta[0] * (beta[1] * xi).exp() + beta[2])
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ndata: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(600);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);

    // Synthetic data from true params + mild noise.
    let (b0, b1, b2) = (2.0_f64, 0.3_f64, 1.0_f64);
    let mut s = 0x51a3_9e77u64;
    let mut noise = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 0.02 - 0.01
    };
    let x: Vec<f64> = (0..ndata).map(|i| i as f64 / ndata as f64 * 3.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| b0 * (b1 * xi).exp() + b2 + noise())
        .collect();
    let beta0 = vec![1.5_f64, 0.25, 0.8];

    ODR_LMSTEP_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = odr(model, beta0.clone(), y.clone(), x.clone()).expect("odr");
    ODR_LMSTEP_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = odr(model, beta0.clone(), y.clone(), x.clone()).expect("odr");
    let bitmism = a
        .beta
        .iter()
        .zip(&b.beta)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();
    println!(
        "# odr::solve_lm_step ndata={ndata} beta={:?} bitmism={bitmism}",
        a.beta
    );

    let bench = |serial: bool| -> f64 {
        ODR_LMSTEP_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(odr(model, beta0.clone(), y.clone(), x.clone()).unwrap());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(odr(model, beta0.clone(), y.clone(), x.clone()).unwrap());
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
    ODR_LMSTEP_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} odr_lmstep serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
