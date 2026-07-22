//! Median-null-gated A/B for the RBF `Φ`-matrix build inside `RbfInterpolator::new`: the ORIG serial
//! double loop vs the parallel-across-rows fill. Both arms live in ONE binary, toggled by
//! `RBF_BUILD_FORCE_SERIAL` and ALTERNATED per iteration inside one measured routine (the shared dense
//! solve runs in both arms, so the A/B isolates the build).
use fsci_interpolate::{RBF_BUILD_FORCE_SERIAL, RbfInterpolator, RbfKernel};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn median(v: &mut [f64]) -> f64 {
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
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1200);
    let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(11);

    let mut s = 1u64;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 10.0
    };
    let points: Vec<Vec<f64>> = (0..n).map(|_| (0..dim).map(|_| rng()).collect()).collect();
    let values: Vec<f64> = (0..n).map(|_| rng() - 5.0).collect();
    let kernel = RbfKernel::Gaussian;

    RBF_BUILD_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = RbfInterpolator::new(&points, &values, kernel, 1.0).unwrap();
    RBF_BUILD_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = RbfInterpolator::new(&points, &values, kernel, 1.0).unwrap();
    let q: Vec<Vec<f64>> = (0..200)
        .map(|_| (0..dim).map(|_| rng()).collect())
        .collect();
    let bitmism = q
        .iter()
        .filter(|p| a.eval(p).to_bits() != b.eval(p).to_bits())
        .count();

    let bench = |force_serial: bool| -> f64 {
        RBF_BUILD_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run =
            || RbfInterpolator::new(black_box(&points), black_box(&values), kernel, 1.0).unwrap();
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..2 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 2.0 * 1e3
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
    RBF_BUILD_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = median(&mut nr);
    let cand_med = median(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# RbfInterpolator::new n={n} dim={dim} kernel=Gaussian (Phi build A/B)");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
