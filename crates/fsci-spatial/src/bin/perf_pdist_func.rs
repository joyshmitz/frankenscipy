//! Median-null-gated A/B for `spatial::pdist_func` (custom-metric condensed pairwise distances): the
//! ORIG serial double loop vs the parallel-across-rows path (contiguous i-chunks preserve the
//! condensed order). Both arms live in ONE binary, toggled by `SPATIAL_CDIST_FUNC_FORCE_SERIAL` and
//! ALTERNATED per iteration. Byte-identical; the euclidean metric (d FMAs + sqrt) is compute-bound.
use fsci_spatial::{SPATIAL_CDIST_FUNC_FORCE_SERIAL, pdist_func};
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

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1400);
    let d: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let data: Vec<Vec<f64>> = (0..n).map(|_| (0..d).map(|_| r()).collect()).collect();

    // Parity: every condensed distance must be bit-identical across the two arms.
    SPATIAL_CDIST_FUNC_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = pdist_func(&data, euclidean);
    SPATIAL_CDIST_FUNC_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = pdist_func(&data, euclidean);
    let bitmism: usize = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
        + usize::from(a.len() != b.len());

    let bench = |force_serial: bool| -> f64 {
        SPATIAL_CDIST_FUNC_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || pdist_func(black_box(&data), euclidean);
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
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
    SPATIAL_CDIST_FUNC_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = med(&mut nr);
    let cand_med = med(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# spatial::pdist_func(euclidean) n={n} d={d} pairs={}", n * (n - 1) / 2);
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
