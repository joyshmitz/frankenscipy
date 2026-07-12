//! Median-null-gated A/B for the PPCC / normality-plot shape sweeps
//! (`boxcox_normplot`, `yeojohnson_normplot`, `ppcc_plot`): the ORIG serial per-shape
//! loop vs the parallel `normplot_sweep` (the `n_shapes` independent grid points fan
//! across cores via `par_pair_index_map`, index-ordered ⇒ BYTE-IDENTICAL). Toggled by
//! `NORMPLOT_FORCE_SERIAL`, alternated per iteration. SciPy's versions loop over lambda
//! in pure Python; here we measure the byte-identical self-speedup of the parallel sweep.
//! Args: n_data [iters] [n_shapes].
use fsci_stats::{NORMPLOT_FORCE_SERIAL, boxcox_normplot, ppcc_plot, yeojohnson_normplot};
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

fn bench(force_serial: bool, f: &dyn Fn() -> Vec<f64>) -> f64 {
    NORMPLOT_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
    let _ = black_box(f());
    let t = Instant::now();
    for _ in 0..3 {
        let _ = black_box(f());
    }
    t.elapsed().as_secs_f64() / 3.0 * 1e3
}

fn parity(name: &str, f: &dyn Fn() -> Vec<f64>) -> usize {
    NORMPLOT_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = f();
    NORMPLOT_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = f();
    let mism = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    println!("# {name}: {} ppcc values, bitmism={mism}", a.len());
    mism
}

fn gate(name: &str, f: &dyn Fn() -> Vec<f64>, iters: usize) {
    let (mut ov, mut fv, mut nr, mut cr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true, f);
        let p = bench(false, f);
        let o2 = bench(true, f);
        nr.push(o / o2);
        cr.push(o / p);
        ov.push(o);
        fv.push(p);
    }
    NORMPLOT_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} {name} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/parallel) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}]",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(11);
    let nshapes: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(80);

    // Positive data in (0.5, 1.5): valid for Box-Cox (needs >0), Yeo-Johnson and PPCC.
    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 + 0.5
    };
    let data: Vec<f64> = (0..n).map(|_| r()).collect();

    let bx: &dyn Fn() -> Vec<f64> = &|| boxcox_normplot(&data, -2.0, 2.0, nshapes).unwrap().1;
    let yj: &dyn Fn() -> Vec<f64> = &|| yeojohnson_normplot(&data, -2.0, 2.0, nshapes).unwrap().1;
    let pp: &dyn Fn() -> Vec<f64> = &|| ppcc_plot(&data, -2.0, 2.0, nshapes).unwrap().1;

    println!("# stats::normplot sweep  n_data={n} n_shapes={nshapes} iters={iters}");
    let bitmism = parity("boxcox_normplot", bx)
        + parity("yeojohnson_normplot", yj)
        + parity("ppcc_plot", pp);
    println!("# total bitmism across all three families = {bitmism}");

    gate("boxcox_normplot", bx, iters);
    gate("yeojohnson_normplot", yj, iters);
    gate("ppcc_plot", pp, iters);
}
