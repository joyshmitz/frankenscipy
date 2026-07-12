//! Median-null-gated A/B for the probability-plot theoretical-quantile maps
//! (`probplot`, `probplot_quantiles`): the ORIG serial `ndtri`/`ppf` map over the
//! order-statistic medians vs the parallel `par_continuous_map` (index-ordered ⇒
//! BYTE-IDENTICAL). Toggled by `PROBPLOT_FORCE_SERIAL`, alternated per iteration.
//! SciPy vectorizes the quantile map in C, so fsci's serial map was the loss; this
//! is the byte-identical self-speedup that flips it. Args: n [iters].
use fsci_stats::{PROBPLOT_FORCE_SERIAL, probplot, probplot_quantiles};
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
    PROBPLOT_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
    let _ = black_box(f());
    let t = Instant::now();
    for _ in 0..3 {
        let _ = black_box(f());
    }
    t.elapsed().as_secs_f64() / 3.0 * 1e3
}

fn parity(name: &str, f: &dyn Fn() -> Vec<f64>) -> usize {
    PROBPLOT_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = f();
    PROBPLOT_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = f();
    let mism = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    println!("# {name}: {} values, bitmism={mism}", a.len());
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
    PROBPLOT_FORCE_SERIAL.store(false, Ordering::Relaxed);
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
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 10.0 - 5.0
    };
    let data: Vec<f64> = (0..n).map(|_| r()).collect();

    let pq: &dyn Fn() -> Vec<f64> = &|| probplot_quantiles(n);
    let pp: &dyn Fn() -> Vec<f64> = &|| probplot(&data).osm;

    println!("# stats::probplot theoretical-quantile map  n={n} iters={iters}");
    let bitmism = parity("probplot_quantiles", pq) + parity("probplot(osm)", pp);
    println!("# total bitmism = {bitmism}");

    gate("probplot_quantiles", pq, iters);
    gate("probplot", pp, iters);
}
