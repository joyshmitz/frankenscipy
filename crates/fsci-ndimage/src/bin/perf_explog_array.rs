//! Median-null-gated A/B for `ndimage::exp_array` and `ndimage::log_array`: the ORIG serial
//! transcendental map vs the work-gated parallel `fill_pixels_parallel` path. Both arms of each op
//! live in ONE binary, toggled by `NDIMAGE_EXP_ARRAY_FORCE_SERIAL` / `NDIMAGE_LOG_ARRAY_FORCE_SERIAL`
//! and ALTERNATED per iteration. `exp`/`ln` are heavy per-element transcendentals (~20-40 cycles) so
//! the maps are COMPUTE-bound (unlike bandwidth-bound add/multiply). Peer: numpy np.exp / np.log.
use fsci_ndimage::{
    NDIMAGE_EXP_ARRAY_FORCE_SERIAL, NDIMAGE_LOG_ARRAY_FORCE_SERIAL, NdArray, exp_array, log_array,
};
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

fn report(
    name: &str,
    npix: usize,
    ov: &[f64],
    fv: &[f64],
    null_r: &mut [f64],
    cand_r: &mut [f64],
    bitmism: usize,
) {
    let null_med = med(null_r);
    let cand_med = med(cand_r);
    let null_lo = null_r.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = null_r.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# ndimage::{name} {npix} elements");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(ov),
        cv(fv),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let npix: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(21);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 2.0 // range [-2, 6): exp finite, some log NEG_INF
    };
    let data: Vec<f64> = (0..npix).map(|_| r()).collect();
    let input = NdArray::new(data, vec![npix]).unwrap();

    // --- exp_array ---
    NDIMAGE_EXP_ARRAY_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let ea = exp_array(&input);
    NDIMAGE_EXP_ARRAY_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let eb = exp_array(&input);
    let ebit = ea
        .data
        .iter()
        .zip(&eb.data)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
        + usize::from(ea.data.len() != eb.data.len());
    let ebench = |fs: bool| -> f64 {
        NDIMAGE_EXP_ARRAY_FORCE_SERIAL.store(fs, Ordering::Relaxed);
        let run = || exp_array(black_box(&input));
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };
    let (mut eov, mut efv, mut enull, mut ecand) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = ebench(true);
        let f = ebench(false);
        let o2 = ebench(true);
        enull.push(o / o2);
        ecand.push(o / f);
        eov.push(o);
        efv.push(f);
    }
    NDIMAGE_EXP_ARRAY_FORCE_SERIAL.store(false, Ordering::Relaxed);

    // --- log_array ---
    NDIMAGE_LOG_ARRAY_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let la = log_array(&input);
    NDIMAGE_LOG_ARRAY_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let lb = log_array(&input);
    let lbit = la
        .data
        .iter()
        .zip(&lb.data)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
        + usize::from(la.data.len() != lb.data.len());
    let lbench = |fs: bool| -> f64 {
        NDIMAGE_LOG_ARRAY_FORCE_SERIAL.store(fs, Ordering::Relaxed);
        let run = || log_array(black_box(&input));
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };
    let (mut lov, mut lfv, mut lnull, mut lcand) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = lbench(true);
        let f = lbench(false);
        let o2 = lbench(true);
        lnull.push(o / o2);
        lcand.push(o / f);
        lov.push(o);
        lfv.push(f);
    }
    NDIMAGE_LOG_ARRAY_FORCE_SERIAL.store(false, Ordering::Relaxed);

    report("exp_array", npix, &eov, &efv, &mut enull, &mut ecand, ebit);
    report("log_array", npix, &lov, &lfv, &mut lnull, &mut lcand, lbit);
}
