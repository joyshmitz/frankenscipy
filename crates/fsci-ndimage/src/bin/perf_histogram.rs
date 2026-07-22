//! Median-null-gated A/B for the GLOBAL (no-labels) `ndimage::histogram`: the ORIG group path
//! (`measurement_label_groups` full-data clone + serial bincount) vs the direct privatized parallel
//! bincount (no clone). Both arms live in ONE binary, toggled by `NDIMAGE_HISTOGRAM_FORCE_SERIAL` and
//! ALTERNATED per iteration, so one `rch exec` invocation measures both on the same worker. The
//! per-pixel divide+floor is compute-bound; scipy.ndimage.histogram is single-threaded.
use fsci_ndimage::{NDIMAGE_HISTOGRAM_FORCE_SERIAL, NdArray, histogram};
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
    let npix: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16_000_000);
    let nbins: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(21);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 255.0
    };
    let data: Vec<f64> = (0..npix).map(|_| r()).collect();
    let input = NdArray::new(data, vec![npix]).unwrap();
    let (min_val, max_val) = (0.0, 255.0);

    // Parity: the single global histogram must be bit-identical across the two arms.
    NDIMAGE_HISTOGRAM_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = histogram(&input, min_val, max_val, nbins, None, None).unwrap();
    NDIMAGE_HISTOGRAM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = histogram(&input, min_val, max_val, nbins, None, None).unwrap();
    let bitmism: usize = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .filter(|(x, y)| x != y)
        .count()
        + usize::from(a.len() != b.len());

    let bench = |force_serial: bool| -> f64 {
        NDIMAGE_HISTOGRAM_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || histogram(black_box(&input), min_val, max_val, nbins, None, None).unwrap();
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
    NDIMAGE_HISTOGRAM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = med(&mut nr);
    let cand_med = med(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# ndimage::histogram (global, no labels) {npix} pixels, {nbins} bins");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
