//! Median-null-gated A/B for `ndimage::power_array`: the ORIG serial `powf` map vs the work-gated
//! parallel `fill_pixels_parallel` path. Both arms live in ONE binary, toggled by
//! `NDIMAGE_POWER_ARRAY_FORCE_SERIAL` and ALTERNATED per iteration inside one measured routine, so a
//! single `rch exec` invocation measures both on the same worker. `powf` is a heavy per-element
//! transcendental → the elementwise map is COMPUTE-bound (unlike bandwidth-bound add/multiply), so a
//! chunked parallel fill should win at large `n`. Peer: numpy `np.power`, single-threaded C.
use fsci_ndimage::{NDIMAGE_POWER_ARRAY_FORCE_SERIAL, NdArray, power_array};
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
        .unwrap_or(4_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(21);
    let exponent: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2.4); // gamma-correction-ish

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 4.0 // positive base for powf
    };
    let data: Vec<f64> = (0..npix).map(|_| r()).collect();
    let input = NdArray::new(data, vec![npix]).unwrap();

    // Parity: every output element must be bit-identical across the two arms.
    NDIMAGE_POWER_ARRAY_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = power_array(&input, exponent);
    NDIMAGE_POWER_ARRAY_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = power_array(&input, exponent);
    let bitmism = a
        .data
        .iter()
        .zip(&b.data)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
        + usize::from(a.data.len() != b.data.len());

    let bench = |force_serial: bool| -> f64 {
        NDIMAGE_POWER_ARRAY_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || power_array(black_box(&input), black_box(exponent));
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
    NDIMAGE_POWER_ARRAY_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = med(&mut nr);
    let cand_med = med(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# ndimage::power_array {npix} elements, exponent={exponent}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
