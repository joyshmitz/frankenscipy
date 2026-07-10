//! Median-null-gated A/B for `generic_filter1d`: the ORIG serial per-pixel loop vs the
//! parallel-across-output-pixels path. Both arms live in ONE binary, toggled by
//! `GENERIC_FILTER1D_FORCE_SERIAL` and ALTERNATED per iteration inside one measured routine, so a
//! single `rch exec` invocation measures both on the same worker. The reducer is a pure Sync
//! closure (the compute-bound case scipy runs single-threaded).
use fsci_ndimage::{BoundaryMode, GENERIC_FILTER1D_FORCE_SERIAL, NdArray, generic_filter1d};
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
    let side: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1500);
    let size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(13);

    let total = side * side;
    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let data: Vec<f64> = (0..total).map(|_| r()).collect();
    let input = NdArray::new(data, vec![side, side]).unwrap();
    // A representative "user reducer": a few transcendentals per window element.
    let reducer = |w: &[f64]| -> f64 {
        let mut acc = 0.0;
        for (i, &v) in w.iter().enumerate() {
            acc += (v * (i as f64 + 1.3)).sin() * (v * 0.5).cos();
        }
        acc
    };

    // Parity: parallel must be byte-identical to serial.
    GENERIC_FILTER1D_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = generic_filter1d(&input, reducer, size, 1, BoundaryMode::Reflect, 0.0).unwrap();
    GENERIC_FILTER1D_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = generic_filter1d(&input, reducer, size, 1, BoundaryMode::Reflect, 0.0).unwrap();
    let bitmism = a
        .data
        .iter()
        .zip(&b.data)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();

    let bench = |force_serial: bool| -> f64 {
        GENERIC_FILTER1D_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || generic_filter1d(black_box(&input), reducer, size, 1, BoundaryMode::Reflect, 0.0).unwrap();
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
    GENERIC_FILTER1D_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = median(&mut nr);
    let cand_med = median(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# generic_filter1d {side}x{side} size={size} axis=1 mode=Reflect");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
