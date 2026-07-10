//! Median-null-gated A/B for `resample_poly_axis_2d`: the ORIG per-line filter rebuild vs the
//! hoisted design (FIR designed once, reused across all lines). Both arms live in ONE binary,
//! toggled by `RESAMPLE_POLY_FORCE_PERROW` and ALTERNATED per iteration inside one measured routine,
//! so a single `rch exec` invocation measures both on the same worker. Wide-short input (many short
//! rows) is where the signal-independent filter design is a large fraction of per-line work.
use fsci_signal::{RESAMPLE_POLY_FORCE_PERROW, resample_poly_axis_2d};
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
    let rows: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20000);
    let cols: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(80);
    let up: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let down: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let iters: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(13);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let x: Vec<Vec<f64>> = (0..rows).map(|_| (0..cols).map(|_| r()).collect()).collect();

    // Parity: hoisted must be byte-identical to per-row.
    RESAMPLE_POLY_FORCE_PERROW.store(true, Ordering::Relaxed);
    let a = resample_poly_axis_2d(&x, up, down, -1).unwrap();
    RESAMPLE_POLY_FORCE_PERROW.store(false, Ordering::Relaxed);
    let b = resample_poly_axis_2d(&x, up, down, -1).unwrap();
    let bitmism: usize = a
        .iter()
        .zip(&b)
        .map(|(ra, rb)| {
            ra.iter()
                .zip(rb)
                .filter(|(p, q)| p.to_bits() != q.to_bits())
                .count()
        })
        .sum();

    let bench = |force_perrow: bool| -> f64 {
        RESAMPLE_POLY_FORCE_PERROW.store(force_perrow, Ordering::Relaxed);
        let run = || resample_poly_axis_2d(black_box(&x), up, down, -1).unwrap();
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };

    let (mut ov, mut hv) = (Vec::new(), Vec::new());
    let (mut null_r, mut cand_r) = (Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true);
        let h = bench(false);
        let o2 = bench(true);
        null_r.push(o / o2);
        cand_r.push(o / h);
        ov.push(o);
        hv.push(h);
    }
    RESAMPLE_POLY_FORCE_PERROW.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = median(&mut nr);
    let cand_med = median(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let hb = hv.iter().copied().fold(f64::MAX, f64::min);
    println!("# resample_poly_axis_2d {rows}x{cols} up={up} down={down} axis=-1");
    println!(
        "{} perrow {ob:.2}ms (cv {:.1}%) hoisted {hb:.2}ms (cv {:.1}%) | CAND(perrow/hoisted) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&hv),
    );
}
