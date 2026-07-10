//! Median-null-gated A/B for `SmoothBivariateSpline::eval_grid`: the ORIG per-cell path (rebuilds
//! the ny x-direction BSplines for every (x,y) cell) vs the hoisted path (x-splines built once for
//! the grid, y-spline once per row). Both arms live in ONE binary, toggled by
//! `SMOOTHBISPLINE_EVAL_GRID_FORCE_PERCELL` and ALTERNATED per iteration inside one measured
//! routine, so a single `rch exec` invocation measures both on the same worker.
use fsci_interpolate::{
    SMOOTHBISPLINE_EVAL_GRID_FORCE_PERCELL, SmoothBivariateSpline, SmoothBivariateSplineOptions,
};
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
    let nsamp: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30); // samples per axis
    let gside: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200); // query grid side
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(13);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    // Scattered samples on a jittered grid.
    let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..nsamp {
        for j in 0..nsamp {
            let xv = i as f64 + r() * 0.3;
            let yv = j as f64 + r() * 0.3;
            xs.push(xv);
            ys.push(yv);
            zs.push((xv * 0.2).sin() * 3.0 + (yv * 0.15).cos() - 0.01 * xv * yv);
        }
    }
    let options = SmoothBivariateSplineOptions {
        kx: 3,
        ky: 3,
        smoothing: Some(nsamp as f64 * nsamp as f64),
        ..SmoothBivariateSplineOptions::default()
    };
    let spline = SmoothBivariateSpline::new(&xs, &ys, &zs, options).expect("spline fit");
    let hi = (nsamp - 1) as f64;
    let gx: Vec<f64> = (0..gside).map(|k| k as f64 / gside as f64 * hi).collect();
    let gy: Vec<f64> = (0..gside).map(|k| k as f64 / gside as f64 * hi).collect();

    // Parity: hoisted must be byte-identical to the per-cell path.
    SMOOTHBISPLINE_EVAL_GRID_FORCE_PERCELL.store(true, Ordering::Relaxed);
    let a = spline.eval_grid(&gx, &gy);
    SMOOTHBISPLINE_EVAL_GRID_FORCE_PERCELL.store(false, Ordering::Relaxed);
    let b = spline.eval_grid(&gx, &gy);
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

    let bench = |percell: bool| -> f64 {
        SMOOTHBISPLINE_EVAL_GRID_FORCE_PERCELL.store(percell, Ordering::Relaxed);
        let run = || spline.eval_grid(black_box(&gx), black_box(&gy));
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
    SMOOTHBISPLINE_EVAL_GRID_FORCE_PERCELL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = median(&mut nr);
    let cand_med = median(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let hb = hv.iter().copied().fold(f64::MAX, f64::min);
    println!("# SmoothBivariateSpline eval_grid {nsamp}x{nsamp} samples, {gside}x{gside} query grid");
    println!(
        "{} percell {ob:.2}ms (cv {:.1}%) hoisted {hb:.2}ms (cv {:.1}%) | CAND(percell/hoisted) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&hv),
    );
}
