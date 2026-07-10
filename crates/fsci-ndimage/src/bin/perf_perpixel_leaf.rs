//! A/B probe for the two per-pixel B-spline levers in `sample_interpolated`.
//!
//! `zoom`/`shift`/diagonal-`affine_transform` are axis-separable and precompute their per-axis
//! supports once. `rotate`, a GENERAL (non-diagonal) `affine_transform`, and `map_coordinates`
//! have coupled coordinates, so their supports are rebuilt per pixel. Two levers live there:
//!
//!   `compact` — `NDIMAGE_BSPLINE_COMPACT_DISABLE`: the cardinal tap loop spanned
//!               `floor(cc) ± order` (`2·order+1` `cardinal_bspline` calls) though only `order+1`
//!               taps can be nonzero. Shipped 6c53716ff (order3 1.37x, order5 1.53x).
//!   `offs`    — `NDIMAGE_SPLINE_OFFSET_DISABLE`: each of the `(order+1)^ndim` tensor leaves
//!               recomputed `Σ idx[d]·stride[d]` via `coeffs.get`. Premultiply taps by stride once
//!               per pixel ⇒ the leaf is a single `data[base]` load.
//!
//! NOISE: remote rch workers cannot be `taskset`-pinned, so the `map_coords_serial` workload is
//! sized UNDER the parallel gate (`npts · (order+1)^ndim < 2^18`) and therefore runs on the
//! SERIAL path by construction — the honest way to time a per-pixel arithmetic lever.
//!
//! Each lever carries its own NULL CONTROL, a row where the knob provably cannot act:
//!   `offs`    → `order=0` (`sample_interpolated` returns before the leaf).
//!   `compact` → `Constant` mode (routes to `fold_wrap_cubic` / `bspline_local_support`).
//! Any run whose control drifts off 1.00x is noise-dominated and must be discarded.
//!
//! Usage: `perf_perpixel_leaf <compact|offs> [order]`
use fsci_ndimage::{
    BoundaryMode, NDIMAGE_BSPLINE_COMPACT_DISABLE, NDIMAGE_SPLINE_OFFSET_DISABLE, NdArray,
    affine_transform, map_coordinates, rotate,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Mean and coefficient-of-variation (%) of a sample set.
fn mean_cv(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, if m > 0.0 { var.sqrt() / m * 100.0 } else { 0.0 })
}

/// A general (non-diagonal) affine, so the diagonal separable gate misses.
const GENERAL_AFFINE: [[f64; 3]; 2] = [[0.9, 0.3, -20.0], [-0.3, 0.9, 15.0]];

/// Kept under the parallel gate: `NPTS * (order+1)^2 < 2^18` for every order <= 5 (6000*36=216000).
const NPTS: usize = 6000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lever = args.get(1).map_or("offs", |s| s.as_str());
    let only_order: Option<usize> = args.get(2).and_then(|s| s.parse().ok());
    let env_usize = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let reps = env_usize("FSCI_AB_REPS", 3);
    let iters = env_usize("FSCI_AB_ITERS", 7);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let big_side = 512usize;
    let big = NdArray::new(
        (0..big_side * big_side).map(|_| r()).collect(),
        vec![big_side, big_side],
    )
    .unwrap();

    // Small input keeps the spline prefilter from diluting the per-point leaf work.
    let small_side = 64usize;
    let small = NdArray::new(
        (0..small_side * small_side).map(|_| r()).collect(),
        vec![small_side, small_side],
    )
    .unwrap();
    // Coupled (non-separable) coordinates.
    let coords: Vec<Vec<f64>> = {
        let (mut rr, mut cc) = (Vec::with_capacity(NPTS), Vec::with_capacity(NPTS));
        for i in 0..NPTS {
            let (y, x) = ((i / small_side) as f64, (i % small_side) as f64);
            rr.push(0.9 * y + 0.05 * x + 3.7);
            cc.push(0.9 * x - 0.05 * y + 2.3);
        }
        vec![rr, cc]
    };

    let offs_lever = lever == "offs";
    println!(
        "# same-binary A/B lever={lever}: {}",
        if offs_lever {
            "ORIG index-space leaf vs flat-offset leaf (per-pixel path)"
        } else {
            "ORIG full tap window vs compact support window"
        }
    );
    println!("# CONTROL rows must read ~1.00x; otherwise the run is noise-dominated.");
    let set_orig = |orig: bool| {
        if offs_lever {
            NDIMAGE_SPLINE_OFFSET_DISABLE.store(orig, Ordering::Relaxed);
        } else {
            NDIMAGE_BSPLINE_COMPACT_DISABLE.store(orig, Ordering::Relaxed);
        }
    };

    for &order in &[0usize, 1, 2, 3, 4, 5] {
        if only_order.is_some_and(|o| o != order) {
            continue;
        }
        // The compact lever does not exist at order 0 (no cardinal loop); it is the offs control.
        if !offs_lever && order == 0 {
            continue;
        }
        for &mode in &[BoundaryMode::Reflect, BoundaryMode::Constant] {
            let kernels: [(&str, Box<dyn Fn() -> Vec<f64>>); 3] = [
                (
                    "rotate_par",
                    Box::new(|| rotate(&big, 33.0, false, order, mode, 0.0).unwrap().data),
                ),
                (
                    "affine_gen_par",
                    Box::new(|| {
                        affine_transform(&big, &GENERAL_AFFINE, order, mode, 0.0)
                            .unwrap()
                            .data
                    }),
                ),
                (
                    "map_coords_serial",
                    Box::new(|| map_coordinates(&small, &coords, order, mode, 0.0).unwrap()),
                ),
            ];

            for (name, f) in &kernels {
                // Parity first: same binary, both arms, bit-level compare.
                set_orig(false);
                let cand = f();
                set_orig(true);
                let orig = f();
                let md = cand
                    .iter()
                    .zip(&orig)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                let bits = cand
                    .iter()
                    .zip(&orig)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();

                let bench = |orig: bool| {
                    set_orig(orig);
                    let _ = black_box(f().len());
                    let t = Instant::now();
                    for _ in 0..reps {
                        black_box(f().len());
                    }
                    t.elapsed().as_secs_f64() / reps as f64 * 1000.0
                };
                // Interleave ORIG/candidate so slow drift hits both arms equally.
                let (mut ov, mut cv) = (Vec::new(), Vec::new());
                for _ in 0..iters {
                    ov.push(bench(true));
                    cv.push(bench(false));
                }
                set_orig(false);
                let (om, ocv) = mean_cv(&ov);
                let (cm, ccv) = mean_cv(&cv);
                let ob = ov.iter().copied().fold(f64::MAX, f64::min);
                let cb = cv.iter().copied().fold(f64::MAX, f64::min);
                let is_control = if offs_lever {
                    order == 0
                } else {
                    mode == BoundaryMode::Constant
                };
                let tag = if is_control { "CONTROL" } else { "       " };
                println!(
                    "{tag} order={order} {mode:?} {name}: orig {ob:.2}ms (mean {om:.2} cv {ocv:.1}%) \
                     cand {cb:.2}ms (mean {cm:.2} cv {ccv:.1}%) best {:.2}x mean {:.2}x \
                     maxdiff={md:.1e} bitmism={bits}",
                    ob / cb,
                    om / cm
                );
            }
        }
    }
}
