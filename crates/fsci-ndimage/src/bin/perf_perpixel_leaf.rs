//! A/B probe for the compact cardinal-B-spline tap window in `compute_axis_support`.
//!
//! A `perf record` of `rotate` order=3 ranked `compute_axis_support` at **61.68%** self time
//! (`sample_spline_recursive` 19.92%, `sample_interpolated` 3.08%). Its tap loop spanned
//! `floor(cc) ± order` — `2·order+1` `cardinal_bspline` calls — but the kernel vanishes exactly
//! outside `|x| < (order+1)/2`, so only `order+1` taps can be nonzero (7→4 at order 3, 11→6 at
//! order 5). This probe isolates that lever on the transforms where supports are recomputed per
//! pixel (coupled coords: `rotate`, general non-diagonal `affine_transform`, `map_coordinates`).
//!
//! Usage: `perf_perpixel_leaf [mode] [order]`
//!   both (default) — interleaved A/B: ORIG full window vs compact window
//!   full | compact — run ONE path in a tight loop (for `perf record` isolation)
use fsci_ndimage::{
    BoundaryMode, NDIMAGE_BSPLINE_COMPACT_DISABLE, NdArray, affine_transform, map_coordinates,
    rotate,
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

/// A general (non-diagonal) affine: rotation-like matrix so the diagonal separable gate misses.
const GENERAL_AFFINE: [[f64; 3]; 2] = [[0.9, 0.3, -20.0], [-0.3, 0.9, 15.0]];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode_sel = args.get(1).map_or("both", |s| s.as_str());
    let only_order: Option<usize> = args.get(2).and_then(|s| s.parse().ok());
    // Tunable so a loaded box can buy stability with longer samples / more interleaves.
    // Under `taskset -c <one core>` `available_parallelism()` is 1, so the transforms take the
    // SERIAL path — the honest way to measure a per-pixel arithmetic lever (no scheduler noise).
    let env_usize = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let reps = env_usize("FSCI_AB_REPS", 4);
    let iters = env_usize("FSCI_AB_ITERS", 5);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let side = 512usize;
    let data: Vec<f64> = (0..side * side).map(|_| r()).collect();
    let img = NdArray::new(data, vec![side, side]).unwrap();

    // map_coordinates over a coupled (non-separable) swirl of sample points.
    let npts = side * side;
    let coords: Vec<Vec<f64>> = {
        let mut rr = Vec::with_capacity(npts);
        let mut cc = Vec::with_capacity(npts);
        for i in 0..npts {
            let (y, x) = ((i / side) as f64, (i % side) as f64);
            rr.push(0.9 * y + 0.05 * x + 3.7);
            cc.push(0.9 * x - 0.05 * y + 2.3);
        }
        vec![rr, cc]
    };

    // Isolation mode: hammer a single path so `perf record` attributes cleanly.
    if mode_sel == "full" || mode_sel == "compact" {
        let order = only_order.unwrap_or(3);
        NDIMAGE_BSPLINE_COMPACT_DISABLE.store(mode_sel == "full", Ordering::Relaxed);
        for _ in 0..30 {
            black_box(
                rotate(&img, 33.0, false, order, BoundaryMode::Reflect, 0.0)
                    .unwrap()
                    .data
                    .len(),
            );
        }
        return;
    }

    println!("# same-binary A/B: ORIG full tap window vs compact support window (per-pixel path)");
    // Read-once knob: `true` = ORIG (evaluate 2*order+1 taps, discard the zeros).
    let set_orig = |orig: bool| NDIMAGE_BSPLINE_COMPACT_DISABLE.store(orig, Ordering::Relaxed);

    for &order in &[1usize, 2, 3, 4, 5] {
        if only_order.is_some_and(|o| o != order) {
            continue;
        }
        // Reflect/Mirror take the cardinal kernel the lever touches. Constant is a HARNESS
        // NULL CONTROL: it routes to fold_wrap_cubic (order 3) or bspline_local_support (else),
        // never the compact loop, so it MUST read ~1.00x. Any run whose control drifts off 1.00x
        // is noise-dominated and gets discarded.
        for &mode in &[
            BoundaryMode::Reflect,
            BoundaryMode::Mirror,
            BoundaryMode::Constant,
        ] {
            // Three non-separable kernels; each returns a flat Vec<f64> for bit comparison.
            let kernels: [(&str, Box<dyn Fn() -> Vec<f64>>); 3] = [
                (
                    "rotate",
                    Box::new(|| rotate(&img, 33.0, false, order, mode, 0.0).unwrap().data),
                ),
                (
                    "affine_gen",
                    Box::new(|| {
                        affine_transform(&img, &GENERAL_AFFINE, order, mode, 0.0)
                            .unwrap()
                            .data
                    }),
                ),
                (
                    "map_coords",
                    Box::new(|| map_coordinates(&img, &coords, order, mode, 0.0).unwrap()),
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
                let (om, ocv) = mean_cv(&ov);
                let (cm, ccv) = mean_cv(&cv);
                let ob = ov.iter().copied().fold(f64::MAX, f64::min);
                let cb = cv.iter().copied().fold(f64::MAX, f64::min);
                println!(
                    "order={order} {mode:?} {name}: full {ob:.2}ms (mean {om:.2} cv {ocv:.1}%) \
                     compact {cb:.2}ms (mean {cm:.2} cv {ccv:.1}%) best {:.2}x mean {:.2}x \
                     maxdiff={md:.1e} bitmism={bits}",
                    ob / cb,
                    om / cm
                );
            }
        }
    }
}
