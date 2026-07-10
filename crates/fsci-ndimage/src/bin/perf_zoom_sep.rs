//! A/B probe for the zoom separable B-spline support precompute + flat-offset tensor combine.
//!
//! Usage: `perf_zoom_sep [mode] [order]`
//!   both (default) — interleaved A/B: per-pixel support recompute vs separable precompute
//!   offs           — interleaved A/B: ORIG index-space leaf vs flat-offset leaf
//!   odom           — interleaved A/B: ORIG per-pixel unravel vs per-chunk row-major odometer
//!   sep | per      — run ONE path in a tight loop (for `perf record` isolation)
use fsci_ndimage::{
    BoundaryMode, NDIMAGE_SPLINE_OFFSET_DISABLE, NDIMAGE_UNRAVEL_ODOMETER_DISABLE,
    NDIMAGE_ZOOM_SEPARABLE_DISABLE, NdArray, zoom,
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode_sel = args.get(1).map_or("both", |s| s.as_str());
    let only_order: Option<usize> = args.get(2).and_then(|s| s.parse().ok());

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

    // Isolation mode: hammer a single path so `perf record` attributes cleanly.
    if mode_sel == "sep" || mode_sel == "per" {
        let disable = mode_sel == "per";
        let order = only_order.unwrap_or(3);
        NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(disable, Ordering::Relaxed);
        for _ in 0..40 {
            black_box(
                zoom(&img, &[2.0, 2.0], order, BoundaryMode::Reflect, 0.0)
                    .unwrap()
                    .data
                    .len(),
            );
        }
        return;
    }

    // `offs` isolates the leaf-address lever: same binary, same separable precompute, only the
    // leaf address differs (ORIG recomputes Σ idx·stride per leaf; candidate uses flat offsets).
    // `odom` isolates the index lever: ORIG heap-allocates two Vecs per pixel in
    // `unravel_with_shape`; candidate seeds a row-major odometer once per thread chunk.
    let offs_ab = mode_sel == "offs";
    let odom_ab = mode_sel == "odom";
    if offs_ab || odom_ab {
        NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(false, Ordering::Relaxed);
        let what = if offs_ab {
            "ORIG index-space leaf vs flat-offset leaf"
        } else {
            "ORIG per-pixel unravel vs per-chunk odometer"
        };
        println!("# same-binary A/B: {what}");
    }
    let set_orig = |orig: bool| {
        if offs_ab {
            NDIMAGE_SPLINE_OFFSET_DISABLE.store(orig, Ordering::Relaxed);
        } else if odom_ab {
            NDIMAGE_UNRAVEL_ODOMETER_DISABLE.store(orig, Ordering::Relaxed);
        } else {
            NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(orig, Ordering::Relaxed);
        }
    };

    for &order in &[1usize, 2, 3, 4, 5] {
        if only_order.is_some_and(|o| o != order) {
            continue;
        }
        // order<2 has no separable path; the offs lever only exists there. The odometer lever
        // covers BOTH the separable and the generic per-pixel path, so order=1 stays in.
        if offs_ab && order < 2 {
            continue;
        }
        for &mode in &[BoundaryMode::Reflect, BoundaryMode::Mirror] {
            let run = |orig: bool| {
                set_orig(orig);
                zoom(&img, &[2.0, 2.0], order, mode, 0.0).unwrap()
            };
            let cand = run(false);
            let orig = run(true);
            let md = cand
                .data
                .iter()
                .zip(&orig.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            let bits = cand
                .data
                .iter()
                .zip(&orig.data)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();

            let bench = |orig: bool| {
                set_orig(orig);
                let f = || {
                    zoom(&img, &[2.0, 2.0], order, mode, 0.0)
                        .unwrap()
                        .data
                        .len()
                };
                let _ = black_box(f());
                let reps = 4;
                let t = Instant::now();
                for _ in 0..reps {
                    black_box(f());
                }
                t.elapsed().as_secs_f64() / reps as f64 * 1000.0
            };
            // Interleave ORIG/candidate so slow drift hits both arms equally.
            let (mut ov, mut cv) = (Vec::new(), Vec::new());
            for _ in 0..5 {
                ov.push(bench(true));
                cv.push(bench(false));
            }
            let (om, ocv) = mean_cv(&ov);
            let (cm, ccv) = mean_cv(&cv);
            let ob = ov.iter().copied().fold(f64::MAX, f64::min);
            let cb = cv.iter().copied().fold(f64::MAX, f64::min);
            let (label, clabel) = if offs_ab {
                ("idxleaf", "offsleaf")
            } else if odom_ab {
                ("unravel", "odometer")
            } else {
                ("perpixel", "separable")
            };
            println!(
                "order={order} {mode:?}: {label} {ob:.2}ms (mean {om:.2} cv {ocv:.1}%) \
                 {clabel} {cb:.2}ms (mean {cm:.2} cv {ccv:.1}%) best {:.2}x mean {:.2}x \
                 maxdiff={md:.1e} bitmism={bits}",
                ob / cb,
                om / cm
            );
        }
    }
}
