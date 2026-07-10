//! A/B probe for `shift`: separable precompute vs the legacy serial per-pixel loop, and
//! (mode `offs`) the flat-offset tensor combine vs the ORIG index-space leaf.
//!
//! Usage: `perf_shift_sep [both|offs]`
use fsci_ndimage::{
    BoundaryMode, NDIMAGE_SPLINE_OFFSET_DISABLE, NDIMAGE_ZOOM_SEPARABLE_DISABLE, NdArray, shift,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn mean_cv(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, if m > 0.0 { var.sqrt() / m * 100.0 } else { 0.0 })
}

fn main() {
    let offs_ab = std::env::args().nth(1).as_deref() == Some("offs");
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
    let sh = [3.4f64, -2.7f64];

    if offs_ab {
        NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(false, Ordering::Relaxed);
        println!("# same-binary A/B: ORIG index-space leaf vs flat-offset leaf (separable path)");
    }
    let set_orig = |orig: bool| {
        if offs_ab {
            NDIMAGE_SPLINE_OFFSET_DISABLE.store(orig, Ordering::Relaxed);
        } else {
            NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(orig, Ordering::Relaxed);
        }
    };

    // The separable path is gated order>=2; the offs lever only exists there.
    let orders: &[usize] = if offs_ab {
        &[2, 3, 5]
    } else {
        &[0, 1, 2, 3, 5]
    };
    for &order in orders {
        let run = |orig: bool| {
            set_orig(orig);
            shift(&img, &sh, order, BoundaryMode::Reflect, 0.0).unwrap()
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
                shift(&img, &sh, order, BoundaryMode::Reflect, 0.0)
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
        let (mut ov, mut cv) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            ov.push(bench(true));
            cv.push(bench(false));
        }
        let (om, ocv) = mean_cv(&ov);
        let (cm, ccv) = mean_cv(&cv);
        let ob = ov.iter().copied().fold(f64::MAX, f64::min);
        let cb = cv.iter().copied().fold(f64::MAX, f64::min);
        let label = if offs_ab { "idxleaf" } else { "orig(serial)" };
        let clabel = if offs_ab { "offsleaf" } else { "new" };
        println!(
            "order={order}: {label} {ob:.2}ms (mean {om:.2} cv {ocv:.1}%) {clabel} {cb:.2}ms \
             (mean {cm:.2} cv {ccv:.1}%) best {:.2}x mean {:.2}x maxdiff={md:.1e} bitmism={bits}",
            ob / cb,
            om / cm
        );
    }
}
