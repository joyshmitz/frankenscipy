#![no_main]

//! Robustness fuzz harness for `fsci_ndimage::gaussian_filter` and
//! `fsci_ndimage::sobel`. Drives arbitrary 2D inputs and parameters and
//! checks: shape preserved, output finite, gaussian non-expansive on
//! extrema, sobel of a uniform image equals zero.
//!
//! Bead: `frankenscipy-p7i1`.

use arbitrary::Arbitrary;
use fsci_ndimage::{BoundaryMode, NdArray, gaussian_filter, sobel};
use libfuzzer_sys::fuzz_target;

const MIN_DIM: usize = 1;
const MAX_DIM: usize = 32;

#[derive(Debug, Arbitrary)]
struct NdimageInput {
    rows: u8,
    cols: u8,
    sigma_log: f64,
    axis: u8,
    mode_kind: u8,
    cval: f64,
    cells: Vec<f64>,
    test_uniform: bool,
}

fn sanitize_finite(value: f64, lo: f64, hi: f64, default: f64) -> f64 {
    if value.is_finite() {
        value.clamp(lo, hi)
    } else {
        default
    }
}

fn pick_mode(byte: u8) -> BoundaryMode {
    match byte % 3 {
        0 => BoundaryMode::Reflect,
        1 => BoundaryMode::Constant,
        _ => BoundaryMode::Nearest,
    }
}

fuzz_target!(|input: NdimageInput| {
    let rows = MIN_DIM + (input.rows as usize % (MAX_DIM - MIN_DIM + 1));
    let cols = MIN_DIM + (input.cols as usize % (MAX_DIM - MIN_DIM + 1));
    let n = rows * cols;
    let mut data = Vec::with_capacity(n);
    for k in 0..n {
        let raw = input.cells.get(k).copied().unwrap_or(0.0);
        data.push(sanitize_finite(raw, -1e3, 1e3, 0.0));
    }
    let img = match NdArray::new(data, vec![rows, cols]) {
        Ok(a) => a,
        Err(_) => return,
    };

    let mode = pick_mode(input.mode_kind);
    let cval = sanitize_finite(input.cval, -10.0, 10.0, 0.0);
    let sigma = 10f64.powf(sanitize_finite(input.sigma_log, -2.0, 1.0, 0.0));

    // ───────────── gaussian_filter ─────────────
    if let Ok(out) = gaussian_filter(&img, sigma, mode, cval) {
        assert_eq!(out.shape, img.shape, "gaussian_filter changed shape");
        for (i, &v) in out.data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "gaussian_filter produced non-finite at i={i}: {v}, sigma={sigma}, mode={mode:?}"
            );
        }
        // Non-expansive on extrema (only meaningful for Reflect/Nearest;
        // Constant mode can pull values toward `cval`).
        if matches!(mode, BoundaryMode::Reflect | BoundaryMode::Nearest) {
            let lo = img.data.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = img.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lo_out = out.data.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi_out = out.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let slack = 1e-9 * hi.abs().max(1.0);
            assert!(
                lo_out >= lo - slack,
                "gaussian min violated: {lo_out} < {lo} (sigma={sigma})"
            );
            assert!(
                hi_out <= hi + slack,
                "gaussian max violated: {hi_out} > {hi} (sigma={sigma})"
            );
        }
    }

    // ───────────── sobel ─────────────
    let axis = (input.axis as usize) % 2;
    if let Ok(out) = sobel(&img, axis, mode, cval) {
        assert_eq!(out.shape, img.shape, "sobel changed shape");
        for (i, &v) in out.data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "sobel produced non-finite at i={i}: {v}"
            );
        }
    }

    // ───────────── sobel of uniform image is 0 ─────────────
    if input.test_uniform {
        let value = sanitize_finite(input.cval, -100.0, 100.0, 1.0);
        let uniform = NdArray::new(vec![value; n], vec![rows, cols]).unwrap();
        let out = sobel(&uniform, axis, mode, cval).unwrap();
        for (i, &v) in out.data.iter().enumerate() {
            assert!(
                v.abs() < 1e-9 * value.abs().max(1.0),
                "sobel(uniform={value}) at i={i} not zero: {v}"
            );
        }
    }
});
