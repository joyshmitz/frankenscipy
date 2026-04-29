#![no_main]

use arbitrary::Arbitrary;
use fsci_ndimage::{
    BoundaryMode, NdArray, gaussian_filter, maximum_filter, median_filter, minimum_filter,
    uniform_filter,
};
use libfuzzer_sys::fuzz_target;

// Ndimage constant-field filter invariant:
// For a constant array with cval set to the same constant, smoothing and rank
// filters should return the same constant array within floating-point tolerance.
//
// This catches:
// - Boundary handling errors that shift values
// - Normalization bugs in smoothing kernels
// - Rank-window ordering mistakes for constant neighborhoods
// - Off-by-one in filter size computation

const MAX_DIM: usize = 16;
const MAX_FILTER_SIZE: usize = 7;
const REL_TOL: f64 = 1e-12;
const ABS_TOL: f64 = 1e-14;

#[derive(Debug, Arbitrary)]
struct FilterInput {
    value: f64,
    width: u8,
    height: u8,
    filter_size: u8,
    mode_variant: u8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn close_enough(a: f64, b: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return true;
    }
    let diff = (a - b).abs();
    diff <= ABS_TOL + REL_TOL * a.abs().max(b.abs())
}

fn filter_name(mode_variant: u8) -> &'static str {
    match (mode_variant / 4) % 5 {
        0 => "uniform_filter",
        1 => "gaussian_filter",
        2 => "median_filter",
        3 => "minimum_filter",
        _ => "maximum_filter",
    }
}

fuzz_target!(|input: FilterInput| {
    let w = (input.width as usize).clamp(1, MAX_DIM);
    let h = (input.height as usize).clamp(1, MAX_DIM);
    let size = (input.filter_size as usize).clamp(1, MAX_FILTER_SIZE);
    let value = sanitize(input.value);

    let data: Vec<f64> = vec![value; w * h];
    let array = match NdArray::new(data.clone(), vec![h, w]) {
        Ok(a) => a,
        Err(_) => return,
    };

    let mode = match input.mode_variant % 4 {
        0 => BoundaryMode::Constant,
        1 => BoundaryMode::Reflect,
        2 => BoundaryMode::Wrap,
        _ => BoundaryMode::Nearest,
    };

    let cval = value;
    let filter = filter_name(input.mode_variant);
    let result = match filter {
        "uniform_filter" => uniform_filter(&array, size, mode, cval),
        "gaussian_filter" => {
            let sigma = (size as f64).mul_add(0.25, 0.25);
            gaussian_filter(&array, sigma, mode, cval)
        }
        "median_filter" => median_filter(&array, size, mode, cval),
        "minimum_filter" => minimum_filter(&array, size, mode, cval),
        "maximum_filter" => maximum_filter(&array, size, mode, cval),
        _ => return,
    };
    let result = match result {
        Ok(r) => r,
        Err(_) => return,
    };

    let result_data = &result.data;
    if result_data.len() != data.len() {
        panic!(
            "Filter output size mismatch: got {} expected {} ({}x{}, size={})",
            result_data.len(),
            data.len(),
            w,
            h,
            size
        );
    }

    let expected = value;

    for (i, &v) in result_data.iter().enumerate() {
        if !close_enough(v, expected) {
            panic!(
                "Ndimage filter non-idempotent at index {}: \
                 got {} expected {} (filter={}, value={}, {}x{}, size={}, mode={:?})",
                i, v, expected, filter, value, w, h, size, mode
            );
        }
    }
});
