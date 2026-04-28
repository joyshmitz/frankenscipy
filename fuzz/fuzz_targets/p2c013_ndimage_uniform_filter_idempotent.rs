#![no_main]

use arbitrary::Arbitrary;
use fsci_ndimage::{BoundaryMode, NdArray, uniform_filter};
use libfuzzer_sys::fuzz_target;

// Ndimage uniform filter idempotence oracle:
// For a constant array (all same value), uniform_filter should return
// the same constant array (within floating-point tolerance).
//
// This catches:
// - Boundary handling errors that shift values
// - Normalization bugs in the averaging kernel
// - Off-by-one in filter size computation

const MAX_SIZE: usize = 64;
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

fuzz_target!(|input: FilterInput| {
    let w = (input.width as usize).clamp(1, MAX_SIZE);
    let h = (input.height as usize).clamp(1, MAX_SIZE);
    let size = (input.filter_size as usize).clamp(1, w.min(h).min(15));
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
    let result = match uniform_filter(&array, size, mode, cval) {
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
                "Uniform filter non-idempotent at index {}: \
                 got {} expected {} (value={}, {}x{}, size={}, mode={:?})",
                i, v, expected, value, w, h, size, mode
            );
        }
    }
});
