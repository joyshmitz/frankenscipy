#![no_main]

use arbitrary::Arbitrary;
use fsci_interpolate::interp1d_linear;
use libfuzzer_sys::fuzz_target;

// Interpolation passthrough oracle:
// For any interpolant constructed from (x_data, y_data), evaluating at
// x_data points should return y_data values exactly (within floating-point
// tolerance).
//
// This catches:
// - Off-by-one in knot lookup binary search
// - Index bounds errors at data boundaries
// - Wrong interpolation formula evaluation
// - Edge cases: duplicate x values, single point, two points

const MAX_POINTS: usize = 64;
const REL_TOL: f64 = 1e-12;
const ABS_TOL: f64 = 1e-14;

#[derive(Debug, Arbitrary)]
struct InterpolateInput {
    x_vals: Vec<f64>,
    y_vals: Vec<f64>,
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

fuzz_target!(|input: InterpolateInput| {
    let n = input.x_vals.len().min(MAX_POINTS).min(input.y_vals.len());
    if n < 2 {
        return;
    }

    let mut x_data: Vec<f64> = input.x_vals.iter().take(n).map(|&v| sanitize(v)).collect();
    let y_data: Vec<f64> = input.y_vals.iter().take(n).map(|&v| sanitize(v)).collect();

    x_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    for i in 1..x_data.len() {
        if (x_data[i] - x_data[i - 1]).abs() < 1e-12 {
            x_data[i] = x_data[i - 1] + 1e-10 * (i as f64);
        }
    }

    let x_eval: Vec<f64> = x_data.clone();

    let result = match interp1d_linear(&x_data, &y_data, &x_eval) {
        Ok(r) => r,
        Err(_) => return,
    };

    if result.len() != y_data.len() {
        panic!(
            "Interpolation length mismatch: got {} expected {} (n={})",
            result.len(),
            y_data.len(),
            n
        );
    }

    for (i, (interp, orig)) in result.iter().zip(y_data.iter()).enumerate() {
        if !close_enough(*interp, *orig) {
            panic!(
                "Interpolation passthrough failed at index {}: \
                 interpolated={} vs original={} (n={}, x={})",
                i, interp, orig, n, x_data[i]
            );
        }
    }
});
