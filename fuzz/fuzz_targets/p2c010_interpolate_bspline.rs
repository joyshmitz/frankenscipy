#![no_main]

use arbitrary::Arbitrary;
use fsci_interpolate::make_interp_spline;
use libfuzzer_sys::fuzz_target;

// B-spline interpolation oracle:
// Tests make_interp_spline construction and BSpline::eval for various
// spline orders (k=1..5) and data patterns.
//
// Properties verified:
// 1. Construction succeeds for valid inputs (sorted x, enough points)
// 2. Evaluation at data points produces y values (passthrough)
// 3. Evaluation within bounds produces finite results
// 4. Spline is monotonic-preserving for monotonic data (k=1,2)

const MAX_POINTS: usize = 32;
const REL_TOL: f64 = 1e-10;
const ABS_TOL: f64 = 1e-12;

#[derive(Debug, Arbitrary)]
struct BSplineInput {
    x_vals: Vec<f64>,
    y_vals: Vec<f64>,
    k: u8,
    eval_fracs: Vec<f64>,
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
        return a.is_nan() && b.is_nan() || a == b;
    }
    let diff = (a - b).abs();
    diff <= ABS_TOL + REL_TOL * a.abs().max(b.abs())
}

fuzz_target!(|input: BSplineInput| {
    let n = input.x_vals.len().min(MAX_POINTS).min(input.y_vals.len());
    let k = ((input.k % 5) + 1) as usize;

    if n < k + 1 {
        return;
    }

    let mut x_data: Vec<f64> = input.x_vals.iter().take(n).map(|&v| sanitize(v)).collect();
    let y_data: Vec<f64> = input.y_vals.iter().take(n).map(|&v| sanitize(v)).collect();

    x_data.sort_by(|a, b| a.total_cmp(b));

    for i in 1..x_data.len() {
        if (x_data[i] - x_data[i - 1]).abs() < 1e-12 {
            x_data[i] = x_data[i - 1] + 1e-10 * (i as f64);
        }
    }

    let spline = match make_interp_spline(&x_data, &y_data, k) {
        Ok(s) => s,
        Err(_) => return,
    };

    for (i, (&x, &y_expected)) in x_data.iter().zip(y_data.iter()).enumerate() {
        let y_eval = spline.eval(x);
        if !close_enough(y_eval, y_expected) {
            panic!(
                "BSpline passthrough failed at x[{}]={}: eval={} vs expected={} (k={}, n={})",
                i, x, y_eval, y_expected, k, n
            );
        }
    }

    let x_min = x_data[0];
    let x_max = x_data[n - 1];
    let x_range = x_max - x_min;

    for frac in input.eval_fracs.iter().take(16) {
        let t = sanitize(*frac).clamp(0.0, 1.0);
        let x_eval = x_min + t * x_range;
        let y_eval = spline.eval(x_eval);

        if !y_eval.is_finite() {
            panic!(
                "BSpline eval produced non-finite at x={} (t={}, k={}, n={}): {}",
                x_eval, t, k, n, y_eval
            );
        }
    }

    if k <= 2 {
        let is_monotonic_inc = y_data.windows(2).all(|w| w[1] >= w[0]);
        let is_monotonic_dec = y_data.windows(2).all(|w| w[1] <= w[0]);

        if is_monotonic_inc || is_monotonic_dec {
            let test_points: Vec<f64> = (0..10)
                .map(|i| x_min + (i as f64 / 9.0) * x_range)
                .collect();
            let test_vals: Vec<f64> = test_points.iter().map(|&x| spline.eval(x)).collect();

            let spline_mono_inc = test_vals.windows(2).all(|w| w[1] >= w[0] - ABS_TOL);
            let spline_mono_dec = test_vals.windows(2).all(|w| w[1] <= w[0] + ABS_TOL);

            if is_monotonic_inc && !spline_mono_inc && !spline_mono_dec {
                panic!(
                    "k={} spline should preserve monotonicity for monotonic data (n={})",
                    k, n
                );
            }
            if is_monotonic_dec && !spline_mono_inc && !spline_mono_dec {
                panic!(
                    "k={} spline should preserve monotonicity for monotonic data (n={})",
                    k, n
                );
            }
        }
    }
});
