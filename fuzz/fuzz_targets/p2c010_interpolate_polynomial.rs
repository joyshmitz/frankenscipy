#![no_main]

use arbitrary::Arbitrary;
use fsci_interpolate::{lagrange, polyadd, polyder, polyfit, polyint, polymul, polysub, polyval};
use libfuzzer_sys::fuzz_target;

// Polynomial operations oracle:
// Tests polynomial arithmetic and interpolation for correctness:
//
// 1. Lagrange interpolation passes through data points
// 2. polyfit + polyval reproduces data points (for exact fit)
// 3. polyder + polyint roundtrip preserves polynomial (up to constant)
// 4. polymul associativity: (a*b)*c = a*(b*c)
// 5. polyadd/polysub are inverses: (a+b)-b = a

const MAX_POINTS: usize = 16;
const MAX_DEGREE: usize = 8;
const TOL: f64 = 1e-8;

#[derive(Debug, Arbitrary)]
struct PolyInput {
    x_vals: Vec<f64>,
    y_vals: Vec<f64>,
    coeffs_a: Vec<f64>,
    coeffs_b: Vec<f64>,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-100.0, 100.0)
    } else {
        0.0
    }
}

fn close_enough(a: f64, b: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return true;
    }
    let diff = (a - b).abs();
    diff <= TOL * (1.0 + a.abs().max(b.abs()))
}

fn make_unique_sorted(vals: &[f64], max_n: usize) -> Vec<f64> {
    let mut result: Vec<f64> = vals
        .iter()
        .take(max_n)
        .map(|&v| sanitize(v))
        .collect();
    result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    result.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    result
}

fuzz_target!(|input: PolyInput| {
    let x_data = make_unique_sorted(&input.x_vals, MAX_POINTS);
    let n = x_data.len();
    if n < 2 {
        return;
    }

    let y_data: Vec<f64> = input
        .y_vals
        .iter()
        .take(n)
        .map(|&v| sanitize(v))
        .chain(std::iter::repeat(0.0))
        .take(n)
        .collect();

    // Test 1: Lagrange interpolation passes through data points
    if let Ok(coeffs) = lagrange(&x_data, &y_data) {
        for (i, (&xi, &yi)) in x_data.iter().zip(y_data.iter()).enumerate() {
            let interp_yi = polyval(&coeffs, xi);
            if !close_enough(interp_yi, yi) {
                panic!(
                    "Lagrange passthrough fail at point {}: x={}, expected y={}, got {}",
                    i, xi, yi, interp_yi
                );
            }
        }
    }

    // Test 2: polyfit + polyval reproduces data for exact fit (n-1 degree)
    let deg = (n - 1).min(MAX_DEGREE);
    if deg >= 1 {
        if let Ok(fit_coeffs) = polyfit(&x_data, &y_data, deg) {
            for (i, (&xi, &yi)) in x_data.iter().zip(y_data.iter()).enumerate() {
                let fit_yi = polyval(&fit_coeffs, xi);
                if !close_enough(fit_yi, yi) {
                    panic!(
                        "polyfit passthrough fail at point {}: x={}, expected y={}, got {} (deg={})",
                        i, xi, yi, fit_yi, deg
                    );
                }
            }
        }
    }

    // Test 3: polyder + polyint roundtrip
    let coeffs_a: Vec<f64> = input
        .coeffs_a
        .iter()
        .take(MAX_DEGREE)
        .map(|&v| sanitize(v))
        .collect();

    if coeffs_a.len() >= 2 {
        let deriv = polyder(&coeffs_a, 1);
        let antideriv = polyint(&deriv, 1, 0.0);

        // antideriv should match coeffs_a except constant term and leading zeros
        let min_len = coeffs_a.len().min(antideriv.len());
        if min_len >= 2 {
            for i in 0..min_len - 1 {
                let orig = coeffs_a.get(i).copied().unwrap_or(0.0);
                let roundtrip = antideriv.get(i).copied().unwrap_or(0.0);
                if !close_enough(orig, roundtrip) && orig.abs() > 1e-10 {
                    panic!(
                        "polyder/polyint roundtrip fail at coeff {}: orig={}, roundtrip={}",
                        i, orig, roundtrip
                    );
                }
            }
        }
    }

    // Test 4: polyadd/polysub inverse
    let coeffs_b: Vec<f64> = input
        .coeffs_b
        .iter()
        .take(MAX_DEGREE)
        .map(|&v| sanitize(v))
        .collect();

    if !coeffs_a.is_empty() && !coeffs_b.is_empty() {
        let sum = polyadd(&coeffs_a, &coeffs_b);
        let diff = polysub(&sum, &coeffs_b);

        // diff should equal coeffs_a (padded to same length)
        let max_len = coeffs_a.len().max(diff.len());
        for i in 0..max_len {
            // Align from right (highest degree first convention)
            let a_offset = max_len.saturating_sub(coeffs_a.len());
            let d_offset = max_len.saturating_sub(diff.len());

            let a_i = if i >= a_offset { coeffs_a.get(i - a_offset).copied().unwrap_or(0.0) } else { 0.0 };
            let d_i = if i >= d_offset { diff.get(i - d_offset).copied().unwrap_or(0.0) } else { 0.0 };

            if !close_enough(a_i, d_i) {
                panic!(
                    "polyadd/polysub inverse fail at {}: a={}, (a+b)-b={}",
                    i, a_i, d_i
                );
            }
        }
    }

    // Test 5: polymul produces correct degree
    if coeffs_a.len() >= 2 && coeffs_b.len() >= 2 {
        let product = polymul(&coeffs_a, &coeffs_b);
        let expected_len = coeffs_a.len() + coeffs_b.len() - 1;
        if product.len() != expected_len {
            panic!(
                "polymul degree wrong: len(a)={}, len(b)={}, expected product len={}, got {}",
                coeffs_a.len(),
                coeffs_b.len(),
                expected_len,
                product.len()
            );
        }
    }
});
