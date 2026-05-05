#![no_main]

use arbitrary::Arbitrary;
use fsci_interpolate::{
    akima1d_interpolate, interp1d_linear, krogh_interpolate, pchip_interpolate,
};
use libfuzzer_sys::fuzz_target;

// 1D interpolation oracle for [frankenscipy-1iavk].
//
// Drives random small (xi, yi) inputs and picks one of four 1D
// interpolators. Verifies three invariants:
//
//   1. Output length matches query length
//   2. Pass-through at input nodes: f(x_i) ≈ y_i
//   3. All output values are finite (no spurious NaN/Inf)
//
// Catches regressions in the akima1d slope estimator, the krogh
// divided-difference table, or the pchip monotone-cubic edge cases.

const MAX_NODES: usize = 16;
const MAX_QUERIES: usize = 16;

#[derive(Debug, Arbitrary)]
struct Interp1dInput {
    raw_x: Vec<f64>,
    raw_y: Vec<f64>,
    raw_query: Vec<f64>,
    method: u8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn build_strictly_increasing_x(raw: &[f64], cap: usize) -> Option<Vec<f64>> {
    let mut xs: Vec<f64> = raw.iter().take(cap).map(|&v| sanitize(v)).collect();
    if xs.len() < 4 {
        return None;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Make strictly increasing by perturbing duplicates by epsilon.
    for i in 1..xs.len() {
        if xs[i] <= xs[i - 1] {
            xs[i] = xs[i - 1] + 1e-6;
        }
    }
    Some(xs)
}

fn dispatch(method: u8) -> &'static str {
    match method % 4 {
        0 => "pchip",
        1 => "akima1d",
        2 => "krogh",
        _ => "interp1d_linear",
    }
}

fn interp(
    name: &str,
    xi: &[f64],
    yi: &[f64],
    x_new: &[f64],
) -> Option<Vec<f64>> {
    match name {
        "pchip" => pchip_interpolate(xi, yi, x_new).ok(),
        "akima1d" => akima1d_interpolate(xi, yi, x_new).ok(),
        "krogh" => krogh_interpolate(xi, yi, x_new).ok(),
        _ => interp1d_linear(xi, yi, x_new).ok(),
    }
}

fuzz_target!(|input: Interp1dInput| {
    let Some(xi) = build_strictly_increasing_x(&input.raw_x, MAX_NODES) else {
        return;
    };
    let n = xi.len();
    let yi: Vec<f64> = input
        .raw_y
        .iter()
        .take(n)
        .map(|&v| sanitize(v))
        .collect();
    if yi.len() != n {
        return;
    }
    let queries: Vec<f64> = input
        .raw_query
        .iter()
        .take(MAX_QUERIES)
        .map(|&v| sanitize(v).clamp(xi[0], xi[n - 1]))
        .collect();
    if queries.is_empty() {
        return;
    }

    let method = dispatch(input.method);
    let Some(out) = interp(method, &xi, &yi, &queries) else {
        return;
    };

    // Property 1: shape preservation.
    if out.len() != queries.len() {
        panic!(
            "{method}: output length {} != query length {}",
            out.len(),
            queries.len()
        );
    }

    // Property 3: all finite.
    for (i, &v) in out.iter().enumerate() {
        if !v.is_finite() {
            panic!(
                "{method}: non-finite output[{i}] = {v} for xi[..]={:?} yi[..]={:?}",
                xi.first(),
                yi.first()
            );
        }
    }

    // Property 2: passing through input nodes.
    let Some(at_nodes) = interp(method, &xi, &yi, &xi) else {
        return;
    };
    if at_nodes.len() != n {
        panic!(
            "{method}: at-nodes length {} != n={n}",
            at_nodes.len()
        );
    }
    for (i, (&got, &exp)) in at_nodes.iter().zip(yi.iter()).enumerate() {
        // Tolerance scales with the magnitude of the data.
        let scale = exp.abs().max(1.0);
        if (got - exp).abs() > 1e-6 * scale + 1e-6 {
            panic!(
                "{method}: f(x_{i}={:.6}) = {got}, expected y_{i} = {exp}",
                xi[i]
            );
        }
    }
});
