//! Metamorphic tests for `fsci-interpolate`.
//!
//! Interpolant condition (passes through data), monotonicity preservation
//! (PCHIP), exactness on linear data, BSpline continuity at knots.
//!
//! Run with: `cargo test -p fsci-interpolate --test metamorphic_tests`

use fsci_interpolate::{
    Akima1DInterpolator, CubicSplineStandalone, Interp1d, Interp1dOptions, InterpKind,
    PchipInterpolator, SplineBc, make_interp_spline,
};

const ATOL: f64 = 1e-10;
const RTOL: f64 = 1e-9;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — interp1d at a data point returns the data value (interpolant
// condition) for Linear, Nearest and CubicSpline kinds.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_interp1d_passes_through_data() {
    let x = vec![0.0, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0];
    let y = vec![0.0, 1.0, 4.0, 12.25, 25.0, 56.25, 100.0]; // y = x²

    for &kind in &[InterpKind::Linear, InterpKind::Nearest, InterpKind::CubicSpline] {
        let opts = Interp1dOptions {
            kind,
            ..Default::default()
        };
        let interp = Interp1d::new(&x, &y, opts).unwrap();
        for (xi, yi) in x.iter().zip(&y) {
            let v = interp.eval(*xi).unwrap();
            assert!(
                close(v, *yi),
                "MR1 {kind:?} at x={xi}: got {v}, expected {yi}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — Linear interpolation is exact for linearly-sampled data.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_interp1d_linear_exact_on_linear_data() {
    let m = 2.5_f64;
    let c = -1.3_f64;
    let x: Vec<f64> = (0..10).map(|i| 0.5 * i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| m * xi + c).collect();
    let opts = Interp1dOptions {
        kind: InterpKind::Linear,
        ..Default::default()
    };
    let interp = Interp1d::new(&x, &y, opts).unwrap();
    // Interpolate at 100 in-domain query points; the linear interpolant
    // must match the underlying line to round-off.
    for k in 0..=100 {
        let xq = x[0] + (x[x.len() - 1] - x[0]) * (k as f64 / 100.0);
        let v = interp.eval(xq).unwrap();
        let expected = m * xq + c;
        assert!(
            close(v, expected),
            "MR2 linear interp not exact at xq={xq}: got {v}, expected {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — CubicSplineStandalone passes through every data point.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cubic_spline_passes_through_data() {
    let x = vec![-2.0_f64, -0.5, 0.7, 1.4, 3.0, 5.5];
    let y = vec![4.0, 0.25, 0.49, 1.96, 9.0, 30.25]; // y = x²
    for bc in [SplineBc::Natural, SplineBc::NotAKnot] {
        let cs = CubicSplineStandalone::new(&x, &y, bc).unwrap();
        for (xi, yi) in x.iter().zip(&y) {
            let v = cs.eval(*xi);
            assert!(
                close(v, *yi),
                "MR3 cubic spline {bc:?} at x={xi}: got {v}, expected {yi}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — PchipInterpolator preserves monotonicity for monotone-increasing
// inputs: pchip(x) is monotone-increasing.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pchip_preserves_monotonicity() {
    // Strictly-increasing y over strictly-increasing x.
    let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![0.0, 0.5, 1.0, 1.7, 2.0, 2.4, 3.5, 4.5, 5.0];
    let pchip = PchipInterpolator::new(&x, &y).unwrap();
    let mut prev = pchip.eval(x[0]);
    for k in 1..=200 {
        let xq = x[0] + (x[x.len() - 1] - x[0]) * (k as f64 / 200.0);
        let v = pchip.eval(xq);
        assert!(
            v >= prev - 1e-12,
            "MR4 pchip non-monotone at k={k} xq={xq}: prev={prev} v={v}"
        );
        prev = v;
    }
    // And it interpolates the data.
    for (xi, yi) in x.iter().zip(&y) {
        let v = pchip.eval(*xi);
        assert!(close(v, *yi), "MR4 pchip at x={xi}: got {v}, expected {yi}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — Akima1DInterpolator passes through data points.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_akima_passes_through_data() {
    let x = vec![0.0, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0];
    let y = vec![1.0, 2.5, 1.0, 3.7, 2.0, 5.5, 0.0];
    let akima = Akima1DInterpolator::new(&x, &y).unwrap();
    for (xi, yi) in x.iter().zip(&y) {
        let v = akima.eval(*xi);
        assert!(close(v, *yi), "MR5 akima at x={xi}: got {v}, expected {yi}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — BSpline of degree k built from data interpolates the data
// (since make_interp_spline produces an interpolating spline).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bspline_interp_passes_through_data() {
    let x: Vec<f64> = (0..9).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    for k in [1usize, 3, 5] {
        let spline = make_interp_spline(&x, &y, k).unwrap();
        for (xi, yi) in x.iter().zip(&y) {
            let v = spline.eval(*xi);
            assert!(
                close(v, *yi),
                "MR6 bspline k={k} at x={xi}: got {v}, expected {yi}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — CubicSplineStandalone is C¹ continuous: at every interior knot,
// the derivative from the left equals the derivative from the right.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cubic_spline_c1_continuity() {
    let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![0.0, 1.0, 0.5, 2.0, 1.5, 3.0];
    let cs = CubicSplineStandalone::new(&x, &y, SplineBc::NotAKnot).unwrap();
    let h = 1e-6;
    for &xi in &x[1..x.len() - 1] {
        let left = (cs.eval(xi) - cs.eval(xi - h)) / h;
        let right = (cs.eval(xi + h) - cs.eval(xi)) / h;
        assert!(
            (left - right).abs() < 1e-4,
            "MR7 cubic spline C¹ break at xi={xi}: left'={left} right'={right}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — Linear interpolation between two adjacent samples lies between
// the two y-values (no overshoot for the Linear kind).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_linear_interp_no_overshoot() {
    let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 5.0, -2.0, 7.0, 1.0];
    let interp = Interp1d::new(
        &x,
        &y,
        Interp1dOptions {
            kind: InterpKind::Linear,
            ..Default::default()
        },
    )
    .unwrap();
    // For each interval, sample 20 interior points; values must lie in
    // [min(y_i, y_{i+1}), max(y_i, y_{i+1})].
    for i in 0..(x.len() - 1) {
        let (lo, hi) = (y[i].min(y[i + 1]), y[i].max(y[i + 1]));
        for k in 0..=20 {
            let alpha = k as f64 / 20.0;
            let xq = x[i] + alpha * (x[i + 1] - x[i]);
            let v = interp.eval(xq).unwrap();
            assert!(
                v >= lo - 1e-12 && v <= hi + 1e-12,
                "MR8 linear overshoot in [{}, {}]: xq={xq} v={v} bounds=[{lo}, {hi}]",
                x[i],
                x[i + 1]
            );
        }
    }
}
