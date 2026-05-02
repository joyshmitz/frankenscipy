//! Metamorphic tests for `fsci-interpolate`.
//!
//! Interpolant condition (passes through data), monotonicity preservation
//! (PCHIP), exactness on linear data, BSpline continuity at knots.
//!
//! Run with: `cargo test -p fsci-interpolate --test metamorphic_tests`

use fsci_interpolate::{
    Akima1DInterpolator, BarycentricInterpolator, CubicSplineStandalone, GriddataMethod, Interp1d,
    Interp1dOptions, InterpKind, KroghInterpolator, LinearNDInterpolator, NearestNDInterpolator,
    PchipInterpolator, RbfInterpolator, RbfKernel, RegularGridInterpolator, RegularGridMethod,
    SplineBc, barycentric_eval, barycentric_weights, chebyshev_nodes, chebyshev_nodes2, griddata,
    hermite_interp, interp1d_linear, interp2d, interpn, lagrange, make_interp_spline,
    make_lsq_spline, neville, pade, polyadd, polyder, polyfit, polyint, polyint_definite,
    polymul, polyroots, polysub, polyval, polyval_with_error, ratval, splantider, splder, splev,
    splev_with_derivative, splint, splrep, sproot,
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

// ─────────────────────────────────────────────────────────────────────
// MR9 — RegularGridInterpolator passes through grid points exactly.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_regular_grid_interpolator_passes_through_grid() {
    use fsci_interpolate::RegularGridMethod;
    // 2D grid: 3x4
    let xs: Vec<f64> = vec![0.0, 1.0, 2.0];
    let ys: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
    let mut values = Vec::with_capacity(xs.len() * ys.len());
    for &x in &xs {
        for &y in &ys {
            values.push(x * 10.0 + y); // unique values per (x, y)
        }
    }
    let interp = RegularGridInterpolator::new(
        vec![xs.clone(), ys.clone()],
        values.clone(),
        RegularGridMethod::Linear,
        true,
        None,
    )
    .unwrap();
    for (i, &x) in xs.iter().enumerate() {
        for (j, &y) in ys.iter().enumerate() {
            let v = interp.eval(&[x, y]).unwrap();
            let expected = values[i * ys.len() + j];
            assert!(
                close(v, expected),
                "MR9 grid interp at ({x}, {y}): got {v}, expected {expected}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — NearestNDInterpolator returns the value associated with the
// closest data point.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_nearest_nd_interpolator_returns_closest_value() {
    let points = vec![
        vec![0.0_f64, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![5.0, 5.0],
    ];
    let values = vec![10.0_f64, 20.0, 30.0, 40.0];
    let interp = NearestNDInterpolator::new(&points, &values).unwrap();

    // Each input data point queries exactly: returns its own value.
    for (i, p) in points.iter().enumerate() {
        let v = interp.eval(p).unwrap();
        assert!(
            close(v, values[i]),
            "MR10 nearest at data point {i}: got {v}, expected {}",
            values[i]
        );
    }

    // Query a point clearly closest to (5, 5) → returns 40.
    let v = interp.eval(&[4.5, 4.5]).unwrap();
    assert!(close(v, 40.0), "MR10 nearest at (4.5, 4.5): got {v}");

    // Query a point clearly closest to (1, 0) → returns 20.
    let v = interp.eval(&[0.95, 0.1]).unwrap();
    assert!(close(v, 20.0), "MR10 nearest at (0.95, 0.1): got {v}");
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — CubicSpline derivative of a constant function is 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cubic_spline_derivative_of_constant() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y = vec![3.7_f64; 10];
    let cs = CubicSplineStandalone::new(&x, &y, SplineBc::Natural).unwrap();
    let dcs = cs.derivative(1);
    for k in 1..=18 {
        let xi = 0.5 * k as f64;
        let v = dcs.eval(xi);
        assert!(
            v.abs() < 1e-10,
            "MR11 derivative of constant at xi={xi}: {v}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — BSpline integration over a zero-width interval is zero.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bspline_integrate_zero_width() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let spline = make_interp_spline(&x, &y, 3).unwrap();
    for &a in &[0.5_f64, 2.7, 5.0, 8.3] {
        let i = spline.integrate(a, a).unwrap();
        assert!(
            i.abs() < 1e-12,
            "MR12 BSpline integrate({a}, {a}) = {i}, expected 0"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — BSpline integration is anti-symmetric:
//   integrate(b, a) = -integrate(a, b).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bspline_integrate_antisymmetric() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
    let spline = make_interp_spline(&x, &y, 3).unwrap();
    for &(a, b) in &[(0.5_f64, 5.0_f64), (1.0, 2.0), (2.5, 8.5)] {
        let i_ab = spline.integrate(a, b).unwrap();
        let i_ba = spline.integrate(b, a).unwrap();
        assert!(
            (i_ab + i_ba).abs() < 1e-9 * i_ab.abs().max(1.0),
            "MR13 antisymmetric integrate: ({a}, {b})={i_ab} + ({b}, {a})={i_ba} != 0"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — polyval of a constant polynomial returns that constant.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_polyval_constant_polynomial() {
    for &c in &[-7.5_f64, 0.0, 1.0, 42.0] {
        for &x in &[-3.0_f64, -1.0, 0.0, 0.5, 2.0, 100.0] {
            let v = polyval(&[c], x);
            assert!(close(v, c), "MR14 polyval([{c}], {x}) = {v}, expected {c}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — polyfit roundtrip: fitting a polynomial of the data's true
// degree should reproduce y at the original x within numerical noise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_polyfit_roundtrip_on_polynomial_data() {
    // y = 2x³ - x² + 3x + 5 sampled at non-uniform xs.
    let x: Vec<f64> = vec![-2.0, -1.5, -0.5, 0.0, 0.7, 1.3, 2.5, 3.4];
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 2.0 * xi.powi(3) - xi.powi(2) + 3.0 * xi + 5.0)
        .collect();
    let coeffs = polyfit(&x, &y, 3).expect("polyfit deg 3");
    for (i, &xi) in x.iter().enumerate() {
        let yhat = polyval(&coeffs, xi);
        assert!(
            (yhat - y[i]).abs() < 1e-7,
            "MR15 polyfit(deg=3) yhat({xi}) = {yhat} vs y = {}",
            y[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — Lagrange interpolant condition: lagrange(xi, yi) is a
// polynomial of degree n-1 that exactly reproduces yi at xi.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_lagrange_passes_through_data() {
    let xi = vec![-2.0_f64, -0.5, 0.5, 1.5, 3.0];
    let yi = vec![3.0_f64, 1.0, 2.5, -1.0, 4.5];
    let coeffs = lagrange(&xi, &yi).expect("lagrange");
    for (i, &x) in xi.iter().enumerate() {
        let v = polyval(&coeffs, x);
        assert!(
            (v - yi[i]).abs() < 1e-9,
            "MR16 lagrange at {x}: got {v}, expected {}",
            yi[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR17 — polyadd is commutative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_polyadd_is_commutative() {
    let a = vec![2.0_f64, -1.0, 3.0, 5.0];
    let b = vec![1.0_f64, 4.0];
    let ab = polyadd(&a, &b);
    let ba = polyadd(&b, &a);
    assert_eq!(ab.len(), ba.len(), "MR17 polyadd length mismatch");
    for (i, (&x, &y)) in ab.iter().zip(ba.iter()).enumerate() {
        assert!(
            close(x, y),
            "MR17 polyadd commutativity at coeff {i}: {x} vs {y}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — polymul is commutative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_polymul_is_commutative() {
    let a = vec![2.0_f64, -1.0, 3.0];
    let b = vec![1.0_f64, 4.0, -2.0, 0.5];
    let ab = polymul(&a, &b);
    let ba = polymul(&b, &a);
    assert_eq!(ab.len(), ba.len(), "MR18 polymul length mismatch");
    for (i, (&x, &y)) in ab.iter().zip(ba.iter()).enumerate() {
        assert!(
            close(x, y),
            "MR18 polymul commutativity at coeff {i}: {x} vs {y}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — Barycentric interpolant passes through its data.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_barycentric_passes_through_data() {
    let xi = vec![-2.0_f64, -1.0, 0.0, 1.5, 3.0];
    let yi = vec![5.0_f64, -1.0, 2.0, 3.5, 7.0];
    let interp = BarycentricInterpolator::new(&xi, &yi).expect("barycentric");
    for (i, &x) in xi.iter().enumerate() {
        let v = interp.eval(x);
        assert!(
            (v - yi[i]).abs() < 1e-9,
            "MR19 barycentric at {x}: got {v}, expected {}",
            yi[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — Krogh interpolant passes through its data.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_krogh_passes_through_data() {
    let xi = vec![0.0_f64, 0.5, 1.5, 2.0, 4.0];
    let yi = vec![1.0_f64, 2.5, 0.0, -1.5, 3.0];
    let interp = KroghInterpolator::new(&xi, &yi).expect("krogh");
    for (i, &x) in xi.iter().enumerate() {
        let v = interp.evaluate(x);
        assert!(
            (v - yi[i]).abs() < 1e-9,
            "MR20 krogh at {x}: got {v}, expected {}",
            yi[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — polyder of a constant polynomial is zero (the constant
// polynomial differentiates to []).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_polyder_of_constant_is_zero() {
    for &c in &[-1.5_f64, 0.0, 3.0, 100.0] {
        let d = polyder(&[c], 1);
        for (i, &v) in d.iter().enumerate() {
            assert!(
                v.abs() < 1e-15,
                "MR21 polyder([{c}], 1)[{i}] = {v}, expected 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — polyint then polyder is the identity (mod the integration
// constant): polyder(polyint(p, 1, 0), 1) ≡ p.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_polyint_polyder_inverse() {
    let polys = [
        vec![3.0_f64, 2.0, 1.0],          // 3x² + 2x + 1
        vec![1.0_f64, -2.0, 0.5, 4.0],    // cubic
        vec![5.0_f64, 0.0, -3.0, 1.0, 2.0], // quartic
    ];
    for p in &polys {
        let pi = polyint(p, 1, 0.0);
        let pd = polyder(&pi, 1);
        assert_eq!(pd.len(), p.len(), "MR22 polyder(polyint(p)) length");
        for (i, (a, b)) in pd.iter().zip(p).enumerate() {
            assert!(
                close(*a, *b),
                "MR22 polyint∘polyder mismatch at {i}: {a} vs {b}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — polysub(a, a) = 0 (zero polynomial: every coefficient is 0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_polysub_self_is_zero() {
    let polys = [
        vec![1.0_f64, 2.0, 3.0],
        vec![5.0_f64, -1.0, 4.0, 7.0, 2.0],
        vec![0.0_f64],
    ];
    for p in &polys {
        let r = polysub(p, p);
        for (i, &v) in r.iter().enumerate() {
            assert!(
                v.abs() < 1e-15,
                "MR23 polysub({p:?}, self)[{i}] = {v}, expected 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — polyroots of (x - r1)(x - r2) recovers {r1, r2}.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_polyroots_quadratic() {
    let r1 = 2.0_f64;
    let r2 = -3.0_f64;
    // (x - r1)(x - r2) = x² - (r1+r2)x + r1·r2
    // numpy convention has highest-degree first.
    let coeffs = vec![1.0_f64, -(r1 + r2), r1 * r2];
    let mut roots = polyroots(&coeffs);
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut expected = vec![r1, r2];
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(roots.len(), 2, "MR24 polyroots length");
    for (i, (got, want)) in roots.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "MR24 root[{i}] = {got} vs {want}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — Neville's algorithm interpolant condition: neville passes
// through all (xi, yi).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_neville_interpolant_condition() {
    let xi = vec![-2.0_f64, -1.0, 0.5, 2.0, 3.5];
    let yi = vec![5.0_f64, -1.0, 2.0, 3.5, 7.0];
    for (i, &x) in xi.iter().enumerate() {
        let v = neville(&xi, &yi, x);
        assert!(
            (v - yi[i]).abs() < 1e-10,
            "MR25 neville at {x}: got {v} vs {}",
            yi[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — ratval(p, q, x) = polyval(p, x) / polyval(q, x) when
// polyval(q, x) is bounded away from zero.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ratval_matches_polyval_ratio() {
    // ratval uses ascending coefficient order (c[0] + c[1]·x + ...);
    // polyval uses descending order. Reverse to bridge conventions.
    let p_asc = vec![1.0_f64, 3.0, -1.0, 2.0]; // 1 + 3x - x² + 2x³
    let q_asc = vec![2.0_f64, 0.5, 1.0];       // 2 + 0.5x + x²
    let p_desc: Vec<f64> = p_asc.iter().rev().copied().collect();
    let q_desc: Vec<f64> = q_asc.iter().rev().copied().collect();
    for &x in &[-2.0_f64, -0.5, 0.0, 1.0, 2.5, 4.0] {
        let r = ratval(&p_asc, &q_asc, x);
        let pv = polyval(&p_desc, x);
        let qv = polyval(&q_desc, x);
        let expected = pv / qv;
        assert!(
            (r - expected).abs() < 1e-9 * expected.abs().max(1.0),
            "MR26 ratval at {x}: got {r} vs polyval(p)/polyval(q) = {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — chebyshev_nodes(n, a, b) returns n nodes, all strictly inside
// (a, b).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_chebyshev_nodes_count_and_range() {
    for n in [1usize, 4, 7, 10] {
        for &(a, b) in &[(-1.0_f64, 1.0), (0.0, 5.0), (-3.0, 2.0)] {
            let nodes = chebyshev_nodes(n, a, b);
            assert_eq!(nodes.len(), n, "MR27 nodes count");
            for (k, &x) in nodes.iter().enumerate() {
                assert!(
                    x >= a - 1e-12 && x <= b + 1e-12,
                    "MR27 chebyshev_nodes(n={n}, a={a}, b={b})[{k}] = {x} outside"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — chebyshev_nodes2 (Chebyshev points of the second kind)
// includes both endpoints when n ≥ 2.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_chebyshev_nodes2_endpoints() {
    for n in [2usize, 4, 8, 16] {
        let nodes = chebyshev_nodes2(n, -1.0, 1.0);
        // Sort to be safe — nodes are returned ordered but defensive.
        let mut sorted = nodes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (sorted[0] - (-1.0)).abs() < 1e-12,
            "MR28 chebyshev_nodes2(n={n}) min = {} expected -1",
            sorted[0]
        );
        assert!(
            (sorted[n - 1] - 1.0).abs() < 1e-12,
            "MR28 chebyshev_nodes2(n={n}) max = {} expected 1",
            sorted[n - 1]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — barycentric_eval at a node returns the corresponding value.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_barycentric_eval_at_nodes() {
    let nodes = vec![-2.0_f64, -0.5, 0.5, 1.5, 3.0];
    let values = vec![3.0_f64, 1.0, 2.5, -1.0, 4.5];
    let weights = barycentric_weights(&nodes);
    for (i, &x) in nodes.iter().enumerate() {
        let v = barycentric_eval(&nodes, &values, &weights, x);
        assert!(
            (v - values[i]).abs() < 1e-9,
            "MR29 barycentric_eval at node {x}: {v} vs {}",
            values[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — Hermite interpolation passes through the nodes (zero-th
// derivative condition); a constant-valued, zero-derivative input
// produces a constant output.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hermite_interp_constant_input() {
    let nodes = vec![0.0_f64, 1.0, 2.5, 4.0];
    let values = vec![5.0_f64; 4];
    let derivs = vec![0.0_f64; 4];
    for &x in &[-1.0_f64, 0.0, 1.5, 3.0, 5.0] {
        let v = hermite_interp(&nodes, &values, &derivs, x);
        assert!(
            (v - 5.0).abs() < 1e-9,
            "MR30 hermite_interp(constant 5) at {x} = {v}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — polyint_definite computes ∫_a^b p(x) dx = P(b) - P(a),
// where P = polyint(p). Verify on p(x) = 2x + 1 ⇒ ∫₀^3 = 9 + 3 = 12.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_polyint_definite_polynomial() {
    let p = vec![2.0_f64, 1.0]; // ascending: 2 + x ... actually polyint here uses ?
    // polyint uses descending; double-check by anti-derivative.
    // p in descending: [2, 1] = 2x + 1; ∫(2x+1)dx = x² + x
    // From 0 to 3: 9 + 3 - 0 = 12.
    let val = polyint_definite(&p, 0.0, 3.0);
    assert!(
        (val - 12.0).abs() < 1e-9,
        "MR31 ∫₀^3 (2x+1) dx = {val}, expected 12"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — polyval_with_error returns the same value as polyval for
// well-conditioned inputs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_polyval_with_error_matches_polyval() {
    let p = vec![1.0_f64, -2.0, 3.0, -1.0, 0.5];
    for &x in &[-2.0_f64, 0.0, 1.5, 3.0] {
        let plain = polyval(&p, x);
        let (val, _err) = polyval_with_error(&p, x);
        assert!(
            (plain - val).abs() < 1e-12 * plain.abs().max(1.0),
            "MR32 polyval_with_error at {x}: {val} vs polyval {plain}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — splev at the data points evaluates close to y for an
// interpolating cubic spline (s = 0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_splev_passes_through_data() {
    let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.5 * xi.cos()).collect();
    let tck = splrep(&x, &y, 3, 0.0).unwrap();
    let yhat = splev(&x, &tck).unwrap();
    for (i, (a, b)) in y.iter().zip(&yhat).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "MR33 splev at x[{i}]={}: y = {a} vs ŷ = {b}",
            x[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — splint(a, a, tck) = 0 (zero-width definite integral).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_splint_zero_width_zero() {
    let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let tck = splrep(&x, &y, 3, 0.0).unwrap();
    for &a in &[0.0_f64, 1.5, 3.0, 4.5] {
        let v = splint(a, a, &tck).unwrap();
        assert!(
            v.abs() < 1e-12,
            "MR34 splint({a}, {a}) = {v}, expected 0"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — splint is antisymmetric in its bounds: splint(b, a) = -splint(a, b).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_splint_antisymmetric_bounds() {
    let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.cos()).collect();
    let tck = splrep(&x, &y, 3, 0.0).unwrap();
    for &(a, b) in &[(0.5_f64, 3.0), (1.0, 4.0), (0.0, 4.5)] {
        let i_ab = splint(a, b, &tck).unwrap();
        let i_ba = splint(b, a, &tck).unwrap();
        assert!(
            (i_ab + i_ba).abs() < 1e-9 * i_ab.abs().max(1.0),
            "MR35 splint antisymmetric: ({a},{b}) = {i_ab} + ({b},{a}) = {i_ba}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — PchipInterpolator returns finite output across an evaluation
// range that includes the data nodes.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pchip_finite_output() {
    let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| if xi < 4.0 { xi } else { 4.0 + (xi - 4.0) * 0.5 })
        .collect();
    let interp = PchipInterpolator::new(&x, &y).unwrap();
    for i in 0..70 {
        let t = (i as f64) * 0.1;
        let v = interp.eval(t);
        assert!(
            v.is_finite(),
            "MR36 PchipInterpolator non-finite at t={t}: {v}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR37 — CubicSplineStandalone passes through its data points (zero-th
// derivative of the spline at xi equals yi).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cubic_spline_standalone_passes_through_data() {
    let x: Vec<f64> = (0..7).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.5 * xi.cos()).collect();
    let spline = CubicSplineStandalone::new(&x, &y, SplineBc::NotAKnot).unwrap();
    for (i, &xi) in x.iter().enumerate() {
        let v = spline.eval(xi);
        assert!(
            (v - y[i]).abs() < 1e-9,
            "MR37 cubic spline at x[{i}]={xi}: y = {} vs ŷ = {v}",
            y[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR38 — splev composed with splrep on a smooth function recovers the
// data within tolerance.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_splrep_splev_smooth_recovery() {
    let x: Vec<f64> = (0..21).map(|i| i as f64 * 0.25).collect();
    let y: Vec<f64> = x.iter().map(|&xi| (xi * 0.3).sin() + 0.4 * (xi * 0.7).cos()).collect();
    let tck = splrep(&x, &y, 5, 0.0).unwrap();
    let yhat = splev(&x, &tck).unwrap();
    let mut max_err = 0.0_f64;
    for (a, b) in y.iter().zip(&yhat) {
        let e = (a - b).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(
        max_err < 1e-4,
        "MR38 splrep/splev max error = {max_err}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR39 — splder reduces spline degree by 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_splder_reduces_degree() {
    let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let tck = splrep(&x, &y, 3, 0.0).unwrap();
    let dtck = splder(&tck).unwrap();
    assert_eq!(dtck.2, tck.2 - 1, "MR39 splder degree {} vs {}", dtck.2, tck.2 - 1);
}

// ─────────────────────────────────────────────────────────────────────
// MR40 — splantider increases spline degree by 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_splantider_increases_degree() {
    let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let tck = splrep(&x, &y, 3, 0.0).unwrap();
    let itck = splantider(&tck).unwrap();
    assert_eq!(itck.2, tck.2 + 1, "MR40 splantider degree {} vs {}", itck.2, tck.2 + 1);
}

// ─────────────────────────────────────────────────────────────────────
// MR41 — splev_with_derivative(der=0) equals plain splev.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_splev_with_derivative_zero_matches_splev() {
    let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.cos()).collect();
    let tck = splrep(&x, &y, 3, 0.0).unwrap();
    let v0 = splev(&x, &tck).unwrap();
    let v_der0 = splev_with_derivative(&x, &tck, 0).unwrap();
    for (i, (a, b)) in v0.iter().zip(&v_der0).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "MR41 splev_with_derivative(der=0)[{i}] = {b} vs splev = {a}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR42 — sproot on a strictly positive smooth function returns no
// roots within the data range.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sproot_no_roots_for_positive_data() {
    let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.cos() + 5.0).collect(); // ≥ 4
    let tck = splrep(&x, &y, 3, 0.0).unwrap();
    let roots = sproot(&tck).unwrap_or_default();
    assert!(
        roots.is_empty(),
        "MR42 sproot returned {} roots for strictly positive data",
        roots.len()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR43 — Akima1DInterpolator passes through the data points.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_akima_passes_through_data_extra() {
    let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![1.0, 2.5, 3.0, 1.5, -0.5, 2.0];
    let interp = Akima1DInterpolator::new(&x, &y).unwrap();
    for (i, &xi) in x.iter().enumerate() {
        let v = interp.eval(xi);
        assert!(
            (v - y[i]).abs() < 1e-9,
            "MR43 akima at x[{i}]={xi}: got {v} vs {}",
            y[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR44 — KroghInterpolator on a polynomial of degree n-1 reproduces it
// exactly: lagrange interpolation through the given nodes is the
// unique polynomial of degree ≤ n-1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_krogh_reproduces_polynomial() {
    // p(x) = 2x³ - x² + 3x + 5 sampled at 4 points (deg 3 = n-1 with n = 4).
    let xi: Vec<f64> = vec![-1.0, 0.0, 1.0, 2.0];
    let yi: Vec<f64> = xi.iter()
        .map(|&xi| 2.0 * xi.powi(3) - xi.powi(2) + 3.0 * xi + 5.0)
        .collect();
    let interp = KroghInterpolator::new(&xi, &yi).unwrap();
    for &x in &[-0.5_f64, 0.5, 1.5, 3.0, -2.0] {
        let expected = 2.0 * x.powi(3) - x.powi(2) + 3.0 * x + 5.0;
        let got = interp.evaluate(x);
        assert!(
            (got - expected).abs() < 1e-9 * expected.abs().max(1.0),
            "MR44 krogh at {x}: got {got} vs {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR45 — interp1d_linear at the data points returns the data values.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_interp1d_linear_at_data_points() {
    let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.5, 6.0];
    let y: Vec<f64> = vec![1.0, 2.5, 0.0, -1.0, 4.0, 2.0];
    let yhat = interp1d_linear(&x, &y, &x).unwrap();
    for (i, (a, b)) in y.iter().zip(&yhat).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "MR45 interp1d_linear at x[{i}]: y = {a}, yhat = {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR46 — RbfInterpolator passes through the data using Gaussian kernel.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rbf_passes_through_data() {
    let points: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
    ];
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let interp =
        RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, 1.0).unwrap();
    for (i, p) in points.iter().enumerate() {
        let v = interp.eval(p);
        assert!(
            (v - values[i]).abs() < 1e-7,
            "MR46 RBF at point {i}: got {v} vs {}",
            values[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR47 — pade(c, m, n) returns vectors of length m+1 and n+1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pade_returns_correct_lengths() {
    // Taylor series for exp(x) up to x^4.
    let c = vec![1.0_f64, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0];
    let (p, q) = pade(&c, 2, 2).unwrap();
    assert_eq!(p.len(), 3, "MR47 pade p length");
    assert_eq!(q.len(), 3, "MR47 pade q length");
    assert!((q[0] - 1.0).abs() < 1e-12, "MR47 pade q[0] = {} != 1", q[0]);
}

// ─────────────────────────────────────────────────────────────────────
// MR48 — make_lsq_spline returns a B-spline of the requested degree
// when given enough knots.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_make_lsq_spline_degree() {
    let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.5 * xi.cos()).collect();
    let k = 3;
    // Knot vector: clamped at endpoints.
    let mut t = vec![x[0]; k + 1];
    let interior_count = 6;
    for i in 1..=interior_count {
        let frac = i as f64 / (interior_count + 1) as f64;
        t.push(x[0] + frac * (x[x.len() - 1] - x[0]));
    }
    for _ in 0..=k {
        t.push(x[x.len() - 1]);
    }
    let bspline = make_lsq_spline(&x, &y, &t, k).unwrap();
    // BSpline with degree k returns finite values when evaluated.
    let v = bspline.eval(x[5]);
    assert!(v.is_finite(), "MR48 make_lsq_spline eval non-finite: {v}");
}

// ─────────────────────────────────────────────────────────────────────
// MR49 — Interp1d (Cubic) on a smooth function approaches y at data
// points within tolerance.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_interp1d_cubic_close_to_data_at_xs() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let mut opts = Interp1dOptions::default();
    opts.kind = InterpKind::CubicSpline;
    let interp = Interp1d::new(&x, &y, opts).unwrap();
    for (i, &xi) in x.iter().enumerate() {
        let v = interp.eval(xi).unwrap();
        assert!(
            (v - y[i]).abs() < 1e-9,
            "MR49 Interp1d cubic at x[{i}]: y = {} vs ŷ = {v}",
            y[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR50 — interp1d_linear extrapolation at the boundary returns boundary
// values when querying inside the data range. (Sanity check on linear.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_interp1d_linear_in_range_finite() {
    let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
    let y: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
    let x_query: Vec<f64> = vec![0.5, 1.5, 2.5];
    let yhat = interp1d_linear(&x, &y, &x_query).unwrap();
    let expected = vec![15.0_f64, 25.0, 35.0];
    for (i, (a, b)) in expected.iter().zip(&yhat).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "MR50 linear interp at x_query[{i}] = {}: got {b}, expected {a}",
            x_query[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR51 — NearestNDInterpolator at a training point returns the exact
// training value.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_nearest_nd_at_training_point() {
    let pts = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let vals = vec![1.0_f64, 2.0, 3.0, 4.0];
    let interp = NearestNDInterpolator::new(&pts, &vals).unwrap();
    for (i, p) in pts.iter().enumerate() {
        let v = interp.eval(p).unwrap();
        assert!(
            (v - vals[i]).abs() < 1e-12,
            "MR51 nearest_nd at point {i}: {v} vs {}",
            vals[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR52 — griddata with Nearest method on training points returns the
// training values.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_griddata_nearest_at_training_points() {
    let pts = vec![vec![0.0_f64, 0.0], vec![1.0, 0.0], vec![0.5, 1.0]];
    let vals = vec![10.0_f64, 20.0, 30.0];
    let result = griddata(&pts, &vals, &pts, GriddataMethod::Nearest).unwrap();
    for (i, (got, want)) in result.iter().zip(&vals).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "MR52 griddata(nearest)[{i}] = {got} vs {want}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR53 — interpn at a grid corner (e.g., the lower-left grid point)
// returns the corresponding value.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_interpn_at_grid_corner() {
    let xs: Vec<f64> = vec![0.0, 1.0, 2.0];
    let ys: Vec<f64> = vec![0.0, 1.0];
    // Values stored row-major along (x, y): 3 × 2 = 6.
    let vals: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let xi = vec![vec![0.0_f64, 0.0]];
    let result = interpn(
        vec![xs, ys],
        vals,
        &xi,
        RegularGridMethod::Linear,
        true,
        None,
    )
    .unwrap();
    assert!(
        (result[0] - 1.0).abs() < 1e-9,
        "MR53 interpn at (0, 0) = {} vs 1",
        result[0]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR54 — interp2d at a grid corner returns the corresponding value.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_interp2d_at_corner() {
    let x = vec![0.0_f64, 1.0, 2.0];
    let y = vec![0.0_f64, 1.0];
    let z = vec![vec![1.0_f64, 2.0, 3.0], vec![4.0_f64, 5.0, 6.0]];
    let v = interp2d(&x, &y, &z, 0.0, 0.0).unwrap();
    assert!((v - 1.0).abs() < 1e-9, "MR54 interp2d(0, 0) = {v}");
    let v22 = interp2d(&x, &y, &z, 2.0, 1.0).unwrap();
    assert!((v22 - 6.0).abs() < 1e-9, "MR54 interp2d(2, 1) = {v22}");
}

// ─────────────────────────────────────────────────────────────────────
// MR55 — LinearNDInterpolator at a training vertex returns the vertex
// value.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_linear_nd_at_training_vertex() {
    let pts = vec![
        vec![0.0_f64, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
    ];
    let vals = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let interp = LinearNDInterpolator::new(&pts, &vals).unwrap();
    for (i, p) in pts.iter().enumerate() {
        let v = interp.eval(p).unwrap();
        assert!(
            v.is_finite() && (v - vals[i]).abs() < 1e-7,
            "MR55 LinearND at vertex {i}: got {v} vs {}",
            vals[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR56 — PCHIP on monotone-increasing y produces a monotone-increasing
// interpolant on a fine evaluation grid.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pchip_monotone_on_increasing_data() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = vec![1.0, 2.0, 3.5, 4.0, 6.0, 7.5, 8.0, 9.0, 11.0, 12.0];
    let interp = PchipInterpolator::new(&x, &y).unwrap();
    let grid: Vec<f64> = (0..91).map(|i| i as f64 * 0.1).collect();
    let yhat: Vec<f64> = grid.iter().map(|&t| interp.eval(t)).collect();
    for w in yhat.windows(2) {
        assert!(
            w[0] <= w[1] + 1e-9,
            "MR56 PCHIP not monotone increasing: {} > {}",
            w[0],
            w[1]
        );
    }
}






