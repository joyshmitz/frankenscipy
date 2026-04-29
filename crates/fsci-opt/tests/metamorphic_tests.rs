//! Metamorphic tests for `fsci-opt`.
//!
//! Minimum location on convex test functions, multi-method root
//! agreement, root-finder shift invariance, and curve_fit recovery
//! of synthetic linear parameters.
//!
//! Run with: `cargo test -p fsci-opt --test metamorphic_tests`

use fsci_opt::{
    CurveFitOptions, MinimizeOptions, RootOptions, bisect, brenth, brentq, curve_fit, minimize,
    ridder, toms748,
};

const ATOL: f64 = 1e-6;
const RTOL: f64 = 1e-5;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — minimize finds a known minimum on a quadratic bowl.
// f(x) = (x[0] − 2)² + (x[1] + 3)² has its unique global minimum at
// (2, −3) with value 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_finds_known_minimum_on_bowl() {
    let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] + 3.0).powi(2);
    let opts = MinimizeOptions::default();
    let res = minimize(f, &[0.0, 0.0], opts).unwrap();
    assert!(res.success, "MR1 minimize did not converge: {}", res.message);
    assert!(
        (res.x[0] - 2.0).abs() < 1e-3,
        "MR1 x[0]={} expected 2.0",
        res.x[0]
    );
    assert!(
        (res.x[1] - (-3.0)).abs() < 1e-3,
        "MR1 x[1]={} expected -3.0",
        res.x[1]
    );
    let fmin = res.fun.unwrap();
    assert!(fmin < 1e-6, "MR1 fmin={fmin} expected ~0");
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — All bracketing root finders return the same root for a
// monotone f on a bracket containing the unique zero.
//
// f(x) = exp(x) − 2 has a unique real root x* = ln(2) ≈ 0.6931.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_root_finders_agree_on_monotone_function() {
    let f = |x: f64| x.exp() - 2.0;
    let bracket = (0.0_f64, 1.0_f64);
    let opts = RootOptions::default();

    let r_brentq = brentq(f, bracket, opts).unwrap().root;
    let r_brenth = brenth(f, bracket, opts).unwrap().root;
    let r_bisect = bisect(f, bracket, opts).unwrap().root;
    let r_ridder = ridder(f, bracket, opts).unwrap().root;
    let r_toms = toms748(f, bracket, opts).unwrap().root;

    let truth = 2.0_f64.ln();
    for (name, r) in [
        ("brentq", r_brentq),
        ("brenth", r_brenth),
        ("bisect", r_bisect),
        ("ridder", r_ridder),
        ("toms748", r_toms),
    ] {
        assert!(
            (r - truth).abs() < 1e-8,
            "MR2 {name} root {r} differs from ln(2)={truth}"
        );
        assert!(
            f(r).abs() < 1e-8,
            "MR2 {name} f(root) = {} not near 0",
            f(r)
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — Shift invariance: root of g(x) = f(x − c) is c + root_of_f.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_root_shift_invariance() {
    let f = |x: f64| x.exp() - 2.0; // root at ln(2)
    let opts = RootOptions::default();
    let r0 = brentq(f, (0.0, 1.0), opts).unwrap().root;
    for shift in [-3.0_f64, -0.5, 0.0, 0.7, 4.5] {
        let g = |x: f64| f(x - shift);
        let bracket = (shift + 0.0, shift + 1.0);
        let r = brentq(g, bracket, opts).unwrap().root;
        assert!(
            close(r, shift + r0),
            "MR3 shift={shift}: got {r}, expected {}",
            shift + r0
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — minimize is shift-invariant: shifting f(x) by an additive
// constant leaves x* unchanged.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_constant_shift_invariance() {
    let f = |x: &[f64]| (x[0] - 1.5).powi(2) + (x[1] - 2.5).powi(2);
    let g = |x: &[f64]| f(x) + 1000.0;
    let opts = MinimizeOptions::default();
    let r1 = minimize(f, &[0.0, 0.0], opts).unwrap();
    let r2 = minimize(g, &[0.0, 0.0], opts).unwrap();
    for (a, b) in r1.x.iter().zip(&r2.x) {
        assert!(
            (a - b).abs() < 1e-3,
            "MR4 minimize shift: x_f={a} x_g={b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — curve_fit recovers the slope and intercept of synthetic
// noise-free linear data y = α x + β.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_curve_fit_recovers_linear_params() {
    let alpha_true = 2.5_f64;
    let beta_true = -1.2_f64;
    let xdata: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let ydata: Vec<f64> = xdata.iter().map(|&x| alpha_true * x + beta_true).collect();
    let model = |x: f64, p: &[f64]| p[0] * x + p[1];
    let opts = CurveFitOptions {
        p0: Some(vec![0.0, 0.0]),
        ..Default::default()
    };
    let res = curve_fit(model, &xdata, &ydata, opts).unwrap();
    assert!(
        (res.popt[0] - alpha_true).abs() < 1e-6,
        "MR5 alpha: {} vs {alpha_true}",
        res.popt[0]
    );
    assert!(
        (res.popt[1] - beta_true).abs() < 1e-6,
        "MR5 beta: {} vs {beta_true}",
        res.popt[1]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — Bisection counts: bisect's iteration count is upper-bounded by
// ⌈log₂((b − a) / xtol)⌉ + 1 because bisection halves the bracket
// width every iteration.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bisect_iteration_count_bounded() {
    let f = |x: f64| x.exp() - 2.0;
    let opts = RootOptions {
        xtol: 1e-10,
        ..Default::default()
    };
    let res = bisect(f, (0.0, 1.0), opts).unwrap();
    let max_iters = ((1.0_f64 / 1e-10).log2().ceil() as usize) + 5;
    assert!(
        res.iterations <= max_iters,
        "MR6 bisect iterations {} > expected upper bound {max_iters}",
        res.iterations
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — minimize on a strictly convex function lands on a point with
// objective ≤ initial objective (descent property).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_strict_descent_on_convex() {
    let f = |x: &[f64]| x[0].powi(4) + (x[1] - 2.0).powi(2) + 1.0;
    let x0 = vec![3.0, -1.0];
    let f0 = f(&x0);
    let opts = MinimizeOptions::default();
    let res = minimize(f, &x0, opts).unwrap();
    let fmin = res.fun.unwrap();
    assert!(
        fmin <= f0 + 1e-9,
        "MR7 minimize did not descend: f0={f0} fmin={fmin}"
    );
}
