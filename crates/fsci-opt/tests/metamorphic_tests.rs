//! Metamorphic tests for `fsci-opt`.
//!
//! Minimum location on convex test functions, multi-method root
//! agreement, root-finder shift invariance, and curve_fit recovery
//! of synthetic linear parameters.
//!
//! Run with: `cargo test -p fsci-opt --test metamorphic_tests`

use fsci_opt::{
    BasinhoppingOptions, CurveFitOptions, DifferentialEvolutionOptions, LeastSquaresOptions,
    MinimizeOptions, MinimizeScalarOptions, RootOptions, basinhopping, bisect, brenth, brentq,
    curve_fit, differential_evolution, fsolve, least_squares, minimize, minimize_scalar, ridder,
    toms748,
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

// ─────────────────────────────────────────────────────────────────────
// MR8 — minimize_scalar finds the analytic minimum of a parabola.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_scalar_finds_parabola_vertex() {
    let f = |x: f64| (x - 3.7).powi(2) + 1.5;
    let res = minimize_scalar(f, (0.0, 10.0), MinimizeScalarOptions::default()).unwrap();
    assert!(res.success, "MR8 minimize_scalar did not converge: {res:?}");
    assert!(
        (res.x - 3.7).abs() < 1e-5,
        "MR8 vertex: got x={}, expected 3.7",
        res.x
    );
    assert!(
        (res.fun - 1.5).abs() < 1e-9,
        "MR8 fun at vertex: got {}, expected 1.5",
        res.fun
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — least_squares recovers the synthetic linear-residual minimizer.
//
// Residuals r_i(p) = y_i - (p[0] · x_i + p[1]) — known global minimum
// at α=2.0, β=-1.0 with cost 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_least_squares_recovers_linear_params() {
    let alpha_true = 2.0_f64;
    let beta_true = -1.0_f64;
    let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| alpha_true * x + beta_true).collect();
    let resid = move |p: &[f64]| -> Vec<f64> {
        xs.iter()
            .zip(&ys)
            .map(|(&x, &y)| y - (p[0] * x + p[1]))
            .collect()
    };
    let res = least_squares(resid, &[0.0, 0.0], LeastSquaresOptions::default()).unwrap();
    assert!(
        (res.x[0] - alpha_true).abs() < 1e-6,
        "MR9 alpha: {} vs {alpha_true}",
        res.x[0]
    );
    assert!(
        (res.x[1] - beta_true).abs() < 1e-6,
        "MR9 beta: {} vs {beta_true}",
        res.x[1]
    );
    assert!(
        res.cost < 1e-12,
        "MR9 cost not zero on noise-free data: {}",
        res.cost
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — fsolve solves a 2x2 linear system: f(x) = A x − b.
// Solution must satisfy A x ≈ b.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fsolve_2x2_linear_system() {
    // A = [[3, 1], [1, 2]],  b = [9, 8] → exact solution x = [2, 3]
    let func = |x: &[f64]| -> Vec<f64> {
        vec![3.0 * x[0] + x[1] - 9.0, x[0] + 2.0 * x[1] - 8.0]
    };
    let res = fsolve(func, &[0.0, 0.0]).unwrap();
    let r = func(&res.x);
    let resid: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        resid < 1e-9,
        "MR10 fsolve residual too large: {resid:e} (x={:?})",
        res.x
    );
    // Sanity-check exact answer for the well-conditioned system.
    assert!((res.x[0] - 2.0).abs() < 1e-6);
    assert!((res.x[1] - 3.0).abs() < 1e-6);
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — differential_evolution finds the global minimum on a sphere
// across rectangular bounds. f(x) = Σ x_i² has minimum 0 at the origin.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_de_finds_sphere_minimum() {
    let f = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    let bounds = vec![(-5.0_f64, 5.0); 4];
    let opts = DifferentialEvolutionOptions {
        seed: Some(42),
        ..DifferentialEvolutionOptions::default()
    };
    let res = differential_evolution(f, &bounds, opts).unwrap();
    let fmin = res.fun.unwrap();
    assert!(
        fmin < 1e-6,
        "MR11 differential_evolution did not converge: fmin={fmin}"
    );
    for &xi in &res.x {
        assert!(
            xi.abs() < 1e-3,
            "MR11 not at origin: {res:?} (component {xi})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — basinhopping starting far from the minimum still finds it.
// f(x) = (x[0] − 1)² + (x[1] + 2)² has min at (1, −2).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_basinhopping_far_start_converges() {
    let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2);
    let opts = BasinhoppingOptions {
        niter: 50,
        seed: Some(7),
        ..BasinhoppingOptions::default()
    };
    let res = basinhopping(f, &[10.0, 10.0], opts).unwrap();
    assert!(
        (res.x[0] - 1.0).abs() < 1e-3,
        "MR12 basinhopping x[0] = {}, expected ~1",
        res.x[0]
    );
    assert!(
        (res.x[1] - (-2.0)).abs() < 1e-3,
        "MR12 basinhopping x[1] = {}, expected ~-2",
        res.x[1]
    );
}
