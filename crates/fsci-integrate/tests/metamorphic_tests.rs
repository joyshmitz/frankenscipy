//! Metamorphic tests for `fsci-integrate` quadrature.
//!
//! Metamorphic relations (MRs) check input-output relationships rather than
//! point values, so they don't require an oracle. Each MR transforms the
//! input in a known way and asserts the corresponding transformation of the
//! output.
//!
//! References:
//!   Chen et al., "Metamorphic Testing: A Review of Challenges and
//!   Opportunities", ACM Comput. Surv. 51(1) (2018).
//!
//! Run with: `cargo test -p fsci-integrate --test metamorphic_tests`

use fsci_integrate::{QuadOptions, cumulative_trapezoid, quad, simpson, trapezoid};

/// Tolerance for metamorphic relation comparisons.
///
/// Adaptive GK15 targets `epsrel ~= 1.49e-8` by default; we allow a small
/// additional slack for round-off accumulated when summing two adaptive
/// estimates.
const MR_RTOL: f64 = 1e-7;
const MR_ATOL: f64 = 1e-9;

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= MR_ATOL + MR_RTOL * expected.abs().max(1.0)
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        close(actual, expected),
        "{label}: actual={actual:.16e} expected={expected:.16e} diff={:.3e}",
        (actual - expected).abs()
    );
}

// Reference test integrands chosen to span polynomial, transcendental, and
// oscillatory behavior.
fn poly_cubic(x: f64) -> f64 {
    1.0 + x + 0.5 * x * x - 0.25 * x.powi(3)
}

fn smooth_exp(x: f64) -> f64 {
    (-0.3 * x).exp() * (1.0 + 0.1 * x)
}

fn oscillatory(x: f64) -> f64 {
    (3.0 * x).sin() + 0.5 * (5.0 * x).cos()
}

fn gaussian(x: f64) -> f64 {
    (-x * x).exp()
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — additivity: ∫_a^b f = ∫_a^c f + ∫_c^b f
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_additivity_polynomial() {
    let opts = QuadOptions::default();
    let whole = quad(poly_cubic, -1.0, 2.5, opts).unwrap();
    let left = quad(poly_cubic, -1.0, 0.7, opts).unwrap();
    let right = quad(poly_cubic, 0.7, 2.5, opts).unwrap();
    assert_close(left.integral + right.integral, whole.integral, "MR1 poly");
}

#[test]
fn mr_quad_additivity_smooth_exp() {
    let opts = QuadOptions::default();
    let whole = quad(smooth_exp, 0.0, 4.0, opts).unwrap();
    for c in [0.5, 1.0, 2.3, 3.7] {
        let left = quad(smooth_exp, 0.0, c, opts).unwrap();
        let right = quad(smooth_exp, c, 4.0, opts).unwrap();
        assert_close(
            left.integral + right.integral,
            whole.integral,
            &format!("MR1 smooth_exp split @ {c}"),
        );
    }
}

#[test]
fn mr_quad_additivity_oscillatory() {
    let opts = QuadOptions::default();
    let whole = quad(oscillatory, -2.0, 3.0, opts).unwrap();
    let left = quad(oscillatory, -2.0, 0.0, opts).unwrap();
    let right = quad(oscillatory, 0.0, 3.0, opts).unwrap();
    assert_close(left.integral + right.integral, whole.integral, "MR1 osc");
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — sign-flip / orientation reversal: ∫_a^b f = −∫_b^a f
//
// `quad` requires a < b in our implementation; the sign-flip identity is
// instead exercised against `trapezoid` and `simpson`, which accept any
// monotone-x ordering.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_trapezoid_sign_flip() {
    let xs: Vec<f64> = (0..21).map(|i| -1.0 + 0.1 * i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| poly_cubic(x)).collect();
    let xs_rev: Vec<f64> = xs.iter().rev().copied().collect();
    let ys_rev: Vec<f64> = ys.iter().rev().copied().collect();
    let forward = trapezoid(&ys, &xs).unwrap();
    let backward = trapezoid(&ys_rev, &xs_rev).unwrap();
    assert_close(
        forward.integral,
        -backward.integral,
        "MR2 trapezoid sign-flip",
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — linearity: ∫(α f + β g) = α ∫f + β ∫g
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_linearity_combination() {
    let alpha = 2.5_f64;
    let beta = -0.75_f64;
    let combined = move |x: f64| alpha * smooth_exp(x) + beta * oscillatory(x);

    let opts = QuadOptions::default();
    let i_combined = quad(combined, 0.0, 3.0, opts).unwrap().integral;
    let i_f = quad(smooth_exp, 0.0, 3.0, opts).unwrap().integral;
    let i_g = quad(oscillatory, 0.0, 3.0, opts).unwrap().integral;

    assert_close(i_combined, alpha * i_f + beta * i_g, "MR3 linearity");
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — translation invariance: ∫_a^b f(x) dx = ∫_{a+c}^{b+c} f(x − c) dx
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_translation_invariance() {
    let opts = QuadOptions::default();
    let base = quad(gaussian, -2.0, 2.0, opts).unwrap().integral;
    for shift in [-1.5, -0.3, 0.0, 0.7, 2.4] {
        let translated = move |x: f64| gaussian(x - shift);
        let shifted = quad(translated, -2.0 + shift, 2.0 + shift, opts)
            .unwrap()
            .integral;
        assert_close(shifted, base, &format!("MR4 translate Δ={shift}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — scaling: ∫_a^b f(x) dx = (1/k) ∫_{ka}^{kb} f(x/k) dx, k > 0
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_scaling_invariance() {
    let opts = QuadOptions::default();
    let base = quad(gaussian, -2.0, 2.0, opts).unwrap().integral;
    for k in [0.5, 1.0, 1.7, 3.0] {
        let scaled = move |x: f64| gaussian(x / k);
        let i = quad(scaled, -2.0 * k, 2.0 * k, opts).unwrap().integral;
        assert_close(i / k, base, &format!("MR5 scale k={k}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — zero-width: ∫_a^a f = 0
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_zero_width() {
    let opts = QuadOptions::default();
    for a in [-3.0_f64, -0.5, 0.0, 1.7, 4.2] {
        let r = quad(smooth_exp, a, a, opts).unwrap();
        assert_eq!(r.integral, 0.0, "MR6 zero-width @ {a}");
        assert_eq!(r.error, 0.0);
        assert!(r.converged);
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — composite quadrature consistency: cumulative_trapezoid's
// final element equals the unconditional trapezoid integral.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cumulative_trapezoid_final_equals_total() {
    let xs: Vec<f64> = (0..50).map(|i| 0.1 * i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| smooth_exp(x)).collect();
    let cum = cumulative_trapezoid(&ys, &xs).unwrap();
    let total = trapezoid(&ys, &xs).unwrap();
    assert_close(
        *cum.last().unwrap(),
        total.integral,
        "MR7 cumulative-trapezoid finalvalue",
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — Simpson is exact for cubics on uniformly-spaced grids: a
// metamorphic invariant under refinement (doubling N changes nothing).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_simpson_cubic_invariant_under_refinement() {
    fn integrate_simpson(n_intervals: usize) -> f64 {
        // n_intervals must be even for Simpson's 1/3 rule.
        let n = if n_intervals % 2 == 0 {
            n_intervals
        } else {
            n_intervals + 1
        };
        let xs: Vec<f64> = (0..=n).map(|i| -1.0 + 3.5 * i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| poly_cubic(x)).collect();
        simpson(&ys, &xs).unwrap().integral
    }
    let coarse = integrate_simpson(20);
    let fine = integrate_simpson(80);
    let ultra = integrate_simpson(320);
    // Simpson is exact for polynomials of degree ≤ 3, so all refinements
    // should agree to round-off.
    assert!(
        (coarse - fine).abs() < 1e-10,
        "Simpson should be exact for cubic: coarse={coarse} fine={fine}"
    );
    assert!(
        (fine - ultra).abs() < 1e-10,
        "Simpson should be exact for cubic: fine={fine} ultra={ultra}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — composition of MR1+MR3: combined splits and linear combinations
// commute. ∫_a^b (αf+βg) = α(∫_a^c f + ∫_c^b f) + β(∫_a^c g + ∫_c^b g)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_composed_split_and_linearity() {
    let opts = QuadOptions::default();
    let alpha = 1.3_f64;
    let beta = -0.4_f64;
    let combined = move |x: f64| alpha * smooth_exp(x) + beta * gaussian(x);

    let lhs = quad(combined, 0.0, 3.0, opts).unwrap().integral;

    let f_left = quad(smooth_exp, 0.0, 1.4, opts).unwrap().integral;
    let f_right = quad(smooth_exp, 1.4, 3.0, opts).unwrap().integral;
    let g_left = quad(gaussian, 0.0, 1.4, opts).unwrap().integral;
    let g_right = quad(gaussian, 1.4, 3.0, opts).unwrap().integral;
    let rhs = alpha * (f_left + f_right) + beta * (g_left + g_right);

    assert_close(lhs, rhs, "MR9 composed");
}
