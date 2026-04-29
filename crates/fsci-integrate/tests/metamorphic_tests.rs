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

use fsci_integrate::{
    DblquadOptions, QuadOptions, SolveIvpOptions, SolverKind, ToleranceValue, cumulative_simpson,
    cumulative_trapezoid, dblquad, quad, romb, simpson, solve_ivp, trapezoid,
};

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

// ─────────────────────────────────────────────────────────────────────
// MR10 — solve_ivp on y' = −y matches the analytic solution y₀ · e⁻ᵗ.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_solve_ivp_exponential_decay() {
    let mut rhs = |_t: f64, y: &[f64]| y.iter().map(|v| -v).collect::<Vec<f64>>();
    let y0 = [1.0_f64];
    let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.5).collect();
    let opts = SolveIvpOptions {
        t_span: (0.0, 5.0),
        y0: &y0,
        method: SolverKind::Rk45,
        t_eval: Some(&t_eval),
        rtol: 1e-9,
        atol: ToleranceValue::Scalar(1e-12),
        ..SolveIvpOptions::default()
    };
    let res = solve_ivp(&mut rhs, &opts).unwrap();
    assert_eq!(res.t.len(), t_eval.len(), "MR10 t length");
    for (i, (&t, row)) in res.t.iter().zip(&res.y).enumerate() {
        let expected = (-t).exp();
        let got = row[0];
        assert!(
            (got - expected).abs() <= 1e-7 * expected.abs().max(1e-12),
            "MR10 decay at i={i} t={t}: got {got}, expected {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — solve_ivp on y' = 0 keeps state constant.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_solve_ivp_zero_rhs_keeps_state_constant() {
    let mut rhs = |_t: f64, y: &[f64]| vec![0.0; y.len()];
    let y0 = [3.7_f64, -2.4, 0.0, 11.5];
    let opts = SolveIvpOptions {
        t_span: (0.0, 100.0),
        y0: &y0,
        method: SolverKind::Rk45,
        rtol: 1e-9,
        atol: ToleranceValue::Scalar(1e-12),
        ..SolveIvpOptions::default()
    };
    let res = solve_ivp(&mut rhs, &opts).unwrap();
    for (i, row) in res.y.iter().enumerate() {
        for (j, (&got, &want)) in row.iter().zip(y0.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "MR11 zero-rhs drift at i={i} j={j}: got {got}, expected {want}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — solve_ivp time-shift invariance for autonomous ODE: shifting
// t_span by Δt produces solutions that agree on the shared time
// segment after re-aligning t.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_solve_ivp_autonomous_time_shift_invariance() {
    let mut rhs = |_t: f64, y: &[f64]| y.iter().map(|v| -0.5 * v).collect::<Vec<f64>>();
    let y0 = [2.0_f64];
    let dt = 7.0;
    let t_eval_base: Vec<f64> = (0..=20).map(|i| i as f64 * 0.25).collect();
    let t_eval_shifted: Vec<f64> = t_eval_base.iter().map(|t| t + dt).collect();

    let opts_base = SolveIvpOptions {
        t_span: (0.0, 5.0),
        y0: &y0,
        method: SolverKind::Rk45,
        t_eval: Some(&t_eval_base),
        rtol: 1e-9,
        atol: ToleranceValue::Scalar(1e-12),
        ..SolveIvpOptions::default()
    };
    let opts_shifted = SolveIvpOptions {
        t_span: (dt, 5.0 + dt),
        y0: &y0,
        method: SolverKind::Rk45,
        t_eval: Some(&t_eval_shifted),
        rtol: 1e-9,
        atol: ToleranceValue::Scalar(1e-12),
        ..SolveIvpOptions::default()
    };
    let r_base = solve_ivp(&mut rhs, &opts_base).unwrap();
    let r_shift = solve_ivp(&mut rhs, &opts_shifted).unwrap();
    assert_eq!(r_base.y.len(), r_shift.y.len(), "MR12 length");
    for (i, (a, b)) in r_base.y.iter().zip(&r_shift.y).enumerate() {
        for (j, (&av, &bv)) in a.iter().zip(b).enumerate() {
            assert!(
                (av - bv).abs() <= 1e-6 * av.abs().max(1.0),
                "MR12 shift mismatch at i={i} j={j}: base={av}, shifted={bv}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — cumulative_simpson final element matches simpson over the
// full grid (the running cumulative sum reaches the same total).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cumulative_simpson_final_equals_simpson_total() {
    // Use an odd number of points so cumulative_simpson covers the
    // full grid with no trapezoid-tail correction.
    let n = 21;
    let xs: Vec<f64> = (0..n).map(|i| 0.1 * i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| smooth_exp(x)).collect();
    let cum = cumulative_simpson(&ys, &xs).unwrap();
    let total = simpson(&ys, &xs).unwrap();
    assert_close(
        *cum.last().unwrap(),
        total.integral,
        "MR13 cumulative_simpson final value matches simpson total",
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — dblquad on a separable integrand factorizes:
// ∫_a^b ∫_c^d f(x) g(y) dy dx = (∫f) · (∫g).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dblquad_separable_factorizes() {
    let f_x = |x: f64| (-x * x).exp();
    let g_y = |y: f64| 1.0 + y * y;
    // f(x) g(y) — note dblquad signature is (y, x).
    let integrand = move |y: f64, x: f64| f_x(x) * g_y(y);
    let res = dblquad(
        integrand,
        0.0,
        1.5, // x range
        |_x| 0.0,
        |_x| 1.0, // y range
        DblquadOptions::default(),
    )
    .unwrap();

    let qopts = QuadOptions::default();
    let int_f = quad(f_x, 0.0, 1.5, qopts).unwrap().integral;
    let int_g = quad(g_y, 0.0, 1.0, qopts).unwrap().integral;
    let expected = int_f * int_g;
    assert!(
        (res.integral - expected).abs() < 1e-7 * expected.abs().max(1.0),
        "MR14 dblquad factorization: got {}, expected {expected}",
        res.integral
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — romb on uniformly-spaced samples of a polynomial reproduces
// the exact integral. Romberg integration is exact for polynomials of
// degree ≤ 2k where k is the number of refinement levels (here k=4
// gives exact results for degree ≤ 8).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_romb_exact_on_polynomial() {
    // n = 2^k + 1 = 17 for k = 4 → exact for polynomials of degree ≤ 8.
    let k = 4;
    let n: usize = (1 << k) + 1;
    let dx = 1.0 / (n as f64 - 1.0); // x in [0, 1]
    let xs: Vec<f64> = (0..n).map(|i| i as f64 * dx).collect();
    // f(x) = 3 x^2 + 2 x + 1, exact integral on [0, 1] = 1 + 1 + 1 = 3.
    let ys: Vec<f64> = xs.iter().map(|&x| 3.0 * x * x + 2.0 * x + 1.0).collect();
    let result = romb(&ys, dx).unwrap();
    let expected = 3.0;
    assert!(
        (result - expected).abs() < 1e-12,
        "MR15 romb exact on quadratic: got {result}, expected {expected}"
    );
}
