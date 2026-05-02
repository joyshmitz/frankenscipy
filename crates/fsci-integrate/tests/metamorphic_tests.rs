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
    cumulative_trapezoid, cumulative_trapezoid_initial, dblquad, fixed_quad, gauss_kronrod_quad,
    gauss_legendre, monte_carlo_integrate, newton_cotes, quad, quad_full_inf, quad_inf, romb,
    romb_func, romberg, simpson, simpson_irregular, simpson_uniform, solve_ivp, trapezoid,
    trapezoid_irregular, tplquad, trapezoid_richardson,
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

// ─────────────────────────────────────────────────────────────────────
// MR16 — quad ∫₀^π sin(x) dx = 2 (textbook closed form).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_sin_over_pi_is_two() {
    let opts = QuadOptions::default();
    let result = quad(|x: f64| x.sin(), 0.0, std::f64::consts::PI, opts).unwrap();
    assert!(
        (result.integral - 2.0).abs() < 1e-9,
        "MR16 ∫₀^π sin(x) dx = {}, expected 2",
        result.integral
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR17 — quad ∫₀^π cos²(x) dx = π/2.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_cos_squared_over_pi() {
    let opts = QuadOptions::default();
    let result = quad(
        |x: f64| x.cos().powi(2),
        0.0,
        std::f64::consts::PI,
        opts,
    )
    .unwrap();
    let expected = std::f64::consts::PI / 2.0;
    assert!(
        (result.integral - expected).abs() < 1e-9,
        "MR17 ∫₀^π cos²(x) dx = {}, expected π/2 = {expected}",
        result.integral
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — quad ∫₀^B e^(-x) dx ≈ 1 - e^(-B). For B = 30, this is
// essentially 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_exp_decay_truncated() {
    let opts = QuadOptions::default();
    let b = 30.0_f64;
    let result = quad(|x: f64| (-x).exp(), 0.0, b, opts).unwrap();
    let expected = 1.0 - (-b).exp();
    assert!(
        (result.integral - expected).abs() < 1e-9,
        "MR18 ∫₀^30 e^(-x) dx = {}, expected {expected}",
        result.integral
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — fixed_quad with n = 5 nodes integrates polynomials of degree
// ≤ 9 exactly (Gauss-Legendre). Verify on a degree-7 polynomial.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fixed_quad_exact_on_degree_seven_polynomial() {
    // p(x) = 3x⁷ + 2x⁵ - x³ + 4x + 1
    // ∫₀^1 p(x) dx = 3/8 + 2/6 - 1/4 + 2 + 1 = 0.375 + 0.333... - 0.25 + 3 = 3.4583...
    let p = |x: f64| 3.0 * x.powi(7) + 2.0 * x.powi(5) - x.powi(3) + 4.0 * x + 1.0;
    let exact = 3.0 / 8.0 + 2.0 / 6.0 - 1.0 / 4.0 + 4.0 / 2.0 + 1.0;
    let (val, _n) = fixed_quad(p, 0.0, 1.0, 5).unwrap();
    assert!(
        (val - exact).abs() < 1e-12,
        "MR19 fixed_quad(p, n=5) = {val} vs exact = {exact}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — romberg(f) and quad(f) agree on a smooth function.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_romberg_matches_quad() {
    // f(x) = sin(x) on [0, π], exact value = 2.
    let f = |x: f64| x.sin();
    let r = romberg(f, 0.0, std::f64::consts::PI, 1e-10, 12);
    let q = quad(f, 0.0, std::f64::consts::PI, QuadOptions::default()).unwrap();
    assert!(
        (r.integral - q.integral).abs() < 1e-7,
        "MR20 romberg = {} vs quad = {}",
        r.integral,
        q.integral
    );
    assert!(
        (q.integral - 2.0).abs() < 1e-9,
        "MR20 quad(sin, 0, π) = {}, expected 2",
        q.integral
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — simpson_uniform with any subdivision is exact on cubics
// (Simpson's rule has degree of precision 3).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_simpson_uniform_exact_on_cubic() {
    // f(x) = 2x³ - x² + 3x + 5 on [0, 4]
    // ∫ = (1/2)·256 - 64/3 + 24 + 20 = 128 - 21.333.. + 44 = 150.666...
    let exact = 0.5 * 256.0 - 64.0 / 3.0 + 24.0 + 20.0;
    for &n_intervals in &[4usize, 8, 16, 32] {
        let h = 4.0 / n_intervals as f64;
        let y: Vec<f64> = (0..=n_intervals)
            .map(|i| {
                let x = i as f64 * h;
                2.0 * x.powi(3) - x.powi(2) + 3.0 * x + 5.0
            })
            .collect();
        let res = simpson_uniform(&y, h).unwrap();
        assert!(
            (res.integral - exact).abs() < 1e-10,
            "MR21 simpson_uniform n={n_intervals} got {} vs exact {exact}",
            res.integral
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — cumulative_trapezoid is monotone non-decreasing for a non-
// negative integrand.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cumulative_trapezoid_monotone_on_nonneg_integrand() {
    // y(x) = x² + 1 on a non-uniform grid — strictly positive.
    let x: Vec<f64> = vec![0.0, 0.3, 0.7, 1.4, 2.1, 3.0, 4.5];
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi + 1.0).collect();
    let cum = cumulative_trapezoid(&y, &x).unwrap();
    for w in cum.windows(2) {
        assert!(
            w[0] <= w[1] + 1e-12,
            "MR22 cumulative_trapezoid not monotone: {} > {}",
            w[0],
            w[1]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — quad of an odd function over a symmetric interval is 0.
// ∫_{-a}^{a} sin(x) dx = 0 for any a.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_odd_function_symmetric_interval_zero() {
    let opts = QuadOptions::default();
    for &a in &[1.0_f64, std::f64::consts::PI, 2.5, 5.0, 10.0] {
        let r = quad(|x: f64| x.sin(), -a, a, opts.clone()).unwrap();
        assert!(
            r.integral.abs() < 1e-9,
            "MR23 ∫_{{-{a}}}^{{{a}}} sin(x) dx = {}, expected 0",
            r.integral
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — Triple integral of a constant 1 over a unit cube equals 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_tplquad_constant_one_unit_cube() {
    let opts = DblquadOptions::default();
    let res = tplquad(
        |_x, _y, _z| 1.0,
        0.0,
        1.0,
        |_| 0.0,
        |_| 1.0,
        |_, _| 0.0,
        |_, _| 1.0,
        opts,
    )
    .unwrap();
    assert!(
        (res.integral - 1.0).abs() < 1e-9,
        "MR24 tplquad(1, unit cube) = {}, expected 1",
        res.integral
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — gauss_legendre with n nodes is exact on polynomials of
// degree ≤ 2n - 1 (textbook degree-of-precision).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gauss_legendre_exact_on_degree_2n_minus_1() {
    // Test n = 4 → exact on degree 7.
    // p(x) = 4x⁷ - 2x⁵ + x³ - x + 1; ∫₀^1 p dx = 4/8 - 2/6 + 1/4 - 1/2 + 1 = 0.916666…
    let p = |x: f64| 4.0 * x.powi(7) - 2.0 * x.powi(5) + x.powi(3) - x + 1.0;
    let exact = 4.0 / 8.0 - 2.0 / 6.0 + 1.0 / 4.0 - 1.0 / 2.0 + 1.0;
    let val = gauss_legendre(p, 0.0, 1.0, 4);
    assert!(
        (val - exact).abs() < 1e-12,
        "MR25 gauss_legendre n=4 on deg 7 = {val} vs {exact}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — cumulative_trapezoid_initial(y, x, c)[0] = c (the initial
// value sets the first sample).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cumulative_trapezoid_initial_first_entry() {
    let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
    for &c in &[0.0_f64, 1.5, -3.0, 100.0] {
        let cum = cumulative_trapezoid_initial(&y, &x, c);
        assert!(
            (cum[0] - c).abs() < 1e-12,
            "MR26 cumulative_trapezoid_initial[0] = {} vs initial = {c}",
            cum[0]
        );
        assert_eq!(
            cum.len(),
            x.len(),
            "MR26 length should equal input"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — Monte Carlo integration of constant 1 over a box returns the
// box volume (within stochastic tolerance).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_monte_carlo_constant_returns_volume() {
    let bounds = vec![(0.0_f64, 2.0), (0.0, 3.0), (0.0, 4.0)]; // volume 24
    let (val, _stderr) = monte_carlo_integrate(|_x| 1.0, &bounds, 10_000, 7);
    assert!(
        (val - 24.0).abs() < 0.5,
        "MR27 MC constant volume = {val}, expected 24"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — quad_full_inf computes ∫_{-∞}^{∞} e^(-x²) dx = √π.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_full_inf_gaussian_integral() {
    let opts = QuadOptions::default();
    let r = quad_full_inf(|x: f64| (-x * x).exp(), opts).unwrap();
    let expected = std::f64::consts::PI.sqrt();
    assert!(
        (r.integral - expected).abs() < 1e-6,
        "MR28 ∫_{{-∞}}^{{∞}} e^(-x²) dx = {} vs √π = {expected}",
        r.integral
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — trapezoid_richardson on a smooth function approximates the
// true integral better than (or as well as) plain trapezoid.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_trapezoid_richardson_improves_smooth_integral() {
    let n = 64;
    let h = std::f64::consts::PI / n as f64;
    let x: Vec<f64> = (0..=n).map(|i| i as f64 * h).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let exact = 2.0_f64; // ∫₀^π sin x dx = 2
    let plain = trapezoid(&y, &x).unwrap().integral;
    let rich = trapezoid_richardson(&y, &x);
    let err_plain = (plain - exact).abs();
    let err_rich = (rich - exact).abs();
    // Richardson should be at least as accurate.
    assert!(
        err_rich <= err_plain + 1e-15,
        "MR29 richardson err = {err_rich} > plain err = {err_plain}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — Composite: MC integral of x² over [0, 1] approximates 1/3.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_monte_carlo_x_squared() {
    let bounds = vec![(0.0_f64, 1.0)];
    let (val, _stderr) = monte_carlo_integrate(|x| x[0] * x[0], &bounds, 50_000, 13);
    assert!(
        (val - 1.0 / 3.0).abs() < 0.02,
        "MR30 MC ∫₀^1 x² ≈ {val}, expected 1/3"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — quad_inf computes ∫₀^∞ e^(-x) dx = 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quad_inf_exp_decay() {
    let opts = QuadOptions::default();
    let r = quad_inf(|x: f64| (-x).exp(), 0.0, opts).unwrap();
    assert!(
        (r.integral - 1.0).abs() < 1e-6,
        "MR31 ∫₀^∞ e^(-x) dx = {} vs 1",
        r.integral
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — newton_cotes weights sum to 1 (closed Newton-Cotes formulas
// integrate the constant 1 over a unit interval to 1).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_newton_cotes_weights_sum_to_one() {
    for n in [1usize, 2, 3, 4] {
        let w = newton_cotes(n).unwrap();
        let s: f64 = w.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "MR32 newton_cotes(n={n}) Σw = {s}, expected 1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — gauss_kronrod_quad and quad agree on a smooth function.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gauss_kronrod_matches_quad() {
    let opts = QuadOptions::default();
    let f = |x: f64| (x * x).exp() * x.cos();
    let gk = gauss_kronrod_quad(f, -1.0, 1.0, opts);
    let q = quad(f, -1.0, 1.0, opts).unwrap();
    assert!(
        (gk.integral - q.integral).abs() < 1e-7,
        "MR33 GK = {} vs quad = {}",
        gk.integral,
        q.integral
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — trapezoid_irregular agrees with trapezoid on a uniform grid.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_trapezoid_irregular_matches_uniform_trapezoid() {
    let n = 16;
    let h = std::f64::consts::PI / n as f64;
    let x: Vec<f64> = (0..=n).map(|i| i as f64 * h).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let irregular = trapezoid_irregular(&y, &x);
    let uniform = trapezoid(&y, &x).unwrap().integral;
    assert!(
        (irregular - uniform).abs() < 1e-12,
        "MR34 trapezoid_irregular = {irregular} vs trapezoid = {uniform}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — simpson_irregular ≈ simpson on a uniform grid for a smooth
// function. (Both should integrate sin x on [0, π] to ≈ 2.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_simpson_irregular_close_to_uniform_simpson() {
    let n = 16;
    let h = std::f64::consts::PI / n as f64;
    let x: Vec<f64> = (0..=n).map(|i| i as f64 * h).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let irregular = simpson_irregular(&y, &x);
    let uniform = simpson(&y, &x).unwrap().integral;
    assert!(
        (irregular - uniform).abs() < 1e-9,
        "MR35 simpson_irregular = {irregular} vs simpson = {uniform}"
    );
    assert!(
        (irregular - 2.0).abs() < 1e-3,
        "MR35 ∫sin on [0, π] ≈ {irregular}, expected 2"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — romb_func on sin x over [0, π] yields ≈ 2.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_romb_func_sin_over_pi() {
    let f = |x: f64| x.sin();
    let r = romb_func(f, 0.0, std::f64::consts::PI, Some(8), None).unwrap();
    assert!(
        (r.integral - 2.0).abs() < 1e-9,
        "MR36 romb_func(sin) = {} vs 2",
        r.integral
    );
}



