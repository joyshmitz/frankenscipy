//! Metamorphic tests for `fsci-special`.
//!
//! Recurrence Γ(x+1) = x Γ(x), gamma/factorial agreement, beta
//! symmetry, ellipk(0) = π/2, gammaln consistency, Bessel symmetry.
//!
//! Run with: `cargo test -p fsci-special --test metamorphic_tests`

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;
use fsci_special::{
    SpecialResult, SpecialTensor, beta, ellipk, erf_scalar, factorial, gamma, gammainc, gammaincc,
    gammaln, i0, i0_scalar, j0, jn,
    orthopoly::{
        eval_chebyt, eval_chebyu, eval_hermite, eval_hermitenorm, eval_laguerre, eval_legendre,
        roots_chebyt, roots_legendre,
    },
};

const ATOL: f64 = 1e-10;
const RTOL: f64 = 1e-9;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

fn real(value: f64) -> SpecialTensor {
    SpecialTensor::RealScalar(value)
}

fn unwrap_real(result: SpecialResult) -> f64 {
    match result.unwrap() {
        SpecialTensor::RealScalar(v) => v,
        other => panic!("expected RealScalar, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — Γ(x + 1) = x · Γ(x) for various non-integer x.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gamma_recurrence() {
    for x in [0.3_f64, 1.7, 2.4, 3.5, 4.2, 5.8, 7.1] {
        let g_x = unwrap_real(gamma(&real(x), RuntimeMode::Strict));
        let g_x1 = unwrap_real(gamma(&real(x + 1.0), RuntimeMode::Strict));
        assert!(
            close(g_x1, x * g_x),
            "MR1 recurrence at x={x}: Γ(x+1)={g_x1} vs x·Γ(x)={}",
            x * g_x
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — Γ(n+1) = n! for non-negative integers.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gamma_matches_factorial() {
    for n in [0u64, 1, 2, 3, 5, 8, 15, 20] {
        let g = unwrap_real(gamma(&real((n + 1) as f64), RuntimeMode::Strict));
        let f = factorial(n);
        let rel_err = ((g - f) / f.abs().max(1.0)).abs();
        assert!(
            rel_err < 1e-9,
            "MR2 Γ({}) = {g}, expected n! = {f}, rel_err={rel_err:e}",
            n + 1
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — beta(a, b) = beta(b, a) (symmetry).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_beta_symmetric() {
    let pairs: [(f64, f64); 5] = [(1.5, 2.5), (3.0, 0.7), (5.5, 4.5), (0.1, 0.9), (10.0, 2.0)];
    for (a, b) in pairs {
        let bab = unwrap_real(beta(&real(a), &real(b), RuntimeMode::Strict));
        let bba = unwrap_real(beta(&real(b), &real(a), RuntimeMode::Strict));
        assert!(close(bab, bba), "MR3 beta symmetry: B({a},{b})={bab} vs B({b},{a})={bba}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — ellipk(0) = π/2 and ellipk(m) ≥ π/2 for m ∈ [0, 1).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ellipk_endpoint_and_monotonicity() {
    let k0 = unwrap_real(ellipk(&real(0.0), RuntimeMode::Strict));
    assert!(close(k0, PI / 2.0), "MR4 ellipk(0)={k0}, expected π/2");
    let mut prev = k0;
    for k in 1..50 {
        let m = k as f64 / 50.0;
        let v = unwrap_real(ellipk(&real(m), RuntimeMode::Strict));
        assert!(
            v >= prev - 1e-12,
            "MR4 ellipk not monotone at m={m}: {v} < prev={prev}"
        );
        assert!(
            v >= PI / 2.0 - 1e-12,
            "MR4 ellipk({m}) below π/2: {v}"
        );
        prev = v;
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — gammaln consistency: gammaln(x) ≈ ln(|gamma(x)|) for x > 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gammaln_matches_log_gamma() {
    for x in [0.5_f64, 1.0, 2.0, 3.5, 7.0, 12.5, 50.0] {
        let g = unwrap_real(gamma(&real(x), RuntimeMode::Strict));
        let lg = unwrap_real(gammaln(&real(x), RuntimeMode::Strict));
        let direct = g.abs().ln();
        let rel_err = (lg - direct).abs() / direct.abs().max(1.0);
        assert!(
            rel_err < 1e-9,
            "MR5 gammaln at x={x}: lg={lg}, ln|Γ|={direct}, rel={rel_err:e}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — Bessel j0(0) = 1 (zeroth-order Bessel at the origin).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bessel_j0_at_zero_is_one() {
    let v = unwrap_real(j0(&real(0.0), RuntimeMode::Strict));
    assert!(close(v, 1.0), "MR6 j0(0) = {v}, expected 1");
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — j_n is even for even n, odd for odd n.
//   j_n(-x) = (-1)^n j_n(x).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_jn_parity() {
    for n in 0..6_i64 {
        for x in [0.5_f64, 1.5, 3.7, 5.0] {
            let pos = unwrap_real(jn(&real(n as f64), &real(x), RuntimeMode::Strict));
            let neg = unwrap_real(jn(&real(n as f64), &real(-x), RuntimeMode::Strict));
            let expected = if n % 2 == 0 { pos } else { -pos };
            assert!(
                close(neg, expected),
                "MR7 j_{n}({x}) parity: j_n(-x)={neg}, expected {expected}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — Modified Bessel i0(0) = 1 and i0(x) ≥ 1 for x ∈ R (i0 is even
//        and convex with minimum 1 at 0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_i0_minimum_at_zero() {
    let v0 = unwrap_real(i0(&real(0.0), RuntimeMode::Strict));
    assert!(close(v0, 1.0), "MR8 i0(0) = {v0}, expected 1");
    for x in [0.1_f64, -0.5, 1.0, -2.5, 5.0, -3.7] {
        let v = i0_scalar(x);
        assert!(
            v >= 1.0 - 1e-12,
            "MR8 i0({x}) = {v} < 1 (i0 has minimum 1 at zero)"
        );
        // i0 is even.
        let v_neg = i0_scalar(-x);
        assert!(
            close(v, v_neg),
            "MR8 i0 not even at x={x}: i0(x)={v}, i0(-x)={v_neg}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — Multiplication formula: Γ(x) · Γ(1-x) · sin(πx) = π
// (reflection formula).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gamma_reflection_formula() {
    for x in [0.3_f64, 0.4, 0.7, 0.9] {
        let g_x = unwrap_real(gamma(&real(x), RuntimeMode::Strict));
        let g_1mx = unwrap_real(gamma(&real(1.0 - x), RuntimeMode::Strict));
        let product = g_x * g_1mx * (PI * x).sin();
        assert!(
            close(product, PI),
            "MR9 reflection at x={x}: Γ(x)·Γ(1-x)·sin(πx) = {product}, expected π"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Orthogonal polynomial relations
// ─────────────────────────────────────────────────────────────────────

// MR10 — Legendre P_n(1) = 1 and P_n(-1) = (-1)^n.
#[test]
fn mr_legendre_endpoints() {
    for n in 0..=10_u32 {
        let p_at_1 = eval_legendre(n, 1.0);
        assert!(close(p_at_1, 1.0), "MR10 P_{n}(1) = {p_at_1}, expected 1");
        let p_at_neg1 = eval_legendre(n, -1.0);
        let expected = if n % 2 == 0 { 1.0 } else { -1.0 };
        assert!(
            close(p_at_neg1, expected),
            "MR10 P_{n}(-1) = {p_at_neg1}, expected {expected}"
        );
    }
}

// MR11 — Chebyshev T_n(cos θ) = cos(n θ).
#[test]
fn mr_chebyt_cos_relation() {
    for n in 0..=8_u32 {
        for k in 0..=20 {
            let theta = k as f64 * PI / 20.0;
            let x = theta.cos();
            let t = eval_chebyt(n, x);
            let expected = (n as f64 * theta).cos();
            assert!(
                close(t, expected),
                "MR11 T_{n}(cos {theta}) = {t}, expected cos({n}θ) = {expected}"
            );
        }
    }
}

// MR12 — Chebyshev U_n(cos θ) sin θ = sin((n+1) θ).
#[test]
fn mr_chebyu_sin_relation() {
    for n in 0..=8_u32 {
        for k in 1..=19 {
            let theta = k as f64 * PI / 20.0;
            let x = theta.cos();
            let u = eval_chebyu(n, x);
            let lhs = u * theta.sin();
            let expected = ((n + 1) as f64 * theta).sin();
            assert!(
                close(lhs, expected),
                "MR12 U_{n}(cos θ) sin θ = {lhs}, expected sin({}θ) = {expected}",
                n + 1
            );
        }
    }
}

// MR13 — Hermite recurrence H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x).
#[test]
fn mr_hermite_three_term_recurrence() {
    for x in [-2.0_f64, -0.5, 0.0, 0.7, 1.5, 3.0] {
        let mut prev = eval_hermite(0, x);
        let mut curr = eval_hermite(1, x);
        for n in 1..=10_u32 {
            let next_expected = 2.0 * x * curr - 2.0 * (n as f64) * prev;
            let next = eval_hermite(n + 1, x);
            assert!(
                close(next, next_expected),
                "MR13 Hermite recurrence at n={n}, x={x}: H_{}(x) = {next}, recurrence = {next_expected}",
                n + 1
            );
            prev = curr;
            curr = next;
        }
    }
}

// MR14 — Probabilist's Hermite recurrence: He_{n+1}(x) = x He_n(x) - n He_{n-1}(x).
#[test]
fn mr_hermitenorm_three_term_recurrence() {
    for x in [-2.0_f64, -0.5, 0.0, 0.7, 1.5, 3.0] {
        let mut prev = eval_hermitenorm(0, x);
        let mut curr = eval_hermitenorm(1, x);
        for n in 1..=10_u32 {
            let next_expected = x * curr - (n as f64) * prev;
            let next = eval_hermitenorm(n + 1, x);
            assert!(
                close(next, next_expected),
                "MR14 He recurrence at n={n}, x={x}: He_{}(x) = {next}, recurrence = {next_expected}",
                n + 1
            );
            prev = curr;
            curr = next;
        }
    }
}

// MR15 — Laguerre L_n(0) = 1 for all n.
#[test]
fn mr_laguerre_at_zero_is_one() {
    for n in 0..=12_u32 {
        let l = eval_laguerre(n, 0.0);
        assert!(close(l, 1.0), "MR15 L_{n}(0) = {l}, expected 1");
    }
}

// MR16 — roots of Legendre P_n give P_n(root_i) ≈ 0.
#[test]
fn mr_roots_legendre_evaluate_to_zero() {
    for n in 2..=10_usize {
        let (xs, weights) = roots_legendre(n);
        assert_eq!(xs.len(), n);
        assert_eq!(weights.len(), n);
        for &x in &xs {
            let v = eval_legendre(n as u32, x);
            assert!(
                v.abs() < 1e-10,
                "MR16 P_{n}(root={x}) = {v}, expected 0"
            );
            // Roots must be in (-1, 1).
            assert!(x.abs() < 1.0, "MR16 root {x} outside open interval");
        }
        // Weights must be positive and sum to 2 (Gauss-Legendre).
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 2.0).abs() < 1e-10,
            "MR16 Legendre weights sum to {sum}, expected 2"
        );
        for &w in &weights {
            assert!(w > 0.0, "MR16 negative Legendre weight: {w}");
        }
    }
}

// MR17 — roots of Chebyshev T_n give T_n(root_i) ≈ 0; weights sum to π.
#[test]
fn mr_roots_chebyt_evaluate_to_zero() {
    for n in 2..=10_usize {
        let (xs, weights) = roots_chebyt(n);
        assert_eq!(xs.len(), n);
        for &x in &xs {
            let v = eval_chebyt(n as u32, x);
            assert!(
                v.abs() < 1e-10,
                "MR17 T_{n}(root={x}) = {v}, expected 0"
            );
            assert!(x.abs() < 1.0, "MR17 root {x} outside (-1, 1)");
        }
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - PI).abs() < 1e-10,
            "MR17 Chebyshev T weights sum to {sum}, expected π"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — Gauss-Legendre n-point quadrature integrates polynomials of
// degree ≤ 2n − 1 exactly on [−1, 1].
//
// We test against ∫_{-1}^{1} x^k dx, which equals 2/(k+1) for even k
// and 0 for odd k. Required tolerance is fairly tight because the
// Gauss-Legendre weights and nodes can be evaluated to machine
// precision.
// ─────────────────────────────────────────────────────────────────────

fn analytic_x_k(k: usize) -> f64 {
    if k % 2 == 1 {
        0.0
    } else {
        2.0 / (k as f64 + 1.0)
    }
}

#[test]
fn mr_gauss_legendre_exactness_on_polynomials() {
    for n in 2..=8_usize {
        let (xs, weights) = roots_legendre(n);
        let max_degree = 2 * n - 1;
        for k in 0..=max_degree {
            let approx: f64 = xs
                .iter()
                .zip(&weights)
                .map(|(x, w)| w * x.powi(k as i32))
                .sum();
            let exact = analytic_x_k(k);
            let bound = 1e-10 + 1e-9 * exact.abs().max(1.0);
            assert!(
                (approx - exact).abs() <= bound,
                "MR18 GL-{n} on x^{k}: got {approx:.16e}, exact {exact:.16e}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — Gauss-Chebyshev T n-point quadrature integrates
//   ∫_{-1}^{1} f(x) / sqrt(1 − x²) dx
// exactly for f a polynomial of degree ≤ 2n − 1. The simplest closed
// form: ∫_{-1}^{1} x^{2k} / sqrt(1 − x²) dx = π · (2k)! / (4^k (k!)²)
// (and zero for odd powers).
// ─────────────────────────────────────────────────────────────────────

fn central_binomial_coefficient(k: usize) -> f64 {
    // C(2k, k) = (2k)! / (k!)²
    let mut c = 1.0_f64;
    for i in 0..k {
        c *= (k + i + 1) as f64;
        c /= (i + 1) as f64;
    }
    c
}

fn analytic_gc_x_k(k: usize) -> f64 {
    if k % 2 == 1 {
        0.0
    } else {
        let half = k / 2;
        PI * central_binomial_coefficient(half) / (4.0_f64.powi(half as i32))
    }
}

#[test]
fn mr_gauss_chebyt_exactness_on_polynomials() {
    for n in 2..=8_usize {
        let (xs, weights) = roots_chebyt(n);
        let max_degree = 2 * n - 1;
        for k in 0..=max_degree {
            let approx: f64 = xs
                .iter()
                .zip(&weights)
                .map(|(x, w)| w * x.powi(k as i32))
                .sum();
            let exact = analytic_gc_x_k(k);
            let bound = 1e-10 + 1e-9 * exact.abs().max(1.0);
            assert!(
                (approx - exact).abs() <= bound,
                "MR19 GCT-{n} on x^{k}: got {approx:.16e}, exact {exact:.16e}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — Boundary values of gammainc:
//   gammainc(a, 0) = 0 and gammainc(a, ∞) = 1.
// gammaincc is the complement, so gammaincc(a, 0) = 1, gammaincc(a, ∞) = 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gammainc_endpoints() {
    for a in [0.5_f64, 1.0, 2.0, 5.0, 10.0] {
        let p_at_0 = unwrap_real(gammainc(&real(a), &real(0.0), RuntimeMode::Strict));
        assert!(
            p_at_0.abs() < 1e-12,
            "MR20 gammainc({a}, 0) = {p_at_0}, expected 0"
        );
        // For x = 100 we should be very close to 1 for all a ≤ 10.
        let p_at_inf = unwrap_real(gammainc(&real(a), &real(100.0), RuntimeMode::Strict));
        assert!(
            (p_at_inf - 1.0).abs() < 1e-9,
            "MR20 gammainc({a}, large) = {p_at_inf}, expected 1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — gammainc(a, x) + gammaincc(a, x) = 1 for any (a > 0, x ≥ 0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gammainc_complement() {
    for a in [0.7_f64, 1.5, 3.0, 7.5] {
        for x in [0.1_f64, 0.5, 1.0, 3.0, 7.0, 15.0] {
            let p = unwrap_real(gammainc(&real(a), &real(x), RuntimeMode::Strict));
            let q = unwrap_real(gammaincc(&real(a), &real(x), RuntimeMode::Strict));
            let sum = p + q;
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "MR21 gammainc({a}, {x}) + gammaincc({a}, {x}) = {sum}, expected 1"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — erf and gammainc relationship: for x ≥ 0,
//   erf(x) = gammainc(1/2, x²).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_erf_equals_gammainc_half() {
    for x in [0.1_f64, 0.5, 1.0, 1.5, 2.5, 4.0] {
        let lhs = erf_scalar(x);
        let rhs = unwrap_real(gammainc(&real(0.5), &real(x * x), RuntimeMode::Strict));
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "MR22 erf({x}) = {lhs}, gammainc(1/2, {}²) = {rhs}",
            x
        );
    }
}
