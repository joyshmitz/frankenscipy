//! Metamorphic tests for `fsci-special`.
//!
//! Recurrence Γ(x+1) = x Γ(x), gamma/factorial agreement, beta
//! symmetry, ellipk(0) = π/2, gammaln consistency, Bessel symmetry.
//!
//! Run with: `cargo test -p fsci-special --test metamorphic_tests`

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;
use fsci_special::{
    SpecialResult, SpecialTensor, beta, ellipk, factorial, gamma, gammaln, i0, i0_scalar, j0, jn,
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
