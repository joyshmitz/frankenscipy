#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, SkewCauchy};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-cttfi]:
// SkewCauchy has 4 closed-form anchor cases but no fuzz
// coverage. Drive Arbitrary-derived (a, x, q) inputs and assert:
//   1. SkewCauchy::new never panics for sanitized a ∈ (-0.99, 0.99).
//   2. pdf(x) is finite, non-negative, and ≤ 1/π (peak at x=0).
//   3. cdf is monotone non-decreasing along ascending x.
//   4. cdf(x) ∈ [0, 1] for finite x.
//   5. ppf(cdf(x)) ≈ x within tol.
//   6. cdf(-1e10) ≈ 0 and cdf(1e10) ≈ 1 (asymptotic tails).

const A_BOUND: f64 = 0.99;
const X_BOUND: f64 = 1.0e6;

#[derive(Debug, Arbitrary)]
struct Input {
    a_seed: f64,
    x_seed: f64,
}

fn sanitize_a(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 0.0;
    }
    let abs = seed.abs() % A_BOUND;
    abs * seed.signum()
}

fn sanitize_x(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 0.0;
    }
    seed.clamp(-X_BOUND, X_BOUND)
}

fuzz_target!(|input: Input| {
    let a = sanitize_a(input.a_seed);
    let x = sanitize_x(input.x_seed);

    // 1. Constructor never panics within (-1, 1).
    let dist = SkewCauchy::new(a);

    // 2. pdf finite, non-negative, ≤ 1/π.
    let p = dist.pdf(x);
    assert!(p.is_finite(), "pdf({x}; a={a}) must be finite, got {p}");
    assert!(p >= 0.0, "pdf must be non-negative, got {p}");
    let inv_pi = std::f64::consts::FRAC_1_PI;
    // 1/π is the global peak (at x=0). Leave a small margin for fp.
    assert!(
        p <= inv_pi + 1e-12,
        "pdf({x}; a={a}) = {p} must be ≤ 1/π ≈ {inv_pi}"
    );

    // 3-4. cdf monotone non-decreasing in x and ∈ [0, 1].
    let mut prev = 0.0;
    for &xi in &[-1.0e6_f64, -1.0, -0.1, 0.0, 0.1, 1.0, 1.0e6] {
        let c = dist.cdf(xi);
        assert!(c.is_finite(), "cdf({xi}; a={a}) must be finite, got {c}");
        assert!(
            (0.0..=1.0).contains(&c),
            "cdf({xi}; a={a}) = {c} must be in [0, 1]"
        );
        assert!(
            c >= prev - 1e-15,
            "cdf must be non-decreasing: cdf at xi={xi} (a={a}) = {c} < prev = {prev}"
        );
        prev = c;
    }

    // 5. ppf(cdf(x)) round-trip — only for moderate x. The closed-form
    // ppf uses tan(...) which amplifies floating-point error near
    // π/2. With |x| ≥ 1e3 and a near ±1, the saturated cdf loses
    // enough trailing bits that the ppf can drift by more than 1
    // ulp of x — that's a numerical limit of the analytic inverse,
    // not a defect.
    if x.abs() < 1.0e3 && a.abs() < 0.9 {
        let q = dist.cdf(x);
        if q > 1e-10 && q < 1.0 - 1e-10 {
            let recovered = dist.ppf(q);
            let scale = x.abs().max(1.0);
            assert!(
                (recovered - x).abs() <= 1e-8 * scale,
                "ppf(cdf({x}; a={a})) = {recovered}, want {x}"
            );
        }
    }

    // 6. Asymptotic tails: cdf(-1e10) ≈ 0, cdf(1e10) ≈ 1.
    let c_low = dist.cdf(-1.0e10);
    let c_high = dist.cdf(1.0e10);
    assert!(c_low < 1.0e-9, "cdf(-1e10; a={a}) = {c_low} must be ≈ 0");
    assert!(
        c_high > 1.0 - 1.0e-9,
        "cdf(1e10; a={a}) = {c_high} must be ≈ 1"
    );
});
