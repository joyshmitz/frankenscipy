#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, Kappa3};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-4397j]:
// Kappa3 has 3 anchor cases + scipy diff but no fuzz coverage.
// Drive Arbitrary-derived (a, x) and assert:
//   1. Constructor never panics for sanitized a ∈ [0.1, 20].
//   2. pdf finite, non-negative on x > 0.
//   3. pdf = 0 strictly outside support (x ≤ 0).
//   4. cdf monotone non-decreasing along ascending grid in [0, 1].
//   5. cdf(≤ 0) = 0; far-tail cdf approaches 1.
//   6. ppf(cdf(x)) round-trip on interior.
//   7. cdf + sf ≈ 1.

const A_MIN: f64 = 0.1;
const A_MAX: f64 = 20.0;
const X_MIN: f64 = 1.0e-6;
const X_MAX: f64 = 100.0;

#[derive(Debug, Arbitrary)]
struct Input {
    a_seed: f64,
    x_seed: f64,
}

fn sanitize_a(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(A_MIN, A_MAX)
}

fn sanitize_x(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(X_MIN, X_MAX)
}

fuzz_target!(|input: Input| {
    let a = sanitize_a(input.a_seed);
    let x = sanitize_x(input.x_seed);

    // 1. Constructor never panics within (0, 20].
    let dist = Kappa3::new(a);

    // 2. pdf finite, non-negative on x > 0.
    let p = dist.pdf(x);
    assert!(p.is_finite(), "pdf({x}; a={a}) must be finite, got {p}");
    assert!(p >= 0.0, "pdf must be non-negative, got {p}");

    // 3. pdf = 0 strictly outside support.
    assert_eq!(dist.pdf(-0.5), 0.0, "pdf(<0) must be 0");
    assert_eq!(dist.pdf(0.0), 0.0, "pdf(0) must be 0");

    // 4-5. cdf monotone non-decreasing on a grid + boundaries.
    assert_eq!(dist.cdf(-0.5), 0.0);
    assert_eq!(dist.cdf(0.0), 0.0);
    let mut prev = 0.0;
    // Grid spans 5 orders of magnitude scaled by x.
    for k in 1..=15 {
        let xi = (k as f64) * x / 7.5;
        let cv = dist.cdf(xi);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({xi}; a={a}) = {cv} must be in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({xi}; a={a}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 5b. Far-tail saturation. Kappa3 has cdf(x; a) → 1 as
    // x → ∞ for any a > 0, but the rate depends strongly on a:
    //   x^a dominates 'a' once x^a ≫ a, after which cdf ≈ 1.
    // For a = 0.15, x^a only reaches 178 at x = 1e15; full
    // saturation needs x ~ 1e20. Use a probe that scales the
    // critical x with the inverse of a, and a tolerance that
    // tracks the heavy tail's convergence floor.
    let probe = 10.0_f64.powf((20.0 / a).min(300.0));
    if probe.is_finite() {
        let c_high = dist.cdf(probe);
        let saturation_tol = if a < 0.5 { 1.0e-2 } else { 1.0e-3 };
        assert!(
            c_high > 1.0 - saturation_tol,
            "cdf({probe}; a={a}) = {c_high} must approach 1 (tol {saturation_tol})"
        );
    }

    // 6. ppf(cdf(x)) round-trip on interior.
    let q = dist.cdf(x);
    if (1e-6..1.0 - 1e-6).contains(&q) {
        let recovered = dist.ppf(q);
        let scale = x.abs().max(1.0);
        assert!(
            (recovered - x).abs() < 1e-9 * scale,
            "ppf(cdf({x}; a={a})) = {recovered}, want {x}"
        );
    }

    // 7. cdf + sf ≈ 1 (basic survival relation).
    let sum = dist.cdf(x) + dist.sf(x);
    assert!(
        (sum - 1.0).abs() < 1e-12,
        "cdf({x}; a={a}) + sf({x}; a={a}) = {sum}, must be 1"
    );
});
