#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{Burr3, ContinuousDistribution};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-uwla7]:
// Burr3 has 3 anchor cases + scipy diff (575 cases) but no
// fuzz coverage. Drive Arbitrary-derived (c, d, x) and assert:
//   1. Constructor never panics for sanitized c, d ∈ [0.1, 20].
//   2. pdf finite, non-negative on x > 0.
//   3. pdf = 0 strictly outside support (x ≤ 0).
//   4. cdf monotone non-decreasing along ascending grid in [0, 1].
//   5. cdf(≤ 0) = 0; far-tail cdf approaches 1.
//   6. ppf(cdf(x)) round-trip on interior.
//   7. cdf + sf ≈ 1.
//
// Tail rate: cdf(x; c, d) = (1 + x^(-c))^(-d). For large x,
// x^(-c) → 0 and cdf → 1; the rate scales with 1/c, so the
// far-tail probe must scale with c (small c → heavy tail).

const PARAM_MIN: f64 = 0.1;
const PARAM_MAX: f64 = 20.0;
const X_MIN: f64 = 1.0e-6;
const X_MAX: f64 = 100.0;

#[derive(Debug, Arbitrary)]
struct Input {
    c_seed: f64,
    d_seed: f64,
    x_seed: f64,
}

fn sanitize_param(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(PARAM_MIN, PARAM_MAX)
}

fn sanitize_x(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(X_MIN, X_MAX)
}

fuzz_target!(|input: Input| {
    let c = sanitize_param(input.c_seed);
    let d = sanitize_param(input.d_seed);
    let x = sanitize_x(input.x_seed);

    // 1. Constructor never panics within validated range.
    let dist = Burr3::new(c, d);

    // 2. pdf finite, non-negative on x > 0.
    let p = dist.pdf(x);
    assert!(
        p.is_finite(),
        "pdf({x}; c={c}, d={d}) must be finite, got {p}"
    );
    assert!(p >= 0.0, "pdf must be non-negative, got {p}");

    // 3. pdf = 0 strictly outside support.
    assert_eq!(dist.pdf(-0.5), 0.0, "pdf(<0) must be 0");
    assert_eq!(dist.pdf(0.0), 0.0, "pdf(0) must be 0");

    // 4-5. cdf monotone non-decreasing on a grid + boundaries.
    assert_eq!(dist.cdf(-0.5), 0.0);
    assert_eq!(dist.cdf(0.0), 0.0);
    let mut prev = 0.0;
    for k in 1..=15 {
        let xi = (k as f64) * x / 7.5;
        let cv = dist.cdf(xi);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({xi}; c={c}, d={d}) = {cv} must be in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({xi}; c={c}, d={d}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 5b. Far-tail saturation. Tail rate is x^(-c), so probe at
    // x = 10^(20/c) gives x^(-c) = 1e-20 — well below saturation
    // floor. Scaled tolerance 1e-2 for very small c (heavy tail).
    let probe = 10.0_f64.powf((20.0 / c).min(300.0));
    if probe.is_finite() {
        let c_high = dist.cdf(probe);
        let saturation_tol = if c < 0.5 { 1.0e-2 } else { 1.0e-3 };
        assert!(
            c_high > 1.0 - saturation_tol,
            "cdf({probe}; c={c}, d={d}) = {c_high} must approach 1 (tol {saturation_tol})"
        );
    }

    // 6. ppf(cdf(x)) round-trip on interior.
    let q = dist.cdf(x);
    if (1e-6..1.0 - 1e-6).contains(&q) {
        let recovered = dist.ppf(q);
        let scale = x.abs().max(1.0);
        assert!(
            (recovered - x).abs() < 1e-9 * scale,
            "ppf(cdf({x}; c={c}, d={d})) = {recovered}, want {x}"
        );
    }

    // 7. cdf + sf ≈ 1 (basic survival relation).
    let sum = dist.cdf(x) + dist.sf(x);
    assert!(
        (sum - 1.0).abs() < 1e-12,
        "cdf({x}; c={c}, d={d}) + sf = {sum}, must be 1"
    );
});
