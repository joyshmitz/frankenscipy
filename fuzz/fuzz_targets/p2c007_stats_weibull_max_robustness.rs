#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, WeibullMax};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-48xts]:
// WeibullMax has 3 anchor + 115-case scipy diff but no fuzz
// coverage. Drive Arbitrary-derived (c, x) where c is
// sanitized to [0.1, 20] and x to [-100, -1e-6], and assert:
//   1. Constructor never panics within validated range.
//   2. pdf finite, non-negative on x < 0.
//   3. pdf = 0 strictly outside support (x > 0).
//   4. cdf monotone non-decreasing along ascending grid in [0, 1].
//   5. cdf(≥ 0) = 1; cdf(very negative) ≈ 0.
//   6. ppf(cdf(x)) round-trip on interior.
//   7. cdf + sf ≈ 1.

const C_MIN: f64 = 0.1;
const C_MAX: f64 = 20.0;
const X_NEG_MIN: f64 = -100.0;
const X_NEG_MAX: f64 = -1.0e-6;

#[derive(Debug, Arbitrary)]
struct Input {
    c_seed: f64,
    x_seed: f64,
}

fn sanitize_c(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(C_MIN, C_MAX)
}

fn sanitize_x(seed: f64) -> f64 {
    if !seed.is_finite() {
        return -1.0;
    }
    let m = -seed.abs();
    m.clamp(X_NEG_MIN, X_NEG_MAX)
}

fuzz_target!(|input: Input| {
    let c = sanitize_c(input.c_seed);
    let x = sanitize_x(input.x_seed);

    // 1. Constructor never panics within (0, 20].
    let dist = WeibullMax::new(c);

    // 2. pdf finite, non-negative on x < 0.
    let p = dist.pdf(x);
    assert!(
        p.is_finite(),
        "pdf({x}; c={c}) must be finite, got {p}"
    );
    assert!(p >= 0.0, "pdf must be non-negative, got {p}");

    // 3. pdf = 0 strictly outside support.
    assert_eq!(dist.pdf(0.5), 0.0, "pdf(>0) must be 0");
    assert_eq!(dist.pdf(0.0), 0.0, "pdf(0) must be 0");

    // 4-5. cdf monotone non-decreasing on a grid (going from
    // very negative toward 0) + boundaries.
    assert_eq!(dist.cdf(0.5), 1.0);
    assert_eq!(dist.cdf(0.0), 1.0);
    let mut prev = 0.0;
    for k in 1..=15 {
        // Grid moves from x toward 0 monotonically.
        let xi = x * ((16 - k) as f64) / 16.0;
        let cv = dist.cdf(xi);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({xi}; c={c}) = {cv} must be in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({xi}; c={c}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 5b. Far-tail saturation (deep negative). For c<1 the tail
    // is heavy — scale probe with 1/c, mirroring kappa3/burr3.
    let probe = -(10.0_f64.powf((20.0 / c).min(300.0)));
    if probe.is_finite() {
        let c_low = dist.cdf(probe);
        let saturation_tol = if c < 0.5 { 1.0e-2 } else { 1.0e-3 };
        assert!(
            c_low < saturation_tol,
            "cdf({probe}; c={c}) = {c_low} must be ≈ 0 (tol {saturation_tol})"
        );
    }

    // 6. ppf(cdf(x)) round-trip on interior.
    let q = dist.cdf(x);
    if (1e-6..1.0 - 1e-6).contains(&q) {
        let recovered = dist.ppf(q);
        let scale = x.abs().max(1.0);
        assert!(
            (recovered - x).abs() < 1e-9 * scale,
            "ppf(cdf({x}; c={c})) = {recovered}, want {x}"
        );
    }

    // 7. cdf + sf ≈ 1 (basic survival relation).
    let sum = dist.cdf(x) + dist.sf(x);
    assert!(
        (sum - 1.0).abs() < 1e-12,
        "cdf({x}; c={c}) + sf = {sum}, must be 1"
    );
});
