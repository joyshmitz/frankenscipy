#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, KsTwoBign};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-p1i9n]:
// KsTwoBign has 3 anchor cases + scipy diff but no fuzz coverage.
// Drive Arbitrary x-values and assert:
//   1. Constructor never panics (parameterless).
//   2. pdf finite, non-negative on x > 0.
//   3. pdf = 0 strictly outside support (x ≤ 0).
//   4. cdf monotone non-decreasing along ascending grid in [0, 1].
//   5. cdf(≤ 0) = 0; far-tail cdf approaches 1.
//   6. ppf(cdf(x)) round-trip on interior.
//   7. cdf + sf ≈ 1.

const X_MAX: f64 = 6.0;

#[derive(Debug, Arbitrary)]
struct Input {
    x_seed: f64,
}

fn sanitize_x(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    // KsTwoBign saturates to ~1 by x ≈ 4 and to numerical 1 well
    // before x ≈ 6, so clamp into a meaningful exploration range.
    seed.abs().fract() * X_MAX + 1.0e-3
}

fuzz_target!(|input: Input| {
    let x = sanitize_x(input.x_seed);

    // 1. Constructor never panics.
    let dist = KsTwoBign;

    // 2. pdf finite, non-negative on x > 0.
    let p = dist.pdf(x);
    assert!(p.is_finite(), "pdf({x}) must be finite, got {p}");
    assert!(p >= 0.0, "pdf must be non-negative, got {p}");

    // 3. pdf = 0 strictly outside support.
    assert_eq!(dist.pdf(-0.5), 0.0, "pdf(<0) must be 0");
    assert_eq!(dist.pdf(0.0), 0.0, "pdf(0) must be 0");

    // 4-5. cdf monotone non-decreasing on [0, X_MAX] grid + boundaries.
    assert_eq!(dist.cdf(-0.5), 0.0);
    assert_eq!(dist.cdf(0.0), 0.0);
    let mut prev = 0.0;
    for k in 1..=24 {
        let xi = (k as f64) * X_MAX / 24.0;
        let cv = dist.cdf(xi);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({xi}) = {cv} must be in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({xi}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 5b. Far-tail saturation. Series truncates well before x = 10.
    let c_high = dist.cdf(10.0);
    assert!(
        c_high > 1.0 - 1.0e-12,
        "cdf(10) = {c_high} must be ≈ 1"
    );

    // 6. ppf(cdf(x)) round-trip on interior.
    let q = dist.cdf(x);
    if (1e-6..1.0 - 1e-6).contains(&q) {
        let recovered = dist.ppf(q);
        let scale = x.abs().max(1.0);
        assert!(
            (recovered - x).abs() < 1e-6 * scale,
            "ppf(cdf({x})) = {recovered}, want {x}"
        );
    }

    // 7. cdf + sf ≈ 1 (basic survival relation).
    let sum = dist.cdf(x) + dist.sf(x);
    assert!(
        (sum - 1.0).abs() < 1e-12,
        "cdf({x}) + sf({x}) = {sum}, must be 1"
    );
});
