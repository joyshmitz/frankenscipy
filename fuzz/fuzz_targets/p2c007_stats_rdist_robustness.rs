#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, RDist};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-h0gc3]:
// RDist has 4 anchor cases + scipy diff but no fuzz coverage.
// Drive Arbitrary-derived (c, x) and assert:
//   1. Constructor never panics for sanitized c ∈ [0.1, 50].
//   2. pdf finite, non-negative on (-1, 1) interior.
//   3. pdf = 0 strictly outside [-1, 1].
//   4. cdf monotone non-decreasing along ascending grid.
//   5. cdf(-1) = 0 (exactly), cdf(1) = 1 (exactly), cdf in [0, 1].
//   6. ppf round-trip on interior.
//   7. pdf symmetric around 0: pdf(x) = pdf(-x).

const C_MIN: f64 = 0.1;
const C_MAX: f64 = 50.0;

#[derive(Debug, Arbitrary)]
struct Input {
    c_seed: f64,
    x_seed: f64,
}

fn sanitize_c(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 2.0;
    }
    seed.abs().clamp(C_MIN, C_MAX)
}

fn sanitize_x_in_support(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 0.0;
    }
    let u = seed.abs().fract();
    -1.0 + 2.0 * u
}

fuzz_target!(|input: Input| {
    let c = sanitize_c(input.c_seed);
    let x = sanitize_x_in_support(input.x_seed);

    // 1. Constructor never panics within (0, 50].
    let dist = RDist::new(c);

    // 2. pdf finite, non-negative on (-1, 1).
    if x.abs() < 1.0 {
        let p = dist.pdf(x);
        assert!(p.is_finite(), "pdf({x}; c={c}) must be finite, got {p}");
        assert!(p >= 0.0, "pdf must be non-negative, got {p}");
    }

    // 3. pdf = 0 strictly outside [-1, 1].
    assert_eq!(dist.pdf(-1.5), 0.0, "pdf(<-1) must be 0");
    assert_eq!(dist.pdf(1.5), 0.0, "pdf(>1) must be 0");

    // 4-5. cdf monotone non-decreasing + boundary values.
    assert_eq!(dist.cdf(-1.5), 0.0, "cdf(<-1) must be 0");
    assert_eq!(dist.cdf(1.5), 1.0, "cdf(>1) must be 1");
    let mut prev = 0.0;
    for k in 0..=20 {
        let xi = -1.0 + (k as f64) * 0.1;
        let cv = dist.cdf(xi);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({xi}; c={c}) = {cv} must be finite and in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({xi}; c={c}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 6. ppf round-trip on interior. Skip near boundaries where the
    // cdf saturates (Beta cdf via regularized_incomplete_beta loses
    // precision when q is near 0 or 1).
    if x.abs() < 0.95 && c >= 0.5 {
        let q = dist.cdf(x);
        if (1e-6..1.0 - 1e-6).contains(&q) {
            let recovered = dist.ppf(q);
            assert!(
                (recovered - x).abs() < 1e-6,
                "ppf(cdf({x}; c={c})) = {recovered}, want {x}"
            );
        }
    }

    // 7. pdf symmetric around 0 (RDist is symmetric beta on [-1, 1]).
    if x.abs() < 1.0 {
        let p_x = dist.pdf(x);
        let p_neg = dist.pdf(-x);
        let scale = p_x.abs().max(1.0);
        assert!(
            (p_x - p_neg).abs() <= 1e-9 * scale,
            "pdf symmetry: pdf({x}; c={c}) = {p_x} vs pdf({}; c={c}) = {p_neg}",
            -x
        );
    }
});
