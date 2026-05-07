#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, TruncPareto};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-rpdre]:
// TruncPareto has 3 anchor cases + scipy diff but no fuzz coverage.
// Drive Arbitrary-derived (b, c, x, q) and assert:
//   1. Constructor never panics for sanitized b ∈ [-10, 10] \ {0}
//      and c ∈ (1.001, 1000].
//   2. pdf finite, non-negative on the support [1, c].
//   3. pdf = 0 strictly outside support.
//   4. cdf monotone non-decreasing along ascending x.
//   5. cdf(1) = 0 and cdf(c) = 1 (exactly).
//   6. ppf(cdf(x)) round-trip on the interior.

const B_BOUND: f64 = 10.0;
const C_MIN: f64 = 1.001;
const C_MAX: f64 = 1000.0;

#[derive(Debug, Arbitrary)]
struct Input {
    b_seed: f64,
    c_seed: f64,
    x_seed: f64,
}

fn sanitize_b(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    let b = seed.clamp(-B_BOUND, B_BOUND);
    if b.abs() < 0.05 {
        // Nudge away from 0 to honour the b ≠ 0 constraint.
        if b >= 0.0 { 0.05 } else { -0.05 }
    } else {
        b
    }
}

fn sanitize_c(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 5.0;
    }
    seed.abs().clamp(C_MIN, C_MAX)
}

fn sanitize_x_in_support(seed: f64, c: f64) -> f64 {
    if !seed.is_finite() {
        return (1.0 + c) * 0.5;
    }
    // Map onto [1, c].
    let u = seed.abs().fract();
    1.0 + u * (c - 1.0)
}

fuzz_target!(|input: Input| {
    let b = sanitize_b(input.b_seed);
    let c = sanitize_c(input.c_seed);
    let x = sanitize_x_in_support(input.x_seed, c);

    // 1. Constructor never panics within (-10, 10) \ {0}, c ∈ (1.001, 1000].
    let dist = TruncPareto::new(b, c);

    // 2. pdf finite, non-negative on [1, c].
    let p = dist.pdf(x);
    assert!(p.is_finite(), "pdf({x}; b={b}, c={c}) must be finite, got {p}");
    assert!(p >= 0.0, "pdf must be non-negative, got {p}");

    // 3. pdf = 0 strictly outside support.
    assert_eq!(dist.pdf(0.5), 0.0, "pdf(<1) must be 0");
    assert_eq!(dist.pdf(c + 1.0), 0.0, "pdf(>c) must be 0");

    // 4-5. cdf monotone non-decreasing + cdf(1)=0 / cdf(c)=1.
    assert_eq!(dist.cdf(0.5), 0.0);
    assert_eq!(dist.cdf(0.999), 0.0);
    assert_eq!(
        dist.cdf(c + 1.0),
        1.0,
        "cdf above c must be 1"
    );
    let cdf_1 = dist.cdf(1.0);
    assert!(
        cdf_1 == 0.0 || cdf_1.abs() < 1e-15,
        "cdf(1) must be 0 (got {cdf_1})"
    );
    let mut prev = 0.0;
    for k in 0..=10 {
        let xi = 1.0 + (k as f64) * (c - 1.0) / 10.0;
        let cv = dist.cdf(xi);
        assert!(cv.is_finite() && (0.0..=1.0).contains(&cv));
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({xi}; b={b}, c={c}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 6. ppf(cdf(x)) round-trip on interior. Skip near boundaries
    // where the closed-form inverse can lose precision via x^(-1/b)
    // for tiny b·log(x).
    let q = dist.cdf(x);
    if (1e-6..1.0 - 1e-6).contains(&q) && b.abs() > 0.1 {
        let recovered = dist.ppf(q);
        let scale = x.abs().max(1.0);
        assert!(
            (recovered - x).abs() < 1e-8 * scale,
            "ppf(cdf({x}; b={b}, c={c})) = {recovered}, want {x}"
        );
    }
});
