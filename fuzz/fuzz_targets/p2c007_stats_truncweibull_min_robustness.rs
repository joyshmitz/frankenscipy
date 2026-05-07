#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, TruncWeibullMin};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-kin4s]:
// TruncWeibullMin has 3 anchor cases + scipy diff but no fuzz
// coverage. Drive Arbitrary-derived (c, a, b, x) and assert:
//   1. Constructor never panics for sanitized
//      c ∈ [0.1, 10], a ∈ [0, 10], b ∈ (a, a + 50].
//   2. pdf finite, non-negative on (a, b].
//   3. pdf = 0 strictly outside (a, b].
//   4. cdf monotone non-decreasing along ascending grid.
//   5. cdf(≤ a) = 0, cdf(≥ b) = 1.
//   6. ppf(cdf(x)) round-trip on interior.

const C_MIN: f64 = 0.1;
const C_MAX: f64 = 10.0;
const A_MAX: f64 = 10.0;
const B_SPAN_MAX: f64 = 50.0;

#[derive(Debug, Arbitrary)]
struct Input {
    c_seed: f64,
    a_seed: f64,
    span_seed: f64,
    x_seed: f64,
}

fn sanitize_c(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(C_MIN, C_MAX)
}

fn sanitize_a(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 0.0;
    }
    seed.abs().min(A_MAX)
}

fn sanitize_span(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(0.1, B_SPAN_MAX)
}

fuzz_target!(|input: Input| {
    let c = sanitize_c(input.c_seed);
    let a = sanitize_a(input.a_seed);
    let span = sanitize_span(input.span_seed);
    let b = a + span;

    // 1. Constructor never panics within validated range.
    let dist = TruncWeibullMin::new(c, a, b);

    // Sanitize x into (a, b].
    let x = if input.x_seed.is_finite() {
        a + input.x_seed.abs().fract() * (b - a) + 1e-12
    } else {
        a + 0.5 * (b - a)
    };
    let x = x.min(b);

    // 2. pdf finite, non-negative on (a, b].
    let p = dist.pdf(x);
    assert!(p.is_finite(), "pdf({x}; c={c}, a={a}, b={b}) must be finite, got {p}");
    assert!(p >= 0.0, "pdf must be non-negative, got {p}");

    // 3. pdf = 0 strictly outside (a, b].
    assert_eq!(
        dist.pdf(a - 0.1_f64.max(0.0).min(a)),
        0.0,
        "pdf below a must be 0"
    );
    if a > 0.0 {
        assert_eq!(dist.pdf(0.0), 0.0, "pdf(0) must be 0 when a > 0");
    }
    assert_eq!(dist.pdf(b + 0.5), 0.0, "pdf above b must be 0");

    // 4-5. cdf monotone non-decreasing + boundary values.
    assert_eq!(dist.cdf(a - 0.1), 0.0);
    assert_eq!(dist.cdf(b + 0.5), 1.0);
    let mut prev = 0.0;
    for k in 0..=10 {
        let xi = a + (k as f64) * (b - a) / 10.0;
        let cv = dist.cdf(xi);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({xi}; c={c}, a={a}, b={b}) = {cv} must be in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({xi}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 6. ppf(cdf(x)) round-trip on interior.
    let q = dist.cdf(x);
    if (1e-8..1.0 - 1e-8).contains(&q) {
        let recovered = dist.ppf(q);
        assert!(
            (recovered - x).abs() < 1e-9 * (b - a),
            "ppf(cdf({x}; c={c}, a={a}, b={b})) = {recovered}, want {x}"
        );
    }
});
