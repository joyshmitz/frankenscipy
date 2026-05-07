#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, SkewNorm};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-eb2vm]:
// SkewNorm has 4 anchor tests + scipy diff but no fuzz coverage.
// Drive Arbitrary-derived (a, x) and assert:
//   1. Constructor never panics for sanitized finite a.
//   2. pdf is finite, non-negative, and ≤ 1 (loose bound:
//      max{2·φ(0)·Φ(0)} = 0.7979).
//   3. cdf is monotone non-decreasing along ascending x grid.
//   4. cdf ∈ [0, 1] for finite x.
//   5. ppf round-trip on interior (skip near saturation).
//   6. Far tails: cdf(-1e6) ≈ 0, cdf(1e6) ≈ 1.

const A_BOUND: f64 = 50.0;
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
    seed.clamp(-A_BOUND, A_BOUND)
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

    // 1. Constructor never panics for sanitized finite a.
    let dist = SkewNorm::new(a);

    // 2. pdf finite, non-negative, ≤ 1.
    let p = dist.pdf(x);
    assert!(p.is_finite(), "pdf({x}; a={a}) must be finite, got {p}");
    assert!(p >= 0.0, "pdf must be non-negative, got {p}");
    assert!(
        p <= 1.0,
        "pdf({x}; a={a}) = {p} must be ≤ 1 (max ~0.798)"
    );

    // 3-4. cdf monotone non-decreasing on a fixed grid + ∈ [0, 1].
    let mut prev = 0.0;
    for &xi in &[-1.0e6_f64, -100.0, -1.0, 0.0, 1.0, 100.0, 1.0e6] {
        let c = dist.cdf(xi);
        assert!(c.is_finite(), "cdf({xi}; a={a}) must be finite, got {c}");
        assert!(
            (0.0..=1.0).contains(&c),
            "cdf({xi}; a={a}) = {c} must be in [0, 1]"
        );
        assert!(
            c >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({xi}; a={a}) = {c} < prev = {prev}"
        );
        prev = c;
    }

    // 5. ppf round-trip on interior — skip when |x| > 5 or |a| > 5
    // (Owen's T 10-pt Gauss-Legendre quadrature loses precision in
    // the deep tail).
    if x.abs() < 5.0 && a.abs() < 5.0 {
        let q = dist.cdf(x);
        if (1.0e-6..1.0 - 1.0e-6).contains(&q) {
            let recovered = dist.ppf(q);
            let scale = x.abs().max(1.0);
            assert!(
                (recovered - x).abs() < 1.0e-4 * scale,
                "ppf(cdf({x}; a={a})) = {recovered}, want {x}"
            );
        }
    }

    // 6. Far tails — only require approximate saturation since
    // the X_BOUND clamp + scaling can leave residual fp slop.
    let c_low = dist.cdf(-X_BOUND);
    let c_high = dist.cdf(X_BOUND);
    assert!(
        c_low < 1.0e-6,
        "cdf(-{X_BOUND}; a={a}) = {c_low} must be ≈ 0"
    );
    assert!(
        c_high > 1.0 - 1.0e-6,
        "cdf({X_BOUND}; a={a}) = {c_high} must be ≈ 1"
    );
});
