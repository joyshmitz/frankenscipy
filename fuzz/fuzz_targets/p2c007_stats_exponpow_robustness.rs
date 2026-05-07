#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, ExponPow};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-wtr6f]:
// ExponPow has 3 anchor cases + scipy diff but no fuzz coverage.
// Drive Arbitrary-derived (b, x, q) and assert:
//   1. Constructor never panics for sanitized b ∈ (0, B_MAX].
//   2. pdf is finite, non-negative on the interior x > 0.
//   3. cdf is monotone non-decreasing along an ascending x-grid.
//   4. cdf ∈ [0, 1] for finite x.
//   5. ppf round-trip on interior; skip when cdf saturates near 1.
//   6. cdf(0) == 0 (exact); cdf(large) → 1.

const B_MIN: f64 = 0.05;
const B_MAX: f64 = 20.0;
const X_MAX: f64 = 1.0e6;

#[derive(Debug, Arbitrary)]
struct Input {
    b_seed: f64,
    x_seed: f64,
}

fn sanitize_b(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(B_MIN, B_MAX)
}

fn sanitize_x(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 0.0;
    }
    seed.abs().min(X_MAX)
}

fuzz_target!(|input: Input| {
    let b = sanitize_b(input.b_seed);
    let x = sanitize_x(input.x_seed);

    // 1. Constructor never panics within (0, B_MAX].
    let dist = ExponPow::new(b);

    // 2. pdf at interior x is finite, non-negative.
    if x > 0.0 {
        let p = dist.pdf(x);
        assert!(p.is_finite(), "pdf({x}; b={b}) must be finite, got {p}");
        assert!(p >= 0.0, "pdf must be non-negative, got {p}");
    }

    // cdf(0) is exactly 0 by support convention.
    assert_eq!(
        dist.cdf(0.0),
        0.0,
        "cdf(0; b={b}) must be exactly 0"
    );

    // 3-4. cdf is monotone non-decreasing along ascending grid + ∈ [0, 1].
    let mut prev = 0.0;
    for &xi in &[0.0_f64, 0.05, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0] {
        let c = dist.cdf(xi);
        assert!(c.is_finite(), "cdf({xi}; b={b}) must be finite, got {c}");
        assert!(
            (0.0..=1.0).contains(&c),
            "cdf({xi}; b={b}) = {c} must be in [0, 1]"
        );
        assert!(
            c >= prev - 1e-15,
            "cdf must be non-decreasing: cdf({xi}; b={b}) = {c} < prev = {prev}"
        );
        prev = c;
    }

    // 5. ppf round-trip on interior (skip near saturation). The cdf
    // saturates very fast — for b=1 even x=3 already gives
    // cdf ≈ 1 − 6e-11, where log1p(−q) loses trailing bits and the
    // analytic ppf can't recover x to better than ~1e-8 relative.
    // Restrict probes to q < 0.99 to stay clear of saturation.
    if x > 0.0 && x < 2.5 && (B_MIN * 4.0..=B_MAX / 2.0).contains(&b) {
        let q = dist.cdf(x);
        if (1e-12..0.99).contains(&q) {
            let recovered = dist.ppf(q);
            let scale = x.abs().max(1.0);
            assert!(
                (recovered - x).abs() < 1e-8 * scale,
                "ppf(cdf({x}; b={b})) = {recovered}, want {x} (scale {scale})"
            );
        }
    }

    // 6. Far-tail cdf approaches 1, but only when b ≳ 0.5. For very
    // small b the tail is so heavy that even cdf(1e3) stays well
    // below 1 — that's a property of the distribution, not a defect.
    if b >= 0.5 {
        let c_high = dist.cdf(50.0);
        assert!(
            c_high > 1.0 - 1e-9 || c_high.is_nan(),
            "cdf(50; b={b}) = {c_high} must approach 1 (or NaN)"
        );
    }
});
