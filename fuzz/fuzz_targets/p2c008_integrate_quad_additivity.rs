#![no_main]

use arbitrary::Arbitrary;
use fsci_integrate::{QuadOptions, quad};
use libfuzzer_sys::fuzz_target;

// Quadrature additivity metamorphic oracle per br-66c2:
// for any continuous integrand f and bounds a <= b <= c,
//
//   ∫_a^c f(x) dx  =  ∫_a^b f(x) dx  +  ∫_b^c f(x) dx
//
// Any adaptive GK quadrature must obey this exactly up to the
// estimated error bounds. A regression that corrupts one bracket
// (off-by-one in the recursion stack, integer overflow in neval,
// stale epsabs handling) will violate this even if the individual
// integrals look plausible in isolation.
//
// Integrand is the bounded smooth function
//   f(x) = x * x + sin(x)
// — analytic on all of R, no singularities, bounded derivatives.
// Bounds are clamped to [-10, 10] so even the quartic antiderivative
// stays well inside f64 range.

const BOUND_LIMIT: f64 = 10.0;
const INTEGRAND_ABS_MAX: f64 = 101.0; // |x^2 + sin(x)| <= 100 + 1 on [-10, 10]
const ABS_TOL: f64 = 1.0e-8;
const REL_TOL: f64 = 1.0e-6;

#[derive(Debug, Arbitrary)]
struct AdditivityInput {
    a: f64,
    b: f64,
    c: f64,
}

fn sanitize(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-BOUND_LIMIT, BOUND_LIMIT)
    } else {
        0.0
    }
}

fn integrand(x: f64) -> f64 {
    x * x + x.sin()
}

fuzz_target!(|input: AdditivityInput| {
    let mut bounds = [
        sanitize(input.a),
        sanitize(input.b),
        sanitize(input.c),
    ];
    bounds.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).expect("sanitized bounds are non-NaN"));
    let [a, b, c] = bounds;

    // Skip when any sub-interval collapses — quad returns 0 exactly
    // in that case which makes the additivity check degenerate.
    if (b - a).abs() < f64::EPSILON || (c - b).abs() < f64::EPSILON {
        return;
    }

    let options = QuadOptions::default();
    let Ok(full) = quad(integrand, a, c, options) else {
        return;
    };
    let Ok(left) = quad(integrand, a, b, options) else {
        return;
    };
    let Ok(right) = quad(integrand, b, c, options) else {
        return;
    };

    if !full.integral.is_finite() || !left.integral.is_finite() || !right.integral.is_finite() {
        return;
    }

    let split_sum = left.integral + right.integral;
    let diff = (split_sum - full.integral).abs();

    // Tolerance budget: each quad call carries its own estimated
    // absolute error; additivity must hold at least within the sum
    // of those errors plus one ULP-level floor.
    let reported_budget = full.error.abs() + left.error.abs() + right.error.abs();
    let envelope = ABS_TOL + REL_TOL * full.integral.abs().max(INTEGRAND_ABS_MAX * (c - a).abs());
    let threshold = envelope + reported_budget;

    assert!(
        diff <= threshold,
        "quad additivity violated: \
         ∫[{a}, {c}] = {} but ∫[{a}, {b}] + ∫[{b}, {c}] = {} ({} + {}); \
         diff={diff:e}, threshold={threshold:e}, \
         reported_budget={reported_budget:e}",
        full.integral,
        split_sum,
        left.integral,
        right.integral,
    );
});
