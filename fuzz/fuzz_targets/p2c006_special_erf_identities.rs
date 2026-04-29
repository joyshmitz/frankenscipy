#![no_main]

//! Identity fuzzing for the real-valued error functions in `fsci-special`.
//!
//! These metamorphic relations are oracle-free: for any real `x` in the
//! finite domain, the underlying analytic functions satisfy
//!
//!   1. `erf(-x) = -erf(x)`             (odd symmetry)
//!   2. `erfc(-x) = 2 - erfc(x)`        (complement symmetry)
//!   3. `erf(x) + erfc(x) = 1`          (partition of unity)
//!   4. `erfinv(erf(x)) ≈ x`            (round-trip on the open domain)
//!
//! libFuzzer drives `f64` inputs that are sanitized to a representative
//! domain. The harness only fails when an identity is broken to a degree
//! beyond what conservative single-precision-round-trip tolerances would
//! ever allow.
//!
//! Bead: `frankenscipy-9r3w`.

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::{erf_scalar, erfc_scalar, erfinv_scalar};
use libfuzzer_sys::fuzz_target;

/// Tolerances chosen to be a few orders of magnitude looser than the
/// double-precision noise floor for these series + asymptotic
/// implementations, so round-off doesn't cause spurious crashes.
const ABS_TOL: f64 = 1.0e-10;
const REL_TOL: f64 = 1.0e-9;

/// Domain clamp. Erf saturates outside ±5.5: `erf(5.5) > 1 - 2e-14`.
/// We pick a slightly tighter bound so the inverse round-trip stays
/// numerically stable.
const X_MAX: f64 = 5.0;

#[derive(Debug, Arbitrary)]
struct ErfInput {
    x_raw: f64,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-X_MAX, X_MAX)
    } else {
        // Map non-finite input to a deterministic point in-domain.
        0.0
    }
}

fn approx_eq(actual: f64, expected: f64, label: &str, x: f64) {
    let diff = (actual - expected).abs();
    let scale = expected.abs().max(actual.abs()).max(1.0);
    let bound = ABS_TOL + REL_TOL * scale;
    if diff > bound {
        panic!(
            "{label}: x={x} actual={actual:.16e} expected={expected:.16e} diff={diff:.3e} bound={bound:.3e}"
        );
    }
}

fuzz_target!(|input: ErfInput| {
    let x = sanitize(input.x_raw);

    // Reference values.
    let e_pos = erf_scalar(x);
    let e_neg = erf_scalar(-x);
    let ec_pos = erfc_scalar(x);
    let ec_neg = erfc_scalar(-x);

    // All outputs must be finite for in-domain inputs.
    assert!(
        e_pos.is_finite() && e_neg.is_finite() && ec_pos.is_finite() && ec_neg.is_finite(),
        "non-finite scalar erf/erfc at x={x}: e+={e_pos} e-={e_neg} ec+={ec_pos} ec-={ec_neg}"
    );

    // Identity 1: erf is odd.
    approx_eq(e_neg, -e_pos, "MR1 erf-odd", x);

    // Identity 2: erfc complement symmetry.
    approx_eq(ec_neg, 2.0 - ec_pos, "MR2 erfc-complement", x);

    // Identity 3: erf + erfc = 1.
    approx_eq(e_pos + ec_pos, 1.0, "MR3 erf+erfc", x);
    approx_eq(e_neg + ec_neg, 1.0, "MR3 erf+erfc (neg)", x);

    // Identity 4: erfinv is a left inverse of erf on the open domain.
    // Round-trip is unstable as |erf(x)| → 1; restrict to where the
    // inverse can be evaluated without saturation noise dominating the
    // result.
    if x.abs() < 4.0 && e_pos.abs() < 1.0 - 1.0e-14 {
        let x_back = erfinv_scalar(e_pos, RuntimeMode::Strict)
            .expect("erfinv on a value strictly inside (-1, 1) must succeed");
        // Erfinv near 1 has steep slope, so allow a slightly looser
        // round-trip tolerance scaled by 1/(1-|y|).
        let slope_factor = 1.0 / (1.0 - e_pos.abs()).max(1e-12);
        let bound = (ABS_TOL + REL_TOL * x.abs().max(1.0)) * slope_factor.min(1e6);
        let diff = (x_back - x).abs();
        if diff > bound {
            panic!(
                "MR4 erfinv-roundtrip: x={x} y=erf(x)={e_pos:.16e} erfinv(y)={x_back:.16e} diff={diff:.3e} bound={bound:.3e}"
            );
        }
    }
});
