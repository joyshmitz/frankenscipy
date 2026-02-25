//! Differential oracle, metamorphic relation, and adversarial tests
//! for fsci-integrate validation functions (bd-3jh.12.6).
//!
//! Oracle values are hand-computed or derived from the fixture file
//! FSCI-P2C-001_validate_tol.json.  Every differential test verifies
//! that the Rust implementation matches the hand-computed oracle within
//! tolerance.

use fsci_integrate::{
    IntegrateValidationError, MIN_RTOL, ToleranceValue, ToleranceWarning, validate_first_step,
    validate_max_step, validate_tol,
};
use fsci_runtime::RuntimeMode;

// ═══════════════════════════════════════════════════════════════════
// §1  Differential Oracle Tests (18)
// ═══════════════════════════════════════════════════════════════════

// -- Fixture case 1: scalar_rtol_clamp --
// rtol=1e-30 below MIN_RTOL → clamped to MIN_RTOL with warning
#[test]
fn diff_fixture_scalar_rtol_clamp() {
    let result = validate_tol(
        ToleranceValue::Scalar(1e-30),
        ToleranceValue::Scalar(1e-8),
        3,
        RuntimeMode::Strict,
    )
    .expect("should succeed with clamping");

    // Oracle: rtol clamped to MIN_RTOL = 100 * f64::EPSILON = 2.220446049250313e-14
    assert_eq!(result.rtol, ToleranceValue::Scalar(MIN_RTOL));
    assert_eq!(result.atol, ToleranceValue::Scalar(1e-8));
    assert_eq!(
        result.warnings,
        vec![ToleranceWarning::RtolClamped { minimum: MIN_RTOL }]
    );
}

// -- Fixture case 2: vector_passthrough --
// Valid vector tolerances in Hardened mode → pass through unchanged
#[test]
fn diff_fixture_vector_passthrough() {
    let result = validate_tol(
        ToleranceValue::Vector(vec![1e-6, 2e-6, 3e-6]),
        ToleranceValue::Vector(vec![1e-9, 2e-9, 3e-9]),
        3,
        RuntimeMode::Hardened,
    )
    .expect("should succeed");

    // Oracle: everything passes through unchanged, no warnings
    assert_eq!(result.rtol, ToleranceValue::Vector(vec![1e-6, 2e-6, 3e-6]));
    assert_eq!(result.atol, ToleranceValue::Vector(vec![1e-9, 2e-9, 3e-9]));
    assert!(result.warnings.is_empty());
    assert_eq!(result.mode, RuntimeMode::Hardened);
}

// -- Fixture case 3: wrong_shape --
// Vector atol length 2 but n=3 → AtolWrongShape error
#[test]
fn diff_fixture_wrong_shape() {
    let err = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Vector(vec![1e-9, 2e-9]),
        3,
        RuntimeMode::Strict,
    )
    .expect_err("should fail with wrong shape");

    // Oracle: error message "`atol` has wrong shape."
    assert_eq!(
        err,
        IntegrateValidationError::AtolWrongShape {
            expected: 3,
            actual: 2
        }
    );
    assert_eq!(err.to_string(), "`atol` has wrong shape.");
}

// -- Fixture case 4: negative_atol --
// atol vector with negative element → AtolMustBePositive error
#[test]
fn diff_fixture_negative_atol() {
    let err = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Vector(vec![1e-9, -2e-9, 3e-9]),
        3,
        RuntimeMode::Hardened,
    )
    .expect_err("should fail with negative atol");

    // Oracle: error message "`atol` must be positive."
    assert_eq!(err, IntegrateValidationError::AtolMustBePositive);
    assert_eq!(err.to_string(), "`atol` must be positive.");
}

// -- Hand-computed oracle: rtol exactly at MIN_RTOL boundary --
#[test]
fn diff_rtol_exact_min() {
    let result = validate_tol(
        ToleranceValue::Scalar(MIN_RTOL),
        ToleranceValue::Scalar(1e-6),
        1,
        RuntimeMode::Strict,
    )
    .expect("MIN_RTOL is valid");

    // Oracle: MIN_RTOL is NOT < MIN_RTOL so no clamping, no warning
    assert!(result.warnings.is_empty());
    assert_eq!(result.rtol, ToleranceValue::Scalar(MIN_RTOL));
}

// -- Hand-computed oracle: rtol just below MIN_RTOL --
#[test]
fn diff_rtol_just_below_min() {
    let just_below = MIN_RTOL * (1.0 - f64::EPSILON);
    let result = validate_tol(
        ToleranceValue::Scalar(just_below),
        ToleranceValue::Scalar(1e-6),
        1,
        RuntimeMode::Strict,
    )
    .expect("should clamp");

    // Oracle: just_below < MIN_RTOL → clamped to MIN_RTOL
    assert_eq!(result.rtol, ToleranceValue::Scalar(MIN_RTOL));
    assert!(!result.warnings.is_empty());
}

// -- Hand-computed oracle: validate_first_step within bounds --
#[test]
fn diff_first_step_within_bounds() {
    // Oracle: 0 < 0.5 <= |10.0 - 0.0| = 10.0 → Ok(0.5)
    let result = validate_first_step(0.5, 0.0, 10.0);
    assert_eq!(result.unwrap(), 0.5);
}

// -- Hand-computed oracle: validate_first_step at boundary --
#[test]
fn diff_first_step_at_boundary() {
    // Oracle: 0 < 10.0 <= |10.0 - 0.0| = 10.0 → Ok(10.0)
    let result = validate_first_step(10.0, 0.0, 10.0);
    assert_eq!(result.unwrap(), 10.0);
}

// -- Hand-computed oracle: validate_first_step exceeds bounds --
#[test]
fn diff_first_step_exceeds() {
    // Oracle: 10.1 > |10.0 - 0.0| = 10.0 → Err(FirstStepExceedsBounds)
    let err = validate_first_step(10.1, 0.0, 10.0).expect_err("exceeds bounds");
    assert_eq!(err, IntegrateValidationError::FirstStepExceedsBounds);
}

// -- Hand-computed oracle: validate_first_step zero --
#[test]
fn diff_first_step_zero() {
    // Oracle: 0.0 <= 0.0 → Err(FirstStepMustBePositive)
    let err = validate_first_step(0.0, 0.0, 10.0).expect_err("zero step");
    assert_eq!(err, IntegrateValidationError::FirstStepMustBePositive);
}

// -- Hand-computed oracle: validate_first_step negative --
#[test]
fn diff_first_step_negative() {
    // Oracle: -1.0 <= 0.0 → Err(FirstStepMustBePositive)
    let err = validate_first_step(-1.0, 0.0, 10.0).expect_err("negative step");
    assert_eq!(err, IntegrateValidationError::FirstStepMustBePositive);
}

// -- Hand-computed oracle: validate_first_step backward integration --
#[test]
fn diff_first_step_backward() {
    // Oracle: 0 < 5.0 <= |0.0 - 10.0| = 10.0 → Ok(5.0)
    let result = validate_first_step(5.0, 10.0, 0.0);
    assert_eq!(result.unwrap(), 5.0);
}

// -- Hand-computed oracle: validate_max_step positive --
#[test]
fn diff_max_step_positive() {
    // Oracle: 1.0 > 0.0 → Ok(1.0)
    assert_eq!(validate_max_step(1.0).unwrap(), 1.0);
}

// -- Hand-computed oracle: validate_max_step infinity --
#[test]
fn diff_max_step_infinity() {
    // Oracle: INFINITY > 0.0 → Ok(INFINITY)
    assert_eq!(validate_max_step(f64::INFINITY).unwrap(), f64::INFINITY);
}

// -- Hand-computed oracle: validate_max_step zero --
#[test]
fn diff_max_step_zero() {
    // Oracle: 0.0 <= 0.0 → Err(MaxStepMustBePositive)
    let err = validate_max_step(0.0).expect_err("zero");
    assert_eq!(err, IntegrateValidationError::MaxStepMustBePositive);
}

// -- Hand-computed oracle: validate_max_step negative --
#[test]
fn diff_max_step_negative() {
    // Oracle: -5.0 <= 0.0 → Err(MaxStepMustBePositive)
    let err = validate_max_step(-5.0).expect_err("negative");
    assert_eq!(err, IntegrateValidationError::MaxStepMustBePositive);
}

// -- Hand-computed oracle: vector rtol with partial clamping --
#[test]
fn diff_vector_rtol_partial_clamp() {
    let result = validate_tol(
        ToleranceValue::Vector(vec![1e-3, 1e-30, 1e-6]),
        ToleranceValue::Scalar(1e-8),
        3,
        RuntimeMode::Strict,
    )
    .expect("should succeed with partial clamping");

    // Oracle: element 1 (1e-30) < MIN_RTOL, gets clamped to MIN_RTOL
    // Elements 0 and 2 are above MIN_RTOL so pass through
    // Warning is emitted because needs_clamp triggers on any(|x| x < MIN_RTOL)
    match &result.rtol {
        ToleranceValue::Vector(v) => {
            assert_eq!(v[0], 1e-3);
            assert_eq!(v[1], MIN_RTOL);
            assert_eq!(v[2], 1e-6);
        }
        _ => panic!("expected vector"),
    }
    assert!(!result.warnings.is_empty());
}

// -- Hand-computed oracle: scalar atol with n=0 (empty system) --
#[test]
fn diff_empty_system() {
    // Oracle: scalar atol has no shape check, n=0 is valid → Ok
    let result = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Scalar(1e-6),
        0,
        RuntimeMode::Strict,
    )
    .expect("empty system should succeed");
    assert!(result.warnings.is_empty());
}

// ═══════════════════════════════════════════════════════════════════
// §2  Metamorphic Relation Tests (8)
// ═══════════════════════════════════════════════════════════════════

// MR1: validate_tol(rtol, atol * c, n) scales atol by c in result
#[test]
fn meta_atol_scaling() {
    let c = 3.0;
    let atol_base = vec![1e-8, 2e-8, 3e-8];
    let atol_scaled: Vec<f64> = atol_base.iter().map(|x| x * c).collect();

    let r1 = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Vector(atol_base.clone()),
        3,
        RuntimeMode::Strict,
    )
    .expect("base should succeed");

    let r2 = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Vector(atol_scaled),
        3,
        RuntimeMode::Strict,
    )
    .expect("scaled should succeed");

    // Metamorphic: r2.atol == r1.atol * c
    match (&r1.atol, &r2.atol) {
        (ToleranceValue::Vector(v1), ToleranceValue::Vector(v2)) => {
            for (a, b) in v1.iter().zip(v2.iter()) {
                assert!((b - a * c).abs() < 1e-20, "scaling not preserved");
            }
        }
        _ => panic!("expected vectors"),
    }
}

// MR2: validate_tol in Strict == validate_tol in Hardened for valid inputs
// (both modes accept/reject the same valid inputs, produce same rtol/atol)
#[test]
fn meta_strict_equals_hardened_for_valid() {
    let cases: Vec<(f64, f64)> = vec![
        (1e-3, 1e-6),
        (1e-6, 1e-9),
        (1e-30, 1e-8), // triggers clamping
    ];

    for (rtol, atol) in cases {
        let r_strict = validate_tol(
            ToleranceValue::Scalar(rtol),
            ToleranceValue::Scalar(atol),
            5,
            RuntimeMode::Strict,
        )
        .expect("strict should succeed");

        let r_hardened = validate_tol(
            ToleranceValue::Scalar(rtol),
            ToleranceValue::Scalar(atol),
            5,
            RuntimeMode::Hardened,
        )
        .expect("hardened should succeed");

        // Metamorphic: rtol and atol values are mode-independent
        assert_eq!(r_strict.rtol, r_hardened.rtol, "rtol should match");
        assert_eq!(r_strict.atol, r_hardened.atol, "atol should match");
        assert_eq!(
            r_strict.warnings.len(),
            r_hardened.warnings.len(),
            "warning count should match"
        );
    }
}

// MR3: validate_first_step(s, t0, tb) == validate_first_step(s, t0+k, tb+k)
// (translation invariance)
#[test]
fn meta_first_step_translation_invariance() {
    let offsets = [0.0, 100.0, -100.0, 1e10, -1e10];
    let step = 0.5;
    let t0 = 0.0;
    let tb = 10.0;

    let base = validate_first_step(step, t0, tb);

    for k in offsets {
        let shifted = validate_first_step(step, t0 + k, tb + k);
        assert_eq!(
            base.is_ok(),
            shifted.is_ok(),
            "translation by {k} changed result"
        );
        if let (Ok(a), Ok(b)) = (&base, &shifted) {
            assert_eq!(a, b, "translated values should match");
        }
    }
}

// MR4: validate_max_step(s) result independent of prior validate_tol calls
// (no state leakage)
#[test]
fn meta_max_step_no_state_leakage() {
    // Call validate_tol first to see if it leaks state
    let _ = validate_tol(
        ToleranceValue::Scalar(1e-30),
        ToleranceValue::Scalar(1e-8),
        3,
        RuntimeMode::Strict,
    );

    let r1 = validate_max_step(1.0);

    // Call a bunch more
    for _ in 0..100 {
        let _ = validate_tol(
            ToleranceValue::Scalar(1e-30),
            ToleranceValue::Vector(vec![1e-6; 100]),
            100,
            RuntimeMode::Hardened,
        );
    }

    let r2 = validate_max_step(1.0);
    assert_eq!(r1, r2, "state leakage detected");
}

// MR5: validate_tol idempotence — applying validate_tol to already-valid output
// produces the same result with no additional warnings
#[test]
fn meta_validate_tol_idempotence() {
    let r1 = validate_tol(
        ToleranceValue::Scalar(1e-30),
        ToleranceValue::Scalar(1e-8),
        3,
        RuntimeMode::Strict,
    )
    .expect("first pass");

    // Apply again with the output values
    let r2 = validate_tol(r1.rtol.clone(), r1.atol.clone(), 3, RuntimeMode::Strict)
        .expect("second pass");

    // Metamorphic: output is a fixpoint — no further clamping needed
    assert_eq!(r1.rtol, r2.rtol, "rtol should be stable");
    assert_eq!(r1.atol, r2.atol, "atol should be stable");
    assert!(
        r2.warnings.is_empty(),
        "second pass should produce no warnings"
    );
}

// MR6: error kind preserved under mode change
// (wrong_shape and negative_atol errors don't depend on mode)
#[test]
fn meta_error_kind_mode_independent() {
    // Wrong shape
    let err_strict = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Vector(vec![1e-9]),
        3,
        RuntimeMode::Strict,
    )
    .expect_err("wrong shape");

    let err_hardened = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Vector(vec![1e-9]),
        3,
        RuntimeMode::Hardened,
    )
    .expect_err("wrong shape");

    assert_eq!(
        err_strict, err_hardened,
        "error kind should be mode-independent"
    );

    // Negative atol
    let err_strict2 = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Scalar(-1.0),
        1,
        RuntimeMode::Strict,
    )
    .expect_err("negative atol");

    let err_hardened2 = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Scalar(-1.0),
        1,
        RuntimeMode::Hardened,
    )
    .expect_err("negative atol");

    assert_eq!(err_strict2, err_hardened2);
}

// MR7: validate_first_step symmetry — same step valid for (t0,tb) and (tb,t0)
#[test]
fn meta_first_step_direction_symmetry() {
    let step = 3.0;
    let fwd = validate_first_step(step, 0.0, 10.0);
    let bwd = validate_first_step(step, 10.0, 0.0);
    assert_eq!(
        fwd.is_ok(),
        bwd.is_ok(),
        "direction should not affect validity"
    );
    assert_eq!(fwd.unwrap(), bwd.unwrap());
}

// MR8: validate_tol monotonicity — if rtol1 < rtol2 and both need clamping,
// both produce the same clamped value
#[test]
fn meta_rtol_clamping_convergence() {
    let r1 = validate_tol(
        ToleranceValue::Scalar(1e-30),
        ToleranceValue::Scalar(1e-6),
        1,
        RuntimeMode::Strict,
    )
    .expect("r1");

    let r2 = validate_tol(
        ToleranceValue::Scalar(1e-100),
        ToleranceValue::Scalar(1e-6),
        1,
        RuntimeMode::Strict,
    )
    .expect("r2");

    // Both are below MIN_RTOL → both clamped to the same value
    assert_eq!(
        r1.rtol, r2.rtol,
        "all sub-minimum rtols converge to MIN_RTOL"
    );
}

// ═══════════════════════════════════════════════════════════════════
// §3  Adversarial Vector Tests (10)
// ═══════════════════════════════════════════════════════════════════

// ADV1: rtol = f64::MIN_POSITIVE (near zero positive)
#[test]
fn adv_rtol_min_positive() {
    let result = validate_tol(
        ToleranceValue::Scalar(f64::MIN_POSITIVE),
        ToleranceValue::Scalar(1e-6),
        1,
        RuntimeMode::Strict,
    );
    // MIN_POSITIVE ≈ 5e-324, far below MIN_RTOL → should clamp
    let validated = result.expect("should clamp, not panic");
    assert_eq!(validated.rtol, ToleranceValue::Scalar(MIN_RTOL));
    assert!(!validated.warnings.is_empty());
}

// ADV2: atol vector with one NaN element
#[test]
fn adv_atol_nan_element() {
    // NaN is not < 0 (NaN < 0 is false), so it passes the negative check
    let result = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Vector(vec![1e-8, f64::NAN, 1e-8]),
        3,
        RuntimeMode::Strict,
    );
    // NaN passes through — implementation follows SciPy's permissive behavior
    assert!(result.is_ok(), "NaN atol should not cause panic or error");
}

// ADV3: atol vector with one Inf element
#[test]
fn adv_atol_inf_element() {
    let result = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Vector(vec![1e-8, f64::INFINITY, 1e-8]),
        3,
        RuntimeMode::Strict,
    );
    // Inf is not < 0, so it's accepted (SciPy allows Inf tolerance)
    assert!(result.is_ok(), "Inf atol should be accepted");
}

// ADV4: n = usize::MAX (dimension overflow potential)
#[test]
fn adv_dimension_overflow() {
    // With scalar atol there's no length check, so this should succeed
    let result = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Scalar(1e-6),
        usize::MAX,
        RuntimeMode::Strict,
    );
    assert!(result.is_ok(), "scalar atol with huge n should not panic");
}

// ADV5: first_step = f64::EPSILON (near zero positive)
#[test]
fn adv_first_step_epsilon() {
    let result = validate_first_step(f64::EPSILON, 0.0, 1.0);
    // EPSILON > 0 and EPSILON <= 1.0 → valid
    assert_eq!(result.unwrap(), f64::EPSILON);
}

// ADV6: t_bound - t0 = f64::EPSILON (extremely tight integration interval)
#[test]
fn adv_first_step_tight_interval() {
    let t0 = 0.0;
    let tb = f64::EPSILON;
    // Step exactly at boundary
    let result = validate_first_step(f64::EPSILON, t0, tb);
    assert!(result.is_ok(), "step == interval should be accepted");

    // Step exceeding the tiny interval
    let result2 = validate_first_step(2.0 * f64::EPSILON, t0, tb);
    assert!(result2.is_err(), "step > interval should be rejected");
}

// ADV7: NaN inputs to validate_first_step
#[test]
fn adv_first_step_nan() {
    // NaN <= 0.0 is false (NaN comparisons always false), so NaN passes
    // the guard and the bounds check (NaN > 1.0 is also false).
    // This matches SciPy's permissive NaN handling.
    let result = validate_first_step(f64::NAN, 0.0, 1.0);
    assert!(
        result.is_ok(),
        "NaN step passes guards due to IEEE 754 semantics"
    );
    assert!(result.unwrap().is_nan());
}

// ADV8: Inf inputs to validate_max_step
#[test]
fn adv_max_step_inf() {
    // SciPy uses INFINITY as default max_step
    let result = validate_max_step(f64::INFINITY);
    assert!(result.is_ok(), "Inf max_step should be accepted");
}

// ADV9: NaN to validate_max_step
#[test]
fn adv_max_step_nan() {
    // NaN <= 0.0 is false, so NaN passes the guard.
    // Matches SciPy's permissive NaN handling.
    let result = validate_max_step(f64::NAN);
    assert!(
        result.is_ok(),
        "NaN max_step passes guard due to IEEE 754 semantics"
    );
    assert!(result.unwrap().is_nan());
}

// ADV10: negative Inf to validate_max_step
#[test]
fn adv_max_step_neg_inf() {
    let result = validate_max_step(f64::NEG_INFINITY);
    assert!(result.is_err(), "negative Inf max_step should be rejected");
}
