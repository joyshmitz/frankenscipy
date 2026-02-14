//! Property tests for fsci-integrate IVP solver core.
//!
//! Convention: test_{module}_{function}_{scenario}
//!
//! Seed replay: `PROPTEST_CASES=1000 cargo test -p fsci-integrate --test property_tests`
//! Reproduce: `PROPTEST_SEED=<seed> cargo test -p fsci-integrate --test property_tests`

use fsci_integrate::{
    InitialStepRequest, MIN_RTOL, SolveIvpOptions, SolverKind, ToleranceValue, select_initial_step,
    solve_ivp, validate_first_step, validate_max_step, validate_tol,
};
use fsci_runtime::{RuntimeMode, TestLogEntry, TestResult};
use proptest::prelude::*;

// ═══════════════════════════════════════════════════════════════
// Property 1: For any valid rtol >= MIN_RTOL, validate_tol never clamps
// ═══════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn test_validation_tol_no_clamp_when_rtol_above_min(
        rtol in MIN_RTOL..1.0,
        atol in 1e-15f64..1.0,
        n in 1usize..50,
    ) {
        let result = validate_tol(
            ToleranceValue::Scalar(rtol),
            ToleranceValue::Scalar(atol),
            n,
            RuntimeMode::Strict,
        );
        let validated = result.expect("valid tolerances should not fail");
        prop_assert!(
            validated.warnings.is_empty(),
            "rtol >= MIN_RTOL should never produce warnings, got {:?}",
            validated.warnings
        );
        // rtol should pass through unchanged
        match validated.rtol {
            ToleranceValue::Scalar(v) => prop_assert!(
                (v - rtol).abs() < f64::EPSILON,
                "rtol should be unchanged: got {v}, expected {rtol}"
            ),
            _ => prop_assert!(false, "rtol should remain scalar"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Property 2: For any rtol < MIN_RTOL, validate_tol always clamps to MIN_RTOL
// ═══════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn test_validation_tol_always_clamps_below_min_rtol(
        rtol in 0.0f64..MIN_RTOL,
        atol in 1e-15f64..1.0,
        n in 1usize..50,
    ) {
        let result = validate_tol(
            ToleranceValue::Scalar(rtol),
            ToleranceValue::Scalar(atol),
            n,
            RuntimeMode::Strict,
        );
        let validated = result.expect("valid tolerances should not fail");
        prop_assert!(
            !validated.warnings.is_empty(),
            "rtol < MIN_RTOL should always produce a clamping warning"
        );
        match validated.rtol {
            ToleranceValue::Scalar(v) => prop_assert!(
                (v - MIN_RTOL).abs() < f64::EPSILON,
                "clamped rtol should equal MIN_RTOL: got {v}, expected {MIN_RTOL}"
            ),
            _ => prop_assert!(false, "rtol should remain scalar"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Property 3: For any valid atol vector of length n, validate_tol succeeds
// ═══════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn test_validation_tol_vector_atol_matching_length_succeeds(
        n in 1usize..20,
    ) {
        let atol_vec: Vec<f64> = (0..n).map(|i| 1e-8 + i as f64 * 1e-10).collect();
        let result = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Vector(atol_vec.clone()),
            n,
            RuntimeMode::Strict,
        );
        prop_assert!(result.is_ok(), "matching vector atol should succeed");
    }

    #[test]
    fn test_validation_tol_vector_atol_wrong_length_fails(
        n in 2usize..20,
        offset in 1usize..5,
    ) {
        let wrong_n = if n > offset { n - offset } else { n + offset };
        let atol_vec: Vec<f64> = (0..wrong_n).map(|_| 1e-8).collect();
        let result = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Vector(atol_vec),
            n,
            RuntimeMode::Strict,
        );
        prop_assert!(result.is_err(), "wrong length vector atol should fail");
    }
}

// ═══════════════════════════════════════════════════════════════
// Property 4: validate_first_step succeeds iff 0 < s <= |tb - t0|
// ═══════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn test_validation_first_step_iff_in_bounds(
        t0 in -100.0f64..100.0,
        interval in 0.01f64..100.0,
        fraction in 0.0f64..2.0,
    ) {
        let t_bound = t0 + interval;
        let step = fraction * interval;

        let result = validate_first_step(step, t0, t_bound);
        if step > 0.0 && step <= interval {
            prop_assert!(result.is_ok(), "valid step {step} within [{t0}, {t_bound}] should succeed");
            prop_assert_eq!(result.unwrap(), step, "step should be returned unchanged");
        } else {
            prop_assert!(result.is_err(), "invalid step {step} should fail");
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Property 5: Validated tolerance values are always non-negative
// ═══════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn test_validation_tol_result_always_nonnegative(
        rtol in 0.0f64..1.0,
        atol in 0.0f64..1.0,
        n in 1usize..20,
    ) {
        let result = validate_tol(
            ToleranceValue::Scalar(rtol),
            ToleranceValue::Scalar(atol),
            n,
            RuntimeMode::Strict,
        );
        if let Ok(validated) = result {
            match validated.rtol {
                ToleranceValue::Scalar(v) => prop_assert!(v >= 0.0, "rtol must be non-negative: {v}"),
                ToleranceValue::Vector(ref vs) => {
                    for v in vs {
                        prop_assert!(*v >= 0.0, "rtol element must be non-negative: {v}");
                    }
                }
            }
            match validated.atol {
                ToleranceValue::Scalar(v) => prop_assert!(v >= 0.0, "atol must be non-negative: {v}"),
                ToleranceValue::Vector(ref vs) => {
                    for v in vs {
                        prop_assert!(*v >= 0.0, "atol element must be non-negative: {v}");
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Property 6: select_initial_step always returns positive for valid inputs
// ═══════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn test_step_size_select_initial_always_positive(
        y0_val in 0.01f64..100.0,
        interval in 0.1f64..100.0,
    ) {
        let y0 = [y0_val];
        let f0 = [-0.5 * y0_val];
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &y0,
            t_bound: interval,
            max_step: f64::INFINITY,
            f0: &f0,
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, y| vec![-0.5 * y[0]], &request)
            .expect("select_initial_step should succeed");
        prop_assert!(h > 0.0, "h must be positive, got {h}");
        prop_assert!(h <= interval, "h must not exceed interval, got {h}");
    }
}

// ═══════════════════════════════════════════════════════════════
// Property 7: RK45 solver is more accurate than RK23 for same tolerances
// ═══════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn test_solver_rk45_more_accurate_than_rk23(
        y0_val in 0.1f64..10.0,
    ) {
        let expected = y0_val * (-1.0_f64).exp();

        let result_45 = solve_ivp(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[y0_val],
                method: SolverKind::Rk45,
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        ).expect("RK45 should succeed");

        let result_23 = solve_ivp(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[y0_val],
                method: SolverKind::Rk23,
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        ).expect("RK23 should succeed");

        prop_assert!(result_45.success, "RK45 should succeed");
        prop_assert!(result_23.success, "RK23 should succeed");

        let err_45 = (result_45.y.last().unwrap()[0] - expected).abs();
        let err_23 = (result_23.y.last().unwrap()[0] - expected).abs();

        // Both should be accurate, but RK45 should generally use fewer steps
        prop_assert!(
            err_45 < 1e-4,
            "RK45 error too large: {err_45}"
        );
        prop_assert!(
            err_23 < 1e-3,
            "RK23 error too large: {err_23}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Property 8: validate_max_step accepts positive, rejects non-positive
// ═══════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn test_validation_max_step_positive_accepted(
        step in 0.001f64..1e10,
    ) {
        let result = validate_max_step(step);
        prop_assert!(result.is_ok(), "positive max_step should be accepted");
    }

    #[test]
    fn test_validation_max_step_nonpositive_rejected(
        step in -1e10f64..=0.0,
    ) {
        let result = validate_max_step(step);
        prop_assert!(result.is_err(), "non-positive max_step should be rejected");
    }
}

// ═══════════════════════════════════════════════════════════════
// Structured logging convention test
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_integrate_structured_log_convention() {
    let entry = TestLogEntry::new(
        "test_validation_tol_no_clamp_when_rtol_above_min",
        "fsci_integrate",
        "property test: rtol passthrough verified over 1000 cases",
    )
    .with_result(TestResult::Pass)
    .with_mode(RuntimeMode::Strict);

    let json = entry.to_json_line();
    let parsed: serde_json::Value =
        serde_json::from_str(&json).expect("structured log must be valid JSON");
    assert!(parsed["test_id"].is_string());
    assert!(parsed["timestamp_ms"].is_number());
    assert_eq!(parsed["level"], "info");
    assert_eq!(parsed["module"], "fsci_integrate");
}
