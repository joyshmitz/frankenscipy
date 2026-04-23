#![no_main]

use arbitrary::Arbitrary;
use fsci_integrate::{IntegrateValidationError, validate_first_step, validate_max_step};
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct StepBoundsInput {
    first_step: f64,
    t0: f64,
    t_bound: f64,
    max_step: f64,
}

// Post-condition oracle per br-66c2: the prior harness only did
// `let _ = fn(input);`, a pure crash-only oracle. Now we assert the
// full input→outcome contract so any silent behavior change in
// validation semantics (e.g. a refactor that admits NaN or returns a
// different value than was passed in) will fail the fuzzer loudly
// instead of surviving as an unchecked panic-free execution.
fuzz_target!(|input: StepBoundsInput| {
    check_validate_first_step(input.first_step, input.t0, input.t_bound);
    check_validate_max_step(input.max_step);
});

fn check_validate_first_step(first_step: f64, t0: f64, t_bound: f64) {
    let result = validate_first_step(first_step, t0, t_bound);
    let range = (t_bound - t0).abs();
    let not_positive_finite = !first_step.is_finite() || first_step <= 0.0;
    let exceeds = !not_positive_finite && first_step > range;

    match result {
        Err(IntegrateValidationError::FirstStepMustBePositive) => {
            assert!(
                not_positive_finite,
                "FirstStepMustBePositive rejected a positive, finite first_step: \
                 first_step={first_step:?}, t0={t0:?}, t_bound={t_bound:?}"
            );
        }
        Err(IntegrateValidationError::FirstStepExceedsBounds) => {
            assert!(
                exceeds,
                "FirstStepExceedsBounds raised but first_step did not exceed range: \
                 first_step={first_step:?}, range={range:?}, t0={t0:?}, t_bound={t_bound:?}"
            );
        }
        Err(other) => panic!(
            "validate_first_step returned unexpected error variant: {other:?}; \
             first_step={first_step:?}, t0={t0:?}, t_bound={t_bound:?}"
        ),
        Ok(returned) => {
            assert!(
                !not_positive_finite,
                "validate_first_step accepted non-positive or non-finite input: \
                 first_step={first_step:?}, returned={returned:?}"
            );
            assert!(
                !exceeds,
                "validate_first_step accepted first_step that exceeds range: \
                 first_step={first_step:?}, range={range:?}, returned={returned:?}"
            );
            assert_eq!(
                returned.to_bits(),
                first_step.to_bits(),
                "validate_first_step must return the input bit-identical on success, \
                 got {returned:?} for {first_step:?}"
            );
        }
    }
}

fn check_validate_max_step(max_step: f64) {
    let result = validate_max_step(max_step);
    // Contract: NaN is rejected, values <= 0 are rejected, +Inf is allowed
    // (interpreted as "no cap"), positive finite values are returned as-is.
    let nan = max_step.is_nan();
    let non_positive = !nan && max_step <= 0.0;

    match result {
        Err(IntegrateValidationError::MaxStepMustBePositive) => {
            assert!(
                nan || non_positive,
                "MaxStepMustBePositive rejected a positive non-NaN max_step: {max_step:?}"
            );
        }
        Err(other) => panic!(
            "validate_max_step returned unexpected error variant: {other:?}; \
             max_step={max_step:?}"
        ),
        Ok(returned) => {
            assert!(
                !nan && !non_positive,
                "validate_max_step accepted NaN or non-positive input: {max_step:?}"
            );
            assert_eq!(
                returned.to_bits(),
                max_step.to_bits(),
                "validate_max_step must return the input bit-identical on success, \
                 got {returned:?} for {max_step:?}"
            );
        }
    }
}
