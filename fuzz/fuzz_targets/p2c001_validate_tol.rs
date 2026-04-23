#![no_main]

use arbitrary::Arbitrary;
use fsci_integrate::{ToleranceValue, validate_tol};
use fsci_runtime::RuntimeMode;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct ValidateTolInput {
    n: u8,
    hardened: bool,
    rtol_vector: bool,
    atol_vector: bool,
    rtol_values: Vec<f64>,
    atol_values: Vec<f64>,
}

fn build_value(as_vector: bool, values: &[f64]) -> ToleranceValue {
    let clipped = values.iter().copied().take(8).collect::<Vec<_>>();
    if as_vector {
        ToleranceValue::Vector(clipped)
    } else {
        let scalar = clipped.first().copied().unwrap_or(0.0);
        ToleranceValue::Scalar(scalar)
    }
}

fn all_positive_finite(value: &ToleranceValue) -> bool {
    match value {
        ToleranceValue::Scalar(v) => v.is_finite() && *v > 0.0,
        ToleranceValue::Vector(values) => values.iter().all(|v| v.is_finite() && *v > 0.0),
    }
}

fn length_consistent(value: &ToleranceValue, n: usize) -> bool {
    match value {
        ToleranceValue::Scalar(_) => true,
        ToleranceValue::Vector(values) => values.len() == n || n == 0,
    }
}

fuzz_target!(|input: ValidateTolInput| {
    let mode = if input.hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    };
    let n = usize::from(input.n);
    let rtol = build_value(input.rtol_vector, &input.rtol_values);
    let atol = build_value(input.atol_vector, &input.atol_values);
    if let Ok(validated) = validate_tol(rtol, atol, n, mode) {
        // br-66c2: post-condition oracle — a successful validation
        // must produce positive, finite rtol/atol with length matching n
        // (or scalar). Failure here means validate_tol returned Ok for
        // an input it should have rejected.
        assert!(
            all_positive_finite(&validated.rtol),
            "validated rtol contains non-positive or non-finite value: {:?}",
            validated.rtol
        );
        assert!(
            all_positive_finite(&validated.atol),
            "validated atol contains non-positive or non-finite value: {:?}",
            validated.atol
        );
        assert!(
            length_consistent(&validated.rtol, n),
            "validated rtol length inconsistent with n={n}: {:?}",
            validated.rtol
        );
        assert!(
            length_consistent(&validated.atol, n),
            "validated atol length inconsistent with n={n}: {:?}",
            validated.atol
        );
        assert_eq!(validated.mode, mode, "mode must round-trip");
    }
});
