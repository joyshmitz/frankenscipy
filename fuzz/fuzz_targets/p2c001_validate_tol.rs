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

fuzz_target!(|input: ValidateTolInput| {
    let mode = if input.hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    };
    let rtol = build_value(input.rtol_vector, &input.rtol_values);
    let atol = build_value(input.atol_vector, &input.atol_values);
    let _ = validate_tol(rtol, atol, usize::from(input.n), mode);
});
