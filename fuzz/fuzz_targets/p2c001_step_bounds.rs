#![no_main]

use arbitrary::Arbitrary;
use fsci_integrate::{validate_first_step, validate_max_step};
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct StepBoundsInput {
    first_step: f64,
    t0: f64,
    t_bound: f64,
    max_step: f64,
}

fuzz_target!(|input: StepBoundsInput| {
    let _ = validate_first_step(input.first_step, input.t0, input.t_bound);
    let _ = validate_max_step(input.max_step);
});
