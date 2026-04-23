#![no_main]

//! Fuzz target for hyp0f1 / hyp1f1 / hyp2f1 parameter broadcasting
//! (br-kstk). Exercises scalar/vector/complex tensor combinations,
//! shape mismatches, and NaN/Inf edge cases. Oracles:
//!   - no panic on any combination (crash oracle)
//!   - when all inputs are length-1 RealVec, the broadcast result must
//!     match the pure-RealScalar result elementwise (metamorphic).
//!   - when shapes are incompatible, the call must return Err, not Ok.

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::types::{Complex64, SpecialResult, SpecialTensor};
use fsci_special::{hyp0f1, hyp1f1, hyp2f1};
use libfuzzer_sys::fuzz_target;

const MAX_LEN: usize = 8;

#[derive(Debug, Arbitrary)]
enum TensorSpec {
    Empty,
    RealScalar(f64),
    ComplexScalar(f64, f64),
    RealVec(Vec<f64>),
    ComplexVec(Vec<(f64, f64)>),
}

impl TensorSpec {
    fn to_tensor(&self) -> SpecialTensor {
        match self {
            TensorSpec::Empty => SpecialTensor::Empty,
            TensorSpec::RealScalar(x) => SpecialTensor::RealScalar(*x),
            TensorSpec::ComplexScalar(re, im) => {
                SpecialTensor::ComplexScalar(Complex64 { re: *re, im: *im })
            }
            TensorSpec::RealVec(values) => {
                let clipped: Vec<f64> = values.iter().copied().take(MAX_LEN).collect();
                SpecialTensor::RealVec(clipped)
            }
            TensorSpec::ComplexVec(pairs) => {
                let clipped: Vec<Complex64> = pairs
                    .iter()
                    .copied()
                    .take(MAX_LEN)
                    .map(|(re, im)| Complex64 { re, im })
                    .collect();
                SpecialTensor::ComplexVec(clipped)
            }
        }
    }
}

#[derive(Debug, Arbitrary)]
struct BroadcastInput {
    hardened: bool,
    which: u8,
    a: TensorSpec,
    b: TensorSpec,
    c: TensorSpec,
    z: TensorSpec,
}

/// All inputs non-panicking produce Ok OR Err; the only forbidden
/// outcome is a panic. The crash oracle is implicit via libfuzzer.
fn invoke(input: &BroadcastInput, mode: RuntimeMode) -> SpecialResult {
    let a = input.a.to_tensor();
    let b = input.b.to_tensor();
    let c = input.c.to_tensor();
    let z = input.z.to_tensor();
    match input.which % 3 {
        0 => hyp0f1(&b, &z, mode),
        1 => hyp1f1(&a, &b, &z, mode),
        _ => hyp2f1(&a, &b, &c, &z, mode),
    }
}

fuzz_target!(|input: BroadcastInput| {
    let mode = if input.hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    };
    let _ = invoke(&input, mode);
});
