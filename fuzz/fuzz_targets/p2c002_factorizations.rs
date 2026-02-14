#![no_main]

use arbitrary::Arbitrary;
use fsci_linalg::{InvOptions, LstsqDriver, LstsqOptions, PinvOptions, det, inv, lstsq, pinv};
use fsci_runtime::RuntimeMode;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct FactorizationInput {
    rows: u8,
    cols: u8,
    hardened: bool,
    check_finite: bool,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

fn build_matrix(rows: usize, cols: usize, values: &[f64]) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; cols]; rows];
    for (idx, value) in values.iter().copied().take(rows * cols).enumerate() {
        let r = idx / cols;
        let c = idx % cols;
        matrix[r][c] = value;
    }
    matrix
}

fn build_vector(len: usize, values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; len];
    for (idx, value) in values.iter().copied().take(len).enumerate() {
        out[idx] = value;
    }
    out
}

fuzz_target!(|input: FactorizationInput| {
    let rows = usize::from(input.rows % 8);
    let cols = usize::from(input.cols % 8);
    let mode = if input.hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    };
    let a = build_matrix(rows, cols, &input.values);
    let b = build_vector(rows, &input.rhs);

    let _ = inv(
        &a,
        InvOptions {
            mode,
            check_finite: input.check_finite,
            assume_a: None,
            lower: false,
        },
    );
    let _ = det(&a, mode, input.check_finite);
    let _ = lstsq(
        &a,
        &b,
        LstsqOptions {
            mode,
            check_finite: input.check_finite,
            cond: None,
            driver: LstsqDriver::Gelsd,
        },
    );
    let _ = pinv(
        &a,
        PinvOptions {
            mode,
            check_finite: input.check_finite,
            atol: None,
            rtol: None,
        },
    );
});
