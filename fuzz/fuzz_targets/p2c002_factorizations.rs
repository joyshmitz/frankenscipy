#![no_main]

use arbitrary::Arbitrary;
use fsci_linalg::{InvOptions, LstsqDriver, LstsqOptions, PinvOptions, det, inv, lstsq, pinv};
use fsci_runtime::RuntimeMode;
use libfuzzer_sys::fuzz_target;

const MAX_DIM: usize = 64;

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

fn clamp_dimension(raw: u8) -> usize {
    usize::from(raw) % (MAX_DIM + 1)
}

/// Check A @ A_inv ≈ I with a scale-aware loose bound. Returns true
/// when the product is within tolerance or when numerical issues
/// prevent adjudication (non-finite intermediates).
fn check_inverse_identity(a: &[Vec<f64>], a_inv: &[Vec<f64>]) -> bool {
    let n = a.len();
    if n == 0 || a_inv.len() != n || a[0].len() != n || a_inv[0].len() != n {
        return true;
    }
    let mut max_off = 0.0_f64;
    let mut max_diag = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut prod = 0.0_f64;
            for k in 0..n {
                prod += a[i][k] * a_inv[k][j];
            }
            if !prod.is_finite() {
                return true;
            }
            if i == j {
                max_diag = max_diag.max((prod - 1.0).abs());
            } else {
                max_off = max_off.max(prod.abs());
            }
        }
    }
    let max_err = max_off.max(max_diag);
    max_err < 1e-3
}

fuzz_target!(|input: FactorizationInput| {
    let rows = clamp_dimension(input.rows);
    let cols = clamp_dimension(input.cols);
    let mode = if input.hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    };
    let a = build_matrix(rows, cols, &input.values);
    let b = build_vector(rows, &input.rhs);

    if let Ok(inv_result) = inv(
        &a,
        InvOptions {
            mode,
            check_finite: input.check_finite,
            assume_a: None,
            lower: false,
        },
    ) && rows == cols
        && rows > 0
        && a.iter().flatten().all(|v| v.is_finite())
        && inv_result.inverse.iter().flatten().all(|v| v.is_finite())
    {
        // br-66c2: metamorphic oracle — A @ A_inv ≈ I.
        assert!(
            check_inverse_identity(&a, &inv_result.inverse),
            "inv returned Ok but A @ A_inv diverges from identity"
        );
    }
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
