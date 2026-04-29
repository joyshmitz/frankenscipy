#![no_main]

//! Robustness fuzzing for `fsci_integrate::solve_ivp`.
//!
//! Drives the IVP solver with arbitrary tolerances, time spans, and a
//! small library of well-behaved RHS functions. Verifies that the
//! solver:
//!   * does not panic for any in-domain inputs,
//!   * returns a strictly-monotone time grid,
//!   * produces only finite output values.
//!
//! Catches regressions in step-size selection, tolerance clamping,
//! and the Rk23/Rk45/Bdf method selectors.
//!
//! Bead: `frankenscipy-r147`.

use arbitrary::Arbitrary;
use fsci_integrate::{SolveIvpOptions, SolverKind, ToleranceValue, solve_ivp};
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct IvpInput {
    rhs_kind: u8,
    method_kind: u8,
    rtol_log: f64,
    atol_log: f64,
    t_start: f64,
    t_span_len: f64,
    y0_a: f64,
    y0_b: f64,
}

fn sanitize_finite(value: f64, lo: f64, hi: f64, default: f64) -> f64 {
    if value.is_finite() {
        value.clamp(lo, hi)
    } else {
        default
    }
}

fn pick_method(byte: u8) -> SolverKind {
    match byte % 3 {
        0 => SolverKind::Rk23,
        1 => SolverKind::Rk45,
        _ => SolverKind::Bdf,
    }
}

fn rhs(kind: u8) -> impl FnMut(f64, &[f64]) -> Vec<f64> {
    let kind = kind % 5;
    move |t: f64, y: &[f64]| match kind {
        // Linear decay: y' = -y, equilibrium at zero, well-behaved.
        0 => y.iter().map(|v| -v).collect(),
        // Damped oscillator: y' = (-y[1], y[0]) on the unit circle.
        1 => {
            if y.len() == 2 {
                vec![-y[1], y[0]]
            } else {
                y.iter().map(|v| -v).collect()
            }
        }
        // Logistic growth: y' = 0.5 y (1 − y), bounded for y ∈ [0, 2].
        2 => y.iter().map(|v| 0.5 * v * (1.0 - *v)).collect(),
        // Forced linear: y' = -0.1 y + sin(t).
        3 => y.iter().map(|v| -0.1 * v + t.sin()).collect(),
        // Constant: y' = 1, drift.
        _ => vec![1.0; y.len()],
    }
}

fuzz_target!(|input: IvpInput| {
    // Tolerances clamped well above the solver minimum.
    let rtol = 10f64.powf(sanitize_finite(input.rtol_log, -10.0, -2.0, -3.0));
    let atol_value = 10f64.powf(sanitize_finite(input.atol_log, -10.0, -2.0, -6.0));
    let atol = ToleranceValue::Scalar(atol_value);

    // Time span: [t_start, t_start + len] with len > 0.
    let t_start = sanitize_finite(input.t_start, -100.0, 100.0, 0.0);
    let span_len = sanitize_finite(input.t_span_len.abs(), 1e-3, 50.0, 1.0);
    let t_end = t_start + span_len;

    // Initial state: pick 1 or 2 dimensions deterministically from the
    // rhs kind to keep the harness deterministic.
    let (mut y0_a, mut y0_b) = (
        sanitize_finite(input.y0_a, -10.0, 10.0, 1.0),
        sanitize_finite(input.y0_b, -10.0, 10.0, 0.0),
    );
    let rhs_kind = input.rhs_kind % 5;
    if rhs_kind == 2 {
        // Logistic stays bounded if y0 ∈ [0, 2].
        y0_a = y0_a.abs().clamp(0.05, 1.95);
        y0_b = y0_b.abs().clamp(0.05, 1.95);
    }
    let dim_two = matches!(rhs_kind, 1);
    let y0_vec: Vec<f64> = if dim_two { vec![y0_a, y0_b] } else { vec![y0_a] };

    let method = pick_method(input.method_kind);
    let mut rhs_fn = rhs(rhs_kind);

    let options = SolveIvpOptions {
        t_span: (t_start, t_end),
        y0: &y0_vec,
        method,
        rtol,
        atol,
        max_step: span_len.max(1e-3),
        ..SolveIvpOptions::default()
    };

    let Ok(result) = solve_ivp(&mut rhs_fn, &options) else {
        return; // Validation rejection is fine.
    };

    // Time grid invariants.
    assert!(
        !result.t.is_empty(),
        "solve_ivp returned empty time grid for valid input"
    );
    assert!(
        result.t[0] == t_start,
        "first sample time != t_start: got {}, expected {t_start}",
        result.t[0]
    );
    for i in 1..result.t.len() {
        assert!(
            result.t[i] > result.t[i - 1],
            "non-monotone time grid at i={i}: t[i-1]={}, t[i]={}",
            result.t[i - 1],
            result.t[i]
        );
        assert!(
            result.t[i].is_finite(),
            "non-finite t[{i}]={}",
            result.t[i]
        );
    }

    // State invariants.
    assert_eq!(
        result.y.len(),
        result.t.len(),
        "y rows ({}) != t length ({})",
        result.y.len(),
        result.t.len()
    );
    for (i, row) in result.y.iter().enumerate() {
        assert_eq!(
            row.len(),
            y0_vec.len(),
            "state row {i} has wrong dim {} vs {}",
            row.len(),
            y0_vec.len()
        );
        for (j, v) in row.iter().enumerate() {
            assert!(
                v.is_finite(),
                "non-finite state at i={i} j={j}: {v}"
            );
        }
    }
});
