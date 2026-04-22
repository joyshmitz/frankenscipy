#![no_main]

use arbitrary::Arbitrary;
use fsci_opt::{approx_fprime, rosen, rosen_der, rosen_hess, rosen_hess_prod};
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct RosenPropertiesInput {
    x: Vec<f64>,
    direction: Vec<f64>,
    epsilon: f64,
}

const EDGE_LIMIT: f64 = 3.0;
const GRAD_ABS_TOL: f64 = 5.0e-4;
const GRAD_REL_TOL: f64 = 2.0e-2;
const HESS_TOL: f64 = 1.0e-10;

fn sanitize_component(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-EDGE_LIMIT, EDGE_LIMIT)
    } else {
        0.0
    }
}

fn sanitize_epsilon(value: f64) -> f64 {
    if value.is_finite() {
        value.abs().clamp(1.0e-7, 1.0e-4)
    } else {
        1.0e-6
    }
}

fn sanitize_state(values: &[f64]) -> Vec<f64> {
    let mut sanitized: Vec<f64> = values
        .iter()
        .copied()
        .take(6)
        .map(sanitize_component)
        .collect();
    while sanitized.len() < 2 {
        sanitized.push(1.0);
    }
    sanitized
}

fn sanitize_direction(values: &[f64], len: usize) -> Vec<f64> {
    let mut sanitized: Vec<f64> = values
        .iter()
        .copied()
        .take(len)
        .map(sanitize_component)
        .collect();
    sanitized.resize(len, 0.0);
    sanitized
}

fn approx_eq(lhs: f64, rhs: f64, epsilon: f64) -> bool {
    let scale = lhs.abs().max(rhs.abs()).max(1.0);
    (lhs - rhs).abs() <= GRAD_ABS_TOL + GRAD_REL_TOL * scale + 32.0 * epsilon
}

fn explicit_hess_prod(hessian: &[Vec<f64>], direction: &[f64]) -> Vec<f64> {
    hessian
        .iter()
        .map(|row| row.iter().zip(direction).map(|(lhs, rhs)| lhs * rhs).sum())
        .collect()
}

fuzz_target!(|input: RosenPropertiesInput| {
    let x = sanitize_state(&input.x);
    let direction = sanitize_direction(&input.direction, x.len());
    let epsilon = sanitize_epsilon(input.epsilon);

    let value = rosen(&x);
    assert!(
        value.is_finite() && value >= 0.0,
        "rosen should stay finite and non-negative for x={x:?}, got {value:?}"
    );

    let gradient = rosen_der(&x);
    assert_eq!(gradient.len(), x.len());
    assert!(gradient.iter().all(|value| value.is_finite()));

    let Ok(numerical) = approx_fprime(&x, rosen, epsilon) else {
        return;
    };
    for (lhs, rhs) in numerical.iter().zip(&gradient) {
        assert!(
            approx_eq(*lhs, *rhs, epsilon),
            "rosen gradient mismatch: numerical={lhs:?} analytical={rhs:?} x={x:?} epsilon={epsilon:?}"
        );
    }

    let hessian = rosen_hess(&x);
    assert_eq!(hessian.len(), x.len());
    assert!(hessian.iter().all(|row| row.len() == x.len()));

    let explicit = explicit_hess_prod(&hessian, &direction);
    let implicit = rosen_hess_prod(&x, &direction);
    assert_eq!(implicit.len(), explicit.len());
    for (lhs, rhs) in implicit.iter().zip(&explicit) {
        assert!(
            (lhs - rhs).abs() <= HESS_TOL,
            "rosen_hess_prod mismatch: implicit={lhs:?} explicit={rhs:?} x={x:?} direction={direction:?}"
        );
    }
});
