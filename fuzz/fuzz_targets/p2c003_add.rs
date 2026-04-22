#![no_main]

use arbitrary::Arbitrary;
use fsci_opt::approx_fprime;
use libfuzzer_sys::fuzz_target;

const EDGE_LIMIT: f64 = 8.0;
const ABS_TOL: f64 = 1.0e-6;
const REL_TOL: f64 = 5.0e-4;

#[derive(Debug, Arbitrary)]
struct ApproxFprimeInput {
    x0: f64,
    x1: f64,
    x2: f64,
    epsilon: f64,
}

fn sanitize_component(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-EDGE_LIMIT, EDGE_LIMIT)
    } else {
        0.0
    }
}

fn sanitize_epsilon(value: f64) -> f64 {
    if value.is_finite() {
        value.abs().clamp(1.0e-8, 1.0e-3)
    } else {
        1.0e-6
    }
}

fn quadratic(x: &[f64]) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    3.0 * x0 * x0 + 2.0 * x0 * x1 + 0.5 * x1 * x1 + x2 * x2 - 4.0 * x2
}

fn quadratic_grad(x: &[f64]) -> [f64; 3] {
    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    [6.0 * x0 + 2.0 * x1, 2.0 * x0 + x1, 2.0 * x2 - 4.0]
}

fn approx_eq(lhs: f64, rhs: f64, epsilon: f64) -> bool {
    let scale = lhs.abs().max(rhs.abs()).max(1.0);
    (lhs - rhs).abs() <= ABS_TOL + REL_TOL * scale + 8.0 * epsilon
}

fuzz_target!(|input: ApproxFprimeInput| {
    let x = [
        sanitize_component(input.x0),
        sanitize_component(input.x1),
        sanitize_component(input.x2),
    ];
    let epsilon = sanitize_epsilon(input.epsilon);

    let Ok(numerical) = approx_fprime(&x, quadratic, epsilon) else {
        return;
    };
    let analytical = quadratic_grad(&x);

    assert_eq!(numerical.len(), analytical.len());
    for (lhs, rhs) in numerical.iter().zip(analytical) {
        assert!(
            approx_eq(*lhs, rhs, epsilon),
            "approx_fprime mismatch: numerical={lhs:?} analytical={rhs:?} x={x:?} epsilon={epsilon:?}"
        );
    }
});
