#![no_main]

use arbitrary::Arbitrary;
use fsci_special::elliprd;
use libfuzzer_sys::fuzz_target;

// Carlson elliptic RD property oracle for [frankenscipy-b57y7].
//
// Verifies invariants on random non-negative inputs:
//
//   1. RD(x, y, z) finite for finite positive input.
//   2. RD(x, y, z) > 0 (positive integrand → positive integral).
//   3. RD is symmetric in (x, y) only:
//        RD(x, y, z) = RD(y, x, z)
//      but NOT necessarily RD(z, y, x). Z plays a distinguished role.
//   4. Diagonal closed form: RD(x, x, x) = x^{-3/2}.
//
// Catches regressions in the duplication-with-sum-term algorithm
// or the 5th-order Taylor correction specific to RD.

const BOUND: f64 = 100.0;
const TOL_REL: f64 = 1e-9;

#[derive(Debug, Arbitrary)]
struct CarlsonRdInput {
    x: f64,
    y: f64,
    z: f64,
}

fn sanitize_positive(value: f64) -> f64 {
    if value.is_finite() {
        value.abs().clamp(1e-3, BOUND)
    } else {
        1.0
    }
}

fn approx_eq(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() < TOL_REL * scale
}

fuzz_target!(|input: CarlsonRdInput| {
    let x = sanitize_positive(input.x);
    let y = sanitize_positive(input.y);
    let z = sanitize_positive(input.z);

    let rd_xyz = elliprd(x, y, z);
    if !rd_xyz.is_finite() {
        panic!("RD({x}, {y}, {z}) = {rd_xyz}: non-finite for finite positive input");
    }
    if rd_xyz <= 0.0 {
        panic!("RD({x}, {y}, {z}) = {rd_xyz}: must be > 0 for positive input");
    }

    // Property 3: symmetry in first two arguments.
    let rd_yxz = elliprd(y, x, z);
    if !approx_eq(rd_xyz, rd_yxz) {
        panic!(
            "RD symmetry x↔y broken: RD({x}, {y}, {z}) = {rd_xyz}, RD({y}, {x}, {z}) = {rd_yxz}"
        );
    }

    // Property 4: RD(x, x, x) = x^{-3/2}.
    let rd_diag = elliprd(x, x, x);
    let expected_diag = x.powf(-1.5);
    if !approx_eq(rd_diag, expected_diag) {
        panic!(
            "RD diagonal broken: RD({x}, {x}, {x}) = {rd_diag}, expected {expected_diag}"
        );
    }
});
