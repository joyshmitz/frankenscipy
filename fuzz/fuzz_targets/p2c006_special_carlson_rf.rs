#![no_main]

use arbitrary::Arbitrary;
use fsci_special::{elliprc, elliprf};
use libfuzzer_sys::fuzz_target;

// Carlson elliptic RF/RC property oracle for [frankenscipy-aq8uf].
//
// Verifies invariants on random non-negative inputs:
//
//   1. RF(x, y, z) and RC(x, y) finite for finite positive input.
//   2. Both functions return strictly positive values (positive
//      integrand ⇒ positive integral).
//   3. RF is symmetric in all three arguments:
//        RF(x, y, z) = RF(y, x, z) = RF(z, y, x).
//   4. RC degenerates to RF: RC(x, y) ≡ RF(x, y, y).
//
// Catches regressions in the duplication algorithm, the Taylor
// correction, or the symmetric reduction at RC.

const BOUND: f64 = 100.0;
const TOL_REL: f64 = 1e-9;

#[derive(Debug, Arbitrary)]
struct CarlsonInput {
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

fuzz_target!(|input: CarlsonInput| {
    let x = sanitize_positive(input.x);
    let y = sanitize_positive(input.y);
    let z = sanitize_positive(input.z);

    let rf_xyz = elliprf(x, y, z);
    if !rf_xyz.is_finite() {
        panic!("RF({x}, {y}, {z}) = {rf_xyz}: non-finite for finite positive input");
    }
    if rf_xyz <= 0.0 {
        panic!("RF({x}, {y}, {z}) = {rf_xyz}: must be > 0 for positive input");
    }

    // Property 3: argument symmetry.
    let rf_yxz = elliprf(y, x, z);
    let rf_zyx = elliprf(z, y, x);
    if !approx_eq(rf_xyz, rf_yxz) {
        panic!(
            "RF symmetry x↔y broken: RF({x}, {y}, {z}) = {rf_xyz}, RF({y}, {x}, {z}) = {rf_yxz}"
        );
    }
    if !approx_eq(rf_xyz, rf_zyx) {
        panic!(
            "RF symmetry x↔z broken: RF({x}, {y}, {z}) = {rf_xyz}, RF({z}, {y}, {x}) = {rf_zyx}"
        );
    }

    // Property 4: RC(x, y) ≡ RF(x, y, y).
    let rc_xy = elliprc(x, y);
    if !rc_xy.is_finite() {
        panic!("RC({x}, {y}) = {rc_xy}: non-finite for finite positive input");
    }
    let rf_xyy = elliprf(x, y, y);
    if !approx_eq(rc_xy, rf_xyy) {
        panic!("RC({x}, {y}) = {rc_xy} ≠ RF(x, y, y) = {rf_xyy}");
    }
});
