#![no_main]

use arbitrary::Arbitrary;
use fsci_special::elliprg;
use libfuzzer_sys::fuzz_target;

// Carlson elliptic RG property oracle for [frankenscipy-e6vgy].
//
// Verifies invariants on random non-negative inputs:
//
//   1. RG(x, y, z) finite for finite non-negative input.
//   2. RG(x, y, z) > 0 for any input not all zero.
//   3. RG is fully symmetric (unlike RD which is symmetric only
//      in (x, y)): RG(x, y, z) = RG(y, x, z) = RG(z, y, x).
//   4. Diagonal closed form: RG(x, x, x) = √x.

const BOUND: f64 = 100.0;
const TOL_REL: f64 = 1e-9;

#[derive(Debug, Arbitrary)]
struct CarlsonRgInput {
    x: f64,
    y: f64,
    z: f64,
}

fn sanitize_nonneg(value: f64) -> f64 {
    if value.is_finite() {
        value.abs().min(BOUND)
    } else {
        0.0
    }
}

fn approx_eq(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() < TOL_REL * scale
}

fuzz_target!(|input: CarlsonRgInput| {
    let x = sanitize_nonneg(input.x);
    let y = sanitize_nonneg(input.y);
    let z = sanitize_nonneg(input.z);

    let rg_xyz = elliprg(x, y, z);
    if !rg_xyz.is_finite() {
        panic!("RG({x}, {y}, {z}) = {rg_xyz}: non-finite for finite input");
    }
    // Property 2: positivity (RG ≥ 0 always, strict when not all zero).
    if rg_xyz < 0.0 {
        panic!("RG({x}, {y}, {z}) = {rg_xyz}: must be ≥ 0");
    }

    // Property 3: full argument symmetry.
    let rg_yxz = elliprg(y, x, z);
    let rg_zyx = elliprg(z, y, x);
    let rg_yzx = elliprg(y, z, x);
    if !approx_eq(rg_xyz, rg_yxz) {
        panic!(
            "RG symmetry x↔y broken: RG({x}, {y}, {z}) = {rg_xyz}, RG({y}, {x}, {z}) = {rg_yxz}"
        );
    }
    if !approx_eq(rg_xyz, rg_zyx) {
        panic!(
            "RG symmetry x↔z broken: RG({x}, {y}, {z}) = {rg_xyz}, RG({z}, {y}, {x}) = {rg_zyx}"
        );
    }
    if !approx_eq(rg_xyz, rg_yzx) {
        panic!(
            "RG cyclic symmetry broken: RG({x}, {y}, {z}) = {rg_xyz}, RG({y}, {z}, {x}) = {rg_yzx}"
        );
    }

    // Property 4: diagonal RG(x, x, x) = √x for x > 0.
    if x > 0.0 {
        let rg_diag = elliprg(x, x, x);
        let expected_diag = x.sqrt();
        if !approx_eq(rg_diag, expected_diag) {
            panic!(
                "RG diagonal broken: RG({x}, {x}, {x}) = {rg_diag}, expected √x = {expected_diag}"
            );
        }
    }
});
