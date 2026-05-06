#![no_main]

use arbitrary::Arbitrary;
use fsci_special::{elliprd, elliprj};
use libfuzzer_sys::fuzz_target;

// Carlson elliptic RJ property oracle for [frankenscipy-2bdkg].
//
// Verifies invariants on random valid inputs:
//
//   1. RJ(x, y, z, p) finite for finite valid input.
//   2. RJ(x, x, x, x) = x^{-3/2} (the integrand collapses).
//   3. RJ is symmetric in (x, y, z) but not in p:
//        RJ(x, y, z, p) = RJ(y, x, z, p) = RJ(z, y, x, p).
//   4. RJ(x, y, z, z) = RD(x, y, z) — the same integral when p = z.

const BOUND: f64 = 100.0;
const TOL_REL: f64 = 1e-7;

#[derive(Debug, Arbitrary)]
struct CarlsonRjInput {
    x: f64,
    y: f64,
    z: f64,
    p: f64,
}

fn sanitize_nonneg(value: f64) -> f64 {
    if value.is_finite() {
        value.abs().min(BOUND)
    } else {
        0.0
    }
}

fn sanitize_positive(value: f64) -> f64 {
    let v = sanitize_nonneg(value);
    if v < 1e-3 { 1e-3 } else { v }
}

fn approx_eq(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() < TOL_REL * scale
}

fuzz_target!(|input: CarlsonRjInput| {
    let x = sanitize_nonneg(input.x);
    let y = sanitize_nonneg(input.y);
    let z = sanitize_nonneg(input.z);
    let p = sanitize_positive(input.p);

    // RJ requires ≤ 1 zero in (x, y, z); skip the divergent corners.
    let zero_count = (x == 0.0) as u32 + (y == 0.0) as u32 + (z == 0.0) as u32;
    if zero_count >= 2 {
        return;
    }

    let rj_xyzp = elliprj(x, y, z, p);
    if !rj_xyzp.is_finite() {
        panic!("RJ({x}, {y}, {z}, {p}) = {rj_xyzp}: non-finite for valid input");
    }

    // Property 3: symmetry in (x, y, z).
    let rj_yxzp = elliprj(y, x, z, p);
    let rj_zyxp = elliprj(z, y, x, p);
    if !approx_eq(rj_xyzp, rj_yxzp) {
        panic!(
            "RJ symmetry x↔y broken: RJ({x}, {y}, {z}, {p}) = {rj_xyzp}, RJ({y}, {x}, {z}, {p}) = {rj_yxzp}"
        );
    }
    if !approx_eq(rj_xyzp, rj_zyxp) {
        panic!(
            "RJ symmetry x↔z broken: RJ({x}, {y}, {z}, {p}) = {rj_xyzp}, RJ({z}, {y}, {x}, {p}) = {rj_zyxp}"
        );
    }

    // Property 2: diagonal.
    if x > 0.0 {
        let rj_diag = elliprj(x, x, x, x);
        let expected_diag = x.powf(-1.5);
        if !approx_eq(rj_diag, expected_diag) {
            panic!(
                "RJ diagonal broken: RJ({x}, {x}, {x}, {x}) = {rj_diag}, expected x^(-3/2) = {expected_diag}"
            );
        }
    }

    // Property 4: RJ(x, y, z, z) = RD(x, y, z).
    if z > 0.0 {
        let rj_pz = elliprj(x, y, z, z);
        let rd = elliprd(x, y, z);
        if rd.is_finite() && rj_pz.is_finite() && !approx_eq(rj_pz, rd) {
            panic!(
                "RJ p=z metamorphic broken: RJ({x}, {y}, {z}, {z}) = {rj_pz}, RD({x}, {y}, {z}) = {rd}"
            );
        }
    }
});
