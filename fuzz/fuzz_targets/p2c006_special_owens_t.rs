#![no_main]

use arbitrary::Arbitrary;
use fsci_special::owens_t_scalar;
use libfuzzer_sys::fuzz_target;

// Owen's T function property oracle for [frankenscipy-ef1iz].
//
// Verifies four mathematical invariants of T(h, a) that hold for
// all real (h, a):
//
//   1. Finite output for finite input.
//   2. |T(h, a)| ≤ 0.25 + ε (Owen's T is bounded by 1/4).
//   3. Symmetry in h: T(-h, a) = T(h, a) (even in h).
//   4. Antisymmetry in a: T(h, -a) = -T(h, a) (odd in a).
//
// Catches regressions in the Gauss-Legendre integration core or
// the symmetry-folding logic of owens_t_scalar.

const BOUND: f64 = 20.0;
// 0.26 leaves a 0.01 numerical-headroom envelope above the
// theoretical |T(h,a)| ≤ 0.25 bound.
const T_MAX: f64 = 0.26;

#[derive(Debug, Arbitrary)]
struct OwensTInput {
    h: f64,
    a: f64,
}

fn sanitize(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-BOUND, BOUND)
    } else {
        0.0
    }
}

fuzz_target!(|input: OwensTInput| {
    let h = sanitize(input.h);
    let a = sanitize(input.a);

    let t = owens_t_scalar(h, a);

    // Property 1: finite output for finite input.
    if !t.is_finite() {
        panic!("owens_t({h}, {a}) = {t}: non-finite for finite input");
    }

    // Property 2: bounded by 1/4 (with small numerical headroom).
    if t.abs() > T_MAX {
        panic!(
            "owens_t({h}, {a}) = {t}: violates |T| ≤ 1/4 bound (with 0.01 headroom)"
        );
    }

    // Property 3: symmetry in h (T is even in h).
    let t_neg_h = owens_t_scalar(-h, a);
    if (t - t_neg_h).abs() > 1e-12 {
        panic!(
            "owens_t symmetry violated: T({h}, {a}) = {t} but T(-h, a) = {t_neg_h}"
        );
    }

    // Property 4: antisymmetry in a (T is odd in a).
    let t_neg_a = owens_t_scalar(h, -a);
    if (t + t_neg_a).abs() > 1e-12 {
        panic!(
            "owens_t antisymmetry violated: T({h}, {a}) = {t} \
             but T(h, -a) = {t_neg_a} (sum should be 0)"
        );
    }
});
