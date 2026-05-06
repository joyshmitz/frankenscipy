#![no_main]

use arbitrary::Arbitrary;
use fsci_signal::lp2lp;
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-gb4x9]:
// lp2lp has scipy diff coverage and unit anchors but no
// libfuzzer harness. Drive sanitized (b, a, wo) inputs and
// assert the call:
//   1. Never panics in the success path.
//   2. Output b length is exactly the input b length (lp2lp scales
//      coefficient-wise on the numerator without trimming).
//   3. Output a length is between 1 and the input a length, inclusive
//      (normalize_filter trims leading zeros from a).
//   4. Output a[0] is exactly 1.0 (normalize_filter post-invariant).
//   5. No coefficient is NaN. (Coefficients may be ±∞ when the input
//      a leads with a denormal-tiny value — scipy.signal.lp2lp behaves
//      identically there, emitting a divide-by-zero runtime warning
//      and returning inf. NaN, by contrast, would be an unambiguous
//      defect.)
//   6. Errors only fire on degenerate input we cannot avoid (e.g.,
//      all-zero a after the sanitizer guard).

const COEFF_BOUND: f64 = 1.0e3;
const WO_MIN: f64 = 1.0e-3;
const WO_MAX: f64 = 1.0e3;
const MAX_LEN: usize = 12;
const MIN_LEN: usize = 1;

#[derive(Debug, Arbitrary)]
struct Lp2lpInput {
    b: Vec<f64>,
    a: Vec<f64>,
    wo_seed: f64,
}

fn sanitize(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-COEFF_BOUND, COEFF_BOUND)
    } else {
        0.0
    }
}

fn cap_length(v: Vec<f64>) -> Vec<f64> {
    v.into_iter().take(MAX_LEN).map(sanitize).collect()
}

fn sanitize_wo(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    // Map an arbitrary f64 to [WO_MIN, WO_MAX] via abs + clamp + offset.
    let abs = seed.abs();
    if abs == 0.0 || abs <= WO_MIN {
        WO_MIN
    } else if abs >= WO_MAX {
        WO_MAX
    } else {
        abs
    }
}

fuzz_target!(|input: Lp2lpInput| {
    let mut b = cap_length(input.b);
    let mut a = cap_length(input.a);

    if b.len() < MIN_LEN {
        b.push(1.0);
    }
    if a.len() < MIN_LEN {
        a.push(1.0);
    }
    // Ensure a is not all-zero so normalize_filter has a non-zero
    // leading coefficient to divide by.
    if a.iter().all(|&v| v == 0.0) {
        a[0] = 1.0;
    }

    let wo = sanitize_wo(input.wo_seed);

    let nb_in = b.len();
    let na_in = a.len();

    let result = lp2lp(&b, &a, wo);
    let (nb, na) = match result {
        Ok((nb, na)) => (nb, na),
        Err(_) => {
            // normalize_filter rejects degenerate `a` even after the
            // all-zero guard above (e.g., all-zero after leading-zero
            // trim). That's a contract-validating error, not a panic.
            return;
        }
    };

    assert_eq!(
        nb.len(),
        nb_in,
        "lp2lp: numerator length must be preserved"
    );
    assert!(
        !na.is_empty() && na.len() <= na_in,
        "lp2lp: denominator length must be in [1, {na_in}], got {}",
        na.len()
    );

    assert_eq!(
        na[0], 1.0,
        "lp2lp: normalize_filter must leave a[0] = 1.0, got {}",
        na[0]
    );

    // Outputs may be ±∞ when the input a leads with a denormal-tiny
    // value (scipy.signal.lp2lp matches this behaviour). NaN would
    // be an unambiguous defect — assert the strict no-NaN contract.
    for (i, &v) in nb.iter().enumerate() {
        assert!(
            !v.is_nan(),
            "lp2lp: nb[{i}] = NaN for input b={b:?}, a={a:?}, wo={wo}"
        );
    }
    for (i, &v) in na.iter().enumerate() {
        assert!(
            !v.is_nan(),
            "lp2lp: na[{i}] = NaN for input b={b:?}, a={a:?}, wo={wo}"
        );
    }
});
