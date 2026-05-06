#![no_main]

use arbitrary::Arbitrary;
use fsci_signal::{CorrelationMode, correlation_lags};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-nexdg]:
// correlation_lags has 6 closed-form anchor cases but no fuzz
// coverage. Drive Arbitrary-derived (in1, in2) sizes through
// all three CorrelationMode variants and assert:
//   1. Never panics for any sanitized (in1, in2 >= 1) input.
//   2. Full result has length in1 + in2 - 1.
//   3. Same result has length in1 (matches the first input,
//      consistent with scipy's `mode='same'` convention for
//      cross-correlation output sizing).
//   4. Valid result has length |in1 - in2| + 1.
//   5. All results are sorted ascending.
//   6. Consecutive lags differ by exactly 1.

const MIN_LEN: usize = 1;
const MAX_LEN: usize = 2048;

#[derive(Debug, Arbitrary)]
struct Input {
    in1_seed: u16,
    in2_seed: u16,
}

fn sanitize(seed: u16) -> usize {
    (seed as usize % MAX_LEN).max(MIN_LEN)
}

fn check_result(lags: &[i64], expected_len: usize, label: &str, in1: usize, in2: usize) {
    assert_eq!(
        lags.len(),
        expected_len,
        "{label}: expected len {expected_len} for (in1={in1}, in2={in2}), got {}",
        lags.len()
    );
    if lags.is_empty() {
        return;
    }
    for w in lags.windows(2) {
        assert_eq!(
            w[1] - w[0],
            1,
            "{label}: consecutive lags must differ by 1, got {} and {}",
            w[0],
            w[1]
        );
    }
}

fuzz_target!(|input: Input| {
    let in1 = sanitize(input.in1_seed);
    let in2 = sanitize(input.in2_seed);

    let full = correlation_lags(in1, in2, CorrelationMode::Full);
    check_result(&full, in1 + in2 - 1, "Full", in1, in2);

    let same = correlation_lags(in1, in2, CorrelationMode::Same);
    check_result(&same, in1, "Same", in1, in2);

    let valid = correlation_lags(in1, in2, CorrelationMode::Valid);
    let valid_expected = if in1 >= in2 {
        in1 - in2 + 1
    } else {
        in2 - in1 + 1
    };
    check_result(&valid, valid_expected, "Valid", in1, in2);

    // Same result must be a contiguous slice of Full.
    if !same.is_empty() {
        let pos = full
            .iter()
            .position(|&v| v == same[0])
            .expect("Same first lag must appear in Full");
        for (offset, &v) in same.iter().enumerate() {
            assert_eq!(
                full[pos + offset], v,
                "Same must be a contiguous slice of Full"
            );
        }
    }
});
