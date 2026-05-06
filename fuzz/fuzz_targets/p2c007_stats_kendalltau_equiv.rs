#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{kendalltau, kendalltau_alternative};
use libfuzzer_sys::fuzz_target;

// Kendall's tau equivalence oracle for [frankenscipy-jfa78].
//
// After the 5d39d94 perf fix tightened the kendalltau O(N²) inner loop
// (drop redundant equality, hoist xi/yi, guard the multiplication),
// verify random inputs + ties don't surface a mismatch:
//
//   1. kendalltau(x, y).statistic equals a naïve in-fuzz reference.
//   2. kendalltau and kendalltau_alternative("two-sided") agree on the
//      statistic (both touched by the same opt).

const BOUND: f64 = 1.0e6;
const MIN_N: usize = 3;
const MAX_N: usize = 64;

#[derive(Debug, Arbitrary)]
struct KendallInput {
    x: Vec<f64>,
    y: Vec<f64>,
}

fn sanitize(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-BOUND, BOUND)
    } else {
        0.0
    }
}

fn naive_tau(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n < 2 {
        return None;
    }
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    let mut x_ties: i64 = 0;
    let mut y_ties: i64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let x_tied = x[i] == x[j];
            let y_tied = y[i] == y[j];
            if x_tied {
                x_ties += 1;
            }
            if y_tied {
                y_ties += 1;
            }
            let product = dx * dy;
            if !x_tied && !y_tied && product > 0.0 {
                concordant += 1;
            } else if !x_tied && !y_tied && product < 0.0 {
                discordant += 1;
            }
        }
    }
    let n_pairs = (n * (n - 1) / 2) as f64;
    let denom = ((n_pairs - x_ties as f64) * (n_pairs - y_ties as f64)).sqrt();
    if denom == 0.0 {
        return None;
    }
    Some((concordant - discordant) as f64 / denom)
}

fuzz_target!(|input: KendallInput| {
    let len = input.x.len().min(input.y.len()).clamp(MIN_N, MAX_N);
    if len < MIN_N {
        return;
    }
    let x: Vec<f64> = input.x.iter().take(len).copied().map(sanitize).collect();
    let y: Vec<f64> = input.y.iter().take(len).copied().map(sanitize).collect();

    let opt = kendalltau(&x, &y);
    let alt = kendalltau_alternative(&x, &y, "two-sided");

    // If the production statistic is NaN (constant input — denom 0), the
    // naïve impl will also produce None; treat both as equivalent and skip.
    let naive = naive_tau(&x, &y);
    if opt.statistic.is_nan() {
        assert!(
            naive.is_none() || naive.is_some_and(|v| v.is_nan()),
            "kendalltau NaN but naive returned a finite tau"
        );
        assert!(
            alt.statistic.is_nan(),
            "kendalltau_alternative diverged from kendalltau on NaN case"
        );
        return;
    }

    let naive_tau_val = naive.expect("optimized produced finite tau but naive was None");
    let scale = opt.statistic.abs().max(naive_tau_val.abs()).max(1.0);
    if (opt.statistic - naive_tau_val).abs() > 1e-9 * scale {
        panic!(
            "kendalltau perf-opt diverges from naive: opt={}, naive={}",
            opt.statistic, naive_tau_val
        );
    }

    if (opt.statistic - alt.statistic).abs() > 1e-12 * scale {
        panic!(
            "kendalltau and kendalltau_alternative two-sided disagree on statistic: opt={}, alt={}",
            opt.statistic, alt.statistic
        );
    }
});
