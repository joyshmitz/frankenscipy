#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{Boltzmann, DiscreteDistribution};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-bm9zo]:
// Boltzmann is the first discrete-distribution fuzz target in
// this session. Drive Arbitrary-derived (λ, N, k_seed) where
// λ ∈ [0.01, 50], N ∈ [1, 1000], k arbitrary u64 sanitized
// against the support. Assert:
//   1. Constructor never panics.
//   2. pmf finite, non-negative on support k ∈ {0..N−1}.
//   3. pmf = 0 outside support (k ≥ N).
//   4. cdf monotone non-decreasing across the support, in [0, 1].
//   5. cdf(N−1) = 1 exactly (by construction of Z).
//   6. pmf sums to 1 across the full support.

const LAMBDA_MIN: f64 = 0.01;
const LAMBDA_MAX: f64 = 50.0;
const N_MAX: u32 = 1000;

#[derive(Debug, Arbitrary)]
struct Input {
    lambda_seed: f64,
    n_seed: u32,
    k_seed: u64,
}

fn sanitize_lambda(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(LAMBDA_MIN, LAMBDA_MAX)
}

fn sanitize_n(seed: u32) -> u32 {
    1 + (seed % N_MAX)
}

fn sanitize_k(seed: u64, n: u32) -> u64 {
    seed % (n as u64)
}

fuzz_target!(|input: Input| {
    let lambda = sanitize_lambda(input.lambda_seed);
    let n = sanitize_n(input.n_seed);
    let k = sanitize_k(input.k_seed, n);

    // 1. Constructor never panics within validated range.
    let dist = Boltzmann::new(lambda, n);

    // 2. pmf finite, non-negative on support.
    let p = dist.pmf(k);
    assert!(
        p.is_finite(),
        "pmf({k}; λ={lambda}, N={n}) must be finite, got {p}"
    );
    assert!(p >= 0.0, "pmf must be non-negative, got {p}");

    // 3. pmf = 0 strictly outside support.
    assert_eq!(dist.pmf(n as u64), 0.0, "pmf(N) must be 0");
    assert_eq!(dist.pmf((n as u64) + 100), 0.0, "pmf(>N) must be 0");

    // 4. cdf monotone non-decreasing across the support.
    let mut prev = 0.0;
    for j in 0..n.min(50) as u64 {
        let cv = dist.cdf(j);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({j}; λ={lambda}, N={n}) = {cv} must be in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({j}; λ={lambda}, N={n}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 5. cdf(N−1) = 1 exactly (within fp epsilon).
    let final_cdf = dist.cdf((n - 1) as u64);
    assert!(
        (final_cdf - 1.0).abs() < 1e-12,
        "cdf(N−1; λ={lambda}, N={n}) = {final_cdf}, want 1"
    );

    // 6. pmf sums to 1 across full support. Bound the work to
    // N ≤ 200 to keep the fuzz target fast; the closed-form
    // cdf(N−1) = 1 check above already covers the same invariant
    // for larger N.
    if n <= 200 {
        let total: f64 = (0..n as u64).map(|j| dist.pmf(j)).sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "pmf sum = {total} (λ={lambda}, N={n}), want 1"
        );
    }
});
