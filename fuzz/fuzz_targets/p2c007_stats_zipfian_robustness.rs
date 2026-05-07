#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{DiscreteDistribution, Zipfian};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-ovr3s]:
// Zipfian is the fourth discrete-distribution fuzz target in
// this session. Drive Arbitrary a ∈ [0.5, 20], n ∈ [1, 200],
// k_seed (u64). Note that scipy.stats.zipfian only requires
// a > 0 (unlike Zipf which needs a > 1); we cover the a < 1
// regime here. Assert:
//   1. Constructor never panics.
//   2. pmf finite, non-negative on support; pmf(0) = 0;
//      pmf(k > n) = 0.
//   3. cdf monotone non-decreasing across the support, in [0, 1].
//   4. cdf(0) = 0; cdf(n) = 1 exactly.
//   5. pmf telescoping: pmf(k) = cdf(k) − cdf(k − 1).
//   6. Normalisation: ∑_{k=1..n} pmf(k) = 1.

const A_MIN: f64 = 0.5;
const A_MAX: f64 = 20.0;
const N_MAX: u32 = 200;

#[derive(Debug, Arbitrary)]
struct Input {
    a_seed: f64,
    n_seed: u32,
    k_seed: u64,
}

fn sanitize_a(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.5;
    }
    seed.abs().clamp(A_MIN, A_MAX)
}

fn sanitize_n(seed: u32) -> u32 {
    1 + (seed % N_MAX)
}

fn sanitize_k(seed: u64, n: u32) -> u64 {
    1 + (seed % n as u64)
}

fuzz_target!(|input: Input| {
    let a = sanitize_a(input.a_seed);
    let n = sanitize_n(input.n_seed);
    let k = sanitize_k(input.k_seed, n);

    // 1. Constructor never panics.
    let dist = Zipfian::new(a, n);

    // 2. pmf finite, non-negative on support; pmf(0) and pmf(k>n) = 0.
    assert_eq!(dist.pmf(0), 0.0, "pmf(0) must be 0 (support starts at k=1)");
    assert_eq!(
        dist.pmf((n as u64) + 1),
        0.0,
        "pmf(>n) must be 0"
    );
    let p = dist.pmf(k);
    assert!(
        p.is_finite(),
        "pmf({k}; a={a}, n={n}) must be finite, got {p}"
    );
    assert!(p >= 0.0, "pmf must be non-negative, got {p}");

    // 3-4. cdf monotone in [0, 1] + boundary.
    assert_eq!(dist.cdf(0), 0.0);
    let mut prev = 0.0;
    for j in 1..=n.min(50) as u64 {
        let cv = dist.cdf(j);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({j}; a={a}, n={n}) = {cv} must be in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({j}; a={a}, n={n}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }
    let final_cdf = dist.cdf(n as u64);
    assert!(
        (final_cdf - 1.0).abs() < 1e-12,
        "cdf(n; a={a}, n={n}) = {final_cdf}, want 1"
    );

    // 5. pmf telescoping.
    if k >= 1 {
        let prev = if k == 1 { 0.0 } else { dist.cdf(k - 1) };
        let diff = dist.cdf(k) - prev;
        assert!(
            (dist.pmf(k) - diff).abs() < 1e-12,
            "pmf({k}; a={a}, n={n}) = {} ≠ cdf-diff {diff}",
            dist.pmf(k)
        );
    }

    // 6. Normalisation. Bound total work to N ≤ 100 since the
    // closed-form cdf(n) = 1 check above already covers the same
    // invariant for larger n.
    if n <= 100 {
        let total: f64 = (1..=n as u64).map(|j| dist.pmf(j)).sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "pmf sum = {total} (a={a}, n={n}), want 1"
        );
    }
});
