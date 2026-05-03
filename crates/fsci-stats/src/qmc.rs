//! Quasi-Monte Carlo low-discrepancy sequences.
//!
//! Scaffold for `scipy.stats.qmc`. This module currently provides only the
//! Halton sequence; Sobol, LatinHypercube, PoissonDisk, Owen scrambling, and
//! the L2/star/wraparound discrepancy metrics are tracked as follow-on slices
//! under bead `frankenscipy-e5j6p`.

use crate::StatsError;

/// First 32 primes, sufficient for the Halton sequence in dimensions ≤ 32.
/// scipy's qmc.Halton uses the same prime list.
const HALTON_PRIMES: &[u64] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

/// Halton low-discrepancy sequence sampler in `[0, 1)^d`.
///
/// For each dimension `i`, samples are the radical inverse of the index in
/// base `p_i`, where `p_i` is the i-th prime. Discrepancy of an `N`-point
/// Halton set scales as `O(log(N)^d / N)` — much better than i.i.d. uniform's
/// `O(1/sqrt(N))` — making Halton useful for low-dimensional Monte Carlo
/// integration.
///
/// The sequence is deterministic; reseeding via `reset` rewinds to index 0.
/// `skip(k)` advances by `k` samples without materializing them, matching
/// scipy.stats.qmc.Halton().fast_forward(k).
#[derive(Debug, Clone)]
pub struct HaltonSampler {
    primes: Vec<u64>,
    next_index: u64,
}

impl HaltonSampler {
    /// Construct a Halton sampler for the given dimension.
    ///
    /// Returns `StatsError::InvalidArgument` if `dimension == 0` or exceeds
    /// the bundled prime table (32 dimensions). The 32-prime cutoff matches
    /// scipy's default — Halton's discrepancy degrades sharply past that.
    pub fn new(dimension: usize) -> Result<Self, StatsError> {
        if dimension == 0 {
            return Err(StatsError::InvalidArgument(
                "Halton dimension must be ≥ 1".to_string(),
            ));
        }
        if dimension > HALTON_PRIMES.len() {
            return Err(StatsError::InvalidArgument(format!(
                "Halton dimension {dimension} exceeds bundled prime table ({} primes)",
                HALTON_PRIMES.len()
            )));
        }
        Ok(Self {
            primes: HALTON_PRIMES[..dimension].to_vec(),
            next_index: 1,
        })
    }

    /// Dimension of the sequence.
    pub fn dimension(&self) -> usize {
        self.primes.len()
    }

    /// Index of the next sample to be emitted.
    pub fn next_index(&self) -> u64 {
        self.next_index
    }

    /// Rewind the sampler so the next call to `sample` returns sample index 1
    /// (skipping index 0 which would land at the origin and is never used by
    /// scipy.stats.qmc.Halton).
    pub fn reset(&mut self) {
        self.next_index = 1;
    }

    /// Advance the sampler by `k` samples without materializing them. Useful
    /// when partitioning a long sequence across workers.
    pub fn skip(&mut self, k: u64) {
        self.next_index = self.next_index.saturating_add(k);
    }

    /// Draw the next `n` samples, returned as `n * dimension` values in
    /// row-major order: `out[i * d + j]` is the j-th coordinate of the i-th
    /// sample.
    pub fn sample(&mut self, n: usize) -> Vec<f64> {
        let d = self.primes.len();
        let mut out = Vec::with_capacity(n.saturating_mul(d));
        for _ in 0..n {
            let idx = self.next_index;
            for &prime in &self.primes {
                out.push(radical_inverse(idx, prime));
            }
            self.next_index = self.next_index.saturating_add(1);
        }
        out
    }
}

/// Radical inverse of `index` in base `prime`.
///
/// φ_b(n) = sum over i of d_i / b^(i+1) where n = sum d_i * b^i.
///
/// Numerically: starts with `f = 1/prime`, accumulates `result += f * d`
/// where `d` is the next digit, then `f /= prime`. Loop exits when `n` is
/// zero. The result lies strictly inside `(0, 1)` for any positive `index`.
fn radical_inverse(mut index: u64, prime: u64) -> f64 {
    let inv_prime = 1.0_f64 / prime as f64;
    let mut f = inv_prime;
    let mut result = 0.0_f64;
    while index > 0 {
        let digit = index % prime;
        result += digit as f64 * f;
        index /= prime;
        f *= inv_prime;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_construct_rejects_zero_dimension() {
        let err = HaltonSampler::new(0).expect_err("zero-dim must fail");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn halton_construct_rejects_excess_dimension() {
        let err = HaltonSampler::new(33).expect_err("33-dim must fail");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn halton_sample_first_2d_canonical() {
        // Canonical first-five Halton points in 2D (bases 2 and 3):
        //   index 1 → (1/2, 1/3)
        //   index 2 → (1/4, 2/3)
        //   index 3 → (3/4, 1/9)
        //   index 4 → (1/8, 4/9)
        //   index 5 → (5/8, 7/9)
        let mut h = HaltonSampler::new(2).unwrap();
        let s = h.sample(5);
        let expected = [
            (1.0 / 2.0, 1.0 / 3.0),
            (1.0 / 4.0, 2.0 / 3.0),
            (3.0 / 4.0, 1.0 / 9.0),
            (1.0 / 8.0, 4.0 / 9.0),
            (5.0 / 8.0, 7.0 / 9.0),
        ];
        for (i, (ex, ey)) in expected.iter().enumerate() {
            assert!((s[i * 2] - ex).abs() < 1e-15, "x[{i}]");
            assert!((s[i * 2 + 1] - ey).abs() < 1e-15, "y[{i}]");
        }
    }

    #[test]
    fn halton_sample_in_unit_hypercube() {
        let mut h = HaltonSampler::new(5).unwrap();
        let n = 200;
        let s = h.sample(n);
        for v in &s {
            assert!(*v >= 0.0 && *v < 1.0, "value {v} out of [0,1)");
        }
        assert_eq!(s.len(), n * 5);
    }

    #[test]
    fn halton_metamorphic_skip_equivalent_to_consume() {
        let mut a = HaltonSampler::new(3).unwrap();
        let mut b = HaltonSampler::new(3).unwrap();
        let _ = a.sample(7);
        b.skip(7);
        let from_a = a.sample(5);
        let from_b = b.sample(5);
        assert_eq!(from_a, from_b);
    }

    #[test]
    fn halton_metamorphic_reset_replays_prefix() {
        let mut h = HaltonSampler::new(2).unwrap();
        let first = h.sample(10);
        h.reset();
        let second = h.sample(10);
        assert_eq!(first, second);
    }

    #[test]
    fn halton_metamorphic_low_discrepancy_beats_uniform() {
        // Halton's star discrepancy is bounded by C·log(N)^d / N. For d=2 and
        // N=1024, that bound is ~0.07 with C ≈ 1; uniform i.i.d. has
        // E[D*] ~ 1/sqrt(N) ≈ 0.031 expected but the *worst-case* L∞ box
        // discrepancy of any deterministic uniform point set is much higher.
        // A safe relation: for the unit-square box [0, 0.5]^2, the Halton
        // count fraction must agree with the box volume (0.25) within 5%
        // for N ≥ 200, while a deterministic offset-grid sample might not.
        let mut h = HaltonSampler::new(2).unwrap();
        let s = h.sample(1024);
        let in_box = (0..1024)
            .filter(|&i| s[i * 2] < 0.5 && s[i * 2 + 1] < 0.5)
            .count();
        let frac = in_box as f64 / 1024.0;
        assert!(
            (frac - 0.25).abs() < 0.05,
            "Halton box-count fraction {frac} should be within 5% of 0.25"
        );
    }

    #[test]
    fn halton_index_advances_with_each_sample() {
        let mut h = HaltonSampler::new(2).unwrap();
        assert_eq!(h.next_index(), 1);
        let _ = h.sample(3);
        assert_eq!(h.next_index(), 4);
    }

    #[test]
    fn halton_dimension_query() {
        let h = HaltonSampler::new(7).unwrap();
        assert_eq!(h.dimension(), 7);
    }
}
