//! Quasi-Monte Carlo low-discrepancy sequences.
//!
//! Compact `scipy.stats.qmc` surface for deterministic integration tests:
//! Halton, Sobol, Latin Hypercube, Poisson-disk sampling, scale, and the
//! centered/L2-star/mixture/wraparound discrepancy metrics.

use crate::StatsError;

/// First 32 primes, sufficient for the Halton sequence in dimensions ≤ 32.
/// scipy's qmc.Halton uses the same prime list.
const HALTON_PRIMES: &[u64] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

/// Common stateful engine surface for QMC samplers.
pub trait QmcEngine {
    /// Dimension of each emitted point.
    fn dimension(&self) -> usize;

    /// Rewind to the engine's initial state.
    fn reset(&mut self);

    /// Advance the engine by `n` points without returning them.
    fn fast_forward(&mut self, n: u64);

    /// Draw `n` points in row-major order.
    fn sample(&mut self, n: usize) -> Vec<f64>;
}

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

impl QmcEngine for HaltonSampler {
    fn dimension(&self) -> usize {
        HaltonSampler::dimension(self)
    }

    fn reset(&mut self) {
        HaltonSampler::reset(self);
    }

    fn fast_forward(&mut self, n: u64) {
        self.skip(n);
    }

    fn sample(&mut self, n: usize) -> Vec<f64> {
        HaltonSampler::sample(self, n)
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

// ══════════════════════════════════════════════════════════════════════
// Sobol sampling
// ══════════════════════════════════════════════════════════════════════

/// Sobol low-discrepancy sequence sampler for one or two dimensions.
///
/// The first two Sobol dimensions are enough for the common QMC integration
/// smoke paths in this crate and match SciPy's unscrambled prefix:
/// `(0,0), (1/2,1/2), (3/4,1/4), (1/4,3/4), ...`.
///
/// `with_digital_shift` applies a deterministic xor shift to each coordinate.
/// This is not a full Owen tree permutation, but it is the standard
/// random-shifted digital net primitive and preserves the low-discrepancy
/// structure while moving the origin off the boundary.
#[derive(Debug, Clone)]
pub struct SobolSampler {
    dimension: usize,
    next_index: u64,
    digital_shift: Vec<u64>,
}

impl SobolSampler {
    /// Construct an unscrambled Sobol sampler.
    pub fn new(dimension: usize) -> Result<Self, StatsError> {
        Self::with_shift_words(dimension, vec![0; dimension])
    }

    /// Construct a deterministic digital-shifted Sobol sampler.
    pub fn with_digital_shift(dimension: usize, seed: u64) -> Result<Self, StatsError> {
        let shifts = (0..dimension)
            .map(|idx| splitmix64(seed.wrapping_add(idx as u64)))
            .collect();
        Self::with_shift_words(dimension, shifts)
    }

    fn with_shift_words(dimension: usize, digital_shift: Vec<u64>) -> Result<Self, StatsError> {
        if dimension == 0 {
            return Err(StatsError::InvalidArgument(
                "Sobol dimension must be ≥ 1".to_string(),
            ));
        }
        if dimension > 2 {
            return Err(StatsError::InvalidArgument(
                "Sobol supports dimensions 1..=2 in this QMC surface".to_string(),
            ));
        }
        Ok(Self {
            dimension,
            next_index: 0,
            digital_shift,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn next_index(&self) -> u64 {
        self.next_index
    }

    pub fn reset(&mut self) {
        self.next_index = 0;
    }

    pub fn skip(&mut self, k: u64) {
        self.next_index = self.next_index.saturating_add(k);
    }

    pub fn sample(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n.saturating_mul(self.dimension));
        for _ in 0..n {
            let idx = self.next_index;
            for dim in 0..self.dimension {
                let bits = sobol_bits(idx, dim) ^ self.digital_shift[dim];
                out.push(bits_to_unit(bits));
            }
            self.next_index = self.next_index.saturating_add(1);
        }
        out
    }
}

impl QmcEngine for SobolSampler {
    fn dimension(&self) -> usize {
        SobolSampler::dimension(self)
    }

    fn reset(&mut self) {
        SobolSampler::reset(self);
    }

    fn fast_forward(&mut self, n: u64) {
        self.skip(n);
    }

    fn sample(&mut self, n: usize) -> Vec<f64> {
        SobolSampler::sample(self, n)
    }
}

fn sobol_bits(index: u64, dimension: usize) -> u64 {
    let mut gray = index ^ (index >> 1);
    let mut bit = 0usize;
    let mut value = 0u64;
    while gray != 0 {
        if gray & 1 == 1 {
            value ^= sobol_direction(dimension, bit);
        }
        gray >>= 1;
        bit += 1;
    }
    value
}

fn sobol_direction(dimension: usize, bit: usize) -> u64 {
    let mut direction = 1u64 << 63;
    if dimension == 0 {
        return direction >> bit.min(63);
    }

    for _ in 0..bit.min(63) {
        direction ^= direction >> 1;
    }
    direction
}

fn bits_to_unit(bits: u64) -> f64 {
    unit_interval_from_u64(bits)
}

// ══════════════════════════════════════════════════════════════════════
// Centered L2 discrepancy
// ══════════════════════════════════════════════════════════════════════

/// Compute the centered L2 discrepancy CD²(P) of an `n × d` point set in
/// `[0, 1)^d`, where `sample` is row-major (`sample[i * d + j]` is the j-th
/// coordinate of the i-th point).
///
/// Matches `scipy.stats.qmc.discrepancy(sample, method='CD')`. The formula is
///
///   CD²(P) = (13/12)^d
///          - (2/N) Σ_i Π_k [ 1 + 0.5·|x_i^k − 0.5|
///                              - 0.5·(x_i^k − 0.5)² ]
///          + (1/N²) Σ_{i,j} Π_k [ 1 + 0.5·|x_i^k − 0.5|
///                                    + 0.5·|x_j^k − 0.5|
///                                    - 0.5·|x_i^k − x_j^k| ]
///
/// Returns `Err(StatsError::InvalidArgument)` when `dimension == 0`, when
/// `sample.len()` is not a multiple of `dimension`, or when any coordinate is
/// outside `[0, 1]`.
pub fn centered_discrepancy(sample: &[f64], dimension: usize) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "centered_discrepancy: dimension must be ≥ 1".to_string(),
        ));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "centered_discrepancy: sample.len() {} not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    let n = sample.len() / dimension;
    if n == 0 {
        return Ok((13.0_f64 / 12.0).powi(dimension as i32));
    }
    for (idx, &v) in sample.iter().enumerate() {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(StatsError::InvalidArgument(format!(
                "centered_discrepancy: sample[{idx}] = {v} outside [0, 1]"
            )));
        }
    }

    let leading = (13.0_f64 / 12.0).powi(dimension as i32);

    // Single-sum term Σ_i Π_k [1 + 0.5|x − 0.5| - 0.5 (x − 0.5)²].
    let mut single = 0.0_f64;
    for i in 0..n {
        let mut prod = 1.0_f64;
        for k in 0..dimension {
            let x = sample[i * dimension + k];
            let centered = x - 0.5;
            prod *= 1.0 + 0.5 * centered.abs() - 0.5 * centered * centered;
        }
        single += prod;
    }

    // Double-sum term Σ_{i,j} Π_k [...]. Quadratic in n; QMC users typically
    // call this on samples with n ≤ a few thousand.
    let mut double = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut prod = 1.0_f64;
            for k in 0..dimension {
                let xi = sample[i * dimension + k];
                let xj = sample[j * dimension + k];
                prod *=
                    1.0 + 0.5 * (xi - 0.5).abs() + 0.5 * (xj - 0.5).abs() - 0.5 * (xi - xj).abs();
            }
            double += prod;
        }
    }

    let n_f = n as f64;
    Ok(leading - 2.0 / n_f * single + double / (n_f * n_f))
}

/// Compute the mixture L2 discrepancy MD²(P) of an `n × d` point set in
/// `[0, 1)^d`, with `sample` row-major.
///
/// Matches `scipy.stats.qmc.discrepancy(sample, method='MD')`. The formula
///
///   MD² = (19/12)^d
///       − (2/N) Σ_i  Π_k [ 5/3 − 1/4·|xi^k − 1/2| − 1/4·(xi^k − 1/2)² ]
///       + (1/N²) Σ_{i,j} Π_k [ 15/8 − 1/4·|xi^k − 1/2| − 1/4·|xj^k − 1/2|
///                                  − 3/4·|xi^k − xj^k|
///                                  + 1/2·(xi^k − xj^k)² ]
///
/// is a "mix" of centered- and wraparound-style weighting and is one of the
/// four standard L2 discrepancy metrics in scipy.stats.qmc.
pub fn mixture_discrepancy(sample: &[f64], dimension: usize) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "mixture_discrepancy: dimension must be ≥ 1".to_string(),
        ));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "mixture_discrepancy: sample.len() {} not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    let n = sample.len() / dimension;
    if n == 0 {
        return Ok((19.0_f64 / 12.0).powi(dimension as i32));
    }
    for (idx, &v) in sample.iter().enumerate() {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(StatsError::InvalidArgument(format!(
                "mixture_discrepancy: sample[{idx}] = {v} outside [0, 1]"
            )));
        }
    }
    let leading = (19.0_f64 / 12.0).powi(dimension as i32);
    // Single-sum term.
    let mut single = 0.0_f64;
    for i in 0..n {
        let mut prod = 1.0_f64;
        for k in 0..dimension {
            let x = sample[i * dimension + k];
            let centered = x - 0.5;
            prod *= 5.0 / 3.0 - 0.25 * centered.abs() - 0.25 * centered * centered;
        }
        single += prod;
    }
    // Double-sum term.
    let mut double = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut prod = 1.0_f64;
            for k in 0..dimension {
                let xi = sample[i * dimension + k];
                let xj = sample[j * dimension + k];
                let d = (xi - xj).abs();
                prod *= 15.0 / 8.0 - 0.25 * (xi - 0.5).abs() - 0.25 * (xj - 0.5).abs() - 0.75 * d
                    + 0.5 * (xi - xj).powi(2);
            }
            double += prod;
        }
    }
    let n_f = n as f64;
    Ok(leading - 2.0 / n_f * single + double / (n_f * n_f))
}

/// Linearly scale a unit-cube `[0, 1)^d` design into the box `[l_bounds, u_bounds]`.
///
/// `sample` is row-major (`sample[i * d + j]` is the j-th coordinate of the
/// i-th point). The result has the same shape; coordinate `j` is mapped via
///   `out_j = l_bounds[j] + (u_bounds[j] - l_bounds[j]) * sample_j`.
///
/// Matches the *forward* direction of `scipy.stats.qmc.scale(sample,
/// l_bounds, u_bounds)`. Returns `Err(StatsError::InvalidArgument)` when the
/// bound vectors disagree with `dimension`, when any non-NaN coordinate is
/// outside `[0, 1]`, or when any `u_bounds[j] < l_bounds[j]`.
pub fn scale(
    sample: &[f64],
    dimension: usize,
    l_bounds: &[f64],
    u_bounds: &[f64],
) -> Result<Vec<f64>, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "qmc::scale: dimension must be ≥ 1".to_string(),
        ));
    }
    if l_bounds.len() != dimension || u_bounds.len() != dimension {
        return Err(StatsError::InvalidArgument(format!(
            "qmc::scale: bounds vectors of length ({}, {}) do not match dimension {dimension}",
            l_bounds.len(),
            u_bounds.len()
        )));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "qmc::scale: sample.len() {} not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    for k in 0..dimension {
        if !l_bounds[k].is_finite() || !u_bounds[k].is_finite() {
            return Err(StatsError::InvalidArgument(format!(
                "qmc::scale: non-finite bound at dim {k}"
            )));
        }
        if u_bounds[k] < l_bounds[k] {
            return Err(StatsError::InvalidArgument(format!(
                "qmc::scale: u_bounds[{k}]={} < l_bounds[{k}]={}",
                u_bounds[k], l_bounds[k]
            )));
        }
    }
    let n = sample.len() / dimension;
    let mut out = Vec::with_capacity(sample.len());
    for i in 0..n {
        for k in 0..dimension {
            let value = sample[i * dimension + k];
            if !value.is_nan() && !(0.0..=1.0).contains(&value) {
                return Err(StatsError::InvalidArgument(
                    "qmc::scale: sample is not in unit hypercube".to_string(),
                ));
            }
            let scale_k = u_bounds[k] - l_bounds[k];
            out.push(l_bounds[k] + scale_k * value);
        }
    }
    Ok(out)
}

/// Update the centered L2 discrepancy `existing_disc` to reflect a new point
/// appended to `existing_sample` — without recomputing the full O((N+1)²)
/// double-sum from scratch.
///
/// Algebra: with `A = (13/12)^d`, `B = Σ_i s1(x_i)`, `C = Σ_{i,j} s2(x_i, x_j)`,
/// the centered discrepancy is `A - 2B/N + C/N²`. We can recover `C` from
/// `existing_disc` once `B` is computed in one O(N·d) pass over the sample,
/// then add the new point's `s1`, cross, and self contributions in another
/// O(N·d) pass — total O(N·d), strictly faster than the O((N+1)²·d) full
/// recomputation.
///
/// Matches the incremental-update pattern in
/// `scipy.stats.qmc.update_discrepancy(x_new, sample, initial_disc)`.
pub fn update_centered_discrepancy(
    existing_sample: &[f64],
    dimension: usize,
    existing_disc: f64,
    new_point: &[f64],
) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "update_centered_discrepancy: dimension must be ≥ 1".to_string(),
        ));
    }
    if !existing_sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "update_centered_discrepancy: existing_sample.len() {} not a multiple of dimension {dimension}",
            existing_sample.len()
        )));
    }
    if new_point.len() != dimension {
        return Err(StatsError::InvalidArgument(format!(
            "update_centered_discrepancy: new_point length {} != dimension {dimension}",
            new_point.len()
        )));
    }
    for &v in existing_sample.iter().chain(new_point.iter()) {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(StatsError::InvalidArgument(format!(
                "update_centered_discrepancy: value {v} outside [0, 1]"
            )));
        }
    }
    let n_old = existing_sample.len() / dimension;
    let n_new = n_old + 1;
    let leading = (13.0_f64 / 12.0).powi(dimension as i32);
    // Compute B_old = Σ_i s1(x_i) on the existing sample.
    let s1 = |point: &[f64]| -> f64 {
        let mut prod = 1.0_f64;
        for &coordinate in point.iter().take(dimension) {
            let centered = coordinate - 0.5;
            prod *= 1.0 + 0.5 * centered.abs() - 0.5 * centered * centered;
        }
        prod
    };
    let s2_pair = |a: &[f64], b: &[f64]| -> f64 {
        let mut prod = 1.0_f64;
        for k in 0..dimension {
            prod *= 1.0 + 0.5 * (a[k] - 0.5).abs() + 0.5 * (b[k] - 0.5).abs()
                - 0.5 * (a[k] - b[k]).abs();
        }
        prod
    };

    if n_old == 0 {
        // Singleton extension.
        let s1z = s1(new_point);
        let s2zz = s2_pair(new_point, new_point);
        let cd = leading - 2.0 * s1z + s2zz;
        return Ok(cd);
    }

    let n_old_f = n_old as f64;
    let mut b_old = 0.0_f64;
    for i in 0..n_old {
        let row = &existing_sample[i * dimension..(i + 1) * dimension];
        b_old += s1(row);
    }
    // Recover C_old from existing_disc:
    //   existing_disc = A - 2 B_old / N + C_old / N²
    //   ⟹ C_old = N² · (existing_disc - A) + 2 N · B_old
    let c_old = n_old_f * n_old_f * (existing_disc - leading) + 2.0 * n_old_f * b_old;

    let s1z = s1(new_point);
    let mut cross = 0.0_f64;
    for i in 0..n_old {
        let row = &existing_sample[i * dimension..(i + 1) * dimension];
        cross += s2_pair(row, new_point);
    }
    let self_term = s2_pair(new_point, new_point);

    let b_new = b_old + s1z;
    let c_new = c_old + 2.0 * cross + self_term;
    let n_new_f = n_new as f64;
    Ok(leading - 2.0 / n_new_f * b_new + c_new / (n_new_f * n_new_f))
}

/// Compute the L2-star discrepancy SD²(P) of an `n × d` point set in
/// `[0, 1)^d`, with `sample` row-major.
///
/// Matches `scipy.stats.qmc.discrepancy(sample, method='L2-star')`. The
/// closed form (Hickernell):
///
///   SD² = (1/3)^d
///       − (2^(1−d) / N) Σ_i Π_k (1 − (xi^k)²)
///       + (1/N²) Σ_{i,j} Π_k (1 − max(xi^k, xj^k))
///
/// SD² weights how well the design covers the lower-left orthants
/// `[0, x]^d` and is the most common deterministic-sequence quality
/// measure in the QMC literature.
pub fn l2_star_discrepancy(sample: &[f64], dimension: usize) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "l2_star_discrepancy: dimension must be ≥ 1".to_string(),
        ));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "l2_star_discrepancy: sample.len() {} not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    let n = sample.len() / dimension;
    if n == 0 {
        return Ok((1.0_f64 / 3.0).powi(dimension as i32));
    }
    for (idx, &v) in sample.iter().enumerate() {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(StatsError::InvalidArgument(format!(
                "l2_star_discrepancy: sample[{idx}] = {v} outside [0, 1]"
            )));
        }
    }
    let leading = (1.0_f64 / 3.0).powi(dimension as i32);
    let two_pow_one_minus_d = 2.0_f64.powi(1 - dimension as i32);
    // Single-sum Σ_i Π_k (1 - x_i^k²).
    let mut single = 0.0_f64;
    for i in 0..n {
        let mut prod = 1.0_f64;
        for k in 0..dimension {
            let x = sample[i * dimension + k];
            prod *= 1.0 - x * x;
        }
        single += prod;
    }
    // Double-sum Σ_ij Π_k (1 - max(x_i^k, x_j^k)).
    let mut double = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut prod = 1.0_f64;
            for k in 0..dimension {
                let xi = sample[i * dimension + k];
                let xj = sample[j * dimension + k];
                prod *= 1.0 - xi.max(xj);
            }
            double += prod;
        }
    }
    let n_f = n as f64;
    Ok(leading - two_pow_one_minus_d / n_f * single + double / (n_f * n_f))
}

/// Compute the wraparound L2 discrepancy WD²(P) of an `n × d` point set in
/// `[0, 1)^d`, with `sample` row-major (`sample[i * d + j]` is coordinate j
/// of point i).
///
/// Matches `scipy.stats.qmc.discrepancy(sample, method='WD')`. The formula
///
///   WD² = -(4/3)^d
///       + (1/N²) Σ_{i,j} Π_k [ 3/2 - |x_i^k − x_j^k| · (1 − |x_i^k − x_j^k|) ]
///
/// is invariant under cyclic shifts of any dimension's coordinates — so it
/// captures the design's quality on the unit *torus*, complementing the
/// centered discrepancy which weights points near the cube's center.
///
/// Returns `Err(StatsError::InvalidArgument)` for the same reasons as
/// `centered_discrepancy`.
pub fn wraparound_discrepancy(sample: &[f64], dimension: usize) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "wraparound_discrepancy: dimension must be ≥ 1".to_string(),
        ));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "wraparound_discrepancy: sample.len() {} not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    let n = sample.len() / dimension;
    if n == 0 {
        return Ok(-(4.0_f64 / 3.0).powi(dimension as i32));
    }
    for (idx, &v) in sample.iter().enumerate() {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(StatsError::InvalidArgument(format!(
                "wraparound_discrepancy: sample[{idx}] = {v} outside [0, 1]"
            )));
        }
    }
    let leading = -(4.0_f64 / 3.0).powi(dimension as i32);
    let mut double = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut prod = 1.0_f64;
            for k in 0..dimension {
                let xi = sample[i * dimension + k];
                let xj = sample[j * dimension + k];
                let d = (xi - xj).abs();
                prod *= 1.5 - d * (1.0 - d);
            }
            double += prod;
        }
    }
    let n_f = n as f64;
    Ok(leading + double / (n_f * n_f))
}

// ══════════════════════════════════════════════════════════════════════
// Latin Hypercube Sampling
// ══════════════════════════════════════════════════════════════════════

/// Latin Hypercube Sampling (LHS) over `[0, 1)^d`.
///
/// For each dimension, the unit interval is partitioned into `n` equal-width
/// strata; exactly one sample is placed in each stratum. The per-dimension
/// stratum order is a uniform random permutation. This is the construction
/// used by `scipy.stats.qmc.LatinHypercube`.
///
/// Two modes:
/// - `centered = true` (default): each sample sits at the midpoint of its
///   stratum, i.e. `(perm_j[i] + 0.5) / n`. Deterministic given the seed.
/// - `centered = false`: each sample is drawn uniformly inside its stratum,
///   i.e. `(perm_j[i] + u) / n` for `u ∈ [0, 1)`.
#[derive(Debug, Clone)]
pub struct LatinHypercubeSampler {
    dimension: usize,
    centered: bool,
    rng_state: u64,
}

impl LatinHypercubeSampler {
    /// Construct a centered LHS sampler in `dimension`-dimensional unit
    /// hypercube, seeded so subsequent `sample(n)` calls are reproducible.
    pub fn new(dimension: usize, seed: u64) -> Result<Self, StatsError> {
        if dimension == 0 {
            return Err(StatsError::InvalidArgument(
                "LatinHypercube dimension must be ≥ 1".to_string(),
            ));
        }
        Ok(Self {
            dimension,
            centered: true,
            // Derive the live RNG state from the seed via splitmix mixing so
            // seed=0 produces a non-trivial state.
            rng_state: splitmix64(seed.wrapping_add(0x9E37_79B9_7F4A_7C15)),
        })
    }

    /// Switch between centered (cell midpoint) and randomized (uniform within
    /// cell) modes. Returns the previous setting.
    pub fn set_centered(&mut self, centered: bool) -> bool {
        let prev = self.centered;
        self.centered = centered;
        prev
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Draw an `n × d` LHS design, row-major (`out[i*d + j]` is sample i,
    /// dimension j).
    pub fn sample(&mut self, n: usize) -> Vec<f64> {
        let d = self.dimension;
        let mut out = vec![0.0_f64; n.saturating_mul(d)];
        if n == 0 {
            return out;
        }
        let inv_n = 1.0_f64 / n as f64;
        for j in 0..d {
            let perm = self.permutation(n);
            for i in 0..n {
                let stratum = perm[i] as f64;
                let u = if self.centered {
                    0.5
                } else {
                    self.next_uniform()
                };
                out[i * d + j] = (stratum + u) * inv_n;
            }
        }
        out
    }

    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..n).collect();
        // Fisher-Yates with the internal splitmix RNG.
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        perm
    }

    fn next_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix64(self.rng_state)
    }

    fn next_uniform(&mut self) -> f64 {
        unit_interval_from_u64(self.next_u64())
    }
}

// ══════════════════════════════════════════════════════════════════════
// Poisson-disk sampling
// ══════════════════════════════════════════════════════════════════════

/// Two-dimensional Poisson-disk sampler over the unit square.
///
/// This is a deterministic dart-throwing implementation: candidate points are
/// drawn from a SplitMix stream and accepted only when they are at least
/// `radius` away from every previous point. It is intentionally compact and
/// bounded for conformance and metamorphic tests; callers requesting more
/// points than the radius permits may receive fewer than `n` points.
#[derive(Debug, Clone)]
pub struct PoissonDiskSampler {
    radius: f64,
    rng_state: u64,
}

impl PoissonDiskSampler {
    pub fn new_2d(radius: f64, seed: u64) -> Result<Self, StatsError> {
        if !radius.is_finite() || radius <= 0.0 || radius >= 1.0 {
            return Err(StatsError::InvalidArgument(
                "PoissonDisk radius must be finite and in (0, 1)".to_string(),
            ));
        }
        Ok(Self {
            radius,
            rng_state: splitmix64(seed.wrapping_add(0xA5A5_A5A5_5A5A_5A5A)),
        })
    }

    pub fn dimension(&self) -> usize {
        2
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Draw up to `n` points in row-major `[x0, y0, x1, y1, ...]` order.
    pub fn sample(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n.saturating_mul(2));
        if n == 0 {
            return out;
        }

        let min_dist2 = self.radius * self.radius;
        let max_attempts = n.saturating_mul(2_000).max(2_000);
        for _ in 0..max_attempts {
            if out.len() / 2 == n {
                break;
            }
            let x = self.next_uniform();
            let y = self.next_uniform();
            if poisson_candidate_is_valid(&out, x, y, min_dist2) {
                out.push(x);
                out.push(y);
            }
        }
        out
    }

    fn next_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix64(self.rng_state)
    }

    fn next_uniform(&mut self) -> f64 {
        unit_interval_from_u64(self.next_u64())
    }
}

fn poisson_candidate_is_valid(sample: &[f64], x: f64, y: f64, min_dist2: f64) -> bool {
    for point in sample.chunks_exact(2) {
        let dx = x - point[0];
        let dy = y - point[1];
        if dx.mul_add(dx, dy * dy) < min_dist2 {
            return false;
        }
    }
    true
}

fn unit_interval_from_u64(bits: u64) -> f64 {
    // 53-bit mantissa division: standard f64 [0,1) sample.
    (bits >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
}

fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Method selector for [`geometric_discrepancy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometricDiscrepancyMethod {
    /// Minimum pairwise Euclidean distance (default).
    MinDist,
    /// Mean edge length of the minimum spanning tree built on the sample.
    Mst,
}

/// Geometric discrepancy of a sample. Higher values correspond to greater
/// sample uniformity (the inverse polarity from the standard discrepancies
/// in this module).
///
/// Matches `scipy.stats.qmc.geometric_discrepancy(sample, method, metric='euclidean')`.
/// The `metric` parameter on the scipy side is currently fixed to euclidean
/// here — sufficient for the QMC sample-quality use cases. The sample buffer
/// is laid out row-major as `n × dimension` (the same convention the rest of
/// the qmc module uses for its `sample(n)` outputs).
///
/// `MinDist` returns `min_{i<j} ||x_i − x_j||`. `Mst` builds the
/// fully-connected weighted graph on the sample with edge weights = pairwise
/// Euclidean distance and returns the mean edge length of its minimum
/// spanning tree (Prim's algorithm; n ≤ a few thousand is the comfortable
/// regime — the algorithm is O(n²) in time and memory).
pub fn geometric_discrepancy(
    sample: &[f64],
    dimension: usize,
    method: GeometricDiscrepancyMethod,
) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "dimension must be >= 1".into(),
        ));
    }
    if sample.len() % dimension != 0 {
        return Err(StatsError::InvalidArgument(format!(
            "sample length {} is not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    let n = sample.len() / dimension;
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "sample must contain at least two points".into(),
        ));
    }
    // Sample must lie in the unit hypercube — scipy raises a ValueError for
    // out-of-range inputs. Mirror that.
    if sample.iter().any(|v| !(0.0..=1.0).contains(v) || !v.is_finite()) {
        return Err(StatsError::InvalidArgument(
            "sample must lie in [0, 1] with finite coordinates".into(),
        ));
    }

    let row = |i: usize| &sample[i * dimension..(i + 1) * dimension];
    let euclid = |i: usize, j: usize| -> f64 {
        let a = row(i);
        let b = row(j);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    };

    match method {
        GeometricDiscrepancyMethod::MinDist => {
            let mut best = f64::INFINITY;
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = euclid(i, j);
                    if d > 0.0 && d < best {
                        best = d;
                    }
                }
            }
            if !best.is_finite() {
                // All pairs collapsed onto a duplicate — match scipy's behaviour
                // of returning the smallest (zero) distance.
                Ok(0.0)
            } else {
                Ok(best)
            }
        }
        GeometricDiscrepancyMethod::Mst => {
            // Prim's algorithm on the dense weighted graph. O(n²) time and
            // O(n) auxiliary memory. Returns the mean MST edge length.
            let mut in_tree = vec![false; n];
            let mut min_edge = vec![f64::INFINITY; n];
            in_tree[0] = true;
            for j in 1..n {
                min_edge[j] = euclid(0, j);
            }
            let mut total = 0.0_f64;
            for _ in 1..n {
                // Pick the cheapest fringe vertex.
                let mut u = usize::MAX;
                let mut best = f64::INFINITY;
                for j in 0..n {
                    if !in_tree[j] && min_edge[j] < best {
                        best = min_edge[j];
                        u = j;
                    }
                }
                if u == usize::MAX {
                    // Sample reduces to fewer than 2 distinct points; fall back.
                    return Ok(0.0);
                }
                in_tree[u] = true;
                total += best;
                // Relax the fringe.
                for j in 0..n {
                    if !in_tree[j] {
                        let w = euclid(u, j);
                        if w < min_edge[j] {
                            min_edge[j] = w;
                        }
                    }
                }
            }
            Ok(total / (n - 1) as f64)
        }
    }
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

    #[test]
    fn sobol_construct_rejects_unsupported_dimensions() {
        assert!(matches!(
            SobolSampler::new(0),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            SobolSampler::new(3),
            Err(StatsError::InvalidArgument(_))
        ));
    }

    #[test]
    fn sobol_sample_first_2d_canonical() {
        let mut sobol = SobolSampler::new(2).unwrap();
        let sample = sobol.sample(5);
        let expected = [
            (0.0, 0.0),
            (0.5, 0.5),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.375, 0.375),
        ];
        for (i, (x, y)) in expected.iter().enumerate() {
            assert!((sample[i * 2] - x).abs() < 1e-15, "x[{i}]");
            assert!((sample[i * 2 + 1] - y).abs() < 1e-15, "y[{i}]");
        }
    }

    #[test]
    fn sobol_metamorphic_skip_equivalent_to_consume() {
        let mut a = SobolSampler::new(2).unwrap();
        let mut b = SobolSampler::new(2).unwrap();
        let _ = a.sample(11);
        b.skip(11);
        assert_eq!(a.sample(8), b.sample(8));
    }

    #[test]
    fn sobol_digital_shift_is_reproducible_and_bounded() {
        let mut a = SobolSampler::with_digital_shift(2, 99).unwrap();
        let mut b = SobolSampler::with_digital_shift(2, 99).unwrap();
        let shifted = a.sample(16);
        assert_eq!(shifted, b.sample(16));
        assert_ne!(shifted, SobolSampler::new(2).unwrap().sample(16));
        for value in shifted {
            assert!((0.0..1.0).contains(&value), "shifted value {value}");
        }
    }

    #[test]
    fn qmc_engine_trait_dispatches_halton_and_sobol() {
        fn draw_prefix(engine: &mut dyn QmcEngine, n: usize) -> Vec<f64> {
            engine.sample(n)
        }

        let mut halton = HaltonSampler::new(2).unwrap();
        let mut sobol = SobolSampler::new(2).unwrap();
        assert_eq!(
            draw_prefix(&mut halton, 2),
            vec![0.5, 1.0 / 3.0, 0.25, 2.0 / 3.0]
        );
        assert_eq!(draw_prefix(&mut sobol, 2), vec![0.0, 0.0, 0.5, 0.5]);
    }

    #[test]
    fn lhs_construct_rejects_zero_dimension() {
        let err = LatinHypercubeSampler::new(0, 0).expect_err("zero-dim must fail");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn lhs_metamorphic_one_per_stratum_centered() {
        // Centered LHS: every sample's coordinate is (k + 0.5) / n for some
        // integer k ∈ [0, n). Each dimension must hit every k exactly once.
        let n = 10;
        let d = 4;
        let mut lhs = LatinHypercubeSampler::new(d, 42).unwrap();
        let s = lhs.sample(n);
        for j in 0..d {
            let mut strata: Vec<i64> = (0..n)
                .map(|i| (s[i * d + j] * n as f64 - 0.5).round() as i64)
                .collect();
            strata.sort_unstable();
            assert_eq!(
                strata,
                (0..n as i64).collect::<Vec<_>>(),
                "dim {j}: each stratum must appear exactly once"
            );
        }
    }

    #[test]
    fn lhs_metamorphic_seed_reproducibility() {
        let mut a = LatinHypercubeSampler::new(3, 1234).unwrap();
        let mut b = LatinHypercubeSampler::new(3, 1234).unwrap();
        assert_eq!(a.sample(20), b.sample(20));
    }

    #[test]
    fn lhs_metamorphic_distinct_seeds_diverge() {
        let mut a = LatinHypercubeSampler::new(3, 1).unwrap();
        let mut b = LatinHypercubeSampler::new(3, 2).unwrap();
        assert_ne!(a.sample(20), b.sample(20));
    }

    #[test]
    fn lhs_randomized_in_unit_hypercube() {
        let mut lhs = LatinHypercubeSampler::new(4, 7).unwrap();
        lhs.set_centered(false);
        let s = lhs.sample(50);
        for v in &s {
            assert!(*v >= 0.0 && *v < 1.0, "value {v} out of [0,1)");
        }
        assert_eq!(s.len(), 50 * 4);
    }

    #[test]
    fn lhs_randomized_one_per_stratum_invariant() {
        // Randomized LHS still places exactly one sample per stratum per
        // dimension; only the within-cell offset is random. Stratum index
        // is `floor(value * n)`.
        let n = 16;
        let d = 3;
        let mut lhs = LatinHypercubeSampler::new(d, 12345).unwrap();
        lhs.set_centered(false);
        let s = lhs.sample(n);
        for j in 0..d {
            let mut strata: Vec<usize> =
                (0..n).map(|i| (s[i * d + j] * n as f64) as usize).collect();
            strata.sort_unstable();
            assert_eq!(
                strata,
                (0..n).collect::<Vec<_>>(),
                "dim {j}: every stratum hit exactly once"
            );
        }
    }

    #[test]
    fn lhs_zero_samples_returns_empty_design() {
        let mut lhs = LatinHypercubeSampler::new(3, 0).unwrap();
        assert!(lhs.sample(0).is_empty());
    }

    #[test]
    fn poisson_disk_rejects_invalid_radius() {
        for radius in [0.0, -0.1, 1.0, f64::NAN] {
            assert!(
                matches!(
                    PoissonDiskSampler::new_2d(radius, 0),
                    Err(StatsError::InvalidArgument(_))
                ),
                "radius {radius} should fail"
            );
        }
    }

    #[test]
    fn poisson_disk_samples_are_separated_and_reproducible() {
        let mut a = PoissonDiskSampler::new_2d(0.08, 123).unwrap();
        let mut b = PoissonDiskSampler::new_2d(0.08, 123).unwrap();
        let sample = a.sample(24);
        assert_eq!(sample, b.sample(24));
        assert_eq!(sample.len(), 48);
        for value in &sample {
            assert!((0.0..1.0).contains(value), "value {value}");
        }
        for i in 0..24 {
            for j in i + 1..24 {
                let dx = sample[i * 2] - sample[j * 2];
                let dy = sample[i * 2 + 1] - sample[j * 2 + 1];
                let dist = dx.mul_add(dx, dy * dy).sqrt();
                assert!(dist >= 0.08 - 1e-15, "distance {dist} too small");
            }
        }
    }

    #[test]
    fn discrepancy_rejects_zero_dimension() {
        let err = centered_discrepancy(&[], 0).expect_err("d=0 must fail");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn discrepancy_rejects_misshaped_sample() {
        let err =
            centered_discrepancy(&[0.1, 0.2, 0.3], 2).expect_err("3 values for d=2 must fail");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn discrepancy_rejects_out_of_range_value() {
        let err = centered_discrepancy(&[0.5, 1.5], 2).expect_err("1.5 must be rejected");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn discrepancy_empty_sample_is_leading_constant() {
        // With n=0 the single and double sums vanish and only the (13/12)^d
        // leading term remains.
        let d = 3;
        let cd = centered_discrepancy(&[], d).unwrap();
        let expected = (13.0_f64 / 12.0).powi(d as i32);
        assert!(
            (cd - expected).abs() < 1e-15,
            "empty-sample CD² should equal (13/12)^d = {expected}, got {cd}"
        );
    }

    #[test]
    fn discrepancy_metamorphic_halton_beats_uniform_grid_2d() {
        // For 64 points in [0,1)^2, a Halton design has CD² much smaller
        // than a deterministic offset uniform grid (which has rows of points
        // exactly aligned, blowing up the double-sum term).
        let n = 64;
        let d = 2;
        let mut h = HaltonSampler::new(d).unwrap();
        let halton = h.sample(n);
        // Aligned grid: every point at (0.5, 0.5) — pathological, maximally
        // bad CD² because all single-sum products are identical and the
        // double-sum is dominated by the single-distance pair.
        let aligned = vec![0.5_f64; n * d];
        let cd_halton = centered_discrepancy(&halton, d).unwrap();
        let cd_aligned = centered_discrepancy(&aligned, d).unwrap();
        assert!(
            cd_halton < cd_aligned,
            "Halton CD²={cd_halton} should beat aligned-point CD²={cd_aligned}"
        );
        // Halton CD² for n=64, d=2 should be small (<0.01).
        assert!(
            cd_halton.abs() < 0.01,
            "Halton CD² {cd_halton} unexpectedly large"
        );
    }

    #[test]
    fn scale_rejects_mismatched_bounds() {
        let err = scale(&[0.5, 0.5], 2, &[0.0], &[1.0]).expect_err("bounds shape must match dim");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn scale_rejects_inverted_bounds() {
        let err =
            scale(&[0.5, 0.5], 2, &[2.0, 0.0], &[1.0, 1.0]).expect_err("u<l must be rejected");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn scale_rejects_coordinates_outside_unit_hypercube() {
        for sample in [
            &[-0.1_f64][..],
            &[1.5],
            &[f64::INFINITY],
            &[f64::NEG_INFINITY],
        ] {
            let err = scale(sample, 1, &[0.0], &[1.0]).expect_err("outside unit cube");
            assert!(matches!(err, StatsError::InvalidArgument(_)));
        }
    }

    #[test]
    fn scale_preserves_scipy_nan_propagation() {
        let out = scale(&[f64::NAN], 1, &[2.0], &[5.0]).expect("NaN propagates");
        assert!(out[0].is_nan());
    }

    #[test]
    fn scale_identity_with_unit_bounds() {
        let sample = [0.0_f64, 0.5, 0.99, 0.25, 0.75, 0.5];
        let out = scale(&sample, 2, &[0.0, 0.0], &[1.0, 1.0]).unwrap();
        for (a, b) in sample.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn scale_metamorphic_inverse_recovers_unit() {
        // forward: x ↦ l + (u-l)·x; inverse: y ↦ (y-l)/(u-l).
        let sample = [0.1_f64, 0.4, 0.7, 0.9, 0.5, 0.3];
        let l = [-2.0_f64, 5.0];
        let u = [3.0_f64, 8.5];
        let scaled = scale(&sample, 2, &l, &u).unwrap();
        for i in 0..3 {
            for k in 0..2 {
                let recovered = (scaled[i * 2 + k] - l[k]) / (u[k] - l[k]);
                let expected = sample[i * 2 + k];
                assert!(
                    (recovered - expected).abs() < 1e-12,
                    "roundtrip at ({i},{k}): got {recovered}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn scale_zero_width_bound_is_collapsed_dimension() {
        // u == l yields the constant value at that dimension.
        let sample = [0.1_f64, 0.5, 0.9];
        let out = scale(&sample, 1, &[2.0], &[2.0]).unwrap();
        for v in &out {
            assert_eq!(*v, 2.0);
        }
    }

    #[test]
    fn update_centered_disc_metamorphic_matches_batch_recompute() {
        // Build a 16-point Halton 2D design, compute CD² on first 16 points
        // batch, then incrementally extend with one more point and compare
        // against the batch CD² of the full 17-point set.
        let mut h = HaltonSampler::new(2).unwrap();
        let initial = h.sample(16);
        let cd_initial = centered_discrepancy(&initial, 2).unwrap();
        let extra = h.sample(1); // one new point
        let cd_via_update = update_centered_discrepancy(&initial, 2, cd_initial, &extra).unwrap();
        let mut combined = initial.clone();
        combined.extend_from_slice(&extra);
        let cd_via_batch = centered_discrepancy(&combined, 2).unwrap();
        assert!(
            (cd_via_update - cd_via_batch).abs() < 1e-12,
            "incremental {cd_via_update} != batch {cd_via_batch}"
        );
    }

    #[test]
    fn update_centered_disc_singleton_starts_from_empty() {
        // Adding a point to an empty sample must equal the CD² of the
        // singleton design.
        let new_point = [0.4_f64, 0.7];
        // CD² of empty sample is the (13/12)^d leading constant.
        let empty_cd = centered_discrepancy(&[], 2).unwrap();
        let after = update_centered_discrepancy(&[], 2, empty_cd, &new_point).unwrap();
        let direct = centered_discrepancy(&new_point, 2).unwrap();
        assert!((after - direct).abs() < 1e-12, "{after} != {direct}");
    }

    #[test]
    fn update_centered_disc_rejects_dimension_mismatch() {
        let err = update_centered_discrepancy(&[0.5, 0.5], 2, 0.0, &[0.5])
            .expect_err("new_point length must match dim");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn l2_star_rejects_invalid_inputs() {
        assert!(matches!(
            l2_star_discrepancy(&[], 0),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            l2_star_discrepancy(&[0.5, 0.5, 0.5], 2),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            l2_star_discrepancy(&[1.5], 1),
            Err(StatsError::InvalidArgument(_))
        ));
    }

    #[test]
    fn l2_star_empty_sample_is_one_third_to_d() {
        for d in 1..=4 {
            let sd = l2_star_discrepancy(&[], d).unwrap();
            let expected = (1.0_f64 / 3.0).powi(d as i32);
            assert!((sd - expected).abs() < 1e-15, "d={d}: got {sd}");
        }
    }

    #[test]
    fn l2_star_metamorphic_single_corner_point() {
        // For n=1 at the corner (0,…,0), the single product Π (1 - 0²) = 1
        // and the double product Π (1 - max(0,0)) = 1.
        // SD² = (1/3)^d − 2^(1−d) + 1.
        for d in 1..=4 {
            let sample = vec![0.0_f64; d];
            let sd = l2_star_discrepancy(&sample, d).unwrap();
            let expected = (1.0_f64 / 3.0).powi(d as i32) - 2.0_f64.powi(1 - d as i32) + 1.0;
            assert!(
                (sd - expected).abs() < 1e-13,
                "d={d}: SD²={sd}, expected={expected}"
            );
        }
    }

    #[test]
    fn l2_star_metamorphic_halton_beats_corner_clustered_2d() {
        let n = 64;
        let d = 2;
        let mut h = HaltonSampler::new(d).unwrap();
        let halton = h.sample(n);
        // All-zeros design: every point at the lower-left corner. SD² is
        // dominated by huge contributions (single sum=N, double sum=N²).
        let clustered = vec![0.0_f64; n * d];
        let sd_halton = l2_star_discrepancy(&halton, d).unwrap();
        let sd_cluster = l2_star_discrepancy(&clustered, d).unwrap();
        assert!(
            sd_halton < sd_cluster,
            "Halton SD²={sd_halton} should beat corner-clustered SD²={sd_cluster}"
        );
    }

    #[test]
    fn mixture_rejects_invalid_inputs() {
        assert!(matches!(
            mixture_discrepancy(&[], 0),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            mixture_discrepancy(&[0.1, 0.2, 0.3], 2),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            mixture_discrepancy(&[1.5], 1),
            Err(StatsError::InvalidArgument(_))
        ));
    }

    #[test]
    fn mixture_empty_sample_is_leading_constant() {
        let d = 3;
        let md = mixture_discrepancy(&[], d).unwrap();
        let expected = (19.0_f64 / 12.0).powi(d as i32);
        assert!(
            (md - expected).abs() < 1e-15,
            "empty MD² should equal (19/12)^d = {expected}, got {md}"
        );
    }

    #[test]
    fn mixture_metamorphic_single_point_at_center() {
        // For n=1 at the center of every dimension, |xi-1/2| = 0 and the
        // single/double products reduce to (5/3)^d / 1 and (15/8)^d / 1.
        // MD² = (19/12)^d - 2·(5/3)^d + (15/8)^d.
        for d in 1..=4 {
            let sample = vec![0.5_f64; d];
            let md = mixture_discrepancy(&sample, d).unwrap();
            let leading = (19.0_f64 / 12.0).powi(d as i32);
            let single = (5.0_f64 / 3.0).powi(d as i32);
            let double = (15.0_f64 / 8.0).powi(d as i32);
            let expected = leading - 2.0 * single + double;
            assert!(
                (md - expected).abs() < 1e-13,
                "d={d}: center-point MD²={md}, expected={expected}"
            );
        }
    }

    #[test]
    fn mixture_metamorphic_halton_beats_aligned_2d() {
        let n = 64;
        let d = 2;
        let mut h = HaltonSampler::new(d).unwrap();
        let halton = h.sample(n);
        let aligned = vec![0.5_f64; n * d];
        let md_halton = mixture_discrepancy(&halton, d).unwrap();
        let md_aligned = mixture_discrepancy(&aligned, d).unwrap();
        assert!(
            md_halton < md_aligned,
            "Halton MD²={md_halton} should beat aligned MD²={md_aligned}"
        );
    }

    #[test]
    fn wraparound_rejects_invalid_inputs() {
        assert!(matches!(
            wraparound_discrepancy(&[], 0),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            wraparound_discrepancy(&[0.1, 0.2, 0.3], 2),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            wraparound_discrepancy(&[0.5, -0.1], 2),
            Err(StatsError::InvalidArgument(_))
        ));
    }

    #[test]
    fn wraparound_metamorphic_single_point_closed_form() {
        // For n=1 in d dimensions the double sum is exactly Π (3/2) = (3/2)^d
        // regardless of the point's coordinates (because xi == xj makes
        // every distance zero). Thus WD² = (3/2)^d - (4/3)^d.
        for d in 1..=4 {
            let mut sample = vec![0.0_f64; d];
            for (k, coordinate) in sample.iter_mut().enumerate().take(d) {
                *coordinate = 0.1 + 0.07 * k as f64;
            }
            let wd = wraparound_discrepancy(&sample, d).unwrap();
            let expected = 1.5_f64.powi(d as i32) - (4.0_f64 / 3.0).powi(d as i32);
            assert!(
                (wd - expected).abs() < 1e-15,
                "d={d}: WD²={wd}, expected={expected}"
            );
        }
    }

    #[test]
    fn wraparound_metamorphic_halton_beats_aligned_points() {
        let n = 64;
        let d = 2;
        let mut h = HaltonSampler::new(d).unwrap();
        let halton = h.sample(n);
        let aligned = vec![0.5_f64; n * d];
        let wd_halton = wraparound_discrepancy(&halton, d).unwrap();
        let wd_aligned = wraparound_discrepancy(&aligned, d).unwrap();
        assert!(
            wd_halton < wd_aligned,
            "Halton WD²={wd_halton} should beat aligned WD²={wd_aligned}"
        );
    }

    #[test]
    fn wraparound_metamorphic_invariant_under_cyclic_shift() {
        // The wraparound L2 discrepancy is, by construction, invariant
        // under any per-dimension cyclic shift on the unit torus —
        // because the kernel K(d) = 3/2 − d(1−d) satisfies K(d) = K(1−d)
        // and shifting all points uniformly preserves pairwise
        // wraparound distances. Verify across shifts in {0.0, 0.13, 0.5,
        // 0.87} for a 4-point 3D Halton sample.
        let d = 3;
        let n = 4;
        let mut h = HaltonSampler::new(d).unwrap();
        let original = h.sample(n);
        let baseline = wraparound_discrepancy(&original, d).unwrap();
        for &shift in &[0.0_f64, 0.13, 0.5, 0.87] {
            let shifted: Vec<f64> = original.iter().map(|x| (x + shift).fract()).collect();
            let wd = wraparound_discrepancy(&shifted, d).unwrap();
            assert!(
                (wd - baseline).abs() < 1e-12,
                "shift={shift}: WD² changed from {baseline} to {wd} — kernel must be torus-invariant"
            );
        }
    }

    #[test]
    fn wraparound_empty_sample_is_negative_leading_constant() {
        // Empty sample: only the -(4/3)^d term survives.
        let d = 2;
        let wd = wraparound_discrepancy(&[], d).unwrap();
        let expected = -(4.0_f64 / 3.0).powi(d as i32);
        assert!((wd - expected).abs() < 1e-15);
    }

    /// Anchor `geometric_discrepancy` against scipy.stats.qmc.geometric_discrepancy
    /// on a deterministic 8-point 2-D Halton fixture (frankenscipy-b8s95).
    #[test]
    fn geometric_discrepancy_matches_scipy_halton_2d() {
        // Coordinates from scipy.stats.qmc.Halton(d=2, seed=42).random(8).
        let sample = vec![
            0.5513058732778853,
            0.15176798070427107,
            0.05130587327788534,
            0.8184346473709377,
            0.8013058732778853,
            0.4851013140376044,
            0.30130587327788534,
            0.262879090904271,
            0.6763058732778853,
            0.9295457584820488,
            0.17630587327788534,
            0.5962124251487155,
            0.9263058732778853,
            0.04065687070427107,
            0.42630587327788534,
            0.7073235362709377,
        ];
        let mindist =
            geometric_discrepancy(&sample, 2, GeometricDiscrepancyMethod::MinDist).unwrap();
        let mst = geometric_discrepancy(&sample, 2, GeometricDiscrepancyMethod::Mst).unwrap();
        // scipy reports 0.25496610764841404 / 0.3286278710953668 to ~16 digits.
        assert!(
            (mindist - 0.25496610764841404).abs() < 1e-10,
            "mindist {mindist} vs scipy 0.25497"
        );
        assert!(
            (mst - 0.3286278710953668).abs() < 1e-10,
            "mst {mst} vs scipy 0.32863"
        );
    }

    #[test]
    fn geometric_discrepancy_validates_inputs() {
        // Empty/single-point samples are rejected.
        let single = vec![0.5_f64, 0.5];
        assert!(
            geometric_discrepancy(&single, 2, GeometricDiscrepancyMethod::MinDist).is_err(),
            "single-point sample must be rejected"
        );
        // Out-of-unit-cube coords are rejected.
        let out = vec![0.5_f64, 1.5];
        assert!(
            geometric_discrepancy(&out, 1, GeometricDiscrepancyMethod::MinDist).is_err(),
            "out-of-unit-cube must be rejected"
        );
        // dimension=0 rejected.
        assert!(
            geometric_discrepancy(&[0.5_f64], 0, GeometricDiscrepancyMethod::MinDist).is_err(),
            "dimension=0 must be rejected"
        );
        // Length not multiple of dimension rejected.
        assert!(
            geometric_discrepancy(&[0.1_f64, 0.2, 0.3], 2, GeometricDiscrepancyMethod::MinDist)
                .is_err(),
            "ragged sample must be rejected"
        );
    }

    #[test]
    fn discrepancy_lhs_is_lower_than_random_offset_1d() {
        // 1D LHS: each cell has exactly one centered sample. CD² ought to
        // be smaller than a degenerate sample where all points cluster
        // (e.g. all at 0.25).
        let n = 32;
        let d = 1;
        let mut lhs = LatinHypercubeSampler::new(d, 0xc0ffee).unwrap();
        let sample = lhs.sample(n);
        let cd_lhs = centered_discrepancy(&sample, d).unwrap();
        let clustered = vec![0.25_f64; n];
        let cd_cluster = centered_discrepancy(&clustered, d).unwrap();
        assert!(
            cd_lhs < cd_cluster,
            "LHS CD²={cd_lhs} should beat clustered CD²={cd_cluster}"
        );
    }
}
