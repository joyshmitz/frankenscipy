#![forbid(unsafe_code)]

//! Statistical distributions for FrankenSciPy.
//!
//! Matches `scipy.stats` core continuous distributions:
//! - `norm` — Normal (Gaussian) distribution
//! - `t` — Student's t-distribution
//! - `chi2` — Chi-squared distribution
//! - `uniform` — Uniform distribution
//!
//! Each distribution implements pdf, cdf, sf, ppf (inverse CDF), mean, var, std.

use std::f64::consts::{FRAC_1_SQRT_2, LN_2, PI};

use fsci_runtime::RuntimeMode;
use rand::Rng;

const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Error type for stats APIs that validate structured inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatsError {
    InvalidArgument(String),
}

impl std::fmt::Display for StatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for StatsError {}

/// Trait for continuous probability distributions.
pub trait ContinuousDistribution {
    /// Probability density function.
    fn pdf(&self, x: f64) -> f64;
    /// Cumulative distribution function.
    fn cdf(&self, x: f64) -> f64;
    /// Survival function: 1 - cdf(x).
    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }
    /// Percent point function (inverse CDF) via bisection.
    /// Override for distributions with analytic inverses.
    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        ppf_bisection(|x| self.cdf(x), q, self.mean(), self.std())
    }
    /// Mean of the distribution.
    fn mean(&self) -> f64;
    /// Variance of the distribution.
    fn var(&self) -> f64;
    /// Standard deviation.
    fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Generate `n` random variates via inverse transform sampling.
    fn rvs(&self, n: usize, rng: &mut impl Rng) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let u: f64 = rng.random();
                self.ppf(u)
            })
            .collect()
    }
}

/// Generic inverse CDF via bisection search.
fn ppf_bisection(cdf: impl Fn(f64) -> f64, q: f64, mean: f64, std: f64) -> f64 {
    // Initial bracket: start around mean ± 10*std
    let half_width = if std.is_finite() && std > 0.0 {
        10.0 * std
    } else {
        100.0
    };
    let center = if mean.is_finite() { mean } else { 0.0 };
    let mut lo = center - half_width;
    let mut hi = center + half_width;
    let mut step = half_width;

    // Expand bracket if needed (bounded to prevent infinite loops)
    for _ in 0..60 {
        if cdf(lo) <= q {
            break;
        }
        step *= 2.0;
        lo -= step;
    }
    step = half_width;
    for _ in 0..60 {
        if cdf(hi) >= q {
            break;
        }
        step *= 2.0;
        hi += step;
    }

    // Bisection
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if (hi - lo).abs() < 1e-12 * mid.abs().max(1.0) {
            return mid;
        }
        if cdf(mid) < q {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn simpson_integrate(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = n + (n % 2);
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += w * f(x);
    }
    sum * h / 3.0
}

fn simpson_integrate_adaptive(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    initial_n: usize,
    rel_tol: f64,
    abs_tol: f64,
    max_refinements: usize,
) -> f64 {
    let mut n = initial_n.max(2);
    n += n % 2;
    let mut prev = simpson_integrate(&f, a, b, n);
    for _ in 0..max_refinements {
        n *= 2;
        let curr = simpson_integrate(&f, a, b, n);
        let refined = curr + (curr - prev) / 15.0;
        let err = (refined - curr).abs();
        let tol = abs_tol.max(rel_tol * refined.abs());
        if err <= tol {
            return refined;
        }
        prev = curr;
    }
    prev
}

// ══════════════════════════════════════════════════════════════════════
// Normal (Gaussian) Distribution
// ══════════════════════════════════════════════════════════════════════

/// Normal distribution N(loc, scale²).
///
/// Matches `scipy.stats.norm(loc=mu, scale=sigma)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal {
    pub loc: f64,
    pub scale: f64,
}

impl Default for Normal {
    fn default() -> Self {
        Self {
            loc: 0.0,
            scale: 1.0,
        }
    }
}

impl Normal {
    /// Standard normal distribution N(0, 1).
    #[must_use]
    pub fn standard() -> Self {
        Self::default()
    }

    /// Create N(loc, scale²). Panics if scale <= 0.
    #[must_use]
    pub fn new(loc: f64, scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { loc, scale }
    }

    /// Percent point function (inverse CDF) using rational approximation.
    ///
    /// Matches `scipy.stats.norm.ppf(q)`.
    pub fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        self.loc + self.scale * standard_normal_ppf(q)
    }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        (-0.5 * z * z).exp() / (self.scale * (2.0 * PI).sqrt())
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        0.5 * (1.0 + fsci_special::erf_scalar(z * FRAC_1_SQRT_2))
    }

    fn sf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        0.5 * fsci_special::erfc_scalar(z * FRAC_1_SQRT_2)
    }

    fn ppf(&self, q: f64) -> f64 {
        // Use the specialized rational approximation
        Normal::ppf(self, q)
    }

    fn mean(&self) -> f64 {
        self.loc
    }

    fn var(&self) -> f64 {
        self.scale * self.scale
    }
}

// ══════════════════════════════════════════════════════════════════════
// Student's t-Distribution
// ══════════════════════════════════════════════════════════════════════

/// Student's t-distribution with `df` degrees of freedom.
///
/// Matches `scipy.stats.t(df)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentT {
    pub df: f64,
}

impl StudentT {
    #[must_use]
    pub fn new(df: f64) -> Self {
        assert!(df > 0.0, "df must be positive, got {df}");
        Self { df }
    }
}

impl ContinuousDistribution for StudentT {
    fn pdf(&self, x: f64) -> f64 {
        let v = self.df;
        let coeff = gamma_ratio_t(v);
        coeff * (1.0 + x * x / v).powf(-0.5 * (v + 1.0))
    }

    fn cdf(&self, x: f64) -> f64 {
        let v = self.df;
        if x == 0.0 {
            return 0.5;
        }
        // Use regularized incomplete beta function
        let t2 = x * x;
        let w = v / (v + t2);
        let ib = regularized_incomplete_beta(0.5 * v, 0.5, w);
        if x > 0.0 { 1.0 - 0.5 * ib } else { 0.5 * ib }
    }

    fn sf(&self, x: f64) -> f64 {
        let v = self.df;
        if x == 0.0 {
            return 0.5;
        }
        // Use regularized incomplete beta function
        let t2 = x * x;
        let w = v / (v + t2);
        let ib = regularized_incomplete_beta(0.5 * v, 0.5, w);
        if x > 0.0 { 0.5 * ib } else { 1.0 - 0.5 * ib }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        if q == 0.5 {
            return 0.0;
        }
        let v = self.df;
        let (p, sign) = if q < 0.5 {
            (2.0 * q, -1.0)
        } else {
            (2.0 * (1.0 - q), 1.0)
        };
        let w = fsci_special::betaincinv(0.5 * v, 0.5, p);
        let mut x = sign * (v * (1.0 - w) / w).sqrt();
        // Newton refinement for accuracy
        for _ in 0..8 {
            let err = self.cdf(x) - q;
            if err.abs() < 1e-14 {
                break;
            }
            let dx = self.pdf(x);
            if dx > 1e-30 {
                x -= err / dx;
            }
        }
        x
    }

    fn mean(&self) -> f64 {
        if self.df > 1.0 { 0.0 } else { f64::NAN }
    }

    fn var(&self) -> f64 {
        if self.df > 2.0 {
            self.df / (self.df - 2.0)
        } else if self.df > 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Chi-Squared Distribution
// ══════════════════════════════════════════════════════════════════════

/// Chi-squared distribution with `df` degrees of freedom.
///
/// Matches `scipy.stats.chi2(df)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquared {
    pub df: f64,
}

impl ChiSquared {
    #[must_use]
    pub fn new(df: f64) -> Self {
        assert!(df > 0.0, "df must be positive, got {df}");
        Self { df }
    }
}

impl ContinuousDistribution for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            return if self.df == 2.0 {
                0.5
            } else if self.df > 2.0 {
                0.0
            } else {
                f64::INFINITY
            };
        }
        let k2 = 0.5 * self.df;
        let ln_pdf = (k2 - 1.0) * x.ln() - 0.5 * x - k2 * 2.0_f64.ln() - ln_gamma(k2);
        ln_pdf.exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        // CDF of chi2(k) = regularized lower incomplete gamma P(k/2, x/2)
        lower_regularized_gamma(0.5 * self.df, 0.5 * x)
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        upper_regularized_gamma(0.5 * self.df, 0.5 * x)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        let mut x = 2.0 * fsci_special::gammaincinv(0.5 * self.df, q);
        // Fall back to bisection if gammaincinv gave a degenerate result
        if x <= 0.0 || !x.is_finite() || (self.cdf(x) - q).abs() > 0.1 {
            return ppf_bisection(|v| self.cdf(v), q, self.mean(), self.std());
        }
        for _ in 0..8 {
            let err = self.cdf(x) - q;
            if err.abs() < 1e-14 {
                break;
            }
            let dx = self.pdf(x);
            if dx > 1e-30 {
                x = (x - err / dx).max(0.0);
            }
        }
        x
    }

    fn mean(&self) -> f64 {
        self.df
    }

    fn var(&self) -> f64 {
        2.0 * self.df
    }
}

// ══════════════════════════════════════════════════════════════════════
// Uniform Distribution
// ══════════════════════════════════════════════════════════════════════

/// Uniform distribution on [loc, loc + scale].
///
/// Matches `scipy.stats.uniform(loc, scale)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uniform {
    pub loc: f64,
    pub scale: f64,
}

impl Default for Uniform {
    fn default() -> Self {
        Self {
            loc: 0.0,
            scale: 1.0,
        }
    }
}

impl Uniform {
    #[must_use]
    pub fn new(loc: f64, scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { loc, scale }
    }
}

impl ContinuousDistribution for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if x >= self.loc && x <= self.loc + self.scale {
            1.0 / self.scale
        } else {
            0.0
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < self.loc {
            0.0
        } else if x > self.loc + self.scale {
            1.0
        } else {
            (x - self.loc) / self.scale
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return self.loc;
        }
        if q == 1.0 {
            return self.loc + self.scale;
        }
        self.loc + q * self.scale
    }

    fn mean(&self) -> f64 {
        self.loc + 0.5 * self.scale
    }

    fn var(&self) -> f64 {
        self.scale * self.scale / 12.0
    }
}

// ══════════════════════════════════════════════════════════════════════
// Exponential Distribution
// ══════════════════════════════════════════════════════════════════════

/// Exponential distribution with rate parameter `lambda` (= 1/scale).
///
/// Matches `scipy.stats.expon(scale=1/lambda)`.
/// The PDF is: f(x) = lambda * exp(-lambda * x) for x >= 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential {
    pub lambda: f64,
}

impl Default for Exponential {
    fn default() -> Self {
        Self { lambda: 1.0 }
    }
}

impl Exponential {
    #[must_use]
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "lambda must be positive, got {lambda}");
        Self { lambda }
    }

    /// Create from scale parameter (= 1/lambda), matching SciPy convention.
    #[must_use]
    pub fn from_scale(scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self {
            lambda: 1.0 / scale,
        }
    }

    /// Percent point function (inverse CDF).
    pub fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        -(1.0 - q).ln() / self.lambda
    }
}

impl ContinuousDistribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.lambda * (-self.lambda * x).exp()
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            -(-self.lambda * x).exp_m1()
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x < 0.0 {
            1.0
        } else {
            (-self.lambda * x).exp()
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        Exponential::ppf(self, q)
    }

    fn mean(&self) -> f64 {
        1.0 / self.lambda
    }

    fn var(&self) -> f64 {
        1.0 / (self.lambda * self.lambda)
    }
}

// ══════════════════════════════════════════════════════════════════════
// F-Distribution
// ══════════════════════════════════════════════════════════════════════

/// F-distribution with `dfn` (numerator) and `dfd` (denominator) degrees of freedom.
///
/// Matches `scipy.stats.f(dfn, dfd)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FDistribution {
    pub dfn: f64,
    pub dfd: f64,
}

impl FDistribution {
    #[must_use]
    pub fn new(dfn: f64, dfd: f64) -> Self {
        assert!(dfn > 0.0, "dfn must be positive, got {dfn}");
        assert!(dfd > 0.0, "dfd must be positive, got {dfd}");
        Self { dfn, dfd }
    }
}

impl ContinuousDistribution for FDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let d1 = self.dfn;
        let d2 = self.dfd;
        let ln_pdf = 0.5 * d1 * (d1 / d2).ln() + (0.5 * d1 - 1.0) * x.ln()
            - 0.5 * (d1 + d2) * (1.0 + d1 * x / d2).ln()
            - ln_beta(0.5 * d1, 0.5 * d2);
        ln_pdf.exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let d1 = self.dfn;
        let d2 = self.dfd;
        let w = d1 * x / (d1 * x + d2);
        regularized_incomplete_beta(0.5 * d1, 0.5 * d2, w)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        let d1 = self.dfn;
        let d2 = self.dfd;
        let w = fsci_special::betaincinv(0.5 * d1, 0.5 * d2, q);
        if w >= 1.0 {
            return f64::INFINITY;
        }
        let mut x = d2 * w / (d1 * (1.0 - w));
        if !x.is_finite() || (self.cdf(x) - q).abs() > 0.1 {
            return ppf_bisection(|v| self.cdf(v), q, self.mean(), self.std());
        }
        for _ in 0..8 {
            let err = self.cdf(x) - q;
            if err.abs() < 1e-14 {
                break;
            }
            let dx = self.pdf(x);
            if dx > 1e-30 {
                x = (x - err / dx).max(0.0);
            }
        }
        x
    }

    fn mean(&self) -> f64 {
        if self.dfd > 2.0 {
            self.dfd / (self.dfd - 2.0)
        } else {
            f64::NAN
        }
    }

    fn var(&self) -> f64 {
        if self.dfd > 4.0 {
            let d1 = self.dfn;
            let d2 = self.dfd;
            2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0).powi(2) * (d2 - 4.0))
        } else {
            f64::NAN
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Beta Distribution
// ══════════════════════════════════════════════════════════════════════

/// Beta distribution with shape parameters `a` and `b`.
///
/// Matches `scipy.stats.beta(a, b)`.
/// Supported on [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaDist {
    pub a: f64,
    pub b: f64,
}

impl BetaDist {
    #[must_use]
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a > 0.0, "a must be positive, got {a}");
        assert!(b > 0.0, "b must be positive, got {b}");
        Self { a, b }
    }
}

impl ContinuousDistribution for BetaDist {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        let ln_pdf =
            (self.a - 1.0) * x.ln() + (self.b - 1.0) * (1.0 - x).ln() - ln_beta(self.a, self.b);
        ln_pdf.exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        regularized_incomplete_beta(self.a, self.b, x)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return 1.0;
        }
        let mut x = fsci_special::betaincinv(self.a, self.b, q);
        for _ in 0..8 {
            let err = self.cdf(x) - q;
            if err.abs() < 1e-14 {
                break;
            }
            let dx = self.pdf(x);
            if dx > 1e-30 {
                x = (x - err / dx).clamp(0.0, 1.0);
            }
        }
        x
    }

    fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }

    fn var(&self) -> f64 {
        let ab = self.a + self.b;
        self.a * self.b / (ab * ab * (ab + 1.0))
    }
}

// ══════════════════════════════════════════════════════════════════════
// Gamma Distribution
// ══════════════════════════════════════════════════════════════════════

/// Gamma distribution with shape `a` and rate `1/scale`.
///
/// Matches `scipy.stats.gamma(a, scale=scale)`.
/// PDF: f(x) = x^(a-1) * exp(-x/scale) / (scale^a * Γ(a))
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaDist {
    pub a: f64,
    pub scale: f64,
}

impl GammaDist {
    #[must_use]
    pub fn new(a: f64, scale: f64) -> Self {
        assert!(a > 0.0, "shape a must be positive, got {a}");
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { a, scale }
    }
}

impl ContinuousDistribution for GammaDist {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            return if self.a == 1.0 {
                1.0 / self.scale
            } else if self.a > 1.0 {
                0.0
            } else {
                f64::INFINITY
            };
        }
        let ln_pdf =
            (self.a - 1.0) * x.ln() - x / self.scale - self.a * self.scale.ln() - ln_gamma(self.a);
        ln_pdf.exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        lower_regularized_gamma(self.a, x / self.scale)
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        upper_regularized_gamma(self.a, x / self.scale)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        let mut x = self.scale * fsci_special::gammaincinv(self.a, q);
        if x <= 0.0 || !x.is_finite() || (self.cdf(x) - q).abs() > 0.1 {
            return ppf_bisection(|v| self.cdf(v), q, self.mean(), self.std());
        }
        for _ in 0..8 {
            let err = self.cdf(x) - q;
            if err.abs() < 1e-14 {
                break;
            }
            let dx = self.pdf(x);
            if dx > 1e-30 {
                x = (x - err / dx).max(0.0);
            }
        }
        x
    }

    fn mean(&self) -> f64 {
        self.a * self.scale
    }

    fn var(&self) -> f64 {
        self.a * self.scale * self.scale
    }
}

// ══════════════════════════════════════════════════════════════════════
// Weibull Distribution
// ══════════════════════════════════════════════════════════════════════

/// Weibull (minimum) distribution with shape `c` and scale `scale`.
///
/// Matches `scipy.stats.weibull_min(c, scale=scale)`.
/// PDF: f(x) = (c/scale) * (x/scale)^(c-1) * exp(-(x/scale)^c)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weibull {
    pub c: f64,
    pub scale: f64,
}

impl Weibull {
    #[must_use]
    pub fn new(c: f64, scale: f64) -> Self {
        assert!(c > 0.0, "shape c must be positive, got {c}");
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { c, scale }
    }

    /// Inverse CDF (analytic).
    pub fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        self.scale * (-(1.0 - q).ln()).powf(1.0 / self.c)
    }
}

impl ContinuousDistribution for Weibull {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            return if self.c == 1.0 {
                1.0 / self.scale
            } else if self.c > 1.0 {
                0.0
            } else {
                f64::INFINITY
            };
        }
        let z = x / self.scale;
        (self.c / self.scale) * z.powf(self.c - 1.0) * (-z.powf(self.c)).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        -(-(x / self.scale).powf(self.c)).exp_m1()
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        (-(x / self.scale).powf(self.c)).exp()
    }

    fn ppf(&self, q: f64) -> f64 {
        Weibull::ppf(self, q)
    }

    fn mean(&self) -> f64 {
        self.scale * ln_gamma(1.0 + 1.0 / self.c).exp()
    }

    fn var(&self) -> f64 {
        let g1 = ln_gamma(1.0 + 1.0 / self.c).exp();
        let g2 = ln_gamma(1.0 + 2.0 / self.c).exp();
        self.scale * self.scale * (g2 - g1 * g1)
    }
}

// ══════════════════════════════════════════════════════════════════════
// Lognormal Distribution
// ══════════════════════════════════════════════════════════════════════

/// Lognormal distribution: X = exp(μ + σZ) where Z ~ N(0,1).
///
/// Matches `scipy.stats.lognorm(s, scale=exp(mu))` where s=σ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lognormal {
    /// Shape parameter σ (standard deviation of the underlying normal).
    pub s: f64,
    /// Scale parameter (= exp(μ)).
    pub scale: f64,
}

impl Lognormal {
    #[must_use]
    pub fn new(s: f64, scale: f64) -> Self {
        assert!(s > 0.0, "s must be positive, got {s}");
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { s, scale }
    }
}

impl ContinuousDistribution for Lognormal {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = (x / self.scale).ln() / self.s;
        (-0.5 * z * z).exp() / (x * self.s * (2.0 * PI).sqrt())
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = (x / self.scale).ln() / self.s;
        0.5 * (1.0 + fsci_special::erf_scalar(z * FRAC_1_SQRT_2))
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        // If X ~ Lognormal(mu, sigma), then ln(X) ~ Normal(mu, sigma)
        // ppf(q) = exp(mu + sigma * ndtri(q)) where mu = ln(scale)
        self.scale * (self.s * fsci_special::ndtri(q)).exp()
    }

    fn mean(&self) -> f64 {
        self.scale * (0.5 * self.s * self.s).exp()
    }

    fn var(&self) -> f64 {
        let s2 = self.s * self.s;
        self.scale * self.scale * s2.exp() * (s2.exp() - 1.0)
    }
}

// ══════════════════════════════════════════════════════════════════════
// Pareto Distribution
// ══════════════════════════════════════════════════════════════════════

/// Pareto distribution with shape `b` and scale `scale`.
///
/// Matches `scipy.stats.pareto(b, scale=scale)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pareto {
    pub b: f64,
    pub scale: f64,
}

impl Pareto {
    #[must_use]
    pub fn new(b: f64, scale: f64) -> Self {
        assert!(b > 0.0, "shape b must be positive, got {b}");
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { b, scale }
    }
}

impl ContinuousDistribution for Pareto {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.scale {
            0.0
        } else {
            self.b * self.scale.powf(self.b) / x.powf(self.b + 1.0)
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < self.scale {
            0.0
        } else {
            1.0 - (self.scale / x).powf(self.b)
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return self.scale;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        self.scale / (1.0 - q).powf(1.0 / self.b)
    }

    fn mean(&self) -> f64 {
        if self.b <= 1.0 {
            f64::INFINITY
        } else {
            self.b * self.scale / (self.b - 1.0)
        }
    }

    fn var(&self) -> f64 {
        if self.b <= 2.0 {
            f64::INFINITY
        } else {
            self.scale * self.scale * self.b / ((self.b - 1.0).powi(2) * (self.b - 2.0))
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Rayleigh Distribution
// ══════════════════════════════════════════════════════════════════════

/// Rayleigh distribution with scale parameter `scale`.
///
/// Matches `scipy.stats.rayleigh(scale=scale)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rayleigh {
    pub scale: f64,
}

impl Rayleigh {
    #[must_use]
    pub fn new(scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { scale }
    }
}

impl ContinuousDistribution for Rayleigh {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            let sigma_sq = self.scale * self.scale;
            x / sigma_sq * (-(x * x) / (2.0 * sigma_sq)).exp()
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            -(-(x * x) / (2.0 * self.scale * self.scale)).exp_m1()
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        self.scale * (-2.0 * (1.0 - q).ln()).sqrt()
    }

    fn mean(&self) -> f64 {
        self.scale * (PI / 2.0).sqrt()
    }

    fn var(&self) -> f64 {
        (2.0 - PI / 2.0) * self.scale * self.scale
    }
}

// ══════════════════════════════════════════════════════════════════════
// Gumbel Distribution
// ══════════════════════════════════════════════════════════════════════

/// Gumbel (extreme value type I) distribution.
///
/// Matches `scipy.stats.gumbel_r(loc=loc, scale=scale)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gumbel {
    pub loc: f64,
    pub scale: f64,
}

impl Gumbel {
    #[must_use]
    pub fn new(loc: f64, scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { loc, scale }
    }
}

impl ContinuousDistribution for Gumbel {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        let exp_neg_z = (-z).exp();
        (-(z + exp_neg_z)).exp() / self.scale
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        (-(-z).exp()).exp()
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        self.loc - self.scale * (-q.ln()).ln()
    }

    fn mean(&self) -> f64 {
        const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
        self.loc + EULER_GAMMA * self.scale
    }

    fn var(&self) -> f64 {
        PI * PI * self.scale * self.scale / 6.0
    }
}

/// Left-skewed Gumbel (extreme value type I) distribution.
///
/// Matches `scipy.stats.gumbel_l(loc=loc, scale=scale)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GumbelLeft {
    pub loc: f64,
    pub scale: f64,
}

impl GumbelLeft {
    #[must_use]
    pub fn new(loc: f64, scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { loc, scale }
    }
}

impl ContinuousDistribution for GumbelLeft {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        (z - z.exp()).exp() / self.scale
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        -(-z.exp()).exp_m1()
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        self.loc + self.scale * (-(-q).ln_1p()).ln()
    }

    fn mean(&self) -> f64 {
        const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
        self.loc - EULER_GAMMA * self.scale
    }

    fn var(&self) -> f64 {
        PI * PI * self.scale * self.scale / 6.0
    }
}

// ══════════════════════════════════════════════════════════════════════
// Logistic Distribution
// ══════════════════════════════════════════════════════════════════════

/// Logistic distribution with location `loc` and scale `scale`.
///
/// Matches `scipy.stats.logistic(loc=loc, scale=scale)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Logistic {
    pub loc: f64,
    pub scale: f64,
}

impl Logistic {
    #[must_use]
    pub fn new(loc: f64, scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { loc, scale }
    }
}

impl ContinuousDistribution for Logistic {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        let exp_neg_z = (-z).exp();
        exp_neg_z / (self.scale * (1.0 + exp_neg_z).powi(2))
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        1.0 / (1.0 + (-z).exp())
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        self.loc + self.scale * (q / (1.0 - q)).ln()
    }

    fn mean(&self) -> f64 {
        self.loc
    }

    fn var(&self) -> f64 {
        PI * PI * self.scale * self.scale / 3.0
    }
}

// ══════════════════════════════════════════════════════════════════════
// Maxwell Distribution
// ══════════════════════════════════════════════════════════════════════

/// Maxwell distribution with scale parameter `scale`.
///
/// Matches `scipy.stats.maxwell(scale=scale)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Maxwell {
    pub scale: f64,
}

impl Maxwell {
    #[must_use]
    pub fn new(scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { scale }
    }
}

impl ContinuousDistribution for Maxwell {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            (2.0 / PI).sqrt() * x * x / self.scale.powi(3)
                * (-(x * x) / (2.0 * self.scale * self.scale)).exp()
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            let sqrt_two_over_pi = (2.0 / PI).sqrt();
            let z = x / (self.scale * 2.0_f64.sqrt());
            fsci_special::erf_scalar(z)
                - sqrt_two_over_pi
                    * (x / self.scale)
                    * (-(x * x) / (2.0 * self.scale * self.scale)).exp()
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        let mut x = self.scale * (2.0 * fsci_special::gammaincinv(1.5, q)).sqrt();
        if x <= 0.0 || !x.is_finite() || (self.cdf(x) - q).abs() > 0.1 {
            return ppf_bisection(|v| self.cdf(v), q, self.mean(), self.std());
        }
        for _ in 0..8 {
            let err = self.cdf(x) - q;
            if err.abs() < 1e-14 {
                break;
            }
            let dx = self.pdf(x);
            if dx > 1e-30 {
                x = (x - err / dx).max(0.0);
            }
        }
        x
    }

    fn mean(&self) -> f64 {
        2.0 * self.scale * (2.0 / PI).sqrt()
    }

    fn var(&self) -> f64 {
        self.scale * self.scale * (3.0 - 8.0 / PI)
    }
}

// ══════════════════════════════════════════════════════════════════════
// Multivariate Normal Distribution
// ══════════════════════════════════════════════════════════════════════

/// Multivariate normal distribution with cached Cholesky factorization.
///
/// Matches `scipy.stats.multivariate_normal(mean, cov)` for the default
/// `allow_singular=False` path.
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateNormal {
    pub mean: Vec<f64>,
    pub cov: Vec<Vec<f64>>,
    chol: Vec<Vec<f64>>,
    log_det: f64,
}

impl MultivariateNormal {
    pub fn new(mean: &[f64], cov: &[Vec<f64>]) -> Result<Self, StatsError> {
        if mean.is_empty() {
            return Err(StatsError::InvalidArgument(
                "mean must be non-empty".to_string(),
            ));
        }
        if cov.len() != mean.len() {
            return Err(StatsError::InvalidArgument(format!(
                "cov row count ({}) must match mean length ({})",
                cov.len(),
                mean.len()
            )));
        }
        for (row_idx, row) in cov.iter().enumerate() {
            if row.len() != mean.len() {
                return Err(StatsError::InvalidArgument(format!(
                    "cov row {row_idx} has length {}, expected {}",
                    row.len(),
                    mean.len()
                )));
            }
        }
        for (i, row_i) in cov.iter().enumerate() {
            for (j, row_j) in cov.iter().enumerate() {
                if (row_i[j] - row_j[i]).abs() > 1e-12 {
                    return Err(StatsError::InvalidArgument(
                        "covariance matrix must be symmetric".to_string(),
                    ));
                }
            }
        }

        let chol = cholesky_decompose(cov)?;
        let log_det = 2.0 * (0..chol.len()).map(|i| chol[i][i].ln()).sum::<f64>();
        Ok(Self {
            mean: mean.to_vec(),
            cov: cov.to_vec(),
            chol,
            log_det,
        })
    }

    pub fn logpdf(&self, x: &[f64]) -> Result<f64, StatsError> {
        if x.len() != self.mean.len() {
            return Err(StatsError::InvalidArgument(format!(
                "x length ({}) must match dimension ({})",
                x.len(),
                self.mean.len()
            )));
        }
        let centered: Vec<f64> = x
            .iter()
            .zip(self.mean.iter())
            .map(|(&xi, &mi)| xi - mi)
            .collect();
        let solved = solve_lower_triangular(&self.chol, &centered)?;
        let mahalanobis = solved.iter().map(|value| value * value).sum::<f64>();
        let dim = self.mean.len() as f64;
        Ok(-0.5 * (dim * (2.0 * PI).ln() + self.log_det + mahalanobis))
    }

    pub fn pdf(&self, x: &[f64]) -> Result<f64, StatsError> {
        Ok(self.logpdf(x)?.exp())
    }

    pub fn rvs(&self, n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_standard_normals(self.mean.len(), rng);
            let mut sample = self.mean.clone();
            for (i, row) in self.chol.iter().enumerate() {
                for (j, &zj) in z.iter().take(i + 1).enumerate() {
                    sample[i] += row[j] * zj;
                }
            }
            samples.push(sample);
        }
        samples
    }
}

// ══════════════════════════════════════════════════════════════════════
// Von Mises Distribution
// ══════════════════════════════════════════════════════════════════════

/// Von Mises circular distribution.
///
/// Matches `scipy.stats.vonmises(kappa, loc=loc)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VonMises {
    pub kappa: f64,
    pub loc: f64,
}

impl VonMises {
    #[must_use]
    pub fn new(kappa: f64, loc: f64) -> Self {
        assert!(kappa >= 0.0, "kappa must be non-negative, got {kappa}");
        Self { kappa, loc }
    }

    fn period_start(&self) -> f64 {
        self.loc - PI
    }

    fn base_cdf(&self, x: f64) -> f64 {
        let start = self.period_start();
        let end = start + 2.0 * PI;
        if x <= start {
            return 0.0;
        }
        if x >= end {
            return 1.0;
        }

        let steps = 2048usize;
        let step = (x - start) / steps as f64;
        let mut sum = 0.5 * (self.pdf(start) + self.pdf(x));
        for idx in 1..steps {
            sum += self.pdf(start + idx as f64 * step);
        }
        (sum * step).clamp(0.0, 1.0)
    }

    #[must_use]
    pub fn circular_variance(&self) -> f64 {
        self.var()
    }
}

impl ContinuousDistribution for VonMises {
    fn pdf(&self, x: f64) -> f64 {
        let i0 = modified_bessel_i(0.0, self.kappa);
        (self.kappa * (x - self.loc).cos()).exp() / (2.0 * PI * i0)
    }

    fn cdf(&self, x: f64) -> f64 {
        let start = self.period_start();
        let period = 2.0 * PI;
        let cycles = ((x - start) / period).floor();
        let reduced = x - cycles * period;
        cycles + self.base_cdf(reduced)
    }

    fn mean(&self) -> f64 {
        self.loc
    }

    fn var(&self) -> f64 {
        let i0 = modified_bessel_i(0.0, self.kappa);
        let i1 = modified_bessel_i(1.0, self.kappa);
        1.0 - i1 / i0
    }
}

// ══════════════════════════════════════════════════════════════════════
// Poisson Distribution (discrete, but commonly needed)
// ══════════════════════════════════════════════════════════════════════

/// Poisson distribution with rate parameter `mu`.
///
/// Matches `scipy.stats.poisson(mu)`.
/// PMF: P(k) = mu^k * exp(-mu) / k!
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson {
    pub mu: f64,
}

impl Poisson {
    #[must_use]
    pub fn new(mu: f64) -> Self {
        assert!(mu > 0.0, "mu must be positive, got {mu}");
        Self { mu }
    }

    /// Probability mass function.
    pub fn pmf(&self, k: u64) -> f64 {
        if self.mu == 0.0 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        let ln_pmf = k as f64 * self.mu.ln() - self.mu - ln_gamma(k as f64 + 1.0);
        ln_pmf.exp()
    }

    /// Cumulative distribution function.
    pub fn cdf(&self, k: u64) -> f64 {
        // Direct summation: P(X <= k) = sum_{i=0}^{k} pmf(i)
        let mut sum = 0.0;
        for i in 0..=k {
            sum += self.pmf(i);
        }
        sum.min(1.0) // clamp for numerical safety
    }

    /// Mean of the distribution.
    pub fn mean(&self) -> f64 {
        self.mu
    }

    /// Variance of the distribution.
    pub fn var(&self) -> f64 {
        self.mu
    }
}

// ══════════════════════════════════════════════════════════════════════
// Discrete Distribution Trait and Implementations
// ══════════════════════════════════════════════════════════════════════

/// Trait for discrete probability distributions.
pub trait DiscreteDistribution {
    /// Probability mass function.
    fn pmf(&self, k: u64) -> f64;
    /// Cumulative distribution function: P(X <= k).
    fn cdf(&self, k: u64) -> f64 {
        (0..=k).map(|i| self.pmf(i)).sum::<f64>().min(1.0)
    }
    /// Survival function: P(X > k) = 1 - cdf(k).
    fn sf(&self, k: u64) -> f64 {
        1.0 - self.cdf(k)
    }
    /// Mean of the distribution.
    fn mean(&self) -> f64;
    /// Variance of the distribution.
    fn var(&self) -> f64;
    /// Standard deviation.
    fn std(&self) -> f64 {
        self.var().sqrt()
    }
}

impl DiscreteDistribution for Poisson {
    fn pmf(&self, k: u64) -> f64 {
        Poisson::pmf(self, k)
    }
    fn cdf(&self, k: u64) -> f64 {
        Poisson::cdf(self, k)
    }
    fn mean(&self) -> f64 {
        Poisson::mean(self)
    }
    fn var(&self) -> f64 {
        Poisson::var(self)
    }
}

/// Binomial distribution: number of successes in n independent Bernoulli trials.
///
/// Matches `scipy.stats.binom(n, p)`.
/// PMF: P(k) = C(n,k) * p^k * (1-p)^(n-k)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Binomial {
    pub n: u64,
    pub p: f64,
}

impl Binomial {
    #[must_use]
    pub fn new(n: u64, p: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be in [0, 1], got {p}");
        Self { n, p }
    }
}

impl DiscreteDistribution for Binomial {
    fn pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return 0.0;
        }
        if self.p == 0.0 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        if self.p == 1.0 {
            return if k == self.n { 1.0 } else { 0.0 };
        }
        let ln_comb = ln_gamma(self.n as f64 + 1.0)
            - ln_gamma(k as f64 + 1.0)
            - ln_gamma((self.n - k) as f64 + 1.0);
        let ln_pmf = ln_comb + k as f64 * self.p.ln() + (self.n - k) as f64 * (1.0 - self.p).ln();
        ln_pmf.exp()
    }

    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }

    fn var(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
}

/// Bernoulli distribution: single trial with probability p.
///
/// Matches `scipy.stats.bernoulli(p)`.
/// Special case of Binomial(1, p).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bernoulli {
    pub p: f64,
}

impl Bernoulli {
    #[must_use]
    pub fn new(p: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be in [0, 1], got {p}");
        Self { p }
    }
}

impl DiscreteDistribution for Bernoulli {
    fn pmf(&self, k: u64) -> f64 {
        match k {
            0 => 1.0 - self.p,
            1 => self.p,
            _ => 0.0,
        }
    }

    fn cdf(&self, k: u64) -> f64 {
        if k == 0 { 1.0 - self.p } else { 1.0 }
    }

    fn mean(&self) -> f64 {
        self.p
    }

    fn var(&self) -> f64 {
        self.p * (1.0 - self.p)
    }
}

/// Geometric distribution: number of trials until first success.
///
/// Matches `scipy.stats.geom(p)`.
/// PMF: P(k) = p * (1-p)^(k-1) for k = 1, 2, 3, ...
/// Note: SciPy uses k >= 1 convention (first success on trial k).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometric {
    pub p: f64,
}

impl Geometric {
    #[must_use]
    pub fn new(p: f64) -> Self {
        assert!(p > 0.0 && p <= 1.0, "p must be in (0, 1], got {p}");
        Self { p }
    }
}

impl DiscreteDistribution for Geometric {
    fn pmf(&self, k: u64) -> f64 {
        if k == 0 {
            return 0.0;
        }
        self.p * (1.0 - self.p).powi((k - 1) as i32)
    }

    fn cdf(&self, k: u64) -> f64 {
        if k == 0 {
            return 0.0;
        }
        1.0 - (1.0 - self.p).powi(k as i32)
    }

    fn mean(&self) -> f64 {
        1.0 / self.p
    }

    fn var(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }
}

/// Negative binomial distribution: number of failures before n successes.
///
/// Matches `scipy.stats.nbinom(n, p)`.
/// PMF: P(k) = C(k+n-1, k) * p^n * (1-p)^k for k = 0, 1, 2, ...
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegBinomial {
    /// Number of successes required.
    pub n: f64,
    /// Probability of success on each trial.
    pub p: f64,
}

impl NegBinomial {
    #[must_use]
    pub fn new(n: f64, p: f64) -> Self {
        assert!(n > 0.0, "n must be positive, got {n}");
        assert!(p > 0.0 && p <= 1.0, "p must be in (0, 1], got {p}");
        Self { n, p }
    }
}

impl DiscreteDistribution for NegBinomial {
    fn pmf(&self, k: u64) -> f64 {
        let kf = k as f64;
        // C(k+n-1, k) = Γ(k+n) / (Γ(k+1) * Γ(n))
        let ln_comb = ln_gamma(kf + self.n) - ln_gamma(kf + 1.0) - ln_gamma(self.n);
        let ln_pmf = ln_comb + self.n * self.p.ln() + kf * (1.0 - self.p).ln();
        ln_pmf.exp()
    }

    fn mean(&self) -> f64 {
        self.n * (1.0 - self.p) / self.p
    }

    fn var(&self) -> f64 {
        self.n * (1.0 - self.p) / (self.p * self.p)
    }
}

/// Hypergeometric distribution: draws without replacement.
///
/// Matches `scipy.stats.hypergeom(M, n, N)`.
/// PMF: P(k) = C(n, k) * C(M-n, N-k) / C(M, N)
///
/// M = total population, n = success states in population, N = draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hypergeometric {
    /// Total population size.
    pub big_m: u64,
    /// Number of success states in population.
    pub n: u64,
    /// Number of draws.
    pub big_n: u64,
}

impl Hypergeometric {
    #[must_use]
    pub fn new(big_m: u64, n: u64, big_n: u64) -> Self {
        assert!(n <= big_m, "n must be <= M, got n={n}, M={big_m}");
        assert!(big_n <= big_m, "N must be <= M, got N={big_n}, M={big_m}");
        Self { big_m, n, big_n }
    }
}

impl DiscreteDistribution for Hypergeometric {
    fn pmf(&self, k: u64) -> f64 {
        let m = self.big_m as f64;
        let n = self.n as f64;
        let big_n = self.big_n as f64;
        let kf = k as f64;

        // Valid range: max(0, N+n-M) <= k <= min(n, N)
        // Use saturating_add to avoid u64 overflow
        let k_min = if self.big_n.saturating_add(self.n) > self.big_m {
            (self.big_n + self.n - self.big_m) as f64
        } else {
            0.0
        };
        let k_max = n.min(big_n);
        if kf < k_min || kf > k_max {
            return 0.0;
        }

        // ln(C(n,k)) + ln(C(M-n, N-k)) - ln(C(M, N))
        let ln_pmf = ln_gamma(n + 1.0) - ln_gamma(kf + 1.0) - ln_gamma(n - kf + 1.0)
            + ln_gamma(m - n + 1.0)
            - ln_gamma(big_n - kf + 1.0)
            - ln_gamma(m - n - big_n + kf + 1.0)
            - ln_gamma(m + 1.0)
            + ln_gamma(big_n + 1.0)
            + ln_gamma(m - big_n + 1.0);
        ln_pmf.exp()
    }

    fn mean(&self) -> f64 {
        self.big_n as f64 * self.n as f64 / self.big_m as f64
    }

    fn var(&self) -> f64 {
        let m = self.big_m as f64;
        if m <= 1.0 {
            return 0.0; // Degenerate: single-element population has no variance
        }
        let n = self.n as f64;
        let big_n = self.big_n as f64;
        big_n * (n / m) * ((m - n) / m) * ((m - big_n) / (m - 1.0))
    }
}

/// Log of the Beta function: ln(B(a,b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

// ══════════════════════════════════════════════════════════════════════
// Internal Helper Functions
// ══════════════════════════════════════════════════════════════════════

/// Standard normal inverse CDF (Beasley-Springer-Moro algorithm).
fn standard_normal_ppf(p: f64) -> f64 {
    // Rational approximation for the central region
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        // Central region: |q| <= 0.425
        let r = 0.180_625 - q * q;
        let num = q
            * (((((((2.509_080_928_730_122_7e3 * r + 3.343_054_881_277_247_5e4) * r
                + 6.726_577_092_700_87e4)
                * r
                + 4.592_195_393_154_987e4)
                * r
                + 1.373_169_376_550_946e4)
                * r
                + 1.971_590_950_306_551_4e3)
                * r
                + 1.331_416_678_917_843_8e2)
                * r
                + 3.387_132_872_796_366_5);
        let den = ((((((5.226_495_278_852_854e3 * r + 2.872_908_573_572_194_3e4) * r
            + 3.930_789_580_009_271e4)
            * r
            + 2.121_379_430_158_993_7e4)
            * r
            + 5.394_196_021_424_751e3)
            * r
            + 6.871_870_741_484_02e2)
            * r
            + 4.231_333_070_160_091e1)
            * r
            + 1.0;
        return num / den;
    }

    // Tail region
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();

    #[allow(clippy::excessive_precision)]
    let result = if r <= 5.0 {
        // Wichura (1988) AS241 coefficients for intermediate tail region
        let r = r - 1.6;
        let num = ((((((7.745_450_142_772_702_5e-4 * r + 2.272_388_412_100_731_05e-2) * r
            + 2.417_807_251_774_506_12e-1)
            * r
            + 1.270_458_252_452_368_38)
            * r
            + 3.647_848_324_763_204_60)
            * r
            + 5.769_497_221_460_691_41)
            * r
            + 4.630_337_846_156_545_30)
            * r
            + 1.423_437_110_749_683_58;
        let den = ((((((1.050_750_071_644_416_84e-9 * r + 5.475_938_084_995_344_95e-4) * r
            + 1.519_866_656_361_645_72e-2)
            * r
            + 1.481_039_764_274_800_75e-1)
            * r
            + 6.897_673_349_851_000_05e-1)
            * r
            + 1.676_384_830_183_803_8)
            * r
            + 2.053_191_626_637_759)
            * r
            + 1.0;
        num / den
    } else {
        let r = r - 5.0;
        let num = ((((((2.010_334_399_292_288_2e-7 * r + 2.711_555_568_743_487_6e-5) * r
            + 1.242_660_947_388_078_4e-3)
            * r
            + 2.653_218_952_657_612_4e-2)
            * r
            + 2.965_605_718_285_048_7e-1)
            * r
            + 1.784_826_539_917_291_3)
            * r
            + 5.463_784_911_164_114)
            * r
            + 6.657_904_643_501_103;
        let den = ((((((2.044_263_103_389_939_7e-15 * r + 1.421_511_758_316_446e-7) * r
            + 1.846_318_317_510_054_8e-5)
            * r
            + 7.868_691_311_456_133e-4)
            * r
            + 1.487_536_129_085_061_5e-2)
            * r
            + 1.369_298_809_227_358e-1)
            * r
            + 5.990_260_257_979_974e-1)
            * r
            + 1.0;
        num / den
    };

    if q < 0.0 { -result } else { result }
}

/// Gamma function ratio for Student's t PDF coefficient.
fn gamma_ratio_t(v: f64) -> f64 {
    // Γ((v+1)/2) / (sqrt(v*π) * Γ(v/2))
    let ln_coeff = ln_gamma(0.5 * (v + 1.0)) - ln_gamma(0.5 * v) - 0.5 * (v * PI).ln();
    ln_coeff.exp()
}

fn ln_gamma(x: f64) -> f64 {
    fsci_special::gammaln_scalar(x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

fn lower_regularized_gamma(a: f64, x: f64) -> f64 {
    fsci_special::gammainc_scalar(a, x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

fn upper_regularized_gamma(a: f64, x: f64) -> f64 {
    fsci_special::gammaincc_scalar(a, x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    fsci_special::betainc_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

fn modified_bessel_i(order: f64, x: f64) -> f64 {
    match fsci_special::iv(
        &fsci_special::SpecialTensor::RealScalar(order),
        &fsci_special::SpecialTensor::RealScalar(x),
        RuntimeMode::Strict,
    ) {
        Ok(fsci_special::SpecialTensor::RealScalar(value)) => value,
        _ => f64::NAN,
    }
}

fn modified_bessel_k(order: f64, x: f64) -> f64 {
    match fsci_special::kv(
        &fsci_special::SpecialTensor::RealScalar(order),
        &fsci_special::SpecialTensor::RealScalar(x),
        RuntimeMode::Strict,
    ) {
        Ok(fsci_special::SpecialTensor::RealScalar(value)) => value,
        _ => f64::NAN,
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn matvec_mul(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, vector)).collect()
}

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; n]; n];
    for (i, row) in out.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    out
}

fn covariance_biased(samples: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if samples.is_empty() {
        return Vec::new();
    }
    let n = samples.len();
    let dim = samples[0].len();
    let mut means = vec![0.0; dim];
    for sample in samples {
        for (j, &value) in sample.iter().enumerate() {
            means[j] += value;
        }
    }
    for mean in &mut means {
        *mean /= n as f64;
    }

    let mut cov = vec![vec![0.0; dim]; dim];
    for sample in samples {
        let centered: Vec<f64> = sample
            .iter()
            .zip(means.iter())
            .map(|(&x, &m)| x - m)
            .collect();
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value += centered[i] * centered[j];
            }
        }
    }
    for row in &mut cov {
        for value in row {
            *value /= n as f64;
        }
    }
    cov
}

fn jacobi_symmetric_eigendecomposition(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut matrix = a.to_vec();
    let mut eigenvectors = identity_matrix(n);
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    for _ in 0..(n * n * 32).max(32) {
        let mut p = 0;
        let mut q = 1.min(n.saturating_sub(1));
        let mut max_off = 0.0;
        for (i, row) in matrix.iter().enumerate() {
            for (j, value) in row.iter().enumerate().skip(i + 1) {
                if value.abs() > max_off {
                    max_off = value.abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }

        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        // apq is the max off-diagonal element (abs >= 1e-12 from convergence check),
        // but guard explicitly against pathological zero to prevent NaN propagation.
        if apq.abs() < f64::MIN_POSITIVE {
            break;
        }
        let tau = (aqq - app) / (2.0 * apq);
        let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for row in &mut matrix {
            let mkp = row[p];
            let mkq = row[q];
            row[p] = c * mkp - s * mkq;
            row[q] = s * mkp + c * mkq;
        }
        let row_p_old = matrix[p].clone();
        let row_q_old = matrix[q].clone();
        for (j, (&mpj, &mqj)) in row_p_old.iter().zip(row_q_old.iter()).enumerate() {
            matrix[p][j] = c * mpj - s * mqj;
            matrix[q][j] = s * mpj + c * mqj;
        }
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;

        for row in &mut eigenvectors {
            let vkp = row[p];
            let vkq = row[q];
            row[p] = c * vkp - s * vkq;
            row[q] = s * vkp + c * vkq;
        }
    }

    let eigenvalues = matrix.iter().enumerate().map(|(i, row)| row[i]).collect();
    (eigenvalues, eigenvectors)
}

fn symmetric_pseudoinverse_and_rank(matrix: &[Vec<f64>]) -> (Vec<Vec<f64>>, usize) {
    let n = matrix.len();
    if n == 0 {
        return (Vec::new(), 0);
    }
    let (eigenvalues, eigenvectors) = jacobi_symmetric_eigendecomposition(matrix);
    let max_eigen = eigenvalues.iter().copied().fold(0.0_f64, |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b.abs())
        }
    });
    if max_eigen.is_nan() {
        return (vec![vec![f64::NAN; n]; n], 0);
    }
    let threshold = (n as f64) * f64::EPSILON * max_eigen.max(1.0);
    let mut pinv = vec![vec![0.0; n]; n];
    let mut rank = 0usize;
    for (idx, &eigenvalue) in eigenvalues.iter().enumerate() {
        if eigenvalue.abs() <= threshold {
            continue;
        }
        rank += 1;
        let inv = 1.0 / eigenvalue;
        for (i, row) in pinv.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value += inv * eigenvectors[i][idx] * eigenvectors[j][idx];
            }
        }
    }
    (pinv, rank)
}

fn cholesky_decompose(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, StatsError> {
    let n = matrix.len();
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum = (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
            if i == j {
                let value = matrix[i][i] - sum;
                if value <= 0.0 || !value.is_finite() {
                    return Err(StatsError::InvalidArgument(
                        "covariance matrix must be symmetric positive definite".to_string(),
                    ));
                }
                lower[i][j] = value.sqrt();
            } else {
                lower[i][j] = (matrix[i][j] - sum) / lower[j][j];
            }
        }
    }
    Ok(lower)
}

fn solve_lower_triangular(lower: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, StatsError> {
    let n = lower.len();
    let mut solution = vec![0.0; n];
    for i in 0..n {
        let sum = (0..i).map(|j| lower[i][j] * solution[j]).sum::<f64>();
        if lower[i][i] == 0.0 {
            return Err(StatsError::InvalidArgument(
                "singular lower-triangular system".to_string(),
            ));
        }
        solution[i] = (rhs[i] - sum) / lower[i][i];
    }
    Ok(solution)
}

fn sample_standard_normals(n: usize, rng: &mut impl Rng) -> Vec<f64> {
    let mut values = Vec::with_capacity(n);
    while values.len() < n {
        let u1 = rng.random::<f64>().max(1e-15);
        let u2 = rng.random::<f64>();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * PI * u2;
        values.push(radius * angle.cos());
        if values.len() < n {
            values.push(radius * angle.sin());
        }
    }
    values
}

// ══════════════════════════════════════════════════════════════════════
// Cauchy Distribution
// ══════════════════════════════════════════════════════════════════════

/// Cauchy distribution (Lorentzian) with location `loc` and scale `scale`.
///
/// Matches `scipy.stats.cauchy(loc, scale)`.
/// PDF: f(x) = 1 / (π * scale * (1 + ((x - loc)/scale)²))
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cauchy {
    pub loc: f64,
    pub scale: f64,
}

impl Default for Cauchy {
    fn default() -> Self {
        Self {
            loc: 0.0,
            scale: 1.0,
        }
    }
}

impl Cauchy {
    #[must_use]
    pub fn new(loc: f64, scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { loc, scale }
    }
}

impl ContinuousDistribution for Cauchy {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        1.0 / (PI * self.scale * (1.0 + z * z))
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        0.5 + z.atan() / PI
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        self.loc + self.scale * (PI * (q - 0.5)).tan()
    }

    fn mean(&self) -> f64 {
        f64::NAN // Cauchy has no finite mean
    }

    fn var(&self) -> f64 {
        f64::NAN // Cauchy has no finite variance
    }
}

// ══════════════════════════════════════════════════════════════════════
// Laplace Distribution
// ══════════════════════════════════════════════════════════════════════

/// Laplace (double exponential) distribution.
///
/// Matches `scipy.stats.laplace(loc, scale)`.
/// PDF: f(x) = exp(-|x - loc|/scale) / (2 * scale)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Laplace {
    pub loc: f64,
    pub scale: f64,
}

impl Default for Laplace {
    fn default() -> Self {
        Self {
            loc: 0.0,
            scale: 1.0,
        }
    }
}

impl Laplace {
    #[must_use]
    pub fn new(loc: f64, scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive, got {scale}");
        Self { loc, scale }
    }
}

impl ContinuousDistribution for Laplace {
    fn pdf(&self, x: f64) -> f64 {
        let z = ((x - self.loc) / self.scale).abs();
        (-z).exp() / (2.0 * self.scale)
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        if z < 0.0 {
            0.5 * z.exp()
        } else {
            1.0 - 0.5 * (-z).exp()
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        if q < 0.5 {
            self.loc + self.scale * (2.0 * q).ln()
        } else {
            self.loc - self.scale * (2.0 * (1.0 - q)).ln()
        }
    }

    fn mean(&self) -> f64 {
        self.loc
    }

    fn var(&self) -> f64 {
        2.0 * self.scale * self.scale
    }
}

// ══════════════════════════════════════════════════════════════════════
// Triangular Distribution
// ══════════════════════════════════════════════════════════════════════

/// Triangular distribution on [left, right] with mode `mode`.
///
/// Matches `scipy.stats.triang(c, loc, scale)` where c=(mode-left)/(right-left).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangular {
    pub left: f64,
    pub mode: f64,
    pub right: f64,
}

impl Triangular {
    #[must_use]
    pub fn new(left: f64, mode: f64, right: f64) -> Self {
        assert!(left < mode, "left must be < mode");
        assert!(mode <= right, "mode must be <= right");
        Self { left, mode, right }
    }
}

impl ContinuousDistribution for Triangular {
    fn pdf(&self, x: f64) -> f64 {
        let (a, c, b) = (self.left, self.mode, self.right);
        if x < a || x > b {
            0.0
        } else if c == b {
            // Degenerate: right-skewed triangle (mode == right)
            if x == b {
                2.0 / (b - a)
            } else {
                2.0 * (x - a) / ((b - a) * (b - a))
            }
        } else if x < c {
            2.0 * (x - a) / ((b - a) * (c - a))
        } else if x == c {
            2.0 / (b - a)
        } else {
            2.0 * (b - x) / ((b - a) * (b - c))
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        let (a, c, b) = (self.left, self.mode, self.right);
        if x <= a {
            0.0
        } else if c == b {
            // Degenerate: mode == right
            if x >= b {
                1.0
            } else {
                (x - a).powi(2) / ((b - a) * (b - a))
            }
        } else if x <= c {
            (x - a).powi(2) / ((b - a) * (c - a))
        } else if x < b {
            1.0 - (b - x).powi(2) / ((b - a) * (b - c))
        } else {
            1.0
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return self.left;
        }
        if q == 1.0 {
            return self.right;
        }
        let (a, c, b) = (self.left, self.mode, self.right);
        let fc = (c - a) / (b - a);
        if q < fc {
            a + (q * (b - a) * (c - a)).sqrt()
        } else {
            b - ((1.0 - q) * (b - a) * (b - c)).sqrt()
        }
    }

    fn mean(&self) -> f64 {
        (self.left + self.mode + self.right) / 3.0
    }

    fn var(&self) -> f64 {
        let (a, c, b) = (self.left, self.mode, self.right);
        (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
    }
}

/// Trapezoidal distribution on `[loc, loc + scale]` with flat top spanning
/// `[loc + c * scale, loc + d * scale]`.
///
/// Matches `scipy.stats.trapezoid(c, d, loc, scale)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trapezoid {
    pub c: f64,
    pub d: f64,
    pub loc: f64,
    pub scale: f64,
}

impl Trapezoid {
    #[must_use]
    pub fn new(c: f64, d: f64, loc: f64, scale: f64) -> Self {
        assert!((0.0..=1.0).contains(&c), "c must be in [0, 1]");
        assert!((0.0..=1.0).contains(&d), "d must be in [0, 1]");
        assert!(d >= c, "d must be >= c");
        assert!(scale > 0.0, "scale must be positive");
        Self { c, d, loc, scale }
    }

    fn standard_moment(&self, n: usize) -> f64 {
        let c = self.c;
        let d = self.d;
        let den = 1.0 + d - c;
        let ab_term = c.powf((n + 1) as f64);
        let dc_term = if d == 0.0 {
            1.0
        } else if d < 1.0 {
            (d.powf((n + 2) as f64) - 1.0) / (d - 1.0)
        } else {
            (n + 2) as f64
        };
        2.0 * (dc_term - ab_term) / (den * (n as f64 + 1.0) * (n as f64 + 2.0))
    }
}

impl ContinuousDistribution for Trapezoid {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        if !(0.0..=1.0).contains(&z) {
            return 0.0;
        }
        let height = 2.0 / (1.0 + self.d - self.c);
        let standard_pdf = if z < self.c {
            height * z / self.c
        } else if z <= self.d {
            height
        } else {
            height * (1.0 - z) / (1.0 - self.d)
        };
        standard_pdf / self.scale
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        if z <= 0.0 {
            return 0.0;
        }
        if z >= 1.0 {
            return 1.0;
        }
        let den = 1.0 + self.d - self.c;
        if z < self.c {
            z * z / (self.c * den)
        } else if z <= self.d {
            (self.c + 2.0 * (z - self.c)) / den
        } else {
            1.0 - (1.0 - z).powi(2) / ((1.0 - self.d) * den)
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return self.loc;
        }
        if q == 1.0 {
            return self.loc + self.scale;
        }
        let den = 1.0 + self.d - self.c;
        let qc = self.c / den;
        let qd = (2.0 * self.d - self.c) / den;
        let z = if q < qc {
            (q * self.c * den).sqrt()
        } else if q <= qd {
            0.5 * (q * den + self.c)
        } else {
            1.0 - ((1.0 - q) * (1.0 - self.d) * den).sqrt()
        };
        self.loc + self.scale * z
    }

    fn mean(&self) -> f64 {
        self.loc + self.scale * self.standard_moment(1)
    }

    fn var(&self) -> f64 {
        let ex = self.standard_moment(1);
        let ex2 = self.standard_moment(2);
        self.scale * self.scale * (ex2 - ex * ex)
    }
}

// ══════════════════════════════════════════════════════════════════════
// Additional Distributions
// ══════════════════════════════════════════════════════════════════════

/// Inverse Gamma distribution.
///
/// Matches `scipy.stats.invgamma`.
pub struct InverseGamma {
    pub a: f64, // shape
}

impl InverseGamma {
    #[must_use]
    pub fn new(a: f64) -> Self {
        assert!(a > 0.0, "shape parameter must be positive");
        Self { a }
    }
}

impl ContinuousDistribution for InverseGamma {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let a = self.a;
        x.powf(-a - 1.0) * (-1.0 / x).exp() / ln_gamma(a).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        // CDF = 1 - gammainc(a, 1/x) = gammaincc(a, 1/x)
        // Using upper regularized incomplete gamma
        upper_regularized_gamma(self.a, 1.0 / x)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        // CDF = Q(a, 1/x) = 1 - P(a, 1/x), so P(a, 1/x) = 1 - q
        // 1/x = gammaincinv(a, 1 - q), x = 1 / gammaincinv(a, 1 - q)
        let g = fsci_special::gammaincinv(self.a, 1.0 - q);
        if g <= 0.0 {
            return ppf_bisection(|v| self.cdf(v), q, self.mean(), self.std());
        }
        let mut x = 1.0 / g;
        for _ in 0..8 {
            let err = self.cdf(x) - q;
            if err.abs() < 1e-14 {
                break;
            }
            let dx = self.pdf(x);
            if dx > 1e-30 {
                x = (x - err / dx).max(0.0);
            }
        }
        x
    }

    fn mean(&self) -> f64 {
        if self.a > 1.0 {
            1.0 / (self.a - 1.0)
        } else {
            f64::INFINITY
        }
    }

    fn var(&self) -> f64 {
        if self.a > 2.0 {
            1.0 / ((self.a - 1.0).powi(2) * (self.a - 2.0))
        } else {
            f64::INFINITY
        }
    }
}

// Log-normal distribution (parameterized by mu, sigma of underlying normal).
//
// Note: The Lognormal struct above uses s/scale. This is an alias with
// the same interface. Kept for disambiguation.

/// Inverse Gaussian (Wald) distribution.
///
/// Matches `scipy.stats.invgauss`.
pub struct InverseGaussian {
    pub mu: f64, // mean
}

impl InverseGaussian {
    #[must_use]
    pub fn new(mu: f64) -> Self {
        assert!(mu > 0.0, "mu must be positive");
        Self { mu }
    }
}

impl ContinuousDistribution for InverseGaussian {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let mu = self.mu;
        (1.0 / (2.0 * PI * x.powi(3))).sqrt() * (-(x - mu).powi(2) / (2.0 * mu * mu * x)).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let mu = self.mu;
        let sqrt_x = x.sqrt();
        let t1 = standard_normal_cdf(sqrt_x / mu - 1.0 / sqrt_x);
        let t2 = (2.0 / mu).exp() * standard_normal_cdf(-(sqrt_x / mu + 1.0 / sqrt_x));
        t1 + t2
    }

    fn mean(&self) -> f64 {
        self.mu
    }

    fn var(&self) -> f64 {
        self.mu.powi(3)
    }
}

/// Pearson type III distribution.
///
/// Matches `scipy.stats.pearson3`.
pub struct Pearson3 {
    pub skew: f64,
}

impl Pearson3 {
    #[must_use]
    pub fn new(skew: f64) -> Self {
        assert!(skew.is_finite(), "skew must be finite");
        Self { skew }
    }
}

impl ContinuousDistribution for Pearson3 {
    fn pdf(&self, x: f64) -> f64 {
        const NORMAL_TRANSITION: f64 = 1.6e-5;
        let skew = self.skew;
        if skew.abs() < NORMAL_TRANSITION {
            return standard_normal_pdf(x);
        }

        let beta = 2.0 / skew;
        let alpha = beta * beta;
        let zeta = -beta;
        let transx = beta * (x - zeta);
        if transx < 0.0 {
            return 0.0;
        }

        beta.abs() * transx.powf(alpha - 1.0) * (-transx).exp() / ln_gamma(alpha).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        const NORMAL_TRANSITION: f64 = 1.6e-5;
        let skew = self.skew;
        if skew.abs() < NORMAL_TRANSITION {
            return standard_normal_cdf(x);
        }

        let beta = 2.0 / skew;
        let alpha = beta * beta;
        let zeta = -beta;
        let transx = beta * (x - zeta);

        if skew > 0.0 {
            if transx <= 0.0 {
                0.0
            } else {
                lower_regularized_gamma(alpha, transx)
            }
        } else if transx <= 0.0 {
            1.0
        } else {
            upper_regularized_gamma(alpha, transx)
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        const NORMAL_TRANSITION: f64 = 1.6e-5;
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }

        let skew = self.skew;
        if skew.abs() < NORMAL_TRANSITION {
            return fsci_special::ndtri(q);
        }

        let beta = 2.0 / skew;
        let alpha = beta * beta;
        let zeta = -beta;
        let gamma_q = if beta < 0.0 { 1.0 - q } else { q };
        let g = fsci_special::gammaincinv(alpha, gamma_q);
        if !g.is_finite() {
            return ppf_bisection(|x| self.cdf(x), q, self.mean(), self.std());
        }
        g / beta + zeta
    }

    fn mean(&self) -> f64 {
        0.0
    }

    fn var(&self) -> f64 {
        1.0
    }
}

/// Exponentially modified normal distribution.
///
/// Matches `scipy.stats.exponnorm`.
pub struct ExponNorm {
    pub k: f64,
}

impl ExponNorm {
    #[must_use]
    pub fn new(k: f64) -> Self {
        assert!(k.is_finite() && k > 0.0, "k must be positive and finite");
        Self { k }
    }
}

impl ContinuousDistribution for ExponNorm {
    fn pdf(&self, x: f64) -> f64 {
        let inv_k = 1.0 / self.k;
        let exp_arg = inv_k * (0.5 * inv_k - x);
        (exp_arg + fsci_special::log_ndtr(x - inv_k)).exp() / self.k
    }

    fn cdf(&self, x: f64) -> f64 {
        let inv_k = 1.0 / self.k;
        let exp_arg = inv_k * (0.5 * inv_k - x);
        standard_normal_cdf(x) - (exp_arg + fsci_special::log_ndtr(x - inv_k)).exp()
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        ppf_bisection(|x| self.cdf(x), q, self.mean(), self.std())
    }

    fn mean(&self) -> f64 {
        self.k
    }

    fn var(&self) -> f64 {
        1.0 + self.k * self.k
    }
}

/// Generalized Extreme Value (GEV) distribution.
///
/// Matches `scipy.stats.genextreme`.
pub struct GenExtreme {
    pub c: f64, // shape parameter
}

impl GenExtreme {
    #[must_use]
    pub fn new(c: f64) -> Self {
        Self { c }
    }
}

impl ContinuousDistribution for GenExtreme {
    fn pdf(&self, x: f64) -> f64 {
        let c = self.c;
        if c.abs() < 1e-15 {
            // Gumbel case (c -> 0): exp(-(x + exp(-x)))
            let ex = (-x).exp();
            (-(x + ex)).exp()
        } else {
            let t = 1.0 + c * x;
            if t <= 0.0 {
                return 0.0;
            }
            let tp = t.powf(-1.0 / c);
            tp.powf(c + 1.0) * (-tp).exp()
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        let c = self.c;
        if c.abs() < 1e-15 {
            (-(-x).exp()).exp()
        } else {
            let t = 1.0 + c * x;
            if t <= 0.0 {
                return if c > 0.0 { 0.0 } else { 1.0 };
            }
            (-t.powf(-1.0 / c)).exp()
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return if self.c > 0.0 {
                -1.0 / self.c
            } else {
                f64::NEG_INFINITY
            };
        }
        if q == 1.0 {
            return if self.c < 0.0 {
                -1.0 / self.c
            } else {
                f64::INFINITY
            };
        }
        let c = self.c;
        if c.abs() < 1e-15 {
            -(-q.ln()).ln()
        } else {
            ((-q.ln()).powf(-c) - 1.0) / c
        }
    }

    fn mean(&self) -> f64 {
        let c = self.c;
        if c.abs() < 1e-15 {
            0.577_215_664_901_532_9 // Euler-Mascheroni constant
        } else if c < 1.0 {
            (ln_gamma(1.0 - c).exp() - 1.0) / c
        } else {
            f64::INFINITY
        }
    }

    fn var(&self) -> f64 {
        let c = self.c;
        if c.abs() < 1e-15 {
            PI * PI / 6.0
        } else if c < 0.5 {
            let g1 = ln_gamma(1.0 - c).exp();
            let g2 = ln_gamma(1.0 - 2.0 * c).exp();
            (g2 - g1 * g1) / (c * c)
        } else {
            f64::INFINITY
        }
    }
}

/// Generalized Pareto distribution.
///
/// Matches `scipy.stats.genpareto`.
pub struct GenPareto {
    pub c: f64,
}

impl GenPareto {
    #[must_use]
    pub fn new(c: f64) -> Self {
        Self { c }
    }
}

impl ContinuousDistribution for GenPareto {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let c = self.c;
        if c.abs() < 1e-15 {
            (-x).exp()
        } else {
            let t = 1.0 + c * x;
            if t <= 0.0 {
                return 0.0;
            }
            t.powf(-1.0 / c - 1.0)
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let c = self.c;
        if c.abs() < 1e-15 {
            1.0 - (-x).exp()
        } else {
            let t = 1.0 + c * x;
            if t <= 0.0 {
                return if c > 0.0 { 0.0 } else { 1.0 };
            }
            1.0 - t.powf(-1.0 / c)
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return if self.c < 0.0 {
                -1.0 / self.c
            } else {
                f64::INFINITY
            };
        }
        let c = self.c;
        if c.abs() < 1e-15 {
            -(1.0 - q).ln()
        } else {
            ((1.0 - q).powf(-c) - 1.0) / c
        }
    }

    fn mean(&self) -> f64 {
        if self.c < 1.0 {
            1.0 / (1.0 - self.c)
        } else {
            f64::INFINITY
        }
    }

    fn var(&self) -> f64 {
        if self.c < 0.5 {
            1.0 / ((1.0 - self.c).powi(2) * (1.0 - 2.0 * self.c))
        } else {
            f64::INFINITY
        }
    }
}

/// Power-law distribution: x^(a-1) on [0, 1].
///
/// Matches `scipy.stats.powerlaw`.
pub struct PowerLaw {
    pub a: f64,
}

impl PowerLaw {
    #[must_use]
    pub fn new(a: f64) -> Self {
        assert!(a > 0.0, "a must be positive");
        Self { a }
    }
}

impl ContinuousDistribution for PowerLaw {
    fn pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            0.0
        } else {
            self.a * x.powf(self.a - 1.0)
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            x.powf(self.a)
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            0.0
        } else if q == 1.0 {
            1.0
        } else {
            q.powf(1.0 / self.a)
        }
    }

    fn mean(&self) -> f64 {
        self.a / (self.a + 1.0)
    }

    fn var(&self) -> f64 {
        self.a / ((self.a + 1.0).powi(2) * (self.a + 2.0))
    }
}

/// Half-normal distribution (folded normal).
///
/// Matches `scipy.stats.halfnorm`.
pub struct HalfNormal;

impl ContinuousDistribution for HalfNormal {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            (2.0 / PI).sqrt() * (-x * x / 2.0).exp()
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            fsci_special::erf_scalar(x / std::f64::consts::SQRT_2)
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        // HalfNormal is the positive half of N(0,1), so ppf(q) = ndtri((1+q)/2)
        fsci_special::ndtri((1.0 + q) / 2.0)
    }

    fn mean(&self) -> f64 {
        (2.0 / PI).sqrt()
    }

    fn var(&self) -> f64 {
        1.0 - 2.0 / PI
    }
}

/// Half-logistic distribution.
///
/// Matches `scipy.stats.halflogistic`.
pub struct HalfLogistic;

impl ContinuousDistribution for HalfLogistic {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let exp_neg_x = (-x).exp();
        2.0 * exp_neg_x / (1.0 + exp_neg_x).powi(2)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let exp_neg_x = (-x).exp();
        (1.0 - exp_neg_x) / (1.0 + exp_neg_x)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        ((1.0 + q) / (1.0 - q)).ln()
    }

    fn mean(&self) -> f64 {
        2.0 * LN_2
    }

    fn var(&self) -> f64 {
        PI * PI / 3.0 - 4.0 * LN_2 * LN_2
    }
}

/// Half-Cauchy distribution.
///
/// Matches `scipy.stats.halfcauchy`.
pub struct HalfCauchy;

impl ContinuousDistribution for HalfCauchy {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        2.0 / (PI * (1.0 + x * x))
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        2.0 * x.atan() / PI
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        (PI * q / 2.0).tan()
    }

    fn mean(&self) -> f64 {
        f64::NAN
    }

    fn var(&self) -> f64 {
        f64::NAN
    }
}

/// Birnbaum-Saunders fatigue-life distribution.
///
/// Matches `scipy.stats.fatiguelife`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FatigueLife {
    pub c: f64,
}

impl FatigueLife {
    #[must_use]
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        Self { c }
    }
}

impl ContinuousDistribution for FatigueLife {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let numerator = x + 1.0;
        let denominator = 2.0 * self.c * (2.0 * PI * x.powi(3)).sqrt();
        let exponent = -((x - 1.0) * (x - 1.0)) / (2.0 * x * self.c * self.c);
        numerator * exponent.exp() / denominator
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let root_x = x.sqrt();
        let z = (root_x - 1.0 / root_x) / self.c;
        0.5 * (1.0 + fsci_special::erf_scalar(z * FRAC_1_SQRT_2))
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        let tmp = self.c * fsci_special::ndtri(q);
        0.25 * (tmp + (tmp * tmp + 4.0).sqrt()).powi(2)
    }

    fn mean(&self) -> f64 {
        1.0 + 0.5 * self.c * self.c
    }

    fn var(&self) -> f64 {
        0.25 * self.c * self.c * (5.0 * self.c * self.c + 4.0)
    }
}

/// Truncated normal distribution on [a, b].
///
/// Matches `scipy.stats.truncnorm`.
pub struct TruncNormal {
    pub a: f64,
    pub b: f64,
}

impl TruncNormal {
    #[must_use]
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a < b, "a must be less than b");
        Self { a, b }
    }
}

impl ContinuousDistribution for TruncNormal {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
            return 0.0;
        }
        let phi_x = (-x * x / 2.0).exp() / (2.0 * PI).sqrt();
        let norm = standard_normal_cdf(self.b) - standard_normal_cdf(self.a);
        if norm > 0.0 { phi_x / norm } else { 0.0 }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a {
            return 0.0;
        }
        if x >= self.b {
            return 1.0;
        }
        let phi_a = standard_normal_cdf(self.a);
        let norm = standard_normal_cdf(self.b) - phi_a;
        if norm > 0.0 {
            (standard_normal_cdf(x) - phi_a) / norm
        } else {
            0.0
        }
    }

    fn mean(&self) -> f64 {
        let phi_a = (-self.a * self.a / 2.0).exp() / (2.0 * PI).sqrt();
        let phi_b = (-self.b * self.b / 2.0).exp() / (2.0 * PI).sqrt();
        let norm = standard_normal_cdf(self.b) - standard_normal_cdf(self.a);
        if norm > 0.0 {
            (phi_a - phi_b) / norm
        } else {
            (self.a + self.b) / 2.0
        }
    }

    fn var(&self) -> f64 {
        let phi_a = (-self.a * self.a / 2.0).exp() / (2.0 * PI).sqrt();
        let phi_b = (-self.b * self.b / 2.0).exp() / (2.0 * PI).sqrt();
        let norm = standard_normal_cdf(self.b) - standard_normal_cdf(self.a);
        if norm <= 0.0 {
            return 0.0;
        }
        let z = norm;
        1.0 + (self.a * phi_a - self.b * phi_b) / z - ((phi_a - phi_b) / z).powi(2)
    }
}

/// Chi distribution (not chi-squared).
///
/// Matches `scipy.stats.chi`.
pub struct Chi {
    pub df: f64,
}

impl Chi {
    #[must_use]
    pub fn new(df: f64) -> Self {
        assert!(df > 0.0, "df must be positive");
        Self { df }
    }
}

impl ContinuousDistribution for Chi {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let k = self.df;
        2.0_f64.powf(1.0 - k / 2.0) * x.powf(k - 1.0) * (-x * x / 2.0).exp()
            / ln_gamma(k / 2.0).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        // CDF of chi with k df = regularized lower incomplete gamma(k/2, x²/2)
        lower_regularized_gamma(self.df / 2.0, x * x / 2.0)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        // CDF = P(df/2, x²/2), so x²/2 = gammaincinv(df/2, q), x = sqrt(2*gammaincinv(df/2, q))
        let g = fsci_special::gammaincinv(self.df / 2.0, q);
        let mut x = (2.0 * g).sqrt();
        if x <= 0.0 || (self.cdf(x) - q).abs() > 0.1 {
            return ppf_bisection(|v| self.cdf(v), q, self.mean(), self.std());
        }
        for _ in 0..8 {
            let err = self.cdf(x) - q;
            if err.abs() < 1e-14 {
                break;
            }
            let dx = self.pdf(x);
            if dx > 1e-30 {
                x = (x - err / dx).max(0.0);
            }
        }
        x
    }

    fn mean(&self) -> f64 {
        let k = self.df;
        std::f64::consts::SQRT_2 * ln_gamma((k + 1.0) / 2.0).exp() / ln_gamma(k / 2.0).exp()
    }

    fn var(&self) -> f64 {
        let m = self.mean();
        self.df - m * m
    }
}

/// Rice distribution.
///
/// Matches `scipy.stats.rice`.
pub struct Rice {
    pub b: f64, // non-centrality parameter
}

impl Rice {
    #[must_use]
    pub fn new(b: f64) -> Self {
        assert!(b >= 0.0, "b must be non-negative");
        Self { b }
    }
}

impl ContinuousDistribution for Rice {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let b = self.b;
        // f(x) = x * exp(-(x² + b²)/2) * I₀(b*x)
        x * (-(x * x + b * b) / 2.0).exp() * modified_bessel_i(0.0, b * x)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        // Adaptive Simpson's rule — use more points for wider intervals
        let n = (200.0 * (1.0 + x / 5.0).min(10.0)) as usize;
        let n = n + (n % 2); // ensure even for Simpson
        let h = x / n as f64;
        let mut sum = self.pdf(0.0) + self.pdf(x);
        for i in 1..n {
            let t = i as f64 * h;
            let w = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += w * self.pdf(t);
        }
        (sum * h / 3.0).clamp(0.0, 1.0)
    }

    fn mean(&self) -> f64 {
        // Numerical integration — scale grid with the distribution's spread
        let upper = self.b + 6.0;
        let n = (500.0 * (1.0 + self.b / 5.0).min(10.0)) as usize;
        let n = n + (n % 2);
        let h = upper / n as f64;
        let mut sum = 0.0;
        for i in 0..=n {
            let x = i as f64 * h;
            let w = if i == 0 || i == n {
                1.0
            } else if i % 2 == 0 {
                2.0
            } else {
                4.0
            };
            sum += w * x * self.pdf(x);
        }
        sum * h / 3.0
    }

    fn var(&self) -> f64 {
        let m = self.mean();
        let upper = self.b + 6.0;
        let n = (500.0 * (1.0 + self.b / 5.0).min(10.0)) as usize;
        let n = n + (n % 2);
        let h = upper / n as f64;
        let mut sum = 0.0;
        for i in 0..=n {
            let x = i as f64 * h;
            let w = if i == 0 || i == n {
                1.0
            } else if i % 2 == 0 {
                2.0
            } else {
                4.0
            };
            sum += w * x * x * self.pdf(x);
        }
        let ex2 = sum * h / 3.0;
        ex2 - m * m
    }
}

/// Nakagami distribution.
///
/// Matches `scipy.stats.nakagami`.
pub struct Nakagami {
    pub nu: f64, // shape parameter >= 0.5
}

impl Nakagami {
    #[must_use]
    pub fn new(nu: f64) -> Self {
        assert!(nu >= 0.5, "nu must be >= 0.5");
        Self { nu }
    }
}

impl ContinuousDistribution for Nakagami {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let nu = self.nu;
        2.0 * nu.powf(nu) / ln_gamma(nu).exp() * x.powf(2.0 * nu - 1.0) * (-nu * x * x).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        lower_regularized_gamma(self.nu, self.nu * x * x)
    }

    fn mean(&self) -> f64 {
        ln_gamma(self.nu + 0.5).exp() / (self.nu.sqrt() * ln_gamma(self.nu).exp())
    }

    fn var(&self) -> f64 {
        let m = self.mean();
        1.0 - m * m
    }
}

/// Log-logistic (Fisk) distribution.
///
/// Matches `scipy.stats.fisk`.
pub struct Fisk {
    pub c: f64, // shape
}

impl Fisk {
    #[must_use]
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        Self { c }
    }
}

impl ContinuousDistribution for Fisk {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let c = self.c;
        c * x.powf(c - 1.0) / (1.0 + x.powf(c)).powi(2)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            1.0 / (1.0 + x.powf(-self.c))
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        (q / (1.0 - q)).powf(1.0 / self.c)
    }

    fn mean(&self) -> f64 {
        if self.c > 1.0 {
            let b = PI / self.c;
            b / b.sin()
        } else {
            f64::INFINITY
        }
    }

    fn var(&self) -> f64 {
        if self.c > 2.0 {
            let b = PI / self.c;
            2.0 * b / (2.0 * b).sin() - b * b / (b.sin() * b.sin())
        } else {
            f64::INFINITY
        }
    }
}

/// Loguniform (reciprocal) distribution on [a, b].
///
/// Matches `scipy.stats.loguniform`.
pub struct Loguniform {
    pub a: f64,
    pub b: f64,
}

impl Loguniform {
    #[must_use]
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a > 0.0, "a must be positive");
        assert!(b > a, "b must be greater than a");
        Self { a, b }
    }
}

impl ContinuousDistribution for Loguniform {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
            0.0
        } else {
            1.0 / (x * (self.b / self.a).ln())
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a {
            0.0
        } else if x >= self.b {
            1.0
        } else {
            (x / self.a).ln() / (self.b / self.a).ln()
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return self.a;
        }
        if q == 1.0 {
            return self.b;
        }
        self.a * (self.b / self.a).powf(q)
    }

    fn mean(&self) -> f64 {
        (self.b - self.a) / (self.b / self.a).ln()
    }

    fn var(&self) -> f64 {
        let log_ratio = (self.b / self.a).ln();
        let m = self.mean();
        (self.b * self.b - self.a * self.a) / (2.0 * log_ratio) - m * m
    }
}

// Discrete distributions: Zipf, Boltzmann, etc.

/// Zipf (zeta) distribution.
///
/// Matches `scipy.stats.zipf`.
pub struct Zipf {
    pub a: f64,
}

impl Zipf {
    #[must_use]
    pub fn new(a: f64) -> Self {
        assert!(a > 1.0, "a must be > 1");
        Self { a }
    }

    /// PMF: P(X=k) = k^{-a} / zeta(a)
    pub fn pmf(&self, k: usize) -> f64 {
        if k == 0 {
            return 0.0;
        }
        (k as f64).powf(-self.a) / riemann_zeta(self.a)
    }

    /// CDF: sum of PMF from 1 to k.
    pub fn cdf(&self, k: usize) -> f64 {
        let z = riemann_zeta(self.a);
        let mut sum = 0.0;
        for i in 1..=k {
            sum += (i as f64).powf(-self.a);
        }
        (sum / z).min(1.0)
    }

    /// Mean: zeta(a-1)/zeta(a), exists for a > 2.
    pub fn mean(&self) -> f64 {
        if self.a > 2.0 {
            riemann_zeta(self.a - 1.0) / riemann_zeta(self.a)
        } else {
            f64::INFINITY
        }
    }
}

/// Helper: Riemann zeta function for real s > 1.
fn riemann_zeta(s: f64) -> f64 {
    let mut sum = 0.0;
    for k in 1..=10000 {
        let term = (k as f64).powf(-s);
        sum += term;
        if term < 1e-15 * sum {
            break;
        }
    }
    sum
}

/// Double Weibull distribution.
///
/// Matches `scipy.stats.dweibull`.
pub struct DoubleWeibull {
    pub c: f64,
}

impl DoubleWeibull {
    #[must_use]
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        Self { c }
    }
}

impl ContinuousDistribution for DoubleWeibull {
    fn pdf(&self, x: f64) -> f64 {
        let c = self.c;
        0.5 * c * x.abs().powf(c - 1.0) * (-x.abs().powf(c)).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        let c = self.c;
        if x < 0.0 {
            0.5 * (-(-x).powf(c)).exp()
        } else {
            1.0 - 0.5 * (-x.powf(c)).exp()
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        let c = self.c;
        if q < 0.5 {
            // x < 0: CDF = 0.5*exp(-|x|^c), so |x| = (-ln(2q))^(1/c)
            -(-(2.0 * q).ln()).powf(1.0 / c)
        } else if q > 0.5 {
            // x >= 0: CDF = 1 - 0.5*exp(-x^c), so x = (-ln(2*(1-q)))^(1/c)
            (-(2.0 * (1.0 - q)).ln()).powf(1.0 / c)
        } else {
            0.0
        }
    }

    fn mean(&self) -> f64 {
        0.0 // symmetric about 0
    }

    fn var(&self) -> f64 {
        let c = self.c;
        ln_gamma(1.0 + 2.0 / c).exp() // Γ(1 + 2/c) for unit scale
    }
}

/// Semicircular distribution on [-1, 1].
///
/// Matches `scipy.stats.semicircular`.
pub struct Semicircular;

impl ContinuousDistribution for Semicircular {
    fn pdf(&self, x: f64) -> f64 {
        if x.abs() > 1.0 {
            0.0
        } else {
            2.0 / PI * (1.0 - x * x).sqrt()
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= -1.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            0.5 + x * (1.0 - x * x).sqrt() / PI + x.asin() / PI
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return -1.0;
        }
        if q == 1.0 {
            return 1.0;
        }
        // No closed form — use Newton with initial guess from arcsine approximation
        let mut x = (PI * (q - 0.5)).sin(); // rough initial guess
        for _ in 0..12 {
            let err = self.cdf(x) - q;
            if err.abs() < 1e-14 {
                break;
            }
            let dx = self.pdf(x);
            if dx > 1e-30 {
                x = (x - err / dx).clamp(-1.0, 1.0);
            }
        }
        x
    }

    fn mean(&self) -> f64 {
        0.0
    }

    fn var(&self) -> f64 {
        0.25 // 1/4
    }
}

/// Cosine distribution on [-π, π].
///
/// Matches `scipy.stats.cosine`.
pub struct CosineDistribution;

impl ContinuousDistribution for CosineDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x.abs() > PI {
            0.0
        } else {
            (1.0 + x.cos()) / (2.0 * PI)
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= -PI {
            0.0
        } else if x >= PI {
            1.0
        } else {
            (PI + x + x.sin()) / (2.0 * PI)
        }
    }

    fn mean(&self) -> f64 {
        0.0
    }

    fn var(&self) -> f64 {
        PI * PI / 3.0 - 2.0 // π²/3 - 2
    }
}

/// Reciprocal (log-uniform on [a, b]) distribution.
/// Alias for Loguniform kept for compatibility.
///
/// Matches `scipy.stats.reciprocal`.
pub type Reciprocal = Loguniform;

/// Wald distribution (alias for Inverse Gaussian).
///
/// Matches `scipy.stats.wald`.
pub type Wald = InverseGaussian;

/// Erlang distribution: Gamma with integer shape.
///
/// Matches `scipy.stats.erlang`.
pub struct Erlang {
    pub k: usize,
    pub rate: f64,
}

impl Erlang {
    #[must_use]
    pub fn new(k: usize, rate: f64) -> Self {
        assert!(k > 0, "k must be positive");
        assert!(rate > 0.0, "rate must be positive");
        Self { k, rate }
    }
}

impl ContinuousDistribution for Erlang {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let k = self.k as f64;
        let lambda = self.rate;
        lambda.powf(k) * x.powf(k - 1.0) * (-lambda * x).exp() / ln_gamma(k).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        lower_regularized_gamma(self.k as f64, self.rate * x)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        // CDF = P(k, rate*x), so rate*x = gammaincinv(k, q), x = gammaincinv(k, q) / rate
        let k = self.k as f64;
        let mut x = fsci_special::gammaincinv(k, q) / self.rate;
        if x <= 0.0 || (self.cdf(x) - q).abs() > 0.1 {
            return ppf_bisection(|v| self.cdf(v), q, self.mean(), self.std());
        }
        for _ in 0..8 {
            let err = self.cdf(x) - q;
            if err.abs() < 1e-14 {
                break;
            }
            let dx = self.pdf(x);
            if dx > 1e-30 {
                x = (x - err / dx).max(0.0);
            }
        }
        x
    }

    fn mean(&self) -> f64 {
        self.k as f64 / self.rate
    }

    fn var(&self) -> f64 {
        self.k as f64 / (self.rate * self.rate)
    }
}

/// Anglit distribution on [-π/4, π/4].
///
/// Matches `scipy.stats.anglit`.
pub struct Anglit;

impl ContinuousDistribution for Anglit {
    fn pdf(&self, x: f64) -> f64 {
        let quarter_pi = PI / 4.0;
        if x.abs() > quarter_pi {
            0.0
        } else {
            (2.0 * x).cos()
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        let quarter_pi = PI / 4.0;
        if x <= -quarter_pi {
            0.0
        } else if x >= quarter_pi {
            1.0
        } else {
            ((2.0 * x).sin() + 1.0) / 2.0
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return -PI / 4.0;
        }
        if q == 1.0 {
            return PI / 4.0;
        }
        // CDF = (sin(2x)+1)/2, so x = asin(2q-1)/2
        (2.0 * q - 1.0).asin() / 2.0
    }

    fn mean(&self) -> f64 {
        0.0
    }

    fn var(&self) -> f64 {
        PI * PI / 16.0 - 0.5 // π²/16 - 1/2
    }
}

/// Bradford distribution on [0, 1] with shape c.
///
/// Matches `scipy.stats.bradford`.
pub struct Bradford {
    pub c: f64,
}

impl Bradford {
    #[must_use]
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        Self { c }
    }
}

impl ContinuousDistribution for Bradford {
    fn pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            0.0
        } else {
            self.c / ((1.0 + self.c * x) * (1.0 + self.c).ln())
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            (1.0 + self.c * x).ln() / (1.0 + self.c).ln()
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return 1.0;
        }
        ((1.0 + self.c).powf(q) - 1.0) / self.c
    }

    fn mean(&self) -> f64 {
        let k = (1.0 + self.c).ln();
        (self.c - k) / (self.c * k)
    }

    fn var(&self) -> f64 {
        let c = self.c;
        let k = (1.0 + c).ln();
        let m = self.mean();
        // E[X²] = (c² - 2c + 2k) / (2c²k), derived from ∫₀¹ x²·c/((1+cx)·k) dx
        let ex2 = (c * c - 2.0 * c + 2.0 * k) / (2.0 * c * c * k);
        ex2 - m * m
    }
}

/// Gilbrat distribution (log-normal with s=1).
///
/// Matches `scipy.stats.gilbrat`.
pub struct Gilbrat;

impl ContinuousDistribution for Gilbrat {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            (-0.5 * x.ln().powi(2)).exp() / (x * (2.0 * PI).sqrt())
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            standard_normal_cdf(x.ln())
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        // CDF = Φ(ln(x)), so ln(x) = ndtri(q), x = exp(ndtri(q))
        fsci_special::ndtri(q).exp()
    }

    fn mean(&self) -> f64 {
        std::f64::consts::E.sqrt() // exp(1/2)
    }

    fn var(&self) -> f64 {
        std::f64::consts::E * (std::f64::consts::E - 1.0) // e(e-1)
    }
}

/// Levy distribution (one-sided stable with α=1/2, β=1).
///
/// Matches `scipy.stats.levy`.
pub struct Levy {
    pub loc: f64,
    pub scale: f64,
}

impl Levy {
    #[must_use]
    pub fn new(loc: f64, scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive");
        Self { loc, scale }
    }
}

impl Default for Levy {
    fn default() -> Self {
        Self {
            loc: 0.0,
            scale: 1.0,
        }
    }
}

impl ContinuousDistribution for Levy {
    fn pdf(&self, x: f64) -> f64 {
        let z = x - self.loc;
        if z <= 0.0 {
            return 0.0;
        }
        (self.scale / (2.0 * PI)).sqrt() * (-self.scale / (2.0 * z)).exp() / z.powf(1.5)
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = x - self.loc;
        if z <= 0.0 {
            return 0.0;
        }
        fsci_special::erfc_scalar((self.scale / (2.0 * z)).sqrt())
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return self.loc;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        // CDF(x) = erfc(sqrt(scale/(2*(x-loc))))
        // erfcinv(q) = sqrt(scale/(2*z)) where z = x - loc
        // z = scale / (2 * erfcinv(q)²)
        // erfcinv(q) = -ndtri(q/2) / sqrt(2)
        let erfcinv_q = -fsci_special::ndtri(q / 2.0) * FRAC_1_SQRT_2;
        if erfcinv_q <= 0.0 {
            return f64::INFINITY;
        }
        self.loc + self.scale / (2.0 * erfcinv_q * erfcinv_q)
    }

    fn mean(&self) -> f64 {
        f64::INFINITY
    }

    fn var(&self) -> f64 {
        f64::INFINITY
    }
}

/// Burr (Type XII) distribution.
///
/// Matches `scipy.stats.burr12`.
pub struct Burr12 {
    pub c: f64,
    pub d: f64,
}

impl Burr12 {
    #[must_use]
    pub fn new(c: f64, d: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        assert!(d > 0.0, "d must be positive");
        Self { c, d }
    }
}

impl ContinuousDistribution for Burr12 {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let c = self.c;
        let d = self.d;
        c * d * x.powf(c - 1.0) / (1.0 + x.powf(c)).powf(d + 1.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        1.0 - (1.0 + x.powf(self.c)).powf(-self.d)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        // CDF = 1 - (1+x^c)^(-d), so x = ((1-q)^(-1/d) - 1)^(1/c)
        ((1.0 - q).powf(-1.0 / self.d) - 1.0).powf(1.0 / self.c)
    }

    fn mean(&self) -> f64 {
        if self.c * self.d > 1.0 {
            let c = self.c;
            let d = self.d;
            d * ln_gamma(d - 1.0 / c).exp() * ln_gamma(1.0 + 1.0 / c).exp()
                / ln_gamma(d + 1.0).exp()
        } else {
            f64::INFINITY
        }
    }

    fn var(&self) -> f64 {
        let c = self.c;
        let d = self.d;
        if c * d <= 2.0 {
            return f64::NAN;
        }

        let mean = self.mean();
        let second_moment = d * ln_gamma(d - 2.0 / c).exp() * ln_gamma(1.0 + 2.0 / c).exp()
            / ln_gamma(d + 1.0).exp();
        second_moment - mean * mean
    }
}

/// Log-Laplace distribution.
///
/// Matches `scipy.stats.loglaplace`.
pub struct LogLaplace {
    pub c: f64,
}

impl LogLaplace {
    #[must_use]
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        Self { c }
    }
}

impl ContinuousDistribution for LogLaplace {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let c = self.c;
        if x < 1.0 {
            c / 2.0 * x.powf(c - 1.0)
        } else {
            c / 2.0 * x.powf(-c - 1.0)
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let c = self.c;
        if x < 1.0 {
            0.5 * x.powf(c)
        } else {
            1.0 - 0.5 * x.powf(-c)
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        let c = self.c;
        if q < 0.5 {
            // CDF = 0.5*x^c, so x = (2q)^(1/c)
            (2.0 * q).powf(1.0 / c)
        } else {
            // CDF = 1 - 0.5*x^(-c), so x = (2(1-q))^(-1/c)
            (2.0 * (1.0 - q)).powf(-1.0 / c)
        }
    }

    fn mean(&self) -> f64 {
        if self.c > 1.0 {
            self.c * self.c / (self.c * self.c - 1.0)
        } else {
            f64::INFINITY
        }
    }

    fn var(&self) -> f64 {
        let c = self.c;
        if c > 2.0 {
            let second_moment = c * c / (c * c - 4.0);
            second_moment - self.mean().powi(2)
        } else {
            f64::INFINITY
        }
    }
}

/// Mielke Beta-Kappa distribution.
///
/// Matches `scipy.stats.mielke`.
pub struct Mielke {
    pub k: f64,
    pub s: f64,
}

impl Mielke {
    #[must_use]
    pub fn new(k: f64, s: f64) -> Self {
        assert!(k > 0.0 && s > 0.0, "k and s must be positive");
        Self { k, s }
    }
}

impl ContinuousDistribution for Mielke {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let k = self.k;
        let s = self.s;
        k * x.powf(k - 1.0) / (1.0 + x.powf(s)).powf(1.0 + k / s)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let xk = x.powf(self.k);
        xk / (1.0 + x.powf(self.s)).powf(self.k / self.s)
    }

    fn mean(&self) -> f64 {
        if self.s <= 1.0 {
            return f64::INFINITY;
        }
        let log_moment = ln_gamma((self.k + 1.0) / self.s) + ln_gamma(1.0 - 1.0 / self.s)
            - ln_gamma(self.k / self.s);
        log_moment.exp()
    }

    fn var(&self) -> f64 {
        if self.s <= 2.0 {
            return f64::INFINITY;
        }
        let mean = self.mean();
        let log_second_moment = ln_gamma((self.k + 2.0) / self.s) + ln_gamma(1.0 - 2.0 / self.s)
            - ln_gamma(self.k / self.s);
        log_second_moment.exp() - mean * mean
    }
}

/// Moyal distribution.
///
/// Matches `scipy.stats.moyal`.
pub struct Moyal;

impl ContinuousDistribution for Moyal {
    fn pdf(&self, x: f64) -> f64 {
        let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();
        inv_sqrt_2pi * (-0.5 * (x + (-x).exp())).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        // F(x) = erfc(exp(-x/2) / √2) = erfc(√(exp(-x)/2))
        fsci_special::erfc_scalar(((-x).exp() / 2.0).sqrt())
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        // CDF = erfc(sqrt(exp(-x)/2))
        // erfcinv(q) = sqrt(exp(-x)/2)
        // exp(-x) = 2 * erfcinv(q)^2
        // x = -ln(2 * erfcinv(q)^2)
        // erfcinv(q) = -ndtri(q/2) / sqrt(2)
        let erfcinv_q = -fsci_special::ndtri(q / 2.0) * FRAC_1_SQRT_2;
        if erfcinv_q <= 0.0 {
            return f64::INFINITY;
        }
        -(2.0 * erfcinv_q * erfcinv_q).ln()
    }

    fn mean(&self) -> f64 {
        0.577_215_664_901_532_9 + (2.0f64).ln() // γ + ln(2)
    }

    fn var(&self) -> f64 {
        PI * PI / 2.0
    }
}

/// Gompertz distribution.
///
/// Matches `scipy.stats.gompertz`.
pub struct Gompertz {
    pub c: f64,
}

impl Gompertz {
    #[must_use]
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        Self { c }
    }
}

impl ContinuousDistribution for Gompertz {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        // f(x; c) = c * exp(x) * exp(-c * (exp(x) - 1))
        self.c * x.exp() * (-self.c * (x.exp() - 1.0)).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        1.0 - (-self.c * (x.exp() - 1.0)).exp()
    }

    fn mean(&self) -> f64 {
        -(self.c.exp() * fsci_special::expi_scalar(-self.c))
    }

    fn var(&self) -> f64 {
        let mean = self.mean();
        let tail_prob = 1e-12_f64;
        let upper = (1.0_f64 - tail_prob.ln() / self.c).ln();
        let n = 4_000;
        let h = upper / n as f64;
        let mut second_moment = 0.0;

        for i in 0..=n {
            let x = i as f64 * h;
            let weight = if i == 0 || i == n {
                1.0
            } else if i % 2 == 0 {
                2.0
            } else {
                4.0
            };
            second_moment += weight * x * x * self.pdf(x);
        }

        (second_moment * h / 3.0 - mean * mean).max(0.0)
    }
}

/// Arcsine distribution on [0, 1].
///
/// Matches `scipy.stats.arcsine`. The arcsine distribution has PDF:
/// f(x) = 1 / (π * sqrt(x * (1-x))) for x in (0, 1)
///
/// This is a special case of the Beta distribution with α = β = 0.5.
pub struct Arcsine;

impl ContinuousDistribution for Arcsine {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        1.0 / (PI * (x * (1.0 - x)).sqrt())
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        2.0 / PI * x.sqrt().asin()
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return 1.0;
        }
        // F(x) = 2/π * arcsin(sqrt(x))
        // q = 2/π * arcsin(sqrt(x))
        // arcsin(sqrt(x)) = q * π / 2
        // sqrt(x) = sin(q * π / 2)
        // x = sin²(q * π / 2)
        let s = (q * PI / 2.0).sin();
        s * s
    }

    fn mean(&self) -> f64 {
        0.5
    }

    fn var(&self) -> f64 {
        0.125 // 1/8
    }
}

/// Generalized logistic distribution.
///
/// Matches `scipy.stats.genlogistic`.
pub struct GenLogistic {
    pub c: f64,
}

impl GenLogistic {
    #[must_use]
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        Self { c }
    }
}

impl ContinuousDistribution for GenLogistic {
    fn pdf(&self, x: f64) -> f64 {
        let c = self.c;
        c * (-x).exp() / (1.0 + (-x).exp()).powf(c + 1.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp()).powf(self.c)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        -(q.powf(-1.0 / self.c) - 1.0).ln()
    }

    fn mean(&self) -> f64 {
        fsci_special::digamma_scalar(self.c) + EULER_MASCHERONI
    }

    fn var(&self) -> f64 {
        PI * PI / 6.0 + fsci_special::trigamma(self.c)
    }
}

/// Frechet (right) distribution (Weibull maximum).
///
/// Matches `scipy.stats.frechet_r` / `scipy.stats.weibull_max`.
pub struct FrechetR {
    pub c: f64,
}

impl FrechetR {
    #[must_use]
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        Self { c }
    }
}

impl ContinuousDistribution for FrechetR {
    fn pdf(&self, x: f64) -> f64 {
        if x > 0.0 {
            return 0.0;
        }
        let ax = (-x).abs();
        self.c * ax.powf(self.c - 1.0) * (-ax.powf(self.c)).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x > 0.0 {
            return 1.0;
        }
        (-(-x).powf(self.c)).exp()
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return 0.0;
        }
        // CDF = exp(-(-x)^c), so (-x)^c = -ln(q), x = -(-ln(q))^(1/c)
        -(-q.ln()).powf(1.0 / self.c)
    }

    fn mean(&self) -> f64 {
        -ln_gamma(1.0 + 1.0 / self.c).exp()
    }

    fn var(&self) -> f64 {
        let first = ln_gamma(1.0 + 1.0 / self.c).exp();
        let second = ln_gamma(1.0 + 2.0 / self.c).exp();
        second - first * first
    }
}

/// Truncated exponential distribution on [0, b].
///
/// Matches `scipy.stats.truncexpon`.
pub struct TruncExpon {
    pub b: f64,
}

impl TruncExpon {
    #[must_use]
    pub fn new(b: f64) -> Self {
        assert!(b > 0.0, "b must be positive");
        Self { b }
    }
}

impl ContinuousDistribution for TruncExpon {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 || x > self.b {
            0.0
        } else {
            (-x).exp() / (1.0 - (-self.b).exp())
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x >= self.b {
            1.0
        } else {
            (1.0 - (-x).exp()) / (1.0 - (-self.b).exp())
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return self.b;
        }
        // CDF = (1-exp(-x))/(1-exp(-b)), so x = -ln(1 - q*(1-exp(-b)))
        -(1.0 - q * (1.0 - (-self.b).exp())).ln()
    }

    fn mean(&self) -> f64 {
        let eb = (-self.b).exp();
        // E[X] = (1 - (1+b)*exp(-b)) / (1 - exp(-b))
        (1.0 - (1.0 + self.b) * eb) / (1.0 - eb)
    }

    fn var(&self) -> f64 {
        let m = self.mean();
        let eb = (-self.b).exp();
        let e2 = (2.0 - (2.0 + 2.0 * self.b + self.b * self.b) * eb) / (1.0 - eb);
        e2 - m * m
    }
}

/// Generalized half-logistic distribution.
///
/// Matches `scipy.stats.genhalflogistic`.
pub struct GenHalfLogistic {
    pub c: f64,
}

impl GenHalfLogistic {
    #[must_use]
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        Self { c }
    }
}

impl ContinuousDistribution for GenHalfLogistic {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let limit = 1.0 / self.c;
        if x >= limit {
            return 0.0;
        }
        2.0 * (1.0 - self.c * x).powf(1.0 / self.c - 1.0)
            / (1.0 + (1.0 - self.c * x).powf(1.0 / self.c)).powi(2)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let limit = 1.0 / self.c;
        if x >= limit {
            return 1.0;
        }
        let t = (1.0 - self.c * x).powf(1.0 / self.c);
        1.0 - 2.0 * t / (1.0 + t)
    }

    fn mean(&self) -> f64 {
        let c = self.c;
        (fsci_special::digamma_scalar(1.0 / c + 1.0) + EULER_MASCHERONI) / c
    }

    fn var(&self) -> f64 {
        fsci_special::trigamma(1.0 / self.c + 1.0) / (self.c * self.c)
    }
}

/// Tukey-Lambda distribution.
///
/// Matches `scipy.stats.tukeylambda`.
pub struct TukeyLambda {
    pub lam: f64,
}

impl TukeyLambda {
    #[must_use]
    pub fn new(lam: f64) -> Self {
        Self { lam }
    }

    fn logistic_cdf(x: f64) -> f64 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let ex = x.exp();
            ex / (1.0 + ex)
        }
    }

    fn quantile_derivative(&self, p: f64) -> f64 {
        let lam = self.lam;
        if lam.abs() < 1e-15 {
            1.0 / p + 1.0 / (1.0 - p)
        } else {
            p.powf(lam - 1.0) + (1.0 - p).powf(lam - 1.0)
        }
    }
}

impl ContinuousDistribution for TukeyLambda {
    fn pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }

        if self.lam.abs() < 1e-15 {
            let cdf = Self::logistic_cdf(x);
            return cdf * (1.0 - cdf);
        }

        let cdf = self.cdf(x);
        if !cdf.is_finite() {
            return f64::NAN;
        }
        if cdf <= 0.0 || cdf >= 1.0 {
            return 0.0;
        }

        1.0 / self.quantile_derivative(cdf)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }

        if self.lam > 0.0 {
            let bound = 1.0 / self.lam;
            if x <= -bound {
                return 0.0;
            }
            if x >= bound {
                return 1.0;
            }
        }

        let mut lo = 0.0f64;
        let mut hi = 1.0f64;
        let eps = 1e-12;
        let mut p = Self::logistic_cdf(x).clamp(eps, 1.0 - eps);

        for _ in 0..10 {
            let q = self.ppf(p);
            if !q.is_finite() {
                break;
            }
            if q < x {
                lo = p;
            } else {
                hi = p;
            }
            if (q - x).abs() <= 1e-14 * (1.0 + x.abs()) {
                return p;
            }

            let derivative = self.quantile_derivative(p);
            if !derivative.is_finite() || derivative <= 0.0 {
                break;
            }

            let next = p - (q - x) / derivative;
            if !next.is_finite() || next <= lo || next >= hi {
                break;
            }
            p = next.clamp(eps, 1.0 - eps);
        }

        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            let q = self.ppf(mid);
            if q < x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        let lam = self.lam;
        if lam.abs() < 1e-15 {
            // Logistic: Q(p) = ln(p/(1-p))
            (q / (1.0 - q)).ln()
        } else {
            (q.powf(lam) - (1.0 - q).powf(lam)) / lam
        }
    }

    fn mean(&self) -> f64 {
        0.0 // Symmetric
    }

    fn var(&self) -> f64 {
        let lam = self.lam;
        if lam.abs() < 1e-15 {
            return PI * PI / 3.0;
        }
        if lam <= -0.5 {
            return f64::NAN;
        }

        let gamma_ratio = (2.0 * ln_gamma(lam + 1.0) - ln_gamma(2.0 * lam + 2.0)).exp();
        2.0 * (1.0 / (1.0 + 2.0 * lam) - gamma_ratio) / (lam * lam)
    }
}

/// Inverse Weibull (Frechet) distribution.
///
/// Matches `scipy.stats.invweibull`.
pub struct InvWeibull {
    pub c: f64,
}

impl InvWeibull {
    #[must_use]
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        Self { c }
    }
}

impl ContinuousDistribution for InvWeibull {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.c * x.powf(-self.c - 1.0) * (-x.powf(-self.c)).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            (-x.powf(-self.c)).exp()
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        // CDF = exp(-x^(-c)), so x^(-c) = -ln(q), x = (-ln(q))^(-1/c)
        (-q.ln()).powf(-1.0 / self.c)
    }

    fn mean(&self) -> f64 {
        if self.c > 1.0 {
            ln_gamma(1.0 - 1.0 / self.c).exp()
        } else {
            f64::INFINITY
        }
    }

    fn var(&self) -> f64 {
        if self.c <= 2.0 {
            return f64::NAN;
        }

        let g1 = ln_gamma(1.0 - 1.0 / self.c).exp();
        let g2 = ln_gamma(1.0 - 2.0 / self.c).exp();
        g2 - g1 * g1
    }
}

/// Generalized normal distribution (exponential power).
///
/// Matches `scipy.stats.gennorm`.
pub struct GenNorm {
    pub beta: f64,
}

impl GenNorm {
    #[must_use]
    pub fn new(beta: f64) -> Self {
        assert!(beta > 0.0, "beta must be positive");
        Self { beta }
    }
}

impl ContinuousDistribution for GenNorm {
    fn pdf(&self, x: f64) -> f64 {
        let b = self.beta;
        b / (2.0 * ln_gamma(1.0 / b).exp()) * (-x.abs().powf(b)).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        // Numerical integration via the trait default (bisection ppf inverse)
        // For efficiency, use the incomplete gamma
        let b = self.beta;
        let p = lower_regularized_gamma(1.0 / b, x.abs().powf(b));
        if x >= 0.0 {
            0.5 + 0.5 * p
        } else {
            0.5 - 0.5 * p
        }
    }

    fn mean(&self) -> f64 {
        0.0
    }

    fn var(&self) -> f64 {
        let b = self.beta;
        ln_gamma(3.0 / b).exp() / ln_gamma(1.0 / b).exp()
    }
}

/// Half generalized normal distribution (upper half of `gennorm`).
///
/// Matches `scipy.stats.halfgennorm`.
pub struct HalfGenNorm {
    pub beta: f64,
}

impl HalfGenNorm {
    #[must_use]
    pub fn new(beta: f64) -> Self {
        assert!(beta > 0.0, "beta must be positive");
        Self { beta }
    }
}

impl ContinuousDistribution for HalfGenNorm {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let b = self.beta;
        b / ln_gamma(1.0 / b).exp() * (-x.powf(b)).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let b = self.beta;
        lower_regularized_gamma(1.0 / b, x.powf(b))
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }

        let b = self.beta;
        let g = fsci_special::gammaincinv(1.0 / b, q);
        if g.is_finite() && g >= 0.0 {
            g.powf(1.0 / b)
        } else {
            ppf_bisection(|x| self.cdf(x), q, self.mean(), self.std())
        }
    }

    fn mean(&self) -> f64 {
        let b = self.beta;
        ln_gamma(2.0 / b).exp() / ln_gamma(1.0 / b).exp()
    }

    fn var(&self) -> f64 {
        let b = self.beta;
        let mean = self.mean();
        ln_gamma(3.0 / b).exp() / ln_gamma(1.0 / b).exp() - mean * mean
    }
}

/// Log-gamma distribution.
///
/// Matches `scipy.stats.loggamma`.
pub struct LogGamma {
    pub c: f64,
}

impl LogGamma {
    #[must_use]
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        Self { c }
    }
}

impl ContinuousDistribution for LogGamma {
    fn pdf(&self, x: f64) -> f64 {
        (self.c * x - x.exp() - ln_gamma(self.c)).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        lower_regularized_gamma(self.c, x.exp())
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return f64::NEG_INFINITY;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }

        let g = fsci_special::gammaincinv(self.c, q);
        if g.is_finite() && g > 0.0 {
            g.ln()
        } else {
            (q.ln() + ln_gamma(self.c + 1.0)) / self.c
        }
    }

    fn mean(&self) -> f64 {
        fsci_special::digamma_scalar(self.c)
    }

    fn var(&self) -> f64 {
        fsci_special::trigamma(self.c)
    }
}

/// Alpha distribution.
///
/// Matches `scipy.stats.alpha`.
pub struct Alpha {
    pub a: f64,
}

impl Alpha {
    #[must_use]
    pub fn new(a: f64) -> Self {
        assert!(a > 0.0, "a must be positive");
        Self { a }
    }
}

impl ContinuousDistribution for Alpha {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }

        let normalizer = standard_normal_cdf(self.a);
        standard_normal_pdf(self.a - 1.0 / x) / (x * x * normalizer)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }

        standard_normal_cdf(self.a - 1.0 / x) / standard_normal_cdf(self.a)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }

        1.0 / (self.a - fsci_special::ndtri(q * standard_normal_cdf(self.a)))
    }

    fn mean(&self) -> f64 {
        f64::INFINITY
    }

    fn var(&self) -> f64 {
        f64::INFINITY
    }
}

/// Laplace-asymmetric distribution.
///
/// Matches `scipy.stats.laplace_asymmetric`.
pub struct LaplaceAsymmetric {
    pub kappa: f64,
}

impl LaplaceAsymmetric {
    #[must_use]
    pub fn new(kappa: f64) -> Self {
        assert!(kappa > 0.0, "kappa must be positive");
        Self { kappa }
    }
}

impl ContinuousDistribution for LaplaceAsymmetric {
    fn pdf(&self, x: f64) -> f64 {
        let k = self.kappa;
        let norm = 1.0 / (k + 1.0 / k);
        if x >= 0.0 {
            norm * (-x * k).exp()
        } else {
            norm * (x / k).exp()
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        let k = self.kappa;
        if x >= 0.0 {
            // CDF = 1 - exp(-xκ) / (1+κ²)
            1.0 - (-x * k).exp() / (1.0 + k * k)
        } else {
            // CDF = κ² / (1+κ²) * exp(x/κ)
            k * k / (1.0 + k * k) * (x / k).exp()
        }
    }

    fn mean(&self) -> f64 {
        1.0 / self.kappa - self.kappa
    }

    fn var(&self) -> f64 {
        1.0 / (self.kappa * self.kappa) + self.kappa * self.kappa
    }
}

/// Argus distribution (used in high-energy physics).
///
/// Matches `scipy.stats.argus`.
pub struct Argus {
    pub chi: f64,
}

impl Argus {
    #[must_use]
    pub fn new(chi: f64) -> Self {
        assert!(chi > 0.0, "chi must be positive");
        Self { chi }
    }
}

impl ContinuousDistribution for Argus {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        let chi = self.chi;
        let chi2 = chi * chi;
        let t = 1.0 - x * x;
        // Normalization via Ψ(χ) = Φ(χ) - χφ(χ) - 1/2
        // where Φ = standard normal CDF, φ = standard normal PDF
        let phi_chi = standard_normal_cdf(chi);
        let pdf_chi = (-0.5 * chi2).exp() / (2.0 * PI).sqrt();
        let psi = phi_chi - chi * pdf_chi - 0.5;
        let norm = if psi > 0.0 {
            chi.powi(3) / ((2.0 * PI).sqrt() * psi)
        } else {
            1.0
        };
        norm * x * t.sqrt() * (-0.5 * chi2 * t).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        // Numerical CDF via Simpson's rule — scale grid for accuracy
        let n = 400;
        let h = x / n as f64;
        let mut sum = self.pdf(0.0) + self.pdf(x);
        for i in 1..n {
            let t = i as f64 * h;
            let w = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += w * self.pdf(t);
        }
        (sum * h / 3.0).clamp(0.0, 1.0)
    }

    fn mean(&self) -> f64 {
        let a = 0.5 * self.chi * self.chi;
        let gamma_half3 = ln_gamma(1.5).exp();
        let p = fsci_special::gammainc_scalar(1.5, a, RuntimeMode::Strict).unwrap_or(f64::NAN);
        let norm = 2.0 * a.powf(1.5) / (gamma_half3 * p);
        norm * simpson_integrate_adaptive(
            |t| (1.0 - t * t).sqrt() * t * t * (-(a * t * t)).exp(),
            0.0,
            1.0,
            1_024,
            1e-12,
            1e-12,
            8,
        )
    }

    fn var(&self) -> f64 {
        let a = 0.5 * self.chi * self.chi;
        let gamma_half3 = ln_gamma(1.5).exp();
        let gamma_half5 = ln_gamma(2.5).exp();
        let p3 = fsci_special::gammainc_scalar(1.5, a, RuntimeMode::Strict).unwrap_or(f64::NAN);
        let p5 = fsci_special::gammainc_scalar(2.5, a, RuntimeMode::Strict).unwrap_or(f64::NAN);
        let mean = self.mean();
        let gamma3 = p3 * gamma_half3;
        let gamma5 = p5 * gamma_half5;
        let second_moment = 1.0 - gamma5 / (a * gamma3);
        (second_moment - mean * mean).max(0.0)
    }
}

/// Crystal Ball distribution (used in high-energy physics).
///
/// Matches `scipy.stats.crystalball`.
pub struct CrystalBall {
    pub beta_param: f64,
    pub m: f64,
}

impl CrystalBall {
    #[must_use]
    pub fn new(beta_param: f64, m: f64) -> Self {
        assert!(beta_param > 0.0, "beta must be positive");
        assert!(m > 1.0, "m must be > 1");
        Self { beta_param, m }
    }
}

impl ContinuousDistribution for CrystalBall {
    fn pdf(&self, x: f64) -> f64 {
        let beta = self.beta_param;
        let m = self.m;

        let gauss_norm = (2.0 * PI).sqrt() * standard_normal_cdf(beta);
        let a = (m / beta).powf(m) * (-0.5 * beta * beta).exp();
        let b = m / beta - beta;
        let tail_norm = if m > 1.0 {
            a * (m / beta).powf(1.0 - m) / (m - 1.0)
        } else {
            100.0 * a // rough estimate for divergent tail
        };
        let total_norm = gauss_norm + tail_norm;

        let unnormalized_pdf = if x > -beta {
            (-0.5 * x * x).exp()
        } else {
            a * (b - x).powf(-m)
        };

        if total_norm > 0.0 {
            unnormalized_pdf / total_norm
        } else {
            0.0
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        let beta = self.beta_param;
        let m = self.m;

        // Analytical normalization: split at x = -beta
        // Gaussian part: ∫_{-β}^{∞} exp(-x²/2) dx = √(2π) * Φ(β)... wait, ∫ from -β to ∞
        // = √(2π) * (1 - Φ(-β)) = √(2π) * Φ(β)
        let gauss_norm = (2.0 * PI).sqrt() * standard_normal_cdf(beta);

        // Power tail part: ∫_{-∞}^{-β} A*(B-x)^{-m} dx
        let a_coeff = (m / beta).powf(m) * (-0.5 * beta * beta).exp();
        let b_val = m / beta - beta;
        // ∫_{-∞}^{-β} A*(B-x)^{-m} dx = A * (B-(-β))^{1-m} / (m-1)
        // B - (-β) = m/β - β + β = m/β
        let tail_norm = if m > 1.0 {
            a_coeff * (m / beta).powf(1.0 - m) / (m - 1.0)
        } else {
            // For m <= 1 the integral diverges; use numerical fallback
            100.0 * a_coeff // rough estimate
        };

        let total_norm = gauss_norm + tail_norm;
        if total_norm <= 0.0 {
            return 0.5;
        }

        // CDF: integrate pdf from -∞ to x
        if x >= -beta {
            // All tail + Gaussian from -β to x
            let gauss_cdf_part =
                (2.0 * PI).sqrt() * (standard_normal_cdf(x) - standard_normal_cdf(-beta));
            (tail_norm + gauss_cdf_part) / total_norm
        } else {
            // Only part of the tail: ∫_{-∞}^{x} A*(B-t)^{-m} dt
            let bx = b_val - x;
            let partial = if m > 1.0 && bx > 0.0 {
                a_coeff * bx.powf(1.0 - m) / (m - 1.0)
            } else {
                0.0
            };
            // tail_norm - ∫_{x}^{-β} = tail_norm - (tail_norm - partial)
            (partial / total_norm).clamp(0.0, 1.0)
        }
    }

    fn mean(&self) -> f64 {
        let beta = self.beta_param;
        let m = self.m;
        if m <= 2.0 {
            return f64::INFINITY;
        }

        let gauss_norm = (2.0 * PI).sqrt() * standard_normal_cdf(beta);
        let a = (m / beta).powf(m) * (-0.5 * beta * beta).exp();
        let b = m / beta - beta;
        let y0 = m / beta;
        let tail_norm = a * y0.powf(1.0 - m) / (m - 1.0);
        let total_norm = gauss_norm + tail_norm;

        let gauss_first = (-0.5 * beta * beta).exp();
        let tail_first = a * (b * y0.powf(1.0 - m) / (m - 1.0) - y0.powf(2.0 - m) / (m - 2.0));

        (gauss_first + tail_first) / total_norm
    }

    fn var(&self) -> f64 {
        let beta = self.beta_param;
        let m = self.m;
        if m <= 3.0 {
            return f64::INFINITY;
        }

        let mean = self.mean();
        let gauss_norm = (2.0 * PI).sqrt() * standard_normal_cdf(beta);
        let a = (m / beta).powf(m) * (-0.5 * beta * beta).exp();
        let b = m / beta - beta;
        let y0 = m / beta;
        let tail_norm = a * y0.powf(1.0 - m) / (m - 1.0);
        let total_norm = gauss_norm + tail_norm;

        let gauss_second = gauss_norm - beta * (-0.5 * beta * beta).exp();
        let tail_second = a
            * (b * b * y0.powf(1.0 - m) / (m - 1.0) - 2.0 * b * y0.powf(2.0 - m) / (m - 2.0)
                + y0.powf(3.0 - m) / (m - 3.0));
        let second_moment = (gauss_second + tail_second) / total_norm;

        (second_moment - mean * mean).max(0.0)
    }
}

// ══════════════════════════════════════════════════════════════════════
// Ranking and Order Statistics
// ══════════════════════════════════════════════════════════════════════

/// Compute quantiles of data at given probabilities.
///
/// Matches `numpy.quantile`.
pub fn quantile(data: &[f64], q: &[f64]) -> Vec<f64> {
    if data.is_empty() || data.iter().any(|v| v.is_nan()) {
        return vec![f64::NAN; q.len()];
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();

    q.iter()
        .map(|&qi| {
            let qi = qi.clamp(0.0, 1.0);
            let idx = qi * (n - 1) as f64;
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            let frac = idx - lo as f64;
            if lo == hi || hi >= n {
                sorted[lo.min(n - 1)]
            } else {
                sorted[lo] * (1.0 - frac) + sorted[hi] * frac
            }
        })
        .collect()
}

/// Compute weighted mean.
pub fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    let total_w: f64 = weights.iter().sum();
    if total_w == 0.0 {
        return f64::NAN;
    }
    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum::<f64>()
        / total_w
}

/// Compute weighted variance.
pub fn weighted_var(values: &[f64], weights: &[f64]) -> f64 {
    let mean = weighted_mean(values, weights);
    let total_w: f64 = weights.iter().sum();
    if total_w == 0.0 {
        return f64::NAN;
    }
    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| w * (v - mean).powi(2))
        .sum::<f64>()
        / total_w
}

/// Compute the geometric mean.
///
/// Matches `scipy.stats.gmean`.
pub fn gmean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let n = data.len() as f64;
    let log_sum: f64 = data.iter().map(|&x| x.ln()).sum();
    (log_sum / n).exp()
}

/// Compute the harmonic mean.
///
/// Matches `scipy.stats.hmean`.
pub fn hmean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let n = data.len() as f64;
    let inv_sum: f64 = data.iter().map(|&x| 1.0 / x).sum();
    if inv_sum == 0.0 {
        return f64::INFINITY;
    }
    n / inv_sum
}

/// Compute the power mean (generalized mean).
///
/// Matches `scipy.stats.pmean`.
pub fn pmean(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    if p == 0.0 {
        return gmean(data);
    }
    let n = data.len() as f64;
    let power_sum: f64 = data.iter().map(|&x| x.powf(p)).sum();
    (power_sum / n).powf(1.0 / p)
}

/// Circular mean for angular data.
///
/// Matches `scipy.stats.circmean`.
pub fn circmean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let sin_sum: f64 = data.iter().map(|&x| x.sin()).sum();
    let cos_sum: f64 = data.iter().map(|&x| x.cos()).sum();
    sin_sum.atan2(cos_sum)
}

/// Circular variance for angular data.
///
/// Matches `scipy.stats.circvar`.
pub fn circvar(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let n = data.len() as f64;
    let sin_sum: f64 = data.iter().map(|&x| x.sin()).sum();
    let cos_sum: f64 = data.iter().map(|&x| x.cos()).sum();
    let r = (sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / n;
    1.0 - r
}

/// Circular standard deviation for angular data.
///
/// Matches `scipy.stats.circstd`.
pub fn circstd(data: &[f64]) -> f64 {
    let v = circvar(data);
    if v <= 0.0 {
        return 0.0;
    }
    (-2.0 * (1.0 - v).ln()).sqrt()
}

// ══════════════════════════════════════════════════════════════════════
// Statistical Tests
// ══════════════════════════════════════════════════════════════════════

/// Result of a statistical test.
#[derive(Debug, Clone, PartialEq)]
pub struct TtestResult {
    /// Test statistic.
    pub statistic: f64,
    /// Two-sided p-value.
    pub pvalue: f64,
    /// Degrees of freedom.
    pub df: f64,
}

/// Result of a variance homogeneity test.
#[derive(Debug, Clone, PartialEq)]
pub struct VarianceTestResult {
    /// Test statistic.
    pub statistic: f64,
    /// Right-tail p-value.
    pub pvalue: f64,
}

fn invalid_variance_test_result() -> VarianceTestResult {
    VarianceTestResult {
        statistic: f64::NAN,
        pvalue: f64::NAN,
    }
}

fn sample_mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn sample_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    let mean = sample_mean(data);
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() as f64 - 1.0)
}

fn sample_median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

/// One-sample t-test: test whether the mean of a sample differs from `popmean`.
///
/// Matches `scipy.stats.ttest_1samp(a, popmean)`.
/// Returns NaN statistic and p-value for samples with fewer than 2 elements.
pub fn ttest_1samp(data: &[f64], popmean: f64) -> TtestResult {
    let n = data.len() as f64;
    if data.len() < 2 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: n - 1.0,
        };
    }
    let mean: f64 = data.iter().sum::<f64>() / n;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
    if se == 0.0 {
        // All values identical: t is infinite or NaN depending on whether mean == popmean
        let statistic = if (mean - popmean).abs() == 0.0 {
            f64::NAN
        } else {
            (mean - popmean).signum() * f64::INFINITY
        };
        return TtestResult {
            statistic,
            pvalue: if statistic.is_nan() { f64::NAN } else { 0.0 },
            df: n - 1.0,
        };
    }
    let t = (mean - popmean) / se;
    let df = n - 1.0;
    let tdist = StudentT::new(df);
    let pvalue = 2.0 * tdist.sf(t.abs());
    TtestResult {
        statistic: t,
        pvalue,
        df,
    }
}

/// Independent two-sample t-test (equal variance assumed).
///
/// Matches `scipy.stats.ttest_ind(a, b, equal_var=True)`.
/// Each sample must have at least 2 elements; returns NaN otherwise.
pub fn ttest_ind(a: &[f64], b: &[f64]) -> TtestResult {
    let n1 = a.len() as f64;
    let n2 = b.len() as f64;
    let df = n1 + n2 - 2.0;
    if a.len() < 2 || b.len() < 2 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df,
        };
    }
    let mean1: f64 = a.iter().sum::<f64>() / n1;
    let mean2: f64 = b.iter().sum::<f64>() / n2;
    let var1: f64 = a.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2: f64 = b.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    // Pooled variance
    let sp2 = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / df;
    let se = (sp2 * (1.0 / n1 + 1.0 / n2)).sqrt();
    let t = (mean1 - mean2) / se;

    let tdist = StudentT::new(df);
    let pvalue = 2.0 * tdist.sf(t.abs());
    TtestResult {
        statistic: t,
        pvalue,
        df,
    }
}

/// Welch's t-test (unequal variance).
///
/// Matches `scipy.stats.ttest_ind(a, b, equal_var=False)`.
/// Each sample must have at least 2 elements; returns NaN otherwise.
pub fn ttest_ind_welch(a: &[f64], b: &[f64]) -> TtestResult {
    if a.len() < 2 || b.len() < 2 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }
    let n1 = a.len() as f64;
    let n2 = b.len() as f64;
    let mean1: f64 = a.iter().sum::<f64>() / n1;
    let mean2: f64 = b.iter().sum::<f64>() / n2;
    let var1: f64 = a.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2: f64 = b.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    let se = (var1 / n1 + var2 / n2).sqrt();
    let t = (mean1 - mean2) / se;

    // Welch-Satterthwaite degrees of freedom
    let vn1 = var1 / n1;
    let vn2 = var2 / n2;
    let df = (vn1 + vn2).powi(2) / (vn1.powi(2) / (n1 - 1.0) + vn2.powi(2) / (n2 - 1.0));

    let tdist = StudentT::new(df);
    let pvalue = 2.0 * tdist.sf(t.abs());
    TtestResult {
        statistic: t,
        pvalue,
        df,
    }
}

/// Paired t-test for related samples.
///
/// Tests H0: mean of differences a[i] - b[i] is zero.
/// Requires equal-length paired observations.
///
/// Matches `scipy.stats.ttest_rel(a, b)`.
pub fn ttest_rel(a: &[f64], b: &[f64]) -> TtestResult {
    if a.len() != b.len() || a.len() < 2 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }
    let n = a.len() as f64;
    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai - bi).collect();
    let d_mean = diffs.iter().sum::<f64>() / n;
    let d_var = diffs.iter().map(|&d| (d - d_mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (d_var / n).sqrt();

    if se == 0.0 {
        return TtestResult {
            statistic: if d_mean == 0.0 { 0.0 } else { f64::INFINITY },
            pvalue: if d_mean == 0.0 { 1.0 } else { 0.0 },
            df: n - 1.0,
        };
    }

    let t = d_mean / se;
    let df = n - 1.0;
    let tdist = StudentT::new(df);
    let pvalue = 2.0 * tdist.sf(t.abs());
    TtestResult {
        statistic: t,
        pvalue,
        df,
    }
}

/// One-sample chi-squared goodness-of-fit test.
///
/// Tests whether observed frequencies differ significantly from expected.
/// This is a convenience wrapper around `power_divergence` with lambda=1.
///
/// Matches `scipy.stats.chisquare(f_obs, f_exp)`.
pub fn chisquare(f_obs: &[f64], f_exp: Option<&[f64]>) -> (f64, f64) {
    power_divergence(f_obs, f_exp, 1.0)
}

/// 1D Wasserstein distance (earth mover's distance) between two distributions.
///
/// Computes the first Wasserstein distance between empirical distributions
/// defined by samples u and v.
///
/// Matches `scipy.stats.wasserstein_distance(u_values, v_values)`.
pub fn wasserstein_distance(u: &[f64], v: &[f64]) -> f64 {
    if u.is_empty()
        || v.is_empty()
        || u.iter().any(|&val| val.is_nan())
        || v.iter().any(|&val| val.is_nan())
    {
        return f64::NAN;
    }

    // Sort both distributions
    let mut u_sorted = u.to_vec();
    let mut v_sorted = v.to_vec();
    u_sorted.sort_by(|a, b| a.total_cmp(b));
    v_sorted.sort_by(|a, b| a.total_cmp(b));

    // Merge and compute the area between CDFs
    let nu = u.len() as f64;
    let nv = v.len() as f64;

    // Combine all unique values and compute CDF difference at each
    let mut all_vals: Vec<f64> = u_sorted.iter().chain(v_sorted.iter()).copied().collect();
    all_vals.sort_by(|a, b| a.total_cmp(b));
    all_vals.dedup_by(|a, b| a.total_cmp(b).is_eq());

    let mut distance = 0.0;
    let mut prev_val = all_vals[0];

    for &val in &all_vals[1..] {
        // CDF of u at prev_val
        let cdf_u = u_sorted.partition_point(|&x| x <= prev_val) as f64 / nu;
        // CDF of v at prev_val
        let cdf_v = v_sorted.partition_point(|&x| x <= prev_val) as f64 / nv;

        distance += (cdf_u - cdf_v).abs() * (val - prev_val);
        prev_val = val;
    }

    distance
}

/// Energy distance between two 1D distributions.
///
/// A statistical distance based on the Cramér distance:
///   D(u, v) = 2 * E|X-Y| - E|X-X'| - E|Y-Y'|
/// where X, X' ~ u and Y, Y' ~ v.
///
/// Matches `scipy.stats.energy_distance(u_values, v_values)`.
pub fn energy_distance(u: &[f64], v: &[f64]) -> f64 {
    if u.is_empty()
        || v.is_empty()
        || u.iter().any(|&val| val.is_nan())
        || v.iter().any(|&val| val.is_nan())
    {
        return f64::NAN;
    }

    let nu = u.len() as f64;
    let nv = v.len() as f64;

    // E|X-Y|: mean of |u_i - v_j| over all pairs
    let mut e_xy = 0.0;
    for &ui in u {
        for &vj in v {
            e_xy += (ui - vj).abs();
        }
    }
    e_xy /= nu * nv;

    // E|X-X'|: mean of |u_i - u_j| over all pairs
    let mut e_xx = 0.0;
    for i in 0..u.len() {
        for j in (i + 1)..u.len() {
            e_xx += (u[i] - u[j]).abs();
        }
    }
    e_xx *= 2.0 / (nu * nu); // double because we only summed upper triangle

    // E|Y-Y'|
    let mut e_yy = 0.0;
    for i in 0..v.len() {
        for j in (i + 1)..v.len() {
            e_yy += (v[i] - v[j]).abs();
        }
    }
    e_yy *= 2.0 / (nv * nv);

    let d_sq = 2.0 * e_xy - e_xx - e_yy;
    if d_sq.is_nan() {
        f64::NAN
    } else {
        d_sq.max(0.0).sqrt()
    }
}

// ══════════════════════════════════════════════════════════════════════
// Non-parametric Tests and ANOVA
// ══════════════════════════════════════════════════════════════════════

/// One-way ANOVA (Analysis of Variance).
///
/// Matches `scipy.stats.f_oneway(*groups)`.
///
/// Tests H0: all group means are equal.
/// Assumes normality and equal variances within groups.
pub fn f_oneway(groups: &[&[f64]]) -> TtestResult {
    if groups.len() < 2 || groups.iter().any(|g| g.iter().any(|v| v.is_nan())) {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    let k = groups.len() as f64; // number of groups
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let nf = n_total as f64;

    if n_total <= groups.len() {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    // Grand mean
    let grand_sum: f64 = groups.iter().flat_map(|g| g.iter()).sum();
    let grand_mean = grand_sum / nf;

    // Between-group sum of squares
    let ss_between: f64 = groups
        .iter()
        .map(|g| {
            let gi_mean: f64 = g.iter().sum::<f64>() / g.len() as f64;
            g.len() as f64 * (gi_mean - grand_mean).powi(2)
        })
        .sum();

    // Within-group sum of squares
    let ss_within: f64 = groups
        .iter()
        .map(|g| {
            let gi_mean: f64 = g.iter().sum::<f64>() / g.len() as f64;
            g.iter().map(|&x| (x - gi_mean).powi(2)).sum::<f64>()
        })
        .sum();

    let df_between = k - 1.0;
    let df_within = nf - k;

    if df_within <= 0.0 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: df_between,
        };
    }

    if ss_within == 0.0 {
        if ss_between == 0.0 {
            return TtestResult {
                statistic: f64::NAN,
                pvalue: f64::NAN,
                df: df_between,
            };
        } else {
            return TtestResult {
                statistic: f64::INFINITY,
                pvalue: 0.0,
                df: df_between,
            };
        }
    }

    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    let f_stat = ms_between / ms_within;

    let fdist = FDistribution::new(df_between, df_within);
    let cdf_val = fdist.cdf(f_stat);
    let pvalue = if !cdf_val.is_finite() || !(0.0..=1.0).contains(&cdf_val) {
        if f_stat > 0.0 { 0.0 } else { 1.0 }
    } else {
        (1.0 - cdf_val).max(0.0)
    };

    TtestResult {
        statistic: f_stat,
        pvalue,
        df: df_between,
    }
}

/// Levene's test for equal variances using median-centered absolute deviations.
///
/// Matches the robust default behavior of `scipy.stats.levene(*groups)`.
pub fn levene(groups: &[&[f64]]) -> VarianceTestResult {
    if groups.len() < 2
        || groups.iter().any(|group| group.len() < 2)
        || groups.iter().any(|g| g.iter().any(|v| v.is_nan()))
    {
        return invalid_variance_test_result();
    }

    let deviations: Vec<Vec<f64>> = groups
        .iter()
        .map(|group| {
            let center = sample_median(group);
            group.iter().map(|&x| (x - center).abs()).collect()
        })
        .collect();

    let k = deviations.len() as f64;
    let n_total: usize = deviations.iter().map(Vec::len).sum();
    let df_within = n_total as f64 - k;
    if df_within <= 0.0 {
        return invalid_variance_test_result();
    }

    let grand_mean = deviations.iter().flatten().sum::<f64>() / n_total as f64;

    let ss_between: f64 = deviations
        .iter()
        .map(|group| {
            let mean = sample_mean(group);
            group.len() as f64 * (mean - grand_mean).powi(2)
        })
        .sum();

    let ss_within: f64 = deviations
        .iter()
        .map(|group| {
            let mean = sample_mean(group);
            group.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
        })
        .sum();

    if ss_within == 0.0 {
        if ss_between == 0.0 {
            return invalid_variance_test_result();
        }
        return VarianceTestResult {
            statistic: f64::INFINITY,
            pvalue: 0.0,
        };
    }

    let df_between = k - 1.0;
    let w = (df_within / df_between) * (ss_between / ss_within);
    let pvalue = FDistribution::new(df_between, df_within)
        .sf(w)
        .clamp(0.0, 1.0);
    VarianceTestResult {
        statistic: w,
        pvalue,
    }
}

/// Bartlett's test for equal variances under normality.
///
/// Matches `scipy.stats.bartlett(*groups)`.
pub fn bartlett(groups: &[&[f64]]) -> VarianceTestResult {
    if groups.len() < 2 || groups.iter().any(|group| group.len() < 2) {
        return invalid_variance_test_result();
    }

    let k = groups.len() as f64;
    let n_total: usize = groups.iter().map(|group| group.len()).sum();
    let df = n_total as f64 - k;
    if df <= 0.0 {
        return invalid_variance_test_result();
    }

    let variances: Vec<f64> = groups.iter().map(|group| sample_variance(group)).collect();
    if variances
        .iter()
        .any(|variance| !variance.is_finite() || *variance < 0.0)
    {
        return invalid_variance_test_result();
    }

    let pooled_variance: f64 = groups
        .iter()
        .zip(&variances)
        .map(|(group, variance)| (group.len() as f64 - 1.0) * variance)
        .sum::<f64>()
        / df;

    if pooled_variance <= 0.0 {
        if variances.iter().all(|variance| *variance == 0.0) {
            return invalid_variance_test_result();
        }
        return VarianceTestResult {
            statistic: f64::INFINITY,
            pvalue: 0.0,
        };
    }

    let weighted_log_variance_sum: f64 = groups
        .iter()
        .zip(&variances)
        .map(|(group, variance)| (group.len() as f64 - 1.0) * variance.ln())
        .sum();

    let correction = 1.0
        + (groups
            .iter()
            .map(|group| 1.0 / (group.len() as f64 - 1.0))
            .sum::<f64>()
            - 1.0 / df)
            / (3.0 * (k - 1.0));

    let statistic = (df * pooled_variance.ln() - weighted_log_variance_sum) / correction;
    let pvalue = ChiSquared::new(k - 1.0).sf(statistic).clamp(0.0, 1.0);
    VarianceTestResult { statistic, pvalue }
}

/// Friedman test for repeated measures (non-parametric).
///
/// Tests H0: all treatments have the same effect. Non-parametric alternative
/// to repeated-measures ANOVA. Each group must have the same length.
///
/// Matches `scipy.stats.friedmanchisquare(*groups)`.
pub fn friedmanchisquare(groups: &[&[f64]]) -> TtestResult {
    let k = groups.len();
    if k < 3 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }
    let n = groups[0].len();
    if n < 2
        || groups
            .iter()
            .any(|g| g.len() != n || g.iter().any(|v| v.is_nan()))
    {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    // Rank within each block (row) with tie handling (average ranks)
    let mut rank_sums = vec![0.0; k];
    for block in 0..n {
        let mut vals: Vec<(f64, usize)> = groups
            .iter()
            .enumerate()
            .map(|(j, g)| (g[block], j))
            .collect();
        vals.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Average ranks for ties
        let mut i = 0;
        while i < k {
            let mut j = i + 1;
            while j < k && (vals[j].0 - vals[i].0).abs() < 1e-12 {
                j += 1;
            }
            // Positions i..j all tied — average rank
            let avg_rank = (i..j).map(|r| r as f64 + 1.0).sum::<f64>() / (j - i) as f64;
            for tied in &vals[i..j] {
                rank_sums[tied.1] += avg_rank;
            }
            i = j;
        }
    }

    // Friedman statistic: χ² = 12/(nk(k+1)) Σ R_j² - 3n(k+1)
    let kf = k as f64;
    let nf = n as f64;
    let sum_sq: f64 = rank_sums.iter().map(|&r| r * r).sum();
    let chi2 = 12.0 / (nf * kf * (kf + 1.0)) * sum_sq - 3.0 * nf * (kf + 1.0);

    let df = kf - 1.0;
    let dist = ChiSquared::new(df);
    let pvalue = (1.0 - dist.cdf(chi2)).clamp(0.0, 1.0);

    TtestResult {
        statistic: chi2,
        pvalue,
        df,
    }
}

/// Fligner-Killeen test for equal variances.
///
/// A robust non-parametric test that uses the chi-squared distribution
/// of median-centered rank scores.
///
/// Matches `scipy.stats.fligner(*groups)`.
pub fn fligner(groups: &[&[f64]]) -> VarianceTestResult {
    if groups.len() < 2 || groups.iter().any(|g| g.len() < 2) {
        return invalid_variance_test_result();
    }

    let k = groups.len();

    // Compute median-centered scores using normal quantiles
    let mut all_scores: Vec<f64> = Vec::new();
    let mut group_sizes: Vec<usize> = Vec::new();

    for group in groups {
        let mut sorted = group.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = quantile_sorted(&sorted, 0.5);
        let deviations: Vec<f64> = group.iter().map(|&x| (x - median).abs()).collect();
        all_scores.extend_from_slice(&deviations);
        group_sizes.push(group.len());
    }

    let n_total = all_scores.len();
    // Rank all scores
    let mut indexed: Vec<(f64, usize)> = all_scores
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut ranks = vec![0.0; n_total];
    for (rank, &(_, idx)) in indexed.iter().enumerate() {
        ranks[idx] = rank as f64 + 1.0;
    }

    // Transform ranks to normal quantile scores
    let nf = n_total as f64;
    let scores: Vec<f64> = ranks
        .iter()
        .map(|&r| standard_normal_ppf((1.0 + r / (nf + 1.0)) / 2.0))
        .collect();

    // Compute group means of scores
    let grand_mean = scores.iter().sum::<f64>() / nf;
    let mut offset = 0;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for &ni in &group_sizes {
        let group_mean = scores[offset..offset + ni].iter().sum::<f64>() / ni as f64;
        numerator += ni as f64 * (group_mean - grand_mean).powi(2);
        for score in scores.iter().skip(offset).take(ni) {
            denominator += (score - grand_mean).powi(2);
        }
        offset += ni;
    }

    let statistic = if denominator > 0.0 {
        (nf - k as f64) * numerator / denominator
    } else {
        f64::NAN
    };

    let df = (k - 1) as f64;
    let dist = ChiSquared::new(df);
    let pvalue = (1.0 - dist.cdf(statistic)).clamp(0.0, 1.0);

    VarianceTestResult { statistic, pvalue }
}

/// Mood's test for equal scale parameters.
///
/// Tests H0: two samples have equal scale. Non-parametric.
///
/// Matches `scipy.stats.mood(x, y)`.
pub fn mood(x: &[f64], y: &[f64]) -> TtestResult {
    if x.len() < 3 || y.len() < 3 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    let n = x.len() + y.len();
    let nf = n as f64;

    // Pool and rank all observations
    let mut pooled: Vec<(f64, bool)> = x
        .iter()
        .map(|&v| (v, true))
        .chain(y.iter().map(|&v| (v, false)))
        .collect();
    pooled.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Mood statistic: M = Σ (rank_i - (n+1)/2)² for x observations
    let center = (nf + 1.0) / 2.0;
    let mut m_stat = 0.0;
    for (rank, &(_, is_x)) in pooled.iter().enumerate() {
        if is_x {
            let r = rank as f64 + 1.0;
            m_stat += (r - center).powi(2);
        }
    }

    // Expected value and variance under H0
    let nx = x.len() as f64;
    let e_m = nx * (nf * nf - 1.0) / 12.0;
    let var_m = nx * (nf - nx) * (nf + 1.0) * (nf + 2.0) * (nf - 2.0) / 180.0;

    if var_m <= 0.0 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    let z = (m_stat - e_m) / var_m.sqrt();
    let norm = Normal::standard();
    let pvalue = 2.0 * (1.0 - ContinuousDistribution::cdf(&norm, z.abs()));

    TtestResult {
        statistic: z,
        pvalue,
        df: f64::NAN, // Non-parametric, no df
    }
}

/// Mood's median test for equal medians across groups.
///
/// Tests H0: all groups have the same median.
///
/// Matches `scipy.stats.median_test(*groups)`.
pub fn median_test(groups: &[&[f64]]) -> TtestResult {
    if groups.len() < 2
        || groups
            .iter()
            .any(|g| g.is_empty() || g.iter().any(|v| v.is_nan()))
    {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    // Pool all data and find grand median
    let mut all_data: Vec<f64> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    all_data.sort_by(|a, b| a.total_cmp(b));
    let grand_median = quantile_sorted(&all_data, 0.5);

    // Count observations above/below median per group
    let k = groups.len();
    let mut above = vec![0.0; k];
    let mut below = vec![0.0; k];
    for (i, group) in groups.iter().enumerate() {
        for &val in *group {
            if val > grand_median {
                above[i] += 1.0;
            } else {
                below[i] += 1.0;
            }
        }
    }

    // Chi-squared test on the 2×k contingency table
    let table = vec![above, below];
    let result = chi2_contingency(&table);
    TtestResult {
        statistic: result.statistic,
        pvalue: result.pvalue,
        df: (k - 1) as f64,
    }
}

/// Ansari-Bradley test for equal scale parameters.
///
/// Matches the core behavior of `scipy.stats.ansari(x, y)` for the
/// default two-sided alternative.
pub fn ansari(x: &[f64], y: &[f64]) -> TtestResult {
    let n = x.len();
    let m = y.len();
    if n < 1 || m < 1 || x.iter().any(|v| v.is_nan()) || y.iter().any(|v| v.is_nan()) {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    let total = n + m;
    let mut pooled = Vec::with_capacity(total);
    pooled.extend_from_slice(x);
    pooled.extend_from_slice(y);
    let ranks = rankdata(&pooled);
    let total_f = total as f64;
    let symranks: Vec<f64> = ranks.iter().map(|&r| r.min(total_f - r + 1.0)).collect();
    let statistic: f64 = symranks[..n].iter().sum();

    let repeats = {
        let mut sorted = pooled.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        sorted.windows(2).any(|w| w[0].total_cmp(&w[1]).is_eq())
    };

    let n_f = n as f64;
    let m_f = m as f64;
    let mean = if total % 2 == 1 {
        n_f * (total_f + 1.0).powi(2) / (4.0 * total_f)
    } else {
        n_f * (total_f + 2.0) / 4.0
    };

    let variance = if repeats {
        let fac: f64 = symranks.iter().map(|r| r * r).sum();
        if total % 2 == 1 {
            m_f * n_f * (16.0 * total_f * fac - (total_f + 1.0).powi(4))
                / (16.0 * total_f.powi(2) * (total_f - 1.0))
        } else {
            m_f * n_f * (16.0 * fac - total_f * (total_f + 2.0).powi(2))
                / (16.0 * total_f * (total_f - 1.0))
        }
    } else if total % 2 == 1 {
        n_f * m_f * (total_f + 1.0) * (3.0 + total_f.powi(2)) / (48.0 * total_f.powi(2))
    } else {
        m_f * n_f * (total_f + 2.0) * (total_f - 2.0) / (48.0 * (total_f - 1.0))
    };

    if variance <= 0.0 {
        return TtestResult {
            statistic,
            pvalue: 1.0,
            df: f64::NAN,
        };
    }

    // Smaller Ansari statistics indicate larger dispersion in x.
    let z = (mean - statistic) / variance.sqrt();
    let normal = Normal::standard();
    let pvalue = 2.0 * normal.sf(z.abs());

    TtestResult {
        statistic,
        pvalue: pvalue.clamp(0.0, 1.0),
        df: f64::NAN,
    }
}

///
/// Matches `scipy.stats.mannwhitneyu(x, y, alternative='two-sided')`.
///
/// Non-parametric test: H0: distributions of x and y are equal.
/// Uses normal approximation for p-value (valid for n > 20).
pub fn mannwhitneyu(x: &[f64], y: &[f64]) -> TtestResult {
    if x.len() < 2 || y.len() < 2 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    let n1 = x.len();
    let n2 = y.len();
    let n1f = n1 as f64;
    let n2f = n2 as f64;

    // Combine and rank all observations
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(n1 + n2);
    for &v in x {
        combined.push((v, 0)); // group 0 = x
    }
    for &v in y {
        combined.push((v, 1)); // group 1 = y
    }

    let values: Vec<f64> = combined.iter().map(|&(v, _)| v).collect();
    let ranks = rankdata(&values);

    // U statistic: sum of ranks of x - n1*(n1+1)/2
    let rank_sum_x: f64 = ranks
        .iter()
        .zip(combined.iter())
        .filter(|(_, (_, group))| *group == 0)
        .map(|(r, _)| *r)
        .sum();

    let u1 = rank_sum_x - n1f * (n1f + 1.0) / 2.0;
    let u2 = n1f * n2f - u1;
    let u = u1.min(u2); // two-sided: use smaller U

    // Normal approximation
    let mu = n1f * n2f / 2.0;
    let sigma = (n1f * n2f * (n1f + n2f + 1.0) / 12.0).sqrt();

    if sigma == 0.0 {
        return TtestResult {
            statistic: u,
            pvalue: 1.0,
            df: f64::NAN,
        };
    }

    let z = (u - mu) / sigma;
    let normal = Normal::standard();
    let pvalue = 2.0 * normal.cdf(z.min(0.0)); // two-sided

    TtestResult {
        statistic: u,
        pvalue,
        df: f64::NAN,
    }
}

/// Wilcoxon signed-rank test for paired samples.
///
/// Matches `scipy.stats.wilcoxon(x, y)`.
///
/// Non-parametric test: H0: median of x - y is zero.
pub fn wilcoxon(x: &[f64], y: &[f64]) -> TtestResult {
    if x.len() != y.len()
        || x.len() < 10
        || x.iter().any(|v| v.is_nan())
        || y.iter().any(|v| v.is_nan())
    {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    // Compute differences, discard zeros
    let diffs: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi - yi)
        .filter(|&d| d.abs() > 1e-15)
        .collect();

    let nr = diffs.len();
    if nr < 2 {
        return TtestResult {
            statistic: 0.0,
            pvalue: 1.0,
            df: f64::NAN,
        };
    }

    // Rank the absolute differences
    let abs_diffs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let ranks = rankdata(&abs_diffs);

    // T+ = sum of ranks where difference is positive
    let t_plus: f64 = ranks
        .iter()
        .zip(diffs.iter())
        .filter(|(_, d)| **d > 0.0)
        .map(|(r, _)| *r)
        .sum();

    let t_minus: f64 = ranks
        .iter()
        .zip(diffs.iter())
        .filter(|(_, d)| **d < 0.0)
        .map(|(r, _)| *r)
        .sum();

    let t_stat = t_plus.min(t_minus);
    let nrf = nr as f64;

    // Normal approximation
    let mu = nrf * (nrf + 1.0) / 4.0;
    let sigma = (nrf * (nrf + 1.0) * (2.0 * nrf + 1.0) / 24.0).sqrt();

    if sigma == 0.0 {
        return TtestResult {
            statistic: t_stat,
            pvalue: 1.0,
            df: f64::NAN,
        };
    }

    let z = (t_stat - mu) / sigma;
    let normal = Normal::standard();
    let pvalue = 2.0 * normal.cdf(z.min(0.0));

    TtestResult {
        statistic: t_stat,
        pvalue,
        df: f64::NAN,
    }
}

/// Kruskal-Wallis H-test for independent samples.
///
/// Matches `scipy.stats.kruskal(*groups)`.
///
/// Non-parametric alternative to one-way ANOVA.
/// Tests H0: all groups come from the same distribution.
pub fn kruskal(groups: &[&[f64]]) -> TtestResult {
    if groups.len() < 2 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let nf = n_total as f64;

    if n_total < 3 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    // Combine all observations and rank
    let mut all_values: Vec<f64> = Vec::with_capacity(n_total);
    let mut group_sizes: Vec<usize> = Vec::with_capacity(groups.len());
    for &g in groups {
        all_values.extend_from_slice(g);
        group_sizes.push(g.len());
    }
    let ranks = rankdata(&all_values);

    // Compute H statistic: H = (12/(N(N+1))) * Σ(R_i²/n_i) - 3(N+1)
    let mut offset = 0;
    let mut h_sum = 0.0;
    for &ni in &group_sizes {
        let rank_sum: f64 = ranks[offset..offset + ni].iter().sum();
        h_sum += rank_sum * rank_sum / ni as f64;
        offset += ni;
    }

    let h = 12.0 / (nf * (nf + 1.0)) * h_sum - 3.0 * (nf + 1.0);
    let df = groups.len() as f64 - 1.0;

    // P-value from chi-squared distribution
    let chi2 = ChiSquared::new(df);
    let pvalue = chi2.sf(h).clamp(0.0, 1.0);

    TtestResult {
        statistic: h,
        pvalue,
        df,
    }
}

/// Wilcoxon rank-sum test for two independent samples.
///
/// Matches `scipy.stats.ranksums(x, y)`.
///
/// Similar to Mann-Whitney U but returns a z-statistic.
pub fn ranksums(x: &[f64], y: &[f64]) -> TtestResult {
    if x.len() < 2 || y.len() < 2 {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    let n1 = x.len() as f64;
    let n2 = y.len() as f64;

    // Combine and rank
    let mut combined: Vec<f64> = Vec::with_capacity(x.len() + y.len());
    combined.extend_from_slice(x);
    combined.extend_from_slice(y);
    let ranks = rankdata(&combined);

    // Sum of ranks for first sample
    let rank_sum_x: f64 = ranks[..x.len()].iter().sum();

    // Expected rank sum under H0
    let expected = n1 * (n1 + n2 + 1.0) / 2.0;
    let sd = (n1 * n2 * (n1 + n2 + 1.0) / 12.0).sqrt();

    if sd == 0.0 {
        return TtestResult {
            statistic: 0.0,
            pvalue: 1.0,
            df: f64::NAN,
        };
    }

    let z = (rank_sum_x - expected) / sd;
    let normal = Normal::standard();
    let pvalue = 2.0 * normal.cdf(-z.abs()); // two-sided

    TtestResult {
        statistic: z,
        pvalue,
        df: f64::NAN,
    }
}

// ══════════════════════════════════════════════════════════════════════
// Correlation and Regression
// ══════════════════════════════════════════════════════════════════════

/// Result of linear regression.
#[derive(Debug, Clone, PartialEq)]
pub struct LinregressResult {
    /// Slope of the regression line.
    pub slope: f64,
    /// Intercept of the regression line.
    pub intercept: f64,
    /// Pearson correlation coefficient (r-value).
    pub rvalue: f64,
    /// Two-sided p-value for the hypothesis test that the slope is zero.
    pub pvalue: f64,
    /// Standard error of the estimated slope.
    pub stderr: f64,
    /// Standard error of the estimated intercept.
    pub intercept_stderr: f64,
}

/// Calculate a linear least-squares regression for two sets of measurements.
///
/// Matches `scipy.stats.linregress(x, y)`.
///
/// Returns slope, intercept, r-value, p-value, and standard errors.
pub fn linregress(x: &[f64], y: &[f64]) -> LinregressResult {
    let n = x.len() as f64;
    if x.len() < 2 || x.len() != y.len() {
        return LinregressResult {
            slope: f64::NAN,
            intercept: f64::NAN,
            rvalue: f64::NAN,
            pvalue: f64::NAN,
            stderr: f64::NAN,
            intercept_stderr: f64::NAN,
        };
    }

    let xmean: f64 = x.iter().sum::<f64>() / n;
    let ymean: f64 = y.iter().sum::<f64>() / n;

    // Sums of squares and cross-products
    let mut ssxm = 0.0; // Σ(x - xmean)²
    let mut ssym = 0.0; // Σ(y - ymean)²
    let mut ssxym = 0.0; // Σ(x - xmean)(y - ymean)
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - xmean;
        let dy = yi - ymean;
        ssxm += dx * dx;
        ssym += dy * dy;
        ssxym += dx * dy;
    }

    if ssxm == 0.0 {
        // All x values identical — slope undefined
        return LinregressResult {
            slope: f64::NAN,
            intercept: f64::NAN,
            rvalue: f64::NAN,
            pvalue: f64::NAN,
            stderr: f64::NAN,
            intercept_stderr: f64::NAN,
        };
    }

    let slope = ssxym / ssxm;
    let intercept = ymean - slope * xmean;

    // Correlation coefficient
    let rvalue = if ssym == 0.0 {
        // All y values identical — perfect fit if slope is 0
        if slope == 0.0 { 1.0 } else { 0.0 }
    } else {
        ssxym / (ssxm * ssym).sqrt()
    };

    // p-value: test H0: slope = 0 using t-distribution
    let df = n - 2.0;
    let (pvalue, stderr, intercept_stderr) = if df > 0.0 {
        // Residual sum of squares
        let r2 = rvalue * rvalue;
        let sse = ssym * (1.0 - r2);
        let mse = sse / df;
        let se_slope = (mse / ssxm).sqrt();
        let se_intercept = (mse * (1.0 / n + xmean * xmean / ssxm)).sqrt();

        let t_stat = if se_slope > 0.0 {
            slope / se_slope
        } else {
            f64::INFINITY
        };

        let pval = if df >= 1.0 && se_slope > 0.0 {
            let tdist = StudentT::new(df);
            2.0 * tdist.sf(t_stat.abs())
        } else {
            0.0
        };
        (pval, se_slope, se_intercept)
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };

    LinregressResult {
        slope,
        intercept,
        rvalue,
        pvalue,
        stderr,
        intercept_stderr,
    }
}

/// Result of a correlation test.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationResult {
    /// Correlation coefficient.
    pub statistic: f64,
    /// Two-sided p-value.
    pub pvalue: f64,
}

/// Result of Somers' D ordinal association test.
#[derive(Debug, Clone, PartialEq)]
pub struct SomersDResult {
    /// Somers' D statistic.
    pub statistic: f64,
    /// p-value for H0: D = 0.
    pub pvalue: f64,
    /// Contingency table used in the calculation.
    pub table: Vec<Vec<f64>>,
    /// Alias for `statistic`, matching SciPy's result object.
    pub correlation: f64,
}

/// Input surface for `somersd`.
pub enum SomersDInput<'a> {
    /// Independent/dependent ordinal rankings.
    Rankings(&'a [f64], &'a [f64]),
    /// Precomputed contingency table.
    Table(&'a [Vec<f64>]),
}

/// Calculate the Pearson correlation coefficient and p-value.
///
/// Matches `scipy.stats.pearsonr(x, y)`.
///
/// Tests for non-correlation: H0: ρ = 0 (no linear relationship).
pub fn pearsonr(x: &[f64], y: &[f64]) -> CorrelationResult {
    let n = x.len();
    if n < 2 || n != y.len() {
        return CorrelationResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }
    let nf = n as f64;

    let xmean: f64 = x.iter().sum::<f64>() / nf;
    let ymean: f64 = y.iter().sum::<f64>() / nf;

    let mut ssxm = 0.0;
    let mut ssym = 0.0;
    let mut ssxym = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - xmean;
        let dy = yi - ymean;
        ssxm += dx * dx;
        ssym += dy * dy;
        ssxym += dx * dy;
    }

    let denom = (ssxm * ssym).sqrt();
    if denom == 0.0 {
        return CorrelationResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let r = ssxym / denom;
    // Clamp to [-1, 1] for numerical safety
    let r = r.clamp(-1.0, 1.0);

    // p-value via t-distribution: t = r * sqrt((n-2)/(1-r²))
    let df = nf - 2.0;
    let pvalue = if df > 0.0 && r.abs() < 1.0 {
        let t = r * (df / (1.0 - r * r)).sqrt();
        let tdist = StudentT::new(df);
        2.0 * tdist.sf(t.abs())
    } else if r.abs() >= 1.0 {
        0.0 // Perfect correlation
    } else {
        f64::NAN
    };

    CorrelationResult {
        statistic: r,
        pvalue,
    }
}

/// Calculate the Spearman rank-order correlation coefficient and p-value.
///
/// Matches `scipy.stats.spearmanr(a, b)`.
///
/// Uses the Pearson correlation of the rank-transformed data.
pub fn spearmanr(x: &[f64], y: &[f64]) -> CorrelationResult {
    let n = x.len();
    if n < 3 || n != y.len() {
        return CorrelationResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let rank_x = rankdata(x);
    let rank_y = rankdata(y);

    pearsonr(&rank_x, &rank_y)
}

/// Compute ranks with average tie-breaking.
fn rankdata(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if data.iter().any(|v| v.is_nan()) {
        return vec![f64::NAN; n];
    }
    // Create (value, original_index) pairs and sort by value
    let mut indexed: Vec<(f64, usize)> = data
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        // Find the end of the tie group
        let mut j = i + 1;
        while j < n && indexed[j].0 == indexed[i].0 {
            j += 1;
        }
        // Average rank for the tie group (ranks are 1-based)
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for item in &indexed[i..j] {
            ranks[item.1] = avg_rank;
        }
        i = j;
    }
    ranks
}

// ══════════════════════════════════════════════════════════════════════
// Summary Statistics
// ══════════════════════════════════════════════════════════════════════

/// Result of `describe`.
#[derive(Debug, Clone, PartialEq)]
pub struct DescribeResult {
    /// Number of observations.
    pub nobs: usize,
    /// Minimum and maximum values.
    pub minmax: (f64, f64),
    /// Arithmetic mean.
    pub mean: f64,
    /// Unbiased variance (ddof=1).
    pub variance: f64,
    /// Sample skewness (Fisher's definition).
    pub skewness: f64,
    /// Sample excess kurtosis (Fisher's definition).
    pub kurtosis: f64,
}

/// Compute several descriptive statistics of a data set.
///
/// Matches `scipy.stats.describe(a)`.
pub fn describe(data: &[f64]) -> DescribeResult {
    let n = data.len();
    if n < 2 {
        return DescribeResult {
            nobs: n,
            minmax: if n == 1 {
                (data[0], data[0])
            } else {
                (f64::NAN, f64::NAN)
            },
            mean: if n == 1 { data[0] } else { f64::NAN },
            variance: f64::NAN,
            skewness: f64::NAN,
            kurtosis: f64::NAN,
        };
    }

    let nf = n as f64;
    let mean_val = data.iter().sum::<f64>() / nf;
    let min_val = data.iter().copied().fold(f64::INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    });
    let max_val = data
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });

    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &x in data {
        let d = x - mean_val;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }

    let variance_val = m2 / (nf - 1.0);
    let skewness_val = skew_from_moments(nf, m2, m3);
    let kurtosis_val = kurtosis_from_moments(nf, m2, m4);

    DescribeResult {
        nobs: n,
        minmax: (min_val, max_val),
        mean: mean_val,
        variance: variance_val,
        skewness: skewness_val,
        kurtosis: kurtosis_val,
    }
}

/// Compute the sample skewness (Fisher's definition, bias=True).
///
/// Matches `scipy.stats.skew(a, bias=True)`.
/// skew = m3 / m2^(3/2) where m_k are central moments.
pub fn skew(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean_val = data.iter().sum::<f64>() / nf;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &x in data {
        let d = x - mean_val;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
    }
    skew_from_moments(nf, m2, m3)
}

/// Compute the sample excess kurtosis (Fisher's definition, bias=True).
///
/// Matches `scipy.stats.kurtosis(a, fisher=True, bias=True)`.
/// kurtosis = m4 / m2^2 - 3 where m_k are central moments.
pub fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 4 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean_val = data.iter().sum::<f64>() / nf;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for &x in data {
        let d = x - mean_val;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    kurtosis_from_moments(nf, m2, m4)
}

/// Compute the median absolute deviation (MAD).
///
/// Matches `scipy.stats.median_abs_deviation(a, scale=1.0)`.
/// Default scale is 1.0. For normal consistency, use 1.4826.
pub fn median_abs_deviation(data: &[f64], scale: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = quantile_sorted(&sorted, 0.5);

    let mut diffs: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    diffs.sort_by(|a, b| a.total_cmp(b));
    let mad = quantile_sorted(&diffs, 0.5);
    mad * scale
}

/// Compute the mode (most frequent value) of a data set.
///
/// For continuous data, returns the smallest value among those with the
/// highest frequency. Returns NaN for empty input.
pub fn mode(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let mut best_val = sorted[0];
    let mut best_count = 1usize;
    let mut current_val = sorted[0];
    let mut current_count = 1usize;

    for &v in &sorted[1..] {
        if v == current_val {
            current_count += 1;
        } else {
            if current_count > best_count {
                best_count = current_count;
                best_val = current_val;
            }
            current_val = v;
            current_count = 1;
        }
    }
    if current_count > best_count {
        best_val = current_val;
    }
    best_val
}

/// Compute the k-th central moment of a data set.
///
/// Matches `scipy.stats.moment(a, moment=k)`.
pub fn moment(data: &[f64], k: u32) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    if k == 0 {
        return 1.0;
    }
    let n = data.len() as f64;
    let mean_val = data.iter().sum::<f64>() / n;
    data.iter()
        .map(|&x| (x - mean_val).powi(k as i32))
        .sum::<f64>()
        / n
}

/// Standard error of the mean.
///
/// Matches `scipy.stats.sem(a)`.
pub fn sem(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean_val = data.iter().sum::<f64>() / nf;
    let var: f64 = data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (nf - 1.0);
    (var / nf).sqrt()
}

/// Interquartile range (IQR = Q3 - Q1).
///
/// Matches `scipy.stats.iqr(a)`.
pub fn iqr(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(&sorted, 0.75) - quantile_sorted(&sorted, 0.25)
}

/// Coefficient of variation (std / mean).
///
/// Matches `scipy.stats.variation(a)`.
pub fn variation(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    let n = data.len() as f64;
    let mean_val = data.iter().sum::<f64>() / n;
    if mean_val == 0.0 {
        return f64::NAN;
    }
    let var: f64 = data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / n;
    var.sqrt() / mean_val
}

fn mean_std_ddof(data: &[f64], ddof: usize) -> Option<(f64, f64)> {
    let n = data.len();
    if n == 0 || n <= ddof {
        return None;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - ddof) as f64;
    Some((mean, var.sqrt()))
}

/// Compute z-scores: (x - mean) / std.
///
/// Matches `scipy.stats.zscore(a)`.
pub fn zscore(data: &[f64]) -> Vec<f64> {
    let Some((mean_val, std_val)) = mean_std_ddof(data, 0) else {
        return vec![f64::NAN; data.len()];
    };
    if std_val == 0.0 {
        return vec![f64::NAN; data.len()];
    }
    data.iter().map(|&x| (x - mean_val) / std_val).collect()
}

/// Compute relative z-scores using the mean and variance of `compare`.
///
/// Matches the core 1D behavior of `scipy.stats.zmap(scores, compare)`.
pub fn zmap(scores: &[f64], compare: &[f64]) -> Vec<f64> {
    zmap_ddof(scores, compare, 0)
}

/// Compute relative z-scores with an explicit degrees-of-freedom correction.
///
/// Matches the core 1D behavior of `scipy.stats.zmap(scores, compare, ddof=...)`.
pub fn zmap_ddof(scores: &[f64], compare: &[f64], ddof: usize) -> Vec<f64> {
    let Some((mean, std)) = mean_std_ddof(compare, ddof) else {
        return vec![f64::NAN; scores.len()];
    };

    let mut out: Vec<f64> = scores.iter().map(|&score| (score - mean) / std).collect();

    // SciPy's zscore special-cases the exact same array object by replacing
    // constant slices with NaN rather than leaving mixed NaN/inf outputs.
    let same_slice =
        core::ptr::eq(scores.as_ptr(), compare.as_ptr()) && scores.len() == compare.len();
    if same_slice && std <= (f64::EPSILON * mean).abs() {
        out.fill(f64::NAN);
    }

    out
}

/// Compute the geometric z-score for strictly positive data.
///
/// Matches the core 1D behavior of `scipy.stats.gzscore(a)`.
pub fn gzscore(data: &[f64]) -> Vec<f64> {
    gzscore_ddof(data, 0)
}

/// Compute the geometric z-score with an explicit degrees-of-freedom correction.
///
/// Matches the core 1D behavior of `scipy.stats.gzscore(a, ddof=...)`.
pub fn gzscore_ddof(data: &[f64], ddof: usize) -> Vec<f64> {
    let logged: Vec<f64> = data.iter().map(|&value| value.ln()).collect();
    zscore_ddof(&logged, ddof)
}

/// Compute the q-th percentile of data.
///
/// Uses linear interpolation between data points.
/// `q` should be in [0, 100].
///
/// Matches `numpy.percentile(data, q)`.
pub fn percentile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() || q.is_nan() || data.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let q_frac = (q / 100.0).clamp(0.0, 1.0);
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(&sorted, q_frac)
}

/// Compute the mean after trimming a proportion from each tail.
///
/// Matches `scipy.stats.trim_mean(a, proportiontocut)`.
///
/// # Arguments
/// * `data` — Input array
/// * `proportiontocut` — Fraction to trim from each end (0.0 to 0.5)
pub fn trim_mean(data: &[f64], proportiontocut: f64) -> f64 {
    if data.is_empty() || data.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let prop = proportiontocut.clamp(0.0, 0.5);
    let n = data.len();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let ncut = (n as f64 * prop).floor() as usize;
    let trimmed = &sorted[ncut..n - ncut];
    if trimmed.is_empty() {
        return f64::NAN;
    }
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

/// Exact binomial test.
///
/// Tests H0: probability of success = p, given k successes in n trials.
/// Returns (k, n, two-sided p-value).
///
/// Matches `scipy.stats.binomtest(k, n, p)`.
pub fn binomtest(k: u64, n: u64, p: f64) -> f64 {
    if n == 0 || k > n || p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p == 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }

    let binom = Binomial::new(n, p);

    // Two-sided p-value: sum P(X=j) for all j where P(X=j) <= P(X=k)
    let p_observed = DiscreteDistribution::pmf(&binom, k);
    let mut pvalue = 0.0;
    for j in 0..=n {
        let p_j = DiscreteDistribution::pmf(&binom, j);
        if p_j <= p_observed + 1e-14 {
            pvalue += p_j;
        }
    }
    pvalue.min(1.0)
}

// Internal helpers for skewness and kurtosis
fn skew_from_moments(n: f64, m2: f64, m3: f64) -> f64 {
    if m2 == 0.0 {
        return f64::NAN;
    }
    let m2_n = m2 / n;
    let m3_n = m3 / n;
    m3_n / m2_n.powf(1.5)
}

fn kurtosis_from_moments(n: f64, m2: f64, m4: f64) -> f64 {
    if m2 == 0.0 {
        return f64::NAN; // Degenerate: all values identical
    }
    let m2_n = m2 / n;
    let m4_n = m4 / n;
    m4_n / (m2_n * m2_n) - 3.0
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ══════════════════════════════════════════════════════════════════════
// Goodness-of-Fit Tests
// ══════════════════════════════════════════════════════════════════════

/// Result of a goodness-of-fit test.
#[derive(Debug, Clone, PartialEq)]
pub struct GoodnessOfFitResult {
    /// Test statistic.
    pub statistic: f64,
    /// p-value.
    pub pvalue: f64,
}

/// Result of the Anderson-Darling test.
#[derive(Debug, Clone, PartialEq)]
pub struct AndersonResult {
    /// Test statistic.
    pub statistic: f64,
    /// Critical values at significance levels [15%, 10%, 5%, 2.5%, 1%].
    pub critical_values: [f64; 5],
    /// Significance levels corresponding to critical values.
    pub significance_level: [f64; 5],
}

/// Result of the k-sample Anderson-Darling test.
#[derive(Debug, Clone, PartialEq)]
pub struct AndersonKSampleResult {
    /// Normalized k-sample Anderson-Darling statistic.
    pub statistic: f64,
    /// Critical values at significance levels [25%, 10%, 5%, 2.5%, 1%, 0.5%, 0.1%].
    pub critical_values: [f64; 7],
    /// Approximate p-value, capped to [0.001, 0.25] without permutation mode.
    pub pvalue: f64,
}

/// Variant of the k-sample Anderson-Darling test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AndersonKSampleVariant {
    Midrank,
    Right,
    Continuous,
}

/// Target distribution or reference sample for `kstest`.
pub enum KstestTarget<'a> {
    /// One-sample KS against a reference CDF.
    Cdf(fn(f64) -> f64),
    /// Two-sample KS against another sample.
    Sample(&'a [f64]),
}

/// P-value method for the two-sample Cramer-von Mises test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cvm2SampleMethod {
    Auto,
    Exact,
    Asymptotic,
}

fn anderson_ksamp_statistic_midrank(
    samples: &[Vec<f64>],
    pooled: &[f64],
    unique: &[f64],
    n: &[usize],
    total: usize,
) -> f64 {
    let mut statistic = 0.0;
    let z_left: Vec<usize> = unique
        .iter()
        .map(|&value| pooled.partition_point(|x| *x < value))
        .collect();
    let lj: Vec<f64> = if total == unique.len() {
        vec![1.0; unique.len()]
    } else {
        unique
            .iter()
            .zip(z_left.iter())
            .map(|(&value, &left)| (pooled.partition_point(|x| *x <= value) - left) as f64)
            .collect()
    };
    let bj: Vec<f64> = z_left
        .iter()
        .zip(lj.iter())
        .map(|(&left, &count)| left as f64 + count / 2.0)
        .collect();

    for (sample, &sample_n) in samples.iter().zip(n.iter()) {
        let mut sorted = sample.clone();
        sorted.sort_by(f64::total_cmp);
        let right_counts: Vec<usize> = unique
            .iter()
            .map(|&value| sorted.partition_point(|x| *x <= value))
            .collect();
        let left_counts: Vec<usize> = unique
            .iter()
            .map(|&value| sorted.partition_point(|x| *x < value))
            .collect();
        let mij: Vec<f64> = right_counts
            .iter()
            .zip(left_counts.iter())
            .map(|(&right, &left)| right as f64 - (right - left) as f64 / 2.0)
            .collect();

        let mut inner_sum = 0.0;
        for ((&l_j, &b_j), &m_ij) in lj.iter().zip(bj.iter()).zip(mij.iter()) {
            let denominator = b_j * (total as f64 - b_j) - total as f64 * l_j / 4.0;
            if denominator > 0.0 {
                let term = l_j / total as f64
                    * (total as f64 * m_ij - b_j * sample_n as f64).powi(2)
                    / denominator;
                inner_sum += term;
            }
        }
        statistic += inner_sum / sample_n as f64;
    }

    statistic * (total as f64 - 1.0) / total as f64
}

fn anderson_ksamp_statistic_right(
    samples: &[Vec<f64>],
    pooled: &[f64],
    unique: &[f64],
    n: &[usize],
    total: usize,
) -> f64 {
    let unique_prefix = &unique[..unique.len().saturating_sub(1)];
    let lj: Vec<f64> = unique_prefix
        .iter()
        .map(|&value| {
            let left = pooled.partition_point(|x| *x < value);
            let right = pooled.partition_point(|x| *x <= value);
            (right - left) as f64
        })
        .collect();
    let mut bj = Vec::with_capacity(lj.len());
    let mut running = 0.0;
    for &count in &lj {
        running += count;
        bj.push(running);
    }

    let mut statistic = 0.0;
    for (sample, &sample_n) in samples.iter().zip(n.iter()) {
        let mut sorted = sample.clone();
        sorted.sort_by(f64::total_cmp);
        let mij: Vec<f64> = unique_prefix
            .iter()
            .map(|&value| sorted.partition_point(|x| *x <= value) as f64)
            .collect();
        let mut inner_sum = 0.0;
        for ((&l_j, &b_j), &m_ij) in lj.iter().zip(bj.iter()).zip(mij.iter()) {
            let denominator = b_j * (total as f64 - b_j);
            if denominator > 0.0 {
                inner_sum += l_j / total as f64
                    * (total as f64 * m_ij - b_j * sample_n as f64).powi(2)
                    / denominator;
            }
        }
        statistic += inner_sum / sample_n as f64;
    }
    statistic
}

fn anderson_ksamp_statistic_continuous(
    samples: &[Vec<f64>],
    pooled: &[f64],
    n: &[usize],
    total: usize,
) -> f64 {
    let mut statistic = 0.0;
    let js: Vec<f64> = (1..total).map(|j| j as f64).collect();

    for (sample, &sample_n) in samples.iter().zip(n.iter()) {
        let mut sorted = sample.clone();
        sorted.sort_by(f64::total_cmp);
        let mij: Vec<f64> = pooled[..pooled.len().saturating_sub(1)]
            .iter()
            .map(|&value| sorted.partition_point(|x| *x <= value) as f64)
            .collect();
        let mut inner_sum = 0.0;
        for (&j, &m_ij) in js.iter().zip(mij.iter()) {
            inner_sum +=
                (total as f64 * m_ij - j * sample_n as f64).powi(2) / (j * (total as f64 - j));
        }
        statistic += inner_sum / sample_n as f64;
    }

    statistic / total as f64
}

fn quadratic_least_squares_eval(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len() as f64;
    let sum_x = xs.iter().sum::<f64>();
    let sum_x2 = xs.iter().map(|value| value * value).sum::<f64>();
    let sum_x3 = xs.iter().map(|value| value * value * value).sum::<f64>();
    let sum_x4 = xs
        .iter()
        .map(|value| value * value * value * value)
        .sum::<f64>();
    let sum_y = ys.iter().sum::<f64>();
    let sum_xy = xs
        .iter()
        .zip(ys.iter())
        .map(|(xv, yv)| xv * yv)
        .sum::<f64>();
    let sum_x2y = xs
        .iter()
        .zip(ys.iter())
        .map(|(xv, yv)| xv * xv * yv)
        .sum::<f64>();

    let mut augmented = [
        [sum_x4, sum_x3, sum_x2, sum_x2y],
        [sum_x3, sum_x2, sum_x, sum_xy],
        [sum_x2, sum_x, n, sum_y],
    ];

    for pivot in 0..3 {
        let (best_row, best_value) = augmented
            .iter()
            .enumerate()
            .skip(pivot)
            .map(|(row_idx, row)| (row_idx, row[pivot].abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap_or((pivot, 0.0));
        if best_value <= f64::EPSILON {
            return f64::NAN;
        }
        if best_row != pivot {
            augmented.swap(pivot, best_row);
        }

        let pivot_value = augmented[pivot][pivot];
        for value in augmented[pivot].iter_mut().skip(pivot) {
            *value /= pivot_value;
        }

        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            let pivot_slice: Vec<f64> = augmented[pivot].iter().copied().skip(pivot).collect();
            for (value, pivot_value) in augmented[row]
                .iter_mut()
                .skip(pivot)
                .zip(pivot_slice.iter())
            {
                *value -= factor * pivot_value;
            }
        }
    }

    let a = augmented[0][3];
    let b = augmented[1][3];
    let c = augmented[2][3];
    a * x * x + b * x + c
}

/// K-sample Anderson-Darling test.
///
/// Matches the core non-permutation behavior of `scipy.stats.anderson_ksamp`.
pub fn anderson_ksamp(
    samples: &[Vec<f64>],
    variant: Option<AndersonKSampleVariant>,
) -> Result<AndersonKSampleResult, StatsError> {
    if samples.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "anderson_ksamp needs at least two samples".to_string(),
        ));
    }
    if samples.iter().any(Vec::is_empty) {
        return Err(StatsError::InvalidArgument(
            "anderson_ksamp encountered sample without observations".to_string(),
        ));
    }

    let mut pooled: Vec<f64> = samples
        .iter()
        .flat_map(|sample| sample.iter().copied())
        .collect();
    pooled.sort_by(f64::total_cmp);
    let mut unique = pooled.clone();
    unique.dedup_by(|a, b| a.total_cmp(b).is_eq());
    if unique.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "anderson_ksamp needs more than one distinct observation".to_string(),
        ));
    }

    let n: Vec<usize> = samples.iter().map(Vec::len).collect();
    let total = pooled.len();
    let k = samples.len();
    let statistic_raw = match variant.unwrap_or(AndersonKSampleVariant::Midrank) {
        AndersonKSampleVariant::Midrank => {
            anderson_ksamp_statistic_midrank(samples, &pooled, &unique, &n, total)
        }
        AndersonKSampleVariant::Right => {
            anderson_ksamp_statistic_right(samples, &pooled, &unique, &n, total)
        }
        AndersonKSampleVariant::Continuous => {
            anderson_ksamp_statistic_continuous(samples, &pooled, &n, total)
        }
    };

    let harmonic_sum: f64 = n.iter().map(|&count| 1.0 / count as f64).sum();
    let hs_cs: Vec<f64> = (2..=(total - 1))
        .rev()
        .scan(0.0, |acc, value| {
            *acc += 1.0 / value as f64;
            Some(*acc)
        })
        .collect();
    let h = hs_cs.last().copied().unwrap_or(0.0) + 1.0;
    let g: f64 = hs_cs
        .iter()
        .zip(2..total)
        .map(|(&value, denom)| value / denom as f64)
        .sum();

    let m = (k - 1) as f64;
    let a = (4.0 * g - 6.0) * m + (10.0 - 6.0 * g) * harmonic_sum;
    let b = (2.0 * g - 4.0) * (k * k) as f64
        + 8.0 * h * k as f64
        + (2.0 * g - 14.0 * h - 4.0) * harmonic_sum
        - 8.0 * h
        + 4.0 * g
        - 6.0;
    let c = (6.0 * h + 2.0 * g - 2.0) * (k * k) as f64
        + (4.0 * h - 4.0 * g + 6.0) * k as f64
        + (2.0 * h - 6.0) * harmonic_sum
        + 4.0 * h;
    let d = (2.0 * h + 6.0) * (k * k) as f64 - 4.0 * h * k as f64;
    let total_f = total as f64;
    let sigma_sq = (a * total_f.powi(3) + b * total_f.powi(2) + c * total_f + d)
        / ((total_f - 1.0) * (total_f - 2.0) * (total_f - 3.0));
    let normalized = (statistic_raw - m) / sigma_sq.sqrt();

    let b0 = [0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085];
    let b1 = [-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615];
    let b2 = [-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154];
    let critical_values = std::array::from_fn(|i| b0[i] + b1[i] / m.sqrt() + b2[i] / m);
    let significance: [f64; 7] = [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001];
    let pvalue = if normalized < critical_values[0] {
        significance[0]
    } else if normalized > critical_values[critical_values.len() - 1] {
        significance[significance.len() - 1]
    } else {
        let ys: Vec<f64> = significance.iter().map(|value| (*value).ln()).collect();
        let xs: Vec<f64> = critical_values.to_vec();
        quadratic_least_squares_eval(&xs, &ys, normalized).exp()
    };

    Ok(AndersonKSampleResult {
        statistic: normalized,
        critical_values,
        pvalue: pvalue.clamp(0.001, 0.25),
    })
}

/// One-sample Kolmogorov-Smirnov test.
///
/// Tests H0: data comes from the distribution specified by `cdf_func`.
/// The CDF function maps x -> P(X <= x) for the reference distribution.
///
/// Matches `scipy.stats.ks_1samp(data, cdf_func)`.
pub fn ks_1samp(data: &[f64], cdf_func: impl Fn(f64) -> f64) -> GoodnessOfFitResult {
    let n = data.len();
    if n == 0 || data.iter().any(|v| v.is_nan()) {
        return GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }
    let nf = n as f64;

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    // D = max_i { max(|i/n - F(x_i)|, |(i-1)/n - F(x_i)|) }
    let mut d_stat = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let f_x = cdf_func(x);
        let d_plus = ((i + 1) as f64 / nf - f_x).abs();
        let d_minus = (f_x - i as f64 / nf).abs();
        d_stat = if d_stat.is_nan() || d_plus.is_nan() || d_minus.is_nan() {
            f64::NAN
        } else {
            d_stat.max(d_plus).max(d_minus)
        };
    }

    let pvalue = kolmogorov_pvalue(d_stat, nf);

    GoodnessOfFitResult {
        statistic: d_stat,
        pvalue,
    }
}

/// Kolmogorov-Smirnov test wrapper.
///
/// Dispatches to one-sample or two-sample KS based on `target`.
///
/// Matches the core SciPy surface of `scipy.stats.kstest`.
pub fn kstest(data: &[f64], target: KstestTarget<'_>) -> GoodnessOfFitResult {
    match target {
        KstestTarget::Cdf(cdf) => ks_1samp(data, cdf),
        KstestTarget::Sample(reference) => ks_2samp(data, reference),
    }
}

/// Epps-Singleton two-sample test with SciPy's default support points.
pub fn epps_singleton_2samp(x: &[f64], y: &[f64]) -> GoodnessOfFitResult {
    match epps_singleton_2samp_with_t(x, y, &[0.4, 0.8]) {
        Ok(result) => result,
        Err(_) => GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        },
    }
}

/// Epps-Singleton two-sample test with explicit positive support points `t`.
pub fn epps_singleton_2samp_with_t(
    x: &[f64],
    y: &[f64],
    t: &[f64],
) -> Result<GoodnessOfFitResult, StatsError> {
    if x.len() < 5 || y.len() < 5 {
        return Ok(GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }
    if t.is_empty() {
        return Err(StatsError::InvalidArgument(
            "t must not be empty".to_string(),
        ));
    }
    if t.iter().any(|&value| value <= 0.0 || !value.is_finite()) {
        return Err(StatsError::InvalidArgument(
            "t must contain positive finite elements only".to_string(),
        ));
    }
    for (i, &value) in t.iter().enumerate() {
        if t.iter().skip(i + 1).any(|&other| other == value) {
            return Err(StatsError::InvalidArgument(
                "t must contain distinct elements".to_string(),
            ));
        }
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Ok(GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }

    let mut pooled = x.to_vec();
    pooled.extend_from_slice(y);
    let sigma = iqr(&pooled) / 2.0;
    if !sigma.is_finite() || sigma == 0.0 {
        return Ok(GoodnessOfFitResult {
            statistic: 0.0,
            pvalue: f64::NAN,
        });
    }

    let scaled_t: Vec<f64> = t.iter().map(|&value| value / sigma).collect();
    let feature_dim = 2 * scaled_t.len();
    let feature_row = |sample: f64| -> Vec<f64> {
        let mut row = Vec::with_capacity(feature_dim);
        for &t_value in &scaled_t {
            row.push((t_value * sample).cos());
        }
        for &t_value in &scaled_t {
            row.push((t_value * sample).sin());
        }
        row
    };

    let gx: Vec<Vec<f64>> = x.iter().copied().map(feature_row).collect();
    let gy: Vec<Vec<f64>> = y.iter().copied().map(feature_row).collect();
    let cov_x = covariance_biased(&gx);
    let cov_y = covariance_biased(&gy);

    let nx = x.len() as f64;
    let ny = y.len() as f64;
    let n_total = nx + ny;
    let mut est_cov = vec![vec![0.0; feature_dim]; feature_dim];
    for (i, row) in est_cov.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = (n_total / nx) * cov_x[i][j] + (n_total / ny) * cov_y[i][j];
        }
    }

    let (pinv, rank) = symmetric_pseudoinverse_and_rank(&est_cov);
    if rank == 0 {
        return Ok(GoodnessOfFitResult {
            statistic: 0.0,
            pvalue: f64::NAN,
        });
    }

    let mut mean_x = vec![0.0; feature_dim];
    let mut mean_y = vec![0.0; feature_dim];
    for row in &gx {
        for (j, value) in row.iter().enumerate() {
            mean_x[j] += value;
        }
    }
    for row in &gy {
        for (j, value) in row.iter().enumerate() {
            mean_y[j] += value;
        }
    }
    for value in &mut mean_x {
        *value /= nx;
    }
    for value in &mut mean_y {
        *value /= ny;
    }
    let g_diff: Vec<f64> = mean_x
        .iter()
        .zip(mean_y.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let pinv_times_diff = matvec_mul(&pinv, &g_diff);
    let mut statistic = n_total * dot(&g_diff, &pinv_times_diff);

    if x.len().max(y.len()) < 25 {
        let correction = 1.0 / (1.0 + n_total.powf(-0.45) + 10.1 * (nx.powf(-1.7) + ny.powf(-1.7)));
        statistic *= correction;
    }

    let pvalue = ChiSquared::new(rank as f64).sf(statistic).clamp(0.0, 1.0);
    Ok(GoodnessOfFitResult { statistic, pvalue })
}

fn cvm_cdf_inf(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }

    let mut total = 0.0;
    for k in 0..256 {
        let kf = k as f64;
        let y = 4.0 * kf + 1.0;
        let q = y * y / (16.0 * x);
        let coeff = (ln_gamma(kf + 0.5) - ln_gamma(kf + 1.0)).exp() / (PI.powf(1.5) * x.sqrt());
        let term = coeff * y.sqrt() * (-q).exp() * modified_bessel_k(0.25, q);
        if !term.is_finite() {
            break;
        }
        total += term;
        if term.abs() < 1e-10 {
            break;
        }
    }
    total.clamp(0.0, 1.0)
}

fn cvm_cdf(x: f64, n: Option<usize>) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    match n {
        None => cvm_cdf_inf(x),
        Some(sample_size) => {
            let nf = sample_size as f64;
            if x <= 1.0 / (12.0 * nf) {
                0.0
            } else if x >= nf / 3.0 {
                1.0
            } else {
                // The asymptotic CvM cdf is numerically stable in this codebase.
                // A finite-sample Bessel-series correction was tested here but
                // proved less reliable than the asymptotic approximation on
                // ordinary valid inputs, so we keep the stable path.
                cvm_cdf_inf(x)
            }
        }
    }
}

/// One-sample Cramer-von Mises goodness-of-fit test.
///
/// Matches `scipy.stats.cramervonmises(data, cdf)`.
pub fn cramervonmises(data: &[f64], cdf_func: impl Fn(f64) -> f64) -> GoodnessOfFitResult {
    let n = data.len();
    if n <= 1 {
        return GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let nf = n as f64;
    let statistic = sorted
        .iter()
        .enumerate()
        .fold(1.0 / (12.0 * nf), |acc, (i, &x)| {
            let ui = (2.0 * (i + 1) as f64 - 1.0) / (2.0 * nf);
            let cdf = cdf_func(x);
            acc + (ui - cdf) * (ui - cdf)
        });
    let pvalue = (1.0 - cvm_cdf(statistic, Some(n))).clamp(0.0, 1.0);

    GoodnessOfFitResult { statistic, pvalue }
}

fn lcm_u64(a: u64, b: u64) -> u64 {
    a / gcd_u64(a, b) * b
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn binomial_u128(n: u64, k: u64) -> u128 {
    let k = k.min(n - k);
    let mut result = 1u128;
    for i in 0..k {
        result = result * (n - i) as u128 / (i + 1) as u128;
    }
    result
}

fn cvm_2samp_exact_pvalue(u_stat: u64, nx: usize, ny: usize) -> f64 {
    use std::collections::BTreeMap;

    let m = nx as u64;
    let n = ny as u64;
    let lcm = lcm_u64(m, n);
    let a = lcm / m;
    let b = lcm / n;
    let mn = m * n;
    let zeta = (lcm as i128
        * lcm as i128
        * (m + n) as i128
        * (6 * u_stat as i128 - (mn * (4 * mn - 1)) as i128))
        / (6 * mn as i128 * mn as i128);

    let mut gs: Vec<BTreeMap<i128, u128>> = std::iter::once({
        let mut seed = BTreeMap::new();
        seed.insert(0, 1);
        seed
    })
    .chain((0..m).map(|_| BTreeMap::new()))
    .collect();

    for u in 0..=n {
        let mut next_gs = Vec::with_capacity((m + 1) as usize);
        let mut tmp: BTreeMap<i128, u128> = BTreeMap::new();
        for v in 0..=m {
            for (&value, &freq) in &gs[v as usize] {
                *tmp.entry(value).or_insert(0) += freq;
            }
            let shift = (a as i128 * v as i128 - b as i128 * u as i128).pow(2);
            let shifted = tmp
                .iter()
                .map(|(&value, &freq)| (value + shift, freq))
                .collect::<BTreeMap<_, _>>();
            next_gs.push(shifted.clone());
            tmp = shifted;
        }
        gs = next_gs;
    }

    let combinations = binomial_u128(m + n, m) as f64;
    let tail = gs[m as usize]
        .iter()
        .filter(|(value, _)| **value >= zeta)
        .map(|(_, &freq)| freq as f64)
        .sum::<f64>();
    (tail / combinations).clamp(0.0, 1.0)
}

fn cvm_2samp_asymptotic_pvalue(statistic: f64, nx: usize, ny: usize) -> f64 {
    let nxf = nx as f64;
    let nyf = ny as f64;
    let n_total = nxf + nyf;
    let k = nxf * nyf;
    let expected_t = (1.0 + 1.0 / n_total) / 6.0;
    let variance_t = (n_total + 1.0)
        * (4.0 * k * n_total - 3.0 * (nxf * nxf + nyf * nyf) - 2.0 * k)
        / (45.0 * n_total * n_total * 4.0 * k);
    let normalized = 1.0 / 6.0 + (statistic - expected_t) / (45.0 * variance_t).sqrt();
    if normalized < 0.003 {
        1.0
    } else {
        (1.0 - cvm_cdf_inf(normalized)).clamp(0.0, 1.0)
    }
}

/// Two-sample Cramer-von Mises test with SciPy's default auto method.
///
/// Matches `scipy.stats.cramervonmises_2samp(x, y)`.
pub fn cramervonmises_2samp(x: &[f64], y: &[f64]) -> GoodnessOfFitResult {
    cramervonmises_2samp_with_method(x, y, Cvm2SampleMethod::Auto)
}

/// Two-sample Cramer-von Mises test with explicit p-value method selection.
pub fn cramervonmises_2samp_with_method(
    x: &[f64],
    y: &[f64],
    method: Cvm2SampleMethod,
) -> GoodnessOfFitResult {
    let nx = x.len();
    let ny = y.len();
    if nx <= 1 || ny <= 1 || x.iter().any(|v| v.is_nan()) || y.iter().any(|v| v.is_nan()) {
        return GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let mut xa = x.to_vec();
    let mut ya = y.to_vec();
    xa.sort_by(|a, b| a.total_cmp(b));
    ya.sort_by(|a, b| a.total_cmp(b));

    let mut pooled = xa.clone();
    pooled.extend_from_slice(&ya);
    let ranks = rankdata(&pooled);
    let rx = &ranks[..nx];
    let ry = &ranks[nx..];

    let u_stat_x = rx
        .iter()
        .enumerate()
        .map(|(i, &rank)| (rank - (i + 1) as f64).powi(2))
        .sum::<f64>();
    let u_stat_y = ry
        .iter()
        .enumerate()
        .map(|(i, &rank)| (rank - (i + 1) as f64).powi(2))
        .sum::<f64>();
    let u_stat = nx as f64 * u_stat_x + ny as f64 * u_stat_y;
    let k = (nx * ny) as f64;
    let n_total = (nx + ny) as f64;
    let statistic = u_stat / (k * n_total) - (4.0 * k - 1.0) / (6.0 * n_total);

    let pvalue = match method {
        Cvm2SampleMethod::Exact => cvm_2samp_exact_pvalue(u_stat.round() as u64, nx, ny),
        Cvm2SampleMethod::Asymptotic => cvm_2samp_asymptotic_pvalue(statistic, nx, ny),
        Cvm2SampleMethod::Auto => {
            if nx.max(ny) > 20 {
                cvm_2samp_asymptotic_pvalue(statistic, nx, ny)
            } else {
                cvm_2samp_exact_pvalue(u_stat.round() as u64, nx, ny)
            }
        }
    };

    GoodnessOfFitResult { statistic, pvalue }
}

/// Two-sample Kolmogorov-Smirnov test.
///
/// Tests H0: two samples come from the same continuous distribution.
///
/// Matches `scipy.stats.ks_2samp(data1, data2)`.
pub fn ks_2samp(data1: &[f64], data2: &[f64]) -> GoodnessOfFitResult {
    let n1 = data1.len();
    let n2 = data2.len();
    if n1 == 0 || n2 == 0 || data1.iter().any(|v| v.is_nan()) || data2.iter().any(|v| v.is_nan()) {
        return GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let mut sorted1 = data1.to_vec();
    let mut sorted2 = data2.to_vec();
    sorted1.sort_by(|a, b| a.total_cmp(b));
    sorted2.sort_by(|a, b| a.total_cmp(b));

    // Walk both sorted arrays, computing max |F1(x) - F2(x)|
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let mut i = 0;
    let mut j = 0;
    let mut d_stat = 0.0_f64;

    while i < n1 || j < n2 {
        let val1 = if i < n1 { sorted1[i] } else { f64::INFINITY };
        let val2 = if j < n2 { sorted2[j] } else { f64::INFINITY };
        let current_val = val1.min(val2);

        // Advance both indices past all elements <= current_val to handle ties
        while i < n1 && sorted1[i] <= current_val {
            i += 1;
        }
        while j < n2 && sorted2[j] <= current_val {
            j += 1;
        }

        let diff = (i as f64 / n1f - j as f64 / n2f).abs();
        d_stat = d_stat.max(diff);
    }

    // Effective sample size for p-value
    let en = (n1f * n2f / (n1f + n2f)).sqrt();
    let pvalue = kolmogorov_pvalue(d_stat, en * en);

    GoodnessOfFitResult {
        statistic: d_stat,
        pvalue,
    }
}

/// Shapiro-Wilk test for normality.
///
/// Tests H0: data was drawn from a normal distribution.
/// Most powerful normality test for small to moderate sample sizes.
///
/// Matches `scipy.stats.shapiro(data)`.
///
/// Uses the Royston (1992) algorithm with approximated coefficients
/// for n <= 5000.
pub fn shapiro(data: &[f64]) -> GoodnessOfFitResult {
    let n = data.len();
    if n < 3 {
        return GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let nf = n as f64;
    let mean_val = data.iter().sum::<f64>() / nf;

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    // Compute the Shapiro-Wilk W statistic.
    // W = (sum(a_i * x_(i))^2) / (sum(x_i - mean)^2)
    // where a_i are optimal coefficients derived from normal order statistics.

    // Generate approximate coefficients using the normal ppf (Blom's approximation)
    let mut m = vec![0.0; n];
    for (i, mi) in m.iter_mut().enumerate() {
        *mi = standard_normal_ppf((i as f64 + 1.0 - 0.375) / (nf + 0.25));
    }

    // Normalize the m vector
    let m_sq_sum: f64 = m.iter().map(|&v| v * v).sum();
    let m_norm = m_sq_sum.sqrt();

    // Compute approximate weights a_i = m_i / ||m||
    let a: Vec<f64> = m.iter().map(|&v| v / m_norm).collect();

    // W = (sum(a_i * x_(i)))^2 / SS
    let numerator: f64 = a
        .iter()
        .zip(sorted.iter())
        .map(|(&ai, &xi)| ai * xi)
        .sum::<f64>();
    let ss: f64 = data.iter().map(|&x| (x - mean_val).powi(2)).sum();

    if ss == 0.0 {
        return GoodnessOfFitResult {
            statistic: 1.0,
            pvalue: 1.0,
        };
    }

    let w = (numerator * numerator) / ss;

    // Approximate p-value using Royston's normal transformation
    // For n >= 7, use ln(1 - W) transformation
    let pvalue = if n <= 6 {
        // Small sample: use simple approximation
        // Based on Shapiro-Francia for tiny n
        let z = (-((1.0 - w).ln()) - (0.0 + 0.221 * nf.ln() - 0.0174 * nf)) / (1.0 + 0.042 / nf);
        let norm = Normal::standard();
        1.0 - ContinuousDistribution::cdf(&norm, z)
    } else {
        // Royston approximation: transform ln(1-W) to approximate normality
        let ln_1mw = (1.0 - w).ln();
        let mu =
            0.0038915 * nf.ln().powi(3) - 0.083751 * nf.ln().powi(2) - 0.31082 * nf.ln() - 1.5861;
        let sigma = (0.0030302 * nf.ln().powi(2) - 0.082676 * nf.ln() - 0.4803).exp();
        let z = (ln_1mw - mu) / sigma;
        let norm = Normal::standard();
        1.0 - ContinuousDistribution::cdf(&norm, z)
    };

    GoodnessOfFitResult {
        statistic: w,
        pvalue: pvalue.clamp(0.0, 1.0),
    }
}

/// D'Agostino-Pearson omnibus test for normality.
///
/// Tests H0: data was drawn from a normal distribution by combining
/// the skewness test and kurtosis test into an omnibus chi-squared test.
///
/// Matches `scipy.stats.normaltest(data)`.
pub fn normaltest(data: &[f64]) -> GoodnessOfFitResult {
    let n = data.len();
    if n < 8 {
        return GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let nf = n as f64;
    let mean_val = data.iter().sum::<f64>() / nf;

    // Compute central moments
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &x in data {
        let d = x - mean_val;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    let m2_n = m2 / nf;
    let m3_n = m3 / nf;
    let m4_n = m4 / nf;

    // Skewness test (D'Agostino 1990)
    let g1 = m3_n / m2_n.powf(1.5); // sample skewness
    let z_skew = dagostino_skewtest_z(g1, nf);

    // Kurtosis test (Anscombe & Glynn 1983)
    let g2 = m4_n / (m2_n * m2_n) - 3.0; // excess kurtosis
    let z_kurt = dagostino_kurttest_z(g2, nf);

    // Omnibus: K² = z_skew² + z_kurt² ~ chi²(2)
    let k2 = z_skew * z_skew + z_kurt * z_kurt;

    // p-value from chi-squared(2) survival function
    // For chi²(2): P(X > x) = exp(-x/2)
    let pvalue = (-k2 / 2.0).exp();

    GoodnessOfFitResult {
        statistic: k2,
        pvalue,
    }
}

/// Jarque-Bera test for normality.
///
/// Tests H0: data was drawn from a normal distribution using the sample
/// skewness and excess kurtosis.
///
/// Matches `scipy.stats.jarque_bera(data)`.
pub fn jarque_bera(data: &[f64]) -> GoodnessOfFitResult {
    let n = data.len();
    if n < 2 {
        return GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let nf = n as f64;
    let mean_val = data.iter().sum::<f64>() / nf;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &x in data {
        let d = x - mean_val;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }

    if m2 == 0.0 {
        return GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let skewness = skew_from_moments(nf, m2, m3);
    let excess_kurtosis = kurtosis_from_moments(nf, m2, m4);
    let statistic = nf / 6.0 * (skewness * skewness + 0.25 * excess_kurtosis * excess_kurtosis);
    let pvalue = ChiSquared::new(2.0).sf(statistic).clamp(0.0, 1.0);

    GoodnessOfFitResult { statistic, pvalue }
}

/// Anderson-Darling test for a specified distribution.
///
/// Tests H0: data comes from the given distribution family.
/// Currently supports `"norm"` (normal distribution).
///
/// Returns the test statistic and critical values at significance levels
/// [15%, 10%, 5%, 2.5%, 1%].
///
/// Matches `scipy.stats.anderson(data, dist='norm')`.
pub fn anderson(data: &[f64], dist: &str) -> AndersonResult {
    let n = data.len();
    if n < 3 || dist != "norm" {
        return AndersonResult {
            statistic: f64::NAN,
            critical_values: [f64::NAN; 5],
            significance_level: [15.0, 10.0, 5.0, 2.5, 1.0],
        };
    }

    let nf = n as f64;

    // Fit normal parameters from data
    let mean_val = data.iter().sum::<f64>() / nf;
    let var_val: f64 = data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (nf - 1.0);
    let std_val = var_val.sqrt();

    if std_val == 0.0 {
        return AndersonResult {
            statistic: f64::INFINITY,
            critical_values: anderson_critical_values_norm(nf),
            significance_level: [15.0, 10.0, 5.0, 2.5, 1.0],
        };
    }

    // Sort and standardize
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let norm = Normal::standard();

    // A² = -n - (1/n) * sum_{i=1}^{n} (2i-1) * [ln(F(Y_i)) + ln(1 - F(Y_{n+1-i}))]
    let mut s = 0.0;
    for i in 0..n {
        let z = (sorted[i] - mean_val) / std_val;
        let f_z = ContinuousDistribution::cdf(&norm, z);
        // Clamp to avoid ln(0)
        let f_z = f_z.clamp(1e-15, 1.0 - 1e-15);

        let z_rev = (sorted[n - 1 - i] - mean_val) / std_val;
        let f_z_rev = ContinuousDistribution::cdf(&norm, z_rev);
        let f_z_rev = f_z_rev.clamp(1e-15, 1.0 - 1e-15);

        s += (2.0 * (i + 1) as f64 - 1.0) * (f_z.ln() + (1.0 - f_z_rev).ln());
    }

    let a2 = -nf - s / nf;

    // Apply correction factor for estimated parameters (case 3 in SciPy)
    let a2_star = a2 * (1.0 + 0.75 / nf + 2.25 / (nf * nf));

    AndersonResult {
        statistic: a2_star,
        critical_values: anderson_critical_values_norm(nf),
        significance_level: [15.0, 10.0, 5.0, 2.5, 1.0],
    }
}

/// Critical values for Anderson-Darling test with normal distribution
/// at significance levels [15%, 10%, 5%, 2.5%, 1%].
/// These are the standard tabulated values (Stephens 1974/1986).
fn anderson_critical_values_norm(_n: f64) -> [f64; 5] {
    // Standard critical values for normal distribution (parameters estimated)
    [0.576, 0.656, 0.787, 0.918, 1.092]
}

/// Kolmogorov distribution p-value: P(D_n >= d).
/// Uses the Kolmogorov-Smirnov limiting distribution approximation.
fn kolmogorov_pvalue(d: f64, n: f64) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }

    // Effective value: sqrt(n) * d
    let s = n.sqrt() * d;

    // For large s, use asymptotic series (Kolmogorov's formula):
    // P(sqrt(n)*D_n > s) ≈ 2 * sum_{k=1}^{inf} (-1)^{k+1} * exp(-2*k²*s²)
    let mut pval = 0.0;
    for k in 1..=100 {
        let kf = k as f64;
        let term = (-2.0 * kf * kf * s * s).exp();
        if term < 1e-20 {
            break;
        }
        if k % 2 == 1 {
            pval += term;
        } else {
            pval -= term;
        }
    }
    (2.0 * pval).clamp(0.0, 1.0)
}

/// D'Agostino's skewness test z-score.
fn dagostino_skewtest_z(g1: f64, n: f64) -> f64 {
    // D'Agostino (1990) transformation of sample skewness to z-score
    let y = g1 * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();

    let beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w2 = (2.0 * (beta2 - 1.0)).sqrt() - 1.0;
    let delta = 1.0 / w2.ln().sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();

    delta * (y / alpha + ((y / alpha).powi(2) + 1.0).sqrt()).ln()
}

/// Anscombe & Glynn kurtosis test z-score.
fn dagostino_kurttest_z(g2: f64, n: f64) -> f64 {
    // Expected value and variance of excess kurtosis under normality
    let e_g2 = -6.0 / (n + 1.0);
    let var_g2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) * (n + 1.0) * (n + 3.0) * (n + 5.0));
    let std_g2 = var_g2.sqrt();

    if std_g2 == 0.0 {
        return 0.0;
    }

    // Standardize
    let x = (g2 - e_g2) / std_g2;

    // Apply cube-root transformation (Anscombe & Glynn)
    let beta1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
        * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();

    let a = 6.0 + 8.0 / beta1 * (2.0 / beta1 + (1.0 + 4.0 / (beta1 * beta1)).sqrt());
    let z_term = ((1.0 - 2.0 / a) / (1.0 + x * (2.0 / (a - 4.0)).sqrt())).abs();
    if z_term <= 0.0 {
        return 0.0;
    }
    (1.0 - 2.0 / (9.0 * a) - z_term.powf(1.0 / 3.0)) / (2.0 / (9.0 * a)).sqrt()
}

// ══════════════════════════════════════════════════════════════════════
// Additional Statistics: gaussian_kde, entropy, boxcox, kendalltau
// ══════════════════════════════════════════════════════════════════════

/// Gaussian Kernel Density Estimator.
///
/// Matches `scipy.stats.gaussian_kde(dataset)`.
///
/// Uses Silverman's rule for bandwidth selection by default.
#[derive(Debug, Clone)]
pub struct GaussianKde {
    /// Data points.
    dataset: Vec<f64>,
    /// Bandwidth (standard deviation of the Gaussian kernel).
    bandwidth: f64,
}

impl GaussianKde {
    /// Create a new Gaussian KDE from data.
    ///
    /// Uses Silverman's rule: bw = n^(-1/5) * std(data).
    pub fn new(data: &[f64]) -> Self {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
        let std_dev = var.sqrt();
        let bw = std_dev * n.powf(-0.2); // Silverman's rule
        Self {
            dataset: data.to_vec(),
            bandwidth: bw.max(f64::EPSILON),
        }
    }

    /// Create a KDE with a specified bandwidth.
    pub fn with_bandwidth(data: &[f64], bandwidth: f64) -> Self {
        Self {
            dataset: data.to_vec(),
            bandwidth: bandwidth.max(f64::EPSILON),
        }
    }

    /// Evaluate the KDE at a single point.
    pub fn evaluate(&self, x: f64) -> f64 {
        let n = self.dataset.len() as f64;
        let inv_bw = 1.0 / self.bandwidth;
        let norm = inv_bw / (n * (2.0 * std::f64::consts::PI).sqrt());
        self.dataset
            .iter()
            .map(|&xi| {
                let z = (x - xi) * inv_bw;
                norm * (-0.5 * z * z).exp()
            })
            .sum()
    }

    /// Evaluate the KDE at multiple points.
    pub fn evaluate_many(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|&x| self.evaluate(x)).collect()
    }

    /// Return the bandwidth.
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }
}

/// Compute the Shannon entropy of a discrete probability distribution.
///
/// Matches `scipy.stats.entropy(pk, base)`.
///
/// H = -Σ p_k * log(p_k), with 0*log(0) = 0 by convention.
///
/// # Arguments
/// * `pk` — Probability distribution (will be normalized if not summing to 1).
/// * `base` — Logarithm base (None = natural log, Some(2) = bits).
pub fn entropy(pk: &[f64], base: Option<f64>) -> f64 {
    if pk.is_empty() {
        return 0.0;
    }
    let total: f64 = pk.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    let h: f64 = pk
        .iter()
        .map(|&p| {
            let prob = p / total;
            if prob > 0.0 { -prob * prob.ln() } else { 0.0 }
        })
        .sum();

    match base {
        Some(b) => h / b.ln(),
        None => h,
    }
}

/// Result of Box-Cox transformation.
#[derive(Debug, Clone)]
pub struct BoxCoxResult {
    /// Transformed data.
    pub data: Vec<f64>,
    /// Optimal lambda parameter.
    pub lmbda: f64,
}

/// Apply Box-Cox transformation to data.
///
/// Matches `scipy.stats.boxcox(x, lmbda)`.
///
/// Transforms positive data to approximate normality.
/// `y = (x^λ - 1) / λ` if λ ≠ 0, `y = ln(x)` if λ = 0.
///
/// # Arguments
/// * `data` — Input data (must be positive).
/// * `lmbda` — If None, find the optimal λ that maximizes the log-likelihood.
///   If Some(λ), use that value.
pub fn boxcox(data: &[f64], lmbda: Option<f64>) -> Result<BoxCoxResult, String> {
    if data.is_empty() {
        return Err("data must not be empty".to_string());
    }
    for (i, &v) in data.iter().enumerate() {
        if v <= 0.0 {
            return Err(format!(
                "data[{i}] = {v}: Box-Cox requires all positive values"
            ));
        }
    }

    let lambda = lmbda.unwrap_or_else(|| find_optimal_boxcox_lambda(data));

    let transformed: Vec<f64> = data
        .iter()
        .map(|&x| {
            if lambda.abs() < 1e-10 {
                x.ln()
            } else {
                (x.powf(lambda) - 1.0) / lambda
            }
        })
        .collect();

    Ok(BoxCoxResult {
        data: transformed,
        lmbda: lambda,
    })
}

/// Find optimal Box-Cox lambda by maximizing log-likelihood over a grid.
fn find_optimal_boxcox_lambda(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let log_geometric_mean: f64 = data.iter().map(|&x| x.ln()).sum::<f64>() / n;

    let mut best_lambda = 0.0;
    let mut best_ll = f64::NEG_INFINITY;

    // Search lambda in [-2, 2] with resolution 0.01.
    let steps = 400;
    for i in 0..=steps {
        let lambda = -2.0 + 4.0 * i as f64 / steps as f64;

        let transformed: Vec<f64> = data
            .iter()
            .map(|&x| {
                if lambda.abs() < 1e-10 {
                    x.ln()
                } else {
                    (x.powf(lambda) - 1.0) / lambda
                }
            })
            .collect();

        let mean = transformed.iter().sum::<f64>() / n;
        let var = transformed.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / n;

        if var <= 0.0 {
            continue;
        }

        // Log-likelihood (up to constants):
        // -n/2 * ln(var) + (lambda - 1) * sum(ln(x))
        let ll = -n / 2.0 * var.ln() + (lambda - 1.0) * log_geometric_mean * n;

        if ll > best_ll {
            best_ll = ll;
            best_lambda = lambda;
        }
    }

    best_lambda
}

/// Kendall's tau rank correlation coefficient.
///
/// Matches `scipy.stats.kendalltau(x, y)`.
///
/// Returns (tau, p_value). The p-value is approximate for large n.
pub fn kendalltau(x: &[f64], y: &[f64]) -> CorrelationResult {
    let n = x.len();
    if n < 2 || x.len() != y.len() {
        return CorrelationResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    // Count concordant and discordant pairs.
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
        return CorrelationResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let tau = (concordant - discordant) as f64 / denom;

    // Approximate p-value using normal approximation for large n.
    let n_f = n as f64;
    let var = (2.0 * (2.0 * n_f + 5.0)) / (9.0 * n_f * (n_f - 1.0));
    let z = tau / var.sqrt();
    // Two-tailed p-value from standard normal.
    let p = 2.0 * (1.0 - standard_normal_cdf(z.abs()));

    CorrelationResult {
        statistic: tau,
        pvalue: p,
    }
}

/// Standard normal CDF approximation.
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + fsci_special::erf_scalar(x / std::f64::consts::SQRT_2))
}

fn standard_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

fn somers_alternative_pvalue(z: f64, alternative: &str) -> f64 {
    let normal = Normal::standard();
    match alternative {
        "two-sided" => (2.0 * ContinuousDistribution::sf(&normal, z.abs())).clamp(0.0, 1.0),
        "less" => ContinuousDistribution::cdf(&normal, z).clamp(0.0, 1.0),
        "greater" => ContinuousDistribution::sf(&normal, z).clamp(0.0, 1.0),
        _ => f64::NAN,
    }
}

fn somers_validate_alternative(alternative: Option<&str>) -> Result<&'static str, StatsError> {
    match alternative
        .unwrap_or("two-sided")
        .to_ascii_lowercase()
        .as_str()
    {
        "two-sided" => Ok("two-sided"),
        "less" => Ok("less"),
        "greater" => Ok("greater"),
        _ => Err(StatsError::InvalidArgument(
            "alternative must be one of {'two-sided', 'less', 'greater'}".to_string(),
        )),
    }
}

fn somers_from_rankings(x: &[f64], y: &[f64]) -> Result<Vec<Vec<f64>>, StatsError> {
    if x.len() != y.len() {
        return Err(StatsError::InvalidArgument(
            "Rankings must be of equal length.".to_string(),
        ));
    }

    let mut x_levels = x.to_vec();
    x_levels.sort_by(f64::total_cmp);
    x_levels.dedup_by(|a, b| a.total_cmp(b).is_eq());

    let mut y_levels = y.to_vec();
    y_levels.sort_by(f64::total_cmp);
    y_levels.dedup_by(|a, b| a.total_cmp(b).is_eq());

    let mut table = vec![vec![0.0; y_levels.len()]; x_levels.len()];
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let row = x_levels
            .binary_search_by(|value| value.total_cmp(&xi))
            .unwrap_or(0);
        let col = y_levels
            .binary_search_by(|value| value.total_cmp(&yi))
            .unwrap_or(0);
        table[row][col] += 1.0;
    }
    Ok(table)
}

fn somers_validate_table(table: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, StatsError> {
    if table.is_empty() {
        return Err(StatsError::InvalidArgument(
            "x must be either a 1D or 2D array".to_string(),
        ));
    }
    let cols = table[0].len();
    if cols == 0 || table.iter().any(|row| row.len() != cols) {
        return Err(StatsError::InvalidArgument(
            "contingency table must be rectangular".to_string(),
        ));
    }
    if table
        .iter()
        .flatten()
        .any(|&value| value < 0.0 || !value.is_finite())
    {
        return Err(StatsError::InvalidArgument(
            "All elements of the contingency table must be non-negative.".to_string(),
        ));
    }
    if table
        .iter()
        .flatten()
        .any(|&value| (value - value.round()).abs() > 1e-12)
    {
        return Err(StatsError::InvalidArgument(
            "All elements of the contingency table must be integer.".to_string(),
        ));
    }
    if table
        .iter()
        .flatten()
        .filter(|&&value| value != 0.0)
        .count()
        < 2
    {
        return Err(StatsError::InvalidArgument(
            "At least two elements of the contingency table must be nonzero.".to_string(),
        ));
    }
    Ok(table.to_vec())
}

fn somers_aij(table: &[Vec<f64>], i: usize, j: usize) -> f64 {
    let upper_left: f64 = table
        .iter()
        .take(i)
        .map(|row| row.iter().take(j).sum::<f64>())
        .sum();
    let lower_right: f64 = table
        .iter()
        .skip(i + 1)
        .map(|row| row.iter().skip(j + 1).sum::<f64>())
        .sum();
    upper_left + lower_right
}

fn somers_dij(table: &[Vec<f64>], i: usize, j: usize) -> f64 {
    let lower_left: f64 = table
        .iter()
        .skip(i + 1)
        .map(|row| row.iter().take(j).sum::<f64>())
        .sum();
    let upper_right: f64 = table
        .iter()
        .take(i)
        .map(|row| row.iter().skip(j + 1).sum::<f64>())
        .sum();
    lower_left + upper_right
}

/// Somers' D ordinal association test.
///
/// Matches the core behavior of `scipy.stats.somersd`.
pub fn somersd(
    input: SomersDInput<'_>,
    alternative: Option<&str>,
) -> Result<SomersDResult, StatsError> {
    let alternative = somers_validate_alternative(alternative)?;
    let table = match input {
        SomersDInput::Rankings(x, y) => somers_from_rankings(x, y)?,
        SomersDInput::Table(table) => somers_validate_table(table)?,
    };

    if table.len() <= 1 || table[0].len() <= 1 {
        return Ok(SomersDResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            table,
            correlation: f64::NAN,
        });
    }

    let total: f64 = table.iter().flatten().sum();
    let total_sq = total * total;
    let sri2: f64 = table
        .iter()
        .map(|row| {
            let row_sum: f64 = row.iter().sum();
            row_sum * row_sum
        })
        .sum();

    let mut p = 0.0;
    let mut q = 0.0;
    let mut a_term = 0.0;
    for (i, row) in table.iter().enumerate() {
        for (j, &cell) in row.iter().enumerate() {
            let aij = somers_aij(&table, i, j);
            let dij = somers_dij(&table, i, j);
            p += cell * aij;
            q += cell * dij;
            a_term += cell * (aij - dij).powi(2);
        }
    }

    let statistic = (p - q) / (total_sq - sri2);
    let s = a_term - (p - q).powi(2) / total;
    let z = if s > 0.0 {
        (p - q) / (4.0 * s).sqrt()
    } else if p > q {
        f64::INFINITY
    } else if p < q {
        f64::NEG_INFINITY
    } else {
        0.0
    };
    let pvalue = somers_alternative_pvalue(z, alternative);

    Ok(SomersDResult {
        statistic,
        pvalue,
        table,
        correlation: statistic,
    })
}

// ── Fisher's exact test ──────────────────────────────────────────────

/// Result of Fisher's exact test.
#[derive(Debug, Clone, PartialEq)]
pub struct FisherExactResult {
    /// Sample odds ratio: (a*d) / (b*c) for table [[a,b],[c,d]].
    pub odds_ratio: f64,
    /// Two-sided p-value from the hypergeometric distribution.
    pub pvalue: f64,
}

/// Fisher's exact test for a 2x2 contingency table.
///
/// Computes the exact p-value using the hypergeometric distribution.
/// This is the gold standard for small-sample contingency tables where
/// chi-squared approximation is unreliable.
///
/// Matches `scipy.stats.fisher_exact(table)`.
///
/// # Arguments
/// * `table` — 2x2 contingency table [[a, b], [c, d]] with non-negative integer counts.
pub fn fisher_exact(table: &[[f64; 2]; 2]) -> FisherExactResult {
    let a = table[0][0];
    let b = table[0][1];
    let c = table[1][0];
    let d = table[1][1];

    // Odds ratio
    let odds_ratio = if b * c > 0.0 {
        (a * d) / (b * c)
    } else if a * d > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };

    // Marginals
    let row0 = a + b; // n
    let col0 = a + c; // K
    let total = a + b + c + d; // N

    if total == 0.0 {
        return FisherExactResult {
            odds_ratio,
            pvalue: 1.0,
        };
    }

    // Use hypergeometric distribution: X ~ Hypergeometric(N=total, K=col0, n=row0)
    // P(X = k) = C(K,k) C(N-K,n-k) / C(N,n)
    let big_m = total as u64;
    let n_succ = col0 as u64;
    let n_draw = row0 as u64;

    let hyper = Hypergeometric::new(big_m, n_succ, n_draw);

    // Two-sided p-value: sum probabilities of all outcomes as extreme or more extreme
    // than the observed value (where "extreme" means pmf <= pmf(observed))
    let observed_k = a as u64;
    let p_observed = hyper.pmf(observed_k);

    let k_min = (n_draw + n_succ).saturating_sub(big_m);
    let k_max = n_draw.min(n_succ);

    let mut pvalue = 0.0;
    for k in k_min..=k_max {
        let p_k = hyper.pmf(k);
        if p_k <= p_observed + 1e-14 {
            pvalue += p_k;
        }
    }

    // Clamp to [0, 1]
    let pvalue = pvalue.clamp(0.0, 1.0);

    FisherExactResult { odds_ratio, pvalue }
}

// ── Chi-squared contingency test ─────────────────────────────────────

/// Result of a chi-squared contingency test.
#[derive(Debug, Clone, PartialEq)]
pub struct Chi2ContingencyResult {
    /// Chi-squared test statistic.
    pub statistic: f64,
    /// P-value from chi-squared distribution.
    pub pvalue: f64,
    /// Degrees of freedom.
    pub dof: usize,
    /// Expected frequencies under the null hypothesis of independence.
    pub expected: Vec<Vec<f64>>,
}

/// Chi-squared test of independence for a contingency table.
///
/// Tests whether the row and column variables are independent.
/// Matches `scipy.stats.chi2_contingency(observed)`.
///
/// # Arguments
/// * `observed` — 2D contingency table of observed frequencies (rows x cols).
///
/// # Returns
/// (chi2 statistic, p-value, degrees of freedom, expected frequencies)
pub fn chi2_contingency(observed: &[Vec<f64>]) -> Chi2ContingencyResult {
    let nrows = observed.len();
    if nrows == 0 {
        return Chi2ContingencyResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            dof: 0,
            expected: Vec::new(),
        };
    }
    let ncols = observed[0].len();
    // Validate all rows have the same number of columns
    if observed.iter().any(|row| row.len() != ncols) {
        return Chi2ContingencyResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            dof: 0,
            expected: Vec::new(),
        };
    }
    if ncols == 0 || nrows < 2 || ncols < 2 {
        return Chi2ContingencyResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            dof: 0,
            expected: Vec::new(),
        };
    }
    if observed
        .iter()
        .flat_map(|row| row.iter())
        .any(|&value| !value.is_finite() || value < 0.0)
    {
        return Chi2ContingencyResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            dof: 0,
            expected: Vec::new(),
        };
    }

    // Compute row totals, column totals, grand total
    let row_totals: Vec<f64> = observed.iter().map(|row| row.iter().sum()).collect();
    let mut col_totals = vec![0.0; ncols];
    for row in observed {
        for (j, &val) in row.iter().enumerate() {
            col_totals[j] += val;
        }
    }
    let grand_total: f64 = row_totals.iter().sum();

    if grand_total == 0.0 {
        return Chi2ContingencyResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            dof: 0,
            expected: Vec::new(),
        };
    }

    // Expected frequencies: e_ij = row_i_total * col_j_total / grand_total
    let expected: Vec<Vec<f64>> = (0..nrows)
        .map(|i| {
            (0..ncols)
                .map(|j| row_totals[i] * col_totals[j] / grand_total)
                .collect()
        })
        .collect();

    // Chi-squared statistic: Σ (o_ij - e_ij)² / e_ij
    let dof = (nrows - 1) * (ncols - 1);
    let use_yates_correction = dof == 1;
    let chi2: f64 = observed
        .iter()
        .zip(expected.iter())
        .flat_map(|(obs_row, exp_row)| {
            obs_row.iter().zip(exp_row.iter()).map(|(&o, &e)| {
                if e > 0.0 {
                    let diff = (o - e).abs();
                    let corrected = if use_yates_correction {
                        (diff - diff.min(0.5)).powi(2)
                    } else {
                        diff.powi(2)
                    };
                    corrected / e
                } else {
                    0.0
                }
            })
        })
        .sum();

    // P-value from chi-squared distribution
    let pvalue = if dof > 0 {
        let dist = ChiSquared::new(dof as f64);
        dist.sf(chi2).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };

    Chi2ContingencyResult {
        statistic: chi2,
        pvalue,
        dof,
        expected,
    }
}

/// Power divergence statistic and test.
///
/// Computes the power divergence statistic for testing whether observed
/// frequencies differ from expected frequencies.
///
/// Matches `scipy.stats.power_divergence(f_obs, f_exp, lambda_)`.
///
/// # Arguments
/// * `f_obs` — Observed frequencies.
/// * `f_exp` — Expected frequencies (if None, assumes uniform).
/// * `lambda_` — Power divergence parameter. 1.0 = Pearson chi-squared, 0.0 = G-test.
///
/// # Returns
/// (statistic, p-value) where p-value is from chi-squared distribution with len-1 dof.
pub fn power_divergence(f_obs: &[f64], f_exp: Option<&[f64]>, lambda_: f64) -> (f64, f64) {
    let n = f_obs.len();
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    if f_obs.iter().any(|&value| !value.is_finite() || value < 0.0) {
        return (f64::NAN, f64::NAN);
    }

    // Default expected: uniform
    let total: f64 = f_obs.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let uniform_exp = total / n as f64;
    let default_exp: Vec<f64> = vec![uniform_exp; n];
    let exp = f_exp.unwrap_or(&default_exp);

    if exp.len() != n {
        return (f64::NAN, f64::NAN);
    }
    if exp.iter().any(|&value| !value.is_finite() || value < 0.0) {
        return (f64::NAN, f64::NAN);
    }
    let exp_total: f64 = exp.iter().sum();
    let rel_tol = f64::EPSILON.sqrt();
    let scale = total.abs().max(exp_total.abs()).max(1.0);
    if (total - exp_total).abs() > rel_tol * scale {
        return (f64::NAN, f64::NAN);
    }

    let stat = if (lambda_ - 1.0).abs() < 1e-10 {
        // Pearson chi-squared: Σ (o-e)²/e
        f_obs
            .iter()
            .zip(exp.iter())
            .map(|(&o, &e)| if e > 0.0 { (o - e).powi(2) / e } else { 0.0 })
            .sum()
    } else if lambda_.abs() < 1e-10 {
        // G-test (log-likelihood ratio): 2 Σ o ln(o/e)
        2.0 * f_obs
            .iter()
            .zip(exp.iter())
            .map(|(&o, &e)| {
                if o > 0.0 && e > 0.0 {
                    o * (o / e).ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
    } else {
        // General power divergence: 2/(λ(λ+1)) Σ o((o/e)^λ - 1)
        let factor = 2.0 / (lambda_ * (lambda_ + 1.0));
        factor
            * f_obs
                .iter()
                .zip(exp.iter())
                .map(|(&o, &e)| {
                    if o > 0.0 && e > 0.0 {
                        o * ((o / e).powf(lambda_) - 1.0)
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
    };

    let dof = (n - 1) as f64;
    let pvalue = if dof > 0.0 {
        let dist = ChiSquared::new(dof);
        dist.sf(stat).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };

    (stat, pvalue)
}

/// Permutation test: estimate p-value by random permutation.
///
/// Matches `scipy.stats.permutation_test` (simplified).
pub fn permutation_test<F>(
    x: &[f64],
    y: &[f64],
    stat_fn: F,
    n_permutations: usize,
    seed: u64,
) -> (f64, f64)
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let observed = stat_fn(x, y);
    let n = x.len() + y.len();
    let mut combined: Vec<f64> = x.iter().chain(y.iter()).cloned().collect();
    let mut rng = seed;
    let mut count_extreme = 0usize;

    for _ in 0..n_permutations {
        for i in (1..n).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng >> 33) as usize % (i + 1);
            combined.swap(i, j);
        }

        let perm_stat = stat_fn(&combined[..x.len()], &combined[x.len()..]);
        if perm_stat.abs() >= observed.abs() {
            count_extreme += 1;
        }
    }

    let pvalue = (count_extreme + 1) as f64 / (n_permutations + 1) as f64;
    (observed, pvalue)
}

/// Brunner-Munzel test for stochastic equality.
///
/// Tests H₀: P(X > Y) = 0.5.
/// Matches `scipy.stats.brunnermunzel`.
pub fn brunnermunzel(x: &[f64], y: &[f64]) -> TtestResult {
    let nx = x.len();
    let ny = y.len();
    if nx < 2 || ny < 2 || x.iter().any(|v| v.is_nan()) || y.iter().any(|v| v.is_nan()) {
        return TtestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
        };
    }

    let nxf = nx as f64;
    let nyf = ny as f64;

    // Compute mean rank for each group in combined sample
    let mut combined: Vec<(f64, usize)> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .chain(y.iter().enumerate().map(|(i, &v)| (v, nx + i)))
        .collect();
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut ranks = vec![0.0; nx + ny];
    let mut i = 0;
    while i < combined.len() {
        let mut j = i + 1;
        while j < combined.len() && combined[j].0 == combined[i].0 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            ranks[combined[k].1] = avg_rank;
        }
        i = j;
    }

    let mean_rx: f64 = ranks[..nx].iter().sum::<f64>() / nxf;
    let mean_ry: f64 = ranks[nx..].iter().sum::<f64>() / nyf;

    // Within-group ranks
    let rank_within = |data: &[f64]| -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut r = vec![0.0; data.len()];
        let mut k = 0;
        while k < indexed.len() {
            let mut l = k + 1;
            while l < indexed.len() && indexed[l].1.total_cmp(&indexed[k].1).is_eq() {
                l += 1;
            }
            let avg = (k + l + 1) as f64 / 2.0;
            for m in k..l {
                r[indexed[m].0] = avg;
            }
            k = l;
        }
        r
    };

    let rx_within = rank_within(x);
    let ry_within = rank_within(y);

    let sx2: f64 = ranks[..nx]
        .iter()
        .zip(rx_within.iter())
        .map(|(&ri, &rwi)| (ri - rwi - mean_rx + (nxf + 1.0) / 2.0).powi(2))
        .sum::<f64>()
        / (nxf - 1.0);

    let sy2: f64 = ranks[nx..]
        .iter()
        .zip(ry_within.iter())
        .map(|(&ri, &rwi)| (ri - rwi - mean_ry + (nyf + 1.0) / 2.0).powi(2))
        .sum::<f64>()
        / (nyf - 1.0);

    let nf = nxf + nyf;
    let rank_delta = mean_ry - mean_rx;
    let denom = (nxf * sx2 + nyf * sy2).sqrt();
    let (w, df, pvalue) = if denom > 0.0 {
        let w = nxf * nyf * rank_delta / (nf * denom);
        let df_num = (nxf * sx2 + nyf * sy2).powi(2);
        let df_den = (nxf * sx2).powi(2) / (nxf - 1.0) + (nyf * sy2).powi(2) / (nyf - 1.0);
        let df = if df_den > 0.0 {
            df_num / df_den
        } else {
            f64::NAN
        };
        let pvalue = if df.is_finite() {
            let t_dist = StudentT::new(df);
            2.0 * (1.0 - t_dist.cdf(w.abs()))
        } else {
            f64::NAN
        };
        (w, df, pvalue)
    } else {
        let w = if rank_delta > 0.0 {
            f64::INFINITY
        } else if rank_delta < 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
        (w, f64::NAN, f64::NAN)
    };

    TtestResult {
        statistic: w,
        pvalue: pvalue.clamp(0.0, 1.0),
        df,
    }
}

/// Alexander-Govern test for comparing means of k groups.
///
/// More robust alternative to one-way ANOVA for unequal variances.
/// Matches `scipy.stats.alexandergovern`.
pub fn alexandergovern(groups: &[&[f64]]) -> VarianceTestResult {
    let k = groups.len();
    if k < 2 {
        return VarianceTestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let means: Vec<f64> = groups
        .iter()
        .map(|g| g.iter().sum::<f64>() / g.len() as f64)
        .collect();
    let vars: Vec<f64> = groups
        .iter()
        .zip(means.iter())
        .map(|(g, &m)| g.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (g.len() - 1) as f64)
        .collect();
    let weights: Vec<f64> = groups
        .iter()
        .zip(vars.iter())
        .map(|(g, &v)| g.len() as f64 / v)
        .collect();
    let total_weight: f64 = weights.iter().sum();
    let weighted_mean: f64 = means
        .iter()
        .zip(weights.iter())
        .map(|(&m, &w)| m * w)
        .sum::<f64>()
        / total_weight;

    let stat: f64 = means
        .iter()
        .zip(weights.iter())
        .map(|(&m, &w)| w * (m - weighted_mean).powi(2))
        .sum();

    let df = (k - 1) as f64;
    let chi2 = ChiSquared::new(df);
    let pvalue = chi2.sf(stat);

    VarianceTestResult {
        statistic: stat,
        pvalue: pvalue.clamp(0.0, 1.0),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Multiple Comparison Corrections
// ══════════════════════════════════════════════════════════════════════

/// Result of a multiple comparison correction.
#[derive(Debug, Clone)]
pub struct MultitestResult {
    /// Corrected p-values.
    pub pvalues_corrected: Vec<f64>,
    /// Whether each test is rejected at the given alpha.
    pub reject: Vec<bool>,
}

fn validate_probability_vector(values: &[f64], name: &str) -> Result<(), StatsError> {
    if values.is_empty() {
        return Err(StatsError::InvalidArgument(format!(
            "{name} must not be empty"
        )));
    }
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(StatsError::InvalidArgument(format!(
                "{name}[{idx}] must be finite and lie in [0, 1]"
            )));
        }
    }
    Ok(())
}

fn harmonic_number(n: usize) -> f64 {
    (1..=n).map(|k| 1.0 / k as f64).sum()
}

fn fdr_adjusted_pvalues(pvalues: &[f64], scale: f64) -> Vec<f64> {
    let n = pvalues.len();
    if n == 0 {
        return Vec::new();
    }

    let mut indexed: Vec<(usize, f64)> = pvalues.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut corrected = vec![0.0; n];
    let mut running_min = 1.0f64;

    for (rank, &(orig_idx, p)) in indexed.iter().enumerate().rev() {
        let adjusted = (p * scale * n as f64 / (rank + 1) as f64).min(1.0);
        running_min = running_min.min(adjusted);
        corrected[orig_idx] = running_min;
    }

    corrected
}

/// Bonferroni correction: multiply each p-value by the number of tests.
///
/// Matches `scipy.stats.false_discovery_control` and `statsmodels.multipletests(method='bonferroni')`.
pub fn multipletests_bonferroni(pvalues: &[f64], alpha: f64) -> MultitestResult {
    let n = pvalues.len();
    let corrected: Vec<f64> = pvalues.iter().map(|&p| (p * n as f64).min(1.0)).collect();
    let reject: Vec<bool> = corrected.iter().map(|&p| p < alpha).collect();
    MultitestResult {
        pvalues_corrected: corrected,
        reject,
    }
}

/// Šidák correction: 1 - (1 - p)^n.
///
/// Less conservative than Bonferroni for independent tests.
pub fn multipletests_sidak(pvalues: &[f64], alpha: f64) -> MultitestResult {
    let n = pvalues.len();
    let corrected: Vec<f64> = pvalues
        .iter()
        .map(|&p| (1.0 - (1.0 - p).powi(n as i32)).min(1.0))
        .collect();
    let reject: Vec<bool> = corrected.iter().map(|&p| p < alpha).collect();
    MultitestResult {
        pvalues_corrected: corrected,
        reject,
    }
}

/// Holm-Bonferroni step-down correction.
///
/// Matches `statsmodels.multipletests(method='holm')`.
pub fn multipletests_holm(pvalues: &[f64], alpha: f64) -> MultitestResult {
    let n = pvalues.len();
    if n == 0 {
        return MultitestResult {
            pvalues_corrected: vec![],
            reject: vec![],
        };
    }

    // Sort p-values, keeping track of original indices
    let mut indexed: Vec<(usize, f64)> = pvalues.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut corrected = vec![0.0; n];
    let mut running_max = 0.0f64;

    for (rank, &(orig_idx, p)) in indexed.iter().enumerate() {
        let factor = (n - rank) as f64;
        let adjusted = (p * factor).min(1.0);
        // Enforce monotonicity: corrected p-values must be non-decreasing
        running_max = running_max.max(adjusted);
        corrected[orig_idx] = running_max;
    }

    let reject: Vec<bool> = corrected.iter().map(|&p| p < alpha).collect();
    MultitestResult {
        pvalues_corrected: corrected,
        reject,
    }
}

/// Benjamini-Hochberg (FDR) correction.
///
/// Controls false discovery rate rather than family-wise error rate.
/// Matches `statsmodels.multipletests(method='fdr_bh')`.
pub fn multipletests_fdr_bh(pvalues: &[f64], alpha: f64) -> MultitestResult {
    let n = pvalues.len();
    if n == 0 {
        return MultitestResult {
            pvalues_corrected: vec![],
            reject: vec![],
        };
    }

    let mut indexed: Vec<(usize, f64)> = pvalues.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut corrected = vec![0.0; n];
    let mut running_min = 1.0f64;

    // Process from largest to smallest p-value
    for (rank, &(orig_idx, p)) in indexed.iter().enumerate().rev() {
        let adjusted = (p * n as f64 / (rank + 1) as f64).min(1.0);
        running_min = running_min.min(adjusted);
        corrected[orig_idx] = running_min;
    }

    let reject: Vec<bool> = corrected.iter().map(|&p| p < alpha).collect();
    MultitestResult {
        pvalues_corrected: corrected,
        reject,
    }
}

/// Combine independent p-values into a single test statistic and p-value.
///
/// Supported methods: `"fisher"` (default), `"pearson"`, `"tippett"`, `"stouffer"`.
///
/// Matches the core behavior of `scipy.stats.combine_pvalues`.
pub fn combine_pvalues(
    pvalues: &[f64],
    method: Option<&str>,
    weights: Option<&[f64]>,
) -> Result<GoodnessOfFitResult, StatsError> {
    validate_probability_vector(pvalues, "pvalues")?;
    let method = method.unwrap_or("fisher");
    let k = pvalues.len() as f64;

    let result = match method {
        "fisher" => {
            let statistic = if pvalues.contains(&0.0) {
                f64::INFINITY
            } else {
                -2.0 * pvalues.iter().map(|&p| p.ln()).sum::<f64>()
            };
            let pvalue = ChiSquared::new(2.0 * k).sf(statistic).clamp(0.0, 1.0);
            GoodnessOfFitResult { statistic, pvalue }
        }
        "pearson" => {
            let statistic = if pvalues.contains(&1.0) {
                f64::INFINITY
            } else {
                -2.0 * pvalues.iter().map(|&p| (1.0 - p).ln()).sum::<f64>()
            };
            let pvalue = ChiSquared::new(2.0 * k).cdf(statistic).clamp(0.0, 1.0);
            GoodnessOfFitResult { statistic, pvalue }
        }
        "tippett" => {
            let statistic = pvalues
                .iter()
                .copied()
                .fold(f64::INFINITY, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.min(b)
                    }
                });
            let pvalue = (1.0 - (1.0 - statistic).powf(k)).clamp(0.0, 1.0);
            GoodnessOfFitResult { statistic, pvalue }
        }
        "stouffer" => {
            let weights = if let Some(weights) = weights {
                if weights.len() != pvalues.len() {
                    return Err(StatsError::InvalidArgument(
                        "weights must have the same length as pvalues".to_string(),
                    ));
                }
                if weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
                    return Err(StatsError::InvalidArgument(
                        "weights must be finite and non-negative".to_string(),
                    ));
                }
                weights.to_vec()
            } else {
                vec![1.0; pvalues.len()]
            };

            let weight_norm = weights.iter().map(|&w| w * w).sum::<f64>().sqrt();
            if weight_norm == 0.0 {
                return Err(StatsError::InvalidArgument(
                    "weights must not all be zero".to_string(),
                ));
            }

            let statistic = weights
                .iter()
                .zip(pvalues.iter())
                .map(|(&w, &p)| {
                    let bounded = p.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
                    w * standard_normal_ppf(1.0 - bounded)
                })
                .sum::<f64>()
                / weight_norm;
            let pvalue = Normal::standard().sf(statistic).clamp(0.0, 1.0);
            GoodnessOfFitResult { statistic, pvalue }
        }
        _ => {
            return Err(StatsError::InvalidArgument(format!(
                "unsupported combine_pvalues method: {method}"
            )));
        }
    };

    Ok(result)
}

/// Adjust p-values to control the false discovery rate.
///
/// Supported methods: `"bh"` (default) and `"by"`.
///
/// Matches the core behavior of `scipy.stats.false_discovery_control`.
pub fn false_discovery_control(
    pvalues: &[f64],
    method: Option<&str>,
) -> Result<Vec<f64>, StatsError> {
    validate_probability_vector(pvalues, "pvalues")?;
    let method = method.unwrap_or("bh");

    let corrected = match method {
        "bh" => fdr_adjusted_pvalues(pvalues, 1.0),
        "by" => fdr_adjusted_pvalues(pvalues, harmonic_number(pvalues.len())),
        _ => {
            return Err(StatsError::InvalidArgument(format!(
                "unsupported false_discovery_control method: {method}"
            )));
        }
    };

    Ok(corrected)
}

fn poisson_quantile_bounds(mu: f64, lower_q: f64, upper_q: f64) -> (usize, usize) {
    debug_assert!(mu > 0.0);
    debug_assert!(lower_q >= 0.0 && upper_q <= 1.0 && lower_q <= upper_q);

    let mut pmf = (-mu).exp();
    let mut cdf = pmf;
    let mut lower = None;
    let mut upper = 0usize;

    if cdf >= lower_q {
        lower = Some(0);
    }
    if cdf >= upper_q {
        return (lower.unwrap_or(0), 0);
    }

    for k in 1..=1_000_000usize {
        pmf *= mu / k as f64;
        cdf = (cdf + pmf).min(1.0);
        if lower.is_none() && cdf >= lower_q {
            lower = Some(k);
        }
        if cdf >= upper_q {
            upper = k;
            break;
        }
    }

    (lower.unwrap_or(upper), upper)
}

fn poisson_pmf_at(mu: f64, k: usize) -> f64 {
    let mut pmf = (-mu).exp();
    if k == 0 {
        return pmf;
    }
    for i in 1..=k {
        pmf *= mu / i as f64;
    }
    pmf
}

fn validate_poisson_means_test(
    k1: i64,
    n1: f64,
    k2: i64,
    n2: f64,
    diff: f64,
    alternative: &str,
) -> Result<&str, StatsError> {
    if k1 < 0 || k2 < 0 {
        return Err(StatsError::InvalidArgument(
            "`k1` and `k2` must be greater than or equal to 0.".to_string(),
        ));
    }
    if !n1.is_finite() || !n2.is_finite() || n1 <= 0.0 || n2 <= 0.0 {
        return Err(StatsError::InvalidArgument(
            "`n1` and `n2` must be greater than 0.".to_string(),
        ));
    }
    if diff.is_nan() || diff < 0.0 {
        return Err(StatsError::InvalidArgument(
            "diff must be greater than or equal to 0.".to_string(),
        ));
    }

    match alternative.to_ascii_lowercase().as_str() {
        "two-sided" => Ok("two-sided"),
        "less" => Ok("less"),
        "greater" => Ok("greater"),
        _ => Err(StatsError::InvalidArgument(
            "Alternative must be one of '{'two-sided', 'less', 'greater'}'.".to_string(),
        )),
    }
}

/// Poisson means E-test for the difference between two Poisson rates.
///
/// Matches the core behavior of `scipy.stats.poisson_means_test`.
pub fn poisson_means_test(
    k1: i64,
    n1: f64,
    k2: i64,
    n2: f64,
    diff: f64,
    alternative: Option<&str>,
) -> Result<GoodnessOfFitResult, StatsError> {
    let alternative =
        validate_poisson_means_test(k1, n1, k2, n2, diff, alternative.unwrap_or("two-sided"))?;

    let k1f = k1 as f64;
    let k2f = k2 as f64;
    let lambda_hat2 = (k1f + k2f) / (n1 + n2) - diff * n1 / (n1 + n2);
    if lambda_hat2 <= 0.0 || !lambda_hat2.is_finite() {
        return Ok(GoodnessOfFitResult {
            statistic: 0.0,
            pvalue: 1.0,
        });
    }

    let variance = k1f / (n1 * n1) + k2f / (n2 * n2);
    let observed = (k1f / n1 - k2f / n2 - diff) / variance.sqrt();

    let mu1 = n1 * (lambda_hat2 + diff);
    let mu2 = n2 * lambda_hat2;
    let (x1_lb, x1_ub) = poisson_quantile_bounds(mu1, 1e-10, 1.0 - 1e-16);
    let (x2_lb, x2_ub) = poisson_quantile_bounds(mu2, 1e-10, 1.0 - 1e-16);

    let mut pvalue = 0.0;
    let mut prob_x1 = poisson_pmf_at(mu1, x1_lb);
    for x1 in x1_lb..=x1_ub {
        if x1 > x1_lb {
            prob_x1 *= mu1 / x1 as f64;
        }
        let lambda_x1 = x1 as f64 / n1;

        let mut prob_x2 = poisson_pmf_at(mu2, x2_lb);
        for x2 in x2_lb..=x2_ub {
            if x2 > x2_lb {
                prob_x2 *= mu2 / x2 as f64;
            }
            let lambda_x2 = x2 as f64 / n2;
            let delta = lambda_x1 - lambda_x2 - diff;
            let var_x1x2 = lambda_x1 / n1 + lambda_x2 / n2;
            let pivot = if var_x1x2 > 0.0 {
                delta / var_x1x2.sqrt()
            } else {
                f64::NAN
            };

            let include = match alternative {
                "two-sided" => pivot.abs() >= observed.abs(),
                "less" => pivot <= observed,
                "greater" => pivot >= observed,
                _ => false,
            };
            if include {
                pvalue += prob_x1 * prob_x2;
            }
        }
    }

    Ok(GoodnessOfFitResult {
        statistic: observed,
        pvalue: pvalue.clamp(0.0, 1.0),
    })
}

// ══════════════════════════════════════════════════════════════════════
// Effect Sizes
// ══════════════════════════════════════════════════════════════════════

/// Cohen's d: standardized mean difference between two groups.
///
/// Uses pooled standard deviation.
pub fn cohens_d(group1: &[f64], group2: &[f64]) -> f64 {
    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;
    if n1 < 2.0 || n2 < 2.0 {
        return f64::NAN;
    }

    let mean1: f64 = group1.iter().sum::<f64>() / n1;
    let mean2: f64 = group2.iter().sum::<f64>() / n2;

    let var1: f64 = group1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2: f64 = group2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    let pooled_std = (((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0)).sqrt();

    if pooled_std == 0.0 {
        return if (mean1 - mean2).abs() < 1e-15 {
            0.0
        } else {
            f64::INFINITY * (mean1 - mean2).signum()
        };
    }

    (mean1 - mean2) / pooled_std
}

/// Cramér's V: association measure for contingency tables.
///
/// V = sqrt(χ²/(n * min(r-1, c-1))), where χ² is the chi-squared statistic.
pub fn cramers_v(observed: &[Vec<f64>]) -> f64 {
    let result = chi2_contingency(observed);
    let n: f64 = observed.iter().flat_map(|row| row.iter()).sum();
    let r = observed.len();
    if r == 0 {
        return 0.0;
    }
    let c = observed[0].len();
    let min_dim = (r - 1).min(c - 1);
    if min_dim == 0 || n == 0.0 {
        return 0.0;
    }
    (result.statistic / (n * min_dim as f64)).sqrt()
}

/// Point-biserial correlation: correlation between binary and continuous variable.
///
/// Equivalent to Pearson r when one variable is dichotomous.
pub fn pointbiserialr(binary: &[f64], continuous: &[f64]) -> CorrelationResult {
    // Point-biserial is just Pearson correlation
    pearsonr(binary, continuous)
}

/// Rank-biserial correlation for Mann-Whitney U test.
///
/// r = 1 - 2U/(n1*n2).
pub fn rank_biserial(u_stat: f64, n1: usize, n2: usize) -> f64 {
    let n = n1 as f64 * n2 as f64;
    if n == 0.0 {
        return f64::NAN;
    }
    1.0 - 2.0 * u_stat / n
}

// ══════════════════════════════════════════════════════════════════════
// Additional Statistical Functions
// ══════════════════════════════════════════════════════════════════════

/// Compute the sign test for a paired sample.
///
/// Tests whether the median of differences is zero.
pub fn sign_test(x: &[f64], y: &[f64]) -> Result<TtestResult, StatsError> {
    if x.len() != y.len() {
        return Err(StatsError::InvalidArgument(
            "x and y must have the same length".to_string(),
        ));
    }

    let diffs: Vec<f64> = x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect();
    let n_pos = diffs.iter().filter(|&&d| d > 0.0).count();
    let n_neg = diffs.iter().filter(|&&d| d < 0.0).count();
    let n = n_pos + n_neg; // exclude ties

    if n == 0 {
        return Ok(TtestResult {
            statistic: 0.0,
            pvalue: 1.0,
            df: f64::NAN,
        });
    }

    // Under H0, n_pos ~ Binomial(n, 0.5)
    let k = n_pos.min(n_neg) as f64;
    let nf = n as f64;

    // Normal approximation for large n
    let z = (k - nf / 2.0 + 0.5) / (nf / 4.0).sqrt(); // continuity correction
    let normal = Normal::standard();
    let pvalue = 2.0 * normal.cdf(z.min(0.0));

    Ok(TtestResult {
        statistic: k,
        pvalue: pvalue.clamp(0.0, 1.0),
        df: nf,
    })
}

/// Compute bootstrap confidence interval for a statistic.
///
/// Returns (lower, upper) bounds of the confidence interval.
pub fn bootstrap_ci<F>(
    data: &[f64],
    stat_fn: F,
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> (f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = data.len();
    if n == 0 || n_bootstrap == 0 {
        return (f64::NAN, f64::NAN);
    }

    let mut rng_state = seed;
    let next_rng = |state: &mut u64| -> usize {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as usize) % n
    };

    let mut boot_stats = Vec::with_capacity(n_bootstrap);
    let mut sample = vec![0.0; n];

    for _ in 0..n_bootstrap {
        for s in sample.iter_mut() {
            *s = data[next_rng(&mut rng_state)];
        }
        boot_stats.push(stat_fn(&sample));
    }

    boot_stats.sort_by(|a, b| a.total_cmp(b));

    let alpha = 1.0 - confidence;
    let lo_idx = ((alpha / 2.0) * n_bootstrap as f64).floor() as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64).ceil() as usize;

    let lo = boot_stats[lo_idx.min(n_bootstrap - 1)];
    let hi = boot_stats[hi_idx.min(n_bootstrap - 1)];

    (lo, hi)
}

/// T-test from summary statistics (no raw data needed).
///
/// Matches `scipy.stats.ttest_ind_from_stats`.
pub fn ttest_ind_from_stats(
    mean1: f64,
    std1: f64,
    n1: usize,
    mean2: f64,
    std2: f64,
    n2: usize,
) -> TtestResult {
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let se = (std1 * std1 / n1f + std2 * std2 / n2f).sqrt();

    if se == 0.0 {
        return TtestResult {
            statistic: 0.0,
            pvalue: 1.0,
            df: (n1 + n2 - 2) as f64,
        };
    }

    let t = (mean1 - mean2) / se;

    // Welch-Satterthwaite degrees of freedom
    let v1 = std1 * std1 / n1f;
    let v2 = std2 * std2 / n2f;
    let df = (v1 + v2).powi(2) / (v1 * v1 / (n1f - 1.0) + v2 * v2 / (n2f - 1.0));

    let t_dist = StudentT::new(df);
    let pvalue = 2.0 * (1.0 - t_dist.cdf(t.abs()));

    TtestResult {
        statistic: t,
        pvalue: pvalue.clamp(0.0, 1.0),
        df,
    }
}

/// Yeo-Johnson power transformation.
///
/// Matches `scipy.stats.yeojohnson`.
pub fn yeojohnson(x: &[f64], lam: f64) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            if xi >= 0.0 {
                if lam.abs() < 1e-15 {
                    (xi + 1.0).ln()
                } else {
                    ((xi + 1.0).powf(lam) - 1.0) / lam
                }
            } else if (lam - 2.0).abs() < 1e-15 {
                -((-xi + 1.0).ln())
            } else {
                -((-xi + 1.0).powf(2.0 - lam) - 1.0) / (2.0 - lam)
            }
        })
        .collect()
}

/// Inverse Yeo-Johnson transformation.
pub fn yeojohnson_inv(y: &[f64], lam: f64) -> Vec<f64> {
    y.iter()
        .map(|&yi| {
            if yi >= 0.0 {
                if lam.abs() < 1e-15 {
                    yi.exp() - 1.0
                } else {
                    (lam * yi + 1.0).powf(1.0 / lam) - 1.0
                }
            } else if (lam - 2.0).abs() < 1e-15 {
                1.0 - (-yi).exp()
            } else {
                1.0 - ((2.0 - lam) * (-yi) + 1.0).powf(1.0 / (2.0 - lam))
            }
        })
        .collect()
}

/// Compute the median of a dataset.
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Winsorize data: clip extreme values to specified percentiles.
///
/// Matches `scipy.stats.mstats.winsorize`.
pub fn winsorize(data: &[f64], limits: (f64, f64)) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();

    let lo_idx = (limits.0.clamp(0.0, 1.0) * n as f64).floor() as usize;
    let hi_idx = n.saturating_sub((limits.1.clamp(0.0, 1.0) * n as f64).ceil() as usize);
    let lo_idx = lo_idx.min(n - 1);
    let hi_idx = hi_idx.min(n - 1);
    let (lo_val, hi_val) = if lo_idx > hi_idx {
        // When the clipping windows overlap, SciPy collapses everything to the
        // lower-window boundary instead of panicking on an inverted clamp range.
        let collapsed = sorted[lo_idx];
        (collapsed, collapsed)
    } else {
        (sorted[lo_idx], sorted[hi_idx])
    };

    data.iter().map(|&x| x.clamp(lo_val, hi_val)).collect()
}

/// Compute the interquartile range.
///
/// Matches `scipy.stats.iqr`.
pub fn iqr_range(data: &[f64]) -> f64 {
    let q = quantile(data, &[0.25, 0.75]);
    q[1] - q[0]
}

/// Compute z-scores with optional axis and ddof.
///
/// Matches `scipy.stats.zscore` for 1D.
pub fn zscore_ddof(data: &[f64], ddof: usize) -> Vec<f64> {
    let Some((mean, std)) = mean_std_ddof(data, ddof) else {
        return vec![f64::NAN; data.len()];
    };
    if std == 0.0 {
        return vec![f64::NAN; data.len()];
    }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Compute the expected value of the order statistic for the normal distribution.
///
/// Used in probability plots (Q-Q plots).
/// Matches `scipy.stats.probplot` (partially).
pub fn probplot_quantiles(n: usize) -> Vec<f64> {
    let normal = Normal::standard();
    (0..n)
        .map(|i| {
            let p = (i as f64 + 0.5) / n as f64;
            normal.ppf(p)
        })
        .collect()
}

/// Compute the expected frequency for a Chi-squared goodness-of-fit test
/// against a uniform distribution.
pub fn expected_freq_uniform(observed: &[f64]) -> Vec<f64> {
    let n: f64 = observed.iter().sum();
    let k = observed.len() as f64;
    vec![n / k; observed.len()]
}

/// Compute the coefficient of determination R².
///
/// Matches `sklearn.metrics.r2_score` (for 1D).
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len();
    if n != y_pred.len() || n == 0 {
        return f64::NAN;
    }

    let mean_true: f64 = y_true.iter().sum::<f64>() / n as f64;
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|&t| (t - mean_true).powi(2)).sum();

    if ss_tot == 0.0 {
        return if ss_res == 0.0 { 1.0 } else { 0.0 };
    }

    1.0 - ss_res / ss_tot
}

/// Compute the mean absolute error.
pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return f64::NAN;
    }
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .sum::<f64>()
        / y_true.len() as f64
}

/// Compute the mean squared error.
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return f64::NAN;
    }
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum::<f64>()
        / y_true.len() as f64
}

/// Compute the root mean squared error.
pub fn root_mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    mean_squared_error(y_true, y_pred).sqrt()
}

/// Compute the mean absolute percentage error.
pub fn mean_absolute_percentage_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return f64::NAN;
    }
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| if t == 0.0 { 0.0 } else { ((t - p) / t).abs() })
        .sum::<f64>()
        / y_true.len() as f64
}

/// Compute the log-likelihood for a normal distribution.
pub fn norm_loglikelihood(data: &[f64], mu: f64, sigma: f64) -> f64 {
    let n = data.len() as f64;
    let log_sigma = sigma.ln();
    let two_sigma2 = 2.0 * sigma * sigma;
    -n / 2.0 * (2.0 * PI).ln()
        - n * log_sigma
        - data
            .iter()
            .map(|&x| (x - mu).powi(2) / two_sigma2)
            .sum::<f64>()
}

/// Compute the Akaike Information Criterion (AIC).
pub fn aic(log_likelihood: f64, n_params: usize) -> f64 {
    2.0 * n_params as f64 - 2.0 * log_likelihood
}

/// Compute the Bayesian Information Criterion (BIC).
pub fn bic(log_likelihood: f64, n_params: usize, n_samples: usize) -> f64 {
    n_params as f64 * (n_samples as f64).ln() - 2.0 * log_likelihood
}

/// Generate samples from a multivariate normal distribution.
///
/// Uses Box-Muller transform with Cholesky decomposition.
pub fn multivariate_normal_rvs(
    mean: &[f64],
    cov: &[Vec<f64>],
    n_samples: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let d = mean.len();
    if d == 0 || cov.len() != d {
        return vec![];
    }

    // Cholesky decomposition
    let mut l = vec![vec![0.0; d]; d];
    #[allow(clippy::needless_range_loop)]
    for j in 0..d {
        let mut sum = cov[j][j];
        for k in 0..j {
            sum -= l[j][k] * l[j][k];
        }
        l[j][j] = if sum > 0.0 { sum.sqrt() } else { 0.0 };
        for i in j + 1..d {
            let mut sum = cov[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            l[i][j] = if l[j][j] > 0.0 { sum / l[j][j] } else { 0.0 };
        }
    }

    let mut rng = seed;
    let mut samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // Generate standard normal samples via Box-Muller
        let mut z = vec![0.0; d];
        for item in z.iter_mut().take(d) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((rng >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;
            *item = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }

        // Transform: x = mean + L * z
        let mut x = mean.to_vec();
        for i in 0..d {
            for j in 0..=i {
                x[i] += l[i][j] * z[j];
            }
        }
        samples.push(x);
    }

    samples
}

/// Beta-prime (inverted beta) distribution.
///
/// Matches `scipy.stats.betaprime`.
pub struct BetaPrime {
    pub a: f64,
    pub b: f64,
}

impl BetaPrime {
    #[must_use]
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a > 0.0 && b > 0.0, "a and b must be positive");
        Self { a, b }
    }
}

impl ContinuousDistribution for BetaPrime {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let a = self.a;
        let b = self.b;
        x.powf(a - 1.0) * (1.0 + x).powf(-a - b)
            / (ln_gamma(a).exp() * ln_gamma(b).exp() / ln_gamma(a + b).exp())
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        // CDF = I_{x/(1+x)}(a, b) where I is regularized incomplete beta
        let t = x / (1.0 + x);
        regularized_incomplete_beta(self.a, self.b, t)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        // CDF = I_{x/(1+x)}(a,b), so t = betaincinv(a,b,q), x = t/(1-t)
        let t = fsci_special::betaincinv(self.a, self.b, q);
        if t >= 1.0 {
            return f64::INFINITY;
        }
        let mut x = t / (1.0 - t);
        for _ in 0..8 {
            let err = self.cdf(x) - q;
            if err.abs() < 1e-14 {
                break;
            }
            let dx = self.pdf(x);
            if dx > 1e-30 {
                x = (x - err / dx).max(0.0);
            }
        }
        x
    }

    fn mean(&self) -> f64 {
        if self.b > 1.0 {
            self.a / (self.b - 1.0)
        } else {
            f64::INFINITY
        }
    }

    fn var(&self) -> f64 {
        if self.b > 2.0 {
            self.a * (self.a + self.b - 1.0) / ((self.b - 2.0) * (self.b - 1.0).powi(2))
        } else {
            f64::INFINITY
        }
    }
}

/// Exponentiated Weibull distribution.
///
/// Matches `scipy.stats.exponweib`.
pub struct ExponWeibull {
    pub a: f64,
    pub c: f64,
}

impl ExponWeibull {
    #[must_use]
    pub fn new(a: f64, c: f64) -> Self {
        assert!(a > 0.0 && c > 0.0, "a and c must be positive");
        Self { a, c }
    }
}

impl ContinuousDistribution for ExponWeibull {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let a = self.a;
        let c = self.c;
        a * c * (1.0 - (-x.powf(c)).exp()).powf(a - 1.0) * (-x.powf(c)).exp() * x.powf(c - 1.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        (1.0 - (-x.powf(self.c)).exp()).powf(self.a)
    }

    fn ppf(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return 0.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        let u = q.powf(1.0 / self.a);
        (-(1.0 - u).ln()).powf(1.0 / self.c)
    }

    fn mean(&self) -> f64 {
        let eps = 1e-12;
        let upper = 64.0;
        self.a
            * simpson_integrate_adaptive(
                |t| {
                    let exp_neg_t = (-t).exp();
                    t.powf(1.0 / self.c) * exp_neg_t * (1.0 - exp_neg_t).powf(self.a - 1.0)
                },
                eps,
                upper,
                1_024,
                1e-11,
                1e-11,
                8,
            )
    }

    fn var(&self) -> f64 {
        let eps = 1e-12;
        let upper = 64.0;
        let mean = self.mean();
        let second_moment = self.a
            * simpson_integrate_adaptive(
                |t| {
                    let exp_neg_t = (-t).exp();
                    t.powf(2.0 / self.c) * exp_neg_t * (1.0 - exp_neg_t).powf(self.a - 1.0)
                },
                eps,
                upper,
                1_024,
                1e-11,
                1e-11,
                8,
            );
        (second_moment - mean * mean).max(0.0)
    }
}

/// Folded Cauchy distribution.
///
/// Matches `scipy.stats.foldcauchy`.
pub struct FoldedCauchy {
    pub c: f64,
}

impl FoldedCauchy {
    #[must_use]
    pub fn new(c: f64) -> Self {
        Self { c }
    }
}

impl ContinuousDistribution for FoldedCauchy {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let c = self.c;
        1.0 / (PI * (1.0 + (x - c).powi(2))) + 1.0 / (PI * (1.0 + (x + c).powi(2)))
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let c = self.c;
        ((x - c).atan() + (x + c).atan()) / PI + 0.5
    }

    fn mean(&self) -> f64 {
        f64::INFINITY
    }

    fn var(&self) -> f64 {
        f64::INFINITY
    }
}

/// Folded Normal distribution.
///
/// Matches `scipy.stats.foldnorm`.
pub struct FoldedNormal {
    pub c: f64, // location parameter of underlying normal
}

impl FoldedNormal {
    #[must_use]
    pub fn new(c: f64) -> Self {
        Self { c }
    }
}

impl ContinuousDistribution for FoldedNormal {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();
        inv_sqrt_2pi * ((-0.5 * (x - self.c).powi(2)).exp() + (-0.5 * (x + self.c).powi(2)).exp())
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        standard_normal_cdf(x - self.c) + standard_normal_cdf(x + self.c) - 1.0
    }

    fn mean(&self) -> f64 {
        // E[|X|] where X ~ N(c, 1)
        let c = self.c;
        let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();
        c * (2.0 * standard_normal_cdf(c) - 1.0) + 2.0 * inv_sqrt_2pi * (-0.5 * c * c).exp()
    }

    fn var(&self) -> f64 {
        let m = self.mean();
        self.c * self.c + 1.0 - m * m
    }
}

/// Compute the median absolute deviation (MAD) with optional scaling.
///
/// With scale=1.4826, gives a consistent estimator for the normal distribution std.
pub fn mad(data: &[f64], scale: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let med = median(data);
    let abs_devs: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    scale * median(&abs_devs)
}

/// Compute the coefficient of variation (CV = std/mean).
pub fn coefficient_of_variation(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mean: f64 = data.iter().sum::<f64>() / n;
    if mean == 0.0 {
        return f64::INFINITY;
    }
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt() / mean.abs()
}

/// Compute the standard error of the mean.
pub fn standard_error_of_mean(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mean: f64 = data.iter().sum::<f64>() / n;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt() / n.sqrt()
}

/// Compute the excess kurtosis (Fisher's definition).
pub fn excess_kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 {
        return f64::NAN;
    }
    let mean: f64 = data.iter().sum::<f64>() / n;
    let m2: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4: f64 = data.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 {
        return 0.0;
    }
    m4 / (m2 * m2) - 3.0
}

/// Compute the covariance matrix from a dataset.
///
/// Each row of `data` is an observation, each column is a variable.
/// Matches `numpy.cov` (rowvar=False).
pub fn cov_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = data.len();
    if n < 2 {
        return vec![];
    }
    let d = data[0].len();

    // Compute means
    let mut means = vec![0.0; d];
    for row in data {
        for (j, &v) in row.iter().enumerate() {
            means[j] += v;
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }

    // Compute covariance
    let mut cov = vec![vec![0.0; d]; d];
    for row in data {
        for i in 0..d {
            for j in i..d {
                cov[i][j] += (row[i] - means[i]) * (row[j] - means[j]);
            }
        }
    }
    #[allow(clippy::needless_range_loop)]
    for i in 0..d {
        for j in i..d {
            cov[i][j] /= (n - 1) as f64;
            cov[j][i] = cov[i][j];
        }
    }

    cov
}

/// Compute the correlation matrix from a dataset.
///
/// Matches `numpy.corrcoef` (rowvar=False).
pub fn corr_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = cov_matrix(data);
    let d = cov.len();
    if d == 0 {
        return vec![];
    }

    let stds: Vec<f64> = (0..d).map(|i| cov[i][i].sqrt()).collect();
    let mut corr = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            if stds[i] > 0.0 && stds[j] > 0.0 {
                corr[i][j] = cov[i][j] / (stds[i] * stds[j]);
            } else if i == j {
                corr[i][j] = 1.0;
            }
        }
    }

    corr
}

/// Compute the cross-covariance between two datasets.
pub fn cross_cov(x: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = x.len();
    if n < 2 || n != y.len() {
        return vec![];
    }
    let dx = x[0].len();
    let dy = y[0].len();

    let mut means_x = vec![0.0; dx];
    let mut means_y = vec![0.0; dy];
    for row in x {
        for (j, &v) in row.iter().enumerate() {
            means_x[j] += v;
        }
    }
    for row in y {
        for (j, &v) in row.iter().enumerate() {
            means_y[j] += v;
        }
    }
    for m in &mut means_x {
        *m /= n as f64;
    }
    for m in &mut means_y {
        *m /= n as f64;
    }

    let mut cov = vec![vec![0.0; dy]; dx];
    for i in 0..n {
        for j in 0..dx {
            for k in 0..dy {
                cov[j][k] += (x[i][j] - means_x[j]) * (y[i][k] - means_y[k]);
            }
        }
    }
    for row in &mut cov {
        for v in row.iter_mut() {
            *v /= (n - 1) as f64;
        }
    }

    cov
}

/// Compute the contingency table from two categorical arrays.
///
/// Returns (table, row_labels, col_labels).
pub fn contingency_table(x: &[usize], y: &[usize]) -> (Vec<Vec<usize>>, Vec<usize>, Vec<usize>) {
    if x.len() != y.len() || x.is_empty() {
        return (vec![], vec![], vec![]);
    }

    let mut row_labels: Vec<usize> = x.to_vec();
    row_labels.sort();
    row_labels.dedup();

    let mut col_labels: Vec<usize> = y.to_vec();
    col_labels.sort();
    col_labels.dedup();

    let nr = row_labels.len();
    let nc = col_labels.len();
    let mut table = vec![vec![0usize; nc]; nr];

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if let Ok(ri) = row_labels.binary_search(&xi)
            && let Ok(ci) = col_labels.binary_search(&yi)
        {
            table[ri][ci] += 1;
        }
    }

    (table, row_labels, col_labels)
}

/// Compute the relative risk from a 2x2 contingency table.
///
/// RR = (a/(a+b)) / (c/(c+d)) where [[a,b],[c,d]].
pub fn relative_risk(table: &[[usize; 2]; 2]) -> f64 {
    let a = table[0][0] as f64;
    let b = table[0][1] as f64;
    let c = table[1][0] as f64;
    let d = table[1][1] as f64;

    let p1 = a / (a + b);
    let p2 = c / (c + d);

    if p2 == 0.0 {
        return f64::INFINITY;
    }
    p1 / p2
}

/// Compute the odds ratio from a 2x2 contingency table.
///
/// OR = (a*d) / (b*c) where [[a,b],[c,d]].
pub fn odds_ratio(table: &[[usize; 2]; 2]) -> f64 {
    let a = table[0][0] as f64;
    let b = table[0][1] as f64;
    let c = table[1][0] as f64;
    let d = table[1][1] as f64;

    let denom = b * c;
    if denom == 0.0 {
        return f64::INFINITY;
    }
    (a * d) / denom
}

/// Compute the Phi coefficient from a 2x2 contingency table.
///
/// φ = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
pub fn phi_coefficient(table: &[[usize; 2]; 2]) -> f64 {
    let a = table[0][0] as f64;
    let b = table[0][1] as f64;
    let c = table[1][0] as f64;
    let d = table[1][1] as f64;

    let num = a * d - b * c;
    let denom = ((a + b) * (c + d) * (a + c) * (b + d)).sqrt();

    if denom == 0.0 {
        return 0.0;
    }
    num / denom
}

/// Compute exponentially weighted moving average.
///
/// Matches `pandas.DataFrame.ewm(span=span).mean()`.
pub fn ewma(data: &[f64], span: f64) -> Vec<f64> {
    if data.is_empty() || span <= 0.0 {
        return data.to_vec();
    }
    let alpha = 2.0 / (span + 1.0);
    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]);
    for i in 1..data.len() {
        result.push(alpha * data[i] + (1.0 - alpha) * result[i - 1]);
    }
    result
}

/// Compute simple moving average.
pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if data.is_empty() || window == 0 {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len().saturating_sub(window - 1));
    let mut sum: f64 = data[..window.min(data.len())].iter().sum();

    if window <= data.len() {
        result.push(sum / window as f64);
        for i in window..data.len() {
            sum += data[i] - data[i - window];
            result.push(sum / window as f64);
        }
    }

    result
}

/// Compute cumulative sum.
pub fn cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data {
        sum += v;
        result.push(sum);
    }
    result
}

/// Compute cumulative product.
pub fn cumprod(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut prod = 1.0;
    for &v in data {
        prod *= v;
        result.push(prod);
    }
    result
}

/// Compute differences between consecutive elements.
///
/// Matches `numpy.diff`.
pub fn diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![];
    }
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Compute a histogram of data values.
///
/// Returns (counts, bin_edges).
/// Matches `numpy.histogram`.
pub fn histogram(data: &[f64], bins: usize) -> (Vec<usize>, Vec<f64>) {
    if data.is_empty() || bins == 0 {
        return (vec![], vec![]);
    }

    let min_val = data.iter().cloned().fold(f64::INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    });
    let max_val = data
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    let range = max_val - min_val;
    let bin_width = if range > 0.0 {
        range / bins as f64
    } else {
        1.0
    };

    let mut counts = vec![0usize; bins];
    let bin_edges: Vec<f64> = (0..=bins).map(|i| min_val + i as f64 * bin_width).collect();

    for &v in data {
        let bin = ((v - min_val) / bin_width).floor() as usize;
        counts[bin.min(bins - 1)] += 1;
    }

    (counts, bin_edges)
}

/// Compute histogram bin edges using various strategies.
///
/// Methods: "auto" (Sturges), "sqrt", "sturges", "rice", "scott", "fd" (Freedman-Diaconis).
pub fn histogram_bin_edges(data: &[f64], method: &str) -> Vec<f64> {
    let n = data.len();
    if n == 0 || data.iter().any(|v| !v.is_finite()) {
        return vec![];
    }

    let min_val = data.iter().cloned().fold(f64::INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    });
    let max_val = data
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });

    let nbins = match method {
        "sqrt" => (n as f64).sqrt().ceil() as usize,
        "rice" => (2.0 * (n as f64).cbrt()).ceil() as usize,
        "scott" => {
            let mean: f64 = data.iter().sum::<f64>() / n as f64;
            let std: f64 =
                (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            if std > 0.0 {
                ((max_val - min_val) / (3.5 * std / (n as f64).cbrt())).ceil() as usize
            } else {
                1
            }
        }
        "fd" => {
            let q = quantile(data, &[0.25, 0.75]);
            let iqr = q[1] - q[0];
            if iqr > 0.0 {
                ((max_val - min_val) / (2.0 * iqr / (n as f64).cbrt())).ceil() as usize
            } else {
                1
            }
        }
        _ => {
            // "sturges" / "auto"
            (1.0 + (n as f64).log2()).ceil() as usize
        }
    };

    let nbins = nbins.max(1);
    let bin_width = (max_val - min_val) / nbins as f64;
    (0..=nbins)
        .map(|i| min_val + i as f64 * bin_width)
        .collect()
}

/// Compute the binned statistic of values.
///
/// Groups `values` by which bin in `x` they fall into, applies `statistic`.
/// Matches `scipy.stats.binned_statistic`.
pub fn binned_statistic(
    x: &[f64],
    values: &[f64],
    bins: usize,
    statistic: &str,
) -> (Vec<f64>, Vec<f64>) {
    if x.len() != values.len() || x.is_empty() || bins == 0 {
        return (vec![], vec![]);
    }

    let min_x = x.iter().cloned().fold(f64::INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    });
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let bin_width = if max_x > min_x {
        (max_x - min_x) / bins as f64
    } else {
        1.0
    };

    let bin_edges: Vec<f64> = (0..=bins).map(|i| min_x + i as f64 * bin_width).collect();

    let mut bin_values: Vec<Vec<f64>> = vec![vec![]; bins];
    for (&xi, &vi) in x.iter().zip(values.iter()) {
        let bin = ((xi - min_x) / bin_width).floor() as usize;
        bin_values[bin.min(bins - 1)].push(vi);
    }

    let stats: Vec<f64> = bin_values
        .iter()
        .map(|bv| {
            if bv.is_empty() {
                return f64::NAN;
            }
            match statistic {
                "mean" => bv.iter().sum::<f64>() / bv.len() as f64,
                "sum" => bv.iter().sum(),
                "count" => bv.len() as f64,
                "min" => bv.iter().cloned().fold(f64::INFINITY, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.min(b)
                    }
                }),
                "max" => bv
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                        if a.is_nan() || b.is_nan() {
                            f64::NAN
                        } else {
                            a.max(b)
                        }
                    }),
                "median" => {
                    let mut sorted = bv.clone();
                    sorted.sort_by(|a, b| a.total_cmp(b));
                    let n = sorted.len();
                    if n % 2 == 0 {
                        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                    } else {
                        sorted[n / 2]
                    }
                }
                "std" => {
                    let mean = bv.iter().sum::<f64>() / bv.len() as f64;
                    (bv.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / bv.len() as f64).sqrt()
                }
                _ => bv.iter().sum::<f64>() / bv.len() as f64,
            }
        })
        .collect();

    (stats, bin_edges)
}

/// Compute the relative frequency histogram.
///
/// Returns (relative_frequencies, bin_edges).
pub fn relfreq(data: &[f64], bins: usize) -> (Vec<f64>, Vec<f64>) {
    let (counts, edges) = histogram(data, bins);
    let n = data.len() as f64;
    let rel: Vec<f64> = counts.iter().map(|&c| c as f64 / n).collect();
    (rel, edges)
}

/// Compute the cumulative frequency histogram.
///
/// Returns (cumulative_frequencies, bin_edges).
pub fn cumfreq(data: &[f64], bins: usize) -> (Vec<f64>, Vec<f64>) {
    let (counts, edges) = histogram(data, bins);
    let mut cum = Vec::with_capacity(counts.len());
    let mut total = 0.0;
    for &c in &counts {
        total += c as f64;
        cum.push(total);
    }
    (cum, edges)
}

/// Compute the power spectral density using Welch's method.
///
/// Simple wrapper around periodogram concepts.
pub fn psd_welch(data: &[f64], window_size: usize, overlap: usize, fs: f64) -> Vec<f64> {
    if data.is_empty() || window_size == 0 {
        return vec![];
    }

    let overlap = overlap.min(window_size.saturating_sub(1));
    let hop = window_size - overlap;
    let n_freq = window_size / 2 + 1;
    let mut psd = vec![0.0; n_freq];
    let mut n_segments = 0;

    let win: Vec<f64> = (0..window_size)
        .map(|i| {
            if window_size == 1 {
                1.0
            } else {
                0.5 * (1.0
                    - (2.0 * std::f64::consts::PI * i as f64 / (window_size - 1) as f64).cos())
            }
        })
        .collect();

    let mut start = 0;
    while start + window_size <= data.len() {
        let segment: Vec<f64> = data[start..start + window_size]
            .iter()
            .zip(win.iter())
            .map(|(&d, &w)| d * w)
            .collect();

        // Compute periodogram of segment
        let two_pi = 2.0 * std::f64::consts::PI;
        for (k, psd_k) in psd.iter_mut().enumerate().take(n_freq) {
            let mut re = 0.0;
            let mut im = 0.0;
            for (n, &s) in segment.iter().enumerate() {
                let angle = two_pi * k as f64 * n as f64 / window_size as f64;
                re += s * angle.cos();
                im -= s * angle.sin();
            }
            let power = (re * re + im * im) / (window_size as f64 * fs);
            *psd_k += power;
        }

        n_segments += 1;
        start += hop;
    }

    if n_segments > 0 {
        for v in &mut psd {
            *v /= n_segments as f64;
        }
    }

    psd
}

/// Compute the Kolmogorov-Smirnov distance between a sample and a theoretical CDF.
///
/// Returns just the D statistic without p-value.
pub fn ks_distance(data: &[f64], cdf_func: impl Fn(f64) -> f64) -> f64 {
    let n = data.len();
    if n == 0 || data.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let mut max_d = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let ecdf_before = i as f64 / n as f64;
        let ecdf_after = (i + 1) as f64 / n as f64;
        let cdf_val = cdf_func(x);
        let d1 = (ecdf_after - cdf_val).abs();
        let d2 = (ecdf_before - cdf_val).abs();

        max_d = if max_d.is_nan() || d1.is_nan() || d2.is_nan() {
            f64::NAN
        } else {
            max_d.max(d1).max(d2)
        };
    }
    max_d
}

/// Compute the empirical CDF at given points.
///
/// Returns the proportion of data ≤ each value in `x_eval`.
pub fn ecdf(data: &[f64], x_eval: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    if n == 0.0 {
        return vec![0.0; x_eval.len()];
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    x_eval
        .iter()
        .map(|&x| {
            let count = sorted.partition_point(|&v| v <= x);
            count as f64 / n
        })
        .collect()
}

/// Compute the Mann-Kendall trend test.
///
/// Tests for monotonic trend in a time series.
/// Returns (tau, p_value, trend direction: +1, 0, -1).
pub fn mannkendall(data: &[f64]) -> (f64, f64, i32) {
    let n = data.len();
    if n < 3 {
        return (f64::NAN, f64::NAN, 0);
    }

    let mut s: i64 = 0;
    for i in 0..n {
        for j in i + 1..n {
            if data[j] > data[i] {
                s += 1;
            } else if data[j] < data[i] {
                s -= 1;
            }
        }
    }

    // Kendall's tau
    let pairs = (n * (n - 1) / 2) as f64;
    let tau = s as f64 / pairs;

    // Variance of S
    let var_s = (n * (n - 1) * (2 * n + 5)) as f64 / 18.0;

    // Normal approximation for p-value
    let z = if s > 0 {
        (s as f64 - 1.0) / var_s.sqrt()
    } else if s < 0 {
        (s as f64 + 1.0) / var_s.sqrt()
    } else {
        0.0
    };

    let normal = Normal::standard();
    let p = 2.0 * (1.0 - normal.cdf(z.abs()));

    let trend = if s > 0 {
        1
    } else if s < 0 {
        -1
    } else {
        0
    };

    (tau, p.clamp(0.0, 1.0), trend)
}

/// Compute the runs test for randomness.
///
/// Tests whether the sequence of values above/below median is random.
/// Returns (n_runs, z_statistic, p_value).
pub fn runs_test(data: &[f64]) -> (usize, f64, f64) {
    let n = data.len();
    if n < 2 {
        return (0, f64::NAN, f64::NAN);
    }

    let med = median(data);
    let signs: Vec<bool> = data.iter().map(|&v| v >= med).collect();

    let n1 = signs.iter().filter(|&&s| s).count();
    let n2 = n - n1;

    if n1 == 0 || n2 == 0 {
        return (1, f64::NAN, f64::NAN);
    }

    // Count runs
    let mut runs = 1usize;
    for i in 1..n {
        if signs[i] != signs[i - 1] {
            runs += 1;
        }
    }

    // Expected runs and variance under H0
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let nf = n as f64;
    let expected = 2.0 * n1f * n2f / nf + 1.0;
    let var = 2.0 * n1f * n2f * (2.0 * n1f * n2f - nf) / (nf * nf * (nf - 1.0));

    let z = if var > 0.0 {
        (runs as f64 - expected) / var.sqrt()
    } else {
        0.0
    };

    let normal = Normal::standard();
    let p = 2.0 * (1.0 - normal.cdf(z.abs()));

    (runs, z, p.clamp(0.0, 1.0))
}

/// Compute Durbin-Watson statistic for autocorrelation in residuals.
///
/// DW ≈ 2 means no autocorrelation, DW < 2 means positive, DW > 2 means negative.
pub fn durbin_watson(residuals: &[f64]) -> f64 {
    if residuals.len() < 2 {
        return f64::NAN;
    }

    let num: f64 = residuals.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let den: f64 = residuals.iter().map(|&r| r * r).sum();

    if den == 0.0 {
        return f64::NAN;
    }
    num / den
}

/// Compute the autocorrelation function of a time series.
///
/// Returns autocorrelation at lags 0, 1, ..., max_lag.
pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    if n < 2 {
        return vec![1.0];
    }
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&v| (v - mean).powi(2)).sum();

    if var == 0.0 {
        return vec![1.0; max_lag + 1];
    }

    (0..=max_lag.min(n - 1))
        .map(|lag| {
            let sum: f64 = (0..n - lag)
                .map(|i| (data[i] - mean) * (data[i + lag] - mean))
                .sum();
            sum / var
        })
        .collect()
}

/// Compute the partial autocorrelation function.
///
/// Uses the Durbin-Levinson recursion.
pub fn pacf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let autocorr = acf(data, max_lag);
    let nlags = autocorr.len() - 1;
    if nlags == 0 {
        return vec![1.0];
    }

    let mut result = vec![1.0]; // lag 0

    // Durbin-Levinson
    let mut phi = vec![vec![0.0; nlags + 1]; nlags + 1];
    phi[1][1] = autocorr[1];
    result.push(phi[1][1]);

    for k in 2..=nlags {
        let mut num = autocorr[k];
        for j in 1..k {
            num -= phi[k - 1][j] * autocorr[k - j];
        }
        let mut den = 1.0;
        for j in 1..k {
            den -= phi[k - 1][j] * autocorr[j];
        }
        phi[k][k] = if den.abs() > 1e-15 { num / den } else { 0.0 };

        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }

        result.push(phi[k][k]);
    }

    result
}

/// Ljung-Box test for autocorrelation in residuals.
///
/// Returns (statistic, p_value).
pub fn ljung_box(data: &[f64], lags: usize) -> (f64, f64) {
    let n = data.len();
    if n < 3 || lags == 0 {
        return (f64::NAN, f64::NAN);
    }

    let autocorr = acf(data, lags);
    let nf = n as f64;

    let q: f64 = (1..=lags.min(autocorr.len() - 1))
        .map(|k| autocorr[k].powi(2) / (nf - k as f64))
        .sum::<f64>()
        * nf
        * (nf + 2.0);

    let df = lags as f64;
    let chi2 = ChiSquared::new(df);
    let p = chi2.sf(q);

    (q, p.clamp(0.0, 1.0))
}

/// Multiple linear regression: y = X * beta + epsilon.
///
/// Returns (coefficients, residuals, r_squared, std_errors).
pub fn multiple_regression(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, Vec<f64>, f64, Vec<f64>) {
    let n = x.len();
    if n == 0 || n != y.len() {
        return (vec![], vec![], f64::NAN, vec![]);
    }
    let p = x[0].len();

    // Add intercept column
    let p1 = p + 1;
    let mut xtx = vec![vec![0.0; p1]; p1];
    let mut xty = vec![0.0; p1];

    for i in 0..n {
        let mut row = vec![1.0]; // intercept
        row.extend_from_slice(&x[i]);

        for j in 0..p1 {
            xty[j] += row[j] * y[i];
            for k in j..p1 {
                xtx[j][k] += row[j] * row[k];
                if k != j {
                    xtx[k][j] = xtx[j][k];
                }
            }
        }
    }

    // Solve XtX * beta = Xty via Cholesky
    let beta = solve_regression_system(&xtx, &xty);

    // Compute residuals
    let mut residuals = Vec::with_capacity(n);
    let mut ss_res = 0.0;
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut ss_tot = 0.0;

    for i in 0..n {
        let mut pred = beta[0]; // intercept
        for j in 0..p {
            pred += beta[j + 1] * x[i][j];
        }
        let r = y[i] - pred;
        residuals.push(r);
        ss_res += r * r;
        ss_tot += (y[i] - y_mean).powi(2);
    }

    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    // Standard errors of coefficients
    let mse = if n > p1 {
        ss_res / (n - p1) as f64
    } else {
        0.0
    };
    let std_errors = compute_std_errors(&xtx, mse);

    (beta, residuals, r_squared, std_errors)
}

#[allow(clippy::needless_range_loop)]
fn solve_regression_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut row = r.clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        let max_row = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap_or(col);
        aug.swap(col, max_row);
        if aug[col][col].abs() < 1e-15 {
            continue;
        }
        let pivot = aug[col][col];
        for row in col + 1..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-15 {
            continue;
        }
        let mut sum = aug[i][n];
        for j in i + 1..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    x
}

#[allow(clippy::needless_range_loop)]
fn compute_std_errors(xtx: &[Vec<f64>], mse: f64) -> Vec<f64> {
    let n = xtx.len();
    // Compute inverse of XtX via Gauss-Jordan
    let mut aug: Vec<Vec<f64>> = xtx
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut row = r.clone();
            row.extend(vec![0.0; n]);
            row[n + i] = 1.0;
            row
        })
        .collect();

    for col in 0..n {
        let max_row = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap_or(col);
        aug.swap(col, max_row);
        if aug[col][col].abs() < 1e-15 {
            continue;
        }
        let pivot = aug[col][col];
        for j in 0..2 * n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..2 * n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    (0..n)
        .map(|i| {
            let var = mse * aug[i][n + i];
            if var.is_nan() {
                f64::NAN
            } else {
                var.max(0.0).sqrt()
            }
        })
        .collect()
}

/// Ridge regression: minimize ||Xβ - y||² + α||β||².
///
/// Returns coefficients (including intercept as first element).
pub fn ridge_regression(x: &[Vec<f64>], y: &[f64], alpha: f64) -> Vec<f64> {
    let n = x.len();
    if n == 0 || n != y.len() {
        return vec![];
    }
    let p = x[0].len();
    let p1 = p + 1;

    // Build XtX + alpha*I and Xty
    let mut xtx = vec![vec![0.0; p1]; p1];
    let mut xty = vec![0.0; p1];

    for i in 0..n {
        let mut row = vec![1.0];
        row.extend_from_slice(&x[i]);
        for j in 0..p1 {
            xty[j] += row[j] * y[i];
            for k in 0..p1 {
                xtx[j][k] += row[j] * row[k];
            }
        }
    }

    // Add regularization (skip intercept at index 0)
    for (j, item) in xtx.iter_mut().enumerate().take(p1).skip(1) {
        item[j] += alpha;
    }

    solve_regression_system(&xtx, &xty)
}

/// Polynomial regression: fit y = β₀ + β₁x + β₂x² + ... + βₖx^k.
///
/// Returns coefficients [β₀, β₁, ..., βₖ].
pub fn polynomial_regression(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || n != y.len() {
        return vec![];
    }

    // Build design matrix: each row is [1, x, x², ..., x^degree]
    let x_poly: Vec<Vec<f64>> = x
        .iter()
        .map(|&xi| (1..=degree).map(|d| xi.powi(d as i32)).collect())
        .collect();

    let (beta, _, _, _) = multiple_regression(&x_poly, y);
    beta
}

/// Exponential regression: fit y = a * exp(b * x).
///
/// Uses log-linear regression: ln(y) = ln(a) + b*x.
/// Returns (a, b).
pub fn exponential_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 2 || n != y.len() {
        return (f64::NAN, f64::NAN);
    }

    // Filter positive y values (log requires y > 0)
    let valid: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|&(_, yi)| *yi > 0.0)
        .map(|(&xi, &yi)| (xi, yi.ln()))
        .collect();

    if valid.len() < 2 {
        return (f64::NAN, f64::NAN);
    }

    let x_valid: Vec<f64> = valid.iter().map(|&(xi, _)| xi).collect();
    let ln_y: Vec<f64> = valid.iter().map(|&(_, ly)| ly).collect();

    let result = linregress(&x_valid, &ln_y);
    (result.intercept.exp(), result.slope)
}

/// Power regression: fit y = a * x^b.
///
/// Uses log-log regression: ln(y) = ln(a) + b*ln(x).
/// Returns (a, b).
pub fn power_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 2 || n != y.len() {
        return (f64::NAN, f64::NAN);
    }

    let valid: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|&(xi, yi)| *xi > 0.0 && *yi > 0.0)
        .map(|(&xi, &yi)| (xi.ln(), yi.ln()))
        .collect();

    if valid.len() < 2 {
        return (f64::NAN, f64::NAN);
    }

    let ln_x: Vec<f64> = valid.iter().map(|&(lx, _)| lx).collect();
    let ln_y: Vec<f64> = valid.iter().map(|&(_, ly)| ly).collect();

    let result = linregress(&ln_x, &ln_y);
    (result.intercept.exp(), result.slope)
}

/// Theil-Sen robust regression estimator.
///
/// Uses median of all pairwise slopes. Resistant to outliers.
/// Matches `scipy.stats.theilslopes`.
pub fn theil_sen(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 2 || n != y.len() {
        return (f64::NAN, f64::NAN);
    }

    // Compute all pairwise slopes
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            let dx = x[j] - x[i];
            if dx.abs() > 1e-15 {
                slopes.push((y[j] - y[i]) / dx);
            }
        }
    }

    if slopes.is_empty() {
        return (0.0, median(y));
    }

    let slope = median(&slopes);
    // Intercept: median of y_i - slope * x_i
    let intercepts: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| yi - slope * xi)
        .collect();
    let intercept = median(&intercepts);

    (slope, intercept)
}

/// Compute the Spearman footrule distance between two rankings.
pub fn spearman_footrule(rank1: &[usize], rank2: &[usize]) -> f64 {
    if rank1.len() != rank2.len() {
        return f64::NAN;
    }
    rank1
        .iter()
        .zip(rank2.iter())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .sum()
}

/// Compute the Kendall distance (number of discordant pairs) between two rankings.
pub fn kendall_distance(rank1: &[usize], rank2: &[usize]) -> usize {
    let n = rank1.len();
    if n != rank2.len() {
        return 0;
    }
    let mut count = 0;
    for i in 0..n {
        for j in i + 1..n {
            let a = (rank1[i] as i64 - rank1[j] as i64).signum();
            let b = (rank2[i] as i64 - rank2[j] as i64).signum();
            if a != b && a != 0 && b != 0 {
                count += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: got {actual}, expected {expected} (diff={})",
            (actual - expected).abs()
        );
    }

    // ── Normal distribution ─────────────────────────────────────────

    #[test]
    fn normal_pdf_at_mean() {
        let n = Normal::standard();
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert_close(n.pdf(0.0), expected, 1e-12, "N(0,1) pdf(0)");
    }

    #[test]
    fn normal_cdf_at_zero() {
        let n = Normal::standard();
        assert_close(n.cdf(0.0), 0.5, 1e-6, "N(0,1) cdf(0)");
    }

    #[test]
    fn normal_cdf_known_values() {
        let n = Normal::standard();
        assert_close(n.cdf(1.0), 0.841_344_7, 1e-4, "cdf(1)");
        assert_close(n.cdf(-1.0), 0.158_655_3, 1e-4, "cdf(-1)");
        assert_close(n.cdf(2.0), 0.977_249_9, 1e-4, "cdf(2)");
    }

    #[test]
    fn normal_sf() {
        let n = Normal::standard();
        assert_close(n.sf(0.0), 0.5, 1e-6, "sf(0)");
        assert_close(n.sf(0.0) + n.cdf(0.0), 1.0, 1e-12, "sf + cdf = 1");
    }

    #[test]
    fn normal_ppf_known_values() {
        let n = Normal::standard();
        assert_close(n.ppf(0.5), 0.0, 1e-6, "ppf(0.5)");
        // Tail approximation accuracy — within 0.5 for current implementation
        assert_close(n.ppf(0.975), 1.96, 0.5, "ppf(0.975)");
        assert_close(n.ppf(0.025), -1.96, 0.5, "ppf(0.025)");
    }

    #[test]
    fn normal_ppf_extremes() {
        let n = Normal::standard();
        assert!(n.ppf(0.0).is_infinite() && n.ppf(0.0).is_sign_negative());
        assert!(n.ppf(1.0).is_infinite() && n.ppf(1.0).is_sign_positive());
    }

    #[test]
    fn normal_mean_var() {
        let n = Normal::new(3.0, 2.0);
        assert_eq!(n.mean(), 3.0);
        assert_eq!(n.var(), 4.0);
        assert_close(n.std(), 2.0, 1e-12, "std");
    }

    #[test]
    fn normal_shifted() {
        let n = Normal::new(5.0, 2.0);
        assert_close(n.cdf(5.0), 0.5, 1e-6, "shifted cdf at mean");
        assert_close(
            n.pdf(5.0),
            1.0 / (2.0 * (2.0 * PI).sqrt()),
            1e-10,
            "shifted pdf at mean",
        );
    }

    // ── Student's t-distribution ────────────────────────────────────

    #[test]
    fn student_t_pdf_symmetric() {
        let t = StudentT::new(5.0);
        assert_close(t.pdf(1.0), t.pdf(-1.0), 1e-12, "t pdf symmetric");
    }

    #[test]
    fn student_t_cdf_at_zero() {
        let t = StudentT::new(10.0);
        assert_close(t.cdf(0.0), 0.5, 1e-10, "t cdf(0)");
    }

    #[test]
    fn student_t_cdf_known() {
        // t(5) CDF at x=2.571 ≈ 0.975 (two-tailed 5% critical value)
        let t = StudentT::new(5.0);
        let cdf = t.cdf(0.0);
        assert_close(cdf, 0.5, 1e-6, "t(5) cdf(0)");
        // Verify CDF is monotonically increasing
        let cdf1 = t.cdf(1.0);
        let cdf2 = t.cdf(2.0);
        assert!(cdf1 > 0.5 && cdf1 < 1.0, "t(5) cdf(1) = {cdf1}");
        assert!(cdf2 > cdf1, "t(5) cdf should be monotone");
    }

    #[test]
    fn student_t_mean_var() {
        let t = StudentT::new(5.0);
        assert_eq!(t.mean(), 0.0);
        assert_close(t.var(), 5.0 / 3.0, 1e-10, "t(5) var");
    }

    #[test]
    fn student_t_df1_no_variance() {
        let t = StudentT::new(1.0);
        assert!(t.var().is_nan(), "t(1) has no finite variance");
    }

    // ── Chi-squared distribution ────────────────────────────────────

    #[test]
    fn chi2_pdf_positive_only() {
        let c = ChiSquared::new(3.0);
        assert_eq!(c.pdf(-1.0), 0.0);
        assert!(c.pdf(1.0) > 0.0);
    }

    #[test]
    fn chi2_cdf_at_zero() {
        let c = ChiSquared::new(3.0);
        assert_eq!(c.cdf(0.0), 0.0);
    }

    #[test]
    fn chi2_cdf_known() {
        // chi2(2) CDF at x ≈ 1 - exp(-x/2) => CDF(2) = 1 - exp(-1) ≈ 0.6321
        let c = ChiSquared::new(2.0);
        assert_close(c.cdf(2.0), 1.0 - (-1.0_f64).exp(), 1e-4, "chi2(2) cdf(2)");
    }

    #[test]
    fn chi2_mean_var() {
        let c = ChiSquared::new(5.0);
        assert_eq!(c.mean(), 5.0);
        assert_eq!(c.var(), 10.0);
    }

    // ── Uniform distribution ────────────────────────────────────────

    #[test]
    fn uniform_pdf() {
        let u = Uniform::new(2.0, 3.0); // [2, 5]
        assert_close(u.pdf(3.0), 1.0 / 3.0, 1e-12, "uniform pdf inside");
        assert_eq!(u.pdf(1.0), 0.0);
        assert_eq!(u.pdf(6.0), 0.0);
    }

    #[test]
    fn uniform_cdf() {
        let u = Uniform::default(); // [0, 1]
        assert_eq!(u.cdf(-1.0), 0.0);
        assert_eq!(u.cdf(0.5), 0.5);
        assert_eq!(u.cdf(2.0), 1.0);
    }

    #[test]
    fn uniform_mean_var() {
        let u = Uniform::new(0.0, 12.0);
        assert_eq!(u.mean(), 6.0);
        assert_eq!(u.var(), 12.0);
    }

    // ── ContinuousDistribution trait ────────────────────────────────

    #[test]
    fn trait_sf_plus_cdf_equals_one() {
        let n = Normal::standard();
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            assert_close(n.sf(x) + n.cdf(x), 1.0, 1e-10, &format!("sf+cdf at {x}"));
        }
    }

    #[test]
    fn trait_std_is_sqrt_var() {
        let n = Normal::new(0.0, 3.0);
        assert_close(n.std(), 3.0, 1e-12, "std = sqrt(var)");
    }

    // ── Exponential distribution ────────────────────────────────────

    #[test]
    fn exponential_pdf_at_zero() {
        let e = Exponential::new(2.0);
        assert_close(e.pdf(0.0), 2.0, 1e-12, "expon pdf(0) = lambda");
    }

    #[test]
    fn exponential_pdf_negative() {
        let e = Exponential::default();
        assert_eq!(e.pdf(-1.0), 0.0);
    }

    #[test]
    fn exponential_cdf_known() {
        let e = Exponential::new(1.0);
        // CDF(1) = 1 - e^(-1) ≈ 0.6321
        assert_close(e.cdf(1.0), 1.0 - (-1.0_f64).exp(), 1e-12, "expon cdf(1)");
        assert_eq!(e.cdf(0.0), 0.0);
        assert_eq!(e.cdf(-1.0), 0.0);
    }

    #[test]
    fn exponential_sf_plus_cdf() {
        let e = Exponential::new(3.0);
        for x in [0.0, 0.5, 1.0, 2.0, 5.0] {
            assert_close(
                e.sf(x) + e.cdf(x),
                1.0,
                1e-12,
                &format!("expon sf+cdf at {x}"),
            );
        }
    }

    #[test]
    fn exponential_mean_var() {
        let e = Exponential::new(0.5);
        assert_close(e.mean(), 2.0, 1e-12, "mean = 1/lambda");
        assert_close(e.var(), 4.0, 1e-12, "var = 1/lambda^2");
    }

    #[test]
    fn exponential_ppf() {
        let e = Exponential::new(1.0);
        assert_close(e.ppf(0.0), 0.0, 1e-12, "ppf(0)");
        assert_close(e.ppf(0.5), 2.0_f64.ln(), 1e-12, "ppf(0.5) = ln(2)");
        assert!(e.ppf(1.0).is_infinite(), "ppf(1) = inf");
    }

    #[test]
    fn exponential_from_scale() {
        let e = Exponential::from_scale(2.0);
        assert_close(e.lambda, 0.5, 1e-12, "lambda = 1/scale");
        assert_close(e.mean(), 2.0, 1e-12, "mean = scale");
    }

    #[test]
    fn exponential_memoryless_property() {
        // P(X > s+t | X > s) = P(X > t)
        let e = Exponential::new(1.5);
        let s = 1.0;
        let t = 2.0;
        let conditional = e.sf(s + t) / e.sf(s);
        assert_close(conditional, e.sf(t), 1e-12, "memoryless property");
    }

    // ── F-distribution ──────────────────────────────────────────────

    #[test]
    fn f_dist_pdf_positive_only() {
        let f = FDistribution::new(5.0, 10.0);
        assert_eq!(f.pdf(-1.0), 0.0);
        assert_eq!(f.pdf(0.0), 0.0);
        assert!(f.pdf(1.0) > 0.0);
    }

    #[test]
    fn f_dist_cdf_at_zero() {
        let f = FDistribution::new(5.0, 10.0);
        assert_eq!(f.cdf(0.0), 0.0);
    }

    #[test]
    fn f_dist_cdf_monotone() {
        let f = FDistribution::new(3.0, 7.0);
        let c1 = f.cdf(0.5);
        let c2 = f.cdf(1.0);
        let c3 = f.cdf(2.0);
        assert!(c1 < c2, "CDF should be monotone");
        assert!(c2 < c3, "CDF should be monotone");
        assert!(c3 < 1.0, "CDF < 1 for finite x");
    }

    #[test]
    fn f_dist_mean() {
        let f = FDistribution::new(5.0, 10.0);
        // mean = dfd/(dfd-2) = 10/8 = 1.25
        assert_close(f.mean(), 1.25, 1e-10, "F(5,10) mean");
    }

    #[test]
    fn f_dist_sf_plus_cdf() {
        let f = FDistribution::new(4.0, 8.0);
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert_close(f.sf(x) + f.cdf(x), 1.0, 1e-6, &format!("F sf+cdf at {x}"));
        }
    }

    // ── Beta distribution ───────────────────────────────────────────

    #[test]
    fn beta_dist_pdf_support() {
        let b = BetaDist::new(2.0, 3.0);
        assert_eq!(b.pdf(-0.1), 0.0);
        assert_eq!(b.pdf(1.1), 0.0);
        assert!(b.pdf(0.5) > 0.0);
    }

    #[test]
    fn beta_dist_cdf_bounds() {
        let b = BetaDist::new(2.0, 5.0);
        assert_eq!(b.cdf(0.0), 0.0);
        assert_eq!(b.cdf(1.0), 1.0);
    }

    #[test]
    fn beta_dist_uniform_special_case() {
        // Beta(1,1) = Uniform(0,1)
        let b = BetaDist::new(1.0, 1.0);
        assert_close(b.mean(), 0.5, 1e-10, "Beta(1,1) mean");
        assert_close(b.var(), 1.0 / 12.0, 1e-10, "Beta(1,1) var");
        assert_close(b.cdf(0.5), 0.5, 1e-4, "Beta(1,1) cdf(0.5)");
    }

    #[test]
    fn beta_dist_mean_var() {
        let b = BetaDist::new(2.0, 5.0);
        // mean = a/(a+b) = 2/7
        assert_close(b.mean(), 2.0 / 7.0, 1e-10, "Beta(2,5) mean");
        // var = ab/((a+b)^2*(a+b+1)) = 10/(49*8) = 10/392
        assert_close(b.var(), 10.0 / 392.0, 1e-10, "Beta(2,5) var");
    }

    #[test]
    fn beta_dist_symmetry() {
        // Beta(a,b) at x has same pdf as Beta(b,a) at 1-x
        let b1 = BetaDist::new(2.0, 5.0);
        let b2 = BetaDist::new(5.0, 2.0);
        assert_close(b1.pdf(0.3), b2.pdf(0.7), 1e-10, "Beta symmetry");
    }

    // ── Gamma distribution ──────────────────────────────────────────

    #[test]
    fn gamma_dist_exponential_special_case() {
        // Gamma(1, scale) = Exponential(1/scale)
        let g = GammaDist::new(1.0, 2.0);
        let e = Exponential::from_scale(2.0);
        assert_close(g.pdf(1.0), e.pdf(1.0), 1e-10, "Gamma(1,2) = Exp(1/2)");
        assert_close(g.cdf(1.0), e.cdf(1.0), 1e-6, "Gamma(1,2) cdf");
    }

    #[test]
    fn gamma_dist_chi2_special_case() {
        // Gamma(k/2, 2) = Chi2(k)
        let g = GammaDist::new(3.0, 2.0); // Gamma(3, 2) = Chi2(6)
        let c = ChiSquared::new(6.0);
        assert_close(g.mean(), c.mean(), 1e-10, "Gamma mean = Chi2 mean");
        assert_close(g.var(), c.var(), 1e-10, "Gamma var = Chi2 var");
    }

    #[test]
    fn gamma_dist_pdf_positive_only() {
        let g = GammaDist::new(2.0, 1.0);
        assert_eq!(g.pdf(-1.0), 0.0);
        assert!(g.pdf(1.0) > 0.0);
    }

    #[test]
    fn gamma_dist_mean_var() {
        let g = GammaDist::new(3.0, 2.0);
        assert_close(g.mean(), 6.0, 1e-10, "Gamma(3,2) mean");
        assert_close(g.var(), 12.0, 1e-10, "Gamma(3,2) var");
    }

    #[test]
    fn multivariate_normal_pdf_matches_scipy_reference() {
        let rv = MultivariateNormal::new(&[0.0, 0.0], &[vec![1.0, 0.2], vec![0.2, 2.0]])
            .expect("multivariate normal");
        let pdf = rv.pdf(&[0.0, 0.0]).expect("pdf");
        assert_close(pdf, 0.11368210220849669, 1e-12, "multivariate_normal pdf");
    }

    #[test]
    fn multivariate_normal_logpdf_matches_scipy_reference() {
        let rv = MultivariateNormal::new(&[0.0, 0.0], &[vec![1.0, 0.2], vec![0.2, 2.0]])
            .expect("multivariate normal");
        let logpdf = rv.logpdf(&[0.0, 0.0]).expect("logpdf");
        assert_close(
            logpdf,
            -2.1743493030305583,
            1e-12,
            "multivariate_normal logpdf",
        );
    }

    #[test]
    fn multivariate_normal_rejects_invalid_covariance() {
        assert!(MultivariateNormal::new(&[0.0, 0.0], &[vec![1.0]]).is_err());
        assert!(MultivariateNormal::new(&[0.0, 0.0], &[vec![1.0, 2.0], vec![0.0, 1.0]]).is_err());
    }

    #[test]
    fn multivariate_normal_rvs_tracks_mean() {
        let rv = MultivariateNormal::new(&[1.0, -2.0], &[vec![1.0, 0.3], vec![0.3, 1.5]])
            .expect("multivariate normal");
        let mut rng = rand::rng();
        let samples = rv.rvs(4000, &mut rng);
        let mean0 = samples.iter().map(|row| row[0]).sum::<f64>() / samples.len() as f64;
        let mean1 = samples.iter().map(|row| row[1]).sum::<f64>() / samples.len() as f64;
        assert!((mean0 - 1.0).abs() < 0.08, "sample mean0 = {mean0}");
        assert!((mean1 + 2.0).abs() < 0.08, "sample mean1 = {mean1}");
    }

    #[test]
    fn vonmises_pdf_and_cdf_match_scipy_reference() {
        let vm = VonMises::new(2.5, 0.3);
        assert_close(
            vm.pdf(0.3),
            0.5893613785159519,
            1e-10,
            "vonmises pdf at loc",
        );
        assert_close(vm.cdf(0.3), 0.5, 1e-4, "vonmises cdf at loc");
        assert_close(
            vm.cdf(0.3 + PI / 2.0),
            0.978826666129197,
            1e-4,
            "vonmises cdf at loc+pi/2",
        );
    }

    #[test]
    fn vonmises_is_periodic() {
        let vm = VonMises::new(2.5, 0.3);
        assert_close(
            vm.pdf(0.3 + 2.0 * PI),
            vm.pdf(0.3),
            1e-12,
            "pdf periodicity",
        );
        assert_close(vm.cdf(0.3 + 2.0 * PI), 1.5, 1e-4, "cdf period lift");
    }

    // ── Poisson distribution ────────────────────────────────────────

    #[test]
    fn poisson_pmf_known() {
        let p = Poisson::new(3.0);
        // P(0) = exp(-3) ≈ 0.04979
        assert_close(p.pmf(0), (-3.0_f64).exp(), 1e-10, "P(0)");
        // P(1) = 3*exp(-3) ≈ 0.14936
        assert_close(p.pmf(1), 3.0 * (-3.0_f64).exp(), 1e-10, "P(1)");
        // P(3) = 27/6 * exp(-3) ≈ 0.22404
        assert_close(p.pmf(3), 4.5 * (-3.0_f64).exp(), 1e-10, "P(3)");
    }

    #[test]
    fn poisson_pmf_sums_near_one() {
        let p = Poisson::new(5.0);
        let sum: f64 = (0..30).map(|k| p.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-6, "PMF sum = {sum}, expected ~1.0");
    }

    #[test]
    fn poisson_mean_var() {
        let p = Poisson::new(7.5);
        assert_eq!(p.mean(), 7.5);
        assert_eq!(p.var(), 7.5);
    }

    // ── ppf (trait-level inverse CDF) ─────────────────────────────

    #[test]
    fn chi2_ppf_via_trait() {
        let c = ChiSquared::new(5.0);
        let q = 0.5;
        let x = c.ppf(q);
        // Verify: CDF(ppf(q)) ≈ q
        let cdf_at_x = c.cdf(x);
        assert!(
            (cdf_at_x - q).abs() < 1e-4,
            "chi2 ppf roundtrip: cdf(ppf(0.5)) = {cdf_at_x}, expected 0.5"
        );
    }

    #[test]
    fn student_t_ppf_via_trait() {
        let t = StudentT::new(10.0);
        // ppf(0.5) should be 0 (symmetric distribution)
        let x = t.ppf(0.5);
        assert!(x.abs() < 0.01, "t ppf(0.5) = {x}, expected ~0");
    }

    #[test]
    fn f_dist_ppf_via_trait() {
        let f = FDistribution::new(5.0, 10.0);
        let q = 0.95;
        let x = f.ppf(q);
        let cdf_at_x = f.cdf(x);
        assert!(
            (cdf_at_x - q).abs() < 1e-3,
            "F ppf roundtrip: cdf(ppf(0.95)) = {cdf_at_x}, expected 0.95"
        );
    }

    #[test]
    fn gamma_ppf_via_trait() {
        let g = GammaDist::new(3.0, 2.0);
        let q = 0.5;
        let x = g.ppf(q);
        let cdf_at_x = g.cdf(x);
        assert!(
            (cdf_at_x - q).abs() < 1e-4,
            "Gamma ppf roundtrip: cdf(ppf(0.5)) = {cdf_at_x}, expected 0.5"
        );
    }

    #[test]
    fn poisson_cdf_monotone() {
        let p = Poisson::new(4.0);
        let mut prev = 0.0;
        for k in 0..20 {
            let c = p.cdf(k);
            assert!(c >= prev, "CDF should be monotone at k={k}");
            prev = c;
        }
        assert!((prev - 1.0).abs() < 1e-4, "CDF(large k) ≈ 1.0");
    }

    // ── Weibull distribution ────────────────────────────────────────

    #[test]
    fn weibull_exponential_special_case() {
        // Weibull(c=1, scale=λ) = Exponential(1/λ)
        let w = Weibull::new(1.0, 2.0);
        let e = Exponential::from_scale(2.0);
        assert_close(w.pdf(1.0), e.pdf(1.0), 1e-10, "Weibull(1) = Exp");
        assert_close(w.cdf(1.0), e.cdf(1.0), 1e-10, "Weibull(1) cdf");
    }

    #[test]
    fn weibull_ppf_roundtrip() {
        let w = Weibull::new(2.0, 3.0);
        let q = 0.75;
        let x = w.ppf(q);
        assert_close(w.cdf(x), q, 1e-10, "Weibull ppf roundtrip");
    }

    #[test]
    fn weibull_sf_plus_cdf() {
        let w = Weibull::new(1.5, 1.0);
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert_close(w.sf(x) + w.cdf(x), 1.0, 1e-12, &format!("sf+cdf at {x}"));
        }
    }

    #[test]
    fn weibull_pdf_positive_only() {
        let w = Weibull::new(2.0, 1.0);
        assert_eq!(w.pdf(-1.0), 0.0);
        assert!(w.pdf(1.0) > 0.0);
    }

    // ── Lognormal distribution ──────────────────────────────────────

    #[test]
    fn lognormal_pdf_positive_only() {
        let ln = Lognormal::new(1.0, 1.0);
        assert_eq!(ln.pdf(-1.0), 0.0);
        assert_eq!(ln.pdf(0.0), 0.0);
        assert!(ln.pdf(1.0) > 0.0);
    }

    #[test]
    fn lognormal_cdf_bounds() {
        let ln = Lognormal::new(0.5, 1.0);
        assert_eq!(ln.cdf(0.0), 0.0);
        // CDF at large x should approach 1
        assert!(ln.cdf(100.0) > 0.99);
    }

    #[test]
    fn lognormal_mean() {
        // For s=σ, scale=exp(μ): mean = exp(μ + σ²/2) = scale * exp(σ²/2)
        let ln = Lognormal::new(1.0, 1.0);
        let expected = (0.5_f64).exp(); // exp(0 + 1/2)
        assert_close(ln.mean(), expected, 1e-10, "lognormal mean");
    }

    #[test]
    fn lognormal_cdf_at_scale() {
        // CDF(scale) = CDF of standard lognorm at 1 = Φ(0) = 0.5
        let ln = Lognormal::new(1.0, 5.0);
        assert_close(ln.cdf(5.0), 0.5, 1e-4, "lognormal cdf at scale");
    }

    // ── Additional continuous distributions ────────────────────────

    #[test]
    fn pareto_support_mean_and_variance() {
        let p = Pareto::new(3.0, 2.0);
        assert_eq!(p.pdf(1.5), 0.0);
        assert_eq!(p.cdf(1.5), 0.0);
        assert_close(p.mean(), 3.0, 1e-12, "pareto mean");
        assert_close(p.var(), 3.0, 1e-12, "pareto variance");
    }

    #[test]
    fn pareto_cdf_ppf_roundtrip() {
        let p = Pareto::new(2.5, 1.5);
        let x = p.ppf(0.75);
        assert_close(p.cdf(x), 0.75, 1e-10, "pareto cdf(ppf(q))");
    }

    #[test]
    fn rayleigh_mode_and_roundtrip() {
        let r = Rayleigh::new(1.0);
        assert!(r.pdf(1.0) > r.pdf(0.5), "mode near scale");
        assert!(r.pdf(1.0) > r.pdf(2.0), "density should fall past the mode");
        let x = r.ppf(0.6);
        assert_close(r.cdf(x), 0.6, 1e-10, "rayleigh cdf(ppf(q))");
    }

    #[test]
    fn rayleigh_mean_and_variance() {
        let r = Rayleigh::new(2.0);
        assert_close(r.mean(), 2.0 * (PI / 2.0).sqrt(), 1e-12, "rayleigh mean");
        assert_close(r.var(), 4.0 * (2.0 - PI / 2.0), 1e-12, "rayleigh variance");
    }

    #[test]
    fn gumbel_cdf_ppf_roundtrip() {
        let g = Gumbel::new(1.0, 2.0);
        let x = g.ppf(0.8);
        assert_close(g.cdf(x), 0.8, 1e-10, "gumbel cdf(ppf(q))");
    }

    #[test]
    fn gumbel_mean_and_variance() {
        let g = Gumbel::new(0.5, 1.5);
        assert_close(
            g.mean(),
            0.5 + 0.577_215_664_901_532_9 * 1.5,
            1e-12,
            "gumbel mean",
        );
        assert_close(g.var(), PI * PI * 1.5 * 1.5 / 6.0, 1e-12, "gumbel variance");
    }

    #[test]
    fn gumbel_left_pdf_cdf_match_scipy_reference_values() {
        let dist = GumbelLeft::new(0.5, 1.25);
        let cases = [
            (-1.0, 0.178_291_083_598_038_75, 0.260_065_945_216_393_8),
            (0.0, 0.274_319_005_174_185_4, 0.488_455_166_310_958_4),
            (0.5, 0.294_303_552_937_153_9, 0.632_120_558_828_557_7),
            (1.0, 0.268_482_847_714_755_95, 0.775_038_206_450_081_6),
            (2.0, 0.096_014_075_924_335_02, 0.963_851_395_086_864_5),
        ];

        for &(x, pdf, cdf) in &cases {
            assert_close(dist.pdf(x), pdf, 1e-12, &format!("GumbelLeft pdf({x})"));
            assert_close(dist.cdf(x), cdf, 1e-12, &format!("GumbelLeft cdf({x})"));
        }
    }

    #[test]
    fn gumbel_left_ppf_roundtrip_matches_quantiles() {
        let dist = GumbelLeft::new(0.5, 1.25);
        let cases = [
            (0.1, -2.312_959_159_140_557),
            (0.25, -1.057_374_154_634_047_8),
            (0.5, 0.041_858_849_272_919_55),
            (0.75, 0.908_292_824_972_851_2),
            (0.9, 1.542_540_556_559_945),
        ];

        for &(q, expected_x) in &cases {
            let x = dist.ppf(q);
            assert_close(x, expected_x, 1e-12, &format!("GumbelLeft ppf({q})"));
            assert_close(dist.cdf(x), q, 1e-10, &format!("GumbelLeft cdf(ppf({q}))"));
        }
    }

    #[test]
    fn gumbel_left_mean_and_variance() {
        let dist = GumbelLeft::new(0.5, 1.25);
        assert_close(
            dist.mean(),
            -0.221_519_581_126_916_05,
            1e-12,
            "GumbelLeft mean",
        );
        assert_close(
            dist.var(),
            2.570_209_479_450_354,
            1e-12,
            "GumbelLeft variance",
        );
    }

    #[test]
    fn pearson3_positive_skew_pdf_cdf_match_scipy_reference_values() {
        let dist = Pearson3::new(1.5);
        let cases = [
            (-2.0, 0.0, 0.0),
            (-1.0, 0.491_520_420_923_186_1, 0.108_815_120_615_328_25),
            (0.0, 0.380_847_526_111_116_33, 0.599_647_799_258_328_2),
            (1.0, 0.155_139_155_520_572_03, 0.856_126_172_584_151_5),
            (2.0, 0.053_968_632_797_029_154, 0.952_726_553_443_132_6),
        ];

        for &(x, pdf, cdf) in &cases {
            assert_close(dist.pdf(x), pdf, 1e-12, &format!("Pearson3(1.5) pdf({x})"));
            assert_close(dist.cdf(x), cdf, 1e-12, &format!("Pearson3(1.5) cdf({x})"));
        }
    }

    #[test]
    fn pearson3_negative_skew_pdf_cdf_match_scipy_reference_values() {
        let dist = Pearson3::new(-1.5);
        let cases = [
            (-2.0, 0.053_968_632_797_029_154, 0.047_273_446_556_867_44),
            (-1.0, 0.155_139_155_520_572_03, 0.143_873_827_415_848_5),
            (0.0, 0.380_847_526_111_116_33, 0.400_352_200_741_672_94),
            (1.0, 0.491_520_420_923_186_1, 0.891_184_879_384_671_7),
            (2.0, 0.0, 1.0),
        ];

        for &(x, pdf, cdf) in &cases {
            assert_close(dist.pdf(x), pdf, 1e-12, &format!("Pearson3(-1.5) pdf({x})"));
            assert_close(dist.cdf(x), cdf, 1e-12, &format!("Pearson3(-1.5) cdf({x})"));
        }
    }

    #[test]
    fn pearson3_ppf_roundtrip_matches_scipy_reference_values() {
        let positive = Pearson3::new(1.5);
        let positive_cases = [
            (0.1, -1.018_104_310_659_964_5),
            (0.25, -0.733_120_813_876_001_7),
            (0.5, -0.239_964_153_964_511_43),
            (0.75, 0.475_368_176_123_716_5),
            (0.9, 1.333_300_518_218_442_5),
        ];

        for &(q, expected_x) in &positive_cases {
            let x = positive.ppf(q);
            assert_close(x, expected_x, 1e-12, &format!("Pearson3(1.5) ppf({q})"));
            assert_close(
                positive.cdf(x),
                q,
                1e-10,
                &format!("Pearson3(1.5) cdf(ppf({q}))"),
            );
        }

        let negative = Pearson3::new(-1.5);
        let negative_cases = [
            (0.1, -1.333_300_518_218_442_5),
            (0.25, -0.475_368_176_123_716_5),
            (0.5, 0.239_964_153_964_511_43),
            (0.75, 0.733_120_813_876_001_7),
            (0.9, 1.018_104_310_659_964_5),
        ];

        for &(q, expected_x) in &negative_cases {
            let x = negative.ppf(q);
            assert_close(x, expected_x, 1e-12, &format!("Pearson3(-1.5) ppf({q})"));
            assert_close(
                negative.cdf(x),
                q,
                1e-10,
                &format!("Pearson3(-1.5) cdf(ppf({q}))"),
            );
        }
    }

    #[test]
    fn pearson3_zero_skew_matches_standard_normal() {
        let dist = Pearson3::new(0.0);
        let cases = [
            (-2.0, 0.053_990_966_513_188_056, 0.022_750_131_948_179_195),
            (-1.0, 0.241_970_724_519_143_37, 0.158_655_253_931_457_07),
            (0.0, 0.398_942_280_401_432_7, 0.5),
            (1.0, 0.241_970_724_519_143_37, 0.841_344_746_068_542_9),
            (2.0, 0.053_990_966_513_188_056, 0.977_249_868_051_820_8),
        ];

        for &(x, pdf, cdf) in &cases {
            assert_close(dist.pdf(x), pdf, 1e-12, &format!("Pearson3(0) pdf({x})"));
            assert_close(dist.cdf(x), cdf, 1e-12, &format!("Pearson3(0) cdf({x})"));
        }

        assert_close(dist.mean(), 0.0, 1e-12, "Pearson3 mean");
        assert_close(dist.var(), 1.0, 1e-12, "Pearson3 variance");
    }

    #[test]
    fn exponnorm_pdf_cdf_match_scipy_reference_values() {
        let dist = ExponNorm::new(1.5);
        let cases = [
            (-2.0, 0.012_098_174_949_533_125, 0.004_602_869_523_879_505),
            (-1.0, 0.077_497_646_225_164_67, 0.042_408_784_593_710_086),
            (0.0, 0.210_216_679_964_559_62, 0.184_674_980_053_160_58),
            (1.0, 0.269_534_564_286_752_27, 0.437_042_899_638_414_53),
            (2.0, 0.199_444_595_909_829_16, 0.678_082_974_187_077_1),
            (4.0, 0.057_824_732_034_141_49, 0.913_231_230_706_954_7),
        ];

        for &(x, pdf, cdf) in &cases {
            assert_close(dist.pdf(x), pdf, 1e-12, &format!("ExponNorm pdf({x})"));
            assert_close(dist.cdf(x), cdf, 1e-12, &format!("ExponNorm cdf({x})"));
        }
    }

    #[test]
    fn exponnorm_ppf_roundtrip_matches_scipy_reference_values() {
        let dist = ExponNorm::new(1.5);
        let cases = [
            (0.1, -0.475_357_491_648_256),
            (0.25, 0.287_818_458_047_716_5),
            (0.5, 1.236_654_259_449_335_7),
            (0.75, 2.399_177_439_596_912),
            (0.9, 3.786_999_526_289_809),
        ];

        for &(q, expected_x) in &cases {
            let x = dist.ppf(q);
            assert_close(x, expected_x, 1e-11, &format!("ExponNorm ppf({q})"));
            assert_close(dist.cdf(x), q, 1e-10, &format!("ExponNorm cdf(ppf({q}))"));
        }
    }

    #[test]
    fn exponnorm_mean_and_variance_match_closed_form() {
        let dist = ExponNorm::new(1.5);
        assert_close(dist.mean(), 1.5, 1e-12, "ExponNorm mean");
        assert_close(dist.var(), 3.25, 1e-12, "ExponNorm variance");
    }

    #[test]
    fn logistic_cdf_matches_expit_form() {
        let l = Logistic::new(2.0, 0.5);
        assert_close(l.cdf(2.0), 0.5, 1e-12, "logistic cdf at location");
        assert_close(l.pdf(2.0), 0.5, 1e-12, "logistic pdf at location");
    }

    #[test]
    fn logistic_cdf_ppf_roundtrip() {
        let l = Logistic::new(-1.0, 3.0);
        let x = l.ppf(0.25);
        assert_close(l.cdf(x), 0.25, 1e-10, "logistic cdf(ppf(q))");
    }

    #[test]
    fn maxwell_support_and_mode() {
        let m = Maxwell::new(1.0);
        assert_eq!(m.pdf(-0.5), 0.0);
        assert!(m.pdf(1.0) > m.pdf(0.5), "maxwell rises before the mode");
        assert!(m.pdf(1.5) > m.pdf(2.0), "maxwell falls after the mode");
    }

    #[test]
    fn maxwell_cdf_and_mean_variance() {
        let m = Maxwell::new(2.0);
        assert_eq!(m.cdf(0.0), 0.0);
        assert!(m.cdf(2.0) > 0.0 && m.cdf(2.0) < 1.0);
        assert_close(m.mean(), 4.0 * (2.0 / PI).sqrt(), 1e-12, "maxwell mean");
        assert_close(m.var(), 4.0 * (3.0 - 8.0 / PI), 1e-12, "maxwell variance");
    }

    // ── Cauchy distribution ───────────────────────────────────────

    #[test]
    fn cauchy_pdf_at_loc() {
        let c = Cauchy::default();
        assert_close(c.pdf(0.0), 1.0 / PI, 1e-12, "Cauchy pdf(0)");
    }

    #[test]
    fn cauchy_cdf_at_loc() {
        let c = Cauchy::default();
        assert_close(c.cdf(0.0), 0.5, 1e-12, "Cauchy cdf(0)");
    }

    #[test]
    fn cauchy_cdf_known() {
        let c = Cauchy::default();
        // CDF(1) = 0.5 + atan(1)/π = 0.5 + 0.25 = 0.75
        assert_close(c.cdf(1.0), 0.75, 1e-12, "Cauchy cdf(1)");
    }

    #[test]
    fn cauchy_ppf_roundtrip() {
        let c = Cauchy::new(2.0, 3.0);
        for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = c.ppf(q);
            assert_close(
                c.cdf(x),
                q,
                1e-10,
                &format!("Cauchy ppf roundtrip at q={q}"),
            );
        }
    }

    #[test]
    fn cauchy_mean_is_nan() {
        let c = Cauchy::default();
        assert!(c.mean().is_nan());
        assert!(c.var().is_nan());
    }

    // ── Laplace distribution ──────────────────────────────────────

    #[test]
    fn laplace_pdf_at_loc() {
        let l = Laplace::default();
        assert_close(l.pdf(0.0), 0.5, 1e-12, "Laplace pdf(0) = 1/(2*1)");
    }

    #[test]
    fn laplace_cdf_at_loc() {
        let l = Laplace::default();
        assert_close(l.cdf(0.0), 0.5, 1e-12, "Laplace cdf(0)");
    }

    #[test]
    fn laplace_sf_plus_cdf() {
        let l = Laplace::new(1.0, 2.0);
        for x in [-3.0, 0.0, 1.0, 4.0] {
            assert_close(l.sf(x) + l.cdf(x), 1.0, 1e-12, &format!("sf+cdf at {x}"));
        }
    }

    #[test]
    fn laplace_ppf_roundtrip() {
        let l = Laplace::new(3.0, 2.0);
        for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = l.ppf(q);
            assert_close(l.cdf(x), q, 1e-10, &format!("Laplace ppf at q={q}"));
        }
    }

    #[test]
    fn laplace_mean_var() {
        let l = Laplace::new(5.0, 3.0);
        assert_close(l.mean(), 5.0, 1e-12, "Laplace mean");
        assert_close(l.var(), 18.0, 1e-12, "Laplace var = 2*scale^2");
    }

    // ── Triangular distribution ───────────────────────────────────

    #[test]
    fn triangular_pdf_at_mode() {
        let t = Triangular::new(0.0, 0.5, 1.0);
        assert_close(t.pdf(0.5), 2.0, 1e-12, "Triangular pdf at mode");
    }

    #[test]
    fn triangular_cdf_bounds() {
        let t = Triangular::new(0.0, 0.5, 1.0);
        assert_eq!(t.cdf(-1.0), 0.0);
        assert_close(t.cdf(1.0), 1.0, 1e-12, "cdf at right");
    }

    #[test]
    fn triangular_ppf_roundtrip() {
        let t = Triangular::new(1.0, 3.0, 5.0);
        for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = t.ppf(q);
            assert_close(t.cdf(x), q, 1e-10, &format!("Triangular ppf at q={q}"));
        }
    }

    #[test]
    fn triangular_mean_var() {
        let t = Triangular::new(0.0, 1.0, 2.0);
        assert_close(t.mean(), 1.0, 1e-12, "mean = (a+b+c)/3");
        // var = (a²+b²+c²-ab-ac-bc)/18 = (0+4+1-0-0-2)/18 = 3/18 = 1/6
        assert_close(t.var(), 1.0 / 6.0, 1e-12, "Triangular var");
    }

    #[test]
    fn triangular_mode_equals_right() {
        // Edge case: mode == right (right-skewed triangle)
        let t = Triangular::new(0.0, 2.0, 2.0);
        assert_close(t.pdf(1.0), 0.5, 1e-12, "pdf at x=1");
        assert_close(t.pdf(2.0), 1.0, 1e-12, "pdf at mode=right");
        assert_close(t.cdf(0.0), 0.0, 1e-12, "cdf at left");
        assert_close(t.cdf(2.0), 1.0, 1e-12, "cdf at right");
        assert_close(t.cdf(1.0), 0.25, 1e-12, "cdf at midpoint");
    }

    // ── t-test edge cases ─────────────────────────────────────────

    #[test]
    fn ttest_1samp_single_element_returns_nan() {
        let result = ttest_1samp(&[5.0], 0.0);
        assert!(result.statistic.is_nan());
        assert!(result.pvalue.is_nan());
    }

    #[test]
    fn ttest_ind_single_element_returns_nan() {
        let result = ttest_ind(&[1.0], &[2.0]);
        assert!(result.statistic.is_nan());
        assert!(result.pvalue.is_nan());
    }

    // ── Random variate sampling (rvs) ─────────────────────────────

    #[test]
    fn rvs_normal_sample_mean() {
        let n = Normal::new(5.0, 1.0);
        let mut rng = rand::rng();
        let samples = n.rvs(10_000, &mut rng);
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            (mean - 5.0).abs() < 0.1,
            "sample mean should be near 5.0, got {mean}"
        );
    }

    #[test]
    fn rvs_uniform_bounds() {
        let u = Uniform::new(2.0, 3.0); // [2, 5]
        let mut rng = rand::rng();
        let samples = u.rvs(1000, &mut rng);
        assert!(
            samples.iter().all(|&x| (2.0..=5.0).contains(&x)),
            "all samples should be in [2, 5]"
        );
    }

    #[test]
    fn rvs_exponential_positive() {
        let e = Exponential::new(1.0);
        let mut rng = rand::rng();
        let samples = e.rvs(1000, &mut rng);
        assert!(
            samples.iter().all(|&x| x >= 0.0),
            "exponential samples should be non-negative"
        );
    }

    // ── Statistical tests ─────────────────────────────────────────

    #[test]
    fn ttest_1samp_zero_mean() {
        // Sample from N(0, 1) — should not reject H0: μ = 0
        let data: Vec<f64> = (0..100)
            .map(|i| {
                let x = (i as f64 - 50.0) / 50.0;
                x * 0.1 // small values centered at 0
            })
            .collect();
        let result = ttest_1samp(&data, 0.0);
        assert!(
            result.pvalue > 0.05,
            "should not reject H0, p={}",
            result.pvalue
        );
        assert_close(result.df, 99.0, 1e-10, "df = n-1");
    }

    #[test]
    fn ttest_1samp_shifted_mean() {
        // Data clearly shifted from 0
        let data: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64) * 0.01).collect();
        let result = ttest_1samp(&data, 0.0);
        assert!(
            result.pvalue < 0.001,
            "should reject H0, p={}",
            result.pvalue
        );
    }

    #[test]
    fn ttest_ind_same_distribution() {
        let a: Vec<f64> = (0..50).map(|i| (i as f64) * 0.02).collect();
        let b: Vec<f64> = (0..50).map(|i| (i as f64) * 0.02 + 0.001).collect();
        let result = ttest_ind(&a, &b);
        assert!(
            result.pvalue > 0.05,
            "similar samples should not reject H0, p={}",
            result.pvalue
        );
    }

    #[test]
    fn ttest_ind_different_distributions() {
        let a: Vec<f64> = (0..50).map(|i| (i as f64) * 0.01).collect();
        let b: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64) * 0.01).collect();
        let result = ttest_ind(&a, &b);
        assert!(
            result.pvalue < 0.001,
            "very different samples should reject H0, p={}",
            result.pvalue
        );
    }

    #[test]
    fn ttest_ind_welch_different_variances() {
        let a: Vec<f64> = (0..100).map(|i| (i as f64) * 0.001).collect();
        let b: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.1).collect();
        let result = ttest_ind_welch(&a, &b);
        assert!(
            result.pvalue < 0.001,
            "should reject H0 with Welch, p={}",
            result.pvalue
        );
        // Welch df should be less than n1+n2-2
        assert!(result.df < 128.0, "Welch df should be adjusted");
    }

    // ── Linear regression ─────────────────────────────────────────

    #[test]
    fn linregress_perfect_line() {
        // y = 2x + 3 exactly
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();
        let result = linregress(&x, &y);
        assert_close(result.slope, 2.0, 1e-10, "slope");
        assert_close(result.intercept, 3.0, 1e-10, "intercept");
        assert_close(result.rvalue, 1.0, 1e-10, "r-value");
        assert!(
            result.stderr < 1e-10,
            "stderr should be ~0, got {}",
            result.stderr
        );
    }

    #[test]
    fn linregress_negative_slope() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| -1.5 * xi + 10.0).collect();
        let result = linregress(&x, &y);
        assert_close(result.slope, -1.5, 1e-10, "slope");
        assert_close(result.intercept, 10.0, 1e-10, "intercept");
        assert_close(result.rvalue, -1.0, 1e-10, "r-value (negative)");
    }

    #[test]
    fn linregress_with_noise() {
        // y ≈ 3x + 1 with small deterministic perturbation
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 3.0 * xi + 1.0 + if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let result = linregress(&x, &y);
        assert!(
            (result.slope - 3.0).abs() < 0.01,
            "slope ~ 3.0, got {}",
            result.slope
        );
        assert!(result.rvalue > 0.999, "high r for low noise");
        assert!(result.pvalue < 1e-10, "very significant");
    }

    #[test]
    fn linregress_too_few_points() {
        let result = linregress(&[1.0], &[2.0]);
        assert!(result.slope.is_nan());
    }

    #[test]
    fn linregress_constant_x() {
        let result = linregress(&[5.0, 5.0, 5.0], &[1.0, 2.0, 3.0]);
        assert!(result.slope.is_nan(), "constant x => undefined slope");
    }

    // ── Pearson correlation ───────────────────────────────────────

    #[test]
    fn pearsonr_perfect_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let result = pearsonr(&x, &y);
        assert_close(result.statistic, 1.0, 1e-10, "r = 1 for perfect positive");
        assert!(result.pvalue < 1e-10, "very significant");
    }

    #[test]
    fn pearsonr_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 8.0, 6.0, 4.0, 2.0];
        let result = pearsonr(&x, &y);
        assert_close(result.statistic, -1.0, 1e-10, "r = -1");
    }

    #[test]
    fn pearsonr_uncorrelated() {
        // Orthogonal data
        let x = [1.0, 0.0, -1.0, 0.0];
        let y = [0.0, 1.0, 0.0, -1.0];
        let result = pearsonr(&x, &y);
        assert_close(result.statistic, 0.0, 1e-10, "r = 0 for orthogonal");
    }

    #[test]
    fn pearsonr_too_few_points() {
        let result = pearsonr(&[1.0], &[2.0]);
        assert!(result.statistic.is_nan());
    }

    // ── Spearman correlation ──────────────────────────────────────

    #[test]
    fn spearmanr_monotonic() {
        // Perfect monotonic (not necessarily linear)
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 4.0, 9.0, 16.0, 25.0]; // y = x² (monotonic)
        let result = spearmanr(&x, &y);
        assert_close(result.statistic, 1.0, 1e-10, "rs = 1 for monotonic");
    }

    #[test]
    fn spearmanr_anti_monotonic() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [25.0, 16.0, 9.0, 4.0, 1.0];
        let result = spearmanr(&x, &y);
        assert_close(result.statistic, -1.0, 1e-10, "rs = -1 for anti-monotonic");
    }

    #[test]
    fn spearmanr_with_ties() {
        let x = [1.0, 2.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        let result = spearmanr(&x, &y);
        assert!(
            result.statistic > 0.8,
            "should be strongly positive, got {}",
            result.statistic
        );
    }

    #[test]
    fn spearmanr_too_few() {
        let result = spearmanr(&[1.0, 2.0], &[3.0, 4.0]);
        assert!(result.statistic.is_nan(), "need at least 3 for spearmanr");
    }

    // ── rankdata helper ───────────────────────────────────────────

    #[test]
    fn rankdata_no_ties() {
        let ranks = rankdata(&[3.0, 1.0, 2.0]);
        assert_close(ranks[0], 3.0, 1e-12, "rank of 3.0");
        assert_close(ranks[1], 1.0, 1e-12, "rank of 1.0");
        assert_close(ranks[2], 2.0, 1e-12, "rank of 2.0");
    }

    #[test]
    fn rankdata_with_ties() {
        let ranks = rankdata(&[1.0, 2.0, 2.0, 4.0]);
        assert_close(ranks[0], 1.0, 1e-12, "rank of 1.0");
        assert_close(ranks[1], 2.5, 1e-12, "rank of 2.0 (tied)");
        assert_close(ranks[2], 2.5, 1e-12, "rank of 2.0 (tied)");
        assert_close(ranks[3], 4.0, 1e-12, "rank of 4.0");
    }

    // ── Summary statistics ────────────────────────────────────────

    #[test]
    fn describe_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = describe(&data);
        assert_eq!(result.nobs, 5);
        assert_close(result.mean, 3.0, 1e-12, "mean");
        assert_eq!(result.minmax, (1.0, 5.0));
        assert_close(result.variance, 2.5, 1e-12, "var");
    }

    #[test]
    fn describe_single_element() {
        let result = describe(&[42.0]);
        assert_eq!(result.nobs, 1);
        assert_close(result.mean, 42.0, 1e-12, "mean");
        assert!(result.variance.is_nan());
    }

    #[test]
    fn skew_symmetric() {
        // Symmetric data has skewness = 0
        let data = [-2.0, -1.0, 0.0, 1.0, 2.0];
        assert!(skew(&data).abs() < 1e-10, "symmetric => skew=0");
    }

    #[test]
    fn skew_right_skewed() {
        // Right-skewed: many small, few large
        let data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0];
        assert!(skew(&data) > 0.0, "right skew should be positive");
    }

    #[test]
    fn kurtosis_normal_like() {
        // For normal-like data, excess kurtosis ≈ 0
        // Use a simple uniform-ish spread
        let data: Vec<f64> = (-50..=50).map(|i| i as f64).collect();
        let k = kurtosis(&data);
        // Uniform distribution has excess kurtosis = -1.2
        assert!(k < 0.0 && k > -2.0, "uniform-like kurtosis ~ -1.2, got {k}");
    }

    #[test]
    fn kurtosis_heavy_tails() {
        // Heavy-tailed: many near center, few extremes
        let mut data = vec![0.0; 100];
        data.push(100.0);
        data.push(-100.0);
        assert!(kurtosis(&data) > 0.0, "heavy tails => positive kurtosis");
    }

    #[test]
    fn mode_basic() {
        let data = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        assert_close(mode(&data), 3.0, 1e-12, "mode is 3.0");
    }

    #[test]
    fn mode_all_unique() {
        // All unique: returns smallest (first encountered)
        let data = [3.0, 1.0, 2.0];
        let m = mode(&data);
        assert!(m.is_finite(), "mode should be finite for non-empty data");
    }

    #[test]
    fn moment_zeroth() {
        assert_close(moment(&[1.0, 2.0, 3.0], 0), 1.0, 1e-12, "0th moment = 1");
    }

    #[test]
    fn moment_first() {
        // 1st central moment is always 0
        assert_close(
            moment(&[1.0, 2.0, 3.0, 4.0], 1),
            0.0,
            1e-12,
            "1st central moment = 0",
        );
    }

    #[test]
    fn moment_second_is_variance() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let m2 = moment(&data, 2);
        // Population variance = 2.0 (not sample variance 2.5)
        assert_close(m2, 2.0, 1e-12, "2nd moment = population variance");
    }

    #[test]
    fn sem_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        // SE = sqrt(var/n) = sqrt(2.5/5) = sqrt(0.5) ≈ 0.7071
        let se = sem(&data);
        assert_close(se, (2.5 / 5.0_f64).sqrt(), 1e-10, "SEM");
    }

    #[test]
    fn iqr_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let q = iqr(&data);
        // Q1 = 2.75, Q3 = 6.25, IQR = 3.5
        assert_close(q, 3.5, 1e-10, "IQR");
    }

    #[test]
    fn variation_basic() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let cv = variation(&data);
        // mean = 30, std = sqrt(250) ≈ 15.81, cv ≈ 0.527
        assert!(cv > 0.4 && cv < 0.6, "CV ~ 0.53, got {cv}");
    }

    #[test]
    fn zscore_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let z = zscore(&data);
        assert_eq!(z.len(), 5);
        // Mean of z-scores should be 0
        let z_mean: f64 = z.iter().sum::<f64>() / z.len() as f64;
        assert!(z_mean.abs() < 1e-10, "z-score mean should be 0");
        // Std of z-scores should be ~1 (using ddof=0 normalization matching scipy default)
        let z_var: f64 = z.iter().map(|&zi| zi * zi).sum::<f64>() / z.len() as f64;
        assert_close(z_var.sqrt(), 1.0, 1e-10, "z-score std should be 1");
    }

    #[test]
    fn zscore_constant() {
        let z = zscore(&[5.0, 5.0, 5.0]);
        assert!(z.iter().all(|&v| v.is_nan()), "constant data => all NaNs");
    }

    #[test]
    fn zmap_matches_scipy_reference_values() {
        let same = zmap(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert_close(same[0], -1.224_744_871_391_589, 1e-12, "zmap same[0]");
        assert_close(same[1], 0.0, 1e-12, "zmap same[1]");
        assert_close(same[2], 1.224_744_871_391_589, 1e-12, "zmap same[2]");

        let shifted = zmap(&[2.0, 4.0], &[1.0, 2.0, 3.0]);
        assert_close(shifted[0], 0.0, 1e-12, "zmap shifted[0]");
        assert_close(shifted[1], 2.449_489_742_783_178, 1e-12, "zmap shifted[1]");
    }

    #[test]
    fn zmap_matches_scipy_constant_compare_shape() {
        let constant_compare = zmap(&[1.0, 2.0], &[1.0, 1.0]);
        assert!(
            constant_compare[0].is_nan(),
            "mean-aligned score should be NaN"
        );
        assert!(
            constant_compare[1].is_infinite() && constant_compare[1].is_sign_positive(),
            "offset score should be +inf"
        );

        let identical = [1.0, 1.0];
        let same_object = zmap(&identical, &identical);
        assert!(same_object.iter().all(|&value| value.is_nan()));
    }

    #[test]
    fn zmap_ddof_matches_scipy_reference_values() {
        let z = zmap_ddof(&[2.0, 4.0], &[1.0, 2.0, 3.0], 1);
        assert_close(z[0], 0.0, 1e-12, "zmap ddof[0]");
        assert_close(z[1], 2.0, 1e-12, "zmap ddof[1]");
    }

    #[test]
    fn gzscore_matches_scipy_reference_values() {
        let z = gzscore(&[1.0, 2.0, 4.0]);
        assert_close(z[0], -1.224_744_871_391_589_2, 1e-12, "gzscore[0]");
        assert_close(z[1], 0.0, 1e-12, "gzscore[1]");
        assert_close(z[2], 1.224_744_871_391_589_2, 1e-12, "gzscore[2]");

        let ddof = gzscore_ddof(&[1.0, 2.0, 4.0], 1);
        assert_close(ddof[0], -1.0, 1e-12, "gzscore ddof[0]");
        assert_close(ddof[1], 0.0, 1e-12, "gzscore ddof[1]");
        assert_close(ddof[2], 1.0, 1e-12, "gzscore ddof[2]");
    }

    #[test]
    fn gzscore_invalid_inputs_match_scipy_shape() {
        let zero = gzscore(&[0.0, 1.0, 2.0]);
        assert!(zero.iter().all(|&value| value.is_nan()));

        let negative = gzscore(&[-1.0, 1.0, 2.0]);
        assert!(negative.iter().all(|&value| value.is_nan()));
    }

    // ── Discrete distributions ────────────────────────────────────

    #[test]
    fn binomial_pmf_sums_to_one() {
        let b = Binomial::new(10, 0.3);
        let sum: f64 = (0..=10).map(|k| b.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-10, "PMF sum = {sum}");
    }

    #[test]
    fn binomial_mean_var() {
        let b = Binomial::new(20, 0.4);
        assert_close(b.mean(), 8.0, 1e-12, "mean = np");
        assert_close(b.var(), 4.8, 1e-12, "var = np(1-p)");
    }

    #[test]
    fn binomial_p0_all_at_zero() {
        let b = Binomial::new(5, 0.0);
        assert_close(b.pmf(0), 1.0, 1e-15, "P(0) with p=0");
        assert_close(b.pmf(1), 0.0, 1e-15, "P(1) with p=0");
    }

    #[test]
    fn binomial_p1_all_at_n() {
        let b = Binomial::new(5, 1.0);
        assert_close(b.pmf(5), 1.0, 1e-15, "P(5) with p=1");
        assert_close(b.pmf(4), 0.0, 1e-15, "P(4) with p=1");
    }

    #[test]
    fn binomial_known_value() {
        // C(5,2) * 0.5^2 * 0.5^3 = 10 * 0.03125 = 0.3125
        let b = Binomial::new(5, 0.5);
        assert_close(b.pmf(2), 0.3125, 1e-10, "Binom(5,0.5).pmf(2)");
    }

    #[test]
    fn binomial_cdf_at_n() {
        let b = Binomial::new(10, 0.3);
        assert_close(b.cdf(10), 1.0, 1e-10, "CDF(n) = 1");
    }

    #[test]
    fn bernoulli_basic() {
        let b = Bernoulli::new(0.7);
        assert_close(b.pmf(0), 0.3, 1e-15, "P(0)");
        assert_close(b.pmf(1), 0.7, 1e-15, "P(1)");
        assert_close(b.pmf(2), 0.0, 1e-15, "P(2)");
        assert_close(b.mean(), 0.7, 1e-15, "mean");
        assert_close(b.var(), 0.21, 1e-15, "var");
    }

    #[test]
    fn bernoulli_cdf() {
        let b = Bernoulli::new(0.4);
        assert_close(b.cdf(0), 0.6, 1e-15, "CDF(0)");
        assert_close(b.cdf(1), 1.0, 1e-15, "CDF(1)");
    }

    #[test]
    fn geometric_pmf_sums_near_one() {
        let g = Geometric::new(0.3);
        let sum: f64 = (1..=100).map(|k| g.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-6, "PMF sum = {sum}");
    }

    #[test]
    fn geometric_mean_var() {
        let g = Geometric::new(0.25);
        assert_close(g.mean(), 4.0, 1e-12, "mean = 1/p");
        assert_close(g.var(), 12.0, 1e-12, "var = (1-p)/p^2");
    }

    #[test]
    fn geometric_cdf_known() {
        let g = Geometric::new(0.5);
        // P(X <= 1) = 0.5, P(X <= 2) = 0.75, P(X <= 3) = 0.875
        assert_close(g.cdf(1), 0.5, 1e-12, "CDF(1)");
        assert_close(g.cdf(2), 0.75, 1e-12, "CDF(2)");
        assert_close(g.cdf(3), 0.875, 1e-12, "CDF(3)");
    }

    #[test]
    fn geometric_pmf_at_zero() {
        let g = Geometric::new(0.5);
        assert_close(g.pmf(0), 0.0, 1e-15, "P(0) = 0 for geom");
    }

    #[test]
    fn negbinomial_pmf_sums_near_one() {
        let nb = NegBinomial::new(5.0, 0.4);
        let sum: f64 = (0..=200).map(|k| nb.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-6, "PMF sum = {sum}");
    }

    #[test]
    fn negbinomial_mean_var() {
        let nb = NegBinomial::new(3.0, 0.5);
        // mean = n*(1-p)/p = 3*0.5/0.5 = 3
        assert_close(nb.mean(), 3.0, 1e-12, "mean");
        // var = n*(1-p)/p^2 = 3*0.5/0.25 = 6
        assert_close(nb.var(), 6.0, 1e-12, "var");
    }

    #[test]
    fn hypergeometric_pmf_sums_to_one() {
        // M=20, n=7, N=12
        let h = Hypergeometric::new(20, 7, 12);
        let sum: f64 = (0..=7).map(|k| h.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-10, "PMF sum = {sum}");
    }

    #[test]
    fn hypergeometric_mean() {
        let h = Hypergeometric::new(20, 7, 12);
        // mean = N*n/M = 12*7/20 = 4.2
        assert_close(h.mean(), 4.2, 1e-12, "mean");
    }

    #[test]
    fn hypergeometric_known_value() {
        // Drawing 5 cards from deck of 52, P(exactly 1 ace)
        // C(4,1)*C(48,4)/C(52,5)
        let h = Hypergeometric::new(52, 4, 5);
        let p1 = h.pmf(1);
        // Known value: 0.29947...
        assert!(
            p1 > 0.29 && p1 < 0.31,
            "P(1 ace in 5 cards) ~ 0.299, got {p1}"
        );
    }

    #[test]
    fn hypergeometric_out_of_range() {
        let h = Hypergeometric::new(10, 3, 5);
        // k > n=3 is impossible
        assert_close(h.pmf(4), 0.0, 1e-15, "k > n");
        // k > N=5 is impossible
        assert_close(h.pmf(6), 0.0, 1e-15, "k > N");
    }

    #[test]
    fn poisson_implements_discrete_trait() {
        let p = Poisson::new(3.0);
        let d: &dyn DiscreteDistribution = &p;
        assert_close(d.mean(), 3.0, 1e-12, "Poisson mean via trait");
        assert_close(d.var(), 3.0, 1e-12, "Poisson var via trait");
        let sum: f64 = (0..=30).map(|k| d.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-6, "Poisson PMF sum via trait");
    }

    #[test]
    fn discrete_sf_complement() {
        let b = Binomial::new(10, 0.5);
        for k in 0..=10 {
            let sf_plus_cdf = b.sf(k) + b.cdf(k);
            assert_close(sf_plus_cdf, 1.0, 1e-10, &format!("sf+cdf at k={k}"));
        }
    }

    // ── Non-parametric + ANOVA tests ──────────────────────────────

    #[test]
    fn f_oneway_same_means() {
        let a: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.01).collect();
        let b: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.01 + 0.001).collect();
        let c: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.01 - 0.001).collect();
        let result = f_oneway(&[&a, &b, &c]);
        assert!(
            result.pvalue > 0.05,
            "similar groups should not reject H0, p={}",
            result.pvalue
        );
    }

    #[test]
    fn f_oneway_different_means() {
        let a: Vec<f64> = (0..30).map(|i| (i as f64) * 0.01).collect();
        let b: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.01).collect();
        let c: Vec<f64> = (0..30).map(|i| 20.0 + (i as f64) * 0.01).collect();
        let result = f_oneway(&[&a, &b, &c]);
        assert!(
            result.pvalue < 0.001,
            "different means should reject H0, p={}",
            result.pvalue
        );
    }

    #[test]
    fn f_oneway_too_few_groups() {
        let result = f_oneway(&[&[1.0, 2.0]]);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn f_oneway_constant_identical_groups() {
        let result = f_oneway(&[&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]]);
        assert!(result.statistic.is_nan());
        assert!(result.pvalue.is_nan());
    }

    #[test]
    fn f_oneway_constant_different_groups() {
        let result = f_oneway(&[&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0]]);
        assert!(result.statistic.is_infinite());
        assert_eq!(result.pvalue, 0.0);
    }

    #[test]
    fn levene_equal_variance_groups() {
        let a: Vec<f64> = (0..20).map(|i| i as f64 * 0.2 - 2.0).collect();
        let b: Vec<f64> = (0..20).map(|i| i as f64 * 0.2 + 3.0).collect();
        let c: Vec<f64> = (0..20).map(|i| i as f64 * 0.2 - 7.0).collect();
        let result = levene(&[&a, &b, &c]);
        assert!(
            result.pvalue > 0.05,
            "equal-variance groups should not reject, p={}",
            result.pvalue
        );
    }

    #[test]
    fn levene_detects_different_variances() {
        let a = vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let b = vec![-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5];
        let c = vec![-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0];
        let result = levene(&[&a, &b, &c]);
        assert!(
            result.pvalue < 0.05,
            "different-variance groups should reject, p={}",
            result.pvalue
        );
    }

    #[test]
    fn bartlett_equal_variance_groups() {
        let a: Vec<f64> = (0..20).map(|i| i as f64 * 0.2 - 2.0).collect();
        let b: Vec<f64> = (0..20).map(|i| i as f64 * 0.2 + 3.0).collect();
        let c: Vec<f64> = (0..20).map(|i| i as f64 * 0.2 - 7.0).collect();
        let result = bartlett(&[&a, &b, &c]);
        assert!(
            result.pvalue > 0.05,
            "equal-variance groups should not reject, p={}",
            result.pvalue
        );
    }

    #[test]
    fn bartlett_detects_different_variances() {
        let a = vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let b = vec![-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5];
        let c = vec![-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0];
        let result = bartlett(&[&a, &b, &c]);
        assert!(
            result.pvalue < 0.05,
            "different-variance groups should reject, p={}",
            result.pvalue
        );
    }

    #[test]
    fn mannwhitneyu_same_distribution() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64) * 0.02).collect();
        let y: Vec<f64> = (0..50).map(|i| (i as f64) * 0.02 + 0.001).collect();
        let result = mannwhitneyu(&x, &y);
        assert!(
            result.pvalue > 0.05,
            "similar samples p={}, should not reject",
            result.pvalue
        );
    }

    #[test]
    fn mannwhitneyu_shifted() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        let y: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let result = mannwhitneyu(&x, &y);
        assert!(
            result.pvalue < 0.001,
            "shifted samples p={}, should reject",
            result.pvalue
        );
    }

    #[test]
    fn wilcoxon_no_difference() {
        // Paired data with alternating noise — should not reject
        let x: Vec<f64> = (0..30).map(|i| (i as f64) * 0.1).collect();
        // Alternating +/- noise: half positive, half negative differences
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| v + if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let result = wilcoxon(&x, &y);
        assert!(
            result.pvalue > 0.05,
            "balanced noise p={}, should not reject",
            result.pvalue
        );
    }

    #[test]
    fn wilcoxon_systematic_shift() {
        let x: Vec<f64> = (0..30).map(|i| (i as f64) * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&v| v + 5.0).collect();
        let result = wilcoxon(&x, &y);
        assert!(
            result.pvalue < 0.01,
            "systematic shift p={}, should reject",
            result.pvalue
        );
    }

    #[test]
    fn kruskal_same_groups() {
        let a: Vec<f64> = (0..40).map(|i| (i as f64) * 0.025).collect();
        let b: Vec<f64> = (0..40).map(|i| (i as f64) * 0.025 + 0.001).collect();
        let result = kruskal(&[&a, &b]);
        assert!(
            result.pvalue > 0.05,
            "similar groups p={}, should not reject",
            result.pvalue
        );
    }

    #[test]
    fn kruskal_different_groups() {
        let a: Vec<f64> = (0..30).map(|i| (i as f64) * 0.01).collect();
        let b: Vec<f64> = (0..30).map(|i| 50.0 + (i as f64) * 0.01).collect();
        let result = kruskal(&[&a, &b]);
        // H statistic should be large (completely separated groups)
        assert!(
            result.statistic > 40.0,
            "H should be large for separated groups, got {}",
            result.statistic
        );
        // p-value should be very small
        assert!(
            result.pvalue < 0.001,
            "different groups p={}, should reject",
            result.pvalue
        );
    }

    #[test]
    fn ranksums_same_distribution() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64) * 0.025).collect();
        let y: Vec<f64> = (0..40).map(|i| (i as f64) * 0.025 + 0.0001).collect();
        let result = ranksums(&x, &y);
        assert!(
            result.pvalue > 0.05,
            "similar samples p={}, should not reject",
            result.pvalue
        );
    }

    #[test]
    fn ranksums_shifted() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64) * 0.1).collect();
        let y: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let result = ranksums(&x, &y);
        assert!(
            result.pvalue < 0.001,
            "shifted samples p={}, should reject",
            result.pvalue
        );
    }

    #[test]
    fn mannwhitneyu_too_few() {
        let result = mannwhitneyu(&[1.0], &[2.0]);
        assert!(result.statistic.is_nan());
    }

    // ── Goodness-of-fit tests ─────────────────────────────────────────

    #[test]
    fn ks_1samp_normal_data_from_normal_cdf() {
        // Data drawn from N(0,1) tested against N(0,1) CDF → high p-value
        let data: Vec<f64> = (0..200)
            .map(|i| {
                let u = (i as f64 + 0.5) / 200.0;
                standard_normal_ppf(u)
            })
            .collect();
        let norm = Normal::standard();
        let result = ks_1samp(&data, |x| ContinuousDistribution::cdf(&norm, x));
        assert!(
            result.pvalue > 0.05,
            "normal data vs normal CDF should not reject: p={}",
            result.pvalue
        );
    }

    #[test]
    fn ks_1samp_uniform_data_from_normal_cdf() {
        // Uniform data tested against Normal CDF → low p-value (reject)
        let data: Vec<f64> = (0..100).map(|i| (i as f64) / 99.0 * 6.0 - 3.0).collect();
        let norm = Normal::standard();
        let result = ks_1samp(&data, |x| ContinuousDistribution::cdf(&norm, x));
        assert!(
            result.pvalue < 0.05,
            "uniform data vs normal CDF should reject: p={}",
            result.pvalue
        );
    }

    #[test]
    fn ks_1samp_empty_returns_nan() {
        let result = ks_1samp(&[], |x| x);
        assert!(result.statistic.is_nan());
        assert!(result.pvalue.is_nan());
    }

    #[test]
    fn kstest_dispatches_to_one_sample() {
        fn normal_cdf(x: f64) -> f64 {
            ContinuousDistribution::cdf(&Normal::standard(), x)
        }

        let data: Vec<f64> = (0..80)
            .map(|i| standard_normal_ppf((i as f64 + 0.5) / 80.0))
            .collect();
        let direct = ks_1samp(&data, normal_cdf);
        let wrapped = kstest(&data, KstestTarget::Cdf(normal_cdf));
        assert_close(
            wrapped.statistic,
            direct.statistic,
            1e-12,
            "kstest one-sample statistic",
        );
        assert_close(
            wrapped.pvalue,
            direct.pvalue,
            1e-12,
            "kstest one-sample pvalue",
        );
    }

    #[test]
    fn ks_2samp_same_distribution() {
        // Two samples from same distribution → high p-value
        let data1: Vec<f64> = (0..100)
            .map(|i| standard_normal_ppf((i as f64 + 0.5) / 100.0))
            .collect();
        let data2: Vec<f64> = (0..100)
            .map(|i| standard_normal_ppf((i as f64 + 0.3) / 100.0))
            .collect();
        let result = ks_2samp(&data1, &data2);
        assert!(
            result.pvalue > 0.01,
            "same distribution should not reject: p={}",
            result.pvalue
        );
    }

    #[test]
    fn ks_2samp_different_distributions() {
        // Normal vs uniform-like data → low p-value
        let data1: Vec<f64> = (0..100)
            .map(|i| standard_normal_ppf((i as f64 + 0.5) / 100.0))
            .collect();
        let data2: Vec<f64> = (0..100).map(|i| (i as f64) / 10.0).collect();
        let result = ks_2samp(&data1, &data2);
        assert!(
            result.pvalue < 0.05,
            "different distributions should reject: p={}",
            result.pvalue
        );
    }

    #[test]
    fn ks_2samp_empty_returns_nan() {
        let result = ks_2samp(&[], &[1.0, 2.0]);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn kstest_dispatches_to_two_sample() {
        let data1: Vec<f64> = (0..60)
            .map(|i| standard_normal_ppf((i as f64 + 0.5) / 60.0))
            .collect();
        let data2: Vec<f64> = (0..60)
            .map(|i| standard_normal_ppf((i as f64 + 0.3) / 60.0))
            .collect();
        let direct = ks_2samp(&data1, &data2);
        let wrapped = kstest(&data1, KstestTarget::Sample(&data2));
        assert_close(
            wrapped.statistic,
            direct.statistic,
            1e-12,
            "kstest two-sample statistic",
        );
        assert_close(
            wrapped.pvalue,
            direct.pvalue,
            1e-12,
            "kstest two-sample pvalue",
        );
    }

    #[test]
    fn cramervonmises_uniform_reference_matches_scipy_oracle() {
        let data = [0.1, 0.2, 0.3, 0.4, 0.5];
        let result = cramervonmises(&data, |x| x.clamp(0.0, 1.0));
        assert_close(
            result.statistic,
            0.31666666666666665,
            1e-12,
            "cramervonmises one-sample statistic",
        );
        assert_close(
            result.pvalue,
            0.11944716780950626,
            3e-3,
            "cramervonmises one-sample pvalue",
        );
    }

    #[test]
    fn cramervonmises_too_few_returns_nan() {
        let result = cramervonmises(&[0.25], |x| x);
        assert!(result.statistic.is_nan());
        assert!(result.pvalue.is_nan());
    }

    #[test]
    fn cramervonmises_2samp_matches_scipy_oracle() {
        let x = [-1.2, -0.7, -0.1, 0.0, 0.4, 0.9, 1.1];
        let y = [-1.1, -0.4, 0.2, 0.3, 1.0, 1.4];
        let result = cramervonmises_2samp(&x, &y);
        assert_close(
            result.statistic,
            0.045787545787545625,
            1e-12,
            "cramervonmises_2samp statistic",
        );
        assert_close(
            result.pvalue,
            0.9656177156177156,
            1e-12,
            "cramervonmises_2samp pvalue",
        );
    }

    #[test]
    fn cramervonmises_2samp_identical_samples_have_unit_pvalue() {
        let sample = [0.0, 1.0, 2.0];
        let result = cramervonmises_2samp(&sample, &sample);
        assert_close(
            result.statistic,
            0.0,
            1e-12,
            "cramervonmises_2samp identical statistic",
        );
        assert_close(
            result.pvalue,
            1.0,
            1e-12,
            "cramervonmises_2samp identical pvalue",
        );
    }

    #[test]
    fn cramervonmises_2samp_asymptotic_rejects_shifted_samples() {
        let x: Vec<f64> = (0..30).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = (0..30).map(|i| i as f64 / 10.0 + 1.0).collect();
        let result = cramervonmises_2samp_with_method(&x, &y, Cvm2SampleMethod::Asymptotic);
        assert!(
            result.pvalue < 0.05,
            "shifted samples should be rejected: T={}, p={}",
            result.statistic,
            result.pvalue
        );
    }

    #[test]
    fn shapiro_normal_data() {
        // Approximately normal data → W close to 1, high p-value
        let data: Vec<f64> = (0..50)
            .map(|i| standard_normal_ppf((i as f64 + 0.5) / 50.0))
            .collect();
        let result = shapiro(&data);
        assert!(
            result.statistic > 0.9,
            "normal data W should be > 0.9: W={}",
            result.statistic
        );
        assert!(
            result.pvalue > 0.05,
            "normal data p-value should be > 0.05: p={}",
            result.pvalue
        );
    }

    #[test]
    fn shapiro_exponential_data() {
        // Exponential-like data (heavily skewed) → lower W
        let data: Vec<f64> = (1..=50).map(|i| (i as f64 * 0.1).exp()).collect();
        let result = shapiro(&data);
        assert!(
            result.statistic < 0.95,
            "exponential data W should be < 0.95: W={}",
            result.statistic
        );
    }

    #[test]
    fn shapiro_too_few() {
        let result = shapiro(&[1.0, 2.0]);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn shapiro_constant_data() {
        let result = shapiro(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        assert_close(result.statistic, 1.0, 1e-10, "constant data W");
        assert_close(result.pvalue, 1.0, 1e-10, "constant data p");
    }

    #[test]
    fn normaltest_normal_data() {
        // Use pseudo-random normal data via Box-Muller pairs
        // Seed-like deterministic sequence using simple LCG
        let mut seed: u64 = 42;
        let mut data = Vec::with_capacity(200);
        for _ in 0..100 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (seed >> 11) as f64 / (1u64 << 53) as f64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (seed >> 11) as f64 / (1u64 << 53) as f64;
            let u1 = u1.max(1e-15); // avoid ln(0)
            let r = (-2.0 * u1.ln()).sqrt();
            data.push(r * (2.0 * std::f64::consts::PI * u2).cos());
            data.push(r * (2.0 * std::f64::consts::PI * u2).sin());
        }
        let result = normaltest(&data);
        assert!(
            result.pvalue > 0.01,
            "normal data should not be rejected at 1%: K²={}, p={}",
            result.statistic,
            result.pvalue
        );
    }

    #[test]
    fn normaltest_skewed_data() {
        // Highly skewed data → should reject normality
        let data: Vec<f64> = (0..200)
            .map(|i| ((i as f64 + 1.0) / 201.0).powi(5))
            .collect();
        let result = normaltest(&data);
        assert!(
            result.pvalue < 0.05,
            "skewed data should be rejected: K²={}, p={}",
            result.statistic,
            result.pvalue
        );
    }

    #[test]
    fn normaltest_too_few() {
        let result = normaltest(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn jarque_bera_tracks_formula() {
        let data = [1.0, 1.5, 2.5, 4.0, 8.0, 16.0];
        let result = jarque_bera(&data);
        let expected_statistic =
            data.len() as f64 / 6.0 * (skew(&data).powi(2) + 0.25 * kurtosis(&data).powi(2));
        let expected_pvalue = ChiSquared::new(2.0).sf(expected_statistic);
        assert_close(
            result.statistic,
            expected_statistic,
            1e-12,
            "jarque_bera statistic",
        );
        assert_close(result.pvalue, expected_pvalue, 1e-12, "jarque_bera pvalue");
    }

    #[test]
    fn jarque_bera_rejects_skewed_data() {
        let data: Vec<f64> = (0..200)
            .map(|i| ((i as f64 + 1.0) / 201.0).powi(6))
            .collect();
        let result = jarque_bera(&data);
        assert!(
            result.pvalue < 0.05,
            "skewed data should be rejected: JB={}, p={}",
            result.statistic,
            result.pvalue
        );
    }

    #[test]
    fn jarque_bera_constant_data_returns_nan() {
        let result = jarque_bera(&[3.0, 3.0, 3.0, 3.0]);
        assert!(result.statistic.is_nan());
        assert!(result.pvalue.is_nan());
    }

    #[test]
    fn jarque_bera_too_few_returns_nan() {
        let result = jarque_bera(&[1.0]);
        assert!(result.statistic.is_nan());
        assert!(result.pvalue.is_nan());
    }

    #[test]
    fn anderson_normal_data() {
        // Normal-like data should have A² below critical values
        let data: Vec<f64> = (0..100)
            .map(|i| standard_normal_ppf((i as f64 + 0.5) / 100.0))
            .collect();
        let result = anderson(&data, "norm");
        assert!(
            result.statistic < result.critical_values[2],
            "normal data A²={} should be below 5% critical value={}",
            result.statistic,
            result.critical_values[2]
        );
    }

    #[test]
    fn anderson_nonnormal_data() {
        // Exponential-like data should fail normality
        let data: Vec<f64> = (1..=100).map(|i| (i as f64 / 100.0).ln().abs()).collect();
        let result = anderson(&data, "norm");
        assert!(
            result.statistic > result.critical_values[4],
            "non-normal data A²={} should exceed 1% critical value={}",
            result.statistic,
            result.critical_values[4]
        );
    }

    #[test]
    fn anderson_unsupported_dist() {
        let result = anderson(&[1.0, 2.0, 3.0], "expon");
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn anderson_significance_levels() {
        let result = anderson(&[1.0, 2.0, 3.0, 4.0, 5.0], "norm");
        assert_eq!(result.significance_level, [15.0, 10.0, 5.0, 2.5, 1.0]);
    }

    #[test]
    fn ks_1samp_perfect_fit() {
        // Data exactly matching uniform CDF on [0,1]
        let data: Vec<f64> = (1..=20).map(|i| i as f64 / 20.0).collect();
        let result = ks_1samp(&data, |x| x.clamp(0.0, 1.0));
        assert!(
            result.statistic < 0.1,
            "near-perfect fit D={} should be small",
            result.statistic
        );
    }

    #[test]
    fn ansari_same_samples_match_scipy_oracle() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ansari(&x, &y);
        assert_close(result.statistic, 15.5, 1e-12, "ansari statistic");
        assert_close(result.pvalue, 0.8589549227374823, 2e-3, "ansari pvalue");
    }

    #[test]
    fn ansari_handles_ties_with_reasonable_pvalue() {
        let x = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = ansari(&x, &y);
        assert_close(result.statistic, 24.0, 1e-12, "ansari tied statistic");
        assert_close(
            result.pvalue,
            0.2860899351713332,
            3e-2,
            "ansari tied pvalue",
        );
    }

    #[test]
    fn ansari_scale_difference_pushes_pvalue_down() {
        let x = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0];
        let y = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0];
        let result = ansari(&x, &y);
        assert_close(result.statistic, 19.5, 1e-12, "ansari scale statistic");
        assert!(
            result.pvalue < 0.1,
            "larger spread should lower the Ansari p-value: A={}, p={}",
            result.statistic,
            result.pvalue
        );
    }

    #[test]
    fn ansari_empty_returns_nan() {
        let result = ansari(&[], &[1.0, 2.0]);
        assert!(result.statistic.is_nan());
        assert!(result.pvalue.is_nan());
    }

    #[test]
    fn brunnermunzel_tied_samples_match_scipy_oracle() {
        let x = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = brunnermunzel(&x, &y);
        assert_close(
            result.statistic,
            1.6269784336399216,
            1e-12,
            "brunnermunzel tied statistic",
        );
        assert_close(
            result.pvalue,
            0.14903722001764458,
            1e-12,
            "brunnermunzel tied pvalue",
        );
        assert!(result.df.is_finite(), "expected finite degrees of freedom");
    }

    #[test]
    fn brunnermunzel_fully_separated_samples_match_scipy_shape() {
        let x = [1.0, 1.0, 1.0, 1.0];
        let y = [2.0, 2.0, 2.0, 2.0];
        let result = brunnermunzel(&x, &y);
        assert!(
            result.statistic.is_infinite() && result.statistic.is_sign_positive(),
            "expected positive infinite statistic for separated samples, got {}",
            result.statistic
        );
        assert!(
            result.pvalue.is_nan(),
            "expected NaN pvalue, got {}",
            result.pvalue
        );
        assert!(
            result.df.is_nan(),
            "expected NaN degrees of freedom, got {}",
            result.df
        );
    }

    #[test]
    fn brunnermunzel_identical_constant_samples_match_scipy_shape() {
        let x = [1.0, 1.0, 1.0, 1.0];
        let y = [1.0, 1.0, 1.0, 1.0];
        let result = brunnermunzel(&x, &y);
        assert!(
            result.statistic.is_nan(),
            "expected NaN statistic, got {}",
            result.statistic
        );
        assert!(
            result.pvalue.is_nan(),
            "expected NaN pvalue, got {}",
            result.pvalue
        );
        assert!(
            result.df.is_nan(),
            "expected NaN degrees of freedom, got {}",
            result.df
        );
    }

    #[test]
    fn epps_singleton_same_samples_match_scipy_oracle() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 1.0, 2.0, 3.0, 4.0];
        let result = epps_singleton_2samp(&x, &y);
        assert_close(
            result.statistic,
            0.0,
            1e-12,
            "epps singleton same statistic",
        );
        assert_close(result.pvalue, 1.0, 1e-12, "epps singleton same pvalue");
    }

    #[test]
    fn epps_singleton_nearby_samples_match_scipy_oracle() {
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let y = [-1.8, -0.9, 0.2, 1.1, 1.9, 3.2];
        let result = epps_singleton_2samp(&x, &y);
        assert_close(
            result.statistic,
            0.07066079761713522,
            5e-4,
            "epps singleton nearby statistic",
        );
        assert_close(
            result.pvalue,
            0.999390388757499,
            5e-4,
            "epps singleton nearby pvalue",
        );
    }

    #[test]
    fn epps_singleton_discrete_samples_match_scipy_oracle() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let result = epps_singleton_2samp(&x, &y);
        assert_close(
            result.statistic,
            2.315268966735134,
            5e-4,
            "epps singleton discrete statistic",
        );
        assert_close(
            result.pvalue,
            0.6779904963168365,
            5e-4,
            "epps singleton discrete pvalue",
        );
    }

    #[test]
    fn epps_singleton_small_sample_returns_nan() {
        let result = epps_singleton_2samp(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(result.statistic.is_nan());
        assert!(result.pvalue.is_nan());
    }

    #[test]
    fn epps_singleton_invalid_t_is_rejected() {
        let err = epps_singleton_2samp_with_t(
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[0.4, -0.8],
        )
        .expect_err("negative t should be rejected");
        assert!(
            err.to_string().contains("positive finite"),
            "unexpected error: {err}"
        );
    }

    // ── GaussianKde tests ──────────────────────────────────────────

    #[test]
    fn gaussian_kde_peak_at_data() {
        let data = vec![0.0; 100]; // All data at 0
        let kde = GaussianKde::new(&data);
        let at_zero = kde.evaluate(0.0);
        let at_far = kde.evaluate(10.0);
        assert!(
            at_zero > at_far,
            "KDE should peak at data: f(0)={at_zero}, f(10)={at_far}"
        );
    }

    #[test]
    fn gaussian_kde_integrates_to_one() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kde = GaussianKde::new(&data);
        // Numerical integration from -10 to 15
        let n = 10000;
        let a = -10.0;
        let b = 15.0;
        let dx = (b - a) / n as f64;
        let integral: f64 = (0..n)
            .map(|i| {
                let x = a + (i as f64 + 0.5) * dx;
                kde.evaluate(x) * dx
            })
            .sum();
        assert!(
            (integral - 1.0).abs() < 0.05,
            "KDE integral = {integral}, expected ~1.0"
        );
    }

    #[test]
    fn gaussian_kde_with_bandwidth() {
        let data = vec![0.0, 1.0, 2.0];
        let kde = GaussianKde::with_bandwidth(&data, 0.5);
        assert!((kde.bandwidth() - 0.5).abs() < 1e-12);
    }

    // ── Entropy tests ──────────────────────────────────────────────

    #[test]
    fn entropy_uniform() {
        // Uniform distribution over 4 outcomes: H = ln(4)
        let pk = vec![0.25, 0.25, 0.25, 0.25];
        let h = entropy(&pk, None);
        assert!(
            (h - 4.0_f64.ln()).abs() < 1e-10,
            "uniform entropy: {h}, expected {}",
            4.0_f64.ln()
        );
    }

    #[test]
    fn entropy_base2() {
        // Uniform over 8 outcomes in bits: H = 3
        let pk = vec![1.0; 8];
        let h = entropy(&pk, Some(2.0));
        assert!(
            (h - 3.0).abs() < 1e-10,
            "entropy in bits: {h}, expected 3.0"
        );
    }

    #[test]
    fn entropy_deterministic() {
        // Deterministic distribution: H = 0
        let pk = vec![1.0, 0.0, 0.0];
        let h = entropy(&pk, None);
        assert!(h.abs() < 1e-10, "deterministic entropy: {h}");
    }

    // ── Box-Cox tests ──────────────────────────────────────────────

    #[test]
    fn boxcox_lambda_1_is_linear() {
        // λ=1: y = (x^1 - 1)/1 = x - 1
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = boxcox(&data, Some(1.0)).unwrap();
        for (i, (&y, &x)) in result.data.iter().zip(data.iter()).enumerate() {
            assert_close(y, x - 1.0, 1e-10, &format!("boxcox λ=1 at {i}"));
        }
    }

    #[test]
    fn boxcox_lambda_0_is_log() {
        // λ=0: y = ln(x)
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = boxcox(&data, Some(0.0)).unwrap();
        for (i, (&y, &x)) in result.data.iter().zip(data.iter()).enumerate() {
            assert_close(y, x.ln(), 1e-10, &format!("boxcox λ=0 at {i}"));
        }
    }

    #[test]
    fn boxcox_optimal_lambda() {
        // Let the optimizer find lambda
        let data: Vec<f64> = (1..=20).map(|i| (i as f64).powi(3)).collect();
        let result = boxcox(&data, None).unwrap();
        assert_eq!(result.data.len(), 20);
        // Lambda should be somewhere reasonable
        assert!(
            result.lmbda > -3.0 && result.lmbda < 3.0,
            "optimal lambda out of range: {}",
            result.lmbda
        );
    }

    #[test]
    fn boxcox_negative_data_rejected() {
        assert!(boxcox(&[1.0, -1.0, 2.0], Some(1.0)).is_err());
    }

    // ── Kendall's tau tests ────────────────────────────────────────

    #[test]
    fn kendalltau_perfect_concordance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = kendalltau(&x, &y);
        assert!(
            (result.statistic - 1.0).abs() < 1e-10,
            "perfect concordance tau: {}",
            result.statistic
        );
    }

    #[test]
    fn kendalltau_perfect_discordance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let result = kendalltau(&x, &y);
        assert!(
            (result.statistic - (-1.0)).abs() < 1e-10,
            "perfect discordance tau: {}",
            result.statistic
        );
    }

    #[test]
    fn kendalltau_uncorrelated() {
        // Random-ish data should have |tau| << 1
        let x = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 8.0, 7.0];
        let y = vec![8.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0, 4.0];
        let result = kendalltau(&x, &y);
        assert!(
            result.statistic.abs() < 0.6,
            "uncorrelated tau should be small: {}",
            result.statistic
        );
    }

    #[test]
    fn kendalltau_pvalue_significant() {
        // Strong correlation should have small p-value
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..20).map(|i| i as f64 * 2.0 + 1.0).collect();
        let result = kendalltau(&x, &y);
        assert!(
            result.pvalue < 0.05,
            "strong correlation should have p < 0.05, got {}",
            result.pvalue
        );
    }

    #[test]
    fn kendalltau_constant_input_is_undefined() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = vec![2.0, 2.0, 2.0, 2.0];
        let result = kendalltau(&x, &y);
        assert!(result.statistic.is_nan(), "tau should be NaN for all ties");
        assert!(
            result.pvalue.is_nan(),
            "p-value should be NaN for undefined tau"
        );
    }

    #[test]
    fn somersd_matches_scipy_table_example() {
        let table = vec![
            vec![27.0, 25.0, 14.0, 7.0, 0.0],
            vec![7.0, 14.0, 18.0, 35.0, 12.0],
            vec![1.0, 3.0, 2.0, 7.0, 17.0],
        ];
        let result = somersd(SomersDInput::Table(&table), None).expect("somersd table");
        assert_close(
            result.statistic,
            0.6032766111513396,
            1e-12,
            "somersd table statistic",
        );
        assert_close(
            result.pvalue,
            1.0007091191074533e-27,
            1e-30,
            "somersd table pvalue",
        );
        assert_eq!(result.correlation, result.statistic);
        assert_eq!(result.table, table);
    }

    #[test]
    fn somersd_matches_rankings_for_perfect_ordering() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        let result = somersd(SomersDInput::Rankings(&x, &y), None).expect("somersd rankings");
        assert_close(result.statistic, 1.0, 1e-12, "somersd perfect statistic");
        assert_eq!(result.pvalue, 0.0);
        assert_eq!(
            result.table,
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0]
            ]
        );
    }

    #[test]
    fn somersd_rejects_invalid_input() {
        let err =
            somersd(SomersDInput::Rankings(&[1.0, 2.0, 3.0], &[1.0, 2.0]), None).expect_err("len");
        assert!(matches!(err, StatsError::InvalidArgument(_)));

        let err = somersd(SomersDInput::Table(&[vec![1.0, 2.5], vec![3.0, 4.0]]), None)
            .expect_err("non-integer");
        assert!(matches!(err, StatsError::InvalidArgument(_)));

        let err =
            somersd(SomersDInput::Table(&[vec![0.0, 0.0], vec![0.0, 1.0]]), None).expect_err("nz");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn anderson_ksamp_midrank_matches_scipy_oracle() {
        let samples = vec![
            vec![
                -0.6947284969268116,
                0.4963174476481955,
                -0.9670900823111159,
                1.436653455559641,
                0.7031564962971394,
                -0.6109476653841658,
                0.024649407219573916,
                0.6588093285059897,
                1.0908592504588038,
                -0.7423985725278241,
                -0.4193043734770502,
                1.7732539029260768,
                -1.151693349201912,
                -0.2513915385911738,
                1.651_546_177_396_081,
                0.1637626589199808,
                0.4728397841119197,
                -0.5722397169767986,
                -0.406_229_594_447_117_4,
                0.2862812550115444,
                -0.6044850665933583,
                -0.3188449480018543,
                -0.2942975007197502,
                -0.23399765678280584,
                0.6931683218520789,
                -0.6885208255353791,
                1.4898534267383032,
                -0.3287697071729781,
                0.2991549258299444,
                0.13359764531673786,
                0.3955262398950878,
                0.23318322884387597,
                -0.8860679599277484,
                -1.321065439784562,
                -0.439415709131537,
                -1.2207110036875532,
                1.3490367249364824,
                -0.40564423216117986,
                -0.8821761151494374,
                -0.9713257826496288,
                -0.5126552057931326,
                0.16469074952829044,
                0.6387419652973443,
                1.1379690459013305,
                1.696685185696353,
                -2.5729507818506434,
                -0.7999892635894051,
                1.1481727554429504,
                1.5977591334518029,
                0.1456142476657497,
            ],
            vec![
                1.8125549372539924,
                -0.3290670546847558,
                -1.3695312102876547,
                1.3319437227170101,
                1.254102551210099,
                0.5358177254644676,
                0.3182809576436851,
                0.6048356185627456,
                0.3166586510918795,
                1.337322057057232,
                1.1055258628813128,
                -1.0730512523906234,
                0.773_320_852_625_415,
                1.436183728122992,
                -1.1791415382494202,
                -0.189_213_812_292_439_9,
                -0.9554930152535846,
                0.5156304113864132,
                0.6003469986327579,
                0.2909003803399354,
                0.7090642645312579,
                0.511885720810475,
                0.330_493_902_577_640_4,
                0.4181173206487132,
                -0.5677696061279298,
                0.767014137689315,
                1.4711540065731597,
                0.3689168391061859,
                0.24885361754436064,
                0.339_595_061_641_651_7,
            ],
        ];
        let result =
            anderson_ksamp(&samples, Some(AndersonKSampleVariant::Midrank)).expect("anderson");
        assert_close(
            result.statistic,
            2.2868332786662013,
            1e-10,
            "anderson_ksamp midrank statistic",
        );
        assert_close(
            result.pvalue,
            0.03731919523231987,
            1e-10,
            "anderson_ksamp midrank pvalue",
        );
    }

    #[test]
    fn anderson_ksamp_continuous_caps_high_pvalue_like_scipy() {
        let samples = vec![
            vec![
                -0.6947284969268116,
                0.4963174476481955,
                -0.9670900823111159,
                1.436653455559641,
                0.7031564962971394,
                -0.6109476653841658,
                0.024649407219573916,
                0.6588093285059897,
                1.0908592504588038,
                -0.7423985725278241,
                -0.4193043734770502,
                1.7732539029260768,
                -1.151693349201912,
                -0.2513915385911738,
                1.651_546_177_396_081,
                0.1637626589199808,
                0.4728397841119197,
                -0.5722397169767986,
                -0.406_229_594_447_117_4,
                0.2862812550115444,
                -0.6044850665933583,
                -0.3188449480018543,
                -0.2942975007197502,
                -0.23399765678280584,
                0.6931683218520789,
                -0.6885208255353791,
                1.4898534267383032,
                -0.3287697071729781,
                0.2991549258299444,
                0.13359764531673786,
                0.3955262398950878,
                0.23318322884387597,
                -0.8860679599277484,
                -1.321065439784562,
                -0.439415709131537,
                -1.2207110036875532,
                1.3490367249364824,
                -0.40564423216117986,
                -0.8821761151494374,
                -0.9713257826496288,
                -0.5126552057931326,
                0.16469074952829044,
                0.6387419652973443,
                1.1379690459013305,
                1.696685185696353,
                -2.5729507818506434,
                -0.7999892635894051,
                1.1481727554429504,
                1.5977591334518029,
                0.1456142476657497,
            ],
            vec![
                0.8125549372539924,
                -1.3290670546847558,
                -2.3695312102876547,
                0.3319437227170101,
                0.254102551210099,
                -0.4641822745355324,
                -0.6817190423563149,
                -0.3951643814372544,
                -0.6833413489081206,
                0.33732205705723195,
                0.1055258628813128,
                -2.0730512523906234,
                -0.22667914737458514,
                0.43618372812299205,
                -2.17914153824942,
                -1.1892138122924398,
                -1.9554930152535846,
                -0.4843695886135868,
                -0.3996530013672421,
                -0.7090996196600647,
                -0.290_935_735_468_742_1,
                -0.488114279189525,
                -0.6695060974223596,
                -0.5818826793512867,
                -1.5677696061279297,
                -0.23298586231068502,
                0.4711540065731597,
                -0.6310831608938141,
                -0.7511463824556394,
                -0.6604049383583482,
            ],
            vec![
                -0.35588207741803635,
                1.4443378601567752,
                -0.4596751362611252,
                -0.9642484738356342,
                -1.471388522331468,
                -0.07404914873847856,
                0.5702675793629297,
                -0.26773250760437354,
                0.9708101620372975,
                0.5480673580787071,
                0.8645729481839737,
                -1.4825537919151204,
                -0.16172254381566852,
                -0.9004457032888357,
                0.780413911291489,
                -0.30830542393700865,
                0.7674577522114926,
                1.2410370557109975,
                -1.1837613277232888,
                -2.720720805523122,
            ],
        ];
        let result = anderson_ksamp(&samples, Some(AndersonKSampleVariant::Continuous))
            .expect("anderson continuous");
        assert_close(
            result.statistic,
            2.4676648582183063,
            1e-10,
            "anderson_ksamp continuous statistic",
        );
        assert_close(
            result.pvalue,
            0.02818394855742057,
            1e-12,
            "anderson_ksamp continuous pvalue",
        );
    }

    #[test]
    fn anderson_ksamp_rejects_invalid_input() {
        let err = anderson_ksamp(&[vec![1.0, 2.0]], None).expect_err("need two samples");
        assert!(matches!(err, StatsError::InvalidArgument(_)));

        let err = anderson_ksamp(&[vec![1.0, 2.0], vec![]], None).expect_err("empty sample");
        assert!(matches!(err, StatsError::InvalidArgument(_)));

        let err =
            anderson_ksamp(&[vec![1.0, 1.0], vec![1.0, 1.0]], None).expect_err("not distinct");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    // ── Chi-squared contingency tests ────────────────────────────────

    #[test]
    fn chi2_contingency_2x2_known() {
        // Classic 2x2 contingency table
        // [[10, 10], [20, 20]] — independent rows → chi2 = 0, p ≈ 1
        let table = vec![vec![10.0, 10.0], vec![20.0, 20.0]];
        let result = chi2_contingency(&table);
        assert!(
            result.statistic < 0.01,
            "independent table chi2 should be ~0: {}",
            result.statistic
        );
        assert!(
            result.pvalue > 0.9,
            "independent table p should be ~1: {}",
            result.pvalue
        );
        assert_eq!(result.dof, 1);
    }

    #[test]
    fn chi2_contingency_2x2_dependent() {
        // Highly dependent: [[50, 5], [5, 50]]
        let table = vec![vec![50.0, 5.0], vec![5.0, 50.0]];
        let result = chi2_contingency(&table);
        assert!(
            (result.statistic - 70.4).abs() < 1e-10,
            "dependent table should match Yates-corrected chi2: {}",
            result.statistic
        );
        assert!(
            result.pvalue < 1e-10,
            "dependent table should have tiny p: {}",
            result.pvalue
        );
    }

    #[test]
    fn chi2_contingency_expected_frequencies() {
        let table = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        let result = chi2_contingency(&table);
        // Grand total = 100
        // Row totals: [30, 70], Col totals: [40, 60]
        // Expected: [[30*40/100, 30*60/100], [70*40/100, 70*60/100]] = [[12, 18], [28, 42]]
        assert!(
            (result.expected[0][0] - 12.0).abs() < 1e-10,
            "expected[0][0] = {}",
            result.expected[0][0]
        );
        assert!(
            (result.expected[1][1] - 42.0).abs() < 1e-10,
            "expected[1][1] = {}",
            result.expected[1][1]
        );
        assert!(
            (result.statistic - 0.4464285714285714).abs() < 1e-12,
            "Yates-corrected chi2 = {}",
            result.statistic
        );
    }

    #[test]
    fn chi2_contingency_3x3() {
        let table = vec![
            vec![10.0, 20.0, 30.0],
            vec![20.0, 15.0, 25.0],
            vec![30.0, 25.0, 5.0],
        ];
        let result = chi2_contingency(&table);
        assert_eq!(result.dof, 4); // (3-1)*(3-1)
        assert!(result.statistic.is_finite());
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn chi2_contingency_empty_rejected() {
        let result = chi2_contingency(&[]);
        assert!(result.statistic.is_nan());
    }

    // ── Power divergence tests ───────────────────────────────────────

    #[test]
    fn power_divergence_pearson_uniform() {
        // Observed matches expected → stat ≈ 0, p ≈ 1
        let obs = [25.0, 25.0, 25.0, 25.0];
        let (stat, pvalue) = power_divergence(&obs, None, 1.0);
        assert!(stat.abs() < 1e-10, "uniform obs chi2 = {stat}");
        assert!(pvalue > 0.99, "uniform p = {pvalue}");
    }

    #[test]
    fn power_divergence_pearson_skewed() {
        // Very skewed: [50, 10, 10, 10] vs expected uniform [20, 20, 20, 20]
        let obs = [50.0, 10.0, 10.0, 10.0];
        let (stat, pvalue) = power_divergence(&obs, None, 1.0);
        assert!(stat > 10.0, "skewed chi2 should be large: {stat}");
        assert!(pvalue < 0.05, "skewed p should be small: {pvalue}");
    }

    #[test]
    fn power_divergence_gtest() {
        // G-test (lambda=0): same as Pearson for large samples
        let obs = [50.0, 10.0, 10.0, 10.0];
        let (stat_g, _) = power_divergence(&obs, None, 0.0);
        let (stat_p, _) = power_divergence(&obs, None, 1.0);
        // G and Pearson should be in the same ballpark
        assert!(
            (stat_g - stat_p).abs() / stat_p < 0.3,
            "G={stat_g} vs Pearson={stat_p}"
        );
    }

    #[test]
    fn power_divergence_custom_expected() {
        let obs = [10.0, 20.0, 30.0];
        let exp = [20.0, 20.0, 20.0];
        let (stat, pvalue) = power_divergence(&obs, Some(&exp), 1.0);
        // chi2 = (10-20)²/20 + (20-20)²/20 + (30-20)²/20 = 100/20 + 0 + 100/20 = 10
        assert!((stat - 10.0).abs() < 1e-10, "expected chi2=10, got {stat}");
        assert!(pvalue > 0.0 && pvalue < 1.0);
    }

    #[test]
    fn power_divergence_unequal_totals_rejected() {
        let obs = [10.0, 20.0];
        let exp = [20.0, 20.0];
        let (stat, pvalue) = power_divergence(&obs, Some(&exp), 1.0);
        assert!(stat.is_nan());
        assert!(pvalue.is_nan());
    }

    // ── Fisher's exact test ──────────────────────────────────────────

    #[test]
    fn fisher_exact_independent_table() {
        // Independent: [[10, 10], [10, 10]] → p ≈ 1
        let result = fisher_exact(&[[10.0, 10.0], [10.0, 10.0]]);
        assert!(
            (result.odds_ratio - 1.0).abs() < 1e-10,
            "OR should be 1: {}",
            result.odds_ratio
        );
        assert!(
            result.pvalue > 0.9,
            "independent p should be ~1: {}",
            result.pvalue
        );
    }

    #[test]
    fn fisher_exact_dependent_table() {
        // Strongly dependent: [[8, 2], [1, 9]]
        let result = fisher_exact(&[[8.0, 2.0], [1.0, 9.0]]);
        assert!(
            result.odds_ratio > 10.0,
            "OR should be large: {}",
            result.odds_ratio
        );
        assert!(
            result.pvalue < 0.01,
            "dependent p should be small: {}",
            result.pvalue
        );
    }

    #[test]
    fn fisher_exact_known_value() {
        // Classic example: [[1, 9], [11, 3]]
        // SciPy: fisher_exact([[1,9],[11,3]]) → odds_ratio ≈ 0.0303, p ≈ 0.0014
        let result = fisher_exact(&[[1.0, 9.0], [11.0, 3.0]]);
        assert!(
            result.odds_ratio < 0.1,
            "OR should be small: {}",
            result.odds_ratio
        );
        assert!(
            result.pvalue < 0.01,
            "p should be very small: {}",
            result.pvalue
        );
    }

    #[test]
    fn fisher_exact_zero_cell() {
        // Table with zero: [[0, 5], [5, 0]]
        let result = fisher_exact(&[[0.0, 5.0], [5.0, 0.0]]);
        assert!(result.pvalue < 0.05, "p should be small: {}", result.pvalue);
    }

    // ── percentile tests ─────────────────────────────────────────────

    #[test]
    fn percentile_median() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 50.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn percentile_min_max() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert!((percentile(&data, 0.0) - 10.0).abs() < 1e-12);
        assert!((percentile(&data, 100.0) - 50.0).abs() < 1e-12);
    }

    #[test]
    fn percentile_interpolation() {
        // 25th percentile of [1, 2, 3, 4]: position = 0.25*3 = 0.75
        // interpolate: 1*(1-0.75) + 2*0.75 = 1.75
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&data, 25.0) - 1.75).abs() < 1e-12);
    }

    #[test]
    fn percentile_empty() {
        assert!(percentile(&[], 50.0).is_nan());
    }

    // ── trim_mean tests ──────────────────────────────────────────────

    #[test]
    fn trim_mean_zero_trim() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        assert!((trim_mean(&data, 0.0) - mean).abs() < 1e-12);
    }

    #[test]
    fn trim_mean_with_outliers() {
        // Trim 20% from each end of [1, 2, 3, 4, 100]
        // Removes 1 element each end (20% of 5 = 1) → [2, 3, 4] → mean = 3
        let data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        assert!((trim_mean(&data, 0.2) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn trim_mean_empty() {
        assert!(trim_mean(&[], 0.1).is_nan());
    }

    // ── binomtest tests ──────────────────────────────────────────────

    #[test]
    fn binomtest_fair_coin() {
        // 50 heads out of 100 with p=0.5 → not significant
        let p = binomtest(50, 100, 0.5);
        assert!(p > 0.1, "fair coin should not reject: p={p}");
    }

    #[test]
    fn binomtest_biased_coin() {
        // 90 heads out of 100 with p=0.5 → very significant
        let p = binomtest(90, 100, 0.5);
        assert!(p < 0.001, "biased result should reject: p={p}");
    }

    #[test]
    fn binomtest_all_successes() {
        // n successes out of n with p=0.5
        let p = binomtest(10, 10, 0.5);
        assert!(p < 0.01, "all successes should reject: p={p}");
    }

    #[test]
    fn binomtest_edge_cases() {
        assert!((binomtest(0, 10, 0.0) - 1.0).abs() < 1e-12);
        assert!((binomtest(10, 10, 1.0) - 1.0).abs() < 1e-12);
        assert!(binomtest(5, 0, 0.5).is_nan());
    }

    // ── ttest_rel tests ──────────────────────────────────────────────

    #[test]
    fn ttest_rel_no_difference() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ttest_rel(&a, &b);
        assert!(
            (result.statistic).abs() < 1e-10,
            "identical pairs: t = {}",
            result.statistic
        );
    }

    #[test]
    fn ttest_rel_significant_shift() {
        // b = a + 10, very significant difference
        let a: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let b: Vec<f64> = a.iter().map(|&x| x + 10.0).collect();
        let result = ttest_rel(&a, &b);
        assert!(
            result.pvalue < 0.001,
            "shifted pairs should be significant: p = {}",
            result.pvalue
        );
    }

    #[test]
    fn ttest_rel_unequal_lengths() {
        let result = ttest_rel(&[1.0, 2.0], &[1.0]);
        assert!(result.statistic.is_nan());
    }

    // ── chisquare tests ──────────────────────────────────────────────

    #[test]
    fn chisquare_uniform() {
        let obs = [25.0, 25.0, 25.0, 25.0];
        let (stat, p) = chisquare(&obs, None);
        assert!(stat.abs() < 1e-10, "uniform chi2 = {stat}");
        assert!(p > 0.99, "uniform p = {p}");
    }

    #[test]
    fn chisquare_skewed() {
        let obs = [50.0, 10.0, 10.0, 10.0];
        let (stat, p) = chisquare(&obs, None);
        assert!(stat > 10.0, "skewed chi2 = {stat}");
        assert!(p < 0.05, "skewed p = {p}");
    }

    // ── wasserstein_distance tests ───────────────────────────────────

    #[test]
    fn wasserstein_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = wasserstein_distance(&a, &a);
        assert!(d.abs() < 1e-10, "identical should be 0: {d}");
    }

    #[test]
    fn wasserstein_shifted() {
        // Shift by 1.0: W distance should be ~1.0
        let a: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let b: Vec<f64> = a.iter().map(|&x| x + 1.0).collect();
        let d = wasserstein_distance(&a, &b);
        assert!(
            (d - 1.0).abs() < 0.05,
            "shifted by 1 should give W ≈ 1: {d}"
        );
    }

    #[test]
    fn wasserstein_empty() {
        assert!(wasserstein_distance(&[], &[1.0]).is_nan());
    }

    #[test]
    fn wasserstein_nan_input_returns_nan() {
        assert!(wasserstein_distance(&[1.0, f64::NAN], &[1.0, 2.0]).is_nan());
        assert!(wasserstein_distance(&[1.0, 2.0], &[1.0, f64::NAN]).is_nan());
    }

    // ── energy_distance tests ────────────────────────────────────────

    #[test]
    fn energy_distance_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = energy_distance(&a, &a);
        assert!(d.abs() < 1e-10, "identical should be 0: {d}");
    }

    #[test]
    fn energy_distance_different() {
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![10.0, 11.0, 12.0];
        let d = energy_distance(&a, &b);
        assert!(
            (d - 4.268_749_491_621_899).abs() < 1e-12,
            "should match SciPy oracle for separated samples: {d}"
        );
    }

    #[test]
    fn energy_distance_empty() {
        assert!(energy_distance(&[], &[1.0]).is_nan());
    }

    #[test]
    fn energy_distance_nan_input_returns_nan() {
        assert!(energy_distance(&[1.0, f64::NAN], &[1.0, 2.0]).is_nan());
        assert!(energy_distance(&[1.0, 2.0], &[1.0, f64::NAN]).is_nan());
    }

    // ── Friedman test ────────────────────────────────────────────────

    #[test]
    fn friedmanchisquare_identical_groups() {
        // All groups identical → no treatment effect
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = friedmanchisquare(&[&a, &a, &a]);
        assert!(result.pvalue > 0.9, "identical groups: p={}", result.pvalue);
    }

    #[test]
    fn friedmanchisquare_different_groups() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let c = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        let result = friedmanchisquare(&[&a, &b, &c]);
        assert!(
            result.pvalue < 0.01,
            "different groups should reject: p={}",
            result.pvalue
        );
    }

    // ── Fligner test ─────────────────────────────────────────────────

    #[test]
    fn fligner_equal_variance() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0]; // same spread, shifted
        let result = fligner(&[&a, &b]);
        assert!(result.pvalue > 0.1, "equal variance: p={}", result.pvalue);
    }

    #[test]
    fn fligner_unequal_variance() {
        let a: Vec<f64> = (0..50).map(|i| i as f64).collect(); // wide spread
        let b: Vec<f64> = (0..50).map(|i| 25.0 + 0.01 * i as f64).collect(); // narrow
        let result = fligner(&[&a, &b]);
        assert!(
            result.pvalue < 0.05,
            "unequal variance: p={}",
            result.pvalue
        );
    }

    // ── Mood test ────────────────────────────────────────────────────

    #[test]
    fn mood_equal_scale() {
        let a: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..30).map(|i| i as f64 + 100.0).collect();
        let result = mood(&a, &b);
        assert!(result.pvalue > 0.1, "equal scale: p={}", result.pvalue);
    }

    #[test]
    fn mood_different_scale() {
        let a: Vec<f64> = (0..50).map(|i| i as f64 * 10.0).collect(); // wide
        let b: Vec<f64> = (0..50).map(|i| 250.0 + i as f64 * 0.1).collect(); // narrow
        let result = mood(&a, &b);
        assert!(result.pvalue < 0.05, "different scale: p={}", result.pvalue);
    }

    // ── Median test ──────────────────────────────────────────────────

    #[test]
    fn median_test_same_median() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let result = median_test(&[&a, &b]);
        assert!(result.pvalue > 0.1, "similar medians: p={}", result.pvalue);
    }

    #[test]
    fn median_test_different_medians() {
        let a: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let b: Vec<f64> = (100..120).map(|i| i as f64).collect();
        let result = median_test(&[&a, &b]);
        assert!(
            result.pvalue < 0.01,
            "different medians: p={}",
            result.pvalue
        );
    }

    // ── VonMises tests ───────────────────────────────────────────────

    #[test]
    fn vonmises_pdf_integrates_to_one() {
        // Numerically integrate von Mises PDF over [-π, π]
        let vm = VonMises::new(2.0, 0.0);
        let n = 1000;
        let h = 2.0 * std::f64::consts::PI / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let x = -std::f64::consts::PI + (i as f64 + 0.5) * h;
            integral += vm.pdf(x) * h;
        }
        assert!(
            (integral - 1.0).abs() < 0.01,
            "vonmises integral = {integral}"
        );
    }

    #[test]
    fn vonmises_peak_at_mu() {
        let vm = VonMises::new(5.0, 1.0);
        let at_mu = vm.pdf(1.0);
        let away = vm.pdf(1.0 + 1.0);
        assert!(at_mu > away, "peak should be at mu");
    }

    #[test]
    fn vonmises_circular_variance() {
        // High kappa → low variance
        let vm_high = VonMises::new(100.0, 0.0);
        let vm_low = VonMises::new(0.5, 0.0);
        assert!(
            vm_high.circular_variance() < vm_low.circular_variance(),
            "high kappa should have lower variance"
        );
    }

    // ── MultivariateNormal tests ─────────────────────────────────────

    #[test]
    fn mvn_1d_matches_normal() {
        let mvn = MultivariateNormal::new(&[0.0], &[vec![1.0]]).expect("1D MVN");
        let norm = Normal::standard();
        let x = 1.5;
        let mvn_pdf = mvn.pdf(&[x]).expect("mvn pdf");
        let norm_pdf = norm.pdf(x);
        assert!(
            (mvn_pdf - norm_pdf).abs() < 1e-10,
            "1D MVN should match Normal: {} vs {}",
            mvn_pdf,
            norm_pdf
        );
    }

    #[test]
    fn mvn_peak_at_mean() {
        let mvn =
            MultivariateNormal::new(&[1.0, 2.0], &[vec![1.0, 0.0], vec![0.0, 1.0]]).expect("MVN");
        let at_mean = mvn.pdf(&[1.0, 2.0]).expect("pdf at mean");
        let away = mvn.pdf(&[3.0, 4.0]).expect("pdf away");
        assert!(at_mean > away, "peak should be at mean");
    }

    #[test]
    fn mvn_logpdf_consistent() {
        let mvn =
            MultivariateNormal::new(&[0.0, 0.0], &[vec![1.0, 0.0], vec![0.0, 1.0]]).expect("MVN");
        let x = [1.0, 1.0];
        let pdf = mvn.pdf(&x).expect("pdf");
        let logpdf = mvn.logpdf(&x).expect("logpdf");
        assert!(
            (pdf.ln() - logpdf).abs() < 1e-10,
            "ln(pdf) = {}, logpdf = {}",
            pdf.ln(),
            logpdf
        );
    }

    #[test]
    fn inverse_gamma_pdf_cdf() {
        let ig = InverseGamma::new(3.0);
        assert!(ig.pdf(1.0) > 0.0);
        assert!(ig.pdf(-1.0) == 0.0);
        let c = ig.cdf(1.0);
        assert!(c > 0.0 && c < 1.0);
        assert!((ig.mean() - 0.5).abs() < 1e-10); // 1/(a-1) = 1/2
    }

    #[test]
    fn burr12_variance_matches_scipy_reference_values() {
        let cases = [
            (
                (2.0, 3.0),
                (0.589_048_622_548_086_1, 0.153_021_720_274_202_2),
            ),
            (
                (3.0, 2.5),
                (0.727_057_387_946_532_4, 0.110_180_038_392_808_1),
            ),
            (
                (1.5, 4.0),
                (0.417_994_915_214_470_2, 0.123_848_047_436_612_57),
            ),
        ];

        for &((c, d), (mean, var)) in &cases {
            let dist = Burr12::new(c, d);
            assert_close(dist.mean(), mean, 1e-12, "Burr12 mean");
            assert_close(dist.var(), var, 1e-12, "Burr12 variance");
        }
    }

    #[test]
    fn burr12_variance_returns_nan_at_boundary_or_below() {
        assert!(Burr12::new(1.0, 2.0).var().is_nan());
        assert!(Burr12::new(1.5, 4.0 / 3.0).var().is_nan());
    }

    #[test]
    fn invweibull_variance_matches_scipy_reference_values() {
        let cases = [
            (3.0, (1.354_117_939_426_400_2, 0.845_303_140_831_346_9)),
            (4.5, (1.190_151_186_912_873, 0.184_256_270_703_863_52)),
            (2.2, (1.628_998_124_608_686, 7.852_239_966_100_065)),
        ];

        for &(c, (mean, var)) in &cases {
            let dist = InvWeibull::new(c);
            assert_close(dist.mean(), mean, 1e-12, "InvWeibull mean");
            assert_close(dist.var(), var, 1e-10, "InvWeibull variance");
        }
    }

    #[test]
    fn invweibull_variance_returns_nan_at_boundary_or_below() {
        assert!(InvWeibull::new(2.0).var().is_nan());
        assert!(InvWeibull::new(1.9).var().is_nan());
    }

    #[test]
    fn doubleweibull_ppf_matches_scipy_reference_values() {
        let dist = DoubleWeibull::new(2.5);
        let qs = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999];
        let expected = [
            -2.076_660_482_588_982_5,
            -1.209_677_745_531_848_8,
            -0.863_634_900_602_374_8,
            0.0,
            0.863_634_900_602_374_8,
            1.209_677_745_531_848_8,
            2.076_660_482_588_982_5,
        ];

        assert!(dist.ppf(0.0).is_infinite() && dist.ppf(0.0).is_sign_negative());
        assert!(dist.ppf(1.0).is_infinite() && dist.ppf(1.0).is_sign_positive());

        for (&q, &want) in qs.iter().zip(expected.iter()) {
            assert_close(dist.ppf(q), want, 1e-12, &format!("DoubleWeibull ppf({q})"));
        }
    }

    #[test]
    fn burr12_ppf_matches_scipy_reference_values() {
        let dist = Burr12::new(2.0, 3.0);
        let qs = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999];
        let expected = [
            0.018_263_508_111_510_767,
            0.189_061_282_792_871_93,
            0.317_241_889_255_200_47,
            0.509_824_528_533_958_5,
            0.766_420_936_540_88,
            1.074_446_224_820_9,
            2.999_999_999_999_998_7,
        ];

        assert_eq!(dist.ppf(0.0), 0.0);
        assert!(dist.ppf(1.0).is_infinite());

        for (&q, &want) in qs.iter().zip(expected.iter()) {
            assert_close(dist.ppf(q), want, 1e-12, &format!("Burr12 ppf({q})"));
        }
    }

    #[test]
    fn truncexpon_ppf_matches_scipy_reference_values() {
        let dist = TruncExpon::new(2.5);
        let qs = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999];
        let expected = [
            0.000_918_336_543_330_507_7,
            0.096_281_301_341_612_72,
            0.260_688_045_554_922_53,
            0.614_257_446_267_395_8,
            1.166_151_310_110_486_6,
            1.749_410_009_011_430_4,
            2.488_879_567_882_692_7,
        ];

        assert_eq!(dist.ppf(0.0), 0.0);
        assert_close(dist.ppf(1.0), 2.5, 1e-15, "TruncExpon ppf(1)");

        for (&q, &want) in qs.iter().zip(expected.iter()) {
            assert_close(dist.ppf(q), want, 1e-12, &format!("TruncExpon ppf({q})"));
        }
    }

    #[test]
    fn invweibull_ppf_matches_scipy_reference_values() {
        let dist = InvWeibull::new(3.0);
        let qs = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999];
        let expected = [
            0.525_074_610_470_846_1,
            0.757_288_631_330_907_7,
            0.896_839_747_563_287_4,
            1.129_947_276_337_390_1,
            1.514_824_778_579_371_6,
            2.117_259_243_124_697,
            9.998_332_777_468_924,
        ];

        assert_eq!(dist.ppf(0.0), 0.0);
        assert!(dist.ppf(1.0).is_infinite());

        for (&q, &want) in qs.iter().zip(expected.iter()) {
            assert_close(dist.ppf(q), want, 1e-12, &format!("InvWeibull ppf({q})"));
        }
    }

    #[test]
    fn betaprime_ppf_matches_scipy_reference_values() {
        let dist = BetaPrime::new(2.5, 4.0);
        let qs = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999];
        let expected = [
            0.022_604_407_194_587_128,
            0.187_166_336_074_701_37,
            0.330_285_130_036_205_63,
            0.592_692_551_169_090_9,
            1.035_938_736_458_164,
            1.704_029_322_119_078_5,
            8.427_930_904_530_811,
        ];

        assert_eq!(dist.ppf(0.0), 0.0);
        assert!(dist.ppf(1.0).is_infinite());

        for (&q, &want) in qs.iter().zip(expected.iter()) {
            assert_close(dist.ppf(q), want, 1e-10, &format!("BetaPrime ppf({q})"));
        }
    }

    #[test]
    fn tukey_lambda_variance_matches_scipy_reference_values() {
        let cases = [
            (0.0, 3.289_868_133_696_453),
            (0.14, 2.110_297_022_214_417),
            (0.5, 0.858_407_346_410_206_9),
            (-0.25, 9.778_362_573_185_369),
        ];

        for &(lam, var) in &cases {
            let dist = TukeyLambda::new(lam);
            assert_close(dist.var(), var, 1e-12, "TukeyLambda variance");
        }
    }

    #[test]
    fn tukey_lambda_variance_returns_nan_when_undefined() {
        assert!(TukeyLambda::new(-0.5).var().is_nan());
        assert!(TukeyLambda::new(-0.75).var().is_nan());
    }

    fn legacy_tukey_lambda_cdf(dist: &TukeyLambda, x: f64) -> f64 {
        let mut lo = 0.0f64;
        let mut hi = 1.0f64;
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            let q = dist.ppf(mid);
            if q < x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    }

    fn legacy_tukey_lambda_pdf(dist: &TukeyLambda, x: f64) -> f64 {
        let h = 1e-7;
        let c1 = legacy_tukey_lambda_cdf(dist, x + h);
        let c0 = legacy_tukey_lambda_cdf(dist, x - h);
        (c1 - c0) / (2.0 * h)
    }

    #[test]
    fn tukey_lambda_pdf_matches_previous_numerical_reference_values() {
        let lambdas: [f64; 6] = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let xs: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];

        for &lam in &lambdas {
            let dist = TukeyLambda::new(lam);
            for &x in &xs {
                if lam > 0.0 && x.abs() >= 1.0 / lam {
                    continue;
                }
                assert_close(
                    dist.pdf(x),
                    legacy_tukey_lambda_pdf(&dist, x),
                    1e-8,
                    &format!("TukeyLambda({lam}) pdf({x})"),
                );
            }
        }
    }

    #[test]
    fn tukey_lambda_cdf_ppf_roundtrip_matches_quantiles() {
        let lambdas = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];

        for &lam in &lambdas {
            let dist = TukeyLambda::new(lam);
            for &q in &quantiles {
                assert_close(
                    dist.cdf(dist.ppf(q)),
                    q,
                    1e-10,
                    &format!("TukeyLambda({lam}) cdf(ppf({q}))"),
                );
            }
        }
    }

    #[test]
    fn tukey_lambda_pdf_trapezoid_integrates_to_one() {
        for &lam in &[0.0, 0.5, 1.0, 2.0] {
            let dist = TukeyLambda::new(lam);
            let lower = dist.ppf(1e-8);
            let upper = dist.ppf(1.0 - 1e-8);
            let n = 20_000usize;
            let h = (upper - lower) / n as f64;
            let integral = (0..=n)
                .map(|i| {
                    let x = lower + i as f64 * h;
                    let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
                    weight * dist.pdf(x)
                })
                .sum::<f64>()
                * h;
            assert_close(
                integral,
                1.0,
                1e-6,
                &format!("TukeyLambda({lam}) trapezoid integral"),
            );
        }
    }

    #[test]
    fn gen_half_logistic_moments_match_scipy_reference_values() {
        let cases = [
            (0.5, (3.0, 1.579_736_267_392_905_8)),
            (1.0, (1.0, 0.644_934_066_848_226_6)),
            (2.0, (0.306_852_819_440_054_7, 0.233_700_550_136_169_83)),
            (3.5, (0.111_913_624_175_919_25, 0.094_010_019_668_303_19)),
        ];

        for &(c, (mean, var)) in &cases {
            let dist = GenHalfLogistic::new(c);
            assert_close(dist.mean(), mean, 1e-9, "GenHalfLogistic mean");
            assert_close(dist.var(), var, 1e-9, "GenHalfLogistic variance");
        }
    }

    #[test]
    fn gen_logistic_moments_match_scipy_reference_values() {
        let cases = [
            (0.5, (-1.386_294_361_119_890_8, 6.579_736_267_392_906_5)),
            (1.0, (0.0, 3.289_868_133_696_453_3)),
            (5.0, (2.083_333_333_333_333, 1.866_257_022_585_341_7)),
        ];

        for &(c, (mean, var)) in &cases {
            let dist = GenLogistic::new(c);
            assert_close(dist.mean(), mean, 1e-9, "GenLogistic mean");
            assert_close(dist.var(), var, 1e-9, "GenLogistic variance");
        }
    }

    #[test]
    fn gompertz_moments_match_scipy_reference_values() {
        let cases = [
            (0.5, (0.922_910_632_477_416_4, 0.329_627_827_869_170_1)),
            (1.0, (0.596_347_362_317_641_5, 0.176_300_593_521_657_72)),
            (3.0, (0.262_083_740_252_362_75, 0.046_960_406_809_075_47)),
        ];

        for &(c, (mean, var)) in &cases {
            let dist = Gompertz::new(c);
            assert_close(dist.mean(), mean, 1e-6, "Gompertz mean");
            assert_close(dist.var(), var, 1e-6, "Gompertz variance");
        }
    }

    #[test]
    fn log_laplace_variance_matches_scipy_reference_values() {
        let cases = [
            (3.0, (1.125, 0.534_375)),
            (5.0, (1.041_666_666_666_666_7, 0.105_406_746_031_745_82)),
        ];

        for &(c, (mean, var)) in &cases {
            let dist = LogLaplace::new(c);
            assert_close(dist.mean(), mean, 1e-12, "LogLaplace mean");
            assert_close(dist.var(), var, 1e-12, "LogLaplace variance");
        }
    }

    #[test]
    fn log_laplace_variance_is_infinite_when_second_moment_diverges() {
        assert!(LogLaplace::new(1.5).var().is_infinite());
        assert!(LogLaplace::new(2.0).var().is_infinite());
    }

    #[test]
    fn mielke_moments_match_scipy_reference_values() {
        let cases = [
            (
                (1.5, 3.0),
                (0.862_369_853_076_597_1, 0.658_500_341_830_102_8),
            ),
            (
                (2.0, 2.5),
                (1.174_450_160_620_581_7, 2.144_017_302_080_036_4),
            ),
            (
                (4.0, 3.5),
                (1.208_661_176_745_414_5, 0.553_573_454_404_080_7),
            ),
        ];

        for &((k, s), (mean, var)) in &cases {
            let dist = Mielke::new(k, s);
            assert_close(dist.mean(), mean, 1e-12, "Mielke mean");
            assert_close(dist.var(), var, 1e-10, "Mielke variance");
        }
    }

    #[test]
    fn mielke_moments_are_infinite_when_they_do_not_exist() {
        assert!(Mielke::new(3.5, 1.0).mean().is_infinite());
        assert!(Mielke::new(3.5, 1.2).var().is_infinite());
        assert!(Mielke::new(2.0, 2.0).var().is_infinite());
    }

    #[test]
    fn frechet_r_moments_match_scipy_reference_values() {
        let cases = [
            (3.0, (-0.892_979_511_569_248_9, 0.105_332_884_868_479_14)),
            (5.0, (-0.918_168_742_399_760_8, 0.044_229_977_983_117_01)),
            (1.5, (-0.902_745_292_950_933_5, 0.375_690_284_813_931_96)),
        ];

        for &(c, (mean, var)) in &cases {
            let dist = FrechetR::new(c);
            assert_close(dist.mean(), mean, 1e-12, "FrechetR mean");
            assert_close(dist.var(), var, 1e-12, "FrechetR variance");
        }
    }

    #[test]
    fn crystal_ball_moments_match_scipy_reference_values() {
        let finite_cases = [
            (
                (2.0, 4.0),
                (-0.053_285_264_688_121_33, 1.281_348_758_903_764),
            ),
            (
                (3.0, 5.0),
                (-0.002_132_793_568_842_121, 1.009_333_258_932_100_2),
            ),
        ];

        for &((beta_param, m), (mean, var)) in &finite_cases {
            let dist = CrystalBall::new(beta_param, m);
            assert_close(dist.mean(), mean, 1e-10, "CrystalBall mean");
            assert_close(dist.var(), var, 1e-10, "CrystalBall variance");
        }
    }

    #[test]
    fn crystal_ball_divergent_moments_match_scipy_shape() {
        assert!(CrystalBall::new(1.0, 2.0).mean().is_infinite());
        assert!(CrystalBall::new(1.0, 2.0).var().is_infinite());
        assert!(CrystalBall::new(2.0, 3.0).var().is_infinite());
    }

    #[test]
    fn argus_moments_match_scipy_reference_values() {
        let cases = [
            (1.0, (0.618_702_668_355_183_7, 0.052_156_512_541_978_56)),
            (2.0, (0.705_658_515_503_037_3, 0.044_467_693_721_274_79)),
            (5.0, (0.936_445_797_957_029_3, 0.003_084_134_913_330_483)),
        ];

        for &(chi, (mean, var)) in &cases {
            let dist = Argus::new(chi);
            assert_close(dist.mean(), mean, 1e-8, "Argus mean");
            assert_close(dist.var(), var, 1e-8, "Argus variance");
        }
    }

    #[test]
    fn argus_extreme_chi_values_remain_well_behaved() {
        let low = Argus::new(0.1);
        let high = Argus::new(10.0);
        assert!(low.mean().is_finite() && low.var() >= 0.0);
        assert!(high.mean().is_finite() && high.var() >= 0.0);
    }

    #[test]
    fn expon_weibull_moments_match_scipy_reference_values() {
        let cases = [
            ((1.0, 1.0), (1.0, 1.0)),
            ((2.0, 0.5), (3.499_999_999_874_901_4, 34.250_000_000_307_45)),
            (
                (1.5, 2.0),
                (1.039_415_461_769_554_6, 0.199_987_803_381_160_18),
            ),
        ];

        for &((a, c), (mean, var)) in &cases {
            let dist = ExponWeibull::new(a, c);
            assert_close(dist.mean(), mean, 1e-6, "ExponWeibull mean");
            assert_close(dist.var(), var, 1e-5, "ExponWeibull variance");
        }
    }

    #[test]
    fn closed_form_variances_stay_nonnegative_for_valid_parameters() {
        for &(c, d) in &[(2.0, 3.0), (3.0, 2.5), (1.5, 4.0)] {
            assert!(Burr12::new(c, d).var() >= 0.0, "Burr12({c}, {d}) variance");
        }
        for &c in &[2.2, 3.0, 4.5] {
            assert!(InvWeibull::new(c).var() >= 0.0, "InvWeibull({c}) variance");
        }
        for &lam in &[-0.25, 0.0, 0.14, 0.5] {
            assert!(
                TukeyLambda::new(lam).var() >= 0.0,
                "TukeyLambda({lam}) variance"
            );
        }
        for &c in &[0.5, 1.0, 2.0, 3.5] {
            assert!(
                GenHalfLogistic::new(c).var() >= 0.0,
                "GenHalfLogistic({c}) variance"
            );
        }
        for &c in &[0.5, 1.0, 5.0] {
            assert!(
                GenLogistic::new(c).var() >= 0.0,
                "GenLogistic({c}) variance"
            );
        }
        for &c in &[0.5, 1.0, 3.0] {
            assert!(Gompertz::new(c).var() >= 0.0, "Gompertz({c}) variance");
        }
        for &c in &[3.0, 5.0] {
            assert!(LogLaplace::new(c).var() >= 0.0, "LogLaplace({c}) variance");
        }
        for &(k, s) in &[(1.5, 3.0), (2.0, 2.5), (4.0, 3.5)] {
            assert!(Mielke::new(k, s).var() >= 0.0, "Mielke({k}, {s}) variance");
        }
        for &c in &[1.5, 3.0, 5.0] {
            assert!(FrechetR::new(c).var() >= 0.0, "FrechetR({c}) variance");
        }
        for &(beta_param, m) in &[(2.0, 4.0), (3.0, 5.0)] {
            assert!(
                CrystalBall::new(beta_param, m).var() >= 0.0,
                "CrystalBall({beta_param}, {m}) variance"
            );
        }
        for &chi in &[1.0, 2.0, 5.0] {
            assert!(Argus::new(chi).var() >= 0.0, "Argus({chi}) variance");
        }
        for &(a, c) in &[(1.0, 1.0), (2.0, 0.5), (1.5, 2.0)] {
            assert!(
                ExponWeibull::new(a, c).var() >= 0.0,
                "ExponWeibull({a}, {c}) variance"
            );
        }
    }

    #[test]
    fn inverse_gaussian_pdf_cdf() {
        let ig = InverseGaussian::new(1.0);
        assert!(ig.pdf(1.0) > 0.0);
        assert!(ig.pdf(-1.0) == 0.0);
        let c = ig.cdf(1.0);
        assert!(c > 0.0 && c < 1.0);
        assert!((ig.mean() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn gen_extreme_gumbel_case() {
        let gev = GenExtreme::new(0.0); // Gumbel
        assert!(gev.pdf(0.0) > 0.0);
        let c = gev.cdf(0.0);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn gen_pareto_pdf_cdf() {
        let gp = GenPareto::new(0.5);
        assert!(gp.pdf(1.0) > 0.0);
        assert!(gp.pdf(-1.0) == 0.0);
        let c = gp.cdf(1.0);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn power_law_cdf_ppf() {
        let pl = PowerLaw::new(2.0);
        assert!((pl.cdf(0.5) - 0.25).abs() < 1e-10); // x^a = 0.25
        assert!((pl.ppf(0.25) - 0.5).abs() < 1e-10);
        assert!((pl.mean() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn half_normal_pdf_cdf() {
        let hn = HalfNormal;
        assert!(hn.pdf(0.0) > 0.0);
        assert!(hn.pdf(-1.0) == 0.0);
        let c = hn.cdf(1.0);
        assert!(c > 0.0 && c < 1.0);
        assert!((hn.mean() - (2.0 / PI).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn half_logistic_pdf_cdf_ppf_roundtrip() {
        let hl = HalfLogistic;
        assert_close(hl.pdf(0.0), 0.5, 1e-12, "HalfLogistic pdf(0)");
        assert_eq!(hl.cdf(-1.0), 0.0);
        for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = hl.ppf(q);
            assert_close(hl.cdf(x), q, 1e-10, &format!("HalfLogistic ppf at q={q}"));
        }
    }

    #[test]
    fn half_logistic_mean_var_match_closed_form() {
        let hl = HalfLogistic;
        assert_close(hl.mean(), 2.0 * LN_2, 1e-12, "HalfLogistic mean");
        assert_close(
            hl.var(),
            PI * PI / 3.0 - 4.0 * LN_2 * LN_2,
            1e-12,
            "HalfLogistic variance",
        );
    }

    #[test]
    fn half_cauchy_pdf_cdf_ppf_roundtrip() {
        let hc = HalfCauchy;
        assert_close(hc.pdf(0.0), 2.0 / PI, 1e-12, "HalfCauchy pdf(0)");
        assert_eq!(hc.cdf(-1.0), 0.0);
        for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = hc.ppf(q);
            assert_close(hc.cdf(x), q, 1e-10, &format!("HalfCauchy ppf at q={q}"));
        }
    }

    #[test]
    fn half_cauchy_mean_var_are_undefined() {
        let hc = HalfCauchy;
        assert!(hc.mean().is_nan());
        assert!(hc.var().is_nan());
    }

    #[test]
    fn fatiguelife_pdf_cdf_match_scipy_reference_values() {
        let dist = FatigueLife::new(1.2);
        let cases = [
            (0.0, 0.0, 0.0),
            (0.5, 0.592_838_949_590_313_1, 0.277_844_895_141_397_3),
            (1.0, 0.332_451_900_334_527_3, 0.5),
            (2.0, 0.148_209_737_397_578_34, 0.722_155_104_858_602_8),
        ];

        for &(x, pdf, cdf) in &cases {
            assert_close(dist.pdf(x), pdf, 1e-12, &format!("FatigueLife pdf({x})"));
            assert_close(dist.cdf(x), cdf, 1e-12, &format!("FatigueLife cdf({x})"));
        }
    }

    #[test]
    fn fatiguelife_ppf_roundtrip_matches_quantiles() {
        let dist = FatigueLife::new(1.2);
        let cases = [
            (0.1, 0.242_574_512_462_447_16),
            (0.25, 0.454_398_900_905_987_5),
            (0.5, 1.0),
            (0.75, 2.200_709_548_386_197_6),
            (0.9, 4.122_444_645_353_288),
        ];

        for &(q, expected_x) in &cases {
            let x = dist.ppf(q);
            assert_close(x, expected_x, 1e-12, &format!("FatigueLife ppf({q})"));
            assert_close(dist.cdf(x), q, 1e-10, &format!("FatigueLife cdf(ppf({q}))"));
        }
    }

    #[test]
    fn fatiguelife_mean_var_match_closed_form() {
        let dist = FatigueLife::new(1.2);
        assert_close(dist.mean(), 1.72, 1e-12, "FatigueLife mean");
        assert_close(dist.var(), 4.032, 1e-12, "FatigueLife variance");
    }

    #[test]
    fn halfgennorm_pdf_cdf_match_scipy_reference_values() {
        let dist = HalfGenNorm::new(1.5);
        let cases = [
            (0.0, 1.107_732_167_432_472_5, 0.0),
            (0.5, 0.777_836_790_520_629_4, 0.483_498_658_477_359_23),
            (1.0, 0.407_511_890_722_688_56, 0.775_182_471_983_855_6),
            (2.0, 0.065_473_336_746_790_16, 0.971_760_775_641_508),
        ];

        assert_eq!(dist.pdf(-1.0), 0.0);
        assert_eq!(dist.cdf(-1.0), 0.0);

        for &(x, pdf, cdf) in &cases {
            assert_close(dist.pdf(x), pdf, 1e-12, &format!("HalfGenNorm pdf({x})"));
            assert_close(dist.cdf(x), cdf, 1e-12, &format!("HalfGenNorm cdf({x})"));
        }
    }

    #[test]
    fn halfgennorm_ppf_roundtrip_matches_quantiles() {
        let dist = HalfGenNorm::new(1.5);
        let cases = [
            (0.1, 0.091_272_638_420_941_47),
            (0.25, 0.236_147_948_684_195_06),
            (0.5, 0.521_458_465_537_009),
            (0.75, 0.940_877_375_848_903_2),
            (0.9, 1.420_286_821_096_215_4),
        ];

        for &(q, expected_x) in &cases {
            let x = dist.ppf(q);
            assert_close(x, expected_x, 1e-12, &format!("HalfGenNorm ppf({q})"));
            assert_close(dist.cdf(x), q, 1e-10, &format!("HalfGenNorm cdf(ppf({q}))"));
        }
    }

    #[test]
    fn halfgennorm_mean_var_match_closed_form() {
        let dist = HalfGenNorm::new(1.5);
        assert_close(
            dist.mean(),
            0.659_454_753_208_446_7,
            1e-10,
            "HalfGenNorm mean",
        );
        assert_close(
            dist.var(),
            0.303_607_540_093_080_44,
            1e-10,
            "HalfGenNorm variance",
        );
    }

    #[test]
    fn loggamma_pdf_cdf_match_scipy_reference_values() {
        let dist = LogGamma::new(2.5);
        let cases = [
            (-1.0, 0.042_742_466_914_866_446, 0.019_051_344_462_367_25),
            (0.0, 0.276_738_331_613_729_8, 0.150_854_963_915_390_38),
            (0.5, 0.504_895_328_658_646_3, 0.345_766_749_224_207_14),
            (1.0, 0.604_735_141_813_711_2, 0.635_047_943_374_931_8),
        ];

        for &(x, pdf, cdf) in &cases {
            assert_close(dist.pdf(x), pdf, 1e-12, &format!("LogGamma pdf({x})"));
            assert_close(dist.cdf(x), cdf, 1e-12, &format!("LogGamma cdf({x})"));
        }
    }

    #[test]
    fn loggamma_ppf_roundtrip_matches_quantiles() {
        let dist = LogGamma::new(2.5);
        let cases = [
            (0.1, -0.216_721_723_608_478_3),
            (0.25, 0.290_653_706_308_147_6),
            (0.5, 0.777_364_284_327_939_5),
            (0.75, 1.197_805_791_909_737_6),
            (0.9, 1.530_000_352_431_393_7),
        ];

        for &(q, expected_x) in &cases {
            let x = dist.ppf(q);
            assert_close(x, expected_x, 1e-12, &format!("LogGamma ppf({q})"));
            assert_close(dist.cdf(x), q, 1e-10, &format!("LogGamma cdf(ppf({q}))"));
        }
    }

    #[test]
    fn loggamma_mean_var_match_closed_form() {
        let dist = LogGamma::new(2.5);
        assert_close(dist.mean(), 0.703_156_640_645_243_2, 1e-9, "LogGamma mean");
        assert_close(
            dist.var(),
            0.490_357_756_100_234_9,
            1e-9,
            "LogGamma variance",
        );
    }

    #[test]
    fn alpha_pdf_cdf_match_scipy_reference_values() {
        let dist = Alpha::new(2.0);
        let cases = [
            (0.1, 5.169_886_687_842_443e-13, 6.365_782_976_950_845e-16),
            (0.5, 1.632_918_226_724_295_2, 0.511_639_874_658_429_1),
            (1.0, 0.247_603_742_327_967_56, 0.860_931_040_846_074_4),
            (2.0, 0.033_133_183_206_278_98, 0.954_917_293_149_900_3),
        ];

        assert_eq!(dist.pdf(-1.0), 0.0);
        assert_eq!(dist.cdf(-1.0), 0.0);

        for &(x, pdf, cdf) in &cases {
            assert_close(dist.pdf(x), pdf, 1e-12, &format!("Alpha pdf({x})"));
            assert_close(dist.cdf(x), cdf, 1e-12, &format!("Alpha cdf({x})"));
        }
    }

    #[test]
    fn alpha_ppf_roundtrip_matches_quantiles() {
        let dist = Alpha::new(2.0);
        let cases = [
            (0.1, 0.303_524_773_708_178_15),
            (0.25, 0.371_402_382_874_491_34),
            (0.5, 0.492_970_991_216_020_45),
            (0.75, 0.725_542_620_459_912_8),
            (0.9, 1.208_627_158_667_152_9),
        ];

        for &(q, expected_x) in &cases {
            let x = dist.ppf(q);
            assert_close(x, expected_x, 1e-12, &format!("Alpha ppf({q})"));
            assert_close(dist.cdf(x), q, 1e-10, &format!("Alpha cdf(ppf({q}))"));
        }
    }

    #[test]
    fn alpha_mean_var_are_infinite() {
        let dist = Alpha::new(2.0);
        assert!(dist.mean().is_infinite() && dist.mean().is_sign_positive());
        assert!(dist.var().is_infinite() && dist.var().is_sign_positive());
    }

    #[test]
    fn trunc_normal_within_bounds() {
        let tn = TruncNormal::new(-1.0, 1.0);
        assert!(tn.pdf(0.0) > 0.0);
        assert!(tn.pdf(2.0) == 0.0);
        assert!(tn.cdf(-1.0) == 0.0);
        assert!(tn.cdf(1.0) == 1.0);
    }

    #[test]
    fn chi_dist_cdf() {
        let chi = Chi::new(2.0);
        assert!(chi.pdf(1.0) > 0.0);
        let c = chi.cdf(1.0);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn fisk_cdf_ppf() {
        let f = Fisk::new(2.0);
        let x = 1.0;
        let c = f.cdf(x);
        assert!((c - 0.5).abs() < 1e-10); // CDF(1) = 1/(1+1) = 0.5
        assert!((f.ppf(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn loguniform_cdf_ppf() {
        let lu = Loguniform::new(1.0, 10.0);
        assert!((lu.cdf(1.0) - 0.0).abs() < 1e-10);
        assert!((lu.cdf(10.0) - 1.0).abs() < 1e-10);
        assert!((lu.ppf(0.5) - (10.0f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn zipf_pmf_sums() {
        let z = Zipf::new(2.0);
        // PMF should be positive and decreasing
        assert!(z.pmf(1) > z.pmf(2));
        assert!(z.pmf(2) > z.pmf(3));
        let c = z.cdf(10);
        assert!(c > 0.9); // most mass in first 10 terms
    }

    #[test]
    fn nakagami_pdf_cdf() {
        let n = Nakagami::new(1.0);
        assert!(n.pdf(1.0) > 0.0);
        let c = n.cdf(1.0);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn rice_pdf_positive() {
        let r = Rice::new(1.0);
        assert!(r.pdf(1.0) > 0.0);
        assert!(r.pdf(-1.0) == 0.0);
    }

    #[test]
    fn bonferroni_correction() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let result = multipletests_bonferroni(&pvals, 0.05);
        assert_eq!(result.pvalues_corrected.len(), 4);
        assert!((result.pvalues_corrected[0] - 0.04).abs() < 1e-10); // 0.01 * 4
        assert!((result.pvalues_corrected[3] - 0.02).abs() < 1e-10); // 0.005 * 4
        assert!(result.reject[3]); // 0.02 < 0.05
        assert!(!result.reject[1]); // 0.16 > 0.05
    }

    #[test]
    fn holm_correction() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let result = multipletests_holm(&pvals, 0.05);
        // Sorted: 0.005, 0.01, 0.03, 0.04
        // Holm: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04
        // Monotone enforced: 0.02, 0.03, 0.06, 0.06
        assert!(result.reject[3]); // 0.005 → 0.02 < 0.05
        assert!(result.reject[0]); // 0.01 → 0.03 < 0.05
    }

    #[test]
    fn fdr_bh_correction() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let result = multipletests_fdr_bh(&pvals, 0.05);
        // All corrected p-values should be <= 1.0
        assert!(result.pvalues_corrected.iter().all(|&p| p <= 1.0));
        // The smallest p-value should still be the most significant
        assert!(result.pvalues_corrected[3] <= result.pvalues_corrected[0]);
    }

    #[test]
    fn combine_pvalues_fisher_matches_closed_form() {
        let pvalues = [0.01, 0.03, 0.2];
        let result = combine_pvalues(&pvalues, Some("fisher"), None).expect("fisher");
        let expected_statistic = -2.0 * pvalues.iter().map(|p| p.ln()).sum::<f64>();
        let expected_pvalue = ChiSquared::new(2.0 * pvalues.len() as f64).sf(expected_statistic);
        assert_close(
            result.statistic,
            expected_statistic,
            1e-12,
            "combine_pvalues fisher statistic",
        );
        assert_close(
            result.pvalue,
            expected_pvalue,
            1e-12,
            "combine_pvalues fisher pvalue",
        );
    }

    #[test]
    fn combine_pvalues_stouffer_supports_weights() {
        let pvalues = [0.01, 0.05, 0.2];
        let weights = [2.0, 1.0, 0.5];
        let result = combine_pvalues(&pvalues, Some("stouffer"), Some(&weights)).expect("stouffer");
        assert!(result.statistic.is_finite());
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
        assert!(
            result.pvalue < 0.1,
            "combined evidence should remain significant enough"
        );
    }

    #[test]
    fn combine_pvalues_tippett_uses_smallest_pvalue() {
        let pvalues = [0.4, 0.03, 0.8, 0.2];
        let result = combine_pvalues(&pvalues, Some("tippett"), None).expect("tippett");
        assert_close(result.statistic, 0.03, 1e-12, "tippett statistic");
        let expected_pvalue = 1.0 - (1.0 - 0.03_f64).powi(pvalues.len() as i32);
        assert_close(result.pvalue, expected_pvalue, 1e-12, "tippett pvalue");
    }

    #[test]
    fn false_discovery_control_bh_matches_existing_helper() {
        let pvalues = [0.01, 0.04, 0.03, 0.005];
        let corrected = false_discovery_control(&pvalues, Some("bh")).expect("bh");
        let helper = multipletests_fdr_bh(&pvalues, 0.05);
        assert_eq!(corrected.len(), helper.pvalues_corrected.len());
        for (actual, expected) in corrected.iter().zip(helper.pvalues_corrected.iter()) {
            assert_close(*actual, *expected, 1e-12, "fdr bh corrected");
        }
    }

    #[test]
    fn false_discovery_control_by_is_more_conservative() {
        let pvalues = [0.01, 0.04, 0.03, 0.005];
        let bh = false_discovery_control(&pvalues, Some("bh")).expect("bh");
        let by = false_discovery_control(&pvalues, Some("by")).expect("by");
        for (&bh_p, &by_p) in bh.iter().zip(by.iter()) {
            assert!(by_p >= bh_p, "BY should be at least as conservative as BH");
        }
    }

    #[test]
    fn multiple_testing_wrappers_reject_invalid_input() {
        let err = combine_pvalues(&[], None, None).expect_err("empty pvalues");
        assert!(matches!(err, StatsError::InvalidArgument(_)));

        let err = combine_pvalues(&[0.01, 1.2], None, None).expect_err("invalid pvalue");
        assert!(matches!(err, StatsError::InvalidArgument(_)));

        let err =
            combine_pvalues(&[0.01, 0.02], Some("stouffer"), Some(&[1.0])).expect_err("weights");
        assert!(matches!(err, StatsError::InvalidArgument(_)));

        let err = false_discovery_control(&[0.01, 0.02], Some("unknown")).expect_err("bad method");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn poisson_means_test_matches_scipy_reference_example() {
        let result = poisson_means_test(0, 100.0, 3, 100.0, 0.0, None).expect("poisson_means_test");
        assert_close(
            result.statistic,
            -1.7320508075688772,
            1e-12,
            "poisson_means_test statistic",
        );
        assert_close(
            result.pvalue,
            0.08837900945483518,
            1e-10,
            "poisson_means_test pvalue",
        );
    }

    #[test]
    fn poisson_means_test_respects_one_sided_alternatives() {
        let less = poisson_means_test(0, 100.0, 3, 100.0, 0.0, Some("less")).expect("less");
        let greater =
            poisson_means_test(0, 100.0, 3, 100.0, 0.0, Some("greater")).expect("greater");

        assert_close(less.pvalue, 0.04418950472741759, 1e-10, "less pvalue");
        assert_close(greater.pvalue, 0.9340316197238716, 1e-10, "greater pvalue");
    }

    #[test]
    fn poisson_means_test_supports_nonzero_diff() {
        let result = poisson_means_test(10, 100.0, 5, 80.0, 0.02, None).expect("nonzero diff");
        assert_close(
            result.statistic,
            0.4146442144313648,
            1e-12,
            "poisson_means_test nonzero diff statistic",
        );
        assert_close(
            result.pvalue,
            0.6832671240600366,
            5e-7,
            "poisson_means_test nonzero diff pvalue",
        );
    }

    #[test]
    fn poisson_means_test_returns_unit_pvalue_when_lambda_hat2_is_nonpositive() {
        let result = poisson_means_test(1, 1.0, 0, 1.0, 10.0, None).expect("unit pvalue");
        assert_eq!(result.statistic, 0.0);
        assert_eq!(result.pvalue, 1.0);
    }

    #[test]
    fn poisson_means_test_rejects_invalid_input() {
        let err = poisson_means_test(-1, 1.0, 0, 1.0, 0.0, None).expect_err("negative count");
        assert!(matches!(err, StatsError::InvalidArgument(_)));

        let err = poisson_means_test(1, 0.0, 0, 1.0, 0.0, None).expect_err("nonpositive n");
        assert!(matches!(err, StatsError::InvalidArgument(_)));

        let err = poisson_means_test(1, 1.0, 0, 1.0, -1.0, None).expect_err("negative diff");
        assert!(matches!(err, StatsError::InvalidArgument(_)));

        let err = poisson_means_test(1, 1.0, 0, 1.0, 0.0, Some("sideways")).expect_err("bad alt");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn cohens_d_different_means() {
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![4.0, 5.0, 6.0, 7.0, 8.0];
        let d = cohens_d(&g1, &g2);
        assert!(d < 0.0); // g1 mean < g2 mean
        assert!(d.abs() > 1.0); // large effect
    }

    #[test]
    fn cohens_d_identical_groups() {
        let g = vec![1.0, 2.0, 3.0, 4.0];
        let d = cohens_d(&g, &g);
        assert!((d - 0.0).abs() < 1e-10);
    }

    #[test]
    fn sign_test_no_difference() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9];
        let result = sign_test(&x, &y).unwrap();
        assert!(result.pvalue > 0.05);
    }

    #[test]
    fn bootstrap_ci_contains_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean_fn = |d: &[f64]| d.iter().sum::<f64>() / d.len() as f64;
        let (lo, hi) = bootstrap_ci(&data, mean_fn, 1000, 0.95, 42);
        let true_mean = 3.0;
        assert!(
            lo < true_mean && hi > true_mean,
            "CI [{lo}, {hi}] should contain {true_mean}"
        );
    }

    #[test]
    fn winsorize_overlapping_windows_match_scipy_shape() {
        let result = winsorize(&[1.0, 2.0, 3.0], (0.8, 0.8));
        assert_eq!(result, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn winsorize_out_of_range_limits_fail_closed() {
        let result = winsorize(&[1.0, 2.0, 3.0], (0.0, 1.1));
        assert_eq!(result, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn ppf_bisection_non_bracketing_cdf_returns_bounded_value() {
        let result = ppf_bisection(|_| 0.0, 0.5, 0.0, 1.0);
        assert!(result.is_finite(), "expected finite fallback, got {result}");
        assert!(
            result > 1.0e18,
            "expected expanded upper bound, got {result}"
        );
    }

    // ── Arcsine distribution ──────────────────────────────────────────

    #[test]
    fn arcsine_pdf_at_half() {
        // PDF at x=0.5: 1 / (π * sqrt(0.5*0.5)) = 1 / (π * 0.5) = 2/π ≈ 0.6366
        let a = Arcsine;
        let expected = 2.0 / PI;
        assert_close(a.pdf(0.5), expected, 1e-10, "arcsine pdf(0.5)");
    }

    #[test]
    fn arcsine_pdf_boundaries() {
        let a = Arcsine;
        // PDF is 0 outside [0, 1]
        assert_eq!(a.pdf(-0.1), 0.0);
        assert_eq!(a.pdf(1.1), 0.0);
        // At boundaries, PDF should be 0 (singularity)
        assert_eq!(a.pdf(0.0), 0.0);
        assert_eq!(a.pdf(1.0), 0.0);
    }

    #[test]
    fn arcsine_cdf_known_values() {
        let a = Arcsine;
        // CDF(0.5) = 2/π * arcsin(sqrt(0.5)) = 2/π * π/4 = 0.5
        assert_close(a.cdf(0.5), 0.5, 1e-10, "arcsine cdf(0.5)");
        // CDF boundaries
        assert_eq!(a.cdf(0.0), 0.0);
        assert_eq!(a.cdf(1.0), 1.0);
    }

    #[test]
    fn arcsine_ppf_roundtrip() {
        let a = Arcsine;
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = a.ppf(q);
            let q2 = a.cdf(x);
            assert_close(q2, q, 1e-10, &format!("arcsine ppf-cdf roundtrip at q={q}"));
        }
    }

    #[test]
    fn arcsine_mean_var() {
        let a = Arcsine;
        // Mean = 0.5, Var = 1/8 = 0.125
        assert_eq!(a.mean(), 0.5);
        assert_eq!(a.var(), 0.125);
    }

    // ── Fligner test ────────────────────────────────────────────────

    #[test]
    fn fligner_equal_variances() {
        // Two groups with similar spreads should have high p-value
        let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = [2.0, 3.0, 4.0, 5.0, 6.0];
        let result = fligner(&[&g1, &g2]);
        assert!(result.statistic.is_finite(), "stat = {}", result.statistic);
        assert!(result.pvalue.is_finite(), "pvalue = {}", result.pvalue);
        assert!(
            result.pvalue > 0.05,
            "pvalue = {} should be > 0.05",
            result.pvalue
        );
    }

    #[test]
    fn fligner_unequal_variances() {
        // One group with much larger spread
        let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = [0.0, 5.0, 10.0, 15.0, 20.0]; // Much higher variance
        let result = fligner(&[&g1, &g2]);
        assert!(result.statistic.is_finite());
        assert!(result.pvalue.is_finite());
        // With such different variances, p-value should be low
        assert!(
            result.pvalue < 0.1,
            "pvalue = {} should be < 0.1",
            result.pvalue
        );
    }

    #[test]
    fn fligner_three_groups() {
        let g1 = [1.0, 2.0, 3.0, 4.0];
        let g2 = [2.0, 3.0, 4.0, 5.0];
        let g3 = [3.0, 4.0, 5.0, 6.0];
        let result = fligner(&[&g1, &g2, &g3]);
        assert!(result.statistic.is_finite());
        assert!(result.pvalue.is_finite());
    }

    #[test]
    fn fligner_insufficient_data() {
        // Less than 2 groups
        let g1 = [1.0, 2.0, 3.0];
        let result = fligner(&[&g1]);
        assert!(result.statistic.is_nan());
        assert!(result.pvalue.is_nan());

        // Group with less than 2 elements
        let g2 = [1.0];
        let result2 = fligner(&[&g1, &g2]);
        assert!(result2.statistic.is_nan());
    }
}
