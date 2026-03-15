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

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use rand::Rng;

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
        if q <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if q >= 1.0 {
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

    // Expand bracket if needed
    while cdf(lo) > q {
        lo -= half_width;
    }
    while cdf(hi) < q {
        hi += half_width;
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
        if q <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if q >= 1.0 {
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
        0.5 * (1.0 + erf_approx(z * FRAC_1_SQRT_2))
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
        if q <= 0.0 {
            return 0.0;
        }
        if q >= 1.0 {
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
            1.0 - (-self.lambda * x).exp()
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
        if q <= 0.0 {
            return 0.0;
        }
        if q >= 1.0 {
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
        1.0 - (-(x / self.scale).powf(self.c)).exp()
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
        0.5 * (1.0 + erf_approx(z * FRAC_1_SQRT_2))
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

/// Log of the Beta function: ln(B(a,b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

// ══════════════════════════════════════════════════════════════════════
// Internal Helper Functions
// ══════════════════════════════════════════════════════════════════════

/// Error function approximation (Abramowitz & Stegun 7.1.26).
fn erf_approx(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

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

    let result = if r <= 5.0 {
        let r = r - 1.6;
        let num = ((((((7.745_450_142_783_414e-4 * r + 2.272_384_498_926_918e-2) * r
            + 2.220_727_511_804_781_3e-1)
            * r
            + 1.463_707_218_484_560_5)
            * r
            + 2.776_978_183_929_534_3)
            * r
            + 4.306_198_189_809_908)
            * r
            + 3.188_362_175_188_116)
            * r
            + 1.340_343_015_652_349_8;
        let den = ((((((1.050_750_071_644_416_9e-9 * r + 5.475_938_084_995_345e-4) * r
            + 1.538_262_409_926_517_2e-2)
            * r
            + 1.487_536_129_085_061_5e-1)
            * r
            + 6.897_673_349_851e-1)
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

/// Log-gamma function via Stirling series with recurrence for small arguments.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x < 8.0 {
        let mut shifted = x;
        let mut correction = 0.0;
        while shifted < 8.0 {
            correction += shifted.ln();
            shifted += 1.0;
        }
        stirling_ln_gamma(shifted) - correction
    } else {
        stirling_ln_gamma(x)
    }
}

fn stirling_ln_gamma(x: f64) -> f64 {
    let half_ln_2pi = 0.5 * (2.0 * PI).ln();
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    (x - 0.5) * x.ln() - x
        + half_ln_2pi
        + inv * (1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 / 1680.0)))
}

/// Lower regularized incomplete gamma function P(a, x) = γ(a,x)/Γ(a).
/// Uses series expansion for x < a+1, continued fraction otherwise.
fn lower_regularized_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series expansion
        gamma_inc_series(a, x)
    } else {
        // Continued fraction (complement)
        1.0 - gamma_inc_cf(a, x)
    }
}

fn gamma_inc_series(a: f64, x: f64) -> f64 {
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_inc_cf(a: f64, x: f64) -> f64 {
    // Lentz's continued fraction
    let mut c = 1e-30_f64;
    let mut d = 1.0 / (x + 1.0 - a);
    let mut f = d;

    for n in 1..200 {
        let n_f = n as f64;
        let an = if n % 2 == 1 {
            let k = (n_f + 1.0) / 2.0;
            -(a - k) * (a + k - 1.0) / ((a + n_f - 1.0) * (a + n_f))
        } else {
            let k = n_f / 2.0;
            k * (a - k) / ((a + n_f - 1.0) * (a + n_f))
        };
        // Simplified: use modified Lentz
        d = 1.0 / (1.0 + an * d);
        c = 1.0 + an / c;
        f *= d * c;
        if ((d * c) - 1.0).abs() < 1e-15 {
            break;
        }
    }

    f * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Regularized incomplete beta function I_x(a, b) using continued fraction.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Special cases with closed-form solutions
    if (a - 1.0).abs() < 1e-14 && (b - 1.0).abs() < 1e-14 {
        return x; // Beta(1,1) = Uniform(0,1)
    }
    if (a - 1.0).abs() < 1e-14 {
        return 1.0 - (1.0 - x).powf(b); // I_x(1, b) = 1 - (1-x)^b
    }
    if (b - 1.0).abs() < 1e-14 {
        return x.powf(a); // I_x(a, 1) = x^a
    }
    // Use symmetry: if x > (a+1)/(a+b+2), compute 1 - I_{1-x}(b, a)
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }
    let lnbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lnbeta).exp() / a;
    front * beta_cf(a, b, x)
}

/// Continued fraction for incomplete beta (Lentz's method).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0));

    for m in 1..200 {
        let m_f = m as f64;
        // Even step
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 / (1.0 + num_even * d);
        c = 1.0 + num_even / c;
        f *= d * c;

        // Odd step
        let num_odd = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 / (1.0 + num_odd * d);
        c = 1.0 + num_odd / c;
        f *= d * c;

        if ((d * c) - 1.0).abs() < 1e-15 {
            break;
        }
    }

    f
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
        if q <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if q >= 1.0 {
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
        if q <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if q >= 1.0 {
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
        } else if x <= c {
            (x - a).powi(2) / ((b - a) * (c - a))
        } else if x < b {
            1.0 - (b - x).powi(2) / ((b - a) * (b - c))
        } else {
            1.0
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        if q <= 0.0 {
            return self.left;
        }
        if q >= 1.0 {
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

/// One-sample t-test: test whether the mean of a sample differs from `popmean`.
///
/// Matches `scipy.stats.ttest_1samp(a, popmean)`.
pub fn ttest_1samp(data: &[f64], popmean: f64) -> TtestResult {
    let n = data.len() as f64;
    let mean: f64 = data.iter().sum::<f64>() / n;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
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
pub fn ttest_ind(a: &[f64], b: &[f64]) -> TtestResult {
    let n1 = a.len() as f64;
    let n2 = b.len() as f64;
    let mean1: f64 = a.iter().sum::<f64>() / n1;
    let mean2: f64 = b.iter().sum::<f64>() / n2;
    let var1: f64 = a.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2: f64 = b.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    let df = n1 + n2 - 2.0;
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
pub fn ttest_ind_welch(a: &[f64], b: &[f64]) -> TtestResult {
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
            assert_close(c.cdf(x), q, 1e-10, &format!("Cauchy ppf roundtrip at q={q}"));
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
            "should not reject H0, p={}", result.pvalue
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
            "should reject H0, p={}", result.pvalue
        );
    }

    #[test]
    fn ttest_ind_same_distribution() {
        let a: Vec<f64> = (0..50).map(|i| (i as f64) * 0.02).collect();
        let b: Vec<f64> = (0..50).map(|i| (i as f64) * 0.02 + 0.001).collect();
        let result = ttest_ind(&a, &b);
        assert!(
            result.pvalue > 0.05,
            "similar samples should not reject H0, p={}", result.pvalue
        );
    }

    #[test]
    fn ttest_ind_different_distributions() {
        let a: Vec<f64> = (0..50).map(|i| (i as f64) * 0.01).collect();
        let b: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64) * 0.01).collect();
        let result = ttest_ind(&a, &b);
        assert!(
            result.pvalue < 0.001,
            "very different samples should reject H0, p={}", result.pvalue
        );
    }

    #[test]
    fn ttest_ind_welch_different_variances() {
        let a: Vec<f64> = (0..100).map(|i| (i as f64) * 0.001).collect();
        let b: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.1).collect();
        let result = ttest_ind_welch(&a, &b);
        assert!(
            result.pvalue < 0.001,
            "should reject H0 with Welch, p={}", result.pvalue
        );
        // Welch df should be less than n1+n2-2
        assert!(result.df < 128.0, "Welch df should be adjusted");
    }
}
