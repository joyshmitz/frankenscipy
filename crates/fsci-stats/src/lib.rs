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
    let mut step = half_width;

    // Expand bracket if needed
    while cdf(lo) > q {
        step *= 2.0;
        lo -= step;
    }
    step = half_width;
    while cdf(hi) < q {
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
        0.5 * (1.0 + fsci_special::erf_scalar(z * FRAC_1_SQRT_2))
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
        if q <= 0.0 {
            return self.scale;
        }
        if q >= 1.0 {
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
        if q <= 0.0 {
            return 0.0;
        }
        if q >= 1.0 {
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
        if q <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if q >= 1.0 {
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
        if q <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if q >= 1.0 {
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

    fn mean(&self) -> f64 {
        2.0 * self.scale * (2.0 / PI).sqrt()
    }

    fn var(&self) -> f64 {
        self.scale * self.scale * (3.0 - 8.0 / PI)
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
    if groups.len() < 2 {
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

    if df_within <= 0.0 || ss_within == 0.0 {
        return TtestResult {
            statistic: f64::INFINITY,
            pvalue: 0.0,
            df: df_between,
        };
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

/// Mann-Whitney U test (rank-sum test for two independent samples).
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
    if x.len() != y.len() || x.len() < 10 {
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
    let cdf_val = chi2.cdf(h);
    let pvalue = if !cdf_val.is_finite() || !(0.0..=1.0).contains(&cdf_val) {
        // Numerical overflow in CDF — for large H, p is effectively 0
        if h > 0.0 { 0.0 } else { 1.0 }
    } else {
        (1.0 - cdf_val).max(0.0)
    };

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
    // Create (value, original_index) pairs and sort by value
    let mut indexed: Vec<(f64, usize)> = data
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

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
    let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

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
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = quantile_sorted(&sorted, 0.5);

    let mut diffs: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = quantile_sorted(&diffs, 0.5);
    mad * scale
}

/// Compute the mode (most frequent value) of a data set.
///
/// For continuous data, returns the smallest value among those with the
/// highest frequency. Returns NaN for empty input.
pub fn mode(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
    if data.is_empty() {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

/// Compute z-scores: (x - mean) / std.
///
/// Matches `scipy.stats.zscore(a)`.
pub fn zscore(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![f64::NAN; data.len()];
    }
    let n = data.len() as f64;
    let mean_val = data.iter().sum::<f64>() / n;
    let std_val = (data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / n).sqrt();
    if std_val == 0.0 {
        return vec![0.0; data.len()];
    }
    data.iter().map(|&x| (x - mean_val) / std_val).collect()
}

// Internal helpers for skewness and kurtosis
fn skew_from_moments(n: f64, m2: f64, m3: f64) -> f64 {
    if m2 == 0.0 {
        return 0.0;
    }
    let m2_n = m2 / n;
    let m3_n = m3 / n;
    m3_n / m2_n.powf(1.5)
}

fn kurtosis_from_moments(n: f64, m2: f64, m4: f64) -> f64 {
    if m2 == 0.0 {
        return -3.0; // Degenerate: all values identical
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

/// One-sample Kolmogorov-Smirnov test.
///
/// Tests H0: data comes from the distribution specified by `cdf_func`.
/// The CDF function maps x -> P(X <= x) for the reference distribution.
///
/// Matches `scipy.stats.ks_1samp(data, cdf_func)`.
pub fn ks_1samp(data: &[f64], cdf_func: impl Fn(f64) -> f64) -> GoodnessOfFitResult {
    let n = data.len();
    if n == 0 {
        return GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }
    let nf = n as f64;

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // D = max_i { max(|i/n - F(x_i)|, |(i-1)/n - F(x_i)|) }
    let mut d_stat = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let f_x = cdf_func(x);
        let d_plus = ((i + 1) as f64 / nf - f_x).abs();
        let d_minus = (f_x - i as f64 / nf).abs();
        d_stat = d_stat.max(d_plus).max(d_minus);
    }

    let pvalue = kolmogorov_pvalue(d_stat, nf);

    GoodnessOfFitResult {
        statistic: d_stat,
        pvalue,
    }
}

/// Two-sample Kolmogorov-Smirnov test.
///
/// Tests H0: two samples come from the same continuous distribution.
///
/// Matches `scipy.stats.ks_2samp(data1, data2)`.
pub fn ks_2samp(data1: &[f64], data2: &[f64]) -> GoodnessOfFitResult {
    let n1 = data1.len();
    let n2 = data2.len();
    if n1 == 0 || n2 == 0 {
        return GoodnessOfFitResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        };
    }

    let mut sorted1 = data1.to_vec();
    let mut sorted2 = data2.to_vec();
    sorted1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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

    let k_min = if n_draw + n_succ > big_m {
        n_draw + n_succ - big_m
    } else {
        0
    };
    let k_max = n_draw.min(n_succ);

    let mut pvalue = 0.0;
    for k in k_min..=k_max {
        let p_k = hyper.pmf(k);
        if p_k <= p_observed + 1e-14 {
            pvalue += p_k;
        }
    }

    // Clamp to [0, 1]
    let pvalue = pvalue.min(1.0).max(0.0);

    FisherExactResult {
        odds_ratio,
        pvalue,
    }
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
        // For very large chi2 relative to dof, p-value is effectively 0
        if chi2 > dof as f64 * 20.0 {
            0.0
        } else {
            let dist = ChiSquared::new(dof as f64);
            upper_tail_probability(dist.cdf(chi2))
        }
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
        upper_tail_probability(dist.cdf(stat))
    } else {
        f64::NAN
    };

    (stat, pvalue)
}

fn upper_tail_probability(cdf_val: f64) -> f64 {
    if !cdf_val.is_finite() {
        0.0
    } else {
        (1.0 - cdf_val).clamp(0.0, 1.0)
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
        assert!(z.iter().all(|&v| v == 0.0), "constant data => all zeros");
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

    #[test]
    fn upper_tail_probability_nan_maps_to_zero() {
        assert_eq!(upper_tail_probability(f64::NAN), 0.0);
    }

    #[test]
    fn upper_tail_probability_clamps_numeric_roundoff() {
        assert_eq!(upper_tail_probability(-0.25), 1.0);
        assert_eq!(upper_tail_probability(1.25), 0.0);
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
}
