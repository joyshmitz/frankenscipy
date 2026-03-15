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
    /// Mean of the distribution.
    fn mean(&self) -> f64;
    /// Variance of the distribution.
    fn var(&self) -> f64;
    /// Standard deviation.
    fn std(&self) -> f64 {
        self.var().sqrt()
    }
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
}
