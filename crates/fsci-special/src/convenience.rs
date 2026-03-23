#![forbid(unsafe_code)]

//! Convenience special functions commonly used in scientific computing.
//!
//! - `sinc` — Normalized sinc function: sin(πx)/(πx)
//! - `xlogy` — x * log(y) with correct handling of x=0
//! - `xlog1py` — x * log1p(y) with correct handling of x=0
//! - `logsumexp` — Log of sum of exponentials (numerically stable)
//! - `expit` — Logistic sigmoid: 1 / (1 + exp(-x))
//! - `logit` — Log-odds: log(p / (1 - p))
//! - `entr` — Elementwise entropy: -x * log(x)
//! - `rel_entr` — Relative entropy (KL divergence element): x * log(x/y)
//! - `ndtr` / `ndtri` — Standard normal CDF and inverse CDF
//! - `kl_div` — KL divergence element with the `-x + y` correction

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind, SpecialResult,
    SpecialTensor, record_special_trace,
};

pub const CONVENIENCE_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "sinc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|x| < 1e-7: Taylor series to avoid 0/0",
            },
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "general: sin(πx)/(πx)",
            },
        ],
        notes: "sinc(0) = 1 by convention. Matches numpy.sinc normalization.",
    },
    DispatchPlan {
        function: "xlogy",
        steps: &[DispatchStep {
            regime: KernelRegime::BackendDelegate,
            when: "direct evaluation with 0*log(y)=0 convention",
        }],
        notes: "Strict mode propagates NaN even when x=0.",
    },
    DispatchPlan {
        function: "rel_entr",
        steps: &[DispatchStep {
            regime: KernelRegime::BackendDelegate,
            when: "direct evaluation using x*log(x/y)",
        }],
        notes: "Matches SciPy domain rules including infinities.",
    },
];

/// Normalized sinc function: sin(πx) / (πx).
///
/// sinc(0) = 1. Matches `numpy.sinc(x)` (not the unnormalized version).
pub fn sinc(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("sinc", x_tensor, mode, |x| Ok(sinc_scalar(x)))
}

/// Compute x * log(y), with the convention that 0 * log(0) = 0.
///
/// Matches `scipy.special.xlogy(x, y)`.
pub fn xlogy(
    x_tensor: &SpecialTensor,
    y_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("xlogy", x_tensor, y_tensor, mode, |x, y| {
        Ok(xlogy_scalar(x, y))
    })
}

/// Compute x * log1p(y), with the convention that 0 * log1p(y) = 0 when x = 0.
///
/// Matches `scipy.special.xlog1py(x, y)`.
pub fn xlog1py(
    x_tensor: &SpecialTensor,
    y_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("xlog1py", x_tensor, y_tensor, mode, |x, y| {
        Ok(xlog1py_scalar(x, y))
    })
}

/// Log of the sum of exponentials (numerically stable).
///
/// logsumexp(a) = log(sum(exp(a))) computed without overflow.
/// Matches `scipy.special.logsumexp(a)`.
pub fn logsumexp(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return max_val;
    }
    let sum_exp: f64 = data.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

/// Logistic sigmoid function: 1 / (1 + exp(-x)).
///
/// Matches `scipy.special.expit(x)`.
pub fn expit(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("expit", x_tensor, mode, |x| Ok(expit_scalar(x)))
}

/// Log-odds function: log(p / (1 - p)).
///
/// Matches `scipy.special.logit(p)`.
/// Domain: p in (0, 1).
pub fn logit(p_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("logit", p_tensor, mode, |p| logit_scalar(p, mode))
}

/// Elementwise entropy: -x * log(x).
///
/// Returns 0 for x = 0, -inf for x < 0.
/// Matches `scipy.special.entr(x)`.
pub fn entr(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("entr", x_tensor, mode, |x| Ok(entr_scalar(x)))
}

/// Relative entropy element: x * log(x/y).
///
/// Matches `scipy.special.rel_entr(x, y)`.
pub fn rel_entr(
    x_tensor: &SpecialTensor,
    y_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("rel_entr", x_tensor, y_tensor, mode, |x, y| {
        Ok(rel_entr_scalar(x, y))
    })
}

/// Standard normal cumulative distribution function Φ(x).
///
/// Matches `scipy.special.ndtr(x)`.
#[must_use]
pub fn ndtr(x: f64) -> f64 {
    0.5 * (1.0 + crate::error::erf_scalar(x / 2.0_f64.sqrt()))
}

/// Inverse standard normal cumulative distribution function Φ⁻¹(y).
///
/// Matches `scipy.special.ndtri(y)`.
#[must_use]
pub fn ndtri(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return f64::NEG_INFINITY;
    }
    if y == 1.0 {
        return f64::INFINITY;
    }
    2.0_f64.sqrt()
        * crate::error::erfinv_scalar(2.0 * y - 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// KL divergence element `x * log(x / y) - x + y`.
///
/// Matches `scipy.special.kl_div(x, y)`.
#[must_use]
pub fn kl_div(x: f64, y: f64) -> f64 {
    kl_div_scalar(x, y)
}

// ══════════════════════════════════════════════════════════════════════
// Scalar Kernels
// ══════════════════════════════════════════════════════════════════════

fn sinc_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    let px = PI * x;
    if px.abs() < 1.0e-7 {
        // Taylor series: sinc(x) ≈ 1 - (πx)²/6 + (πx)⁴/120
        let px2 = px * px;
        1.0 - px2 / 6.0 + px2 * px2 / 120.0
    } else {
        px.sin() / px
    }
}

fn xlogy_scalar(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 { 0.0 } else { x * y.ln() }
}

fn xlog1py_scalar(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 { 0.0 } else { x * (1.0 + y).ln() }
}

fn expit_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        let ex = (-x).exp();
        1.0 / (1.0 + ex)
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

fn logit_scalar(p: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if p.is_nan() {
        return Ok(f64::NAN);
    }
    if p <= 0.0 || p >= 1.0 {
        return match mode {
            RuntimeMode::Strict => {
                if p <= 0.0 {
                    Ok(f64::NEG_INFINITY)
                } else {
                    Ok(f64::INFINITY)
                }
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "logit",
                    mode,
                    "domain_error",
                    format!("p={p}"),
                    "fail_closed",
                    "p must be in (0, 1)",
                    false,
                );
                Err(SpecialError {
                    function: "logit",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "p must be in (0, 1)",
                })
            }
        };
    }
    Ok((p / (1.0 - p)).ln())
}

fn entr_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        0.0
    } else if x < 0.0 {
        f64::NEG_INFINITY
    } else {
        -x * x.ln()
    }
}

fn rel_entr_scalar(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 && y >= 0.0 {
        0.0
    } else if x > 0.0 && y > 0.0 {
        x * (x / y).ln()
    } else {
        f64::INFINITY
    }
}

fn kl_div_scalar(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 && y >= 0.0 {
        y
    } else if x > 0.0 && y > 0.0 {
        x * (x / y).ln() - x + y
    } else {
        f64::INFINITY
    }
}

// ══════════════════════════════════════════════════════════════════════
// Fresnel integrals
// ══════════════════════════════════════════════════════════════════════

/// Fresnel integrals S(z) and C(z).
///
/// S(z) = ∫₀ᶻ sin(πt²/2) dt
/// C(z) = ∫₀ᶻ cos(πt²/2) dt
///
/// Returns (S, C) as a pair of real scalars.
///
/// Uses rational approximation for small z and asymptotic expansion for large z.
pub fn fresnel(z: f64) -> (f64, f64) {
    if z.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let ax = z.abs();
    if ax < 1e-15 {
        return (0.0, 0.0);
    }

    let (s, c) = if ax < 1.6 {
        fresnel_series(ax)
    } else if ax < 6.0 {
        fresnel_mid(ax)
    } else {
        fresnel_asymptotic(ax)
    };

    if z < 0.0 { (-s, -c) } else { (s, c) }
}

/// Power series for Fresnel integrals (small arguments).
fn fresnel_series(x: f64) -> (f64, f64) {
    fresnel_taylor(x)
}

/// Taylor series computation of Fresnel integrals.
fn fresnel_taylor(x: f64) -> (f64, f64) {
    let x2 = x * x;
    let t = std::f64::consts::FRAC_PI_2 * x2;

    // S(x) = x * Σ (-1)^n t^{2n+1} / ((2n+1)!(4n+3))  where t = πx²/2
    // C(x) = x * Σ (-1)^n t^{2n} / ((2n)!(4n+1))
    let mut s = 0.0;
    let mut c = 0.0;
    let mut s_term = t / 3.0; // n=0: t/(1!*3)
    let mut c_term = 1.0; // n=0: 1/(0!*1)

    s += s_term;
    c += c_term;

    for n in 1..60 {
        let nf = n as f64;
        c_term *= -t * t / ((2.0 * nf) * (2.0 * nf - 1.0));
        c_term *= (4.0 * nf - 3.0) / (4.0 * nf + 1.0);
        c += c_term;

        s_term *= -t * t / ((2.0 * nf + 1.0) * (2.0 * nf));
        s_term *= (4.0 * nf - 1.0) / (4.0 * nf + 3.0);
        s += s_term;

        if s_term.abs() < 1e-16 && c_term.abs() < 1e-16 {
            break;
        }
    }

    (x * s, x * c)
}

/// Asymptotic expansion for Fresnel integrals (large arguments).
fn fresnel_asymptotic(x: f64) -> (f64, f64) {
    // Asymptotic: S(x) ≈ 1/2 - f(x)cos(πx²/2) - g(x)sin(πx²/2)
    //             C(x) ≈ 1/2 + f(x)sin(πx²/2) - g(x)cos(πx²/2)

    let pix = std::f64::consts::PI * x;
    let pix2 = pix * x;
    let half_pix2 = std::f64::consts::FRAC_PI_2 * x * x;

    let mut f_term = 1.0;
    let mut g_term = 1.0;
    let mut f = f_term;
    let mut g = g_term;

    for n in 1..20 {
        let nf = n as f64;
        f_term *= -(2.0 * nf) * (2.0 * nf - 1.0) / (pix2 * pix2);
        g_term *= -(2.0 * nf + 1.0) * (2.0 * nf) / (pix2 * pix2);
        f += f_term;
        g += g_term;
        if f_term.abs() < 1e-16 && g_term.abs() < 1e-16 {
            break;
        }
    }

    f /= pix;
    g /= pix2;

    let sin_t = half_pix2.sin();
    let cos_t = half_pix2.cos();

    let s = 0.5 - f * cos_t - g * sin_t;
    let c = 0.5 + f * sin_t - g * cos_t;
    (s, c)
}

/// Mid-range Fresnel integrals (bridge between series and asymptotic).
fn fresnel_mid(x: f64) -> (f64, f64) {
    // For moderate x, use numerical integration via composite Simpson
    let n = 200;
    let h = x / n as f64;
    let half_pi = std::f64::consts::FRAC_PI_2;

    let mut s = 0.0;
    let mut c = 0.0;

    // Simpson's rule
    for i in 0..=n {
        let t = i as f64 * h;
        let arg = half_pi * t * t;
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        s += w * arg.sin();
        c += w * arg.cos();
    }

    s *= h / 3.0;
    c *= h / 3.0;
    (s, c)
}

// ══════════════════════════════════════════════════════════════════════
// Dawson function
// ══════════════════════════════════════════════════════════════════════

/// Dawson function D(x) = exp(-x²) ∫₀ˣ exp(t²) dt.
///
/// Related to the imaginary error function: D(x) = √π/2 * exp(-x²) * erfi(x)
///
/// Uses Rybicki's algorithm with a Gaussian kernel series for efficiency.
pub fn dawsn(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }

    let sign = x.signum();
    let ax = x.abs();

    // For small x, use Taylor series: D(x) ≈ x - 2x³/3 + 4x⁵/15 - ...
    if ax < 0.2 {
        let x2 = ax * ax;
        let result = ax
            * (1.0 - 2.0 * x2 / 3.0 + 4.0 * x2 * x2 / 15.0 - 8.0 * x2 * x2 * x2 / 105.0
                + 16.0 * x2.powi(4) / 945.0);
        return sign * result;
    }

    // Rybicki's algorithm: D(x) ≈ (1/√π) Σ exp(-(x - n*h)²) for suitable h
    // Using the Cephes-style polynomial approximation approach instead
    if ax < 3.9 {
        return sign * dawsn_mid(ax);
    }

    // Asymptotic: D(x) ≈ 1/(2x) + 1/(4x³) + 3/(8x⁵) + ...
    sign * dawsn_asymptotic(ax)
}

/// Dawson function for moderate arguments via numerical integration.
fn dawsn_mid(x: f64) -> f64 {
    // Use composite Simpson's rule for ∫₀ˣ exp(t²-x²) dt
    let n = 200;
    let h = x / n as f64;
    let x2 = x * x;

    let mut sum = 0.0;
    for i in 0..=n {
        let t = i as f64 * h;
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * (t * t - x2).exp();
    }
    sum * h / 3.0
}

/// Dawson function asymptotic expansion for large arguments.
fn dawsn_asymptotic(x: f64) -> f64 {
    let x2 = x * x;
    let inv_2x2 = 0.5 / x2;
    let mut term = 1.0;
    let mut sum = 1.0;

    for n in 1..20 {
        term *= (2 * n - 1) as f64 * inv_2x2;
        sum += term;
        if term.abs() < 1e-16 * sum.abs() {
            break;
        }
    }

    sum / (2.0 * x)
}

// ══════════════════════════════════════════════════════════════════════
// Struve functions
// ══════════════════════════════════════════════════════════════════════

/// Struve function H_v(x) for integer order v.
///
/// H_v(x) = (x/2)^{v+1} Σ_{k=0}^∞ (-1)^k (x/2)^{2k} / (Γ(k+3/2) Γ(k+v+3/2))
///
/// Appears in electromagnetics and acoustics (e.g., radiation impedance).
pub fn struve(v: f64, x: f64) -> f64 {
    if x.is_nan() || v.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x.abs() > 30.0 && v.abs() < x.abs() / 2.0 {
        return struve_asymptotic(v, x);
    }
    struve_series(v, x)
}

/// Modified Struve function L_v(x).
///
/// L_v(x) = -i * exp(-i*v*π/2) * H_v(ix) (for real x, this is real)
/// L_v(x) = (x/2)^{v+1} Σ_{k=0}^∞ (x/2)^{2k} / (Γ(k+3/2) Γ(k+v+3/2))
///
/// Note: same as Struve series but without the (-1)^k alternating sign.
pub fn modstruve(v: f64, x: f64) -> f64 {
    if x.is_nan() || v.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    modstruve_series(v, x)
}

/// Struve function via power series.
fn struve_series(v: f64, x: f64) -> f64 {
    let half_x = x / 2.0;
    let half_x_sq = half_x * half_x;

    // H_v(x) = (x/2)^{v+1} Σ (-1)^k (x/2)^{2k} / (Γ(k+3/2) Γ(k+v+3/2))
    let mut sum = 0.0;
    let mut term = 1.0 / (gamma_fn(1.5) * gamma_fn(v + 1.5));

    for k in 0..100 {
        sum += term;
        let kf = k as f64;
        term *= -half_x_sq / ((kf + 1.5) * (kf + v + 1.5));
        if term.abs() < 1e-16 * sum.abs().max(1e-300) {
            break;
        }
    }

    sum * half_x.powf(v + 1.0)
}

/// Modified Struve function via power series.
fn modstruve_series(v: f64, x: f64) -> f64 {
    let half_x = x / 2.0;
    let half_x_sq = half_x * half_x;

    let mut sum = 0.0;
    let mut term = 1.0 / (gamma_fn(1.5) * gamma_fn(v + 1.5));

    for k in 0..100 {
        sum += term;
        let kf = k as f64;
        term *= half_x_sq / ((kf + 1.5) * (kf + v + 1.5));
        if term.abs() < 1e-16 * sum.abs().max(1e-300) {
            break;
        }
    }

    sum * half_x.powf(v + 1.0)
}

/// Struve asymptotic expansion for large x.
fn struve_asymptotic(v: f64, x: f64) -> f64 {
    // H_v(x) ≈ Y_v(x) + (1/π) Σ Γ(k+1/2) / (Γ(v+1/2-k) (x/2)^{2k-v+1})
    // For large x, approximate using the leading asymptotic behavior
    // H_0(x) ≈ Y_0(x) + 2/(πx) for large x
    let pix = std::f64::consts::PI * x;

    // Simple asymptotic: good for v=0
    if (v - 0.0).abs() < 0.5 {
        // H_0(x) ≈ Y_0(x) + 2/(πx)
        // Y_0(x) ≈ sqrt(2/(πx)) sin(x - π/4)
        let y0_approx = (2.0 / pix).sqrt() * (x - std::f64::consts::FRAC_PI_4).sin();
        return y0_approx + 2.0 / pix;
    }

    // Fall back to series for other orders
    struve_series(v, x)
}

/// Simple gamma function for use in Struve computation.
fn gamma_fn(x: f64) -> f64 {
    // Use Lanczos approximation
    if x <= 0.0 && x.fract().abs() < 1e-14 {
        return f64::INFINITY;
    }
    if (x - 0.5).abs() < 1e-14 {
        return std::f64::consts::PI.sqrt();
    }
    if (x - 1.5).abs() < 1e-14 {
        return std::f64::consts::PI.sqrt() / 2.0;
    }
    if (x - 2.5).abs() < 1e-14 {
        return 3.0 * std::f64::consts::PI.sqrt() / 4.0;
    }

    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        return PI / ((PI * x).sin() * gamma_fn(1.0 - x));
    }

    let z = x - 1.0;
    let mut s = COEFFS[0];
    for (idx, coeff) in COEFFS.iter().enumerate().skip(1) {
        s += coeff / (z + idx as f64);
    }
    let t = z + G + 0.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * s
}

// ══════════════════════════════════════════════════════════════════════
// Number Theory Functions
// ══════════════════════════════════════════════════════════════════════

/// Bernoulli number B_n.
///
/// Returns the nth Bernoulli number. B_0=1, B_1=-1/2, B_2=1/6, B_3=0, ...
/// Odd Bernoulli numbers beyond B_1 are zero.
///
/// Matches `scipy.special.bernoulli(n)`.
pub fn bernoulli(n: u32) -> f64 {
    // Precomputed for small n
    match n {
        0 => 1.0,
        1 => -0.5,
        2 => 1.0 / 6.0,
        4 => -1.0 / 30.0,
        6 => 1.0 / 42.0,
        8 => -1.0 / 30.0,
        10 => 5.0 / 66.0,
        12 => -691.0 / 2730.0,
        14 => 7.0 / 6.0,
        16 => -3617.0 / 510.0,
        18 => 43867.0 / 798.0,
        20 => -174611.0 / 330.0,
        _ => {
            if n % 2 == 1 {
                return 0.0; // Odd Bernoulli numbers (n >= 3) are zero
            }
            // Use the relationship with zeta function:
            // B_{2n} = (-1)^{n+1} * 2 * (2n)! / (2π)^{2n} * ζ(2n)
            let nf = n as f64;
            let sign = if (n / 2) & 1 == 0 { -1.0 } else { 1.0 };
            let zeta_val = crate::gamma::zeta(nf);
            sign * 2.0 * gamma_fn(nf + 1.0) / (2.0 * PI).powf(nf) * zeta_val
        }
    }
}

/// Euler number E_n.
///
/// Returns the nth Euler number. E_0=1, E_1=0, E_2=-1, E_3=0, E_4=5, ...
/// Odd Euler numbers are zero.
///
/// Matches `scipy.special.euler(n)`.
pub fn euler(n: u32) -> f64 {
    if n % 2 == 1 {
        return 0.0;
    }
    match n {
        0 => 1.0,
        2 => -1.0,
        4 => 5.0,
        6 => -61.0,
        8 => 1385.0,
        10 => -50521.0,
        12 => 2702765.0,
        14 => -199360981.0,
        _ => {
            // Compute via alternating sum formula or recurrence
            // E_{2n} = -Σ_{k=0}^{n-1} C(2n, 2k) E_{2k}
            let mut e = vec![0.0; (n / 2 + 1) as usize];
            e[0] = 1.0;
            for m in 1..=(n / 2) as usize {
                let mut sum = 0.0;
                for (k, &ek) in e.iter().enumerate().take(m) {
                    sum += comb_f64(2 * m as u64, 2 * k as u64) * ek;
                }
                e[m] = -sum;
            }
            e[(n / 2) as usize]
        }
    }
}

/// Hurwitz zeta function ζ(s, a) = Σ_{n=0}^∞ 1/(n+a)^s.
///
/// Generalizes the Riemann zeta function: ζ(s) = ζ(s, 1).
///
/// Matches `scipy.special.zeta(s, a)` (the two-argument form).
pub fn hurwitz_zeta(s: f64, a: f64) -> f64 {
    if s.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 {
        return f64::NAN;
    }
    if s <= 1.0 {
        return f64::INFINITY; // Pole at s=1
    }

    // Direct summation for small a or moderate s
    let mut sum = 0.0;
    let max_terms = 10000;
    for n in 0..max_terms {
        let term = 1.0 / (n as f64 + a).powf(s);
        sum += term;
        if term < 1e-16 * sum.abs() && n > 10 {
            break;
        }
    }

    // Euler-Maclaurin correction for the tail
    let n = max_terms as f64;
    let tail = (n + a).powf(1.0 - s) / (s - 1.0);
    sum += tail;

    sum
}

fn comb_f64(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

// ══════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════

fn map_real<F>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError>,
{
    match input {
        SpecialTensor::RealScalar(x) => kernel(*x).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .copied()
            .map(kernel)
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        _ => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "unsupported_type",
                "fail_closed",
                "unsupported input type for convenience scalar function",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "unsupported input type",
            })
        }
    }
}

fn map_real_binary<F>(
    function: &'static str,
    lhs: &SpecialTensor,
    rhs: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<f64, SpecialError>,
{
    match (lhs, rhs) {
        (SpecialTensor::RealScalar(left), SpecialTensor::RealScalar(right)) => {
            kernel(*left, *right).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(left), SpecialTensor::RealScalar(right)) => left
            .iter()
            .copied()
            .map(|value| kernel(value, *right))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(left), SpecialTensor::RealVec(right)) => right
            .iter()
            .copied()
            .map(|value| kernel(*left, value))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(left), SpecialTensor::RealVec(right)) => {
            if left.len() != right.len() {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("lhs_len={},rhs_len={}", left.len(), right.len()),
                    "fail_closed",
                    "vector inputs must have matching lengths",
                    false,
                );
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            left.iter()
                .copied()
                .zip(right.iter().copied())
                .map(|(l, r)| kernel(l, r))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        _ => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "unsupported_types",
                "fail_closed",
                "unsupported input type combination for binary convenience function",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "unsupported input type combination",
            })
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Kelvin Functions
// ══════════════════════════════════════════════════════════════════════

/// Kelvin function ber(x): real part of J_0(x * sqrt(j)).
///
/// Matches `scipy.special.ber`.
pub fn ber(x: f64) -> f64 {
    // ber(x) = Re[J_0(x * e^{jπ/4})] = Σ (-1)^k (x/2)^{4k} / ((2k)!)^2
    let x2 = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..50 {
        term *= -x2 * x2 / ((2 * k - 1) as f64 * (2 * k) as f64).powi(2);
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum
}

/// Kelvin function bei(x): imaginary part of J_0(x * sqrt(j)).
///
/// Matches `scipy.special.bei`.
pub fn bei(x: f64) -> f64 {
    // bei(x) = Im[J_0(x * e^{jπ/4})] = Σ (-1)^k (x/2)^{4k+2} / ((2k+1)!)^2 ... no
    // Actually: bei(x) = Σ_{k=0}^∞ (-1)^k (x/2)^{4k+2} / ((2k)!(2k+1)!) ... no
    // Correct series: bei(x) = Σ_{k=0}^∞ (-1)^k (x/2)^{4k+2} / ((2k+1)!)^2
    // Wait, let me re-derive. J_0(z) = Σ (-1)^n (z/2)^{2n} / (n!)^2
    // With z = x*sqrt(j) = x*e^{jπ/4}:
    // (z/2)^{2n} = (x/2)^{2n} * e^{jnπ/2}
    // e^{jnπ/2} cycles: 1, j, -1, -j, 1, ...
    // J_0(x*e^{jπ/4}) = Σ (-1)^n (x/2)^{2n} / (n!)^2 * e^{jnπ/2}
    //
    // Real parts (n mod 4 == 0: factor 1, n mod 4 == 2: factor -1):
    // ber(x) = Σ_{k=0} (x/2)^{4k}/(2k)!^2 - (x/2)^{4k+2}/((2k+1)!)^2 ... hmm not quite
    //
    // Let me just use: n=0: e^0=1 real, n=1: e^{jπ/2}=j imag, n=2: e^{jπ}=-1 real, n=3: e^{j3π/2}=-j imag
    // So (-1)^n * e^{jnπ/2}: n=0: 1, n=1: -j, n=2: 1, n=3: j, n=4: 1, ...
    // Wait: (-1)^0 * e^0 = 1, (-1)^1 * e^{jπ/2} = -j, (-1)^2 * e^{jπ} = -1, (-1)^3 * e^{j3π/2} = j
    // Hmm that gives: real parts at n=0: 1, n=2: -1, n=4: 1, n=6: -1, ...
    // And: imag parts at n=1: -1, n=3: 1, n=5: -1, n=7: 1, ...
    //
    // ber(x) = 1 - (x/2)^4/(2!)^2 + (x/2)^8/(4!)^2 - ... = Σ_{k=0} (-1)^k (x/2)^{4k} / ((2k)!)^2
    // bei(x) = -(x/2)^2/(1!)^2 + (x/2)^6/(3!)^2 - ... = Σ_{k=0} (-1)^{k+1} (x/2)^{4k+2} / ((2k+1)!)^2
    //
    // Wait that gives bei with a leading negative. Let me reconsider.
    // Actually: (-1)^1 * e^{jπ/2} = (-1)(j) = -j. The imaginary part is -1.
    // (-1)^3 * e^{j3π/2} = (-1)(-j) = j. The imaginary part is +1.
    //
    // bei(x) = Σ_{n odd} (-1)^n (x/2)^{2n} / (n!)^2 * sin(nπ/2)
    // n=1: (-1)^1 * (x/2)^2 / 1 * sin(π/2) = -(x/2)^2
    // n=3: (-1)^3 * (x/2)^6 / 36 * sin(3π/2) = -(x/2)^6/36 * (-1) = (x/2)^6/36
    //
    // So bei(x) = -(x/2)^2 + (x/2)^6/36 - (x/2)^{10}/(5!)^2 + ...
    //           = Σ_{k=0} (-1)^{k+1} (x/2)^{4k+2} / ((2k+1)!)^2

    let x2 = x * x / 4.0; // (x/2)^2
    let mut term = -x2; // first term: -(x/2)^2
    let mut sum = term;
    for k in 1..50 {
        // Ratio: next/current = -(x/2)^4 / ((2k+1) * (2k))^2 ... let me compute:
        // term_k = (-1)^{k+1} (x/2)^{4k+2} / ((2k+1)!)^2
        // term_{k+1}/term_k = -1 * (x/2)^4 / ((2k+2)*(2k+3))^2
        term *= -x2 * x2 / ((2 * k) as f64 * (2 * k + 1) as f64).powi(2);
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum
}

/// Kelvin function ker(x): real part of K_0(x * sqrt(j)).
///
/// Matches `scipy.special.ker`.
pub fn ker(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // For small x, use the series representation:
    // ker(x) = -(ln(x/2) + γ) * ber(x) + (π/4) * bei(x) + Σ h(k) terms
    // where γ is the Euler-Mascheroni constant.
    //
    // For simplicity, use numerical integration of the integral representation:
    // ker(x) = ∫_0^∞ cos(x*sinh(t) - x*cosh(t)·something) dt (complex)
    //
    // Actually, use the relation: ker(x) + j·kei(x) = K_0(x·e^{jπ/4})
    // And K_0 can be computed from the series for small arguments.
    //
    // K_0(z) = -(ln(z/2) + γ) I_0(z) + Σ_{k=0}^∞ (z/2)^{2k} ψ(k+1) / (k!)^2
    // where ψ is the digamma function and ψ(1) = -γ, ψ(k+1) = -γ + Σ_{j=1}^k 1/j

    let gamma_em = 0.577_215_664_901_532_9;
    let x_half = x / 2.0;
    let ln_x2 = x_half.ln();

    // Compute ber(x) and bei(x) for the log term
    let ber_x = ber(x);
    let bei_x = bei(x);

    // ker(x) = -(ln(x/2) + γ) * ber(x) + (π/4) * bei(x) + series_correction
    // The series correction involves harmonic numbers.
    let x2 = x_half * x_half;
    let mut term = 1.0; // (x/2)^0 / (0!)^2 = 1
    let mut harmonic = 0.0; // H_0 = 0
    let mut correction = 0.0; // ψ(1) = -γ, but first term has k=0

    for k in 0..50 {
        if k > 0 {
            term *= x2 * x2 / ((2 * k - 1) as f64 * (2 * k) as f64).powi(2);
            harmonic += 1.0 / (2 * k - 1) as f64 + 1.0 / (2 * k) as f64;
        }
        // The correction uses (H_{2k} from the real part of ψ contributions)
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        correction += sign * term * harmonic;
    }

    -(ln_x2 + gamma_em) * ber_x + (std::f64::consts::PI / 4.0) * bei_x + correction
}

/// Kelvin function kei(x): imaginary part of K_0(x * sqrt(j)).
///
/// Matches `scipy.special.kei`.
pub fn kei(x: f64) -> f64 {
    if x <= 0.0 {
        return -std::f64::consts::PI / 4.0; // kei(0) = -π/4
    }
    // kei(x) = -(ln(x/2) + γ) * bei(x) - (π/4) * ber(x) + series_correction

    let gamma_em = 0.577_215_664_901_532_9;
    let x_half = x / 2.0;
    let ln_x2 = x_half.ln();

    let ber_x = ber(x);
    let bei_x = bei(x);

    let x2 = x_half * x_half;
    let mut term = -x2; // first imaginary series term
    let mut harmonic = 1.0; // H_1 = 1
    let mut correction = term * harmonic;

    for k in 1..50 {
        term *= -x2 * x2 / ((2 * k) as f64 * (2 * k + 1) as f64).powi(2);
        harmonic += 1.0 / (2 * k) as f64 + 1.0 / (2 * k + 1) as f64;
        correction += term * harmonic;
    }

    -(ln_x2 + gamma_em) * bei_x - (std::f64::consts::PI / 4.0) * ber_x + correction
}

// ══════════════════════════════════════════════════════════════════════
// Exponential Integral Variants
// ══════════════════════════════════════════════════════════════════════

/// Generalized exponential integral E_n(x) = ∫_1^∞ t^{-n} e^{-xt} dt.
///
/// Matches `scipy.special.expn`.
pub fn expn(n: usize, x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if n > 1 { 1.0 / (n as f64 - 1.0) } else { f64::INFINITY };
    }
    if n == 0 {
        return (-x).exp() / x;
    }
    if n == 1 {
        // E_1(x) = -Ei(-x) for x > 0
        // Use series for small x, continued fraction for large x
        if x < 1.0 {
            let gamma_em = 0.577_215_664_901_532_9;
            let mut sum = -gamma_em - x.ln();
            let mut term = -x;
            sum += term;
            for k in 2..100 {
                term *= -x / k as f64;
                let contrib = term / k as f64;
                sum += contrib;
                if contrib.abs() < sum.abs() * 1e-16 {
                    break;
                }
            }
            return sum;
        }
        // Continued fraction for E_1(x) when x >= 1
        let mut result = 0.0;
        for k in (1..=20).rev() {
            result = k as f64 / (1.0 + k as f64 / (x + result));
        }
        return (-x).exp() / (x + result);
    }

    // General n > 1: use recurrence E_{n+1}(x) = (e^{-x} - x E_n(x)) / n
    // Start from E_1 and recur upward
    let mut e_prev = expn(1, x);
    for j in 1..n {
        let e_next = ((-x).exp() - x * e_prev) / j as f64;
        e_prev = e_next;
    }
    e_prev
}

/// Exponential integral Ei(x) = PV ∫_{-∞}^{x} e^t/t dt (scalar version).
///
/// Matches `scipy.special.expi` for scalar inputs.
pub fn expi_scalar(x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return -expn(1, -x);
    }
    // For x > 0, use series
    let gamma_em = 0.577_215_664_901_532_9;
    let mut sum = gamma_em + x.ln();
    let mut term = x;
    sum += term;
    for k in 2..200 {
        term *= x / k as f64;
        let contrib = term / k as f64;
        sum += contrib;
        if contrib.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum
}
