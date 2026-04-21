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
//! - `nrdtrimn` — Recover the normal mean from a CDF value, scale, and quantile
//! - `kl_div` — KL divergence element with the `-x + y` correction

use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

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
    let weights = vec![1.0; data.len()];
    logsumexp_weighted_unchecked(data, &weights)
}

/// Weighted log of the sum of exponentials (numerically stable).
///
/// Matches `scipy.special.logsumexp(a, b=b)`.
pub fn logsumexp_with_b(data: &[f64], b: &[f64]) -> Result<f64, SpecialError> {
    if data.len() != b.len() {
        return Err(SpecialError {
            function: "logsumexp_with_b",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "data and weights must have the same length",
        });
    }
    Ok(logsumexp_weighted_unchecked(data, b))
}

/// Log of the sum of exponentials reduced along a 2D axis.
///
/// Matches `scipy.special.logsumexp(a, axis=axis)` for 2D inputs.
pub fn logsumexp_axis_2d(data: &[Vec<f64>], axis: usize) -> Result<Vec<f64>, SpecialError> {
    logsumexp_axis_2d_impl(data, axis, None)
}

/// Weighted log of the sum of exponentials reduced along a 2D axis.
///
/// Matches `scipy.special.logsumexp(a, axis=axis, b=b)` for 2D inputs, including
/// NumPy-style broadcasting where each weight dimension is either 1 or matches
/// the corresponding data dimension.
pub fn logsumexp_axis_2d_with_b(
    data: &[Vec<f64>],
    axis: usize,
    b: &[Vec<f64>],
) -> Result<Vec<f64>, SpecialError> {
    logsumexp_axis_2d_impl(data, axis, Some(b))
}

fn logsumexp_axis_2d_impl(
    data: &[Vec<f64>],
    axis: usize,
    b: Option<&[Vec<f64>]>,
) -> Result<Vec<f64>, SpecialError> {
    let (rows, cols) = rectangular_shape(data, "logsumexp_axis_2d")?;
    if axis > 1 {
        return Err(SpecialError {
            function: "logsumexp_axis_2d",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "axis must be 0 or 1",
        });
    }

    let weight_shape = b
        .map(|weights| {
            let (weight_rows, weight_cols) =
                rectangular_shape(weights, "logsumexp_axis_2d_with_b")?;
            if !dimension_is_broadcastable(weight_rows, rows)
                || !dimension_is_broadcastable(weight_cols, cols)
            {
                return Err(SpecialError {
                    function: "logsumexp_axis_2d_with_b",
                    kind: SpecialErrorKind::DomainError,
                    mode: RuntimeMode::Strict,
                    detail: "weights are not broadcast-compatible with data",
                });
            }
            Ok((weight_rows, weight_cols))
        })
        .transpose()?;

    match axis {
        0 => {
            let mut reduced = Vec::with_capacity(cols);
            for col in 0..cols {
                let column = data
                    .iter()
                    .map(|row_values| row_values[col])
                    .collect::<Vec<_>>();
                let value =
                    if let (Some(weights), Some((weight_rows, weight_cols))) = (b, weight_shape) {
                        let column_weights = data
                            .iter()
                            .enumerate()
                            .map(|(row, _)| weight_at(weights, weight_rows, weight_cols, row, col))
                            .collect::<Vec<_>>();
                        logsumexp_with_b(&column, &column_weights)?
                    } else {
                        logsumexp(&column)
                    };
                reduced.push(value);
            }
            Ok(reduced)
        }
        1 => {
            let mut reduced = Vec::with_capacity(rows);
            for (row, row_values) in data.iter().enumerate() {
                let value =
                    if let (Some(weights), Some((weight_rows, weight_cols))) = (b, weight_shape) {
                        let row_weights = (0..cols)
                            .map(|col| weight_at(weights, weight_rows, weight_cols, row, col))
                            .collect::<Vec<_>>();
                        logsumexp_with_b(row_values, &row_weights)?
                    } else {
                        logsumexp(row_values)
                    };
                reduced.push(value);
            }
            Ok(reduced)
        }
        _ => unreachable!("axis validated above"),
    }
}

fn logsumexp_weighted_unchecked(data: &[f64], b: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NEG_INFINITY;
    }

    let mut max_val = f64::NEG_INFINITY;
    let mut saw_active_term = false;
    for (&value, &weight) in data.iter().zip(b.iter()) {
        if value.is_nan() || weight.is_nan() {
            return f64::NAN;
        }
        if weight == 0.0 {
            continue;
        }
        saw_active_term = true;
        max_val = max_val.max(value);
    }

    if !saw_active_term || max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }

    if max_val == f64::INFINITY {
        let infinite_weight_sum = data
            .iter()
            .zip(b.iter())
            .filter(|(value, weight)| **weight != 0.0 && **value == f64::INFINITY)
            .map(|(_, weight)| *weight)
            .sum::<f64>();
        return if infinite_weight_sum > 0.0 {
            f64::INFINITY
        } else if infinite_weight_sum == 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }

    let sum_exp = data
        .iter()
        .zip(b.iter())
        .filter(|(_, weight)| **weight != 0.0)
        .map(|(value, weight)| *weight * (*value - max_val).exp())
        .sum::<f64>();
    if sum_exp > 0.0 {
        max_val + sum_exp.ln()
    } else if sum_exp == 0.0 {
        f64::NEG_INFINITY
    } else {
        f64::NAN
    }
}

fn rectangular_shape(
    matrix: &[Vec<f64>],
    function: &'static str,
) -> Result<(usize, usize), SpecialError> {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);
    if matrix.iter().any(|row| row.len() != cols) {
        return Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "matrix rows must all have the same length",
        });
    }
    Ok((rows, cols))
}

fn dimension_is_broadcastable(source: usize, target: usize) -> bool {
    source == target || source == 1 || source == 0 || target == 0
}

fn weight_at(
    weights: &[Vec<f64>],
    weight_rows: usize,
    weight_cols: usize,
    row: usize,
    col: usize,
) -> f64 {
    let source_row = if weight_rows <= 1 { 0 } else { row };
    let source_col = if weight_cols <= 1 { 0 } else { col };
    weights[source_row][source_col]
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
    // Use erfc for improved tail accuracy (avoids catastrophic cancellation for x << 0).
    0.5 * crate::error::erfc_scalar(-x * FRAC_1_SQRT_2)
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
    SQRT_2 * crate::error::erfinv_scalar(2.0 * y - 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Recover the mean of a normal distribution from a CDF value, standard deviation, and quantile.
///
/// Matches `scipy.special.nrdtrimn(p, std, x)`.
#[must_use]
pub fn nrdtrimn(p: f64, std: f64, x: f64) -> f64 {
    if p.is_nan() || std.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if std <= 0.0 || !(0.0 < p && p < 1.0) {
        return f64::NAN;
    }
    if std == f64::INFINITY {
        if x.is_finite() {
            return if p < 0.5 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        }
        if x.is_sign_positive() {
            return if p < 0.5 { f64::INFINITY } else { f64::NAN };
        }
        return if p < 0.5 { f64::NAN } else { f64::NEG_INFINITY };
    }
    x - std * ndtri(p)
}

/// Recover the standard deviation of a normal distribution from a mean, CDF value, and quantile.
///
/// Matches `scipy.special.nrdtrisd(mn, p, x)`.
#[must_use]
pub fn nrdtrisd(mn: f64, p: f64, x: f64) -> f64 {
    const NRDTRISD_P50_DENOM: f64 = 6.637_989_419_862_078e-17;

    if mn.is_nan() || p.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if !(0.0 < p && p < 1.0) {
        return f64::NAN;
    }
    let delta = x - mn;
    if p == 0.5 {
        return delta / NRDTRISD_P50_DENOM;
    }
    delta / ndtri(p)
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
fn dawsn_impl(x: f64) -> f64 {
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
// Sine and Cosine Integrals
// ══════════════════════════════════════════════════════════════════════

/// Sine integral Si(x) and cosine integral Ci(x).
///
/// Si(x) = ∫₀ˣ sin(t)/t dt
/// Ci(x) = γ + ln|x| + ∫₀ˣ (cos(t)-1)/t dt
///
/// where γ ≈ 0.5772... is the Euler-Mascheroni constant.
///
/// Matches `scipy.special.sici(x)` which returns (Si(x), Ci(x)).
///
/// # Arguments
/// * `x` - Real argument
///
/// # Returns
/// Tuple (Si(x), Ci(x))
pub fn sici(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, f64::NEG_INFINITY);
    }

    let ax = x.abs();

    // Series converges for all x, but asymptotic is faster for large x
    // Use series for x < 20 where it converges quickly with good accuracy
    let (si, ci) = if ax < 20.0 {
        sici_series(ax)
    } else {
        sici_asymptotic(ax)
    };

    // Si(-x) = -Si(x), Ci(-x) = Ci(x) + i*π (we ignore imaginary part for real x > 0)
    if x < 0.0 { (-si, ci) } else { (si, ci) }
}

/// Power series for Si and Ci.
fn sici_series(x: f64) -> (f64, f64) {
    const EULER_GAMMA: f64 = 0.5772156649015329;

    let x2 = x * x;

    // Si(x) = x - x³/(3·3!) + x⁵/(5·5!) - x⁷/(7·7!) + ...
    //       = Σ (-1)^n x^{2n+1} / ((2n+1)·(2n+1)!)
    let mut si = x;
    let mut si_term = x;

    for n in 1..150 {
        let nf = n as f64;
        si_term *= -x2 / ((2.0 * nf) * (2.0 * nf + 1.0));
        let contribution = si_term / (2.0 * nf + 1.0);
        si += contribution;
        if contribution.abs() < 1e-16 * si.abs() {
            break;
        }
    }

    // Ci(x) = γ + ln(x) + Σ (-1)^n x^{2n} / ((2n)·(2n)!) for n ≥ 1
    //       = γ + ln(x) - x²/(2·2!) + x⁴/(4·4!) - ...
    let mut ci = EULER_GAMMA + x.ln();
    let mut ci_term = 1.0;

    for n in 1..150 {
        let nf = n as f64;
        ci_term *= -x2 / ((2.0 * nf - 1.0) * (2.0 * nf));
        let contribution = ci_term / (2.0 * nf);
        ci += contribution;
        if contribution.abs() < 1e-16 * ci.abs().max(1.0) {
            break;
        }
    }

    (si, ci)
}

/// Asymptotic expansion for Si and Ci (large arguments).
fn sici_asymptotic(x: f64) -> (f64, f64) {
    // For large x:
    // Si(x) ≈ π/2 - f(x)cos(x) - g(x)sin(x)
    // Ci(x) ≈ f(x)sin(x) - g(x)cos(x)
    // where f(x) = (1/x)[1 - 2!/x² + 4!/x⁴ - ...] (alternating factorials)
    //       g(x) = (1/x²)[1 - 3!/x² + 5!/x⁴ - ...]

    let x_inv = 1.0 / x;
    let x2_inv = x_inv * x_inv;

    let (sin_x, cos_x) = x.sin_cos();
    let half_pi = std::f64::consts::FRAC_PI_2;

    // Compute the auxiliary functions f and g via their series
    let mut f_aux = 0.0;
    let mut g_aux = 0.0;
    let mut term_f = 1.0;
    let mut term_g = 1.0;

    for n in 0..10 {
        f_aux += term_f;
        g_aux += term_g;

        // term_f(n+1) = -term_f(n) * (2n+2)(2n+1) / x²
        // term_g(n+1) = -term_g(n) * (2n+3)(2n+2) / x²
        let nf = n as f64;
        term_f *= -(2.0 * nf + 2.0) * (2.0 * nf + 1.0) * x2_inv;
        term_g *= -(2.0 * nf + 3.0) * (2.0 * nf + 2.0) * x2_inv;

        if term_f.abs() < 1e-15 && term_g.abs() < 1e-15 {
            break;
        }
        if term_f.abs() > 1e8 || term_g.abs() > 1e8 {
            break; // Asymptotic series diverging
        }
    }

    let f_val = f_aux * x_inv;
    let g_val = g_aux * x2_inv;

    let si = half_pi - f_val * cos_x - g_val * sin_x;
    let ci = f_val * sin_x - g_val * cos_x;

    (si, ci)
}

/// Hyperbolic sine integral Shi(x) and hyperbolic cosine integral Chi(x).
///
/// Shi(x) = ∫₀ˣ sinh(t)/t dt
/// Chi(x) = γ + ln|x| + ∫₀ˣ (cosh(t)-1)/t dt
///
/// where γ ≈ 0.5772... is the Euler-Mascheroni constant.
///
/// Matches `scipy.special.shichi(x)` which returns (Shi(x), Chi(x)).
///
/// # Arguments
/// * `x` - Real argument
///
/// # Returns
/// Tuple (Shi(x), Chi(x))
pub fn shichi(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, f64::NEG_INFINITY);
    }

    let ax = x.abs();

    // Shi and Chi are computed using series for all x
    // (they don't have the oscillatory behavior of Si/Ci)
    let (shi, chi) = shichi_series(ax);

    // Shi(-x) = -Shi(x), Chi(-x) = Chi(x)
    if x < 0.0 { (-shi, chi) } else { (shi, chi) }
}

/// Power series for Shi and Chi.
fn shichi_series(x: f64) -> (f64, f64) {
    const EULER_GAMMA: f64 = 0.5772156649015329;

    let x2 = x * x;

    // Shi(x) = x + x³/(3·3!) + x⁵/(5·5!) + x⁷/(7·7!) + ...
    //        = Σ x^{2n+1} / ((2n+1)·(2n+1)!)
    let mut shi = x;
    let mut shi_term = x;

    for n in 1..80 {
        let nf = n as f64;
        shi_term *= x2 / ((2.0 * nf) * (2.0 * nf + 1.0));
        let contribution = shi_term / (2.0 * nf + 1.0);
        shi += contribution;
        if contribution.abs() < 1e-16 * shi.abs() {
            break;
        }
    }

    // Chi(x) = γ + ln(x) + Σ x^{2n} / ((2n)·(2n)!) for n ≥ 1
    //        = γ + ln(x) + x²/(2·2!) + x⁴/(4·4!) + ...
    let mut chi = EULER_GAMMA + x.ln();
    let mut chi_term = 1.0;

    for n in 1..80 {
        let nf = n as f64;
        chi_term *= x2 / ((2.0 * nf - 1.0) * (2.0 * nf));
        let contribution = chi_term / (2.0 * nf);
        chi += contribution;
        if contribution.abs() < 1e-16 * chi.abs().max(1.0) {
            break;
        }
    }

    (shi, chi)
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
    // scipy.special.bei uses the positive-leading Kelvin convention:
    // bei(x) = (x/2)^2 - (x/2)^6/(3!)^2 + (x/2)^10/(5!)^2 - ...
    //        = Σ (-1)^k (x/2)^{4k+2} / ((2k+1)!)^2

    let x2 = x * x / 4.0; // (x/2)^2
    let mut term = x2;
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

    -(ln_x2 + gamma_em) * bei_x - (std::f64::consts::PI / 4.0) * ber_x - correction
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
        return if n > 1 {
            1.0 / (n as f64 - 1.0)
        } else {
            f64::INFINITY
        };
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
            // Series: E_1(x) = -γ - ln(x) - Σ_{k=1}^∞ (-x)^k / (k·k!)
            //       = -γ - ln(x) + x - x²/4 + x³/18 - ...
            let mut term = x; // first term: +x (from -(-x)^1/(1·1!))
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

// ══════════════════════════════════════════════════════════════════════
// Polygamma Functions
// ══════════════════════════════════════════════════════════════════════

/// Trigamma function ψ₁(x) = d²ln(Γ(x))/dx².
///
/// Matches `scipy.special.polygamma(1, x)`.
pub fn trigamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    if x < 0.0 {
        let pi = std::f64::consts::PI;
        let sin_pi_x = (pi * x).sin();
        return (pi * pi) / (sin_pi_x * sin_pi_x) - trigamma(1.0 - x);
    }

    let mut val = x;
    let mut result = 0.0;
    while val < 8.0 {
        result += 1.0 / (val * val);
        val += 1.0;
    }

    let inv_x = 1.0 / val;
    let inv_x2 = inv_x * inv_x;
    result += inv_x + inv_x2 / 2.0 + inv_x2 * inv_x / 6.0 - inv_x2 * inv_x2 * inv_x / 30.0
        + inv_x2 * inv_x2 * inv_x2 * inv_x / 42.0;

    result
}

/// Tetragamma function ψ₂(x) = d³ln(Γ(x))/dx³.
///
/// Matches `scipy.special.polygamma(2, x)`.
pub fn tetragamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }
    if x < 0.0 {
        let pi = std::f64::consts::PI;
        let sin_pi_x = (pi * x).sin();
        let cos_pi_x = (pi * x).cos();
        return tetragamma(1.0 - x)
            - 2.0 * pi * pi * pi * cos_pi_x / (sin_pi_x * sin_pi_x * sin_pi_x);
    }

    let mut val = x;
    let mut result = 0.0;
    while val < 8.0 {
        result -= 2.0 / (val * val * val);
        val += 1.0;
    }

    let inv_x = 1.0 / val;
    let inv_x2 = inv_x * inv_x;
    let inv_x3 = inv_x2 * inv_x;
    result += -inv_x2 - inv_x3 - inv_x2 * inv_x2 / 2.0 + inv_x2 * inv_x2 * inv_x2 / 6.0;

    result
}

/// Digamma function ψ(x) = d(ln Γ(x))/dx (scalar).
pub fn digamma_scalar(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }

    let mut val = x;
    let mut result = 0.0;

    if val < 0.0 {
        result -= std::f64::consts::PI / (std::f64::consts::PI * val).tan();
        val = 1.0 - val;
    }

    while val < 8.0 {
        result -= 1.0 / val;
        val += 1.0;
    }

    let inv_x = 1.0 / val;
    let inv_x2 = inv_x * inv_x;
    result += val.ln() - inv_x / 2.0 - inv_x2 / 12.0 + inv_x2 * inv_x2 / 120.0
        - inv_x2 * inv_x2 * inv_x2 / 252.0;

    result
}

// ══════════════════════════════════════════════════════════════════════
// Combinatorial & Utility Functions
// ══════════════════════════════════════════════════════════════════════

/// Rising factorial (Pochhammer symbol): (x)_n = x(x+1)...(x+n-1).
///
/// Matches `scipy.special.poch`.
pub fn poch(x: f64, n: f64) -> f64 {
    if n == 0.0 {
        return 1.0;
    }
    if n == n.floor() && n > 0.0 && n <= 20.0 {
        let ni = n as usize;
        let mut result = 1.0;
        for k in 0..ni {
            result *= x + k as f64;
        }
        return result;
    }
    let log_result = crate::gammaln_scalar(x + n, fsci_runtime::RuntimeMode::Strict)
        .unwrap_or(f64::NAN)
        - crate::gammaln_scalar(x, fsci_runtime::RuntimeMode::Strict).unwrap_or(f64::NAN);
    log_result.exp()
}

/// Softmax function: exp(x_i) / Σ exp(x_j), numerically stable.
///
/// Matches `scipy.special.softmax`.
pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let exp_x: Vec<f64> = x.iter().map(|&xi| (xi - max_x).exp()).collect();
    let sum_exp: f64 = exp_x.iter().sum();
    exp_x.iter().map(|&e| e / sum_exp).collect()
}

/// Log-softmax: log(softmax(x)), numerically stable.
///
/// Matches `scipy.special.log_softmax`.
pub fn log_softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let shifted: Vec<f64> = x.iter().map(|&xi| xi - max_x).collect();
    let log_sum_exp = shifted.iter().map(|&s| s.exp()).sum::<f64>().ln();
    shifted.iter().map(|&s| s - log_sum_exp).collect()
}

/// Spence's function (dilogarithm): Li₂(z) via integral ∫₁ᶻ ln(t)/(1-t) dt.
///
/// Matches `scipy.special.spence`.
pub fn spence(x: f64) -> f64 {
    if x == 1.0 {
        return 0.0;
    }
    if x == 0.0 {
        return std::f64::consts::PI * std::f64::consts::PI / 6.0;
    }
    if x < 0.0 {
        return f64::NAN;
    }

    // For x > 2, use transformation: spence(x) = -spence(1/x) - ln(x)²/2
    if x > 2.0 {
        return -spence(1.0 / x) - x.ln().powi(2) / 2.0;
    }

    // Numerical integration: ∫₁ˣ ln(t)/(1-t) dt via Simpson's rule
    // Scale grid with interval width for accuracy near singularities
    let n = (500.0 * (1.0 + (x - 1.0).abs())).min(2000.0) as usize;
    let n = n + (n % 2); // ensure even for Simpson
    let a = 1.0;
    let b = x;
    let h = (b - a) / n as f64;
    let f = |t: f64| {
        if (1.0 - t).abs() < 1e-15 {
            -1.0 // L'Hôpital limit: ln(t)/(1-t) → -1 as t → 1
        } else {
            t.ln() / (1.0 - t)
        }
    };

    let mut integral = f(a) + f(b);
    for i in 1..n {
        let t = a + i as f64 * h;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        integral += w * f(t);
    }

    integral * h / 3.0
}

// ══════════════════════════════════════════════════════════════════════
// Additional Special Functions
// ══════════════════════════════════════════════════════════════════════

/// Wright Omega function: solution of y + ln(y) = z.
///
/// Matches `scipy.special.wrightomega`.
pub fn wrightomega(z: f64) -> f64 {
    // Initial guess via Lambert W approximation
    let mut w = if z > 1.0 {
        z - z.ln()
    } else if z > -2.0 {
        z.exp() / (1.0 + z.exp())
    } else {
        z.exp()
    };

    // Newton iteration: f(w) = w + ln(w) - z, f'(w) = 1 + 1/w
    for _ in 0..50 {
        if w <= 0.0 {
            w = 1e-15;
        }
        let residual = w + w.ln() - z;
        if residual.abs() < 1e-15 {
            break;
        }
        w -= residual / (1.0 + 1.0 / w);
    }

    w
}

/// Iterated exponential function (tetration): exp(exp(...exp(x)...)) applied n times.
///
/// exp2(x) = exp(exp(x))
pub fn exp2_iterated(x: f64) -> f64 {
    x.exp().exp()
}

/// Normalized sinc function squared: (sin(πx)/(πx))².
pub fn sinc_squared(x: f64) -> f64 {
    let s = sinc_scalar(x);
    s * s
}

/// Inverse hyperbolic sine (sinh⁻¹): ln(x + √(x²+1)).
///
/// Matches `numpy.arcsinh`.
pub fn arcsinh(x: f64) -> f64 {
    x.asinh()
}

/// Inverse hyperbolic cosine (cosh⁻¹): ln(x + √(x²-1)).
///
/// Matches `numpy.arccosh`.
pub fn arccosh(x: f64) -> f64 {
    x.acosh()
}

/// Inverse hyperbolic tangent (tanh⁻¹): 0.5 * ln((1+x)/(1-x)).
///
/// Matches `numpy.arctanh`.
pub fn arctanh(x: f64) -> f64 {
    x.atanh()
}

/// Compute the squared modulus of the Gamma function |Γ(a + ib)|².
///
/// Useful in scattering theory and other physics applications.
pub fn gamma_mod_squared(a: f64, b: f64) -> f64 {
    // |Γ(a + ib)|² = π * b / (sinh(πb)) * Π_{k=0}^{∞} 1/(1 + b²/(a+k)²)
    // For simplicity, use |Γ(z)|² = Γ(z) * Γ(z̄) = Γ(a+ib) * Γ(a-ib)
    // This equals π / (a * Π_{k=1}^{N} ((a+k)² + b²) / k²) approximately
    //
    // Use the relation: |Γ(a+ib)|² = π*b / (sinh(πb)) for a = 0
    // General case: recurrence + asymptotic
    if b == 0.0 {
        let mode = fsci_runtime::RuntimeMode::Strict;
        let g = crate::gammaln_scalar(a, mode).unwrap_or(f64::NAN);
        return (2.0 * g).exp();
    }

    if a < 0.0 {
        let pi = std::f64::consts::PI;
        let sin_pi_a = (pi * a).sin();
        let cos_pi_a = (pi * a).cos();
        let sinh_pi_b = (pi * b).sinh();
        let cosh_pi_b = (pi * b).cosh();
        let sin_mod_sq = sin_pi_a * sin_pi_a * cosh_pi_b * cosh_pi_b
            + cos_pi_a * cos_pi_a * sinh_pi_b * sinh_pi_b;
        return (pi * pi) / (sin_mod_sq * gamma_mod_squared(1.0 - a, -b));
    }

    // Numerical: use Stirling's approximation shifted to large argument
    let mut val_a = a;
    let mut product = 1.0;
    while val_a < 8.0 {
        product *= val_a * val_a + b * b;
        val_a += 1.0;
    }

    // Stirling: ln|Γ(a+ib)| ≈ (a-0.5)*ln(a²+b²)/2 - b*atan(b/a) - a + 0.5*ln(2π) + ...
    let r2 = val_a * val_a + b * b;
    let theta = b.atan2(val_a);
    let log_mod =
        (val_a - 0.5) * r2.ln() / 2.0 - b * theta - val_a + 0.5 * (2.0 * std::f64::consts::PI).ln();

    (2.0 * log_mod).exp() / product
}

// ══════════════════════════════════════════════════════════════════════
// Convenience Wrappers
// ══════════════════════════════════════════════════════════════════════

/// Log of the binomial coefficient: ln(C(n, k)).
///
/// Matches `scipy.special.gammaln` combination.
pub fn log_comb(n: f64, k: f64) -> f64 {
    let mode = fsci_runtime::RuntimeMode::Strict;
    let lgn1 = crate::gammaln_scalar(n + 1.0, mode).unwrap_or(f64::NAN);
    let lgk1 = crate::gammaln_scalar(k + 1.0, mode).unwrap_or(f64::NAN);
    let lgnk1 = crate::gammaln_scalar(n - k + 1.0, mode).unwrap_or(f64::NAN);
    lgn1 - lgk1 - lgnk1
}

/// Regularized incomplete beta function I_x(a, b).
///
/// Scalar convenience wrapper for `scipy.special.betainc`.
fn betainc_conv(a: f64, b: f64, x: f64) -> f64 {
    crate::betainc_scalar(a, b, x, fsci_runtime::RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse of the regularized incomplete beta function.
///
/// Finds x such that I_x(a, b) = y.
/// Matches `scipy.special.betaincinv`.
pub fn betaincinv(a: f64, b: f64, y: f64) -> f64 {
    if y <= 0.0 {
        return 0.0;
    }
    if y >= 1.0 {
        return 1.0;
    }

    let mode = fsci_runtime::RuntimeMode::Strict;
    let ln_beta = crate::betaln_scalar(a, b, mode).unwrap_or(f64::NAN);

    // Initial guess: use mean of Beta distribution as starting point
    let mut x = a / (a + b);

    // Bracketed Newton with bisection fallback
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;

    for _ in 0..100 {
        let val = betainc_conv(a, b, x);
        let err = val - y;
        if err.abs() < 1e-15 {
            break;
        }

        // Update brackets
        if val < y {
            lo = x;
        } else {
            hi = x;
        }

        // Newton step: dI_x/dx = x^(a-1) * (1-x)^(b-1) / B(a,b)
        let dpx = if x > 0.0 && x < 1.0 {
            ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta).exp()
        } else {
            0.0
        };

        if dpx > 1e-30 {
            let x_new = x - err / dpx;
            if x_new > lo && x_new < hi {
                x = x_new;
            } else {
                x = 0.5 * (lo + hi);
            }
        } else {
            x = 0.5 * (lo + hi);
        }
    }
    x
}

/// Regularized incomplete gamma function P(a, x).
///
/// Scalar wrapper matching `scipy.special.gammainc`.
pub fn gammainc_conv(a: f64, x: f64) -> f64 {
    crate::gammainc_scalar(a, x, fsci_runtime::RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Upper regularized incomplete gamma function Q(a, x) = 1 - P(a, x).
///
/// Scalar wrapper matching `scipy.special.gammaincc`.
pub fn gammaincc_conv(a: f64, x: f64) -> f64 {
    crate::gammaincc_scalar(a, x, fsci_runtime::RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse of the regularized incomplete gamma function.
///
/// Finds x such that P(a, x) = y.
/// Matches `scipy.special.gammaincinv`.
pub fn gammaincinv(a: f64, y: f64) -> f64 {
    if !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return f64::INFINITY;
    }

    let mode = fsci_runtime::RuntimeMode::Strict;
    let ln_gamma_a = crate::gammaln_scalar(a, mode).unwrap_or(f64::NAN);

    // Initial guess using Wilson-Hilferty approximation for chi-squared quantiles
    let x0 = if y < 0.5 {
        // For small y, use inverse of the leading term: P(a,x) ~ x^a / (a * Gamma(a))
        // x ~ (y * a * Gamma(a))^(1/a)
        (y * a * ln_gamma_a.exp()).powf(1.0 / a)
    } else {
        // For larger y, start near a (the mean of Gamma(a,1))
        a
    };

    // Bracketed Newton: maintain [lo, hi] where P(a, lo) < y < P(a, hi)
    let mut lo = 0.0_f64;
    let mut hi = a + 4.0 * a.sqrt() + 10.0; // generous upper bound
    // Expand hi if needed
    while gammainc_conv(a, hi) < y {
        hi *= 2.0;
    }

    let mut x = x0.clamp(lo + 1e-300, hi);

    for _ in 0..100 {
        let p = gammainc_conv(a, x);
        let err = p - y;
        if err.abs() < 1e-14 {
            break;
        }

        // Update brackets
        if p < y {
            lo = x;
        } else {
            hi = x;
        }

        // Newton step: dP/dx = x^(a-1) * e^(-x) / Gamma(a)
        let dpx = x.powf(a - 1.0) * (-x).exp() / ln_gamma_a.exp();
        if dpx.abs() > 1e-30 {
            let x_new = x - err / dpx;
            // Accept Newton step only if it stays in bracket
            if x_new > lo && x_new < hi {
                x = x_new;
            } else {
                x = 0.5 * (lo + hi);
            }
        } else {
            x = 0.5 * (lo + hi);
        }
    }
    x
}

/// Evaluate the complementary error function erfc(x) = 1 - erf(x).
///
/// Scalar convenience wrapper.
pub fn erfc_conv(x: f64) -> f64 {
    crate::erfc_scalar(x)
}

/// Inverse complementary error function.
///
/// Finds x such that erfc(x) = y.
/// Matches `scipy.special.erfcinv`.
pub fn erfcinv_conv(y: f64) -> f64 {
    crate::erfinv_scalar(1.0 - y, fsci_runtime::RuntimeMode::Strict).unwrap_or(f64::NAN)
}

// ══════════════════════════════════════════════════════════════════════
// Additional Special Functions
// ══════════════════════════════════════════════════════════════════════

/// Scaled complementary error function: exp(x²) * erfc(x).
///
/// Avoids overflow for large x. Matches `scipy.special.erfcx`.
pub fn erfcx(x: f64) -> f64 {
    if x < 0.0 {
        // For negative x, exp(x²) * erfc(x) = exp(x²) * (2 - erfc(-x))
        // This can overflow, but erfc(-x) is near 2 and exp(x²) grows
        (x * x).exp() * crate::erfc_scalar(x)
    } else if x < 25.0 {
        (x * x).exp() * crate::erfc_scalar(x)
    } else {
        // Asymptotic: erfcx(x) ≈ 1/(x√π) * (1 - 1/(2x²) + 3/(4x⁴) - ...)
        let inv_x = 1.0 / x;
        let inv_x2 = inv_x * inv_x;
        inv_x / std::f64::consts::PI.sqrt() * (1.0 - 0.5 * inv_x2 + 0.75 * inv_x2 * inv_x2)
    }
}

/// Imaginary error function: erfi(x) = -i * erf(ix) = 2/√π ∫₀ˣ exp(t²) dt.
///
/// Matches `scipy.special.erfi`.
fn erfi_impl(x: f64) -> f64 {
    // erfi(x) = 2x/√π * Σ_{k=0}^∞ x^{2k} / (k! * (2k+1))
    if x.abs() < 6.0 {
        let x2 = x * x;
        let mut term = 1.0;
        let mut sum = 1.0;
        for k in 1..100 {
            term *= x2 / k as f64;
            let contrib = term / (2 * k + 1) as f64;
            sum += contrib;
            if contrib.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        2.0 * x / std::f64::consts::PI.sqrt() * sum
    } else {
        // For large |x|, erfi grows like exp(x²)/(x√π)
        x.signum() * erfcx(-x.abs()) * (x * x).exp()
            - x.signum() / (x.abs() * std::f64::consts::PI.sqrt())
    }
}

pub fn erfi(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("erfi", x_tensor, mode, |x| Ok(erfi_scalar(x)))
}

pub fn erfi_scalar(x: f64) -> f64 {
    erfi_impl(x)
}

/// Owen's T function: T(h, a) = (1/2π) ∫₀ᵃ exp(-h²(1+t²)/2) / (1+t²) dt.
///
/// Used in bivariate normal distribution. Matches `scipy.special.owens_t`.
pub fn owens_t(h: f64, a: f64) -> f64 {
    if a == 0.0 {
        return 0.0;
    }
    if h == 0.0 {
        return a.atan() / (2.0 * std::f64::consts::PI);
    }

    // Numerical integration via Gauss-Legendre (10-point)
    let gl_nodes = [
        -0.973_906_528_517_171_7,
        -0.865_063_366_688_984_5,
        -0.679_409_568_299_024_4,
        -0.433_395_394_129_247_2,
        -0.148_874_338_981_631_2,
        0.148_874_338_981_631_2,
        0.433_395_394_129_247_2,
        0.679_409_568_299_024_4,
        0.865_063_366_688_984_5,
        0.973_906_528_517_171_7,
    ];
    let gl_weights = [
        0.066_671_344_308_688_1,
        0.149_451_349_150_580_6,
        0.219_086_362_515_982,
        0.269_266_719_309_996_4,
        0.295_524_224_714_752_9,
        0.295_524_224_714_752_9,
        0.269_266_719_309_996_4,
        0.219_086_362_515_982,
        0.149_451_349_150_580_6,
        0.066_671_344_308_688_1,
    ];

    let mid = a / 2.0;
    let half = a / 2.0;
    let h2 = h * h;

    let mut sum = 0.0;
    for (&node, &weight) in gl_nodes.iter().zip(gl_weights.iter()) {
        let t = mid + half * node;
        let integrand = (-0.5 * h2 * (1.0 + t * t)).exp() / (1.0 + t * t);
        sum += weight * integrand;
    }

    sum * half / (2.0 * std::f64::consts::PI)
}

/// Relative error exponential: (exp(x) - 1) / x, accurate near x=0.
///
/// Matches `scipy.special.exprel`.
pub fn exprel(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("exprel", x_tensor, mode, |x| Ok(exprel_scalar(x)))
}

pub fn exprel_scalar(x: f64) -> f64 {
    if x.abs() < 1e-5 {
        // Taylor series: 1 + x/2 + x²/6 + x³/24 + ...
        1.0 + x / 2.0 + x * x / 6.0 + x * x * x / 24.0
    } else {
        x.exp_m1() / x
    }
}

/// Box-Cox transformation: (x^λ - 1) / λ for λ ≠ 0, ln(x) for λ = 0.
///
/// Matches `scipy.special.boxcox`.
pub fn boxcox_transform(x: f64, lam: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    if lam.abs() < 1e-15 {
        x.ln()
    } else {
        (x.powf(lam) - 1.0) / lam
    }
}

/// Inverse Box-Cox transformation.
///
/// Matches `scipy.special.inv_boxcox`.
pub fn inv_boxcox(y: f64, lam: f64) -> f64 {
    if lam.abs() < 1e-15 {
        y.exp()
    } else {
        (lam * y + 1.0).powf(1.0 / lam)
    }
}

/// Box-Cox transformation with offset: ((x+1)^λ - 1) / λ.
///
/// Matches `scipy.special.boxcox1p`.
pub fn boxcox1p(x: f64, lam: f64) -> f64 {
    boxcox_transform(1.0 + x, lam)
}

/// Inverse Box-Cox transformation with offset.
///
/// Matches `scipy.special.inv_boxcox1p`.
pub fn inv_boxcox1p(y: f64, lam: f64) -> f64 {
    inv_boxcox(y, lam) - 1.0
}

/// Log of the number of combinations: ln(C(n, k)).
///
/// More numerically stable than computing C(n,k) directly.
/// Matches `scipy.special.gammaln`-based combination counting.
pub fn log_ndtr(x: f64) -> f64 {
    // log(Φ(x)) where Φ is the standard normal CDF
    // For large negative x, use asymptotic to avoid log(tiny)
    if x > 6.0 {
        // Φ(x) ≈ 1, log(1) ≈ 0 with correction
        let t = ndtr(x);
        if t > 0.0 { t.ln() } else { 0.0 }
    } else if x > -20.0 {
        let t = ndtr(x);
        if t > 0.0 { t.ln() } else { f64::NEG_INFINITY }
    } else {
        // Asymptotic: log Φ(x) ≈ -x²/2 - log(-x√(2π)) for x << 0
        -0.5 * x * x - (-x * (2.0 * std::f64::consts::PI).sqrt()).ln()
    }
}

/// Compute the Dawson integral approximation for large arguments.
///
/// For small x, Dawson(x) ≈ x - 2x³/3 + ...
///
/// Matches `scipy.special.dawsn` (scalar convenience).
pub fn dawsn(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("dawsn", x_tensor, mode, |x| Ok(dawsn_scalar(x)))
}

pub fn dawsn_scalar(x: f64) -> f64 {
    dawsn_impl(x)
}

/// Compute the Struve function H_v(x) (scalar convenience).
pub fn struve_scalar(v: f64, x: f64) -> f64 {
    struve(v, x)
}

/// Compute the modified Struve function L_v(x) (scalar convenience).
pub fn modstruve_scalar(v: f64, x: f64) -> f64 {
    modstruve(v, x)
}

/// Compute the Debye function D_n(x) = (n/x^n) ∫₀ˣ t^n/(e^t - 1) dt.
///
/// Matches `scipy.special.debye` for n=1,2,3,4.
pub fn debye(n: usize, x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    if x < 0.0 {
        return f64::NAN;
    }

    // Numerical integration via Simpson's rule
    let npts = (200.0 * (1.0 + x / 5.0).min(10.0)) as usize;
    let npts = npts + (npts % 2);
    let h = x / npts as f64;

    let integrand = |t: f64| -> f64 {
        if t < 1e-15 {
            // L'Hôpital: t^n / (e^t - 1) → t^(n-1) for small t
            t.powi(n as i32 - 1)
        } else {
            t.powi(n as i32) / (t.exp() - 1.0)
        }
    };

    let mut sum = integrand(0.0) + integrand(x);
    for i in 1..npts {
        let t = i as f64 * h;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += w * integrand(t);
    }
    let integral = sum * h / 3.0;

    n as f64 / x.powi(n as i32) * integral
}

/// Lambert W function principal branch W_0(x).
///
/// Finds w such that w * exp(w) = x.
/// Scalar convenience wrapper matching `scipy.special.lambertw`.
pub fn lambertw_scalar(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if (x - (-1.0 / std::f64::consts::E)).abs() < f64::EPSILON {
        return -1.0;
    }
    if x < -1.0 / std::f64::consts::E {
        return f64::NAN;
    }

    // Initial guess
    let mut w = if x < 1.0 { x } else { x.ln() - x.ln().ln() };

    // Halley's method
    for _ in 0..50 {
        let ew = w.exp();
        let wew = w * ew;
        let f = wew - x;
        if f.abs() < 1e-15 * x.abs().max(1.0) {
            break;
        }
        let fp = ew * (w + 1.0);
        let fpp = ew * (w + 2.0);
        w -= f / (fp - f * fpp / (2.0 * fp));
    }

    w
}

/// Zeta function for real s > 1 (convenience alias).
pub fn zeta_scalar(s: f64) -> f64 {
    if s <= 1.0 {
        return f64::NAN;
    }
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

/// Arithmetic-geometric mean of two positive numbers.
pub fn agm(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    let mut an = a;
    let mut bn = b;
    for _ in 0..50 {
        let next_a = (an + bn) / 2.0;
        let next_b = (an * bn).sqrt();
        if (next_a - next_b).abs() < 1e-15 * next_a {
            return next_a;
        }
        an = next_a;
        bn = next_b;
    }
    (an + bn) / 2.0
}

/// Clausen function Cl₂(θ) = Σ sin(kθ)/k².
pub fn clausen(theta: f64) -> f64 {
    let mut sum = 0.0;
    for k in 1..1000 {
        let term = (k as f64 * theta).sin() / (k as f64 * k as f64);
        sum += term;
        if term.abs() < 1e-15 * sum.abs().max(1e-30) {
            break;
        }
    }
    sum
}

/// Central difference derivative.
pub fn central_diff<F>(f: F, x: f64, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Second derivative via central difference.
pub fn central_diff2<F>(f: F, x: f64, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}

/// Gradient of a multivariate function via central differences.
pub fn gradient_approx<F>(f: F, x: &[f64], h: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut grad = Vec::with_capacity(n);
    for i in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        grad.push((f(&xp) - f(&xm)) / (2.0 * h));
    }
    grad
}

/// Jacobian of a vector function via central differences.
pub fn jacobian_approx<F>(f: F, x: &[f64], h: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let f0 = f(x);
    let m = f0.len();
    let mut jac = vec![vec![0.0; n]; m];

    for j in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[j] += h;
        xm[j] -= h;
        let fp = f(&xp);
        let fm = f(&xm);
        for i in 0..m {
            jac[i][j] = (fp[i] - fm[i]) / (2.0 * h);
        }
    }

    jac
}

/// Hessian of a scalar function via central differences.
pub fn hessian_approx<F>(f: F, x: &[f64], h: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut hess = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            let mut xpp = x.to_vec();
            let mut xpm = x.to_vec();
            let mut xmp = x.to_vec();
            let mut xmm = x.to_vec();

            xpp[i] += h;
            xpp[j] += h;
            xpm[i] += h;
            xpm[j] -= h;
            xmp[i] -= h;
            xmp[j] += h;
            xmm[i] -= h;
            xmm[j] -= h;

            hess[i][j] = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * h * h);
            hess[j][i] = hess[i][j];
        }
    }

    hess
}

/// Kolmogorov distribution CDF.
///
/// Computes the complementary CDF of the Kolmogorov distribution,
/// P(D_n > x) where D_n is the Kolmogorov-Smirnov statistic.
///
/// Uses the series: K(x) = 1 - 2 * sum_{k=1}^{inf} (-1)^{k-1} * exp(-2*k^2*x^2)
///
/// Matches `scipy.special.kolmogorov(y)`.
#[must_use]
pub fn kolmogorov(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
    }
    if y <= 0.0 {
        return 1.0;
    }
    if y >= 3.0 {
        // Asymptotic: K(y) ~ 2*exp(-2*y^2) for large y
        return 2.0 * (-2.0 * y * y).exp();
    }

    // Series expansion: K(y) = 1 - 2 * sum_{k=1}^{inf} (-1)^{k-1} * exp(-2*k^2*y^2)
    let y2 = y * y;
    let mut sum = 0.0;
    let mut sign = 1.0;

    for k in 1..100 {
        let kf = k as f64;
        let term = sign * (-2.0 * kf * kf * y2).exp();
        sum += term;
        if term.abs() < 1e-16 * sum.abs().max(1e-30) {
            break;
        }
        sign = -sign;
    }

    2.0 * sum
}

/// Inverse Kolmogorov distribution CDF.
///
/// Returns y such that kolmogorov(y) = p.
///
/// Matches `scipy.special.kolmogi(p)`.
#[must_use]
pub fn kolmogi(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p >= 1.0 {
        return 0.0;
    }

    // Initial guess from asymptotic: p ~ 2*exp(-2*y^2) => y ~ sqrt(-ln(p/2)/2)
    let y0 = if p < 0.5 {
        (-(p / 2.0).ln() / 2.0).sqrt()
    } else {
        0.5 // Start from center for larger p
    };

    // Newton-Raphson iteration
    let mut y = y0;
    for _ in 0..50 {
        let f = kolmogorov(y) - p;
        if f.abs() < 1e-14 {
            break;
        }

        // Derivative: dK/dy = -8*y * sum_{k=1}^{inf} (-1)^{k-1} * k^2 * exp(-2*k^2*y^2)
        let y2 = y * y;
        let mut dsum = 0.0;
        let mut sign = 1.0;
        for k in 1..100 {
            let kf = k as f64;
            let term = sign * kf * kf * (-2.0 * kf * kf * y2).exp();
            dsum += term;
            if term.abs() < 1e-16 {
                break;
            }
            sign = -sign;
        }
        let df = -8.0 * y * dsum;

        if df.abs() < 1e-30 {
            break;
        }

        let delta = f / df;
        y -= delta;
        y = y.max(1e-15);

        if delta.abs() < 1e-14 * y {
            break;
        }
    }

    y
}

/// One-sided Kolmogorov-Smirnov distribution (Smirnov distribution).
///
/// Computes P(D_n^+ > d) where D_n^+ is the one-sided KS statistic
/// for sample size n.
///
/// Uses the asymptotic approximation: P(D_n^+ > d) ≈ exp(-2 * n * d^2)
/// which is accurate for practical purposes.
///
/// Matches `scipy.special.smirnov(n, d)`.
#[must_use]
pub fn smirnov(n: i32, d: f64) -> f64 {
    if n <= 0 || d.is_nan() {
        return f64::NAN;
    }
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }

    let nf = n as f64;

    // Use asymptotic approximation: P(D_n^+ > d) ≈ exp(-2 * n * d^2)
    // This is accurate for most practical purposes
    let x = 2.0 * nf * d * d;

    // Add correction terms for better accuracy at small n
    if n < 20 {
        // For small n, use Birnbaum-Tingey formula with first-order correction
        // P(D_n^+ > d) ≈ exp(-2nd^2) * (1 + O(1/n))
        let correction = 1.0 + (2.0 / 3.0 - d) * d.sqrt() / nf.sqrt();
        return ((-x).exp() * correction).clamp(0.0, 1.0);
    }

    (-x).exp()
}

/// Inverse one-sided Kolmogorov-Smirnov distribution.
///
/// Returns d such that smirnov(n, d) = p.
///
/// Matches `scipy.special.smirnovi(n, p)`.
#[must_use]
pub fn smirnovi(n: i32, p: f64) -> f64 {
    if n <= 0 || p.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }

    let nf = n as f64;

    // Initial guess from asymptotic: p = exp(-2 * n * d^2) => d = sqrt(-ln(p) / (2n))
    let d0 = (-(p.ln()) / (2.0 * nf)).sqrt().min(0.99);

    // Newton-Raphson iteration
    let mut d = d0;
    for _ in 0..50 {
        let f = smirnov(n, d) - p;
        if f.abs() < 1e-14 {
            break;
        }

        // Numerical derivative
        let h = 1e-8 * d.max(1e-8);
        let df = (smirnov(n, d + h) - smirnov(n, d - h)) / (2.0 * h);

        if df.abs() < 1e-30 {
            break;
        }

        let delta = f / df;
        d -= delta;
        d = d.clamp(1e-15, 1.0 - 1e-15);

        if delta.abs() < 1e-14 * d {
            break;
        }
    }

    d
}

/// Cosine of angle given in degrees.
///
/// Matches `scipy.special.cosdg(x)`.
#[must_use]
pub fn cosdg(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Convert degrees to radians and compute cosine
    // Handle exact values for common angles
    let x_mod = x.rem_euclid(360.0);
    if x_mod == 0.0 || x_mod == 360.0 {
        return 1.0;
    }
    if x_mod == 90.0 || x_mod == 270.0 {
        return 0.0;
    }
    if x_mod == 180.0 {
        return -1.0;
    }
    (x * std::f64::consts::PI / 180.0).cos()
}

/// Sine of angle given in degrees.
///
/// Matches `scipy.special.sindg(x)`.
#[must_use]
pub fn sindg(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Handle exact values for common angles
    let x_mod = x.rem_euclid(360.0);
    if x_mod == 0.0 || x_mod == 180.0 || x_mod == 360.0 {
        return 0.0;
    }
    if x_mod == 90.0 {
        return 1.0;
    }
    if x_mod == 270.0 {
        return -1.0;
    }
    (x * std::f64::consts::PI / 180.0).sin()
}

/// Tangent of angle given in degrees.
///
/// Matches `scipy.special.tandg(x)`.
#[must_use]
pub fn tandg(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Handle exact values for common angles
    let x_mod = x.rem_euclid(180.0);
    if x_mod == 0.0 {
        return 0.0;
    }
    if x_mod == 90.0 {
        return f64::INFINITY;
    }
    if x_mod == 45.0 {
        return 1.0;
    }
    if x_mod == 135.0 {
        return -1.0;
    }
    (x * std::f64::consts::PI / 180.0).tan()
}

/// Cotangent of angle given in degrees.
///
/// Matches `scipy.special.cotdg(x)`.
#[must_use]
pub fn cotdg(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Handle exact values for common angles
    let x_mod = x.rem_euclid(180.0);
    if x_mod == 0.0 {
        return f64::INFINITY;
    }
    if x_mod == 90.0 {
        return 0.0;
    }
    if x_mod == 45.0 {
        return 1.0;
    }
    if x_mod == 135.0 {
        return -1.0;
    }
    1.0 / (x * std::f64::consts::PI / 180.0).tan()
}

/// Convert angle from degrees to radians.
///
/// Matches `scipy.special.radian(d, m, s)` where d=degrees, m=minutes, s=seconds.
#[must_use]
pub fn radian(degrees: f64, minutes: f64, seconds: f64) -> f64 {
    if degrees.is_nan() || minutes.is_nan() || seconds.is_nan() {
        return f64::NAN;
    }
    let total_degrees = degrees + minutes / 60.0 + seconds / 3600.0;
    total_degrees * std::f64::consts::PI / 180.0
}

/// Cube root that handles negative numbers correctly.
///
/// Unlike `x.powf(1.0/3.0)`, this returns real values for negative x.
/// cbrt(-8) = -2, not NaN.
///
/// Matches `scipy.special.cbrt(x)`.
#[must_use]
pub fn cbrt(x: f64) -> f64 {
    x.cbrt()
}

/// Base-2 exponential: 2^x.
///
/// Matches `scipy.special.exp2(x)`.
#[must_use]
pub fn exp2(x: f64) -> f64 {
    x.exp2()
}

/// Base-10 exponential: 10^x.
///
/// Matches `scipy.special.exp10(x)`.
#[must_use]
pub fn exp10(x: f64) -> f64 {
    (x * std::f64::consts::LN_10).exp()
}

/// Base-2 logarithm.
///
/// Matches `scipy.special.log2(x)` (numpy ufunc).
#[must_use]
pub fn log2(x: f64) -> f64 {
    x.log2()
}

/// Base-10 logarithm.
///
/// Matches `scipy.special.log10(x)` (numpy ufunc).
#[must_use]
pub fn log10(x: f64) -> f64 {
    x.log10()
}

/// Round to nearest integer.
///
/// Rounds half-way cases away from zero.
/// Matches `scipy.special.round(x)`.
#[must_use]
pub fn round(x: f64) -> f64 {
    x.round()
}

/// Floor function - largest integer not greater than x.
///
/// Matches numpy `floor(x)`.
#[must_use]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// Ceiling function - smallest integer not less than x.
///
/// Matches numpy `ceil(x)`.
#[must_use]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// Truncate towards zero.
///
/// Matches `scipy.special.fix(x)` and numpy `trunc(x)`.
#[must_use]
pub fn trunc(x: f64) -> f64 {
    x.trunc()
}

/// Sign function.
///
/// Returns -1 for x < 0, 0 for x == 0, 1 for x > 0.
/// Returns NaN for NaN input.
///
/// Matches numpy `sign(x)`.
#[must_use]
pub fn sign(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Heaviside step function.
///
/// H(x) = 0 for x < 0, h0 for x == 0, 1 for x > 0.
///
/// Matches numpy `heaviside(x, h0)`.
#[must_use]
pub fn heaviside(x: f64, h0: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        0.0
    } else {
        h0
    }
}

/// Euclidean distance / hypotenuse: sqrt(x² + y²).
///
/// Computed without overflow for large inputs.
/// Matches numpy `hypot(x, y)`.
#[must_use]
pub fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// Copy sign of y to magnitude of x.
///
/// Returns a value with magnitude of x and sign of y.
/// Matches numpy `copysign(x, y)`.
#[must_use]
pub fn copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
}

/// Multiply x by 2 raised to the power exp.
///
/// ldexp(x, exp) = x * 2^exp
/// Matches numpy `ldexp(x, exp)`.
#[must_use]
pub fn ldexp(x: f64, exp: i32) -> f64 {
    // Use the formula: x * 2^exp
    // For safety, use libm's implementation via powi
    x * 2.0_f64.powi(exp)
}

/// Extract mantissa and exponent from x.
///
/// Returns (mantissa, exponent) such that x = mantissa * 2^exponent
/// where 0.5 <= |mantissa| < 1.0 (or mantissa == 0 if x == 0).
///
/// Matches numpy `frexp(x)`.
#[must_use]
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }

    let bits = x.to_bits();
    let sign = bits >> 63;
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let mantissa_bits = bits & 0x000f_ffff_ffff_ffff;

    if exp == 0 {
        // Subnormal number - normalize it
        let normalized = x * 2.0_f64.powi(64);
        let (m, e) = frexp(normalized);
        return (m, e - 64);
    }

    // Normal number: reconstruct mantissa in [0.5, 1.0)
    let new_exp: u64 = 0x3fe; // Exponent for [0.5, 1.0)
    let new_bits = (sign << 63) | (new_exp << 52) | mantissa_bits;
    let mantissa = f64::from_bits(new_bits);

    (mantissa, exp - 0x3fe)
}

/// Absolute value.
///
/// Matches numpy `fabs(x)`.
#[must_use]
pub fn fabs(x: f64) -> f64 {
    x.abs()
}

/// Clip/clamp value to range [min, max].
///
/// Matches numpy `clip(x, min, max)`.
#[must_use]
pub fn clip(x: f64, min: f64, max: f64) -> f64 {
    x.clamp(min, max)
}

/// Sine of π*x with higher accuracy for integer/half-integer arguments.
///
/// sinpi(x) = sin(π*x)
///
/// This function computes sin(π*x) more accurately than sin(PI*x),
/// especially for integer and half-integer values where the result
/// should be exactly 0 or ±1.
///
/// Matches `numpy.sinpi(x)`.
#[must_use]
pub fn sinpi(x: f64) -> f64 {
    use std::f64::consts::PI;

    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::NAN;
    }

    // Reduce to [-1, 1) range for better accuracy
    let mut y = x % 2.0;
    if y > 1.0 {
        y -= 2.0;
    } else if y < -1.0 {
        y += 2.0;
    }

    // For exact integers, return 0
    if y == 0.0 || y == 1.0 || y == -1.0 {
        return 0.0;
    }

    // For half-integers, return ±1
    if y == 0.5 {
        return 1.0;
    }
    if y == -0.5 {
        return -1.0;
    }

    // Use sin(π*y) = sin(π*(1-y)) for y > 0.5
    if y > 0.5 {
        (PI * (1.0 - y)).sin()
    } else if y < -0.5 {
        -(PI * (1.0 + y)).sin()
    } else {
        (PI * y).sin()
    }
}

/// Cosine of π*x with higher accuracy for integer/half-integer arguments.
///
/// cospi(x) = cos(π*x)
///
/// This function computes cos(π*x) more accurately than cos(PI*x),
/// especially for integer and half-integer values where the result
/// should be exactly ±1 or 0.
///
/// Matches `numpy.cospi(x)`.
#[must_use]
pub fn cospi(x: f64) -> f64 {
    use std::f64::consts::PI;

    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::NAN;
    }

    // Reduce to [-1, 1) range
    let mut y = x % 2.0;
    if y > 1.0 {
        y -= 2.0;
    } else if y < -1.0 {
        y += 2.0;
    }

    // For exact integers, return ±1
    if y == 0.0 {
        return 1.0;
    }
    if y == 1.0 || y == -1.0 {
        return -1.0;
    }

    // For half-integers, return 0
    if y == 0.5 || y == -0.5 {
        return 0.0;
    }

    // Use cos(π*y) = -cos(π*(1-y)) for y > 0.5
    if y > 0.5 {
        -(PI * (1.0 - y)).cos()
    } else if y < -0.5 {
        -(PI * (1.0 + y)).cos()
    } else {
        (PI * y).cos()
    }
}

/// Compute log(1 + x) with better accuracy for small x.
///
/// For small x, the direct computation log(1+x) loses precision.
/// This function uses a numerically stable algorithm.
///
/// Matches `numpy.log1p(x)`.
#[must_use]
pub fn log1p(x: f64) -> f64 {
    x.ln_1p()
}

/// Compute exp(x) - 1 with better accuracy for small x.
///
/// For small x, exp(x) is close to 1, so exp(x)-1 loses precision.
/// This function uses a numerically stable algorithm.
///
/// Matches `numpy.expm1(x)`.
#[must_use]
pub fn expm1(x: f64) -> f64 {
    x.exp_m1()
}

/// Compute log(exp(x) + exp(y)) without overflow.
///
/// This is useful for adding probabilities in log-space.
/// logaddexp(x, y) = log(exp(x) + exp(y))
///                 = max(x, y) + log1p(exp(-|x - y|))
///
/// Matches `numpy.logaddexp(x, y)`.
#[must_use]
pub fn logaddexp(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == f64::NEG_INFINITY {
        return y;
    }
    if y == f64::NEG_INFINITY {
        return x;
    }
    if x == f64::INFINITY || y == f64::INFINITY {
        return f64::INFINITY;
    }

    // Use the stable formula: max(x,y) + log1p(exp(-|x-y|))
    let (larger, smaller) = if x >= y { (x, y) } else { (y, x) };
    larger + (smaller - larger).exp().ln_1p()
}

/// Compute log2(2^x + 2^y) without overflow.
///
/// This is useful for adding values in log2-space.
/// logaddexp2(x, y) = log2(2^x + 2^y)
///                  = max(x, y) + log2(1 + 2^{-|x - y|})
///
/// Matches `numpy.logaddexp2(x, y)`.
#[must_use]
pub fn logaddexp2(x: f64, y: f64) -> f64 {
    use std::f64::consts::LN_2;

    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == f64::NEG_INFINITY {
        return y;
    }
    if y == f64::NEG_INFINITY {
        return x;
    }
    if x == f64::INFINITY || y == f64::INFINITY {
        return f64::INFINITY;
    }

    // Use the stable formula: max(x,y) + log2(1 + 2^(-|x-y|))
    let (larger, smaller) = if x >= y { (x, y) } else { (y, x) };
    let diff = smaller - larger;
    // 2^diff = exp(diff * ln(2))
    let two_pow_diff = (diff * LN_2).exp();
    larger + two_pow_diff.ln_1p() / LN_2
}

/// Return the next floating-point value after x towards y.
///
/// If x == y, returns y.
///
/// Matches `numpy.nextafter(x, y)`.
#[must_use]
pub fn nextafter(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == y {
        return y;
    }

    // Use bit manipulation to find the next representable value
    if x == 0.0 {
        // From zero, step toward y
        let tiny = f64::from_bits(1);
        return if y > 0.0 { tiny } else { -tiny };
    }

    let bits = x.to_bits();
    let next_bits = if (y > x) == (x > 0.0) {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next_bits)
}

/// Return the spacing between x and the nearest adjacent number.
///
/// This is the positive distance to the next representable floating
/// point value larger in magnitude than x.
///
/// Matches `numpy.spacing(x)`.
#[must_use]
pub fn spacing(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::NAN;
    }

    let ax = x.abs();
    if ax == 0.0 {
        // Smallest positive subnormal
        return f64::from_bits(1);
    }

    // Spacing is the difference to the next representable number
    let bits = ax.to_bits();
    let next = f64::from_bits(bits + 1);
    next - ax
}

/// Extract the fractional and integer parts of x.
///
/// Returns (fractional, integer) where x = fractional + integer
/// and fractional has the same sign as x with |fractional| < 1.
///
/// Matches `numpy.modf(x)` (which returns (frac, int)).
#[must_use]
pub fn modf(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x.is_infinite() {
        return (0.0_f64.copysign(x), x);
    }

    let int_part = x.trunc();
    let frac_part = x - int_part;
    (frac_part, int_part)
}

/// Test if x is negative (sign bit is set).
///
/// Returns true if sign bit is set, false otherwise.
/// Note: signbit(-0.0) is true, signbit(NaN) depends on the NaN's sign bit.
///
/// Matches `numpy.signbit(x)`.
#[must_use]
pub fn signbit(x: f64) -> bool {
    x.is_sign_negative()
}

/// Test if x is NaN.
///
/// Matches `numpy.isnan(x)`.
#[must_use]
pub fn isnan(x: f64) -> bool {
    x.is_nan()
}

/// Test if x is positive or negative infinity.
///
/// Matches `numpy.isinf(x)`.
#[must_use]
pub fn isinf(x: f64) -> bool {
    x.is_infinite()
}

/// Test if x is finite (not infinity or NaN).
///
/// Matches `numpy.isfinite(x)`.
#[must_use]
pub fn isfinite(x: f64) -> bool {
    x.is_finite()
}

/// Test if x is positive infinity.
///
/// Matches `numpy.isposinf(x)`.
#[must_use]
pub fn isposinf(x: f64) -> bool {
    x == f64::INFINITY
}

/// Test if x is negative infinity.
///
/// Matches `numpy.isneginf(x)`.
#[must_use]
pub fn isneginf(x: f64) -> bool {
    x == f64::NEG_INFINITY
}

/// Return 1/x (reciprocal).
///
/// Matches `numpy.reciprocal(x)`.
#[must_use]
pub fn reciprocal(x: f64) -> f64 {
    1.0 / x
}

/// Return x² (square).
///
/// Matches `numpy.square(x)`.
#[must_use]
pub fn square(x: f64) -> f64 {
    x * x
}

/// Return the positive part of x (max(x, 0)).
///
/// Matches `numpy.positive(x)` conceptually, though numpy.positive
/// just returns x. This matches the mathematical positive part [x]⁺.
#[must_use]
pub fn positive(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Return the negative part of x (max(-x, 0)).
///
/// This matches the mathematical negative part [x]⁻ = max(-x, 0).
#[must_use]
pub fn negative(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x < 0.0 {
        -x
    } else {
        0.0
    }
}

/// Convert degrees to radians.
///
/// deg2rad(x) = x * π / 180
///
/// Matches `numpy.deg2rad(x)` and `numpy.radians(x)`.
#[must_use]
pub fn deg2rad(x: f64) -> f64 {
    x * std::f64::consts::PI / 180.0
}

/// Convert radians to degrees.
///
/// rad2deg(x) = x * 180 / π
///
/// Matches `numpy.rad2deg(x)` and `numpy.degrees(x)`.
#[must_use]
pub fn rad2deg(x: f64) -> f64 {
    x * 180.0 / std::f64::consts::PI
}

/// Round to nearest integer (as float).
///
/// Uses round-half-to-even (banker's rounding) like numpy.rint.
/// This differs from `round` which uses round-half-away-from-zero.
///
/// Matches `numpy.rint(x)`.
#[must_use]
pub fn rint(x: f64) -> f64 {
    // Rust's round_ties_even provides banker's rounding
    x.round_ties_even()
}

/// Round towards zero (truncate to integer, return as float).
///
/// This is equivalent to trunc but emphasizes the "fix" semantic
/// from numpy where values are "fixed" towards zero.
///
/// Matches `numpy.fix(x)`.
#[must_use]
pub fn fix(x: f64) -> f64 {
    x.trunc()
}

/// Return quotient and remainder of division.
///
/// divmod(x, y) returns (floor(x/y), x % y) where the remainder
/// has the same sign as the divisor y (Python/numpy convention).
///
/// Matches `numpy.divmod(x, y)`.
#[must_use]
pub fn divmod(x: f64, y: f64) -> (f64, f64) {
    if y == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if x.is_nan() || y.is_nan() {
        return (f64::NAN, f64::NAN);
    }

    // Floor division and modulo (Python-style)
    let q = (x / y).floor();
    let r = x - q * y;
    (q, r)
}

/// Element-wise maximum, propagating NaNs.
///
/// If either input is NaN, returns NaN.
///
/// Matches `numpy.maximum(x, y)`.
#[must_use]
pub fn maximum(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x >= y {
        x
    } else {
        y
    }
}

/// Element-wise minimum, propagating NaNs.
///
/// If either input is NaN, returns NaN.
///
/// Matches `numpy.minimum(x, y)`.
#[must_use]
pub fn minimum(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x <= y {
        x
    } else {
        y
    }
}

/// Element-wise maximum, ignoring NaNs.
///
/// If one input is NaN, returns the other. If both are NaN, returns NaN.
///
/// Matches `numpy.fmax(x, y)`.
#[must_use]
pub fn fmax(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        y
    } else if y.is_nan() || x >= y {
        x
    } else {
        y
    }
}

/// Element-wise minimum, ignoring NaNs.
///
/// If one input is NaN, returns the other. If both are NaN, returns NaN.
///
/// Matches `numpy.fmin(x, y)`.
#[must_use]
pub fn fmin(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        y
    } else if y.is_nan() || x <= y {
        x
    } else {
        y
    }
}

/// Compute x raised to the power y.
///
/// Handles special cases like 0^0 = 1 and negative bases with
/// non-integer exponents (returns NaN).
///
/// Matches `numpy.power(x, y)`.
#[must_use]
pub fn power(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// Compute the absolute difference |x - y|.
///
/// Useful for computing distances and tolerances.
#[must_use]
pub fn fdiff(x: f64, y: f64) -> f64 {
    (x - y).abs()
}

/// Replace NaN with zero and infinity with large finite numbers.
///
/// nan_to_num(x, nan, posinf, neginf) replaces:
/// - NaN with `nan` (default 0.0)
/// - +inf with `posinf` (default f64::MAX)
/// - -inf with `neginf` (default f64::MIN)
///
/// Matches `numpy.nan_to_num(x)`.
#[must_use]
pub fn nan_to_num(x: f64, nan: f64, posinf: f64, neginf: f64) -> f64 {
    if x.is_nan() {
        nan
    } else if x == f64::INFINITY {
        posinf
    } else if x == f64::NEG_INFINITY {
        neginf
    } else {
        x
    }
}

/// Rectified Linear Unit (ReLU): max(0, x).
///
/// The ReLU activation function, commonly used in neural networks.
///
/// Matches `scipy.special.relu(x)` (proposed).
#[must_use]
pub fn relu(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Softplus: log(1 + exp(x)).
///
/// A smooth approximation to ReLU. Computed in a numerically stable way.
///
/// Matches `scipy.special.softplus(x)` (proposed).
#[must_use]
pub fn softplus(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // For large positive x, softplus(x) ≈ x
    // For large negative x, softplus(x) ≈ exp(x) ≈ 0
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Huber loss function.
///
/// huber(delta, x) =
///   0.5 * x^2                  if |x| <= delta
///   delta * (|x| - 0.5*delta)  if |x| > delta
///
/// A robust loss function that is quadratic for small errors and
/// linear for large errors.
///
/// Matches `scipy.special.huber(delta, x)`.
#[must_use]
pub fn huber(delta: f64, x: f64) -> f64 {
    if delta.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if delta <= 0.0 {
        return f64::NAN;
    }

    let ax = x.abs();
    if ax <= delta {
        0.5 * x * x
    } else {
        delta * (ax - 0.5 * delta)
    }
}

/// Pseudo-Huber loss function.
///
/// pseudo_huber(delta, x) = delta^2 * (sqrt(1 + (x/delta)^2) - 1)
///
/// A smooth approximation to the Huber loss. Unlike Huber, it has
/// continuous derivatives of all orders.
///
/// Matches `scipy.special.pseudo_huber(delta, x)`.
#[must_use]
pub fn pseudo_huber(delta: f64, x: f64) -> f64 {
    if delta.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if delta <= 0.0 {
        return f64::NAN;
    }

    let ratio = x / delta;
    delta * delta * ((1.0 + ratio * ratio).sqrt() - 1.0)
}

/// Exponential Linear Unit (ELU).
///
/// elu(x, alpha) =
///   x                    if x > 0
///   alpha * (exp(x) - 1) if x <= 0
///
/// A smooth activation function that allows negative outputs.
#[must_use]
pub fn elu(x: f64, alpha: f64) -> f64 {
    if x.is_nan() || alpha.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

/// Leaky Rectified Linear Unit.
///
/// leaky_relu(x, alpha) =
///   x         if x > 0
///   alpha * x if x <= 0
///
/// Unlike ReLU, allows small negative values to pass through.
#[must_use]
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x.is_nan() || alpha.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 { x } else { alpha * x }
}

/// Gaussian Error Linear Unit (GELU).
///
/// gelu(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
///
/// A smooth activation function used in transformers and modern NNs.
/// This is the exact formula; for the approximate version, use gelu_approx.
#[must_use]
pub fn gelu(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    let sqrt_2 = std::f64::consts::SQRT_2;
    x * 0.5 * (1.0 + crate::erf_scalar(x / sqrt_2))
}

/// Scaled Exponential Linear Unit (SELU).
///
/// selu(x) = scale * elu(x, alpha)
/// where scale ≈ 1.0507 and alpha ≈ 1.6733
///
/// Self-normalizing activation function.
#[must_use]
pub fn selu(x: f64) -> f64 {
    const ALPHA: f64 = 1.6732632423543772;
    const SCALE: f64 = 1.0507009873554805;

    if x.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 {
        SCALE * x
    } else {
        SCALE * ALPHA * (x.exp() - 1.0)
    }
}

/// Swish activation function.
///
/// swish(x, beta) = x * sigmoid(beta * x) = x / (1 + exp(-beta * x))
///
/// Also known as SiLU (Sigmoid Linear Unit) when beta = 1.
#[must_use]
pub fn swish(x: f64, beta: f64) -> f64 {
    if x.is_nan() || beta.is_nan() {
        return f64::NAN;
    }
    let bx = beta * x;
    if bx >= 0.0 {
        let ex = (-bx).exp();
        x / (1.0 + ex)
    } else {
        let ex = bx.exp();
        x * ex / (1.0 + ex)
    }
}

/// Mish activation function.
///
/// mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
///
/// A self-regularized non-monotonic activation function.
#[must_use]
pub fn mish(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // softplus(x) = ln(1 + exp(x)), computed stably
    let sp = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    };
    x * sp.tanh()
}

/// Hard sigmoid activation function.
///
/// hard_sigmoid(x) = clip((x + 3) / 6, 0, 1)
///
/// A piecewise linear approximation to sigmoid, faster to compute.
#[must_use]
pub fn hard_sigmoid(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

/// Hard swish activation function.
///
/// hard_swish(x) = x * hard_sigmoid(x) = x * clip((x + 3) / 6, 0, 1)
///
/// A piecewise linear approximation to swish, used in MobileNetV3.
#[must_use]
pub fn hard_swish(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x * hard_sigmoid(x)
}

/// Hard tanh activation function.
///
/// hard_tanh(x, min, max) = clip(x, min, max)
///
/// A clipped version of the identity function, approximating tanh.
#[must_use]
pub fn hard_tanh(x: f64, min_val: f64, max_val: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x.clamp(min_val, max_val)
}

/// Log-cosh loss function.
///
/// log_cosh(x) = log(cosh(x))
///
/// A smooth approximation to absolute value loss. For large |x|,
/// log_cosh(x) ≈ |x| - ln(2).
#[must_use]
pub fn log_cosh(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // For numerical stability:
    // log(cosh(x)) = log((exp(x) + exp(-x))/2)
    //              = log(exp(x) + exp(-x)) - log(2)
    // For large |x|: log(cosh(x)) ≈ |x| - log(2)
    let ax = x.abs();
    if ax > 20.0 {
        ax - std::f64::consts::LN_2
    } else {
        x.cosh().ln()
    }
}

/// Softsign activation function.
///
/// softsign(x) = x / (1 + |x|)
///
/// A smooth, bounded activation function similar to tanh but with
/// slower saturation.
#[must_use]
pub fn softsign(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x / (1.0 + x.abs())
}

/// Threshold function.
///
/// threshold(x, threshold, value) =
///   x     if x > threshold
///   value otherwise
///
/// A simple step function with configurable threshold and fill value.
#[must_use]
pub fn threshold(x: f64, thresh: f64, value: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > thresh { x } else { value }
}

/// Sigmoid Linear Unit (SiLU), same as swish with beta=1.
///
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// Also known as swish-1.
#[must_use]
pub fn silu(x: f64) -> f64 {
    swish(x, 1.0)
}

/// Log-expit function (log of logistic sigmoid).
///
/// log_expit(x) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
///
/// Numerically stable computation of log(expit(x)).
#[must_use]
pub fn log_expit_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // log(expit(x)) = -log(1 + exp(-x)) = -softplus(-x)
    -softplus(-x)
}

/// Complementary log-log function.
///
/// cloglog(p) = log(-log(1 - p))
///
/// The cloglog link function, used in survival analysis and
/// generalized linear models. Maps (0, 1) to (-∞, +∞).
///
/// Returns -∞ for p ≤ 0, +∞ for p ≥ 1.
#[must_use]
pub fn cloglog(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Use log1p for numerical stability when p is near 0
    if p < 0.5 {
        // log(-log(1-p)) where 1-p is close to 1
        // ln_1p(-p) = ln(1-p), then negate and take ln
        (-((-p).ln_1p())).ln()
    } else {
        (-((1.0 - p).ln())).ln()
    }
}

/// Inverse complementary log-log function.
///
/// cloglog_inv(x) = 1 - exp(-exp(x))
///
/// The inverse of cloglog. Maps (-∞, +∞) to (0, 1).
/// Also known as the Gumbel CDF.
#[must_use]
pub fn cloglog_inv(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // 1 - exp(-exp(x))
    // Use expm1 for numerical stability
    if x < -37.0 {
        // exp(x) is essentially 0, so 1 - exp(-0) = 0
        0.0
    } else if x > 709.0 {
        // exp(x) overflows, result is 1
        1.0
    } else {
        -(-x.exp()).exp_m1()
    }
}

/// Log-log link function.
///
/// loglog(p) = -log(-log(p))
///
/// The log-log link function, used in extreme value distributions.
/// Maps (0, 1) to (-∞, +∞). The negative of the Gumbel quantile function.
///
/// Returns -∞ for p ≤ 0, +∞ for p ≥ 1.
#[must_use]
pub fn loglog(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    -(-p.ln()).ln()
}

/// Inverse log-log link function.
///
/// loglog_inv(x) = exp(-exp(-x))
///
/// The inverse of loglog. Maps (-∞, +∞) to (0, 1).
/// This is the standard Gumbel (minimum) CDF.
#[must_use]
pub fn loglog_inv(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // exp(-exp(-x))
    if x < -709.0 {
        // exp(-x) overflows, exp(-inf) = 0
        0.0
    } else if x > 37.0 {
        // exp(-x) is essentially 0, exp(-0) = 1
        1.0
    } else {
        (-(-x).exp()).exp()
    }
}

/// Cauchy link function (cauchit).
///
/// cauchit(p) = tan(π * (p - 0.5))
///
/// The inverse Cauchy CDF, used as a link function for heavy-tailed
/// distributions. Maps (0, 1) to (-∞, +∞).
///
/// Returns -∞ for p ≤ 0, +∞ for p ≥ 1.
#[must_use]
pub fn cauchit(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    (std::f64::consts::PI * (p - 0.5)).tan()
}

/// Inverse Cauchy link function.
///
/// cauchit_inv(x) = 0.5 + arctan(x) / π
///
/// The Cauchy CDF. Maps (-∞, +∞) to (0, 1).
#[must_use]
pub fn cauchit_inv(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    0.5 + x.atan() / std::f64::consts::PI
}

/// Hard shrinkage function.
///
/// hardshrink(x, λ) = x if |x| > λ, else 0
///
/// A thresholding function that sets values with magnitude below λ to zero.
/// Used in sparse signal processing and neural networks.
#[must_use]
pub fn hardshrink(x: f64, lambda: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() > lambda { x } else { 0.0 }
}

/// Soft shrinkage function (soft thresholding).
///
/// softshrink(x, λ) = sign(x) * max(|x| - λ, 0)
///                  = x - λ  if x > λ
///                  = x + λ  if x < -λ
///                  = 0      otherwise
///
/// Shrinks values toward zero by λ. Used in LASSO regression,
/// wavelet denoising, and neural networks.
#[must_use]
pub fn softshrink(x: f64, lambda: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Tanh shrinkage function.
///
/// tanhshrink(x) = x - tanh(x)
///
/// A smooth shrinkage function that subtracts the bounded tanh.
/// Approaches 0 for small x, approaches x for large |x|.
#[must_use]
pub fn tanhshrink(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x - x.tanh()
}

/// CELU activation function (Continuously Differentiable ELU).
///
/// celu(x, α) = max(0, x) + min(0, α * (exp(x/α) - 1))
///
/// A continuously differentiable variant of ELU. Unlike ELU,
/// CELU is C¹ continuous (has continuous first derivative).
#[must_use]
pub fn celu(x: f64, alpha: f64) -> f64 {
    if x.is_nan() || alpha.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        x
    } else {
        alpha * ((x / alpha).exp() - 1.0)
    }
}

/// LogSigmoid activation function.
///
/// logsigmoid(x) = log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -softplus(-x)
///
/// Numerically stable log of the sigmoid function.
/// Equivalent to log_expit_scalar but with a more common ML name.
#[must_use]
pub fn logsigmoid(x: f64) -> f64 {
    log_expit_scalar(x)
}

/// Log of 1 minus exp(x), computed in a numerically stable way.
///
/// log1mexp(x) = log(1 - exp(x))
///
/// For x < 0, this computes log(1 - exp(x)) stably.
/// Uses log1p for x close to 0 and direct computation otherwise.
///
/// Returns NaN for x > 0 (since 1 - exp(x) < 0).
#[must_use]
pub fn log1mexp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 {
        return f64::NAN; // log of negative number
    }
    if x == 0.0 {
        return f64::NEG_INFINITY; // log(0)
    }
    // For x < 0:
    // If x is close to 0 (say x > -0.693 = -ln(2)), use log1p(-exp(x))
    // Otherwise use log(1 - exp(x)) directly
    if x > -std::f64::consts::LN_2 {
        // x close to 0: exp(x) close to 1, use log1p for accuracy
        (-x.exp()).ln_1p()
    } else {
        // x far from 0: exp(x) small, direct computation is fine
        (1.0 - x.exp()).ln()
    }
}

/// Log of 1 plus exp(x), computed in a numerically stable way.
///
/// log1pexp(x) = log(1 + exp(x))
///
/// This is the same as softplus(x). Provided as an alias for
/// compatibility with other libraries.
#[must_use]
pub fn log1pexp(x: f64) -> f64 {
    softplus(x)
}

/// x * log(x) with proper handling of x = 0.
///
/// xlogx(x) = x * log(x) for x > 0
///          = 0         for x = 0
///          = NaN       for x < 0
///
/// Used in entropy calculations where 0 * log(0) = 0 by convention.
#[must_use]
pub fn xlogx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0; // By L'Hôpital's rule, lim x*log(x) as x->0+ = 0
    }
    x * x.ln()
}

/// Negative entropy function: x * log(x).
///
/// negentropy(x) = x * log(x)
///
/// The negation of entropy contribution. Same as xlogx.
/// Returns 0 for x = 0, NaN for x < 0.
#[must_use]
pub fn negentropy(x: f64) -> f64 {
    xlogx(x)
}

/// Binary cross-entropy loss (logistic loss).
///
/// binary_cross_entropy(p, q) = -p * log(q) - (1-p) * log(1-q)
///
/// Computes the cross-entropy between true label p and predicted
/// probability q. Both p and q should be in [0, 1].
#[must_use]
pub fn binary_cross_entropy(p: f64, q: f64) -> f64 {
    if p.is_nan() || q.is_nan() {
        return f64::NAN;
    }
    if q <= 0.0 || q >= 1.0 {
        if (p == 0.0 && q == 0.0) || (p == 1.0 && q == 1.0) {
            return 0.0; // 0 * log(0) = 0 by convention
        }
        return f64::INFINITY; // log(0) or log(negative)
    }
    -p * q.ln() - (1.0 - p) * (1.0 - q).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logsumexp_with_weights_matches_scipy_contract_point() {
        let value =
            logsumexp_with_b(&[1.0, 2.0, 3.0], &[1.0, 2.0, 0.5]).expect("weighted logsumexp");
        assert!((value - 3.315_609_082_086_973_5).abs() < 1.0e-12);
    }

    #[test]
    fn logsumexp_axis_2d_matches_scipy_axis_zero_contract_point() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let reduced = logsumexp_axis_2d(&data, 0).expect("axis logsumexp");
        assert_eq!(reduced.len(), 2);
        assert!((reduced[0] - 3.126_928_011_042_972_7).abs() < 1.0e-12);
        assert!((reduced[1] - 4.126_928_011_042_972).abs() < 1.0e-12);
    }

    #[test]
    fn logsumexp_axis_2d_with_broadcast_weights_matches_scipy_contract_point() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![vec![1.0, 2.0]];
        let reduced =
            logsumexp_axis_2d_with_b(&data, 1, &weights).expect("axis logsumexp with weights");
        assert_eq!(reduced.len(), 2);
        assert!((reduced[0] - 2.861_994_804_058_251_2).abs() < 1.0e-12);
        assert!((reduced[1] - 4.861_994_804_058_251).abs() < 1.0e-12);
    }

    #[test]
    fn logsumexp_axis_2d_rejects_non_broadcastable_weights() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![vec![1.0, 2.0, 3.0]];
        let err = logsumexp_axis_2d_with_b(&data, 1, &weights).expect_err("shape mismatch");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
    }

    #[test]
    fn nrdtrimn_recovers_mean() {
        let mean = 3.0;
        let std = 2.0;
        let x = 6.0;
        let p = ndtr((x - mean) / std);
        let recovered = nrdtrimn(p, std, x);
        assert!(
            (recovered - mean).abs() <= 1.0e-12,
            "nrdtrimn mean recovery mismatch: expected={mean}, got={recovered}"
        );
    }

    #[test]
    fn nrdtrimn_matches_scipy_contract_points() {
        assert!((nrdtrimn(0.8, 2.0, 1.0) - (-0.683_242_467_145_828_8)).abs() <= 1.0e-12);
        assert!((nrdtrimn(0.2, 2.0, 1.0) - 2.683_242_467_145_828_6).abs() <= 1.0e-12);
        assert!((nrdtrimn(0.5, 1.0, 1.0) - 1.0).abs() <= 1.0e-12);
        assert!(nrdtrimn(0.2, f64::INFINITY, 1.0).is_infinite());
        assert!(nrdtrimn(0.2, f64::INFINITY, 1.0).is_sign_positive());
        assert!(nrdtrimn(0.5, f64::INFINITY, 1.0).is_infinite());
        assert!(nrdtrimn(0.5, f64::INFINITY, 1.0).is_sign_negative());
        assert!(nrdtrimn(0.5, f64::INFINITY, f64::INFINITY).is_nan());
        assert!(nrdtrimn(0.8, f64::INFINITY, f64::INFINITY).is_nan());
        assert!(nrdtrimn(0.2, f64::INFINITY, f64::NEG_INFINITY).is_nan());
        assert!(nrdtrimn(0.5, f64::INFINITY, f64::NEG_INFINITY).is_infinite());
        assert!(nrdtrimn(0.5, f64::INFINITY, f64::NEG_INFINITY).is_sign_negative());
        assert!(nrdtrimn(0.8, f64::INFINITY, f64::NEG_INFINITY).is_infinite());
        assert!(nrdtrimn(0.8, f64::INFINITY, f64::NEG_INFINITY).is_sign_negative());
    }

    #[test]
    fn nrdtrimn_rejects_invalid_inputs_like_scipy() {
        assert!(nrdtrimn(0.0, 2.0, 1.0).is_nan());
        assert!(nrdtrimn(1.0, 2.0, 1.0).is_nan());
        assert!(nrdtrimn(0.8, 0.0, 1.0).is_nan());
        assert!(nrdtrimn(0.8, -1.0, 1.0).is_nan());
        assert!(nrdtrimn(f64::NAN, 2.0, 1.0).is_nan());
        assert!(nrdtrimn(0.8, f64::NAN, 1.0).is_nan());
        assert!(nrdtrimn(0.8, 2.0, f64::NAN).is_nan());
    }

    #[test]
    fn nrdtrisd_recovers_std() {
        let mean = 3.0;
        let std = 2.0;
        let x = 6.0;
        let p = ndtr((x - mean) / std);
        let recovered = nrdtrisd(mean, p, x);
        assert!(
            (recovered - std).abs() <= 1.0e-12,
            "nrdtrisd std recovery mismatch: expected={std}, got={recovered}"
        );
    }

    #[test]
    fn nrdtrisd_matches_scipy_contract_points() {
        assert!(nrdtrisd(0.5, 0.5, 0.5) == 0.0);
        assert!((nrdtrisd(3.0, 0.933_192_798_731_141_9, 6.0) - 2.0).abs() <= 1.0e-12);
        assert!(
            (nrdtrisd(1.0, 0.5, 0.5) - (-7.532_401_279_578_852e15)).abs()
                <= 1.0e-12 * 7.532_401_279_578_852e15
        );
        assert!(nrdtrisd(1.0, 0.2, f64::INFINITY).is_infinite());
        assert!(nrdtrisd(1.0, 0.2, f64::INFINITY).is_sign_negative());
        assert!(nrdtrisd(1.0, 0.5, f64::INFINITY).is_infinite());
        assert!(nrdtrisd(1.0, 0.5, f64::INFINITY).is_sign_positive());
        assert!(nrdtrisd(f64::INFINITY, 0.2, 1.0).is_infinite());
        assert!(nrdtrisd(f64::INFINITY, 0.2, 1.0).is_sign_positive());
        assert!(nrdtrisd(f64::NEG_INFINITY, 0.8, 1.0).is_infinite());
        assert!(nrdtrisd(f64::NEG_INFINITY, 0.8, 1.0).is_sign_positive());
    }

    #[test]
    fn nrdtrisd_rejects_invalid_inputs_like_scipy() {
        assert!(nrdtrisd(0.0, 0.0, 1.0).is_nan());
        assert!(nrdtrisd(0.0, 1.0, 1.0).is_nan());
        assert!(nrdtrisd(f64::NAN, 0.8, 1.0).is_nan());
        assert!(nrdtrisd(0.0, f64::NAN, 1.0).is_nan());
        assert!(nrdtrisd(0.0, 0.8, f64::NAN).is_nan());
    }

    #[test]
    fn kelvin_functions_match_scipy_reference_points() {
        assert!((ber(1.0) - 0.984_381_781_213_087).abs() < 1e-12);
        assert!((bei(1.0) - 0.249_566_040_036_659_72).abs() < 1e-12);
        assert!((ker(1.0) - 0.286_706_208_728_316_04).abs() < 1e-10);
        assert!((kei(1.0) - (-0.494_994_636_518_72)).abs() < 1e-10);
    }

    #[test]
    fn kolmogorov_basic() {
        // kolmogorov(0) = 1 (survival function at 0)
        assert!((kolmogorov(0.0) - 1.0).abs() < 1e-10);

        // kolmogorov is monotonically decreasing
        assert!(kolmogorov(0.5) > kolmogorov(1.0));
        assert!(kolmogorov(1.0) > kolmogorov(1.5));
        assert!(kolmogorov(1.5) > kolmogorov(2.0));

        // Known value: kolmogorov(1.0) ≈ 0.27
        let k1 = kolmogorov(1.0);
        assert!(
            (k1 - 0.27).abs() < 0.02,
            "kolmogorov(1.0) = {k1}, expected ~0.27"
        );

        // For large y, kolmogorov(y) -> 0
        assert!(kolmogorov(3.0) < 0.001);
    }

    #[test]
    fn kolmogi_inverse() {
        // kolmogi should be inverse of kolmogorov
        for &y in &[0.5, 1.0, 1.5, 2.0, 2.5] {
            let p = kolmogorov(y);
            if p > 0.001 && p < 0.999 {
                let y_recovered = kolmogi(p);
                assert!(
                    (y_recovered - y).abs() < 0.001,
                    "kolmogi failed: y={y}, p={p}, y_recovered={y_recovered}"
                );
            }
        }
    }

    #[test]
    fn kolmogi_endpoints() {
        // kolmogi(0) = +inf
        assert!(kolmogi(0.0).is_infinite() && kolmogi(0.0).is_sign_positive());
        // kolmogi(1) = 0
        assert!((kolmogi(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn smirnov_basic() {
        // smirnov(n, 0) = 1
        assert!((smirnov(10, 0.0) - 1.0).abs() < 1e-10);
        // smirnov(n, 1) = 0
        assert!((smirnov(10, 1.0) - 0.0).abs() < 1e-10);

        // smirnov is monotonically decreasing in d for larger d values
        for n in &[10, 20, 50] {
            assert!(
                smirnov(*n, 0.3) > smirnov(*n, 0.5),
                "n={n}: smirnov(0.3) > smirnov(0.5)"
            );
            assert!(
                smirnov(*n, 0.5) > smirnov(*n, 0.7),
                "n={n}: smirnov(0.5) > smirnov(0.7)"
            );
        }

        // smirnov(n, d) is in [0, 1]
        for n in &[10, 20, 50, 100] {
            for &d in &[0.1, 0.3, 0.5, 0.7] {
                let s = smirnov(*n, d);
                assert!(
                    (0.0..=1.0).contains(&s),
                    "smirnov({n}, {d}) = {s} out of range"
                );
            }
        }
    }

    #[test]
    fn smirnovi_inverse() {
        // smirnovi should be inverse of smirnov (within tolerance)
        for &n in &[20, 50, 100] {
            for &d in &[0.2, 0.3, 0.4, 0.5] {
                let p = smirnov(n, d);
                if p > 0.01 && p < 0.99 {
                    let d_recovered = smirnovi(n, p);
                    assert!(
                        (d_recovered - d).abs() < 0.05,
                        "smirnovi failed: n={n}, d={d}, p={p}, d_recovered={d_recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn degree_trig_exact_values() {
        // Test exact values at common angles
        assert!((cosdg(0.0) - 1.0).abs() < 1e-15);
        assert!((cosdg(90.0) - 0.0).abs() < 1e-15);
        assert!((cosdg(180.0) - (-1.0)).abs() < 1e-15);
        assert!((cosdg(270.0) - 0.0).abs() < 1e-15);
        assert!((cosdg(360.0) - 1.0).abs() < 1e-15);

        assert!((sindg(0.0) - 0.0).abs() < 1e-15);
        assert!((sindg(90.0) - 1.0).abs() < 1e-15);
        assert!((sindg(180.0) - 0.0).abs() < 1e-15);
        assert!((sindg(270.0) - (-1.0)).abs() < 1e-15);

        assert!((tandg(0.0) - 0.0).abs() < 1e-15);
        assert!((tandg(45.0) - 1.0).abs() < 1e-15);
        assert!(tandg(90.0).is_infinite());

        assert!(cotdg(0.0).is_infinite());
        assert!((cotdg(45.0) - 1.0).abs() < 1e-15);
        assert!((cotdg(90.0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn degree_trig_general() {
        // Test general values match radians conversion
        for &deg in &[30.0, 45.0, 60.0, 120.0, 150.0, 210.0, 300.0] {
            let rad = deg * std::f64::consts::PI / 180.0;
            assert!(
                (cosdg(deg) - rad.cos()).abs() < 1e-14,
                "cosdg({deg}) failed"
            );
            assert!(
                (sindg(deg) - rad.sin()).abs() < 1e-14,
                "sindg({deg}) failed"
            );
        }
    }

    #[test]
    fn radian_basic() {
        // 180 degrees = π radians
        assert!((radian(180.0, 0.0, 0.0) - std::f64::consts::PI).abs() < 1e-14);

        // 90 degrees = π/2 radians
        assert!((radian(90.0, 0.0, 0.0) - std::f64::consts::FRAC_PI_2).abs() < 1e-14);

        // 45 degrees 30 minutes = 45.5 degrees
        let expected = 45.5 * std::f64::consts::PI / 180.0;
        assert!((radian(45.0, 30.0, 0.0) - expected).abs() < 1e-14);

        // 1 degree 1 minute 1 second
        let expected = (1.0 + 1.0 / 60.0 + 1.0 / 3600.0) * std::f64::consts::PI / 180.0;
        assert!((radian(1.0, 1.0, 1.0) - expected).abs() < 1e-14);
    }

    #[test]
    fn cbrt_basic() {
        // Positive values
        assert!((cbrt(8.0) - 2.0).abs() < 1e-14);
        assert!((cbrt(27.0) - 3.0).abs() < 1e-14);
        assert!((cbrt(1.0) - 1.0).abs() < 1e-14);

        // Negative values (key feature - handles negative inputs)
        assert!((cbrt(-8.0) - (-2.0)).abs() < 1e-14);
        assert!((cbrt(-27.0) - (-3.0)).abs() < 1e-14);

        // Zero
        assert!((cbrt(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn exp_log_functions() {
        // exp2: 2^x
        assert!((exp2(0.0) - 1.0).abs() < 1e-14);
        assert!((exp2(1.0) - 2.0).abs() < 1e-14);
        assert!((exp2(3.0) - 8.0).abs() < 1e-14);
        assert!((exp2(-1.0) - 0.5).abs() < 1e-14);

        // exp10: 10^x
        assert!((exp10(0.0) - 1.0).abs() < 1e-14);
        assert!((exp10(1.0) - 10.0).abs() < 1e-13);
        assert!((exp10(2.0) - 100.0).abs() < 1e-12);

        // log2
        assert!((log2(1.0) - 0.0).abs() < 1e-14);
        assert!((log2(2.0) - 1.0).abs() < 1e-14);
        assert!((log2(8.0) - 3.0).abs() < 1e-14);

        // log10
        assert!((log10(1.0) - 0.0).abs() < 1e-14);
        assert!((log10(10.0) - 1.0).abs() < 1e-14);
        assert!((log10(100.0) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn round_basic() {
        assert!((round(1.4) - 1.0).abs() < 1e-14);
        assert!((round(1.5) - 2.0).abs() < 1e-14);
        assert!((round(1.6) - 2.0).abs() < 1e-14);
        assert!((round(-1.4) - (-1.0)).abs() < 1e-14);
        assert!((round(-1.5) - (-2.0)).abs() < 1e-14);
        assert!((round(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn floor_ceil_trunc() {
        // floor
        assert!((floor(1.7) - 1.0).abs() < 1e-14);
        assert!((floor(-1.7) - (-2.0)).abs() < 1e-14);
        assert!((floor(2.0) - 2.0).abs() < 1e-14);

        // ceil
        assert!((ceil(1.3) - 2.0).abs() < 1e-14);
        assert!((ceil(-1.3) - (-1.0)).abs() < 1e-14);
        assert!((ceil(2.0) - 2.0).abs() < 1e-14);

        // trunc (towards zero)
        assert!((trunc(1.7) - 1.0).abs() < 1e-14);
        assert!((trunc(-1.7) - (-1.0)).abs() < 1e-14);
        assert!((trunc(2.0) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn sign_basic() {
        assert!((sign(5.0) - 1.0).abs() < 1e-14);
        assert!((sign(-5.0) - (-1.0)).abs() < 1e-14);
        assert!((sign(0.0) - 0.0).abs() < 1e-14);
        assert!(sign(f64::NAN).is_nan());
    }

    #[test]
    fn heaviside_basic() {
        assert!((heaviside(1.0, 0.5) - 1.0).abs() < 1e-14);
        assert!((heaviside(-1.0, 0.5) - 0.0).abs() < 1e-14);
        assert!((heaviside(0.0, 0.5) - 0.5).abs() < 1e-14);
        assert!((heaviside(0.0, 0.0) - 0.0).abs() < 1e-14);
        assert!((heaviside(0.0, 1.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hypot_basic() {
        // Classic 3-4-5 triangle
        assert!((hypot(3.0, 4.0) - 5.0).abs() < 1e-14);
        // 5-12-13 triangle
        assert!((hypot(5.0, 12.0) - 13.0).abs() < 1e-14);
        // Handles zeros
        assert!((hypot(3.0, 0.0) - 3.0).abs() < 1e-14);
        assert!((hypot(0.0, 4.0) - 4.0).abs() < 1e-14);
        // Negative values
        assert!((hypot(-3.0, 4.0) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn copysign_basic() {
        assert!((copysign(3.0, 1.0) - 3.0).abs() < 1e-14);
        assert!((copysign(3.0, -1.0) - (-3.0)).abs() < 1e-14);
        assert!((copysign(-3.0, 1.0) - 3.0).abs() < 1e-14);
        assert!((copysign(-3.0, -1.0) - (-3.0)).abs() < 1e-14);
    }

    #[test]
    fn ldexp_basic() {
        // ldexp(x, n) = x * 2^n
        assert!((ldexp(1.0, 0) - 1.0).abs() < 1e-14);
        assert!((ldexp(1.0, 1) - 2.0).abs() < 1e-14);
        assert!((ldexp(1.0, 2) - 4.0).abs() < 1e-14);
        assert!((ldexp(1.0, -1) - 0.5).abs() < 1e-14);
        assert!((ldexp(3.0, 2) - 12.0).abs() < 1e-14);
    }

    #[test]
    fn frexp_basic() {
        // frexp returns (mantissa, exp) where x = mantissa * 2^exp
        // and 0.5 <= |mantissa| < 1.0
        let (m, e) = frexp(1.0);
        assert!((m - 0.5).abs() < 1e-14);
        assert_eq!(e, 1);

        let (m, e) = frexp(2.0);
        assert!((m - 0.5).abs() < 1e-14);
        assert_eq!(e, 2);

        let (m, e) = frexp(8.0);
        assert!((m - 0.5).abs() < 1e-14);
        assert_eq!(e, 4);

        let (m, e) = frexp(0.0);
        assert!((m - 0.0).abs() < 1e-14);
        assert_eq!(e, 0);

        // Verify roundtrip: ldexp(frexp(x)) == x
        for &x in &[0.5, 1.0, 2.0, std::f64::consts::PI, 100.0, 0.001] {
            let (m, e) = frexp(x);
            assert!((ldexp(m, e) - x).abs() < 1e-14);
        }
    }

    #[test]
    fn fabs_basic() {
        assert!((fabs(3.0) - 3.0).abs() < 1e-14);
        assert!((fabs(-3.0) - 3.0).abs() < 1e-14);
        assert!((fabs(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn clip_basic() {
        // Value within range
        assert!((clip(5.0, 0.0, 10.0) - 5.0).abs() < 1e-14);
        // Value below min
        assert!((clip(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-14);
        // Value above max
        assert!((clip(15.0, 0.0, 10.0) - 10.0).abs() < 1e-14);
        // At boundaries
        assert!((clip(0.0, 0.0, 10.0) - 0.0).abs() < 1e-14);
        assert!((clip(10.0, 0.0, 10.0) - 10.0).abs() < 1e-14);
    }

    #[test]
    fn sinpi_basic() {
        // sinpi(0) = 0
        assert!((sinpi(0.0) - 0.0).abs() < 1e-14);
        // sinpi(0.5) = 1
        assert!((sinpi(0.5) - 1.0).abs() < 1e-14);
        // sinpi(1) = 0
        assert!((sinpi(1.0) - 0.0).abs() < 1e-14);
        // sinpi(-0.5) = -1
        assert!((sinpi(-0.5) - (-1.0)).abs() < 1e-14);
        // sinpi(1.5) = -1
        assert!((sinpi(1.5) - (-1.0)).abs() < 1e-14);
        // sinpi(2) = 0
        assert!((sinpi(2.0) - 0.0).abs() < 1e-14);
        // Non-special values
        assert!((sinpi(0.25) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-14);
        // Negative
        assert!((sinpi(-0.25) - (-std::f64::consts::FRAC_1_SQRT_2)).abs() < 1e-14);
    }

    #[test]
    fn cospi_basic() {
        // cospi(0) = 1
        assert!((cospi(0.0) - 1.0).abs() < 1e-14);
        // cospi(0.5) = 0
        assert!((cospi(0.5) - 0.0).abs() < 1e-14);
        // cospi(1) = -1
        assert!((cospi(1.0) - (-1.0)).abs() < 1e-14);
        // cospi(-0.5) = 0
        assert!((cospi(-0.5) - 0.0).abs() < 1e-14);
        // cospi(2) = 1
        assert!((cospi(2.0) - 1.0).abs() < 1e-14);
        // Non-special values
        assert!((cospi(0.25) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-14);
        // Negative
        assert!((cospi(-0.25) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-14);
    }

    #[test]
    fn log1p_basic() {
        // log1p(0) = 0
        assert!((log1p(0.0) - 0.0).abs() < 1e-14);
        // log1p(e-1) = 1
        assert!((log1p(std::f64::consts::E - 1.0) - 1.0).abs() < 1e-14);
        // Small values - this is where log1p shines
        let small = 1e-15;
        assert!((log1p(small) - small).abs() < 1e-28);
    }

    #[test]
    fn expm1_basic() {
        // expm1(0) = 0
        assert!((expm1(0.0) - 0.0).abs() < 1e-14);
        // expm1(1) = e-1
        assert!((expm1(1.0) - (std::f64::consts::E - 1.0)).abs() < 1e-14);
        // Small values - this is where expm1 shines
        let small = 1e-15;
        assert!((expm1(small) - small).abs() < 1e-28);
    }

    #[test]
    fn logaddexp_basic() {
        // logaddexp(0, 0) = log(2)
        assert!((logaddexp(0.0, 0.0) - std::f64::consts::LN_2).abs() < 1e-14);
        // logaddexp(x, -inf) = x
        assert!((logaddexp(1.0, f64::NEG_INFINITY) - 1.0).abs() < 1e-14);
        // logaddexp(-inf, x) = x
        assert!((logaddexp(f64::NEG_INFINITY, 2.0) - 2.0).abs() < 1e-14);
        // For large difference, result ≈ max
        assert!((logaddexp(100.0, 0.0) - 100.0).abs() < 1e-10);
        // Symmetric
        assert!((logaddexp(1.0, 2.0) - logaddexp(2.0, 1.0)).abs() < 1e-14);
    }

    #[test]
    fn logaddexp2_basic() {
        // logaddexp2(0, 0) = 1  (log2(2^0 + 2^0) = log2(2) = 1)
        assert!((logaddexp2(0.0, 0.0) - 1.0).abs() < 1e-14);
        // logaddexp2(x, -inf) = x
        assert!((logaddexp2(1.0, f64::NEG_INFINITY) - 1.0).abs() < 1e-14);
        // logaddexp2(-inf, x) = x
        assert!((logaddexp2(f64::NEG_INFINITY, 2.0) - 2.0).abs() < 1e-14);
        // For large difference, result ≈ max
        assert!((logaddexp2(100.0, 0.0) - 100.0).abs() < 1e-10);
        // Symmetric
        assert!((logaddexp2(1.0, 2.0) - logaddexp2(2.0, 1.0)).abs() < 1e-14);
    }

    #[test]
    fn nextafter_basic() {
        // Moving toward larger value increases
        let next = nextafter(1.0, 2.0);
        assert!(next > 1.0);
        assert!(next < 1.0 + 1e-14);

        // Moving toward smaller value decreases
        let prev = nextafter(1.0, 0.0);
        assert!(prev < 1.0);
        assert!(prev > 1.0 - 1e-14);

        // From zero
        let tiny = nextafter(0.0, 1.0);
        assert!(tiny > 0.0);
        let neg_tiny = nextafter(0.0, -1.0);
        assert!(neg_tiny < 0.0);

        // Same value returns itself
        assert_eq!(nextafter(1.0, 1.0), 1.0);
    }

    #[test]
    fn spacing_basic() {
        // Spacing at 1.0 is machine epsilon
        let eps = spacing(1.0);
        assert!((eps - f64::EPSILON).abs() < 1e-30);

        // Spacing is always positive
        assert!(spacing(1.0) > 0.0);
        assert!(spacing(-1.0) > 0.0);
        assert!(spacing(0.0) > 0.0);

        // Spacing gets larger for larger numbers
        assert!(spacing(1e10) > spacing(1.0));
    }

    #[test]
    fn modf_basic() {
        // Positive values
        let (frac, int) = modf(3.5);
        assert!((frac - 0.5).abs() < 1e-14);
        assert!((int - 3.0).abs() < 1e-14);

        // Negative values
        let (frac, int) = modf(-3.5);
        assert!((frac - (-0.5)).abs() < 1e-14);
        assert!((int - (-3.0)).abs() < 1e-14);

        // Integer values
        let (frac, int) = modf(4.0);
        assert!((frac - 0.0).abs() < 1e-14);
        assert!((int - 4.0).abs() < 1e-14);

        // Zero
        let (frac, int) = modf(0.0);
        assert!((frac - 0.0).abs() < 1e-14);
        assert!((int - 0.0).abs() < 1e-14);
    }

    #[test]
    fn signbit_basic() {
        // Positive
        assert!(!signbit(1.0));
        // Negative
        assert!(signbit(-1.0));
        // Zero
        assert!(!signbit(0.0));
        // Negative zero
        assert!(signbit(-0.0));
    }

    #[test]
    fn float_classification() {
        // isnan
        assert!(isnan(f64::NAN));
        assert!(!isnan(1.0));
        assert!(!isnan(f64::INFINITY));

        // isinf
        assert!(isinf(f64::INFINITY));
        assert!(isinf(f64::NEG_INFINITY));
        assert!(!isinf(1.0));
        assert!(!isinf(f64::NAN));

        // isfinite
        assert!(isfinite(1.0));
        assert!(isfinite(0.0));
        assert!(!isfinite(f64::INFINITY));
        assert!(!isfinite(f64::NEG_INFINITY));
        assert!(!isfinite(f64::NAN));

        // isposinf
        assert!(isposinf(f64::INFINITY));
        assert!(!isposinf(f64::NEG_INFINITY));
        assert!(!isposinf(1.0));

        // isneginf
        assert!(isneginf(f64::NEG_INFINITY));
        assert!(!isneginf(f64::INFINITY));
        assert!(!isneginf(-1.0));
    }

    #[test]
    fn reciprocal_basic() {
        assert!((reciprocal(2.0) - 0.5).abs() < 1e-14);
        assert!((reciprocal(4.0) - 0.25).abs() < 1e-14);
        assert!((reciprocal(-2.0) - (-0.5)).abs() < 1e-14);
        assert!(reciprocal(0.0).is_infinite());
    }

    #[test]
    fn square_basic() {
        assert!((square(2.0) - 4.0).abs() < 1e-14);
        assert!((square(3.0) - 9.0).abs() < 1e-14);
        assert!((square(-2.0) - 4.0).abs() < 1e-14);
        assert!((square(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn positive_negative_parts() {
        // Positive part
        assert!((positive(3.0) - 3.0).abs() < 1e-14);
        assert!((positive(-3.0) - 0.0).abs() < 1e-14);
        assert!((positive(0.0) - 0.0).abs() < 1e-14);

        // Negative part
        assert!((negative(3.0) - 0.0).abs() < 1e-14);
        assert!((negative(-3.0) - 3.0).abs() < 1e-14);
        assert!((negative(0.0) - 0.0).abs() < 1e-14);

        // positive(x) + negative(x) == |x|
        for &x in &[-3.0, -1.0, 0.0, 1.0, 3.0] {
            assert!((positive(x) + negative(x) - x.abs()).abs() < 1e-14);
        }
    }

    #[test]
    fn deg2rad_rad2deg() {
        use std::f64::consts::PI;

        // deg2rad
        assert!((deg2rad(0.0) - 0.0).abs() < 1e-14);
        assert!((deg2rad(180.0) - PI).abs() < 1e-14);
        assert!((deg2rad(90.0) - PI / 2.0).abs() < 1e-14);
        assert!((deg2rad(360.0) - 2.0 * PI).abs() < 1e-14);
        assert!((deg2rad(-90.0) - (-PI / 2.0)).abs() < 1e-14);

        // rad2deg
        assert!((rad2deg(0.0) - 0.0).abs() < 1e-14);
        assert!((rad2deg(PI) - 180.0).abs() < 1e-12);
        assert!((rad2deg(PI / 2.0) - 90.0).abs() < 1e-12);
        assert!((rad2deg(2.0 * PI) - 360.0).abs() < 1e-12);

        // Roundtrip
        for &deg in &[0.0, 45.0, 90.0, 180.0, 270.0, 360.0] {
            assert!((rad2deg(deg2rad(deg)) - deg).abs() < 1e-12);
        }
    }

    #[test]
    fn rint_basic() {
        // Round to nearest, ties to even (banker's rounding)
        assert!((rint(1.4) - 1.0).abs() < 1e-14);
        assert!((rint(1.6) - 2.0).abs() < 1e-14);
        assert!((rint(-1.4) - (-1.0)).abs() < 1e-14);
        assert!((rint(-1.6) - (-2.0)).abs() < 1e-14);

        // Ties go to even
        assert!((rint(0.5) - 0.0).abs() < 1e-14); // 0 is even
        assert!((rint(1.5) - 2.0).abs() < 1e-14); // 2 is even
        assert!((rint(2.5) - 2.0).abs() < 1e-14); // 2 is even
        assert!((rint(3.5) - 4.0).abs() < 1e-14); // 4 is even
    }

    #[test]
    fn fix_basic() {
        // Round towards zero
        assert!((fix(1.7) - 1.0).abs() < 1e-14);
        assert!((fix(-1.7) - (-1.0)).abs() < 1e-14);
        assert!((fix(2.9) - 2.0).abs() < 1e-14);
        assert!((fix(-2.9) - (-2.0)).abs() < 1e-14);
        assert!((fix(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn divmod_basic() {
        // Basic division
        let (q, r) = divmod(7.0, 3.0);
        assert!((q - 2.0).abs() < 1e-14);
        assert!((r - 1.0).abs() < 1e-14);

        // Negative dividend
        let (q, r) = divmod(-7.0, 3.0);
        assert!((q - (-3.0)).abs() < 1e-14);
        assert!((r - 2.0).abs() < 1e-14);

        // Negative divisor
        let (q, r) = divmod(7.0, -3.0);
        assert!((q - (-3.0)).abs() < 1e-14);
        assert!((r - (-2.0)).abs() < 1e-14);

        // Both negative
        let (q, r) = divmod(-7.0, -3.0);
        assert!((q - 2.0).abs() < 1e-14);
        assert!((r - (-1.0)).abs() < 1e-14);

        // Division by zero
        let (q, r) = divmod(1.0, 0.0);
        assert!(q.is_nan());
        assert!(r.is_nan());
    }

    #[test]
    fn maximum_minimum_basic() {
        // maximum - basic
        assert!((maximum(3.0, 5.0) - 5.0).abs() < 1e-14);
        assert!((maximum(5.0, 3.0) - 5.0).abs() < 1e-14);
        assert!((maximum(-1.0, 1.0) - 1.0).abs() < 1e-14);

        // maximum - propagates NaN
        assert!(maximum(f64::NAN, 1.0).is_nan());
        assert!(maximum(1.0, f64::NAN).is_nan());
        assert!(maximum(f64::NAN, f64::NAN).is_nan());

        // minimum - basic
        assert!((minimum(3.0, 5.0) - 3.0).abs() < 1e-14);
        assert!((minimum(5.0, 3.0) - 3.0).abs() < 1e-14);
        assert!((minimum(-1.0, 1.0) - (-1.0)).abs() < 1e-14);

        // minimum - propagates NaN
        assert!(minimum(f64::NAN, 1.0).is_nan());
        assert!(minimum(1.0, f64::NAN).is_nan());
        assert!(minimum(f64::NAN, f64::NAN).is_nan());
    }

    #[test]
    fn fmax_fmin_basic() {
        // fmax - basic
        assert!((fmax(3.0, 5.0) - 5.0).abs() < 1e-14);
        assert!((fmax(5.0, 3.0) - 5.0).abs() < 1e-14);

        // fmax - ignores NaN
        assert!((fmax(f64::NAN, 1.0) - 1.0).abs() < 1e-14);
        assert!((fmax(1.0, f64::NAN) - 1.0).abs() < 1e-14);
        assert!(fmax(f64::NAN, f64::NAN).is_nan());

        // fmin - basic
        assert!((fmin(3.0, 5.0) - 3.0).abs() < 1e-14);
        assert!((fmin(5.0, 3.0) - 3.0).abs() < 1e-14);

        // fmin - ignores NaN
        assert!((fmin(f64::NAN, 1.0) - 1.0).abs() < 1e-14);
        assert!((fmin(1.0, f64::NAN) - 1.0).abs() < 1e-14);
        assert!(fmin(f64::NAN, f64::NAN).is_nan());
    }

    #[test]
    fn power_basic() {
        assert!((power(2.0, 3.0) - 8.0).abs() < 1e-14);
        assert!((power(3.0, 2.0) - 9.0).abs() < 1e-14);
        assert!((power(4.0, 0.5) - 2.0).abs() < 1e-14);
        assert!((power(2.0, -1.0) - 0.5).abs() < 1e-14);
        assert!((power(0.0, 0.0) - 1.0).abs() < 1e-14); // 0^0 = 1 by convention
    }

    #[test]
    fn fdiff_basic() {
        assert!((fdiff(5.0, 3.0) - 2.0).abs() < 1e-14);
        assert!((fdiff(3.0, 5.0) - 2.0).abs() < 1e-14);
        assert!((fdiff(-1.0, 1.0) - 2.0).abs() < 1e-14);
        assert!((fdiff(0.0, 0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn nan_to_num_basic() {
        // NaN replacement
        assert!((nan_to_num(f64::NAN, 0.0, f64::MAX, f64::MIN) - 0.0).abs() < 1e-14);
        assert!((nan_to_num(f64::NAN, 99.0, f64::MAX, f64::MIN) - 99.0).abs() < 1e-14);

        // Infinity replacement
        assert!((nan_to_num(f64::INFINITY, 0.0, 1e10, -1e10) - 1e10).abs() < 1e-14);
        assert!((nan_to_num(f64::NEG_INFINITY, 0.0, 1e10, -1e10) - (-1e10)).abs() < 1e-14);

        // Finite values unchanged
        assert!((nan_to_num(5.0, 0.0, f64::MAX, f64::MIN) - 5.0).abs() < 1e-14);
        assert!((nan_to_num(-3.0, 0.0, f64::MAX, f64::MIN) - (-3.0)).abs() < 1e-14);
    }

    #[test]
    fn relu_basic() {
        // Positive values pass through
        assert!((relu(5.0) - 5.0).abs() < 1e-14);
        assert!((relu(0.1) - 0.1).abs() < 1e-14);

        // Negative values become zero
        assert!((relu(-5.0) - 0.0).abs() < 1e-14);
        assert!((relu(-0.1) - 0.0).abs() < 1e-14);

        // Zero stays zero
        assert!((relu(0.0) - 0.0).abs() < 1e-14);

        // NaN propagates
        assert!(relu(f64::NAN).is_nan());
    }

    #[test]
    fn softplus_basic() {
        // softplus(0) = ln(2)
        assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 1e-14);

        // For large positive x, softplus(x) ≈ x
        assert!((softplus(100.0) - 100.0).abs() < 1e-10);

        // For large negative x, softplus(x) ≈ 0
        assert!(softplus(-100.0) < 1e-40);

        // softplus is always positive
        assert!(softplus(-10.0) > 0.0);
        assert!(softplus(0.0) > 0.0);
        assert!(softplus(10.0) > 0.0);

        // NaN propagates
        assert!(softplus(f64::NAN).is_nan());
    }

    #[test]
    fn huber_basic() {
        let delta = 1.0;

        // For |x| <= delta, huber = 0.5 * x^2
        assert!((huber(delta, 0.5) - 0.125).abs() < 1e-14);
        assert!((huber(delta, -0.5) - 0.125).abs() < 1e-14);
        assert!((huber(delta, 0.0) - 0.0).abs() < 1e-14);

        // For |x| > delta, huber = delta * (|x| - 0.5*delta)
        assert!((huber(delta, 2.0) - 1.5).abs() < 1e-14);
        assert!((huber(delta, -2.0) - 1.5).abs() < 1e-14);

        // At boundary
        assert!((huber(delta, 1.0) - 0.5).abs() < 1e-14);

        // Invalid delta
        assert!(huber(0.0, 1.0).is_nan());
        assert!(huber(-1.0, 1.0).is_nan());
    }

    #[test]
    fn pseudo_huber_basic() {
        let delta = 1.0;

        // pseudo_huber(delta, 0) = 0
        assert!((pseudo_huber(delta, 0.0) - 0.0).abs() < 1e-14);

        // For small x, pseudo_huber ≈ 0.5 * x^2
        let small = 0.01;
        let expected = 0.5 * small * small;
        assert!((pseudo_huber(delta, small) - expected).abs() < 1e-6);

        // Symmetric
        assert!((pseudo_huber(delta, 2.0) - pseudo_huber(delta, -2.0)).abs() < 1e-14);

        // Invalid delta
        assert!(pseudo_huber(0.0, 1.0).is_nan());
        assert!(pseudo_huber(-1.0, 1.0).is_nan());
    }

    #[test]
    fn elu_basic() {
        // Positive values pass through
        assert!((elu(2.0, 1.0) - 2.0).abs() < 1e-14);

        // Negative values: alpha * (exp(x) - 1)
        // elu(-1, 1) = 1 * (exp(-1) - 1) ≈ -0.632
        assert!((elu(-1.0, 1.0) - ((-1.0_f64).exp() - 1.0)).abs() < 1e-14);

        // Zero
        assert!((elu(0.0, 1.0) - 0.0).abs() < 1e-14);

        // Different alpha
        assert!((elu(-1.0, 2.0) - 2.0 * ((-1.0_f64).exp() - 1.0)).abs() < 1e-14);
    }

    #[test]
    fn leaky_relu_basic() {
        // Positive values pass through
        assert!((leaky_relu(2.0, 0.1) - 2.0).abs() < 1e-14);

        // Negative values scaled by alpha
        assert!((leaky_relu(-2.0, 0.1) - (-0.2)).abs() < 1e-14);

        // Zero
        assert!((leaky_relu(0.0, 0.1) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn gelu_basic() {
        // gelu(0) = 0 (since Φ(0) = 0.5)
        assert!((gelu(0.0) - 0.0).abs() < 1e-14);

        // For large positive x, gelu(x) ≈ x
        assert!((gelu(10.0) - 10.0).abs() < 1e-6);

        // For large negative x, gelu(x) ≈ 0
        assert!(gelu(-10.0).abs() < 1e-6);

        // gelu is smooth and monotonic for x > 0
        assert!(gelu(1.0) > gelu(0.5));
        assert!(gelu(2.0) > gelu(1.0));
    }

    #[test]
    fn selu_basic() {
        // Positive values scaled by ~1.0507
        let scale = 1.0507009873554805;
        assert!((selu(1.0) - scale * 1.0).abs() < 1e-10);

        // Negative values
        assert!(selu(-1.0) < 0.0);

        // Zero
        assert!((selu(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn swish_basic() {
        // swish(0, beta) = 0
        assert!((swish(0.0, 1.0) - 0.0).abs() < 1e-14);

        // For large positive x, swish(x, 1) ≈ x
        assert!((swish(10.0, 1.0) - 10.0).abs() < 1e-3);

        // For large negative x, swish(x, 1) ≈ 0
        assert!(swish(-10.0, 1.0).abs() < 1e-3);

        // swish is smooth
        assert!(swish(1.0, 1.0) > swish(0.0, 1.0));
    }

    #[test]
    fn mish_basic() {
        // mish(0) = 0
        assert!((mish(0.0) - 0.0).abs() < 1e-14);

        // For large positive x, mish(x) ≈ x
        assert!((mish(10.0) - 10.0).abs() < 1e-3);

        // For large negative x, mish(x) ≈ 0
        assert!(mish(-10.0).abs() < 1e-3);

        // mish is smooth and has slight negative region
        assert!(mish(-0.5) < 0.0);
    }

    #[test]
    fn hard_sigmoid_basic() {
        // hard_sigmoid(-3) = 0
        assert!((hard_sigmoid(-3.0) - 0.0).abs() < 1e-14);
        // hard_sigmoid(3) = 1
        assert!((hard_sigmoid(3.0) - 1.0).abs() < 1e-14);
        // hard_sigmoid(0) = 0.5
        assert!((hard_sigmoid(0.0) - 0.5).abs() < 1e-14);
        // Linear region
        assert!((hard_sigmoid(1.0) - (4.0 / 6.0)).abs() < 1e-14);
    }

    #[test]
    fn hard_swish_basic() {
        // hard_swish(0) = 0 * 0.5 = 0
        assert!((hard_swish(0.0) - 0.0).abs() < 1e-14);
        // hard_swish(-3) = -3 * 0 = 0
        assert!((hard_swish(-3.0) - 0.0).abs() < 1e-14);
        // hard_swish(3) = 3 * 1 = 3
        assert!((hard_swish(3.0) - 3.0).abs() < 1e-14);
        // For large positive x, hard_swish(x) ≈ x
        assert!((hard_swish(10.0) - 10.0).abs() < 1e-14);
    }

    #[test]
    fn hard_tanh_basic() {
        // Clipped to range
        assert!((hard_tanh(0.5, -1.0, 1.0) - 0.5).abs() < 1e-14);
        assert!((hard_tanh(2.0, -1.0, 1.0) - 1.0).abs() < 1e-14);
        assert!((hard_tanh(-2.0, -1.0, 1.0) - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn log_cosh_basic() {
        // log_cosh(0) = 0
        assert!((log_cosh(0.0) - 0.0).abs() < 1e-14);
        // Symmetric
        assert!((log_cosh(2.0) - log_cosh(-2.0)).abs() < 1e-14);
        // For large |x|, log_cosh(x) ≈ |x| - ln(2)
        let large = 30.0;
        assert!((log_cosh(large) - (large - std::f64::consts::LN_2)).abs() < 1e-10);
        // Always non-negative
        assert!(log_cosh(-5.0) >= 0.0);
    }

    #[test]
    fn softsign_basic() {
        // softsign(0) = 0
        assert!((softsign(0.0) - 0.0).abs() < 1e-14);
        // Bounded by [-1, 1]
        assert!(softsign(100.0) < 1.0);
        assert!(softsign(-100.0) > -1.0);
        // Antisymmetric
        assert!((softsign(2.0) + softsign(-2.0)).abs() < 1e-14);
        // softsign(1) = 0.5
        assert!((softsign(1.0) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn threshold_basic() {
        // Above threshold: pass through
        assert!((threshold(5.0, 0.0, -1.0) - 5.0).abs() < 1e-14);
        // Below threshold: use value
        assert!((threshold(-5.0, 0.0, -1.0) - (-1.0)).abs() < 1e-14);
        // At threshold: use value (not strictly greater)
        assert!((threshold(0.0, 0.0, -1.0) - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn silu_basic() {
        // silu is just swish with beta=1
        assert!((silu(0.0) - swish(0.0, 1.0)).abs() < 1e-14);
        assert!((silu(2.0) - swish(2.0, 1.0)).abs() < 1e-14);
        assert!((silu(-2.0) - swish(-2.0, 1.0)).abs() < 1e-14);
    }

    #[test]
    fn log_expit_scalar_basic() {
        // log_expit(0) = log(0.5) = -ln(2)
        assert!((log_expit_scalar(0.0) - (-std::f64::consts::LN_2)).abs() < 1e-14);
        // log_expit should equal log(expit_scalar(x))
        assert!((log_expit_scalar(2.0) - expit_scalar(2.0).ln()).abs() < 1e-14);
        // For large negative x, log_expit(x) ≈ x
        assert!((log_expit_scalar(-50.0) - (-50.0)).abs() < 1e-10);
        // Symmetric property: log_expit(x) + log_expit(-x) = -log(2) - |x|... actually just test consistency
        assert!((log_expit_scalar(5.0) - expit_scalar(5.0).ln()).abs() < 1e-14);
        assert!((log_expit_scalar(-5.0) - expit_scalar(-5.0).ln()).abs() < 1e-14);
    }

    fn expect_real_scalar(tensor: SpecialTensor) -> Result<f64, String> {
        match tensor {
            SpecialTensor::RealScalar(value) => Ok(value),
            other => Err(format!("expected real scalar, got {other:?}")),
        }
    }

    fn expect_real_vec(tensor: SpecialTensor) -> Result<Vec<f64>, String> {
        match tensor {
            SpecialTensor::RealVec(values) => Ok(values),
            other => Err(format!("expected real vector, got {other:?}")),
        }
    }

    #[test]
    fn exprel_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = exprel(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - exprel_scalar(1.0)).abs() < 1e-14);

        let vector = exprel(
            &SpecialTensor::RealVec(vec![-1.0e-6, 0.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-1.0e-6, 0.0, 1.0].map(exprel_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn exprel_near_zero_series_stays_stable() -> Result<(), String> {
        let result = exprel(&SpecialTensor::RealScalar(1.0e-8), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let value = expect_real_scalar(result)?;
        let expected = 1.0 + 0.5e-8 + 1.0e-16 / 6.0 + 1.0e-24 / 24.0;
        assert!((value - expected).abs() < 1e-16);
        Ok(())
    }

    #[test]
    fn dawsn_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = dawsn(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - dawsn_scalar(1.0)).abs() < 1e-14);

        let vector = dawsn(
            &SpecialTensor::RealVec(vec![-1.0, 0.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-1.0, 0.0, 1.0].map(dawsn_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn dawsn_tensor_dispatch_preserves_odd_symmetry() -> Result<(), String> {
        let positive = dawsn(&SpecialTensor::RealScalar(2.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let negative = dawsn(&SpecialTensor::RealScalar(-2.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let positive_value = expect_real_scalar(positive)?;
        let negative_value = expect_real_scalar(negative)?;
        assert!((positive_value + negative_value).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn erfi_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = erfi(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - erfi_scalar(1.0)).abs() < 1e-14);

        let vector = erfi(
            &SpecialTensor::RealVec(vec![-1.0, 0.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-1.0, 0.0, 1.0].map(erfi_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn erfi_tensor_dispatch_preserves_odd_symmetry() -> Result<(), String> {
        let positive = erfi(&SpecialTensor::RealScalar(2.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let negative = erfi(&SpecialTensor::RealScalar(-2.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let positive_value = expect_real_scalar(positive)?;
        let negative_value = expect_real_scalar(negative)?;
        assert!((positive_value + negative_value).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn cloglog_basic() {
        // cloglog(0.5) = log(-log(0.5)) = log(ln(2)) ≈ -0.3665
        let expected = std::f64::consts::LN_2.ln();
        assert!((cloglog(0.5) - expected).abs() < 1e-14);

        // Monotonically increasing
        assert!(cloglog(0.3) < cloglog(0.5));
        assert!(cloglog(0.5) < cloglog(0.7));

        // Boundary behavior
        assert!(cloglog(0.0).is_infinite() && cloglog(0.0) < 0.0);
        assert!(cloglog(1.0).is_infinite() && cloglog(1.0) > 0.0);

        // Near boundaries should still work
        assert!(cloglog(1e-10).is_finite());
        assert!(cloglog(1.0 - 1e-10).is_finite());
    }

    #[test]
    fn cloglog_inv_basic() {
        // cloglog_inv is the Gumbel CDF: 1 - exp(-exp(x))
        // cloglog_inv(0) = 1 - exp(-1) ≈ 0.6321
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((cloglog_inv(0.0) - expected).abs() < 1e-14);

        // Monotonically increasing
        assert!(cloglog_inv(-2.0) < cloglog_inv(0.0));
        assert!(cloglog_inv(0.0) < cloglog_inv(2.0));

        // Bounded by (0, 1)
        assert!(cloglog_inv(-100.0) >= 0.0);
        assert!(cloglog_inv(100.0) <= 1.0);
    }

    #[test]
    fn cloglog_inverse_relationship() {
        // cloglog and cloglog_inv are inverses
        let p = 0.3;
        assert!((cloglog_inv(cloglog(p)) - p).abs() < 1e-14);

        let x = 1.5;
        assert!((cloglog(cloglog_inv(x)) - x).abs() < 1e-14);

        // Test more values
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert!((cloglog_inv(cloglog(p)) - p).abs() < 1e-13);
        }
    }

    #[test]
    fn loglog_basic() {
        // loglog(p) = -log(-log(p))
        // loglog(exp(-1)) = -log(-log(exp(-1))) = -log(1) = 0
        let p_at_zero = (-1.0_f64).exp();
        assert!((loglog(p_at_zero) - 0.0).abs() < 1e-14);

        // Monotonically increasing (like cloglog)
        assert!(loglog(0.3) < loglog(0.5));
        assert!(loglog(0.5) < loglog(0.7));

        // Boundary behavior (opposite direction of cloglog)
        assert!(loglog(0.0).is_infinite() && loglog(0.0) < 0.0);
        assert!(loglog(1.0).is_infinite() && loglog(1.0) > 0.0);
    }

    #[test]
    fn loglog_inv_basic() {
        // loglog_inv(x) = exp(-exp(-x)) - the Gumbel (minimum) CDF
        // loglog_inv(0) = exp(-exp(0)) = exp(-1) ≈ 0.3679
        let expected = (-1.0_f64).exp();
        assert!((loglog_inv(0.0) - expected).abs() < 1e-14);

        // Monotonically increasing
        assert!(loglog_inv(-2.0) < loglog_inv(0.0));
        assert!(loglog_inv(0.0) < loglog_inv(2.0));

        // Bounded by (0, 1)
        assert!(loglog_inv(-100.0) >= 0.0);
        assert!(loglog_inv(100.0) <= 1.0);
    }

    #[test]
    fn loglog_inverse_relationship() {
        // loglog and loglog_inv are inverses
        let p = 0.3;
        assert!((loglog_inv(loglog(p)) - p).abs() < 1e-14);

        let x = -1.5;
        assert!((loglog(loglog_inv(x)) - x).abs() < 1e-14);

        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert!((loglog_inv(loglog(p)) - p).abs() < 1e-13);
        }
    }

    #[test]
    fn cauchit_basic() {
        // cauchit(0.5) = tan(0) = 0
        assert!((cauchit(0.5) - 0.0).abs() < 1e-14);

        // cauchit(0.75) = tan(π/4) = 1
        assert!((cauchit(0.75) - 1.0).abs() < 1e-14);

        // cauchit(0.25) = tan(-π/4) = -1
        assert!((cauchit(0.25) - (-1.0)).abs() < 1e-14);

        // Monotonically increasing
        assert!(cauchit(0.3) < cauchit(0.5));
        assert!(cauchit(0.5) < cauchit(0.7));

        // Boundary behavior
        assert!(cauchit(0.0).is_infinite() && cauchit(0.0) < 0.0);
        assert!(cauchit(1.0).is_infinite() && cauchit(1.0) > 0.0);
    }

    #[test]
    fn cauchit_inv_basic() {
        // cauchit_inv(0) = 0.5
        assert!((cauchit_inv(0.0) - 0.5).abs() < 1e-14);

        // cauchit_inv(1) = 0.75
        assert!((cauchit_inv(1.0) - 0.75).abs() < 1e-14);

        // cauchit_inv(-1) = 0.25
        assert!((cauchit_inv(-1.0) - 0.25).abs() < 1e-14);

        // Bounded by (0, 1) for finite x
        assert!(cauchit_inv(-1000.0) > 0.0);
        assert!(cauchit_inv(1000.0) < 1.0);
    }

    #[test]
    fn cauchit_inverse_relationship() {
        // cauchit and cauchit_inv are inverses
        let p = 0.3;
        assert!((cauchit_inv(cauchit(p)) - p).abs() < 1e-14);

        let x = 2.5;
        assert!((cauchit(cauchit_inv(x)) - x).abs() < 1e-14);

        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert!((cauchit_inv(cauchit(p)) - p).abs() < 1e-13);
        }
    }

    #[test]
    fn hardshrink_basic() {
        // hardshrink(x, λ) = x if |x| > λ, else 0
        assert!((hardshrink(2.0, 0.5) - 2.0).abs() < 1e-14);
        assert!((hardshrink(-2.0, 0.5) - (-2.0)).abs() < 1e-14);
        assert!((hardshrink(0.3, 0.5) - 0.0).abs() < 1e-14);
        assert!((hardshrink(-0.3, 0.5) - 0.0).abs() < 1e-14);
        // At threshold
        assert!((hardshrink(0.5, 0.5) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn softshrink_basic() {
        // softshrink shrinks toward zero by λ
        assert!((softshrink(2.0, 0.5) - 1.5).abs() < 1e-14);
        assert!((softshrink(-2.0, 0.5) - (-1.5)).abs() < 1e-14);
        assert!((softshrink(0.3, 0.5) - 0.0).abs() < 1e-14);
        assert!((softshrink(-0.3, 0.5) - 0.0).abs() < 1e-14);
        // At threshold
        assert!((softshrink(0.5, 0.5) - 0.0).abs() < 1e-14);
        assert!((softshrink(-0.5, 0.5) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn tanhshrink_basic() {
        // tanhshrink(x) = x - tanh(x)
        assert!((tanhshrink(0.0) - 0.0).abs() < 1e-14);
        // For small x, tanhshrink ≈ x^3/3
        assert!((tanhshrink(0.1) - (0.1 - 0.1_f64.tanh())).abs() < 1e-14);
        // For large |x|, tanhshrink ≈ x - sign(x)
        assert!((tanhshrink(10.0) - 9.0).abs() < 1e-5);
        assert!((tanhshrink(-10.0) - (-9.0)).abs() < 1e-5);
    }

    #[test]
    fn celu_basic() {
        // celu(x, α) is like elu but C¹ continuous
        // For x >= 0, celu(x) = x
        assert!((celu(2.0, 1.0) - 2.0).abs() < 1e-14);
        assert!((celu(0.0, 1.0) - 0.0).abs() < 1e-14);
        // For x < 0, celu(x, α) = α * (exp(x/α) - 1)
        let expected = 1.0 * ((-1.0_f64 / 1.0).exp() - 1.0);
        assert!((celu(-1.0, 1.0) - expected).abs() < 1e-14);
        // Different alpha
        let expected2 = 2.0 * ((-1.0_f64 / 2.0).exp() - 1.0);
        assert!((celu(-1.0, 2.0) - expected2).abs() < 1e-14);
    }

    #[test]
    fn logsigmoid_basic() {
        // logsigmoid is same as log_expit_scalar
        assert!((logsigmoid(0.0) - log_expit_scalar(0.0)).abs() < 1e-14);
        assert!((logsigmoid(2.0) - log_expit_scalar(2.0)).abs() < 1e-14);
        assert!((logsigmoid(-2.0) - log_expit_scalar(-2.0)).abs() < 1e-14);
        // logsigmoid(0) = log(0.5) = -ln(2)
        assert!((logsigmoid(0.0) - (-std::f64::consts::LN_2)).abs() < 1e-14);
    }

    #[test]
    fn log1mexp_basic() {
        // log1mexp(x) = log(1 - exp(x)) for x < 0
        // log1mexp(-ln(2)) = log(1 - 0.5) = log(0.5) = -ln(2)
        assert!((log1mexp(-std::f64::consts::LN_2) - (-std::f64::consts::LN_2)).abs() < 1e-14);

        // For x -> -inf, exp(x) -> 0, so log1mexp(x) -> log(1) = 0
        assert!((log1mexp(-100.0) - 0.0).abs() < 1e-40);

        // log1mexp(0) = log(0) = -inf
        assert!(log1mexp(0.0).is_infinite() && log1mexp(0.0) < 0.0);

        // log1mexp(x) is NaN for x > 0
        assert!(log1mexp(1.0).is_nan());
    }

    #[test]
    fn log1pexp_basic() {
        // log1pexp is same as softplus
        assert!((log1pexp(0.0) - softplus(0.0)).abs() < 1e-14);
        assert!((log1pexp(2.0) - softplus(2.0)).abs() < 1e-14);
        assert!((log1pexp(-2.0) - softplus(-2.0)).abs() < 1e-14);
    }

    #[test]
    fn xlogx_basic() {
        // xlogx(0) = 0 by convention
        assert!((xlogx(0.0) - 0.0).abs() < 1e-14);

        // xlogx(1) = 1 * log(1) = 0
        assert!((xlogx(1.0) - 0.0).abs() < 1e-14);

        // xlogx(e) = e * log(e) = e
        assert!((xlogx(std::f64::consts::E) - std::f64::consts::E).abs() < 1e-14);

        // xlogx(x) for x > 0
        assert!((xlogx(2.0) - 2.0 * 2.0_f64.ln()).abs() < 1e-14);

        // xlogx(x) is NaN for x < 0
        assert!(xlogx(-1.0).is_nan());
    }

    #[test]
    fn negentropy_basic() {
        // negentropy is same as xlogx
        assert!((negentropy(0.0) - xlogx(0.0)).abs() < 1e-14);
        assert!((negentropy(0.5) - xlogx(0.5)).abs() < 1e-14);
        assert!((negentropy(2.0) - xlogx(2.0)).abs() < 1e-14);
    }

    #[test]
    fn binary_cross_entropy_basic() {
        // BCE(1, 1) = 0 (perfect prediction)
        assert!((binary_cross_entropy(1.0, 1.0) - 0.0).abs() < 1e-14);

        // BCE(0, 0) = 0 (perfect prediction)
        assert!((binary_cross_entropy(0.0, 0.0) - 0.0).abs() < 1e-14);

        // BCE(1, 0.5) = -log(0.5) = ln(2)
        assert!((binary_cross_entropy(1.0, 0.5) - std::f64::consts::LN_2).abs() < 1e-14);

        // BCE(0, 0.5) = -log(0.5) = ln(2)
        assert!((binary_cross_entropy(0.0, 0.5) - std::f64::consts::LN_2).abs() < 1e-14);

        // BCE(0.5, 0.5) = -0.5*log(0.5) - 0.5*log(0.5) = ln(2)
        assert!((binary_cross_entropy(0.5, 0.5) - std::f64::consts::LN_2).abs() < 1e-14);
    }
}
