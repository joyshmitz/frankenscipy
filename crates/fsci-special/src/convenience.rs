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
    SpecialTensor,
};

pub const CONVENIENCE_DISPATCH_PLAN: &[DispatchPlan] = &[DispatchPlan {
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
}];

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
    if x == 0.0 { 0.0 } else { x * y.ln() }
}

fn xlog1py_scalar(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * (1.0 + y).ln() }
}

fn expit_scalar(x: f64) -> f64 {
    if x >= 0.0 {
        let ex = (-x).exp();
        1.0 / (1.0 + ex)
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

fn logit_scalar(p: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if p <= 0.0 || p >= 1.0 {
        return match mode {
            RuntimeMode::Strict => {
                if p <= 0.0 {
                    Ok(f64::NEG_INFINITY)
                } else {
                    Ok(f64::INFINITY)
                }
            }
            RuntimeMode::Hardened => Err(SpecialError {
                function: "logit",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "p must be in (0, 1)",
            }),
        };
    }
    Ok((p / (1.0 - p)).ln())
}

fn entr_scalar(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if x < 0.0 {
        f64::NEG_INFINITY
    } else {
        -x * x.ln()
    }
}

fn rel_entr_scalar(x: f64, y: f64) -> f64 {
    if x == 0.0 && y >= 0.0 {
        0.0
    } else if x > 0.0 && y > 0.0 {
        x * (x / y).ln()
    } else {
        f64::INFINITY
    }
}

fn kl_div_scalar(x: f64, y: f64) -> f64 {
    if x == 0.0 && y >= 0.0 {
        y
    } else if x > 0.0 && y > 0.0 {
        x * (x / y).ln() - x + y
    } else {
        f64::INFINITY
    }
}

// ══════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════

fn map_real<F>(
    function: &'static str,
    input: &SpecialTensor,
    _mode: RuntimeMode,
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
            .map(&kernel)
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        _ => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode: _mode,
            detail: "unsupported input type",
        }),
    }
}

fn map_real_binary<F>(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<f64, SpecialError>,
{
    match (a, b) {
        (SpecialTensor::RealScalar(x), SpecialTensor::RealScalar(y)) => {
            kernel(*x, *y).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(xs), SpecialTensor::RealVec(ys)) => {
            if xs.len() != ys.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "input vectors must have same length",
                });
            }
            xs.iter()
                .zip(ys.iter())
                .map(|(&x, &y)| kernel(x, y))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        _ => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "unsupported input type combination",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_scalar(result: SpecialResult) -> f64 {
        match result.expect("should succeed") {
            SpecialTensor::RealScalar(v) => v,
            other => panic!("expected RealScalar, got {other:?}"),
        }
    }

    fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: got {actual}, expected {expected}"
        );
    }

    #[test]
    fn sinc_at_zero() {
        let result = eval_scalar(sinc(&SpecialTensor::RealScalar(0.0), RuntimeMode::Strict));
        assert_close(result, 1.0, 1e-15, "sinc(0)");
    }

    #[test]
    fn sinc_at_integer() {
        // sinc(n) = 0 for non-zero integers
        for n in 1..=5 {
            let result = eval_scalar(sinc(
                &SpecialTensor::RealScalar(n as f64),
                RuntimeMode::Strict,
            ));
            assert!(result.abs() < 1e-15, "sinc({n}) should be 0, got {result}");
        }
    }

    #[test]
    fn sinc_at_half() {
        // sinc(0.5) = sin(π/2) / (π/2) = 1/(π/2) = 2/π
        let result = eval_scalar(sinc(&SpecialTensor::RealScalar(0.5), RuntimeMode::Strict));
        assert_close(result, 2.0 / PI, 1e-12, "sinc(0.5)");
    }

    #[test]
    fn xlogy_zero_x() {
        let result = eval_scalar(xlogy(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            RuntimeMode::Strict,
        ));
        assert_close(result, 0.0, 1e-15, "xlogy(0, 0) = 0");
    }

    #[test]
    fn xlogy_normal() {
        let result = eval_scalar(xlogy(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(std::f64::consts::E),
            RuntimeMode::Strict,
        ));
        assert_close(result, 2.0, 1e-12, "xlogy(2, e) = 2");
    }

    #[test]
    fn logsumexp_basic() {
        // logsumexp([0, 0]) = log(2)
        assert_close(logsumexp(&[0.0, 0.0]), 2.0_f64.ln(), 1e-12, "logsumexp");
    }

    #[test]
    fn logsumexp_large_values() {
        // Should handle large values without overflow
        let result = logsumexp(&[1000.0, 1001.0]);
        assert!(result.is_finite(), "should not overflow");
        assert_close(
            result,
            1001.0 + (1.0 + (-1.0_f64).exp()).ln(),
            1e-10,
            "logsumexp large",
        );
    }

    #[test]
    fn logsumexp_empty() {
        assert!(logsumexp(&[]).is_infinite() && logsumexp(&[]).is_sign_negative());
    }

    #[test]
    fn expit_zero() {
        let result = eval_scalar(expit(&SpecialTensor::RealScalar(0.0), RuntimeMode::Strict));
        assert_close(result, 0.5, 1e-15, "expit(0) = 0.5");
    }

    #[test]
    fn expit_large_positive() {
        let result = eval_scalar(expit(
            &SpecialTensor::RealScalar(100.0),
            RuntimeMode::Strict,
        ));
        assert_close(result, 1.0, 1e-10, "expit(100) ≈ 1");
    }

    #[test]
    fn expit_large_negative() {
        let result = eval_scalar(expit(
            &SpecialTensor::RealScalar(-100.0),
            RuntimeMode::Strict,
        ));
        assert_close(result, 0.0, 1e-10, "expit(-100) ≈ 0");
    }

    #[test]
    fn logit_expit_inverse() {
        // logit(expit(x)) = x
        for &x in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            let p = eval_scalar(expit(&SpecialTensor::RealScalar(x), RuntimeMode::Strict));
            let back = eval_scalar(logit(&SpecialTensor::RealScalar(p), RuntimeMode::Strict));
            assert_close(back, x, 1e-10, &format!("logit(expit({x}))"));
        }
    }

    #[test]
    fn entr_at_zero() {
        let result = eval_scalar(entr(&SpecialTensor::RealScalar(0.0), RuntimeMode::Strict));
        assert_close(result, 0.0, 1e-15, "entr(0) = 0");
    }

    #[test]
    fn entr_at_one() {
        let result = eval_scalar(entr(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict));
        assert_close(result, 0.0, 1e-15, "entr(1) = -1*ln(1) = 0");
    }

    #[test]
    fn entr_positive() {
        let result = eval_scalar(entr(&SpecialTensor::RealScalar(0.5), RuntimeMode::Strict));
        assert_close(result, -0.5 * 0.5_f64.ln(), 1e-12, "entr(0.5)");
    }

    #[test]
    fn rel_entr_equal() {
        // rel_entr(x, x) = x * log(1) = 0
        let result = eval_scalar(rel_entr(
            &SpecialTensor::RealScalar(1.0),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Strict,
        ));
        assert_close(result, 0.0, 1e-15, "rel_entr(1,1) = 0");
    }

    #[test]
    fn rel_entr_zero_x() {
        let result = eval_scalar(rel_entr(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Strict,
        ));
        assert_close(result, 0.0, 1e-15, "rel_entr(0,1) = 0");
    }
}
