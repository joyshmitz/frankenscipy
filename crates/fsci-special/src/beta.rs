#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::gamma;
use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind, SpecialResult,
    SpecialTensor, not_yet_implemented, record_special_trace,
};

pub const BETA_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "beta",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "use gamma/gammaln composition in stable space",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large-parameter regime uses logspace stabilization",
            },
        ],
        notes: "Symmetry beta(a,b)=beta(b,a) must hold in strict mode and hardened mode.",
    },
    DispatchPlan {
        function: "betaln",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "direct logspace composition",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "a+b sufficiently large",
            },
        ],
        notes: "Primary path for underflow-prone beta regions.",
    },
    DispatchPlan {
        function: "betainc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "x in lower-tail region",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "x in upper-tail region",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "parameter shifts for stability",
            },
        ],
        notes: "Strict mode preserves SciPy endpoint behavior at x=0 and x=1.",
    },
];

pub fn beta(a: &SpecialTensor, b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("beta", a, b, mode, |x, y| beta_scalar(x, y, mode))
}

pub fn betaln(a: &SpecialTensor, b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("betaln", a, b, mode, |x, y| betaln_scalar(x, y, mode))
}

pub fn betainc(
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_ternary("betainc", a, b, x, mode, |av, bv, xv| {
        betainc_scalar(av, bv, xv, mode)
    })
}

/// Beta distribution CDF.
///
/// Matches `scipy.special.btdtr(a, b, x)`.
#[must_use]
pub fn btdtr(a: f64, b: f64, x: f64) -> f64 {
    betainc_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse beta distribution CDF.
///
/// Matches `scipy.special.btdtri(a, b, y)`.
#[must_use]
pub fn btdtri(a: f64, b: f64, y: f64) -> f64 {
    if a.is_nan() || b.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return 1.0;
    }

    invert_monotone_unit_interval(|x| btdtr(a, b, x), y)
}

/// F-distribution CDF.
///
/// Matches `scipy.special.fdtr(dfn, dfd, x)`.
#[must_use]
pub fn fdtr(dfn: f64, dfd: f64, x: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let z = dfn * x / (dfn * x + dfd);
    btdtr(0.5 * dfn, 0.5 * dfd, z)
}

/// Inverse F-distribution CDF.
///
/// Matches `scipy.special.fdtri(dfn, dfd, y)`.
#[must_use]
pub fn fdtri(dfn: f64, dfd: f64, y: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 || !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return f64::INFINITY;
    }

    let z = btdtri(0.5 * dfn, 0.5 * dfd, y);
    dfd * z / (dfn * (1.0 - z))
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
        (SpecialTensor::RealScalar(lhs), SpecialTensor::RealScalar(rhs)) => {
            kernel(*lhs, *rhs).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(lhs), SpecialTensor::RealScalar(rhs)) => lhs
            .iter()
            .copied()
            .map(|value| kernel(value, *rhs))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(lhs), SpecialTensor::RealVec(rhs)) => rhs
            .iter()
            .copied()
            .map(|value| kernel(*lhs, value))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(lhs), SpecialTensor::RealVec(rhs)) => {
            if lhs.len() != rhs.len() {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("lhs_len={},rhs_len={}", lhs.len(), rhs.len()),
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
            lhs.iter()
                .copied()
                .zip(rhs.iter().copied())
                .map(|(left, right)| kernel(left, right))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::ComplexScalar(_), _)
        | (SpecialTensor::ComplexVec(_), _)
        | (_, SpecialTensor::ComplexScalar(_))
        | (_, SpecialTensor::ComplexVec(_)) => {
            not_yet_implemented(function, mode, "complex-valued path pending")
        }
        _ => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "input=empty",
                "fail_closed",
                "empty tensor is not a valid special-function input",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "empty tensor is not a valid special-function input",
            })
        }
    }
}

fn map_real_ternary<F>(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    c: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64, f64) -> Result<f64, SpecialError>,
{
    match (a, b, c) {
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealScalar(cv),
        ) => kernel(*av, *bv, *cv).map(SpecialTensor::RealScalar),
        (
            SpecialTensor::RealVec(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealScalar(cv),
        ) => av
            .iter()
            .copied()
            .map(|lhs| kernel(lhs, *bv, *cv))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealVec(bv),
            SpecialTensor::RealScalar(cv),
        ) => bv
            .iter()
            .copied()
            .map(|rhs| kernel(*av, rhs, *cv))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealVec(cv),
        ) => cv
            .iter()
            .copied()
            .map(|xv| kernel(*av, *bv, xv))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::ComplexScalar(_), _, _)
        | (SpecialTensor::ComplexVec(_), _, _)
        | (_, SpecialTensor::ComplexScalar(_), _)
        | (_, SpecialTensor::ComplexVec(_), _)
        | (_, _, SpecialTensor::ComplexScalar(_))
        | (_, _, SpecialTensor::ComplexVec(_)) => {
            not_yet_implemented(function, mode, "complex-valued path pending")
        }
        _ => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "unsupported_broadcast_pattern",
                "fail_closed",
                "unsupported broadcast pattern for ternary inputs",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "unsupported broadcast pattern for ternary inputs",
            })
        }
    }
}

fn beta_scalar(a: f64, b: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    // Symmetry beta(a,b)=beta(b,a)
    let (a, b) = if a < b { (b, a) } else { (a, b) };

    let log_value = betaln_scalar(a, b, mode)?;
    const LN_MAX: f64 = 709.782_712_893_384;
    const LN_MIN: f64 = -745.133_219_101_941_1;

    if log_value > LN_MAX {
        if matches!(mode, RuntimeMode::Hardened) {
            record_special_trace(
                "beta",
                mode,
                "overflow_risk",
                format!("a={a},b={b},log_beta={log_value}"),
                "fail_closed",
                "beta overflow risk",
                true,
            );
            return Err(SpecialError {
                function: "beta",
                kind: SpecialErrorKind::OverflowRisk,
                mode,
                detail: "beta overflow risk",
            });
        }
        record_special_trace(
            "beta",
            mode,
            "overflow_risk",
            format!("a={a},b={b},log_beta={log_value}"),
            "returned_inf",
            "strict overflow fallback",
            true,
        );
        return Ok(f64::INFINITY);
    }
    if log_value < LN_MIN {
        record_special_trace(
            "beta",
            mode,
            "underflow_risk",
            format!("a={a},b={b},log_beta={log_value}"),
            "returned_zero",
            "underflow-safe clamp to zero",
            true,
        );
        return Ok(0.0);
    }

    Ok(log_value.exp())
}

fn betaln_scalar(a: f64, b: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if a.is_nan() || b.is_nan() {
        return Ok(f64::NAN);
    }
    if matches!(mode, RuntimeMode::Hardened) && (a <= 0.0 || b <= 0.0) {
        record_special_trace(
            "betaln",
            mode,
            "domain_error",
            format!("a={a},b={b}"),
            "fail_closed",
            "betaln principal domain requires positive parameters",
            false,
        );
        return Err(SpecialError {
            function: "betaln",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "betaln principal domain requires positive parameters",
        });
    }
    if a <= 0.0 || b <= 0.0 {
        record_special_trace(
            "betaln",
            mode,
            "domain_error",
            format!("a={a},b={b}"),
            "returned_nan",
            "strict domain fallback",
            false,
        );
        return Ok(f64::NAN);
    }

    let lg_a = gammaln_scalar(a, RuntimeMode::Strict)?;
    let lg_b = gammaln_scalar(b, RuntimeMode::Strict)?;
    let lg_ab = gammaln_scalar(a + b, RuntimeMode::Strict)?;
    Ok(lg_a + lg_b - lg_ab)
}

fn betainc_scalar(a: f64, b: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if !(0.0..=1.0).contains(&x) {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "fail_closed",
                    "betainc domain requires x in [0, 1]",
                    false,
                );
                Err(SpecialError {
                    function: "betainc",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "betainc domain requires x in [0, 1]",
                })
            }
        };
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }
    if a <= 0.0 || b <= 0.0 {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "fail_closed",
                    "betainc requires positive shape parameters",
                    false,
                );
                Err(SpecialError {
                    function: "betainc",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "betainc requires positive shape parameters",
                })
            }
        };
    }

    let ln_beta = betaln_scalar(a, b, RuntimeMode::Strict)?;
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        Ok(front * betacf(a, b, x) / a)
    } else {
        Ok(1.0 - front * betacf(b, a, 1.0 - x) / b)
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITERS: usize = 200;
    const EPS: f64 = 3.0e-14;
    const MIN_NUM: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < MIN_NUM {
        d = MIN_NUM;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITERS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < MIN_NUM {
            d = MIN_NUM;
        }
        c = 1.0 + aa / c;
        if c.abs() < MIN_NUM {
            c = MIN_NUM;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa2 = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < MIN_NUM {
            d = MIN_NUM;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < MIN_NUM {
            c = MIN_NUM;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= EPS {
            break;
        }
    }

    h
}

fn invert_monotone_unit_interval(cdf: impl Fn(f64) -> f64, target: f64) -> f64 {
    let mut lo = 0.0;
    let mut hi = 1.0;

    for _ in 0..100 {
        let mid = lo + (hi - lo) * 0.5;
        let value = cdf(mid);
        if !value.is_finite() {
            hi = mid;
            continue;
        }
        if value < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < f64::EPSILON * 2.0 {
            break;
        }
    }

    lo + (hi - lo) * 0.5
}

fn gammaln_scalar(value: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    let tensor = SpecialTensor::RealScalar(value);
    let result = gamma::gammaln(&tensor, mode)?;
    match result {
        SpecialTensor::RealScalar(v) => Ok(v),
        _ => Err(SpecialError {
            function: "betaln",
            kind: SpecialErrorKind::NotYetImplemented,
            mode,
            detail: "unexpected non-scalar gammaln output",
        }),
    }
}
