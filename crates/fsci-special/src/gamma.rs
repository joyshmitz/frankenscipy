#![forbid(unsafe_code)]

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind, SpecialResult,
    SpecialTensor, not_yet_implemented, record_special_trace,
};

pub const GAMMA_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "gamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "Re(z) < 0.5 and not at poles",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "0.5 <= Re(z) < 8.0",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "Re(z) >= 8.0",
            },
        ],
        notes: "Strict mode preserves SciPy signed-zero and pole semantics; hardened mode adds fail-closed diagnostics.",
    },
    DispatchPlan {
        function: "gammaln",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "real axis shift into stable window",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |x| Stirling region",
            },
        ],
        notes: "Keep +inf pole parity with SciPy and avoid direct exp(gammaln) overflow back-conversions.",
    },
    DispatchPlan {
        function: "gammainc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "x < a + 1 via stable lower-tail expansion",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "x >= a + 1 via stable upper-tail continued fraction",
            },
        ],
        notes: "Regularized lower incomplete gamma P(a, x); strict mode preserves NaN domain fallback.",
    },
    DispatchPlan {
        function: "gammaincc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "complement from lower-tail expansion",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "direct upper-tail continued fraction for stability",
            },
        ],
        notes: "Regularized upper incomplete gamma Q(a, x)=1-P(a, x); hardened mode fail-closes malformed domains.",
    },
    DispatchPlan {
        function: "digamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "Re(z) <= 0 and away from poles",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "shift toward asymptotic region",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z|",
            },
        ],
        notes: "Negative-axis cancellation regions require dedicated numerical guards before D2 implementation.",
    },
    DispatchPlan {
        function: "polygamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "order n fixed and argument shifted off poles",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large argument regime",
            },
        ],
        notes: "Polygamma order must remain integer and nonnegative in both modes.",
    },
    DispatchPlan {
        function: "rgamma",
        steps: &[DispatchStep {
            regime: KernelRegime::BackendDelegate,
            when: "evaluate via reciprocal-safe gamma path",
        }],
        notes: "Prefer reciprocal from stabilized gammaln/gamma path to reduce overflow risk.",
    },
];

const SQRT_2PI: f64 = 2.506_628_274_631_000_7;
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    0.000_009_984_369_578_019_572,
    0.000_000_150_563_273_514_931_16,
];

pub fn gamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("gamma", z, mode, |x| gamma_scalar(x, mode))
}

pub fn gammaln(x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("gammaln", x, mode, |v| gammaln_scalar(v, mode))
}

pub fn digamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("digamma", z, mode, |x| digamma_scalar(x, mode))
}

pub fn gammainc(a: &SpecialTensor, x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("gammainc", a, x, mode, |shape, arg| {
        gammainc_scalar(shape, arg, mode)
    })
}

pub fn gammaincc(a: &SpecialTensor, x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("gammaincc", a, x, mode, |shape, arg| {
        gammaincc_scalar(shape, arg, mode)
    })
}

pub fn polygamma(n: usize, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match n {
        0 => digamma(z, mode),
        1 => map_real_input("polygamma", z, mode, |x| trigamma_scalar(x, mode)),
        2 => map_real_input("polygamma", z, mode, |x| tetragamma_scalar(x, mode)),
        _ => map_real_input("polygamma", z, mode, |x| {
            polygamma_higher_scalar(n, x, mode)
        }),
    }
}

pub fn rgamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("rgamma", z, mode, |x| rgamma_scalar(x, mode))
}

/// Gamma distribution CDF with shape `a` and scale `b`.
///
/// Matches `scipy.special.gdtr(a, b, x)`.
#[must_use]
pub fn gdtr(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    gammainc_scalar(a, x / b, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse gamma distribution CDF with shape `a` and scale `b`.
///
/// Matches `scipy.special.gdtri(a, b, y)`.
#[must_use]
pub fn gdtri(a: f64, b: f64, y: f64) -> f64 {
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
        return f64::INFINITY;
    }

    let mut lo = 0.0;
    let mut hi = (a + 1.0).max(1.0);
    while gdtr(a, 1.0, hi) < y {
        hi *= 2.0;
        if !hi.is_finite() {
            return f64::INFINITY;
        }
    }

    for _ in 0..160 {
        let mid = 0.5 * (lo + hi);
        let value = gdtr(a, 1.0, mid);
        if value < y {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() <= 1.0e-12 * mid.abs().max(1.0) {
            break;
        }
    }

    0.5 * (lo + hi) * b
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

fn map_real_input<F>(
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
        SpecialTensor::ComplexScalar(_) | SpecialTensor::ComplexVec(_) => {
            not_yet_implemented(function, mode, "complex-valued path pending")
        }
        SpecialTensor::Empty => {
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

fn gamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "gamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "gamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "gamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "gamma pole at nonpositive integer",
        });
    }
    let value = gamma_core(x);
    if !value.is_finite() && !is_negative_integer_pole(x) && x != 0.0 {
        record_special_trace(
            "gamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

pub fn gammaln_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if is_negative_integer_pole(x) || x == 0.0 {
        if matches!(mode, RuntimeMode::Hardened) {
            record_special_trace(
                "gammaln",
                mode,
                "pole_input",
                format!("input={x}"),
                "fail_closed",
                "gammaln pole at nonpositive integer",
                false,
            );
            return Err(SpecialError {
                function: "gammaln",
                kind: SpecialErrorKind::PoleInput,
                mode,
                detail: "gammaln pole at nonpositive integer",
            });
        } else {
            return Ok(f64::INFINITY);
        }
    }

    let output = if x >= 0.5 {
        if x < 100.0 {
            lngamma_lanczos(x)
        } else {
            lngamma_positive(x)
        }
    } else {
        // Reflection formula: ln|Γ(x)| = ln(π) - ln|sin(πx)| - ln|Γ(1-x)|
        // Valid for all x < 0.5 (except poles handled above)
        let g1mx = if 1.0 - x < 100.0 {
            lngamma_lanczos(1.0 - x)
        } else {
            lngamma_positive(1.0 - x)
        };
        PI.ln() - (PI * x).sin().abs().ln() - g1mx
    };

    if !output.is_finite() && !x.is_infinite() {
        record_special_trace(
            "gammaln",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={output}"),
            false,
        );
    }
    Ok(output)
}

fn lngamma_lanczos(x: f64) -> f64 {
    let mut coeff_sum = LANCZOS_COEFFS[0];
    for (idx, coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        coeff_sum += coeff / (x + (idx as f64 - 1.0));
    }

    let t = x + 6.5;
    // ln(Γ(x)) = ln(sqrt(2π)) + (x-0.5)ln(t) - t + ln(coeff_sum)
    0.5 * (2.0 * PI).ln() + (x - 0.5) * t.ln() - t + coeff_sum.ln()
}

fn digamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "digamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "digamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "digamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "digamma pole at nonpositive integer",
        });
    }
    let value = digamma_core(x);
    if !value.is_finite() {
        record_special_trace(
            "digamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn trigamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "polygamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "trigamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "polygamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "trigamma pole at nonpositive integer",
        });
    }
    let value = trigamma_core(x);
    if !value.is_finite() {
        record_special_trace(
            "polygamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn tetragamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "polygamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "tetragamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "polygamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "tetragamma pole at nonpositive integer",
        });
    }
    let value = crate::convenience::tetragamma(x);
    if !value.is_finite() {
        record_special_trace(
            "polygamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn polygamma_higher_scalar(order: usize, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "polygamma",
            mode,
            "pole_input",
            format!("order={order},input={x}"),
            "fail_closed",
            "higher-order polygamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "polygamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "higher-order polygamma pole at nonpositive integer",
        });
    }
    let value = polygamma_higher_core(order, x);
    if !value.is_finite() {
        record_special_trace(
            "polygamma",
            mode,
            "non_finite_output",
            format!("order={order},input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn rgamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    let gamma_value = gamma_scalar(x, mode)?;
    let value = 1.0 / gamma_value;
    if !value.is_finite() {
        record_special_trace(
            "rgamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

pub fn gammainc_scalar(a: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    validate_incomplete_gamma_domain("gammainc", a, x, mode)?;
    let (p, _) = regularized_gamma_pair(a, x, mode)?;
    Ok(p)
}

pub fn gammaincc_scalar(a: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    validate_incomplete_gamma_domain("gammaincc", a, x, mode)?;
    let (_, q) = regularized_gamma_pair(a, x, mode)?;
    Ok(q)
}

fn validate_incomplete_gamma_domain(
    function: &'static str,
    a: f64,
    x: f64,
    mode: RuntimeMode,
) -> Result<(), SpecialError> {
    if a.is_nan() || x.is_nan() {
        return Ok(());
    }
    if !a.is_finite() || a <= 0.0 || x < 0.0 {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("a={a},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(())
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("a={a},x={x}"),
                    "fail_closed",
                    "regularized incomplete gamma requires finite a>0 and x>=0",
                    false,
                );
                Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "regularized incomplete gamma requires finite a>0 and x>=0",
                })
            }
        };
    }
    Ok(())
}

fn regularized_gamma_pair(a: f64, x: f64, mode: RuntimeMode) -> Result<(f64, f64), SpecialError> {
    if a.is_nan() || x.is_nan() {
        return Ok((f64::NAN, f64::NAN));
    }
    if !a.is_finite() || a <= 0.0 || x < 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }
    if x == 0.0 {
        return Ok((0.0, 1.0));
    }
    if x.is_infinite() {
        return Ok((1.0, 0.0));
    }

    const EPS: f64 = 1.0e-14;
    const MAX_ITERS: usize = 200;
    const FPMIN: f64 = 1.0e-300;

    let lg = gammaln_scalar(a, RuntimeMode::Strict)?;
    let prefactor = (-x + a * x.ln() - lg).exp();
    let (p, q) = if x < a + 1.0 {
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        for _ in 0..MAX_ITERS {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() <= sum.abs() * EPS {
                break;
            }
        }
        let lower = prefactor * sum;
        let p_value = clamp_unit_interval(lower);
        let q_value = clamp_unit_interval(1.0 - p_value);
        (p_value, q_value)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=MAX_ITERS {
            let i_f = i as f64;
            let an = -i_f * (i_f - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() <= EPS {
                break;
            }
        }
        let upper = prefactor * h;
        let q_value = clamp_unit_interval(upper);
        let p_value = clamp_unit_interval(1.0 - q_value);
        (p_value, q_value)
    };

    if !p.is_finite() {
        record_special_trace(
            "gammainc",
            mode,
            "non_finite_output",
            format!("a={a},x={x}"),
            "returned_non_finite",
            format!("output={p}"),
            false,
        );
    }
    if !q.is_finite() {
        record_special_trace(
            "gammaincc",
            mode,
            "non_finite_output",
            format!("a={a},x={x}"),
            "returned_non_finite",
            format!("output={q}"),
            false,
        );
    }

    Ok((p, q))
}

fn clamp_unit_interval(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    value.clamp(0.0, 1.0)
}

fn gamma_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if is_negative_integer_pole(x) {
        // SciPy returns NaN for negative integer poles
        return f64::NAN;
    }

    if x < 0.5 {
        // Reflection formula: Γ(x) = π / (sin(πx) * Γ(1-x))
        let sin_pi_x = (PI * x).sin();
        if sin_pi_x == 0.0 {
            // This matches x = 0 handled above, or integer x < 0
            if x == 0.0 {
                return if x.is_sign_negative() {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
            }
            return f64::NAN;
        }
        return PI / (sin_pi_x * gamma_core(1.0 - x));
    }

    gamma_lanczos(x)
}

fn gamma_lanczos(x: f64) -> f64 {
    let x_minus_1 = x - 1.0;
    let mut coeff_sum = LANCZOS_COEFFS[0];
    for (idx, coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        coeff_sum += coeff / (x_minus_1 + idx as f64);
    }

    let t = x_minus_1 + 7.5;
    SQRT_2PI * t.powf(x_minus_1 + 0.5) * (-t).exp() * coeff_sum
}

fn digamma_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if is_negative_integer_pole(x) {
        return f64::NAN;
    }

    if x < 0.0 {
        return digamma_core(1.0 - x) - PI / (PI * x).tan();
    }

    let mut shifted = x;
    let mut acc = 0.0;
    while shifted < 8.0 {
        acc -= 1.0 / shifted;
        shifted += 1.0;
    }

    let inv = 1.0 / shifted;
    let inv2 = inv * inv;
    acc + shifted.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

fn trigamma_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { f64::NAN };
    }
    if is_negative_integer_pole(x) {
        return f64::NAN;
    }

    if x < 0.0 {
        let sin_pi_x = (PI * x).sin();
        return (PI * PI) / (sin_pi_x * sin_pi_x) - trigamma_core(1.0 - x);
    }

    let mut shifted = x;
    let mut acc = 0.0;
    while shifted < 8.0 {
        acc += 1.0 / (shifted * shifted);
        shifted += 1.0;
    }

    let inv = 1.0 / shifted;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let inv5 = inv3 * inv2;
    let inv7 = inv5 * inv2;
    acc + inv + 0.5 * inv2 + inv3 / 6.0 - inv5 / 30.0 + inv7 / 42.0
}

fn polygamma_higher_core(order: usize, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return polygamma_sign(order) * f64::INFINITY;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { f64::NAN };
    }
    if is_negative_integer_pole(x) {
        return f64::NAN;
    }

    let sign = polygamma_sign(order);
    let factorial = factorial_f64(order);
    let order_plus_one = order as f64 + 1.0;

    let mut shifted = x;
    let mut correction = 0.0;
    while shifted < 8.0 {
        correction += sign * factorial / shifted.powf(order_plus_one);
        shifted += 1.0;
    }

    correction + sign * factorial * crate::convenience::hurwitz_zeta(order_plus_one, shifted)
}

fn polygamma_sign(order: usize) -> f64 {
    if order % 2 == 1 { 1.0 } else { -1.0 }
}

fn factorial_f64(order: usize) -> f64 {
    (1..=order).fold(1.0, |acc, value| acc * value as f64)
}

fn is_negative_integer_pole(x: f64) -> bool {
    x.is_finite() && x < 0.0 && x.fract() == 0.0
}

/// Log-gamma function computed directly via Stirling series.
/// Avoids the overflow that gamma_core hits for large arguments.
/// Only valid for x > 0.
fn lngamma_positive(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    if x < 8.0 {
        // Shift upward using recurrence: lnΓ(x) = lnΓ(x+n) - ln(x(x+1)...(x+n-1))
        let mut shifted = x;
        let mut correction = 0.0;
        while shifted < 8.0 {
            correction += shifted.ln();
            shifted += 1.0;
        }
        stirling_lngamma(shifted) - correction
    } else {
        stirling_lngamma(x)
    }
}

/// Stirling series for lnΓ(x), valid for large x.
fn stirling_lngamma(x: f64) -> f64 {
    let half_ln_2pi = 0.5 * (2.0 * PI).ln();
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // Bernoulli number series: lnΓ(x) ≈ (x-0.5)*ln(x) - x + 0.5*ln(2π)
    //   + 1/(12x) - 1/(360x³) + 1/(1260x⁵) - 1/(1680x⁷)
    (x - 0.5) * x.ln() - x
        + half_ln_2pi
        + inv * (1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 / 1680.0)))
}

// ══════════════════════════════════════════════════════════════════════
// Combinatorial Functions
// ══════════════════════════════════════════════════════════════════════

/// Compute n! (factorial).
///
/// Matches `scipy.special.factorial(n, exact=True)`.
/// Returns f64 to handle large factorials (overflows to infinity).
pub fn factorial(n: u64) -> f64 {
    // Use lookup table for small values, gamma for large
    if n <= 20 {
        FACTORIALS[n as usize]
    } else {
        // n! = gamma(n+1)
        gamma_core((n + 1) as f64)
    }
}

/// Compute the double factorial n!! = n * (n-2) * (n-4) * ... * (1 or 2).
///
/// Matches `scipy.special.factorial2(n, exact=True)`.
///
/// For odd n: n!! = n * (n-2) * ... * 3 * 1
/// For even n: n!! = n * (n-2) * ... * 4 * 2
///
/// Special cases:
/// - 0!! = 1 (by convention)
/// - (-1)!! = 1 (by convention)
/// - For negative n < -1, returns 0
pub fn factorial2(n: i64) -> f64 {
    match n.cmp(&0) {
        std::cmp::Ordering::Less => {
            // (-1)!! = 1 by convention, others are 0
            if n == -1 { 1.0 } else { 0.0 }
        }
        std::cmp::Ordering::Equal => 1.0, // 0!! = 1
        std::cmp::Ordering::Greater => {
            if n <= 33 {
                // Direct computation for small values
                let mut result = 1.0_f64;
                let mut k = n;
                while k > 1 {
                    result *= k as f64;
                    k -= 2;
                }
                result
            } else {
                // For large n, use relation to gamma function:
                // n!! = 2^(n/2) * Gamma(n/2 + 1) for even n
                // n!! = 2^((n+1)/2) * Gamma((n+1)/2) / sqrt(pi) for odd n
                let nf = n as f64;
                if n % 2 == 0 {
                    // Even: n!! = 2^(n/2) * (n/2)!
                    let half_n = nf / 2.0;
                    2.0_f64.powf(half_n) * gamma_core(half_n + 1.0)
                } else {
                    // Odd: n!! = n! / (n/2)! / 2^(n/2)
                    // Or equivalently: n!! = sqrt(2/pi) * 2^((n+1)/2) * Gamma((n+1)/2 + 1/2)
                    let half_n_plus_1 = (nf + 1.0) / 2.0;
                    2.0_f64.powf(half_n_plus_1) * gamma_core(half_n_plus_1) / PI.sqrt()
                }
            }
        }
    }
}

const FACTORIALS: [f64; 21] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5_040.0,
    40_320.0,
    362_880.0,
    3_628_800.0,
    39_916_800.0,
    479_001_600.0,
    6_227_020_800.0,
    87_178_291_200.0,
    1_307_674_368_000.0,
    20_922_789_888_000.0,
    355_687_428_096_000.0,
    6_402_373_705_728_000.0,
    121_645_100_408_832_000.0,
    2_432_902_008_176_640_000.0,
];

/// Compute the binomial coefficient C(n, k) = n! / (k! * (n-k)!).
///
/// Matches `scipy.special.comb(n, k, exact=True)` for nonneg integer inputs.
/// Returns 0 for invalid inputs (k > n or negative-equivalent).
pub fn comb(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    // Use symmetry: C(n, k) = C(n, n-k) to minimize multiplications
    let k = k.min(n - k);
    if k <= 20 && n <= 170 {
        // Direct computation to avoid floating-point drift
        let mut result = 1.0_f64;
        for i in 0..k {
            result *= (n - i) as f64;
            result /= (i + 1) as f64;
        }
        result
    } else {
        // Use log-gamma for large values: ln(C(n,k)) = lnΓ(n+1) - lnΓ(k+1) - lnΓ(n-k+1)
        let lnc = lngamma_positive((n + 1) as f64)
            - lngamma_positive((k + 1) as f64)
            - lngamma_positive((n - k + 1) as f64);
        lnc.exp()
    }
}

/// Compute the number of k-permutations of n: P(n, k) = n! / (n-k)!.
///
/// Matches `scipy.special.perm(n, k, exact=True)` for nonneg integer inputs.
/// Returns 0 for invalid inputs (k > n).
pub fn perm(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 {
        return 1.0;
    }
    if k <= 20 && n <= 170 {
        let mut result = 1.0_f64;
        for i in 0..k {
            result *= (n - i) as f64;
        }
        result
    } else {
        let lnp = lngamma_positive((n + 1) as f64) - lngamma_positive((n - k + 1) as f64);
        lnp.exp()
    }
}

// ══════════════════════════════════════════════════════════════════════
// Riemann Zeta Function
// ══════════════════════════════════════════════════════════════════════

/// Compute the Riemann zeta function ζ(s) for real s.
///
/// Matches `scipy.special.zeta(s)`.
///
/// Uses direct summation with Euler-Maclaurin acceleration for s > 1,
/// and the reflection formula for s < 0.
pub fn zeta(s: f64) -> f64 {
    if s.is_nan() {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY; // pole
    }
    if s == 0.0 {
        return -0.5; // ζ(0) = -1/2
    }
    if s > 1.0 {
        zeta_positive(s)
    } else if s < 0.0 {
        // Reflection formula: ζ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s) * ζ(1-s)
        let s1 = 1.0 - s;
        let z1 = zeta_positive(s1);
        let sin_half_pi_s = (PI * s / 2.0).sin();
        let gamma_1_minus_s = gamma_core(s1);
        // Compute parts carefully to avoid overflow/underflow
        let factor = 2.0_f64.powf(s) * PI.powf(s - 1.0) * sin_half_pi_s * gamma_1_minus_s;
        factor * z1
    } else {
        // 0 < s < 1: use Dirichlet eta function relation
        // ζ(s) = η(s) / (1 - 2^(1-s)) where η(s) = sum (-1)^(n+1) / n^s
        let eta = dirichlet_eta(s);
        let denom = 1.0 - 2.0_f64.powf(1.0 - s);
        if denom == 0.0 {
            return f64::NAN;
        }
        eta / denom
    }
}

/// Zeta for s > 1 via Euler-Maclaurin summation.
fn zeta_positive(s: f64) -> f64 {
    // Direct sum for first N terms + Euler-Maclaurin correction
    // Use N=20 for better performance/accuracy balance
    let n = 20;
    let mut sum = 0.0_f64;
    for k in 1..=n {
        sum += (k as f64).powf(-s);
    }

    // Euler-Maclaurin remainder: integral from N to infinity of x^(-s) dx
    let n_f = n as f64;
    let s_m_1 = s - 1.0;
    let integral = n_f.powf(-s_m_1) / s_m_1;
    let half_last = 0.5 * n_f.powf(-s);

    // Bernoulli correction terms: s/12 * n^(-s-1) - s(s+1)(s+2)/720 * n^(-s-3) + ...
    let term1 = (s / 12.0) * n_f.powf(-s - 1.0);
    let term2 = (s * (s + 1.0) * (s + 2.0) / 720.0) * n_f.powf(-s - 3.0);

    sum + integral - half_last + term1 - term2
}

/// Dirichlet eta function η(s) = sum_{n=1}^∞ (-1)^{n+1} / n^s.
/// Converges for s > 0. Uses Borwein's acceleration.
fn dirichlet_eta(s: f64) -> f64 {
    // Simple partial sum with enough terms for convergence
    let n = 100;
    let mut sum = 0.0;
    for k in 1..=n {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        sum += sign * (k as f64).powf(-s);
    }
    sum
}

/// Compute the Riemann zeta function complement: zetac(s) = zeta(s) - 1.
///
/// Matches `scipy.special.zetac(s)`.
///
/// This is useful when zeta(s) is close to 1 (i.e., for large s > 1),
/// where direct computation of zeta(s) - 1 would suffer from catastrophic
/// cancellation. For large s, zetac(s) ≈ 2^(-s).
///
/// # Arguments
/// * `s` - Real argument
///
/// # Returns
/// ζ(s) - 1 for any real s ≠ 1
pub fn zetac(s: f64) -> f64 {
    if s.is_nan() {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY; // pole of zeta
    }
    if s > 10.0 {
        // For large s > 10, zetac(s) = sum_{n=2}^∞ n^(-s) ≈ 2^(-s) + 3^(-s) + ...
        // Direct computation avoids subtraction from 1
        let mut sum = 0.0_f64;
        for n in 2..=100 {
            let term = (n as f64).powf(-s);
            if term < 1e-18 {
                break;
            }
            sum += term;
        }
        sum
    } else {
        // For smaller s, just compute zeta(s) - 1
        zeta(s) - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(value: f64) -> SpecialTensor {
        SpecialTensor::RealScalar(value)
    }

    fn get_scalar(result: SpecialResult) -> f64 {
        match result.expect("polygamma should succeed") {
            SpecialTensor::RealScalar(value) => value,
            _ => panic!("expected real scalar"),
        }
    }

    #[test]
    fn polygamma_order_two_matches_tetragamma_scalar() {
        let x = 1.5;
        let actual = get_scalar(polygamma(2, &scalar(x), RuntimeMode::Strict));
        let expected = crate::convenience::tetragamma(x);
        assert!((actual - expected).abs() <= 1.0e-12);
    }

    #[test]
    fn polygamma_order_two_supports_real_vectors() {
        let input = SpecialTensor::RealVec(vec![0.5, 1.0, 2.5]);
        let result = polygamma(2, &input, RuntimeMode::Strict).expect("vector order-2 polygamma");
        match result {
            SpecialTensor::RealVec(values) => {
                let expected = [
                    crate::convenience::tetragamma(0.5),
                    crate::convenience::tetragamma(1.0),
                    crate::convenience::tetragamma(2.5),
                ];
                assert_eq!(values.len(), expected.len());
                for (actual, expected_value) in values.iter().zip(expected.iter()) {
                    assert!((*actual - *expected_value).abs() <= 1.0e-12);
                }
            }
            _ => panic!("expected real vector"),
        }
    }

    #[test]
    fn polygamma_order_two_hardened_rejects_poles() {
        let err = polygamma(2, &scalar(-1.0), RuntimeMode::Hardened)
            .expect_err("hardened mode should reject tetragamma poles");
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
    }

    #[test]
    fn polygamma_order_three_matches_known_value_at_one() {
        let actual = get_scalar(polygamma(3, &scalar(1.0), RuntimeMode::Strict));
        let expected = PI.powi(4) / 15.0;
        assert!((actual - expected).abs() <= 1.0e-11);
    }

    #[test]
    fn polygamma_higher_orders_support_real_vectors() {
        let input = SpecialTensor::RealVec(vec![1.0, 2.0, 0.5]);
        let result = polygamma(3, &input, RuntimeMode::Strict).expect("vector order-3 polygamma");
        match result {
            SpecialTensor::RealVec(values) => {
                let expected = [PI.powi(4) / 15.0, PI.powi(4) / 15.0 - 6.0, PI.powi(4)];
                assert_eq!(values.len(), expected.len());
                for (actual, expected_value) in values.iter().zip(expected.iter()) {
                    assert!((*actual - *expected_value).abs() <= 1.0e-10);
                }
            }
            _ => panic!("expected real vector"),
        }
    }

    #[test]
    fn polygamma_order_four_satisfies_recurrence_off_negative_axis() {
        let x = -0.25;
        let lhs = get_scalar(polygamma(4, &scalar(x + 1.0), RuntimeMode::Strict));
        let rhs = get_scalar(polygamma(4, &scalar(x), RuntimeMode::Strict)) + 24.0 / x.powi(5);
        assert!((lhs - rhs).abs() <= 1.0e-10);
    }

    #[test]
    fn polygamma_higher_orders_hardened_reject_poles() {
        let err = polygamma(3, &scalar(-1.0), RuntimeMode::Hardened)
            .expect_err("hardened mode should reject higher-order poles");
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
    }

    // ── factorial2 tests ─────────────────────────────────────────────────

    #[test]
    fn factorial2_small_even() {
        // 0!! = 1, 2!! = 2, 4!! = 8, 6!! = 48, 8!! = 384, 10!! = 3840
        assert_eq!(factorial2(0), 1.0);
        assert_eq!(factorial2(2), 2.0);
        assert_eq!(factorial2(4), 8.0);
        assert_eq!(factorial2(6), 48.0);
        assert_eq!(factorial2(8), 384.0);
        assert_eq!(factorial2(10), 3840.0);
    }

    #[test]
    fn factorial2_small_odd() {
        // 1!! = 1, 3!! = 3, 5!! = 15, 7!! = 105, 9!! = 945
        assert_eq!(factorial2(1), 1.0);
        assert_eq!(factorial2(3), 3.0);
        assert_eq!(factorial2(5), 15.0);
        assert_eq!(factorial2(7), 105.0);
        assert_eq!(factorial2(9), 945.0);
    }

    #[test]
    fn factorial2_negative() {
        // (-1)!! = 1 by convention, others are 0
        assert_eq!(factorial2(-1), 1.0);
        assert_eq!(factorial2(-2), 0.0);
        assert_eq!(factorial2(-5), 0.0);
    }

    #[test]
    fn factorial2_large() {
        // Test gamma-based computation for large values
        // 20!! = 3715891200
        let computed = factorial2(20);
        let expected = 3_715_891_200.0_f64;
        assert!((computed - expected).abs() / expected < 1e-10);
    }

    // ── zetac tests ──────────────────────────────────────────────────────

    #[test]
    fn zetac_small_s() {
        // zetac(2) = zeta(2) - 1 = π²/6 - 1 ≈ 0.6449340668
        let result = zetac(2.0);
        let expected = PI * PI / 6.0 - 1.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn zetac_large_s() {
        // For large s, zetac(s) ≈ 2^(-s)
        let s = 20.0;
        let result = zetac(s);
        let approx = 2.0_f64.powf(-s);
        // Should be close to 2^(-s), within a few percent
        assert!((result - approx).abs() / approx < 0.01);
    }

    #[test]
    fn zetac_consistency() {
        // zetac(s) should equal zeta(s) - 1
        for s in [3.0, 4.0, 5.0, 10.0] {
            let zetac_result = zetac(s);
            let zeta_minus_one = zeta(s) - 1.0;
            assert!((zetac_result - zeta_minus_one).abs() < 1e-12);
        }
    }

    #[test]
    fn zetac_pole() {
        // At s=1, zetac should return infinity
        assert!(zetac(1.0).is_infinite());
    }
}
