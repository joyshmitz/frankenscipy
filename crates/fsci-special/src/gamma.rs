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
        _ => not_yet_implemented(
            "polygamma",
            mode,
            "orders greater than 1 are pending D2 follow-up",
        ),
    }
}

pub fn rgamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("rgamma", z, mode, |x| rgamma_scalar(x, mode))
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
    if !value.is_finite() {
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

fn gammaln_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
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
    }
    let value = gamma_core(x);
    let output = value.abs().ln();
    if !output.is_finite() {
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

fn gammainc_scalar(a: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    validate_incomplete_gamma_domain("gammainc", a, x, mode)?;
    let (p, _) = regularized_gamma_pair(a, x, mode)?;
    Ok(p)
}

fn gammaincc_scalar(a: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
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
        return f64::NAN;
    }

    if x < 0.5 {
        return PI / ((PI * x).sin() * gamma_core(1.0 - x));
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

fn is_negative_integer_pole(x: f64) -> bool {
    x.is_finite() && x < 0.0 && x.fract() == 0.0
}
