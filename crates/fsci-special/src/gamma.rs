#![forbid(unsafe_code)]

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind, SpecialResult,
    SpecialTensor, not_yet_implemented,
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
        SpecialTensor::Empty => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn gamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        return Err(SpecialError {
            function: "gamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "gamma pole at nonpositive integer",
        });
    }
    Ok(gamma_core(x))
}

fn gammaln_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        return Err(SpecialError {
            function: "gammaln",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "gammaln pole at nonpositive integer",
        });
    }
    let value = gamma_core(x);
    Ok(value.abs().ln())
}

fn digamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        return Err(SpecialError {
            function: "digamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "digamma pole at nonpositive integer",
        });
    }
    Ok(digamma_core(x))
}

fn trigamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        return Err(SpecialError {
            function: "polygamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "trigamma pole at nonpositive integer",
        });
    }
    Ok(trigamma_core(x))
}

fn rgamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    let gamma_value = gamma_scalar(x, mode)?;
    Ok(1.0 / gamma_value)
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
