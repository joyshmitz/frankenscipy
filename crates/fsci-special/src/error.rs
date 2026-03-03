#![forbid(unsafe_code)]

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind, SpecialResult,
    SpecialTensor, not_yet_implemented,
};

pub const ERROR_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "erf",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|z| < 1",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "|z| >= 1",
            },
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "complex continuation region",
            },
        ],
        notes: "Strict mode preserves endpoint parity: erf(0)=0, erf(+/-inf)=+/-1.",
    },
    DispatchPlan {
        function: "erfc",
        steps: &[DispatchStep {
            regime: KernelRegime::BackendDelegate,
            when: "use dedicated complement kernel to avoid 1-erf cancellation",
        }],
        notes: "Central requirement: erf(x)+erfc(x)=1 within tolerance on finite reals.",
    },
    DispatchPlan {
        function: "erfinv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|y| < 0.9",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "0.9 <= |y| < 1",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "polish via Newton/Halley refinement",
            },
        ],
        notes: "Endpoints y=+/-1 map to +/-inf with strict SciPy parity.",
    },
    DispatchPlan {
        function: "erfcinv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "map to erfinv(1-y) with tail-stable correction",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "optional refinement iteration",
            },
        ],
        notes: "Domain is [0,2] in strict mode with hardened fail-closed diagnostics for malformed inputs.",
    },
];

const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;
const TWO_INV_SQRT_PI: f64 = 2.0 * INV_SQRT_PI;

pub fn erf(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("erf", z, mode, |x| Ok(erf_scalar(x)))
}

pub fn erfc(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("erfc", z, mode, |x| Ok(erfc_scalar(x)))
}

pub fn erfinv(y: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("erfinv", y, mode, |v| erfinv_scalar(v, mode))
}

pub fn erfcinv(y: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("erfcinv", y, mode, |v| erfcinv_scalar(v, mode))
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

fn erf_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    if x == f64::NEG_INFINITY {
        return -1.0;
    }
    if x == 0.0 {
        return x;
    }

    // Abramowitz & Stegun 7.1.26 approximation.
    let sign = x.signum();
    let x_abs = x.abs();
    let p = 0.327_591_1;
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let t = 1.0 / (1.0 + p * x_abs);
    let poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t;
    sign * (1.0 - poly * (-x_abs * x_abs).exp())
}

fn erfc_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x == f64::NEG_INFINITY {
        return 2.0;
    }
    if x < 0.0 {
        return 2.0 - erfc_scalar(-x);
    }

    let p = 0.327_591_1;
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let t = 1.0 / (1.0 + p * x);
    let poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t;
    poly * (-x * x).exp()
}

fn erfinv_scalar(y: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if y.is_nan() {
        return Ok(f64::NAN);
    }
    if y == 1.0 {
        return Ok(f64::INFINITY);
    }
    if y == -1.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if y.abs() > 1.0 {
        return match mode {
            RuntimeMode::Strict => Ok(f64::NAN),
            RuntimeMode::Hardened => Err(SpecialError {
                function: "erfinv",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "erfinv domain is [-1, 1]",
            }),
        };
    }
    if y == 0.0 {
        return Ok(y);
    }

    // Winitzki approximation with two Newton refinements.
    let a = 0.147;
    let ln_term = (1.0 - y * y).ln();
    let first = 2.0 / (PI * a) + ln_term / 2.0;
    let second = (first * first - ln_term / a).sqrt();
    let mut x = (second - first).sqrt().copysign(y);

    for _ in 0..2 {
        let fx = erf_scalar(x) - y;
        let dfx = TWO_INV_SQRT_PI * (-x * x).exp();
        if dfx == 0.0 {
            break;
        }
        x -= fx / dfx;
    }

    Ok(x)
}

fn erfcinv_scalar(y: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if y.is_nan() {
        return Ok(f64::NAN);
    }
    if y == 0.0 {
        return Ok(f64::INFINITY);
    }
    if y == 2.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if !(0.0..=2.0).contains(&y) {
        return match mode {
            RuntimeMode::Strict => Ok(f64::NAN),
            RuntimeMode::Hardened => Err(SpecialError {
                function: "erfcinv",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "erfcinv domain is [0, 2]",
            }),
        };
    }

    erfinv_scalar(1.0 - y, mode)
}
