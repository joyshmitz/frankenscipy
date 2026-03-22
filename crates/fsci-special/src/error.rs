#![forbid(unsafe_code)]

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, not_yet_implemented, record_special_trace,
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
    map_unary_input(
        "erf",
        z,
        mode,
        |x| Ok(erf_scalar(x)),
        |value| Ok(erf_complex_scalar(value)),
    )
}

pub fn erfc(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_unary_input(
        "erfc",
        z,
        mode,
        |x| Ok(erfc_scalar(x)),
        |value| Ok(erfc_complex_scalar(value)),
    )
}

pub fn erfinv(y: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("erfinv", y, mode, |v| erfinv_scalar(v, mode))
}

pub fn erfcinv(y: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("erfcinv", y, mode, |v| erfcinv_scalar(v, mode))
}

fn map_unary_input<F, G>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    real_kernel: F,
    complex_kernel: G,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError>,
    G: Fn(Complex64) -> Result<Complex64, SpecialError>,
{
    match input {
        SpecialTensor::RealScalar(x) => real_kernel(*x).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .copied()
            .map(real_kernel)
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(value) => {
            complex_kernel(*value).map(SpecialTensor::ComplexScalar)
        }
        SpecialTensor::ComplexVec(values) => values
            .iter()
            .copied()
            .map(complex_kernel)
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
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

pub fn erf_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return x.signum();
    }
    erf_complex_scalar(Complex64::from_real(x)).re
}

fn erf_complex_scalar(z: Complex64) -> Complex64 {
    if z.re < 0.0 {
        return -erf_complex_scalar(-z);
    }
    if z.abs() <= 4.5 || z.re < 1.0 {
        return erf_complex_series(z);
    }
    Complex64::from_real(1.0) - erfc_complex_asymptotic(z)
}

pub fn erfc_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { 2.0 };
    }
    erfc_complex_scalar(Complex64::from_real(x)).re
}

fn erfc_complex_scalar(z: Complex64) -> Complex64 {
    if z.re < 0.0 {
        return Complex64::from_real(2.0) - erfc_complex_scalar(-z);
    }
    if z.abs() <= 4.5 || z.re < 1.0 {
        return Complex64::from_real(1.0) - erf_complex_series(z);
    }
    erfc_complex_asymptotic(z)
}

fn erf_complex_series(z: Complex64) -> Complex64 {
    let z2 = z * z;
    let mut term = z;
    let mut sum = term;

    for n in 0..80 {
        let numer = -z2 * ((2 * n + 1) as f64);
        let denom = ((n + 1) * (2 * n + 3)) as f64;
        term = term * numer / denom;
        sum = sum + term;
        if n >= 4 && term.abs() <= 1.0e-16 * sum.abs().max(1.0) {
            break;
        }
    }

    sum * TWO_INV_SQRT_PI
}

fn erfc_complex_asymptotic(z: Complex64) -> Complex64 {
    let z2 = z * z;
    let mut term = Complex64::from_real(1.0);
    let mut sum = term;

    let mut last_term_abs = term.abs();
    for n in 0..60 {
        let factor = -((2 * n + 1) as f64);
        let next_term = term * factor / (z2 * 2.0);
        let next_term_abs = next_term.abs();

        if next_term_abs > last_term_abs {
            break; // Series starts to diverge
        }

        term = next_term;
        last_term_abs = next_term_abs;
        sum = sum + term;

        if n >= 4 && term.abs() <= 1.0e-16 * sum.abs().max(1.0) {
            break;
        }
    }

    ((-z2).exp() * sum / z) / PI.sqrt()
}

pub fn erfinv_scalar(y: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
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
            RuntimeMode::Strict => {
                record_special_trace(
                    "erfinv",
                    mode,
                    "domain_error",
                    format!("input={y}"),
                    "returned_nan",
                    "out-of-domain strict fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "erfinv",
                    mode,
                    "domain_error",
                    format!("input={y}"),
                    "fail_closed",
                    "erfinv domain is [-1, 1]",
                    false,
                );
                Err(SpecialError {
                    function: "erfinv",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "erfinv domain is [-1, 1]",
                })
            }
        };
    }
    if y == 0.0 {
        return Ok(y);
    }

    let mut x = inv_norm_cdf_scalar(0.5 * (y + 1.0)) / 2.0_f64.sqrt();

    // Newton-Raphson refinement for double precision accuracy.
    if y.abs() < 0.7 {
        // Use erf for central values
        for _ in 0..2 {
            let fx = erf_scalar(x) - y;
            let dfx = TWO_INV_SQRT_PI * (-x * x).exp();
            if dfx.abs() < 1e-300 {
                break;
            }
            x -= fx / dfx;
        }
    } else {
        // Use erfc for tail values to avoid 1 - small precision loss
        let yc = if y > 0.0 { 1.0 - y } else { 1.0 + y };
        let sign = y.signum();
        x = x.abs();
        for _ in 0..2 {
            let fx = erfc_scalar(x) - yc;
            let dfx = -TWO_INV_SQRT_PI * (-x * x).exp();
            if dfx.abs() < 1e-300 {
                break;
            }
            x -= fx / dfx;
        }
        x *= sign;
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
            RuntimeMode::Strict => {
                record_special_trace(
                    "erfcinv",
                    mode,
                    "domain_error",
                    format!("input={y}"),
                    "returned_nan",
                    "out-of-domain strict fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "erfcinv",
                    mode,
                    "domain_error",
                    format!("input={y}"),
                    "fail_closed",
                    "erfcinv domain is [0, 2]",
                    false,
                );
                Err(SpecialError {
                    function: "erfcinv",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "erfcinv domain is [0, 2]",
                })
            }
        };
    }

    if y == 1.0 {
        return Ok(0.0);
    }

    Ok(-inv_norm_cdf_scalar(0.5 * y) / 2.0_f64.sqrt())
}

fn inv_norm_cdf_scalar(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    // Acklam's rational approximation for the inverse normal CDF.
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    const A: [f64; 6] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_69e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}
