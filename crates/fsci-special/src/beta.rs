#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::gamma;
use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind, SpecialResult,
    SpecialTensor, not_yet_implemented,
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
        _ => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
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
        _ => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "unsupported broadcast pattern for ternary inputs",
        }),
    }
}

fn beta_scalar(a: f64, b: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    let log_value = betaln_scalar(a, b, mode)?;
    Ok(log_value.exp())
}

fn betaln_scalar(a: f64, b: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if a.is_nan() || b.is_nan() {
        return Ok(f64::NAN);
    }
    if matches!(mode, RuntimeMode::Hardened) && (a <= 0.0 || b <= 0.0) {
        return Err(SpecialError {
            function: "betaln",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "betaln principal domain requires positive parameters",
        });
    }
    if a <= 0.0 || b <= 0.0 {
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
            RuntimeMode::Strict => Ok(f64::NAN),
            RuntimeMode::Hardened => Err(SpecialError {
                function: "betainc",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "betainc domain requires x in [0, 1]",
            }),
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
            RuntimeMode::Strict => Ok(f64::NAN),
            RuntimeMode::Hardened => Err(SpecialError {
                function: "betainc",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "betainc requires positive shape parameters",
            }),
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
