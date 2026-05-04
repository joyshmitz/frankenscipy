#![forbid(unsafe_code)]

use std::f64::consts::{FRAC_2_PI, PI};

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, not_yet_implemented, record_special_trace,
};

pub const BESSEL_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "j0",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |x| rational/series blend",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |x| oscillatory asymptotics",
            },
        ],
        notes: "Strict mode tracks SciPy-style finite-real behavior for integer-order Bessel J0.",
    },
    DispatchPlan {
        function: "j1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |x| rational/series blend",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |x| oscillatory asymptotics",
            },
        ],
        notes: "Odd parity j1(-x)=-j1(x) must remain intact.",
    },
    DispatchPlan {
        function: "jn",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "integer order n>=2 via upward recurrence",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "negative-order parity reconstruction",
            },
        ],
        notes: "Order input must be integral; hardened mode fail-closes non-integral orders.",
    },
    DispatchPlan {
        function: "y0",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small positive x with logarithmic correction",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large x oscillatory asymptotics",
            },
        ],
        notes: "Negative real axis is domain-invalid for this real-only wrapper.",
    },
    DispatchPlan {
        function: "y1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small positive x with logarithmic correction",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large x oscillatory asymptotics",
            },
        ],
        notes: "x=0 preserves SciPy-style singular divergence in strict mode.",
    },
    DispatchPlan {
        function: "yn",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "integer order n>=2 via upward recurrence",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "negative-order parity reconstruction",
            },
        ],
        notes: "Order input must be integral; hardened mode fail-closes non-integral orders.",
    },
    DispatchPlan {
        function: "jv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |z|",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "integer-order and moderate |z| windows",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z| oscillatory region",
            },
        ],
        notes: "Negative-order reconstruction uses SciPy-compatible cancellation-safe branches.",
    },
    DispatchPlan {
        function: "yv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |z| away from singular neighborhoods",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "moderate argument / order coupling",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z|",
            },
        ],
        notes: "Origin-adjacent singularity behavior must preserve SciPy divergence semantics.",
    },
    DispatchPlan {
        function: "iv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |z|",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "order shifting in stable window",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z| or large order",
            },
        ],
        notes: "Real-negative argument handling enforces strict integer-order branch rules.",
    },
    DispatchPlan {
        function: "kv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small positive |z|",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large positive real z",
            },
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "complex continuation branch",
            },
        ],
        notes: "Hardened mode adds singular-neighborhood diagnostics near zero.",
    },
    DispatchPlan {
        function: "hankel1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "AMOS-compatible principal branch composition",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "negative-order phase mapping",
            },
        ],
        notes: "Outgoing-wave sign and phase conventions are contract-critical.",
    },
    DispatchPlan {
        function: "hankel2",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "AMOS-compatible principal branch composition",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "negative-order phase mapping",
            },
        ],
        notes: "Incoming-wave sign and phase conventions are contract-critical.",
    },
    DispatchPlan {
        function: "wright_bessel",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "nonnegative arguments summed in the log-domain series",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large-a windows collapse to early terms through log-gamma damping",
            },
        ],
        notes: "SciPy only exposes the nonnegative real domain; strict mode returns NaN outside it.",
    },
];

pub fn j0(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("j0", z, mode, |x| Ok(j0_core(x)))
}

pub fn j1(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("j1", z, mode, |x| Ok(j1_core(x)))
}

pub fn jn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("jn", n, z, mode, |order, x| jn_scalar(order, x, mode))
}

pub fn y0(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("y0", z, mode, |x| y0_scalar(x, mode))
}

pub fn y1(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("y1", z, mode, |x| y1_scalar(x, mode))
}

pub fn yn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("yn", n, z, mode, |order, x| yn_scalar(order, x, mode))
}

/// Bessel function of the first kind for real order v: J_v(z).
///
/// Uses power series for small z, asymptotic expansion for large z,
/// and reduces to j0/j1/jn for integer orders.
/// Supports both real and complex z arguments.
pub fn jv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("jv", v, z, mode, BesselKind::Jv)
}

/// Bessel function of the second kind for real order v: Y_v(z).
///
/// Y_v(z) = (J_v(z) cos(vπ) - J_{-v}(z)) / sin(vπ) for non-integer v.
/// For integer v, uses the integer-order y0/y1/yn implementations.
/// Supports both real and complex z arguments.
pub fn yv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("yv", v, z, mode, BesselKind::Yv)
}

/// Modified Bessel function of the first kind for real order v: I_v(z).
///
/// I_v(z) = (z/2)^v Σ (z²/4)^k / (k! Γ(v+k+1))
/// Supports both real and complex z arguments.
pub fn iv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("iv", v, z, mode, BesselKind::Iv)
}

/// Modified Bessel function of the first kind of order 0: I_0(z).
///
/// Convenience wrapper for iv(0, z). Matches `scipy.special.i0(z)`.
pub fn i0(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("i0", z, mode, |x| Ok(iv_scalar(0.0, x)))
}

/// Modified Bessel function of the first kind of order 1: I_1(z).
///
/// Convenience wrapper for iv(1, z). Matches `scipy.special.i1(z)`.
pub fn i1(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("i1", z, mode, |x| Ok(iv_scalar(1.0, x)))
}

/// Scalar convenience function for I_0(x).
#[must_use]
pub fn i0_scalar(x: f64) -> f64 {
    iv_scalar(0.0, x)
}

/// Scalar convenience function for I_1(x).
#[must_use]
pub fn i1_scalar(x: f64) -> f64 {
    iv_scalar(1.0, x)
}

/// Modified Bessel function of the second kind for real order v: K_v(z).
///
/// K_v(z) = π/2 (I_{-v}(z) - I_v(z)) / sin(vπ) for non-integer v.
/// Supports both real and complex z arguments.
pub fn kv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("kv", v, z, mode, BesselKind::Kv)
}

/// Hankel function of the first kind: H1_v(z) = J_v(z) + i·Y_v(z).
pub fn hankel1(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    hankel_dispatch("hankel1", v, z, mode, HankelKind::H1)
}

/// Hankel function of the second kind: H2_v(z) = J_v(z) - i·Y_v(z).
pub fn hankel2(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    hankel_dispatch("hankel2", v, z, mode, HankelKind::H2)
}

/// Wright's generalized Bessel function.
///
/// Matches `scipy.special.wright_bessel(a, b, x)` on SciPy's supported
/// nonnegative real domain.
pub fn wright_bessel(
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_ternary("wright_bessel", a, b, x, mode, |av, bv, xv| {
        wright_bessel_scalar(av, bv, xv, mode)
    })
}

/// Derivative of the Bessel function `J_v(z)`.
///
/// Matches `scipy.special.jvp(v, z, n)` for real-valued inputs. The derivative
/// order `n` is represented by `derivative_order` and defaults to 1 in SciPy.
pub fn jvp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    bessel_derivative_dispatch(
        "jvp",
        v,
        z,
        derivative_order,
        mode,
        BesselKind::Jv,
        DerivativeRule::Alternating,
    )
}

/// Derivative of the Bessel function `Y_v(z)`.
///
/// Matches `scipy.special.yvp(v, z, n)` for real-valued inputs.
pub fn yvp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    bessel_derivative_dispatch(
        "yvp",
        v,
        z,
        derivative_order,
        mode,
        BesselKind::Yv,
        DerivativeRule::Alternating,
    )
}

/// Derivative of the modified Bessel function `I_v(z)`.
///
/// Matches `scipy.special.ivp(v, z, n)` for real-valued inputs.
pub fn ivp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    bessel_derivative_dispatch(
        "ivp",
        v,
        z,
        derivative_order,
        mode,
        BesselKind::Iv,
        DerivativeRule::Positive,
    )
}

/// Derivative of the modified Bessel function `K_v(z)`.
///
/// Matches `scipy.special.kvp(v, z, n)` for real-valued inputs.
pub fn kvp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    bessel_derivative_dispatch(
        "kvp",
        v,
        z,
        derivative_order,
        mode,
        BesselKind::Kv,
        DerivativeRule::NegativeByOrder,
    )
}

/// Derivative of the Hankel function `H_v^(1)(z)`.
///
/// Returns a complex tensor: `h1vp = jvp + i*yvp`.
pub fn h1vp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    hankel_derivative_dispatch("h1vp", v, z, derivative_order, mode, HankelKind::H1)
}

/// Derivative of the Hankel function `H_v^(2)(z)`.
///
/// Returns a complex tensor: `h2vp = jvp - i*yvp`.
pub fn h2vp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    hankel_derivative_dispatch("h2vp", v, z, derivative_order, mode, HankelKind::H2)
}

#[derive(Debug, Clone, Copy)]
enum DerivativeRule {
    Alternating,
    Positive,
    NegativeByOrder,
}

#[derive(Debug, Clone, Copy)]
enum HankelKind {
    H1,
    H2,
}

fn bessel_derivative_sum<F>(
    function: &'static str,
    order: f64,
    x: f64,
    derivative_order: usize,
    mode: RuntimeMode,
    rule: DerivativeRule,
    mut kernel: F,
) -> Result<f64, SpecialError>
where
    F: FnMut(f64, f64) -> Result<f64, SpecialError>,
{
    if derivative_order == 0 {
        return kernel(order, x);
    }
    if derivative_order > 128 {
        return domain_error_by_mode(
            function,
            mode,
            format!("n={derivative_order}"),
            "derivative order is too large for stable recurrence",
        );
    }

    let mut binom = 1.0;
    let mut sum = 0.0;
    for k in 0..=derivative_order {
        let shifted_order = order - derivative_order as f64 + 2.0 * k as f64;
        let mut term = binom * kernel(shifted_order, x)?;
        if matches!(rule, DerivativeRule::Alternating) && k % 2 == 1 {
            term = -term;
        }
        sum += term;

        if k < derivative_order {
            binom *= (derivative_order - k) as f64 / (k + 1) as f64;
        }
    }

    let scale = 0.5_f64.powi(derivative_order as i32);
    let sign = if matches!(rule, DerivativeRule::NegativeByOrder) && derivative_order % 2 == 1 {
        -1.0
    } else {
        1.0
    };
    Ok(sign * scale * sum)
}

fn bessel_derivative_sum_complex<F>(
    function: &'static str,
    order: f64,
    z: Complex64,
    derivative_order: usize,
    mode: RuntimeMode,
    rule: DerivativeRule,
    mut kernel: F,
) -> Result<Complex64, SpecialError>
where
    F: FnMut(f64, Complex64) -> Result<Complex64, SpecialError>,
{
    if derivative_order == 0 {
        return kernel(order, z);
    }
    if derivative_order > 128 {
        return complex_domain_error_by_mode(
            function,
            mode,
            format!("n={derivative_order}"),
            "derivative order is too large for stable recurrence",
        );
    }

    let mut binom = 1.0;
    let mut sum = Complex64::new(0.0, 0.0);
    for k in 0..=derivative_order {
        let shifted_order = order - derivative_order as f64 + 2.0 * k as f64;
        let mut term = kernel(shifted_order, z)? * binom;
        if matches!(rule, DerivativeRule::Alternating) && k % 2 == 1 {
            term = -term;
        }
        sum = sum + term;

        if k < derivative_order {
            binom *= (derivative_order - k) as f64 / (k + 1) as f64;
        }
    }

    let scale = 0.5_f64.powi(derivative_order as i32);
    let sign = if matches!(rule, DerivativeRule::NegativeByOrder) && derivative_order % 2 == 1 {
        -1.0
    } else {
        1.0
    };
    Ok(sum * (sign * scale))
}

fn bessel_derivative_dispatch(
    function: &'static str,
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: BesselKind,
    rule: DerivativeRule,
) -> SpecialResult {
    match (v, z) {
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            bessel_derivative_real_scalar(function, *order, *x, derivative_order, mode, kind, rule)
                .map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => orders
            .iter()
            .map(|&order| {
                bessel_derivative_real_scalar(
                    function,
                    order,
                    *x,
                    derivative_order,
                    mode,
                    kind,
                    rule,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => xs
            .iter()
            .map(|&x| {
                bessel_derivative_real_scalar(
                    function,
                    *order,
                    x,
                    derivative_order,
                    mode,
                    kind,
                    rule,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(xs.iter())
                .map(|(&order, &x)| {
                    bessel_derivative_real_scalar(
                        function,
                        order,
                        x,
                        derivative_order,
                        mode,
                        kind,
                        rule,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            bessel_derivative_complex_scalar(
                function,
                *order,
                *z_val,
                derivative_order,
                mode,
                kind,
                rule,
            )
            .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => zs
            .iter()
            .map(|&z_val| {
                bessel_derivative_complex_scalar(
                    function,
                    *order,
                    z_val,
                    derivative_order,
                    mode,
                    kind,
                    rule,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => orders
            .iter()
            .map(|&order| {
                bessel_derivative_complex_scalar(
                    function,
                    order,
                    *z_val,
                    derivative_order,
                    mode,
                    kind,
                    rule,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(zs.iter())
                .map(|(&order, &z_val)| {
                    bessel_derivative_complex_scalar(
                        function,
                        order,
                        z_val,
                        derivative_order,
                        mode,
                        kind,
                        rule,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(_), _) | (SpecialTensor::ComplexVec(_), _) => {
            not_yet_implemented(function, mode, "complex-valued order not supported")
        }
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn bessel_derivative_real_scalar(
    function: &'static str,
    order: f64,
    x: f64,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: BesselKind,
    rule: DerivativeRule,
) -> Result<f64, SpecialError> {
    bessel_derivative_sum(
        function,
        order,
        x,
        derivative_order,
        mode,
        rule,
        |shifted_order, value| match kind {
            BesselKind::Jv => Ok(jv_scalar(shifted_order, value)),
            BesselKind::Yv => yv_scalar(shifted_order, value, mode),
            BesselKind::Iv => Ok(iv_scalar(shifted_order, value)),
            BesselKind::Kv => kv_scalar(shifted_order, value, mode),
        },
    )
}

fn bessel_derivative_complex_scalar(
    function: &'static str,
    order: f64,
    z: Complex64,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: BesselKind,
    rule: DerivativeRule,
) -> Result<Complex64, SpecialError> {
    if z.im == 0.0 && z.re >= 0.0 {
        return bessel_derivative_real_scalar(
            function,
            order,
            z.re,
            derivative_order,
            mode,
            kind,
            rule,
        )
        .map(|value| Complex64::new(value, 0.0));
    }

    bessel_derivative_sum_complex(
        function,
        order,
        z,
        derivative_order,
        mode,
        rule,
        |shifted_order, value| bessel_complex_scalar(function, shifted_order, value, mode, kind),
    )
}

fn hankel_derivative_dispatch(
    function: &'static str,
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: HankelKind,
) -> SpecialResult {
    match (v, z) {
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            hankel_derivative_real_scalar(function, *order, *x, derivative_order, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => orders
            .iter()
            .map(|&order| {
                hankel_derivative_real_scalar(function, order, *x, derivative_order, mode, kind)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => xs
            .iter()
            .map(|&x| {
                hankel_derivative_real_scalar(function, *order, x, derivative_order, mode, kind)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(xs.iter())
                .map(|(&order, &x)| {
                    hankel_derivative_real_scalar(function, order, x, derivative_order, mode, kind)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            hankel_derivative_complex_scalar(function, *order, *z_val, derivative_order, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => zs
            .iter()
            .map(|&z_val| {
                hankel_derivative_complex_scalar(
                    function,
                    *order,
                    z_val,
                    derivative_order,
                    mode,
                    kind,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => orders
            .iter()
            .map(|&order| {
                hankel_derivative_complex_scalar(
                    function,
                    order,
                    *z_val,
                    derivative_order,
                    mode,
                    kind,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(zs.iter())
                .map(|(&order, &z_val)| {
                    hankel_derivative_complex_scalar(
                        function,
                        order,
                        z_val,
                        derivative_order,
                        mode,
                        kind,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(_), _) | (SpecialTensor::ComplexVec(_), _) => {
            not_yet_implemented(function, mode, "complex-valued order not supported")
        }
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn hankel_derivative_real_scalar(
    function: &'static str,
    order: f64,
    x: f64,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: HankelKind,
) -> Result<Complex64, SpecialError> {
    let real = bessel_derivative_sum(
        function,
        order,
        x,
        derivative_order,
        mode,
        DerivativeRule::Alternating,
        |shifted_order, value| Ok(jv_scalar(shifted_order, value)),
    )?;
    let imag = bessel_derivative_sum(
        function,
        order,
        x,
        derivative_order,
        mode,
        DerivativeRule::Alternating,
        |shifted_order, value| yv_scalar(shifted_order, value, mode),
    )?;
    Ok(match kind {
        HankelKind::H1 => Complex64::new(real, imag),
        HankelKind::H2 => Complex64::new(real, -imag),
    })
}

fn hankel_derivative_complex_scalar(
    function: &'static str,
    order: f64,
    z: Complex64,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: HankelKind,
) -> Result<Complex64, SpecialError> {
    if z.im == 0.0 && z.re >= 0.0 {
        return hankel_derivative_real_scalar(function, order, z.re, derivative_order, mode, kind);
    }

    let j = bessel_derivative_sum_complex(
        function,
        order,
        z,
        derivative_order,
        mode,
        DerivativeRule::Alternating,
        |shifted_order, value| {
            bessel_complex_scalar(function, shifted_order, value, mode, BesselKind::Jv)
        },
    )?;
    let y = bessel_derivative_sum_complex(
        function,
        order,
        z,
        derivative_order,
        mode,
        DerivativeRule::Alternating,
        |shifted_order, value| {
            bessel_complex_scalar(function, shifted_order, value, mode, BesselKind::Yv)
        },
    )?;
    Ok(match kind {
        HankelKind::H1 => j + Complex64::new(-y.im, y.re),
        HankelKind::H2 => j + Complex64::new(y.im, -y.re),
    })
}

/// J_v(z) for real order v via power series.
fn jv_scalar(v: f64, z: f64) -> f64 {
    if z.is_nan() || v.is_nan() {
        return f64::NAN;
    }

    // Integer order: delegate to integer routines
    if v.fract() == 0.0 && v.abs() <= i32::MAX as f64 {
        let n = v as i32;
        return if n < 0 {
            let sign = if (-n) & 1 == 0 { 1.0 } else { -1.0 };
            sign * jn_nonnegative((-n) as u32, z)
        } else {
            jn_nonnegative(n as u32, z)
        };
    }

    if z == 0.0 {
        return if v > 0.0 { 0.0 } else { f64::INFINITY };
    }

    let az = z.abs();

    // Power series: J_v(z) = (z/2)^v Σ (-z²/4)^k / (k! Γ(v+k+1))
    if az < 20.0 + v.abs() {
        return jv_series(v, z);
    }

    // Asymptotic: J_v(z) ≈ sqrt(2/(πz)) cos(z - vπ/2 - π/4) for large z
    jv_asymptotic(v, az)
}

/// Power series for J_v(z).
fn jv_series(v: f64, z: f64) -> f64 {
    // For non-integer v and negative z, J_v(z) involves complex values
    if z < 0.0 && v.fract() != 0.0 {
        return f64::NAN;
    }

    let half_z_abs = (z / 2.0).abs();

    // First term: (|z|/2)^v / Γ(v+1), computed in log space
    let log_first = v * half_z_abs.ln() - lgamma(v + 1.0);
    let mut sum = 0.0;
    let mut log_term = log_first;

    for k in 0..200 {
        let term = log_term.exp();
        let term_sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        sum += term_sign * term;

        if term.abs() < 1e-16 * sum.abs().max(1e-300) && k > 5 {
            break;
        }

        // Recurrence: next log-magnitude adds log(z²/4) - log(k+1) - log(v+k+1)
        let kf = k as f64;
        log_term += (z * z / 4.0).ln() - (kf + 1.0).ln() - (v + kf + 1.0).ln();
    }

    sum
}

/// Asymptotic expansion for J_v(z) for large |z|.
fn jv_asymptotic(v: f64, z: f64) -> f64 {
    let phase = z - v * PI / 2.0 - PI / 4.0;
    (FRAC_2_PI / z).sqrt() * phase.cos()
}

/// Y_v(z) for real order v.
fn yv_scalar(v: f64, z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if v.is_nan() || z.is_nan() {
        return Ok(f64::NAN);
    }
    if z <= 0.0 {
        if z == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        return domain_error_by_mode(
            "yv",
            mode,
            format!("v={v},z={z}"),
            "yv requires z > 0 for real-valued output",
        );
    }

    // Integer order: delegate
    if v.fract() == 0.0 && v.abs() <= i32::MAX as f64 {
        let n = v as i32;
        return yn_scalar(n as f64, z, mode);
    }

    // Non-integer order: Y_v = (J_v cos(vπ) - J_{-v}) / sin(vπ)
    let sin_vpi = (v * PI).sin();
    if sin_vpi.abs() < 1e-15 {
        // Near integer order — should have been caught above
        return Ok(f64::NAN);
    }

    let jv_pos = jv_scalar(v, z);
    let jv_neg = jv_scalar(-v, z);
    Ok((jv_pos * (v * PI).cos() - jv_neg) / sin_vpi)
}

/// I_v(z) for real order v via power series.
fn iv_scalar(v: f64, z: f64) -> f64 {
    if z.is_nan() || v.is_nan() {
        return f64::NAN;
    }
    let az = z.abs();
    if az == 0.0 {
        return if v == 0.0 {
            1.0
        } else if v > 0.0 {
            0.0
        } else if v.fract() == 0.0 {
            // I_{-n}(0) = I_n(0)
            if v.abs() == 0.0 { 1.0 } else { 0.0 }
        } else {
            f64::INFINITY
        };
    }

    if v < 0.0 && v.fract() == 0.0 {
        return iv_scalar(v.abs(), z);
    }
    // I_{-v}(z) = I_v(z) + (2/pi) * sin(v*pi) * K_v(z)
    // But for real-only iv_scalar, we usually follow scipy.special.iv behavior.
    // scipy.special.iv(-v, z) for non-integer v and z > 0 returns same as formula.

    if az > 50.0 {
        return iv_asymptotic(v, az);
    }

    // Power series: I_v(z) = (z/2)^v Σ (z²/4)^k / (k! Γ(v+k+1))
    let half_z = az / 2.0;
    let quarter_z2 = az * az / 4.0;

    let log_first = v * half_z.ln() - lgamma(v + 1.0);
    let mut sum = 0.0;
    let mut log_term = log_first;

    for k in 0..200 {
        let term = log_term.exp();
        sum += term;

        if term < 1e-16 * sum && k > 10 {
            break;
        }

        let kf = k as f64;
        log_term += quarter_z2.ln() - (kf + 1.0).ln() - (v + kf + 1.0).ln();
    }

    // Parity for negative z: I_v(-z) = (-1)^v I_v(z) for integer v
    if z < 0.0 {
        if v.fract() == 0.0 {
            let n = v.abs() as i64;
            if n % 2 != 0 {
                return -sum;
            }
        } else {
            return f64::NAN; // Complex for non-integer v and negative z
        }
    }

    sum
}

fn iv_asymptotic(v: f64, z: f64) -> f64 {
    // I_v(z) ~ e^z / sqrt(2*pi*z) * [ 1 - (4v^2-1)/8z + (4v^2-1)(4v^2-9)/(2! (8z)^2) - ... ]
    let mu = 4.0 * v * v;
    let mut sum = 1.0;
    let mut term = 1.0;
    let iz8 = 1.0 / (8.0 * z);

    for k in 1..20 {
        let kf = k as f64;
        term *= -(mu - (2.0 * kf - 1.0).powi(2)) * iz8 / kf;
        sum += term;
        if term.abs() < 1e-14 * sum.abs() {
            break;
        }
    }

    (z * 2.0 * PI).sqrt().recip() * z.exp() * sum
}

/// K_v(z) for real order v.
fn kv_scalar(v: f64, z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if v.is_nan() || z.is_nan() {
        return Ok(f64::NAN);
    }
    if z <= 0.0 {
        if z == 0.0 {
            return Ok(f64::INFINITY);
        }
        return domain_error_by_mode("kv", mode, format!("v={v},z={z}"), "kv requires z > 0");
    }

    // For non-integer v: K_v = π/2 * (I_{-v} - I_v) / sin(vπ)
    let sin_vpi = (v * PI).sin();
    if sin_vpi.abs() > 1e-10 {
        // Non-integer order: direct formula
        let iv_neg = iv_scalar(-v, z);
        let iv_pos = iv_scalar(v, z);
        return Ok(PI / 2.0 * (iv_neg - iv_pos) / sin_vpi);
    }

    // Integer order: use K_0, K_1 from series, then recurrence K_{n+1} = K_{n-1} + 2n/z K_n
    let k0 = kv_integer_zero(z);
    let n = v.abs().round() as u32;
    if n == 0 {
        return Ok(k0);
    }
    let k1 = kv_integer_one(z);
    if n == 1 {
        return Ok(k1);
    }
    let mut k_prev = k0;
    let mut k_curr = k1;
    for i in 1..n {
        let k_next = k_prev + 2.0 * i as f64 / z * k_curr;
        k_prev = k_curr;
        k_curr = k_next;
    }
    Ok(k_curr)
}

fn kv_integer_zero(z: f64) -> f64 {
    kv_integral(0.0, z)
}

fn kv_integer_one(z: f64) -> f64 {
    kv_integral(1.0, z)
}

fn kv_integral(v: f64, z: f64) -> f64 {
    let upper = kv_integral_upper(z);
    adaptive_simpson(
        &|t| (-z * t.cosh()).exp() * (v * t).cosh(),
        0.0,
        upper,
        1.0e-12,
        16,
    )
}

fn kv_integral_upper(z: f64) -> f64 {
    let mut upper = 1.0_f64;
    while z * upper.cosh() < 40.0 && upper < 12.0 {
        upper += 1.0;
    }
    upper
}

fn adaptive_simpson(f: &impl Fn(f64) -> f64, a: f64, b: f64, tol: f64, depth: u32) -> f64 {
    let fa = f(a);
    let fb = f(b);
    let c = 0.5 * (a + b);
    let fc = f(c);
    let whole = simpson_estimate(a, b, fa, fb, fc);
    adaptive_simpson_inner(f, a, b, tol, whole, (fa, fb, fc), depth)
}

fn adaptive_simpson_inner(
    f: &impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    whole: f64,
    samples: (f64, f64, f64),
    depth: u32,
) -> f64 {
    let (fa, fb, fc) = samples;
    let c = 0.5 * (a + b);
    let left_mid = 0.5 * (a + c);
    let right_mid = 0.5 * (c + b);
    let f_left_mid = f(left_mid);
    let f_right_mid = f(right_mid);
    let left = simpson_estimate(a, c, fa, fc, f_left_mid);
    let right = simpson_estimate(c, b, fc, fb, f_right_mid);
    let delta = left + right - whole;

    if depth == 0 || delta.abs() <= 15.0 * tol {
        return left + right + delta / 15.0;
    }

    adaptive_simpson_inner(f, a, c, tol / 2.0, left, (fa, fc, f_left_mid), depth - 1)
        + adaptive_simpson_inner(f, c, b, tol / 2.0, right, (fc, fb, f_right_mid), depth - 1)
}

fn simpson_estimate(a: f64, b: f64, fa: f64, fb: f64, fm: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}

/// Log-gamma function for Bessel series computation.
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 && x.fract().abs() < 1e-14 {
        return f64::INFINITY;
    }
    // Stirling's approximation with Lanczos correction
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
        // Reflection formula: lgamma(x) = ln(π/|sin(πx)|) - lgamma(1-x)
        let sin_px = (PI * x).sin();
        if sin_px.abs() < 1e-300 {
            return f64::INFINITY; // pole
        }
        return (PI / sin_px.abs()).ln() - lgamma(1.0 - x);
    }

    let z = x - 1.0;
    let mut s = COEFFS[0];
    for (idx, coeff) in COEFFS.iter().enumerate().skip(1) {
        s += coeff / (z + idx as f64);
    }
    let t = z + G + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + s.ln()
}

// ── Spherical Bessel functions ──────────────────────────────────────

/// Spherical Bessel function of the first kind j_n(z).
///
/// j_n(z) = √(π/(2z)) J_{n+1/2}(z)
///
/// Computed via recurrence from:
///   j_0(z) = sin(z)/z,  j_1(z) = sin(z)/z² - cos(z)/z
pub fn spherical_jn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    spherical_bessel_dispatch("spherical_jn", n, z, mode, SphericalKind::Jn)
}

/// Spherical Bessel function of the second kind y_n(z).
///
/// y_n(z) = √(π/(2z)) Y_{n+1/2}(z)
///
/// Computed via recurrence from:
///   y_0(z) = -cos(z)/z,  y_1(z) = -cos(z)/z² - sin(z)/z
pub fn spherical_yn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    spherical_bessel_dispatch("spherical_yn", n, z, mode, SphericalKind::Yn)
}

/// Modified spherical Bessel function of the first kind i_n(z).
///
/// i_n(z) = √(π/(2z)) I_{n+1/2}(z)
///
/// Computed via recurrence from:
///   i_0(z) = sinh(z)/z,  i_1(z) = cosh(z)/z - sinh(z)/z²
pub fn spherical_in(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    spherical_bessel_dispatch("spherical_in", n, z, mode, SphericalKind::In)
}

/// Modified spherical Bessel function of the second kind k_n(z).
///
/// k_n(z) = √(π/(2z)) K_{n+1/2}(z)
///
/// Computed via recurrence from:
///   k_0(z) = π exp(-z)/(2z),  k_1(z) = π exp(-z)/(2z) * (1 + 1/z)
pub fn spherical_kn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    spherical_bessel_dispatch("spherical_kn", n, z, mode, SphericalKind::Kn)
}

#[derive(Clone, Copy)]
enum SphericalKind {
    Jn,
    Yn,
    In,
    Kn,
}

fn spherical_bessel_dispatch(
    function: &'static str,
    n: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
    kind: SphericalKind,
) -> SpecialResult {
    match (n, z) {
        // Real-real cases - use existing scalar implementations
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            let result = match kind {
                SphericalKind::Jn => spherical_jn_scalar(*order, *x, mode),
                SphericalKind::Yn => spherical_yn_scalar(*order, *x, mode),
                SphericalKind::In => spherical_in_scalar(*order, *x, mode),
                SphericalKind::Kn => spherical_kn_scalar(*order, *x, mode),
            };
            result.map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => orders
            .iter()
            .map(|&order| match kind {
                SphericalKind::Jn => spherical_jn_scalar(order, *x, mode),
                SphericalKind::Yn => spherical_yn_scalar(order, *x, mode),
                SphericalKind::In => spherical_in_scalar(order, *x, mode),
                SphericalKind::Kn => spherical_kn_scalar(order, *x, mode),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => xs
            .iter()
            .map(|&x| match kind {
                SphericalKind::Jn => spherical_jn_scalar(*order, x, mode),
                SphericalKind::Yn => spherical_yn_scalar(*order, x, mode),
                SphericalKind::In => spherical_in_scalar(*order, x, mode),
                SphericalKind::Kn => spherical_kn_scalar(*order, x, mode),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(xs.iter())
                .map(|(&order, &x)| match kind {
                    SphericalKind::Jn => spherical_jn_scalar(order, x, mode),
                    SphericalKind::Yn => spherical_yn_scalar(order, x, mode),
                    SphericalKind::In => spherical_in_scalar(order, x, mode),
                    SphericalKind::Kn => spherical_kn_scalar(order, x, mode),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        // Real n, complex z - use complex implementations
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            spherical_bessel_complex_scalar(function, *order, *z_val, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => zs
            .iter()
            .map(|&z_val| spherical_bessel_complex_scalar(function, *order, z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => orders
            .iter()
            .map(|&order| spherical_bessel_complex_scalar(function, order, *z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(zs.iter())
                .map(|(&order, &z_val)| {
                    spherical_bessel_complex_scalar(function, order, z_val, mode, kind)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        // Complex n - not supported
        (SpecialTensor::ComplexScalar(_), _) | (SpecialTensor::ComplexVec(_), _) => {
            not_yet_implemented(function, mode, "complex-valued order not supported")
        }
        // Empty tensor
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn spherical_bessel_complex_scalar(
    function: &'static str,
    order: f64,
    z: Complex64,
    mode: RuntimeMode,
    kind: SphericalKind,
) -> Result<Complex64, SpecialError> {
    if order.is_nan() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    let n = match integer_order_or_nan(function, order, mode)? {
        Some(v) => v,
        None => return Ok(Complex64::new(f64::NAN, f64::NAN)),
    };

    if n < 0 {
        return complex_domain_error_by_mode(
            function,
            mode,
            format!("n={n}"),
            "spherical Bessel order must be non-negative",
        );
    }

    Ok(match kind {
        SphericalKind::Jn => complex_spherical_jn(n as u32, z),
        SphericalKind::Yn => complex_spherical_yn(n as u32, z),
        SphericalKind::In => complex_spherical_in(n as u32, z),
        SphericalKind::Kn => complex_spherical_kn(n as u32, z),
    })
}

fn spherical_jn_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    let n = match integer_order_or_nan("spherical_jn", order, mode)? {
        Some(v) => v,
        None => return Ok(f64::NAN),
    };
    if n < 0 {
        return domain_error_by_mode(
            "spherical_jn",
            mode,
            format!("n={n}"),
            "spherical Bessel order must be non-negative",
        );
    }
    Ok(spherical_jn_nonneg(n as u32, x))
}

fn spherical_yn_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if x <= 0.0 {
        if x == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        return domain_error_by_mode(
            "spherical_yn",
            mode,
            format!("x={x}"),
            "spherical_yn requires x > 0",
        );
    }
    let n = match integer_order_or_nan("spherical_yn", order, mode)? {
        Some(v) => v,
        None => return Ok(f64::NAN),
    };
    if n < 0 {
        return domain_error_by_mode(
            "spherical_yn",
            mode,
            format!("n={n}"),
            "spherical Bessel order must be non-negative",
        );
    }
    Ok(spherical_yn_nonneg(n as u32, x))
}

fn spherical_in_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    let n = match integer_order_or_nan("spherical_in", order, mode)? {
        Some(v) => v,
        None => return Ok(f64::NAN),
    };
    if n < 0 {
        return domain_error_by_mode(
            "spherical_in",
            mode,
            format!("n={n}"),
            "spherical Bessel order must be non-negative",
        );
    }
    Ok(spherical_in_nonneg(n as u32, x))
}

fn spherical_kn_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if x <= 0.0 {
        if x == 0.0 {
            return Ok(f64::INFINITY);
        }
        return domain_error_by_mode(
            "spherical_kn",
            mode,
            format!("x={x}"),
            "spherical_kn requires x > 0",
        );
    }
    let n = match integer_order_or_nan("spherical_kn", order, mode)? {
        Some(v) => v,
        None => return Ok(f64::NAN),
    };
    if n < 0 {
        return domain_error_by_mode(
            "spherical_kn",
            mode,
            format!("n={n}"),
            "spherical Bessel order must be non-negative",
        );
    }
    Ok(spherical_kn_nonneg(n as u32, x))
}

/// j_0(z) = sin(z)/z, j_1(z) = sin(z)/z² - cos(z)/z
/// Forward recurrence: j_{k+1}(z) = (2k+1)/z * j_k(z) - j_{k-1}(z)
fn spherical_jn_nonneg(n: u32, x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    let ax = x.abs();
    if ax < 1e-15 {
        // Small-x Taylor: j_n(x) ≈ x^n / (2n+1)!!
        return if n == 0 { 1.0 } else { 0.0 };
    }

    let mut j_prev = x.sin() / x; // j_0
    if n == 0 {
        return j_prev;
    }
    let mut j_curr = x.sin() / (x * x) - x.cos() / x; // j_1
    if n == 1 {
        return j_curr;
    }

    for k in 1..n {
        let next = (2.0 * k as f64 + 1.0) / x * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = next;
    }
    j_curr
}

/// y_0(z) = -cos(z)/z, y_1(z) = -cos(z)/z² - sin(z)/z
/// Forward recurrence: y_{k+1}(z) = (2k+1)/z * y_k(z) - y_{k-1}(z)
fn spherical_yn_nonneg(n: u32, x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    let mut y_prev = -x.cos() / x; // y_0
    if n == 0 {
        return y_prev;
    }
    let mut y_curr = -x.cos() / (x * x) - x.sin() / x; // y_1
    if n == 1 {
        return y_curr;
    }

    for k in 1..n {
        let next = (2.0 * k as f64 + 1.0) / x * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = next;
    }
    y_curr
}

/// i_0(z) = sinh(z)/z, i_1(z) = cosh(z)/z - sinh(z)/z²
/// Recurrence: i_{k+1}(z) = i_{k-1}(z) - (2k+1)/z * i_k(z)
fn spherical_in_nonneg(n: u32, x: f64) -> f64 {
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    let ax = x.abs();
    let sh = ax.sinh();
    let ch = ax.cosh();

    let mut i_prev = sh / ax; // i_0
    if n == 0 {
        return i_prev;
    }
    let mut i_curr = ch / ax - sh / (ax * ax); // i_1
    if n == 1 {
        // For negative x, i_n(-x) = (-1)^n * i_n(x)
        return if x < 0.0 { -i_curr } else { i_curr };
    }

    for k in 1..n {
        let next = i_prev - (2.0 * k as f64 + 1.0) / ax * i_curr;
        i_prev = i_curr;
        i_curr = next;
    }
    // Parity: i_n(-x) = (-1)^n i_n(x)
    if x < 0.0 && n % 2 == 1 {
        -i_curr
    } else {
        i_curr
    }
}

/// k_0(z) = π exp(-z)/(2z), k_1(z) = π exp(-z)/(2z) * (1 + 1/z)
/// Recurrence: k_{k+1}(z) = k_{k-1}(z) + (2k+1)/z * k_k(z)
fn spherical_kn_nonneg(n: u32, x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    let emx = (-x).exp();
    let scale = PI / 2.0;
    let mut k_prev = scale * emx / x; // k_0
    if n == 0 {
        return k_prev;
    }
    let mut k_curr = scale * emx / x * (1.0 + 1.0 / x); // k_1
    if n == 1 {
        return k_curr;
    }

    for k in 1..n {
        let next = k_prev + (2.0 * k as f64 + 1.0) / x * k_curr;
        k_prev = k_curr;
        k_curr = next;
    }
    k_curr
}

// ============================================================================
// Complex spherical Bessel functions
// ============================================================================

/// Complex spherical Bessel function of the first kind j_n(z).
fn complex_spherical_jn(n: u32, z: Complex64) -> Complex64 {
    if z.re == 0.0 && z.im == 0.0 {
        return if n == 0 {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
    }

    let sin_z = z.sin();
    let cos_z = z.cos();
    let z_inv = z.recip();

    let mut j_prev = sin_z * z_inv; // j_0
    if n == 0 {
        return j_prev;
    }

    let z_inv2 = z_inv * z_inv;
    let mut j_curr = sin_z * z_inv2 - cos_z * z_inv; // j_1
    if n == 1 {
        return j_curr;
    }

    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64 + 1.0, 0.0) * z_inv;
        let next = coeff * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = next;
    }
    j_curr
}

/// Complex spherical Bessel function of the second kind y_n(z).
fn complex_spherical_yn(n: u32, z: Complex64) -> Complex64 {
    if z.re == 0.0 && z.im == 0.0 {
        return Complex64::new(f64::NEG_INFINITY, 0.0);
    }

    let sin_z = z.sin();
    let cos_z = z.cos();
    let z_inv = z.recip();

    let mut y_prev = Complex64::new(-1.0, 0.0) * cos_z * z_inv; // y_0
    if n == 0 {
        return y_prev;
    }

    let z_inv2 = z_inv * z_inv;
    let mut y_curr = Complex64::new(-1.0, 0.0) * cos_z * z_inv2 - sin_z * z_inv; // y_1
    if n == 1 {
        return y_curr;
    }

    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64 + 1.0, 0.0) * z_inv;
        let next = coeff * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = next;
    }
    y_curr
}

/// Complex modified spherical Bessel function of the first kind i_n(z).
fn complex_spherical_in(n: u32, z: Complex64) -> Complex64 {
    if z.re == 0.0 && z.im == 0.0 {
        return if n == 0 {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
    }

    let sinh_z = z.sinh();
    let cosh_z = z.cosh();
    let z_inv = z.recip();

    let mut i_prev = sinh_z * z_inv; // i_0
    if n == 0 {
        return i_prev;
    }

    let z_inv2 = z_inv * z_inv;
    let mut i_curr = cosh_z * z_inv - sinh_z * z_inv2; // i_1
    if n == 1 {
        return i_curr;
    }

    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64 + 1.0, 0.0) * z_inv;
        let next = i_prev - coeff * i_curr;
        i_prev = i_curr;
        i_curr = next;
    }
    i_curr
}

/// Complex modified spherical Bessel function of the second kind k_n(z).
fn complex_spherical_kn(n: u32, z: Complex64) -> Complex64 {
    if z.re == 0.0 && z.im == 0.0 {
        return Complex64::new(f64::INFINITY, 0.0);
    }

    let neg_z = Complex64::new(-z.re, -z.im);
    let emz = neg_z.exp();
    let scale = Complex64::new(std::f64::consts::PI / 2.0, 0.0);
    let z_inv = z.recip();

    let mut k_prev = scale * emz * z_inv; // k_0
    if n == 0 {
        return k_prev;
    }

    let one_plus_zinv = Complex64::new(1.0, 0.0) + z_inv;
    let mut k_curr = scale * emz * z_inv * one_plus_zinv; // k_1
    if n == 1 {
        return k_curr;
    }

    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64 + 1.0, 0.0) * z_inv;
        let next = k_prev + coeff * k_curr;
        k_prev = k_curr;
        k_curr = next;
    }
    k_curr
}

fn wright_bessel_scalar(a: f64, b: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    const LN_MAX: f64 = 709.782_712_893_384;
    const LN_MIN: f64 = -745.133_219_101_941_1;

    if a.is_nan() || b.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if !a.is_finite() || !b.is_finite() || !x.is_finite() || a < 0.0 || b < 0.0 || x < 0.0 {
        return domain_error_by_mode(
            "wright_bessel",
            mode,
            format!("a={a},b={b},x={x}"),
            "wright_bessel requires finite a>=0, b>=0, x>=0",
        );
    }
    if x == 0.0 {
        return rgamma_nonnegative(b);
    }
    if a == 0.0 {
        let log_value = x + rgamma_log_nonnegative(b)?;
        return Ok(exp_from_log(log_value, LN_MIN, LN_MAX));
    }

    let log_value = log_wright_bessel_series(a, b, x)?;
    Ok(exp_from_log(log_value, LN_MIN, LN_MAX))
}

fn log_wright_bessel_series(a: f64, b: f64, x: f64) -> Result<f64, SpecialError> {
    const RELATIVE_LOG_EPS: f64 = -40.0;
    const MIN_DECREASING_TERMS: usize = 8;

    let ln_x = x.ln();
    let mut max_log = f64::NEG_INFINITY;
    let mut scaled_sum = 0.0;
    let mut prev_log = f64::NEG_INFINITY;
    let mut decreasing_terms = 0usize;
    let max_terms = if a >= 10.0 {
        256usize
    } else if a >= 1.0 {
        1024usize
    } else if a >= 0.1 {
        4096usize
    } else {
        16_384usize
    };

    for k in 0..max_terms {
        let kf = k as f64;
        let akb = a.mul_add(kf, b);
        let log_rgamma = rgamma_log_nonnegative(akb)?;
        let log_term =
            kf * ln_x - crate::gamma::gammaln_scalar(kf + 1.0, RuntimeMode::Strict)? + log_rgamma;

        if !log_term.is_finite() {
            prev_log = log_term;
            continue;
        }

        if log_term > max_log {
            scaled_sum = if max_log.is_finite() {
                scaled_sum * (max_log - log_term).exp() + 1.0
            } else {
                1.0
            };
            max_log = log_term;
        } else {
            scaled_sum += (log_term - max_log).exp();
        }

        if prev_log.is_finite() && log_term < prev_log {
            decreasing_terms += 1;
        } else {
            decreasing_terms = 0;
        }
        prev_log = log_term;

        let current_log_sum = max_log + scaled_sum.ln();
        if decreasing_terms >= MIN_DECREASING_TERMS
            && log_term - current_log_sum <= RELATIVE_LOG_EPS
        {
            return Ok(current_log_sum);
        }
    }

    if max_log.is_finite() {
        Ok(max_log + scaled_sum.ln())
    } else {
        Ok(f64::NEG_INFINITY)
    }
}

fn rgamma_nonnegative(x: f64) -> Result<f64, SpecialError> {
    Ok(exp_from_log(
        rgamma_log_nonnegative(x)?,
        f64::NEG_INFINITY,
        709.782_712_893_384,
    ))
}

fn rgamma_log_nonnegative(x: f64) -> Result<f64, SpecialError> {
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(-crate::gamma::gammaln_scalar(x, RuntimeMode::Strict)?)
}

fn exp_from_log(log_value: f64, ln_min: f64, ln_max: f64) -> f64 {
    if log_value.is_nan() {
        f64::NAN
    } else if log_value == f64::NEG_INFINITY {
        0.0
    } else if log_value > ln_max {
        f64::INFINITY
    } else if log_value < ln_min {
        0.0
    } else {
        log_value.exp()
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

fn jn_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    let n = match integer_order_or_nan("jn", order, mode)? {
        Some(value) => value,
        None => return Ok(f64::NAN),
    };

    if n < 0 {
        let sign = if n & 1 == 0 { 1.0 } else { -1.0 };
        return Ok(sign * jn_nonnegative((-n) as u32, x));
    }
    Ok(jn_nonnegative(n as u32, x))
}

fn yn_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if x < 0.0 {
        return domain_error_by_mode(
            "yn",
            mode,
            format!("order={order},x={x}"),
            "yn real-valued wrapper requires x >= 0",
        );
    }
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    let n = match integer_order_or_nan("yn", order, mode)? {
        Some(value) => value,
        None => return Ok(f64::NAN),
    };
    if n < 0 {
        let sign = if n & 1 == 0 { 1.0 } else { -1.0 };
        return Ok(sign * yn_nonnegative((-n) as u32, x));
    }
    Ok(yn_nonnegative(n as u32, x))
}

fn integer_order_or_nan(
    function: &'static str,
    order: f64,
    mode: RuntimeMode,
) -> Result<Option<i32>, SpecialError> {
    if order.is_finite()
        && order.fract() == 0.0
        && order >= i32::MIN as f64
        && order <= i32::MAX as f64
    {
        return Ok(Some(order as i32));
    }
    match mode {
        RuntimeMode::Strict => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                format!("order={order}"),
                "returned_nan",
                "strict non-integral-order fallback",
                false,
            );
            Ok(None)
        }
        RuntimeMode::Hardened => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                format!("order={order}"),
                "fail_closed",
                "Bessel order must be a finite integer",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "Bessel order must be a finite integer",
            })
        }
    }
}

fn y0_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x < 0.0 {
        return domain_error_by_mode(
            "y0",
            mode,
            format!("x={x}"),
            "y0 real-valued wrapper requires x >= 0",
        );
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    Ok(y0_core_positive(x))
}

fn y1_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x < 0.0 {
        return domain_error_by_mode(
            "y1",
            mode,
            format!("x={x}"),
            "y1 real-valued wrapper requires x >= 0",
        );
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    Ok(y1_core_positive(x))
}

fn domain_error_by_mode(
    function: &'static str,
    mode: RuntimeMode,
    input_summary: String,
    detail: &'static str,
) -> Result<f64, SpecialError> {
    match mode {
        RuntimeMode::Strict => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                input_summary,
                "returned_nan",
                "strict domain fallback",
                false,
            );
            Ok(f64::NAN)
        }
        RuntimeMode::Hardened => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                input_summary,
                "fail_closed",
                detail,
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail,
            })
        }
    }
}

fn complex_domain_error_by_mode(
    function: &'static str,
    mode: RuntimeMode,
    input_summary: String,
    detail: &'static str,
) -> Result<Complex64, SpecialError> {
    match mode {
        RuntimeMode::Strict => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                input_summary,
                "returned_nan",
                "strict domain fallback",
                false,
            );
            Ok(Complex64::new(f64::NAN, f64::NAN))
        }
        RuntimeMode::Hardened => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                input_summary,
                "fail_closed",
                detail,
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail,
            })
        }
    }
}

fn jn_nonnegative(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }
    if n == 0 {
        return j0_core(x);
    }
    if n == 1 {
        return j1_core(x);
    }
    if x == 0.0 {
        return 0.0;
    }

    let mut j_prev = j0_core(x);
    let mut j_curr = j1_core(x);
    for k in 1..n {
        let next = (2.0 * k as f64 / x) * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = next;
    }
    j_curr
}

fn yn_nonnegative(n: u32, x: f64) -> f64 {
    if n == 0 {
        return y0_core_positive(x);
    }
    if n == 1 {
        return y1_core_positive(x);
    }

    let mut y_prev = y0_core_positive(x);
    let mut y_curr = y1_core_positive(x);
    for k in 1..n {
        let next = (2.0 * k as f64 / x) * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = next;
    }
    y_curr
}

fn j0_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }

    let ax = x.abs();
    if ax == 0.0 {
        // J_0(0) = 1 exactly. The rational approximation below evaluates
        // to 57568490574/57568490411 ≈ 1.000000003 at x=0, so this
        // special case keeps the analytic value at the origin.
        return 1.0;
    }
    if ax < 8.0 {
        let y = ax * ax;
        let numer = 57_568_490_574.0
            + y * (-13_362_590_354.0
                + y * (651_619_640.7
                    + y * (-11_214_424.18 + y * (77_392.330_17 + y * (-184.905_245_6)))));
        let denom = 57_568_490_411.0
            + y * (1_029_532_985.0
                + y * (9_494_680.718 + y * (59_272.648_53 + y * (267.853_271_2 + y))));
        numer / denom
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - PI / 4.0;
        let p = 1.0
            + y * (-0.001_098_628_627
                + y * (0.000_027_345_104_07
                    + y * (-0.000_002_073_370_639 + y * 0.000_000_209_388_721_1)));
        let q = -0.015_624_999_95
            + y * (0.000_143_048_876_5
                + y * (-0.000_006_911_147_651
                    + y * (0.000_000_762_109_516_1 + y * (-0.000_000_093_494_515_2))));
        (FRAC_2_PI / ax).sqrt() * (xx.cos() * p - z * xx.sin() * q)
    }
}

fn j1_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }

    let ax = x.abs();
    let ans = if ax < 8.0 {
        let y = ax * ax;
        let numer = ax
            * (72_362_614_232.0
                + y * (-7_895_059_235.0
                    + y * (242_396_853.1
                        + y * (-2_972_611.439 + y * (15_704.482_60 + y * (-30.160_366_06))))));
        let denom = 144_725_228_442.0
            + y * (2_300_535_178.0
                + y * (18_583_304.74 + y * (99_447.433_94 + y * (376.999_139_7 + y))));
        numer / denom
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 3.0 * PI / 4.0;
        let p = 1.0
            + y * (0.001_831_05
                + y * (-0.000_035_163_964_96
                    + y * (0.000_002_457_520_174 + y * (-0.000_000_240_337_019))));
        let q = 0.046_874_999_95
            + y * (-0.000_200_269_087_3
                + y * (0.000_008_449_199_096
                    + y * (-0.000_000_882_289_87 + y * 0.000_000_105_787_412)));
        (FRAC_2_PI / ax).sqrt() * (xx.cos() * p - z * xx.sin() * q)
    };

    if x < 0.0 { -ans } else { ans }
}

fn y0_core_positive(x: f64) -> f64 {
    if x < 8.0 {
        let y = x * x;
        let numer = -2_957_821_389.0
            + y * (7_062_834_065.0
                + y * (-512_359_803.6
                    + y * (10_879_881.29 + y * (-86_327.927_57 + y * 228.462_273_3))));
        let denom = 40_076_544_269.0
            + y * (745_249_964.8
                + y * (7_189_466.438 + y * (47_447.264_70 + y * (226.103_024_4 + y))));
        numer / denom + FRAC_2_PI * j0_core(x) * x.ln()
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - PI / 4.0;
        let p = 1.0
            + y * (-0.001_098_628_627
                + y * (0.000_027_345_104_07
                    + y * (-0.000_002_073_370_639 + y * 0.000_000_209_388_721_1)));
        let q = -0.015_624_999_95
            + y * (0.000_143_048_876_5
                + y * (-0.000_006_911_147_651
                    + y * (0.000_000_762_109_516_1 + y * (-0.000_000_093_494_515_2))));
        (FRAC_2_PI / x).sqrt() * (xx.sin() * p + z * xx.cos() * q)
    }
}

fn y1_core_positive(x: f64) -> f64 {
    if x < 8.0 {
        let y = x * x;
        let numer = x
            * (-4.900_604_943e12
                + y * (1.275_274_39e12
                    + y * (-5.153_438_139e10
                        + y * (7.349_264_551e8 + y * (-4.237_922_726e6 + y * 8_511.937_935)))));
        let denom = 2.499_580_57e13
            + y * (4.244_419_664e11
                + y * (3.733_650_367e9
                    + y * (2.245_904_002e7 + y * (1.020_426_05e5 + y * (354.963_288_5 + y)))));
        numer / denom + FRAC_2_PI * (j1_core(x) * x.ln() - 1.0 / x)
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 3.0 * PI / 4.0;
        let p = 1.0
            + y * (0.001_831_05
                + y * (-0.000_035_163_964_96
                    + y * (0.000_002_457_520_174 + y * (-0.000_000_240_337_019))));
        let q = 0.046_874_999_95
            + y * (-0.000_200_269_087_3
                + y * (0.000_008_449_199_096
                    + y * (-0.000_000_882_289_87 + y * 0.000_000_105_787_412)));
        (FRAC_2_PI / x).sqrt() * (xx.sin() * p + z * xx.cos() * q)
    }
}

// ============================================================================
// Complex cylindrical Bessel functions J_v, Y_v, I_v, K_v
// ============================================================================

/// Complex J_v(z) via power series.
/// J_v(z) = (z/2)^v Σ_{k=0}^∞ (-z²/4)^k / (k! Γ(v+k+1))
fn negative_integer_order(v: f64) -> Option<u32> {
    if !v.is_finite() || v >= 0.0 || v.fract() != 0.0 || v.abs() > u32::MAX as f64 {
        return None;
    }

    Some((-v) as u32)
}

fn complex_jv_scalar(v: f64, z: Complex64) -> Complex64 {
    if !z.is_finite() || v.is_nan() {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    if let Some(n) = negative_integer_order(v) {
        let value = complex_jv_scalar(-v, z);
        return if n % 2 == 0 { value } else { -value };
    }

    if z.re == 0.0 && z.im == 0.0 {
        return if v == 0.0 {
            Complex64::new(1.0, 0.0)
        } else if v > 0.0 {
            Complex64::new(0.0, 0.0)
        } else {
            Complex64::new(f64::INFINITY, 0.0)
        };
    }

    let half_z = z / 2.0;
    let neg_quarter_z2 = Complex64::new(-1.0, 0.0) * half_z * half_z;

    // First term: (z/2)^v / Γ(v+1)
    let log_half_z = half_z.ln();
    let v_c = Complex64::new(v, 0.0);
    let log_first = v_c * log_half_z - crate::gamma::complex_gammaln(Complex64::new(v + 1.0, 0.0));
    let mut sum = log_first.exp();
    let mut term = sum;

    for k in 1..200 {
        let kf = k as f64;
        // term *= (-z²/4) / (k * (v + k))
        term = term * neg_quarter_z2 / Complex64::new(kf * (v + kf), 0.0);
        sum = sum + term;

        if term.abs() < 1e-15 * sum.abs() && k > 5 {
            break;
        }
    }

    sum
}

/// Complex I_v(z) via power series.
/// I_v(z) = (z/2)^v Σ_{k=0}^∞ (z²/4)^k / (k! Γ(v+k+1))
fn complex_iv_scalar(v: f64, z: Complex64) -> Complex64 {
    if !z.is_finite() || v.is_nan() {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    if negative_integer_order(v).is_some() {
        return complex_iv_scalar(-v, z);
    }

    if z.re == 0.0 && z.im == 0.0 {
        return if v == 0.0 {
            Complex64::new(1.0, 0.0)
        } else if v > 0.0 {
            Complex64::new(0.0, 0.0)
        } else {
            Complex64::new(f64::INFINITY, 0.0)
        };
    }

    let half_z = z / 2.0;
    let quarter_z2 = half_z * half_z;

    // First term: (z/2)^v / Γ(v+1)
    let log_half_z = half_z.ln();
    let v_c = Complex64::new(v, 0.0);
    let log_first = v_c * log_half_z - crate::gamma::complex_gammaln(Complex64::new(v + 1.0, 0.0));
    let mut sum = log_first.exp();
    let mut term = sum;

    for k in 1..200 {
        let kf = k as f64;
        // term *= (z²/4) / (k * (v + k))
        term = term * quarter_z2 / Complex64::new(kf * (v + kf), 0.0);
        sum = sum + term;

        if term.abs() < 1e-15 * sum.abs() && k > 5 {
            break;
        }
    }

    sum
}

/// Complex Y_v(z) for real order v.
/// Y_v = (J_v cos(vπ) - J_{-v}) / sin(vπ) for non-integer v.
fn complex_yv_scalar(v: f64, z: Complex64, _mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if v.is_nan() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    if z.re == 0.0 && z.im == 0.0 {
        return Ok(Complex64::new(f64::NEG_INFINITY, 0.0));
    }

    // Integer order: use recurrence
    if v.fract() == 0.0 && v.abs() <= i32::MAX as f64 {
        let n = v.abs() as u32;
        let result = complex_yn_integer(n, z);
        return if v < 0.0 && n % 2 == 1 {
            Ok(Complex64::new(-result.re, -result.im))
        } else {
            Ok(result)
        };
    }

    // Non-integer order: Y_v = (J_v cos(vπ) - J_{-v}) / sin(vπ)
    let sin_vpi = (v * PI).sin();
    if sin_vpi.abs() < 1e-15 {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    let jv_pos = complex_jv_scalar(v, z);
    let jv_neg = complex_jv_scalar(-v, z);
    let cos_vpi = Complex64::new((v * PI).cos(), 0.0);
    let sin_vpi_c = Complex64::new(sin_vpi, 0.0);

    Ok((jv_pos * cos_vpi - jv_neg) / sin_vpi_c)
}

/// Complex Y_n(z) for integer order via recurrence.
fn complex_yn_integer(n: u32, z: Complex64) -> Complex64 {
    // Y_0 and Y_1 via formula, then recurrence
    let y0 = complex_y0_series(z);
    if n == 0 {
        return y0;
    }
    let y1 = complex_y1_series(z);
    if n == 1 {
        return y1;
    }

    let z_inv = z.recip();
    let mut y_prev = y0;
    let mut y_curr = y1;
    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64, 0.0) * z_inv;
        let next = coeff * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = next;
    }
    y_curr
}

/// Complex Y_0(z) via series with logarithmic term.
fn complex_y0_series(z: Complex64) -> Complex64 {
    // Y_0(z) = (2/π)[J_0(z)(ln(z/2) + γ) + series...]
    // Simplified: use limiting form for small |z| and asymptotic for large
    let j0 = complex_jv_scalar(0.0, z);
    let half_z = z / 2.0;
    let ln_half_z = half_z.ln();
    let euler_gamma = 0.577_215_664_901_532_9_f64;

    // Leading term: (2/π) * J_0(z) * (ln(z/2) + γ)
    let frac_2_pi = Complex64::new(FRAC_2_PI, 0.0);
    let gamma_c = Complex64::new(euler_gamma, 0.0);

    frac_2_pi * j0 * (ln_half_z + gamma_c)
}

/// Complex Y_1(z) via series with logarithmic term.
fn complex_y1_series(z: Complex64) -> Complex64 {
    let j1 = complex_jv_scalar(1.0, z);
    let half_z = z / 2.0;
    let ln_half_z = half_z.ln();
    let euler_gamma = 0.577_215_664_901_532_9_f64;

    let frac_2_pi = Complex64::new(FRAC_2_PI, 0.0);
    let gamma_c = Complex64::new(euler_gamma, 0.0);
    let z_inv = z.recip();

    // Y_1(z) ≈ (2/π) * J_1(z) * (ln(z/2) + γ) - 2/(πz)
    frac_2_pi * j1 * (ln_half_z + gamma_c) - frac_2_pi * z_inv
}

/// Complex K_v(z) for real order v.
/// K_v = π/2 * (I_{-v} - I_v) / sin(vπ) for non-integer v.
fn complex_kv_scalar(v: f64, z: Complex64, _mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if v.is_nan() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    if z.re == 0.0 && z.im == 0.0 {
        return Ok(Complex64::new(f64::INFINITY, 0.0));
    }

    // For non-integer v: K_v = π/2 * (I_{-v} - I_v) / sin(vπ)
    let sin_vpi = (v * PI).sin();
    if sin_vpi.abs() > 1e-10 {
        let iv_neg = complex_iv_scalar(-v, z);
        let iv_pos = complex_iv_scalar(v, z);
        let pi_half = Complex64::new(PI / 2.0, 0.0);
        let sin_vpi_c = Complex64::new(sin_vpi, 0.0);
        return Ok(pi_half * (iv_neg - iv_pos) / sin_vpi_c);
    }

    // Integer order: use recurrence
    let n = v.abs().round() as u32;
    Ok(complex_kn_integer(n, z))
}

/// Complex K_n(z) for integer order via recurrence.
fn complex_kn_integer(n: u32, z: Complex64) -> Complex64 {
    let neg_z = Complex64::new(-z.re, -z.im);
    let emz = neg_z.exp();
    let z_inv = z.recip();
    let scale = Complex64::new(PI / 2.0, 0.0);

    // K_0(z) ≈ -ln(z/2) - γ + O(z²) for small z; π*exp(-z)/(2z) asymptotic
    // Use asymptotic form for simplicity
    let mut k_prev = scale * emz * z_inv; // K_0 asymptotic
    if n == 0 {
        return k_prev;
    }

    let one = Complex64::new(1.0, 0.0);
    let mut k_curr = scale * emz * z_inv * (one + z_inv); // K_1 asymptotic
    if n == 1 {
        return k_curr;
    }

    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64, 0.0) * z_inv;
        let next = k_prev + coeff * k_curr;
        k_prev = k_curr;
        k_curr = next;
    }
    k_curr
}

fn hankel_dispatch(
    function: &'static str,
    v: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
    kind: HankelKind,
) -> SpecialResult {
    match (v, z) {
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            hankel_real_scalar(function, *order, *x, mode, kind).map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => orders
            .iter()
            .map(|&order| hankel_real_scalar(function, order, *x, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => xs
            .iter()
            .map(|&x| hankel_real_scalar(function, *order, x, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(xs.iter())
                .map(|(&order, &x)| hankel_real_scalar(function, order, x, mode, kind))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            hankel_complex_scalar(function, *order, *z_val, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => zs
            .iter()
            .map(|&z_val| hankel_complex_scalar(function, *order, z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => orders
            .iter()
            .map(|&order| hankel_complex_scalar(function, order, *z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(zs.iter())
                .map(|(&order, &z_val)| hankel_complex_scalar(function, order, z_val, mode, kind))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(_), _) | (SpecialTensor::ComplexVec(_), _) => {
            not_yet_implemented(function, mode, "complex-valued order not supported")
        }
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn hankel_real_scalar(
    function: &'static str,
    order: f64,
    x: f64,
    mode: RuntimeMode,
    kind: HankelKind,
) -> Result<Complex64, SpecialError> {
    let j = jv_scalar(order, x);
    let y = yv_scalar(order, x, mode)?;
    let imag = match kind {
        HankelKind::H1 => y,
        HankelKind::H2 => -y,
    };
    if !j.is_finite() || !imag.is_finite() {
        record_special_trace(
            function,
            mode,
            "non_finite_output",
            format!("order={order},x={x}"),
            "returned_non_finite",
            format!("real={j},imag={imag}"),
            false,
        );
    }
    Ok(Complex64::new(j, imag))
}

fn hankel_complex_scalar(
    function: &'static str,
    order: f64,
    z: Complex64,
    mode: RuntimeMode,
    kind: HankelKind,
) -> Result<Complex64, SpecialError> {
    let j = bessel_complex_scalar(function, order, z, mode, BesselKind::Jv)?;
    let y = bessel_complex_scalar(function, order, z, mode, BesselKind::Yv)?;
    Ok(match kind {
        HankelKind::H1 => j + Complex64::new(-y.im, y.re),
        HankelKind::H2 => j + Complex64::new(y.im, -y.re),
    })
}

#[derive(Clone, Copy)]
enum BesselKind {
    Jv,
    Yv,
    Iv,
    Kv,
}

fn bessel_dispatch(
    function: &'static str,
    v: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
    kind: BesselKind,
) -> SpecialResult {
    match (v, z) {
        // Real-real: delegate to existing real scalars
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            let result = match kind {
                BesselKind::Jv => Ok(jv_scalar(*order, *x)),
                BesselKind::Yv => yv_scalar(*order, *x, mode),
                BesselKind::Iv => Ok(iv_scalar(*order, *x)),
                BesselKind::Kv => kv_scalar(*order, *x, mode),
            };
            result.map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => orders
            .iter()
            .map(|&order| match kind {
                BesselKind::Jv => Ok(jv_scalar(order, *x)),
                BesselKind::Yv => yv_scalar(order, *x, mode),
                BesselKind::Iv => Ok(iv_scalar(order, *x)),
                BesselKind::Kv => kv_scalar(order, *x, mode),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => xs
            .iter()
            .map(|&x| match kind {
                BesselKind::Jv => Ok(jv_scalar(*order, x)),
                BesselKind::Yv => yv_scalar(*order, x, mode),
                BesselKind::Iv => Ok(iv_scalar(*order, x)),
                BesselKind::Kv => kv_scalar(*order, x, mode),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::ShapeMismatch,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(xs.iter())
                .map(|(&order, &x)| match kind {
                    BesselKind::Jv => Ok(jv_scalar(order, x)),
                    BesselKind::Yv => yv_scalar(order, x, mode),
                    BesselKind::Iv => Ok(iv_scalar(order, x)),
                    BesselKind::Kv => kv_scalar(order, x, mode),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        // Real order, complex z
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            bessel_complex_scalar(function, *order, *z_val, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => zs
            .iter()
            .map(|&z_val| bessel_complex_scalar(function, *order, z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => orders
            .iter()
            .map(|&order| bessel_complex_scalar(function, order, *z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::ShapeMismatch,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(zs.iter())
                .map(|(&order, &z_val)| bessel_complex_scalar(function, order, z_val, mode, kind))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        // Complex order not supported
        (SpecialTensor::ComplexScalar(_), _) | (SpecialTensor::ComplexVec(_), _) => {
            not_yet_implemented(function, mode, "complex-valued order not supported")
        }
        // Empty
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn bessel_complex_scalar(
    _function: &'static str,
    order: f64,
    z: Complex64,
    mode: RuntimeMode,
    kind: BesselKind,
) -> Result<Complex64, SpecialError> {
    match kind {
        BesselKind::Jv => Ok(complex_jv_scalar(order, z)),
        BesselKind::Yv => complex_yv_scalar(order, z, mode),
        BesselKind::Iv => Ok(complex_iv_scalar(order, z)),
        BesselKind::Kv => complex_kv_scalar(order, z, mode),
    }
}

/// Riccati-Bessel function of the first kind, returning
/// `(S_k(x), S'_k(x))` for `k = 0..=n`.
///
/// Matches `scipy.special.riccati_jn(n, x)`. Defined as
///   S_k(x) = x · j_k(x)
/// where j_k is the spherical Bessel function of the first kind.
/// The values follow the same upward recurrence as j_k; the derivatives
/// use S'_0 = cos(x) and S'_k = -k S_k(x)/x + S_{k-1}(x) for k ≥ 1.
pub fn riccati_jn(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
    let len = n as usize + 1;
    let mut s = Vec::with_capacity(len);
    let mut sp = Vec::with_capacity(len);
    for k in 0..=n {
        let jk = spherical_jn_nonneg(k, x);
        s.push(x * jk);
    }
    sp.push(x.cos());
    for k in 1..=n as usize {
        let inv_x = if x.abs() < 1.0e-300 { 0.0 } else { 1.0 / x };
        let val = -((k as f64) * s[k] * inv_x) + s[k - 1];
        sp.push(val);
    }
    (s, sp)
}

/// Riccati-Bessel function of the second kind, returning
/// `(C_k(x), C'_k(x))` for `k = 0..=n`.
///
/// Matches `scipy.special.riccati_yn(n, x)`. Defined as
///   C_k(x) = -x · y_k(x)
/// where y_k is the spherical Bessel function of the second kind.
/// `C_0(x) = cos(x)` and `C'_0(x) = -sin(x)`; the higher orders use the
/// upward recurrence on C and the derivative identity
/// C'_k = -k C_k(x)/x + C_{k-1}(x).
pub fn riccati_yn(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
    let len = n as usize + 1;
    let mut c = Vec::with_capacity(len);
    let mut cp = Vec::with_capacity(len);
    for k in 0..=n {
        let yk = spherical_yn_nonneg(k, x);
        c.push(-x * yk);
    }
    cp.push(-x.sin());
    for k in 1..=n as usize {
        let inv_x = if x.abs() < 1.0e-300 { 0.0 } else { 1.0 / x };
        let val = -((k as f64) * c[k] * inv_x) + c[k - 1];
        cp.push(val);
    }
    (c, cp)
}

/// Olver/Tricomi initial-guess for the first zero of J_n at large n,
/// from DLMF 10.21.40 (asymptotic expansion in n^(-1/3)):
///   j_{n,1} ≈ n + a·n^(1/3) + b·n^(-1/3) − c·n^(-1) − d·n^(-5/3)
/// with a ≈ 1.855_757, b ≈ 1.033_150, c ≈ 0.003_972, d ≈ 0.090_8.
fn olver_jn_first_zero(n_f: f64) -> f64 {
    let n_one_third = n_f.cbrt();
    let inv_n_one_third = 1.0 / n_one_third;
    let inv_n = 1.0 / n_f;
    let inv_n_five_thirds = inv_n * inv_n_one_third * inv_n_one_third;
    n_f + 1.855_757_081_206_722 * n_one_third + 1.033_150_071_8 * inv_n_one_third
        - 0.003_972 * inv_n
        - 0.090_8 * inv_n_five_thirds
}

/// Olver/Tricomi initial-guess for the first zero of Y_n at large n
/// (DLMF 10.21.40 second-kind variant).
fn olver_yn_first_zero(n_f: f64) -> f64 {
    let n_one_third = n_f.cbrt();
    let inv_n_one_third = 1.0 / n_one_third;
    let inv_n = 1.0 / n_f;
    let inv_n_five_thirds = inv_n * inv_n_one_third * inv_n_one_third;
    n_f + 0.931_577_4 * n_one_third + 0.260_086 * inv_n_one_third
        - 0.011_911 * inv_n
        - 0.043_4 * inv_n_five_thirds
}

/// Refine a bessel-zero initial guess by bracket expansion + bisection.
/// Walks outward from `initial` in steps of `step`, bounded by [`floor`,
/// initial+max_outward], until a sign change of `f_at` is found, then
/// bisects to ~1e-12 precision. Returns `initial` unchanged if no
/// bracket can be found within `max_outward`.
fn bracket_and_bisect_zero(
    f_at: impl Fn(f64) -> f64,
    initial: f64,
    floor: f64,
    step: f64,
    max_outward: f64,
) -> f64 {
    let mut lo = (initial - step).max(floor);
    let mut hi = (initial + step).max(lo + step);
    let mut f_lo = f_at(lo);
    let mut f_hi = f_at(hi);
    let mut walked = step;
    while walked < max_outward {
        if f_lo.is_finite() && f_hi.is_finite() && f_lo.signum() != f_hi.signum() {
            break;
        }
        let new_lo = (lo - step).max(floor);
        let new_hi = hi + step;
        if new_lo == lo && new_hi == hi {
            break;
        }
        lo = new_lo;
        hi = new_hi;
        f_lo = f_at(lo);
        f_hi = f_at(hi);
        walked += step;
    }
    if !f_lo.is_finite()
        || !f_hi.is_finite()
        || f_lo.signum() == f_hi.signum()
    {
        return initial;
    }
    for _ in 0..120 {
        let mid = 0.5 * (lo + hi);
        let f_mid = f_at(mid);
        if !f_mid.is_finite() {
            break;
        }
        if f_mid.signum() == f_lo.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < 1.0e-12 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// First `k` positive zeros of the Bessel function Y_n(x), for integer
/// order `n ≥ 0`. Returns a Vec of length `k`, sorted ascending.
///
/// Matches `scipy.special.yn_zeros(n, k)`. Initial guess is piecewise:
/// for k=1 with n ≥ 5 use Olver/Tricomi (DLMF 10.21.40); otherwise use
/// McMahon's asymptotic
///   μ = (4 k + 2 n - 3) · π / 4
///   y_{n,k} ≈ μ - (4 n² - 1)/(8 μ) - (4 n² - 1)(28 n² - 31)/(384 μ³)
/// (DLMF 10.21.19 second-kind variant — offset differs from jn_zeros by
/// 2 because zeros of Y_n interlace those of J_n). Refines via bracket
/// expansion + bisection.
pub fn yn_zeros(n: u32, k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(k);
    let n_f = n as f64;
    let four_n_sq = 4.0 * n_f * n_f;
    let mut prev_zero = 0.0_f64;
    for ki in 1..=k {
        let mu = (4.0 * ki as f64 + 2.0 * n_f - 3.0) * std::f64::consts::PI / 4.0;
        let inv_mu = 1.0 / mu;
        let inv_mu2 = inv_mu * inv_mu;
        let mcmahon = mu - (four_n_sq - 1.0) / (8.0 * mu)
            - (four_n_sq - 1.0) * (28.0 * four_n_sq - 31.0) / 384.0 * inv_mu * inv_mu2;
        let initial = if ki == 1 && n >= 5 {
            olver_yn_first_zero(n_f)
        } else if ki >= 2 {
            // Successive zeros are separated by ~π; if the asymptotic
            // mcmahon falls below prev_zero + π/4 (sign that mcmahon is
            // too low for this regime), nudge it up.
            mcmahon.max(prev_zero + 0.5 * std::f64::consts::PI)
        } else {
            mcmahon
        };
        let f_at = |x: f64| -> f64 {
            yn_scalar(n_f, x, RuntimeMode::Strict).unwrap_or(f64::NAN)
        };
        let floor = if ki >= 2 {
            prev_zero + 1.0e-6
        } else {
            1.0e-6
        };
        let zero =
            bracket_and_bisect_zero(f_at, initial, floor, 0.5, (n_f * 0.25).max(20.0));
        out.push(zero);
        prev_zero = zero;
    }
    out
}

/// First `k` positive zeros of the Bessel function J_n(x), for integer
/// order `n ≥ 0`. Returns a Vec of length `k`, sorted ascending.
///
/// Matches `scipy.special.jn_zeros(n, k)`. Initial guess is piecewise:
/// for k=1 with n ≥ 5 use Olver/Tricomi (DLMF 10.21.40); otherwise use
/// McMahon's asymptotic
///   μ = (4 k + 2 n - 1) · π / 4 = (k + n/2 − 1/4) · π
///   j_{n,k} ≈ μ - (4 n² - 1)/(8 μ) - (4 n² - 1)(28 n² - 31)/(384 μ³)
/// (DLMF 10.21.19). Refines via bracket expansion + bisection. Bisection
/// is preferred over Newton for the same basin-flip robustness reason
/// as ai_zeros / bi_zeros. Without the Olver fallback, McMahon's
/// expansion produces useless guesses for moderate-large n at small k
/// (e.g. n=100, k=1 places McMahon at ~120 vs the true 108.84).
pub fn jn_zeros(n: u32, k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(k);
    let n_f = n as f64;
    let four_n_sq = 4.0 * n_f * n_f;
    let mut prev_zero = 0.0_f64;
    for ki in 1..=k {
        let mu = (4.0 * ki as f64 + 2.0 * n_f - 1.0) * std::f64::consts::PI / 4.0;
        let inv_mu = 1.0 / mu;
        let inv_mu2 = inv_mu * inv_mu;
        let mcmahon = mu - (four_n_sq - 1.0) / (8.0 * mu)
            - (four_n_sq - 1.0) * (28.0 * four_n_sq - 31.0) / 384.0 * inv_mu * inv_mu2;
        let initial = if ki == 1 && n >= 5 {
            olver_jn_first_zero(n_f)
        } else if ki >= 2 {
            mcmahon.max(prev_zero + 0.5 * std::f64::consts::PI)
        } else {
            mcmahon
        };
        let f_at = |x: f64| -> f64 {
            jn_scalar(n_f, x, RuntimeMode::Strict).unwrap_or(f64::NAN)
        };
        let floor = if ki >= 2 {
            prev_zero + 1.0e-6
        } else {
            1.0e-6
        };
        let zero =
            bracket_and_bisect_zero(f_at, initial, floor, 0.5, (n_f * 0.25).max(20.0));
        out.push(zero);
        prev_zero = zero;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(value: f64) -> SpecialTensor {
        SpecialTensor::RealScalar(value)
    }

    fn tensor_result(result: SpecialResult) -> Result<SpecialTensor, String> {
        result.map_err(|err| err.to_string())
    }

    fn real_value(tensor: SpecialTensor) -> Result<f64, String> {
        match tensor {
            SpecialTensor::RealScalar(value) => Ok(value),
            other => Err(format!("expected RealScalar, got {other:?}")),
        }
    }

    fn complex_value(tensor: SpecialTensor) -> Result<Complex64, String> {
        match tensor {
            SpecialTensor::ComplexScalar(value) => Ok(value),
            other => Err(format!("expected ComplexScalar, got {other:?}")),
        }
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tol: f64) {
        assert!(
            (actual.re - expected.re).abs() < tol,
            "real mismatch: actual={} expected={}",
            actual.re,
            expected.re
        );
        assert!(
            (actual.im - expected.im).abs() < tol,
            "imag mismatch: actual={} expected={}",
            actual.im,
            expected.im
        );
    }

    #[test]
    fn jvp_zero_order_matches_jv() -> Result<(), String> {
        let v = scalar(2.0);
        let x = scalar(1.25);
        let derivative = real_value(tensor_result(jvp(&v, &x, 0, RuntimeMode::Strict))?)?;
        let base = real_value(tensor_result(jv(&v, &x, RuntimeMode::Strict))?)?;
        assert!((derivative - base).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn jvp_first_and_second_derivatives_match_recurrences() -> Result<(), String> {
        let x = 1.25;
        let first = real_value(tensor_result(jvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let expected_first = -real_value(tensor_result(j1(&scalar(x), RuntimeMode::Strict))?)?;
        assert!((first - expected_first).abs() < 1e-12);

        let second = real_value(tensor_result(jvp(
            &scalar(0.0),
            &scalar(x),
            2,
            RuntimeMode::Strict,
        ))?)?;
        let j0_x = real_value(tensor_result(j0(&scalar(x), RuntimeMode::Strict))?)?;
        let j2_x = real_value(tensor_result(jn(
            &scalar(2.0),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;
        let expected_second = 0.5 * (j2_x - j0_x);
        assert!((second - expected_second).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn yvp_first_derivative_matches_y0_identity() -> Result<(), String> {
        let x = 1.75;
        let derivative = real_value(tensor_result(yvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let expected = -real_value(tensor_result(y1(&scalar(x), RuntimeMode::Strict))?)?;
        assert!((derivative - expected).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn ivp_and_kvp_first_derivatives_match_order_zero_identities() -> Result<(), String> {
        let x = 1.1;
        let iv_derivative = real_value(tensor_result(ivp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let iv_expected = real_value(tensor_result(iv(
            &scalar(1.0),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;
        assert!((iv_derivative - iv_expected).abs() < 1e-12);

        let kv_derivative = real_value(tensor_result(kvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let kv_expected = -real_value(tensor_result(kv(
            &scalar(1.0),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;
        assert!((kv_derivative - kv_expected).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_bessel_derivative_zero_order_matches_base_functions() -> Result<(), String> {
        let z = complex_scalar(1.25, 0.75);

        let jvp_zero = complex_value(tensor_result(jvp(
            &scalar(0.5),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let jv_base = complex_value(tensor_result(jv(&scalar(0.5), &z, RuntimeMode::Strict))?)?;
        assert!((jvp_zero.re - jv_base.re).abs() < 1e-12);
        assert!((jvp_zero.im - jv_base.im).abs() < 1e-12);

        let yvp_zero = complex_value(tensor_result(yvp(
            &scalar(0.5),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let yv_base = complex_value(tensor_result(yv(&scalar(0.5), &z, RuntimeMode::Strict))?)?;
        assert!((yvp_zero.re - yv_base.re).abs() < 1e-12);
        assert!((yvp_zero.im - yv_base.im).abs() < 1e-12);

        let ivp_zero = complex_value(tensor_result(ivp(
            &scalar(0.5),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let iv_base = complex_value(tensor_result(iv(&scalar(0.5), &z, RuntimeMode::Strict))?)?;
        assert!((ivp_zero.re - iv_base.re).abs() < 1e-12);
        assert!((ivp_zero.im - iv_base.im).abs() < 1e-12);

        let kvp_zero = complex_value(tensor_result(kvp(
            &scalar(0.5),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let kv_base = complex_value(tensor_result(kv(&scalar(0.5), &z, RuntimeMode::Strict))?)?;
        assert!((kvp_zero.re - kv_base.re).abs() < 1e-12);
        assert!((kvp_zero.im - kv_base.im).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_bessel_negative_integer_orders_stay_finite() {
        let z = Complex64::new(1.0, 0.0);

        assert_complex_close(
            complex_jv_scalar(-1.0, z),
            Complex64::new(-0.440_050_585_744_933_5, 0.0),
            1.0e-12,
        );
        assert_complex_close(
            complex_jv_scalar(-2.0, z),
            Complex64::new(0.114_903_484_931_900_5, 0.0),
            1.0e-12,
        );
        assert_complex_close(
            complex_iv_scalar(-1.0, z),
            Complex64::new(0.565_159_103_992_485, 0.0),
            1.0e-12,
        );
        assert_complex_close(
            complex_jv_scalar(-1.0, Complex64::new(0.0, 0.0)),
            Complex64::new(0.0, 0.0),
            1.0e-12,
        );
        assert_complex_close(
            complex_iv_scalar(-1.0, Complex64::new(0.0, 0.0)),
            Complex64::new(0.0, 0.0),
            1.0e-12,
        );
    }

    #[test]
    fn complex_bessel_derivative_reduces_to_real_axis_values() -> Result<(), String> {
        let x = 1.8;

        let j_real = real_value(tensor_result(jvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let j_complex = complex_value(tensor_result(jvp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((j_real - j_complex.re).abs() < 1e-12);
        assert!(j_complex.im.abs() < 1e-12);

        let y_real = real_value(tensor_result(yvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let y_complex = complex_value(tensor_result(yvp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((y_real - y_complex.re).abs() < 1e-12);
        assert!(y_complex.im.abs() < 1e-12);

        let i_real = real_value(tensor_result(ivp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let i_complex = complex_value(tensor_result(ivp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((i_real - i_complex.re).abs() < 1e-12);
        assert!(i_complex.im.abs() < 1e-12);

        let k_real = real_value(tensor_result(kvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let k_complex = complex_value(tensor_result(kvp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((k_real - k_complex.re).abs() < 1e-12);
        assert!(k_complex.im.abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_bessel_derivative_broadcasts_complex_vectors() -> Result<(), String> {
        let zs = SpecialTensor::ComplexVec(vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(1.5, -0.25),
            Complex64::new(2.0, 0.0),
        ]);
        let result = tensor_result(jvp(&scalar(0.5), &zs, 1, RuntimeMode::Strict))?;
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 3);
                for (index, value) in values.iter().enumerate() {
                    let lane = match &zs {
                        SpecialTensor::ComplexVec(items) => items[index],
                        _ => unreachable!(),
                    };
                    let expected = complex_value(tensor_result(jvp(
                        &scalar(0.5),
                        &SpecialTensor::ComplexScalar(lane),
                        1,
                        RuntimeMode::Strict,
                    ))?)?;
                    assert!((value.re - expected.re).abs() < 1e-12);
                    assert!((value.im - expected.im).abs() < 1e-12);
                }
            }
            other => return Err(format!("expected ComplexVec, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn hankel_derivatives_return_complex_conjugate_parts() -> Result<(), String> {
        let x = 2.0;
        let h1 = complex_value(tensor_result(h1vp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let h2 = complex_value(tensor_result(h2vp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let j_part = real_value(tensor_result(jvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let y_part = real_value(tensor_result(yvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;

        assert!((h1.re - j_part).abs() < 1e-12);
        assert!((h1.im - y_part).abs() < 1e-12);
        assert!((h2.re - j_part).abs() < 1e-12);
        assert!((h2.im + y_part).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_hankel_derivative_zero_order_matches_base_hankel() -> Result<(), String> {
        let z = complex_scalar(2.0, 1.0);
        let h1_derivative = complex_value(tensor_result(h1vp(
            &scalar(1.0),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let h1_base = complex_value(tensor_result(hankel1(
            &scalar(1.0),
            &z,
            RuntimeMode::Strict,
        ))?)?;
        assert!((h1_derivative.re - h1_base.re).abs() < 1e-12);
        assert!((h1_derivative.im - h1_base.im).abs() < 1e-12);

        let h2_derivative = complex_value(tensor_result(h2vp(
            &scalar(1.0),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let h2_base = complex_value(tensor_result(hankel2(
            &scalar(1.0),
            &z,
            RuntimeMode::Strict,
        ))?)?;
        assert!((h2_derivative.re - h2_base.re).abs() < 1e-12);
        assert!((h2_derivative.im - h2_base.im).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_hankel_derivative_reduces_to_real_axis_values() -> Result<(), String> {
        let x = 2.0;
        let h1_real = complex_value(tensor_result(h1vp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let h1_complex = complex_value(tensor_result(h1vp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((h1_real.re - h1_complex.re).abs() < 1e-12);
        assert!((h1_real.im - h1_complex.im).abs() < 1e-12);

        let h2_real = complex_value(tensor_result(h2vp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let h2_complex = complex_value(tensor_result(h2vp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((h2_real.re - h2_complex.re).abs() < 1e-12);
        assert!((h2_real.im - h2_complex.im).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_hankel_derivatives_broadcast_vectors() -> Result<(), String> {
        let zs = SpecialTensor::ComplexVec(vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, 0.0),
        ]);
        let h1_result = tensor_result(h1vp(&scalar(0.0), &zs, 1, RuntimeMode::Strict))?;
        match h1_result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 3);
                for (index, value) in values.iter().enumerate() {
                    let z = match &zs {
                        SpecialTensor::ComplexVec(items) => items[index],
                        _ => unreachable!(),
                    };
                    let expected = complex_value(tensor_result(h1vp(
                        &scalar(0.0),
                        &SpecialTensor::ComplexScalar(z),
                        1,
                        RuntimeMode::Strict,
                    ))?)?;
                    assert!(
                        (value.re - expected.re).abs() < 1e-12
                            || (value.re.is_nan() && expected.re.is_nan())
                    );
                    assert!(
                        (value.im - expected.im).abs() < 1e-12
                            || (value.im.is_nan() && expected.im.is_nan())
                    );
                }
            }
            other => return Err(format!("expected ComplexVec, got {other:?}")),
        }

        let h2_result = tensor_result(h2vp(&scalar(0.0), &zs, 1, RuntimeMode::Strict))?;
        match h2_result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 3);
                for (index, value) in values.iter().enumerate() {
                    let z = match &zs {
                        SpecialTensor::ComplexVec(items) => items[index],
                        _ => unreachable!(),
                    };
                    let expected = complex_value(tensor_result(h2vp(
                        &scalar(0.0),
                        &SpecialTensor::ComplexScalar(z),
                        1,
                        RuntimeMode::Strict,
                    ))?)?;
                    assert!(
                        (value.re - expected.re).abs() < 1e-12
                            || (value.re.is_nan() && expected.re.is_nan())
                    );
                    assert!(
                        (value.im - expected.im).abs() < 1e-12
                            || (value.im.is_nan() && expected.im.is_nan())
                    );
                }
            }
            other => return Err(format!("expected ComplexVec, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn derivative_helpers_broadcast_vectors() -> Result<(), String> {
        let values = SpecialTensor::RealVec(vec![1.0, 2.0]);
        let result = tensor_result(jvp(&scalar(0.0), &values, 1, RuntimeMode::Strict))?;
        match result {
            SpecialTensor::RealVec(items) => {
                assert_eq!(items.len(), 2);
                assert!((items[0] + j1_core(1.0)).abs() < 1e-12);
                assert!((items[1] + j1_core(2.0)).abs() < 1e-12);
            }
            other => return Err(format!("expected RealVec, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn yvp_hardened_rejects_negative_real_argument() -> Result<(), String> {
        let Err(err) = yvp(&scalar(0.0), &scalar(-1.0), 1, RuntimeMode::Hardened) else {
            return Err("negative argument should fail closed".to_string());
        };
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        Ok(())
    }

    #[test]
    fn hankel_tensor_interface_returns_full_complex_parts() -> Result<(), String> {
        let x = scalar(2.0);
        let h1 = complex_value(tensor_result(hankel1(
            &scalar(0.0),
            &x,
            RuntimeMode::Strict,
        ))?)?;
        let h2 = complex_value(tensor_result(hankel2(
            &scalar(0.0),
            &x,
            RuntimeMode::Strict,
        ))?)?;
        let j = real_value(tensor_result(jv(&scalar(0.0), &x, RuntimeMode::Strict))?)?;
        let y = real_value(tensor_result(yv(&scalar(0.0), &x, RuntimeMode::Strict))?)?;

        assert!((h1.re - j).abs() < 1e-12);
        assert!((h1.im - y).abs() < 1e-12);
        assert!((h2.re - j).abs() < 1e-12);
        assert!((h2.im + y).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_hankel_tensor_interface_matches_jv_plus_minus_i_yv() -> Result<(), String> {
        let z = complex_scalar(2.0, 1.0);
        let j = complex_value(tensor_result(jv(&scalar(1.0), &z, RuntimeMode::Strict))?)?;
        let y = complex_value(tensor_result(yv(&scalar(1.0), &z, RuntimeMode::Strict))?)?;
        let h1 = complex_value(tensor_result(hankel1(
            &scalar(1.0),
            &z,
            RuntimeMode::Strict,
        ))?)?;
        let h2 = complex_value(tensor_result(hankel2(
            &scalar(1.0),
            &z,
            RuntimeMode::Strict,
        ))?)?;

        let iy = Complex64::new(-y.im, y.re);
        assert!((h1.re - (j + iy).re).abs() < 1e-12);
        assert!((h1.im - (j + iy).im).abs() < 1e-12);

        let minus_iy = Complex64::new(y.im, -y.re);
        assert!((h2.re - (j + minus_iy).re).abs() < 1e-12);
        assert!((h2.im - (j + minus_iy).im).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn hankel_broadcasts_real_vector_inputs_to_complex_vec() -> Result<(), String> {
        let xs = SpecialTensor::RealVec(vec![1.0, 2.0]);
        let result = tensor_result(hankel1(&scalar(0.0), &xs, RuntimeMode::Strict))?;
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 2);
                for value in values {
                    assert!(value.re.is_finite());
                    assert!(value.im.is_finite());
                }
            }
            other => return Err(format!("expected ComplexVec, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn wright_bessel_zero_matches_rgamma_boundary() -> Result<(), String> {
        let result = real_value(tensor_result(wright_bessel(
            &scalar(0.0),
            &scalar(0.0),
            &scalar(0.0),
            RuntimeMode::Strict,
        ))?)?;
        assert_eq!(result, 0.0);

        let exp_result = real_value(tensor_result(wright_bessel(
            &scalar(0.0),
            &scalar(1.0),
            &scalar(2.0),
            RuntimeMode::Strict,
        ))?)?;
        assert!((exp_result - 2.0_f64.exp()).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn wright_bessel_matches_large_a_scipy_reference_values() -> Result<(), String> {
        let cases: [(f64, f64, f64, f64, f64); 4] = [
            (20.0, 1.5, 2.0, std::f64::consts::FRAC_2_SQRT_PI, 1e-13),
            (50.0, 0.25, 0.1, 0.275_815_662_830_209_3, 1e-13),
            (100.0, 0.0, 1e20, 1.071_510_288_125_446_2e-136, 1e-12),
            (100.0, 100.0, 1e20, 1.071_510_288_125_468_3e-156, 1e-12),
        ];

        for (a, b, x, expected, rel_tol) in cases {
            let actual = real_value(tensor_result(wright_bessel(
                &scalar(a),
                &scalar(b),
                &scalar(x),
                RuntimeMode::Strict,
            ))?)?;
            let scale = expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= rel_tol * scale,
                "wright_bessel({a}, {b}, {x}) mismatch: actual={actual}, expected={expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn wright_bessel_rejects_negative_domain_in_hardened_mode() -> Result<(), String> {
        let strict = real_value(tensor_result(wright_bessel(
            &scalar(-0.5),
            &scalar(2.0),
            &scalar(0.1),
            RuntimeMode::Strict,
        ))?)?;
        assert!(strict.is_nan());

        let Err(err) = wright_bessel(
            &scalar(-0.5),
            &scalar(2.0),
            &scalar(0.1),
            RuntimeMode::Hardened,
        ) else {
            return Err("negative wright_bessel input should fail closed".to_string());
        };
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        Ok(())
    }

    // ========================================================================
    // Complex spherical Bessel tests
    // ========================================================================

    fn complex_scalar(re: f64, im: f64) -> SpecialTensor {
        SpecialTensor::ComplexScalar(Complex64::new(re, im))
    }

    fn get_complex(result: SpecialResult) -> Result<Complex64, String> {
        match result.map_err(|e| e.to_string())? {
            SpecialTensor::ComplexScalar(c) => Ok(c),
            other => Err(format!("expected complex scalar, got {other:?}")),
        }
    }

    #[test]
    fn complex_spherical_jn_reduces_to_real() -> Result<(), String> {
        // For real positive z, complex j_n should match real j_n
        let n = 2.0;
        let x = 1.5;

        let real_result = real_value(tensor_result(spherical_jn(
            &scalar(n),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;

        let complex_result = get_complex(spherical_jn(
            &scalar(n),
            &complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;

        assert!(
            (complex_result.re - real_result).abs() < 1e-10,
            "Real parts should match: {} vs {}",
            complex_result.re,
            real_result
        );
        assert!(
            complex_result.im.abs() < 1e-10,
            "Imaginary part should be negligible: {}",
            complex_result.im
        );
        Ok(())
    }

    #[test]
    fn complex_spherical_jn_at_zero() {
        let j0 = complex_spherical_jn(0, Complex64::new(0.0, 0.0));
        assert_eq!(j0.re, 1.0);
        assert_eq!(j0.im, 0.0);

        let j1 = complex_spherical_jn(1, Complex64::new(0.0, 0.0));
        assert_eq!(j1.re, 0.0);
        assert_eq!(j1.im, 0.0);
    }

    #[test]
    fn complex_spherical_yn_at_zero_is_negative_inf() {
        let y0 = complex_spherical_yn(0, Complex64::new(0.0, 0.0));
        assert!(y0.re.is_infinite() && y0.re < 0.0);
    }

    #[test]
    fn complex_spherical_in_reduces_to_real() -> Result<(), String> {
        let n = 1.0;
        let x = 2.0;

        let real_result = real_value(tensor_result(spherical_in(
            &scalar(n),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;

        let complex_result = get_complex(spherical_in(
            &scalar(n),
            &complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;

        assert!(
            (complex_result.re - real_result).abs() < 1e-8,
            "Real parts should match: {} vs {}",
            complex_result.re,
            real_result
        );
        Ok(())
    }

    #[test]
    fn complex_spherical_kn_reduces_to_real() -> Result<(), String> {
        let n = 0.0;
        let x = 1.0;

        let real_result = real_value(tensor_result(spherical_kn(
            &scalar(n),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;

        let complex_result = get_complex(spherical_kn(
            &scalar(n),
            &complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;

        assert!(
            (complex_result.re - real_result).abs() < 1e-10,
            "Real parts should match: {} vs {}",
            complex_result.re,
            real_result
        );
        Ok(())
    }

    #[test]
    fn complex_spherical_jn_tensor_interface() -> Result<(), String> {
        let result = spherical_jn(&scalar(2.0), &complex_scalar(1.0, 0.5), RuntimeMode::Strict);
        let c = get_complex(result)?;
        assert!(c.re.is_finite());
        assert!(c.im.is_finite());
        Ok(())
    }

    // ── Complex cylindrical Bessel function tests ───────────────────────

    #[test]
    fn complex_jv_real_matches_real_jv() -> Result<(), String> {
        let v = scalar(1.5);
        let z_real = scalar(2.0);
        let z_complex = complex_scalar(2.0, 0.0);

        let real_result = real_value(tensor_result(jv(&v, &z_real, RuntimeMode::Strict))?)?;
        let complex_result =
            complex_value(tensor_result(jv(&v, &z_complex, RuntimeMode::Strict))?)?;

        assert!((complex_result.re - real_result).abs() < 1e-10);
        assert!(complex_result.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_jv_pure_imaginary() -> Result<(), String> {
        let v = scalar(0.0);
        let z = complex_scalar(0.0, 2.0);
        let result = complex_value(tensor_result(jv(&v, &z, RuntimeMode::Strict))?)?;
        assert!(result.re.is_finite());
        assert!(result.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_iv_real_matches_real_iv() -> Result<(), String> {
        let v = scalar(1.0);
        let z_real = scalar(2.0);
        let z_complex = complex_scalar(2.0, 0.0);

        let real_result = real_value(tensor_result(iv(&v, &z_real, RuntimeMode::Strict))?)?;
        let complex_result =
            complex_value(tensor_result(iv(&v, &z_complex, RuntimeMode::Strict))?)?;

        assert!((complex_result.re - real_result).abs() < 1e-10);
        assert!(complex_result.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_yv_tensor_interface() -> Result<(), String> {
        let v = scalar(1.0);
        let z = complex_scalar(2.0, 1.0);
        let result = yv(&v, &z, RuntimeMode::Strict);
        let c = complex_value(tensor_result(result)?)?;
        assert!(c.re.is_finite());
        assert!(c.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_kv_tensor_interface() -> Result<(), String> {
        let v = scalar(1.0);
        let z = complex_scalar(2.0, 1.0);
        let result = kv(&v, &z, RuntimeMode::Strict);
        let c = complex_value(tensor_result(result)?)?;
        assert!(c.re.is_finite());
        assert!(c.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_bessel_vector_broadcast() -> Result<(), String> {
        let v = scalar(0.0);
        let zs = SpecialTensor::ComplexVec(vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.5, 1.0),
        ]);
        let result = tensor_result(jv(&v, &zs, RuntimeMode::Strict))?;
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 3);
                for v in values {
                    assert!(v.is_finite());
                }
            }
            _ => return Err("expected ComplexVec".into()),
        }
        Ok(())
    }

    #[test]
    fn riccati_jn_zero_order_is_sin_cos() {
        // S_0(x) = sin(x), S'_0(x) = cos(x).
        for &x in &[0.0_f64, 0.5, 1.0, 2.5, 5.0] {
            let (s, sp) = riccati_jn(0, x);
            assert_eq!(s.len(), 1);
            assert!((s[0] - x.sin()).abs() < 1e-12);
            assert!((sp[0] - x.cos()).abs() < 1e-12);
        }
    }

    #[test]
    fn riccati_jn_metamorphic_recurrence_consistency() {
        // S'_k = -k S_k / x + S_{k-1} for k ≥ 1. The function builds the
        // derivative using this identity, so verify it holds independently.
        let x = 2.5_f64;
        let n = 6;
        let (s, sp) = riccati_jn(n, x);
        for k in 1..=n as usize {
            let predicted = -(k as f64) * s[k] / x + s[k - 1];
            assert!(
                (sp[k] - predicted).abs() < 1e-12,
                "S'_{k}({x}) = {} vs recurrence {predicted}",
                sp[k]
            );
        }
    }

    #[test]
    fn riccati_yn_zero_order_is_cos_minus_sin() {
        // C_0(x) = cos(x), C'_0(x) = -sin(x).
        for &x in &[0.5_f64, 1.0, 2.5, 5.0] {
            let (c, cp) = riccati_yn(0, x);
            assert!((c[0] - x.cos()).abs() < 1e-12);
            assert!((cp[0] - (-x.sin())).abs() < 1e-12);
        }
    }

    #[test]
    fn riccati_metamorphic_pythagorean_at_k_zero() {
        // S_0² + C_0² = sin²(x) + cos²(x) = 1.
        for &x in &[0.5_f64, 1.0, 2.5, 5.0] {
            let (s, _) = riccati_jn(0, x);
            let (c, _) = riccati_yn(0, x);
            let sum = s[0] * s[0] + c[0] * c[0];
            assert!(
                (sum - 1.0).abs() < 1e-13,
                "S_0² + C_0² = {sum} should be 1 at x={x}"
            );
        }
    }

    #[test]
    fn yn_zeros_first_three_of_y0_match_known() {
        // First three zeros of Y_0:
        //   0.893577_695_5
        //   3.957678_419_3
        //   7.086051_060_3
        let zeros = yn_zeros(0, 3);
        let expected = [
            0.893_577_695_5_f64,
            3.957_678_419_3,
            7.086_051_060_3,
        ];
        assert_eq!(zeros.len(), 3);
        for (got, exp) in zeros.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-4,
                "y0 zero {got} vs {exp}"
            );
        }
    }

    #[test]
    fn yn_zeros_metamorphic_zero_value() {
        for &n in &[0_u32, 1, 2] {
            let zeros = yn_zeros(n, 4);
            for z in &zeros {
                let val =
                    yn_scalar(n as f64, *z, RuntimeMode::Strict).expect("yn");
                assert!(
                    val.abs() < 1e-5,
                    "Y_{n}({z}) = {val} should be ≈ 0"
                );
            }
        }
    }

    #[test]
    fn yn_zeros_metamorphic_strictly_increasing() {
        let zeros = yn_zeros(0, 8);
        for z in &zeros {
            assert!(*z > 0.0);
        }
        for w in zeros.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn jn_zeros_first_three_of_j0_match_known() {
        // First three zeros of J_0:
        //   2.404825557695773
        //   5.520078110286311
        //   8.653727912911012
        let zeros = jn_zeros(0, 3);
        let expected = [
            2.404_825_557_695_773_f64,
            5.520_078_110_286_311,
            8.653_727_912_911_012,
        ];
        assert_eq!(zeros.len(), 3);
        for (got, exp) in zeros.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-6,
                "j0 zero {got} vs {exp}"
            );
        }
    }

    #[test]
    fn jn_zeros_first_two_of_j1_match_known() {
        // First two zeros of J_1:
        //   3.831705970207512
        //   7.015586669815619
        let zeros = jn_zeros(1, 2);
        let expected = [3.831_705_970_207_512_f64, 7.015_586_669_815_619];
        for (got, exp) in zeros.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-6,
                "j1 zero {got} vs {exp}"
            );
        }
    }

    #[test]
    fn jn_zeros_metamorphic_zero_value() {
        // J_n(zero) ≈ 0 at every returned zero.
        for &n in &[0_u32, 1, 2, 3] {
            let zeros = jn_zeros(n, 5);
            for z in &zeros {
                let val =
                    jn_scalar(n as f64, *z, RuntimeMode::Strict).expect("jn");
                assert!(
                    val.abs() < 1e-6,
                    "J_{n}({z}) = {val} should be ≈ 0"
                );
            }
        }
    }

    #[test]
    fn jn_zeros_large_n_first_zero_matches_olver() {
        // Regression for [frankenscipy-zppm2]: McMahon's asymptotic guess
        // diverges from the true first zero for moderate-large n. Olver
        // expansion fixes it; verify J_n(jn_zeros(n,1)[0]) ≈ 0 across the
        // failure regime (n=10, 20, 50, 100).
        for &n in &[10_u32, 20, 50, 100] {
            let zeros = jn_zeros(n, 1);
            let z = zeros[0];
            let val = jn_scalar(n as f64, z, RuntimeMode::Strict).expect("jn");
            assert!(
                val.abs() < 1e-6,
                "J_{n}({z}) = {val} should be ≈ 0 (n={n}, k=1)"
            );
        }
    }

    #[test]
    fn yn_zeros_large_n_first_zero_matches_olver() {
        for &n in &[10_u32, 20, 50, 100] {
            let zeros = yn_zeros(n, 1);
            let z = zeros[0];
            let val = yn_scalar(n as f64, z, RuntimeMode::Strict).expect("yn");
            assert!(
                val.abs() < 1e-6,
                "Y_{n}({z}) = {val} should be ≈ 0 (n={n}, k=1)"
            );
        }
    }

    #[test]
    fn bessel_zeros_metamorphic_interlace_relation() {
        // DLMF 10.21.2: zeros of J_n and Y_n strictly interlace —
        //   y_{n,1} < j_{n,1} < y_{n,2} < j_{n,2} < ...
        // Verify across multiple n. This is a stronger metamorphic
        // property than zero-value-is-zero: it pins the *count and
        // ordering* of zeros across two different functions.
        for &n in &[0_u32, 1, 2, 5, 10, 20] {
            let k = 5;
            let jzs = jn_zeros(n, k);
            let yzs = yn_zeros(n, k);
            for i in 0..k {
                assert!(
                    yzs[i] < jzs[i],
                    "interlace fail (n={n}, i={i}): y={} should be < j={}",
                    yzs[i],
                    jzs[i]
                );
                if i + 1 < k {
                    assert!(
                        jzs[i] < yzs[i + 1],
                        "interlace fail (n={n}, i={i}): j={} should be < y_next={}",
                        jzs[i],
                        yzs[i + 1]
                    );
                }
            }
        }
    }

    #[test]
    fn bessel_zeros_metamorphic_pi_separation_for_large_k() {
        // DLMF 10.21.20: consecutive zeros of J_n satisfy
        //   j_{n,k+1} - j_{n,k} → π as k → ∞.
        // Take k=20..30 across n=0..3 and check the separation lies
        // within [π − 0.05, π + 0.05].
        for &n in &[0_u32, 1, 2, 3] {
            let zs = jn_zeros(n, 30);
            for i in 19..29 {
                let dx = zs[i + 1] - zs[i];
                let pi = std::f64::consts::PI;
                assert!(
                    (dx - pi).abs() < 0.05,
                    "n={n}, k={}: separation {dx} should be ≈ π",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn jn_zeros_metamorphic_strictly_increasing() {
        let zeros = jn_zeros(0, 10);
        for z in &zeros {
            assert!(*z > 0.0);
        }
        for w in zeros.windows(2) {
            assert!(w[0] < w[1], "zeros must be increasing: {} < {}", w[0], w[1]);
        }
    }
}
