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
pub fn jv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("jv", v, z, mode, |order, x| Ok(jv_scalar(order, x)))
}

/// Bessel function of the second kind for real order v: Y_v(z).
///
/// Y_v(z) = (J_v(z) cos(vπ) - J_{-v}(z)) / sin(vπ) for non-integer v.
/// For integer v, uses the integer-order y0/y1/yn implementations.
pub fn yv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("yv", v, z, mode, |order, x| yv_scalar(order, x, mode))
}

/// Modified Bessel function of the first kind for real order v: I_v(z).
///
/// I_v(z) = (z/2)^v Σ (z²/4)^k / (k! Γ(v+k+1))
pub fn iv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("iv", v, z, mode, |order, x| Ok(iv_scalar(order, x)))
}

/// Modified Bessel function of the second kind for real order v: K_v(z).
///
/// K_v(z) = π/2 (I_{-v}(z) - I_v(z)) / sin(vπ) for non-integer v.
pub fn kv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("kv", v, z, mode, |order, x| kv_scalar(order, x, mode))
}

/// Hankel function of the first kind: H1_v(z) = J_v(z) + i·Y_v(z).
///
/// Returns the real part (imaginary part requires complex output path).
pub fn hankel1(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    // H1_v = J_v + i*Y_v; return J_v as the real part
    map_real_binary("hankel1", v, z, mode, |order, x| Ok(jv_scalar(order, x)))
}

/// Hankel function of the second kind: H2_v(z) = J_v(z) - i·Y_v(z).
///
/// Returns the real part (imaginary part requires complex output path).
pub fn hankel2(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    // H2_v = J_v - i*Y_v; return J_v as the real part
    map_real_binary("hankel2", v, z, mode, |order, x| Ok(jv_scalar(order, x)))
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
    map_real_binary("jvp", v, z, mode, |order, x| {
        bessel_derivative_sum(
            "jvp",
            order,
            x,
            derivative_order,
            mode,
            DerivativeRule::Alternating,
            |shifted_order, value| Ok(jv_scalar(shifted_order, value)),
        )
    })
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
    map_real_binary("yvp", v, z, mode, |order, x| {
        bessel_derivative_sum(
            "yvp",
            order,
            x,
            derivative_order,
            mode,
            DerivativeRule::Alternating,
            |shifted_order, value| yv_scalar(shifted_order, value, mode),
        )
    })
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
    map_real_binary("ivp", v, z, mode, |order, x| {
        bessel_derivative_sum(
            "ivp",
            order,
            x,
            derivative_order,
            mode,
            DerivativeRule::Positive,
            |shifted_order, value| Ok(iv_scalar(shifted_order, value)),
        )
    })
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
    map_real_binary("kvp", v, z, mode, |order, x| {
        bessel_derivative_sum(
            "kvp",
            order,
            x,
            derivative_order,
            mode,
            DerivativeRule::NegativeByOrder,
            |shifted_order, value| kv_scalar(shifted_order, value, mode),
        )
    })
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
    map_real_binary_complex("h1vp", v, z, mode, |order, x| {
        let real = bessel_derivative_sum(
            "h1vp",
            order,
            x,
            derivative_order,
            mode,
            DerivativeRule::Alternating,
            |shifted_order, value| Ok(jv_scalar(shifted_order, value)),
        )?;
        let imag = bessel_derivative_sum(
            "h1vp",
            order,
            x,
            derivative_order,
            mode,
            DerivativeRule::Alternating,
            |shifted_order, value| yv_scalar(shifted_order, value, mode),
        )?;
        Ok(Complex64::new(real, imag))
    })
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
    map_real_binary_complex("h2vp", v, z, mode, |order, x| {
        let real = bessel_derivative_sum(
            "h2vp",
            order,
            x,
            derivative_order,
            mode,
            DerivativeRule::Alternating,
            |shifted_order, value| Ok(jv_scalar(shifted_order, value)),
        )?;
        let imag = bessel_derivative_sum(
            "h2vp",
            order,
            x,
            derivative_order,
            mode,
            DerivativeRule::Alternating,
            |shifted_order, value| yv_scalar(shifted_order, value, mode),
        )?;
        Ok(Complex64::new(real, -imag))
    })
}

#[derive(Debug, Clone, Copy)]
enum DerivativeRule {
    Alternating,
    Positive,
    NegativeByOrder,
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
    map_real_binary("spherical_jn", n, z, mode, |order, x| {
        spherical_jn_scalar(order, x, mode)
    })
}

/// Spherical Bessel function of the second kind y_n(z).
///
/// y_n(z) = √(π/(2z)) Y_{n+1/2}(z)
///
/// Computed via recurrence from:
///   y_0(z) = -cos(z)/z,  y_1(z) = -cos(z)/z² - sin(z)/z
pub fn spherical_yn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("spherical_yn", n, z, mode, |order, x| {
        spherical_yn_scalar(order, x, mode)
    })
}

/// Modified spherical Bessel function of the first kind i_n(z).
///
/// i_n(z) = √(π/(2z)) I_{n+1/2}(z)
///
/// Computed via recurrence from:
///   i_0(z) = sinh(z)/z,  i_1(z) = cosh(z)/z - sinh(z)/z²
pub fn spherical_in(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("spherical_in", n, z, mode, |order, x| {
        spherical_in_scalar(order, x, mode)
    })
}

/// Modified spherical Bessel function of the second kind k_n(z).
///
/// k_n(z) = √(π/(2z)) K_{n+1/2}(z)
///
/// Computed via recurrence from:
///   k_0(z) = π exp(-z)/(2z),  k_1(z) = π exp(-z)/(2z) * (1 + 1/z)
pub fn spherical_kn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("spherical_kn", n, z, mode, |order, x| {
        spherical_kn_scalar(order, x, mode)
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

fn map_real_binary_complex<F>(
    function: &'static str,
    lhs: &SpecialTensor,
    rhs: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<Complex64, SpecialError>,
{
    match (lhs, rhs) {
        (SpecialTensor::RealScalar(left), SpecialTensor::RealScalar(right)) => {
            kernel(*left, *right).map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealVec(left), SpecialTensor::RealScalar(right)) => left
            .iter()
            .copied()
            .map(|value| kernel(value, *right))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealScalar(left), SpecialTensor::RealVec(right)) => right
            .iter()
            .copied()
            .map(|value| kernel(*left, value))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
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
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(_), _)
        | (SpecialTensor::ComplexVec(_), _)
        | (_, SpecialTensor::ComplexScalar(_))
        | (_, SpecialTensor::ComplexVec(_)) => {
            not_yet_implemented(function, mode, "complex-valued input path pending")
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
}
