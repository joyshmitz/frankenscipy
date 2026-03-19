#![forbid(unsafe_code)]

use std::f64::consts::{FRAC_2_PI, PI};

use fsci_runtime::RuntimeMode;

use crate::types::{
    not_yet_implemented, record_special_trace, DispatchPlan, DispatchStep, KernelRegime,
    SpecialError, SpecialErrorKind, SpecialResult, SpecialTensor,
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
    let half_z = z / 2.0;
    let neg_quarter_z2 = -(z * z) / 4.0;

    // First term: (z/2)^v / Γ(v+1)
    let log_first = v * half_z.abs().ln() - lgamma(v + 1.0);
    let mut sum = 0.0;
    let mut log_term = log_first;
    let sign_z = if z < 0.0 && v.fract() != 0.0 {
        // For non-integer v and negative z, J_v(z) involves complex values
        // Return NaN for real-only path
        return f64::NAN;
    } else {
        1.0
    };

    for k in 0..200 {
        let term = log_term.exp();
        let term_sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        sum += term_sign * term;

        if term.abs() < 1e-16 * sum.abs().max(1e-300) && k > 5 {
            break;
        }

        // Next term: multiply by (-z²/4) / ((k+1)(v+k+1))
        let kf = k as f64;
        log_term += (z * z / 4.0).ln() - (kf + 1.0).ln() - (v + kf + 1.0).ln();
    }

    sign_z * sum
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
    if z == 0.0 {
        return if v == 0.0 { 1.0 } else if v > 0.0 { 0.0 } else { f64::INFINITY };
    }

    let az = z.abs();

    // Power series: I_v(z) = (z/2)^v Σ (z²/4)^k / (k! Γ(v+k+1))
    let half_z = az / 2.0;
    let quarter_z2 = az * az / 4.0;

    let log_first = v * half_z.ln() - lgamma(v + 1.0);
    let mut sum = 0.0;
    let mut log_term = log_first;

    for k in 0..200 {
        sum += log_term.exp();

        if log_term.exp() < 1e-16 * sum.max(1e-300) && k > 5 {
            break;
        }

        let kf = k as f64;
        log_term += quarter_z2.ln() - (kf + 1.0).ln() - (v + kf + 1.0).ln();
    }

    // Parity for negative z: I_v(-z) = (-1)^v I_v(z) for integer v
    if z < 0.0 && v.fract() == 0.0 {
        let n = v as i64;
        if n % 2 != 0 {
            return -sum;
        }
    }

    sum
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
        return domain_error_by_mode(
            "kv",
            mode,
            format!("v={v},z={z}"),
            "kv requires z > 0",
        );
    }

    // Non-integer order: K_v = π/2 * (I_{-v} - I_v) / sin(vπ)
    let sin_vpi = (v * PI).sin();
    if sin_vpi.abs() < 1e-15 {
        // Integer or near-integer order: use limiting form
        // K_n(z) ≈ (n-1)!/2 * (2/z)^n for small z, or asymptotic for large z
        // Use asymptotic: K_v(z) ≈ sqrt(π/(2z)) exp(-z) for large z
        if z > 2.0 {
            let factor = (PI / (2.0 * z)).sqrt() * (-z).exp();
            return Ok(factor);
        }
        // Small z: compute via series difference limit
        // Use K_0 and K_1 as base, then recurrence
        let k0 = kv_zero_order(z);
        if v.abs() < 0.5 {
            return Ok(k0);
        }
        let k1 = kv_first_order(z);
        if (v.abs() - 1.0).abs() < 0.5 {
            return Ok(k1);
        }
        // Recurrence: K_{n+1}(z) = K_{n-1}(z) + 2n/z K_n(z)
        let n = v.abs().round() as u32;
        let mut k_prev = k0;
        let mut k_curr = k1;
        for i in 1..n {
            let k_next = k_prev + 2.0 * i as f64 / z * k_curr;
            k_prev = k_curr;
            k_curr = k_next;
        }
        return Ok(k_curr);
    }

    let iv_neg = iv_scalar(-v, z);
    let iv_pos = iv_scalar(v, z);
    Ok(PI / 2.0 * (iv_neg - iv_pos) / sin_vpi)
}

/// K_0(z) via polynomial approximation.
fn kv_zero_order(z: f64) -> f64 {
    if z <= 2.0 {
        let t = z * z / 4.0;
        -(z / 2.0).ln() * j0_core(z)
            + (-0.577_215_664_9
                + t * (0.422_784_335_1
                    + t * (0.230_831_625_8
                        + t * (0.034_873_012_6 + t * (0.002_623_720_2 + t * 0.000_107_587_4)))))
    } else {
        let t = 2.0 / z;
        (-z).exp() / z.sqrt()
            * (1.253_314_137_3
                + t * (-0.078_565_724_0
                    + t * (0.021_009_292_8
                        + t * (-0.010_944_476_7 + t * (0.010_032_511_9 + t * (-0.016_382_637_5))))))
    }
}

/// K_1(z) via polynomial approximation.
fn kv_first_order(z: f64) -> f64 {
    if z <= 2.0 {
        let t = z * z / 4.0;
        z * (z / 2.0).ln() * j1_core(z)
            + 1.0 / z
                * (1.0
                    + t * (0.466_943_581_8
                        + t * (-0.146_547_888_8
                            + t * (-0.016_588_089_5
                                + t * (-0.001_023_145_7 + t * (-0.000_039_827_4))))))
    } else {
        let t = 2.0 / z;
        (-z).exp() / z.sqrt()
            * (1.253_314_137_3
                + t * (0.235_697_172_0
                    + t * (-0.063_027_878_2
                        + t * (0.032_819_357_6
                            + t * (-0.029_752_279_3 + t * (0.048_710_583_5))))))
    }
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
        return (PI / (PI * x).sin()).ln() - lgamma(1.0 - x);
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
/// k_n(z) = √(2/(πz)) K_{n+1/2}(z)
///
/// Computed via recurrence from:
///   k_0(z) = exp(-z)/z,  k_1(z) = exp(-z)/z * (1 + 1/z)
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

/// k_0(z) = exp(-z)/z, k_1(z) = exp(-z)/z * (1 + 1/z)
/// Recurrence: k_{k+1}(z) = k_{k-1}(z) + (2k+1)/z * k_k(z)
fn spherical_kn_nonneg(n: u32, x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    let emx = (-x).exp();
    let mut k_prev = emx / x; // k_0
    if n == 0 {
        return k_prev;
    }
    let mut k_curr = emx / x * (1.0 + 1.0 / x); // k_1
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

    if x < 0.0 {
        -ans
    } else {
        ans
    }
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
