#![forbid(unsafe_code)]

//! Elliptic integrals and Lambert W function.
//!
//! Provides:
//! - `ellipk` — Complete elliptic integral of the first kind K(m)
//! - `ellipe` — Complete elliptic integral of the second kind E(m)
//! - `ellipkinc` — Incomplete elliptic integral of the first kind F(φ, m)
//! - `ellipeinc` — Incomplete elliptic integral of the second kind E(φ, m)
//! - `lambertw` — Lambert W function (principal branch W₀)
//! - `exp1` — Exponential integral E₁(z)
//! - `expi` — Exponential integral Ei(x)

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind, SpecialResult,
    SpecialTensor, record_special_trace,
};

pub const ELLIPTIC_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "ellipk",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "m near 0: polynomial approximation",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "m near 1: logarithmic singularity handling",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "general m: arithmetic-geometric mean iteration",
            },
        ],
        notes: "Domain: m in [0, 1). K(m) -> inf as m -> 1.",
    },
    DispatchPlan {
        function: "ellipe",
        steps: &[DispatchStep {
            regime: KernelRegime::Recurrence,
            when: "arithmetic-geometric mean with E accumulator",
        }],
        notes: "Domain: m in [0, 1]. E(0) = π/2, E(1) = 1.",
    },
    DispatchPlan {
        function: "lambertw",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "x near 0: series expansion",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "general x: Halley iteration from initial guess",
            },
        ],
        notes: "Principal branch W₀ for x >= -1/e. W₀(0) = 0, W₀(e) = 1.",
    },
];

// ══════════════════════════════════════════════════════════════════════
// Complete Elliptic Integrals (AGM method)
// ══════════════════════════════════════════════════════════════════════

/// Complete elliptic integral of the first kind K(m).
///
/// K(m) = ∫₀^{π/2} dθ / sqrt(1 - m sin²θ)
///
/// Uses the arithmetic-geometric mean (AGM) iteration.
/// Domain: m in [0, 1).
pub fn ellipk(m_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("ellipk", m_tensor, mode, |m| ellipk_scalar(m, mode))
}

/// Complete elliptic integral of the second kind E(m).
///
/// E(m) = ∫₀^{π/2} sqrt(1 - m sin²θ) dθ
///
/// Uses the AGM method with E accumulator.
/// Domain: m in [0, 1].
pub fn ellipe(m_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("ellipe", m_tensor, mode, |m| ellipe_scalar(m, mode))
}

/// Incomplete elliptic integral of the first kind F(φ, m).
///
/// F(φ, m) = ∫₀^φ dθ / sqrt(1 - m sin²θ)
///
/// Uses Carlson's RF form via Gauss transformation.
pub fn ellipkinc(
    phi_tensor: &SpecialTensor,
    m_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("ellipkinc", phi_tensor, m_tensor, mode, |phi, m| {
        ellipkinc_scalar(phi, m, mode)
    })
}

/// Incomplete elliptic integral of the second kind E(φ, m).
///
/// E(φ, m) = ∫₀^φ sqrt(1 - m sin²θ) dθ
pub fn ellipeinc(
    phi_tensor: &SpecialTensor,
    m_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("ellipeinc", phi_tensor, m_tensor, mode, |phi, m| {
        ellipeinc_scalar(phi, m, mode)
    })
}

// ══════════════════════════════════════════════════════════════════════
// Lambert W Function
// ══════════════════════════════════════════════════════════════════════

/// Lambert W function, principal branch W₀(x).
///
/// Solves w * exp(w) = x for w.
/// Domain: x >= -1/e.
pub fn lambertw(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("lambertw", x_tensor, mode, |x| lambertw_scalar(x, mode))
}

// ══════════════════════════════════════════════════════════════════════
// Exponential Integrals
// ══════════════════════════════════════════════════════════════════════

/// Exponential integral E₁(z) = ∫₁^∞ exp(-zt)/t dt for z > 0.
pub fn exp1(z_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("exp1", z_tensor, mode, |z| exp1_scalar(z, mode))
}

/// Exponential integral Ei(x) = -PV∫_{-x}^∞ exp(-t)/t dt for x > 0.
pub fn expi(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("expi", x_tensor, mode, |x| expi_scalar(x, mode))
}

/// Generalized exponential integral E_n(x) = ∫₁^∞ exp(-xt)/t^n dt.
///
/// Matches `scipy.special.expn(n, x)`.
///
/// # Arguments
/// * `n` - Order (non-negative integer)
/// * `x` - Argument (must be > 0 for n=0,1; can be 0 for n >= 2)
pub fn expn_scalar(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN;
    }

    // Special cases
    if n == 0 {
        if x == 0.0 {
            return f64::INFINITY;
        }
        return (-x).exp() / x;
    }

    if x == 0.0 {
        if n == 1 {
            return f64::INFINITY;
        }
        // E_n(0) = 1/(n-1) for n >= 2
        return 1.0 / (n - 1) as f64;
    }

    if x == f64::INFINITY {
        return 0.0;
    }

    // Use E_1 and recurrence relation: n * E_{n+1}(x) = e^{-x} - x * E_n(x)
    // Rearranged: E_n(x) computed from E_1(x) using upward recurrence
    let e1 = exp1_scalar(x, fsci_runtime::RuntimeMode::Strict).unwrap_or(f64::NAN);
    if n == 1 {
        return e1;
    }

    // Upward recurrence from E_1 to E_n
    let exp_neg_x = (-x).exp();
    let mut e_prev = e1;
    for k in 1..n {
        // E_{k+1}(x) = (e^{-x} - x * E_k(x)) / k
        let e_next = (exp_neg_x - x * e_prev) / k as f64;
        e_prev = e_next;
    }
    e_prev
}

// ══════════════════════════════════════════════════════════════════════
// Scalar Kernels
// ══════════════════════════════════════════════════════════════════════

fn ellipk_scalar(m: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if m.is_nan() {
        return Ok(f64::NAN);
    }
    if !(0.0..=1.0).contains(&m) {
        return domain_error("ellipk", mode, "m must be in [0, 1)");
    }
    if m >= 1.0 {
        return Ok(f64::INFINITY);
    }
    if m == 0.0 {
        return Ok(PI / 2.0);
    }

    // AGM iteration: K(m) = π / (2 * agm(1, sqrt(1-m)))
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    for _ in 0..50 {
        let a_new = 0.5 * (a + b);
        let b_new = (a * b).sqrt();
        if (a_new - b_new).abs() < 1.0e-15 * a_new {
            return Ok(PI / (2.0 * a_new));
        }
        a = a_new;
        b = b_new;
    }
    Ok(PI / (2.0 * a))
}

fn ellipe_scalar(m: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if m.is_nan() {
        return Ok(f64::NAN);
    }
    if !(0.0..=1.0).contains(&m) {
        return domain_error("ellipe", mode, "m must be in [0, 1]");
    }
    if m == 0.0 {
        return Ok(PI / 2.0);
    }
    if m == 1.0 {
        return Ok(1.0);
    }

    // AGM with E accumulator
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    let mut c2_sum = m; // sum of c_n^2 * 2^n
    let mut power = 1.0; // 2^n

    for _ in 0..50 {
        let a_new = 0.5 * (a + b);
        let b_new = (a * b).sqrt();
        let c = 0.5 * (a - b);
        power *= 2.0;
        c2_sum += c * c * power;
        a = a_new;
        b = b_new;
        if c.abs() < 1.0e-15 {
            break;
        }
    }

    // E(m) = K(m) * (1 - sum(c_n^2 * 2^n) / 2)
    // More precisely: E = (π/(2*agm)) * (1 - sum/2)
    let k = PI / (2.0 * a);
    Ok(k * (1.0 - c2_sum / 2.0))
}

fn ellipkinc_scalar(phi: f64, m: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if phi.is_nan() || m.is_nan() {
        return Ok(f64::NAN);
    }
    if !(0.0..=1.0).contains(&m) {
        return domain_error("ellipkinc", mode, "m must be in [0, 1]");
    }
    if phi == 0.0 {
        return Ok(0.0);
    }
    if (phi - PI / 2.0).abs() < 1e-15 {
        return ellipk_scalar(m, mode);
    }

    // Numerical integration via Gauss-Legendre 15-point quadrature
    let result = gauss_legendre_elliptic_f(phi, m);
    Ok(result)
}

fn ellipeinc_scalar(phi: f64, m: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if phi.is_nan() || m.is_nan() {
        return Ok(f64::NAN);
    }
    if !(0.0..=1.0).contains(&m) {
        return domain_error("ellipeinc", mode, "m must be in [0, 1]");
    }
    if phi == 0.0 {
        return Ok(0.0);
    }
    if (phi - PI / 2.0).abs() < 1e-15 {
        return ellipe_scalar(m, mode);
    }

    // Numerical integration via Gauss-Legendre 15-point quadrature
    let result = gauss_legendre_elliptic_e(phi, m);
    Ok(result)
}

fn lambertw_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    let min_x = -1.0 / std::f64::consts::E;
    if x < min_x - 1.0e-12 {
        return domain_error("lambertw", mode, "x must be >= -1/e for principal branch");
    }
    if (x - min_x).abs() < 1.0e-12 {
        return Ok(-1.0); // W₀(-1/e) = -1
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if (x - std::f64::consts::E).abs() < f64::EPSILON {
        return Ok(1.0);
    }

    // Initial guess
    let mut w = if x < 0.0 {
        // For x in (-1/e, 0): W is in (-1, 0). Use a quadratic approximation near -1/e.
        let p = (2.0 * (std::f64::consts::E * x + 1.0)).sqrt();
        -1.0 + p - p * p / 3.0
    } else if x < 0.5 {
        // Near 0: W(x) ≈ x - x²
        x * (1.0 - x)
    } else if x <= std::f64::consts::E {
        // Moderate x: use log-based estimate
        let lx = (1.0 + x).ln();
        lx * (1.0 - lx / (2.0 + lx))
    } else {
        // Large x: W(x) ≈ ln(x) - ln(ln(x))
        let lx = x.ln();
        lx - lx.ln()
    };

    // Halley's iteration: w_{n+1} = w_n - (w*e^w - x) / (e^w*(w+1) - (w+2)*(w*e^w - x)/(2w+2))
    for _ in 0..50 {
        let ew = w.exp();
        let wew = w * ew;
        let f = wew - x;
        if f.abs() < 1.0e-15 * (1.0 + x.abs()) {
            return Ok(w);
        }
        let denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0);
        if denom.abs() < 1.0e-30 {
            break;
        }
        w -= f / denom;
    }
    Ok(w)
}

fn exp1_scalar(z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if z.is_nan() {
        return Ok(f64::NAN);
    }
    if z <= 0.0 {
        return domain_error("exp1", mode, "z must be > 0 for real E1");
    }
    if z == f64::INFINITY {
        return Ok(0.0);
    }

    if z <= 2.0 {
        // Series: E1(z) = -γ - ln(z) + Σ_{n=1}^∞ (-1)^{n+1} * z^n / (n * n!)
        let euler_gamma = 0.577_215_664_901_532_9;
        let mut sum = -euler_gamma - z.ln();
        let mut term = 1.0_f64;
        for n in 1..300 {
            term *= z / n as f64;
            let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
            let contribution = sign * term / n as f64;
            sum += contribution;
            if contribution.abs() < 1.0e-15 * sum.abs() {
                break;
            }
        }
        Ok(sum)
    } else {
        // Continued fraction (Lentz's method):
        // E1(z) = exp(-z) * (1/(z + 1/(1 + 1/(z + 2/(1 + 2/(z + ...))))))
        // Using the form: E1(z) = exp(-z) * CF where CF = 1/(z+) 1/(1+) 1/(z+) 2/(1+) 2/(z+) ...
        // Rewritten as standard CF: b_0=0, a_1=1, b_1=z, then alternating
        // a_{2k}=k, b_{2k}=1, a_{2k+1}=k, b_{2k+1}=z
        let mut d = 1.0 / z;
        let mut c = 1.0 / 1.0e-30_f64;
        let mut h = d;

        for n in 1..100 {
            let a_n = ((n + 1) / 2) as f64;
            let b_n = if n % 2 == 1 { 1.0 } else { z };
            d = 1.0 / (b_n + a_n * d);
            c = b_n + a_n / c;
            let delta = c * d;
            h *= delta;
            if (delta - 1.0).abs() < 1.0e-15 {
                break;
            }
        }
        Ok((-z).exp() * h)
    }
}

fn expi_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x == f64::INFINITY {
        return Ok(f64::INFINITY);
    }
    if x < 0.0 {
        // Ei(x) = -E1(-x) for x < 0
        return exp1_scalar(-x, mode).map(|v| -v);
    }

    // Series: Ei(x) = γ + ln(x) + sum_{n=1}^∞ x^n / (n * n!)
    let euler_gamma = 0.577_215_664_901_532_9;
    let mut sum = euler_gamma + x.ln();
    let mut term = x;
    sum += term;
    for n in 2..100 {
        term *= x / n as f64;
        let contribution = term / n as f64;
        sum += contribution;
        if contribution.abs() < 1.0e-15 * sum.abs() {
            break;
        }
    }
    Ok(sum)
}

// ══════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════

fn map_real<F>(
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
            .map(&kernel)
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(_) | SpecialTensor::ComplexVec(_) => {
            record_special_trace(
                function,
                mode,
                "not_implemented",
                "input=complex",
                "fail_closed",
                "complex-valued path pending",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "complex-valued path pending",
            })
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
                .map(|(left_value, right_value)| kernel(left_value, right_value))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::ComplexScalar(_), _)
        | (SpecialTensor::ComplexVec(_), _)
        | (_, SpecialTensor::ComplexScalar(_))
        | (_, SpecialTensor::ComplexVec(_)) => {
            record_special_trace(
                function,
                mode,
                "not_implemented",
                "input=complex",
                "fail_closed",
                "complex-valued path pending",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "complex-valued path pending",
            })
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

fn domain_error(
    function: &'static str,
    mode: RuntimeMode,
    detail: &'static str,
) -> Result<f64, SpecialError> {
    match mode {
        RuntimeMode::Strict => Ok(f64::NAN),
        RuntimeMode::Hardened => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail,
        }),
    }
}

/// 15-point Gauss-Legendre quadrature for incomplete elliptic integral F(φ, m).
fn gauss_legendre_elliptic_f(phi: f64, m: f64) -> f64 {
    // Gauss-Legendre nodes and weights for [-1, 1], n=15
    const NODES: [f64; 8] = [
        0.987_992_518_020_485_4,
        0.937_273_392_400_706,
        0.848_206_583_410_427_2,
        0.724_417_731_360_170_1,
        0.570_972_172_608_538_8,
        0.394_151_347_077_563_4,
        0.201_194_093_997_435,
        0.0,
    ];
    const WEIGHTS: [f64; 8] = [
        0.030_753_241_996_117_3,
        0.070_366_047_488_108_1,
        0.107_159_220_467_171_9,
        0.139_570_677_926_154_1,
        0.166_269_205_816_994,
        0.186_161_000_015_562_2,
        0.198_431_485_327_111_6,
        0.202_578_241_925_561_3,
    ];

    let half_phi = 0.5 * phi;
    let mut sum = 0.0;
    for i in 0..8 {
        let t_pos = half_phi * (1.0 + NODES[i]);
        let t_neg = half_phi * (1.0 - NODES[i]);
        let f_pos = 1.0 / (1.0 - m * t_pos.sin().powi(2)).sqrt();
        let f_neg = 1.0 / (1.0 - m * t_neg.sin().powi(2)).sqrt();
        if i == 7 {
            // Center node (weight only once, symmetric around 0 maps to center)
            sum += WEIGHTS[i] * f_pos;
        } else {
            sum += WEIGHTS[i] * (f_pos + f_neg);
        }
    }
    half_phi * sum
}

/// 15-point Gauss-Legendre quadrature for incomplete elliptic integral E(φ, m).
fn gauss_legendre_elliptic_e(phi: f64, m: f64) -> f64 {
    const NODES: [f64; 8] = [
        0.987_992_518_020_485_4,
        0.937_273_392_400_706,
        0.848_206_583_410_427_2,
        0.724_417_731_360_170_1,
        0.570_972_172_608_538_8,
        0.394_151_347_077_563_4,
        0.201_194_093_997_435,
        0.0,
    ];
    const WEIGHTS: [f64; 8] = [
        0.030_753_241_996_117_3,
        0.070_366_047_488_108_1,
        0.107_159_220_467_171_9,
        0.139_570_677_926_154_1,
        0.166_269_205_816_994,
        0.186_161_000_015_562_2,
        0.198_431_485_327_111_6,
        0.202_578_241_925_561_3,
    ];

    let half_phi = 0.5 * phi;
    let mut sum = 0.0;
    for i in 0..8 {
        let t_pos = half_phi * (1.0 + NODES[i]);
        let t_neg = half_phi * (1.0 - NODES[i]);
        let f_pos = (1.0 - m * t_pos.sin().powi(2)).sqrt();
        let f_neg = (1.0 - m * t_neg.sin().powi(2)).sqrt();
        if i == 7 {
            sum += WEIGHTS[i] * f_pos;
        } else {
            sum += WEIGHTS[i] * (f_pos + f_neg);
        }
    }
    half_phi * sum
}

// ══════════════════════════════════════════════════════════════════════
// Jacobi elliptic functions
// ══════════════════════════════════════════════════════════════════════

/// Jacobi elliptic functions sn(u, m), cn(u, m), dn(u, m), and amplitude ph(u, m).
///
/// Computed via the arithmetic-geometric mean (descending Landen transformation).
///
/// Parameters:
/// - `u`: argument (real)
/// - `m`: parameter (0 <= m <= 1)
///
/// Returns (sn, cn, dn, ph) where:
/// - sn(u, m) = sin(am(u, m))
/// - cn(u, m) = cos(am(u, m))
/// - dn(u, m) = √(1 - m·sn²)
/// - ph(u, m) = am(u, m) (the Jacobi amplitude)
///
/// Identities: sn² + cn² = 1, dn² + m·sn² = 1
pub fn ellipj(u: f64, m: f64) -> (f64, f64, f64, f64) {
    if m.is_nan() || u.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }

    // Special case: m = 0 => sn = sin(u), cn = cos(u), dn = 1
    if m.abs() < 1e-15 {
        return (u.sin(), u.cos(), 1.0, u);
    }

    // Special case: m = 1 => sn = tanh(u), cn = dn = sech(u)
    if (m - 1.0).abs() < 1e-15 {
        let sn = u.tanh();
        let cn = 1.0 / u.cosh();
        return (
            sn,
            cn,
            cn,
            2.0 * u.exp().atan() - std::f64::consts::FRAC_PI_2,
        );
    }

    // AGM-based computation via descending Landen transformation
    const MAX_ITER: usize = 20;
    let mut a = [0.0; MAX_ITER + 1];
    let mut b = [0.0; MAX_ITER + 1];
    let mut c = [0.0; MAX_ITER + 1];

    a[0] = 1.0;
    b[0] = (1.0 - m).sqrt();
    c[0] = m.sqrt();

    let mut n = 0;
    while c[n].abs() > 1e-16 && n < MAX_ITER {
        let a_new = (a[n] + b[n]) / 2.0;
        let b_new = (a[n] * b[n]).sqrt();
        let c_new = (a[n] - b[n]) / 2.0;
        n += 1;
        a[n] = a_new;
        b[n] = b_new;
        c[n] = c_new;
    }

    // Compute amplitude by back-substitution
    let mut phi = (1u64 << n) as f64 * a[n] * u;
    for k in (1..=n).rev() {
        phi = (phi + (c[k] / a[k] * phi.sin()).clamp(-1.0, 1.0).asin()) / 2.0;
    }

    let sn = phi.sin();
    let cn = phi.cos();
    let dn = (1.0 - m * sn * sn).sqrt();

    (sn, cn, dn, phi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: got {actual}, expected {expected} (diff={})",
            (actual - expected).abs()
        );
    }

    fn eval_scalar(result: SpecialResult) -> f64 {
        match result.expect("should succeed") {
            SpecialTensor::RealScalar(v) => v,
            other => panic!("expected RealScalar, got {other:?}"),
        }
    }

    // ── Complete elliptic integrals ───────────────────────────────

    #[test]
    fn ellipk_at_zero() {
        let m = SpecialTensor::RealScalar(0.0);
        let result = eval_scalar(ellipk(&m, RuntimeMode::Strict));
        assert_close(result, PI / 2.0, 1e-12, "K(0) = π/2");
    }

    #[test]
    fn ellipk_at_half() {
        // K(0.5) ≈ 1.854_074_677
        let m = SpecialTensor::RealScalar(0.5);
        let result = eval_scalar(ellipk(&m, RuntimeMode::Strict));
        assert_close(result, 1.854_074_677, 1e-8, "K(0.5)");
    }

    #[test]
    fn ellipk_at_one_is_infinity() {
        let m = SpecialTensor::RealScalar(1.0);
        let result = eval_scalar(ellipk(&m, RuntimeMode::Strict));
        assert!(result.is_infinite(), "K(1) should be infinite");
    }

    #[test]
    fn ellipe_at_zero() {
        let m = SpecialTensor::RealScalar(0.0);
        let result = eval_scalar(ellipe(&m, RuntimeMode::Strict));
        assert_close(result, PI / 2.0, 1e-12, "E(0) = π/2");
    }

    #[test]
    fn ellipe_at_one() {
        let m = SpecialTensor::RealScalar(1.0);
        let result = eval_scalar(ellipe(&m, RuntimeMode::Strict));
        assert_close(result, 1.0, 1e-12, "E(1) = 1");
    }

    #[test]
    fn ellipe_at_half() {
        // E(0.5) ≈ 1.350_643_881
        let m = SpecialTensor::RealScalar(0.5);
        let result = eval_scalar(ellipe(&m, RuntimeMode::Strict));
        assert_close(result, 1.350_643_881, 1e-8, "E(0.5)");
    }

    #[test]
    fn ellipk_ellipe_legendre_relation() {
        // Legendre's relation: K(m)*E(1-m) + E(m)*K(1-m) - K(m)*K(1-m) = π/2
        let m = 0.3;
        let m1 = 1.0 - m;
        let km = eval_scalar(ellipk(&SpecialTensor::RealScalar(m), RuntimeMode::Strict));
        let em = eval_scalar(ellipe(&SpecialTensor::RealScalar(m), RuntimeMode::Strict));
        let km1 = eval_scalar(ellipk(&SpecialTensor::RealScalar(m1), RuntimeMode::Strict));
        let em1 = eval_scalar(ellipe(&SpecialTensor::RealScalar(m1), RuntimeMode::Strict));
        let legendre = km * em1 + em * km1 - km * km1;
        assert_close(legendre, PI / 2.0, 1e-6, "Legendre relation");
    }

    // ── Incomplete elliptic integrals ─────────────────────────────

    #[test]
    fn ellipkinc_at_pi_half_equals_complete() {
        let phi = SpecialTensor::RealScalar(PI / 2.0);
        let m = SpecialTensor::RealScalar(0.5);
        let incomplete = eval_scalar(ellipkinc(&phi, &m, RuntimeMode::Strict));
        let complete = eval_scalar(ellipk(&m, RuntimeMode::Strict));
        assert_close(incomplete, complete, 1e-6, "F(π/2, m) = K(m)");
    }

    #[test]
    fn ellipeinc_at_pi_half_equals_complete() {
        let phi = SpecialTensor::RealScalar(PI / 2.0);
        let m = SpecialTensor::RealScalar(0.5);
        let incomplete = eval_scalar(ellipeinc(&phi, &m, RuntimeMode::Strict));
        let complete = eval_scalar(ellipe(&m, RuntimeMode::Strict));
        assert_close(incomplete, complete, 1e-6, "E(π/2, m) = E(m)");
    }

    #[test]
    fn ellipkinc_at_zero_phi() {
        let phi = SpecialTensor::RealScalar(0.0);
        let m = SpecialTensor::RealScalar(0.5);
        let result = eval_scalar(ellipkinc(&phi, &m, RuntimeMode::Strict));
        assert_close(result, 0.0, 1e-12, "F(0, m) = 0");
    }

    #[test]
    fn ellipkinc_broadcasts_scalar_phi_over_m_vector() {
        let phi = SpecialTensor::RealScalar(PI / 2.0);
        let m = SpecialTensor::RealVec(vec![0.0, 0.5]);
        let result =
            ellipkinc(&phi, &m, RuntimeMode::Strict).expect("scalar phi should broadcast over m");
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values.len(), 2);
                assert_close(values[0], PI / 2.0, 1e-12, "F(π/2, 0) = K(0)");
                assert_close(values[1], 1.854_074_677, 1e-6, "F(π/2, 0.5) = K(0.5)");
            }
            other => panic!("expected RealVec, got {other:?}"),
        }
    }

    #[test]
    fn ellipkinc_broadcasts_vector_phi_over_scalar_m() {
        let phi = SpecialTensor::RealVec(vec![0.0, PI / 2.0]);
        let m = SpecialTensor::RealScalar(0.5);
        let result = ellipkinc(&phi, &m, RuntimeMode::Strict)
            .expect("vector phi should broadcast over scalar m");
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values.len(), 2);
                assert_close(values[0], 0.0, 1e-12, "F(0, m) = 0");
                assert_close(values[1], 1.854_074_677, 1e-6, "F(π/2, 0.5) = K(0.5)");
            }
            other => panic!("expected RealVec, got {other:?}"),
        }
    }

    #[test]
    fn ellipeinc_supports_pairwise_vector_inputs() {
        let phi = SpecialTensor::RealVec(vec![0.0, PI / 2.0]);
        let m = SpecialTensor::RealVec(vec![0.0, 0.5]);
        let result = ellipeinc(&phi, &m, RuntimeMode::Strict).expect("pairwise vector ellipeinc");
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values.len(), 2);
                assert_close(values[0], 0.0, 1e-12, "E(0, 0) = 0");
                assert_close(values[1], 1.350_643_881, 1e-6, "E(π/2, 0.5) = E(0.5)");
            }
            other => panic!("expected RealVec, got {other:?}"),
        }
    }

    #[test]
    fn ellipkinc_rejects_mismatched_vector_lengths() {
        let phi = SpecialTensor::RealVec(vec![0.0, PI / 2.0]);
        let m = SpecialTensor::RealVec(vec![0.5]);
        let err = ellipkinc(&phi, &m, RuntimeMode::Strict)
            .expect_err("mismatched vector lengths should error");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
    }

    // ── Lambert W function ────────────────────────────────────────

    #[test]
    fn lambertw_at_zero() {
        let x = SpecialTensor::RealScalar(0.0);
        let result = eval_scalar(lambertw(&x, RuntimeMode::Strict));
        assert_close(result, 0.0, 1e-12, "W(0) = 0");
    }

    #[test]
    fn lambertw_at_e() {
        let x = SpecialTensor::RealScalar(std::f64::consts::E);
        let result = eval_scalar(lambertw(&x, RuntimeMode::Strict));
        assert_close(result, 1.0, 1e-10, "W(e) = 1");
    }

    #[test]
    fn lambertw_at_neg_inv_e() {
        let x = SpecialTensor::RealScalar(-1.0 / std::f64::consts::E);
        let result = eval_scalar(lambertw(&x, RuntimeMode::Strict));
        assert_close(result, -1.0, 1e-10, "W(-1/e) = -1");
    }

    #[test]
    fn lambertw_identity() {
        // W(x) * exp(W(x)) = x
        for &x in &[0.5, 1.0, 2.0, 10.0, 100.0] {
            let w = eval_scalar(lambertw(&SpecialTensor::RealScalar(x), RuntimeMode::Strict));
            let check = w * w.exp();
            assert_close(check, x, 1e-10, &format!("W({x})*exp(W({x})) = {x}"));
        }
    }

    #[test]
    fn lambertw_domain_error_hardened() {
        let x = SpecialTensor::RealScalar(-1.0); // < -1/e
        let result = lambertw(&x, RuntimeMode::Hardened);
        assert!(result.is_err(), "should reject x < -1/e in hardened mode");
    }

    // ── Exponential integrals ─────────────────────────────────────

    #[test]
    fn exp1_known_value() {
        // E1(1) ≈ 0.219_383_934_4
        let z = SpecialTensor::RealScalar(1.0);
        let result = eval_scalar(exp1(&z, RuntimeMode::Strict));
        assert_close(result, 0.219_383_934_4, 1e-8, "E1(1)");
    }

    #[test]
    fn exp1_large_z() {
        // E1(20) ≈ 9.8355e-11 (verified against Wolfram Alpha)
        let z = 20.0;
        let result = eval_scalar(exp1(&SpecialTensor::RealScalar(z), RuntimeMode::Strict));
        // The asymptotic series exp(-z)/z is only approximate; use known reference
        assert_close(result, 9.835_525_290_7e-11, 1e-15, "E1(20)");
    }

    #[test]
    fn expi_known_value() {
        // Ei(1) ≈ 1.895_117_816_4
        let x = SpecialTensor::RealScalar(1.0);
        let result = eval_scalar(expi(&x, RuntimeMode::Strict));
        assert_close(result, 1.895_117_816_4, 1e-6, "Ei(1)");
    }

    #[test]
    fn expi_at_zero_is_neg_inf() {
        let x = SpecialTensor::RealScalar(0.0);
        let result = eval_scalar(expi(&x, RuntimeMode::Strict));
        assert!(
            result.is_infinite() && result.is_sign_negative(),
            "Ei(0) = -inf"
        );
    }

    #[test]
    fn exp1_domain_error_hardened() {
        let z = SpecialTensor::RealScalar(-1.0);
        let result = exp1(&z, RuntimeMode::Hardened);
        assert!(result.is_err(), "should reject z <= 0 in hardened mode");
    }

    // ── Vector inputs ─────────────────────────────────────────────

    #[test]
    fn ellipk_vector_input() {
        let m = SpecialTensor::RealVec(vec![0.0, 0.5]);
        let result = ellipk(&m, RuntimeMode::Strict).expect("should succeed");
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values.len(), 2);
                assert_close(values[0], PI / 2.0, 1e-12, "K(0)");
                assert_close(values[1], 1.854_074_677, 1e-8, "K(0.5)");
            }
            other => panic!("expected RealVec, got {other:?}"),
        }
    }

    #[test]
    fn lambertw_vector_input() {
        let x = SpecialTensor::RealVec(vec![0.0, std::f64::consts::E]);
        let result = lambertw(&x, RuntimeMode::Strict).expect("should succeed");
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values.len(), 2);
                assert_close(values[0], 0.0, 1e-12, "W(0)");
                assert_close(values[1], 1.0, 1e-10, "W(e)");
            }
            other => panic!("expected RealVec, got {other:?}"),
        }
    }

    // ── Jacobi elliptic functions ────────────────────────────────────

    #[test]
    fn ellipj_at_zero() {
        // sn(0, m) = 0, cn(0, m) = 1, dn(0, m) = 1
        let (sn, cn, dn, ph) = ellipj(0.0, 0.5);
        assert_close(sn, 0.0, 1e-12, "sn(0) = 0");
        assert_close(cn, 1.0, 1e-12, "cn(0) = 1");
        assert_close(dn, 1.0, 1e-12, "dn(0) = 1");
        assert_close(ph, 0.0, 1e-12, "am(0) = 0");
    }

    #[test]
    fn ellipj_m_zero_is_trig() {
        // m=0: sn = sin, cn = cos, dn = 1
        let u = 1.0;
        let (sn, cn, dn, _) = ellipj(u, 0.0);
        assert_close(sn, u.sin(), 1e-12, "sn(u,0) = sin(u)");
        assert_close(cn, u.cos(), 1e-12, "cn(u,0) = cos(u)");
        assert_close(dn, 1.0, 1e-12, "dn(u,0) = 1");
    }

    #[test]
    fn ellipj_m_one_is_hyp() {
        // m=1: sn = tanh, cn = dn = sech
        let u = 1.0;
        let (sn, cn, dn, _) = ellipj(u, 1.0);
        assert_close(sn, u.tanh(), 1e-12, "sn(u,1) = tanh(u)");
        assert_close(cn, 1.0 / u.cosh(), 1e-12, "cn(u,1) = sech(u)");
        assert_close(dn, 1.0 / u.cosh(), 1e-12, "dn(u,1) = sech(u)");
    }

    #[test]
    fn ellipj_pythagorean_identity() {
        // sn² + cn² = 1
        for m in [0.1, 0.3, 0.5, 0.7, 0.9] {
            for u in [0.5, 1.0, 2.0, 3.0] {
                let (sn, cn, _, _) = ellipj(u, m);
                let sum = sn * sn + cn * cn;
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "sn²+cn²=1 failed: u={u}, m={m}, sum={sum}"
                );
            }
        }
    }

    #[test]
    fn ellipj_dn_identity() {
        // dn² + m*sn² = 1
        for m in [0.1, 0.3, 0.5, 0.7, 0.9] {
            for u in [0.5, 1.0, 2.0, 3.0] {
                let (sn, _, dn, _) = ellipj(u, m);
                let sum = dn * dn + m * sn * sn;
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "dn²+m·sn²=1 failed: u={u}, m={m}, sum={sum}"
                );
            }
        }
    }

    #[test]
    fn ellipj_nan_passthrough() {
        let (sn, cn, dn, ph) = ellipj(f64::NAN, 0.5);
        assert!(sn.is_nan());
        assert!(cn.is_nan());
        assert!(dn.is_nan());
        assert!(ph.is_nan());
    }

    // ── Generalized exponential integral E_n ──────────────────────────

    #[test]
    fn expn_special_cases() {
        // E_0(x) = exp(-x)/x
        assert_close(expn_scalar(0, 1.0), (-1.0_f64).exp(), 1e-12, "E_0(1)");
        assert_close(expn_scalar(0, 2.0), (-2.0_f64).exp() / 2.0, 1e-12, "E_0(2)");

        // E_1(1) ≈ 0.219_383_934_4 (matches exp1)
        assert_close(expn_scalar(1, 1.0), 0.219_383_934_4, 1e-8, "E_1(1)");

        // E_n(0) = 1/(n-1) for n >= 2
        assert_close(expn_scalar(2, 0.0), 1.0, 1e-12, "E_2(0) = 1");
        assert_close(expn_scalar(3, 0.0), 0.5, 1e-12, "E_3(0) = 0.5");
        assert_close(expn_scalar(4, 0.0), 1.0 / 3.0, 1e-12, "E_4(0) = 1/3");
    }

    #[test]
    fn expn_known_values() {
        // Values verified against SciPy/Wolfram Alpha
        // E_2(1) ≈ 0.148_495_506_8
        assert_close(expn_scalar(2, 1.0), 0.148_495_506_8, 1e-6, "E_2(1)");

        // E_3(1) ≈ 0.109_691_632_2
        assert_close(expn_scalar(3, 1.0), 0.109_691_632_2, 1e-6, "E_3(1)");

        // E_2(0.5) ≈ 0.326_643_070_9
        assert_close(expn_scalar(2, 0.5), 0.326_643_070_9, 1e-6, "E_2(0.5)");
    }

    #[test]
    fn expn_large_x() {
        // For large x, E_n(x) ≈ exp(-x)/x
        let x = 10.0_f64;
        let asymp = (-x).exp() / x;
        // E_1(10) should be close to but slightly larger than asymptotic
        let e1 = expn_scalar(1, x);
        assert!(e1 > 0.9 * asymp && e1 < 1.5 * asymp, "E_1(10) near asymptotic");
    }

    #[test]
    fn expn_recurrence() {
        // Recurrence: E_{n+1}(x) = (exp(-x) - x*E_n(x)) / n for n >= 1
        let x = 2.0;
        for n in 1..5 {
            let en = expn_scalar(n, x);
            let en1 = expn_scalar(n + 1, x);
            let computed = ((-x).exp() - x * en) / n as f64;
            assert_close(en1, computed, 1e-6, &format!("E_{} recurrence", n + 1));
        }
    }
}
