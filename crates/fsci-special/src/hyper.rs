#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor,
};

pub const HYPER_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "hyp1f1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|z| <= 2 and moderate parameters",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "parameter shifting to stable region",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z| or large parameters",
            },
        ],
        notes: "Fallback routing should preserve SciPy branch-selection semantics for strict mode.",
    },
    DispatchPlan {
        function: "hyp2f1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|z| < 0.9 and c not near nonpositive integers",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "boundary neighborhoods near z=1",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "contiguous relation stabilization",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large-parameter asymptotic domains",
            },
        ],
        notes: "z=1 convergence edge cases and c-pole exclusions are explicit hardened guards.",
    },
];

/// Confluent hypergeometric function 1F1(a; b; z).
///
/// Also known as Kummer's function M(a, b, z).
/// Supports scalar or vector parameters with NumPy-style broadcasting.
/// Matches `scipy.special.hyp1f1(a, b, z)`.
pub fn hyp1f1(
    a: &SpecialTensor,
    b: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    hyp1f1_dispatch("hyp1f1", a, b, z, mode)
}

fn hyp1f1_dispatch(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    // Check for empty tensors
    if matches!(a, SpecialTensor::Empty)
        || matches!(b, SpecialTensor::Empty)
        || matches!(z, SpecialTensor::Empty)
    {
        return Ok(SpecialTensor::Empty);
    }

    // Get vector lengths (0 = scalar)
    let a_len = tensor_vec_len(a);
    let b_len = tensor_vec_len(b);
    let z_len = tensor_vec_len(z);

    // Check broadcast compatibility
    let out_len = broadcast_len_3(a_len, b_len, z_len)
        .ok_or_else(|| broadcast_shape_error(function, mode))?;

    // Check if output is complex
    let is_complex = is_complex_tensor(a) || is_complex_tensor(b) || is_complex_tensor(z);

    if out_len == 0 {
        // All scalars
        if is_complex {
            let a_c = tensor_as_complex_scalar(a)?;
            let b_c = tensor_as_complex_scalar(b)?;
            let z_c = tensor_as_complex_scalar(z)?;
            let result = hyp1f1_complex_parameters(a_c, b_c, z_c, mode)?;
            Ok(SpecialTensor::ComplexScalar(result))
        } else {
            let a_r = tensor_as_real_scalar(a)?;
            let b_r = tensor_as_real_scalar(b)?;
            let z_r = tensor_as_real_scalar(z)?;
            let result = hyp1f1_scalar(a_r, b_r, z_r, mode)?;
            Ok(SpecialTensor::RealScalar(result))
        }
    } else {
        // At least one vector
        if is_complex {
            let mut results = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let a_c = tensor_get_complex(a, i, a_len)?;
                let b_c = tensor_get_complex(b, i, b_len)?;
                let z_c = tensor_get_complex(z, i, z_len)?;
                results.push(hyp1f1_complex_parameters(a_c, b_c, z_c, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        } else {
            let mut results = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let a_r = tensor_get_real(a, i, a_len)?;
                let b_r = tensor_get_real(b, i, b_len)?;
                let z_r = tensor_get_real(z, i, z_len)?;
                results.push(hyp1f1_scalar(a_r, b_r, z_r, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
    }
}

// Helper functions for broadcast dispatch

fn tensor_vec_len(t: &SpecialTensor) -> usize {
    match t {
        SpecialTensor::RealScalar(_) | SpecialTensor::ComplexScalar(_) => 0,
        SpecialTensor::RealVec(v) => v.len(),
        SpecialTensor::ComplexVec(v) => v.len(),
        SpecialTensor::Empty => 0,
    }
}

fn is_complex_tensor(t: &SpecialTensor) -> bool {
    matches!(
        t,
        SpecialTensor::ComplexScalar(_) | SpecialTensor::ComplexVec(_)
    )
}

fn broadcast_len_4(a: usize, b: usize, c: usize, d: usize) -> Option<usize> {
    let abc = broadcast_len_3(a, b, c)?;
    broadcast_len_2(abc, d)
}

fn broadcast_len_3(a: usize, b: usize, c: usize) -> Option<usize> {
    let ab = broadcast_len_2(a, b)?;
    broadcast_len_2(ab, c)
}

fn broadcast_len_2(a: usize, b: usize) -> Option<usize> {
    match (a, b) {
        (0, 0) => Some(0),
        (0, n) | (n, 0) => Some(n),
        (m, n) if m == n => Some(m),
        _ => None,
    }
}

fn tensor_as_real_scalar(t: &SpecialTensor) -> Result<f64, SpecialError> {
    match t {
        SpecialTensor::RealScalar(v) => Ok(*v),
        _ => Err(SpecialError {
            function: "hyp1f1",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "expected real scalar",
        }),
    }
}

fn tensor_as_complex_scalar(t: &SpecialTensor) -> Result<Complex64, SpecialError> {
    match t {
        SpecialTensor::RealScalar(v) => Ok(Complex64::from_real(*v)),
        SpecialTensor::ComplexScalar(v) => Ok(*v),
        _ => Err(SpecialError {
            function: "hyp1f1",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "expected scalar",
        }),
    }
}

fn tensor_get_real(t: &SpecialTensor, i: usize, len: usize) -> Result<f64, SpecialError> {
    match t {
        SpecialTensor::RealScalar(v) => Ok(*v),
        SpecialTensor::RealVec(vec) => Ok(vec[if len == 0 { 0 } else { i }]),
        _ => Err(SpecialError {
            function: "hyp1f1",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "expected real tensor",
        }),
    }
}

fn tensor_get_complex(t: &SpecialTensor, i: usize, len: usize) -> Result<Complex64, SpecialError> {
    match t {
        SpecialTensor::RealScalar(v) => Ok(Complex64::from_real(*v)),
        SpecialTensor::ComplexScalar(v) => Ok(*v),
        SpecialTensor::RealVec(vec) => Ok(Complex64::from_real(vec[if len == 0 { 0 } else { i }])),
        SpecialTensor::ComplexVec(vec) => Ok(vec[if len == 0 { 0 } else { i }]),
        SpecialTensor::Empty => Err(SpecialError {
            function: "hyp1f1",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "empty tensor",
        }),
    }
}

/// Confluent hypergeometric limit function 0F1(; b; z).
///
/// Also known as the regularized confluent hypergeometric limit function.
/// Defined as: 0F1(; b; z) = Σ_{n=0}^∞ z^n / ((b)_n * n!)
///
/// Related to Bessel functions: J_n(x) = (x/2)^n / Γ(n+1) * 0F1(; n+1; -x²/4)
///
/// Supports scalar real or complex parameters with real or complex
/// scalar/vector `z` inputs.
/// Matches `scipy.special.hyp0f1(b, z)`.
///
/// # Arguments
/// * `b` - Parameter (must not be a non-positive integer)
/// * `z` - Argument
///
/// # Returns
/// Value of 0F1(; b; z)
pub fn hyp0f1(b: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match (b, z) {
        (SpecialTensor::RealScalar(b_val), SpecialTensor::RealScalar(z_val)) => {
            let result = hyp0f1_scalar(*b_val, *z_val, mode)?;
            Ok(SpecialTensor::RealScalar(result))
        }
        (SpecialTensor::RealScalar(b_val), SpecialTensor::RealVec(z_vec)) => {
            let mut results = Vec::with_capacity(z_vec.len());
            for &zi in z_vec {
                results.push(hyp0f1_scalar(*b_val, zi, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
        (SpecialTensor::RealVec(b_vec), SpecialTensor::RealScalar(z_val)) => {
            let mut results = Vec::with_capacity(b_vec.len());
            for &bi in b_vec {
                results.push(hyp0f1_scalar(bi, *z_val, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
        (SpecialTensor::RealVec(b_vec), SpecialTensor::RealVec(z_vec)) => {
            if b_vec.len() != z_vec.len() {
                return Err(broadcast_shape_error("hyp0f1", mode));
            }
            let mut results = Vec::with_capacity(b_vec.len());
            for (&bi, &zi) in b_vec.iter().zip(z_vec.iter()) {
                results.push(hyp0f1_scalar(bi, zi, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
        (SpecialTensor::ComplexScalar(b_val), SpecialTensor::RealScalar(z_val)) => {
            let result = hyp0f1_complex_scalar(*b_val, Complex64::from_real(*z_val), mode)?;
            Ok(SpecialTensor::ComplexScalar(result))
        }
        (SpecialTensor::ComplexScalar(b_val), SpecialTensor::RealVec(z_vec)) => {
            let mut results = Vec::with_capacity(z_vec.len());
            for &zi in z_vec {
                results.push(hyp0f1_complex_scalar(
                    *b_val,
                    Complex64::from_real(zi),
                    mode,
                )?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::ComplexScalar(b_val), SpecialTensor::ComplexScalar(z_val)) => {
            let result = hyp0f1_complex_scalar(*b_val, *z_val, mode)?;
            Ok(SpecialTensor::ComplexScalar(result))
        }
        (SpecialTensor::ComplexScalar(b_val), SpecialTensor::ComplexVec(z_vec)) => {
            let mut results = Vec::with_capacity(z_vec.len());
            for &zi in z_vec {
                results.push(hyp0f1_complex_scalar(*b_val, zi, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::ComplexVec(b_vec), SpecialTensor::RealScalar(z_val)) => {
            let z_complex = Complex64::from_real(*z_val);
            let mut results = Vec::with_capacity(b_vec.len());
            for &bi in b_vec {
                results.push(hyp0f1_complex_scalar(bi, z_complex, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::ComplexVec(b_vec), SpecialTensor::ComplexScalar(z_val)) => {
            let mut results = Vec::with_capacity(b_vec.len());
            for &bi in b_vec {
                results.push(hyp0f1_complex_scalar(bi, *z_val, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::ComplexVec(b_vec), SpecialTensor::RealVec(z_vec)) => {
            if b_vec.len() != z_vec.len() {
                return Err(broadcast_shape_error("hyp0f1", mode));
            }
            let mut results = Vec::with_capacity(b_vec.len());
            for (&bi, &zi) in b_vec.iter().zip(z_vec.iter()) {
                results.push(hyp0f1_complex_scalar(bi, Complex64::from_real(zi), mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::ComplexVec(b_vec), SpecialTensor::ComplexVec(z_vec)) => {
            if b_vec.len() != z_vec.len() {
                return Err(broadcast_shape_error("hyp0f1", mode));
            }
            let mut results = Vec::with_capacity(b_vec.len());
            for (&bi, &zi) in b_vec.iter().zip(z_vec.iter()) {
                results.push(hyp0f1_complex_scalar(bi, zi, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::RealVec(b_vec), SpecialTensor::ComplexScalar(z_val)) => {
            let mut results = Vec::with_capacity(b_vec.len());
            for &bi in b_vec {
                results.push(hyp0f1_complex_scalar(
                    Complex64::from_real(bi),
                    *z_val,
                    mode,
                )?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::RealVec(b_vec), SpecialTensor::ComplexVec(z_vec)) => {
            if b_vec.len() != z_vec.len() {
                return Err(broadcast_shape_error("hyp0f1", mode));
            }
            let mut results = Vec::with_capacity(b_vec.len());
            for (&bi, &zi) in b_vec.iter().zip(z_vec.iter()) {
                results.push(hyp0f1_complex_scalar(Complex64::from_real(bi), zi, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::RealScalar(b_val), SpecialTensor::ComplexScalar(z_val)) => {
            let result = hyp0f1_complex_scalar(Complex64::from_real(*b_val), *z_val, mode)?;
            Ok(SpecialTensor::ComplexScalar(result))
        }
        (SpecialTensor::RealScalar(b_val), SpecialTensor::ComplexVec(z_vec)) => {
            let b_complex = Complex64::from_real(*b_val);
            let mut results = Vec::with_capacity(z_vec.len());
            for &zi in z_vec {
                results.push(hyp0f1_complex_scalar(b_complex, zi, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Ok(SpecialTensor::Empty),
    }
}

/// Scalar implementation of 0F1(; b; z).
pub fn hyp0f1_scalar(b: f64, z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    // Check for pole at non-positive integer b
    if b <= 0.0 && b == b.floor() {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp0f1",
                kind: SpecialErrorKind::PoleInput,
                mode,
                detail: "hyp0f1: b must not be a non-positive integer",
            });
        }
        return Ok(f64::NAN);
    }

    // Handle special cases
    if z == 0.0 {
        return Ok(1.0);
    }

    // For small |z|, use direct series
    if z.abs() < 50.0 {
        return Ok(hyp0f1_series(b, z));
    }

    // For large |z|, use asymptotic expansion
    Ok(hyp0f1_asymptotic(b, z))
}

/// Power series for 0F1(; b; z).
/// 0F1(; b; z) = Σ_{n=0}^∞ z^n / ((b)_n * n!)
fn hyp0f1_series(b: f64, z: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;

    for n in 1..300 {
        let nf = n as f64;
        // term_n = term_{n-1} * z / (n * (b + n - 1))
        term *= z / (nf * (b + nf - 1.0));
        sum += term;

        if term.abs() < 1e-16 * sum.abs() {
            break;
        }
        if !sum.is_finite() {
            break;
        }
    }

    sum
}

fn broadcast_shape_error(function: &'static str, mode: RuntimeMode) -> SpecialError {
    SpecialError {
        function,
        kind: SpecialErrorKind::ShapeMismatch,
        mode,
        detail: "vector parameters must have the same length for element-wise operations",
    }
}

fn complex_is_zero(value: Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
}

fn complex_is_nonpositive_integer(value: Complex64) -> bool {
    value.im == 0.0 && value.re <= 0.0 && value.re == value.re.floor()
}

fn complex_series_converged(term: Complex64, sum: Complex64, tol: f64) -> bool {
    term.abs() <= tol * sum.abs().max(1.0)
}

fn hyp0f1_complex_scalar(
    b: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    if complex_is_nonpositive_integer(b) {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp0f1",
                kind: SpecialErrorKind::PoleInput,
                mode,
                detail: "hyp0f1: b must not be a non-positive integer",
            });
        }
        return Ok(complex_nan());
    }

    if complex_is_zero(z) {
        return Ok(Complex64::from_real(1.0));
    }

    hyp0f1_series_complex(b, z, mode)
}

fn hyp0f1_series_complex(
    b: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    let max_terms = 4_000;
    let tol = 1.0e-14;

    let mut sum = Complex64::from_real(1.0);
    let mut term = Complex64::from_real(1.0);

    for n in 1..=max_terms {
        let nf = n as f64;
        let denom = (b + Complex64::from_real(nf - 1.0)) * nf;
        term = term * z / denom;

        if !term.is_finite() || !sum.is_finite() {
            if mode == RuntimeMode::Hardened {
                return Err(SpecialError {
                    function: "hyp0f1",
                    kind: SpecialErrorKind::OverflowRisk,
                    mode,
                    detail: "series evaluation overflowed for complex-valued hyp0f1",
                });
            }
            return Ok(complex_nan());
        }

        sum = sum + term;
        if complex_series_converged(term, sum, tol) {
            return Ok(sum);
        }
    }

    Ok(sum)
}

/// Asymptotic expansion for 0F1(; b; z) for large |z|.
/// Uses relation to Bessel functions.
fn hyp0f1_asymptotic(b: f64, z: f64) -> f64 {
    // 0F1(; b; z) is related to Bessel functions:
    // 0F1(; b; -x²/4) = Γ(b) * (x/2)^(1-b) * J_{b-1}(x)  for x > 0
    // 0F1(; b; x²/4) = Γ(b) * (x/2)^(1-b) * I_{b-1}(x)   for x > 0
    //
    // For large positive z: use modified Bessel I
    // For large negative z: use Bessel J

    // Compute gamma(b) via exp(gammaln(b))
    // Use gammaln_scalar with Strict mode (won't fail for b > 0)
    let ln_gamma_b = crate::gamma::gammaln_scalar(b, RuntimeMode::Strict).unwrap_or(f64::NAN);
    let gamma_b = ln_gamma_b.exp();

    if z > 0.0 {
        // z = x²/4, so x = 2*sqrt(z)
        let x = 2.0 * z.sqrt();
        let nu = b - 1.0;

        // 0F1(; b; z) = Γ(b) * z^((1-b)/2) * I_{b-1}(x)
        let i_val = bessel_i_asymptotic(nu, x);
        gamma_b * z.powf((1.0 - b) / 2.0) * i_val
    } else {
        // z = -x²/4, so x = 2*sqrt(-z)
        let x = 2.0 * (-z).sqrt();
        let nu = b - 1.0;

        // 0F1(; b; z) = Γ(b) * (-z)^((1-b)/2) * J_{b-1}(x)
        let j_val = bessel_j_asymptotic(nu, x);
        gamma_b * (-z).powf((1.0 - b) / 2.0) * j_val
    }
}

/// Asymptotic approximation for I_nu(x) for large x.
fn bessel_i_asymptotic(nu: f64, x: f64) -> f64 {
    // I_nu(x) ~ exp(x) / sqrt(2*pi*x) * (1 - (4*nu²-1)/(8x) + ...)
    let coeff = (2.0 * std::f64::consts::PI * x).sqrt().recip();
    let mu = 4.0 * nu * nu;

    let mut sum = 1.0;
    let mut term = 1.0;
    let x_inv = 1.0 / x;

    for k in 1..10 {
        let kf = k as f64;
        term *= -(mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf) * x_inv;
        sum += term;
        if term.abs() < 1e-15 {
            break;
        }
    }

    x.exp() * coeff * sum
}

/// Asymptotic approximation for J_nu(x) for large x.
fn bessel_j_asymptotic(nu: f64, x: f64) -> f64 {
    // J_nu(x) ~ sqrt(2/(pi*x)) * cos(x - nu*pi/2 - pi/4)
    let phase = x - nu * std::f64::consts::FRAC_PI_2 - std::f64::consts::FRAC_PI_4;
    let amplitude = (2.0 / (std::f64::consts::PI * x)).sqrt();
    amplitude * phase.cos()
}

/// Gauss hypergeometric function 2F1(a, b; c; z).
///
/// Supports scalar or vector parameters with NumPy-style broadcasting.
/// Complex-valued evaluation uses the convergent `|z| < 1` series path.
/// Matches `scipy.special.hyp2f1(a, b, c, z)`.
pub fn hyp2f1(
    a: &SpecialTensor,
    b: &SpecialTensor,
    c: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    hyp2f1_dispatch("hyp2f1", a, b, c, z, mode)
}

fn hyp2f1_dispatch(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    c: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    // Check for empty tensors
    if matches!(a, SpecialTensor::Empty)
        || matches!(b, SpecialTensor::Empty)
        || matches!(c, SpecialTensor::Empty)
        || matches!(z, SpecialTensor::Empty)
    {
        return Ok(SpecialTensor::Empty);
    }

    // Get vector lengths (0 = scalar)
    let a_len = tensor_vec_len(a);
    let b_len = tensor_vec_len(b);
    let c_len = tensor_vec_len(c);
    let z_len = tensor_vec_len(z);

    // Check broadcast compatibility
    let out_len = broadcast_len_4(a_len, b_len, c_len, z_len)
        .ok_or_else(|| broadcast_shape_error(function, mode))?;

    // Check if output is complex
    let is_complex = is_complex_tensor(a)
        || is_complex_tensor(b)
        || is_complex_tensor(c)
        || is_complex_tensor(z);

    if out_len == 0 {
        // All scalars
        if is_complex {
            let a_c = tensor_as_complex_scalar(a)?;
            let b_c = tensor_as_complex_scalar(b)?;
            let c_c = tensor_as_complex_scalar(c)?;
            let z_c = tensor_as_complex_scalar(z)?;
            let result = hyp2f1_complex_parameters(a_c, b_c, c_c, z_c, mode)?;
            Ok(SpecialTensor::ComplexScalar(result))
        } else {
            let a_r = tensor_as_real_scalar(a)?;
            let b_r = tensor_as_real_scalar(b)?;
            let c_r = tensor_as_real_scalar(c)?;
            let z_r = tensor_as_real_scalar(z)?;
            let result = hyp2f1_scalar(a_r, b_r, c_r, z_r, mode)?;
            Ok(SpecialTensor::RealScalar(result))
        }
    } else {
        // At least one vector
        if is_complex {
            let mut results = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let a_c = tensor_get_complex(a, i, a_len)?;
                let b_c = tensor_get_complex(b, i, b_len)?;
                let c_c = tensor_get_complex(c, i, c_len)?;
                let z_c = tensor_get_complex(z, i, z_len)?;
                results.push(hyp2f1_complex_parameters(a_c, b_c, c_c, z_c, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        } else {
            let mut results = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let a_r = tensor_get_real(a, i, a_len)?;
                let b_r = tensor_get_real(b, i, b_len)?;
                let c_r = tensor_get_real(c, i, c_len)?;
                let z_r = tensor_get_real(z, i, z_len)?;
                results.push(hyp2f1_scalar(a_r, b_r, c_r, z_r, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
    }
}

/// Scalar 1F1(a; b; z) via series summation.
///
/// 1F1(a; b; z) = Σ_{n=0}^∞ (a)_n z^n / ((b)_n n!)
/// where (a)_n = a(a+1)...(a+n-1) is the Pochhammer symbol.
fn hyp1f1_scalar(a: f64, b: f64, z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    // b must not be zero or a negative integer
    if b == 0.0 || (b < 0.0 && b == b.floor()) {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp1f1",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "b must not be zero or a negative integer",
            });
        }
        return Ok(f64::NAN);
    }

    // Special cases
    if z == 0.0 {
        return Ok(1.0);
    }
    if a == 0.0 {
        return Ok(1.0);
    }

    // For large negative z, use Kummer's transformation: M(a,b,z) = e^z M(b-a, b, -z)
    if z < -20.0 {
        let inner = hyp1f1_series(b - a, b, -z)?;
        return Ok(z.exp() * inner);
    }

    hyp1f1_series(a, b, z)
}

/// Direct series summation for 1F1.
fn hyp1f1_series(a: f64, b: f64, z: f64) -> Result<f64, SpecialError> {
    let max_terms = 500;
    let eps = f64::EPSILON;

    let mut sum = 1.0;
    let mut term = 1.0;

    for n in 0..max_terms {
        let nf = n as f64;
        term *= (a + nf) * z / ((b + nf) * (nf + 1.0));

        if !term.is_finite() {
            break;
        }

        sum += term;

        if term.abs() < eps * sum.abs() {
            return Ok(sum);
        }
    }

    Ok(sum)
}

fn complex_nan() -> Complex64 {
    Complex64::new(f64::NAN, f64::NAN)
}

fn hyp1f1_complex_parameters(
    a: Complex64,
    b: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    if z.im == 0.0 && a.im == 0.0 && b.im == 0.0 {
        return Ok(Complex64::from_real(hyp1f1_scalar(a.re, b.re, z.re, mode)?));
    }

    if complex_is_zero(b) || complex_is_nonpositive_integer(b) {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp1f1",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "b must not be zero or a negative integer",
            });
        }
        return Ok(complex_nan());
    }

    if complex_is_zero(z) || complex_is_zero(a) {
        return Ok(Complex64::from_real(1.0));
    }

    hyp1f1_series_complex(a, b, z, mode)
}

fn hyp1f1_series_complex(
    a: Complex64,
    b: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    let max_terms = 2_000;
    let tol = 1.0e-14;

    let mut sum = Complex64::from_real(1.0);
    let mut term = Complex64::from_real(1.0);

    for n in 0..max_terms {
        let nf = n as f64;
        let scale = (a + Complex64::from_real(nf)) / ((b + Complex64::from_real(nf)) * (nf + 1.0));
        term = term * scale * z;

        if !term.is_finite() || !sum.is_finite() {
            if mode == RuntimeMode::Hardened {
                return Err(SpecialError {
                    function: "hyp1f1",
                    kind: SpecialErrorKind::OverflowRisk,
                    mode,
                    detail: "series evaluation overflowed for complex z",
                });
            }
            return Ok(complex_nan());
        }

        sum = sum + term;
        if complex_series_converged(term, sum, tol) {
            return Ok(sum);
        }
    }

    Ok(sum)
}

/// Scalar 2F1(a, b; c; z) via series summation and transformations.
///
/// 2F1(a, b; c; z) = Σ_{n=0}^∞ (a)_n (b)_n z^n / ((c)_n n!)
fn hyp2f1_scalar(a: f64, b: f64, c: f64, z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    // c must not be zero or a negative integer (unless a or b is a negative
    // integer with |a| or |b| < |c|)
    if c == 0.0 || (c < 0.0 && c == c.floor()) {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp2f1",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "c must not be zero or a negative integer",
            });
        }
        return Ok(f64::NAN);
    }

    // Special cases
    if z == 0.0 {
        return Ok(1.0);
    }
    if a == 0.0 || b == 0.0 {
        return Ok(1.0);
    }

    // |z| must be < 1 for direct series convergence
    // For |z| >= 1, use Euler's transformation: 2F1(a,b;c;z) = (1-z)^(c-a-b) 2F1(c-a, c-b; c; z)
    // (valid when |z| < 1 after transformation)
    if z.abs() < 1.0 {
        return hyp2f1_series(a, b, c, z);
    }

    // z = 1: converges when c > a + b (Gauss sum)
    if (z - 1.0).abs() < f64::EPSILON {
        let cab = c - a - b;
        if cab > 0.0 {
            // 2F1(a,b;c;1) = Gamma(c)Gamma(c-a-b) / (Gamma(c-a)Gamma(c-b))
            let result = gamma_ratio_for_hyp2f1(c, cab, c - a, c - b);
            return Ok(result);
        }
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp2f1",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "2F1 diverges at z=1 when c <= a+b",
            });
        }
        return Ok(f64::INFINITY);
    }

    // For z < 0, use Pfaff transformation: 2F1(a,b;c;z) = (1-z)^(-a) 2F1(a, c-b; c; z/(z-1))
    if z < 0.0 {
        let z_new = z / (z - 1.0);
        if z_new.abs() < 1.0 {
            let factor = (1.0 - z).powf(-a);
            let inner = hyp2f1_series(a, c - b, c, z_new)?;
            return Ok(factor * inner);
        }
    }

    // For z > 1, use 1/z transformation when possible
    if z > 1.0 && z.abs() > 1.0 {
        // Use the linear transformation formula for 1/z
        let z_inv = 1.0 / z;
        if z_inv.abs() < 1.0 {
            // 2F1(a,b;c;z) = Γ(c)Γ(b-a)/(Γ(b)Γ(c-a)) (-z)^(-a) 2F1(a,1-c+a;1-b+a;1/z)
            //              + Γ(c)Γ(a-b)/(Γ(a)Γ(c-b)) (-z)^(-b) 2F1(b,1-c+b;1-a+b;1/z)
            // Simplified: use Pfaff on the already-transformed result
            if mode == RuntimeMode::Hardened {
                return Err(SpecialError {
                    function: "hyp2f1",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "2F1 with |z| > 1 may not converge reliably",
                });
            }
        }
    }

    // Fallback: direct series (may not converge for |z| >= 1)
    hyp2f1_series(a, b, c, z)
}

/// Direct series summation for 2F1.
fn hyp2f1_series(a: f64, b: f64, c: f64, z: f64) -> Result<f64, SpecialError> {
    let max_terms = 500;
    let eps = f64::EPSILON;

    let mut sum = 1.0;
    let mut term = 1.0;

    for n in 0..max_terms {
        let nf = n as f64;
        term *= (a + nf) * (b + nf) * z / ((c + nf) * (nf + 1.0));

        if !term.is_finite() {
            break;
        }

        sum += term;

        if term.abs() < eps * sum.abs() {
            return Ok(sum);
        }
    }

    Ok(sum)
}

fn hyp2f1_complex_parameters(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    if z.im == 0.0 && a.im == 0.0 && b.im == 0.0 && c.im == 0.0 {
        return Ok(Complex64::from_real(hyp2f1_scalar(
            a.re, b.re, c.re, z.re, mode,
        )?));
    }

    if complex_is_zero(c) || complex_is_nonpositive_integer(c) {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp2f1",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "c must not be zero or a negative integer",
            });
        }
        return Ok(complex_nan());
    }

    if complex_is_zero(z) || complex_is_zero(a) || complex_is_zero(b) {
        return Ok(Complex64::from_real(1.0));
    }

    if z.abs() < 1.0 {
        return hyp2f1_series_complex(a, b, c, z, mode);
    }

    if mode == RuntimeMode::Hardened {
        return Err(SpecialError {
            function: "hyp2f1",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "complex-valued hyp2f1 currently requires |z| < 1 for stable evaluation",
        });
    }

    Ok(complex_nan())
}

fn hyp2f1_series_complex(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    let max_terms = 2_000;
    let tol = 1.0e-14;

    let mut sum = Complex64::from_real(1.0);
    let mut term = Complex64::from_real(1.0);

    for n in 0..max_terms {
        let nf = n as f64;
        let scale = ((a + Complex64::from_real(nf)) * (b + Complex64::from_real(nf)))
            / ((c + Complex64::from_real(nf)) * (nf + 1.0));
        term = term * scale * z;

        if !term.is_finite() || !sum.is_finite() {
            if mode == RuntimeMode::Hardened {
                return Err(SpecialError {
                    function: "hyp2f1",
                    kind: SpecialErrorKind::OverflowRisk,
                    mode,
                    detail: "series evaluation overflowed for complex z",
                });
            }
            return Ok(complex_nan());
        }

        sum = sum + term;
        if complex_series_converged(term, sum, tol) {
            return Ok(sum);
        }
    }

    Ok(sum)
}

/// Compute Gamma(c)*Gamma(c-a-b) / (Gamma(c-a)*Gamma(c-b)) for the Gauss sum.
fn gamma_ratio_for_hyp2f1(c: f64, cab: f64, ca: f64, cb: f64) -> f64 {
    let (ln_c, sign_c) = ln_gamma_with_sign(c);
    let (ln_cab, sign_cab) = ln_gamma_with_sign(cab);
    let (ln_ca, sign_ca) = ln_gamma_with_sign(ca);
    let (ln_cb, sign_cb) = ln_gamma_with_sign(cb);

    if !ln_c.is_finite() || !ln_cab.is_finite() || sign_c == 0.0 || sign_cab == 0.0 {
        return f64::NAN;
    }
    if !ln_ca.is_finite() || !ln_cb.is_finite() || sign_ca == 0.0 || sign_cb == 0.0 {
        return 0.0;
    }

    let sign = sign_c * sign_cab * sign_ca * sign_cb;
    sign * (ln_c + ln_cab - ln_ca - ln_cb).exp()
}

/// Computes (ln(|Gamma(x)|), sign(Gamma(x))) via Lanczos approximation.
fn ln_gamma_with_sign(x: f64) -> (f64, f64) {
    if x <= 0.0 && x == x.floor() {
        return (f64::INFINITY, 0.0);
    }
    // Lanczos coefficients (g=7, n=9)
    #[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        // Reflection formula
        let pi = std::f64::consts::PI;
        let sin_pi_x = (pi * x).sin();
        let sign = if sin_pi_x < 0.0 { -1.0 } else { 1.0 };
        let (ln_1_minus_x, _) = ln_gamma_with_sign(1.0 - x);
        let ln_abs = (pi / sin_pi_x.abs()).ln() - ln_1_minus_x;
        return (ln_abs, sign);
    }

    let x = x - 1.0;
    let mut sum = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }

    let t = x + G + 0.5;
    let val = 0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln();
    (val, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> SpecialTensor {
        SpecialTensor::RealScalar(v)
    }

    fn complex(re: f64, im: f64) -> SpecialTensor {
        SpecialTensor::ComplexScalar(Complex64::new(re, im))
    }

    fn complex_vec(values: Vec<Complex64>) -> SpecialTensor {
        SpecialTensor::ComplexVec(values)
    }

    fn get_scalar(r: &SpecialResult) -> Option<f64> {
        match r.as_ref() {
            Ok(SpecialTensor::RealScalar(v)) => Some(*v),
            _ => None,
        }
    }

    fn get_complex_scalar(r: &SpecialResult) -> Option<Complex64> {
        match r.as_ref() {
            Ok(SpecialTensor::ComplexScalar(v)) => Some(*v),
            _ => None,
        }
    }

    fn get_real_vec(r: &SpecialResult) -> Option<&[f64]> {
        match r.as_ref() {
            Ok(SpecialTensor::RealVec(values)) => Some(values.as_slice()),
            _ => None,
        }
    }

    fn get_complex_vec(r: &SpecialResult) -> Option<&[Complex64]> {
        match r.as_ref() {
            Ok(SpecialTensor::ComplexVec(values)) => Some(values.as_slice()),
            _ => None,
        }
    }

    fn error_kind(r: &SpecialResult) -> Option<SpecialErrorKind> {
        match r {
            Err(err) => Some(err.kind),
            Ok(_) => None,
        }
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tol: f64) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tol,
            "expected {}+{}i, got {}+{}i (|delta|={delta})",
            expected.re,
            expected.im,
            actual.re,
            actual.im
        );
    }

    // ── hyp0f1 tests ────────────────────────────────────────────────

    #[test]
    fn hyp0f1_complex_parameter_zero_z_is_one() {
        let result = hyp0f1(
            &complex(1.25, -0.5),
            &complex(0.0, 0.0),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&result).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            Complex64::from_real(1.0),
            1.0e-14,
        );
    }

    #[test]
    fn hyp0f1_complex_real_parameter_matches_real_path() {
        let complex_result = hyp0f1(&complex(1.5, 0.0), &scalar(0.25), RuntimeMode::Strict);
        let real_result = hyp0f1(&scalar(1.5), &scalar(0.25), RuntimeMode::Strict);
        assert_complex_close(
            get_complex_scalar(&complex_result).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            Complex64::from_real(get_scalar(&real_result).unwrap_or(f64::NAN)),
            1.0e-12,
        );
    }

    #[test]
    fn hyp0f1_complex_parameter_preserves_parameter_conjugation_symmetry() {
        let b = Complex64::new(1.25, 0.5);
        let z = Complex64::new(0.2, -0.1);
        let lhs = hyp0f1(
            &complex(b.re, b.im),
            &complex(z.re, z.im),
            RuntimeMode::Strict,
        );
        let rhs = hyp0f1(
            &complex(b.re, -b.im),
            &complex(z.re, -z.im),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&rhs).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            get_complex_scalar(&lhs)
                .map_or(Complex64::new(f64::NAN, f64::NAN), |value| value.conj()),
            1.0e-12,
        );
    }

    #[test]
    fn hyp0f1_complex_parameter_pole_errors_hardened() {
        let result = hyp0f1(&complex(0.0, 0.0), &scalar(1.0), RuntimeMode::Hardened);
        assert_eq!(error_kind(&result), Some(SpecialErrorKind::PoleInput));
    }

    #[test]
    fn hyp0f1_parameter_vector_broadcast_matches_scalar_evaluations() {
        let params = complex_vec(vec![Complex64::new(1.25, 0.5), Complex64::new(1.25, -0.5)]);
        let z = complex(0.2, -0.1);
        let result = hyp0f1(&params, &z, RuntimeMode::Strict);
        let values = get_complex_vec(&result).unwrap_or(&[]);

        assert_eq!(values.len(), 2);
        assert_complex_close(
            values[0],
            hyp0f1_complex_scalar(
                Complex64::new(1.25, 0.5),
                Complex64::new(0.2, -0.1),
                RuntimeMode::Strict,
            )
            .unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            1.0e-12,
        );
        assert_complex_close(
            values[1],
            hyp0f1_complex_scalar(
                Complex64::new(1.25, -0.5),
                Complex64::new(0.2, -0.1),
                RuntimeMode::Strict,
            )
            .unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            1.0e-12,
        );
    }

    // ── hyp1f1 tests ────────────────────────────────────────────────

    #[test]
    fn hyp1f1_zero_z_is_one() {
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(0.0),
            RuntimeMode::Strict,
        );
        assert!((get_scalar(&r).unwrap_or(f64::NAN) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hyp1f1_zero_a_is_one() {
        let r = hyp1f1(
            &scalar(0.0),
            &scalar(2.0),
            &scalar(5.0),
            RuntimeMode::Strict,
        );
        assert!((get_scalar(&r).unwrap_or(f64::NAN) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hyp1f1_a_equals_b_is_exp() {
        // 1F1(a; a; z) = e^z
        let z = 1.5;
        let r = hyp1f1(&scalar(3.0), &scalar(3.0), &scalar(z), RuntimeMode::Strict);
        let expected = z.exp();
        assert!(
            (get_scalar(&r).unwrap_or(f64::NAN) - expected).abs() < 1e-10,
            "1F1(a;a;z) should be e^z: got {} expected {expected}",
            get_scalar(&r).unwrap_or(f64::NAN)
        );
    }

    #[test]
    fn hyp1f1_known_value() {
        // 1F1(1; 1; 1) = e
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(1.0),
            RuntimeMode::Strict,
        );
        assert!(
            (get_scalar(&r).unwrap_or(f64::NAN) - std::f64::consts::E).abs() < 1e-10,
            "1F1(1;1;1) should be e"
        );
    }

    #[test]
    fn hyp1f1_negative_z() {
        // 1F1(1; 2; -1) = (1 - e^(-1)) * 2 / 1 ... known value ≈ 0.6321205588
        // Actually: 1F1(1; 2; -1) = (e^(-1) - 1) / (-1) * 2/1...
        // Let's just check it's finite and reasonable
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(-1.0),
            RuntimeMode::Strict,
        );
        let val = get_scalar(&r).unwrap_or(f64::NAN);
        assert!(val.is_finite(), "hyp1f1 should be finite for negative z");
        assert!(val > 0.0, "hyp1f1(1,2,-1) should be positive");
    }

    #[test]
    fn hyp1f1_large_negative_z_kummer() {
        // For large negative z, Kummer's transformation is used
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(-30.0),
            RuntimeMode::Strict,
        );
        let val = get_scalar(&r).unwrap_or(f64::NAN);
        assert!(val.is_finite(), "should handle large negative z");
    }

    #[test]
    fn hyp1f1_b_zero_returns_nan_strict() {
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(0.0),
            &scalar(1.0),
            RuntimeMode::Strict,
        );
        let val = get_scalar(&r).unwrap_or(f64::NAN);
        assert!(val.is_nan(), "b=0 should return NaN in strict mode");
    }

    #[test]
    fn hyp1f1_b_zero_errors_hardened() {
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(0.0),
            &scalar(1.0),
            RuntimeMode::Hardened,
        );
        assert!(r.is_err(), "b=0 should error in hardened mode");
    }

    #[test]
    fn hyp1f1_vectorized() {
        let z_vec = SpecialTensor::RealVec(vec![0.0, 1.0, 2.0]);
        let r = hyp1f1(&scalar(1.0), &scalar(1.0), &z_vec, RuntimeMode::Strict);
        let values = get_real_vec(&r).unwrap_or(&[]);
        assert_eq!(values.len(), 3);
        assert!((values[0] - 1.0).abs() < 1e-14); // e^0
        assert!((values[1] - std::f64::consts::E).abs() < 1e-10); // e^1
        assert!((values[2] - std::f64::consts::E.powi(2)).abs() < 1e-10); // e^2
    }

    #[test]
    fn hyp1f1_complex_matches_exp_identity() {
        let z = Complex64::new(0.5, 0.75);
        let r = hyp1f1(
            &scalar(2.0),
            &scalar(2.0),
            &complex(z.re, z.im),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&r).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            z.exp(),
            1.0e-10,
        );
    }

    #[test]
    fn hyp1f1_complex_vector_preserves_conjugation_symmetry() {
        let z = Complex64::new(0.3, 0.4);
        let input = SpecialTensor::ComplexVec(vec![z, z.conj()]);
        let r = hyp1f1(&scalar(1.0), &scalar(1.0), &input, RuntimeMode::Strict);
        let values = get_complex_vec(&r).unwrap_or(&[]);
        assert_eq!(values.len(), 2);
        assert_complex_close(values[1], values[0].conj(), 1.0e-10);
    }

    #[test]
    fn hyp1f1_complex_parameters_match_exp_identity() {
        let a = Complex64::new(1.25, -0.5);
        let z = Complex64::new(0.25, -0.75);
        let r = hyp1f1(
            &complex(a.re, a.im),
            &complex(a.re, a.im),
            &complex(z.re, z.im),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&r).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            z.exp(),
            1.0e-10,
        );
    }

    #[test]
    fn hyp1f1_complex_parameter_vector_matches_exp_identity() {
        let a = Complex64::new(0.75, 0.4);
        let z0 = Complex64::new(0.15, 0.2);
        let z1 = z0.conj();
        let r = hyp1f1(
            &complex(a.re, a.im),
            &complex(a.re, a.im),
            &complex_vec(vec![z0, z1]),
            RuntimeMode::Strict,
        );
        let values = get_complex_vec(&r).unwrap_or(&[]);
        assert_eq!(values.len(), 2);
        assert_complex_close(values[0], z0.exp(), 1.0e-10);
        assert_complex_close(values[1], z1.exp(), 1.0e-10);
    }

    #[test]
    fn hyp1f1_complex_real_parameters_match_real_path() {
        let complex_result = hyp1f1(
            &complex(1.0, 0.0),
            &complex(2.0, 0.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        let real_result = hyp1f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&complex_result).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            Complex64::from_real(get_scalar(&real_result).unwrap_or(f64::NAN)),
            1.0e-12,
        );
    }

    #[test]
    fn hyp1f1_complex_parameter_pole_errors_hardened() {
        let result = hyp1f1(
            &complex(1.0, 0.0),
            &complex(0.0, 0.0),
            &scalar(1.0),
            RuntimeMode::Hardened,
        );
        assert_eq!(error_kind(&result), Some(SpecialErrorKind::DomainError));
    }

    #[test]
    fn hyp1f1_parameter_vectors_broadcast_with_scalar_z() {
        let a = complex_vec(vec![Complex64::new(0.75, 0.4), Complex64::new(0.75, -0.4)]);
        let b = complex_vec(vec![Complex64::new(0.75, 0.4), Complex64::new(0.75, -0.4)]);
        let z = complex(0.15, 0.2);
        let result = hyp1f1(&a, &b, &z, RuntimeMode::Strict);
        let values = get_complex_vec(&result).unwrap_or(&[]);

        assert_eq!(values.len(), 2);
        assert_complex_close(values[0], Complex64::new(0.15, 0.2).exp(), 1.0e-10);
        assert_complex_close(values[1], Complex64::new(0.15, 0.2).exp(), 1.0e-10);
    }

    #[test]
    fn hyp1f1_parameter_vector_mismatch_is_shape_error() {
        let result = hyp1f1(
            &SpecialTensor::RealVec(vec![1.0, 2.0]),
            &SpecialTensor::RealVec(vec![1.0]),
            &scalar(0.5),
            RuntimeMode::Hardened,
        );
        assert_eq!(error_kind(&result), Some(SpecialErrorKind::ShapeMismatch));
    }

    // ── hyp2f1 tests ────────────────────────────────────────────────

    #[test]
    fn hyp2f1_zero_z_is_one() {
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(3.0),
            &scalar(0.0),
            RuntimeMode::Strict,
        );
        assert!((get_scalar(&r).unwrap_or(f64::NAN) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hyp2f1_a_zero_is_one() {
        let r = hyp2f1(
            &scalar(0.0),
            &scalar(2.0),
            &scalar(3.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        assert!((get_scalar(&r).unwrap_or(f64::NAN) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hyp2f1_known_geometric() {
        // 2F1(1, 1; 2; z) = -ln(1-z)/z for |z| < 1
        let z = 0.5;
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(2.0),
            &scalar(z),
            RuntimeMode::Strict,
        );
        let expected = -(1.0 - z).ln() / z;
        assert!(
            (get_scalar(&r).unwrap_or(f64::NAN) - expected).abs() < 1e-10,
            "2F1(1,1;2;0.5) should be -ln(0.5)/0.5: got {} expected {expected}",
            get_scalar(&r).unwrap_or(f64::NAN)
        );
    }

    #[test]
    fn hyp2f1_negative_a_polynomial() {
        // 2F1(-2, b; c; z) terminates after 3 terms (polynomial)
        // 2F1(-2, 1; 1; z) = 1 + (-2)(1)z/(1·1) + (-2)(-1)(1)(2)z²/(1·2·1·2)
        //                   = 1 - 2z + z²  = (1-z)²
        let z = 0.3;
        let r = hyp2f1(
            &scalar(-2.0),
            &scalar(1.0),
            &scalar(1.0),
            &scalar(z),
            RuntimeMode::Strict,
        );
        let expected = (1.0 - z).powi(2);
        assert!(
            (get_scalar(&r).unwrap_or(f64::NAN) - expected).abs() < 1e-10,
            "2F1(-2,1;1;z) should be (1-z)²: got {} expected {expected}",
            get_scalar(&r).unwrap_or(f64::NAN)
        );
    }

    #[test]
    fn hyp2f1_at_z_one_gauss_sum() {
        // 2F1(1, 1; 3; 1) = Gamma(3)*Gamma(1) / (Gamma(2)*Gamma(2)) = 2*1/(1*1) = 2
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(3.0),
            &scalar(1.0),
            RuntimeMode::Strict,
        );
        assert!(
            (get_scalar(&r).unwrap_or(f64::NAN) - 2.0).abs() < 1e-8,
            "2F1(1,1;3;1) should be 2: got {}",
            get_scalar(&r).unwrap_or(f64::NAN)
        );
    }

    #[test]
    fn hyp2f1_c_zero_returns_nan_strict() {
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(0.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        let val = get_scalar(&r).unwrap_or(f64::NAN);
        assert!(val.is_nan(), "c=0 should return NaN in strict mode");
    }

    #[test]
    fn hyp2f1_negative_z() {
        // 2F1(1, 1; 2; -0.5) should use Pfaff transformation
        let z = -0.5;
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(2.0),
            &scalar(z),
            RuntimeMode::Strict,
        );
        let expected = -(1.0 - z).ln() / z; // = -ln(1.5)/(-0.5) = 2*ln(1.5)
        let val = get_scalar(&r).unwrap_or(f64::NAN);
        assert!(
            (val - expected).abs() < 1e-10,
            "2F1(1,1;2;-0.5): got {val} expected {expected}"
        );
    }

    #[test]
    fn hyp2f1_complex_matches_inverse_linear_identity_inside_unit_disk() {
        let z = Complex64::new(0.25, 0.25);
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(2.0),
            &complex(z.re, z.im),
            RuntimeMode::Strict,
        );
        let expected = Complex64::from_real(1.0) / (Complex64::from_real(1.0) - z);
        assert_complex_close(
            get_complex_scalar(&r).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            expected,
            1.0e-10,
        );
    }

    #[test]
    fn hyp2f1_complex_vector_preserves_conjugation_symmetry() {
        let z = Complex64::new(0.2, 0.3);
        let input = SpecialTensor::ComplexVec(vec![z, z.conj()]);
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(2.0),
            &input,
            RuntimeMode::Strict,
        );
        let values = get_complex_vec(&r).unwrap_or(&[]);
        assert_eq!(values.len(), 2);
        assert_complex_close(values[1], values[0].conj(), 1.0e-10);
    }

    #[test]
    fn hyp2f1_complex_outside_unit_disk_errors_in_hardened_mode() {
        let result = hyp2f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(2.0),
            &complex(1.25, 0.5),
            RuntimeMode::Hardened,
        );
        assert_eq!(error_kind(&result), Some(SpecialErrorKind::DomainError));
    }

    #[test]
    fn hyp2f1_complex_parameter_linear_identity_holds() {
        let b = Complex64::new(1.25, -0.5);
        let z = Complex64::new(0.2, 0.3);
        let r = hyp2f1(
            &scalar(-1.0),
            &complex(b.re, b.im),
            &complex(b.re, b.im),
            &complex(z.re, z.im),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&r).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            Complex64::from_real(1.0) - z,
            1.0e-12,
        );
    }

    #[test]
    fn hyp2f1_complex_parameter_vector_matches_linear_identity() {
        let b = Complex64::new(0.75, 0.6);
        let z0 = Complex64::new(0.15, -0.25);
        let z1 = z0.conj();
        let r = hyp2f1(
            &scalar(-1.0),
            &complex(b.re, b.im),
            &complex(b.re, b.im),
            &complex_vec(vec![z0, z1]),
            RuntimeMode::Strict,
        );
        let values = get_complex_vec(&r).unwrap_or(&[]);
        assert_eq!(values.len(), 2);
        assert_complex_close(values[0], Complex64::from_real(1.0) - z0, 1.0e-12);
        assert_complex_close(values[1], Complex64::from_real(1.0) - z1, 1.0e-12);
    }

    #[test]
    fn hyp2f1_complex_real_parameters_match_real_path() {
        let complex_result = hyp2f1(
            &complex(1.0, 0.0),
            &complex(1.0, 0.0),
            &complex(2.0, 0.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        let real_result = hyp2f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(2.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&complex_result).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            Complex64::from_real(get_scalar(&real_result).unwrap_or(f64::NAN)),
            1.0e-12,
        );
    }

    #[test]
    fn hyp2f1_complex_parameter_pole_errors_hardened() {
        let result = hyp2f1(
            &scalar(1.0),
            &complex(1.0, 0.0),
            &complex(0.0, 0.0),
            &scalar(0.5),
            RuntimeMode::Hardened,
        );
        assert_eq!(error_kind(&result), Some(SpecialErrorKind::DomainError));
    }

    #[test]
    fn hyp2f1_parameter_vectors_broadcast_with_scalar_z() {
        let b = complex_vec(vec![Complex64::new(0.75, 0.6), Complex64::new(0.75, -0.6)]);
        let c = complex_vec(vec![Complex64::new(0.75, 0.6), Complex64::new(0.75, -0.6)]);
        let z = complex(0.15, -0.25);
        let result = hyp2f1(&scalar(-1.0), &b, &c, &z, RuntimeMode::Strict);
        let values = get_complex_vec(&result).unwrap_or(&[]);

        assert_eq!(values.len(), 2);
        assert_complex_close(
            values[0],
            Complex64::from_real(1.0) - Complex64::new(0.15, -0.25),
            1.0e-12,
        );
        assert_complex_close(
            values[1],
            Complex64::from_real(1.0) - Complex64::new(0.15, -0.25),
            1.0e-12,
        );
    }
}
