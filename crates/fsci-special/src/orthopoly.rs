#![forbid(unsafe_code)]

//! Orthogonal polynomial evaluation.
//!
//! All polynomials are computed via stable three-term recurrence relations.
//! These are the building blocks for Gaussian quadrature, spectral methods,
//! and spherical harmonics.
//!
//! - `eval_legendre(n, x)` — Legendre P_n(x)
//! - `eval_chebyt(n, x)` — Chebyshev T_n(x) (first kind)
//! - `eval_chebyu(n, x)` — Chebyshev U_n(x) (second kind)
//! - `eval_hermite(n, x)` — Physicist's Hermite H_n(x)
//! - `eval_hermitenorm(n, x)` — Probabilist's Hermite He_n(x)
//! - `eval_laguerre(n, x)` — Laguerre L_n(x)
//! - `eval_genlaguerre(n, alpha, x)` — Generalized Laguerre L_n^α(x)
//! - `eval_jacobi(n, alpha, beta, x)` — Jacobi P_n^{α,β}(x)
//! - `eval_gegenbauer(n, alpha, x)` — Gegenbauer C_n^α(x)
//! - `roots_legendre(n)` — Gauss-Legendre nodes and weights
//! - `roots_chebyt(n)` — Gauss-Chebyshev nodes and weights
//! - `roots_hermite(n)` — Gauss-Hermite nodes and weights
//! - `roots_laguerre(n)` — Gauss-Laguerre nodes and weights
//! - `roots_jacobi(n, alpha, beta)` — Gauss-Jacobi nodes and weights

use std::f64::consts::PI;

use crate::bessel::{jv_scalar, spherical_jn_scalar};
use fsci_runtime::RuntimeMode;


/// Evaluate the Legendre polynomial P_n(x) of degree n at point x.
///
/// Uses the three-term recurrence:
///   P_0(x) = 1, P_1(x) = x,
///   (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
///
/// Domain: x in [-1, 1] for orthogonality, but evaluates for any real x.
pub fn eval_legendre(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if !x.is_finite() {
        return f64::NAN;
    }
    if n == 1 {
        return x;
    }
    let mut p_prev = 1.0; // P_0
    let mut p_curr = x; // P_1
    for k in 1..n {
        let kf = k as f64;
        let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

/// Compute Legendre polynomial values P_k(x) and derivatives P'_k(x) for
/// k = 0..=n using the standard three-term recurrence.
///
/// Matches `scipy.special.lpn(n, x)` which returns `(Pn, Pn_deriv)` arrays
/// of length n+1.
///
/// Recurrence (values):
///   P_0(x) = 1, P_1(x) = x
///   (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)
///
/// Recurrence (derivatives):
///   P'_0(x) = 0, P'_1(x) = 1
///   (1 - x^2) P'_k(x) = k (P_{k-1}(x) - x P_k(x))
/// At |x| = 1 the derivative is given by the closed form
///   P'_k(±1) = (±1)^(k+1) * k(k+1) / 2
/// to avoid division by 1 - x^2 = 0.
pub fn lpn(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
    let len = n as usize + 1;
    let mut values = Vec::with_capacity(len);
    let mut derivs = Vec::with_capacity(len);
    if n == 0 {
        values.push(1.0);
        derivs.push(0.0);
        return (values, derivs);
    }
    if !x.is_finite() {
        return (vec![f64::NAN; len], vec![f64::NAN; len]);
    }
    values.push(1.0); // P_0
    values.push(x); // P_1
    for k in 1..n {
        let kf = k as f64;
        let p_next = ((2.0 * kf + 1.0) * x * values[k as usize] - kf * values[(k - 1) as usize])
            / (kf + 1.0);
        values.push(p_next);
    }
    let near_unit = (x.abs() - 1.0).abs() <= f64::EPSILON;
    if near_unit {
        let sign: f64 = if x > 0.0 { 1.0 } else { -1.0 };
        for k in 0..=n {
            let kf = k as f64;
            // P'_k(±1) = (±1)^(k+1) * k(k+1) / 2.
            let factor = sign.powi(k as i32 + 1) * kf * (kf + 1.0) * 0.5;
            derivs.push(factor);
        }
    } else {
        let one_minus_xsq = 1.0 - x * x;
        derivs.push(0.0); // P'_0
        for k in 1..=n {
            let kf = k as f64;
            let p_k = values[k as usize];
            let p_km1 = values[(k - 1) as usize];
            derivs.push(kf * (p_km1 - x * p_k) / one_minus_xsq);
        }
    }
    (values, derivs)
}

/// Compute Legendre polynomials of the second kind Q_k(x) and derivatives
/// Q'_k(x) for k = 0..=n on the real interval |x| < 1.
///
/// Matches `scipy.special.lqn(n, x)` which returns `(Qn, Qn_deriv)` arrays
/// of length n+1.
///
/// Q_0(x) = atanh(x) = (1/2) ln((1+x)/(1-x))
/// Q_1(x) = x * Q_0(x) - 1
/// (k+1) Q_{k+1}(x) = (2k+1) x Q_k(x) - k Q_{k-1}(x)
/// (1 - x^2) Q'_k(x) = k (Q_{k-1}(x) - x Q_k(x))
///
/// Returns NaN-filled arrays at the singular points x = ±1 (where Q_0
/// diverges) and outside (-1, 1), matching scipy's domain convention for
/// real-valued evaluation.
pub fn lqn(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
    let len = n as usize + 1;
    if !x.is_finite() || x.abs() >= 1.0 {
        return (vec![f64::NAN; len], vec![f64::NAN; len]);
    }
    let mut values = Vec::with_capacity(len);
    let mut derivs = Vec::with_capacity(len);
    let q0 = x.atanh();
    values.push(q0);
    if n == 0 {
        derivs.push(1.0 / (1.0 - x * x));
        return (values, derivs);
    }
    values.push(x * q0 - 1.0);
    for k in 1..n {
        let kf = k as f64;
        let q_next = ((2.0 * kf + 1.0) * x * values[k as usize] - kf * values[(k - 1) as usize])
            / (kf + 1.0);
        values.push(q_next);
    }
    // Derivatives via the same recurrence shape as P: (1-x²) Q'_k = k(Q_{k-1} − x Q_k).
    // For k=0 the formula collapses to Q'_0 = 1/(1-x²) directly.
    let one_minus_xsq = 1.0 - x * x;
    derivs.push(1.0 / one_minus_xsq);
    for k in 1..=n {
        let kf = k as f64;
        let q_k = values[k as usize];
        let q_km1 = values[(k - 1) as usize];
        derivs.push(kf * (q_km1 - x * q_k) / one_minus_xsq);
    }
    (values, derivs)
}

/// Associated Legendre Q array `Q_l^m(x)` for `0 ≤ m ≤ m_max` and
/// `0 ≤ l ≤ n_max`, on the open real interval `|x| < 1`.
///
/// Matches `scipy.special.lqmn(m, n, x)` shape (the values portion).
/// Returns a `(m_max+1) × (n_max+1)` row-major matrix.
///
/// Algorithm: row 0 is the standard Legendre Q sequence from `lqn`. Cells
/// with `l < m` are seeded from SciPy's derivative definition,
/// `Q_l^m(x) = (-1)^m (1-x^2)^(m/2) d^m Q_l(x)/dx^m`, then each row
/// is continued through the standard degree recurrence until the diagonal.
/// Cells with `l ≥ m` are built from row m−1 using DLMF 14.10.4:
///   √(1−x²) · Q_l^m(x) = (l−m+1) x · Q_l^{m−1}(x) − (l+m−1) · Q_{l−1}^{m−1}(x)
/// for l = m..=n_max.
///
/// For `|x| ≥ 1` or non-finite `x`, returns NaN-filled arrays — matching
/// the lqn singular convention (`Q_0(x) = atanh(x)` diverges at the
/// endpoints).
pub fn lqmn(m_max: u32, n_max: u32, x: f64) -> Vec<Vec<f64>> {
    let rows = m_max as usize + 1;
    let cols = n_max as usize + 1;
    let mut out = vec![vec![0.0_f64; cols]; rows];
    if !x.is_finite() || x.abs() >= 1.0 {
        for row in &mut out {
            for cell in row.iter_mut() {
                *cell = f64::NAN;
            }
        }
        return out;
    }
    // Row 0: standard Legendre Q.
    let (q0, _) = lqn(n_max, x);
    for (l, &q) in q0.iter().enumerate() {
        out[0][l] = q;
    }
    let one_minus_xsq = 1.0 - x * x;
    let denom = one_minus_xsq.sqrt();
    if denom == 0.0 {
        // Defensive — shouldn't happen since we already rejected |x| ≥ 1.
        return out;
    }
    let atanh_derivs = atanh_derivatives(m_max, x);
    let mut sqrt_power = 1.0_f64;

    // Row m for m ≥ 1.
    for m in 1..=m_max {
        let m_idx = m as usize;
        sqrt_power *= denom;
        let signed_sqrt_power = if m % 2 == 0 { sqrt_power } else { -sqrt_power };

        // Below the diagonal, scipy.special.lqmn follows the derivative
        // definition rather than the finite-degree polynomial convention.
        out[m_idx][0] = signed_sqrt_power * atanh_derivs[m_idx];
        if cols > 1 {
            out[m_idx][1] = signed_sqrt_power
                * (x * atanh_derivs[m_idx] + (m as f64) * atanh_derivs[m_idx - 1]);
        }
        if n_max >= 2 {
            for l_prev in 1..n_max {
                let l = l_prev + 1;
                if l >= m {
                    break;
                }
                let l_prev_idx = l_prev as usize;
                let lf = l_prev as f64;
                let mf = m as f64;
                out[m_idx][l as usize] = ((2.0 * lf + 1.0) * x * out[m_idx][l_prev_idx]
                    - (lf + mf) * out[m_idx][l_prev_idx - 1])
                    / (lf - mf + 1.0);
            }
        }

        for l in m..=n_max {
            let l_idx = l as usize;
            let lf = l as f64;
            let mf = m as f64;
            let q_l_prev = out[m_idx - 1][l_idx];
            let q_lm1_prev = out[m_idx - 1][l_idx - 1];
            out[m_idx][l_idx] =
                ((lf - mf + 1.0) * x * q_l_prev - (lf + mf - 1.0) * q_lm1_prev) / denom;
        }
    }
    out
}

fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0_f64, |acc, &c| acc * x + c)
}

fn next_atanh_derivative_poly(poly: &[f64], order: usize) -> Vec<f64> {
    let mut next = vec![0.0_f64; poly.len() + 2];
    for (degree, &coeff) in poly.iter().enumerate().skip(1) {
        let derivative_coeff = (degree as f64) * coeff;
        next[degree - 1] += derivative_coeff;
        next[degree + 1] -= derivative_coeff;
    }
    let scale = 2.0 * (order as f64);
    for (degree, &coeff) in poly.iter().enumerate() {
        next[degree + 1] += scale * coeff;
    }
    next
}

fn atanh_derivatives(max_order: u32, x: f64) -> Vec<f64> {
    let max_order = max_order as usize;
    let mut derivs = vec![0.0_f64; max_order + 1];
    derivs[0] = x.atanh();
    if max_order == 0 {
        return derivs;
    }

    let one_minus_xsq = 1.0 - x * x;
    let mut denom_power = one_minus_xsq;
    let mut poly = vec![1.0_f64]; // P_1 where d/dx atanh(x) = P_1(x)/(1-x^2).
    for (order, deriv) in derivs.iter_mut().enumerate().take(max_order + 1).skip(1) {
        *deriv = eval_poly(&poly, x) / denom_power;
        if order < max_order {
            poly = next_atanh_derivative_poly(&poly, order);
            denom_power *= one_minus_xsq;
        }
    }
    derivs
}

/// Associated Legendre polynomial array `P_l^m(x)` for `0 ≤ m ≤ m_max` and
/// `0 ≤ l ≤ n_max`. Returns a `(m_max+1) × (n_max+1)` row-major matrix
/// indexed as `out[m][l]`.
///
/// Matches `scipy.special.lpmn(m, n, x)` shape (the values portion). For
/// each fixed `m`, the row `out[m][..]` is filled by:
///   P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^(m/2)
///   P_(m+1)^m(x) = (2m+1) x P_m^m(x)
///   (l-m+1) P_(l+1)^m(x) = (2l+1) x P_l^m(x) - (l+m) P_(l-1)^m(x)
///
/// Cells where `m > l` are 0.0 by convention.
pub fn lpmn(m_max: u32, n_max: u32, x: f64) -> Vec<Vec<f64>> {
    let rows = m_max as usize + 1;
    let cols = n_max as usize + 1;
    let mut out = vec![vec![0.0_f64; cols]; rows];
    if !x.is_finite() {
        for row in &mut out {
            for cell in row.iter_mut() {
                *cell = f64::NAN;
            }
        }
        return out;
    }
    let somx2 = (1.0 - x * x).max(0.0).sqrt();
    // For each m, compute P_m^m, P_(m+1)^m, then recurrence to fill row m.
    for m in 0..=m_max {
        // P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^(m/2).
        let mut pmm = 1.0_f64;
        if m > 0 {
            let mut fact = 1.0_f64;
            for _ in 1..=m {
                pmm *= -fact * somx2;
                fact += 2.0;
            }
        }
        let m_idx = m as usize;
        // Cells with l < m are zero by convention; leave them at 0.0.
        if m_idx >= cols {
            continue;
        }
        out[m_idx][m_idx] = pmm;
        if (m_idx + 1) < cols {
            // P_(m+1)^m(x) = (2m+1) x P_m^m(x)
            out[m_idx][m_idx + 1] = x * (2.0 * m as f64 + 1.0) * pmm;
            // Forward recurrence on l.
            for l in (m + 2)..=n_max {
                let lf = l as f64;
                let mf = m as f64;
                let prev = out[m_idx][l as usize - 2];
                let curr = out[m_idx][l as usize - 1];
                out[m_idx][l as usize] =
                    ((2.0 * (lf - 1.0) + 1.0) * x * curr - (lf - 1.0 + mf) * prev) / (lf - mf);
            }
        }
    }
    out
}

/// Evaluate the Chebyshev polynomial of the first kind T_n(x).
///
/// Uses the three-term recurrence:
///   T_0(x) = 1, T_1(x) = x,
///   T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
///
/// Identity: T_n(cos θ) = cos(nθ).
pub fn eval_chebyt(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut t_prev = 1.0;
    let mut t_curr = x;
    for _ in 1..n {
        let t_next = 2.0 * x * t_curr - t_prev;
        t_prev = t_curr;
        t_curr = t_next;
    }
    t_curr
}

/// Evaluate the Chebyshev polynomial of the second kind U_n(x).
///
/// Uses the three-term recurrence:
///   U_0(x) = 1, U_1(x) = 2x,
///   U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
pub fn eval_chebyu(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }
    let mut u_prev = 1.0;
    let mut u_curr = 2.0 * x;
    for _ in 1..n {
        let u_next = 2.0 * x * u_curr - u_prev;
        u_prev = u_curr;
        u_curr = u_next;
    }
    u_curr
}

/// Evaluate the Chebyshev polynomial of the first kind C_n on [-2, 2].
///
/// Matches `scipy.special.eval_chebyc(n, x)`. Defined by C_n(x) = 2·T_n(x/2),
/// so it inherits the three-term recurrence
///   C_0(x) = 2, C_1(x) = x,
///   C_{n+1}(x) = x C_n(x) - C_{n-1}(x).
/// Resolves [frankenscipy-1qax5].
pub fn eval_chebyc(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 2.0;
    }
    if n == 1 {
        return x;
    }
    let mut c_prev = 2.0;
    let mut c_curr = x;
    for _ in 1..n {
        let c_next = x * c_curr - c_prev;
        c_prev = c_curr;
        c_curr = c_next;
    }
    c_curr
}

/// Evaluate the Chebyshev polynomial of the second kind S_n on [-2, 2].
///
/// Matches `scipy.special.eval_chebys(n, x)`. Defined by S_n(x) = U_n(x/2),
/// so it satisfies
///   S_0(x) = 1, S_1(x) = x,
///   S_{n+1}(x) = x S_n(x) - S_{n-1}(x).
/// Resolves [frankenscipy-1qax5].
pub fn eval_chebys(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut s_prev = 1.0;
    let mut s_curr = x;
    for _ in 1..n {
        let s_next = x * s_curr - s_prev;
        s_prev = s_curr;
        s_curr = s_next;
    }
    s_curr
}

/// Evaluate the physicist's Hermite polynomial H_n(x).
///
/// Uses the three-term recurrence:
///   H_0(x) = 1, H_1(x) = 2x,
///   H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
///
/// Weight function: exp(-x²). Used in quantum mechanics.
pub fn eval_hermite(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }
    let mut h_prev = 1.0;
    let mut h_curr = 2.0 * x;
    for k in 1..n {
        let h_next = 2.0 * x * h_curr - 2.0 * k as f64 * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    h_curr
}

/// Evaluate the probabilist's Hermite polynomial He_n(x).
///
/// Uses the three-term recurrence:
///   He_0(x) = 1, He_1(x) = x,
///   He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)
///
/// Weight function: exp(-x²/2). Used in probability/statistics.
pub fn eval_hermitenorm(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut h_prev = 1.0;
    let mut h_curr = x;
    for k in 1..n {
        let h_next = x * h_curr - k as f64 * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    h_curr
}

/// Evaluate the Laguerre polynomial L_n(x).
///
/// Uses the three-term recurrence:
///   L_0(x) = 1, L_1(x) = 1 - x,
///   (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
///
/// Weight function: exp(-x) on [0, ∞).
pub fn eval_laguerre(n: u32, x: f64) -> f64 {
    eval_genlaguerre(n, 0.0, x)
}

/// Evaluate the generalized Laguerre polynomial L_n^α(x).
///
/// Uses the three-term recurrence:
///   L_0^α(x) = 1, L_1^α(x) = 1 + α - x,
///   (n+1) L_{n+1}^α(x) = (2n+1+α-x) L_n^α(x) - (n+α) L_{n-1}^α(x)
///
/// Weight function: x^α exp(-x) on [0, ∞).
pub fn eval_genlaguerre(n: u32, alpha: f64, x: f64) -> f64 {
    if n == 0 {
        if alpha.is_nan() || alpha == f64::NEG_INFINITY || x.is_nan() {
            return f64::NAN;
        }
        return 1.0;
    }
    if !alpha.is_finite() || !x.is_finite() {
        return f64::NAN;
    }
    if n == 1 {
        return 1.0 + alpha - x;
    }
    let mut l_prev = 1.0;
    let mut l_curr = 1.0 + alpha - x;
    for k in 1..n {
        let kf = k as f64;
        let l_next = ((2.0 * kf + 1.0 + alpha - x) * l_curr - (kf + alpha) * l_prev) / (kf + 1.0);
        l_prev = l_curr;
        l_curr = l_next;
    }
    l_curr
}

/// Generalized Laguerre with the legacy scipy argument order
/// (x first, then n, then k). Matches `scipy.special.assoc_laguerre(x, n, k=0.0)`.
/// Routes through `eval_genlaguerre(n, k, x)` — the entry point exists
/// for callers that reach for the historical name.
/// Resolves [frankenscipy-fdefl].
pub fn assoc_laguerre(x: f64, n: u32, k: f64) -> f64 {
    eval_genlaguerre(n, k, x)
}

/// Evaluate the Jacobi polynomial P_n^{α,β}(x).
///
/// The most general classical orthogonal polynomial on [-1, 1].
/// Weight function: (1-x)^α (1+x)^β.
///
/// Special cases:
/// - α = β = 0: Legendre P_n(x)
/// - α = β = -1/2: Chebyshev T_n(x) (up to normalization)
/// - α = β: Gegenbauer C_n^{α+1/2}(x) (up to normalization)
///
/// Uses the three-term recurrence (DLMF 18.9.2).
pub fn eval_jacobi(n: u32, alpha: f64, beta: f64, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if !alpha.is_finite() || !beta.is_finite() || !x.is_finite() {
        return f64::NAN;
    }
    if n == 1 {
        return 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x);
    }
    let mut p_prev = 1.0;
    let mut p_curr = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x);
    for k in 1..n {
        let kf = k as f64;
        let ab = alpha + beta;
        let two_k_ab = 2.0 * kf + ab;

        // Recurrence coefficients (DLMF 18.9.2)
        let a1 = 2.0 * (kf + 1.0) * (kf + ab + 1.0) * (two_k_ab);
        let a2 = (two_k_ab + 1.0) * (alpha * alpha - beta * beta);
        let a3 = (two_k_ab) * (two_k_ab + 1.0) * (two_k_ab + 2.0);
        let a4 = 2.0 * (kf + alpha) * (kf + beta) * (two_k_ab + 2.0);

        let p_next = if a1.abs() > 1e-30 {
            ((a2 + a3 * x) * p_curr - a4 * p_prev) / a1
        } else {
            0.0
        };

        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

/// Evaluate the Gegenbauer (ultraspherical) polynomial C_n^α(x).
///
/// Uses the three-term recurrence:
///   C_0^α(x) = 1, C_1^α(x) = 2αx,
///   (n+1) C_{n+1}^α(x) = 2(n+α) x C_n^α(x) - (n+2α-1) C_{n-1}^α(x)
///
/// Weight function: (1-x²)^{α-1/2} on [-1, 1].
/// Requires α > -1/2, α ≠ 0.
pub fn eval_gegenbauer(n: u32, alpha: f64, x: f64) -> f64 {
    if n == 0 {
        if alpha.is_nan() || x.is_nan() {
            return f64::NAN;
        }
        return 1.0;
    }
    if !alpha.is_finite() || !x.is_finite() {
        return f64::NAN;
    }
    if n == 1 {
        return 2.0 * alpha * x;
    }
    let mut c_prev = 1.0;
    let mut c_curr = 2.0 * alpha * x;
    for k in 1..n {
        let kf = k as f64;
        let c_next =
            (2.0 * (kf + alpha) * x * c_curr - (kf + 2.0 * alpha - 1.0) * c_prev) / (kf + 1.0);
        c_prev = c_curr;
        c_curr = c_next;
    }
    c_curr
}

/// Legendre polynomial of the first kind `P_n(z)` together with its derivatives
/// up to order `diff_n`.
///
/// Returns `[P_n(z), P_n'(z), …, P_n^(diff_n)(z)]` (length `diff_n + 1`),
/// matching `scipy.special.legendre_p(n, z, diff_n=diff_n)`. The `m`-th
/// derivative uses the exact, everywhere-stable Gegenbauer identity
/// `P_n^(m)(z) = (2m−1)!! · C_{n−m}^{m+1/2}(z)` (zero for `m > n`), so it is
/// valid for all `z`, including `|z| = 1` where the ODE recurrence is singular.
/// SciPy only implements `diff_n ∈ {0, 1, 2}`; this accepts any order.
#[must_use]
pub fn legendre_p(n: u32, z: f64, diff_n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(diff_n + 1);
    let mut double_factorial = 1.0_f64; // (2m−1)!! accumulated across m
    for m in 0..=diff_n {
        if m > 0 {
            double_factorial *= (2 * m - 1) as f64;
        }
        if (m as u32) > n {
            out.push(0.0);
        } else {
            out.push(double_factorial * eval_gegenbauer(n - m as u32, m as f64 + 0.5, z));
        }
    }
    out
}

/// All Legendre polynomials of the first kind up to degree `n`, with all
/// derivatives up to order `diff_n`.
///
/// Returns a `(diff_n + 1) × (n + 1)` table where entry `[i][j]` is the `i`-th
/// derivative of the degree-`j` Legendre polynomial at `z`, matching
/// `scipy.special.legendre_p_all(n, z, diff_n=diff_n)` (output shape
/// `(diff_n + 1, n + 1)`). Each column is one [`legendre_p`] evaluation, so it
/// inherits the everywhere-stable Gegenbauer derivative identity.
#[must_use]
pub fn legendre_p_all(n: u32, z: f64, diff_n: usize) -> Vec<Vec<f64>> {
    let mut table = vec![vec![0.0_f64; n as usize + 1]; diff_n + 1];
    for j in 0..=n {
        let col = legendre_p(j, z, diff_n);
        for (i, &v) in col.iter().enumerate() {
            table[i][j as usize] = v;
        }
    }
    table
}

/// `(n − |m|)! / (n + |m|)!`, evaluated as `1 / ∏_{k=n−|m|+1}^{n+|m|} k` to stay
/// finite for large `n`. Requires `|m| ≤ n`.
fn assoc_inverse_factorial_ratio(n: u32, abs_m: u32) -> f64 {
    let mut denom = 1.0_f64;
    for k in (n - abs_m + 1)..=(n + abs_m) {
        denom *= f64::from(k);
    }
    1.0 / denom
}

/// Associated Legendre function of the first kind `P_n^m(z)`, optionally
/// normalized, matching `scipy.special.assoc_legendre_p(n, m, z, norm=norm)`
/// (the default `branch_cut=2`, `diff_n=0` case for real `z`).
///
/// With `norm = false` this is exactly [`lpmv`]`(m, n, z)`. With `norm = true`
/// it carries the additional factor `√((2n+1)/2 · (n−|m|)!/(n+|m|)!)`, and for
/// `m < 0` the sign `(−1)^|m|` relative to the `|m|` value (SciPy's convention).
/// `|m| > n` gives `0`.
#[must_use]
pub fn assoc_legendre_p(n: u32, m: i32, z: f64, norm: bool) -> f64 {
    if !norm {
        return lpmv(m, n, z);
    }
    let abs_m = m.unsigned_abs();
    if abs_m > n {
        return 0.0;
    }
    let factor =
        ((2.0 * f64::from(n) + 1.0) / 2.0 * assoc_inverse_factorial_ratio(n, abs_m)).sqrt();
    let base = factor * lpmv(abs_m as i32, n, z);
    if m < 0 && abs_m % 2 == 1 {
        -base
    } else {
        base
    }
}

/// Spherical Legendre function `Y_n^m`-style normalized associated Legendre
/// polynomial evaluated at the polar angle `theta`, matching
/// `scipy.special.sph_legendre_p(n, m, theta)` (the `diff_n = 0` case).
///
/// Equals `√((2n+1)/(4π) · (n−|m|)!/(n+|m|)!) · P_|m|^n(cos θ)` with the
/// `(−1)^|m|` sign for `m < 0`, the angular factor used to build real spherical
/// harmonics. `|m| > n` gives `0`.
#[must_use]
pub fn sph_legendre_p(n: u32, m: i32, theta: f64) -> f64 {
    let abs_m = m.unsigned_abs();
    if abs_m > n {
        return 0.0;
    }
    let factor = ((2.0 * f64::from(n) + 1.0) / (4.0 * PI)
        * assoc_inverse_factorial_ratio(n, abs_m))
    .sqrt();
    let base = factor * lpmv(abs_m as i32, n, theta.cos());
    if m < 0 && abs_m % 2 == 1 {
        -base
    } else {
        base
    }
}

/// Ascending eigenvalues of a symmetric tridiagonal matrix (the Mathieu
/// characteristic-value recurrence matrices), reusing the Golub–Welsch QL.
fn symmetric_tridiagonal_eigenvalues(diagonal: &[f64], offdiagonal: &[f64]) -> Vec<f64> {
    let (mut values, _) = gw_tridiagonal_eigen_first_row(diagonal, offdiagonal)
        .unwrap_or_else(|| (vec![f64::NAN; diagonal.len()], Vec::new()));
    values.sort_by(f64::total_cmp);
    values
}

/// Fourier-mode count for the Mathieu recurrence matrix: large enough that the
/// `m`-th characteristic value has converged for non-centrality `q`.
fn mathieu_matrix_dim(m: u32, q: f64) -> usize {
    m as usize + 2 * (q.abs() + 1.0).sqrt().ceil() as usize + 40
}

/// Characteristic value `a_m(q)` of the even-periodic Mathieu functions `ce_m`.
///
/// Matches `scipy.special.mathieu_a(m, q)`. Computed as the `⌊m/2⌋`-th smallest
/// eigenvalue of the symmetric tridiagonal Fourier-recurrence matrix (DLMF 28.4):
/// for even `m`, diagonal `(2k)²` with first off-diagonal `q√2`; for odd `m`,
/// diagonal `(2k+1)²` with `d₀ = 1 + q`; all other off-diagonals `q`.
#[must_use]
pub fn mathieu_a(m: u32, q: f64) -> f64 {
    let n = mathieu_matrix_dim(m, q);
    if m % 2 == 0 {
        let diag: Vec<f64> = (0..n).map(|k| (2 * k as i64).pow(2) as f64).collect();
        let mut off = vec![q; n - 1];
        off[0] = q * std::f64::consts::SQRT_2;
        symmetric_tridiagonal_eigenvalues(&diag, &off)[m as usize / 2]
    } else {
        let mut diag: Vec<f64> = (0..n).map(|k| (2 * k as i64 + 1).pow(2) as f64).collect();
        diag[0] = 1.0 + q;
        let off = vec![q; n - 1];
        symmetric_tridiagonal_eigenvalues(&diag, &off)[(m as usize - 1) / 2]
    }
}

/// Characteristic value `b_m(q)` of the odd-periodic Mathieu functions `se_m`
/// (`m ≥ 1`).
///
/// Matches `scipy.special.mathieu_b(m, q)`. The `⌊(m−1)/2⌋`-th (odd `m`) or
/// `m/2−1`-th (even `m`) smallest eigenvalue of the symmetric tridiagonal
/// recurrence matrix (DLMF 28.4): for even `m`, diagonal `(2k+2)²`; for odd `m`,
/// diagonal `(2k+1)²` with `d₀ = 1 − q`; all off-diagonals `q`. `m = 0` is
/// undefined and returns NaN, as SciPy does.
#[must_use]
pub fn mathieu_b(m: u32, q: f64) -> f64 {
    if m == 0 {
        return f64::NAN;
    }
    let n = mathieu_matrix_dim(m, q);
    if m % 2 == 0 {
        let diag: Vec<f64> = (0..n).map(|k| (2 * k as i64 + 2).pow(2) as f64).collect();
        let off = vec![q; n - 1];
        symmetric_tridiagonal_eigenvalues(&diag, &off)[m as usize / 2 - 1]
    } else {
        let mut diag: Vec<f64> = (0..n).map(|k| (2 * k as i64 + 1).pow(2) as f64).collect();
        diag[0] = 1.0 - q;
        let off = vec![q; n - 1];
        symmetric_tridiagonal_eigenvalues(&diag, &off)[(m as usize - 1) / 2]
    }
}

/// Thomas algorithm: solve the tridiagonal system with sub-diagonal `sub`,
/// diagonal `diag`, super-diagonal `sup`, and right-hand side `rhs`.
fn thomas_solve(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let mut c = vec![0.0_f64; n];
    let mut d = vec![0.0_f64; n];
    c[0] = sup[0] / diag[0];
    d[0] = rhs[0] / diag[0];
    for i in 1..n {
        let denom = diag[i] - sub[i] * c[i - 1];
        if i < n - 1 {
            c[i] = sup[i] / denom;
        }
        d[i] = (rhs[i] - sub[i] * d[i - 1]) / denom;
    }
    let mut x = vec![0.0_f64; n];
    x[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d[i] - c[i] * x[i + 1];
    }
    x
}

/// Unit eigenvector of a symmetric tridiagonal matrix for the known eigenvalue
/// `lambda`, by inverse iteration with a tiny shift off the eigenvalue.
fn tridiagonal_eigenvector(diag: &[f64], off: &[f64], lambda: f64) -> Vec<f64> {
    let n = diag.len();
    let mu = lambda + (lambda.abs() + 1.0) * 1e-11;
    let sub: Vec<f64> = std::iter::once(0.0).chain(off.iter().copied()).collect();
    let sup: Vec<f64> = off.iter().copied().chain(std::iter::once(0.0)).collect();
    let shifted: Vec<f64> = diag.iter().map(|d| d - mu).collect();
    let mut x = vec![1.0_f64; n];
    for _ in 0..3 {
        let y = thomas_solve(&sub, &shifted, &sup, &x);
        let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        for (xi, yi) in x.iter_mut().zip(y.iter()) {
            *xi = yi / norm;
        }
    }
    x
}

/// Fourier coefficients of a periodic Mathieu function and the harmonic offset
/// `k0` (harmonics are `k0 + 2i`). `even = true` builds `ce_m` (cosine series,
/// using `mathieu_a`), `even = false` builds `se_m` (sine series, `mathieu_b`).
/// The coefficient vector is the unit eigenvector of the same recurrence matrix
/// as the characteristic value, with SciPy's sign convention (the `k = m`
/// harmonic coefficient is positive).
fn mathieu_fourier(m: u32, q: f64, even: bool) -> (Vec<f64>, u32) {
    let n = mathieu_matrix_dim(m, q);
    let (diag, off, lambda, k0, undo_sqrt2): (Vec<f64>, Vec<f64>, f64, u32, bool) = if even {
        if m % 2 == 0 {
            let diag = (0..n).map(|k| (2 * k as i64).pow(2) as f64).collect();
            let mut off = vec![q; n - 1];
            off[0] = q * std::f64::consts::SQRT_2;
            (diag, off, mathieu_a(m, q), 0, true)
        } else {
            let mut diag: Vec<f64> = (0..n).map(|k| (2 * k as i64 + 1).pow(2) as f64).collect();
            diag[0] = 1.0 + q;
            (diag, vec![q; n - 1], mathieu_a(m, q), 1, false)
        }
    } else if m % 2 == 0 {
        let diag = (0..n).map(|k| (2 * k as i64 + 2).pow(2) as f64).collect();
        (diag, vec![q; n - 1], mathieu_b(m, q), 2, false)
    } else {
        let mut diag: Vec<f64> = (0..n).map(|k| (2 * k as i64 + 1).pow(2) as f64).collect();
        diag[0] = 1.0 - q;
        (diag, vec![q; n - 1], mathieu_b(m, q), 1, false)
    };
    let mut v = tridiagonal_eigenvector(&diag, &off, lambda);
    if undo_sqrt2 {
        v[0] /= std::f64::consts::SQRT_2;
    }
    // Fix the global sign with the DLMF convention: ce_m(0,q) = Σ A_k > 0 for the
    // cosine series, and se_m'(0,q) = Σ k·B_k > 0 for the sine series.
    let sign_quantity: f64 = if even {
        v.iter().sum()
    } else {
        v.iter()
            .enumerate()
            .map(|(i, &vi)| f64::from(k0 + 2 * i as u32) * vi)
            .sum()
    };
    if sign_quantity < 0.0 {
        for value in &mut v {
            *value = -*value;
        }
    }
    (v, k0)
}

/// Even periodic Mathieu function `ce_m(x, q)` and its derivative (w.r.t. the
/// argument in radians), matching `scipy.special.mathieu_cem(m, q, x)`.
///
/// `x` is given in **degrees**, as SciPy requires. The function is the cosine
/// Fourier series `ce_m(x) = Σ A_k cos(k x)` whose coefficients are obtained
/// from [`mathieu_fourier`]; the returned derivative is `d ce_m / dx` with `x`
/// in radians.
#[must_use]
pub fn mathieu_cem(m: u32, q: f64, x: f64) -> (f64, f64) {
    let (a, k0) = mathieu_fourier(m, q, true);
    let xr = x.to_radians();
    let mut value = 0.0_f64;
    let mut derivative = 0.0_f64;
    for (i, &ai) in a.iter().enumerate() {
        let k = f64::from(k0 + 2 * i as u32);
        value += ai * (k * xr).cos();
        derivative -= ai * k * (k * xr).sin();
    }
    (value, derivative)
}

/// Odd periodic Mathieu function `se_m(x, q)` (`m ≥ 1`) and its derivative
/// (w.r.t. the argument in radians), matching `scipy.special.mathieu_sem(m, q, x)`.
///
/// `x` is given in **degrees**. The function is the sine Fourier series
/// `se_m(x) = Σ B_k sin(k x)`; `se_0 ≡ 0`, so `m = 0` returns `(0, 0)` as SciPy
/// does.
#[must_use]
pub fn mathieu_sem(m: u32, q: f64, x: f64) -> (f64, f64) {
    if m == 0 {
        return (0.0, 0.0);
    }
    let (b, k0) = mathieu_fourier(m, q, false);
    let xr = x.to_radians();
    let mut value = 0.0_f64;
    let mut derivative = 0.0_f64;
    for (i, &bi) in b.iter().enumerate() {
        let k = f64::from(k0 + 2 * i as u32);
        value += bi * (k * xr).sin();
        derivative += bi * k * (k * xr).cos();
    }
    (value, derivative)
}

/// Bessel `J_n(u)` and its derivative `J_n'(u) = (J_{n-1}(u) − J_{n+1}(u))/2`.
fn bessel_j_and_deriv(n: u32, u: f64) -> (f64, f64) {
    let j = jv_scalar(f64::from(n), u);
    let deriv = if n == 0 {
        -jv_scalar(1.0, u)
    } else {
        0.5 * (jv_scalar(f64::from(n) - 1.0, u) - jv_scalar(f64::from(n) + 1.0, u))
    };
    (j, deriv)
}

/// `J_a(u1) J_b(u2)` and its `z`-derivative, with `u1 = √q e^{-z}`,
/// `u2 = √q e^{z}` (so `du1/dz = -u1`, `du2/dz = u2`).
fn bessel_product(a: u32, b: u32, u1: f64, u2: f64) -> (f64, f64) {
    let (ja, ja_p) = bessel_j_and_deriv(a, u1);
    let (jb, jb_p) = bessel_j_and_deriv(b, u2);
    (ja * jb, -u1 * ja_p * jb + u2 * ja * jb_p)
}

/// Modified (radial) Mathieu function of the first kind and its `z`-derivative,
/// from the Bessel-product series with the angular Fourier coefficients.
/// `even` selects `Mc` (cosine/`ce` coefficients) vs `Ms` (sine/`se`).
fn mathieu_mod1(m: u32, q: f64, z: f64, even: bool) -> (f64, f64) {
    let (coeffs, _) = mathieu_fourier(m, q, even);
    let sq = q.sqrt();
    let u1 = sq * (-z).exp();
    let u2 = sq * z.exp();
    let mut value = 0.0_f64;
    let mut deriv = 0.0_f64;
    for (i, &ci) in coeffs.iter().enumerate() {
        let iu = i as u32;
        let alt = if i % 2 == 0 { 1.0 } else { -1.0 };
        let (p, d) = if even {
            if m % 2 == 0 {
                // Mc_{2n}: Σ (-1)^i A_i J_i(u1) J_i(u2)
                bessel_product(iu, iu, u1, u2)
            } else {
                // Mc_{2n+1}: Σ (-1)^i A_i [J_i J_{i+1} + J_{i+1} J_i]
                let (p1, d1) = bessel_product(iu, iu + 1, u1, u2);
                let (p2, d2) = bessel_product(iu + 1, iu, u1, u2);
                (p1 + p2, d1 + d2)
            }
        } else if m % 2 == 1 {
            // Ms_{2n+1}: Σ (-1)^i B_i [J_i J_{i+1} − J_{i+1} J_i]
            let (p1, d1) = bessel_product(iu, iu + 1, u1, u2);
            let (p2, d2) = bessel_product(iu + 1, iu, u1, u2);
            (p1 - p2, d1 - d2)
        } else {
            // Ms_{2n+2}: Σ (-1)^i B_i [J_i J_{i+2} − J_{i+2} J_i]
            let (p1, d1) = bessel_product(iu, iu + 2, u1, u2);
            let (p2, d2) = bessel_product(iu + 2, iu, u1, u2);
            (p1 - p2, d1 - d2)
        };
        value += alt * ci * p;
        deriv += alt * ci * d;
    }
    // Series-sign factor (-1)^n from the m = 2n / 2n+1 / 2n+2 parameterization,
    // divided by the leading angular coefficient.
    let n_index = if even { m / 2 } else { (m - 1) / 2 };
    let sign = if n_index % 2 == 0 { 1.0 } else { -1.0 };
    let scale = sign / coeffs[0];
    (value * scale, deriv * scale)
}

/// Even modified Mathieu function of the first kind `Mc1_m(x, q)` and its
/// derivative w.r.t. `x`, matching `scipy.special.mathieu_modcem1(m, q, x)`.
#[must_use]
pub fn mathieu_modcem1(m: u32, q: f64, x: f64) -> (f64, f64) {
    mathieu_mod1(m, q, x, true)
}

/// Odd modified Mathieu function of the first kind `Ms1_m(x, q)` (`m ≥ 1`) and
/// its derivative w.r.t. `x`, matching `scipy.special.mathieu_modsem1(m, q, x)`.
#[must_use]
pub fn mathieu_modsem1(m: u32, q: f64, x: f64) -> (f64, f64) {
    mathieu_mod1(m, q, x, false)
}

/// Characteristic value of a spheroidal wave function of order `m`, `n` (`n ≥ m`)
/// and parameter `c`; `prolate` selects `c²` (prolate) vs `−c²` (oblate).
///
/// The expansion coefficients of the spheroidal angular function obey a
/// three-term recurrence (DLMF 30.8): `A_r d_{r+2} + (B_r − λ) d_r + C_r d_{r-2}
/// = 0` over `r` of the same parity as `n − m`. The characteristic value `λ` is
/// the `⌊(n−m)/2⌋`-th eigenvalue of that (symmetrized) tridiagonal matrix. The
/// symmetric off-diagonal `√(A_{r_k} C_{r_{k+1}})` is real because both factors
/// carry one power of `±c²`, so their product is `∝ c⁴ ≥ 0`.
fn spheroidal_cv(m: u32, n: u32, c: f64, prolate: bool) -> f64 {
    if n < m {
        return f64::NAN;
    }
    let cc = if prolate { c * c } else { -c * c };
    let mf = f64::from(m);
    let parity = (n - m) % 2;
    let dim = (n - m) as usize / 2 + 2 * c.abs().ceil() as usize + 50;
    let r_of = |k: usize| (parity as usize + 2 * k) as f64;
    let a_coef = |r: f64| {
        (2.0 * mf + r + 2.0) * (2.0 * mf + r + 1.0)
            / ((2.0 * mf + 2.0 * r + 3.0) * (2.0 * mf + 2.0 * r + 5.0))
            * cc
    };
    let b_coef = |r: f64| {
        (mf + r) * (mf + r + 1.0)
            + (2.0 * (mf + r) * (mf + r + 1.0) - 2.0 * mf * mf - 1.0)
                / ((2.0 * mf + 2.0 * r - 1.0) * (2.0 * mf + 2.0 * r + 3.0))
                * cc
    };
    let c_coef = |r: f64| {
        r * (r - 1.0) / ((2.0 * mf + 2.0 * r - 3.0) * (2.0 * mf + 2.0 * r - 1.0)) * cc
    };
    let diag: Vec<f64> = (0..dim).map(|k| b_coef(r_of(k))).collect();
    let off: Vec<f64> = (0..dim - 1)
        .map(|k| (a_coef(r_of(k)) * c_coef(r_of(k + 1))).sqrt())
        .collect();
    symmetric_tridiagonal_eigenvalues(&diag, &off)[(n - m) as usize / 2]
}

/// Characteristic value of the prolate spheroidal wave functions of order `m`,
/// `n` (`n ≥ m`) and parameter `c`. Matches `scipy.special.pro_cv(m, n, c)`.
/// `n < m` returns NaN, as SciPy does.
#[must_use]
pub fn pro_cv(m: u32, n: u32, c: f64) -> f64 {
    spheroidal_cv(m, n, c, true)
}

/// Characteristic value of the oblate spheroidal wave functions of order `m`,
/// `n` (`n ≥ m`) and parameter `c`. Matches `scipy.special.obl_cv(m, n, c)`.
/// `n < m` returns NaN, as SciPy does.
#[must_use]
pub fn obl_cv(m: u32, n: u32, c: f64) -> f64 {
    spheroidal_cv(m, n, c, false)
}

/// Associated Legendre `P_l^m(x)` in the spheroidal (no Condon–Shortley phase)
/// convention: `(−1)^m · lpmv(m, l, x)`.
fn assoc_legendre_no_cs(m: u32, l: u32, x: f64) -> f64 {
    let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
    sign * lpmv(m as i32, l, x)
}

/// Derivative w.r.t. `x` of [`assoc_legendre_no_cs`] for `|x| < 1`, via
/// `dP_l^m/dx = [l x P_l^m − (l+m) P_{l-1}^m] / (x²−1)`.
fn assoc_legendre_no_cs_deriv(m: u32, l: u32, x: f64) -> f64 {
    if l == 0 {
        return 0.0;
    }
    let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
    let (lf, mf) = (f64::from(l), f64::from(m));
    sign * (lf * x * lpmv(m as i32, l, x) - (lf + mf) * lpmv(m as i32, l - 1, x)) / (x * x - 1.0)
}

/// Unit eigenvector of a (possibly non-symmetric) tridiagonal matrix with
/// sub-diagonal `sub`, diagonal `diag`, super-diagonal `sup` for the known
/// eigenvalue `lambda`, by inverse iteration.
fn tridiagonal_eigenvector_nonsym(sub: &[f64], diag: &[f64], sup: &[f64], lambda: f64) -> Vec<f64> {
    let n = diag.len();
    let mu = lambda + (lambda.abs() + 1.0) * 1e-11;
    let shifted: Vec<f64> = diag.iter().map(|d| d - mu).collect();
    let mut x = vec![1.0_f64; n];
    for _ in 0..4 {
        let y = thomas_solve(sub, &shifted, sup, &x);
        let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        for (xi, yi) in x.iter_mut().zip(y.iter()) {
            *xi = yi / norm;
        }
    }
    x
}

/// Flammer-normalized expansion coefficients `d_r` of the spheroidal angular
/// function of order `m`, `n` and parameter `c` (`prolate` selects `±c²`), with
/// the harmonic for index `k` being `P_{m + (n−m mod 2) + 2k}^m`.
///
/// The `d_r` are the eigenvector of the non-symmetric DLMF 30.8 recurrence for
/// the given characteristic value `cv`, scaled so the angular function reduces to
/// the associated Legendre `P_n^m` at `x = 0` (its value for `n − m` even, its
/// derivative for `n − m` odd) — SciPy's Flammer convention.
fn spheroidal_coefficients(m: u32, n: u32, c: f64, prolate: bool, cv: f64) -> (Vec<f64>, u32) {
    let cc = if prolate { c * c } else { -c * c };
    let mf = f64::from(m);
    let parity = (n - m) % 2;
    let dim = (n - m) as usize / 2 + 2 * c.abs().ceil() as usize + 50;
    let r_of = |k: usize| (parity as usize + 2 * k) as f64;
    let a_coef = |r: f64| {
        (2.0 * mf + r + 2.0) * (2.0 * mf + r + 1.0)
            / ((2.0 * mf + 2.0 * r + 3.0) * (2.0 * mf + 2.0 * r + 5.0))
            * cc
    };
    let b_coef = |r: f64| {
        (mf + r) * (mf + r + 1.0)
            + (2.0 * (mf + r) * (mf + r + 1.0) - 2.0 * mf * mf - 1.0)
                / ((2.0 * mf + 2.0 * r - 1.0) * (2.0 * mf + 2.0 * r + 3.0))
                * cc
    };
    let c_coef = |r: f64| {
        r * (r - 1.0) / ((2.0 * mf + 2.0 * r - 3.0) * (2.0 * mf + 2.0 * r - 1.0)) * cc
    };
    let diag: Vec<f64> = (0..dim).map(|k| b_coef(r_of(k))).collect();
    let sub: Vec<f64> = (0..dim).map(|k| c_coef(r_of(k))).collect();
    let sup: Vec<f64> = (0..dim).map(|k| a_coef(r_of(k))).collect();
    let mut d = tridiagonal_eigenvector_nonsym(&sub, &diag, &sup, cv);

    // Flammer normalization: scale so the function (n−m even) or its derivative
    // (n−m odd) matches the associated Legendre P_n^m at x = 0.
    let degree = |k: usize| m + parity + 2 * k as u32;
    let (raw, target) = if parity == 0 {
        (
            (0..dim)
                .map(|k| d[k] * assoc_legendre_no_cs(m, degree(k), 0.0))
                .sum::<f64>(),
            assoc_legendre_no_cs(m, n, 0.0),
        )
    } else {
        (
            (0..dim)
                .map(|k| d[k] * assoc_legendre_no_cs_deriv(m, degree(k), 0.0))
                .sum::<f64>(),
            assoc_legendre_no_cs_deriv(m, n, 0.0),
        )
    };
    let kappa = target / raw;
    for value in &mut d {
        *value *= kappa;
    }
    (d, parity)
}

/// Spheroidal angular function of the first kind and its derivative at `x`
/// (`|x| < 1`), given the characteristic value `cv`; `prolate` selects prolate
/// vs oblate.
fn spheroidal_ang1(m: u32, n: u32, c: f64, x: f64, prolate: bool, cv: f64) -> (f64, f64) {
    if n < m {
        return (f64::NAN, f64::NAN);
    }
    let (d, parity) = spheroidal_coefficients(m, n, c, prolate, cv);
    let mut value = 0.0_f64;
    let mut derivative = 0.0_f64;
    for (k, &dk) in d.iter().enumerate() {
        let l = m + parity + 2 * k as u32;
        value += dk * assoc_legendre_no_cs(m, l, x);
        derivative += dk * assoc_legendre_no_cs_deriv(m, l, x);
    }
    (value, derivative)
}

/// Prolate spheroidal angular function of the first kind `S_mn^{(1)}(c, x)` and
/// its derivative w.r.t. `x`, matching `scipy.special.pro_ang1(m, n, c, x)`
/// (`n ≥ m`, `|x| < 1`).
#[must_use]
pub fn pro_ang1(m: u32, n: u32, c: f64, x: f64) -> (f64, f64) {
    spheroidal_ang1(m, n, c, x, true, spheroidal_cv(m, n, c, true))
}

/// As [`pro_ang1`] but with a precomputed characteristic value `cv` (from
/// [`pro_cv`]), matching `scipy.special.pro_ang1_cv(m, n, c, cv, x)`.
#[must_use]
pub fn pro_ang1_cv(m: u32, n: u32, c: f64, cv: f64, x: f64) -> (f64, f64) {
    spheroidal_ang1(m, n, c, x, true, cv)
}

/// As [`obl_ang1`] but with a precomputed characteristic value `cv` (from
/// [`obl_cv`]), matching `scipy.special.obl_ang1_cv(m, n, c, cv, x)`.
#[must_use]
pub fn obl_ang1_cv(m: u32, n: u32, c: f64, cv: f64, x: f64) -> (f64, f64) {
    spheroidal_ang1(m, n, c, x, false, cv)
}

/// Oblate spheroidal angular function of the first kind `S_mn^{(1)}(c, x)` and
/// its derivative w.r.t. `x`, matching `scipy.special.obl_ang1(m, n, c, x)`
/// (`n ≥ m`, `|x| < 1`).
#[must_use]
pub fn obl_ang1(m: u32, n: u32, c: f64, x: f64) -> (f64, f64) {
    spheroidal_ang1(m, n, c, x, false, spheroidal_cv(m, n, c, false))
}

/// Spherical Bessel function of the first kind `j_l(z)` (scalar).
fn sph_jn(l: u32, z: f64) -> f64 {
    spherical_jn_scalar(f64::from(l), z, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Derivative `j_l'(z)` via `j_l'(z) = j_{l-1}(z) − (l+1)/z · j_l(z)`, with
/// `j_0'(z) = −j_1(z)`.
fn sph_jn_deriv(l: u32, z: f64) -> f64 {
    if l == 0 {
        -sph_jn(1, z)
    } else {
        sph_jn(l - 1, z) - f64::from(l + 1) / z * sph_jn(l, z)
    }
}

/// Spheroidal radial function of the first kind and its derivative w.r.t. `x`.
///
/// `R_mn^{(1)}(c, x) = (1 + s/x²)^{m/2} · Σ_k φ_k w_k d_k j_{m+r_k}(c x) /
/// Σ_k w_k d_k`, where `s = −1` (prolate, `x > 1`) or `s = +1` (oblate, `x > 0`),
/// `w_k = (2m+r_k)!/r_k!`, `φ_k = (−1)^{(r_k+m−n)/2}`, `r_k = (n−m mod 2) + 2k`,
/// `d_k` the Flammer angular coefficients, and `j_l` the spherical Bessel
/// function of the first kind (Flammer / DLMF 30.9).
fn spheroidal_rad1(m: u32, n: u32, c: f64, x: f64, prolate: bool, cv: f64) -> (f64, f64) {
    if n < m {
        return (f64::NAN, f64::NAN);
    }
    let (d, parity) = spheroidal_coefficients(m, n, c, prolate, cv);
    let s = if prolate { -1.0 } else { 1.0 };
    let z = c * x;
    // w_k = (2m+r_k)!/r_k! = Π_{j=1}^{2m} (r_k + j); φ_k = (−1)^{(r_k+m−n)/2}.
    let mut series = 0.0_f64;
    let mut series_deriv = 0.0_f64;
    let mut norm = 0.0_f64;
    for (k, &dk) in d.iter().enumerate() {
        let r = parity + 2 * k as u32;
        let l = m + r;
        let mut w = 1.0_f64;
        for j in 1..=2 * m {
            w *= f64::from(r + j);
        }
        // φ_k = (−1)^{(l−n)/2}; l−n is even, but can be negative, so use i64.
        let exponent = (i64::from(l) - i64::from(n)) / 2;
        let phase = if exponent.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
        let wd = w * dk;
        norm += wd;
        series += phase * wd * sph_jn(l, z);
        series_deriv += phase * wd * c * sph_jn_deriv(l, z);
    }
    let t = series / norm;
    let t_deriv = series_deriv / norm;
    let pref = (1.0 + s / (x * x)).powf(f64::from(m) / 2.0);
    let pref_deriv = if m == 0 {
        0.0
    } else {
        -f64::from(m) * s / (x * x * x) * (1.0 + s / (x * x)).powf(f64::from(m) / 2.0 - 1.0)
    };
    (pref * t, pref_deriv * t + pref * t_deriv)
}

/// Prolate spheroidal radial function of the first kind `R_mn^{(1)}(c, x)` and
/// its derivative w.r.t. `x`, matching `scipy.special.pro_rad1(m, n, c, x)`
/// (`n ≥ m`, `x > 1`).
#[must_use]
pub fn pro_rad1(m: u32, n: u32, c: f64, x: f64) -> (f64, f64) {
    spheroidal_rad1(m, n, c, x, true, spheroidal_cv(m, n, c, true))
}

/// As [`pro_rad1`] but with a precomputed characteristic value `cv` (from
/// [`pro_cv`]), matching `scipy.special.pro_rad1_cv(m, n, c, cv, x)`.
#[must_use]
pub fn pro_rad1_cv(m: u32, n: u32, c: f64, cv: f64, x: f64) -> (f64, f64) {
    spheroidal_rad1(m, n, c, x, true, cv)
}

/// Oblate spheroidal radial function of the first kind `R_mn^{(1)}(c, x)` and
/// its derivative w.r.t. `x`, matching `scipy.special.obl_rad1(m, n, c, x)`
/// (`n ≥ m`, `x > 0`).
#[must_use]
pub fn obl_rad1(m: u32, n: u32, c: f64, x: f64) -> (f64, f64) {
    spheroidal_rad1(m, n, c, x, false, spheroidal_cv(m, n, c, false))
}

/// As [`obl_rad1`] but with a precomputed characteristic value `cv` (from
/// [`obl_cv`]), matching `scipy.special.obl_rad1_cv(m, n, c, cv, x)`.
#[must_use]
pub fn obl_rad1_cv(m: u32, n: u32, c: f64, cv: f64, x: f64) -> (f64, f64) {
    spheroidal_rad1(m, n, c, x, false, cv)
}

/// Characteristic values of the prolate spheroidal wave functions for modes
/// `(m, m), (m, m+1), …, (m, n)`, matching `scipy.special.pro_cv_seq(m, n, c)`.
#[must_use]
pub fn pro_cv_seq(m: u32, n: u32, c: f64) -> Vec<f64> {
    (m..=n).map(|nn| pro_cv(m, nn, c)).collect()
}

/// Characteristic values of the oblate spheroidal wave functions for modes
/// `(m, m), (m, m+1), …, (m, n)`, matching `scipy.special.obl_cv_seq(m, n, c)`.
#[must_use]
pub fn obl_cv_seq(m: u32, n: u32, c: f64) -> Vec<f64> {
    (m..=n).map(|nn| obl_cv(m, nn, c)).collect()
}

/// Evaluate the shifted Legendre polynomial P_n*(x) = P_n(2x - 1).
///
/// The shifted Legendre polynomials are orthogonal on [0, 1] instead of [-1, 1].
///
/// Matches `scipy.special.eval_sh_legendre(n, x)`.
pub fn eval_sh_legendre(n: u32, x: f64) -> f64 {
    eval_legendre(n, 2.0 * x - 1.0)
}

/// Evaluate the shifted Chebyshev polynomial T_n*(x) = T_n(2x - 1).
///
/// The shifted Chebyshev polynomials are orthogonal on [0, 1] instead of [-1, 1].
///
/// Matches `scipy.special.eval_sh_chebyt(n, x)`.
pub fn eval_sh_chebyt(n: u32, x: f64) -> f64 {
    eval_chebyt(n, 2.0 * x - 1.0)
}

/// Evaluate the shifted Chebyshev polynomial U_n*(x) = U_n(2x - 1).
pub fn eval_sh_chebyu(n: u32, x: f64) -> f64 {
    eval_chebyu(n, 2.0 * x - 1.0)
}

/// Evaluate the shifted Jacobi polynomial G_n^{(p, q)}(x) on [0, 1].
///
/// Matches `scipy.special.eval_sh_jacobi(n, p, q, x)`, the shifted Jacobi
/// polynomial `G_n^{(p, q)}(x) = P_n^{(p − q, q − 1)}(2x − 1) / C(2n + p − 1, n)`.
/// The standard Jacobi value is computed by the well-tested `eval_jacobi`; the
/// binomial `C(2n + p − 1, n)` (a real-upper-index falling factorial over n!)
/// is the normalization scipy divides by — previously omitted, which left the
/// result off by that factor (up to ~3.8e6 at n = 12).
pub fn eval_sh_jacobi(n: u32, p: f64, q: f64, x: f64) -> f64 {
    // C(2n + p − 1, n) = Π_{j=0}^{n−1} (2n + p − 1 − j) / (j + 1), exact for
    // integer n and real upper index.
    let a = 2.0 * f64::from(n) + p - 1.0;
    let mut norm = 1.0_f64;
    for j in 0..n {
        norm *= (a - f64::from(j)) / (f64::from(j) + 1.0);
    }
    eval_jacobi(n, p - q, q - 1.0, 2.0 * x - 1.0) / norm
}

/// Compute Gauss-Legendre quadrature nodes and weights on [-1, 1].
#[must_use]
pub fn roots_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_jacobi(n, 0.0, 0.0)
}

/// Compute Gauss-Chebyshev quadrature nodes and weights on [-1, 1].
///
/// The weight function is `(1 - x^2)^(-1/2)`.
#[must_use]
pub fn roots_chebyt(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut nodes = Vec::with_capacity(n);
    let weight = PI / n as f64;
    let weights = vec![weight; n];

    for k in 0..n {
        let theta = PI * (2.0 * k as f64 + 1.0) / (2.0 * n as f64);
        nodes.push(theta.cos());
    }
    nodes.sort_by(|a, b| a.total_cmp(b));

    (nodes, weights)
}

/// Compute Gauss-Chebyshev (second kind) quadrature nodes and weights on [-1, 1].
///
/// The weight function is `sqrt(1 - x^2)`.
/// The nodes are the roots of U_n(x), the Chebyshev polynomial of the second kind.
///
/// Matches `scipy.special.roots_chebyu(n)`.
#[must_use]
pub fn roots_chebyu(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let n1 = n as f64 + 1.0;
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);

    for k in 1..=n {
        let theta = PI * k as f64 / n1;
        nodes.push(theta.cos());
        // Weight for Chebyshev U: w_k = π/(n+1) * sin²(kπ/(n+1))
        let sin_theta = theta.sin();
        weights.push(PI / n1 * sin_theta * sin_theta);
    }

    // Sort nodes in ascending order (they come out descending from cos)
    // Also reorder weights accordingly
    let mut pairs: Vec<(f64, f64)> = nodes.into_iter().zip(weights).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (sorted_nodes, sorted_weights): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    (sorted_nodes, sorted_weights)
}

/// Compute Gauss-Chebyshev (first kind, scaled) quadrature nodes and weights
/// on `[-2, 2]`. The polynomials `C_n(x) = 2 T_n(x/2)` are orthogonal on
/// this interval with weight `1 / sqrt(1 − x²/4)`.
///
/// Roots: `x_k = 2 cos((2k+1)π / (2n))` for `k ∈ 0..n` (the T-roots scaled by 2).
/// Weights: `2π / n` (uniform) — the weight function integrates to `2π` over
/// `[-2, 2]`, matching scipy.special.roots_chebyc.
#[must_use]
pub fn roots_chebyc(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut nodes = Vec::with_capacity(n);
    let weights = vec![2.0 * PI / n as f64; n];
    for k in 0..n {
        let theta = PI * (2.0 * k as f64 + 1.0) / (2.0 * n as f64);
        nodes.push(2.0 * theta.cos());
    }
    nodes.sort_by(|a, b| a.total_cmp(b));
    (nodes, weights)
}

/// Compute Gauss-Chebyshev (second kind, scaled) quadrature nodes and weights
/// on `[-2, 2]`. The polynomials `S_n(x) = U_n(x/2)` are orthogonal on this
/// interval with weight `sqrt(1 − x²/4)`.
///
/// Roots: `x_k = 2 cos((k+1)π / (n+1))` (the U-roots scaled by 2).
/// Weights: `(2π/(n+1)) · sin²((k+1)π/(n+1))` — twice the Chebyshev-U weight,
/// since the `[-2, 2]` interval doubles the integral, matching
/// scipy.special.roots_chebys.
#[must_use]
pub fn roots_chebys(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let n1 = n as f64 + 1.0;
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n);
    for k in 1..=n {
        let theta = PI * k as f64 / n1;
        let sin_theta = theta.sin();
        let weight = 2.0 * PI / n1 * sin_theta * sin_theta;
        pairs.push((2.0 * theta.cos(), weight));
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (nodes, weights): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
    (nodes, weights)
}

/// Compute Gauss-Legendre quadrature nodes and weights on `[0, 1]` via the
/// shifted Legendre polynomials `P*_n(x) = P_n(2x − 1)`.
///
/// Matches `scipy.special.roots_sh_legendre(n)`. Nodes are mapped from the
/// unshifted Legendre roots via `x_shift = (1 + x_unshift) / 2`; weights are
/// scaled by `1/2` so the quadrature integrates against the unit weight on
/// `[0, 1]`.
#[must_use]
pub fn roots_sh_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    shift_unit_to_zero_one(roots_legendre(n), 0.5)
}

/// Compute Gauss-Chebyshev (first kind) quadrature on `[0, 1]` via the
/// shifted polynomials `T*_n(x) = T_n(2x − 1)`.
///
/// Matches `scipy.special.roots_sh_chebyt(n)`. The node shift is the same as
/// `roots_sh_legendre`, but the Chebyshev-T weight function is unchanged
/// under the change of variables (`weight_scale = 1`).
#[must_use]
pub fn roots_sh_chebyt(n: usize) -> (Vec<f64>, Vec<f64>) {
    shift_unit_to_zero_one(roots_chebyt(n), 1.0)
}

/// Compute Gauss-Chebyshev (second kind) quadrature on `[0, 1]` via the
/// shifted polynomials `U*_n(x) = U_n(2x − 1)`.
///
/// Matches `scipy.special.roots_sh_chebyu(n)`. The Chebyshev-U weight
/// function picks up a factor of `1/4` under the change of variables.
#[must_use]
pub fn roots_sh_chebyu(n: usize) -> (Vec<f64>, Vec<f64>) {
    shift_unit_to_zero_one(roots_chebyu(n), 0.25)
}

/// Compute shifted Gauss-Jacobi quadrature nodes and weights on `[0, 1]`.
///
/// Matches `scipy.special.roots_sh_jacobi(n, p, q)`. SciPy's shifted
/// parameters map to standard Jacobi as `alpha = p - q`, `beta = q - 1`.
/// Nodes are shifted from `[-1, 1]` to `[0, 1]`; weights are scaled by
/// `2^-p` for the transformed weight `(1 - x)^(p - q) x^(q - 1)`.
#[must_use]
pub fn roots_sh_jacobi(n: usize, p: f64, q: f64) -> (Vec<f64>, Vec<f64>) {
    if !p.is_finite() || !q.is_finite() {
        return invalid_quadrature(n);
    }

    let alpha = p - q;
    let beta = q - 1.0;
    if alpha <= -1.0 || beta <= -1.0 {
        return invalid_quadrature(n);
    }

    shift_unit_to_zero_one(roots_jacobi(n, alpha, beta), 2.0_f64.powf(-p))
}

/// Map `(nodes, weights)` from the canonical `[-1, 1]` interval to the
/// shifted `[0, 1]` interval. The node transformation is `x ↦ (1 + x) / 2`.
///
/// `weight_scale` is the constant factor relating the canonical and shifted
/// weight functions under that change of variables. It depends on the weight
/// function, not just the Jacobian: `1/2` for the unit weight (shifted
/// Legendre), `1` for the Chebyshev-T weight `1/√(x(1−x))`, and `1/4` for the
/// Chebyshev-U weight `√(x(1−x))`.
fn shift_unit_to_zero_one(
    (nodes, weights): (Vec<f64>, Vec<f64>),
    weight_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let shifted_nodes = nodes.into_iter().map(|x| 0.5 * (1.0 + x)).collect();
    let shifted_weights = weights.into_iter().map(|w| weight_scale * w).collect();
    (shifted_nodes, shifted_weights)
}

/// Compute Gauss-Hermite quadrature nodes and weights on `(-∞, ∞)`.
///
/// The weight function is `exp(-x^2)`.
#[must_use]
pub fn roots_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    golub_welsch(n, PI.sqrt(), |_k| 0.0, |k| ((k as f64) / 2.0).sqrt(), true)
}

/// Compute Gauss-Hermite quadrature nodes and weights for probabilist's Hermite polynomials.
///
/// The weight function is `exp(-x²/2)` on `(-∞, ∞)`.
///
/// Matches `scipy.special.roots_hermitenorm(n)`.
///
/// # Arguments
/// * `n` - Number of quadrature points
///
/// # Returns
/// Tuple of (nodes, weights) for the quadrature rule
#[must_use]
pub fn roots_hermitenorm(n: usize) -> (Vec<f64>, Vec<f64>) {
    // mu0 = integral of exp(-x²/2) from -∞ to ∞ = sqrt(2π)
    let mu0 = (2.0 * PI).sqrt();
    // Probabilist's recurrence: He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)
    // So b_k = sqrt(k)
    golub_welsch(n, mu0, |_k| 0.0, |k| (k as f64).sqrt(), true)
}

/// Compute Gauss-Laguerre quadrature nodes and weights on `[0, ∞)`.
///
/// The weight function is `exp(-x)`.
#[must_use]
pub fn roots_laguerre(n: usize) -> (Vec<f64>, Vec<f64>) {
    golub_welsch(n, 1.0, |k| 2.0 * k as f64 + 1.0, |k| k as f64, false)
}

/// Compute generalized Gauss-Laguerre quadrature nodes and weights on `[0, ∞)`.
///
/// The weight function is `x^alpha * exp(-x)` where alpha > -1.
///
/// Matches `scipy.special.roots_genlaguerre(n, alpha)`.
///
/// # Arguments
/// * `n` - Number of quadrature points
/// * `alpha` - Parameter of the weight function (must be > -1)
///
/// # Returns
/// Tuple of (nodes, weights) for the quadrature rule
#[must_use]
pub fn roots_genlaguerre(n: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    if !alpha.is_finite() || alpha <= -1.0 {
        return invalid_quadrature(n);
    }

    // mu0 = integral of x^alpha * exp(-x) from 0 to infinity = Gamma(alpha + 1)
    let mu0 = gamma_half_integer_or_lanczos(alpha + 1.0);

    golub_welsch(
        n,
        mu0,
        // Diagonal elements: a_k = 2k + alpha + 1
        |k| 2.0 * k as f64 + alpha + 1.0,
        // Off-diagonal elements: b_k = sqrt(k * (k + alpha))
        |k| {
            let kf = k as f64;
            (kf * (kf + alpha)).sqrt()
        },
        false,
    )
}

/// Compute Gauss-Gegenbauer (ultraspherical) quadrature nodes and weights on `[-1, 1]`.
///
/// The weight function is `(1 - x²)^(alpha - 1/2)` where alpha > -1/2.
///
/// Matches `scipy.special.roots_gegenbauer(n, alpha)`.
///
/// # Arguments
/// * `n` - Number of quadrature points
/// * `alpha` - Parameter of the weight function (must be > -1/2)
///
/// # Returns
/// Tuple of (nodes, weights) for the quadrature rule
#[must_use]
pub fn roots_gegenbauer(n: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    if !alpha.is_finite() || alpha <= -0.5 {
        return invalid_quadrature(n);
    }
    // Gegenbauer with parameter alpha corresponds to Jacobi with alpha = beta = alpha - 1/2
    roots_jacobi(n, alpha - 0.5, alpha - 0.5)
}

/// Compute Gauss-Jacobi quadrature nodes and weights on `[-1, 1]`.
///
/// The weight function is `(1 - x)^alpha (1 + x)^beta`.
#[must_use]
pub fn roots_jacobi(n: usize, alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    if !alpha.is_finite() || !beta.is_finite() || alpha <= -1.0 || beta <= -1.0 {
        return invalid_quadrature(n);
    }

    let mu0 = 2.0_f64.powf(alpha + beta + 1.0) * beta_fn(alpha + 1.0, beta + 1.0);
    let symmetric = (alpha - beta).abs() <= 1e-14;

    golub_welsch(
        n,
        mu0,
        |k| {
            let k = k as f64;
            if (alpha + beta).abs() <= 1e-14 {
                if k == 0.0 {
                    (beta - alpha) / (alpha + beta + 2.0)
                } else {
                    0.0
                }
            } else if k == 0.0 {
                (beta - alpha) / (alpha + beta + 2.0)
            } else {
                (beta * beta - alpha * alpha)
                    / ((2.0 * k + alpha + beta) * (2.0 * k + alpha + beta + 2.0))
            }
        },
        |k| {
            let k = k as f64;
            let sum = 2.0 * k + alpha + beta;
            2.0 / sum
                * (((k + alpha) * (k + beta)) / (sum + 1.0)).sqrt()
                * if k == 1.0 {
                    1.0
                } else {
                    (k * (k + alpha + beta) / (sum - 1.0)).sqrt()
                }
        },
        symmetric,
    )
}

/// SciPy legacy alias for `roots_legendre`.
#[must_use]
pub fn p_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_legendre(n)
}

/// SciPy legacy alias for `roots_sh_legendre`.
#[must_use]
pub fn ps_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_sh_legendre(n)
}

/// SciPy legacy alias for `roots_chebyt`.
#[must_use]
pub fn t_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_chebyt(n)
}

/// SciPy legacy alias for `roots_sh_chebyt`.
#[must_use]
pub fn ts_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_sh_chebyt(n)
}

/// SciPy legacy alias for `roots_chebyu`.
#[must_use]
pub fn u_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_chebyu(n)
}

/// SciPy legacy alias for `roots_sh_chebyu`.
#[must_use]
pub fn us_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_sh_chebyu(n)
}

/// SciPy legacy alias for `roots_chebyc`.
#[must_use]
pub fn c_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_chebyc(n)
}

/// SciPy legacy alias for `roots_chebys`.
#[must_use]
pub fn s_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_chebys(n)
}

/// SciPy legacy alias for `roots_hermite`.
#[must_use]
pub fn h_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_hermite(n)
}

/// SciPy legacy alias for `roots_hermitenorm`.
#[must_use]
pub fn he_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_hermitenorm(n)
}

/// SciPy legacy alias for `roots_laguerre`.
#[must_use]
pub fn l_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_laguerre(n)
}

/// SciPy legacy alias for `roots_genlaguerre`.
#[must_use]
pub fn la_roots(n: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    roots_genlaguerre(n, alpha)
}

/// SciPy legacy alias for `roots_gegenbauer`.
#[must_use]
pub fn cg_roots(n: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    roots_gegenbauer(n, alpha)
}

/// SciPy legacy alias for `roots_jacobi`.
#[must_use]
pub fn j_roots(n: usize, alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    roots_jacobi(n, alpha, beta)
}

/// SciPy legacy alias for `roots_sh_jacobi`.
#[must_use]
pub fn js_roots(n: usize, p: f64, q: f64) -> (Vec<f64>, Vec<f64>) {
    roots_sh_jacobi(n, p, q)
}

fn invalid_quadrature(n: usize) -> (Vec<f64>, Vec<f64>) {
    (vec![f64::NAN; n], vec![f64::NAN; n])
}

fn golub_welsch<FDiag, FOff>(
    n: usize,
    mu0: f64,
    diag: FDiag,
    offdiag: FOff,
    symmetrize: bool,
) -> (Vec<f64>, Vec<f64>)
where
    FDiag: Fn(usize) -> f64,
    FOff: Fn(usize) -> f64,
{
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // The Golub-Welsch Jacobi matrix is symmetric TRIDIAGONAL and the quadrature
    // weights need ONLY w_i = mu0·v_i[0]² — the first component of each
    // eigenvector. Tracking the full n×n eigenvector matrix is O(n³); instead run
    // the same symmetric-tridiagonal QL but accumulate only the first eigenvector
    // row, dropping that to O(n²). `d` is the diagonal, `e` the n-1 off-diagonals.
    let d: Vec<f64> = (0..n).map(&diag).collect();
    let e: Vec<f64> = (1..n).map(&offdiag).collect();
    let (eigenvalues, first_row) = gw_tridiagonal_eigen_first_row(&d, &e).unwrap_or_else(|| {
        let mut z0 = vec![0.0; n];
        z0[0] = 1.0;
        (vec![0.0; n], z0)
    });

    // Sort ascending by eigenvalue (matching `eigh_tridiagonal`), carrying the
    // first-component along, so nodes/weights are bit-identical to the full
    // eigenvector path.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut nodes: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut weights: Vec<f64> = indices
        .iter()
        .map(|&i| mu0 * first_row[i] * first_row[i])
        .collect();

    if symmetrize {
        symmetrize_pairs(&mut nodes, &mut weights);
    }

    (nodes, weights)
}

/// Symmetric-tridiagonal QL eigensolve returning eigenvalues and ONLY the first
/// row of the eigenvector matrix, in O(n²). This is a faithful port of
/// `fsci_linalg::symmetric_tridiagonal_qr_eigen` (the routine `eigh_tridiagonal`
/// uses) that applies the identical Givens rotations but rotates a single length-n
/// row `z0` instead of the full n×n matrix. Because the diagonal/off-diagonal
/// updates and rotation coefficients are identical, `z0[j]` equals the full
/// solver's `eigenvectors[(0, j)]` bit-for-bit — so Golub-Welsch weights
/// `mu0·z0[j]²` match the full-eigenvector path while the eigenvector work drops
/// from O(n³) to O(n²).
fn gw_tridiagonal_eigen_first_row(
    diagonal: &[f64],
    offdiagonal: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let size = diagonal.len();
    if offdiagonal.len() != size.saturating_sub(1) {
        return None;
    }
    if size == 0 {
        return Some((Vec::new(), Vec::new()));
    }

    let scale = diagonal
        .iter()
        .chain(offdiagonal.iter())
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        let mut z0 = vec![0.0; size];
        z0[0] = 1.0;
        return Some((vec![0.0; size], z0));
    }

    let mut diag: Vec<f64> = diagonal.iter().map(|value| value / scale).collect();
    let mut off: Vec<f64> = offdiagonal.iter().map(|value| value / scale).collect();
    // First row of the identity eigenvector matrix.
    let mut z0 = vec![0.0_f64; size];
    z0[0] = 1.0;
    let tolerance = f64::EPSILON;
    let max_iterations = 64 * size; // BIDIAG_TRIDIAGONAL_QR_MAX_ITERS_PER_DIM * size
    let (mut start, mut end) = gw_delimit_subproblem(&diag, &mut off, size - 1, tolerance);
    let mut iterations = 0_usize;

    while end != start {
        let subdim = end - start + 1;
        if subdim > 2 {
            let tail_prev = end - 1;
            let shift = gw_wilkinson_shift(diag[tail_prev], diag[end], off[tail_prev]);
            let mut x = diag[start] - shift;
            let mut y = off[start];

            for idx in start..end {
                let Some((c, s, norm)) = gw_cancel_y(x, y) else {
                    break;
                };
                if idx > start {
                    off[idx - 1] = norm;
                }

                let left = diag[idx];
                let right = diag[idx + 1];
                let bridge = off[idx];
                let cc = c * c;
                let ss = s * s;
                let cs = c * s;

                diag[idx] = cc * left + ss * right - 2.0 * cs * bridge;
                diag[idx + 1] = ss * left + cc * right + 2.0 * cs * bridge;
                off[idx] = cs * (left - right) + (cc - ss) * bridge;

                if idx + 1 < end {
                    x = off[idx];
                    y = -s * off[idx + 1];
                    off[idx + 1] *= c;
                }

                // Rotate columns idx, idx+1 of the first eigenvector row only.
                let zl = z0[idx];
                let zr = z0[idx + 1];
                z0[idx] = c * zl - s * zr;
                z0[idx + 1] = s * zl + c * zr;
            }

            if off[tail_prev].abs() <= tolerance * (diag[tail_prev].abs() + diag[end].abs()) {
                off[tail_prev] = 0.0;
                end -= 1;
            }
        } else {
            gw_diagonalize_two_by_two(&mut diag, &mut off, &mut z0, start);
            end -= 1;
        }

        (start, end) = gw_delimit_subproblem(&diag, &mut off, end, tolerance);
        iterations += 1;
        if iterations > max_iterations {
            return None;
        }
    }

    for value in &mut diag {
        *value *= scale;
    }

    Some((diag, z0))
}

fn gw_wilkinson_shift(tmm: f64, tnn: f64, tmn: f64) -> f64 {
    let tmn_sq = tmn * tmn;
    if tmn_sq == 0.0 {
        return tnn;
    }
    let delta = 0.5 * (tmm - tnn);
    tnn - tmn_sq / (delta + delta.signum() * (delta * delta + tmn_sq).sqrt())
}

fn gw_cancel_y(x: f64, y: f64) -> Option<(f64, f64, f64)> {
    if y == 0.0 {
        return None;
    }
    let norm = x.hypot(y);
    if norm == 0.0 {
        return None;
    }
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let c = x.abs() / norm;
    let s = -y / (sign * norm);
    Some((c, s, sign * norm))
}

fn gw_delimit_subproblem(
    diagonal: &[f64],
    offdiagonal: &mut [f64],
    end: usize,
    tolerance: f64,
) -> (usize, usize) {
    let mut sub_end = end;
    while sub_end > 0 {
        let prev = sub_end - 1;
        let scale = diagonal[sub_end].abs() + diagonal[prev].abs();
        if offdiagonal[prev].abs() > tolerance * scale {
            break;
        }
        offdiagonal[prev] = 0.0;
        sub_end -= 1;
    }

    if sub_end == 0 {
        return (0, 0);
    }

    let mut sub_start = sub_end - 1;
    while sub_start > 0 {
        let prev = sub_start - 1;
        let scale = diagonal[sub_start].abs() + diagonal[prev].abs();
        if offdiagonal[prev] == 0.0 || offdiagonal[prev].abs() <= tolerance * scale {
            offdiagonal[prev] = 0.0;
            break;
        }
        sub_start -= 1;
    }

    (sub_start, sub_end)
}

fn gw_diagonalize_two_by_two(diag: &mut [f64], off: &mut [f64], z0: &mut [f64], start: usize) {
    let o = off[start];
    if o == 0.0 {
        return;
    }
    let left = diag[start];
    let right = diag[start + 1];
    let tau = (right - left) / (2.0 * o);
    let tangent = if tau == 0.0 {
        1.0
    } else {
        tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + tangent * tangent).sqrt();
    let s = tangent * c;

    diag[start] = left - tangent * o;
    diag[start + 1] = right + tangent * o;
    off[start] = 0.0;

    let zl = z0[start];
    let zr = z0[start + 1];
    z0[start] = c * zl - s * zr;
    z0[start + 1] = s * zl + c * zr;
}

fn symmetrize_pairs(nodes: &mut [f64], weights: &mut [f64]) {
    let n = nodes.len();
    for i in 0..(n / 2) {
        let j = n - 1 - i;
        let node = (nodes[j] - nodes[i]) / 2.0;
        let weight = (weights[i] + weights[j]) / 2.0;
        nodes[i] = -node;
        nodes[j] = node;
        weights[i] = weight;
        weights[j] = weight;
    }
    if n % 2 == 1 {
        nodes[n / 2] = 0.0;
    }
}

fn beta_fn(a: f64, b: f64) -> f64 {
    gamma_half_integer_or_lanczos(a) * gamma_half_integer_or_lanczos(b)
        / gamma_half_integer_or_lanczos(a + b)
}

fn gamma_half_integer_or_lanczos(x: f64) -> f64 {
    if x <= 0.0 && x.fract().abs() <= 1e-14 {
        return f64::NAN;
    }
    if (x - 0.5).abs() <= 1e-14 {
        return PI.sqrt();
    }

    lanczos_gamma(x)
}

fn lanczos_gamma(z: f64) -> f64 {
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

    if z < 0.5 {
        return PI / ((PI * z).sin() * lanczos_gamma(1.0 - z));
    }

    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (idx, coeff) in COEFFS.iter().enumerate().skip(1) {
        x += coeff / (z + idx as f64);
    }

    let t = z + G + 0.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

// ── Associated Legendre functions ─────────────────────────────────────

/// Evaluate the associated Legendre function P_l^m(x) (with Condon-Shortley phase).
///
/// Parameters:
/// - `m`: order (integer, |m| <= l)
/// - `l`: degree (non-negative integer)
/// - `x`: evaluation point, typically in [-1, 1]
///
/// Uses upward recurrence in l starting from P_m^m:
///   P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^{m/2}
///   P_{m+1}^m(x) = x (2m+1) P_m^m(x)
///   (l-m) P_l^m(x) = x(2l-1) P_{l-1}^m(x) - (l+m-1) P_{l-2}^m(x)
///
/// For negative m: P_l^{-m}(x) = (-1)^m (l-m)!/(l+m)! P_l^m(x)
pub fn lpmv(m: i32, l: u32, x: f64) -> f64 {
    if m == 0 && l == 0 {
        return 1.0;
    }
    if !x.is_finite() {
        return f64::NAN;
    }

    let li = l as i32;
    let am = m.unsigned_abs();

    // |m| > l is zero by definition
    if am > l {
        return 0.0;
    }

    // Compute P_l^{|m|}(x)
    let plm = lpmv_nonneg_m(am, l, x);

    if m < 0 {
        // P_l^{-m}(x) = (-1)^m * (l-|m|)!/(l+|m|)! * P_l^{|m|}(x)
        let sign = if am.is_multiple_of(2) { 1.0 } else { -1.0 };
        let mut ratio = 1.0;
        for k in (li - am as i32 + 1)..=(li + am as i32) {
            ratio /= k as f64;
        }
        sign * ratio * plm
    } else {
        plm
    }
}

/// Compute P_l^m(x) for m >= 0 using stable upward recurrence.
fn lpmv_nonneg_m(m: u32, l: u32, x: f64) -> f64 {
    // Start with P_m^m(x) = (-1)^m * (2m-1)!! * (1-x²)^{m/2}
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = (1.0 - x * x).max(0.0).sqrt();
        let mut fact = 1.0;
        for _i in 1..=m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }

    if l == m {
        return pmm;
    }

    // P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
    let pmm1 = x * (2.0 * m as f64 + 1.0) * pmm;
    if l == m + 1 {
        return pmm1;
    }

    // Upward recurrence in l
    let mut p_prev = pmm;
    let mut p_curr = pmm1;
    for ll in (m + 2)..=l {
        let llf = ll as f64;
        let mf = m as f64;
        let p_next = ((2.0 * llf - 1.0) * x * p_curr - (llf + mf - 1.0) * p_prev) / (llf - mf);
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

// ── Spherical harmonics ──────────────────────────────────────────────

use crate::types::Complex64;

/// Complex spherical harmonic Y_l^m(θ, φ).
///
/// Uses the physics convention (ISO 31-11):
///   Y_l^m(θ, φ) = √((2l+1)/(4π) * (l-|m|)!/(l+|m|)!) * P_l^{|m|}(cos φ) * exp(i·m·θ)
///
/// Parameters:
/// - `m`: order (integer, |m| <= l)
/// - `l`: degree (non-negative integer)
/// - `theta`: azimuthal angle in [0, 2π)
/// - `phi`: polar angle (colatitude) in [0, π]
///
/// Note: Follows SciPy convention where theta=azimuthal and phi=polar.
pub fn sph_harm(m: i32, l: u32, theta: f64, phi: f64) -> Complex64 {
    let am = m.unsigned_abs();

    if am > l {
        return Complex64 { re: 0.0, im: 0.0 };
    }

    if l == 0 {
        return Complex64 {
            re: 1.0 / (4.0 * PI).sqrt(),
            im: 0.0,
        };
    }

    if !theta.is_finite() || !phi.is_finite() {
        return Complex64 {
            re: f64::NAN,
            im: f64::NAN,
        };
    }

    // Normalization: sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
    let lf = l as f64;
    let li = l as i32;

    // Compute ln((l-|m|)!/(l+|m|)!) for numerical stability
    let mut log_ratio = 0.0;
    for k in (li - am as i32 + 1)..=(li + am as i32) {
        log_ratio -= (k as f64).ln();
    }
    let norm = ((2.0 * lf + 1.0) / (4.0 * std::f64::consts::PI) * log_ratio.exp()).sqrt();

    // Associated Legendre at cos(phi) — Condon-Shortley phase included
    let plm = lpmv_nonneg_m(am, l, phi.cos());

    // Negative orders: scipy uses Y_l^{-m} = (-1)^m · conj(Y_l^m). The
    // normalization and Legendre factor here use |m|, so the (-1)^|m|
    // sign for negative m must be reintroduced explicitly.
    let sign = if m < 0 && !am.is_multiple_of(2) {
        -1.0
    } else {
        1.0
    };

    // Phase factor exp(i·m·θ)
    let angle = m as f64 * theta;

    Complex64 {
        re: sign * norm * plm * angle.cos(),
        im: sign * norm * plm * angle.sin(),
    }
}

/// Complex spherical harmonic `Y_n^m(theta, phi)`.
///
/// Matches `scipy.special.sph_harm_y(n, m, theta, phi)`, where `theta` is the
/// polar angle and `phi` is the azimuthal angle. This differs from the older
/// `sph_harm` helper above, which follows SciPy's deprecated argument order.
pub fn sph_harm_y(n: u32, m: i32, theta: f64, phi: f64) -> Complex64 {
    sph_harm(m, n, phi, theta)
}

// ---------------------------------------------------------------------------
// Polynomial coefficient constructors
//
// SciPy exposes `legendre(n)`, `chebyt(n)`, ... as `orthopoly1d` objects whose
// coefficient array `.c` is ordered highest-degree-first (the NumPy `poly1d`
// convention). We mirror that surface by returning the coefficient `Vec<f64>`
// in the same order. Coefficients are built exactly from each family's
// three-term recurrence acting on the coefficient vectors (ascending powers),
// then reversed to highest-degree-first — this avoids the float noise SciPy
// incurs from constructing the polynomial out of its (approximate) roots.
// ---------------------------------------------------------------------------

/// Build the degree-`n` coefficients (ascending powers, `c[i]` multiplies `x^i`)
/// from a three-term recurrence
/// `P_{k+1}(x) = (a_k·x + b_k)·P_k(x) + c_k·P_{k-1}(x)`, given `P_0` and `P_1`.
fn recurrence_coeffs(
    n: u32,
    p0: Vec<f64>,
    p1: Vec<f64>,
    abc: impl Fn(u32) -> (f64, f64, f64),
) -> Vec<f64> {
    if n == 0 {
        return p0;
    }
    let mut prev = p0;
    let mut curr = p1;
    for k in 1..n {
        let (a, b, c) = abc(k);
        let mut next = vec![0.0_f64; (k + 2) as usize];
        for (i, &ci) in curr.iter().enumerate() {
            next[i + 1] += a * ci; // x · curr shifts powers up by one
            next[i] += b * ci;
        }
        for (i, &pi) in prev.iter().enumerate() {
            next[i] += c * pi;
        }
        prev = curr;
        curr = next;
    }
    curr
}

/// Reverse ascending-power coefficients to the highest-degree-first order SciPy
/// (and NumPy `poly1d`) uses.
fn to_descending(mut ascending: Vec<f64>) -> Vec<f64> {
    ascending.reverse();
    ascending
}

/// Legendre polynomial `P_n` coefficients, highest degree first.
///
/// Matches `scipy.special.legendre(n).c`. Recurrence
/// `(k+1)P_{k+1} = (2k+1)x P_k − k P_{k-1}`, with `P_0 = 1`, `P_1 = x`.
#[must_use]
pub fn legendre(n: u32) -> Vec<f64> {
    to_descending(recurrence_coeffs(n, vec![1.0], vec![0.0, 1.0], |k| {
        let kf = k as f64;
        ((2.0 * kf + 1.0) / (kf + 1.0), 0.0, -kf / (kf + 1.0))
    }))
}

/// Chebyshev polynomial of the first kind `T_n` coefficients, highest degree
/// first.
///
/// Matches `scipy.special.chebyt(n).c`. Recurrence `T_{k+1} = 2x T_k − T_{k-1}`,
/// with `T_0 = 1`, `T_1 = x`.
#[must_use]
pub fn chebyt(n: u32) -> Vec<f64> {
    to_descending(recurrence_coeffs(n, vec![1.0], vec![0.0, 1.0], |_| {
        (2.0, 0.0, -1.0)
    }))
}

/// Chebyshev polynomial of the second kind `U_n` coefficients, highest degree
/// first.
///
/// Matches `scipy.special.chebyu(n).c`. Recurrence `U_{k+1} = 2x U_k − U_{k-1}`,
/// with `U_0 = 1`, `U_1 = 2x`.
#[must_use]
pub fn chebyu(n: u32) -> Vec<f64> {
    to_descending(recurrence_coeffs(n, vec![1.0], vec![0.0, 2.0], |_| {
        (2.0, 0.0, -1.0)
    }))
}

/// Physicist's Hermite polynomial `H_n` coefficients, highest degree first.
///
/// Matches `scipy.special.hermite(n).c`. Recurrence
/// `H_{k+1} = 2x H_k − 2k H_{k-1}`, with `H_0 = 1`, `H_1 = 2x`.
#[must_use]
pub fn hermite(n: u32) -> Vec<f64> {
    to_descending(recurrence_coeffs(n, vec![1.0], vec![0.0, 2.0], |k| {
        (2.0, 0.0, -2.0 * k as f64)
    }))
}

/// Probabilist's Hermite polynomial `He_n` coefficients, highest degree first.
///
/// Matches `scipy.special.hermitenorm(n).c`. Recurrence
/// `He_{k+1} = x He_k − k He_{k-1}`, with `He_0 = 1`, `He_1 = x`.
#[must_use]
pub fn hermitenorm(n: u32) -> Vec<f64> {
    to_descending(recurrence_coeffs(n, vec![1.0], vec![0.0, 1.0], |k| {
        (1.0, 0.0, -(k as f64))
    }))
}

/// Laguerre polynomial `L_n` coefficients, highest degree first.
///
/// Matches `scipy.special.laguerre(n).c`. Recurrence
/// `(k+1)L_{k+1} = (2k+1−x)L_k − k L_{k-1}`, with `L_0 = 1`, `L_1 = 1 − x`.
#[must_use]
pub fn laguerre(n: u32) -> Vec<f64> {
    to_descending(recurrence_coeffs(n, vec![1.0], vec![1.0, -1.0], |k| {
        let kf = k as f64;
        (
            -1.0 / (kf + 1.0),
            (2.0 * kf + 1.0) / (kf + 1.0),
            -kf / (kf + 1.0),
        )
    }))
}

/// Generalized (associated) Laguerre polynomial `L_n^α` coefficients, highest
/// degree first.
///
/// Matches `scipy.special.genlaguerre(n, alpha).c`. Recurrence
/// `(k+1)L_{k+1}^α = (2k+1+α−x)L_k^α − (k+α)L_{k-1}^α`, with `L_0^α = 1`,
/// `L_1^α = 1 + α − x`. (Mirrors [`eval_genlaguerre`].)
#[must_use]
pub fn genlaguerre(n: u32, alpha: f64) -> Vec<f64> {
    to_descending(recurrence_coeffs(n, vec![1.0], vec![1.0 + alpha, -1.0], |k| {
        let kf = k as f64;
        (
            -1.0 / (kf + 1.0),
            (2.0 * kf + 1.0 + alpha) / (kf + 1.0),
            -(kf + alpha) / (kf + 1.0),
        )
    }))
}

/// Gegenbauer (ultraspherical) polynomial `C_n^α` coefficients, highest degree
/// first.
///
/// Matches `scipy.special.gegenbauer(n, alpha).c`. Recurrence
/// `(k+1)C_{k+1}^α = 2(k+α)x C_k^α − (k+2α−1)C_{k-1}^α`, with `C_0^α = 1`,
/// `C_1^α = 2αx`. (Mirrors [`eval_gegenbauer`]; `α = 0` yields the zero
/// polynomial for `n ≥ 1`, as SciPy does.)
#[must_use]
pub fn gegenbauer(n: u32, alpha: f64) -> Vec<f64> {
    to_descending(recurrence_coeffs(n, vec![1.0], vec![0.0, 2.0 * alpha], |k| {
        let kf = k as f64;
        (
            2.0 * (kf + alpha) / (kf + 1.0),
            0.0,
            -(kf + 2.0 * alpha - 1.0) / (kf + 1.0),
        )
    }))
}

/// Jacobi polynomial `P_n^{α,β}` coefficients, highest degree first.
///
/// Matches `scipy.special.jacobi(n, alpha, beta).c`. Uses the DLMF 18.9.2
/// three-term recurrence (mirroring [`eval_jacobi`]), with `P_0 = 1`,
/// `P_1 = ½[(α−β) + (α+β+2)x]`. Where the recurrence's leading factor
/// degenerates (`|a1| ≤ 1e-30`, e.g. `α+β` near `−(k+1)`/`−2k`), the step
/// collapses to the zero polynomial, matching the scalar evaluator.
#[must_use]
pub fn jacobi(n: u32, alpha: f64, beta: f64) -> Vec<f64> {
    let p1 = vec![0.5 * (alpha - beta), 0.5 * (alpha + beta + 2.0)];
    to_descending(recurrence_coeffs(n, vec![1.0], p1, |k| {
        let kf = k as f64;
        let ab = alpha + beta;
        let two_k_ab = 2.0 * kf + ab;
        let a1 = 2.0 * (kf + 1.0) * (kf + ab + 1.0) * two_k_ab;
        let a2 = (two_k_ab + 1.0) * (alpha * alpha - beta * beta);
        let a3 = two_k_ab * (two_k_ab + 1.0) * (two_k_ab + 2.0);
        let a4 = 2.0 * (kf + alpha) * (kf + beta) * (two_k_ab + 2.0);
        if a1.abs() > 1e-30 {
            (a3 / a1, a2 / a1, -a4 / a1)
        } else {
            (0.0, 0.0, 0.0)
        }
    }))
}

/// Compose `p(a·x + b)` for `p` given in ascending-power coefficients, returning
/// ascending-power coefficients of the same degree (`a ≠ 0`). Horner over the
/// degree-1 inner polynomial `a·x + b`.
fn compose_linear(ascending: &[f64], a: f64, b: f64) -> Vec<f64> {
    let mut res = vec![*ascending.last().expect("polynomial has at least one coeff")];
    for &c in ascending.iter().rev().skip(1) {
        let mut next = vec![0.0_f64; res.len() + 1];
        for (i, &v) in res.iter().enumerate() {
            next[i] += b * v; // (a·x + b) · res
            next[i + 1] += a * v;
        }
        next[0] += c;
        res = next;
    }
    res
}

/// Ascending-power coefficients of one of the base constructors (which return
/// descending order).
fn ascending_of(descending: Vec<f64>) -> Vec<f64> {
    let mut a = descending;
    a.reverse();
    a
}

/// Chebyshev polynomial of the first kind on `[-2, 2]`, `C_n(x) = 2·T_n(x/2)`,
/// coefficients highest degree first.
///
/// Matches `scipy.special.chebyc(n).c`.
#[must_use]
pub fn chebyc(n: u32) -> Vec<f64> {
    let mut c = compose_linear(&ascending_of(chebyt(n)), 0.5, 0.0);
    for v in &mut c {
        *v *= 2.0;
    }
    to_descending(c)
}

/// Chebyshev polynomial of the second kind on `[-2, 2]`, `S_n(x) = U_n(x/2)`,
/// coefficients highest degree first.
///
/// Matches `scipy.special.chebys(n).c`.
#[must_use]
pub fn chebys(n: u32) -> Vec<f64> {
    to_descending(compose_linear(&ascending_of(chebyu(n)), 0.5, 0.0))
}

/// Shifted Legendre polynomial `P*_n(x) = P_n(2x − 1)` coefficients, highest
/// degree first.
///
/// Matches `scipy.special.sh_legendre(n).c`.
#[must_use]
pub fn sh_legendre(n: u32) -> Vec<f64> {
    to_descending(compose_linear(&ascending_of(legendre(n)), 2.0, -1.0))
}

/// Shifted Chebyshev polynomial of the first kind `T*_n(x) = T_n(2x − 1)`
/// coefficients, highest degree first.
///
/// Matches `scipy.special.sh_chebyt(n).c`.
#[must_use]
pub fn sh_chebyt(n: u32) -> Vec<f64> {
    to_descending(compose_linear(&ascending_of(chebyt(n)), 2.0, -1.0))
}

/// Shifted Chebyshev polynomial of the second kind `U*_n(x) = U_n(2x − 1)`
/// coefficients, highest degree first.
///
/// Matches `scipy.special.sh_chebyu(n).c`.
#[must_use]
pub fn sh_chebyu(n: u32) -> Vec<f64> {
    to_descending(compose_linear(&ascending_of(chebyu(n)), 2.0, -1.0))
}

/// Shifted Jacobi polynomial `G_n^{(p,q)}(x)` coefficients, highest degree
/// first.
///
/// Matches `scipy.special.sh_jacobi(n, p, q).c`:
/// `G_n^{(p,q)}(x) = P_n^{(p−q, q−1)}(2x − 1) / C(2n+p−1, n)`, the same
/// `C(2n+p−1, n) = Π_{j<n}(2n+p−1−j)/(j+1)` normalization used by
/// [`eval_sh_jacobi`].
#[must_use]
pub fn sh_jacobi(n: u32, p: f64, q: f64) -> Vec<f64> {
    let a = 2.0 * f64::from(n) + p - 1.0;
    let mut norm = 1.0_f64;
    for j in 0..n {
        norm *= (a - f64::from(j)) / (f64::from(j) + 1.0);
    }
    let mut c = compose_linear(&ascending_of(jacobi(n, p - q, q - 1.0)), 2.0, -1.0);
    for v in &mut c {
        *v /= norm;
    }
    to_descending(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_coeffs(actual: &[f64], expected: &[f64], msg: &str) {
        assert_eq!(actual.len(), expected.len(), "{msg}: degree/length");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-9,
                "{msg}[{i}]: got {a}, expected {e} (diff={})",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn polynomial_coeffs_match_scipy() {
        // frankenscipy: golden from scipy.special.{legendre,chebyt,...}(n).c (1.17.1),
        // highest-degree-first. SciPy's near-zero noise (~1e-15) is absorbed by tol.
        assert_coeffs(&legendre(0), &[1.0], "legendre0");
        assert_coeffs(&legendre(1), &[1.0, 0.0], "legendre1");
        assert_coeffs(&legendre(5), &[7.875, 0.0, -8.75, 0.0, 1.875, 0.0], "legendre5");
        assert_coeffs(
            &legendre(6),
            &[14.4375, 0.0, -19.6875, 0.0, 6.5625, 0.0, -0.3125],
            "legendre6",
        );
        assert_coeffs(&chebyt(4), &[8.0, 0.0, -8.0, 0.0, 1.0], "chebyt4");
        assert_coeffs(
            &chebyt(7),
            &[64.0, 0.0, -112.0, 0.0, 56.0, 0.0, -7.0, 0.0],
            "chebyt7",
        );
        assert_coeffs(&chebyu(3), &[8.0, 0.0, -4.0, 0.0], "chebyu3");
        assert_coeffs(
            &chebyu(6),
            &[64.0, 0.0, -80.0, 0.0, 24.0, 0.0, -1.0],
            "chebyu6",
        );
        assert_coeffs(&hermite(3), &[8.0, 0.0, -12.0, 0.0], "hermite3");
        assert_coeffs(&hermite(5), &[32.0, 0.0, -160.0, 0.0, 120.0, 0.0], "hermite5");
        assert_coeffs(
            &hermitenorm(4),
            &[1.0, 0.0, -6.0, 0.0, 3.0],
            "hermitenorm4",
        );
        assert_coeffs(
            &hermitenorm(6),
            &[1.0, 0.0, -15.0, 0.0, 45.0, 0.0, -15.0],
            "hermitenorm6",
        );
        assert_coeffs(
            &laguerre(3),
            &[
                -0.166_666_666_666_666_66,
                1.5,
                -3.0,
                1.0,
            ],
            "laguerre3",
        );
        assert_coeffs(
            &laguerre(5),
            &[
                -0.008_333_333_333_333_333,
                0.208_333_333_333_333_34,
                -1.666_666_666_666_666_7,
                5.0,
                -5.0,
                1.0,
            ],
            "laguerre5",
        );
    }

    #[test]
    fn parameterized_polynomial_coeffs_match_scipy() {
        // frankenscipy: golden from scipy.special.{genlaguerre,gegenbauer,jacobi}(n, ...).c
        // (1.17.1), highest-degree-first.
        assert_coeffs(
            &genlaguerre(3, 0.5),
            &[-0.166_666_666_666_666_66, 1.75, -4.375, 2.1875],
            "genlaguerre(3,0.5)",
        );
        assert_coeffs(
            &genlaguerre(4, 1.5),
            &[
                0.041_666_666_666_666_664,
                -0.916_666_666_666_666_7,
                6.1875,
                -14.4375,
                9.023_437_5,
            ],
            "genlaguerre(4,1.5)",
        );
        assert_coeffs(&gegenbauer(3, 2.0), &[32.0, 0.0, -12.0, 0.0], "gegenbauer(3,2.0)");
        assert_coeffs(
            &gegenbauer(4, 0.5),
            &[4.375, 0.0, -3.75, 0.0, 0.375],
            "gegenbauer(4,0.5)",
        );
        // α = 0 degenerates to the zero polynomial (length n+1), as SciPy does.
        assert_coeffs(&gegenbauer(3, 0.0), &[0.0, 0.0, 0.0, 0.0], "gegenbauer(3,0.0)");
        assert_coeffs(
            &jacobi(3, 1.0, 2.0),
            &[10.5, -3.5, -3.5, 0.5],
            "jacobi(3,1.0,2.0)",
        );
        assert_coeffs(
            &jacobi(4, 0.5, 0.5),
            &[7.875, 0.0, -5.906_25, 0.0, 0.492_187_5],
            "jacobi(4,0.5,0.5)",
        );
    }

    #[test]
    fn legendre_p_matches_scipy() {
        // frankenscipy: golden from scipy.special.legendre_p(n, z, diff_n=...) (1.17.1).
        let close = |got: &[f64], want: &[f64], msg: &str| {
            assert_eq!(got.len(), want.len(), "{msg}: len");
            for (g, w) in got.iter().zip(want.iter()) {
                assert!((g - w).abs() < 1e-10, "{msg}: got {g}, want {w}");
            }
        };
        // value only
        close(&legendre_p(3, 0.4, 0), &[-0.44], "P_3(0.4)");
        // value + first + second derivative
        close(&legendre_p(5, 0.3, 2), &[0.34538625, -0.16856249999999984, -11.4975], "P_5(0.3) d2");
        close(&legendre_p(2, 1.0, 2), &[1.0, 3.0, 3.0], "P_2(1) d2");
        close(&legendre_p(6, -1.0, 2), &[1.0, -21.0, 210.0], "P_6(-1) d2");
        // diff_n exceeding the degree -> trailing zero derivatives
        close(&legendre_p(0, 0.5, 2), &[1.0, 0.0, 0.0], "P_0(0.5) d2");
        close(&legendre_p(1, 0.5, 2), &[0.5, 1.0, 0.0], "P_1(0.5) d2");
    }

    #[test]
    fn legendre_p_all_matches_scipy() {
        // frankenscipy: golden from scipy.special.legendre_p_all(n, z, diff_n=...) (1.17.1),
        // shape (diff_n+1, n+1); entry [i][j] = i-th derivative of degree-j polynomial.
        let check = |got: &[Vec<f64>], want: &[Vec<f64>], msg: &str| {
            assert_eq!(got.len(), want.len(), "{msg}: rows");
            for (gr, wr) in got.iter().zip(want.iter()) {
                assert_eq!(gr.len(), wr.len(), "{msg}: cols");
                for (g, w) in gr.iter().zip(wr.iter()) {
                    assert!((g - w).abs() < 1e-10, "{msg}: got {g}, want {w}");
                }
            }
        };
        check(
            &legendre_p_all(3, 0.4, 0),
            &[vec![1.0, 0.4, -0.26, -0.44]],
            "all(3,0.4,0)",
        );
        check(
            &legendre_p_all(3, 0.4, 1),
            &[vec![1.0, 0.4, -0.26, -0.44], vec![0.0, 1.0, 1.2, -0.3]],
            "all(3,0.4,1)",
        );
        check(
            &legendre_p_all(4, -0.6, 2),
            &[
                vec![1.0, -0.6, 0.04, 0.36, -0.408],
                vec![0.0, 1.0, -1.8, 1.2, 0.72],
                vec![0.0, 0.0, 3.0, -9.0, 11.4],
            ],
            "all(4,-0.6,2)",
        );
    }

    #[test]
    fn spheroidal_cv_seq_and_cv_variants_match_scipy() {
        // frankenscipy: golden from scipy.special.pro_cv_seq / obl_cv_seq (1.17.1).
        let seq_close = |got: &[f64], want: &[f64], msg: &str| {
            assert_eq!(got.len(), want.len(), "{msg}: len");
            for (g, w) in got.iter().zip(want.iter()) {
                assert!((g - w).abs() < 1e-9, "{msg}: got {g}, want {w}");
            }
        };
        seq_close(
            &pro_cv_seq(0, 3, 2.0),
            &[1.127734064849933, 4.287128543955809, 8.225713001105891, 14.100203876205317],
            "pro_cv_seq(0,3,2)",
        );
        seq_close(
            &obl_cv_seq(1, 4, 2.0),
            &[1.1185533907435148, 4.222747333357612, 10.163245057199756, 18.09765523018453],
            "obl_cv_seq(1,4,2)",
        );
        // The _cv variants must reproduce the base functions when fed the matching cv.
        let pair_eq = |a: (f64, f64), b: (f64, f64), msg: &str| {
            assert!((a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12, "{msg}");
        };
        let cv = pro_cv(1, 2, 2.0);
        pair_eq(pro_ang1_cv(1, 2, 2.0, cv, 0.3), pro_ang1(1, 2, 2.0, 0.3), "pro_ang1_cv");
        pair_eq(pro_rad1_cv(1, 2, 2.0, cv, 1.3), pro_rad1(1, 2, 2.0, 1.3), "pro_rad1_cv");
        let ocv = obl_cv(2, 5, 3.0);
        pair_eq(obl_ang1_cv(2, 5, 3.0, ocv, 0.2), obl_ang1(2, 5, 3.0, 0.2), "obl_ang1_cv");
        pair_eq(obl_rad1_cv(2, 5, 3.0, ocv, 1.0), obl_rad1(2, 5, 3.0, 1.0), "obl_rad1_cv");
    }

    #[test]
    fn spheroidal_rad1_match_scipy() {
        // frankenscipy: golden (value, derivative) from scipy.special.pro_rad1 / obl_rad1 (1.17.1).
        let c = |got: (f64, f64), want: (f64, f64), msg: &str| {
            assert!((got.0 - want.0).abs() < 1e-8, "{msg} value: got {} want {}", got.0, want.0);
            assert!((got.1 - want.1).abs() < 1e-8, "{msg} deriv: got {} want {}", got.1, want.1);
        };
        c(pro_rad1(0, 1, 1.0, 1.5), (0.4138205450234367, 0.14462549507897315), "pro(0,1,1,1.5)");
        c(pro_rad1(1, 2, 2.0, 1.3), (0.20800143088699186, 0.38525174496351966), "pro(1,2,2,1.3)");
        c(pro_rad1(0, 2, 3.0, 1.4), (0.3427807385468018, -0.2848804090728431), "pro(0,2,3,1.4)");
        c(pro_rad1(2, 3, 2.0, 1.6), (0.12304660475662652, 0.2341694344697482), "pro(2,3,2,1.6)");
        c(pro_rad1(0, 3, 5.0, 2.5), (0.08751848334486662, -0.041034086997124256), "pro(0,3,5,2.5)");
        c(obl_rad1(0, 1, 1.0, 0.5), (0.15622885398178357, 0.29654103600390647), "obl(0,1,1,0.5)");
        c(obl_rad1(1, 2, 2.0, 0.8), (0.19179313880509874, 0.2380821457479905), "obl(1,2,2,0.8)");
        c(obl_rad1(2, 5, 3.0, 1.0), (0.03647796691020278, 0.09888356007741966), "obl(2,5,3,1.0)");
        let (nv, nd) = pro_rad1(3, 1, 2.0, 1.5);
        assert!(nv.is_nan() && nd.is_nan(), "pro_rad1 n<m -> NaN");
    }

    #[test]
    fn spheroidal_ang1_match_scipy() {
        // frankenscipy: golden (value, derivative) from scipy.special.pro_ang1 / obl_ang1 (1.17.1).
        let c = |got: (f64, f64), want: (f64, f64), msg: &str| {
            assert!((got.0 - want.0).abs() < 1e-8, "{msg} value: got {} want {}", got.0, want.0);
            assert!((got.1 - want.1).abs() < 1e-8, "{msg} deriv: got {} want {}", got.1, want.1);
        };
        c(pro_ang1(0, 1, 1.0, 0.5), (0.4877531776005848, 0.9269534809430655), "pro(0,1,1,.5)");
        c(pro_ang1(1, 2, 2.0, 0.3), (0.8374625906386085, 2.37627584419033), "pro(1,2,2,.3)");
        c(pro_ang1(0, 2, 3.0, 0.4), (-0.0913823901653181, 1.854929773816917), "pro(0,2,3,.4)");
        c(pro_ang1(2, 3, 2.0, 0.6), (5.325478515120345, -2.511554419239202), "pro(2,3,2,.6)");
        c(pro_ang1(0, 3, 5.0, -0.7), (-0.21455871795642947, 2.0630078366395237), "pro(0,3,5,-.7)");
        c(obl_ang1(0, 1, 1.0, 0.5), (0.5127556415606529, 1.0769923985071272), "obl(0,1,1,.5)");
        c(obl_ang1(1, 2, 2.0, 0.3), (0.8816684962327807, 2.803968096264711), "obl(1,2,2,.3)");
        c(obl_ang1(2, 5, 3.0, 0.2), (-9.117409468924578, -32.01395997554938), "obl(2,5,3,.2)");
        let (nan_v, nan_d) = pro_ang1(3, 1, 2.0, 0.5);
        assert!(nan_v.is_nan() && nan_d.is_nan(), "pro_ang1 n<m -> NaN");
    }

    #[test]
    fn spheroidal_cv_match_scipy() {
        // frankenscipy: golden from scipy.special.pro_cv / obl_cv (1.17.1).
        let c = |got: f64, want: f64, msg: &str| {
            assert!((got - want).abs() < 1e-9, "{msg}: got {got}, want {want}");
        };
        c(pro_cv(0, 0, 0.0), 0.0, "pro(0,0,0)");
        c(pro_cv(0, 1, 1.0), 2.5930845799771327, "pro(0,1,1)");
        c(pro_cv(1, 2, 2.0), 7.653149562003566, "pro(1,2,2)");
        c(pro_cv(2, 5, 3.0), 33.936151070297626, "pro(2,5,3)");
        c(pro_cv(0, 3, 5.0), 26.58735960739739, "pro(0,3,5)");
        c(pro_cv(3, 7, 10.0), 96.40807000927285, "pro(3,7,10)");
        c(obl_cv(0, 1, 1.0), 1.3932063104484202, "obl(0,1,1)");
        c(obl_cv(1, 2, 2.0), 4.222747333357613, "obl(1,2,2)");
        c(obl_cv(2, 5, 3.0), 26.10501393807085, "obl(2,5,3)");
        c(obl_cv(0, 4, 8.0), -2.7990336507689504, "obl(0,4,8)");
        // n < m is undefined -> NaN, as SciPy does.
        assert!(pro_cv(3, 1, 2.0).is_nan(), "pro n<m -> NaN");
    }

    #[test]
    fn mathieu_mod1_match_scipy() {
        // frankenscipy: golden (value, derivative) from scipy.special.mathieu_modcem1 /
        // mathieu_modsem1 (1.17.1).
        let c = |got: (f64, f64), want: (f64, f64), msg: &str| {
            assert!((got.0 - want.0).abs() < 1e-9, "{msg} value: got {} want {}", got.0, want.0);
            assert!((got.1 - want.1).abs() < 1e-9, "{msg} deriv: got {} want {}", got.1, want.1);
        };
        c(mathieu_modcem1(0, 5.0, 1.0), (0.024144064633139083, 2.03208182190974), "modcem1(0,5,1)");
        c(mathieu_modcem1(1, 2.0, 0.8), (0.21245270237182737, -1.329777625044168), "modcem1(1,2,.8)");
        c(mathieu_modcem1(2, 5.0, 1.2), (-0.20381071990853145, 1.6697575783227967), "modcem1(2,5,1.2)");
        c(mathieu_modcem1(3, 10.0, 0.5), (0.1830202349061459, -1.5534178465951263), "modcem1(3,10,.5)");
        c(mathieu_modcem1(4, 5.0, 0.8), (0.40727697154630377, 0.1479917173082351), "modcem1(4,5,.8)");
        c(mathieu_modsem1(1, 5.0, 1.0), (-0.30984611185156535, 0.29087720606156564), "modsem1(1,5,1)");
        c(mathieu_modsem1(2, 5.0, 1.2), (-0.26281891612299607, 1.111022300400359), "modsem1(2,5,1.2)");
        c(mathieu_modsem1(3, 10.0, 0.5), (0.3426517831198076, -0.7748887304796102), "modsem1(3,10,.5)");
        c(mathieu_modsem1(4, 5.0, 0.8), (0.3957103341885613, 0.23662248116583467), "modsem1(4,5,.8)");
    }

    #[test]
    fn mathieu_cem_sem_match_scipy() {
        // frankenscipy: golden (value, derivative) from scipy.special.mathieu_cem / mathieu_sem
        // (1.17.1); x in degrees, derivative w.r.t. the radian argument.
        let c = |got: (f64, f64), want: (f64, f64), msg: &str| {
            assert!((got.0 - want.0).abs() < 1e-9, "{msg} value: got {} want {}", got.0, want.0);
            assert!((got.1 - want.1).abs() < 1e-9, "{msg} deriv: got {} want {}", got.1, want.1);
        };
        c(mathieu_cem(0, 5.0, 30.0), (0.17026946498276374, 0.5821239463837029), "cem(0,5,30)");
        c(mathieu_cem(1, 5.0, 30.0), (0.5522262635274278, 1.1204184009096463), "cem(1,5,30)");
        c(mathieu_cem(2, 5.0, 30.0), (0.9065012518998091, 0.3020225775739051), "cem(2,5,30)");
        c(mathieu_cem(3, 10.0, 70.0), (-0.6679194810078108, -1.8140894302215176), "cem(3,10,70)");
        c(mathieu_cem(0, 0.0, 30.0), (0.7071067811865475, 0.0), "cem(0,0,30)");
        c(mathieu_cem(1, -5.0, 30.0), (0.7616380565259091, -1.5978093281414765), "cem(1,-5,30)");
        c(mathieu_sem(1, 5.0, 30.0), (0.16346317035467603, 0.6046025948049121), "sem(1,5,30)");
        c(mathieu_sem(2, 5.0, 30.0), (0.5046345717209944, 1.343445828193893), "sem(2,5,30)");
        c(mathieu_sem(3, 10.0, 70.0), (0.24404071190296778, -4.819189584712689), "sem(3,10,70)");
        c(mathieu_sem(1, 0.0, 45.0), (0.7071067811865477, 0.7071067811865474), "sem(1,0,45)");
        c(mathieu_sem(0, 5.0, 30.0), (0.0, 0.0), "sem(0) -> 0");
        // Large-q cases that exercise the DLMF global-sign convention.
        c(mathieu_cem(4, 50.0, 45.0), (1.1707453018410003, 3.034983794842312), "cem(4,50,45)");
        c(mathieu_cem(3, 20.0, 45.0), (1.1751025369177404, 0.7073419712275579), "cem(3,20,45)");
        c(mathieu_cem(2, 50.0, 60.0), (1.2957144976987285, 3.0058128316097212), "cem(2,50,60)");
        c(mathieu_cem(6, 50.0, 30.0), (1.0228643512401936, 2.9714973827127293), "cem(6,50,30)");
        c(mathieu_sem(3, 50.0, 45.0), (0.37420062721132025, 2.5337601186886447), "sem(3,50,45)");
        c(mathieu_sem(4, 20.0, 60.0), (0.5398091214798074, -5.776690451353455), "sem(4,20,60)");
    }

    #[test]
    fn mathieu_characteristic_values_match_scipy() {
        // frankenscipy: golden from scipy.special.mathieu_a / mathieu_b (1.17.1).
        let c = |got: f64, want: f64, msg: &str| {
            assert!((got - want).abs() < 1e-9, "{msg}: got {got}, want {want}");
        };
        c(mathieu_a(0, 5.0), -5.800046020851508, "a(0,5)");
        c(mathieu_a(1, 5.0), 1.8581875415477505, "a(1,5)");
        c(mathieu_a(2, 5.0), 7.449109739529178, "a(2,5)");
        c(mathieu_a(3, 5.0), 11.548832036343402, "a(3,5)");
        c(mathieu_a(0, 3.0), -2.8343918899043112, "a(0,3)");
        c(mathieu_a(11, 50.0), 132.34094554702267, "a(11,50)");
        c(mathieu_a(2, -10.0), 7.717369849779622, "a(2,-10)");
        c(mathieu_b(1, 5.0), -5.790080598637771, "b(1,5)");
        c(mathieu_b(2, 5.0), 2.0994604454866654, "b(2,5)");
        c(mathieu_b(3, 5.0), 9.2363277136937, "b(3,5)");
        c(mathieu_b(4, 5.0), 16.648219937169777, "b(4,5)");
        c(mathieu_b(4, 10.0), 17.381380678623046, "b(4,10)");
        assert!(mathieu_b(0, 5.0).is_nan(), "b(0,5) -> NaN");
        // q = 0 reduces to a_m(0) = b_m(0) = m^2.
        c(mathieu_a(4, 0.0), 16.0, "a(4,0)");
        c(mathieu_b(3, 0.0), 9.0, "b(3,0)");
    }

    #[test]
    fn assoc_and_sph_legendre_p_match_scipy() {
        // frankenscipy: golden from scipy.special.assoc_legendre_p / sph_legendre_p (1.17.1).
        let c = |got: f64, want: f64, msg: &str| {
            assert!((got - want).abs() < 1e-12, "{msg}: got {got}, want {want}");
        };
        // assoc, unnormalized == lpmv(m, n, z) including signed negative m
        c(assoc_legendre_p(3, 2, 0.4, false), 5.04, "assocF(3,2,0.4)");
        c(assoc_legendre_p(4, -1, -0.3, false), 0.08478134652593106, "assocF(4,-1,-0.3)");
        // assoc, normalized
        c(assoc_legendre_p(3, 2, 0.4, true), 0.8607438643406062, "assocN(3,2,0.4)");
        c(assoc_legendre_p(3, -1, 0.4, true), -0.14849242404917484, "assocN(3,-1,0.4)");
        c(assoc_legendre_p(4, -3, 0.5, true), 1.0189249274664447, "assocN(4,-3,0.5)");
        c(assoc_legendre_p(0, 0, 0.5, true), 0.7071067811865475, "assocN(0,0,0.5)");
        c(assoc_legendre_p(2, 3, 0.5, true), 0.0, "assocN m>n -> 0");
        // spherical
        c(sph_legendre_p(3, 2, 0.7), 0.324400748475796, "sph(3,2,0.7)");
        c(sph_legendre_p(3, -1, 0.7), 0.4007648002460704, "sph(3,-1,0.7)");
        c(sph_legendre_p(4, 0, 1.2), -0.03550969424401845, "sph(4,0,1.2)");
        c(sph_legendre_p(5, 3, 2.0), -0.14528713144968125, "sph(5,3,2.0)");
        c(sph_legendre_p(2, 3, 0.5), 0.0, "sph m>n -> 0");
    }

    #[test]
    fn shifted_polynomial_coeffs_match_scipy() {
        // frankenscipy: golden from scipy.special.{chebyc,chebys,sh_*}(n, ...).c (1.17.1),
        // highest-degree-first. These are linear-argument compositions of the base
        // polynomials: chebyc=2·T_n(x/2), chebys=U_n(x/2), sh_*=base(2x−1).
        assert_coeffs(&chebyc(4), &[1.0, 0.0, -4.0, 0.0, 2.0], "chebyc4");
        assert_coeffs(&chebys(4), &[1.0, 0.0, -3.0, 0.0, 1.0], "chebys4");
        assert_coeffs(&sh_legendre(3), &[20.0, -30.0, 12.0, -1.0], "sh_legendre3");
        assert_coeffs(&sh_chebyt(3), &[32.0, -48.0, 18.0, -1.0], "sh_chebyt3");
        assert_coeffs(&sh_chebyu(3), &[64.0, -96.0, 40.0, -4.0], "sh_chebyu3");
        assert_coeffs(
            &sh_jacobi(3, 2.0, 1.0),
            &[1.0, -1.285_714_285_714, 0.428_571_428_571, -0.028_571_428_571],
            "sh_jacobi(3,2,1)",
        );
        assert_coeffs(
            &sh_jacobi(4, 3.0, 2.0),
            &[1.0, -2.0, 1.333_333_333_333, -0.333_333_333_333, 0.023_809_523_81],
            "sh_jacobi(4,3,2)",
        );
    }

    fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: got {actual}, expected {expected} (diff={})",
            (actual - expected).abs()
        );
    }

    fn assert_pairs_close(
        actual: (Vec<f64>, Vec<f64>),
        expected: (Vec<f64>, Vec<f64>),
        tol: f64,
        msg: &str,
    ) {
        assert_eq!(actual.0.len(), expected.0.len(), "{msg}: node length");
        assert_eq!(actual.1.len(), expected.1.len(), "{msg}: weight length");
        for (idx, (actual_node, expected_node)) in
            actual.0.iter().zip(expected.0.iter()).enumerate()
        {
            assert_close(
                *actual_node,
                *expected_node,
                tol,
                &format!("{msg} node {idx}"),
            );
        }
        for (idx, (actual_weight, expected_weight)) in
            actual.1.iter().zip(expected.1.iter()).enumerate()
        {
            assert_close(
                *actual_weight,
                *expected_weight,
                tol,
                &format!("{msg} weight {idx}"),
            );
        }
    }

    // ── Legendre ──────────────────────────────────────────────────

    #[test]
    fn legendre_p0_is_one() {
        assert_close(eval_legendre(0, 0.5), 1.0, 1e-15, "P_0");
    }

    #[test]
    fn legendre_p0_ignores_nonfinite_input_like_scipy() {
        assert_eq!(eval_legendre(0, f64::NAN), 1.0);
        assert_eq!(eval_legendre(0, f64::INFINITY), 1.0);
        assert_eq!(eval_legendre(0, f64::NEG_INFINITY), 1.0);
    }

    #[test]
    fn legendre_p1_is_x() {
        assert_close(eval_legendre(1, 0.7), 0.7, 1e-15, "P_1");
    }

    #[test]
    fn legendre_p2() {
        // P_2(x) = (3x² - 1) / 2
        let x = 0.6;
        let expected = (3.0 * x * x - 1.0) / 2.0;
        assert_close(eval_legendre(2, x), expected, 1e-14, "P_2");
    }

    #[test]
    fn legendre_p3() {
        // P_3(x) = (5x³ - 3x) / 2
        let x = 0.4;
        let expected = (5.0 * x * x * x - 3.0 * x) / 2.0;
        assert_close(eval_legendre(3, x), expected, 1e-14, "P_3");
    }

    #[test]
    fn legendre_symmetry() {
        // P_n(-x) = (-1)^n P_n(x)
        for n in 0..=8 {
            let x = 0.35;
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert_close(
                eval_legendre(n, -x),
                sign * eval_legendre(n, x),
                1e-12,
                &format!("P_{n} symmetry"),
            );
        }
    }

    #[test]
    fn legendre_at_one() {
        // P_n(1) = 1 for all n
        for n in 0..=10 {
            assert_close(eval_legendre(n, 1.0), 1.0, 1e-12, &format!("P_{n}(1)"));
        }
    }

    // ── Chebyshev T ───────────────────────────────────────────────

    #[test]
    fn chebyt_t0_is_one() {
        assert_close(eval_chebyt(0, 0.5), 1.0, 1e-15, "T_0");
    }

    #[test]
    fn chebyt_t1_is_x() {
        assert_close(eval_chebyt(1, 0.3), 0.3, 1e-15, "T_1");
    }

    #[test]
    fn chebyt_t2() {
        // T_2(x) = 2x² - 1
        let x = 0.6;
        assert_close(eval_chebyt(2, x), 2.0 * x * x - 1.0, 1e-14, "T_2");
    }

    #[test]
    fn chebyt_cos_identity() {
        // T_n(cos θ) = cos(nθ)
        let theta: f64 = 0.7;
        let x = theta.cos();
        for n in 0..=10 {
            let expected = (n as f64 * theta).cos();
            assert_close(eval_chebyt(n, x), expected, 1e-10, &format!("T_{n}(cos θ)"));
        }
    }

    // ── Chebyshev U ───────────────────────────────────────────────

    #[test]
    fn chebyu_u0_is_one() {
        assert_close(eval_chebyu(0, 0.5), 1.0, 1e-15, "U_0");
    }

    #[test]
    fn chebyu_u1_is_2x() {
        assert_close(eval_chebyu(1, 0.3), 0.6, 1e-15, "U_1");
    }

    #[test]
    fn chebyu_u2() {
        // U_2(x) = 4x² - 1
        let x = 0.5;
        assert_close(eval_chebyu(2, x), 4.0 * x * x - 1.0, 1e-14, "U_2");
    }

    // ── Hermite (physicist) ───────────────────────────────────────

    #[test]
    fn hermite_h0_is_one() {
        assert_close(eval_hermite(0, 1.0), 1.0, 1e-15, "H_0");
    }

    #[test]
    fn hermite_h1_is_2x() {
        assert_close(eval_hermite(1, 3.0), 6.0, 1e-15, "H_1");
    }

    #[test]
    fn hermite_h2() {
        // H_2(x) = 4x² - 2
        let x = 1.5;
        assert_close(eval_hermite(2, x), 4.0 * x * x - 2.0, 1e-14, "H_2");
    }

    #[test]
    fn hermite_h3() {
        // H_3(x) = 8x³ - 12x
        let x = 2.0;
        assert_close(eval_hermite(3, x), 8.0 * x * x * x - 12.0 * x, 1e-12, "H_3");
    }

    #[test]
    fn hermite_three_term_recurrence_holds() {
        // /testing-metamorphic for [frankenscipy-8hsiu]:
        //   H_{n+1}(x) = 2x · H_n(x) − 2n · H_{n-1}(x)
        // for n ≥ 1. Pin across high-degree n and several x values
        // — catches drift in the recurrence that low-degree closed
        // forms wouldn't see.
        for &x in &[-2.0_f64, -0.5, 0.5, 2.0] {
            for n in 1u32..=10 {
                let h_nm1 = eval_hermite(n - 1, x);
                let h_n = eval_hermite(n, x);
                let h_np1 = eval_hermite(n + 1, x);
                let lhs = h_np1;
                let rhs = 2.0 * x * h_n - 2.0 * n as f64 * h_nm1;
                let scale = lhs.abs().max(rhs.abs()).max(1.0);
                assert!(
                    (lhs - rhs).abs() < 1e-9 * scale,
                    "Hermite recurrence broken at n={n}, x={x}: \
                     H_{n_plus_one}({x}) = {lhs}, expected {rhs}",
                    n_plus_one = n + 1
                );
            }
        }
    }

    #[test]
    fn legendre_three_term_recurrence_holds() {
        //   (n+1) P_{n+1}(x) = (2n+1) · x · P_n(x) − n · P_{n-1}(x)
        for &x in &[-0.9_f64, -0.3, 0.3, 0.7] {
            for n in 1u32..=10 {
                let p_nm1 = eval_legendre(n - 1, x);
                let p_n = eval_legendre(n, x);
                let p_np1 = eval_legendre(n + 1, x);
                let lhs = (n + 1) as f64 * p_np1;
                let rhs = (2 * n + 1) as f64 * x * p_n - n as f64 * p_nm1;
                let scale = lhs.abs().max(rhs.abs()).max(1.0);
                assert!(
                    (lhs - rhs).abs() < 1e-9 * scale,
                    "Legendre recurrence broken at n={n}, x={x}: \
                     {n}+1 · P_{n_plus_one}({x}) = {lhs}, expected {rhs}",
                    n_plus_one = n + 1
                );
            }
        }
    }

    #[test]
    fn gegenbauer_three_term_recurrence_holds() {
        //   (n+1) C_{n+1}^α(x) = 2(n+α) · x · C_n^α(x)
        //                       − (n + 2α − 1) · C_{n-1}^α(x)
        for &alpha in &[0.5_f64, 1.0, 1.5, 2.5] {
            for &x in &[-0.7_f64, -0.2, 0.4, 0.8] {
                for n in 1u32..=8 {
                    let c_nm1 = eval_gegenbauer(n - 1, alpha, x);
                    let c_n = eval_gegenbauer(n, alpha, x);
                    let c_np1 = eval_gegenbauer(n + 1, alpha, x);
                    let lhs = (n + 1) as f64 * c_np1;
                    let rhs =
                        2.0 * (n as f64 + alpha) * x * c_n - (n as f64 + 2.0 * alpha - 1.0) * c_nm1;
                    let scale = lhs.abs().max(rhs.abs()).max(1.0);
                    assert!(
                        (lhs - rhs).abs() < 1e-9 * scale,
                        "Gegenbauer recurrence broken at n={n}, α={alpha}, \
                         x={x}: lhs={lhs}, rhs={rhs}"
                    );
                }
            }
        }
    }

    // ── Hermite (probabilist) ─────────────────────────────────────

    #[test]
    fn hermitenorm_he0_is_one() {
        assert_close(eval_hermitenorm(0, 1.0), 1.0, 1e-15, "He_0");
    }

    #[test]
    fn hermitenorm_he1_is_x() {
        assert_close(eval_hermitenorm(1, 2.5), 2.5, 1e-15, "He_1");
    }

    #[test]
    fn hermitenorm_he2() {
        // He_2(x) = x² - 1
        let x = 3.0;
        assert_close(eval_hermitenorm(2, x), x * x - 1.0, 1e-14, "He_2");
    }

    // ── Laguerre ──────────────────────────────────────────────────

    #[test]
    fn laguerre_l0_is_one() {
        assert_close(eval_laguerre(0, 5.0), 1.0, 1e-15, "L_0");
    }

    #[test]
    fn laguerre_l1() {
        // L_1(x) = 1 - x
        let x = 3.0;
        assert_close(eval_laguerre(1, x), 1.0 - x, 1e-14, "L_1");
    }

    #[test]
    fn laguerre_l2() {
        // L_2(x) = (x² - 4x + 2) / 2
        let x = 2.0;
        assert_close(
            eval_laguerre(2, x),
            (x * x - 4.0 * x + 2.0) / 2.0,
            1e-14,
            "L_2",
        );
    }

    // ── Generalized Laguerre ──────────────────────────────────────

    #[test]
    fn genlaguerre_alpha0_is_laguerre() {
        for n in 0..=5 {
            let x = 1.5;
            assert_close(
                eval_genlaguerre(n, 0.0, x),
                eval_laguerre(n, x),
                1e-12,
                &format!("L_{n}^0 = L_{n}"),
            );
        }
    }

    #[test]
    fn genlaguerre_l1_alpha() {
        // L_1^α(x) = 1 + α - x
        let alpha = 2.5;
        let x = 3.0;
        assert_close(
            eval_genlaguerre(1, alpha, x),
            1.0 + alpha - x,
            1e-14,
            "L_1^α",
        );
    }

    #[test]
    fn laguerre_degree_zero_nonfinite_inputs_match_scipy() {
        assert_eq!(eval_laguerre(0, f64::INFINITY), 1.0);
        assert_eq!(eval_laguerre(0, f64::NEG_INFINITY), 1.0);
        assert!(eval_laguerre(0, f64::NAN).is_nan());
        assert_eq!(eval_genlaguerre(0, f64::INFINITY, f64::INFINITY), 1.0);
        assert!(eval_genlaguerre(0, f64::NEG_INFINITY, 0.0).is_nan());
    }

    // ── Jacobi ────────────────────────────────────────────────────

    #[test]
    fn jacobi_is_legendre_when_ab_zero() {
        // P_n^{0,0}(x) = P_n(x) (Legendre)
        for n in 0..=6 {
            let x = 0.4;
            assert_close(
                eval_jacobi(n, 0.0, 0.0, x),
                eval_legendre(n, x),
                1e-10,
                &format!("Jacobi(0,0) = Legendre for n={n}"),
            );
        }
    }

    #[test]
    fn jacobi_p0_is_one() {
        assert_close(eval_jacobi(0, 1.5, 2.0, 0.3), 1.0, 1e-15, "P_0^{α,β}");
    }

    #[test]
    fn jacobi_p0_ignores_nonfinite_parameters_like_scipy() {
        assert_eq!(eval_jacobi(0, f64::NAN, f64::INFINITY, f64::NAN), 1.0);
        assert_eq!(
            eval_jacobi(0, f64::NEG_INFINITY, f64::NAN, f64::INFINITY),
            1.0
        );
    }

    #[test]
    fn jacobi_p1() {
        // P_1^{α,β}(x) = (α-β)/2 + (α+β+2)x/2
        let (a, b, x) = (1.0, 2.0, 0.5);
        let expected = 0.5 * ((a - b) + (a + b + 2.0) * x);
        assert_close(eval_jacobi(1, a, b, x), expected, 1e-14, "P_1^{α,β}");
    }

    // ── Gegenbauer ────────────────────────────────────────────────

    #[test]
    fn gegenbauer_c0_is_one() {
        assert_close(eval_gegenbauer(0, 1.5, 0.5), 1.0, 1e-15, "C_0^α");
    }

    #[test]
    fn gegenbauer_c0_nonfinite_inputs_match_scipy() {
        assert_eq!(eval_gegenbauer(0, f64::INFINITY, f64::INFINITY), 1.0);
        assert_eq!(
            eval_gegenbauer(0, f64::NEG_INFINITY, f64::NEG_INFINITY),
            1.0
        );
        assert!(eval_gegenbauer(0, f64::NAN, 0.0).is_nan());
        assert!(eval_gegenbauer(0, 0.5, f64::NAN).is_nan());
    }

    #[test]
    fn gegenbauer_c1_is_2ax() {
        let (alpha, x) = (2.0, 0.3);
        assert_close(
            eval_gegenbauer(1, alpha, x),
            2.0 * alpha * x,
            1e-14,
            "C_1^α",
        );
    }

    #[test]
    fn gegenbauer_c2() {
        // C_2^α(x) = 2α(1+α)x² - α (from recurrence)
        // Verify via recurrence: C_2 = (2(1+α)x * 2αx - (2α)) / 2
        //   = (4α(1+α)x² - 2α) / 2 = 2α(1+α)x² - α
        let (alpha, x) = (1.5, 0.4);
        let expected = 2.0 * alpha * (1.0 + alpha) * x * x - alpha;
        assert_close(eval_gegenbauer(2, alpha, x), expected, 1e-12, "C_2^α");
    }

    #[test]
    fn gegenbauer_half_is_legendre() {
        // C_n^{1/2}(x) = P_n(x) (Legendre)
        for n in 0..=6 {
            let x = 0.6;
            assert_close(
                eval_gegenbauer(n, 0.5, x),
                eval_legendre(n, x),
                1e-10,
                &format!("C_{n}^(1/2) = P_{n}"),
            );
        }
    }

    // ── Quadrature roots and weights ─────────────────────────────

    #[test]
    fn roots_legendre_weights_sum_to_two_and_are_symmetric() {
        let (x, w) = roots_legendre(5);
        assert_eq!(x.len(), 5);
        assert_eq!(w.len(), 5);
        assert_close(w.iter().sum::<f64>(), 2.0, 1e-12, "legendre weight sum");
        for i in 0..(x.len() / 2) {
            let j = x.len() - 1 - i;
            assert_close(x[i], -x[j], 1e-12, "legendre symmetry");
            assert_close(w[i], w[j], 1e-12, "legendre weight symmetry");
        }
    }

    #[test]
    fn roots_legendre_integrates_polynomials_exactly_through_degree_two_n_minus_one() {
        let n = 4;
        let (x, w) = roots_legendre(n);
        for k in 0..(2 * n) {
            let numerical = x
                .iter()
                .zip(&w)
                .map(|(&xi, &wi)| wi * xi.powi(k as i32))
                .sum::<f64>();
            let exact = if k % 2 == 1 {
                0.0
            } else {
                2.0 / (k as f64 + 1.0)
            };
            assert_close(
                numerical,
                exact,
                1e-10,
                &format!("legendre exactness degree {k}"),
            );
        }
    }

    #[test]
    fn roots_chebyt_match_closed_form_and_uniform_weights() {
        let n = 4;
        let (x, w) = roots_chebyt(n);
        let expected = [
            -0.923_879_532_511_286_7,
            -0.382_683_432_365_089_84,
            0.382_683_432_365_089_84,
            0.923_879_532_511_286_7,
        ];
        for (actual, expected) in x.iter().zip(expected) {
            assert_close(*actual, expected, 1e-12, "chebyt node");
        }
        for weight in w {
            assert_close(weight, PI / n as f64, 1e-12, "chebyt weight");
        }
    }

    #[test]
    fn roots_chebyu_match_closed_form() {
        let n = 4;
        let (x, w) = roots_chebyu(n);
        // U_4(x) roots are cos(kπ/5) for k=1,2,3,4
        let expected_nodes = [
            -0.809_016_994_374_947_4, // cos(4π/5)
            -0.309_016_994_374_947_4, // cos(3π/5)
            0.309_016_994_374_947_4,  // cos(2π/5)
            0.809_016_994_374_947_4,  // cos(π/5)
        ];
        for (actual, expected) in x.iter().zip(expected_nodes) {
            assert_close(*actual, expected, 1e-12, "chebyu node");
        }
        // Weights should be π/(n+1) * sin²(kπ/(n+1))
        let weight_sum: f64 = w.iter().sum();
        assert_close(weight_sum, PI / 2.0, 1e-10, "chebyu weight sum = π/2");
    }

    #[test]
    fn roots_hermite_are_symmetric_and_weight_sum_is_sqrt_pi() {
        let (x, w) = roots_hermite(5);
        assert_close(
            w.iter().sum::<f64>(),
            PI.sqrt(),
            1e-10,
            "hermite weight sum",
        );
        for i in 0..(x.len() / 2) {
            let j = x.len() - 1 - i;
            assert_close(x[i], -x[j], 1e-12, "hermite symmetry");
            assert_close(w[i], w[j], 1e-12, "hermite weight symmetry");
        }
    }

    #[test]
    fn roots_hermite_integrates_even_moments() {
        let (x, w) = roots_hermite(4);
        let zeroth = x
            .iter()
            .zip(&w)
            .map(|(&xi, &wi)| wi * xi.powi(0))
            .sum::<f64>();
        let second = x
            .iter()
            .zip(&w)
            .map(|(&xi, &wi)| wi * xi.powi(2))
            .sum::<f64>();
        let fourth = x
            .iter()
            .zip(&w)
            .map(|(&xi, &wi)| wi * xi.powi(4))
            .sum::<f64>();

        assert_close(zeroth, PI.sqrt(), 1e-10, "hermite zeroth moment");
        assert_close(second, PI.sqrt() / 2.0, 1e-10, "hermite second moment");
        assert_close(fourth, 3.0 * PI.sqrt() / 4.0, 1e-9, "hermite fourth moment");
    }

    #[test]
    fn roots_laguerre_are_positive_and_weight_sum_is_one() {
        let (x, w) = roots_laguerre(5);
        assert!(
            x.iter().all(|&xi| xi > 0.0),
            "laguerre nodes should be positive"
        );
        assert_close(w.iter().sum::<f64>(), 1.0, 1e-10, "laguerre weight sum");
    }

    #[test]
    fn roots_laguerre_integrates_low_degree_polynomials() {
        let (x, w) = roots_laguerre(4);
        let first = x.iter().zip(&w).map(|(&xi, &wi)| wi * xi).sum::<f64>();
        let second = x
            .iter()
            .zip(&w)
            .map(|(&xi, &wi)| wi * xi.powi(2))
            .sum::<f64>();
        assert_close(first, 1.0, 1e-10, "laguerre first moment");
        assert_close(second, 2.0, 1e-9, "laguerre second moment");
    }

    #[test]
    fn roots_jacobi_matches_legendre_when_alpha_beta_zero() {
        let (x_j, w_j) = roots_jacobi(5, 0.0, 0.0);
        let (x_l, w_l) = roots_legendre(5);
        for (actual, expected) in x_j.iter().zip(&x_l) {
            assert_close(*actual, *expected, 1e-12, "jacobi nodes vs legendre");
        }
        for (actual, expected) in w_j.iter().zip(&w_l) {
            assert_close(*actual, *expected, 1e-12, "jacobi weights vs legendre");
        }
    }

    #[test]
    fn roots_jacobi_half_half_has_expected_weight_sum_and_symmetry() {
        let (x, w) = roots_jacobi(3, 0.5, 0.5);
        assert_close(w.iter().sum::<f64>(), PI / 2.0, 1e-10, "jacobi mu0");
        assert_close(x[0], -x[2], 1e-12, "jacobi symmetry");
        assert_close(w[0], w[2], 1e-12, "jacobi weight symmetry");
        assert_close(x[1], 0.0, 1e-12, "jacobi center root");
    }

    #[test]
    fn roots_genlaguerre_alpha0_matches_laguerre() {
        // When alpha=0, generalized Laguerre = ordinary Laguerre
        let (x_gen, w_gen) = roots_genlaguerre(5, 0.0);
        let (x_lag, w_lag) = roots_laguerre(5);
        for (actual, expected) in x_gen.iter().zip(&x_lag) {
            assert_close(*actual, *expected, 1e-10, "genlaguerre(alpha=0) nodes");
        }
        for (actual, expected) in w_gen.iter().zip(&w_lag) {
            assert_close(*actual, *expected, 1e-10, "genlaguerre(alpha=0) weights");
        }
    }

    #[test]
    fn roots_genlaguerre_positive_nodes_and_weight_sum() {
        // For alpha > -1, all nodes are positive
        // Weight sum = Gamma(alpha + 1)
        let alpha = 1.5;
        let (x, w) = roots_genlaguerre(5, alpha);
        assert!(
            x.iter().all(|&xi| xi > 0.0),
            "genlaguerre nodes should be positive"
        );
        // Gamma(2.5) = 1.5 * Gamma(1.5) = 1.5 * sqrt(pi)/2 ≈ 1.3293
        let expected_sum = 1.5 * PI.sqrt() / 2.0;
        assert_close(
            w.iter().sum::<f64>(),
            expected_sum,
            1e-10,
            "genlaguerre weight sum = Gamma(alpha+1)",
        );
    }

    #[test]
    fn roots_genlaguerre_invalid_alpha_returns_nan_rule() {
        for alpha in [-1.0, -2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let (x, w) = roots_genlaguerre(3, alpha);
            assert_eq!(x.len(), 3);
            assert_eq!(w.len(), 3);
            assert!(x.iter().all(|v| v.is_nan()));
            assert!(w.iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn roots_gegenbauer_alpha_half_matches_legendre() {
        // Gegenbauer with alpha=0.5 corresponds to Legendre
        // (weight function (1-x²)^0 = 1)
        let (x_g, w_g) = roots_gegenbauer(5, 0.5);
        let (x_l, w_l) = roots_legendre(5);
        for (actual, expected) in x_g.iter().zip(&x_l) {
            assert_close(
                *actual,
                *expected,
                1e-12,
                "gegenbauer(0.5) vs legendre nodes",
            );
        }
        for (actual, expected) in w_g.iter().zip(&w_l) {
            assert_close(
                *actual,
                *expected,
                1e-12,
                "gegenbauer(0.5) vs legendre weights",
            );
        }
    }

    #[test]
    fn roots_gegenbauer_symmetric_and_weight_sum() {
        // Gegenbauer with alpha=1 has weight function (1-x²)^0.5
        // mu0 = integral of (1-x²)^0.5 from -1 to 1 = π/2
        let (x, w) = roots_gegenbauer(4, 1.0);
        assert_close(x[0], -x[3], 1e-12, "gegenbauer symmetry");
        assert_close(x[1], -x[2], 1e-12, "gegenbauer symmetry");
        assert_close(w[0], w[3], 1e-12, "gegenbauer weight symmetry");
        assert_close(
            w.iter().sum::<f64>(),
            PI / 2.0,
            1e-10,
            "gegenbauer weight sum",
        );
    }

    #[test]
    fn roots_gegenbauer_invalid_alpha_returns_nan_rule() {
        for alpha in [-0.5, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let (x, w) = roots_gegenbauer(3, alpha);
            assert_eq!(x.len(), 3);
            assert_eq!(w.len(), 3);
            assert!(x.iter().all(|v| v.is_nan()));
            assert!(w.iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn roots_jacobi_invalid_parameters_return_nan_rule() {
        for (alpha, beta) in [
            (-1.0, 0.0),
            (0.0, -1.0),
            (f64::NAN, 0.0),
            (0.0, f64::NAN),
            (f64::INFINITY, 0.0),
            (0.0, f64::NEG_INFINITY),
        ] {
            let (x, w) = roots_jacobi(3, alpha, beta);
            assert_eq!(x.len(), 3);
            assert_eq!(w.len(), 3);
            assert!(x.iter().all(|v| v.is_nan()));
            assert!(w.iter().all(|v| v.is_nan()));
        }
    }

    // ── Associated Legendre tests ────────────────────────────────────

    #[test]
    fn lpmv_m0_equals_legendre() {
        // P_l^0(x) = P_l(x) (ordinary Legendre)
        for l in 0..=8 {
            let x = 0.6;
            assert_close(
                lpmv(0, l, x),
                eval_legendre(l, x),
                1e-12,
                &format!("P_{l}^0 = P_{l}"),
            );
        }
    }

    #[test]
    fn lpmv_known_values() {
        // P_1^1(x) = -(1-x²)^{1/2}
        let x = 0.5;
        assert_close(lpmv(1, 1, x), -(1.0 - x * x).sqrt(), 1e-12, "P_1^1");

        // P_2^1(x) = -3x(1-x²)^{1/2}
        assert_close(
            lpmv(1, 2, x),
            -3.0 * x * (1.0 - x * x).sqrt(),
            1e-12,
            "P_2^1",
        );

        // P_2^2(x) = 3(1-x²)
        assert_close(lpmv(2, 2, x), 3.0 * (1.0 - x * x), 1e-12, "P_2^2");
    }

    #[test]
    fn lpmv_negative_m_relation() {
        // P_l^{-m}(x) = (-1)^m * (l-m)!/(l+m)! * P_l^m(x)
        let x = 0.4;
        let l = 3_u32;
        let m = 2_i32;
        let positive = lpmv(m, l, x);
        let negative = lpmv(-m, l, x);
        // (l-m)!/(l+m)! = 1!/(5!) = 1/120
        let ratio = 1.0 / 120.0;
        let sign = 1.0; // (-1)^2 = 1
        assert_close(negative, sign * ratio * positive, 1e-12, "P_l^{-m}");
    }

    #[test]
    fn lpmv_m_exceeds_l_is_zero() {
        assert_eq!(lpmv(5, 3, 0.5), 0.0);
        assert_eq!(lpmv(-5, 3, 0.5), 0.0);
    }

    #[test]
    fn lpmv_p00_ignores_nonfinite_input_like_scipy() {
        assert_eq!(lpmv(0, 0, f64::NAN), 1.0);
        assert_eq!(lpmv(0, 0, f64::INFINITY), 1.0);
        assert_eq!(lpmv(0, 0, f64::NEG_INFINITY), 1.0);
    }

    #[test]
    fn lpmv_at_endpoints() {
        // P_l^0(1) = 1, P_l^m(1) = 0 for m > 0
        for l in 0..=5 {
            assert_close(lpmv(0, l, 1.0), 1.0, 1e-12, &format!("P_{l}^0(1)"));
            if l > 0 {
                assert_close(lpmv(1, l, 1.0), 0.0, 1e-12, &format!("P_{l}^1(1)"));
            }
        }
    }

    // ── Spherical harmonics tests ────────────────────────────────────

    #[test]
    fn sph_harm_y00_is_constant() {
        // Y_0^0 = 1/√(4π)
        let expected = 1.0 / (4.0 * PI).sqrt();
        for theta in [0.0, 1.0, 3.0, 5.0] {
            for phi in [0.0, 0.5, 1.5, PI] {
                let y = sph_harm(0, 0, theta, phi);
                assert_close(y.re, expected, 1e-12, "Y_0^0 re");
                assert_close(y.im, 0.0, 1e-12, "Y_0^0 im");
            }
        }
    }

    #[test]
    fn sph_harm_normalization() {
        // |Y_l^m|² integrated over sphere = 1
        // Approximate via numerical integration (trapezoidal on theta/phi grid)
        let n_theta = 100;
        let n_phi = 50;
        let d_theta = 2.0 * PI / n_theta as f64;
        let d_phi = PI / n_phi as f64;

        for (l, m) in [(1, 0), (1, 1), (2, -1), (2, 2), (3, -2)] {
            let mut integral = 0.0;
            for i in 0..n_theta {
                let theta = (i as f64 + 0.5) * d_theta;
                for j in 0..n_phi {
                    let phi = (j as f64 + 0.5) * d_phi;
                    let y = sph_harm(m, l, theta, phi);
                    let abs2 = y.re * y.re + y.im * y.im;
                    integral += abs2 * phi.sin() * d_theta * d_phi;
                }
            }
            assert!(
                (integral - 1.0).abs() < 0.02,
                "Y_{l}^{m} normalization: integral={integral}"
            );
        }
    }

    #[test]
    fn sph_harm_y_m_exceeds_l() {
        let y = sph_harm(5, 3, 1.0, 1.0);
        assert_eq!(y.re, 0.0);
        assert_eq!(y.im, 0.0);
    }

    #[test]
    fn sph_harm_y_y00_matches_scipy_constant() {
        let expected = 1.0 / (4.0 * PI).sqrt();
        let y = sph_harm_y(0, 0, 1.0, 0.5);
        assert_close(y.re, expected, 1e-12, "sph_harm_y Y_0^0 re");
        assert_close(y.im, 0.0, 1e-12, "sph_harm_y Y_0^0 im");
    }

    #[test]
    fn sph_harm_y_input_independent_cases_allow_nonfinite_angles() {
        let expected = 1.0 / (4.0 * PI).sqrt();
        let y = sph_harm_y(0, 0, f64::INFINITY, f64::NAN);
        assert_close(y.re, expected, 1e-12, "Y_0^0 re");
        assert_close(y.im, 0.0, 1e-12, "Y_0^0 im");

        let outside_order = sph_harm_y(0, 1, f64::NAN, f64::INFINITY);
        assert_close(outside_order.re, 0.0, 1e-12, "|m| > l re");
        assert_close(outside_order.im, 0.0, 1e-12, "|m| > l im");
    }

    #[test]
    fn sph_harm_y_uses_scipy_complex_angle_convention() {
        let y10 = sph_harm_y(1, 0, 0.5, 1.0);
        assert_close(y10.re, 0.42878904414183583, 1e-12, "Y_1^0 re");
        assert_close(y10.im, 0.0, 1e-12, "Y_1^0 im");

        let y11 = sph_harm_y(1, 1, 0.5, 1.0);
        assert_close(y11.re, -0.08949498165189648, 1e-12, "Y_1^1 re");
        assert_close(y11.im, -0.13938017574251232, 1e-12, "Y_1^1 im");
    }

    #[test]
    fn sph_harm_y_negative_order_matches_scipy() {
        // scipy.special.sph_harm_y applies Y_n^{-m} = (-1)^m·conj(Y_n^m);
        // odd negative orders therefore flip sign relative to a plain
        // conjugate. Reference values from scipy.special.sph_harm_y.
        let cases: &[(u32, i32, f64, f64)] = &[
            (1, -1, 0.089_494_981_651_896_48, -0.139_380_175_742_512_32),
            (2, -1, 0.175_619_068_974_414_95, -0.273_510_494_617_455_9),
            (2, -2, -0.036_947_463_710_166_74, -0.080_731_681_053_122_66),
            (3, -1, 0.238_650_704_669_713_22, -0.371_676_450_946_947_1),
            (3, -2, -0.085_787_030_722_668_29, -0.187_448_081_879_870_83),
            (
                3,
                -3,
                -0.045_516_042_712_139_326,
                -0.006_488_154_543_036_636,
            ),
        ];
        for &(n, m, re, im) in cases {
            let y = sph_harm_y(n, m, 0.5, 1.0);
            assert_close(y.re, re, 1e-12, &format!("Y_{n}^{m} re"));
            assert_close(y.im, im, 1e-12, &format!("Y_{n}^{m} im"));
        }
    }

    // ── Shifted polynomial tests ─────────────────────────────────────

    #[test]
    fn sh_legendre_at_endpoints() {
        // P_n*(0) = P_n(-1) = (-1)^n
        for n in 0..=6 {
            let expected = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert_close(
                eval_sh_legendre(n, 0.0),
                expected,
                1e-12,
                &format!("P*_{n}(0)"),
            );
        }
        // P_n*(1) = P_n(1) = 1
        for n in 0..=6 {
            assert_close(eval_sh_legendre(n, 1.0), 1.0, 1e-12, &format!("P*_{n}(1)"));
        }
    }

    #[test]
    fn sh_legendre_midpoint() {
        // P_n*(0.5) = P_n(0) = 0 for odd n, nonzero for even n
        assert_close(eval_sh_legendre(1, 0.5), 0.0, 1e-12, "P*_1(0.5)");
        assert_close(eval_sh_legendre(3, 0.5), 0.0, 1e-12, "P*_3(0.5)");
    }

    #[test]
    fn sh_chebyt_at_endpoints() {
        // T_n*(0) = T_n(-1) = (-1)^n
        for n in 0..=6 {
            let expected = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert_close(
                eval_sh_chebyt(n, 0.0),
                expected,
                1e-12,
                &format!("T*_{n}(0)"),
            );
        }
    }

    #[test]
    fn sh_chebyu_at_midpoint() {
        // U_n*(0.5) = U_n(0) = 0 for odd n
        assert_close(eval_sh_chebyu(1, 0.5), 0.0, 1e-12, "U*_1(0.5)");
        assert_close(eval_sh_chebyu(3, 0.5), 0.0, 1e-12, "U*_3(0.5)");
    }

    #[test]
    fn lqmn_first_row_matches_lqn() {
        let x = 0.4;
        let arr = lqmn(2, 6, x);
        let (vals, _) = lqn(6, x);
        for l in 0..=6 {
            assert!(
                (arr[0][l as usize] - vals[l as usize]).abs() < 1e-13,
                "lqmn[0][{l}] = {}, lqn = {}",
                arr[0][l as usize],
                vals[l as usize]
            );
        }
    }

    #[test]
    fn lqmn_metamorphic_dlmf_recurrence() {
        // The internal builder uses DLMF 14.10.4
        //   √(1-x²) Q_l^m = (l-m+1) x Q_l^{m-1} - (l+m-1) Q_{l-1}^{m-1}
        // Verify the same identity holds for the produced grid.
        let x = 0.3;
        let arr = lqmn(3, 5, x);
        let denom = (1.0_f64 - x * x).sqrt();
        for m in 1..=3_u32 {
            for l in m..=5_u32 {
                let lf = l as f64;
                let mf = m as f64;
                let lhs = arr[m as usize][l as usize] * denom;
                let rhs = (lf - mf + 1.0) * x * arr[m as usize - 1][l as usize]
                    - (lf + mf - 1.0) * arr[m as usize - 1][l as usize - 1];
                assert!(
                    (lhs - rhs).abs() < 1e-12,
                    "DLMF 14.10.4 violated at (m={m}, l={l})"
                );
            }
        }
    }

    #[test]
    fn lqmn_outside_unit_interval_returns_nan() {
        let arr = lqmn(2, 3, 1.5);
        for row in &arr {
            for &v in row {
                assert!(v.is_nan());
            }
        }
        let arr = lqmn(2, 3, -1.0);
        for row in &arr {
            for &v in row {
                assert!(v.is_nan());
            }
        }
    }

    #[test]
    fn lqmn_below_diagonal_matches_scipy_derivative_convention() {
        let arr = lqmn(3, 3, 0.5);
        let expected = [
            [-1.154_700_538_379_251_5, 0.0, 0.0],
            [1.333_333_333_333_333_3, 2.666_666_666_666_666_5, 0.0],
            [
                -5.388_602_512_436_506,
                -6.158_402_871_356_006,
                -12.316_805_742_712_011,
            ],
        ];
        for m in 1..=3 {
            for l in 0..m {
                assert!(
                    (arr[m][l] - expected[m - 1][l]).abs() < 1e-12,
                    "lqmn[{m}][{l}] = {}, expected {}",
                    arr[m][l],
                    expected[m - 1][l]
                );
            }
        }
    }

    #[test]
    fn lpmn_metamorphic_matches_lpmv_pointwise() {
        // For each (m, l) in 0..=4, lpmn[m][l] must agree with lpmv(m, l, x).
        let xs = [-0.7_f64, -0.3, 0.0, 0.3, 0.7];
        for &x in &xs {
            let arr = lpmn(4, 4, x);
            for m in 0..=4 {
                for l in 0..=4 {
                    let from_array = arr[m as usize][l as usize];
                    let from_scalar = lpmv(m as i32, l, x);
                    if l < m {
                        // m > l ⇒ 0 by convention
                        assert!(
                            from_array.abs() < 1e-15,
                            "lpmn[{m}][{l}] should be 0, got {from_array}"
                        );
                    } else {
                        assert!(
                            (from_array - from_scalar).abs() < 1e-12,
                            "lpmn[{m}][{l}]={from_array} vs lpmv={from_scalar} at x={x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lpmn_first_row_matches_lpn() {
        // m=0 row is just standard Legendre P_l(x).
        let x = 0.4;
        let arr = lpmn(2, 6, x);
        let (vals, _) = lpn(6, x);
        for l in 0..=6 {
            assert!(
                (arr[0][l as usize] - vals[l as usize]).abs() < 1e-13,
                "row m=0, l={l}: {} vs {}",
                arr[0][l as usize],
                vals[l as usize]
            );
        }
    }

    #[test]
    fn lpmn_propagates_nan_for_non_finite_x() {
        let arr = lpmn(2, 3, f64::NAN);
        for row in &arr {
            for &v in row {
                assert!(v.is_nan(), "expected NaN, got {v}");
            }
        }
    }

    #[test]
    fn lpn_zero_order_returns_singletons() {
        let (vals, ders) = lpn(0, 0.5);
        assert_eq!(vals, vec![1.0]);
        assert_eq!(ders, vec![0.0]);
    }

    #[test]
    fn lpn_first_kind_known_values_at_half() {
        // Hand-computed P_k(0.5):
        // P_0=1, P_1=0.5, P_2=-0.125, P_3=-0.4375, P_4=-0.2890625
        let (vals, _) = lpn(4, 0.5);
        let expected = [1.0, 0.5, -0.125, -0.4375, -0.2890625];
        for (k, e) in expected.iter().enumerate() {
            assert_close(vals[k], *e, 1e-15, &format!("P_{k}(0.5)"));
        }
    }

    #[test]
    fn lpn_metamorphic_endpoint_values() {
        // P_k(1) = 1 for all k; P_k(-1) = (-1)^k.
        let (vals_plus, _) = lpn(8, 1.0);
        let (vals_minus, _) = lpn(8, -1.0);
        for k in 0..=8 {
            assert!(
                (vals_plus[k] - 1.0).abs() < 1e-12,
                "P_{k}(1) = {} should be 1",
                vals_plus[k]
            );
            let expected = if k % 2 == 0 { 1.0 } else { -1.0 };
            assert!(
                (vals_minus[k] - expected).abs() < 1e-12,
                "P_{k}(-1) should be {expected}, got {}",
                vals_minus[k]
            );
        }
    }

    #[test]
    fn lpn_metamorphic_parity() {
        // P_k(-x) = (-1)^k P_k(x).
        let x = 0.37;
        let (a, _) = lpn(7, x);
        let (b, _) = lpn(7, -x);
        for k in 0..=7 {
            let expected = if k % 2 == 0 { a[k] } else { -a[k] };
            assert!(
                (b[k] - expected).abs() < 1e-12,
                "parity violated at k={k}: P_k(-x)={}, expected {expected}",
                b[k]
            );
        }
    }

    #[test]
    fn lpn_derivative_endpoint_closed_form() {
        // P'_k(±1) = (±1)^(k+1) k(k+1) / 2.
        let (_, ders_plus) = lpn(5, 1.0);
        let (_, ders_minus) = lpn(5, -1.0);
        for k in 1..=5 {
            let kf = k as f64;
            let half_kk1 = 0.5 * kf * (kf + 1.0);
            assert!(
                (ders_plus[k] - half_kk1).abs() < 1e-12,
                "P'_{k}(1) should be {half_kk1}, got {}",
                ders_plus[k]
            );
            let expected = if k % 2 == 0 { -half_kk1 } else { half_kk1 };
            assert!(
                (ders_minus[k] - expected).abs() < 1e-12,
                "P'_{k}(-1) should be {expected}, got {}",
                ders_minus[k]
            );
        }
    }

    #[test]
    fn lpn_value_matches_eval_legendre() {
        // The lpn array's k-th entry must agree with eval_legendre(k, x) for
        // every k. This is the metamorphic invariant tying lpn to the
        // pre-existing scalar implementation.
        let xs = [-0.9, -0.3, 0.0, 0.3, 0.9];
        for &x in &xs {
            let (vals, _) = lpn(10, x);
            for (k, value) in vals.iter().enumerate().take(11) {
                let scalar = eval_legendre(k as u32, x);
                assert!(
                    (*value - scalar).abs() < 1e-12,
                    "lpn[{k}] vs eval_legendre at x={x}: {} vs {scalar}",
                    value
                );
            }
        }
    }

    #[test]
    fn lpn_non_finite_x_propagates_nan() {
        let (vals, ders) = lpn(3, f64::NAN);
        assert_eq!(vals.len(), 4);
        assert_eq!(ders.len(), 4);
        for v in vals.iter().chain(ders.iter()) {
            assert!(v.is_nan(), "expected NaN, got {v}");
        }
    }

    #[test]
    fn lqn_zero_order_q0_is_atanh() {
        let (vals, ders) = lqn(0, 0.5);
        assert_close(vals[0], 0.5_f64.atanh(), 1e-15, "Q_0(0.5)");
        // Q'_0 = 1/(1-x²) = 1/0.75
        assert_close(ders[0], 1.0 / 0.75, 1e-15, "Q'_0(0.5)");
    }

    #[test]
    fn lqn_q1_relation_to_q0() {
        // Q_1(x) = x · Q_0(x) - 1.
        let x = 0.3;
        let (vals, _) = lqn(1, x);
        let expected = x * x.atanh() - 1.0;
        assert_close(vals[1], expected, 1e-15, "Q_1(0.3)");
    }

    #[test]
    fn lqn_metamorphic_parity() {
        // Q_k(-x) = (-1)^(k+1) Q_k(x).
        let x = 0.4;
        let (a, _) = lqn(6, x);
        let (b, _) = lqn(6, -x);
        for k in 0..=6 {
            let expected = if k % 2 == 0 { -a[k] } else { a[k] };
            assert_close(b[k], expected, 1e-12, &format!("parity at k={k}"));
        }
    }

    #[test]
    fn lqn_metamorphic_origin_zero_for_even_indices() {
        // Q_{2m}(0) = 0 for all m ≥ 0.
        let (vals, _) = lqn(8, 0.0);
        for k in (0..=8).step_by(2) {
            assert!(
                vals[k].abs() < 1e-15,
                "Q_{k}(0) should be 0, got {}",
                vals[k]
            );
        }
        // Q_1(0) = -1, Q_3(0) = 2/3, Q_5(0) = -8/15, Q_7(0) = 16/35.
        assert_close(vals[1], -1.0, 1e-15, "Q_1(0)");
        assert_close(vals[3], 2.0 / 3.0, 1e-15, "Q_3(0)");
        assert_close(vals[5], -8.0 / 15.0, 1e-15, "Q_5(0)");
        assert_close(vals[7], 16.0 / 35.0, 1e-15, "Q_7(0)");
    }

    #[test]
    fn lqn_outside_unit_interval_returns_nan() {
        let (vals_above, ders_above) = lqn(3, 1.5);
        for v in vals_above.iter().chain(ders_above.iter()) {
            assert!(v.is_nan(), "expected NaN, got {v}");
        }
        let (vals_at_one, _) = lqn(2, 1.0);
        for v in &vals_at_one {
            assert!(v.is_nan(), "expected NaN at x=1, got {v}");
        }
    }

    #[test]
    fn roots_sh_legendre_in_unit_interval() {
        let (nodes, weights) = roots_sh_legendre(7);
        assert_eq!(nodes.len(), 7);
        assert_eq!(weights.len(), 7);
        for x in &nodes {
            assert!(*x > 0.0 && *x < 1.0, "node {x} outside (0, 1)");
        }
        for w in &weights {
            assert!(*w > 0.0, "weight {w} should be positive");
        }
    }

    #[test]
    fn roots_sh_legendre_metamorphic_weight_sum_is_one() {
        // The shifted-Legendre weights are halved so they integrate against
        // unit weight on [0, 1]. Their sum equals the integral of 1 on
        // [0, 1] = 1.
        for n in 1..=8 {
            let (_, w) = roots_sh_legendre(n);
            let sum: f64 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "n={n}: shifted-Legendre weight sum {sum} != 1"
            );
        }
    }

    #[test]
    fn roots_sh_legendre_metamorphic_node_relation() {
        // Shifted node = (1 + unshifted node) / 2.
        let n = 6;
        let (sh_nodes, sh_weights) = roots_sh_legendre(n);
        let (un_nodes, un_weights) = roots_legendre(n);
        for (sh, un) in sh_nodes.iter().zip(un_nodes.iter()) {
            let expected = 0.5 * (1.0 + un);
            assert!(
                (sh - expected).abs() < 1e-13,
                "shifted node {sh} should be (1+{un})/2 = {expected}"
            );
        }
        for (sh, un) in sh_weights.iter().zip(un_weights.iter()) {
            assert!(
                (sh - 0.5 * un).abs() < 1e-13,
                "shifted weight {sh} should be {un}/2"
            );
        }
    }

    #[test]
    fn roots_sh_chebyt_in_unit_interval_with_uniform_weights() {
        let n = 5;
        let (nodes, weights) = roots_sh_chebyt(n);
        for x in &nodes {
            assert!(*x > 0.0 && *x < 1.0);
        }
        // Shifted Chebyshev-T weights are uniform π/n: the Chebyshev-T
        // weight function is invariant under the [-1,1]→[0,1] shift.
        let expected = std::f64::consts::PI / n as f64;
        for w in &weights {
            assert!((w - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn roots_sh_chebyu_in_unit_interval() {
        let (nodes, weights) = roots_sh_chebyu(6);
        assert_eq!(nodes.len(), 6);
        for x in &nodes {
            assert!(*x > 0.0 && *x < 1.0);
        }
        for w in &weights {
            assert!(*w > 0.0);
        }
    }

    #[test]
    fn roots_sh_n_zero_returns_empty() {
        for (a, b) in [roots_sh_legendre(0), roots_sh_chebyt(0), roots_sh_chebyu(0)] {
            assert!(a.is_empty() && b.is_empty());
        }
    }

    #[test]
    fn roots_chebyc_basic_dimensions_and_range() {
        let (nodes, weights) = roots_chebyc(5);
        assert_eq!(nodes.len(), 5);
        assert_eq!(weights.len(), 5);
        for x in &nodes {
            assert!(*x > -2.0 && *x < 2.0, "node {x} outside (-2, 2)");
        }
        // Uniform weights 2π / n.
        let expected = 2.0 * std::f64::consts::PI / 5.0;
        for w in &weights {
            assert!((w - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn roots_chebyc_metamorphic_scales_chebyt_by_two() {
        // C_n's roots are exactly 2 × T_n's roots.
        let n = 7;
        let (c_nodes, _) = roots_chebyc(n);
        let (t_nodes, _) = roots_chebyt(n);
        assert_eq!(c_nodes.len(), t_nodes.len());
        for (c, t) in c_nodes.iter().zip(t_nodes.iter()) {
            assert!(
                (c - 2.0 * t).abs() < 1e-13,
                "C-root {c} should be 2 × T-root {t}"
            );
        }
    }

    #[test]
    fn roots_chebyc_weight_sum_equals_two_pi() {
        // ∑ w_k = n · (2π / n) = 2π for any n — the weight function
        // 1/√(1−x²/4) integrates to 2π over [-2, 2].
        for n in 1..=8 {
            let (_, weights) = roots_chebyc(n);
            let sum: f64 = weights.iter().sum();
            assert!(
                (sum - 2.0 * std::f64::consts::PI).abs() < 1e-13,
                "n={n}: weight sum {sum} != 2π"
            );
        }
    }

    #[test]
    fn roots_chebys_basic_dimensions_and_range() {
        let (nodes, weights) = roots_chebys(6);
        assert_eq!(nodes.len(), 6);
        assert_eq!(weights.len(), 6);
        for x in &nodes {
            assert!(*x > -2.0 && *x < 2.0, "S-node {x} outside (-2, 2)");
        }
        for w in &weights {
            assert!(*w > 0.0, "weight {w} should be positive");
        }
    }

    #[test]
    fn roots_chebys_metamorphic_scales_chebyu_by_two() {
        // S_n's roots are exactly 2 × U_n's roots.
        let n = 6;
        let (s_nodes, s_weights) = roots_chebys(n);
        let (u_nodes, u_weights) = roots_chebyu(n);
        assert_eq!(s_nodes.len(), u_nodes.len());
        for (s, u) in s_nodes.iter().zip(u_nodes.iter()) {
            assert!(
                (s - 2.0 * u).abs() < 1e-13,
                "S-root {s} should be 2 × U-root {u}"
            );
        }
        // S weights are twice the U weights: the [-2, 2] interval doubles
        // the integral of the weight function relative to [-1, 1].
        for (sw, uw) in s_weights.iter().zip(u_weights.iter()) {
            assert!(
                (sw - 2.0 * uw).abs() < 1e-15,
                "S weight {sw} should be 2 × U weight {uw}"
            );
        }
    }

    #[test]
    fn roots_chebyc_n_zero_returns_empty() {
        let (n, w) = roots_chebyc(0);
        assert!(n.is_empty() && w.is_empty());
        let (n, w) = roots_chebys(0);
        assert!(n.is_empty() && w.is_empty());
    }

    #[test]
    fn roots_cheby_scaled_weights_match_scipy() {
        // Weights pinned against scipy.special for n = 5. fsci previously
        // diverged by a constant scale factor (frankenscipy-s2qfz).
        let check = |label: &str, got: &[f64], want: &[f64]| {
            assert_eq!(got.len(), want.len(), "{label}: length");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!((g - w).abs() < 1e-13, "{label}[{i}]: got {g}, want {w}");
            }
        };

        let (_, w) = roots_chebyc(5);
        check("chebyc", &w, &[1.256_637_061_435_917_2; 5]);

        let (_, w) = roots_chebys(5);
        check(
            "chebys",
            &w,
            &[
                0.261_799_387_799_149_35,
                0.785_398_163_397_448_4,
                1.047_197_551_196_597_6,
                0.785_398_163_397_448_2,
                0.261_799_387_799_149_35,
            ],
        );

        let (_, w) = roots_sh_chebyt(5);
        check("sh_chebyt", &w, &[0.628_318_530_717_958_6; 5]);

        let (_, w) = roots_sh_chebyu(5);
        check(
            "sh_chebyu",
            &w,
            &[
                0.032_724_923_474_893_66,
                0.098_174_770_424_681_03,
                0.130_899_693_899_574_68,
                0.098_174_770_424_681_01,
                0.032_724_923_474_893_66,
            ],
        );
    }

    #[test]
    fn lqn_derivative_recurrence_consistency() {
        // (1 - x²) Q'_k(x) = k (Q_{k-1} − x Q_k) — the same recurrence we
        // use to compute the derivative. Validate symmetrically by computing
        // it from the values array directly and comparing against the lqn
        // output.
        let x = 0.6;
        let (vals, ders) = lqn(5, x);
        let one_minus_xsq = 1.0 - x * x;
        for k in 1..=5 {
            let kf = k as f64;
            let direct = kf * (vals[k - 1] - x * vals[k]) / one_minus_xsq;
            assert_close(
                ders[k],
                direct,
                1e-12,
                &format!("derivative recurrence at k={k}"),
            );
        }
    }

    #[test]
    fn eval_sh_jacobi_matches_scipy() {
        // eval_sh_jacobi(n, p, q, x) = P_n^{(p−q, q−1)}(2x−1) / C(2n+p−1, n).
        // The normalization (previously omitted) is what scipy divides by.
        // Golden values from scipy.special.eval_sh_jacobi (SciPy 1.17.1).
        for &(n, p, q, x, golden) in &[
            (3_u32, 5.0_f64, 2.0, 0.3, 0.005_666_666_666_666_663),
            (4, 4.0, 3.0, 0.5, 0.001_893_939_393_939_394),
            (5, 6.5, 1.5, 0.75, 0.000_590_244_822_145_379_2),
            (0, 3.0, 1.0, 0.1, 1.0),
            (1, 4.0, 2.0, 0.9, 0.5),
        ] {
            let direct = eval_sh_jacobi(n, p, q, x);
            assert!(
                (direct - golden).abs() <= 1e-12 + 1e-10 * golden.abs(),
                "eval_sh_jacobi({n}, {p}, {q}, {x}) = {direct} vs scipy {golden}"
            );
        }
    }

    #[test]
    fn roots_sh_jacobi_matches_shifted_jacobi_relation() {
        for &(n, p, q) in &[(3_usize, 0.5_f64, 1.25_f64), (5, 1.5, 0.75), (8, 2.0, 2.0)] {
            let (shifted_nodes, shifted_weights) = roots_sh_jacobi(n, p, q);
            let (jacobi_nodes, jacobi_weights) = roots_jacobi(n, p - q, q - 1.0);
            let scale = 2.0_f64.powf(-p);
            for i in 0..n {
                assert!(
                    (shifted_nodes[i] - 0.5 * (1.0 + jacobi_nodes[i])).abs() < 1e-12,
                    "roots_sh_jacobi node {i} for n={n}, p={p}, q={q}"
                );
                assert!(
                    (shifted_weights[i] - scale * jacobi_weights[i]).abs() < 1e-12,
                    "roots_sh_jacobi weight {i} for n={n}, p={p}, q={q}"
                );
            }
        }
    }

    #[test]
    fn scipy_root_aliases_delegate_to_long_names() {
        assert_pairs_close(p_roots(5), roots_legendre(5), 1e-12, "p_roots");
        assert_pairs_close(ps_roots(5), roots_sh_legendre(5), 1e-12, "ps_roots");
        assert_pairs_close(t_roots(5), roots_chebyt(5), 1e-12, "t_roots");
        assert_pairs_close(ts_roots(5), roots_sh_chebyt(5), 1e-12, "ts_roots");
        assert_pairs_close(u_roots(5), roots_chebyu(5), 1e-12, "u_roots");
        assert_pairs_close(us_roots(5), roots_sh_chebyu(5), 1e-12, "us_roots");
        assert_pairs_close(c_roots(5), roots_chebyc(5), 1e-12, "c_roots");
        assert_pairs_close(s_roots(5), roots_chebys(5), 1e-12, "s_roots");
        assert_pairs_close(h_roots(5), roots_hermite(5), 1e-12, "h_roots");
        assert_pairs_close(he_roots(5), roots_hermitenorm(5), 1e-12, "he_roots");
        assert_pairs_close(l_roots(5), roots_laguerre(5), 1e-12, "l_roots");
        assert_pairs_close(
            la_roots(5, 0.5),
            roots_genlaguerre(5, 0.5),
            1e-12,
            "la_roots",
        );
        assert_pairs_close(
            cg_roots(5, 1.5),
            roots_gegenbauer(5, 1.5),
            1e-12,
            "cg_roots",
        );
        assert_pairs_close(
            j_roots(5, 0.5, 0.3),
            roots_jacobi(5, 0.5, 0.3),
            1e-12,
            "j_roots",
        );
        assert_pairs_close(
            js_roots(5, 1.5, 0.75),
            roots_sh_jacobi(5, 1.5, 0.75),
            1e-12,
            "js_roots",
        );
    }

    #[test]
    fn assoc_laguerre_routes_through_eval_genlaguerre() {
        // [frankenscipy-fdefl] /porting-to-rust: assoc_laguerre(x, n, k)
        // ≡ eval_genlaguerre(n, k, x) for any (x, n, k). Pin across
        // multiple cases including k=0 collapse to eval_laguerre.
        for &(x, n, k) in &[
            (0.0_f64, 0_u32, 0.0_f64),
            (0.5, 3, 0.0),
            (1.5, 4, 1.5),
            (2.7, 5, 0.25),
            (-0.3, 2, 3.0),
        ] {
            let direct = assoc_laguerre(x, n, k);
            let via = eval_genlaguerre(n, k, x);
            assert!(
                (direct - via).abs() < 1e-12 || (direct.is_nan() && via.is_nan()),
                "assoc_laguerre({x}, {n}, {k}) = {direct} via eval_genlaguerre = {via}"
            );
        }
        // k = 0 must collapse to standard Laguerre L_n.
        for n in 0..=5_u32 {
            let direct = assoc_laguerre(1.7, n, 0.0);
            let lag = eval_laguerre(n, 1.7);
            assert!(
                (direct - lag).abs() < 1e-12,
                "assoc_laguerre(1.7, {n}, 0) = {direct} but eval_laguerre = {lag}"
            );
        }
    }

    #[test]
    fn eval_chebyc_matches_closed_form_low_degree() {
        // [frankenscipy-1qax5] Chebyshev C polynomials on [−2, 2]:
        //   C_0(x) = 2
        //   C_1(x) = x
        //   C_2(x) = x² − 2
        //   C_3(x) = x³ − 3x
        //   C_4(x) = x⁴ − 4x² + 2
        for &x in &[-2.0_f64, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0] {
            assert!((eval_chebyc(0, x) - 2.0).abs() < 1e-12);
            assert!((eval_chebyc(1, x) - x).abs() < 1e-12);
            assert!((eval_chebyc(2, x) - (x * x - 2.0)).abs() < 1e-12);
            assert!((eval_chebyc(3, x) - (x.powi(3) - 3.0 * x)).abs() < 1e-12);
            let c4_expected = x.powi(4) - 4.0 * x * x + 2.0;
            assert!(
                (eval_chebyc(4, x) - c4_expected).abs() < 1e-12,
                "C_4({x}) = {} expected {c4_expected}",
                eval_chebyc(4, x)
            );
        }
        // Identity C_n(x) = 2·T_n(x/2). Pin across n=2..=5, x=1.7.
        for n in 2..=5_u32 {
            let from_t = 2.0 * eval_chebyt(n, 1.7 / 2.0);
            let direct = eval_chebyc(n, 1.7);
            assert!(
                (from_t - direct).abs() < 1e-12,
                "C_{n}(1.7): from T = {from_t}, direct = {direct}"
            );
        }
    }

    #[test]
    fn eval_chebys_matches_closed_form_low_degree() {
        // [frankenscipy-1qax5] Chebyshev S polynomials on [−2, 2]:
        //   S_0(x) = 1
        //   S_1(x) = x
        //   S_2(x) = x² − 1
        //   S_3(x) = x³ − 2x
        //   S_4(x) = x⁴ − 3x² + 1
        for &x in &[-2.0_f64, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0] {
            assert!((eval_chebys(0, x) - 1.0).abs() < 1e-12);
            assert!((eval_chebys(1, x) - x).abs() < 1e-12);
            assert!((eval_chebys(2, x) - (x * x - 1.0)).abs() < 1e-12);
            assert!((eval_chebys(3, x) - (x.powi(3) - 2.0 * x)).abs() < 1e-12);
            let s4_expected = x.powi(4) - 3.0 * x * x + 1.0;
            assert!(
                (eval_chebys(4, x) - s4_expected).abs() < 1e-12,
                "S_4({x}) = {} expected {s4_expected}",
                eval_chebys(4, x)
            );
        }
        // Identity S_n(x) = U_n(x/2). Pin across n=2..=5, x=0.7.
        for n in 2..=5_u32 {
            let from_u = eval_chebyu(n, 0.7 / 2.0);
            let direct = eval_chebys(n, 0.7);
            assert!(
                (from_u - direct).abs() < 1e-12,
                "S_{n}(0.7): from U = {from_u}, direct = {direct}"
            );
        }
    }

    #[test]
    fn eval_legendre_matches_scipy_reference_values() {
        // scipy.special.eval_legendre(n, x)
        // P_0(x) = 1, P_1(x) = x, P_2(x) = (3x^2 - 1)/2
        let x = 0.5;
        assert!((eval_legendre(0, x) - 1.0).abs() < 1e-10, "P_0(0.5)");
        assert!((eval_legendre(1, x) - 0.5).abs() < 1e-10, "P_1(0.5)");
        let p2_expected = (3.0 * x * x - 1.0) / 2.0; // -0.125
        assert!(
            (eval_legendre(2, x) - p2_expected).abs() < 1e-10,
            "P_2(0.5) got {}, expected {}",
            eval_legendre(2, x),
            p2_expected
        );
    }

    #[test]
    fn eval_chebyt_matches_scipy_reference_values() {
        // scipy.special.eval_chebyt(n, x)
        // T_0(x) = 1, T_1(x) = x, T_2(x) = 2x^2 - 1
        let x = 0.5;
        assert!((eval_chebyt(0, x) - 1.0).abs() < 1e-10, "T_0(0.5)");
        assert!((eval_chebyt(1, x) - 0.5).abs() < 1e-10, "T_1(0.5)");
        let t2_expected = 2.0 * x * x - 1.0; // -0.5
        assert!(
            (eval_chebyt(2, x) - t2_expected).abs() < 1e-10,
            "T_2(0.5) got {}, expected {}",
            eval_chebyt(2, x),
            t2_expected
        );
    }

    #[test]
    fn eval_hermite_matches_scipy_reference_values() {
        // scipy.special.eval_hermite(n, x) - physicist's Hermite
        // H_0(x) = 1, H_1(x) = 2x, H_2(x) = 4x^2 - 2
        let x = 1.0;
        assert!((eval_hermite(0, x) - 1.0).abs() < 1e-10, "H_0(1)");
        assert!((eval_hermite(1, x) - 2.0).abs() < 1e-10, "H_1(1)");
        let h2_expected = 4.0 * x * x - 2.0; // 2
        assert!(
            (eval_hermite(2, x) - h2_expected).abs() < 1e-10,
            "H_2(1) got {}, expected {}",
            eval_hermite(2, x),
            h2_expected
        );
    }

    #[test]
    fn roots_legendre_matches_scipy_reference_values() {
        // scipy.special.roots_legendre(3)
        // nodes ≈ [-0.7746, 0, 0.7746], weights ≈ [0.5556, 0.8889, 0.5556]
        let (nodes, weights) = roots_legendre(3);
        assert_eq!(nodes.len(), 3);
        assert_eq!(weights.len(), 3);
        // Check symmetry: nodes should be symmetric around 0
        assert!(
            (nodes[0] + nodes[2]).abs() < 1e-10,
            "nodes should be symmetric"
        );
        assert!((nodes[1]).abs() < 1e-10, "middle node should be ~0");
        // Weights should be positive and sum to 2 (integral of 1 over [-1,1])
        let weight_sum: f64 = weights.iter().sum();
        assert!(
            (weight_sum - 2.0).abs() < 1e-10,
            "weights should sum to 2, got {}",
            weight_sum
        );
    }

    #[test]
    fn eval_laguerre_matches_scipy_reference_values() {
        // scipy.special.eval_laguerre(n, x)
        // L_0(x) = 1, L_1(x) = 1-x, L_2(x) = (x^2 - 4x + 2)/2
        let x = 1.0;
        assert!((eval_laguerre(0, x) - 1.0).abs() < 1e-10, "L_0(1)");
        assert!(
            (eval_laguerre(1, x) - 0.0).abs() < 1e-10,
            "L_1(1) = 1-1 = 0"
        );
        let l2_expected = (x * x - 4.0 * x + 2.0) / 2.0; // -0.5
        assert!(
            (eval_laguerre(2, x) - l2_expected).abs() < 1e-10,
            "L_2(1) got {}, expected {}",
            eval_laguerre(2, x),
            l2_expected
        );
    }

    #[test]
    fn eval_jacobi_matches_scipy_reference_values() {
        // scipy.special.eval_jacobi(n, alpha, beta, x)
        // P_0^(a,b)(x) = 1
        // P_1^(a,b)(x) = (a+b+2)*x/2 + (a-b)/2
        assert!(
            (eval_jacobi(0, 0.5, 0.5, 0.5) - 1.0).abs() < 1e-10,
            "P_0^(0.5,0.5)(0.5) = 1"
        );
        // P_1^(1,1)(0) = (1-1)/2 = 0
        assert!(
            eval_jacobi(1, 1.0, 1.0, 0.0).abs() < 1e-10,
            "P_1^(1,1)(0) = 0"
        );
    }

    #[test]
    fn eval_gegenbauer_matches_scipy_reference_values() {
        // scipy.special.eval_gegenbauer(n, alpha, x)
        // C_0^alpha(x) = 1
        // C_1^alpha(x) = 2*alpha*x
        assert!(
            (eval_gegenbauer(0, 0.5, 0.7) - 1.0).abs() < 1e-10,
            "C_0^0.5(0.7) = 1"
        );
        let c1_expected = 2.0 * 0.5 * 0.7; // = 0.7
        assert!(
            (eval_gegenbauer(1, 0.5, 0.7) - c1_expected).abs() < 1e-10,
            "C_1^0.5(0.7) got {}, expected {}",
            eval_gegenbauer(1, 0.5, 0.7),
            c1_expected
        );
    }

    #[test]
    fn eval_chebyu_matches_scipy_reference_values() {
        // scipy.special.eval_chebyu(n, x) - Chebyshev of second kind
        // U_0(x) = 1, U_1(x) = 2x, U_2(x) = 4x^2 - 1
        let x = 0.5;
        assert!((eval_chebyu(0, x) - 1.0).abs() < 1e-10, "U_0(0.5)");
        assert!(
            (eval_chebyu(1, x) - 1.0).abs() < 1e-10,
            "U_1(0.5) = 2*0.5 = 1"
        );
        let u2_expected = 4.0 * x * x - 1.0; // = 0
        assert!(
            (eval_chebyu(2, x) - u2_expected).abs() < 1e-10,
            "U_2(0.5) got {}, expected {}",
            eval_chebyu(2, x),
            u2_expected
        );
    }
}
