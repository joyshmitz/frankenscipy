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

use fsci_linalg::{DecompOptions, eigh};

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
        return 1.0;
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
        return 1.0;
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

/// Compute Gauss-Hermite quadrature nodes and weights on `(-∞, ∞)`.
///
/// The weight function is `exp(-x^2)`.
#[must_use]
pub fn roots_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    golub_welsch(n, PI.sqrt(), |_k| 0.0, |k| ((k as f64) / 2.0).sqrt(), true)
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
    assert!(alpha > -1.0, "alpha must be greater than -1");

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

/// Compute Gauss-Jacobi quadrature nodes and weights on `[-1, 1]`.
///
/// The weight function is `(1 - x)^alpha (1 + x)^beta`.
#[must_use]
pub fn roots_jacobi(n: usize, alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(alpha > -1.0, "alpha must be greater than -1");
    assert!(beta > -1.0, "beta must be greater than -1");

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

    let mut jacobi = vec![vec![0.0; n]; n];
    for (i, row) in jacobi.iter_mut().enumerate() {
        row[i] = diag(i);
    }
    for k in 1..n {
        let beta = offdiag(k);
        jacobi[k - 1][k] = beta;
        jacobi[k][k - 1] = beta;
    }

    let eig = eigh(&jacobi, DecompOptions::default()).unwrap_or(fsci_linalg::EighResult {
        eigenvalues: vec![0.0; n],
        eigenvectors: vec![vec![0.0; n]; n],
    });
    let mut nodes = eig.eigenvalues;
    let mut weights: Vec<f64> = (0..n)
        .map(|col| mu0 * eig.eigenvectors[0][col] * eig.eigenvectors[0][col])
        .collect();

    if symmetrize {
        symmetrize_pairs(&mut nodes, &mut weights);
    }

    (nodes, weights)
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

    // Phase factor exp(i·m·θ)
    let angle = m as f64 * theta;

    Complex64 {
        re: norm * plm * angle.cos(),
        im: norm * plm * angle.sin(),
    }
}

/// Real spherical harmonic Y_l^m(θ, φ).
///
/// The real form is defined as:
///   m > 0: Y_l^m = √2 * (-1)^m * Re(Y_l^{|m|})
///   m = 0: Y_l^0 (already real)
///   m < 0: Y_l^m = √2 * (-1)^m * Im(Y_l^{|m|})
pub fn sph_harm_y(l: u32, m: i32, theta: f64, phi: f64) -> f64 {
    let am = m.unsigned_abs();
    if am > l {
        return 0.0;
    }

    let ylm = sph_harm(am as i32, l, theta, phi);

    if m > 0 {
        let sign = if am.is_multiple_of(2) { 1.0 } else { -1.0 };
        std::f64::consts::SQRT_2 * sign * ylm.re
    } else if m < 0 {
        let sign = if am.is_multiple_of(2) { 1.0 } else { -1.0 };
        std::f64::consts::SQRT_2 * sign * ylm.im
    } else {
        ylm.re
    }
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

    // ── Legendre ──────────────────────────────────────────────────

    #[test]
    fn legendre_p0_is_one() {
        assert_close(eval_legendre(0, 0.5), 1.0, 1e-15, "P_0");
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
    fn sph_harm_y_real_y00_constant() {
        let expected = 1.0 / (4.0 * PI).sqrt();
        assert_close(sph_harm_y(0, 0, 1.0, 0.5), expected, 1e-12, "real Y_0^0");
    }

    #[test]
    fn sph_harm_y_real_orthogonality() {
        // Real spherical harmonics are also orthonormal
        let n_theta = 80;
        let n_phi = 40;
        let d_theta = 2.0 * PI / n_theta as f64;
        let d_phi = PI / n_phi as f64;

        // Compute <Y_1^0 | Y_1^1> — should be ~0
        let mut cross = 0.0;
        for i in 0..n_theta {
            let theta = (i as f64 + 0.5) * d_theta;
            for j in 0..n_phi {
                let phi = (j as f64 + 0.5) * d_phi;
                let y10 = sph_harm_y(1, 0, theta, phi);
                let y11 = sph_harm_y(1, 1, theta, phi);
                cross += y10 * y11 * phi.sin() * d_theta * d_phi;
            }
        }
        assert!(cross.abs() < 0.02, "Y_1^0 · Y_1^1 orthogonality: {cross}");
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
}
