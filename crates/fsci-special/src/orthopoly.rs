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
        let c_next = (2.0 * (kf + alpha) * x * c_curr - (kf + 2.0 * alpha - 1.0) * c_prev)
            / (kf + 1.0);
        c_prev = c_curr;
        c_curr = c_next;
    }
    c_curr
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
            assert_close(
                eval_chebyt(n, x),
                expected,
                1e-10,
                &format!("T_{n}(cos θ)"),
            );
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
        assert_close(
            eval_hermite(3, x),
            8.0 * x * x * x - 12.0 * x,
            1e-12,
            "H_3",
        );
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
}
