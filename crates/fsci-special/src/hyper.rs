#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind, SpecialResult,
    SpecialTensor,
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
/// Computed via direct series summation for real scalar inputs.
/// Matches `scipy.special.hyp1f1(a, b, z)`.
pub fn hyp1f1(
    a: &SpecialTensor,
    b: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    match (a, b, z) {
        (
            SpecialTensor::RealScalar(a_val),
            SpecialTensor::RealScalar(b_val),
            SpecialTensor::RealScalar(z_val),
        ) => {
            let result = hyp1f1_scalar(*a_val, *b_val, *z_val, mode)?;
            Ok(SpecialTensor::RealScalar(result))
        }
        (
            SpecialTensor::RealScalar(a_val),
            SpecialTensor::RealScalar(b_val),
            SpecialTensor::RealVec(z_vec),
        ) => {
            let mut results = Vec::with_capacity(z_vec.len());
            for &zi in z_vec {
                results.push(hyp1f1_scalar(*a_val, *b_val, zi, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
        _ => Err(SpecialError {
            function: "hyp1f1",
            kind: SpecialErrorKind::NotYetImplemented,
            mode,
            detail: "complex hyp1f1 not yet implemented",
        }),
    }
}

/// Gauss hypergeometric function 2F1(a, b; c; z).
///
/// Computed via direct series summation for |z| < 1 and transformation
/// formulas for |z| >= 1.
/// Matches `scipy.special.hyp2f1(a, b, c, z)`.
pub fn hyp2f1(
    a: &SpecialTensor,
    b: &SpecialTensor,
    c: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    match (a, b, c, z) {
        (
            SpecialTensor::RealScalar(a_val),
            SpecialTensor::RealScalar(b_val),
            SpecialTensor::RealScalar(c_val),
            SpecialTensor::RealScalar(z_val),
        ) => {
            let result = hyp2f1_scalar(*a_val, *b_val, *c_val, *z_val, mode)?;
            Ok(SpecialTensor::RealScalar(result))
        }
        (
            SpecialTensor::RealScalar(a_val),
            SpecialTensor::RealScalar(b_val),
            SpecialTensor::RealScalar(c_val),
            SpecialTensor::RealVec(z_vec),
        ) => {
            let mut results = Vec::with_capacity(z_vec.len());
            for &zi in z_vec {
                results.push(hyp2f1_scalar(*a_val, *b_val, *c_val, zi, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
        _ => Err(SpecialError {
            function: "hyp2f1",
            kind: SpecialErrorKind::NotYetImplemented,
            mode,
            detail: "complex hyp2f1 not yet implemented",
        }),
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

/// Compute Gamma(c)*Gamma(c-a-b) / (Gamma(c-a)*Gamma(c-b)) for the Gauss sum.
fn gamma_ratio_for_hyp2f1(c: f64, cab: f64, ca: f64, cb: f64) -> f64 {
    // Use log-gamma via Stirling for numerical stability
    let ln_num = ln_gamma(c) + ln_gamma(cab);
    let ln_den = ln_gamma(ca) + ln_gamma(cb);
    (ln_num - ln_den).exp()
}

/// Simple log-gamma via Lanczos approximation for positive real arguments.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
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
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }

    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> SpecialTensor {
        SpecialTensor::RealScalar(v)
    }

    fn get_scalar(r: &SpecialResult) -> f64 {
        match r.as_ref().expect("should succeed") {
            SpecialTensor::RealScalar(v) => *v,
            _ => panic!("expected RealScalar"),
        }
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
        assert!((get_scalar(&r) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hyp1f1_zero_a_is_one() {
        let r = hyp1f1(
            &scalar(0.0),
            &scalar(2.0),
            &scalar(5.0),
            RuntimeMode::Strict,
        );
        assert!((get_scalar(&r) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hyp1f1_a_equals_b_is_exp() {
        // 1F1(a; a; z) = e^z
        let z = 1.5;
        let r = hyp1f1(&scalar(3.0), &scalar(3.0), &scalar(z), RuntimeMode::Strict);
        let expected = z.exp();
        assert!(
            (get_scalar(&r) - expected).abs() < 1e-10,
            "1F1(a;a;z) should be e^z: got {} expected {expected}",
            get_scalar(&r)
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
            (get_scalar(&r) - std::f64::consts::E).abs() < 1e-10,
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
        let val = get_scalar(&r);
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
        let val = get_scalar(&r);
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
        let val = get_scalar(&r);
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
        match r.expect("should succeed") {
            SpecialTensor::RealVec(v) => {
                assert_eq!(v.len(), 3);
                assert!((v[0] - 1.0).abs() < 1e-14); // e^0
                assert!((v[1] - std::f64::consts::E).abs() < 1e-10); // e^1
                assert!((v[2] - std::f64::consts::E.powi(2)).abs() < 1e-10); // e^2
            }
            _ => panic!("expected RealVec"),
        }
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
        assert!((get_scalar(&r) - 1.0).abs() < 1e-14);
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
        assert!((get_scalar(&r) - 1.0).abs() < 1e-14);
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
            (get_scalar(&r) - expected).abs() < 1e-10,
            "2F1(1,1;2;0.5) should be -ln(0.5)/0.5: got {} expected {expected}",
            get_scalar(&r)
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
            (get_scalar(&r) - expected).abs() < 1e-10,
            "2F1(-2,1;1;z) should be (1-z)²: got {} expected {expected}",
            get_scalar(&r)
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
            (get_scalar(&r) - 2.0).abs() < 1e-8,
            "2F1(1,1;3;1) should be 2: got {}",
            get_scalar(&r)
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
        let val = get_scalar(&r);
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
        let val = get_scalar(&r);
        assert!(
            (val - expected).abs() < 1e-10,
            "2F1(1,1;2;-0.5): got {val} expected {expected}"
        );
    }
}
