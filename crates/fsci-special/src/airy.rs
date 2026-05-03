#![forbid(unsafe_code)]

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, record_special_trace,
};

pub const AIRY_DISPATCH_PLAN: &[DispatchPlan] = &[DispatchPlan {
    function: "airy",
    steps: &[
        DispatchStep {
            regime: KernelRegime::Series,
            when: "|x| < 4: convergent Taylor series around x=0",
        },
        DispatchStep {
            regime: KernelRegime::Asymptotic,
            when: "|x| >= 4: asymptotic expansion",
        },
    ],
    notes: "Returns (Ai, Ai', Bi, Bi'). Matches scipy.special.airy(x).",
}];

/// Result of the Airy function evaluation: (Ai, Ai', Bi, Bi').
#[derive(Debug, Clone, PartialEq)]
pub struct AiryResult {
    pub ai: f64,
    pub aip: f64,
    pub bi: f64,
    pub bip: f64,
}

/// Result of the complex Airy function evaluation: (Ai, Ai', Bi, Bi').
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct ComplexAiryResult {
    ai: Complex64,
    aip: Complex64,
    bi: Complex64,
    bip: Complex64,
}

impl ComplexAiryResult {
    fn from_real(result: AiryResult) -> Self {
        Self {
            ai: Complex64::from_real(result.ai),
            aip: Complex64::from_real(result.aip),
            bi: Complex64::from_real(result.bi),
            bip: Complex64::from_real(result.bip),
        }
    }

    fn nan() -> Self {
        let nan = Complex64::new(f64::NAN, f64::NAN);
        Self {
            ai: nan,
            aip: nan,
            bi: nan,
            bip: nan,
        }
    }
}

/// Compute Airy functions Ai(x), Ai'(x), Bi(x), Bi'(x).
///
/// Matches `scipy.special.airy(x)` which returns `(Ai, Aip, Bi, Bip)`.
pub fn airy(x: &SpecialTensor, mode: RuntimeMode) -> Result<Vec<SpecialTensor>, SpecialError> {
    match x {
        SpecialTensor::RealScalar(val) => {
            let result = airy_scalar(*val, mode)?;
            Ok(vec![
                SpecialTensor::RealScalar(result.ai),
                SpecialTensor::RealScalar(result.aip),
                SpecialTensor::RealScalar(result.bi),
                SpecialTensor::RealScalar(result.bip),
            ])
        }
        SpecialTensor::RealVec(values) => {
            let mut ai_vec = Vec::with_capacity(values.len());
            let mut aip_vec = Vec::with_capacity(values.len());
            let mut bi_vec = Vec::with_capacity(values.len());
            let mut bip_vec = Vec::with_capacity(values.len());
            for &val in values {
                let result = airy_scalar(val, mode)?;
                ai_vec.push(result.ai);
                aip_vec.push(result.aip);
                bi_vec.push(result.bi);
                bip_vec.push(result.bip);
            }
            Ok(vec![
                SpecialTensor::RealVec(ai_vec),
                SpecialTensor::RealVec(aip_vec),
                SpecialTensor::RealVec(bi_vec),
                SpecialTensor::RealVec(bip_vec),
            ])
        }
        SpecialTensor::ComplexScalar(val) => {
            let result = airy_complex_scalar(*val, mode)?;
            Ok(vec![
                SpecialTensor::ComplexScalar(result.ai),
                SpecialTensor::ComplexScalar(result.aip),
                SpecialTensor::ComplexScalar(result.bi),
                SpecialTensor::ComplexScalar(result.bip),
            ])
        }
        SpecialTensor::ComplexVec(values) => {
            let mut ai_vec = Vec::with_capacity(values.len());
            let mut aip_vec = Vec::with_capacity(values.len());
            let mut bi_vec = Vec::with_capacity(values.len());
            let mut bip_vec = Vec::with_capacity(values.len());
            for &val in values {
                let result = airy_complex_scalar(val, mode)?;
                ai_vec.push(result.ai);
                aip_vec.push(result.aip);
                bi_vec.push(result.bi);
                bip_vec.push(result.bip);
            }
            Ok(vec![
                SpecialTensor::ComplexVec(ai_vec),
                SpecialTensor::ComplexVec(aip_vec),
                SpecialTensor::ComplexVec(bi_vec),
                SpecialTensor::ComplexVec(bip_vec),
            ])
        }
        SpecialTensor::Empty => Err(SpecialError {
            function: "airy",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid input",
        }),
    }
}

/// Scalar convenience: compute just Ai(x).
pub fn ai(x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_airy_component("ai", x, mode, |result| result.ai, |result| result.ai)
}

/// Scalar convenience: compute just Bi(x).
pub fn bi(x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_airy_component("bi", x, mode, |result| result.bi, |result| result.bi)
}

/// First `n` zeros of Ai(x), strictly negative and monotonically decreasing.
///
/// Matches `scipy.special.ai_zeros(nt)` (the zeros portion). Uses the
/// asymptotic series
///   t = (3π(4k − 1)/8)^(2/3)
///   a_k ≈ -t · (1 + 5/(48 t²) − 5/(36 t⁴) + ...)
/// as the initial guess for the k-th zero, then refines via Newton's
/// method using Ai and Ai'.
pub fn ai_zeros(n: usize) -> Vec<f64> {
    airy_zeros_inner(n, true)
}

/// First `n` zeros of Bi(x), strictly negative and monotonically decreasing.
///
/// Matches `scipy.special.bi_zeros(nt)`. Uses the asymptotic series
///   t = (3π(4k − 3)/8)^(2/3)
///   b_k ≈ -t · (1 + 5/(48 t²) − 5/(36 t⁴) + ...)
/// (same series shape as Ai zeros — only the (4k − 1) vs (4k − 3)
/// argument differs) as the initial guess, then refines via Newton's
/// method using Bi and Bi'.
pub fn bi_zeros(n: usize) -> Vec<f64> {
    airy_zeros_inner(n, false)
}

fn airy_zeros_inner(n: usize, ai_kind: bool) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for k in 1..=n {
        let kf = k as f64;
        let t_arg = if ai_kind {
            3.0 * std::f64::consts::PI * (4.0 * kf - 1.0) / 8.0
        } else {
            3.0 * std::f64::consts::PI * (4.0 * kf - 3.0) / 8.0
        };
        let t = t_arg.powf(2.0 / 3.0);
        let inv_t2 = 1.0 / (t * t);
        // Both Ai and Bi zeros use the same series shape; only the t_arg
        // argument distinguishes them: Ai uses (4k - 1), Bi uses (4k - 3).
        let initial = -t * (1.0 + 5.0 / 48.0 * inv_t2 - 5.0 / 36.0 * inv_t2 * inv_t2);
        // Refine via bisection inside a 0.5-radius bracket around the
        // asymptotic guess. The asymptotic series is accurate to <0.1 for
        // every zero, so the bracket reliably captures the true zero;
        // bisection is more robust than Newton near the slope reversals
        // inside Ai/Bi where Newton can flip basins.
        let radius = 0.5_f64;
        let mut lo = initial - radius;
        let mut hi = initial + radius;
        let f_at = |x: f64| -> f64 {
            match airy_scalar(x, RuntimeMode::Strict) {
                Ok(r) => {
                    if ai_kind {
                        r.ai
                    } else {
                        r.bi
                    }
                }
                Err(_) => f64::NAN,
            }
        };
        let mut f_lo = f_at(lo);
        let mut f_hi = f_at(hi);
        // If the bracket doesn't straddle zero, expand outward in steps of
        // 0.2 up to a few extra units.
        let mut tries = 0;
        while f_lo.is_finite()
            && f_hi.is_finite()
            && f_lo.signum() == f_hi.signum()
            && tries < 8
        {
            lo -= 0.2;
            hi += 0.2;
            f_lo = f_at(lo);
            f_hi = f_at(hi);
            tries += 1;
        }
        if !f_lo.is_finite() || !f_hi.is_finite() || f_lo.signum() == f_hi.signum() {
            // Fallback: keep the asymptotic guess if bracket couldn't be
            // formed (only realistically happens on numerical pathologies).
            out.push(initial);
            continue;
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
                f_hi = f_mid;
            }
            if (hi - lo) < 1.0e-12 {
                break;
            }
        }
        out.push(0.5 * (lo + hi));
    }
    out
}

fn airy_scalar(x: f64, mode: RuntimeMode) -> Result<AiryResult, SpecialError> {
    if x.is_nan() {
        return Ok(AiryResult {
            ai: f64::NAN,
            aip: f64::NAN,
            bi: f64::NAN,
            bip: f64::NAN,
        });
    }

    if x.is_infinite() {
        if x.is_sign_positive() {
            // x -> +inf: Ai -> 0, Bi -> inf
            return Ok(AiryResult {
                ai: 0.0,
                aip: 0.0,
                bi: f64::INFINITY,
                bip: f64::INFINITY,
            });
        }
        // x -> -inf: oscillatory
        return Ok(AiryResult {
            ai: 0.0,
            aip: 0.0,
            bi: 0.0,
            bip: 0.0,
        });
    }

    if x.abs() < 4.0 {
        airy_series(x, mode)
    } else {
        airy_asymptotic(x, mode)
    }
}

fn airy_complex_scalar(z: Complex64, mode: RuntimeMode) -> Result<ComplexAiryResult, SpecialError> {
    if z.im == 0.0 {
        return airy_scalar(z.re, mode).map(ComplexAiryResult::from_real);
    }

    if !z.is_finite() {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "airy",
                kind: SpecialErrorKind::NonFiniteInput,
                mode,
                detail: "complex-valued Airy requires finite real and imaginary parts",
            });
        }
        return Ok(ComplexAiryResult::nan());
    }

    airy_series_complex(z, mode)
}

/// Taylor series for Airy functions around x=0.
///
/// Ai(x) = c1 * f(x) - c2 * g(x)
/// Bi(x) = sqrt(3) * (c1 * f(x) + c2 * g(x))
///
/// where f and g are the two independent series solutions.
fn airy_series(x: f64, _mode: RuntimeMode) -> Result<AiryResult, SpecialError> {
    // Constants: Ai(0) = 1/(3^(2/3) * Gamma(2/3)), Ai'(0) = -1/(3^(1/3) * Gamma(1/3))
    let c1 = 0.355_028_053_887_817_2; // 1 / (3^(2/3) * Gamma(2/3))
    let c2 = 0.258_819_403_792_806_8; // 1 / (3^(1/3) * Gamma(1/3))

    // Series for f(x) = sum_{k=0}^inf x^(3k) * prod_{j=1}^k 1/((3j-2)(3j-1)) / (3k)!
    // Series for g(x) = sum_{k=0}^inf x^(3k+1) * ...
    // Computed via recurrence on terms.

    let mut f = 1.0_f64; // f(x) = 1 + ...
    let mut fp = 0.0_f64; // f'(x) = 0 + ...
    let mut g = x; // g(x) = x + ...
    let mut gp = 1.0_f64; // g'(x) = 1 + ...

    if x != 0.0 {
        let x3 = x * x * x;
        let mut t_f = 1.0_f64;
        let mut t_g = x;

        for k in 1..50 {
            let k3 = 3 * k;
            // f term: multiply by x^3 / ((3k-1)*(3k))
            t_f *= x3 / ((k3 as f64 - 1.0) * k3 as f64);
            f += t_f;
            fp += t_f * k3 as f64 / x; // derivative contributes k3 * x^(3k-1) coefficient

            // g term: multiply by x^3 / ((3k)*(3k+1))
            t_g *= x3 / (k3 as f64 * (k3 as f64 + 1.0));
            g += t_g;
            gp += t_g * (k3 as f64 + 1.0) / x;

            if t_f.abs() < 1e-16 * f.abs() && t_g.abs() < 1e-16 * g.abs() {
                break;
            }
        }
    }

    let sqrt3 = 3.0_f64.sqrt();

    let ai = c1 * f - c2 * g;
    let aip = c1 * fp - c2 * gp;
    let bi = sqrt3 * (c1 * f + c2 * g);
    let bip = sqrt3 * (c1 * fp + c2 * gp);

    Ok(AiryResult { ai, aip, bi, bip })
}

fn airy_series_complex(z: Complex64, mode: RuntimeMode) -> Result<ComplexAiryResult, SpecialError> {
    let c1 = 0.355_028_053_887_817_2; // 1 / (3^(2/3) * Gamma(2/3))
    let c2 = 0.258_819_403_792_806_8; // 1 / (3^(1/3) * Gamma(1/3))
    let sqrt3 = 3.0_f64.sqrt();
    let zero = Complex64::from_real(0.0);
    let one = Complex64::from_real(1.0);

    if z == zero {
        return Ok(ComplexAiryResult {
            ai: Complex64::from_real(c1),
            aip: Complex64::from_real(-c2),
            bi: Complex64::from_real(sqrt3 * c1),
            bip: Complex64::from_real(sqrt3 * c2),
        });
    }

    let z3 = z * z * z;
    let max_terms = 256;
    let tol = 1.0e-14;

    let mut f = one;
    let mut fp = zero;
    let mut g = z;
    let mut gp = one;
    let mut t_f = one;
    let mut t_g = z;

    for k in 1..=max_terms {
        let k3 = (3 * k) as f64;

        t_f = t_f * z3 / ((k3 - 1.0) * k3);
        let delta_fp = (t_f * k3) / z;
        f = f + t_f;
        fp = fp + delta_fp;

        t_g = t_g * z3 / (k3 * (k3 + 1.0));
        let delta_gp = (t_g * (k3 + 1.0)) / z;
        g = g + t_g;
        gp = gp + delta_gp;

        if !t_f.is_finite()
            || !t_g.is_finite()
            || !delta_fp.is_finite()
            || !delta_gp.is_finite()
            || !f.is_finite()
            || !g.is_finite()
            || !fp.is_finite()
            || !gp.is_finite()
        {
            if mode == RuntimeMode::Hardened {
                return Err(SpecialError {
                    function: "airy",
                    kind: SpecialErrorKind::OverflowRisk,
                    mode,
                    detail: "complex Airy series evaluation overflowed",
                });
            }
            return Ok(ComplexAiryResult::nan());
        }

        if t_f.abs() <= tol * f.abs().max(1.0)
            && t_g.abs() <= tol * g.abs().max(1.0)
            && delta_fp.abs() <= tol * fp.abs().max(1.0)
            && delta_gp.abs() <= tol * gp.abs().max(1.0)
        {
            let ai = f * c1 - g * c2;
            let aip = fp * c1 - gp * c2;
            let bi = (f * c1 + g * c2) * sqrt3;
            let bip = (fp * c1 + gp * c2) * sqrt3;
            return Ok(ComplexAiryResult { ai, aip, bi, bip });
        }
    }

    if mode == RuntimeMode::Hardened {
        return Err(SpecialError {
            function: "airy",
            kind: SpecialErrorKind::CancellationRisk,
            mode,
            detail: "complex Airy series did not converge within the term budget",
        });
    }

    let ai = f * c1 - g * c2;
    let aip = fp * c1 - gp * c2;
    let bi = (f * c1 + g * c2) * sqrt3;
    let bip = (fp * c1 + gp * c2) * sqrt3;
    Ok(ComplexAiryResult { ai, aip, bi, bip })
}

/// Asymptotic expansion for |x| >= 4.
fn airy_asymptotic(x: f64, _mode: RuntimeMode) -> Result<AiryResult, SpecialError> {
    let abs_x = x.abs();
    let zeta = (2.0 / 3.0) * abs_x * abs_x.sqrt(); // (2/3) * |x|^(3/2)

    if x > 0.0 {
        // Exponentially decaying/growing regime
        let prefactor = 1.0 / (2.0 * PI.sqrt() * abs_x.powf(0.25));
        let exp_neg = (-zeta).exp();
        let exp_pos = zeta.exp();

        // Leading asymptotic terms with corrections
        let (c_ai, c_aip, c_bi, c_bip) = asymptotic_coefficients(zeta);

        let ai = prefactor * exp_neg * c_ai;
        let aip = -prefactor * abs_x.sqrt() * exp_neg * c_aip;
        let bi = prefactor * exp_pos * c_bi;
        let bip = prefactor * abs_x.sqrt() * exp_pos * c_bip;

        Ok(AiryResult { ai, aip, bi, bip })
    } else {
        // Oscillatory regime for x < 0
        let prefactor = 1.0 / (PI.sqrt() * abs_x.powf(0.25));

        let (l, m, n, o) = oscillatory_coefficients(zeta);

        let phase = zeta + PI / 4.0;
        let cos_phase = phase.cos();
        let sin_phase = phase.sin();

        // Ai(-x) ~ pi^-1/2 x^-1/4 [ L sin(zeta+pi/4) - M cos(zeta+pi/4) ]
        let ai = prefactor * (l * sin_phase - m * cos_phase);
        // Ai'(-x) ~ -pi^-1/2 x^1/4 [ N cos(zeta+pi/4) + O sin(zeta+pi/4) ]
        let aip = -prefactor * abs_x.sqrt() * (n * cos_phase + o * sin_phase);
        // Bi(-x) ~ pi^-1/2 x^-1/4 [ L cos(zeta+pi/4) + M sin(zeta+pi/4) ]
        let bi = prefactor * (l * cos_phase + m * sin_phase);
        // Bi'(-x) ~ pi^-1/2 x^1/4 [ -N sin(zeta+pi/4) + O cos(zeta+pi/4) ]
        let bip = prefactor * abs_x.sqrt() * (-n * sin_phase + o * cos_phase);

        Ok(AiryResult { ai, aip, bi, bip })
    }
}

/// Asymptotic correction coefficients for x > 0.
fn asymptotic_coefficients(zeta: f64) -> (f64, f64, f64, f64) {
    // u_k coefficients for asymptotic expansion
    // u_0 = 1, u_1 = 5/72, u_2 = 385/10368, ...
    let iz = 1.0 / zeta;
    let iz2 = iz * iz;

    let u1 = 5.0 / 72.0;
    let u2 = 385.0 / 10368.0;
    let u3 = 85085.0 / 2239488.0;
    let u4 = 37_182_145.0 / 644_972_544.0;

    // Ai ~ (1 - u1/z + u2/z^2 - u3/z^3 + ...)
    let c_ai = 1.0 - u1 * iz + u2 * iz2 - u3 * iz * iz2 + u4 * iz2 * iz2;
    // Ai' ~ (1 + v1/z - v2/z^2 + ...)
    let v1 = 7.0 / 72.0;
    let v2 = 455.0 / 10368.0;
    let v3 = 95095.0 / 2239488.0;
    let v4 = 40_415_375.0 / 644_972_544.0;
    let c_aip = 1.0 + v1 * iz - v2 * iz2 + v3 * iz * iz2 - v4 * iz2 * iz2;
    let c_bi = c_ai; // same asymptotic correction for Bi
    let c_bip = c_aip;

    (c_ai, c_aip, c_bi, c_bip)
}

/// Oscillatory asymptotic coefficients for x < 0.
fn oscillatory_coefficients(zeta: f64) -> (f64, f64, f64, f64) {
    let iz = 1.0 / zeta;
    let iz2 = iz * iz;

    // u_k for functions
    let u1 = 5.0 / 72.0;
    let u2 = 385.0 / 10368.0;
    let u3 = 85085.0 / 2239488.0;
    let u4 = 37_182_145.0 / 644_972_544.0;

    let l = 1.0 - u2 * iz2 + u4 * iz2 * iz2;
    let m = u1 * iz - u3 * iz * iz2;

    // v_k for derivatives
    let v1 = 7.0 / 72.0;
    let v2 = 455.0 / 10368.0;
    let v3 = 95095.0 / 2239488.0;
    let v4 = 40_415_375.0 / 644_972_544.0;

    let n = 1.0 - v2 * iz2 + v4 * iz2 * iz2;
    let o = v1 * iz - v3 * iz * iz2;

    (l, m, n, o)
}

fn map_airy_component<F, G>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    real_kernel: F,
    complex_kernel: G,
) -> SpecialResult
where
    F: Fn(AiryResult) -> f64,
    G: Fn(ComplexAiryResult) -> Complex64,
{
    match input {
        SpecialTensor::RealScalar(x) => airy_scalar(*x, mode)
            .map(real_kernel)
            .map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .copied()
            .map(|x| airy_scalar(x, mode).map(&real_kernel))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(value) => airy_complex_scalar(*value, mode)
            .map(complex_kernel)
            .map(SpecialTensor::ComplexScalar),
        SpecialTensor::ComplexVec(values) => values
            .iter()
            .copied()
            .map(|x| airy_complex_scalar(x, mode).map(&complex_kernel))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: expected {expected}, got {actual} (diff={})",
            (actual - expected).abs()
        );
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tol: f64, msg: &str) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tol,
            "{msg}: expected {}+{}i, got {}+{}i (|delta|={delta})",
            expected.re,
            expected.im,
            actual.re,
            actual.im,
        );
    }

    fn complex_scalar(value: Complex64) -> SpecialTensor {
        SpecialTensor::ComplexScalar(value)
    }

    #[test]
    fn airy_at_zero() {
        // Ai(0) ≈ 0.35502805, Ai'(0) ≈ -0.25881940
        // Bi(0) ≈ 0.61492663, Bi'(0) ≈ 0.44828836
        let x = SpecialTensor::RealScalar(0.0);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(0)");
        assert_eq!(result.len(), 4);
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let aip = match &result[1] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bip = match &result[3] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };

        assert_close(ai, 0.355_028_053_887_817_2, 1e-10, "Ai(0)");
        assert_close(aip, -0.258_819_403_792_806_8, 1e-10, "Ai'(0)");
        assert_close(bi, 0.614_926_627_446_001, 1e-8, "Bi(0)");
        assert_close(bip, 0.448_288_357_353_826, 1e-8, "Bi'(0)");
    }

    #[test]
    fn airy_positive_small() {
        // Ai(1) ≈ 0.13529242, Bi(1) ≈ 1.20742359
        let x = SpecialTensor::RealScalar(1.0);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(1)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_close(ai, 0.135_292_416_312_881_4, 1e-6, "Ai(1)");
        assert_close(bi, 1.207_423_594_952_871, 1e-4, "Bi(1)");
    }

    #[test]
    fn airy_negative() {
        // Ai(-1) ≈ 0.53556088, Bi(-1) ≈ 0.10399739
        let x = SpecialTensor::RealScalar(-1.0);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(-1)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_close(ai, 0.535_560_883_292_352, 1e-6, "Ai(-1)");
        assert_close(bi, 0.103_997_389_496_945_9, 1e-4, "Bi(-1)");
    }

    #[test]
    fn airy_large_positive() {
        // Ai(10) ≈ 1.1047533e-10
        let x = SpecialTensor::RealScalar(10.0);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(10)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert!(ai > 0.0, "Ai(10) should be positive");
        assert!(ai < 1e-8, "Ai(10) should be very small, got {ai}");
    }

    #[test]
    fn airy_nan_passthrough() {
        let x = SpecialTensor::RealScalar(f64::NAN);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(NaN)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert!(ai.is_nan());
    }

    #[test]
    fn airy_positive_infinity() {
        let x = SpecialTensor::RealScalar(f64::INFINITY);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(inf)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_eq!(ai, 0.0, "Ai(+inf)=0");
        assert!(bi.is_infinite() && bi.is_sign_positive(), "Bi(+inf)=+inf");
    }

    #[test]
    fn ai_convenience_function() {
        let x = SpecialTensor::RealScalar(0.0);
        let result = ai(&x, RuntimeMode::Strict).expect("ai(0)");
        let val = match result {
            SpecialTensor::RealScalar(v) => v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_close(val, 0.355_028_053_887_817_2, 1e-10, "ai(0)");
    }

    #[test]
    fn bi_convenience_function() {
        let x = SpecialTensor::RealScalar(0.0);
        let result = bi(&x, RuntimeMode::Strict).expect("bi(0)");
        let val = match result {
            SpecialTensor::RealScalar(v) => v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_close(val, 0.614_926_627_446_001, 1e-8, "bi(0)");
    }

    #[test]
    fn airy_vector_input() {
        let x = SpecialTensor::RealVec(vec![0.0, 1.0, -1.0]);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(vec)");
        assert_eq!(result.len(), 4);
        match &result[0] {
            SpecialTensor::RealVec(v) => {
                assert_eq!(v.len(), 3);
                assert_close(v[0], 0.355_028_053_887_817_2, 1e-10, "Ai(0)");
            }
            _ => {
                panic!("expected vector");
            }
        }
    }

    #[test]
    fn airy_empty_input_error() {
        let x = SpecialTensor::Empty;
        let err = airy(&x, RuntimeMode::Strict).expect_err("empty input");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
    }

    #[test]
    fn airy_large_negative() {
        // Ai(-10) ≈ 0.0402412384827037, Bi(-10) ≈ -0.314680808611115
        let x = SpecialTensor::RealScalar(-10.0);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(-10)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_close(ai, 0.040_241_238_482_703_7, 1e-6, "Ai(-10)");
        assert_close(bi, -0.314_680_808_611_115, 1e-6, "Bi(-10)");
    }

    #[test]
    fn airy_complex_zero_matches_known_constants() {
        let z = complex_scalar(Complex64::new(0.0, 0.0));
        let result = airy(&z, RuntimeMode::Strict).expect("airy(0+0i)");
        match &result[0] {
            SpecialTensor::ComplexScalar(value) => {
                assert_complex_close(
                    *value,
                    Complex64::from_real(0.355_028_053_887_817_2),
                    1.0e-12,
                    "Ai(0+0i)",
                );
            }
            _ => panic!("expected complex scalar"),
        }
        match &result[2] {
            SpecialTensor::ComplexScalar(value) => {
                assert_complex_close(
                    *value,
                    Complex64::from_real(0.614_926_627_446_001),
                    1.0e-10,
                    "Bi(0+0i)",
                );
            }
            _ => panic!("expected complex scalar"),
        }
    }

    #[test]
    fn airy_complex_real_axis_matches_real_path() {
        let real = airy(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict).expect("airy(1)");
        let complex = airy(
            &complex_scalar(Complex64::new(1.0, 0.0)),
            RuntimeMode::Strict,
        )
        .expect("airy(1+0i)");

        for index in 0..4 {
            let expected = match &real[index] {
                SpecialTensor::RealScalar(value) => Complex64::from_real(*value),
                _ => panic!("expected real scalar"),
            };
            match &complex[index] {
                SpecialTensor::ComplexScalar(value) => {
                    assert_complex_close(*value, expected, 1.0e-12, "real-axis consistency");
                }
                _ => panic!("expected complex scalar"),
            }
        }
    }

    #[test]
    fn airy_complex_vector_preserves_conjugation_symmetry() {
        let z = Complex64::new(0.5, 0.75);
        let input = SpecialTensor::ComplexVec(vec![z, z.conj()]);
        let result = airy(&input, RuntimeMode::Strict).expect("airy complex vector");

        for output in result {
            match output {
                SpecialTensor::ComplexVec(values) => {
                    assert_eq!(values.len(), 2);
                    assert_complex_close(
                        values[1],
                        values[0].conj(),
                        1.0e-10,
                        "conjugation symmetry",
                    );
                }
                _ => panic!("expected complex vector"),
            }
        }
    }

    #[test]
    fn ai_and_bi_support_complex_inputs() {
        let z = complex_scalar(Complex64::new(0.25, -0.5));
        let ai_result = ai(&z, RuntimeMode::Strict).expect("ai complex");
        let bi_result = bi(&z, RuntimeMode::Strict).expect("bi complex");
        let airy_result = airy(&z, RuntimeMode::Strict).expect("airy complex");

        match (&ai_result, &airy_result[0]) {
            (SpecialTensor::ComplexScalar(ai_value), SpecialTensor::ComplexScalar(airy_ai)) => {
                assert_complex_close(*ai_value, *airy_ai, 1.0e-12, "ai convenience");
            }
            _ => panic!("expected complex scalar"),
        }

        match (&bi_result, &airy_result[2]) {
            (SpecialTensor::ComplexScalar(bi_value), SpecialTensor::ComplexScalar(airy_bi)) => {
                assert_complex_close(*bi_value, *airy_bi, 1.0e-12, "bi convenience");
            }
            _ => panic!("expected complex scalar"),
        }
    }

    #[test]
    fn ai_zeros_first_five_match_known_values() {
        // Tabulated zeros of Ai (Abramowitz & Stegun, Table 10.13):
        //  a_1 = -2.33810741045976...
        //  a_2 = -4.08794944413097...
        //  a_3 = -5.52055982809555...
        //  a_4 = -6.78670809007175...
        //  a_5 = -7.94413358712085...
        let zeros = ai_zeros(5);
        let expected = [
            -2.338_107_410_459_77_f64,
            -4.087_949_444_130_97,
            -5.520_559_828_095_55,
            -6.786_708_090_071_75,
            -7.944_133_587_120_85,
        ];
        assert_eq!(zeros.len(), 5);
        for (got, exp) in zeros.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-4,
                "ai_zero {got} vs expected {exp}"
            );
        }
    }

    #[test]
    fn ai_zeros_metamorphic_zero_value() {
        // At every returned zero, Ai(zero) must be small. Tolerance reflects
        // the underlying Ai accuracy at the negative zero crossings (where
        // the Airy series is oscillatory and harder to evaluate to full
        // f64 precision).
        let zeros = ai_zeros(8);
        for z in &zeros {
            let r = airy_scalar(*z, RuntimeMode::Strict).expect("airy_scalar");
            assert!(r.ai.abs() < 1e-6, "Ai({z}) = {} is not ~0", r.ai);
        }
    }

    #[test]
    fn ai_zeros_metamorphic_strictly_decreasing() {
        // Zeros are strictly negative and monotonically decreasing.
        let zeros = ai_zeros(10);
        for z in &zeros {
            assert!(*z < 0.0, "ai_zero {z} should be negative");
        }
        for w in zeros.windows(2) {
            assert!(w[0] > w[1], "zeros must be decreasing: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn bi_zeros_first_three_match_known_values() {
        // Tabulated zeros of Bi (Abramowitz & Stegun):
        //  b_1 = -1.17371322270913...
        //  b_2 = -3.27109330283635...
        //  b_3 = -4.83073784166202...
        let zeros = bi_zeros(3);
        let expected = [
            -1.173_713_222_709_13_f64,
            -3.271_093_302_836_35,
            -4.830_737_841_662_02,
        ];
        assert_eq!(zeros.len(), 3);
        for (got, exp) in zeros.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-4,
                "bi_zero {got} vs expected {exp}"
            );
        }
    }

    #[test]
    fn bi_zeros_metamorphic_zero_value() {
        let zeros = bi_zeros(6);
        for z in &zeros {
            let r = airy_scalar(*z, RuntimeMode::Strict).expect("airy_scalar");
            assert!(r.bi.abs() < 1e-6, "Bi({z}) = {} is not ~0", r.bi);
        }
    }
}
