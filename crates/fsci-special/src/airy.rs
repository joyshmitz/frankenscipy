#![forbid(unsafe_code)]

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind, SpecialResult,
    SpecialTensor, not_yet_implemented, record_special_trace,
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
        SpecialTensor::ComplexScalar(_) | SpecialTensor::ComplexVec(_) => Err(SpecialError {
            function: "airy",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "complex-valued Airy not yet implemented",
        }),
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
    map_real_input("ai", x, mode, |val| {
        let result = airy_scalar(val, mode)?;
        Ok(result.ai)
    })
}

/// Scalar convenience: compute just Bi(x).
pub fn bi(x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("bi", x, mode, |val| {
        let result = airy_scalar(val, mode)?;
        Ok(result.bi)
    })
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

    #[test]
    fn airy_at_zero() {
        // Ai(0) ≈ 0.35502805, Ai'(0) ≈ -0.25881940
        // Bi(0) ≈ 0.61492663, Bi'(0) ≈ 0.44828836
        let x = SpecialTensor::RealScalar(0.0);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(0)");
        assert_eq!(result.len(), 4);
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => panic!("expected scalar"),
        };
        let aip = match &result[1] {
            SpecialTensor::RealScalar(v) => *v,
            _ => panic!("expected scalar"),
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => panic!("expected scalar"),
        };
        let bip = match &result[3] {
            SpecialTensor::RealScalar(v) => *v,
            _ => panic!("expected scalar"),
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
            _ => panic!("expected scalar"),
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => panic!("expected scalar"),
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
            _ => panic!("expected scalar"),
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => panic!("expected scalar"),
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
            _ => panic!("expected scalar"),
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
            _ => panic!("expected scalar"),
        };
        assert!(ai.is_nan());
    }

    #[test]
    fn airy_positive_infinity() {
        let x = SpecialTensor::RealScalar(f64::INFINITY);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(inf)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => panic!("expected scalar"),
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => panic!("expected scalar"),
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
            _ => panic!("expected scalar"),
        };
        assert_close(val, 0.355_028_053_887_817_2, 1e-10, "ai(0)");
    }

    #[test]
    fn bi_convenience_function() {
        let x = SpecialTensor::RealScalar(0.0);
        let result = bi(&x, RuntimeMode::Strict).expect("bi(0)");
        let val = match result {
            SpecialTensor::RealScalar(v) => v,
            _ => panic!("expected scalar"),
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
            _ => panic!("expected vector"),
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
            _ => panic!("expected scalar"),
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => panic!("expected scalar"),
        };
        assert_close(ai, 0.040_241_238_482_703_7, 1e-6, "Ai(-10)");
        assert_close(bi, -0.314_680_808_611_115, 1e-6, "Bi(-10)");
    }
}
