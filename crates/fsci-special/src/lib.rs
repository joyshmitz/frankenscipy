#![forbid(unsafe_code)]

pub mod bessel;
pub mod beta;
pub mod error;
pub mod gamma;
pub mod hyper;
pub mod types;

pub use bessel::{BESSEL_DISPATCH_PLAN, hankel1, hankel2, iv, jv, kv, yv};
pub use beta::{BETA_DISPATCH_PLAN, beta, betainc, betaln};
pub use error::{ERROR_DISPATCH_PLAN, erf, erfc, erfcinv, erfinv};
pub use gamma::{GAMMA_DISPATCH_PLAN, digamma, gamma, gammaln, polygamma, rgamma};
pub use hyper::{HYPER_DISPATCH_PLAN, hyp1f1, hyp2f1};
pub use types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor,
};

#[cfg(test)]
mod tests {
    use fsci_runtime::RuntimeMode;

    use super::*;

    #[test]
    fn scalar_kernels_follow_primary_contract_points() {
        let one = SpecialTensor::RealScalar(1.0);
        let two = SpecialTensor::RealScalar(2.0);
        let zero = SpecialTensor::RealScalar(0.0);

        let gamma_one = gamma(&one, RuntimeMode::Strict).expect("gamma(1) should evaluate");
        assert_real_scalar_close(gamma_one, 1.0, 1e-12);

        let beta_half = beta(&one, &two, RuntimeMode::Strict).expect("beta(1,2) should evaluate");
        assert_real_scalar_close(beta_half, 0.5, 1e-12);

        let erf_zero = erf(&zero, RuntimeMode::Strict).expect("erf(0) should evaluate");
        assert_real_scalar_close(erf_zero, 0.0, 1e-12);

        let erfc_zero = erfc(&zero, RuntimeMode::Strict).expect("erfc(0) should evaluate");
        assert_real_scalar_close(erfc_zero, 1.0, 1e-12);

        let inv_zero = erfinv(&zero, RuntimeMode::Strict).expect("erfinv(0) should evaluate");
        assert_real_scalar_close(inv_zero, 0.0, 1e-12);
    }

    #[test]
    fn hardened_mode_rejects_domain_violations() {
        let out_of_domain = SpecialTensor::RealScalar(1.1);
        let pole = SpecialTensor::RealScalar(-2.0);

        let strict_inv = erfinv(&out_of_domain, RuntimeMode::Strict).expect("strict returns NaN");
        assert_real_scalar_nan(strict_inv);

        let hardened_inv =
            erfinv(&out_of_domain, RuntimeMode::Hardened).expect_err("hardened rejects domain");
        assert_eq!(hardened_inv.kind, SpecialErrorKind::DomainError);

        let strict_gamma = gamma(&pole, RuntimeMode::Strict).expect("strict returns NaN");
        assert_real_scalar_nan(strict_gamma);

        let hardened_gamma =
            gamma(&pole, RuntimeMode::Hardened).expect_err("hardened rejects pole");
        assert_eq!(hardened_gamma.kind, SpecialErrorKind::PoleInput);
    }

    #[test]
    fn pending_families_remain_explicitly_unimplemented() {
        let scalar = SpecialTensor::RealScalar(1.0);

        let bessel_err = jv(&scalar, &scalar, RuntimeMode::Strict).expect_err("placeholder");
        assert_eq!(bessel_err.kind, SpecialErrorKind::NotYetImplemented);

        let hyper_err = hyp2f1(&scalar, &scalar, &scalar, &scalar, RuntimeMode::Strict)
            .expect_err("placeholder");
        assert_eq!(hyper_err.kind, SpecialErrorKind::NotYetImplemented);
    }

    #[test]
    fn dispatch_plans_cover_packet_boundaries() {
        assert!(
            GAMMA_DISPATCH_PLAN
                .iter()
                .any(|entry| entry.function == "gamma")
        );
        assert!(
            BETA_DISPATCH_PLAN
                .iter()
                .any(|entry| entry.function == "beta")
        );
        assert!(
            BESSEL_DISPATCH_PLAN
                .iter()
                .any(|entry| entry.function == "jv")
        );
        assert!(
            ERROR_DISPATCH_PLAN
                .iter()
                .any(|entry| entry.function == "erf")
        );
        assert!(
            HYPER_DISPATCH_PLAN
                .iter()
                .any(|entry| entry.function == "hyp2f1")
        );
    }

    fn assert_real_scalar_close(actual: SpecialTensor, expected: f64, tol: f64) {
        match actual {
            SpecialTensor::RealScalar(value) => {
                assert!(
                    (value - expected).abs() <= tol,
                    "expected {expected}, got {value}"
                );
            }
            _ => panic!("expected real scalar output"),
        }
    }

    fn assert_real_scalar_nan(actual: SpecialTensor) {
        match actual {
            SpecialTensor::RealScalar(value) => assert!(value.is_nan(), "expected NaN"),
            _ => panic!("expected real scalar output"),
        }
    }
}
