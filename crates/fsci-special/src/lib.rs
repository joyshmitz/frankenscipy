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
    fn packet_function_boundaries_are_exposed() {
        let scalar = SpecialTensor::RealScalar(1.0);
        let expected = SpecialErrorKind::NotYetImplemented;

        let gamma_err = gamma(&scalar, RuntimeMode::Strict).expect_err("skeleton placeholder");
        assert_eq!(gamma_err.kind, expected);

        let beta_err = beta(&scalar, &scalar, RuntimeMode::Strict).expect_err("placeholder");
        assert_eq!(beta_err.kind, expected);

        let bessel_err = jv(&scalar, &scalar, RuntimeMode::Strict).expect_err("placeholder");
        assert_eq!(bessel_err.kind, expected);

        let erf_err = erf(&scalar, RuntimeMode::Strict).expect_err("placeholder");
        assert_eq!(erf_err.kind, expected);

        let hyper_err = hyp2f1(&scalar, &scalar, &scalar, &scalar, RuntimeMode::Strict)
            .expect_err("placeholder");
        assert_eq!(hyper_err.kind, expected);
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
}
