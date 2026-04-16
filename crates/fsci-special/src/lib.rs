#![forbid(unsafe_code)]

pub mod airy;
pub mod bessel;
pub mod beta;
pub mod convenience;
pub mod elliptic;
pub mod error;
pub mod gamma;
pub mod hyper;
pub mod orthopoly;
pub mod types;

pub use airy::{AIRY_DISPATCH_PLAN, AiryResult, ai, airy, bi};
pub use bessel::{
    BESSEL_DISPATCH_PLAN, hankel1, hankel2, iv, j0, j1, jn, jv, kv, spherical_in, spherical_jn,
    spherical_kn, spherical_yn, y0, y1, yn, yv,
};
pub use beta::{
    BETA_DISPATCH_PLAN, bdtr, bdtrc, bdtri, beta, betainc, betainc_scalar, betaln, betaln_scalar,
    btdtr, btdtri, fdtr, fdtrc, fdtri, nbdtr, nbdtrc, nbdtri, stdtr, stdtri,
};
pub use convenience::{
    CONVENIENCE_DISPATCH_PLAN,
    // Numerical differentiation and AGM
    agm,
    arccosh,
    arcsinh,
    arctanh,
    bei,
    ber,
    bernoulli,
    // Convenience wrappers
    betaincinv,
    boxcox_transform,
    boxcox1p,
    central_diff,
    central_diff2,
    clausen,
    dawsn,
    dawsn_scalar,
    debye,
    digamma_scalar,
    entr,
    erfc_conv,
    erfcinv_conv,
    erfcx,
    erfi,
    euler,
    expi_scalar,
    expit,
    expn,
    exprel,
    fresnel,
    gamma_mod_squared,
    gammainc_conv,
    gammaincc_conv,
    gammaincinv,
    gradient_approx,
    hessian_approx,
    hurwitz_zeta,
    inv_boxcox,
    inv_boxcox1p,
    jacobian_approx,
    kei,
    ker,
    kl_div,
    kolmogi,
    kolmogorov,
    lambertw_scalar,
    log_comb,
    log_ndtr,
    log_softmax,
    logit,
    logsumexp,
    modstruve,
    modstruve_scalar,
    ndtr,
    ndtri,
    owens_t,
    poch,
    rel_entr,
    shichi,
    sici,
    sinc,
    sinc_squared,
    smirnov,
    smirnovi,
    softmax,
    spence,
    struve,
    struve_scalar,
    tetragamma,
    trigamma,
    wrightomega,
    xlog1py,
    xlogy,
    zeta_scalar,
};
pub use elliptic::{
    ELLIPTIC_DISPATCH_PLAN, ellipe, ellipeinc, ellipj, ellipk, ellipkinc, ellipkm1, exp1, expi,
    expn_scalar, lambertw,
};
pub use error::{
    ERROR_DISPATCH_PLAN, erf, erf_scalar, erfc, erfc_scalar, erfcinv, erfinv, erfinv_scalar,
};
pub use gamma::{
    GAMMA_DISPATCH_PLAN, chdtr, chdtrc, chdtri, comb, digamma, factorial, factorial2, gamma,
    gammainc, gammainc_scalar, gammaincc, gammaincc_scalar, gammaln, gammaln_scalar, gdtr, gdtrc,
    gdtri, pdtr, pdtrc, pdtri, perm, polygamma, rgamma, zeta, zetac,
};
pub use hyper::{HYPER_DISPATCH_PLAN, hyp0f1, hyp0f1_scalar, hyp1f1, hyp2f1};
pub use orthopoly::{
    eval_chebyt, eval_chebyu, eval_gegenbauer, eval_genlaguerre, eval_hermite, eval_hermitenorm,
    eval_jacobi, eval_laguerre, eval_legendre, eval_sh_chebyt, eval_sh_chebyu, eval_sh_legendre,
    lpmv, roots_chebyt, roots_chebyu, roots_gegenbauer, roots_genlaguerre, roots_hermite,
    roots_hermitenorm, roots_jacobi, roots_laguerre, roots_legendre, sph_harm, sph_harm_y,
};
pub use types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, SpecialTraceEntry, take_special_traces,
};

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

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

        let gammainc_one =
            gammainc(&one, &one, RuntimeMode::Strict).expect("gammainc(1,1) should evaluate");
        assert_real_scalar_close(gammainc_one, 1.0 - (-1.0_f64).exp(), 1e-10);

        let gammaincc_one =
            gammaincc(&one, &one, RuntimeMode::Strict).expect("gammaincc(1,1) should evaluate");
        assert_real_scalar_close(gammaincc_one, (-1.0_f64).exp(), 1e-10);

        let j0_zero = j0(&zero, RuntimeMode::Strict).expect("j0(0) should evaluate");
        assert_real_scalar_close(j0_zero, 1.0, 1e-8);

        let j1_zero = j1(&zero, RuntimeMode::Strict).expect("j1(0) should evaluate");
        assert_real_scalar_close(j1_zero, 0.0, 1e-8);
    }

    #[test]
    fn erf_infinity_regression() {
        use fsci_runtime::RuntimeMode;
        let inf = SpecialTensor::RealScalar(f64::INFINITY);
        let neg_inf = SpecialTensor::RealScalar(f64::NEG_INFINITY);

        let res_inf = erf(&inf, RuntimeMode::Strict).unwrap();
        let res_neg_inf = erf(&neg_inf, RuntimeMode::Strict).unwrap();

        match res_inf {
            SpecialTensor::RealScalar(v) => assert_eq!(v, 1.0, "erf(inf) should be 1.0"),
            _ => {
                panic!("expected scalar");
            }
        }
        match res_neg_inf {
            SpecialTensor::RealScalar(v) => assert_eq!(v, -1.0, "erf(-inf) should be -1.0"),
            _ => {
                panic!("expected scalar");
            }
        }
    }

    #[test]
    fn iv_large_argument_regression() {
        use fsci_runtime::RuntimeMode;
        // I_0.5(500) = sqrt(2/(pi*500)) * sinh(500) ≈ 1.13e215
        let v = SpecialTensor::RealScalar(0.5);
        let z = SpecialTensor::RealScalar(500.0);
        let result = iv(&v, &z, RuntimeMode::Strict).expect("iv(0.5, 500)");
        match result {
            SpecialTensor::RealScalar(val) => {
                assert!(
                    val > 1e214 && val < 1e216,
                    "iv(0.5, 500) should be ~1e215, got {val}"
                );
            }
            _ => {
                panic!("expected scalar");
            }
        }
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
    fn trace_log_captures_domain_policy_events_as_json_lines() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();

        let out_of_domain = SpecialTensor::RealScalar(1.1);
        let _ = erfinv(&out_of_domain, RuntimeMode::Strict).expect("strict returns NaN");
        let _ = erfinv(&out_of_domain, RuntimeMode::Hardened).expect_err("hardened rejects");

        let traces = take_special_traces();
        assert!(!traces.is_empty(), "expected at least one trace entry");
        assert!(traces.iter().any(|entry| {
            entry.function == "erfinv"
                && entry.category == "domain_error"
                && entry.action_taken == "returned_nan"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "erfinv"
                && entry.category == "domain_error"
                && entry.action_taken == "fail_closed"
        }));

        let first = traces.first().expect("trace entries exist");
        let json = first.to_json_line();
        assert!(json.starts_with('{'));
        assert!(json.contains("\"function\""));
        assert!(json.contains("\"action_taken\""));
        assert!(json.contains("\"category\""));
    }

    #[test]
    fn beta_overflow_path_diverges_by_runtime_mode() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();

        let tiny = SpecialTensor::RealScalar(1.0e-308);
        let strict = beta(&tiny, &tiny, RuntimeMode::Strict).expect("strict must evaluate");
        match strict {
            SpecialTensor::RealScalar(value) => assert!(
                value.is_infinite() || value > 1.0e308,
                "expected very large or infinite strict beta, got {value}"
            ),
            _ => {
                panic!("expected real scalar output");
            }
        }

        let hardened = beta(&tiny, &tiny, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened.kind, SpecialErrorKind::OverflowRisk);

        let traces = take_special_traces();
        assert!(traces.iter().any(|entry| {
            entry.function == "beta"
                && entry.category == "overflow_risk"
                && entry.action_taken == "returned_inf"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "beta"
                && entry.category == "overflow_risk"
                && entry.action_taken == "fail_closed"
        }));
    }

    #[test]
    fn erfcinv_domain_policy_diverges_by_runtime_mode() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();
        let out_of_domain = SpecialTensor::RealScalar(3.0);

        let strict = erfcinv(&out_of_domain, RuntimeMode::Strict).expect("strict returns NaN");
        assert_real_scalar_nan(strict);

        let hardened =
            erfcinv(&out_of_domain, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened.kind, SpecialErrorKind::DomainError);

        let traces = take_special_traces();
        assert!(traces.iter().any(|entry| {
            entry.function == "erfcinv"
                && entry.category == "domain_error"
                && entry.action_taken == "returned_nan"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "erfcinv"
                && entry.category == "domain_error"
                && entry.action_taken == "fail_closed"
        }));
    }

    #[test]
    fn betaln_and_betainc_domain_policy_diverge_by_runtime_mode() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();
        let neg = SpecialTensor::RealScalar(-1.0);
        let pos = SpecialTensor::RealScalar(1.0);
        let x_bad = SpecialTensor::RealScalar(2.0);

        let strict_betaln = betaln(&neg, &pos, RuntimeMode::Strict).expect("strict returns NaN");
        assert_real_scalar_nan(strict_betaln);
        let hardened_betaln =
            betaln(&neg, &pos, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened_betaln.kind, SpecialErrorKind::DomainError);

        let strict_betainc =
            betainc(&pos, &pos, &x_bad, RuntimeMode::Strict).expect("strict returns NaN");
        assert_real_scalar_nan(strict_betainc);
        let hardened_betainc =
            betainc(&pos, &pos, &x_bad, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened_betainc.kind, SpecialErrorKind::DomainError);

        let traces = take_special_traces();
        assert!(traces.iter().any(|entry| {
            entry.function == "betaln"
                && entry.category == "domain_error"
                && entry.action_taken == "returned_nan"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "betaln"
                && entry.category == "domain_error"
                && entry.action_taken == "fail_closed"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "betainc"
                && entry.category == "domain_error"
                && entry.action_taken == "returned_nan"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "betainc"
                && entry.category == "domain_error"
                && entry.action_taken == "fail_closed"
        }));
    }

    #[test]
    fn gammainc_domain_policy_diverges_by_runtime_mode() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();
        let invalid_a = SpecialTensor::RealScalar(0.0);
        let one = SpecialTensor::RealScalar(1.0);

        let strict = gammainc(&invalid_a, &one, RuntimeMode::Strict).expect("strict returns NaN");
        assert_real_scalar_nan(strict);

        let hardened =
            gammainc(&invalid_a, &one, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened.kind, SpecialErrorKind::DomainError);

        let traces = take_special_traces();
        assert!(traces.iter().any(|entry| {
            entry.function == "gammainc"
                && entry.category == "domain_error"
                && entry.action_taken == "returned_nan"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "gammainc"
                && entry.category == "domain_error"
                && entry.action_taken == "fail_closed"
        }));
    }

    #[test]
    fn bessel_subset_wrappers_execute_required_surface() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();
        let zero = SpecialTensor::RealScalar(0.0);
        let one = SpecialTensor::RealScalar(1.0);
        let three = SpecialTensor::RealScalar(3.0);
        let neg_one = SpecialTensor::RealScalar(-1.0);

        let y0_zero = y0(&zero, RuntimeMode::Strict).expect("strict y0(0) should diverge");
        match y0_zero {
            SpecialTensor::RealScalar(value) => {
                assert!(value.is_infinite() && value.is_sign_negative())
            }
            _ => {
                panic!("expected real scalar");
            }
        }

        let y1_zero = y1(&zero, RuntimeMode::Strict).expect("strict y1(0) should diverge");
        match y1_zero {
            SpecialTensor::RealScalar(value) => {
                assert!(value.is_infinite() && value.is_sign_negative())
            }
            _ => {
                panic!("expected real scalar");
            }
        }

        let jn_zero = jn(&zero, &one, RuntimeMode::Strict).expect("jn(0,x)=j0(x)");
        let j0_one = j0(&one, RuntimeMode::Strict).expect("j0(1)");
        assert_real_scalars_close(jn_zero, j0_one, 1e-12);

        let yn_one = yn(&one, &three, RuntimeMode::Strict).expect("yn(1,x)=y1(x)");
        let y1_three = y1(&three, RuntimeMode::Strict).expect("y1(3)");
        assert_real_scalars_close(yn_one, y1_three, 1e-12);

        let strict_neg =
            y0(&neg_one, RuntimeMode::Strict).expect("strict returns NaN for negative domain");
        assert_real_scalar_nan(strict_neg);
        let hardened_neg = y0(&neg_one, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened_neg.kind, SpecialErrorKind::DomainError);
    }

    #[test]
    fn error_function_unit_matrix_and_structured_logs() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_005_u64;
        let mode = RuntimeMode::Strict;

        let zero = SpecialTensor::RealScalar(0.0);
        let inf = SpecialTensor::RealScalar(f64::INFINITY);
        let neg_inf = SpecialTensor::RealScalar(f64::NEG_INFINITY);

        let erf_zero = erf(&zero, mode).expect("erf(0)");
        assert_real_scalar_close(erf_zero.clone(), 0.0, 1e-12);
        push_test_log(test_log_json(
            "error-erf-zero",
            "erf",
            "x=0",
            "0",
            scalar_to_string(&erf_zero),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let erf_inf = erf(&inf, mode).expect("erf(inf)");
        assert_real_scalar_close(erf_inf.clone(), 1.0, 1e-12);
        push_test_log(test_log_json(
            "error-erf-inf",
            "erf",
            "x=inf",
            "1",
            scalar_to_string(&erf_inf),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let erf_neg_inf = erf(&neg_inf, mode).expect("erf(-inf)");
        assert_real_scalar_close(erf_neg_inf.clone(), -1.0, 1e-12);
        push_test_log(test_log_json(
            "error-erf-neg-inf",
            "erf",
            "x=-inf",
            "-1",
            scalar_to_string(&erf_neg_inf),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let erfc_zero = erfc(&zero, mode).expect("erfc(0)");
        assert_real_scalar_close(erfc_zero.clone(), 1.0, 1e-12);
        push_test_log(test_log_json(
            "error-erfc-zero",
            "erfc",
            "x=0",
            "1",
            scalar_to_string(&erfc_zero),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let erfc_inf = erfc(&inf, mode).expect("erfc(inf)");
        assert_real_scalar_close(erfc_inf.clone(), 0.0, 1e-12);
        push_test_log(test_log_json(
            "error-erfc-inf",
            "erfc",
            "x=inf",
            "0",
            scalar_to_string(&erfc_inf),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let erfc_neg_inf = erfc(&neg_inf, mode).expect("erfc(-inf)");
        assert_real_scalar_close(erfc_neg_inf.clone(), 2.0, 1e-12);
        push_test_log(test_log_json(
            "error-erfc-neg-inf",
            "erfc",
            "x=-inf",
            "2",
            scalar_to_string(&erfc_neg_inf),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        for x in [-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0] {
            let tensor = SpecialTensor::RealScalar(x);
            let erf_v = erf(&tensor, mode).expect("erf finite");
            let erfc_v = erfc(&tensor, mode).expect("erfc finite");
            let lhs = scalar_value(&erf_v) + scalar_value(&erfc_v);
            assert!(
                (lhs - 1.0).abs() <= 2.0e-7,
                "erf+erfc identity failed for x={x}: {lhs}"
            );
            push_test_log(test_log_json(
                "error-erf-plus-erfc",
                "erf_erfc_identity",
                format!("x={x}"),
                "1",
                format!("{lhs:.16e}"),
                2.0e-7,
                mode,
                seed,
                "pass",
            ));
        }

        let nan = SpecialTensor::RealScalar(f64::NAN);
        assert_real_scalar_nan(erf(&nan, mode).expect("erf NaN propagation"));
        assert_real_scalar_nan(erfc(&nan, mode).expect("erfc NaN propagation"));
        assert_real_scalar_nan(erfinv(&nan, mode).expect("erfinv NaN propagation"));
        assert_real_scalar_nan(erfcinv(&nan, mode).expect("erfcinv NaN propagation"));

        let logs = take_test_logs();
        assert!(!logs.is_empty(), "expected structured test logs");
        assert_logs_follow_schema(&logs);
    }

    #[test]
    fn gamma_beta_unit_matrix_and_structured_logs() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_006_u64;
        let mode = RuntimeMode::Strict;

        for (n, expected) in [(1.0, 1.0), (2.0, 1.0), (3.0, 2.0), (4.0, 6.0), (5.0, 24.0)] {
            let n_tensor = SpecialTensor::RealScalar(n);
            let actual = gamma(&n_tensor, mode).expect("gamma integer point");
            assert_real_scalar_close(actual.clone(), expected, 1e-9);
            push_test_log(test_log_json(
                "gamma-integer-factorial",
                "gamma",
                format!("x={n}"),
                format!("{expected}"),
                scalar_to_string(&actual),
                1e-9,
                mode,
                seed,
                "pass",
            ));
        }

        let gammaln_one = gammaln(&SpecialTensor::RealScalar(1.0), mode).expect("gammaln(1)");
        assert_real_scalar_close(gammaln_one.clone(), 0.0, 1e-12);
        push_test_log(test_log_json(
            "gamma-gammaln-one",
            "gammaln",
            "x=1",
            "0",
            scalar_to_string(&gammaln_one),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let gammaln_five = gammaln(&SpecialTensor::RealScalar(5.0), mode).expect("gammaln(5)");
        assert_real_scalar_close(gammaln_five.clone(), 24.0_f64.ln(), 1e-9);
        push_test_log(test_log_json(
            "gamma-gammaln-five",
            "gammaln",
            "x=5",
            "ln(24)",
            scalar_to_string(&gammaln_five),
            1e-9,
            mode,
            seed,
            "pass",
        ));

        let a = SpecialTensor::RealScalar(2.0);
        let x0 = SpecialTensor::RealScalar(0.0);
        let x1 = SpecialTensor::RealScalar(1.0);
        let xinf = SpecialTensor::RealScalar(f64::INFINITY);

        assert_real_scalar_close(gammainc(&a, &x0, mode).expect("gammainc(a,0)"), 0.0, 1e-12);
        assert_real_scalar_close(
            gammaincc(&a, &x0, mode).expect("gammaincc(a,0)"),
            1.0,
            1e-12,
        );
        assert_real_scalar_close(
            gammainc(&a, &xinf, mode).expect("gammainc(a,inf)"),
            1.0,
            1e-12,
        );
        assert_real_scalar_close(
            gammaincc(&a, &xinf, mode).expect("gammaincc(a,inf)"),
            0.0,
            1e-12,
        );

        let p = gammainc(&a, &x1, mode).expect("gammainc");
        let q = gammaincc(&a, &x1, mode).expect("gammaincc");
        let complement = scalar_value(&p) + scalar_value(&q);
        assert!((complement - 1.0).abs() <= 2.0e-12);
        push_test_log(test_log_json(
            "gamma-complement",
            "gammainc+gammaincc",
            "a=2,x=1",
            "1",
            format!("{complement:.16e}"),
            2.0e-12,
            mode,
            seed,
            "pass",
        ));

        let beta_left = beta(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(2.0),
            mode,
        )
        .expect("beta left");
        let beta_right = beta(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(0.5),
            mode,
        )
        .expect("beta right");
        assert_real_scalars_close(beta_left.clone(), beta_right.clone(), 1e-10);
        push_test_log(test_log_json(
            "beta-symmetry",
            "beta",
            "a=0.5,b=2 vs a=2,b=0.5",
            "equal",
            format!(
                "lhs={},rhs={}",
                scalar_to_string(&beta_left),
                scalar_to_string(&beta_right)
            ),
            1e-10,
            mode,
            seed,
            "pass",
        ));

        assert_real_scalar_close(
            betainc(
                &SpecialTensor::RealScalar(2.5),
                &SpecialTensor::RealScalar(3.0),
                &SpecialTensor::RealScalar(0.0),
                mode,
            )
            .expect("betainc x=0"),
            0.0,
            1e-12,
        );
        assert_real_scalar_close(
            betainc(
                &SpecialTensor::RealScalar(2.5),
                &SpecialTensor::RealScalar(3.0),
                &SpecialTensor::RealScalar(1.0),
                mode,
            )
            .expect("betainc x=1"),
            1.0,
            1e-12,
        );

        let logs = take_test_logs();
        assert!(!logs.is_empty(), "expected structured test logs");
        assert_logs_follow_schema(&logs);
    }

    #[test]
    fn bessel_reference_points_and_structured_logs() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_007_u64;
        let mode = RuntimeMode::Strict;

        let y0_one = y0(&SpecialTensor::RealScalar(1.0), mode).expect("y0(1)");
        assert_real_scalar_close(y0_one.clone(), 0.088_256_964_2, 7.5e-3);
        push_test_log(test_log_json(
            "bessel-y0-one",
            "y0",
            "x=1",
            "0.0882569642",
            scalar_to_string(&y0_one),
            7.5e-3,
            mode,
            seed,
            "pass",
        ));

        let y1_one = y1(&SpecialTensor::RealScalar(1.0), mode).expect("y1(1)");
        assert_real_scalar_close(y1_one.clone(), -0.781_212_821_3, 7.5e-3);
        push_test_log(test_log_json(
            "bessel-y1-one",
            "y1",
            "x=1",
            "-0.7812128213",
            scalar_to_string(&y1_one),
            7.5e-3,
            mode,
            seed,
            "pass",
        ));

        let j0_zero_cross = j0(&SpecialTensor::RealScalar(2.404_825_557_7), mode).expect("j0 zero");
        assert!(
            scalar_value(&j0_zero_cross).abs() <= 7.5e-4,
            "j0 first root mismatch: {}",
            scalar_value(&j0_zero_cross)
        );
        push_test_log(test_log_json(
            "bessel-j0-first-root",
            "j0",
            "x=2.4048255577",
            "0",
            scalar_to_string(&j0_zero_cross),
            7.5e-4,
            mode,
            seed,
            "pass",
        ));

        for x in [20.0, 30.0, 40.0, 50.0] {
            let j0_v = j0(&SpecialTensor::RealScalar(x), mode).expect("j0 large x");
            let y0_v = y0(&SpecialTensor::RealScalar(x), mode).expect("y0 large x");
            let lhs = scalar_value(&j0_v).powi(2) + scalar_value(&y0_v).powi(2);
            let rhs = 2.0 / (std::f64::consts::PI * x);
            assert!(
                (lhs - rhs).abs() <= 3.5e-3,
                "Bessel asymptotic envelope mismatch at x={x}: lhs={lhs} rhs={rhs}"
            );
            push_test_log(test_log_json(
                "bessel-asymptotic-envelope",
                "j0,y0",
                format!("x={x}"),
                format!("{rhs:.16e}"),
                format!("{lhs:.16e}"),
                3.5e-3,
                mode,
                seed,
                "pass",
            ));
        }

        let logs = take_test_logs();
        assert!(!logs.is_empty(), "expected structured test logs");
        assert_logs_follow_schema(&logs);
    }

    #[test]
    fn property_erf_plus_erfc_identity_over_grid() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_101_u64;
        let mode = RuntimeMode::Strict;

        for i in -48..=48 {
            let x = i as f64 / 8.0;
            let tensor = SpecialTensor::RealScalar(x);
            let lhs = scalar_value(&erf(&tensor, mode).expect("erf"))
                + scalar_value(&erfc(&tensor, mode).expect("erfc"));
            assert!((lhs - 1.0).abs() <= 2.0e-7, "identity mismatch for x={x}");
            push_test_log(test_log_json(
                "prop-erf-plus-erfc",
                "erf+erfc",
                format!("x={x}"),
                "1",
                format!("{lhs:.16e}"),
                2.0e-7,
                mode,
                seed,
                "pass",
            ));
        }
        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn property_gamma_recurrence_and_beta_symmetry() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_102_u64;
        let mode = RuntimeMode::Strict;

        for k in 1..=30 {
            let x = 0.1 * k as f64;
            let g_x = scalar_value(&gamma(&SpecialTensor::RealScalar(x), mode).expect("gamma(x)"));
            let g_x1 = scalar_value(
                &gamma(&SpecialTensor::RealScalar(x + 1.0), mode).expect("gamma(x+1)"),
            );
            let rhs = x * g_x;
            let rel = ((g_x1 - rhs) / rhs.max(1e-12)).abs();
            assert!(
                rel <= 2.5e-9,
                "gamma recurrence mismatch at x={x}: rel={rel}"
            );
            push_test_log(test_log_json(
                "prop-gamma-recurrence",
                "gamma",
                format!("x={x}"),
                format!("{rhs:.16e}"),
                format!("{g_x1:.16e}"),
                2.5e-9,
                mode,
                seed,
                "pass",
            ));
        }

        for a in [0.5, 1.0, 1.5, 2.0, 5.0] {
            for b in [0.25, 1.0, 2.5, 4.0] {
                let lhs = beta(
                    &SpecialTensor::RealScalar(a),
                    &SpecialTensor::RealScalar(b),
                    mode,
                )
                .expect("beta lhs");
                let rhs = beta(
                    &SpecialTensor::RealScalar(b),
                    &SpecialTensor::RealScalar(a),
                    mode,
                )
                .expect("beta rhs");
                assert_real_scalars_close(lhs.clone(), rhs.clone(), 1.0e-10);
                push_test_log(test_log_json(
                    "prop-beta-symmetry",
                    "beta",
                    format!("a={a},b={b}"),
                    scalar_to_string(&rhs),
                    scalar_to_string(&lhs),
                    1.0e-10,
                    mode,
                    seed,
                    "pass",
                ));
            }
        }

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn property_inverse_composition_and_gamma_complement() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_103_u64;
        let mode = RuntimeMode::Strict;

        for x in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0] {
            let erf_x = erf(&SpecialTensor::RealScalar(x), mode).expect("erf(x)");
            let inv = erfinv(&erf_x, mode).expect("erfinv(erf(x))");
            let back = scalar_value(&inv);
            assert!(
                (back - x).abs() <= 2.0e-6,
                "inverse mismatch at x={x}: got {back}"
            );
            push_test_log(test_log_json(
                "prop-erfinv-erf",
                "erfinv(erf(x))",
                format!("x={x}"),
                format!("{x:.16e}"),
                format!("{back:.16e}"),
                2.0e-6,
                mode,
                seed,
                "pass",
            ));
        }

        for a in [0.25, 0.5, 1.0, 2.0, 5.0] {
            for x in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0] {
                let p = gammainc(
                    &SpecialTensor::RealScalar(a),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("gammainc");
                let q = gammaincc(
                    &SpecialTensor::RealScalar(a),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("gammaincc");
                let sum = scalar_value(&p) + scalar_value(&q);
                assert!(
                    (sum - 1.0).abs() <= 3.0e-12,
                    "gammainc complement mismatch at a={a} x={x}"
                );
                push_test_log(test_log_json(
                    "prop-gammainc-complement",
                    "gammainc+gammaincc",
                    format!("a={a},x={x}"),
                    "1",
                    format!("{sum:.16e}"),
                    3.0e-12,
                    mode,
                    seed,
                    "pass",
                ));
            }
        }

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn unit_vector_broadcast_and_shape_edge_cases() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_200_u64;
        let mode = RuntimeMode::Strict;

        let vec_input = SpecialTensor::RealVec(vec![-1.0, 0.0, 1.0]);
        let erf_vec = erf(&vec_input, mode).expect("erf vec");
        let erfc_vec = erfc(&vec_input, mode).expect("erfc vec");
        let (erf_vals, erfc_vals) = match (&erf_vec, &erfc_vec) {
            (SpecialTensor::RealVec(lhs), SpecialTensor::RealVec(rhs)) => (lhs, rhs),
            _ => panic!("expected vector outputs"),
        };
        assert_eq!(erf_vals.len(), 3);
        assert_eq!(erfc_vals.len(), 3);
        for (idx, (lhs, rhs)) in erf_vals.iter().zip(erfc_vals.iter()).enumerate() {
            let sum = lhs + rhs;
            assert!(
                (sum - 1.0).abs() <= 3.0e-7,
                "vector erf+erfc identity failed at idx={idx}: {sum}"
            );
            push_test_log(test_log_json(
                "unit-erf-erfc-vector",
                "erf,erfc",
                format!("idx={idx},x={}", [-1.0_f64, 0.0, 1.0][idx]),
                "1",
                format!("{sum:.16e}"),
                3.0e-7,
                mode,
                seed,
                "pass",
            ));
        }

        let beta_vec = beta(
            &SpecialTensor::RealVec(vec![1.0, 2.0, 3.0]),
            &SpecialTensor::RealScalar(2.0),
            mode,
        )
        .expect("beta vector+scalar");
        let beta_vals = match beta_vec {
            SpecialTensor::RealVec(values) => values,
            _ => panic!("expected vector output"),
        };
        assert_eq!(beta_vals.len(), 3);
        assert!((beta_vals[0] - 0.5).abs() <= 1e-12);
        assert!((beta_vals[1] - (1.0 / 6.0)).abs() <= 1e-12);
        assert!((beta_vals[2] - (1.0 / 12.0)).abs() <= 1e-12);
        push_test_log(test_log_json(
            "unit-beta-vector-scalar",
            "beta",
            "a=[1,2,3],b=2",
            "[0.5,0.1666...,0.0833...]",
            format!(
                "[{:.16e},{:.16e},{:.16e}]",
                beta_vals[0], beta_vals[1], beta_vals[2]
            ),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let mismatch = beta(
            &SpecialTensor::RealVec(vec![1.0, 2.0]),
            &SpecialTensor::RealVec(vec![1.0]),
            RuntimeMode::Hardened,
        )
        .expect_err("mismatched vectors should fail");
        assert_eq!(mismatch.kind, SpecialErrorKind::DomainError);
        push_test_log(test_log_json(
            "unit-beta-mismatch-domain",
            "beta",
            "a_len=2,b_len=1,hardened",
            "DomainError",
            format!("{:?}", mismatch.kind),
            0.0,
            RuntimeMode::Hardened,
            seed,
            "pass",
        ));

        let j0_vec = j0(&SpecialTensor::RealVec(vec![0.0, 1.0, 2.0]), mode).expect("j0 vec");
        let j0_vals = match j0_vec {
            SpecialTensor::RealVec(values) => values,
            _ => panic!("expected vector output"),
        };
        assert_eq!(j0_vals.len(), 3);
        assert!((j0_vals[0] - 1.0).abs() <= 1e-8);
        push_test_log(test_log_json(
            "unit-j0-vector",
            "j0",
            "x=[0,1,2]",
            "len=3",
            format!(
                "[{:.16e},{:.16e},{:.16e}]",
                j0_vals[0], j0_vals[1], j0_vals[2]
            ),
            1e-8,
            mode,
            seed,
            "pass",
        ));

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn unit_domain_boundary_matrix_strict_vs_hardened() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_201_u64;

        let erfinv_left =
            erfinv(&SpecialTensor::RealScalar(-1.0), RuntimeMode::Strict).expect("erfinv(-1)");
        let erfinv_right =
            erfinv(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict).expect("erfinv(1)");
        match erfinv_left {
            SpecialTensor::RealScalar(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            _ => {
                panic!("expected scalar");
            }
        }
        match erfinv_right {
            SpecialTensor::RealScalar(v) => assert!(v.is_infinite() && v.is_sign_positive()),
            _ => {
                panic!("expected scalar");
            }
        }

        let erfcinv_zero =
            erfcinv(&SpecialTensor::RealScalar(0.0), RuntimeMode::Strict).expect("erfcinv(0)");
        let erfcinv_two =
            erfcinv(&SpecialTensor::RealScalar(2.0), RuntimeMode::Strict).expect("erfcinv(2)");
        match erfcinv_zero {
            SpecialTensor::RealScalar(v) => assert!(v.is_infinite() && v.is_sign_positive()),
            _ => {
                panic!("expected scalar");
            }
        }
        match erfcinv_two {
            SpecialTensor::RealScalar(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            _ => {
                panic!("expected scalar");
            }
        }

        let strict_bad_order = jn(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Strict,
        )
        .expect("strict returns NaN for non-integer order");
        assert_real_scalar_nan(strict_bad_order);
        let hardened_bad_order = jn(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Hardened,
        )
        .expect_err("hardened rejects non-integer order");
        assert_eq!(hardened_bad_order.kind, SpecialErrorKind::DomainError);

        let strict_bad_gamma = gammaincc(
            &SpecialTensor::RealScalar(-1.0),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .expect("strict returns NaN for invalid shape");
        assert_real_scalar_nan(strict_bad_gamma);
        let hardened_bad_gamma = gammaincc(
            &SpecialTensor::RealScalar(-1.0),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Hardened,
        )
        .expect_err("hardened rejects invalid shape");
        assert_eq!(hardened_bad_gamma.kind, SpecialErrorKind::DomainError);

        push_test_log(test_log_json(
            "unit-boundary-policy-matrix",
            "erfinv/erfcinv/jn/gammaincc",
            "boundary + malformed domain set",
            "strict nonfinite fallback / hardened domain error",
            "validated",
            0.0,
            RuntimeMode::Strict,
            seed,
            "pass",
        ));
        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn property_erfcinv_erfc_roundtrip_grid() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_300_u64;
        let mode = RuntimeMode::Strict;

        for i in 1..=120 {
            let y = (i as f64) / 60.0;
            let inv = erfcinv(&SpecialTensor::RealScalar(y), mode).expect("erfcinv(y)");
            let back = erfc(&inv, mode).expect("erfc(erfcinv(y))");
            let back_v = scalar_value(&back);
            assert!(
                (back_v - y).abs() <= 4.0e-6,
                "erfcinv/erfc roundtrip mismatch y={y}: {back_v}"
            );
            push_test_log(test_log_json(
                "prop-erfcinv-erfc-roundtrip",
                "erfc(erfcinv(y))",
                format!("y={y:.16e}"),
                format!("{y:.16e}"),
                format!("{back_v:.16e}"),
                4.0e-6,
                mode,
                seed,
                "pass",
            ));
        }

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn erfinv_tail_contract_points() {
        let mode = RuntimeMode::Strict;
        let pairs = [
            (0.999_999, 3.458_910_737_275_499, 1.0e-7),
            (0.999_999_999_999, 5.042_031_898_572_696, 2.0e-5),
            (-0.999_999, -3.458_910_737_275_499, 1.0e-7),
        ];

        for (y, expected, tol) in pairs {
            let actual = erfinv(&SpecialTensor::RealScalar(y), mode).expect("erfinv tail point");
            assert_real_scalar_close(actual, expected, tol);
        }
    }

    #[test]
    fn erfcinv_tiny_argument_remains_finite_and_accurate() {
        let mode = RuntimeMode::Strict;
        let actual =
            erfcinv(&SpecialTensor::RealScalar(1.0e-20), mode).expect("erfcinv tiny argument");
        assert_real_scalar_close(actual, 6.601_580_622_355_143, 1.0e-8);
    }

    #[test]
    fn property_bessel_envelope_large_x_grid() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_301_u64;
        let mode = RuntimeMode::Strict;

        for k in 0..200 {
            let x = 20.0 + 0.5 * k as f64;
            let j0_v = j0(&SpecialTensor::RealScalar(x), mode).expect("j0");
            let y0_v = y0(&SpecialTensor::RealScalar(x), mode).expect("y0");
            let lhs = scalar_value(&j0_v).powi(2) + scalar_value(&y0_v).powi(2);
            let rhs = 2.0 / (std::f64::consts::PI * x);
            assert!(
                (lhs - rhs).abs() <= 3.5e-3,
                "bessel envelope mismatch at x={x}: lhs={lhs} rhs={rhs}"
            );
            push_test_log(test_log_json(
                "prop-bessel-envelope",
                "j0^2+y0^2",
                format!("x={x:.16e}"),
                format!("{rhs:.16e}"),
                format!("{lhs:.16e}"),
                3.5e-3,
                mode,
                seed,
                "pass",
            ));
        }

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn property_jn_three_term_recurrence_grid() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_302_u64;
        let mode = RuntimeMode::Strict;

        for n in 1..=8 {
            for step in 1..=120 {
                let x = 0.5 + 0.05 * step as f64;
                let n_tensor = SpecialTensor::RealScalar(n as f64);
                let n_prev_tensor = SpecialTensor::RealScalar((n - 1) as f64);
                let n_next_tensor = SpecialTensor::RealScalar((n + 1) as f64);
                let x_tensor = SpecialTensor::RealScalar(x);

                let jn_v = scalar_value(&jn(&n_tensor, &x_tensor, mode).expect("jn"));
                let jn_prev = scalar_value(&jn(&n_prev_tensor, &x_tensor, mode).expect("jn-1"));
                let jn_next = scalar_value(&jn(&n_next_tensor, &x_tensor, mode).expect("jn+1"));
                let rhs = (2.0 * n as f64 / x) * jn_v - jn_prev;
                assert!(
                    (jn_next - rhs).abs() <= 2.0e-6,
                    "jn recurrence mismatch n={n} x={x}: lhs={jn_next} rhs={rhs}"
                );
                push_test_log(test_log_json(
                    "prop-jn-recurrence",
                    "J_{n+1}=(2n/x)J_n-J_{n-1}",
                    format!("n={n},x={x:.16e}"),
                    format!("{rhs:.16e}"),
                    format!("{jn_next:.16e}"),
                    2.0e-6,
                    mode,
                    seed,
                    "pass",
                ));
            }
        }

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn previously_pending_families_now_implemented() {
        let scalar = SpecialTensor::RealScalar(1.0);

        // jv is now implemented — verify it returns a value
        let jv_result = jv(&scalar, &scalar, RuntimeMode::Strict);
        assert!(
            jv_result.is_ok(),
            "jv(1,1) should succeed now that it's implemented"
        );

        // hyp2f1 is now implemented — verify it returns a value instead
        let hyper_result = hyp2f1(&scalar, &scalar, &scalar, &scalar, RuntimeMode::Strict);
        assert!(
            hyper_result.is_ok(),
            "hyp2f1(1,1,1,1) should succeed now that it's implemented"
        );
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

    #[test]
    fn ndtr_contract_points() {
        assert!((ndtr(0.0) - 0.5).abs() <= 1.0e-12);
        assert!((ndtr(1.959_963_984_540_054) - 0.975).abs() <= 5.0e-5);
        let tail = ndtr(-8.0);
        assert!(tail > 0.0 && tail < 1.0e-14, "ndtr(-8) tail={tail}");
        assert!(ndtr(8.0) > 1.0 - 1.0e-12, "ndtr(8) should be ~1");
        assert_eq!(ndtr(f64::NEG_INFINITY), 0.0);
        assert_eq!(ndtr(f64::INFINITY), 1.0);
    }

    #[test]
    fn ndtr_ndtri_roundtrip() {
        for x in [-4.0, -1.5, -0.25, 0.0, 0.5, 2.0, 4.0] {
            let back = ndtri(ndtr(x));
            assert!(
                (back - x).abs() <= 2.0e-4,
                "ndtri(ndtr(x)) mismatch at x={x}: {back}"
            );
        }
    }

    #[test]
    fn ndtri_domain_and_endpoints() {
        assert_eq!(ndtri(0.0), f64::NEG_INFINITY);
        assert_eq!(ndtri(1.0), f64::INFINITY);
        assert!(ndtri(-0.1).is_nan());
        assert!(ndtri(1.1).is_nan());
    }

    #[test]
    fn beta_distribution_cdf_and_inverse_roundtrip() {
        let value = btdtr(2.0, 5.0, 0.3);
        assert!(
            (value - 0.579_825).abs() <= 1.0e-6,
            "unexpected beta cdf: {value}"
        );

        let x = btdtri(2.0, 5.0, 0.3);
        let back = btdtr(2.0, 5.0, x);
        assert!(
            (back - 0.3).abs() <= 1.0e-8,
            "beta inverse roundtrip failed: {back}"
        );
    }

    #[test]
    fn beta_inverse_endpoints_and_invalid_inputs() {
        assert_eq!(btdtri(2.0, 5.0, 0.0), 0.0);
        assert_eq!(btdtri(2.0, 5.0, 1.0), 1.0);
        assert!(btdtr(2.0, 5.0, -0.1).is_nan());
        assert!(btdtri(-1.0, 5.0, 0.5).is_nan());
    }

    #[test]
    fn f_distribution_cdf_and_inverse_roundtrip() {
        let x = 1.75;
        let y = fdtr(5.0, 7.0, x);
        let back = fdtri(5.0, 7.0, y);
        assert!(
            (back - x).abs() <= 1.0e-8,
            "f inverse roundtrip failed: {back}"
        );
    }

    #[test]
    fn f_distribution_endpoints() {
        assert_eq!(fdtr(5.0, 7.0, 0.0), 0.0);
        assert_eq!(fdtri(5.0, 7.0, 0.0), 0.0);
        assert_eq!(fdtri(5.0, 7.0, 1.0), f64::INFINITY);
        assert!(fdtr(-1.0, 7.0, 1.0).is_nan());
    }

    #[test]
    fn gamma_distribution_cdf_and_inverse_roundtrip() {
        let x = 3.2;
        let y = gdtr(2.5, 1.25, x);
        let back = gdtri(2.5, 1.25, y);
        assert!(
            (back - x).abs() <= 1.0e-8,
            "gamma inverse roundtrip failed: {back}"
        );
    }

    #[test]
    fn gamma_distribution_exponential_special_case() {
        let x = 1.5_f64;
        let scale = 2.0_f64;
        let expected = 1.0 - (-(x / scale)).exp();
        assert!((gdtr(1.0, scale, x) - expected).abs() <= 1.0e-10);
    }

    #[test]
    fn gamma_inverse_endpoints_and_invalid_inputs() {
        assert_eq!(gdtri(2.0, 3.0, 0.0), 0.0);
        assert_eq!(gdtri(2.0, 3.0, 1.0), f64::INFINITY);
        assert!(gdtr(-1.0, 3.0, 1.0).is_nan());
        assert!(gdtri(2.0, -3.0, 0.5).is_nan());
    }

    #[test]
    fn kl_div_matches_rel_entr_offset() {
        let x = 0.7;
        let y = 1.4;
        let rel = rel_entr(
            &SpecialTensor::RealScalar(x),
            &SpecialTensor::RealScalar(y),
            RuntimeMode::Strict,
        )
        .expect("rel_entr");
        let expected = scalar_value(&rel) - x + y;
        assert!((kl_div(x, y) - expected).abs() <= 1.0e-12);
        assert_eq!(kl_div(0.0, 2.0), 2.0);
        assert_eq!(kl_div(1.0, 1.0), 0.0);
    }

    #[test]
    fn kl_div_invalid_inputs_are_infinite() {
        assert!(kl_div(-1.0, 1.0).is_infinite());
        assert!(kl_div(1.0, 0.0).is_infinite());
    }

    #[test]
    fn kl_div_nan_inputs_propagate() {
        assert!(kl_div(f64::NAN, 1.0).is_nan());
        assert!(kl_div(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn complex_erf_erfc_support_known_values_and_identities() {
        let mode = RuntimeMode::Strict;
        let i = SpecialTensor::ComplexScalar(Complex64::new(0.0, 1.0));

        let erf_i = erf(&i, mode).expect("erf(i)");
        assert_complex_scalar_close(
            erf_i.clone(),
            Complex64::new(0.0, 1.650_425_758_797_542_8),
            1.0e-11,
        );

        let erfc_i = erfc(&i, mode).expect("erfc(i)");
        assert_complex_scalar_close(
            erfc_i,
            Complex64::new(1.0, -1.650_425_758_797_542_8),
            1.0e-11,
        );

        let z = SpecialTensor::ComplexScalar(Complex64::new(1.0, 1.0));
        let z_conj = SpecialTensor::ComplexScalar(Complex64::new(1.0, -1.0));
        let erf_z = erf(&z, mode).expect("erf(1+i)");
        let erfc_z = erfc(&z, mode).expect("erfc(1+i)");
        let erf_conj = erf(&z_conj, mode).expect("erf(1-i)");

        assert_complex_scalar_close(
            add_complex_scalars(&erf_z, &erfc_z),
            Complex64::from_real(1.0),
            1.0e-11,
        );
        assert_complex_scalar_close(erf_conj, complex_scalar_value(&erf_z).conj(), 1.0e-11);
    }

    #[test]
    fn complex_vector_erf_preserves_shape_and_odd_symmetry() {
        let mode = RuntimeMode::Strict;
        let input =
            SpecialTensor::ComplexVec(vec![Complex64::new(0.0, 0.5), Complex64::new(0.0, -0.5)]);

        let output = erf(&input, mode).expect("complex vector erf");
        let values = match output {
            SpecialTensor::ComplexVec(values) => values,
            _ => panic!("expected complex vector output"),
        };

        assert_eq!(values.len(), 2);
        assert_complex_close(values[0], -values[1], 1.0e-12);
    }

    fn assert_real_scalar_close(actual: SpecialTensor, expected: f64, tol: f64) {
        match actual {
            SpecialTensor::RealScalar(value) => {
                assert!(
                    (value - expected).abs() <= tol,
                    "expected {expected}, got {value}"
                );
            }
            _ => {
                panic!("expected real scalar output");
            }
        }
    }

    fn assert_real_scalar_nan(actual: SpecialTensor) {
        match actual {
            SpecialTensor::RealScalar(value) => assert!(value.is_nan(), "expected NaN"),
            _ => {
                panic!("expected real scalar output");
            }
        }
    }

    fn assert_real_scalars_close(actual: SpecialTensor, expected: SpecialTensor, tol: f64) {
        match (actual, expected) {
            (SpecialTensor::RealScalar(lhs), SpecialTensor::RealScalar(rhs)) => {
                assert!((lhs - rhs).abs() <= tol, "expected {rhs}, got {lhs}");
            }
            _ => panic!("expected scalar outputs"),
        }
    }

    fn assert_complex_scalar_close(actual: SpecialTensor, expected: Complex64, tol: f64) {
        match actual {
            SpecialTensor::ComplexScalar(value) => assert_complex_close(value, expected, tol),
            _ => panic!("expected complex scalar output"),
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

    fn trace_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static TRACE_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        TRACE_TEST_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("trace test lock poisoned")
    }

    fn scalar_value(tensor: &SpecialTensor) -> f64 {
        match tensor {
            SpecialTensor::RealScalar(value) => *value,
            _ => panic!("expected scalar tensor"),
        }
    }

    fn complex_scalar_value(tensor: &SpecialTensor) -> Complex64 {
        match tensor {
            SpecialTensor::ComplexScalar(value) => *value,
            _ => panic!("expected complex scalar tensor"),
        }
    }

    fn add_complex_scalars(lhs: &SpecialTensor, rhs: &SpecialTensor) -> SpecialTensor {
        SpecialTensor::ComplexScalar(complex_scalar_value(lhs) + complex_scalar_value(rhs))
    }

    fn scalar_to_string(tensor: &SpecialTensor) -> String {
        format!("{:.16e}", scalar_value(tensor))
    }

    fn test_logs() -> &'static Mutex<Vec<String>> {
        static TEST_LOGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
        TEST_LOGS.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn clear_test_logs() {
        if let Ok(mut logs) = test_logs().lock() {
            logs.clear();
        }
    }

    fn push_test_log(line: String) {
        if let Ok(mut logs) = test_logs().lock() {
            logs.push(line);
        }
    }

    fn take_test_logs() -> Vec<String> {
        if let Ok(mut logs) = test_logs().lock() {
            let mut out = Vec::with_capacity(logs.len());
            std::mem::swap(&mut out, &mut *logs);
            return out;
        }
        Vec::new()
    }

    #[allow(clippy::too_many_arguments)]
    fn test_log_json(
        test_id: impl Into<String>,
        function: impl Into<String>,
        input: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
        tolerance: f64,
        mode: RuntimeMode,
        seed: u64,
        result: impl Into<String>,
    ) -> String {
        let test_id = test_id.into();
        let function = function.into();
        let input = input.into();
        let expected = expected.into();
        let actual = actual.into();
        let result = result.into();
        format!(
            "{{\"test_id\":\"{}\",\"function\":\"{}\",\"input\":\"{}\",\"expected\":\"{}\",\"actual\":\"{}\",\"tolerance\":{},\"mode\":\"{:?}\",\"seed\":{},\"timestamp_ms\":{},\"result\":\"{}\"}}",
            escape_json(&test_id),
            escape_json(&function),
            escape_json(&input),
            escape_json(&expected),
            escape_json(&actual),
            tolerance,
            mode,
            seed,
            now_unix_ms(),
            escape_json(&result),
        )
    }

    fn assert_logs_follow_schema(logs: &[String]) {
        assert!(!logs.is_empty(), "expected non-empty structured logs");
        for line in logs {
            assert!(line.starts_with('{'));
            assert!(line.contains("\"test_id\""));
            assert!(line.contains("\"function\""));
            assert!(line.contains("\"input\""));
            assert!(line.contains("\"expected\""));
            assert!(line.contains("\"actual\""));
            assert!(line.contains("\"tolerance\""));
            assert!(line.contains("\"mode\""));
            assert!(line.contains("\"seed\""));
            assert!(line.contains("\"timestamp_ms\""));
            assert!(line.contains("\"result\""));
        }
    }

    fn now_unix_ms() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis())
    }

    fn escape_json(input: &str) -> String {
        input
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }

    // ── Combinatorial function tests ────────────────────────────────

    #[test]
    fn factorial_small_values() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3_628_800.0);
        assert_eq!(factorial(20), 2_432_902_008_176_640_000.0);
    }

    #[test]
    fn factorial_large_values() {
        // 21! should use gamma path
        let f21 = factorial(21);
        assert!((f21 - 51_090_942_171_709_440_000.0).abs() / f21 < 1e-10);
        // Large factorials overflow to infinity in f64
        // f64 max is ~1.8e308, and 171! ≈ 1.24e309, so 171+ overflows
        let f170 = factorial(170);
        // 170! ≈ 7.26e370, also overflows
        assert!(f170.is_infinite());
    }

    #[test]
    fn comb_basic() {
        assert_eq!(comb(5, 0), 1.0);
        assert_eq!(comb(5, 5), 1.0);
        assert_eq!(comb(5, 2), 10.0);
        assert_eq!(comb(10, 3), 120.0);
        assert_eq!(comb(20, 10), 184_756.0);
    }

    #[test]
    fn comb_edge_cases() {
        assert_eq!(comb(0, 0), 1.0);
        assert_eq!(comb(5, 6), 0.0); // k > n
        assert_eq!(comb(100, 1), 100.0);
        assert_eq!(comb(100, 99), 100.0); // symmetry
    }

    #[test]
    fn comb_large() {
        // C(200, 100) is a huge number; verify it's finite and reasonable
        let c = comb(200, 100);
        assert!(c.is_finite());
        assert!(c > 1e58); // known to be ~9.05e58
    }

    #[test]
    fn perm_basic() {
        assert_eq!(perm(5, 0), 1.0);
        assert_eq!(perm(5, 1), 5.0);
        assert_eq!(perm(5, 2), 20.0);
        assert_eq!(perm(5, 5), 120.0);
        assert_eq!(perm(10, 3), 720.0);
    }

    #[test]
    fn perm_edge_cases() {
        assert_eq!(perm(0, 0), 1.0);
        assert_eq!(perm(5, 6), 0.0); // k > n
    }

    #[test]
    fn perm_relation_to_comb() {
        // P(n, k) = C(n, k) * k!
        for n in 0..15_u64 {
            for k in 0..=n {
                let p = perm(n, k);
                let c = comb(n, k);
                let kfact = factorial(k);
                assert!(
                    (p - c * kfact).abs() < 1e-6 * p.max(1.0),
                    "P({n},{k})={p} != C({n},{k})*{k}!={}",
                    c * kfact
                );
            }
        }
    }

    // ── Zeta function tests ─────────────────────────────────────────

    #[test]
    fn zeta_known_values() {
        // ζ(2) = π²/6
        let z2 = zeta(2.0);
        let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!(
            (z2 - expected).abs() < 1e-3,
            "zeta(2) = {z2}, expected {expected}"
        );

        // ζ(4) = π⁴/90
        let z4 = zeta(4.0);
        let expected4 = std::f64::consts::PI.powi(4) / 90.0;
        assert!(
            (z4 - expected4).abs() < 1e-4,
            "zeta(4) = {z4}, expected {expected4}"
        );
    }

    #[test]
    fn zeta_at_zero() {
        assert!((zeta(0.0) - (-0.5)).abs() < 1e-12, "zeta(0) = -1/2");
    }

    #[test]
    fn zeta_pole_at_one() {
        assert!(zeta(1.0).is_infinite());
    }

    #[test]
    fn zeta_negative_even_integers() {
        // ζ(-2n) = 0 for positive integers n (trivial zeros)
        assert!(zeta(-2.0).abs() < 1e-6, "zeta(-2) = {}", zeta(-2.0));
        assert!(zeta(-4.0).abs() < 1e-4, "zeta(-4) = {}", zeta(-4.0));
    }

    #[test]
    fn zeta_nan_passthrough() {
        assert!(zeta(f64::NAN).is_nan());
    }

    // ── Spherical Bessel function tests ──────────────────────────────

    #[test]
    fn spherical_jn_j0_is_sinc() {
        // j_0(x) = sin(x)/x
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
            let result = spherical_jn(
                &SpecialTensor::RealScalar(0.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("spherical_jn(0, x)");
            let expected = x.sin() / x;
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    #[test]
    fn spherical_jn_at_zero() {
        let mode = RuntimeMode::Strict;
        // j_0(0) = 1
        let j0_0 = spherical_jn(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("j_0(0)");
        assert_real_scalar_close(j0_0, 1.0, 1e-12);

        // j_n(0) = 0 for n > 0
        for n in 1..=5 {
            let jn_0 = spherical_jn(
                &SpecialTensor::RealScalar(n as f64),
                &SpecialTensor::RealScalar(0.0),
                mode,
            )
            .unwrap_or_else(|_| panic!("j_{n}(0)"));
            assert_real_scalar_close(jn_0, 0.0, 1e-12);
        }
    }

    #[test]
    fn spherical_jn_recurrence() {
        // (2n+1) j_n(z) = z (j_{n-1}(z) + j_{n+1}(z))
        let mode = RuntimeMode::Strict;
        for n in 1..=6 {
            for x in [1.0, 2.5, 5.0, 10.0] {
                let jn_v = scalar_value(
                    &spherical_jn(
                        &SpecialTensor::RealScalar(n as f64),
                        &SpecialTensor::RealScalar(x),
                        mode,
                    )
                    .expect("jn"),
                );
                let jn_prev = scalar_value(
                    &spherical_jn(
                        &SpecialTensor::RealScalar((n - 1) as f64),
                        &SpecialTensor::RealScalar(x),
                        mode,
                    )
                    .expect("jn-1"),
                );
                let jn_next = scalar_value(
                    &spherical_jn(
                        &SpecialTensor::RealScalar((n + 1) as f64),
                        &SpecialTensor::RealScalar(x),
                        mode,
                    )
                    .expect("jn+1"),
                );
                let lhs = (2.0 * n as f64 + 1.0) * jn_v;
                let rhs = x * (jn_prev + jn_next);
                assert!(
                    (lhs - rhs).abs() < 1e-8,
                    "spherical jn recurrence n={n} x={x}: lhs={lhs} rhs={rhs}"
                );
            }
        }
    }

    #[test]
    fn spherical_yn_y0_is_neg_cos_over_x() {
        // y_0(x) = -cos(x)/x
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0, 5.0, 10.0] {
            let result = spherical_yn(
                &SpecialTensor::RealScalar(0.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("spherical_yn(0, x)");
            let expected = -x.cos() / x;
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    #[test]
    fn spherical_yn_at_zero_is_neg_inf() {
        let mode = RuntimeMode::Strict;
        let result = spherical_yn(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("y_0(0)");
        match result {
            SpecialTensor::RealScalar(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            _ => {
                panic!("expected scalar");
            }
        }
    }

    #[test]
    fn spherical_in_i0_is_sinh_over_x() {
        // i_0(x) = sinh(x)/x
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0, 5.0] {
            let result = spherical_in(
                &SpecialTensor::RealScalar(0.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("spherical_in(0, x)");
            let expected = x.sinh() / x;
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    #[test]
    fn spherical_in_at_zero() {
        let mode = RuntimeMode::Strict;
        let i0_0 = spherical_in(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("i_0(0)");
        assert_real_scalar_close(i0_0, 1.0, 1e-12);

        let i1_0 = spherical_in(
            &SpecialTensor::RealScalar(1.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("i_1(0)");
        assert_real_scalar_close(i1_0, 0.0, 1e-12);
    }

    #[test]
    fn spherical_kn_k0_is_exp_neg_x_over_x() {
        // k_0(x) = exp(-x)/x
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0, 5.0] {
            let result = spherical_kn(
                &SpecialTensor::RealScalar(0.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("spherical_kn(0, x)");
            let expected = (-x).exp() / x;
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    #[test]
    fn spherical_kn_at_zero_is_inf() {
        let mode = RuntimeMode::Strict;
        let result = spherical_kn(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("k_0(0)");
        match result {
            SpecialTensor::RealScalar(v) => assert!(v.is_infinite()),
            _ => {
                panic!("expected scalar");
            }
        }
    }

    #[test]
    fn spherical_bessel_negative_order_rejected_in_hardened() {
        let mode = RuntimeMode::Hardened;
        let neg = SpecialTensor::RealScalar(-1.0);
        let x = SpecialTensor::RealScalar(1.0);

        let err = spherical_jn(&neg, &x, mode).expect_err("negative order");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
    }

    #[test]
    fn spherical_jn_j1_formula() {
        // j_1(x) = sin(x)/x² - cos(x)/x
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 3.0, 7.0] {
            let result = spherical_jn(
                &SpecialTensor::RealScalar(1.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("j_1(x)");
            let expected = x.sin() / (x * x) - x.cos() / x;
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    #[test]
    fn spherical_kn_k1_formula() {
        // k_1(x) = exp(-x)/x * (1 + 1/x)
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 3.0] {
            let result = spherical_kn(
                &SpecialTensor::RealScalar(1.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("k_1(x)");
            let expected = (-x).exp() / x * (1.0 + 1.0 / x);
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    // ── Real-order Bessel function tests ─────────────────────────────

    #[test]
    fn jv_integer_order_matches_jn() {
        // jv(n, x) should match jn(n, x) for integer n
        let mode = RuntimeMode::Strict;
        for n in 0..=5 {
            let x = 2.5;
            let jv_result = scalar_value(
                &jv(
                    &SpecialTensor::RealScalar(n as f64),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("jv"),
            );
            let jn_result = scalar_value(
                &jn(
                    &SpecialTensor::RealScalar(n as f64),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("jn"),
            );
            assert!(
                (jv_result - jn_result).abs() < 1e-8,
                "jv({n}, {x}) = {jv_result}, jn = {jn_result}"
            );
        }
    }

    #[test]
    fn jv_half_order() {
        // J_{1/2}(z) = sqrt(2/(πz)) sin(z)
        let mode = RuntimeMode::Strict;
        for x in [1.0, 2.0, 5.0, 10.0] {
            let result = scalar_value(
                &jv(
                    &SpecialTensor::RealScalar(0.5),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("jv(0.5)"),
            );
            let expected = (2.0 / (std::f64::consts::PI * x)).sqrt() * x.sin();
            assert!(
                (result - expected).abs() < 0.01,
                "J_1/2({x}): got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn yv_positive_z() {
        // Y_v should be finite for positive z
        let mode = RuntimeMode::Strict;
        let result = yv(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(1.0),
            mode,
        )
        .expect("yv(0.5, 1)");
        match result {
            SpecialTensor::RealScalar(v) => {
                assert!(v.is_finite(), "Y_0.5(1) should be finite: {v}")
            }
            _ => {
                panic!("expected scalar");
            }
        }
    }

    #[test]
    fn yv_negative_z_strict_returns_nan() {
        let mode = RuntimeMode::Strict;
        let result = yv(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(-1.0),
            mode,
        )
        .expect("yv strict");
        assert_real_scalar_nan(result);
    }

    #[test]
    fn iv_at_zero() {
        let mode = RuntimeMode::Strict;
        // I_0(0) = 1
        let result = iv(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("iv(0,0)");
        assert_real_scalar_close(result, 1.0, 1e-12);
        // I_v(0) = 0 for v > 0
        let result = iv(
            &SpecialTensor::RealScalar(1.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("iv(1,0)");
        assert_real_scalar_close(result, 0.0, 1e-12);
    }

    #[test]
    fn iv_half_order() {
        // I_{1/2}(z) = sqrt(2/(πz)) sinh(z)
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0] {
            let result = scalar_value(
                &iv(
                    &SpecialTensor::RealScalar(0.5),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("iv(0.5)"),
            );
            let expected = (2.0 / (std::f64::consts::PI * x)).sqrt() * x.sinh();
            assert!(
                (result - expected).abs() < 0.01,
                "I_1/2({x}): got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn kv_positive_z() {
        let mode = RuntimeMode::Strict;
        let result = kv(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(1.0),
            mode,
        )
        .expect("kv(0.5, 1)");
        match result {
            SpecialTensor::RealScalar(v) => {
                assert!(v.is_finite() && v > 0.0, "K_0.5(1) should be positive: {v}")
            }
            _ => {
                panic!("expected scalar");
            }
        }
    }

    #[test]
    fn kv_at_zero_is_inf() {
        let mode = RuntimeMode::Strict;
        let result = kv(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("kv(0,0)");
        match result {
            SpecialTensor::RealScalar(v) => assert!(v.is_infinite()),
            _ => {
                panic!("expected scalar");
            }
        }
    }

    #[test]
    fn kv_integer_order_positive_and_finite() {
        // Small integer-order K_n(1) values should be finite and match reference values.
        let mode = RuntimeMode::Strict;
        for (n, target) in [
            (0.0, 0.421_024_438_240_708_34),
            (1.0, 0.601_907_230_197_234_6),
            (2.0, 1.624_838_898_635_177_4),
        ] {
            let result = kv(
                &SpecialTensor::RealScalar(n),
                &SpecialTensor::RealScalar(1.0),
                mode,
            )
            .unwrap_or_else(|_| panic!("kv({n}, 1)"));
            match result {
                SpecialTensor::RealScalar(v) => {
                    assert!(
                        v.is_finite() && v > 0.0,
                        "K_{n}(1) should be positive finite: {v}"
                    );
                    assert!(
                        (v - target).abs() <= 2.0e-10,
                        "K_{n}(1) mismatch: expected {target}, got {v}"
                    );
                }
                _ => {
                    panic!("expected scalar");
                }
            }
        }
    }

    #[test]
    fn kv_half_order_formula() {
        // K_{1/2}(z) = sqrt(π/(2z)) * exp(-z)
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0] {
            let result = scalar_value(
                &kv(
                    &SpecialTensor::RealScalar(0.5),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("kv(0.5)"),
            );
            let expected = (std::f64::consts::PI / (2.0 * x)).sqrt() * (-x).exp();
            assert!(
                (result - expected).abs() < 0.01,
                "K_1/2({x}): got {result}, expected {expected}"
            );
        }
    }

    // ── Bernoulli / Euler / Hurwitz zeta tests ───────────────────────

    #[test]
    fn bernoulli_known_values() {
        assert!((bernoulli(0) - 1.0).abs() < 1e-12);
        assert!((bernoulli(1) - (-0.5)).abs() < 1e-12);
        assert!((bernoulli(2) - (1.0 / 6.0)).abs() < 1e-12);
        assert!((bernoulli(3)).abs() < 1e-12); // odd > 1 are zero
        assert!((bernoulli(4) - (-1.0 / 30.0)).abs() < 1e-12);
    }

    #[test]
    fn euler_known_values() {
        assert!((euler(0) - 1.0).abs() < 1e-12);
        assert!((euler(1)).abs() < 1e-12); // odd are zero
        assert!((euler(2) - (-1.0)).abs() < 1e-12);
        assert!((euler(4) - 5.0).abs() < 1e-12);
        assert!((euler(6) - (-61.0)).abs() < 1e-12);
    }

    #[test]
    fn hurwitz_zeta_reduces_to_riemann() {
        // ζ(s, 1) = ζ(s) (Riemann zeta)
        let riemann = zeta(2.0);
        let hurwitz = hurwitz_zeta(2.0, 1.0);
        assert!(
            (hurwitz - riemann).abs() < 1e-4,
            "hurwitz(2,1) = {hurwitz}, riemann(2) = {riemann}"
        );
    }

    #[test]
    fn hurwitz_zeta_known_value() {
        // ζ(2, 0.5) = π² / 2 ≈ 4.9348
        let result = hurwitz_zeta(2.0, 0.5);
        let expected = std::f64::consts::PI * std::f64::consts::PI / 2.0;
        assert!(
            (result - expected).abs() < 0.01,
            "ζ(2, 0.5) = {result}, expected {expected}"
        );
    }

    #[test]
    fn hurwitz_zeta_nan_inputs() {
        assert!(hurwitz_zeta(f64::NAN, 1.0).is_nan());
        assert!(hurwitz_zeta(2.0, -1.0).is_nan());
    }

    #[test]
    fn hurwitz_zeta_pole() {
        assert!(hurwitz_zeta(1.0, 1.0).is_infinite());
    }

    // ── sici tests ───────────────────────────────────────────────────

    #[test]
    fn sici_zero() {
        let (si, ci) = sici(0.0);
        assert_eq!(si, 0.0);
        assert!(ci.is_infinite() && ci < 0.0); // Ci(0) = -∞
    }

    #[test]
    fn sici_small_positive() {
        // Si(1) ≈ 0.9460831, Ci(1) ≈ 0.3374039
        let (si, ci) = sici(1.0);
        assert!((si - 0.9460831).abs() < 1e-5, "Si(1) = {si}");
        assert!((ci - 0.3374039).abs() < 1e-5, "Ci(1) = {ci}");
    }

    #[test]
    fn sici_large() {
        // Si(10) ≈ 1.6583475, Ci(10) ≈ -0.04545643
        let (si, ci) = sici(10.0);
        assert!((si - 1.6583475).abs() < 1e-5, "Si(10) = {si}");
        assert!((ci - (-0.04545643)).abs() < 1e-5, "Ci(10) = {ci}");
    }

    #[test]
    fn sici_negative() {
        // Si(-x) = -Si(x)
        let (si_pos, ci_pos) = sici(2.0);
        let (si_neg, ci_neg) = sici(-2.0);
        assert!((si_neg + si_pos).abs() < 1e-10);
        assert!((ci_neg - ci_pos).abs() < 1e-10);
    }

    // ── shichi tests ─────────────────────────────────────────────────

    #[test]
    fn shichi_zero() {
        let (shi, chi) = shichi(0.0);
        assert_eq!(shi, 0.0);
        assert!(chi.is_infinite() && chi < 0.0); // Chi(0) = -∞
    }

    #[test]
    fn shichi_small_positive() {
        // Shi(1) ≈ 1.0572509, Chi(1) ≈ 0.8378669
        let (shi, chi) = shichi(1.0);
        assert!((shi - 1.0572509).abs() < 1e-5, "Shi(1) = {shi}");
        assert!((chi - 0.8378669).abs() < 1e-5, "Chi(1) = {chi}");
    }

    #[test]
    fn shichi_negative() {
        // Shi(-x) = -Shi(x), Chi(-x) = Chi(x)
        let (shi_pos, chi_pos) = shichi(2.0);
        let (shi_neg, chi_neg) = shichi(-2.0);
        assert!((shi_neg + shi_pos).abs() < 1e-10);
        assert!((chi_neg - chi_pos).abs() < 1e-10);
    }

    // ── hyp0f1 tests ─────────────────────────────────────────────────

    #[test]
    fn hyp0f1_at_zero() {
        // 0F1(; b; 0) = 1 for any b
        let result = hyp0f1_scalar(2.0, 0.0, RuntimeMode::Strict).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn hyp0f1_small_positive() {
        // 0F1(; 1; z) = I_0(2*sqrt(z)) for z > 0
        // At z = 1: 0F1(; 1; 1) = I_0(2) ≈ 2.2795853
        let result = hyp0f1_scalar(1.0, 1.0, RuntimeMode::Strict).unwrap();
        assert!((result - 2.2795853).abs() < 0.01, "0F1(;1;1) = {result}");
    }

    #[test]
    fn hyp0f1_negative_z() {
        // 0F1(; 1; -z²/4) = J_0(z) for integer order
        // 0F1(; 1; -1) = J_0(2) ≈ 0.2238907
        let result = hyp0f1_scalar(1.0, -1.0, RuntimeMode::Strict).unwrap();
        assert!((result - 0.2238907).abs() < 0.01, "0F1(;1;-1) = {result}");
    }

    #[test]
    fn hyp0f1_pole() {
        // 0F1(; 0; z) is undefined (pole at b=0)
        let result = hyp0f1_scalar(0.0, 1.0, RuntimeMode::Strict).unwrap();
        assert!(result.is_nan());
    }

    #[test]
    fn hyp0f1_hardened_pole() {
        // In hardened mode, pole should return error
        let result = hyp0f1_scalar(-1.0, 1.0, RuntimeMode::Hardened);
        assert!(result.is_err());
    }
}
