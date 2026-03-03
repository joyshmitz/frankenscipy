#![forbid(unsafe_code)]

pub mod bessel;
pub mod beta;
pub mod error;
pub mod gamma;
pub mod hyper;
pub mod types;

pub use bessel::{BESSEL_DISPATCH_PLAN, hankel1, hankel2, iv, j0, j1, jn, jv, kv, y0, y1, yn, yv};
pub use beta::{BETA_DISPATCH_PLAN, beta, betainc, betaln};
pub use error::{ERROR_DISPATCH_PLAN, erf, erfc, erfcinv, erfinv};
pub use gamma::{
    GAMMA_DISPATCH_PLAN, digamma, gamma, gammainc, gammaincc, gammaln, polygamma, rgamma,
};
pub use hyper::{HYPER_DISPATCH_PLAN, hyp1f1, hyp2f1};
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
            _ => panic!("expected real scalar output"),
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
            _ => panic!("expected real scalar"),
        }

        let y1_zero = y1(&zero, RuntimeMode::Strict).expect("strict y1(0) should diverge");
        match y1_zero {
            SpecialTensor::RealScalar(value) => {
                assert!(value.is_infinite() && value.is_sign_negative())
            }
            _ => panic!("expected real scalar"),
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
                &format!("x={x}"),
                "1",
                &format!("{lhs:.16e}"),
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
                &format!("x={n}"),
                &format!("{expected}"),
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
            &format!("{complement:.16e}"),
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
            &format!(
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
                &format!("x={x}"),
                &format!("{rhs:.16e}"),
                &format!("{lhs:.16e}"),
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
                &format!("x={x}"),
                "1",
                &format!("{lhs:.16e}"),
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
                &format!("x={x}"),
                &format!("{rhs:.16e}"),
                &format!("{g_x1:.16e}"),
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
                    &format!("a={a},b={b}"),
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
                &format!("x={x}"),
                &format!("{x:.16e}"),
                &format!("{back:.16e}"),
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
                    &format!("a={a},x={x}"),
                    "1",
                    &format!("{sum:.16e}"),
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

    fn assert_real_scalars_close(actual: SpecialTensor, expected: SpecialTensor, tol: f64) {
        match (actual, expected) {
            (SpecialTensor::RealScalar(lhs), SpecialTensor::RealScalar(rhs)) => {
                assert!((lhs - rhs).abs() <= tol, "expected {rhs}, got {lhs}");
            }
            _ => panic!("expected scalar outputs"),
        }
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
}
