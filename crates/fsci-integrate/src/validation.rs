#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

pub const EPS: f64 = f64::EPSILON;
pub const MIN_RTOL: f64 = 100.0 * EPS;

#[derive(Debug, Clone, PartialEq)]
pub enum ToleranceValue {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl ToleranceValue {
    fn map(self, mut f: impl FnMut(f64) -> f64) -> Self {
        match self {
            Self::Scalar(value) => Self::Scalar(f(value)),
            Self::Vector(values) => Self::Vector(values.into_iter().map(f).collect()),
        }
    }

    fn any(self, mut predicate: impl FnMut(f64) -> bool) -> bool {
        match self {
            Self::Scalar(value) => predicate(value),
            Self::Vector(values) => values.into_iter().any(predicate),
        }
    }

    fn len_if_vector(&self) -> Option<usize> {
        match self {
            Self::Scalar(_) => None,
            Self::Vector(values) => Some(values.len()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToleranceWarning {
    RtolClamped { minimum: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedTolerance {
    pub rtol: ToleranceValue,
    pub atol: ToleranceValue,
    pub mode: RuntimeMode,
    pub warnings: Vec<ToleranceWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrateValidationError {
    FirstStepMustBePositive,
    FirstStepExceedsBounds,
    MaxStepMustBePositive,
    AtolWrongShape { expected: usize, actual: usize },
    AtolMustBePositive,
    NotYetImplemented { function: &'static str },
}

impl std::fmt::Display for IntegrateValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstStepMustBePositive => write!(f, "`first_step` must be positive."),
            Self::FirstStepExceedsBounds => write!(f, "`first_step` exceeds bounds."),
            Self::MaxStepMustBePositive => write!(f, "`max_step` must be positive."),
            Self::AtolWrongShape { .. } => write!(f, "`atol` has wrong shape."),
            Self::AtolMustBePositive => write!(f, "`atol` must be positive."),
            Self::NotYetImplemented { function } => {
                write!(f, "`{function}` is planned but not implemented yet.")
            }
        }
    }
}

impl std::error::Error for IntegrateValidationError {}

pub fn validate_first_step(
    first_step: f64,
    t0: f64,
    t_bound: f64,
) -> Result<f64, IntegrateValidationError> {
    if first_step <= 0.0 {
        return Err(IntegrateValidationError::FirstStepMustBePositive);
    }
    if first_step > (t_bound - t0).abs() {
        return Err(IntegrateValidationError::FirstStepExceedsBounds);
    }
    Ok(first_step)
}

pub fn validate_max_step(max_step: f64) -> Result<f64, IntegrateValidationError> {
    if max_step <= 0.0 {
        return Err(IntegrateValidationError::MaxStepMustBePositive);
    }
    Ok(max_step)
}

pub fn validate_tol(
    rtol: ToleranceValue,
    atol: ToleranceValue,
    n: usize,
    mode: RuntimeMode,
) -> Result<ValidatedTolerance, IntegrateValidationError> {
    let mut warnings = Vec::new();
    let needs_clamp = rtol.clone().any(|x| x < MIN_RTOL);
    let rtol = if needs_clamp {
        warnings.push(ToleranceWarning::RtolClamped { minimum: MIN_RTOL });
        rtol.map(|x| x.max(MIN_RTOL))
    } else {
        rtol
    };

    if let Some(len) = atol.len_if_vector()
        && len != n
    {
        return Err(IntegrateValidationError::AtolWrongShape {
            expected: n,
            actual: len,
        });
    }

    if atol.clone().any(|x| x < 0.0) {
        return Err(IntegrateValidationError::AtolMustBePositive);
    }

    Ok(ValidatedTolerance {
        rtol,
        atol,
        mode,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── validate_tol scalar tests ────────────────────────────────

    // 1. rtol within range -> passthrough (Strict)
    #[test]
    fn test_validation_tol_scalar_rtol_within_range_strict() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(1e-6),
            3,
            RuntimeMode::Strict,
        )
        .expect("valid tolerances");
        assert!(report.warnings.is_empty());
        assert_eq!(report.rtol, ToleranceValue::Scalar(1e-3));
    }

    // 1b. rtol within range -> passthrough (Hardened)
    #[test]
    fn test_validation_tol_scalar_rtol_within_range_hardened() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(1e-6),
            3,
            RuntimeMode::Hardened,
        )
        .expect("valid tolerances");
        assert!(report.warnings.is_empty());
        assert_eq!(report.mode, RuntimeMode::Hardened);
    }

    // 2. rtol below MIN_RTOL -> clamped with warning
    #[test]
    fn test_validation_tol_scalar_rtol_below_min_clamped() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-30),
            ToleranceValue::Scalar(1e-8),
            3,
            RuntimeMode::Strict,
        )
        .expect("tolerance should validate");
        assert_eq!(
            report,
            ValidatedTolerance {
                rtol: ToleranceValue::Scalar(MIN_RTOL),
                atol: ToleranceValue::Scalar(1e-8),
                mode: RuntimeMode::Strict,
                warnings: vec![ToleranceWarning::RtolClamped { minimum: MIN_RTOL }],
            }
        );
    }

    // 3. rtol = 0.0 -> clamped
    #[test]
    fn test_validation_tol_scalar_rtol_zero_clamped() {
        let report = validate_tol(
            ToleranceValue::Scalar(0.0),
            ToleranceValue::Scalar(1e-6),
            1,
            RuntimeMode::Strict,
        )
        .expect("zero rtol should be clamped");
        assert!(!report.warnings.is_empty());
        match report.rtol {
            ToleranceValue::Scalar(v) => assert_eq!(v, MIN_RTOL),
            _ => panic!("expected scalar"),
        }
    }

    // 4. negative rtol -> clamped (SciPy clamps, doesn't error)
    #[test]
    fn test_validation_tol_scalar_negative_rtol_clamped() {
        // SciPy's behavior: negative rtol gets clamped to MIN_RTOL
        let report = validate_tol(
            ToleranceValue::Scalar(-1.0),
            ToleranceValue::Scalar(1e-6),
            1,
            RuntimeMode::Strict,
        )
        .expect("negative rtol should be clamped");
        assert!(!report.warnings.is_empty());
    }

    // ── validate_tol vector tests ────────────────────────────────

    // 5. matching dimension -> passthrough
    #[test]
    fn test_validation_tol_vector_matching_dim() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Vector(vec![1e-6, 1e-7, 1e-8]),
            3,
            RuntimeMode::Strict,
        )
        .expect("matching vector atol should succeed");
        assert!(report.warnings.is_empty());
    }

    // 6. wrong dimension -> AtolWrongShape error
    #[test]
    fn test_validation_tol_vector_wrong_dim() {
        let err = validate_tol(
            ToleranceValue::Scalar(1e-6),
            ToleranceValue::Vector(vec![1e-9, 1e-9]),
            3,
            RuntimeMode::Strict,
        )
        .expect_err("wrong atol shape must fail");
        assert_eq!(
            err,
            IntegrateValidationError::AtolWrongShape {
                expected: 3,
                actual: 2
            }
        );
    }

    // 7. negative element -> AtolMustBePositive error
    #[test]
    fn test_validation_tol_vector_negative_element() {
        let err = validate_tol(
            ToleranceValue::Scalar(1e-6),
            ToleranceValue::Vector(vec![1e-9, -1e-9, 1e-9]),
            3,
            RuntimeMode::Hardened,
        )
        .expect_err("negative atol must fail");
        assert_eq!(err, IntegrateValidationError::AtolMustBePositive);
    }

    // 8. NaN input in atol -> appropriate behavior
    #[test]
    fn test_validation_tol_nan_atol() {
        // NaN is not < 0 so it passes the negative check but is pathological
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(f64::NAN),
            1,
            RuntimeMode::Strict,
        );
        // NaN passes the any(|x| x < 0.0) check because NaN < 0.0 is false
        assert!(report.is_ok());
    }

    // 9. Inf input in atol -> accepted (SciPy allows)
    #[test]
    fn test_validation_tol_inf_atol() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(f64::INFINITY),
            1,
            RuntimeMode::Strict,
        );
        assert!(report.is_ok());
    }

    // ── validate_first_step tests ────────────────────────────────

    // 10. positive within bounds -> accepted
    #[test]
    fn test_validation_first_step_positive_within_bounds() {
        let result = validate_first_step(0.5, 0.0, 1.0);
        assert_eq!(result.unwrap(), 0.5);
    }

    // 11. zero -> FirstStepMustBePositive error
    #[test]
    fn test_validation_first_step_zero() {
        let err = validate_first_step(0.0, 0.0, 1.0).expect_err("must reject zero");
        assert_eq!(err, IntegrateValidationError::FirstStepMustBePositive);
    }

    // 12. negative -> FirstStepMustBePositive error
    #[test]
    fn test_validation_first_step_negative() {
        let err = validate_first_step(-0.1, 0.0, 1.0).expect_err("must reject negative");
        assert_eq!(err, IntegrateValidationError::FirstStepMustBePositive);
    }

    // 13. exceeds bounds -> FirstStepExceedsBounds error
    #[test]
    fn test_validation_first_step_exceeds_bounds() {
        let err = validate_first_step(2.0, 0.0, 1.0).expect_err("must reject out-of-bounds step");
        assert_eq!(err, IntegrateValidationError::FirstStepExceedsBounds);
    }

    // ── validate_max_step tests ──────────────────────────────────

    // 14. positive -> accepted
    #[test]
    fn test_validation_max_step_positive() {
        assert_eq!(validate_max_step(1.0).unwrap(), 1.0);
    }

    // 15. zero -> MaxStepMustBePositive error
    #[test]
    fn test_validation_max_step_zero() {
        let err = validate_max_step(0.0).expect_err("must reject zero");
        assert_eq!(err, IntegrateValidationError::MaxStepMustBePositive);
    }

    // 16. negative -> MaxStepMustBePositive error
    #[test]
    fn test_validation_max_step_negative() {
        let err = validate_max_step(-1.0).expect_err("must reject negative max step");
        assert_eq!(err, IntegrateValidationError::MaxStepMustBePositive);
    }

    // 17. Inf -> accepted (SciPy allows this as the default)
    #[test]
    fn test_validation_max_step_infinity() {
        assert_eq!(validate_max_step(f64::INFINITY).unwrap(), f64::INFINITY);
    }

    // ── Mode-specific tests ──────────────────────────────────────

    // 18. Strict mode: rtol clamping preserves SciPy semantics
    #[test]
    fn test_validation_tol_strict_clamping_scipy_semantics() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-20),
            ToleranceValue::Scalar(1e-8),
            2,
            RuntimeMode::Strict,
        )
        .expect("strict mode should clamp");
        assert_eq!(report.mode, RuntimeMode::Strict);
        assert!(!report.warnings.is_empty());
        match report.rtol {
            ToleranceValue::Scalar(v) => assert!(v >= MIN_RTOL),
            _ => panic!("expected scalar"),
        }
    }

    // 19. Hardened mode: rtol clamping + finite check
    #[test]
    fn test_validation_tol_hardened_clamping() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-20),
            ToleranceValue::Scalar(1e-8),
            2,
            RuntimeMode::Hardened,
        )
        .expect("hardened mode should clamp");
        assert_eq!(report.mode, RuntimeMode::Hardened);
        assert!(!report.warnings.is_empty());
    }

    // 20. Round-trip: reasonable inputs -> no warnings
    #[test]
    fn test_validation_tol_roundtrip_no_warnings() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Vector(vec![1e-6, 1e-7]),
            2,
            RuntimeMode::Strict,
        )
        .expect("reasonable inputs");
        assert!(report.warnings.is_empty());
        assert_eq!(report.atol, ToleranceValue::Vector(vec![1e-6, 1e-7]));
    }

    // ── Edge cases ───────────────────────────────────────────────

    // 21. Empty system (n=0)
    #[test]
    fn test_validation_tol_empty_system() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(1e-6),
            0,
            RuntimeMode::Strict,
        )
        .expect("empty system should be valid");
        assert!(report.warnings.is_empty());
    }

    // 22. Very large n
    #[test]
    fn test_validation_tol_large_n() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(1e-6),
            10000,
            RuntimeMode::Strict,
        )
        .expect("large n should be valid with scalar atol");
        assert!(report.warnings.is_empty());
    }

    // 23. Extreme small tolerance (1e-300)
    #[test]
    fn test_validation_tol_extreme_small() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-300),
            ToleranceValue::Scalar(1e-300),
            1,
            RuntimeMode::Strict,
        )
        .expect("extreme small should be clamped");
        assert!(!report.warnings.is_empty());
    }

    // 24. Extreme large tolerance (1e300)
    #[test]
    fn test_validation_tol_extreme_large() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e300),
            ToleranceValue::Scalar(1e300),
            1,
            RuntimeMode::Strict,
        )
        .expect("extreme large should be valid");
        assert!(report.warnings.is_empty());
    }

    // 25. First step at exact boundary
    #[test]
    fn test_validation_first_step_exact_boundary() {
        let result = validate_first_step(1.0, 0.0, 1.0);
        assert_eq!(result.unwrap(), 1.0);
    }

    // 26. Backward integration first step
    #[test]
    fn test_validation_first_step_backward() {
        let result = validate_first_step(0.5, 1.0, 0.0);
        assert_eq!(result.unwrap(), 0.5);
    }

    // 27. Vector atol with zero elements
    #[test]
    fn test_validation_tol_vector_zero_element() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Vector(vec![0.0, 1e-6]),
            2,
            RuntimeMode::Strict,
        )
        .expect("zero atol element should be valid");
        assert!(report.warnings.is_empty());
    }
}
