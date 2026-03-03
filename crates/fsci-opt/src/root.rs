#![forbid(unsafe_code)]

use crate::types::{ConvergenceStatus, OptError, RootMethod, RootOptions};

#[derive(Debug, Clone, PartialEq)]
pub struct RootResult {
    pub root: f64,
    pub converged: bool,
    pub status: ConvergenceStatus,
    pub iterations: usize,
    pub function_calls: usize,
    pub method: RootMethod,
    pub message: String,
}

impl RootResult {
    #[must_use]
    pub fn terminal(
        method: RootMethod,
        root: f64,
        converged: bool,
        status: ConvergenceStatus,
        iterations: usize,
        function_calls: usize,
        message: impl Into<String>,
    ) -> Self {
        Self {
            root,
            converged,
            status,
            iterations,
            function_calls,
            method,
            message: message.into(),
        }
    }
}

pub fn root_scalar<F>(
    f: F,
    bracket: Option<(f64, f64)>,
    x0: Option<f64>,
    x1: Option<f64>,
    options: RootOptions,
) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    validate_root_options(options)?;

    let method = if let Some(method) = options.method {
        method
    } else if bracket.is_some() {
        RootMethod::Brentq
    } else if x0.is_some() && x1.is_some() {
        RootMethod::Ridder
    } else {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "unable to select a root solver: provide bracket or explicit method",
            ),
        });
    };

    match method {
        RootMethod::Brentq => {
            let interval = require_bracket(bracket)?;
            brentq(&f, interval, options)
        }
        RootMethod::Brenth => {
            let interval = require_bracket(bracket)?;
            brenth(&f, interval, options)
        }
        RootMethod::Bisect => {
            let interval = require_bracket(bracket)?;
            bisect(&f, interval, options)
        }
        RootMethod::Ridder => {
            let interval = require_bracket(bracket)?;
            ridder(&f, interval, options)
        }
    }
}

pub fn brentq<F>(f: F, bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    validate_root_options(options)?;
    validate_bracket_finite(bracket)?;
    let mut eval = RootEvaluator::new(&f, options);
    let mut state = prepare_bracket(&mut eval, bracket, RootMethod::Brentq)?;
    if state.fa.abs() <= options.xtol {
        return Ok(RootResult::terminal(
            RootMethod::Brentq,
            state.a,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "brentq accepted bracket start as root",
        ));
    }
    if state.fb.abs() <= options.xtol {
        return Ok(RootResult::terminal(
            RootMethod::Brentq,
            state.b,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "brentq accepted bracket end as root",
        ));
    }

    let mut c = state.a;
    let mut fc = state.fa;
    let mut d = state.b - state.a;
    let mut e = d;

    for iter in 1..=options.maxiter {
        if state.fb.abs() <= options.xtol {
            return Ok(RootResult::terminal(
                RootMethod::Brentq,
                state.b,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "brentq converged (|f(x)| <= xtol)",
            ));
        }

        if same_sign(state.fa, state.fb) {
            state.a = c;
            state.fa = fc;
            d = state.b - state.a;
            e = d;
        }

        if state.fa.abs() < state.fb.abs() {
            c = state.b;
            fc = state.fb;
            state.b = state.a;
            state.fb = state.fa;
            state.a = c;
            state.fa = fc;
        }

        let tol = options.xtol + options.rtol * state.b.abs();
        let m = 0.5 * (state.a - state.b);
        if m.abs() <= tol {
            return Ok(RootResult::terminal(
                RootMethod::Brentq,
                state.b,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "brentq converged (interval width <= tol)",
            ));
        }

        if e.abs() >= tol && fc.abs() > state.fb.abs() {
            let s = state.fb / fc;
            let (mut p, mut q) = if (state.a - c).abs() <= f64::EPSILON {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q_val = fc / state.fa;
                let r_val = state.fb / state.fa;
                (
                    s * (2.0 * m * q_val * (q_val - r_val) - (state.b - c) * (r_val - 1.0)),
                    (q_val - 1.0) * (r_val - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            let min1 = 3.0 * m * q.abs() - (tol * q).abs();
            let min2 = (e * q).abs();
            if 2.0 * p < min1.min(min2) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = d;
            }
        } else {
            d = m;
            e = d;
        }

        c = state.b;
        fc = state.fb;
        if d.abs() > tol {
            state.b += d;
        } else if m > 0.0 {
            state.b += tol;
        } else {
            state.b -= tol;
        }
        state.fb = eval.evaluate(state.b)?;
    }

    Ok(RootResult::terminal(
        RootMethod::Brentq,
        state.b,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        eval.function_calls,
        "brentq failed to converge within maxiter",
    ))
}

pub fn brenth<F>(f: F, bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    let mut result = brentq(f, bracket, options)?;
    result.method = RootMethod::Brenth;
    result.message = result.message.replace("brentq", "brenth");
    Ok(result)
}

pub fn bisect<F>(f: F, bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    validate_root_options(options)?;
    validate_bracket_finite(bracket)?;
    let mut eval = RootEvaluator::new(&f, options);
    let mut state = prepare_bracket(&mut eval, bracket, RootMethod::Bisect)?;
    if state.fa.abs() <= options.xtol {
        return Ok(RootResult::terminal(
            RootMethod::Bisect,
            state.a,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "bisection accepted bracket start as root",
        ));
    }
    if state.fb.abs() <= options.xtol {
        return Ok(RootResult::terminal(
            RootMethod::Bisect,
            state.b,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "bisection accepted bracket end as root",
        ));
    }

    for iter in 1..=options.maxiter {
        let mid = 0.5 * (state.a + state.b);
        let f_mid = eval.evaluate(mid)?;
        let interval_tol = options.xtol + options.rtol * mid.abs();
        if f_mid.abs() <= options.xtol || (state.b - state.a).abs() <= interval_tol {
            return Ok(RootResult::terminal(
                RootMethod::Bisect,
                mid,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "bisection converged",
            ));
        }
        if same_sign(state.fa, f_mid) {
            state.a = mid;
            state.fa = f_mid;
        } else {
            state.b = mid;
            state.fb = f_mid;
        }
    }

    let root = 0.5 * (state.a + state.b);
    Ok(RootResult::terminal(
        RootMethod::Bisect,
        root,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        eval.function_calls,
        "bisection failed to converge within maxiter",
    ))
}

pub fn ridder<F>(f: F, bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    validate_root_options(options)?;
    validate_bracket_finite(bracket)?;
    let mut eval = RootEvaluator::new(&f, options);
    let mut state = prepare_bracket(&mut eval, bracket, RootMethod::Ridder)?;
    if state.fa.abs() <= options.xtol {
        return Ok(RootResult::terminal(
            RootMethod::Ridder,
            state.a,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "ridder accepted bracket start as root",
        ));
    }
    if state.fb.abs() <= options.xtol {
        return Ok(RootResult::terminal(
            RootMethod::Ridder,
            state.b,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "ridder accepted bracket end as root",
        ));
    }

    for iter in 1..=options.maxiter {
        let mid = 0.5 * (state.a + state.b);
        let f_mid = eval.evaluate(mid)?;
        if f_mid.abs() <= options.xtol {
            return Ok(RootResult::terminal(
                RootMethod::Ridder,
                mid,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "ridder converged (|f(mid)| <= xtol)",
            ));
        }
        let disc = f_mid * f_mid - state.fa * state.fb;
        if disc <= 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Ridder,
                mid,
                false,
                ConvergenceStatus::PrecisionLoss,
                iter,
                eval.function_calls,
                "ridder encountered non-positive discriminant",
            ));
        }
        let s = disc.sqrt();
        let sign = if (state.fa - state.fb) < 0.0 {
            -1.0
        } else {
            1.0
        };
        let x_new = mid + (mid - state.a) * sign * f_mid / s;
        let f_new = eval.evaluate(x_new)?;

        let interval_tol = options.xtol + options.rtol * x_new.abs();
        if f_new.abs() <= options.xtol || (state.b - state.a).abs() <= interval_tol {
            return Ok(RootResult::terminal(
                RootMethod::Ridder,
                x_new,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "ridder converged",
            ));
        }

        if !same_sign(f_mid, f_new) {
            state.a = mid;
            state.fa = f_mid;
            state.b = x_new;
            state.fb = f_new;
        } else if !same_sign(state.fa, f_new) {
            state.b = x_new;
            state.fb = f_new;
        } else {
            state.a = x_new;
            state.fa = f_new;
        }
    }

    let root = 0.5 * (state.a + state.b);
    Ok(RootResult::terminal(
        RootMethod::Ridder,
        root,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        eval.function_calls,
        "ridder failed to converge within maxiter",
    ))
}

fn require_bracket(bracket: Option<(f64, f64)>) -> Result<(f64, f64), OptError> {
    bracket.ok_or_else(|| OptError::InvalidArgument {
        detail: String::from("bracket is required for bracketing root methods"),
    })
}

fn validate_root_options(options: RootOptions) -> Result<(), OptError> {
    if options.xtol <= 0.0 || !options.xtol.is_finite() {
        return Err(OptError::InvalidArgument {
            detail: String::from("xtol must be finite and > 0"),
        });
    }
    if options.rtol < 0.0 || !options.rtol.is_finite() {
        return Err(OptError::InvalidArgument {
            detail: String::from("rtol must be finite and >= 0"),
        });
    }
    if options.maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxiter must be >= 1"),
        });
    }
    Ok(())
}

fn validate_bracket_finite(bracket: (f64, f64)) -> Result<(), OptError> {
    if !bracket.0.is_finite() || !bracket.1.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("bracket endpoints must be finite"),
        });
    }
    if bracket.0 >= bracket.1 {
        return Err(OptError::InvalidBounds {
            detail: String::from("bracket must satisfy a < b"),
        });
    }
    Ok(())
}

fn same_sign(lhs: f64, rhs: f64) -> bool {
    lhs.is_sign_positive() == rhs.is_sign_positive()
}

#[derive(Debug, Clone, Copy)]
struct BracketState {
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
}

struct RootEvaluator<'a, F>
where
    F: Fn(f64) -> f64,
{
    f: &'a F,
    options: RootOptions,
    function_calls: usize,
}

impl<'a, F> RootEvaluator<'a, F>
where
    F: Fn(f64) -> f64,
{
    fn new(f: &'a F, options: RootOptions) -> Self {
        Self {
            f,
            options,
            function_calls: 0,
        }
    }

    fn evaluate(&mut self, x: f64) -> Result<f64, OptError> {
        if !x.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("root solver evaluated a non-finite point"),
            });
        }
        let value = (self.f)(x);
        self.function_calls += 1;
        if !value.is_finite() {
            return match self.options.mode {
                fsci_runtime::RuntimeMode::Strict => Err(OptError::InvalidArgument {
                    detail: String::from("root function returned non-finite value"),
                }),
                fsci_runtime::RuntimeMode::Hardened => Err(OptError::NonFiniteInput {
                    detail: String::from("hardened mode rejects non-finite function values"),
                }),
            };
        }
        Ok(value)
    }
}

fn prepare_bracket<F>(
    eval: &mut RootEvaluator<'_, F>,
    bracket: (f64, f64),
    _method: RootMethod,
) -> Result<BracketState, OptError>
where
    F: Fn(f64) -> f64,
{
    let fa = eval.evaluate(bracket.0)?;
    let fb = eval.evaluate(bracket.1)?;
    if same_sign(fa, fb) && fa.abs() > eval.options.xtol && fb.abs() > eval.options.xtol {
        return Err(OptError::SignChangeRequired {
            detail: String::from("bracketing methods require f(a) and f(b) to have opposite signs"),
        });
    }

    Ok(BracketState {
        a: bracket.0,
        b: bracket.1,
        fa,
        fb,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsci_runtime::RuntimeMode;
    use proptest::prelude::*;
    use serde::Serialize;

    use crate::{
        ConvergenceStatus, RootMethod, RootOptions, bisect, brenth, brentq, ridder, root_scalar,
    };

    #[derive(Debug, Serialize)]
    struct TestLogEntry<'a> {
        test_id: &'a str,
        optimizer: &'a str,
        problem: &'a str,
        n_dim: usize,
        mode: &'a str,
        converged: bool,
        nfev: usize,
        final_f: Option<f64>,
        seed: u64,
        timestamp_ms: u64,
    }

    fn test_log_sink() -> &'static Mutex<Vec<String>> {
        static TEST_LOGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
        TEST_LOGS.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn now_unix_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64)
    }

    fn mode_name(mode: RuntimeMode) -> &'static str {
        match mode {
            RuntimeMode::Strict => "strict",
            RuntimeMode::Hardened => "hardened",
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_test_log(
        test_id: &str,
        optimizer: &str,
        problem: &str,
        mode: RuntimeMode,
        converged: bool,
        nfev: usize,
        final_f: Option<f64>,
        seed: u64,
    ) {
        let entry = TestLogEntry {
            test_id,
            optimizer,
            problem,
            n_dim: 1,
            mode: mode_name(mode),
            converged,
            nfev,
            final_f,
            seed,
            timestamp_ms: now_unix_ms(),
        };
        let payload = serde_json::to_string(&entry).expect("serialize test log");
        let parsed: serde_json::Value =
            serde_json::from_str(&payload).expect("re-parse serialized log payload");
        assert!(parsed.get("test_id").is_some());
        assert!(parsed.get("optimizer").is_some());
        assert!(parsed.get("timestamp_ms").is_some());
        test_log_sink().lock().expect("log sink lock").push(payload);
    }

    fn cubic(x: f64) -> f64 {
        x * x * x - 2.0
    }

    fn monotone_linear(x: f64) -> f64 {
        x - 3.0
    }

    fn discontinuous_step(x: f64) -> f64 {
        if x < 0.5 { -1.0 } else { 1.0 }
    }

    #[test]
    fn bisect_finds_root() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            xtol: 1.0e-10,
            rtol: 1.0e-10,
            maxiter: 200,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = bisect(cubic, (0.0, 2.0), options).expect("bisect executes");
        assert!(result.converged, "{}", result.message);
        assert_eq!(result.status, ConvergenceStatus::Success);
        assert!((result.root - 2f64.cbrt()).abs() < 1.0e-8);
        push_test_log(
            "bisect-monotone-root",
            "bisect",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            301,
        );
    }

    #[test]
    fn brentq_finds_root() {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            xtol: 1.0e-12,
            rtol: 1.0e-12,
            maxiter: 100,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = brentq(cubic, (0.0, 2.0), options).expect("brentq executes");
        assert!(result.converged, "{}", result.message);
        assert!((result.root - 2f64.cbrt()).abs() < 1.0e-10);
        push_test_log(
            "brentq-polynomial-root",
            "brentq",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            302,
        );
    }

    #[test]
    fn brentq_accepts_root_at_bracket_endpoint() {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            xtol: 1.0e-12,
            rtol: 1.0e-12,
            maxiter: 100,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result =
            brentq(|x| x - 1.0, (1.0, 2.0), options).expect("brentq should accept endpoint root");
        assert!(result.converged);
        assert!((result.root - 1.0).abs() <= f64::EPSILON);
        push_test_log(
            "brentq-endpoint-root",
            "brentq",
            "linear",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(result.root - 1.0),
            303,
        );
    }

    #[test]
    fn brentq_rejects_missing_sign_change() {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let err = brentq(cubic, (2.0, 3.0), options).expect_err("must fail without sign change");
        assert!(matches!(err, crate::OptError::SignChangeRequired { .. }));
        push_test_log(
            "brentq-sign-change-error",
            "brentq",
            "cubic",
            RuntimeMode::Strict,
            false,
            0,
            None,
            304,
        );
    }

    #[test]
    fn brenth_matches_brentq_on_same_problem() {
        let options = RootOptions {
            method: Some(RootMethod::Brenth),
            xtol: 1.0e-12,
            rtol: 1.0e-12,
            maxiter: 150,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let brentq_result = brentq(cubic, (0.0, 2.0), options).expect("brentq executes");
        let brenth_result = brenth(cubic, (0.0, 2.0), options).expect("brenth executes");
        assert!((brenth_result.root - brentq_result.root).abs() < 1.0e-10);
        push_test_log(
            "brenth-matches-brentq",
            "brenth",
            "cubic",
            RuntimeMode::Strict,
            brenth_result.converged,
            brenth_result.function_calls,
            Some(cubic(brenth_result.root)),
            305,
        );
    }

    #[test]
    fn root_scalar_uses_default_method_for_bracket() {
        let options = RootOptions {
            method: None,
            xtol: 1.0e-12,
            rtol: 1.0e-12,
            maxiter: 120,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = root_scalar(cubic, Some((0.0, 2.0)), None, None, options)
            .expect("root_scalar executes");
        assert!(result.converged);
        assert_eq!(result.method, RootMethod::Brentq);
        push_test_log(
            "root-scalar-default-method",
            "root_scalar",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            306,
        );
    }

    #[test]
    fn sign_change_validation_is_enforced() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let err = bisect(cubic, (2.0, 3.0), options).expect_err("must fail without sign change");
        assert!(matches!(err, crate::OptError::SignChangeRequired { .. }));
        push_test_log(
            "bisect-sign-change-error",
            "bisect",
            "cubic",
            RuntimeMode::Strict,
            false,
            0,
            None,
            307,
        );
    }

    #[test]
    fn bisect_respects_tolerance_convergence() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            xtol: 1.0e-6,
            rtol: 1.0e-6,
            maxiter: 80,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = bisect(monotone_linear, (0.0, 5.0), options).expect("bisect executes");
        assert!(result.converged);
        assert!((result.root - 3.0).abs() < 5.0e-6);
        push_test_log(
            "bisect-tolerance-convergence",
            "bisect",
            "linear",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(monotone_linear(result.root)),
            308,
        );
    }

    #[test]
    fn brentq_handles_discontinuous_function_with_defined_output() {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            xtol: 1.0e-8,
            rtol: 1.0e-8,
            maxiter: 120,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = brentq(discontinuous_step, (0.0, 1.0), options).expect("brentq executes");
        assert!(result.root.is_finite());
        assert!((0.0..=1.0).contains(&result.root));
        push_test_log(
            "brentq-discontinuous-defined",
            "brentq",
            "step-discontinuity",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(discontinuous_step(result.root)),
            309,
        );
    }

    #[test]
    fn bracketing_methods_agree_on_root_within_tolerance() {
        let options = RootOptions {
            xtol: 1.0e-10,
            rtol: 1.0e-10,
            maxiter: 200,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let bis = bisect(
            cubic,
            (0.0, 2.0),
            RootOptions {
                method: Some(RootMethod::Bisect),
                ..options
            },
        )
        .expect("bisect executes");
        let brq = brentq(
            cubic,
            (0.0, 2.0),
            RootOptions {
                method: Some(RootMethod::Brentq),
                ..options
            },
        )
        .expect("brentq executes");
        let rid = ridder(
            cubic,
            (0.0, 2.0),
            RootOptions {
                method: Some(RootMethod::Ridder),
                ..options
            },
        )
        .expect("ridder executes");
        assert!((bis.root - brq.root).abs() < 1.0e-7);
        assert!((rid.root - brq.root).abs() < 1.0e-7);
        push_test_log(
            "root-method-agreement",
            "all",
            "cubic",
            RuntimeMode::Strict,
            true,
            bis.function_calls + brq.function_calls + rid.function_calls,
            Some(cubic(brq.root)),
            310,
        );
    }

    #[test]
    fn tight_bracket_converges_quickly() {
        let root = 2f64.cbrt();
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            xtol: 1.0e-14,
            rtol: 1.0e-14,
            maxiter: 100,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result =
            brentq(cubic, (root - 1.0e-4, root + 1.0e-4), options).expect("brentq executes");
        assert!(result.converged);
        assert!(result.iterations <= 10);
        push_test_log(
            "tight-bracket-fast",
            "brentq",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            311,
        );
    }

    #[test]
    fn wide_bracket_still_converges() {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            xtol: 1.0e-10,
            rtol: 1.0e-10,
            maxiter: 250,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = brentq(cubic, (-1000.0, 1000.0), options).expect("brentq executes");
        assert!(result.converged);
        assert!((result.root - 2f64.cbrt()).abs() < 1.0e-7);
        push_test_log(
            "wide-bracket-converges",
            "brentq",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            312,
        );
    }

    #[test]
    fn ridder_finds_root_for_cubic() {
        let options = RootOptions {
            method: Some(RootMethod::Ridder),
            xtol: 1.0e-10,
            rtol: 1.0e-10,
            maxiter: 120,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = ridder(cubic, (0.0, 2.0), options).expect("ridder executes");
        assert!(result.converged);
        assert!((result.root - 2f64.cbrt()).abs() < 1.0e-7);
        push_test_log(
            "ridder-cubic-root",
            "ridder",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            313,
        );
    }

    #[test]
    fn root_scalar_without_bracket_returns_error_for_ridder_selection() {
        let options = RootOptions {
            method: None,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let err = root_scalar(cubic, None, Some(0.0), Some(1.0), options)
            .expect_err("ridder path still requires a bracket");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
        push_test_log(
            "root-scalar-missing-bracket",
            "root_scalar",
            "cubic",
            RuntimeMode::Strict,
            false,
            0,
            None,
            314,
        );
    }

    #[test]
    fn root_scalar_without_inputs_reports_selection_error() {
        let options = RootOptions {
            method: None,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let err =
            root_scalar(cubic, None, None, None, options).expect_err("must reject missing inputs");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
        push_test_log(
            "root-scalar-selection-error",
            "root_scalar",
            "cubic",
            RuntimeMode::Strict,
            false,
            0,
            None,
            315,
        );
    }

    #[test]
    fn hardened_mode_rejects_non_finite_function_values() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            mode: RuntimeMode::Hardened,
            ..RootOptions::default()
        };
        let err = bisect(
            |x| if x > 0.8 { f64::NAN } else { x - 0.2 },
            (0.1, 0.9),
            options,
        )
        .expect_err("hardened mode should reject non-finite values");
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));
        push_test_log(
            "root-hardened-non-finite",
            "bisect",
            "nan-branch",
            RuntimeMode::Hardened,
            false,
            0,
            None,
            316,
        );
    }

    #[test]
    fn strict_mode_non_finite_function_values_map_to_invalid_argument() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let err = bisect(
            |x| if x > 0.8 { f64::NAN } else { x - 0.2 },
            (0.1, 0.9),
            options,
        )
        .expect_err("strict mode should map non-finite values to invalid argument");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
        push_test_log(
            "root-strict-non-finite",
            "bisect",
            "nan-branch",
            RuntimeMode::Strict,
            false,
            0,
            None,
            317,
        );
    }

    #[test]
    fn bisect_reports_max_iterations_when_budget_is_too_low() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            xtol: 1.0e-18,
            rtol: 1.0e-18,
            maxiter: 1,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = bisect(|x| x - 0.3, (0.0, 1.0), options).expect("bisect executes");
        assert!(!result.converged);
        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        push_test_log(
            "bisect-maxiter-budget",
            "bisect",
            "linear",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(result.root - 0.3),
            318,
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 500,
            failure_persistence: None,
            .. ProptestConfig::default()
        })]

        #[test]
        fn property_brentq_returns_small_residual_for_valid_bracket(
            target in -50.0f64..50.0f64,
        ) {
            let f = |x: f64| x - target;
            let options = RootOptions {
                method: Some(RootMethod::Brentq),
                xtol: 1.0e-10,
                rtol: 1.0e-10,
                maxiter: 80,
                mode: RuntimeMode::Strict,
                ..RootOptions::default()
            };
            let bracket = (target - 1.0, target + 1.0);
            let result = brentq(f, bracket, options).expect("brentq executes");
            prop_assert!(result.converged);
            prop_assert!(f(result.root).abs() <= 2.0 * options.xtol);
            push_test_log(
                "property-brentq-small-residual",
                "brentq",
                "linear-shift",
                RuntimeMode::Strict,
                result.converged,
                result.function_calls,
                Some(f(result.root)),
                401,
            );
        }
    }
}
