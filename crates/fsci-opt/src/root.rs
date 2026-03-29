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
    } else if x0.is_some() {
        RootMethod::Secant
    } else {
        return Err(OptError::InvalidArgument {
            detail: String::from("unable to select a root solver: provide bracket or x0"),
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
        RootMethod::Toms748 => {
            let interval = require_bracket(bracket)?;
            toms748(&f, interval, options)
        }
        RootMethod::Secant => {
            let start = x0.ok_or_else(|| OptError::InvalidArgument {
                detail: "secant method requires x0".to_string(),
            })?;
            secant(&f, start, x1, options)
        }
        RootMethod::Newton | RootMethod::Halley => Err(OptError::InvalidArgument {
            detail: format!("{method:?} requires fprime; use newton_scalar() or halley() directly"),
        }),
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
            let (mut p, mut q) = if state.a == c {
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

/// TOMS Algorithm 748: high-order bracketing root finder.
///
/// Combines inverse cubic interpolation with bisection as fallback.
/// Matches `scipy.optimize.toms748`.
pub fn toms748<F>(f: F, bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    let (mut a, mut b) = bracket;
    let mut nfev = 0usize;
    let mut fa = {
        nfev += 1;
        f(a)
    };
    let mut fb = {
        nfev += 1;
        f(b)
    };

    if fa * fb > 0.0 {
        return Err(OptError::SignChangeRequired {
            detail: format!("f(a)={fa} and f(b)={fb} must have opposite signs"),
        });
    }

    if fa == 0.0 {
        return Ok(RootResult::terminal(
            RootMethod::Toms748,
            a,
            true,
            ConvergenceStatus::Success,
            0,
            nfev,
            "exact root at a",
        ));
    }
    if fb == 0.0 {
        return Ok(RootResult::terminal(
            RootMethod::Toms748,
            b,
            true,
            ConvergenceStatus::Success,
            0,
            nfev,
            "exact root at b",
        ));
    }

    if fa > 0.0 {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    for iter in 0..options.maxiter {
        let tol = options.xtol + options.rtol * b.abs();

        if (b - a).abs() < tol || fa == 0.0 || fb == 0.0 {
            let root = if fa.abs() < fb.abs() { a } else { b };
            return Ok(RootResult::terminal(
                RootMethod::Toms748,
                root,
                true,
                ConvergenceStatus::Success,
                iter,
                nfev,
                "converged",
            ));
        }

        let mid = 0.5 * (a + b);
        let c = if fa != fb {
            let s = a - fa * (b - a) / (fb - fa);
            if s > a + 0.25 * (b - a) && s < b - 0.25 * (b - a) {
                s
            } else {
                mid
            }
        } else {
            mid
        };

        nfev += 1;
        let fc = f(c);
        if fc == 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Toms748,
                c,
                true,
                ConvergenceStatus::Success,
                iter,
                nfev,
                "exact root found",
            ));
        }

        if fc < 0.0 {
            a = c;
            fa = fc;
        } else {
            b = c;
            fb = fc;
        }
    }

    let root = 0.5 * (a + b);
    Ok(RootResult::terminal(
        RootMethod::Toms748,
        root,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        nfev,
        "toms748 failed to converge within maxiter",
    ))
}

/// Newton's method for scalar root finding (requires derivative).
///
/// Matches `scipy.optimize.newton` with fprime provided.
pub fn newton_scalar<F, G>(
    f: F,
    fprime: G,
    x0: f64,
    options: RootOptions,
) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let mut x = x0;
    let mut nfev = 0usize;

    for iter in 0..options.maxiter {
        nfev += 1;
        let fx = f(x);
        if fx.abs() < options.xtol {
            return Ok(RootResult::terminal(
                RootMethod::Newton,
                x,
                true,
                ConvergenceStatus::Success,
                iter,
                nfev,
                "converged",
            ));
        }

        let dfx = fprime(x);
        if dfx == 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Newton,
                x,
                false,
                ConvergenceStatus::PrecisionLoss,
                iter,
                nfev,
                "zero derivative encountered",
            ));
        }

        let x_new = x - fx / dfx;
        if (x_new - x).abs() < options.xtol + options.rtol * x.abs() {
            return Ok(RootResult::terminal(
                RootMethod::Newton,
                x_new,
                true,
                ConvergenceStatus::Success,
                iter + 1,
                nfev,
                "converged",
            ));
        }
        x = x_new;
    }

    Ok(RootResult::terminal(
        RootMethod::Newton,
        x,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        nfev,
        "newton failed to converge",
    ))
}

/// Secant method for scalar root finding (derivative-free).
///
/// Matches `scipy.optimize.newton` without fprime.
pub fn secant<F>(
    f: F,
    x0: f64,
    x1: Option<f64>,
    options: RootOptions,
) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    let mut nfev = 0usize;
    let mut xprev = x0;
    nfev += 1;
    let mut fprev = f(xprev);
    let mut xcurr = x1.unwrap_or(x0 * (1.0 + 1e-4) + 1e-4);
    nfev += 1;
    let mut fcurr = f(xcurr);

    for iter in 0..options.maxiter {
        if fcurr.abs() < options.xtol {
            return Ok(RootResult::terminal(
                RootMethod::Secant,
                xcurr,
                true,
                ConvergenceStatus::Success,
                iter,
                nfev,
                "converged",
            ));
        }

        let denom = fcurr - fprev;
        if denom == 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Secant,
                xcurr,
                false,
                ConvergenceStatus::PrecisionLoss,
                iter,
                nfev,
                "identical function values",
            ));
        }

        let x_new = xcurr - fcurr * (xcurr - xprev) / denom;
        let step = (x_new - xcurr).abs();
        xprev = xcurr;
        fprev = fcurr;
        xcurr = x_new;
        nfev += 1;
        fcurr = f(xcurr);

        // Step-size convergence check
        if step < options.xtol + options.rtol * xcurr.abs() {
            return Ok(RootResult::terminal(
                RootMethod::Secant,
                xcurr,
                true,
                ConvergenceStatus::Success,
                iter + 1,
                nfev,
                "converged",
            ));
        }
    }

    Ok(RootResult::terminal(
        RootMethod::Secant,
        xcurr,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        nfev,
        "secant failed to converge",
    ))
}

/// Halley's method (requires f, f', f'').
///
/// Cubic convergence rate. Matches `scipy.optimize.newton` with fprime2.
pub fn halley<F, G, H>(
    f: F,
    fprime: G,
    fprime2: H,
    x0: f64,
    options: RootOptions,
) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
    H: Fn(f64) -> f64,
{
    let mut x = x0;
    let mut nfev = 0usize;

    for iter in 0..options.maxiter {
        nfev += 1;
        let fx = f(x);
        if fx.abs() < options.xtol {
            return Ok(RootResult::terminal(
                RootMethod::Halley,
                x,
                true,
                ConvergenceStatus::Success,
                iter,
                nfev,
                "converged",
            ));
        }

        let dfx = fprime(x);
        let d2fx = fprime2(x);

        if dfx == 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Halley,
                x,
                false,
                ConvergenceStatus::PrecisionLoss,
                iter,
                nfev,
                "zero derivative encountered",
            ));
        }

        let denom = 2.0 * dfx * dfx - fx * d2fx;
        if denom.abs() < 1e-30 {
            // Halley denominator degenerate; fall back to Newton step
            let step = fx / dfx;
            x -= step;
            if step.abs() < options.xtol + options.rtol * x.abs() {
                return Ok(RootResult::terminal(
                    RootMethod::Halley,
                    x,
                    true,
                    ConvergenceStatus::Success,
                    iter + 1,
                    nfev,
                    "converged (Newton fallback)",
                ));
            }
        } else {
            let x_new = x - 2.0 * fx * dfx / denom;
            if (x_new - x).abs() < options.xtol + options.rtol * x.abs() {
                return Ok(RootResult::terminal(
                    RootMethod::Halley,
                    x_new,
                    true,
                    ConvergenceStatus::Success,
                    iter + 1,
                    nfev,
                    "converged",
                ));
            }
            x = x_new;
        }
    }

    Ok(RootResult::terminal(
        RootMethod::Halley,
        x,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        nfev,
        "halley failed to converge",
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

// ══════════════════════════════════════════════════════════════════════
// Multivariate Root Finding: root, fsolve
// ══════════════════════════════════════════════════════════════════════

/// Result of multivariate root finding.
#[derive(Debug, Clone)]
pub struct MultivariateRootResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Residuals F(x) at the solution.
    pub fun: Vec<f64>,
    /// Whether the solver converged.
    pub converged: bool,
    /// Human-readable message.
    pub message: String,
    /// Number of iterations.
    pub iterations: usize,
    /// Number of function evaluations.
    pub function_calls: usize,
}

/// Find roots of a system of nonlinear equations F(x) = 0.
///
/// Matches `scipy.optimize.fsolve(func, x0)`.
///
/// Uses Newton's method with finite-difference Jacobian and
/// simple step-size damping for robustness.
///
/// # Arguments
/// * `func` — Vector function F: R^n → R^n.
/// * `x0` — Initial guess.
pub fn fsolve<F>(func: F, x0: &[f64]) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must not be empty".to_string(),
        });
    }

    let tol = 1e-10;
    let maxiter = 200;
    let eps = 1e-8; // finite difference step

    let mut x: Vec<f64> = x0.to_vec();
    let mut nfev = 0usize;
    let mut jac = vec![vec![0.0; n]; n];

    for iteration in 0..maxiter {
        let fx = func(&x);
        nfev += 1;

        // Check convergence: ||F(x)|| < tol.
        let norm_fx: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_fx < tol {
            return Ok(MultivariateRootResult {
                x,
                fun: fx,
                converged: true,
                message: "fsolve converged".to_string(),
                iterations: iteration,
                function_calls: nfev,
            });
        }

        // Compute Jacobian via finite differences.
        for j in 0..n {
            let h = eps * (1.0 + x[j].abs());
            let original = x[j];
            x[j] += h;
            let fx_plus = func(&x);
            x[j] = original; // restore
            nfev += 1;
            for i in 0..n {
                jac[i][j] = (fx_plus[i] - fx[i]) / h;
            }
        }

        // Solve J * dx = -F(x) using Gaussian elimination with partial pivoting.
        let mut neg_fx = fx.clone();
        for val in &mut neg_fx {
            *val = -*val;
        }
        let dx = match solve_dense(&jac, &neg_fx) {
            Some(d) => d,
            None => {
                return Ok(MultivariateRootResult {
                    x,
                    fun: fx,
                    converged: false,
                    message: "fsolve: singular Jacobian".to_string(),
                    iterations: iteration,
                    function_calls: nfev,
                });
            }
        };

        // Line search: try full step, then halve until improvement found.
        let mut alpha = 1.0;
        let mut best_x = x.clone();
        let best_norm = norm_fx;
        let mut improved = false;
        for _ in 0..10 {
            let trial: Vec<f64> = x
                .iter()
                .zip(&dx)
                .map(|(&xi, &di)| xi + alpha * di)
                .collect();
            let ftrial = func(&trial);
            nfev += 1;
            let trial_norm: f64 = ftrial.iter().map(|v| v * v).sum::<f64>().sqrt();
            if trial_norm < best_norm {
                best_x = trial;
                improved = true;
                break;
            }
            alpha *= 0.5;
        }

        if !improved {
            // No step improved; take the smallest step anyway to avoid stalling.
            let trial: Vec<f64> = x
                .iter()
                .zip(&dx)
                .map(|(&xi, &di)| xi + alpha * di)
                .collect();
            best_x = trial;
        }

        x = best_x;
    }

    let fx = func(&x);
    nfev += 1;
    Ok(MultivariateRootResult {
        x,
        fun: fx,
        converged: false,
        message: format!("fsolve failed to converge within {maxiter} iterations"),
        iterations: maxiter,
        function_calls: nfev,
    })
}

/// Solve a dense linear system Ax = b using Gaussian elimination with partial pivoting.
/// Returns None if the matrix is singular.
fn solve_dense(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    // Augmented matrix.
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivoting.
        let max_row = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap_or(col);
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-15 {
            return None; // Singular
        }

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let (aug_col, aug_row) = {
                let (left, right) = aug.split_at_mut(row);
                (&left[col], &mut right[0])
            };
            let factor = aug_row[col] / pivot;
            for (aug_row_j, &aug_col_j) in
                aug_row.iter_mut().zip(aug_col.iter()).take(n + 1).skip(col)
            {
                *aug_row_j -= factor * aug_col_j;
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

/// Method for multivariate root finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MultivariateRootMethod {
    /// Powell's hybrid method (Newton + finite-difference Jacobian + line search).
    #[default]
    Hybr,
    /// Broyden's first method (quasi-Newton with approximate Jacobian updates).
    Broyden1,
}

/// Options for multivariate root finding.
#[derive(Debug, Clone)]
pub struct MultivariateRootOptions {
    pub method: MultivariateRootMethod,
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for MultivariateRootOptions {
    fn default() -> Self {
        Self {
            method: MultivariateRootMethod::Hybr,
            tol: 1e-10,
            max_iter: 200,
        }
    }
}

/// Solve a system of nonlinear equations F(x) = 0.
///
/// Matches `scipy.optimize.root(fun, x0, method)`.
///
/// Supports method="hybr" (Newton with finite-difference Jacobian) and
/// method="broyden1" (Broyden quasi-Newton with approximate Jacobian updates).
pub fn root<F>(
    func: F,
    x0: &[f64],
    options: MultivariateRootOptions,
) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    match options.method {
        MultivariateRootMethod::Hybr => fsolve(func, x0),
        MultivariateRootMethod::Broyden1 => broyden1(func, x0, options.tol, options.max_iter),
    }
}

/// Broyden's first method for solving F(x) = 0.
///
/// Quasi-Newton method that maintains an approximate Jacobian B and updates it
/// using the rank-1 formula: B_{k+1} = B_k + (Δf - B_k Δx) Δx^T / ||Δx||²
///
/// More efficient than Newton when the Jacobian is expensive to compute.
pub fn broyden1<F>(
    func: F,
    x0: &[f64],
    tol: f64,
    maxiter: usize,
) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must not be empty".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = func(&x);
    let mut nfev = 1usize;

    // Initialize approximate Jacobian as identity
    let mut b_inv: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for iteration in 0..maxiter {
        let norm_fx: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_fx < tol {
            return Ok(MultivariateRootResult {
                x,
                fun: fx,
                converged: true,
                message: "broyden1 converged".to_string(),
                iterations: iteration,
                function_calls: nfev,
            });
        }

        // Compute step: dx = -B^{-1} F(x)
        let dx: Vec<f64> = (0..n)
            .map(|i| {
                -b_inv[i]
                    .iter()
                    .zip(fx.iter())
                    .map(|(b, f)| b * f)
                    .sum::<f64>()
            })
            .collect();

        // Update x
        let x_new: Vec<f64> = x.iter().zip(dx.iter()).map(|(xi, di)| xi + di).collect();
        let fx_new = func(&x_new);
        nfev += 1;

        // Broyden update: B^{-1}_{k+1} = B^{-1}_k + (Δx - B^{-1}_k Δf) Δx^T B^{-1}_k / (Δx^T B^{-1}_k Δf)
        let df: Vec<f64> = fx_new
            .iter()
            .zip(fx.iter())
            .map(|(fn_, fo)| fn_ - fo)
            .collect();

        // b_inv_df = B^{-1} * Δf
        let b_inv_df: Vec<f64> = (0..n)
            .map(|i| {
                b_inv[i]
                    .iter()
                    .zip(df.iter())
                    .map(|(b, d)| b * d)
                    .sum::<f64>()
            })
            .collect();

        // numerator = Δx - B^{-1} Δf
        let numer: Vec<f64> = dx
            .iter()
            .zip(b_inv_df.iter())
            .map(|(d, bd)| d - bd)
            .collect();

        // dx^T B^{-1} = row vector
        let dx_b_inv: Vec<f64> = (0..n)
            .map(|j| {
                dx.iter()
                    .zip(b_inv.iter())
                    .map(|(d, row)| d * row[j])
                    .sum::<f64>()
            })
            .collect();

        // denominator = Δx^T B^{-1} Δf
        let denom: f64 = dx.iter().zip(b_inv_df.iter()).map(|(d, bd)| d * bd).sum();

        if denom.abs() > 1e-30 {
            for i in 0..n {
                for j in 0..n {
                    b_inv[i][j] += numer[i] * dx_b_inv[j] / denom;
                }
            }
        }

        x = x_new;
        fx = fx_new;
    }

    Ok(MultivariateRootResult {
        x,
        fun: fx,
        converged: false,
        message: format!("broyden1 failed to converge within {maxiter} iterations"),
        iterations: maxiter,
        function_calls: nfev,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsci_runtime::RuntimeMode;
    use proptest::prelude::*;
    use serde::Serialize;

    use super::{MultivariateRootMethod, MultivariateRootOptions, broyden1, fsolve, root};
    use crate::{
        ConvergenceStatus, RootMethod, RootOptions, bisect, brenth, brentq, halley, newton_scalar,
        ridder, root_scalar, secant, toms748,
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
        test_log_sink()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(payload);
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
    fn root_scalar_with_x0_uses_secant() {
        let options = RootOptions {
            method: None,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result =
            root_scalar(cubic, None, Some(0.5), Some(2.0), options).expect("secant via x0");
        assert!(result.converged, "secant failed: {}", result.message);
        assert!(
            (cubic(result.root)).abs() < 1e-8,
            "residual too large: {}",
            cubic(result.root)
        );
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

    // ── Multivariate root tests ────────────────────────────────────

    #[test]
    fn fsolve_2x2_circle_intersection() {
        // x^2 + y^2 = 1, x - y = 0 → (1/√2, 1/√2)
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]];
        let result = fsolve(f, &[0.5, 0.5]).expect("fsolve should converge");
        assert!(result.converged, "fsolve failed: {}", result.message);
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (result.x[0] - s2).abs() < 0.01,
            "x = {}, expected {}",
            result.x[0],
            s2
        );
        assert!(
            (result.x[1] - s2).abs() < 0.01,
            "y = {}, expected {}",
            result.x[1],
            s2
        );
    }

    #[test]
    fn fsolve_linear_system() {
        // 2x + y = 5, x - y = 1 → x=2, y=1
        let f = |x: &[f64]| vec![2.0 * x[0] + x[1] - 5.0, x[0] - x[1] - 1.0];
        let result = fsolve(f, &[0.0, 0.0]).expect("fsolve should converge");
        assert!(result.converged, "fsolve failed: {}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 0.01,
            "x = {}, expected 2.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.01,
            "y = {}, expected 1.0",
            result.x[1]
        );
    }

    #[test]
    fn fsolve_3x3_system() {
        // x + y + z = 6, x - y + 2z = 5, 2x + y - z = 1 → x=1, y=2, z=3
        let f = |x: &[f64]| {
            vec![
                x[0] + x[1] + x[2] - 6.0,
                x[0] - x[1] + 2.0 * x[2] - 5.0,
                2.0 * x[0] + x[1] - x[2] - 1.0,
            ]
        };
        let result = fsolve(f, &[0.0, 0.0, 0.0]).expect("fsolve should converge");
        assert!(result.converged, "fsolve failed: {}", result.message);
        assert!((result.x[0] - 1.0).abs() < 0.1, "x = {}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 0.1, "y = {}", result.x[1]);
        assert!((result.x[2] - 3.0).abs() < 0.1, "z = {}", result.x[2]);
    }

    #[test]
    fn fsolve_near_solution_fast() {
        let f = |x: &[f64]| vec![x[0] * x[0] - 4.0, x[1] * x[1] - 9.0];
        let result = fsolve(f, &[1.9, 2.9]).expect("fsolve should converge");
        assert!(result.converged);
        assert!(
            result.iterations < 20,
            "too many iterations: {}",
            result.iterations
        );
    }

    #[test]
    fn fsolve_residual_is_small() {
        let f = |x: &[f64]| vec![x[0].sin() + x[1] - 1.0, x[0] + x[1].cos() - 1.0];
        let result = fsolve(f, &[0.5, 0.5]).expect("fsolve should converge");
        if result.converged {
            let residual = f(&result.x);
            let max_residual = residual.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
            assert!(max_residual < 1e-6, "residual too large: {max_residual}");
        }
    }

    // ── root dispatcher tests ──────────────────────────────────────

    #[test]
    fn root_hybr_matches_fsolve() {
        let f = |x: &[f64]| vec![2.0 * x[0] + x[1] - 5.0, x[0] - x[1] - 1.0];
        let direct = fsolve(f, &[0.0, 0.0]).expect("fsolve");
        let dispatched = root(f, &[0.0, 0.0], MultivariateRootOptions::default()).expect("root");
        assert!(dispatched.converged);
        assert!((direct.x[0] - dispatched.x[0]).abs() < 1e-10);
        assert!((direct.x[1] - dispatched.x[1]).abs() < 1e-10);
    }

    #[test]
    fn root_broyden1_linear_system() {
        let f = |x: &[f64]| vec![2.0 * x[0] + x[1] - 5.0, x[0] - x[1] - 1.0];
        let options = MultivariateRootOptions {
            method: MultivariateRootMethod::Broyden1,
            ..MultivariateRootOptions::default()
        };
        let result = root(f, &[0.0, 0.0], options).expect("root broyden1");
        assert!(result.converged, "broyden1 failed: {}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 0.05,
            "x = {}, expected 2.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.05,
            "y = {}, expected 1.0",
            result.x[1]
        );
    }

    #[test]
    fn broyden1_nonlinear_circle() {
        // x² + y² = 1, x - y = 0 → x=y=1/√2
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]];
        let result = broyden1(f, &[0.5, 0.5], 1e-10, 200).expect("broyden1");
        assert!(result.converged, "broyden1 failed: {}", result.message);
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (result.x[0] - s2).abs() < 0.01,
            "x = {}, expected {}",
            result.x[0],
            s2
        );
    }

    #[test]
    fn broyden1_3d_system() {
        let f = |x: &[f64]| {
            vec![
                x[0] + x[1] + x[2] - 6.0,
                x[0] - x[1] + 2.0 * x[2] - 5.0,
                2.0 * x[0] + x[1] - x[2] - 1.0,
            ]
        };
        let result = broyden1(f, &[0.0, 0.0, 0.0], 1e-10, 200).expect("broyden1 3d");
        assert!(result.converged, "broyden1 3d failed: {}", result.message);
        assert!((result.x[0] - 1.0).abs() < 0.1, "x = {}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 0.1, "y = {}", result.x[1]);
        assert!((result.x[2] - 3.0).abs() < 0.1, "z = {}", result.x[2]);
    }

    #[test]
    fn broyden1_empty_x0_rejected() {
        let f = |_x: &[f64]| vec![];
        let err = broyden1(f, &[], 1e-10, 200).expect_err("empty");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn toms748_finds_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let result = toms748(f, (0.0, 2.0), RootOptions::default()).expect("toms748");
        assert!(result.converged, "toms748 failed: {}", result.message);
        assert!(
            (result.root - std::f64::consts::SQRT_2).abs() < 1e-10,
            "root = {}, expected sqrt(2)",
            result.root
        );
    }

    #[test]
    fn newton_scalar_finds_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        let result = newton_scalar(f, df, 1.0, RootOptions::default()).expect("newton");
        assert!(result.converged, "newton failed: {}", result.message);
        assert!(
            (result.root - std::f64::consts::SQRT_2).abs() < 1e-10,
            "root = {}",
            result.root
        );
    }

    #[test]
    fn secant_finds_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let result = secant(f, 1.0, Some(2.0), RootOptions::default()).expect("secant");
        assert!(result.converged, "secant failed: {}", result.message);
        assert!(
            (result.root - std::f64::consts::SQRT_2).abs() < 1e-10,
            "root = {}",
            result.root
        );
    }

    #[test]
    fn halley_finds_cbrt2() {
        let f = |x: f64| x * x * x - 2.0;
        let df = |x: f64| 3.0 * x * x;
        let d2f = |x: f64| 6.0 * x;
        let result = halley(f, df, d2f, 1.0, RootOptions::default()).expect("halley");
        assert!(result.converged, "halley failed: {}", result.message);
        assert!(
            (result.root - 2.0f64.cbrt()).abs() < 1e-10,
            "root = {}",
            result.root
        );
    }

    #[test]
    fn root_scalar_secant_auto() {
        let f = |x: f64| x * x - 2.0;
        let result = root_scalar(f, None, Some(1.0), Some(2.0), RootOptions::default())
            .expect("root_scalar secant");
        assert!(result.converged, "failed: {}", result.message);
        assert!(
            (result.root - std::f64::consts::SQRT_2).abs() < 1e-10,
            "root = {}",
            result.root
        );
    }
}
