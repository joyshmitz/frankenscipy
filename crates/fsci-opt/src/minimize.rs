#![forbid(unsafe_code)]

use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::types::{
    ConvergenceStatus, MinimizeOptions, OptError, OptimizeMethod, OptimizeResult,
    OptimizeTraceEntry,
};

pub fn minimize<F>(fun: F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    if x0.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("x0 must be a finite 1-D vector with at least one element"),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x0 must not contain NaN or Inf"),
        });
    }

    match options.method.unwrap_or(OptimizeMethod::Bfgs) {
        OptimizeMethod::Bfgs => bfgs(&fun, x0, options),
        OptimizeMethod::ConjugateGradient => cg_pr_plus(&fun, x0, options),
        OptimizeMethod::Powell => powell(&fun, x0, options),
    }
}

pub fn bfgs<F>(fun: &F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = options.tol.unwrap_or(1.0e-6).max(1.0e-12);
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((2000 * n).max(400));
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = match objective.eval(&x) {
        Ok(value) => value,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };
    let mut njev = 0usize;
    let mut grad = match finite_diff_gradient(&mut objective, &x, options.gradient_eps) {
        Ok(value) => {
            njev += 1;
            value
        }
        Err(err) => return Ok(result_from_error(&x, 0, objective.nfev, njev, err)),
    };

    let mut h_inv = identity_matrix(n);
    let mut nit = 0usize;

    for iteration in 0..maxiter {
        let grad_norm = l2_norm(&grad);
        if grad_norm <= tol {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (gradient norm <= tol)"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(grad.clone()),
                hess_inv: Some(h_inv.clone()),
                maxcv: None,
            };
            log_completion(OptimizeMethod::Bfgs, options, iteration, &result);
            return Ok(result);
        }

        if let Some(callback) = options.callback
            && !callback(&x)
        {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(grad.clone()),
                hess_inv: Some(h_inv.clone()),
                maxcv: None,
            };
            log_completion(OptimizeMethod::Bfgs, options, iteration, &result);
            return Ok(result);
        }

        let mut direction = matrix_vector_mul(&h_inv, &scale_vector(&grad, -1.0));
        if dot(&direction, &grad) >= 0.0 {
            direction = scale_vector(&grad, -1.0);
        }
        let search = match armijo_backtracking(&mut objective, &x, f, &grad, &direction) {
            Ok(Some(value)) => value,
            Ok(None) => {
                let result = OptimizeResult {
                    x: x.clone(),
                    fun: Some(f),
                    success: false,
                    status: ConvergenceStatus::PrecisionLoss,
                    message: String::from("line search failed to find a sufficient decrease"),
                    nfev: objective.nfev,
                    njev,
                    nhev: 0,
                    nit,
                    jac: Some(grad.clone()),
                    hess_inv: Some(h_inv.clone()),
                    maxcv: None,
                };
                log_completion(OptimizeMethod::Bfgs, options, iteration, &result);
                return Ok(result);
            }
            Err(err) => return Ok(result_from_error(&x, nit, objective.nfev, njev, err)),
        };

        let next_grad = match finite_diff_gradient(&mut objective, &search.x, options.gradient_eps)
        {
            Ok(value) => {
                njev += 1;
                value
            }
            Err(err) => return Ok(result_from_error(&search.x, nit, objective.nfev, njev, err)),
        };

        let s = sub_vectors(&search.x, &x);
        let y = sub_vectors(&next_grad, &grad);
        let ys = dot(&y, &s);
        if ys > 1.0e-12 {
            let rho = 1.0 / ys;
            h_inv = bfgs_inverse_update(&h_inv, &s, &y, rho);
        } else {
            h_inv = identity_matrix(n);
        }

        log_iteration(
            OptimizeMethod::Bfgs,
            options,
            iteration + 1,
            search.f,
            l2_norm(&next_grad),
            search.alpha,
            objective.nfev,
        );

        x = search.x;
        f = search.f;
        grad = next_grad;
        nit = iteration + 1;
    }

    let result = OptimizeResult {
        x: x.clone(),
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: String::from("maximum iterations exceeded"),
        nfev: objective.nfev,
        njev,
        nhev: 0,
        nit,
        jac: Some(grad),
        hess_inv: Some(h_inv),
        maxcv: None,
    };
    log_completion(OptimizeMethod::Bfgs, options, nit, &result);
    Ok(result)
}

pub fn cg_pr_plus<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = options.tol.unwrap_or(1.0e-6).max(1.0e-12);
    let maxiter = options.maxiter.unwrap_or((250 * n).max(120));
    let maxfev = options.maxfev.unwrap_or((2500 * n).max(500));
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = match objective.eval(&x) {
        Ok(value) => value,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };
    let mut njev = 0usize;
    let mut grad = match finite_diff_gradient(&mut objective, &x, options.gradient_eps) {
        Ok(value) => {
            njev += 1;
            value
        }
        Err(err) => return Ok(result_from_error(&x, 0, objective.nfev, njev, err)),
    };
    let mut direction = scale_vector(&grad, -1.0);
    let mut nit = 0usize;

    for iteration in 0..maxiter {
        let grad_norm = l2_norm(&grad);
        if grad_norm <= tol {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (gradient norm <= tol)"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(
                OptimizeMethod::ConjugateGradient,
                options,
                iteration,
                &result,
            );
            return Ok(result);
        }

        if let Some(callback) = options.callback
            && !callback(&x)
        {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(
                OptimizeMethod::ConjugateGradient,
                options,
                iteration,
                &result,
            );
            return Ok(result);
        }

        if dot(&direction, &grad) >= 0.0 {
            direction = scale_vector(&grad, -1.0);
        }
        let search = match armijo_backtracking(&mut objective, &x, f, &grad, &direction) {
            Ok(Some(value)) => value,
            Ok(None) => {
                let result = OptimizeResult {
                    x: x.clone(),
                    fun: Some(f),
                    success: false,
                    status: ConvergenceStatus::PrecisionLoss,
                    message: String::from("line search failed to find a sufficient decrease"),
                    nfev: objective.nfev,
                    njev,
                    nhev: 0,
                    nit,
                    jac: Some(grad.clone()),
                    hess_inv: None,
                    maxcv: None,
                };
                log_completion(
                    OptimizeMethod::ConjugateGradient,
                    options,
                    iteration,
                    &result,
                );
                return Ok(result);
            }
            Err(err) => return Ok(result_from_error(&x, nit, objective.nfev, njev, err)),
        };

        let next_grad = match finite_diff_gradient(&mut objective, &search.x, options.gradient_eps)
        {
            Ok(value) => {
                njev += 1;
                value
            }
            Err(err) => return Ok(result_from_error(&search.x, nit, objective.nfev, njev, err)),
        };

        let denom = dot(&grad, &grad).max(1.0e-18);
        let grad_delta = sub_vectors(&next_grad, &grad);
        let beta_pr = dot(&next_grad, &grad_delta) / denom;
        let beta = beta_pr.max(0.0);
        direction = sub_vectors(
            &scale_vector(&next_grad, -1.0),
            &scale_vector(&direction, -beta),
        );

        log_iteration(
            OptimizeMethod::ConjugateGradient,
            options,
            iteration + 1,
            search.f,
            l2_norm(&next_grad),
            search.alpha,
            objective.nfev,
        );

        x = search.x;
        f = search.f;
        grad = next_grad;
        nit = iteration + 1;
    }

    let result = OptimizeResult {
        x: x.clone(),
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: String::from("maximum iterations exceeded"),
        nfev: objective.nfev,
        njev,
        nhev: 0,
        nit,
        jac: Some(grad),
        hess_inv: None,
        maxcv: None,
    };
    log_completion(OptimizeMethod::ConjugateGradient, options, nit, &result);
    Ok(result)
}

pub fn powell<F>(fun: &F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = options.tol.unwrap_or(1.0e-6).max(1.0e-12);
    let maxiter = options.maxiter.unwrap_or((150 * n).max(80));
    let maxfev = options.maxfev.unwrap_or((3000 * n).max(800));
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = match objective.eval(&x) {
        Ok(value) => value,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };
    let mut directions = identity_matrix(n);

    for iteration in 0..maxiter {
        if let Some(callback) = options.callback
            && !callback(&x)
        {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev: 0,
                nhev: 0,
                nit: iteration,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::Powell, options, iteration, &result);
            return Ok(result);
        }

        let x_start = x.clone();
        let f_start = f;
        let mut largest_drop = 0.0;
        let mut largest_drop_idx = 0usize;

        for (dir_idx, direction) in directions.iter().enumerate() {
            let search = match golden_section_direction_search(
                &mut objective,
                &x,
                f,
                direction,
                tol.max(1.0e-4),
            ) {
                Ok(value) => value,
                Err(err) => return Ok(result_from_error(&x, iteration, objective.nfev, 0, err)),
            };
            let drop = (f - search.f).max(0.0);
            if drop > largest_drop {
                largest_drop = drop;
                largest_drop_idx = dir_idx;
            }
            x = search.x;
            f = search.f;
            log_iteration(
                OptimizeMethod::Powell,
                options,
                iteration + 1,
                f,
                0.0,
                search.alpha,
                objective.nfev,
            );
        }

        let move_vec = sub_vectors(&x, &x_start);
        let move_norm = l2_norm(&move_vec);
        let f_delta = (f_start - f).abs();
        if move_norm <= tol || f_delta <= tol {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (step/f-value change <= tol)"),
                nfev: objective.nfev,
                njev: 0,
                nhev: 0,
                nit: iteration + 1,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::Powell, options, iteration + 1, &result);
            return Ok(result);
        }

        if move_norm > 1.0e-12 {
            directions[largest_drop_idx] = scale_vector(&move_vec, 1.0 / move_norm);
        }
    }

    let result = OptimizeResult {
        x: x.clone(),
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: String::from("maximum iterations exceeded"),
        nfev: objective.nfev,
        njev: 0,
        nhev: 0,
        nit: maxiter,
        jac: None,
        hess_inv: None,
        maxcv: None,
    };
    log_completion(OptimizeMethod::Powell, options, maxiter, &result);
    Ok(result)
}

#[must_use]
pub fn take_optimize_traces() -> Vec<OptimizeTraceEntry> {
    match trace_log().lock() {
        Ok(mut guard) => std::mem::take(&mut *guard),
        Err(_) => Vec::new(),
    }
}

#[derive(Debug, Clone)]
struct LineSearchStep {
    alpha: f64,
    x: Vec<f64>,
    f: f64,
}

struct Objective<'a, F>
where
    F: Fn(&[f64]) -> f64,
{
    fun: &'a F,
    mode: fsci_runtime::RuntimeMode,
    maxfev: usize,
    nfev: usize,
}

impl<'a, F> Objective<'a, F>
where
    F: Fn(&[f64]) -> f64,
{
    fn new(fun: &'a F, mode: fsci_runtime::RuntimeMode, maxfev: usize) -> Self {
        Self {
            fun,
            mode,
            maxfev: maxfev.max(1),
            nfev: 0,
        }
    }

    fn eval(&mut self, x: &[f64]) -> Result<f64, OptError> {
        if self.nfev >= self.maxfev {
            return Err(OptError::EvaluationBudgetExceeded {
                detail: format!("max function evaluations exceeded ({})", self.maxfev),
            });
        }
        let value = (self.fun)(x);
        self.nfev += 1;
        if !value.is_finite() {
            return match self.mode {
                fsci_runtime::RuntimeMode::Strict => Err(OptError::InvalidArgument {
                    detail: String::from("objective evaluated to non-finite value"),
                }),
                fsci_runtime::RuntimeMode::Hardened => Err(OptError::NonFiniteInput {
                    detail: String::from("hardened mode rejects non-finite objective values"),
                }),
            };
        }
        Ok(value)
    }
}

fn finite_diff_gradient<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    gradient_eps: f64,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let mut gradient = vec![0.0; x.len()];
    for (idx, component) in x.iter().enumerate() {
        let step = gradient_eps * (1.0 + component.abs());
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[idx] += step;
        x_minus[idx] -= step;
        let f_plus = objective.eval(&x_plus)?;
        let f_minus = objective.eval(&x_minus)?;
        gradient[idx] = (f_plus - f_minus) / (2.0 * step);
    }
    Ok(gradient)
}

fn armijo_backtracking<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    fx: f64,
    grad: &[f64],
    direction: &[f64],
) -> Result<Option<LineSearchStep>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let directional_derivative = dot(grad, direction);
    if directional_derivative >= 0.0 {
        return Ok(None);
    }

    let c1 = 1.0e-4;
    let mut alpha = 1.0;
    for _ in 0..24 {
        let candidate_x = add_scaled(x, direction, alpha);
        let candidate_f = objective.eval(&candidate_x)?;
        if candidate_f <= fx + c1 * alpha * directional_derivative {
            return Ok(Some(LineSearchStep {
                alpha,
                x: candidate_x,
                f: candidate_f,
            }));
        }
        alpha *= 0.5;
        if alpha < 1.0e-12 {
            break;
        }
    }
    Ok(None)
}

fn golden_section_direction_search<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    fx: f64,
    direction: &[f64],
    tolerance: f64,
) -> Result<LineSearchStep, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut left = -1.5;
    let mut right = 1.5;
    let mut c = right - phi * (right - left);
    let mut d = left + phi * (right - left);

    let mut fc = objective.eval(&add_scaled(x, direction, c))?;
    let mut fd = objective.eval(&add_scaled(x, direction, d))?;

    for _ in 0..40 {
        if (right - left).abs() <= tolerance {
            break;
        }
        if fc < fd {
            right = d;
            d = c;
            fd = fc;
            c = right - phi * (right - left);
            fc = objective.eval(&add_scaled(x, direction, c))?;
        } else {
            left = c;
            c = d;
            fc = fd;
            d = left + phi * (right - left);
            fd = objective.eval(&add_scaled(x, direction, d))?;
        }
    }

    let alpha = 0.5 * (left + right);
    let candidate_x = add_scaled(x, direction, alpha);
    let candidate_f = objective.eval(&candidate_x)?;
    if candidate_f <= fx {
        return Ok(LineSearchStep {
            alpha,
            x: candidate_x,
            f: candidate_f,
        });
    }

    Ok(LineSearchStep {
        alpha: 0.0,
        x: x.to_vec(),
        f: fx,
    })
}

fn result_from_error(
    x: &[f64],
    nit: usize,
    nfev: usize,
    njev: usize,
    error: OptError,
) -> OptimizeResult {
    let (status, message) = match error {
        OptError::EvaluationBudgetExceeded { detail } => {
            (ConvergenceStatus::MaxEvaluations, detail)
        }
        OptError::NonFiniteInput { detail } => (ConvergenceStatus::NanEncountered, detail),
        OptError::InvalidArgument { detail } | OptError::InvalidBounds { detail } => {
            (ConvergenceStatus::InvalidInput, detail)
        }
        OptError::SignChangeRequired { detail } => (ConvergenceStatus::InvalidInput, detail),
        OptError::NotImplemented { detail } => (ConvergenceStatus::NotImplemented, detail),
    };
    OptimizeResult {
        x: x.to_vec(),
        fun: None,
        success: false,
        status,
        message,
        nfev,
        njev,
        nhev: 0,
        nit,
        jac: None,
        hess_inv: None,
        maxcv: None,
    }
}

fn validate_minimize_options(options: MinimizeOptions) -> Result<(), OptError> {
    if let Some(maxiter) = options.maxiter
        && maxiter == 0
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxiter must be >= 1"),
        });
    }
    if let Some(maxfev) = options.maxfev
        && maxfev == 0
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxfev must be >= 1"),
        });
    }
    if let Some(tol) = options.tol
        && (!tol.is_finite() || tol <= 0.0)
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("tol must be finite and > 0"),
        });
    }
    if !options.gradient_eps.is_finite() || options.gradient_eps <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("gradient_eps must be finite and > 0"),
        });
    }
    Ok(())
}

fn bfgs_inverse_update(h_inv: &[Vec<f64>], s: &[f64], y: &[f64], rho: f64) -> Vec<Vec<f64>> {
    let n = s.len();
    let mut identity = identity_matrix(n);

    for row in 0..n {
        for col in 0..n {
            identity[row][col] -= rho * s[row] * y[col];
        }
    }
    let left = matrix_mul(&identity, h_inv);

    let mut identity_t = identity_matrix(n);
    for row in 0..n {
        for col in 0..n {
            identity_t[row][col] -= rho * y[row] * s[col];
        }
    }
    let core = matrix_mul(&left, &identity_t);
    let rank1 = outer_product(s, s);
    add_matrices(&core, &scale_matrix(&rank1, rho))
}

fn log_iteration(
    method: OptimizeMethod,
    options: MinimizeOptions,
    iter_num: usize,
    f_val: f64,
    grad_norm: f64,
    step_size: f64,
    total_nfev: usize,
) {
    let trace = OptimizeTraceEntry {
        ts_unix_ms: now_unix_ms(),
        event: String::from("iteration"),
        method,
        iter_num,
        f_val: Some(f_val),
        grad_norm: Some(grad_norm),
        step_size: Some(step_size),
        mode: options.mode,
        reason: None,
        final_x: None,
        final_f: None,
        total_nfev,
        fixture_id: options.fixture_id.map(ToOwned::to_owned),
        seed: options.seed,
    };
    push_trace(trace);
}

fn log_completion(
    method: OptimizeMethod,
    options: MinimizeOptions,
    iter_num: usize,
    result: &OptimizeResult,
) {
    let trace = OptimizeTraceEntry {
        ts_unix_ms: now_unix_ms(),
        event: String::from("completion"),
        method,
        iter_num,
        f_val: result.fun,
        grad_norm: None,
        step_size: None,
        mode: options.mode,
        reason: Some(format!("{:?}: {}", result.status, result.message)),
        final_x: Some(result.x.clone()),
        final_f: result.fun,
        total_nfev: result.nfev,
        fixture_id: options.fixture_id.map(ToOwned::to_owned),
        seed: options.seed,
    };
    push_trace(trace);
}

fn trace_log() -> &'static Mutex<Vec<OptimizeTraceEntry>> {
    static TRACE_LOG: OnceLock<Mutex<Vec<OptimizeTraceEntry>>> = OnceLock::new();
    TRACE_LOG.get_or_init(|| Mutex::new(Vec::new()))
}

fn push_trace(entry: OptimizeTraceEntry) {
    if let Ok(mut guard) = trace_log().lock() {
        guard.push(entry);
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn l2_norm(vec: &[f64]) -> f64 {
    dot(vec, vec).sqrt()
}

fn scale_vector(input: &[f64], scale: f64) -> Vec<f64> {
    input.iter().map(|value| value * scale).collect()
}

fn add_scaled(lhs: &[f64], rhs: &[f64], scale: f64) -> Vec<f64> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| left + scale * right)
        .collect()
}

fn sub_vectors(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect()
}

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; n]; n];
    for (idx, row) in out.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    out
}

fn matrix_vector_mul(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, vector)).collect()
}

fn outer_product(lhs: &[f64], rhs: &[f64]) -> Vec<Vec<f64>> {
    lhs.iter()
        .map(|a| rhs.iter().map(|b| a * b).collect())
        .collect()
}

fn matrix_mul(lhs: &[Vec<f64>], rhs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = lhs.len();
    let m = rhs.first().map_or(0, |row| row.len());
    let mut out = vec![vec![0.0; m]; n];
    for row in 0..n {
        for col in 0..m {
            let mut accum = 0.0;
            for (k, rhs_row) in rhs.iter().enumerate() {
                accum += lhs[row][k] * rhs_row[col];
            }
            out[row][col] = accum;
        }
    }
    out
}

fn scale_matrix(matrix: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|entry| entry * scale).collect())
        .collect()
}

fn add_matrices(lhs: &[Vec<f64>], rhs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left_row, right_row)| {
            left_row
                .iter()
                .zip(right_row.iter())
                .map(|(left, right)| left + right)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use fsci_runtime::RuntimeMode;

    use crate::{
        ConvergenceStatus, MinimizeOptions, OptimizeMethod, bfgs, cg_pr_plus, minimize, powell,
        take_optimize_traces,
    };

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|value| value * value).sum()
    }

    #[test]
    fn bfgs_converges_on_quadratic() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-7),
            maxiter: Some(200),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&sphere, &[2.0, -3.0], options).expect("bfgs executes");
        assert!(result.success, "{}", result.message);
        assert_eq!(result.status, ConvergenceStatus::Success);
        assert!(result.fun.expect("objective") < 1.0e-8);
        assert!(result.x.iter().all(|value| value.abs() < 1.0e-4));
    }

    #[test]
    fn cg_converges_on_quadratic() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-7),
            maxiter: Some(200),
            maxfev: Some(30_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = cg_pr_plus(&sphere, &[4.0, -1.5], options).expect("cg executes");
        assert!(result.success, "{}", result.message);
        assert!(result.fun.expect("objective") < 1.0e-8);
    }

    #[test]
    fn powell_reduces_objective() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-5),
            maxiter: Some(50),
            maxfev: Some(40_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = sphere(&[3.0, -2.0]);
        let result = powell(&sphere, &[3.0, -2.0], options).expect("powell executes");
        assert!(result.fun.expect("objective") < initial);
    }

    #[test]
    fn minimize_dispatches_to_selected_method() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-7),
            maxiter: Some(120),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[1.0, -1.0], options).expect("minimize executes");
        assert!(result.success, "{}", result.message);
    }

    #[test]
    fn hardened_mode_rejects_non_finite_objective_values() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            mode: RuntimeMode::Hardened,
            ..MinimizeOptions::default()
        };
        let result = minimize(|_| f64::NAN, &[1.0, 2.0], options)
            .expect("execution should return an OptimizeResult");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::NanEncountered);
    }

    #[test]
    fn trace_log_contains_iteration_and_completion_events() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-6),
            maxiter: Some(40),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let _ = minimize(sphere, &[1.0, 1.0], options).expect("execution succeeds");
        let traces = take_optimize_traces();
        assert!(!traces.is_empty());
        assert!(traces.iter().any(|entry| entry.event == "iteration"));
        assert!(traces.iter().any(|entry| entry.event == "completion"));
    }
}
