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
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsci_runtime::RuntimeMode;
    use proptest::prelude::*;
    use serde::Serialize;

    use crate::{
        ConvergenceStatus, MinimizeOptions, OptimizeMethod, OptimizeResult, bfgs, cg_pr_plus,
        minimize, powell, take_optimize_traces,
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

    fn callback_points() -> &'static Mutex<Vec<Vec<f64>>> {
        static CALLBACK_POINTS: OnceLock<Mutex<Vec<Vec<f64>>>> = OnceLock::new();
        CALLBACK_POINTS.get_or_init(|| Mutex::new(Vec::new()))
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
        n_dim: usize,
        mode: RuntimeMode,
        result: &OptimizeResult,
        seed: u64,
    ) {
        let entry = TestLogEntry {
            test_id,
            optimizer,
            problem,
            n_dim,
            mode: mode_name(mode),
            converged: result.success,
            nfev: result.nfev,
            final_f: result.fun,
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

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|value| value * value).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        let x0 = x[0];
        let x1 = x[1];
        (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
    }

    fn himmelblau(x: &[f64]) -> f64 {
        let x0 = x[0];
        let x1 = x[1];
        (x0 * x0 + x1 - 11.0).powi(2) + (x0 + x1 * x1 - 7.0).powi(2)
    }

    fn flat_quartic(x: &[f64]) -> f64 {
        (x[0] - 1.0).powi(4) + (x[1] + 2.0).powi(4)
    }

    fn step_plateau(x: &[f64]) -> f64 {
        if x[0] >= 0.5 { 1.0 } else { 0.0 }
    }

    fn nonconvex_saddle(x: &[f64]) -> f64 {
        x[0] * x[0] - x[1] * x[1] + 0.1 * x[1].powi(4)
    }

    fn abs_sum(x: &[f64]) -> f64 {
        x.iter().map(|v| v.abs()).sum()
    }

    fn one_dim_quadratic(x: &[f64]) -> f64 {
        (x[0] - 1.5).powi(2) + 1.0
    }

    fn zero_function(_: &[f64]) -> f64 {
        0.0
    }

    fn callback_record_and_stop(x: &[f64]) -> bool {
        callback_points()
            .lock()
            .expect("callback points lock")
            .push(x.to_vec());
        false
    }

    #[test]
    fn optimize_result_success_fields_are_populated() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-8),
            maxiter: Some(200),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[2.0, -3.0], options).expect("minimize executes");
        assert!(result.success, "{}", result.message);
        assert_eq!(result.status, ConvergenceStatus::Success);
        assert!(result.fun.is_some());
        assert!(result.nfev >= 1);
        assert!(result.x.iter().all(|value| value.abs() < 1.0e-4));
        push_test_log(
            "optimize-result-success-fields",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            101,
        );
    }

    #[test]
    fn optimize_result_reports_max_iterations_exceeded() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-14),
            maxiter: Some(1),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[-1.2, 1.0], options).expect("minimize executes");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        assert!(result.message.contains("maximum iterations"));
        push_test_log(
            "optimize-result-maxiter",
            "bfgs",
            "rosenbrock",
            2,
            RuntimeMode::Strict,
            &result,
            102,
        );
    }

    #[test]
    fn optimize_result_reports_line_search_failure() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-12),
            maxiter: Some(50),
            maxfev: Some(20_000),
            gradient_eps: 1.0e-6,
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result =
            minimize(step_plateau, &[0.499_999], options).expect("minimize returns a result");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::PrecisionLoss);
        assert!(result.message.contains("line search failed"));
        push_test_log(
            "optimize-result-linesearch-failure",
            "bfgs",
            "step-plateau",
            1,
            RuntimeMode::Strict,
            &result,
            103,
        );
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
        push_test_log(
            "bfgs-quadratic",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            104,
        );
    }

    #[test]
    fn bfgs_converges_on_rosenbrock() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-7),
            maxiter: Some(800),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&rosenbrock, &[-1.2, 1.0], options).expect("bfgs executes");
        assert!(result.fun.expect("objective") < 1.0e-6);
        assert!((result.x[0] - 1.0).abs() < 1.0e-2);
        assert!((result.x[1] - 1.0).abs() < 1.0e-2);
        push_test_log(
            "bfgs-rosenbrock",
            "bfgs",
            "rosenbrock",
            2,
            RuntimeMode::Strict,
            &result,
            105,
        );
    }

    #[test]
    fn bfgs_finds_local_minimum_on_nonconvex_surface() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-7),
            maxiter: Some(800),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&himmelblau, &[0.0, 0.0], options).expect("bfgs executes");
        assert!(result.fun.expect("objective") < 1.0e-5);
        push_test_log(
            "bfgs-nonconvex-local-min",
            "bfgs",
            "himmelblau",
            2,
            RuntimeMode::Strict,
            &result,
            106,
        );
    }

    #[test]
    fn bfgs_zero_gradient_at_start_converges_immediately() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-8),
            maxiter: Some(200),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&sphere, &[0.0, 0.0], options).expect("bfgs executes");
        assert!(result.success);
        assert_eq!(result.nit, 0);
        push_test_log(
            "bfgs-zero-gradient-start",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            107,
        );
    }

    #[test]
    fn bfgs_handles_very_flat_function() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-7),
            maxiter: Some(800),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = flat_quartic(&[0.5, -1.5]);
        let result = bfgs(&flat_quartic, &[0.5, -1.5], options).expect("bfgs executes");
        assert!(result.fun.expect("objective") <= initial);
        push_test_log(
            "bfgs-flat-function",
            "bfgs",
            "flat-quartic",
            2,
            RuntimeMode::Strict,
            &result,
            108,
        );
    }

    #[test]
    fn bfgs_hardened_mode_rejects_nan_gradient_path() {
        let objective = |x: &[f64]| {
            if x[0] < 0.0 { f64::NAN } else { x[0] * x[0] }
        };
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            mode: RuntimeMode::Hardened,
            ..MinimizeOptions::default()
        };
        let result = minimize(objective, &[0.0], options).expect("returns an OptimizeResult");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::NanEncountered);
        push_test_log(
            "bfgs-hardened-nan-gradient",
            "bfgs",
            "nan-gradient",
            1,
            RuntimeMode::Hardened,
            &result,
            109,
        );
    }

    #[test]
    fn bfgs_callback_receives_intermediate_points() {
        callback_points().lock().expect("callback lock").clear();
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            callback: Some(callback_record_and_stop),
            maxiter: Some(40),
            maxfev: Some(10_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&sphere, &[2.5, -1.5], options).expect("bfgs executes");
        let points = callback_points().lock().expect("callback lock");
        assert!(!points.is_empty());
        assert_eq!(points[0], vec![2.5, -1.5]);
        assert_eq!(result.status, ConvergenceStatus::CallbackStop);
        push_test_log(
            "bfgs-callback-points",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            110,
        );
    }

    #[test]
    fn bfgs_gradient_tolerance_threshold_stops_early() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(10.0),
            maxiter: Some(40),
            maxfev: Some(10_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&sphere, &[0.1, -0.1], options).expect("bfgs executes");
        assert!(result.success);
        assert_eq!(result.nit, 0);
        push_test_log(
            "bfgs-gtol-threshold",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            111,
        );
    }

    #[test]
    fn cg_quadratic_converges_with_small_iterations() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-8),
            maxiter: Some(80),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = cg_pr_plus(&sphere, &[4.0, -1.5], options).expect("cg executes");
        assert!(result.success, "{}", result.message);
        assert!(result.nit <= 20);
        push_test_log(
            "cg-quadratic",
            "cg_pr_plus",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            112,
        );
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
        push_test_log(
            "cg-sphere-reference",
            "cg_pr_plus",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            113,
        );
    }

    #[test]
    fn cg_rosenbrock_reduces_objective() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-7),
            maxiter: Some(600),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = rosenbrock(&[-1.2, 1.0]);
        let result = cg_pr_plus(&rosenbrock, &[-1.2, 1.0], options).expect("cg executes");
        assert!(result.fun.expect("objective") < initial);
        push_test_log(
            "cg-rosenbrock",
            "cg_pr_plus",
            "rosenbrock",
            2,
            RuntimeMode::Strict,
            &result,
            114,
        );
    }

    #[test]
    fn cg_high_dimensional_problem_converges() {
        let diag_quadratic = |x: &[f64]| {
            x.iter()
                .enumerate()
                .map(|(idx, value)| (idx as f64 + 1.0) * value * value)
                .sum::<f64>()
        };
        let x0 = vec![1.0; 50];
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-6),
            maxiter: Some(600),
            maxfev: Some(200_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = cg_pr_plus(&diag_quadratic, &x0, options).expect("cg executes");
        assert!(result.fun.expect("objective") < 1.0e-6);
        push_test_log(
            "cg-high-dimensional",
            "cg_pr_plus",
            "diag-quadratic",
            50,
            RuntimeMode::Strict,
            &result,
            115,
        );
    }

    #[test]
    fn cg_handles_nonconvex_surface_without_invalid_status() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-6),
            maxiter: Some(200),
            maxfev: Some(100_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = nonconvex_saddle(&[1.0, 1.0]);
        let result = cg_pr_plus(&nonconvex_saddle, &[1.0, 1.0], options).expect("cg executes");
        assert!(result.fun.expect("objective") <= initial + 1.0e-6);
        assert_ne!(result.status, ConvergenceStatus::InvalidInput);
        push_test_log(
            "cg-nonconvex",
            "cg_pr_plus",
            "saddle",
            2,
            RuntimeMode::Strict,
            &result,
            116,
        );
    }

    #[test]
    fn cg_zero_function_trivial_convergence() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-8),
            maxiter: Some(40),
            maxfev: Some(10_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = cg_pr_plus(&zero_function, &[10.0, -10.0], options).expect("cg executes");
        assert!(result.success);
        assert_eq!(result.nit, 0);
        push_test_log(
            "cg-zero-function",
            "cg_pr_plus",
            "constant-zero",
            2,
            RuntimeMode::Strict,
            &result,
            117,
        );
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
        push_test_log(
            "powell-reduces-objective",
            "powell",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            118,
        );
    }

    #[test]
    fn powell_quadratic_converges_near_exact_minimum() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-6),
            maxiter: Some(120),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = powell(&sphere, &[2.0, -2.0], options).expect("powell executes");
        assert!(result.fun.expect("objective") < 1.0e-5);
        push_test_log(
            "powell-quadratic",
            "powell",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            119,
        );
    }

    #[test]
    fn powell_nonsmooth_objective_best_effort() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-6),
            maxiter: Some(120),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = abs_sum(&[3.0, -4.0]);
        let result = powell(&abs_sum, &[3.0, -4.0], options).expect("powell executes");
        assert!(result.fun.expect("objective") <= initial);
        push_test_log(
            "powell-nonsmooth",
            "powell",
            "absolute-sum",
            2,
            RuntimeMode::Strict,
            &result,
            120,
        );
    }

    #[test]
    fn powell_one_dimensional_path_reduces_objective() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-8),
            maxiter: Some(100),
            maxfev: Some(50_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = one_dim_quadratic(&[-3.0]);
        let result = powell(&one_dim_quadratic, &[-3.0], options).expect("powell executes");
        assert!(result.fun.expect("objective") < initial);
        push_test_log(
            "powell-one-dimensional",
            "powell",
            "one-dim-quadratic",
            1,
            RuntimeMode::Strict,
            &result,
            121,
        );
    }

    #[test]
    fn powell_constant_function_terminates_quickly() {
        let constant = |_x: &[f64]| 5.0;
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-8),
            maxiter: Some(100),
            maxfev: Some(50_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = powell(&constant, &[3.0, -1.0], options).expect("powell executes");
        assert!(result.success);
        assert!(result.nit <= 1);
        push_test_log(
            "powell-constant-function",
            "powell",
            "constant",
            2,
            RuntimeMode::Strict,
            &result,
            122,
        );
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
        push_test_log(
            "dispatch-selected-method",
            "minimize",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            123,
        );
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
        push_test_log(
            "hardened-non-finite",
            "minimize",
            "nan-objective",
            2,
            RuntimeMode::Hardened,
            &result,
            124,
        );
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
        let completion = traces
            .iter()
            .find(|entry| entry.event == "completion")
            .expect("completion trace exists");
        let result = OptimizeResult {
            x: completion.final_x.clone().unwrap_or_default(),
            fun: completion.final_f,
            success: completion
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("Success")),
            status: ConvergenceStatus::Success,
            message: completion.reason.clone().unwrap_or_default(),
            nfev: completion.total_nfev,
            njev: 0,
            nhev: 0,
            nit: completion.iter_num,
            jac: None,
            hess_inv: None,
            maxcv: None,
        };
        push_test_log(
            "trace-log-schema",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            125,
        );
    }

    #[test]
    fn invalid_gradient_epsilon_is_rejected() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            gradient_eps: 0.0,
            ..MinimizeOptions::default()
        };
        let err = bfgs(&sphere, &[1.0, 1.0], options).expect_err("invalid options should fail");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn evaluation_budget_exceeded_maps_to_max_evaluations() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            maxfev: Some(1),
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[1.0, 1.0], options).expect("execution returns result");
        assert_eq!(result.status, ConvergenceStatus::MaxEvaluations);
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 500,
            failure_persistence: None,
            .. ProptestConfig::default()
        })]

        #[test]
        fn property_minimize_improves_objective(
            x0 in proptest::array::uniform2(-4.0f64..4.0f64),
            method_pick in 0u8..3u8,
        ) {
            let method = match method_pick % 3 {
                0 => OptimizeMethod::Bfgs,
                1 => OptimizeMethod::ConjugateGradient,
                _ => OptimizeMethod::Powell,
            };
            let options = MinimizeOptions {
                method: Some(method),
                tol: Some(1.0e-6),
                maxiter: Some(40),
                maxfev: Some(20_000),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            };
            let result = minimize(sphere, &x0, options).expect("minimize executes");
            prop_assert!(result.fun.is_some());
            let final_f = result.fun.unwrap_or(f64::INFINITY);
            prop_assert!(final_f <= sphere(&x0) + 1.0e-8);
            push_test_log(
                "property-minimize-improves-objective",
                "minimize",
                "sphere",
                2,
                RuntimeMode::Strict,
                &result,
                201,
            );
        }

        #[test]
        fn property_nfev_is_non_zero_for_non_trivial_runs(
            x0 in proptest::array::uniform2(-3.0f64..3.0f64),
        ) {
            let norm = (x0[0] * x0[0] + x0[1] * x0[1]).sqrt();
            prop_assume!(norm > 1.0e-4);
            let options = MinimizeOptions {
                method: Some(OptimizeMethod::Bfgs),
                tol: Some(1.0e-9),
                maxiter: Some(5),
                maxfev: Some(5_000),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            };
            let result = bfgs(&sphere, &x0, options).expect("bfgs executes");
            prop_assert!(result.nfev >= 1);
            push_test_log(
                "property-nfev-non-zero",
                "bfgs",
                "sphere",
                2,
                RuntimeMode::Strict,
                &result,
                202,
            );
        }

        #[test]
        fn property_bfgs_hessian_inverse_is_symmetric_positive_diagonal(
            x0 in proptest::array::uniform2(-2.5f64..2.5f64),
        ) {
            let norm = (x0[0] * x0[0] + x0[1] * x0[1]).sqrt();
            prop_assume!(norm > 0.2);
            let options = MinimizeOptions {
                method: Some(OptimizeMethod::Bfgs),
                tol: Some(1.0e-6),
                maxiter: Some(40),
                maxfev: Some(40_000),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            };
            let result = bfgs(&sphere, &x0, options).expect("bfgs executes");
            let h_inv = result.hess_inv.as_ref().expect("hessian inverse is present");
            prop_assert_eq!(h_inv.len(), 2);
            prop_assert!((h_inv[0][1] - h_inv[1][0]).abs() <= 1.0e-6);
            prop_assert!(h_inv[0][0] > 0.0);
            prop_assert!(h_inv[1][1] > 0.0);
            push_test_log(
                "property-bfgs-hinv-spd",
                "bfgs",
                "sphere",
                2,
                RuntimeMode::Strict,
                &result,
                203,
            );
        }

        #[test]
        fn property_cg_first_iteration_is_descent(
            x0 in proptest::array::uniform2(-4.0f64..4.0f64),
        ) {
            let norm = (x0[0] * x0[0] + x0[1] * x0[1]).sqrt();
            prop_assume!(norm > 0.2);
            let _ = take_optimize_traces();
            let options = MinimizeOptions {
                method: Some(OptimizeMethod::ConjugateGradient),
                tol: Some(1.0e-6),
                maxiter: Some(40),
                maxfev: Some(40_000),
                fixture_id: Some("property-cg-descent"),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            };
            let initial = sphere(&x0);
            let result = cg_pr_plus(&sphere, &x0, options).expect("cg executes");
            let traces = take_optimize_traces();
            let first_iteration = traces
                .iter()
                .find(|entry| {
                    entry.event == "iteration"
                        && entry.method == OptimizeMethod::ConjugateGradient
                        && entry.fixture_id.as_deref() == Some("property-cg-descent")
                });
            if let Some(first) = first_iteration {
                prop_assert!(first.grad_norm.unwrap_or(f64::INFINITY).is_finite());
                let first_f = first.f_val.unwrap_or(f64::INFINITY);
                let final_f = result.fun.unwrap_or(f64::INFINITY);
                prop_assert!(first_f <= initial + 1.0e-6 || final_f <= initial + 1.0e-8);
            }
            push_test_log(
                "property-cg-descent",
                "cg_pr_plus",
                "sphere",
                2,
                RuntimeMode::Strict,
                &result,
                204,
            );
        }
    }
}
