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
        OptimizeMethod::NelderMead => nelder_mead(&fun, x0, options),
        OptimizeMethod::LBfgsB => lbfgsb(&fun, x0, options, None),
        OptimizeMethod::NewtonCg => newton_cg(&fun, x0, options),
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

/// Nelder-Mead simplex (downhill simplex) method for derivative-free optimization.
///
/// Matches `scipy.optimize.minimize(f, x0, method='Nelder-Mead')`.
/// Uses adaptive parameters when n >= 2 (matching SciPy's `adaptive=True` default for n>=2).
pub fn nelder_mead<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    // Nelder-Mead doesn't use gradient_eps, but we still validate other options
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

    let n = x0.len();
    let tol = options.tol.unwrap_or(1.0e-8);
    let xatol = tol;
    let fatol = tol;
    let maxiter = options.maxiter.unwrap_or(200 * n);
    let maxfev = options.maxfev.unwrap_or(200 * n);
    let mut objective = Objective::new(fun, options.mode, maxfev);

    // Adaptive parameters (SciPy convention for n >= 2)
    let (rho, chi, psi, sigma) = if n >= 2 {
        let dim = n as f64;
        (
            1.0,
            1.0 + 2.0 / dim,
            0.75 - 1.0 / (2.0 * dim),
            1.0 - 1.0 / dim,
        )
    } else {
        (1.0, 2.0, 0.5, 0.5) // standard parameters
    };

    // Build initial simplex: n+1 vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());

    for j in 0..n {
        let mut vertex = x0.to_vec();
        let h = if x0[j].abs() > 1e-12 {
            0.05 * x0[j]
        } else {
            0.00025
        };
        vertex[j] += h;
        simplex.push(vertex);
    }

    // Evaluate at all vertices
    let mut f_values: Vec<f64> = Vec::with_capacity(n + 1);
    for vertex in &simplex {
        let fval = match objective.eval(vertex) {
            Ok(v) => v,
            Err(err) => return Ok(result_from_error(x0, 0, objective.nfev, 0, err)),
        };
        f_values.push(fval);
    }

    let mut nit = 0usize;

    for iteration in 0..maxiter {
        nit = iteration + 1;

        // Sort simplex by function values
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| {
            f_values[a]
                .partial_cmp(&f_values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sorted_simplex: Vec<Vec<f64>> = indices.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_f: Vec<f64> = indices.iter().map(|&i| f_values[i]).collect();
        simplex = sorted_simplex;
        f_values = sorted_f;

        // Check convergence: range of function values and simplex diameter
        let f_range = f_values[n] - f_values[0];
        let mut max_delta = 0.0_f64;
        for vertex in simplex.iter().skip(1) {
            let delta: f64 = vertex
                .iter()
                .zip(simplex[0].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            max_delta = max_delta.max(delta);
        }

        if f_range <= fatol && max_delta <= xatol {
            let result = OptimizeResult {
                x: simplex[0].clone(),
                fun: Some(f_values[0]),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (Nelder-Mead)"),
                nfev: objective.nfev,
                njev: 0,
                nhev: 0,
                nit,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::NelderMead, options, iteration, &result);
            return Ok(result);
        }

        if let Some(callback) = options.callback
            && !callback(&simplex[0])
        {
            let result = OptimizeResult {
                x: simplex[0].clone(),
                fun: Some(f_values[0]),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev: 0,
                nhev: 0,
                nit,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::NelderMead, options, iteration, &result);
            return Ok(result);
        }

        // Centroid of best n vertices (exclude worst)
        let mut centroid = vec![0.0; n];
        for vertex in simplex.iter().take(n) {
            for (j, c) in centroid.iter_mut().enumerate() {
                *c += vertex[j];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // Reflection: x_r = centroid + rho * (centroid - worst)
        let worst = &simplex[n];
        let x_r: Vec<f64> = centroid
            .iter()
            .zip(worst.iter())
            .map(|(c, w)| c + rho * (c - w))
            .collect();
        let f_r = match objective.eval(&x_r) {
            Ok(v) => v,
            Err(err) => return Ok(result_from_error(&simplex[0], nit, objective.nfev, 0, err)),
        };

        if f_r < f_values[0] {
            // Expansion: x_e = centroid + chi * (x_r - centroid)
            let x_e: Vec<f64> = centroid
                .iter()
                .zip(x_r.iter())
                .map(|(c, r)| c + chi * (r - c))
                .collect();
            let f_e = match objective.eval(&x_e) {
                Ok(v) => v,
                Err(err) => return Ok(result_from_error(&simplex[0], nit, objective.nfev, 0, err)),
            };
            if f_e < f_r {
                simplex[n] = x_e;
                f_values[n] = f_e;
            } else {
                simplex[n] = x_r;
                f_values[n] = f_r;
            }
        } else if f_r < f_values[n - 1] {
            // Accept reflection
            simplex[n] = x_r;
            f_values[n] = f_r;
        } else {
            // Contraction
            if f_r < f_values[n] {
                // Outside contraction: x_c = centroid + psi * (x_r - centroid)
                let x_c: Vec<f64> = centroid
                    .iter()
                    .zip(x_r.iter())
                    .map(|(c, r)| c + psi * (r - c))
                    .collect();
                let f_c = match objective.eval(&x_c) {
                    Ok(v) => v,
                    Err(err) => {
                        return Ok(result_from_error(&simplex[0], nit, objective.nfev, 0, err));
                    }
                };
                if f_c <= f_r {
                    simplex[n] = x_c;
                    f_values[n] = f_c;
                } else {
                    // Shrink
                    nelder_mead_shrink(&mut simplex, &mut f_values, sigma, &mut objective, n)?;
                }
            } else {
                // Inside contraction: x_cc = centroid - psi * (centroid - worst)
                let x_cc: Vec<f64> = centroid
                    .iter()
                    .zip(worst.iter())
                    .map(|(c, w)| c - psi * (c - w))
                    .collect();
                let f_cc = match objective.eval(&x_cc) {
                    Ok(v) => v,
                    Err(err) => {
                        return Ok(result_from_error(&simplex[0], nit, objective.nfev, 0, err));
                    }
                };
                if f_cc < f_values[n] {
                    simplex[n] = x_cc;
                    f_values[n] = f_cc;
                } else {
                    // Shrink
                    nelder_mead_shrink(&mut simplex, &mut f_values, sigma, &mut objective, n)?;
                }
            }
        }

        log_iteration(
            OptimizeMethod::NelderMead,
            options,
            iteration,
            f_values[0],
            0.0,
            0.0,
            objective.nfev,
        );
    }

    // Max iterations reached
    let mut indices: Vec<usize> = (0..=n).collect();
    indices.sort_by(|&a, &b| {
        f_values[a]
            .partial_cmp(&f_values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best_idx = indices[0];

    let result = OptimizeResult {
        x: simplex[best_idx].clone(),
        fun: Some(f_values[best_idx]),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: format!("maximum iterations reached ({maxiter})"),
        nfev: objective.nfev,
        njev: 0,
        nhev: 0,
        nit,
        jac: None,
        hess_inv: None,
        maxcv: None,
    };
    log_completion(OptimizeMethod::NelderMead, options, nit, &result);
    Ok(result)
}

fn nelder_mead_shrink<F>(
    simplex: &mut [Vec<f64>],
    f_values: &mut [f64],
    sigma: f64,
    objective: &mut Objective<'_, F>,
    n: usize,
) -> Result<(), OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let best = simplex[0].clone();
    for i in 1..=n {
        for j in 0..n {
            simplex[i][j] = best[j] + sigma * (simplex[i][j] - best[j]);
        }
        f_values[i] = objective.eval(&simplex[i]).map_err(|e| match e {
            OptError::EvaluationBudgetExceeded { detail } => {
                OptError::EvaluationBudgetExceeded { detail }
            }
            other => other,
        })?;
    }
    Ok(())
}

/// Bound constraint: (lower, upper) for each variable.
/// `None` means unbounded in that direction.
pub type Bound = (Option<f64>, Option<f64>);

/// L-BFGS-B: Limited-memory BFGS with box constraints.
///
/// Matches `scipy.optimize.minimize(f, x0, method='L-BFGS-B', bounds=...)`.
/// Uses two-loop recursion with limited memory (m=10 corrections).
pub fn lbfgsb<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
    bounds: Option<&[Bound]>,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = options.tol.unwrap_or(1.0e-6).max(1.0e-12);
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((2000 * n).max(400));
    let m = 10; // number of correction pairs stored
    let mut objective = Objective::new(fun, options.mode, maxfev);

    // Project x0 onto bounds
    let mut x = x0.to_vec();
    if let Some(bounds) = bounds {
        project_onto_bounds(&mut x, bounds);
    }

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

    // L-BFGS correction history
    let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rho_history: Vec<f64> = Vec::with_capacity(m);

    let mut nit = 0usize;

    for iteration in 0..maxiter {
        nit = iteration + 1;

        // Project gradient for bound-constrained case
        let projected_grad = if let Some(bounds) = bounds {
            projected_gradient(&x, &grad, bounds)
        } else {
            grad.clone()
        };

        let grad_norm = l2_norm(&projected_grad);
        if grad_norm <= tol {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (L-BFGS-B gradient norm <= tol)"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::LBfgsB, options, iteration, &result);
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
            log_completion(OptimizeMethod::LBfgsB, options, iteration, &result);
            return Ok(result);
        }

        // Two-loop recursion for L-BFGS direction
        let direction = lbfgs_two_loop(&grad, &s_history, &y_history, &rho_history);

        // Negate for descent direction
        let direction: Vec<f64> = direction.iter().map(|&d| -d).collect();

        // Line search with bound projection
        let mut alpha = 1.0;
        let directional_deriv = dot(&grad, &direction);
        if directional_deriv >= 0.0 {
            // Not a descent direction — reset history and use steepest descent
            s_history.clear();
            y_history.clear();
            rho_history.clear();
            let direction: Vec<f64> = grad.iter().map(|&g| -g).collect();
            alpha = 1.0 / l2_norm(&direction).max(1.0);
            let candidate_x = add_scaled(&x, &direction, alpha);
            let mut projected_candidate = candidate_x;
            if let Some(bounds) = bounds {
                project_onto_bounds(&mut projected_candidate, bounds);
            }
            match objective.eval(&projected_candidate) {
                Ok(fv) => {
                    let s = sub_vectors(&projected_candidate, &x);
                    x = projected_candidate;
                    f = fv;
                    let new_grad =
                        match finite_diff_gradient(&mut objective, &x, options.gradient_eps) {
                            Ok(g) => {
                                njev += 1;
                                g
                            }
                            Err(err) => {
                                return Ok(result_from_error(&x, nit, objective.nfev, njev, err));
                            }
                        };
                    let y = sub_vectors(&new_grad, &grad);
                    let sy = dot(&s, &y);
                    if sy > 1e-10 {
                        push_lbfgs_history(
                            &mut s_history,
                            &mut y_history,
                            &mut rho_history,
                            s,
                            y,
                            sy,
                            m,
                        );
                    }
                    grad = new_grad;
                }
                Err(err) => return Ok(result_from_error(&x, nit, objective.nfev, njev, err)),
            }
            continue;
        }

        // Armijo backtracking with bound projection
        let c1 = 1e-4;
        let mut step_accepted = false;
        for _ in 0..24 {
            let mut candidate_x = add_scaled(&x, &direction, alpha);
            if let Some(bounds) = bounds {
                project_onto_bounds(&mut candidate_x, bounds);
            }
            match objective.eval(&candidate_x) {
                Ok(fv) => {
                    if fv <= f + c1 * alpha * directional_deriv {
                        let s = sub_vectors(&candidate_x, &x);
                        x = candidate_x;
                        f = fv;
                        let new_grad =
                            match finite_diff_gradient(&mut objective, &x, options.gradient_eps) {
                                Ok(g) => {
                                    njev += 1;
                                    g
                                }
                                Err(err) => {
                                    return Ok(result_from_error(
                                        &x,
                                        nit,
                                        objective.nfev,
                                        njev,
                                        err,
                                    ));
                                }
                            };
                        let y = sub_vectors(&new_grad, &grad);
                        let sy = dot(&s, &y);
                        if sy > 1e-10 {
                            push_lbfgs_history(
                                &mut s_history,
                                &mut y_history,
                                &mut rho_history,
                                s,
                                y,
                                sy,
                                m,
                            );
                        }
                        grad = new_grad;
                        step_accepted = true;
                        break;
                    }
                }
                Err(err) => return Ok(result_from_error(&x, nit, objective.nfev, njev, err)),
            }
            alpha *= 0.5;
            if alpha < 1e-12 {
                break;
            }
        }

        if !step_accepted {
            break;
        }

        log_iteration(
            OptimizeMethod::LBfgsB,
            options,
            iteration,
            f,
            grad_norm,
            alpha,
            objective.nfev,
        );
    }

    let result = OptimizeResult {
        x: x.clone(),
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: format!("maximum iterations reached ({maxiter})"),
        nfev: objective.nfev,
        njev,
        nhev: 0,
        nit,
        jac: Some(grad),
        hess_inv: None,
        maxcv: None,
    };
    log_completion(OptimizeMethod::LBfgsB, options, nit, &result);
    Ok(result)
}

/// Project x onto box constraints.
fn project_onto_bounds(x: &mut [f64], bounds: &[Bound]) {
    for (xi, bound) in x.iter_mut().zip(bounds.iter()) {
        if let Some(lo) = bound.0 {
            *xi = xi.max(lo);
        }
        if let Some(hi) = bound.1 {
            *xi = xi.min(hi);
        }
    }
}

/// Compute projected gradient: zero out components at active bounds.
fn projected_gradient(x: &[f64], grad: &[f64], bounds: &[Bound]) -> Vec<f64> {
    let mut pg = grad.to_vec();
    for (i, (xi, bound)) in x.iter().zip(bounds.iter()).enumerate() {
        if let Some(lo) = bound.0
            && (*xi - lo).abs() < 1e-14
            && pg[i] > 0.0
        {
            pg[i] = 0.0;
        }
        if let Some(hi) = bound.1
            && (*xi - hi).abs() < 1e-14
            && pg[i] < 0.0
        {
            pg[i] = 0.0;
        }
    }
    pg
}

/// L-BFGS two-loop recursion to compute H*g.
fn lbfgs_two_loop(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
) -> Vec<f64> {
    let k = s_hist.len();
    if k == 0 {
        return grad.to_vec();
    }

    let mut q = grad.to_vec();
    let mut alpha_cache = vec![0.0; k];

    // Forward loop
    for i in (0..k).rev() {
        alpha_cache[i] = rho_hist[i] * dot(&s_hist[i], &q);
        q = add_scaled(&q, &y_hist[i], -alpha_cache[i]);
    }

    // Initial Hessian scaling: H0 = (s^T y / y^T y) * I
    let sy = dot(&s_hist[k - 1], &y_hist[k - 1]);
    let yy = dot(&y_hist[k - 1], &y_hist[k - 1]);
    let gamma = if yy > 1e-15 { sy / yy } else { 1.0 };
    let mut r = scale_vector(&q, gamma);

    // Backward loop
    for i in 0..k {
        let beta = rho_hist[i] * dot(&y_hist[i], &r);
        r = add_scaled(&r, &s_hist[i], alpha_cache[i] - beta);
    }

    r
}

fn push_lbfgs_history(
    s_hist: &mut Vec<Vec<f64>>,
    y_hist: &mut Vec<Vec<f64>>,
    rho_hist: &mut Vec<f64>,
    s: Vec<f64>,
    y: Vec<f64>,
    sy: f64,
    max_m: usize,
) {
    if s_hist.len() >= max_m {
        s_hist.remove(0);
        y_hist.remove(0);
        rho_hist.remove(0);
    }
    s_hist.push(s);
    y_hist.push(y);
    rho_hist.push(1.0 / sy);
}

/// Newton-CG method: Newton's method with CG inner solver for the Newton equation.
///
/// Matches `scipy.optimize.minimize(f, x0, method='Newton-CG')`.
/// Uses finite-difference Hessian-vector products and CG to approximately
/// solve H*d = -g at each outer iteration.
pub fn newton_cg<F>(
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
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((2000 * n).max(400));
    let eps = options.gradient_eps;
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = match objective.eval(&x) {
        Ok(v) => v,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };
    let mut njev = 0usize;
    let mut grad = match finite_diff_gradient(&mut objective, &x, eps) {
        Ok(v) => {
            njev += 1;
            v
        }
        Err(err) => return Ok(result_from_error(&x, 0, objective.nfev, 0, err)),
    };

    for iteration in 0..maxiter {
        let grad_norm = l2_norm(&grad);
        if grad_norm <= tol {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (Newton-CG)"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit: iteration,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::NewtonCg, options, iteration, &result);
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
                nit: iteration,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::NewtonCg, options, iteration, &result);
            return Ok(result);
        }

        // Inner CG loop to solve H*d = -g approximately
        // Using Hessian-vector products via finite differences: H*v ≈ (∇f(x+εv) - ∇f(x)) / ε
        let cg_tol = grad_norm.min(0.5); // Eisenstat-Walker forcing term
        let (direction, nhvp) = cg_newton_direction(&mut objective, &x, &grad, eps, cg_tol, n)?;
        njev += nhvp; // each HVP requires one gradient evaluation

        // Line search along direction
        let directional_deriv = dot(&grad, &direction);
        if directional_deriv >= 0.0 {
            // Not a descent direction — use steepest descent
            let neg_grad: Vec<f64> = grad.iter().map(|&g| -g).collect();
            let alpha = 1.0 / grad_norm.max(1.0);
            let candidate = add_scaled(&x, &neg_grad, alpha);
            match objective.eval(&candidate) {
                Ok(fv) => {
                    x = candidate;
                    f = fv;
                }
                Err(err) => {
                    return Ok(result_from_error(&x, iteration, objective.nfev, njev, err));
                }
            }
        } else {
            // Armijo backtracking
            let c1 = 1e-4;
            let mut alpha = 1.0;
            let mut accepted = false;
            for _ in 0..24 {
                let candidate = add_scaled(&x, &direction, alpha);
                match objective.eval(&candidate) {
                    Ok(fv) => {
                        if fv <= f + c1 * alpha * directional_deriv {
                            x = candidate;
                            f = fv;
                            accepted = true;
                            break;
                        }
                    }
                    Err(err) => {
                        return Ok(result_from_error(&x, iteration, objective.nfev, njev, err));
                    }
                }
                alpha *= 0.5;
                if alpha < 1e-12 {
                    break;
                }
            }
            if !accepted {
                break;
            }
        }

        // Update gradient
        grad = match finite_diff_gradient(&mut objective, &x, eps) {
            Ok(v) => {
                njev += 1;
                v
            }
            Err(err) => return Ok(result_from_error(&x, iteration, objective.nfev, njev, err)),
        };

        log_iteration(
            OptimizeMethod::NewtonCg,
            options,
            iteration,
            f,
            l2_norm(&grad),
            1.0,
            objective.nfev,
        );
    }

    let result = OptimizeResult {
        x,
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: format!("maximum iterations reached ({maxiter})"),
        nfev: objective.nfev,
        njev,
        nhev: 0,
        nit: maxiter,
        jac: Some(grad),
        hess_inv: None,
        maxcv: None,
    };
    log_completion(OptimizeMethod::NewtonCg, options, maxiter, &result);
    Ok(result)
}

/// Inner CG solver for Newton equation H*d = -g.
/// Uses Hessian-vector products via finite differences.
/// Returns (direction, number_of_hvps).
fn cg_newton_direction<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    grad: &[f64],
    eps: f64,
    tol: f64,
    n: usize,
) -> Result<(Vec<f64>, usize), OptError>
where
    F: Fn(&[f64]) -> f64,
{
    // CG to solve H*d = -g
    let neg_g: Vec<f64> = grad.iter().map(|&g| -g).collect();
    let mut d = vec![0.0; n];
    let mut r = neg_g.clone(); // r = -g - H*d = -g (since d=0)
    let mut p = r.clone();
    let mut rs = dot(&r, &r);
    let mut nhvp = 0usize;

    let max_cg_iter = n.min(20);
    let target = tol * tol * dot(grad, grad);

    for _ in 0..max_cg_iter {
        if rs < target {
            break;
        }

        // Hessian-vector product: H*p ≈ (∇f(x + ε*p) - ∇f(x)) / ε
        let hp = hessian_vector_product(objective, x, grad, &p, eps)?;
        nhvp += 1;

        let p_hp = dot(&p, &hp);
        if p_hp <= 0.0 {
            // Negative curvature — use current d if nonzero, else use steepest descent
            if dot(&d, &d) > 0.0 {
                return Ok((d, nhvp));
            }
            return Ok((neg_g, nhvp));
        }

        let alpha = rs / p_hp;
        for i in 0..n {
            d[i] += alpha * p[i];
            r[i] -= alpha * hp[i];
        }

        let rs_new = dot(&r, &r);
        let beta = rs_new / rs;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs = rs_new;
    }

    Ok((d, nhvp))
}

/// Compute Hessian-vector product H*v via finite differences of gradient.
fn hessian_vector_product<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    grad_at_x: &[f64],
    v: &[f64],
    eps: f64,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let v_norm = l2_norm(v);
    if v_norm < f64::EPSILON {
        return Ok(vec![0.0; x.len()]);
    }
    let step = eps * (1.0 + l2_norm(x)) / v_norm;
    let x_pert = add_scaled(x, v, step);
    let grad_pert = finite_diff_gradient(objective, &x_pert, eps)?;
    let mut hv = Vec::with_capacity(x.len());
    for (gp, g) in grad_pert.iter().zip(grad_at_x.iter()) {
        hv.push((gp - g) / step);
    }
    Ok(hv)
}

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
    let mut x_perturbed = x.to_vec();
    for (idx, component) in x.iter().enumerate() {
        let step = gradient_eps * (1.0 + component.abs());

        let original = x_perturbed[idx];
        x_perturbed[idx] = original + step;
        let f_plus = objective.eval(&x_perturbed)?;

        x_perturbed[idx] = original - step;
        let f_minus = objective.eval(&x_perturbed)?;

        x_perturbed[idx] = original; // restore
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

    // H_new = (I - rho*s*y^T) * H * (I - rho*y*s^T) + rho*s*s^T
    //
    // Let A = (I - rho*s*y^T) * H = H - rho*s*(y^T * H)
    // Then H_new = A * (I - rho*y*s^T) + rho*s*s^T = A - rho*(A*y)*s^T + rho*s*s^T

    // 1. v^T = y^T * H  (O(n^2))
    let mut v = vec![0.0; n];
    for j in 0..n {
        for i in 0..n {
            v[j] += y[i] * h_inv[i][j];
        }
    }

    // 2. A = H - rho * s * v^T  (O(n^2))
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = h_inv[i][j] - rho * s[i] * v[j];
        }
    }

    // 3. u = A * y  (O(n^2))
    let mut u = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            u[i] += a[i][j] * y[j];
        }
    }

    // 4. H_new = A - rho * u * s^T + rho * s * s^T (O(n^2))
    let mut h_new = a;
    for i in 0..n {
        for j in 0..n {
            h_new[i][j] += rho * (s[i] * s[j] - u[i] * s[j]);
        }
    }

    h_new
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

// ══════════════════════════════════════════════════════════════════════
// Scalar Minimization — Public API
// ══════════════════════════════════════════════════════════════════════

/// Options for scalar (1-D) minimization.
#[derive(Debug, Clone, Copy)]
pub struct MinimizeScalarOptions {
    /// Convergence tolerance on x.
    pub tol: f64,
    /// Maximum number of iterations.
    pub maxiter: usize,
}

impl Default for MinimizeScalarOptions {
    fn default() -> Self {
        Self {
            tol: 1.48e-8, // matches SciPy's default
            maxiter: 500,
        }
    }
}

/// Result of scalar minimization.
#[derive(Debug, Clone, PartialEq)]
pub struct MinimizeScalarResult {
    /// The solution (minimizer).
    pub x: f64,
    /// Function value at the minimizer.
    pub fun: f64,
    /// Whether the optimizer converged.
    pub success: bool,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Number of iterations.
    pub nit: usize,
}

/// Minimize a scalar function using Brent's method.
///
/// Finds a local minimum of `f` within the bracket `(a, b)`.
/// Optionally, an interior point `xatol` can be provided.
/// Matches `scipy.optimize.minimize_scalar(f, bracket=(a, b), method='brent')`.
pub fn minimize_scalar<F>(
    f: F,
    bracket: (f64, f64),
    options: MinimizeScalarOptions,
) -> Result<MinimizeScalarResult, OptError>
where
    F: Fn(f64) -> f64,
{
    let (mut a, mut b) = bracket;
    if !a.is_finite() || !b.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: "bracket bounds must be finite".to_string(),
        });
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    if (a - b).abs() < f64::EPSILON {
        return Err(OptError::InvalidBounds {
            detail: "bracket bounds must not be equal".to_string(),
        });
    }

    // Brent's method with golden section and parabolic interpolation
    let golden_ratio = 0.5 * (3.0 - 5.0_f64.sqrt()); // ~0.381966

    let mut x = a + golden_ratio * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut nfev = 1;

    let mut d = 0.0_f64;
    let mut e = 0.0_f64;

    for nit in 0..options.maxiter {
        let midpoint = 0.5 * (a + b);
        let tol1 = options.tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        // Convergence check
        if (x - midpoint).abs() <= (tol2 - 0.5 * (b - a)) {
            return Ok(MinimizeScalarResult {
                x,
                fun: fx,
                success: true,
                nfev,
                nit,
            });
        }

        // Try parabolic interpolation
        let mut use_golden = true;

        if e.abs() > tol1 {
            // Parabolic fit through (v, fv), (w, fw), (x, fx)
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            let mut q_val = 2.0 * (q - r);

            if q_val > 0.0 {
                p = -p;
            } else {
                q_val = -q_val;
            }

            if p.abs() < (0.5 * q_val * e).abs() && p > q_val * (a - x) && p < q_val * (b - x) {
                // Accept parabolic step
                d = p / q_val;
                let u_test = x + d;
                if (u_test - a) < tol2 || (b - u_test) < tol2 {
                    d = if x < midpoint { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }

        if use_golden {
            // Golden section step
            e = if x < midpoint { b - x } else { a - x };
            d = golden_ratio * e;
        } else {
            e = d;
        }

        // Evaluate at new point
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = f(u);
        nfev += 1;

        // Update bracket
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || (w - x).abs() < f64::EPSILON {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || (v - x).abs() < f64::EPSILON || (v - w).abs() < f64::EPSILON {
                v = u;
                fv = fu;
            }
        }
    }

    Ok(MinimizeScalarResult {
        x,
        fun: fx,
        success: false,
        nfev,
        nit: options.maxiter,
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
        ConvergenceStatus, MinimizeOptions, MinimizeScalarOptions, OptError, OptimizeMethod,
        OptimizeResult, bfgs, cg_pr_plus, minimize, minimize_scalar, powell, take_optimize_traces,
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

    // ── minimize_scalar tests ───────────────────────────────────────

    #[test]
    fn minimize_scalar_quadratic() {
        // f(x) = (x - 3)^2, minimum at x = 3
        let result = minimize_scalar(
            |x| (x - 3.0).powi(2),
            (0.0, 10.0),
            MinimizeScalarOptions::default(),
        )
        .expect("minimize_scalar works");
        assert!(result.success, "should converge");
        assert!(
            (result.x - 3.0).abs() < 1e-6,
            "minimizer should be near 3, got {}",
            result.x
        );
        assert!(result.fun < 1e-10, "minimum value should be near 0");
    }

    #[test]
    fn minimize_scalar_cos() {
        // f(x) = cos(x), minimum at x = pi in [2, 4]
        let result = minimize_scalar(f64::cos, (2.0, 4.0), MinimizeScalarOptions::default())
            .expect("minimize_scalar works");
        assert!(result.success);
        assert!(
            (result.x - std::f64::consts::PI).abs() < 1e-6,
            "minimizer should be near pi, got {}",
            result.x
        );
        assert!(
            (result.fun - (-1.0)).abs() < 1e-10,
            "min value should be -1"
        );
    }

    #[test]
    fn minimize_scalar_linear_converges() {
        // f(x) = x, monotone, should find the left endpoint
        let result = minimize_scalar(|x| x, (0.0, 10.0), MinimizeScalarOptions::default())
            .expect("minimize_scalar works");
        assert!(result.x < 1.0, "should be near 0, got {}", result.x);
    }

    #[test]
    fn minimize_scalar_equal_bounds_error() {
        let err = minimize_scalar(|x| x * x, (5.0, 5.0), MinimizeScalarOptions::default())
            .expect_err("equal bounds");
        assert!(matches!(err, OptError::InvalidBounds { .. }));
    }

    #[test]
    fn minimize_scalar_nan_bounds_error() {
        let err = minimize_scalar(|x| x * x, (f64::NAN, 5.0), MinimizeScalarOptions::default())
            .expect_err("nan bounds");
        assert!(matches!(err, OptError::NonFiniteInput { .. }));
    }

    #[test]
    fn minimize_scalar_reversed_bracket() {
        // (10, 0) should be auto-swapped to (0, 10)
        let result = minimize_scalar(
            |x| (x - 3.0).powi(2),
            (10.0, 0.0),
            MinimizeScalarOptions::default(),
        )
        .expect("minimize_scalar works");
        assert!(result.success);
        assert!(
            (result.x - 3.0).abs() < 1e-6,
            "minimizer should be near 3, got {}",
            result.x
        );
    }

    // ── Nelder-Mead tests ───────────────────────────────────────────

    #[test]
    fn nelder_mead_sphere_converges() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            tol: Some(1e-8),
            maxiter: Some(1000),
            maxfev: Some(5000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[2.0, -3.0], options).expect("minimize executes");
        assert!(result.success, "should converge: {}", result.message);
        assert!(result.x.iter().all(|v| v.abs() < 1e-4), "x={:?}", result.x);
        assert!(result.fun.unwrap() < 1e-8);
        push_test_log(
            "nelder-mead-sphere",
            "nelder_mead",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            200,
        );
    }

    #[test]
    fn nelder_mead_rosenbrock() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            tol: Some(1e-8),
            maxiter: Some(5000),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[-1.0, 1.0], options).expect("minimize executes");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 1e-3 && (result.x[1] - 1.0).abs() < 1e-3,
            "x={:?}",
            result.x
        );
    }

    #[test]
    fn nelder_mead_1d() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            tol: Some(1e-10),
            ..MinimizeOptions::default()
        };
        let result = minimize(one_dim_quadratic, &[5.0], options).expect("minimize executes");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 1.5).abs() < 1e-4,
            "minimizer should be near 1.5, got {}",
            result.x[0]
        );
    }

    #[test]
    fn nelder_mead_flat_function() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            maxiter: Some(500),
            ..MinimizeOptions::default()
        };
        let result = minimize(zero_function, &[1.0, 2.0], options).expect("minimize executes");
        // Should converge since f is constant everywhere
        assert!(result.success, "{}", result.message);
        assert_eq!(result.fun, Some(0.0));
    }

    #[test]
    fn nelder_mead_max_iter_reached() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            maxiter: Some(2),
            maxfev: Some(10_000),
            tol: Some(1e-15),
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[5.0, 5.0], options).expect("minimize executes");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    }

    #[test]
    fn nelder_mead_callback_stops() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            callback: Some(callback_record_and_stop),
            ..MinimizeOptions::default()
        };
        callback_points().lock().unwrap().clear();
        let result = minimize(sphere, &[2.0, -3.0], options).expect("minimize executes");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::CallbackStop);
    }

    #[test]
    fn nelder_mead_empty_x0_rejected() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            ..MinimizeOptions::default()
        };
        let err = minimize(sphere, &[], options).expect_err("should reject empty");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn nelder_mead_nonfinite_x0_rejected() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            ..MinimizeOptions::default()
        };
        let err = minimize(sphere, &[f64::NAN, 1.0], options).expect_err("should reject NaN");
        assert!(matches!(err, OptError::NonFiniteInput { .. }));
    }

    #[test]
    fn nelder_mead_himmelblau() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            tol: Some(1e-8),
            maxiter: Some(2000),
            maxfev: Some(10_000),
            ..MinimizeOptions::default()
        };
        let result = minimize(himmelblau, &[0.0, 0.0], options).expect("minimize executes");
        assert!(result.success, "should converge: {}", result.message);
        // Himmelblau has 4 minima, all with f=0
        assert!(result.fun.unwrap() < 1e-6, "f={:?}", result.fun);
    }

    #[test]
    fn nelder_mead_higher_dim() {
        // 5-D sphere
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            tol: Some(1e-6),
            maxiter: Some(5000),
            maxfev: Some(50_000),
            ..MinimizeOptions::default()
        };
        let result =
            minimize(sphere, &[1.0, -2.0, 3.0, -4.0, 5.0], options).expect("minimize executes");
        assert!(result.success, "should converge: {}", result.message);
        assert!(result.x.iter().all(|v| v.abs() < 1e-3), "x={:?}", result.x);
    }

    #[test]
    fn nelder_mead_traces_are_emitted() {
        let _ = take_optimize_traces(); // clear
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            maxiter: Some(10),
            ..MinimizeOptions::default()
        };
        let _ = minimize(sphere, &[1.0, 1.0], options);
        let traces = take_optimize_traces();
        let nm_traces: Vec<_> = traces
            .iter()
            .filter(|t| t.method == OptimizeMethod::NelderMead)
            .collect();
        assert!(
            !nm_traces.is_empty(),
            "nelder-mead should emit trace entries"
        );
    }

    // ── L-BFGS-B tests ─────────────────────────────────────────────

    #[test]
    fn lbfgsb_unconstrained_sphere() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            tol: Some(1e-8),
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[2.0, -3.0], options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(result.x.iter().all(|v| v.abs() < 1e-4), "x={:?}", result.x);
    }

    #[test]
    fn lbfgsb_with_bounds() {
        use crate::minimize::lbfgsb;
        // Minimize (x-3)^2 + (y-3)^2 with bounds x in [0, 2], y in [0, 2]
        // Optimum should be at (2, 2)
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2);
        let bounds = vec![(Some(0.0), Some(2.0)), (Some(0.0), Some(2.0))];
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            tol: Some(1e-8),
            ..MinimizeOptions::default()
        };
        let result = lbfgsb(&f, &[0.0, 0.0], options, Some(&bounds)).expect("lbfgsb");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 0.01,
            "x[0] should be at upper bound 2, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.01,
            "x[1] should be at upper bound 2, got {}",
            result.x[1]
        );
    }

    #[test]
    fn lbfgsb_lower_bound_only() {
        use crate::minimize::lbfgsb;
        // Minimize (x+5)^2 with lower bound x >= 0. Optimum at x=0.
        let f = |x: &[f64]| (x[0] + 5.0).powi(2);
        let bounds = vec![(Some(0.0), None)];
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            tol: Some(1e-8),
            ..MinimizeOptions::default()
        };
        let result = lbfgsb(&f, &[10.0], options, Some(&bounds)).expect("lbfgsb");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            result.x[0].abs() < 0.01,
            "x should be at lower bound 0, got {}",
            result.x[0]
        );
    }

    #[test]
    fn lbfgsb_rosenbrock_unconstrained() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            tol: Some(1e-6),
            maxiter: Some(2000),
            maxfev: Some(20_000),
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[0.0, 0.0], options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 0.01 && (result.x[1] - 1.0).abs() < 0.01,
            "x={:?}",
            result.x
        );
    }

    // ── Newton-CG tests ─────────────────────────────────────────────

    #[test]
    fn newton_cg_sphere_converges() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NewtonCg),
            tol: Some(1e-8),
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[2.0, -3.0], options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(result.x.iter().all(|v| v.abs() < 1e-3), "x={:?}", result.x);
    }

    #[test]
    fn newton_cg_rosenbrock() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NewtonCg),
            tol: Some(1e-6),
            maxiter: Some(500),
            maxfev: Some(50_000),
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[0.0, 0.0], options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 0.05 && (result.x[1] - 1.0).abs() < 0.05,
            "x={:?}",
            result.x
        );
    }

    #[test]
    fn newton_cg_1d_quadratic() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NewtonCg),
            tol: Some(1e-10),
            ..MinimizeOptions::default()
        };
        let result = minimize(one_dim_quadratic, &[5.0], options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 1.5).abs() < 1e-4,
            "minimizer should be 1.5, got {}",
            result.x[0]
        );
    }
}
