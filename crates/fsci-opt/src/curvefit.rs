#![forbid(unsafe_code)]

//! Nonlinear least-squares curve fitting (Levenberg-Marquardt).
//!
//! Provides `curve_fit` (matching `scipy.optimize.curve_fit`) and `least_squares`
//! (matching `scipy.optimize.least_squares`).

use fsci_runtime::RuntimeMode;

use crate::types::OptError;

/// Result from `least_squares`.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastSquaresResult {
    /// Solution parameters.
    pub x: Vec<f64>,
    /// Residual vector at the solution.
    pub fun: Vec<f64>,
    /// Sum of squared residuals (cost = 0.5 * ||fun||^2).
    pub cost: f64,
    /// Whether the solver converged.
    pub success: bool,
    /// Status message.
    pub message: String,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Number of Jacobian evaluations.
    pub njev: usize,
    /// Number of iterations.
    pub nit: usize,
    /// Jacobian at the solution (row-major, m rows x n cols).
    pub jac: Vec<Vec<f64>>,
}

/// Options for `least_squares`.
#[derive(Debug, Clone, Copy)]
pub struct LeastSquaresOptions {
    /// Convergence tolerance on the gradient (gtol).
    pub gtol: f64,
    /// Convergence tolerance on the change in parameters (xtol).
    pub xtol: f64,
    /// Convergence tolerance on the change in cost (ftol).
    pub ftol: f64,
    /// Maximum number of iterations.
    pub max_nfev: Option<usize>,
    /// Finite-difference step for Jacobian approximation.
    pub diff_step: f64,
    /// Runtime mode.
    pub mode: RuntimeMode,
}

impl Default for LeastSquaresOptions {
    fn default() -> Self {
        Self {
            gtol: 1.0e-8,
            xtol: 1.0e-8,
            ftol: 1.0e-8,
            max_nfev: None,
            diff_step: 1.4901161193847656e-8, // sqrt(machine eps)
            mode: RuntimeMode::Strict,
        }
    }
}

/// Result from `curve_fit`.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveFitResult {
    /// Optimal parameters.
    pub popt: Vec<f64>,
    /// Estimated covariance matrix of parameters (n x n).
    pub pcov: Vec<Vec<f64>>,
    /// Underlying least-squares result.
    pub ls_result: LeastSquaresResult,
}

/// Options for `curve_fit`.
#[derive(Debug, Clone, Default)]
pub struct CurveFitOptions {
    /// Initial guess for parameters.
    pub p0: Option<Vec<f64>>,
    /// Least-squares options.
    pub ls_options: LeastSquaresOptions,
    /// Whether to compute absolute sigma (unscaled covariance).
    pub absolute_sigma: bool,
}

/// Solve a nonlinear least-squares problem using the Levenberg-Marquardt algorithm.
///
/// Finds `x` that minimizes `0.5 * sum(residuals(x)^2)` where `residuals` maps
/// parameters to a vector of residuals.
///
/// Equivalent to `scipy.optimize.least_squares` with `method='lm'`.
pub fn least_squares<F>(
    residuals: F,
    x0: &[f64],
    options: LeastSquaresOptions,
) -> Result<LeastSquaresResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("x0 must have at least one element"),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x0 must not contain NaN or Inf"),
        });
    }

    let max_nfev = options.max_nfev.unwrap_or(100 * (n + 1));

    let mut x = x0.to_vec();
    let mut r = residuals(&x);
    let m = r.len();
    let mut nfev = 1usize;
    let mut njev = 0usize;

    if m < n {
        return Err(OptError::InvalidArgument {
            detail: format!("number of residuals ({m}) must be >= number of parameters ({n})"),
        });
    }

    // Check for non-finite residuals
    if r.iter().any(|v| !v.is_finite()) {
        return match options.mode {
            RuntimeMode::Strict => Err(OptError::NonFiniteInput {
                detail: String::from("residuals contain non-finite values at x0"),
            }),
            RuntimeMode::Hardened => Err(OptError::NonFiniteInput {
                detail: String::from("hardened mode rejects non-finite residuals"),
            }),
        };
    }

    let mut cost = 0.5 * dot_vec(&r, &r);
    let mut jac = finite_diff_jacobian(&residuals, &x, &r, options.diff_step);
    nfev += n;
    njev += 1;

    // Initial damping parameter (Marquardt strategy)
    let mut mu = 1.0e-3 * max_diag_jtj(&jac);
    if mu == 0.0 {
        mu = 1.0;
    }
    let mut nu = 2.0;
    let mut nit = 0usize;

    for _ in 0..max_nfev {
        nit += 1;

        // Compute J^T * r and J^T * J
        let jtj = jtj_matrix(&jac);
        let jtr = jt_vec(&jac, &r);

        // Check gradient convergence: ||J^T r||_inf <= gtol
        let grad_inf = jtr.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if grad_inf <= options.gtol {
            return Ok(LeastSquaresResult {
                x,
                fun: r,
                cost,
                success: true,
                message: String::from("gradient converged (||J^T r||_inf <= gtol)"),
                nfev,
                njev,
                nit,
                jac,
            });
        }

        // Solve (J^T J + mu I) * step = -J^T r
        let step = solve_damped_normal_equations(&jtj, &jtr, mu, n);

        let step_norm = l2_norm(&step);
        let x_norm = l2_norm(&x);

        // Check parameter convergence
        if step_norm <= options.xtol * (options.xtol + x_norm) {
            return Ok(LeastSquaresResult {
                x,
                fun: r,
                cost,
                success: true,
                message: String::from("parameter change converged (xtol)"),
                nfev,
                njev,
                nit,
                jac,
            });
        }

        // Trial step
        let x_new: Vec<f64> = x.iter().zip(step.iter()).map(|(a, b)| a + b).collect();
        let r_new = residuals(&x_new);
        nfev += 1;
        let cost_new = 0.5 * dot_vec(&r_new, &r_new);

        // Gain ratio (actual reduction / predicted reduction)
        let predicted_reduction = {
            let jstep = mat_vec(&jac, &step);
            -dot_vec(&jtr, &step) - 0.5 * dot_vec(&jstep, &jstep)
        };

        let actual_reduction = cost - cost_new;

        if predicted_reduction > 0.0 {
            let rho = actual_reduction / predicted_reduction;

            if rho > 0.25 {
                // Accept step
                x = x_new;
                r = r_new;
                let old_cost = cost;
                cost = cost_new;

                // Check cost convergence
                let cost_change = (old_cost - cost) / (1.0 + cost);
                if cost_change.abs() <= options.ftol {
                    jac = finite_diff_jacobian(&residuals, &x, &r, options.diff_step);
                    nfev += n;
                    njev += 1;
                    return Ok(LeastSquaresResult {
                        x,
                        fun: r,
                        cost,
                        success: true,
                        message: String::from("cost change converged (ftol)"),
                        nfev,
                        njev,
                        nit,
                        jac,
                    });
                }

                // Recompute Jacobian
                jac = finite_diff_jacobian(&residuals, &x, &r, options.diff_step);
                nfev += n;
                njev += 1;

                // Decrease damping
                mu *= f64::max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
            } else {
                // Reject step, increase damping
                mu *= nu;
                nu *= 2.0;
            }
        } else {
            // Reject step, increase damping
            mu *= nu;
            nu *= 2.0;
        }

        if nfev >= max_nfev {
            return Ok(LeastSquaresResult {
                x,
                fun: r,
                cost,
                success: false,
                message: format!("max function evaluations exceeded ({max_nfev})"),
                nfev,
                njev,
                nit,
                jac,
            });
        }
    }

    Ok(LeastSquaresResult {
        x,
        fun: r,
        cost,
        success: false,
        message: String::from("max iterations exceeded"),
        nfev,
        njev,
        nit,
        jac,
    })
}

/// Nonlinear curve fitting.
///
/// Finds the parameters `popt` such that `f(xdata, popt)` best fits `ydata`
/// in the least-squares sense.
///
/// Equivalent to `scipy.optimize.curve_fit`.
///
/// # Arguments
/// * `f` - Model function `f(x, params) -> y` where `x` is a single data point
/// * `xdata` - Independent variable data
/// * `ydata` - Dependent variable data (same length as xdata)
/// * `options` - Curve fitting options (initial guess, tolerances, etc.)
pub fn curve_fit<F>(
    f: F,
    xdata: &[f64],
    ydata: &[f64],
    options: CurveFitOptions,
) -> Result<CurveFitResult, OptError>
where
    F: Fn(f64, &[f64]) -> f64,
{
    if xdata.len() != ydata.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "xdata and ydata must have same length (got {} and {})",
                xdata.len(),
                ydata.len()
            ),
        });
    }
    if xdata.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("xdata must not be empty"),
        });
    }

    let p0 = options.p0.ok_or_else(|| OptError::InvalidArgument {
        detail: String::from("p0 (initial parameter guess) is required"),
    })?;
    let n_params = p0.len();

    if xdata.len() < n_params {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "number of data points ({}) must be >= number of parameters ({n_params})",
                xdata.len()
            ),
        });
    }

    // Build residual function: r_i(p) = f(x_i, p) - y_i
    let residuals = |params: &[f64]| -> Vec<f64> {
        xdata
            .iter()
            .zip(ydata.iter())
            .map(|(&xi, &yi)| f(xi, params) - yi)
            .collect()
    };

    let ls_result = least_squares(residuals, &p0, options.ls_options)?;

    // Compute covariance: pcov = (J^T J)^{-1} * s^2
    // where s^2 = cost * 2 / (m - n) for relative sigma
    let pcov = compute_covariance(
        &ls_result.jac,
        ls_result.cost,
        xdata.len(),
        n_params,
        options.absolute_sigma,
    );

    Ok(CurveFitResult {
        popt: ls_result.x.clone(),
        pcov,
        ls_result,
    })
}

// ────────────────────────── helpers ──────────────────────────

fn dot_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f64]) -> f64 {
    dot_vec(v, v).sqrt()
}

/// Compute finite-difference Jacobian (m x n) at point x with residuals r.
fn finite_diff_jacobian<F>(residuals: &F, x: &[f64], r0: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let m = r0.len();
    let mut jac = vec![vec![0.0; n]; m];
    let mut x_perturbed = x.to_vec();

    for j in 0..n {
        let step = eps * (1.0 + x[j].abs());
        let original = x_perturbed[j];
        x_perturbed[j] += step;
        let r_plus = residuals(&x_perturbed);
        x_perturbed[j] = original; // restore
        for i in 0..m {
            jac[i][j] = (r_plus[i] - r0[i]) / step;
        }
    }
    jac
}

/// Compute J^T * J (n x n).
fn jtj_matrix(jac: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = jac.first().map_or(0, Vec::len);
    let mut jtj = vec![vec![0.0; n]; n];
    for row in jac {
        for i in 0..n {
            for j in i..n {
                let v = row[i] * row[j];
                jtj[i][j] += v;
                if i != j {
                    jtj[j][i] += v;
                }
            }
        }
    }
    jtj
}

/// Compute J^T * v.
fn jt_vec(jac: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let n = jac.first().map_or(0, Vec::len);
    let mut result = vec![0.0; n];
    for (row, &vi) in jac.iter().zip(v.iter()) {
        for (j, &jval) in row.iter().enumerate() {
            result[j] += jval * vi;
        }
    }
    result
}

/// Compute J * v.
fn mat_vec(jac: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    jac.iter().map(|row| dot_vec(row, v)).collect()
}

/// Max diagonal element of J^T * J.
fn max_diag_jtj(jac: &[Vec<f64>]) -> f64 {
    let n = jac.first().map_or(0, Vec::len);
    let mut max_val = 0.0_f64;
    for j in 0..n {
        let mut diag = 0.0;
        for row in jac {
            diag += row[j] * row[j];
        }
        max_val = max_val.max(diag);
    }
    max_val
}

/// Solve (A + mu * I) * x = -b via Cholesky-like approach.
/// Falls back to diagonal if matrix is singular.
fn solve_damped_normal_equations(jtj: &[Vec<f64>], jtr: &[f64], mu: f64, n: usize) -> Vec<f64> {
    // Build damped matrix
    let mut a = jtj.to_vec();
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += mu;
    }

    // Solve via Cholesky decomposition
    if let Some(step) = cholesky_solve(&a, jtr, n) {
        // Return -step (we solve A*step = -jtr, but passed jtr not -jtr)
        step.iter().map(|v| -v).collect()
    } else {
        // Fallback: diagonal solve
        (0..n).map(|i| -jtr[i] / (jtj[i][i] + mu)).collect()
    }
}

/// Compute Cholesky decomposition A = L * L^T.
fn cholesky_decompose(a: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let mut low = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for (&low_ik, &low_jk) in low[i].iter().zip(low[j].iter()).take(j) {
                sum -= low_ik * low_jk;
            }
            if i == j {
                if sum <= 0.0 {
                    return None; // Not positive definite
                }
                low[i][j] = sum.sqrt();
            } else {
                low[i][j] = sum / low[j][j];
            }
        }
    }
    Some(low)
}

/// Solve L * L^T * x = b given L.
fn cholesky_solve_with_l(low: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    // Forward substitution: L * y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= low[i][j] * y[j];
        }
        y[i] = sum / low[i][i];
    }

    // Back substitution: L^T * x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= low[j][i] * x[j];
        }
        x[i] = sum / low[i][i];
    }
    x
}

/// Cholesky decomposition and solve: A * x = b.
fn cholesky_solve(a: &[Vec<f64>], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let low = cholesky_decompose(a, n)?;
    Some(cholesky_solve_with_l(&low, b, n))
}

/// Compute parameter covariance matrix from Jacobian.
fn compute_covariance(
    jac: &[Vec<f64>],
    cost: f64,
    m: usize,
    n: usize,
    absolute_sigma: bool,
) -> Vec<Vec<f64>> {
    let jtj = jtj_matrix(jac);

    // Invert J^T J via Cholesky
    let mut pcov = vec![vec![0.0; n]; n];
    if let Some(low) = cholesky_decompose(&jtj, n) {
        for col in 0..n {
            let mut e = vec![0.0; n];
            e[col] = 1.0;
            let column = cholesky_solve_with_l(&low, &e, n);
            for (row, val) in column.iter().enumerate() {
                pcov[row][col] = *val;
            }
        }
    } else {
        // If J^T J is singular, return infinity on diagonal
        for row in &mut pcov {
            for val in row.iter_mut() {
                *val = f64::INFINITY;
            }
        }
        return pcov;
    }

    // Scale by residual variance if not absolute_sigma
    if !absolute_sigma && m > n {
        let s2 = 2.0 * cost / (m - n) as f64;
        for row in &mut pcov {
            for val in row.iter_mut() {
                *val *= s2;
            }
        }
    }

    pcov
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_squares_linear_fit() {
        // Fit y = a*x + b to data
        let xdata = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ydata = [2.1, 3.9, 6.1, 7.9, 10.1]; // approx y = 2x + 0.1

        let residuals = |p: &[f64]| -> Vec<f64> {
            xdata
                .iter()
                .zip(ydata.iter())
                .map(|(&x, &y)| p[0] * x + p[1] - y)
                .collect()
        };

        let result = least_squares(residuals, &[1.0, 0.0], LeastSquaresOptions::default())
            .expect("should converge");
        assert!(result.success, "{}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 0.1,
            "slope ~ 2.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.1).abs() < 0.2,
            "intercept ~ 0.1, got {}",
            result.x[1]
        );
    }

    #[test]
    fn least_squares_exponential_decay() {
        // Fit y = a * exp(-b * x)
        let xdata: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let ydata: Vec<f64> = xdata.iter().map(|&x| 3.0 * (-0.5 * x).exp()).collect();

        let residuals = |p: &[f64]| -> Vec<f64> {
            xdata
                .iter()
                .zip(ydata.iter())
                .map(|(&x, &y)| p[0] * (-p[1] * x).exp() - y)
                .collect()
        };

        let result = least_squares(residuals, &[1.0, 1.0], LeastSquaresOptions::default())
            .expect("should converge");
        assert!(result.success, "{}", result.message);
        assert!(
            (result.x[0] - 3.0).abs() < 1.0e-4,
            "a ~ 3.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.5).abs() < 1.0e-4,
            "b ~ 0.5, got {}",
            result.x[1]
        );
    }

    #[test]
    fn least_squares_rejects_empty_x0() {
        let residuals = |_: &[f64]| -> Vec<f64> { vec![] };
        let err = least_squares(residuals, &[], LeastSquaresOptions::default())
            .expect_err("should reject empty x0");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn least_squares_rejects_underdetermined() {
        let residuals = |_: &[f64]| -> Vec<f64> { vec![1.0] };
        let err = least_squares(residuals, &[1.0, 2.0, 3.0], LeastSquaresOptions::default())
            .expect_err("should reject underdetermined");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn curve_fit_quadratic() {
        // Fit y = a*x^2 + b*x + c
        let xdata: Vec<f64> = (-10..=10).map(|i| i as f64).collect();
        let ydata: Vec<f64> = xdata.iter().map(|&x| 2.0 * x * x - 3.0 * x + 1.0).collect();

        let model = |x: f64, p: &[f64]| p[0] * x * x + p[1] * x + p[2];

        let options = CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };

        let result = curve_fit(model, &xdata, &ydata, options).expect("should converge");
        assert!(result.ls_result.success, "{}", result.ls_result.message);
        assert!(
            (result.popt[0] - 2.0).abs() < 1.0e-6,
            "a ~ 2.0, got {}",
            result.popt[0]
        );
        assert!(
            (result.popt[1] - (-3.0)).abs() < 1.0e-6,
            "b ~ -3.0, got {}",
            result.popt[1]
        );
        assert!(
            (result.popt[2] - 1.0).abs() < 1.0e-6,
            "c ~ 1.0, got {}",
            result.popt[2]
        );
    }

    #[test]
    fn curve_fit_sinusoidal() {
        // Fit y = A * sin(omega * x + phi)
        let xdata: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let ydata: Vec<f64> = xdata.iter().map(|&x| 2.5 * (1.5 * x + 0.3).sin()).collect();

        let model = |x: f64, p: &[f64]| p[0] * (p[1] * x + p[2]).sin();

        let options = CurveFitOptions {
            p0: Some(vec![2.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };

        let result = curve_fit(model, &xdata, &ydata, options).expect("should converge");
        assert!(result.ls_result.success, "{}", result.ls_result.message);
        assert!(
            (result.popt[0] - 2.5).abs() < 0.1,
            "A ~ 2.5, got {}",
            result.popt[0]
        );
    }

    #[test]
    fn curve_fit_returns_covariance() {
        let xdata: Vec<f64> = (0..20).map(|i| i as f64).collect();
        // Linear: y = 2x + 1 with noise
        let ydata: Vec<f64> = xdata.iter().map(|&x| 2.0 * x + 1.0).collect();

        let model = |x: f64, p: &[f64]| p[0] * x + p[1];

        let options = CurveFitOptions {
            p0: Some(vec![1.0, 0.0]),
            absolute_sigma: true,
            ..CurveFitOptions::default()
        };

        let result = curve_fit(model, &xdata, &ydata, options).expect("should converge");
        // Covariance should be 2x2 and finite
        assert_eq!(result.pcov.len(), 2);
        assert_eq!(result.pcov[0].len(), 2);
        assert!(result.pcov[0][0].is_finite());
        assert!(result.pcov[1][1].is_finite());
    }

    #[test]
    fn curve_fit_xdata_ydata_length_mismatch() {
        let model = |_x: f64, _p: &[f64]| 0.0;
        let err = curve_fit(
            model,
            &[1.0, 2.0],
            &[1.0],
            CurveFitOptions {
                p0: Some(vec![1.0]),
                ..CurveFitOptions::default()
            },
        )
        .expect_err("should reject mismatched lengths");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn curve_fit_requires_p0() {
        let model = |_x: f64, _p: &[f64]| 0.0;
        let err = curve_fit(model, &[1.0], &[1.0], CurveFitOptions::default())
            .expect_err("should require p0");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn least_squares_gaussian_fit() {
        // Fit y = A * exp(-((x - mu)/sigma)^2 / 2)
        let xdata: Vec<f64> = (-30..=30).map(|i| i as f64 * 0.2).collect();
        let ydata: Vec<f64> = xdata
            .iter()
            .map(|&x| 5.0 * (-((x - 1.0) / 2.0).powi(2) / 2.0).exp())
            .collect();

        let residuals = |p: &[f64]| -> Vec<f64> {
            xdata
                .iter()
                .zip(ydata.iter())
                .map(|(&x, &y)| p[0] * (-((x - p[1]) / p[2]).powi(2) / 2.0).exp() - y)
                .collect()
        };

        let result = least_squares(residuals, &[3.0, 0.0, 1.0], LeastSquaresOptions::default())
            .expect("should converge");
        assert!(result.success, "{}", result.message);
        assert!(
            (result.x[0] - 5.0).abs() < 0.1,
            "A ~ 5.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.1,
            "mu ~ 1.0, got {}",
            result.x[1]
        );
        assert!(
            (result.x[2].abs() - 2.0).abs() < 0.1,
            "|sigma| ~ 2.0, got {}",
            result.x[2]
        );
    }
}
