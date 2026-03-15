#![forbid(unsafe_code)]

//! Boundary Value Problem (BVP) solver using the shooting method.
//!
//! Solves two-point BVPs of the form:
//!   y' = f(t, y),  y(a) partially known, y(b) partially known
//!
//! Uses single shooting with Newton iteration to find missing initial conditions.

use crate::api::{SolveIvpOptions, SolverKind, solve_ivp};
use crate::validation::ToleranceValue;

/// Options for BVP solver.
#[derive(Debug, Clone)]
pub struct BvpOptions {
    /// Tolerance for the boundary condition residual.
    pub tol: f64,
    /// Maximum Newton iterations.
    pub max_iter: usize,
    /// IVP solver tolerances.
    pub rtol: f64,
    pub atol: f64,
}

impl Default for BvpOptions {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 50,
            rtol: 1e-8,
            atol: 1e-10,
        }
    }
}

/// Result of BVP solve.
#[derive(Debug, Clone)]
pub struct BvpResult {
    /// Solution time points.
    pub t: Vec<f64>,
    /// Solution values at each time point (each entry is a state vector).
    pub y: Vec<Vec<f64>>,
    /// Whether the solver converged.
    pub converged: bool,
    /// Number of Newton iterations.
    pub iterations: usize,
    /// Final boundary condition residual norm.
    pub residual: f64,
}

/// Error type for BVP operations.
#[derive(Debug, Clone, PartialEq)]
pub enum BvpError {
    IvpFailed(String),
    DidNotConverge { iterations: usize, residual: f64 },
    InvalidArgument(String),
}

impl std::fmt::Display for BvpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IvpFailed(msg) => write!(f, "IVP solve failed: {msg}"),
            Self::DidNotConverge {
                iterations,
                residual,
            } => write!(
                f,
                "BVP did not converge after {iterations} iterations (residual={residual})"
            ),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for BvpError {}

/// Solve a two-point BVP using single shooting.
///
/// Given:
///   y' = f(t, y)
///   bc(y(a), y(b)) = 0  (boundary condition residual)
///
/// The solver adjusts the initial conditions y(a) until bc(y(a), y(b)) = 0.
///
/// # Arguments
/// * `f` - ODE right-hand side f(t, y) -> y'
/// * `bc` - Boundary condition residual bc(ya, yb) -> residual vector (must have same length as y)
/// * `t_span` - Integration interval (a, b)
/// * `y_guess` - Initial guess for y(a)
/// * `options` - Solver options
pub fn solve_bvp<F, BC>(
    f: &mut F,
    bc: &BC,
    t_span: (f64, f64),
    y_guess: &[f64],
    options: BvpOptions,
) -> Result<BvpResult, BvpError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
    BC: Fn(&[f64], &[f64]) -> Vec<f64>,
{
    let n = y_guess.len();
    if n == 0 {
        return Err(BvpError::InvalidArgument(
            "y_guess must be non-empty".to_string(),
        ));
    }

    let mut y0 = y_guess.to_vec();

    for iteration in 0..options.max_iter {
        // Solve IVP with current y0
        let ivp_result = solve_ivp_internal(f, t_span, &y0, options.rtol, options.atol)?;
        let yb = ivp_result.y.last().unwrap().clone();

        // Evaluate boundary condition residual
        let residual = bc(&y0, &yb);
        let residual_norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();

        if residual_norm < options.tol {
            return Ok(BvpResult {
                t: ivp_result.t,
                y: ivp_result.y,
                converged: true,
                iterations: iteration,
                residual: residual_norm,
            });
        }

        // Compute Jacobian of bc w.r.t. y0 using finite differences
        let mut jac = vec![vec![0.0; n]; n];
        for j in 0..n {
            // Scale perturbation by parameter magnitude for numerical stability
            let eps_j = 1e-7 * (1.0 + y0[j].abs());
            let mut y0_pert = y0.clone();
            y0_pert[j] += eps_j;
            let ivp_pert = solve_ivp_internal(f, t_span, &y0_pert, options.rtol, options.atol)?;
            let yb_pert = ivp_pert.y.last().unwrap();
            let residual_pert = bc(&y0_pert, yb_pert);
            for i in 0..n {
                jac[i][j] = (residual_pert[i] - residual[i]) / eps_j;
            }
        }

        // Newton step: solve J * delta = -residual
        let delta = solve_small_system(&jac, &residual.iter().map(|r| -r).collect::<Vec<_>>());

        // Update y0
        for j in 0..n {
            y0[j] += delta[j];
        }
    }

    // Final evaluation
    let ivp_result = solve_ivp_internal(f, t_span, &y0, options.rtol, options.atol)?;
    let yb = ivp_result.y.last().unwrap();
    let residual = bc(&y0, yb);
    let residual_norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();

    if residual_norm < options.tol {
        Ok(BvpResult {
            t: ivp_result.t,
            y: ivp_result.y,
            converged: true,
            iterations: options.max_iter,
            residual: residual_norm,
        })
    } else {
        Err(BvpError::DidNotConverge {
            iterations: options.max_iter,
            residual: residual_norm,
        })
    }
}

/// Internal IVP solver wrapper.
fn solve_ivp_internal<F>(
    f: &mut F,
    t_span: (f64, f64),
    y0: &[f64],
    rtol: f64,
    atol: f64,
) -> Result<crate::api::SolveIvpResult, BvpError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let result = solve_ivp(
        f,
        &SolveIvpOptions {
            t_span,
            y0,
            method: SolverKind::Rk45,
            rtol,
            atol: ToleranceValue::Scalar(atol),
            ..SolveIvpOptions::default()
        },
    )
    .map_err(|e| BvpError::IvpFailed(format!("{e}")))?;

    if !result.success {
        return Err(BvpError::IvpFailed(result.message));
    }
    Ok(result)
}

/// Solve a small dense linear system Ax = b using Gaussian elimination.
fn solve_small_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return if a[0][0].abs() > f64::EPSILON {
            vec![b[0] / a[0][0]]
        } else {
            vec![0.0]
        };
    }

    // Augmented matrix
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for (row, aug_row) in aug.iter().enumerate().skip(col + 1) {
            if aug_row[col].abs() > max_val {
                max_val = aug_row[col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        if aug[col][col].abs() < f64::EPSILON * 1e6 {
            continue;
        }

        let pivot_row = aug[col].clone();
        for aug_row in aug.iter_mut().skip(col + 1) {
            let factor = aug_row[col] / pivot_row[col];
            for (j, pivot_val) in pivot_row.iter().enumerate().skip(col) {
                aug_row[j] -= factor * pivot_val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < f64::EPSILON * 1e6 {
            continue;
        }
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bvp_linear_ode() {
        // y'' = 0, y(0) = 0, y(1) = 1 => y = t
        // Convert to system: y0' = y1, y1' = 0
        let mut f = |_t: f64, y: &[f64]| vec![y[1], 0.0];
        let bc = |ya: &[f64], yb: &[f64]| vec![ya[0] - 0.0, yb[0] - 1.0];

        let result = solve_bvp(&mut f, &bc, (0.0, 1.0), &[0.0, 0.5], BvpOptions::default())
            .expect("BVP should converge");
        assert!(result.converged, "should converge");
        let y_final = result.y.last().unwrap();
        assert!(
            (y_final[0] - 1.0).abs() < 1e-6,
            "y(1) = {}, expected 1.0",
            y_final[0]
        );
    }

    #[test]
    fn bvp_exponential_boundary() {
        // y' = y, y(0) = 1 => y(t) = exp(t)
        // BC: y(0) = 1 (trivially satisfied by guess)
        let mut f = |_t: f64, y: &[f64]| vec![y[0]];
        let bc = |ya: &[f64], _yb: &[f64]| vec![ya[0] - 1.0];

        let result = solve_bvp(
            &mut f,
            &bc,
            (0.0, 1.0),
            &[1.0], // exact initial condition
            BvpOptions::default(),
        )
        .expect("BVP should converge");
        assert!(result.converged);
        let y_final = result.y.last().unwrap()[0];
        assert!(
            (y_final - std::f64::consts::E).abs() < 1e-4,
            "y(1) = {y_final}, expected e"
        );
    }

    #[test]
    fn bvp_second_order_quadratic() {
        // y'' = 2, y(0) = 0, y(1) = 1 => y = t² (exact for this BC)
        // System: y0' = y1, y1' = 2
        let mut f = |_t: f64, y: &[f64]| vec![y[1], 2.0];
        let bc = |ya: &[f64], yb: &[f64]| vec![ya[0] - 0.0, yb[0] - 1.0];

        let result = solve_bvp(&mut f, &bc, (0.0, 1.0), &[0.0, 0.0], BvpOptions::default())
            .expect("BVP should converge");
        assert!(result.converged);
        let y_final = result.y.last().unwrap();
        assert!(
            (y_final[0] - 1.0).abs() < 1e-4,
            "y(1) = {}, expected 1.0",
            y_final[0]
        );
    }

    #[test]
    fn bvp_empty_guess_rejected() {
        let mut f = |_t: f64, _y: &[f64]| vec![];
        let bc = |_ya: &[f64], _yb: &[f64]| vec![];
        let err = solve_bvp(&mut f, &bc, (0.0, 1.0), &[], BvpOptions::default())
            .expect_err("empty guess");
        assert!(matches!(err, BvpError::InvalidArgument(_)));
    }
}
