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
    if !t_span.0.is_finite() || !t_span.1.is_finite() {
        return Err(BvpError::InvalidArgument(
            "t_span endpoints must be finite".to_string(),
        ));
    }
    if y_guess.iter().any(|value| !value.is_finite()) {
        return Err(BvpError::InvalidArgument(
            "y_guess values must be finite".to_string(),
        ));
    }
    validate_bvp_options(&options)?;

    let mut y0 = y_guess.to_vec();

    for iteration in 0..options.max_iter {
        // Solve IVP with current y0
        let ivp_result = solve_ivp_internal(f, t_span, &y0, options.rtol, options.atol)?;
        let yb = ivp_result
            .y
            .last()
            .ok_or_else(|| BvpError::IvpFailed("IVP result is empty".to_string()))?
            .clone();

        // Evaluate boundary condition residual
        let residual = bc(&y0, &yb);
        validate_boundary_residual_len(&residual, n)?;
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
            let yb_pert = ivp_pert.y.last().ok_or_else(|| {
                BvpError::IvpFailed("IVP perturbation result is empty".to_string())
            })?;
            let residual_pert = bc(&y0_pert, yb_pert);
            validate_boundary_residual_len(&residual_pert, n)?;
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
    let yb = ivp_result
        .y
        .last()
        .ok_or_else(|| BvpError::IvpFailed("Final IVP result is empty".to_string()))?;
    let residual = bc(&y0, yb);
    validate_boundary_residual_len(&residual, n)?;
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

/// Batched boundary-value problems: solve N independent BVPs that share the dynamics/BC SHAPE but
/// differ by a parameter vector, one [`BvpResult`] per set. This is the vmap-over-solver primitive
/// for BVPs — a parameter study (vary a nonlinearity strength, a boundary value, a forcing term)
/// loops `solve_bvp` in Python, N collocation-Newton solves SERIALLY, each calling the Python RHS
/// at every mesh node every Newton iteration. fsci `solve_bvp_many` (`f: Fn(t, y, params)->Vec`,
/// `bc: Fn(ya, yb, params)->Vec`) fans the N independent solves across cores and inlines both
/// callbacks. Result `i` is byte-identical to the per-parameter `solve_bvp` call.
pub fn solve_bvp_many<F, BC>(
    f: F,
    bc: BC,
    t_span: (f64, f64),
    y_guess: &[f64],
    param_rows: &[Vec<f64>],
    options: BvpOptions,
) -> Vec<Result<BvpResult, BvpError>>
where
    F: Fn(f64, &[f64], &[f64]) -> Vec<f64> + Sync,
    BC: Fn(&[f64], &[f64], &[f64]) -> Vec<f64> + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let f_ref = &f;
    let bc_ref = &bc;
    let solve_one = move |params: &[f64]| {
        // solve_bvp wants `&mut F: FnMut` and `&BC: Fn`; wrap the shared user closures so each
        // member gets a fresh local closure capturing its own parameter row.
        let mut local_f = |t: f64, y: &[f64]| f_ref(t, y, params);
        let local_bc = |ya: &[f64], yb: &[f64]| bc_ref(ya, yb, params);
        solve_bvp(&mut local_f, &local_bc, t_span, y_guess, options.clone())
    };

    // Each BVP is an independent, expensive (collocation-Newton) solve → fan whole parameter sets
    // across cores, capped by the row count; a tiny sweep stays serial.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<BvpResult, BvpError>>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * chunk;
                if lo >= nrows {
                    return None;
                }
                let hi = (lo + chunk).min(nrows);
                Some(scope.spawn(move || {
                    (lo..hi)
                        .map(|i| solve_one(&param_rows[i]))
                        .collect::<Vec<_>>()
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("solve_bvp_many worker panicked"))
            .collect()
    });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

fn validate_bvp_options(options: &BvpOptions) -> Result<(), BvpError> {
    if options.max_iter == 0 {
        return Err(BvpError::InvalidArgument(
            "max_iter must be positive".to_string(),
        ));
    }
    if !options.tol.is_finite() || options.tol < 0.0 {
        return Err(BvpError::InvalidArgument(
            "tol must be finite and non-negative".to_string(),
        ));
    }
    if !options.rtol.is_finite() || options.rtol <= 0.0 {
        return Err(BvpError::InvalidArgument(
            "rtol must be finite and positive".to_string(),
        ));
    }
    if !options.atol.is_finite() || options.atol <= 0.0 {
        return Err(BvpError::InvalidArgument(
            "atol must be finite and positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_boundary_residual_len(residual: &[f64], expected: usize) -> Result<(), BvpError> {
    if residual.len() != expected {
        return Err(BvpError::InvalidArgument(format!(
            "bc residual length must match y_guess length (expected {expected}, got {})",
            residual.len()
        )));
    }
    if residual.iter().any(|value| !value.is_finite()) {
        return Err(BvpError::InvalidArgument(
            "bc residual values must be finite".to_string(),
        ));
    }
    Ok(())
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
    fn solve_bvp_many_byte_identical_to_per_param() {
        // Parameter sweep of a nonlinear BVP: y0'=y1, y1'=p*(1+y0^2); y0(0)=0, y0(1)=1.
        // The batched solve must equal looping solve_bvp per parameter, bit-for-bit.
        let f = |_t: f64, y: &[f64], p: &[f64]| vec![y[1], p[0] * (1.0 + y[0] * y[0])];
        let bc = |ya: &[f64], yb: &[f64], _p: &[f64]| vec![ya[0], yb[0] - 1.0];
        let mut s = 21u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            0.5 + 1.5 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 8usize; // crosses the serial->parallel gate
        let params: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng()]).collect();
        let opts = BvpOptions::default();

        let batched = solve_bvp_many(f, bc, (0.0, 1.0), &[0.0, 0.0], &params, opts.clone());
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let mut local_f = |t: f64, y: &[f64]| f(t, y, p);
            let local_bc = |ya: &[f64], yb: &[f64]| bc(ya, yb, p);
            let single = solve_bvp(
                &mut local_f,
                &local_bc,
                (0.0, 1.0),
                &[0.0, 0.0],
                opts.clone(),
            )
            .expect("single solve");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(
                many.converged, single.converged,
                "converged mismatch param {i}"
            );
            assert_eq!(many.t.len(), single.t.len(), "mesh size mismatch param {i}");
            for (a, b) in many.t.iter().zip(&single.t) {
                assert_eq!(a.to_bits(), b.to_bits(), "t mismatch param {i}");
            }
            for (ra, rb) in many.y.iter().zip(&single.y) {
                for (a, b) in ra.iter().zip(rb) {
                    assert_eq!(a.to_bits(), b.to_bits(), "y mismatch param {i}");
                }
            }
        }
        // The sweep exercises genuine converging solves.
        assert!(
            batched
                .iter()
                .filter(|r| r.as_ref().map(|x| x.converged).unwrap_or(false))
                .count()
                >= nrows / 2
        );
        assert!(solve_bvp_many(f, bc, (0.0, 1.0), &[0.0, 0.0], &[], opts).is_empty());
    }

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

    #[test]
    fn bvp_rejects_non_finite_span_and_initial_guess() {
        for span in [
            (f64::NAN, 1.0),
            (0.0, f64::NAN),
            (f64::NEG_INFINITY, 1.0),
            (0.0, f64::INFINITY),
        ] {
            let mut f = |_t: f64, _y: &[f64]| vec![0.0];
            let bc = |ya: &[f64], _yb: &[f64]| vec![ya[0]];
            let err = solve_bvp(&mut f, &bc, span, &[0.0], BvpOptions::default())
                .expect_err("non-finite span");
            assert!(matches!(err, BvpError::InvalidArgument(msg) if msg.contains("t_span")));
        }

        for bad_y0 in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut f = |_t: f64, _y: &[f64]| vec![0.0];
            let bc = |ya: &[f64], _yb: &[f64]| vec![ya[0]];
            let err = solve_bvp(&mut f, &bc, (0.0, 1.0), &[bad_y0], BvpOptions::default())
                .expect_err("non-finite y_guess");
            assert!(matches!(err, BvpError::InvalidArgument(msg) if msg.contains("y_guess")));
        }
    }

    #[test]
    fn bvp_rejects_invalid_boundary_tolerance() {
        for tol in [f64::NAN, f64::INFINITY, -1e-6] {
            let mut f = |_t: f64, _y: &[f64]| vec![0.0];
            let bc = |_ya: &[f64], _yb: &[f64]| vec![1.0];
            let options = BvpOptions {
                tol,
                ..BvpOptions::default()
            };
            let err = solve_bvp(&mut f, &bc, (0.0, 1.0), &[0.0], options)
                .expect_err("invalid boundary tolerance");
            assert!(matches!(err, BvpError::InvalidArgument(msg) if msg.contains("tol")));
        }
    }

    #[test]
    fn bvp_rejects_invalid_ivp_tolerances() {
        for (rtol, atol, expected) in [
            (f64::NAN, 1e-10, "rtol"),
            (f64::INFINITY, 1e-10, "rtol"),
            (0.0, 1e-10, "rtol"),
            (-1e-8, 1e-10, "rtol"),
            (1e-8, f64::NAN, "atol"),
            (1e-8, f64::INFINITY, "atol"),
            (1e-8, 0.0, "atol"),
            (1e-8, -1e-10, "atol"),
        ] {
            let mut f = |_t: f64, _y: &[f64]| vec![0.0];
            let bc = |_ya: &[f64], _yb: &[f64]| vec![1.0];
            let options = BvpOptions {
                rtol,
                atol,
                ..BvpOptions::default()
            };
            let err = solve_bvp(&mut f, &bc, (0.0, 1.0), &[0.0], options)
                .expect_err("invalid IVP tolerance");
            assert!(
                matches!(&err, BvpError::InvalidArgument(msg) if msg.contains(expected)),
                "expected {expected} validation error, got {err:?}"
            );
        }
    }

    #[test]
    fn bvp_rejects_zero_iteration_budget() {
        let mut f = |_t: f64, _y: &[f64]| vec![0.0];
        let bc = |ya: &[f64], _yb: &[f64]| vec![ya[0]];
        let options = BvpOptions {
            max_iter: 0,
            ..BvpOptions::default()
        };
        let err = solve_bvp(&mut f, &bc, (0.0, 1.0), &[0.0], options).expect_err("zero max_iter");
        assert!(matches!(
            err,
            BvpError::InvalidArgument(msg) if msg.contains("max_iter")
        ));
    }

    #[test]
    fn bvp_rejects_short_boundary_residual() {
        let mut f = |_t: f64, _y: &[f64]| vec![0.0, 0.0];
        let bc = |_ya: &[f64], _yb: &[f64]| vec![0.0];
        let err = solve_bvp(&mut f, &bc, (0.0, 1.0), &[0.0, 0.0], BvpOptions::default())
            .expect_err("short boundary residual");
        assert!(matches!(err, BvpError::InvalidArgument(msg) if msg.contains("expected 2, got 1")));
    }

    #[test]
    fn bvp_rejects_long_boundary_residual() {
        let mut f = |_t: f64, _y: &[f64]| vec![0.0, 0.0];
        let bc = |_ya: &[f64], _yb: &[f64]| vec![0.0, 0.0, 0.0];
        let err = solve_bvp(&mut f, &bc, (0.0, 1.0), &[0.0, 0.0], BvpOptions::default())
            .expect_err("long boundary residual");
        assert!(matches!(err, BvpError::InvalidArgument(msg) if msg.contains("expected 2, got 3")));
    }

    #[test]
    fn bvp_rejects_non_finite_boundary_residual() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut f = |_t: f64, _y: &[f64]| vec![0.0, 0.0];
            let bc = move |_ya: &[f64], _yb: &[f64]| vec![0.0, bad];
            let err = solve_bvp(&mut f, &bc, (0.0, 1.0), &[0.0, 0.0], BvpOptions::default())
                .expect_err("non-finite boundary residual");
            assert!(
                matches!(err, BvpError::InvalidArgument(msg) if msg.contains("residual values"))
            );
        }
    }
}
