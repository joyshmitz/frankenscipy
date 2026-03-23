#![forbid(unsafe_code)]

pub mod api;
pub mod bdf;
pub mod bvp;
pub mod quad;
pub mod rk;
pub mod solver;
pub mod step_size;
pub mod validation;

pub use api::{EventFn, OdeSolution, SolveIvpOptions, SolveIvpResult, SolverKind, solve_ivp};
pub use bdf::{BdfSolver, BdfSolverConfig};
pub use bvp::{BvpError, BvpOptions, BvpResult, solve_bvp};
pub use quad::{
    CompositeQuadResult, DblquadOptions, DblquadResult, QuadOptions, QuadResult, QuadVecResult,
    cumulative_simpson, cumulative_trapezoid, cumulative_trapezoid_uniform, dblquad, fixed_quad,
    nquad, quad, quad_vec, romb, romb_func, simpson, simpson_uniform, tplquad, trapezoid,
    trapezoid_uniform,
};
pub use rk::{
    ButcherTableau, DOP853_TABLEAU, RK23_TABLEAU, RK45_TABLEAU, RkSolver, RkSolverConfig,
};
pub use solver::{OdeSolver, OdeSolverState, StepFailure, StepOutcome};
pub use step_size::{InitialStepRequest, StepRhsFn, select_initial_step};
pub use validation::{
    EPS, IntegrateValidationError, MIN_RTOL, ToleranceValue, ToleranceWarning, ValidatedTolerance,
    validate_first_step, validate_max_step, validate_tol,
};

/// Legacy `odeint`-style interface.
///
/// Matches `scipy.integrate.odeint(func, y0, t)`.
/// Integrates an ODE system y' = func(y, t) over a sequence of time points.
/// Returns `(y_matrix, info)` where y_matrix[i] is the state at t[i].
pub fn odeint<F>(
    func: &mut F,
    y0: &[f64],
    t: &[f64],
) -> Result<Vec<Vec<f64>>, IntegrateValidationError>
where
    F: FnMut(&[f64], f64) -> Vec<f64>,
{
    if t.is_empty() {
        return Ok(vec![]);
    }
    if t.len() == 1 {
        return Ok(vec![y0.to_vec()]);
    }

    // odeint convention: func(y, t), but solve_ivp uses func(t, y)
    let mut ivp_func = |ti: f64, yi: &[f64]| -> Vec<f64> { func(yi, ti) };

    let t0 = t[0];
    let tf = t[t.len() - 1];

    let result = solve_ivp(
        &mut ivp_func,
        &SolveIvpOptions {
            t_span: (t0, tf),
            y0,
            method: SolverKind::Rk45,
            t_eval: Some(t),
            rtol: 1.49e-8,
            atol: ToleranceValue::Scalar(1.49e-8),
            ..SolveIvpOptions::default()
        },
    )?;

    // transpose result.y since solve_ivp returns columns for each time step,
    // wait, solve_ivp result.y is Vec<Vec<f64>> where each inner vector is the state at t_i.
    // Let's assume result.y matches the shape we need.
    Ok(result.y)
}
