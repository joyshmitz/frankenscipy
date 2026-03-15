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
    CompositeQuadResult, DblquadOptions, DblquadResult, QuadOptions, QuadResult,
    cumulative_trapezoid, cumulative_trapezoid_uniform, dblquad, quad, simpson, simpson_uniform,
    trapezoid, trapezoid_uniform,
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
    if t.len() < 2 {
        return Ok(vec![y0.to_vec()]);
    }

    let mut results = Vec::with_capacity(t.len());
    results.push(y0.to_vec());

    let mut current_y = y0.to_vec();

    for i in 0..t.len() - 1 {
        // odeint convention: func(y, t), but solve_ivp uses func(t, y)
        let mut ivp_func = |ti: f64, yi: &[f64]| -> Vec<f64> { func(yi, ti) };

        let result = solve_ivp(
            &mut ivp_func,
            &SolveIvpOptions {
                t_span: (t[i], t[i + 1]),
                y0: &current_y,
                method: SolverKind::Rk45,
                rtol: 1.49e-8,
                atol: ToleranceValue::Scalar(1.49e-8),
                ..SolveIvpOptions::default()
            },
        )?;

        if let Some(y_final) = result.y.last() {
            current_y = y_final.clone();
        }
        results.push(current_y.clone());
    }

    Ok(results)
}
