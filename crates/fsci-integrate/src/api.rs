#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::rk::{RK23_TABLEAU, RK45_TABLEAU, RkSolver, RkSolverConfig};
use crate::solver::{OdeSolver, OdeSolverState};
use crate::validation::{ToleranceValue, validate_tol};
use crate::{IntegrateValidationError, validate_first_step, validate_max_step};

pub type EventFn = fn(f64, &[f64]) -> f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    Rk23,
    Rk45,
    Dop853,
    Radau,
    Bdf,
    Lsoda,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OdeSolution {
    pub knots: Vec<f64>,
    pub values: Vec<Vec<f64>>,
    pub alt_segment: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveIvpOptions<'a> {
    pub t_span: (f64, f64),
    pub y0: &'a [f64],
    pub method: SolverKind,
    pub t_eval: Option<&'a [f64]>,
    pub dense_output: bool,
    pub events: Option<Vec<EventFn>>,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub first_step: Option<f64>,
    pub max_step: f64,
    pub mode: RuntimeMode,
}

impl Default for SolveIvpOptions<'_> {
    fn default() -> Self {
        Self {
            t_span: (0.0, 0.0),
            y0: &[],
            method: SolverKind::Rk45,
            t_eval: None,
            dense_output: false,
            events: None,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            first_step: None,
            max_step: f64::INFINITY,
            mode: RuntimeMode::Strict,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveIvpResult {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    pub sol: Option<OdeSolution>,
    pub t_events: Option<Vec<Vec<f64>>>,
    pub y_events: Option<Vec<Vec<Vec<f64>>>>,
    pub nfev: usize,
    pub njev: usize,
    pub nlu: usize,
    pub status: i32,
    pub message: String,
    pub success: bool,
}

const MSG_SUCCESS: &str = "The solver successfully reached the end of the integration interval.";
const MSG_FAILED: &str = "Integration step failed.";

fn validate_t_eval(t_eval: &[f64], t0: f64, tf: f64) -> Result<(), IntegrateValidationError> {
    let t_min = t0.min(tf);
    let t_max = t0.max(tf);
    if t_eval.iter().any(|&te| te < t_min || te > t_max) {
        return Err(IntegrateValidationError::TEvalOutOfSpan);
    }

    let is_sorted = if tf >= t0 {
        t_eval.windows(2).all(|window| window[1] > window[0])
    } else {
        t_eval.windows(2).all(|window| window[1] < window[0])
    };
    if !is_sorted {
        return Err(IntegrateValidationError::TEvalNotSorted);
    }

    Ok(())
}

fn interpolate_state(
    y_old: &[f64],
    y_new: &[f64],
    t_old: f64,
    t_new: f64,
    t_eval: f64,
) -> Vec<f64> {
    let frac = if (t_new - t_old).abs() > 0.0 {
        (t_eval - t_old) / (t_new - t_old)
    } else {
        0.0
    };
    y_old
        .iter()
        .zip(y_new.iter())
        .map(|(y0, y1)| y0 + frac * (y1 - y0))
        .collect()
}

/// Solve an initial value problem for a system of ODEs.
///
/// # Contract (P2C-001-D2)
/// - Supports `SolverKind::Rk45` and `SolverKind::Rk23`.
/// - Validates tolerances, step parameters, and t_eval ordering.
/// - Threads `RuntimeMode` through all validation and solver paths.
/// - Returns `SolveIvpResult` with status 0 on success, -1 on failure.
pub fn solve_ivp<F>(
    fun: &mut F,
    options: &SolveIvpOptions<'_>,
) -> Result<SolveIvpResult, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let (t0, tf) = options.t_span;
    let n = options.y0.len();
    let direction = if tf >= t0 { 1.0 } else { -1.0 };

    // Validate tolerances
    validate_max_step(options.max_step)?;
    if let Some(first_step) = options.first_step {
        validate_first_step(first_step, t0, tf)?;
    }
    let _ = validate_tol(
        ToleranceValue::Scalar(options.rtol),
        options.atol.clone(),
        n,
        options.mode,
    )?;

    // Validate t_eval
    if let Some(t_eval) = options.t_eval {
        validate_t_eval(t_eval, t0, tf)?;
    }

    // Select tableau
    let tableau = match options.method {
        SolverKind::Rk45 => &RK45_TABLEAU,
        SolverKind::Rk23 => &RK23_TABLEAU,
        _ => {
            return Err(IntegrateValidationError::NotYetImplemented {
                function: "solve_ivp: only RK45 and RK23 are currently implemented",
            });
        }
    };

    let config = RkSolverConfig {
        t0,
        y0: options.y0,
        t_bound: tf,
        rtol: options.rtol,
        atol: options.atol.clone(),
        max_step: options.max_step,
        first_step: options.first_step,
        mode: options.mode,
        tableau,
    };

    let mut solver = RkSolver::new(fun, config)?;

    // Collect time and state snapshots. When t_eval is provided, return exactly that grid.
    let mut ts = Vec::new();
    let mut ys: Vec<Vec<f64>> = Vec::new();
    let mut next_t_eval_index = 0usize;
    if let Some(t_eval) = options.t_eval {
        if matches!(t_eval.first(), Some(&first) if first == t0) {
            ts.push(t0);
            ys.push(options.y0.to_vec());
            next_t_eval_index = 1;
        }
    } else {
        ts.push(t0);
        ys.push(options.y0.to_vec());
    }

    let mut status: i32 = -1;

    // Integration loop
    while solver.state() == OdeSolverState::Running {
        match solver.step_with(fun) {
            Ok(outcome) => {
                let t = solver.t();
                let y = solver.y().to_vec();

                if let Some(t_eval) = options.t_eval {
                    let t_old = solver.t_old().unwrap_or(t0);
                    let y_old = solver.y_old().unwrap_or(options.y0);
                    while let Some(&te) = t_eval.get(next_t_eval_index) {
                        let in_range = if direction > 0.0 {
                            te > t_old && te <= t
                        } else {
                            te < t_old && te >= t
                        };
                        if !in_range {
                            break;
                        }

                        ts.push(te);
                        ys.push(interpolate_state(y_old, &y, t_old, t, te));
                        next_t_eval_index += 1;
                    }
                } else {
                    ts.push(t);
                    ys.push(y);
                }

                if outcome.state == OdeSolverState::Finished {
                    status = 0;
                }
            }
            Err(_) => {
                status = -1;
                break;
            }
        }
    }

    let (message, success) = if status >= 0 {
        (MSG_SUCCESS.to_owned(), true)
    } else {
        (MSG_FAILED.to_owned(), false)
    };

    Ok(SolveIvpResult {
        t: ts,
        y: ys,
        sol: None,
        t_events: None,
        y_events: None,
        nfev: solver.nfev(),
        njev: 0,
        nlu: 0,
        status,
        message,
        success,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_ivp_exponential_decay() {
        let result = solve_ivp(
            &mut |_t, y| vec![-0.5 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 10.0),
                y0: &[2.0],
                method: SolverKind::Rk45,
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp should succeed");

        assert!(result.success, "integration should succeed");
        assert_eq!(result.status, 0);

        // Check final value: y(10) = 2 * exp(-5) ≈ 0.01348
        let y_final = result.y.last().unwrap()[0];
        let expected = 2.0 * (-5.0_f64).exp();
        assert!(
            (y_final - expected).abs() < 1e-4,
            "y(10) = {y_final}, expected ≈ {expected}"
        );
    }

    #[test]
    fn solve_ivp_harmonic_oscillator() {
        // y'' + y = 0 => [y, y'] with y(0)=[1,0]
        // exact: y(t) = cos(t), y'(t) = -sin(t)
        let result = solve_ivp(
            &mut |_t, y| vec![y[1], -y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 2.0 * std::f64::consts::PI),
                y0: &[1.0, 0.0],
                method: SolverKind::Rk45,
                rtol: 1e-8,
                atol: ToleranceValue::Scalar(1e-10),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp should succeed");

        assert!(result.success);
        // After one full period, should return to initial conditions
        let y_final = &result.y.last().unwrap();
        assert!(
            (y_final[0] - 1.0).abs() < 1e-4,
            "y[0](2pi) should be ≈ 1, got {}",
            y_final[0]
        );
        assert!(
            y_final[1].abs() < 1e-4,
            "y[1](2pi) should be ≈ 0, got {}",
            y_final[1]
        );
    }

    #[test]
    fn solve_ivp_rk23() {
        let result = solve_ivp(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[1.0],
                method: SolverKind::Rk23,
                rtol: 1e-4,
                atol: ToleranceValue::Scalar(1e-6),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp should succeed");

        assert!(result.success);
        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!(
            (y_final - expected).abs() < 1e-3,
            "RK23 y(1) = {y_final}, expected ≈ {expected}"
        );
    }

    #[test]
    fn solve_ivp_empty_system() {
        let result = solve_ivp(
            &mut |_t, _y| vec![],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[],
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp should succeed for empty system");
        assert!(result.success);
    }

    #[test]
    fn solve_ivp_with_first_step() {
        let result = solve_ivp(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[1.0],
                first_step: Some(0.01),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp should succeed");
        assert!(result.success);
    }

    #[test]
    fn solve_ivp_t_eval_returns_requested_grid_only() {
        let t_eval = [0.25, 0.5, 0.75, 1.0];
        let result = solve_ivp(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[1.0],
                method: SolverKind::Rk45,
                t_eval: Some(&t_eval),
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp should succeed");

        assert_eq!(result.t, t_eval);
        assert_eq!(result.y.len(), t_eval.len());
        for (&t, y) in result.t.iter().zip(result.y.iter()) {
            let expected = (-t).exp();
            assert!(
                (y[0] - expected).abs() < 5e-3,
                "y({t}) = {}, expected ≈ {expected}",
                y[0]
            );
        }
    }

    #[test]
    fn solve_ivp_backward_t_eval_returns_requested_grid_only() {
        let t_eval = [1.0, 0.5, 0.0];
        let result = solve_ivp(
            &mut |_t, y| vec![y[0]],
            &SolveIvpOptions {
                t_span: (1.0, 0.0),
                y0: &[std::f64::consts::E],
                method: SolverKind::Rk45,
                t_eval: Some(&t_eval),
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("backward solve_ivp should succeed");

        assert_eq!(result.t, t_eval);
        assert_eq!(result.y.len(), t_eval.len());
        for (&t, y) in result.t.iter().zip(result.y.iter()) {
            let expected = t.exp();
            assert!(
                (y[0] - expected).abs() < 5e-3,
                "y({t}) = {}, expected ≈ {expected}",
                y[0]
            );
        }
    }

    #[test]
    fn solve_ivp_t_eval_out_of_span_fails() {
        let err = solve_ivp(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[1.0],
                t_eval: Some(&[0.25, 1.5]),
                ..SolveIvpOptions::default()
            },
        )
        .expect_err("out-of-span t_eval should fail");

        assert_eq!(err, IntegrateValidationError::TEvalOutOfSpan);
    }

    #[test]
    fn solve_ivp_t_eval_must_be_sorted_for_direction() {
        let forward_err = solve_ivp(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[1.0],
                t_eval: Some(&[0.75, 0.25]),
                ..SolveIvpOptions::default()
            },
        )
        .expect_err("forward unsorted t_eval should fail");
        assert_eq!(forward_err, IntegrateValidationError::TEvalNotSorted);

        let backward_err = solve_ivp(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                t_span: (1.0, 0.0),
                y0: &[1.0],
                t_eval: Some(&[0.25, 0.75]),
                ..SolveIvpOptions::default()
            },
        )
        .expect_err("backward unsorted t_eval should fail");
        assert_eq!(backward_err, IntegrateValidationError::TEvalNotSorted);
    }

    #[test]
    fn solve_ivp_unsupported_method() {
        let result = solve_ivp(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[1.0],
                method: SolverKind::Radau,
                ..SolveIvpOptions::default()
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn solve_ivp_three_body_lotka_volterra() {
        // dx/dt = x(a - b*y), dy/dt = y(-c + d*x)
        // a=1.5, b=1, c=3, d=1
        let result = solve_ivp(
            &mut |_t, y| vec![y[0] * (1.5 - y[1]), y[1] * (-3.0 + y[0])],
            &SolveIvpOptions {
                t_span: (0.0, 5.0),
                y0: &[10.0, 5.0],
                method: SolverKind::Rk45,
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("Lotka-Volterra should succeed");

        assert!(result.success);
        assert!(result.t.len() > 2, "should have multiple time points");
        // Verify conservation: V = d*x - c*ln(x) + b*y - a*ln(y) should be approximately constant
        let v0 =
            result.y[0][0] - 3.0 * result.y[0][0].ln() + result.y[0][1] - 1.5 * result.y[0][1].ln();
        let y_last = result.y.last().unwrap();
        let v_final = y_last[0] - 3.0 * y_last[0].ln() + y_last[1] - 1.5 * y_last[1].ln();
        assert!(
            (v_final - v0).abs() < 0.1,
            "Lotka-Volterra invariant should be approximately conserved: V0={v0}, Vf={v_final}"
        );
    }
}
