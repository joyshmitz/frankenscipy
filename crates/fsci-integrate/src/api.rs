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
        let direction = if tf >= t0 { 1.0 } else { -1.0 };
        let t_min = t0.min(tf);
        let t_max = t0.max(tf);
        for &te in t_eval {
            if te < t_min || te > t_max {
                return Err(IntegrateValidationError::NotYetImplemented {
                    function: "solve_ivp: t_eval values not within t_span",
                });
            }
        }
        // Check ordering
        for w in t_eval.windows(2) {
            if direction > 0.0 && w[1] <= w[0] {
                return Err(IntegrateValidationError::NotYetImplemented {
                    function: "solve_ivp: t_eval not properly sorted",
                });
            }
            if direction < 0.0 && w[1] >= w[0] {
                return Err(IntegrateValidationError::NotYetImplemented {
                    function: "solve_ivp: t_eval not properly sorted (backward)",
                });
            }
        }
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

    // Collect time and state snapshots
    let mut ts = vec![t0];
    let mut ys: Vec<Vec<f64>> = vec![options.y0.to_vec()];

    let direction = if tf >= t0 { 1.0 } else { -1.0 };
    let mut status: i32 = -1;

    // Integration loop
    while solver.state() == OdeSolverState::Running {
        match solver.step_with(fun) {
            Ok(outcome) => {
                let t = solver.t();
                let y = solver.y().to_vec();

                if let Some(t_eval) = options.t_eval {
                    // Only store t_eval points (using linear interpolation for simplicity).
                    // Full dense output interpolation is future work.
                    let t_old = solver.t_old().unwrap_or(t0);
                    for &te in t_eval {
                        let in_range = if direction > 0.0 {
                            te > t_old && te <= t
                        } else {
                            te < t_old && te >= t
                        };
                        if in_range && !ts.contains(&te) {
                            // Linear interpolation between t_old and t
                            let frac = if (t - t_old).abs() > 0.0 {
                                (te - t_old) / (t - t_old)
                            } else {
                                0.0
                            };
                            let y_old = if let Some(idx) = ts.last().map(|_| ys.len() - 1) {
                                &ys[idx]
                            } else {
                                options.y0
                            };
                            let y_interp: Vec<f64> = y_old
                                .iter()
                                .zip(y.iter())
                                .map(|(yo, yn)| yo + frac * (yn - yo))
                                .collect();
                            ts.push(te);
                            ys.push(y_interp);
                        }
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
