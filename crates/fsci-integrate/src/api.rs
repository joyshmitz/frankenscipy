#![forbid(unsafe_code)]

use fsci_opt::root::brentq;
use fsci_opt::types::RootOptions;
use fsci_runtime::RuntimeMode;

use crate::bdf::{BdfSolver, BdfSolverConfig};
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
    f_old: &[f64],
    f_new: &[f64],
    t_old: f64,
    t_new: f64,
    t_eval: f64,
) -> Vec<f64> {
    let h = t_new - t_old;
    if h.abs() <= 0.0 {
        return y_old.to_vec();
    }
    let x = (t_eval - t_old) / h;
    let x2 = x * x;
    let x3 = x2 * x;

    // Hermite basis functions
    let h00 = 2.0 * x3 - 3.0 * x2 + 1.0;
    let h10 = x3 - 2.0 * x2 + x;
    let h01 = -2.0 * x3 + 3.0 * x2;
    let h11 = x3 - x2;

    y_old
        .iter()
        .zip(y_new.iter())
        .zip(f_old.iter())
        .zip(f_new.iter())
        .map(|(((y0, y1), f0), f1)| h00 * y0 + h10 * h * f0 + h01 * y1 + h11 * h * f1)
        .collect()
}

fn solve_event_equation(
    event_fn: EventFn,
    t_old: f64,
    t_new: f64,
    y_old: &[f64],
    y_new: &[f64],
    f_old: &[f64],
    f_new: &[f64],
) -> f64 {
    let f = |t: f64| {
        let y = interpolate_state(y_old, y_new, f_old, f_new, t_old, t_new, t);
        event_fn(t, &y)
    };

    let options = RootOptions {
        xtol: 1e-12,
        maxiter: 100,
        ..Default::default()
    };

    match brentq(f, (t_old, t_new), options) {
        Ok(res) => res.root,
        Err(_) => 0.5 * (t_old + t_new), // fallback to midpoint
    }
}

/// Solve an initial value problem for a system of ODEs.
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

    if let Some(t_eval) = options.t_eval {
        validate_t_eval(t_eval, t0, tf)?;
    }

    // Select tableau
    let tableau = match options.method {
        SolverKind::Rk45 => &RK45_TABLEAU,
        SolverKind::Rk23 => &RK23_TABLEAU,
        SolverKind::Dop853 => &crate::rk::DOP853_TABLEAU,
        SolverKind::Radau | SolverKind::Bdf => {
            return solve_ivp_via_bdf(fun, options);
        }
        _ => {
            return Err(IntegrateValidationError::NotYetImplemented {
                function: "solve_ivp: only RK45, RK23, DOP853, Radau, and BDF are implemented",
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

    let mut ts = Vec::new();
    let mut ys: Vec<Vec<f64>> = Vec::new();
    let mut next_t_eval_index = 0usize;

    // Event initialization
    let mut t_events: Option<Vec<Vec<f64>>> = options
        .events
        .as_ref()
        .map(|evs| vec![Vec::new(); evs.len()]);
    let mut y_events: Option<Vec<Vec<Vec<f64>>>> = options
        .events
        .as_ref()
        .map(|evs| vec![Vec::new(); evs.len()]);
    let mut event_vals = options
        .events
        .as_ref()
        .map(|evs| evs.iter().map(|&ev| ev(t0, options.y0)).collect::<Vec<_>>());

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

    while solver.state() == OdeSolverState::Running {
        match solver.step_with(fun) {
            Ok(outcome) => {
                let t = solver.t();
                let y = solver.y().to_vec();
                let f = solver.f().to_vec();
                let t_old = solver.t_old().unwrap_or(t0);
                let y_old = solver.y_old().unwrap_or(options.y0).to_vec();
                let f_old = solver.f_old().unwrap_or(&f).to_vec();

                // Check events
                if let (Some(evs), Some(old_vals)) = (options.events.as_ref(), event_vals.as_mut())
                {
                    let mut triggered_terminal = false;
                    for (i, &ev_fn) in evs.iter().enumerate() {
                        let val = ev_fn(t, &y);
                        if old_vals[i].signum() != val.signum() && old_vals[i] != 0.0 {
                            let t_event =
                                solve_event_equation(ev_fn, t_old, t, &y_old, &y, &f_old, &f);
                            let y_event =
                                interpolate_state(&y_old, &y, &f_old, &f, t_old, t, t_event);

                            if let Some(tes) = t_events.as_mut() {
                                tes[i].push(t_event);
                            }
                            if let Some(yes) = y_events.as_mut() {
                                yes[i].push(y_event.clone());
                            }

                            // Adjust final state to event
                            ts.push(t_event);
                            ys.push(y_event);
                            triggered_terminal = true;
                        }
                        old_vals[i] = val;
                    }
                    if triggered_terminal {
                        status = 1; // Event terminated
                        break;
                    }
                }

                if let Some(t_eval) = options.t_eval {
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
                        ys.push(interpolate_state(&y_old, &y, &f_old, &f, t_old, t, te));
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

    let (message, success) = match status {
        0 => (MSG_SUCCESS.to_owned(), true),
        1 => ("A termination event occurred.".to_owned(), true),
        _ => (MSG_FAILED.to_owned(), false),
    };

    Ok(SolveIvpResult {
        t: ts,
        y: ys,
        sol: None,
        t_events,
        y_events,
        nfev: solver.nfev(),
        njev: 0,
        nlu: 0,
        status,
        message,
        success,
    })
}

fn solve_ivp_via_bdf<F>(
    fun: &mut F,
    options: &SolveIvpOptions<'_>,
) -> Result<SolveIvpResult, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let (t0, tf) = options.t_span;
    let n = options.y0.len();
    let direction = if tf >= t0 { 1.0 } else { -1.0 };

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
    if let Some(t_eval) = options.t_eval {
        validate_t_eval(t_eval, t0, tf)?;
    }

    let config = BdfSolverConfig {
        t0,
        y0: options.y0,
        t_bound: tf,
        rtol: options.rtol,
        atol: options.atol.clone(),
        max_step: options.max_step,
        first_step: options.first_step,
        mode: options.mode,
        max_order: 5,
    };

    let mut solver = BdfSolver::new(fun, config)?;

    let mut ts = Vec::new();
    let mut ys = Vec::new();
    let mut next_t_eval_index = 0usize;

    let mut t_events: Option<Vec<Vec<f64>>> = options
        .events
        .as_ref()
        .map(|evs| vec![Vec::new(); evs.len()]);
    let mut y_events: Option<Vec<Vec<Vec<f64>>>> = options
        .events
        .as_ref()
        .map(|evs| vec![Vec::new(); evs.len()]);
    let mut event_vals = options
        .events
        .as_ref()
        .map(|evs| evs.iter().map(|&ev| ev(t0, options.y0)).collect::<Vec<_>>());

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

    let mut status = -1;

    while solver.state() == OdeSolverState::Running {
        match solver.step_with(fun) {
            Ok(_) => {
                let t = solver.t();
                let y = solver.y().to_vec();
                let f = solver.f().to_vec();
                let t_old = solver.t_old().unwrap_or(t0);
                let y_old = solver.y_old().unwrap_or(options.y0).to_vec();
                let f_old = solver.f_old().unwrap_or(&f).to_vec();

                if let (Some(evs), Some(old_vals)) = (options.events.as_ref(), event_vals.as_mut())
                {
                    let mut triggered_terminal = false;
                    let mut event_hits: Vec<(f64, Vec<f64>)> = Vec::new();
                    for (i, &ev_fn) in evs.iter().enumerate() {
                        let val = ev_fn(t, &y);
                        if old_vals[i].signum() != val.signum() && old_vals[i] != 0.0 {
                            let t_event =
                                solve_event_equation(ev_fn, t_old, t, &y_old, &y, &f_old, &f);
                            let y_event =
                                interpolate_state(&y_old, &y, &f_old, &f, t_old, t, t_event);
                            if let Some(tes) = t_events.as_mut() {
                                tes[i].push(t_event);
                            }
                            if let Some(yes) = y_events.as_mut() {
                                yes[i].push(y_event.clone());
                            }
                            event_hits.push((t_event, y_event));
                            triggered_terminal = true;
                        }
                        old_vals[i] = val;
                    }
                    if triggered_terminal {
                        let first_idx = terminal_event_index(direction, &event_hits);
                        let (t_event, y_event) = &event_hits[first_idx];
                        append_solution_until(
                            &mut ts,
                            &mut ys,
                            options.t_eval,
                            &mut next_t_eval_index,
                            direction,
                            t_old,
                            *t_event,
                            |te| interpolate_state(&y_old, &y, &f_old, &f, t_old, t, te),
                        );
                        if ts.last().copied() != Some(*t_event) {
                            ts.push(*t_event);
                            ys.push(y_event.clone());
                        }
                        status = 1;
                        break;
                    }
                }

                if options.t_eval.is_some() {
                    append_solution_until(
                        &mut ts,
                        &mut ys,
                        options.t_eval,
                        &mut next_t_eval_index,
                        direction,
                        t_old,
                        t,
                        |te| interpolate_state(&y_old, &y, &f_old, &f, t_old, t, te),
                    );
                } else {
                    ts.push(t);
                    ys.push(y);
                }
                if solver.state() == OdeSolverState::Finished {
                    status = 0;
                }
            }
            Err(_) => break,
        }
    }

    let success = status >= 0;
    let message = match status {
        0 => MSG_SUCCESS,
        1 => "A termination event occurred.",
        _ => MSG_FAILED,
    };

    Ok(SolveIvpResult {
        t: ts,
        y: ys,
        sol: None,
        t_events,
        y_events,
        nfev: solver.nfev(),
        njev: 0,
        nlu: 0,
        status,
        message: message.to_owned(),
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

        let y_final = result.y.last().unwrap()[0];
        let expected = 2.0 * (-5.0_f64).exp();
        assert!(
            (y_final - expected).abs() < 1e-4,
            "y(10) = {y_final}, expected ≈ {expected}"
        );
    }

    #[test]
    fn solve_ivp_with_event() {
        // y' = 1, y(0) = 0. Event at y = 5.
        fn event_at_5(_t: f64, y: &[f64]) -> f64 {
            y[0] - 5.0
        }

        let result = solve_ivp(
            &mut |_t, _y| vec![1.0],
            &SolveIvpOptions {
                t_span: (0.0, 10.0),
                y0: &[0.0],
                method: SolverKind::Rk45,
                events: Some(vec![event_at_5]),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp with event should succeed");

        assert!(result.success);
        assert_eq!(result.status, 1); // event termination
        assert!((result.t.last().unwrap() - 5.0).abs() < 1e-8);
        assert!((result.y.last().unwrap()[0] - 5.0).abs() < 1e-8);

        let t_events = result.t_events.unwrap();
        assert_eq!(t_events[0].len(), 1);
        assert!((t_events[0][0] - 5.0).abs() < 1e-8);
    }

    #[test]
    fn solve_ivp_terminal_event_inside_long_step_truncates_main_solution() {
        fn event_at_point_three(_t: f64, y: &[f64]) -> f64 {
            y[0] - 0.3
        }

        let result = solve_ivp(
            &mut |_t, _y| vec![1.0],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[0.0],
                method: SolverKind::Rk45,
                first_step: Some(1.0),
                max_step: 1.0,
                events: Some(vec![event_at_point_three]),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp with interior event should succeed");

        assert!(result.success);
        assert_eq!(result.status, 1);
        assert!((result.t.last().expect("event time") - 0.3).abs() < 1e-6);
        assert!((result.y.last().expect("event state")[0] - 0.3).abs() < 1e-6);
    }
}
