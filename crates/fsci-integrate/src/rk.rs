#![forbid(unsafe_code)]

//! Explicit Runge-Kutta ODE solvers (RK23 and RK45).
//!
//! Implements the Dormand-Prince RK5(4) and Bogacki-Shampine RK3(2) methods
//! with adaptive step-size control matching SciPy's `_ivp/rk.py`.

use fsci_runtime::RuntimeMode;

use crate::solver::{OdeSolver, OdeSolverState, StepFailure, StepOutcome};
use crate::step_size::InitialStepRequest;
use crate::validation::{ToleranceValue, validate_tol};
use crate::{
    IntegrateValidationError, select_initial_step, validate_first_step, validate_max_step,
};

// Step-size control constants (from SciPy's rk.py).
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

/// RMS norm: ||x|| / sqrt(n).
fn rms_norm(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = x.iter().map(|v| v * v).sum();
    (sum_sq / x.len() as f64).sqrt()
}

/// Butcher tableau for an explicit Runge-Kutta method.
pub struct ButcherTableau {
    /// A coefficients (lower-triangular, row-major, n_stages × n_stages).
    pub a: &'static [&'static [f64]],
    /// B coefficients (weights for combining stages, length n_stages).
    pub b: &'static [f64],
    /// C coefficients (time increments, length n_stages).
    pub c: &'static [f64],
    /// E coefficients (error estimation, length n_stages + 1).
    pub e: &'static [f64],
    /// Number of stages.
    pub n_stages: usize,
    /// Order of the main method.
    pub order: usize,
    /// Order of the error estimator.
    pub error_estimator_order: usize,
}

// ═══════════════════════════════════════════════════════════════
// RK45: Dormand-Prince 5(4) Butcher tableau
// ═══════════════════════════════════════════════════════════════

static RK45_C: &[f64] = &[0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0];

static RK45_A0: &[f64] = &[];
static RK45_A1: &[f64] = &[1.0 / 5.0];
static RK45_A2: &[f64] = &[3.0 / 40.0, 9.0 / 40.0];
static RK45_A3: &[f64] = &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];
static RK45_A4: &[f64] = &[
    19372.0 / 6561.0,
    -25360.0 / 2187.0,
    64448.0 / 6561.0,
    -212.0 / 729.0,
];
static RK45_A5: &[f64] = &[
    9017.0 / 3168.0,
    -355.0 / 33.0,
    46732.0 / 5247.0,
    49.0 / 176.0,
    -5103.0 / 18656.0,
];

static RK45_A: &[&[f64]] = &[RK45_A0, RK45_A1, RK45_A2, RK45_A3, RK45_A4, RK45_A5];

static RK45_B: &[f64] = &[
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
];

static RK45_E: &[f64] = &[
    -71.0 / 57600.0,
    0.0,
    71.0 / 16695.0,
    -71.0 / 1920.0,
    17253.0 / 339200.0,
    -22.0 / 525.0,
    1.0 / 40.0,
];

pub static RK45_TABLEAU: ButcherTableau = ButcherTableau {
    a: RK45_A,
    b: RK45_B,
    c: RK45_C,
    e: RK45_E,
    n_stages: 6,
    order: 5,
    error_estimator_order: 4,
};

// ═══════════════════════════════════════════════════════════════
// RK23: Bogacki-Shampine 3(2) Butcher tableau
// ═══════════════════════════════════════════════════════════════

static RK23_C: &[f64] = &[0.0, 1.0 / 2.0, 3.0 / 4.0];

static RK23_A0: &[f64] = &[];
static RK23_A1: &[f64] = &[1.0 / 2.0];
static RK23_A2: &[f64] = &[0.0, 3.0 / 4.0];

static RK23_A: &[&[f64]] = &[RK23_A0, RK23_A1, RK23_A2];

static RK23_B: &[f64] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0];

static RK23_E: &[f64] = &[5.0 / 72.0, -1.0 / 12.0, -1.0 / 9.0, 1.0 / 8.0];

pub static RK23_TABLEAU: ButcherTableau = ButcherTableau {
    a: RK23_A,
    b: RK23_B,
    c: RK23_C,
    e: RK23_E,
    n_stages: 3,
    order: 3,
    error_estimator_order: 2,
};

// ═══════════════════════════════════════════════════════════════
// Core RK step function
// ═══════════════════════════════════════════════════════════════

/// Perform a single explicit Runge-Kutta step.
///
/// Returns `(y_new, f_new)` where `f_new = fun(t + h, y_new)`.
/// `k` is filled with the stage derivatives (k[0] = f, k[n_stages] = f_new).
fn rk_step<F>(
    fun: &mut F,
    t: f64,
    y: &[f64],
    f: &[f64],
    h: f64,
    tableau: &ButcherTableau,
    k: &mut [Vec<f64>],
) -> (Vec<f64>, Vec<f64>)
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let n = y.len();
    k[0] = f.to_vec();

    for s in 1..tableau.n_stages {
        let a_row = tableau.a[s];
        let c_s = tableau.c[s];
        let mut dy = vec![0.0; n];
        for (j, &a_sj) in a_row.iter().enumerate() {
            if a_sj != 0.0 {
                for i in 0..n {
                    dy[i] += a_sj * k[j][i];
                }
            }
        }
        let y_stage: Vec<f64> = y
            .iter()
            .zip(dy.iter())
            .map(|(yi, di)| yi + h * di)
            .collect();
        k[s] = fun(t + c_s * h, &y_stage);
    }

    // y_new = y + h * sum(B[s] * K[s])
    let mut y_new = y.to_vec();
    for (s, &b_s) in tableau.b.iter().enumerate() {
        if b_s != 0.0 {
            for i in 0..n {
                y_new[i] += h * b_s * k[s][i];
            }
        }
    }

    let f_new = fun(t + h, &y_new);
    k[tableau.n_stages] = f_new.clone();

    (y_new, f_new)
}

/// Estimate local error using the E coefficients.
fn estimate_error(k: &[Vec<f64>], e: &[f64], h: f64, n: usize) -> Vec<f64> {
    let mut err = vec![0.0; n];
    for (s, &e_s) in e.iter().enumerate() {
        if e_s != 0.0 {
            for i in 0..n {
                err[i] += e_s * k[s][i];
            }
        }
    }
    for v in &mut err {
        *v *= h;
    }
    err
}

/// Compute the RMS error norm: ||error / scale|| / sqrt(n).
fn error_norm(error: &[f64], scale: &[f64]) -> f64 {
    rms_norm(
        &error
            .iter()
            .zip(scale.iter())
            .map(|(e, s)| e / s)
            .collect::<Vec<_>>(),
    )
}

// ═══════════════════════════════════════════════════════════════
// Runge-Kutta solver state machine
// ═══════════════════════════════════════════════════════════════

/// Configuration for constructing an RK solver.
pub struct RkSolverConfig<'a> {
    pub t0: f64,
    pub y0: &'a [f64],
    pub t_bound: f64,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub max_step: f64,
    pub first_step: Option<f64>,
    pub mode: RuntimeMode,
    pub tableau: &'static ButcherTableau,
}

/// An explicit Runge-Kutta ODE solver with adaptive step-size control.
///
/// Supports RK45 (Dormand-Prince) and RK23 (Bogacki-Shampine) via
/// interchangeable Butcher tableaux.
pub struct RkSolver {
    mode: RuntimeMode,
    state: OdeSolverState,
    tableau: &'static ButcherTableau,
    // Problem definition
    n: usize,
    t: f64,
    y: Vec<f64>,
    t_old: Option<f64>,
    y_old: Option<Vec<f64>>,
    t_bound: f64,
    direction: f64,
    // Tolerances
    rtol: f64,
    atol: ToleranceValue,
    max_step: f64,
    // Solver state
    f: Vec<f64>,
    h_abs: f64,
    error_exponent: f64,
    k: Vec<Vec<f64>>,
    // Statistics
    nfev: usize,
}

impl RkSolver {
    /// Create a new RK solver from configuration.
    ///
    /// # Contract (P2C-001-D2)
    /// - Validates tolerances and step parameters.
    /// - Calls `select_initial_step` if no `first_step` is provided.
    /// - Threads `RuntimeMode` through all validation paths.
    pub fn new<F>(fun: &mut F, config: RkSolverConfig<'_>) -> Result<Self, IntegrateValidationError>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = config.y0.len();

        // Validate tolerances
        let _ = validate_tol(
            ToleranceValue::Scalar(config.rtol),
            config.atol.clone(),
            n,
            config.mode,
        )?;

        // Validate max_step
        if config.max_step.is_finite() {
            validate_max_step(config.max_step)?;
        }

        let direction = if config.t_bound != config.t0 {
            (config.t_bound - config.t0).signum()
        } else {
            1.0
        };

        // Evaluate f0
        let f0 = fun(config.t0, config.y0);
        let nfev = 1;

        // Determine initial step size
        let h_abs = if let Some(first_step) = config.first_step {
            validate_first_step(first_step, config.t0, config.t_bound)?
        } else {
            let step_request = InitialStepRequest {
                t0: config.t0,
                y0: config.y0,
                t_bound: config.t_bound,
                max_step: config.max_step,
                f0: &f0,
                direction,
                order: config.tableau.error_estimator_order as f64,
                rtol: config.rtol,
                atol: config.atol.clone(),
                mode: config.mode,
            };
            select_initial_step(fun, &step_request)?
        };

        let error_exponent = -1.0 / (config.tableau.error_estimator_order as f64 + 1.0);

        // Allocate K storage: n_stages + 1 vectors of length n
        let k = vec![vec![0.0; n]; config.tableau.n_stages + 1];

        Ok(Self {
            mode: config.mode,
            state: OdeSolverState::Running,
            tableau: config.tableau,
            n,
            t: config.t0,
            y: config.y0.to_vec(),
            t_old: None,
            y_old: None,
            t_bound: config.t_bound,
            direction,
            rtol: config.rtol,
            atol: config.atol,
            max_step: config.max_step,
            f: f0,
            h_abs,
            error_exponent,
            k,
            nfev,
        })
    }

    /// Current time.
    pub fn t(&self) -> f64 {
        self.t
    }

    /// Current state vector.
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// Previous time (after at least one successful step).
    pub fn t_old(&self) -> Option<f64> {
        self.t_old
    }

    /// Number of function evaluations.
    pub fn nfev(&self) -> usize {
        self.nfev
    }

    /// Perform one adaptive step, advancing the solver.
    ///
    /// This is the core step implementation matching SciPy's `RungeKutta._step_impl`.
    pub fn step_with<F>(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        if self.state != OdeSolverState::Running {
            return Err(StepFailure::RuntimeError(
                "Attempt to step on a finished or failed solver.",
            ));
        }

        // Handle empty system or already at boundary
        if self.n == 0 || self.t == self.t_bound {
            self.t_old = Some(self.t);
            self.t = self.t_bound;
            self.state = OdeSolverState::Finished;
            return Ok(StepOutcome {
                message: None,
                state: OdeSolverState::Finished,
            });
        }

        let t = self.t;

        // Minimum step: 10 * machine epsilon at current t
        let min_step = 10.0 * (next_after(t, self.direction * f64::INFINITY) - t).abs();

        let mut h_abs = self.h_abs.clamp(min_step, self.max_step);

        let mut step_accepted = false;
        let mut step_rejected = false;

        while !step_accepted {
            if h_abs < min_step {
                self.state = OdeSolverState::Failed;
                return Err(StepFailure::StepSizeTooSmall);
            }

            let h = h_abs * self.direction;
            let mut t_new = t + h;

            // Clamp to boundary
            if self.direction * (t_new - self.t_bound) > 0.0 {
                t_new = self.t_bound;
            }

            let h = t_new - t;
            h_abs = h.abs();

            // Perform the RK step
            let (y_new, f_new) = rk_step(fun, t, &self.y, &self.f, h, self.tableau, &mut self.k);
            self.nfev += self.tableau.n_stages; // one eval per non-zero stage + final eval

            // Compute error scale
            let scale = self.compute_scale(&self.y, &y_new);

            // Estimate error
            let err = estimate_error(&self.k, self.tableau.e, h, self.n);
            let err_norm = error_norm(&err, &scale);

            if err_norm < 1.0 {
                // Step accepted
                let factor = if err_norm == 0.0 {
                    MAX_FACTOR
                } else {
                    MAX_FACTOR.min(SAFETY * err_norm.powf(self.error_exponent))
                };

                let factor = if step_rejected {
                    factor.min(1.0)
                } else {
                    factor
                };

                h_abs *= factor;
                step_accepted = true;

                self.y_old = Some(self.y.clone());
                self.t_old = Some(t);
                self.t = t_new;
                self.y = y_new;
                self.h_abs = h_abs;
                self.f = f_new;
            } else {
                // Step rejected: decrease step size
                h_abs *= MIN_FACTOR.max(SAFETY * err_norm.powf(self.error_exponent));
                step_rejected = true;
            }
        }

        if self.direction * (self.t - self.t_bound) >= 0.0 {
            self.state = OdeSolverState::Finished;
        }

        Ok(StepOutcome {
            message: None,
            state: self.state,
        })
    }

    /// Compute the error scale: atol + max(|y|, |y_new|) * rtol.
    fn compute_scale(&self, y: &[f64], y_new: &[f64]) -> Vec<f64> {
        match &self.atol {
            ToleranceValue::Scalar(atol) => y
                .iter()
                .zip(y_new.iter())
                .map(|(yi, yni)| atol + yi.abs().max(yni.abs()) * self.rtol)
                .collect(),
            ToleranceValue::Vector(atol_vec) => y
                .iter()
                .zip(y_new.iter())
                .zip(atol_vec.iter())
                .map(|((yi, yni), ai)| ai + yi.abs().max(yni.abs()) * self.rtol)
                .collect(),
        }
    }
}

/// `next_after` equivalent: the next representable f64 toward `toward`.
fn next_after(from: f64, toward: f64) -> f64 {
    if from == toward {
        return from;
    }
    if from.is_nan() || toward.is_nan() {
        return f64::NAN;
    }
    if from == 0.0 {
        if toward > 0.0 {
            return f64::MIN_POSITIVE * f64::EPSILON;
        }
        return -f64::MIN_POSITIVE * f64::EPSILON;
    }
    let bits = from.to_bits();
    let next_bits = if (toward > from) == (from > 0.0) {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next_bits)
}

impl OdeSolver for RkSolver {
    fn mode(&self) -> RuntimeMode {
        self.mode
    }

    fn state(&self) -> OdeSolverState {
        self.state
    }

    fn step(&mut self) -> Result<StepOutcome, StepFailure> {
        // This trait method cannot accept `fun` — for trait-object usage,
        // we store f but cannot call fun. Use `step_with` directly.
        Err(StepFailure::NotYetImplemented(
            "OdeSolver::step requires step_with(fun) for RkSolver",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rk_step_single_euler_like() {
        // Trivial test: single stage with A=[], B=[1], C=[0] would be Euler
        // But we test with the actual RK45 tableau on a simple system
        let n = 1;
        let mut k = vec![vec![0.0; n]; RK45_TABLEAU.n_stages + 1];
        let (y_new, _f_new) = rk_step(
            &mut |_t, y| vec![-0.5 * y[0]],
            0.0,
            &[2.0],
            &[-1.0], // f(0, 2) = -1
            0.1,
            &RK45_TABLEAU,
            &mut k,
        );
        // y_new should be close to 2 * exp(-0.5 * 0.1) ≈ 1.90245
        assert!(
            (y_new[0] - 2.0 * (-0.05_f64).exp()).abs() < 1e-6,
            "RK45 step should be accurate for exp decay, got {}",
            y_new[0]
        );
    }

    #[test]
    fn rk23_step_exponential() {
        let n = 1;
        let mut k = vec![vec![0.0; n]; RK23_TABLEAU.n_stages + 1];
        let (y_new, _f_new) = rk_step(
            &mut |_t, y| vec![-0.5 * y[0]],
            0.0,
            &[2.0],
            &[-1.0],
            0.1,
            &RK23_TABLEAU,
            &mut k,
        );
        // RK23 is order 3, so the error should be O(h^4) ≈ 1e-4
        assert!(
            (y_new[0] - 2.0 * (-0.05_f64).exp()).abs() < 1e-4,
            "RK23 step should be reasonably accurate, got {}",
            y_new[0]
        );
    }

    #[test]
    fn solver_exponential_decay() {
        // y' = -y, y(0) = 1, integrate to t=1
        let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { vec![-y[0]] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let mut solver = RkSolver::new(&mut fun, config).expect("solver creation should succeed");
        assert_eq!(solver.state(), OdeSolverState::Running);

        let mut steps = 0;
        while solver.state() == OdeSolverState::Running {
            solver
                .step_with(&mut |_t, y| vec![-y[0]])
                .expect("step should succeed");
            steps += 1;
            if steps > 1000 {
                panic!("too many steps");
            }
        }

        assert_eq!(solver.state(), OdeSolverState::Finished);
        let expected = (-1.0_f64).exp();
        assert!(
            (solver.y()[0] - expected).abs() < 1e-6,
            "y(1) should be close to e^-1 ≈ {expected}, got {}",
            solver.y()[0]
        );
    }

    #[test]
    fn solver_two_component_system() {
        // y' = [y[1], -y[0]] (simple harmonic oscillator)
        // y(0) = [1, 0], exact: y(t) = [cos(t), -sin(t)]
        let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { vec![y[1], -y[0]] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0, 0.0],
            t_bound: std::f64::consts::PI,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let mut solver = RkSolver::new(&mut fun, config).expect("solver creation should succeed");

        let mut steps = 0;
        while solver.state() == OdeSolverState::Running {
            solver
                .step_with(&mut |_t, y| vec![y[1], -y[0]])
                .expect("step should succeed");
            steps += 1;
            if steps > 1000 {
                panic!("too many steps");
            }
        }

        // At t=pi: y ≈ [-1, 0]
        assert!(
            (solver.y()[0] - (-1.0)).abs() < 1e-6,
            "y[0](pi) should be close to -1, got {}",
            solver.y()[0]
        );
        assert!(
            solver.y()[1].abs() < 1e-5,
            "y[1](pi) should be close to 0, got {}",
            solver.y()[1]
        );
    }

    #[test]
    fn solver_empty_system() {
        let mut fun = |_t: f64, _y: &[f64]| -> Vec<f64> { vec![] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[],
            t_bound: 1.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let mut solver = RkSolver::new(&mut fun, config).expect("solver creation should succeed");
        let outcome = solver
            .step_with(&mut |_t, _y| vec![])
            .expect("step should succeed");
        assert_eq!(outcome.state, OdeSolverState::Finished);
    }

    #[test]
    fn solver_with_first_step() {
        let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { vec![-y[0]] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: Some(0.01),
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let solver = RkSolver::new(&mut fun, config).expect("solver creation should succeed");
        assert_eq!(solver.state(), OdeSolverState::Running);
    }

    #[test]
    fn solver_rk23_exponential() {
        let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { vec![-y[0]] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK23_TABLEAU,
        };
        let mut solver = RkSolver::new(&mut fun, config).expect("solver creation should succeed");

        let mut steps = 0;
        while solver.state() == OdeSolverState::Running {
            solver
                .step_with(&mut |_t, y| vec![-y[0]])
                .expect("step should succeed");
            steps += 1;
            if steps > 5000 {
                panic!("too many steps");
            }
        }

        let expected = (-1.0_f64).exp();
        assert!(
            (solver.y()[0] - expected).abs() < 1e-4,
            "RK23: y(1) should be close to e^-1 ≈ {expected}, got {}",
            solver.y()[0]
        );
    }

    #[test]
    fn next_after_basic() {
        let x = next_after(1.0, 2.0);
        assert!(x > 1.0);
        assert!(x < 1.0 + 1e-15);
        let y = next_after(1.0, 0.0);
        assert!(y < 1.0);
    }
}
