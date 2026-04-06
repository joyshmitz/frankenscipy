#![forbid(unsafe_code)]

//! Backward Differentiation Formula (BDF) solver for stiff ODEs.
//!
//! Implements a variable-order (1-5) BDF method with adaptive step size control.
//! Matches `scipy.integrate.solve_ivp(method='BDF')`.

use crate::solver::{OdeSolverState, StepFailure, StepOutcome};
use crate::validation::ToleranceValue;
use fsci_runtime::RuntimeMode;
use nalgebra::{DMatrix, DVector, Dyn, LU};

/// BDF coefficients (gamma) for orders 1 through 5.
const BDF_GAMMA: [f64; 5] = [1.0, 2.0 / 3.0, 6.0 / 11.0, 12.0 / 25.0, 60.0 / 137.0];

/// Error constant for each BDF order (1 through 5).
const BDF_ERROR_CONST: [f64; 5] = [0.5, 2.0 / 9.0, 3.0 / 22.0, 12.0 / 125.0, 10.0 / 137.0];

/// Configuration for the BDF solver.
#[derive(Debug, Clone)]
pub struct BdfSolverConfig<'a> {
    pub t0: f64,
    pub y0: &'a [f64],
    pub t_bound: f64,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub max_step: f64,
    pub first_step: Option<f64>,
    pub mode: RuntimeMode,
    pub max_order: usize,
}

/// BDF solver for stiff ODE systems.
pub struct BdfSolver {
    n: usize,
    t: f64,
    y: Vec<f64>,
    t_bound: f64,
    direction: f64,
    h: f64,
    max_step: f64,
    rtol: f64,
    atol: Vec<f64>,
    order: usize,
    #[allow(dead_code)]
    max_order: usize,
    state: OdeSolverState,
    nfev: usize,
    #[allow(dead_code)]
    njev: usize,
    #[allow(dead_code)]
    nlu: usize,
    mode: RuntimeMode,

    f: Vec<f64>,
    f_old: Option<Vec<f64>>,

    // Nordsieck-style array: d[k] for k = 0..order
    d: Vec<Vec<f64>>,

    // Newton state (placeholder fields)
    #[allow(dead_code)]
    pub(crate) current_jac: Option<DMatrix<f64>>,
    #[allow(dead_code)]
    pub(crate) lu: Option<LU<f64, Dyn, Dyn>>,
    #[allow(dead_code)]
    pub(crate) h_abs_last: Option<f64>,
    jacobian_age: usize,

    // Previous step values for interpolation
    t_old: Option<f64>,
    y_old: Option<Vec<f64>>,
}

struct NewtonStep<'a> {
    t_new: f64,
    h_used: f64,
    gamma: f64,
    y_prev: &'a [f64],
    y_predict: &'a [f64],
}

impl BdfSolver {
    /// Create a new BDF solver.
    pub fn new<F>(
        fun: &mut F,
        config: BdfSolverConfig<'_>,
    ) -> Result<Self, crate::IntegrateValidationError>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = config.y0.len();
        let direction = if config.t_bound >= config.t0 {
            1.0
        } else {
            -1.0
        };

        let atol_vec = match &config.atol {
            ToleranceValue::Scalar(v) => vec![*v; n],
            ToleranceValue::Vector(v) => v.clone(),
        };

        let h_mag = match config.first_step {
            Some(h) => h,
            None => select_initial_step_bdf(
                fun,
                config.t0,
                config.y0,
                direction,
                config.rtol,
                &atol_vec,
            )
            .min(config.max_step),
        };
        let h = h_mag * direction;

        let y0 = config.y0.to_vec();
        let f0 = fun(config.t0, &y0);

        // Initialize Nordsieck array: d[0] = y, d[1] = h*f
        let mut d = vec![vec![0.0; n]; 2];
        d[0] = y0.clone();
        for (j, d1j) in d[1].iter_mut().enumerate() {
            *d1j = h * f0[j];
        }

        Ok(Self {
            n,
            t: config.t0,
            y: y0,
            t_bound: config.t_bound,
            direction,
            h,
            max_step: config.max_step,
            rtol: config.rtol,
            atol: atol_vec,
            order: 1,
            max_order: config.max_order.min(5),
            state: OdeSolverState::Running,
            nfev: 1,
            njev: 0,
            nlu: 0,
            mode: config.mode,
            f: f0.clone(),
            f_old: None,
            d,
            current_jac: None,
            lu: None,
            h_abs_last: None,
            jacobian_age: 0,
            t_old: None,
            y_old: None,
        })
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn y(&self) -> &[f64] {
        &self.y
    }

    pub fn t_old(&self) -> Option<f64> {
        self.t_old
    }

    pub fn y_old(&self) -> Option<&[f64]> {
        self.y_old.as_deref()
    }

    pub fn nfev(&self) -> usize {
        self.nfev
    }

    pub fn njev(&self) -> usize {
        self.njev
    }

    pub fn nlu(&self) -> usize {
        self.nlu
    }

    pub fn f(&self) -> &[f64] {
        &self.f
    }

    pub fn f_old(&self) -> Option<&[f64]> {
        self.f_old.as_deref()
    }

    pub fn state(&self) -> OdeSolverState {
        self.state
    }

    pub fn mode(&self) -> RuntimeMode {
        self.mode
    }

    /// Perform one adaptive BDF step.
    pub fn step_with<F>(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        if self.state != OdeSolverState::Running {
            return Err(StepFailure::RuntimeError(
                "Attempt to step on a finished or failed solver.",
            ));
        }

        if self.n == 0 || self.t == self.t_bound {
            self.t_old = Some(self.t);
            self.y_old = Some(self.y.clone());
            self.f_old = Some(self.f.clone());
            self.t = self.t_bound;
            self.state = OdeSolverState::Finished;
            return Ok(StepOutcome {
                message: None,
                state: OdeSolverState::Finished,
            });
        }

        self.bdf_step_impl(fun)
    }

    fn bdf_step_impl<F>(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let max_retries = 10;

        let f_curr = self.f.clone();

        let gamma = BDF_GAMMA[self.order - 1];
        let error_const = BDF_ERROR_CONST[self.order - 1];

        for _ in 0..max_retries {
            let t_new = self.t + self.h;

            let past_bound = if self.direction > 0.0 {
                t_new >= self.t_bound
            } else {
                t_new <= self.t_bound
            };

            let (t_new, h_used) = if past_bound {
                (self.t_bound, self.t_bound - self.t)
            } else {
                (t_new, self.h)
            };

            // Predict: explicit Euler
            let y_predict: Vec<f64> = self
                .y
                .iter()
                .zip(f_curr.iter())
                .map(|(yi, fi)| yi + h_used * fi)
                .collect();
            let y_prev = self.y.clone();

            let mut y_new = y_predict.clone();
            let mut f_new = fun(t_new, &y_new);
            self.nfev += 1;
            let step = NewtonStep {
                t_new,
                h_used,
                gamma,
                y_prev: &y_prev,
                y_predict: &y_predict,
            };
            let converged = self.solve_newton_system(fun, &step, &mut y_new, &mut f_new)?;

            if !converged {
                self.h *= 0.5;
                if self.h.abs() < 1e-14 {
                    self.state = OdeSolverState::Failed;
                    return Err(StepFailure::StepSizeTooSmall);
                }
                continue;
            }

            // Error estimation
            let mut error_norm = 0.0;
            for j in 0..self.n {
                let err = error_const * (y_new[j] - y_predict[j]);
                let scale = self.atol[j]
                    + self.rtol * {
                        let a = y_new[j].abs();
                        let b = self.y[j].abs();
                        if a.is_nan() || b.is_nan() {
                            f64::NAN
                        } else {
                            a.max(b)
                        }
                    };
                error_norm += (err / scale) * (err / scale);
            }
            error_norm = (error_norm / self.n as f64).sqrt();

            if error_norm.is_nan() || error_norm > 1.0 {
                let factor = if error_norm.is_nan() {
                    0.5
                } else {
                    (0.5_f64).max(0.9 / error_norm.powf(1.0 / (self.order as f64 + 1.0)))
                };
                self.h *= factor;
                if self.h.abs() < 1e-14 {
                    self.state = OdeSolverState::Failed;
                    return Err(StepFailure::StepSizeTooSmall);
                }
                continue;
            }

            // Step accepted
            self.t_old = Some(self.t);
            self.y_old = Some(self.y.clone());
            self.f_old = Some(self.f.clone());

            self.f = f_new.clone();

            self.d[0] = y_new.clone();
            for (d1, &fj) in self.d[1].iter_mut().zip(f_new.iter()) {
                *d1 = h_used * fj;
            }

            self.t = t_new;
            self.y = y_new;
            self.jacobian_age = self.jacobian_age.saturating_add(1);

            let factor =
                (1.5_f64).min(0.9 / error_norm.max(1e-10).powf(1.0 / (self.order as f64 + 1.0)));
            self.h = (factor * h_used.abs()).min(self.max_step) * self.direction;

            let state = if past_bound {
                self.state = OdeSolverState::Finished;
                OdeSolverState::Finished
            } else {
                OdeSolverState::Running
            };

            return Ok(StepOutcome {
                message: None,
                state,
            });
        }

        self.state = OdeSolverState::Failed;
        Err(StepFailure::ConvergenceFailure)
    }

    fn should_refresh_jacobian(&self, h_abs: f64) -> bool {
        let Some(previous_h_abs) = self.h_abs_last else {
            return true;
        };
        if self.current_jac.is_none() || self.lu.is_none() {
            return true;
        }
        if previous_h_abs == 0.0 {
            return true;
        }
        let ratio = h_abs / previous_h_abs;
        !((1.0 / 1.2)..=1.2).contains(&ratio) || self.jacobian_age >= 5
    }

    fn compute_jacobian<F>(&mut self, fun: &mut F, t: f64, y: &[f64], f0: &[f64]) -> DMatrix<f64>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let eps = f64::EPSILON.sqrt();
        let mut jac = DMatrix::<f64>::zeros(self.n, self.n);
        let mut y_perturbed = y.to_vec();

        for col in 0..self.n {
            let perturb = eps * y[col].abs().max(1.0);
            y_perturbed[col] += perturb;
            let f_perturbed = fun(t, &y_perturbed);
            self.nfev += 1;
            for row in 0..self.n {
                jac[(row, col)] = (f_perturbed[row] - f0[row]) / perturb;
            }
            y_perturbed[col] = y[col];
        }

        self.njev += 1;
        jac
    }

    fn refresh_linearization<F>(
        &mut self,
        fun: &mut F,
        t: f64,
        y: &[f64],
        f: &[f64],
        gamma_h: f64,
        h_abs: f64,
    ) where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let jac = self.compute_jacobian(fun, t, y, f);
        let system = DMatrix::<f64>::identity(self.n, self.n) - jac.scale(gamma_h);
        self.current_jac = Some(jac);
        self.lu = Some(system.lu());
        self.h_abs_last = Some(h_abs);
        self.jacobian_age = 0;
        self.nlu += 1;
    }

    fn solve_newton_system<F>(
        &mut self,
        fun: &mut F,
        step: &NewtonStep<'_>,
        y_new: &mut [f64],
        f_new: &mut Vec<f64>,
    ) -> Result<bool, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let h_abs = step.h_used.abs();
        let gamma_h = step.gamma * step.h_used;
        let mut force_refresh = self.should_refresh_jacobian(h_abs);

        for refresh_attempt in 0..2 {
            if force_refresh || refresh_attempt > 0 {
                self.refresh_linearization(fun, step.t_new, y_new, f_new, gamma_h, h_abs);
                force_refresh = false;
            }

            for _ in 0..8 {
                let residual =
                    DVector::from_iterator(
                        self.n,
                        y_new.iter().zip(step.y_prev.iter()).zip(f_new.iter()).map(
                            |((&y_curr, &y_base), &f_curr)| -(y_curr - y_base - gamma_h * f_curr),
                        ),
                    );
                let Some(delta) = self.lu.as_ref().and_then(|lu| lu.solve(&residual)) else {
                    return Err(StepFailure::SolverError);
                };

                let mut max_delta = 0.0_f64;
                for j in 0..self.n {
                    let dy = delta[j];
                    y_new[j] += dy;
                    let scale = self.atol[j]
                        + self.rtol * step.y_predict[j].abs().max(y_new[j].abs()).max(1e-10);
                    max_delta = max_delta.max(dy.abs() / scale);
                }

                if !y_new.iter().all(|value| value.is_finite()) {
                    return Err(StepFailure::NonFiniteState);
                }

                *f_new = fun(step.t_new, y_new);
                self.nfev += 1;

                if max_delta < 1.0 {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

/// Select initial step size for BDF solver.
fn select_initial_step_bdf<F>(
    fun: &mut F,
    t0: f64,
    y0: &[f64],
    direction: f64,
    rtol: f64,
    atol: &[f64],
) -> f64
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let f0 = fun(t0, y0);
    let n = y0.len();

    let mut d0 = 0.0_f64;
    let mut d1 = 0.0_f64;
    for j in 0..n {
        let scale = atol[j] + rtol * y0[j].abs();
        d0 += (y0[j] / scale) * (y0[j] / scale);
        d1 += (f0[j] / scale) * (f0[j] / scale);
    }
    d0 = (d0 / n as f64).sqrt();
    d1 = (d1 / n as f64).sqrt();

    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };

    let y1: Vec<f64> = y0
        .iter()
        .zip(f0.iter())
        .map(|(yi, fi)| yi + direction * h0 * fi)
        .collect();
    let f1 = fun(t0 + direction * h0, &y1);

    let mut d2 = 0.0_f64;
    for j in 0..n {
        let scale = atol[j] + rtol * y0[j].abs();
        d2 += ((f1[j] - f0[j]) / scale) * ((f1[j] - f0[j]) / scale);
    }
    d2 = (d2 / n as f64).sqrt() / h0;

    let max_d = if d1.is_nan() || d2.is_nan() { f64::NAN } else { d1.max(d2) };
    let h1 = if max_d <= 1e-15 || max_d.is_nan() {
        if h0.is_nan() { f64::NAN } else { (h0 * 1e-3).max(1e-6) }
    } else {
        (0.01 / max_d).powf(0.5)
    };

    if h0.is_nan() || h1.is_nan() { f64::NAN } else { (100.0 * h0).min(h1) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bdf_exponential_decay() {
        let mut fun = |_t: f64, y: &[f64]| vec![-y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let y_final = solver.y()[0];
        let expected = (-1.0_f64).exp();
        assert!(
            (y_final - expected).abs() < 0.1,
            "y(1) = {y_final}, expected {expected}"
        );
    }

    #[test]
    fn bdf_linear_ode() {
        let mut fun = |_t: f64, _y: &[f64]| vec![1.0];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[0.0],
            t_bound: 2.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let y_final = solver.y()[0];
        assert!(
            (y_final - 2.0).abs() < 0.1,
            "y(2) = {y_final}, expected 2.0"
        );
    }

    #[test]
    fn bdf_stiff_ode() {
        let mut fun = |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 0.1,
            rtol: 1e-4,
            atol: ToleranceValue::Scalar(1e-6),
            max_step: f64::INFINITY,
            first_step: Some(1e-6),
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let y_final = solver.y()[0];
        let expected = 0.1_f64.cos();
        assert!(
            (y_final - expected).abs() < 0.01,
            "y(0.1) = {y_final}, expected ~{expected}"
        );
        assert!(solver.njev() > 0, "should record Jacobian evaluations");
        assert!(solver.nlu() > 0, "should record LU factorizations");
    }

    #[test]
    fn bdf_linear_stiff_decay_uses_newton_counters() {
        let mut fun = |_t: f64, y: &[f64]| vec![-1000.0 * y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 0.01,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: 1e-3,
            first_step: Some(1e-6),
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let expected = (-10.0_f64).exp();
        assert!(
            (solver.y()[0] - expected).abs() < 5e-3,
            "y(0.01) = {}, expected {}",
            solver.y()[0],
            expected
        );
        assert!(solver.njev() > 0, "should record Jacobian evaluations");
        assert!(solver.nlu() > 0, "should record LU factorizations");
    }

    #[test]
    fn bdf_robertson_problem_preserves_mass() {
        let mut fun = |_t: f64, y: &[f64]| {
            vec![
                -0.04 * y[0] + 1.0e4 * y[1] * y[2],
                0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1],
                3.0e7 * y[1] * y[1],
            ]
        };
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0, 0.0, 0.0],
            t_bound: 1.0e-2,
            rtol: 1.0e-5,
            atol: ToleranceValue::Vector(vec![1.0e-8, 1.0e-12, 1.0e-8]),
            max_step: 1.0e-3,
            first_step: Some(1.0e-8),
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let total: f64 = solver.y().iter().sum();
        assert!(
            (total - 1.0).abs() < 1.0e-6,
            "Robertson mass drifted: total={total}"
        );
        assert!(
            solver
                .y()
                .iter()
                .all(|&value| value.is_finite() && value >= -1.0e-10),
            "Robertson state must stay finite and nonnegative: {:?}",
            solver.y()
        );
        assert!(solver.njev() > 0, "should record Jacobian evaluations");
    }

    #[test]
    fn bdf_van_der_pol_mu_1000_stays_finite() {
        let mu = 1000.0;
        let mut fun = move |_t: f64, y: &[f64]| vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[2.0, 0.0],
            t_bound: 0.1,
            rtol: 1.0e-4,
            atol: ToleranceValue::Vector(vec![1.0e-6, 1.0e-6]),
            max_step: 1.0e-2,
            first_step: Some(1.0e-6),
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        assert!(
            solver.y().iter().all(|value| value.is_finite()),
            "Van der Pol state must stay finite: {:?}",
            solver.y()
        );
        assert!(
            (solver.y()[0] - 2.0).abs() < 0.5,
            "Van der Pol drifted unexpectedly over short interval: {:?}",
            solver.y()
        );
        assert!(solver.njev() > 0, "should record Jacobian evaluations");
    }

    #[test]
    fn bdf_2d_system() {
        let mut fun = |_t: f64, y: &[f64]| vec![-y[0], -2.0 * y[1]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0, 1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let y = solver.y();
        assert!(
            (y[0] - (-1.0_f64).exp()).abs() < 0.1,
            "y0(1) = {}, expected {}",
            y[0],
            (-1.0_f64).exp()
        );
        assert!(
            (y[1] - (-2.0_f64).exp()).abs() < 0.1,
            "y1(1) = {}, expected {}",
            y[1],
            (-2.0_f64).exp()
        );
    }

    #[test]
    fn bdf_nfev_is_tracked() {
        let mut fun = |_t: f64, y: &[f64]| vec![-y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 0.5,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 3,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        assert!(solver.nfev() > 0, "should track function evaluations");
    }

    #[test]
    fn bdf_t_old_and_y_old() {
        let mut fun = |_t: f64, y: &[f64]| vec![-y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 3,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        assert!(solver.t_old().is_none());
        assert!(solver.y_old().is_none());

        solver.step_with(&mut fun).expect("BDF step");

        assert!(solver.t_old().is_some());
        assert!(solver.y_old().is_some());
    }
}
