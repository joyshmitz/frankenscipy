#![forbid(unsafe_code)]

//! Radau IIA implicit Runge-Kutta solver for stiff ODEs (3-stage, order 5),
//! matching `scipy.integrate.solve_ivp(method='Radau')`.
//!
//! Implemented as the mathematically-equivalent real `3n×3n` simplified-Newton
//! collocation: solve `Z_i = h Σ_j A_ij f(t + c_j h, y + Z_j)` for the stage
//! corrections `Z`, with the Newton matrix `I_{3n} − h (A ⊗ J)` and a lazily
//! refreshed finite-difference Jacobian `J`. The embedded order-3 error estimate
//! reuses scipy's `(MU_REAL/h) I − J` factor and the `E` coefficients. This yields
//! the same collocation solution as scipy's complex-eigenvalue formulation (only
//! the inner linear algebra differs), so results match scipy to Newton tolerance.

use crate::bdf::select_initial_step_bdf;
use crate::solver::{OdeSolverState, StepFailure, StepOutcome};
use crate::validation::{
    ToleranceValue, validate_first_step, validate_max_step, validate_rhs_shape, validate_tol,
};
use fsci_runtime::RuntimeMode;
use nalgebra::{DMatrix, DVector};

const NEWTON_MAXITER: usize = 6;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 8.0;
const ERR_EXP: f64 = -0.25; // embedded estimator is order 3 → 1/(3+1).

/// Configuration for the Radau solver (mirrors `BdfSolverConfig`).
#[derive(Debug, Clone)]
pub struct RadauSolverConfig<'a> {
    pub t0: f64,
    pub y0: &'a [f64],
    pub t_bound: f64,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub max_step: f64,
    pub first_step: Option<f64>,
    pub mode: RuntimeMode,
}

fn rms_norm(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let s: f64 = x.iter().map(|&v| v * v).sum();
    (s / x.len() as f64).sqrt()
}

/// Radau IIA solver state.
pub struct RadauSolver {
    n: usize,
    t: f64,
    y: Vec<f64>,
    t_bound: f64,
    direction: f64,
    h: f64,
    max_step: f64,
    rtol: f64,
    atol: Vec<f64>,
    mode: RuntimeMode,
    state: OdeSolverState,

    // Radau IIA tableau (3-stage, order 5).
    c: [f64; 3],
    a: [[f64; 3]; 3],
    e: [f64; 3],
    mu_real: f64,

    nfev: usize,
    njev: usize,
    nlu: usize,

    f: Vec<f64>,
    f_old: Option<Vec<f64>>,
    t_old: Option<f64>,
    y_old: Option<Vec<f64>>,

    // Lazy Jacobian / factorization caches.
    jac: Option<DMatrix<f64>>,
}

impl RadauSolver {
    pub fn new<F>(
        fun: &mut F,
        config: RadauSolverConfig<'_>,
    ) -> Result<Self, crate::IntegrateValidationError>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = config.y0.len();
        let _ = validate_tol(
            ToleranceValue::Scalar(config.rtol),
            config.atol.clone(),
            n,
            config.mode,
        )?;
        if config.max_step.is_finite() || config.max_step.is_nan() {
            validate_max_step(config.max_step)?;
        }
        if let Some(first) = config.first_step {
            validate_first_step(first, config.t0, config.t_bound)?;
        }

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
            )?
            .min(config.max_step),
        };
        let h = h_mag * direction;

        let y0 = config.y0.to_vec();
        let f0 = fun(config.t0, &y0);
        validate_rhs_shape(f0.len(), n)?;

        let s6 = 6.0_f64.sqrt();
        let c = [(4.0 - s6) / 10.0, (4.0 + s6) / 10.0, 1.0];
        let a = [
            [
                (88.0 - 7.0 * s6) / 360.0,
                (296.0 - 169.0 * s6) / 1800.0,
                (-2.0 + 3.0 * s6) / 225.0,
            ],
            [
                (296.0 + 169.0 * s6) / 1800.0,
                (88.0 + 7.0 * s6) / 360.0,
                (-2.0 - 3.0 * s6) / 225.0,
            ],
            [(16.0 - s6) / 36.0, (16.0 + s6) / 36.0, 1.0 / 9.0],
        ];
        let e = [
            (-13.0 - 7.0 * s6) / 3.0,
            (-13.0 + 7.0 * s6) / 3.0,
            -1.0 / 3.0,
        ];
        let mu_real = 3.0 + 3.0_f64.powf(2.0 / 3.0) - 3.0_f64.powf(1.0 / 3.0);

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
            mode: config.mode,
            state: OdeSolverState::Running,
            c,
            a,
            e,
            mu_real,
            nfev: 1,
            njev: 0,
            nlu: 0,
            f: f0,
            f_old: None,
            t_old: None,
            y_old: None,
            jac: None,
        })
    }

    pub fn t(&self) -> f64 {
        self.t
    }
    pub fn y(&self) -> &[f64] {
        &self.y
    }
    pub fn f(&self) -> &[f64] {
        &self.f
    }
    pub fn t_old(&self) -> Option<f64> {
        self.t_old
    }
    pub fn y_old(&self) -> Option<&[f64]> {
        self.y_old.as_deref()
    }
    pub fn f_old(&self) -> Option<&[f64]> {
        self.f_old.as_deref()
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
    pub fn state(&self) -> OdeSolverState {
        self.state
    }
    pub fn mode(&self) -> RuntimeMode {
        self.mode
    }

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
        self.radau_step(fun)
    }

    fn compute_jacobian<F>(&mut self, fun: &mut F, t: f64, y: &[f64], f0: &[f64]) -> DMatrix<f64>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let eps = f64::EPSILON.sqrt();
        let mut jac = DMatrix::<f64>::zeros(self.n, self.n);
        let mut yp = y.to_vec();
        for col in 0..self.n {
            let perturb = eps * y[col].abs().max(1.0);
            yp[col] += perturb;
            let fp = fun(t, &yp);
            self.nfev += 1;
            for row in 0..self.n {
                jac[(row, col)] = (fp[row] - f0[row]) / perturb;
            }
            yp[col] = y[col];
        }
        self.njev += 1;
        jac
    }

    #[allow(clippy::needless_range_loop)]
    fn radau_step<F>(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = self.n;
        let newton_tol = (10.0 * f64::EPSILON / self.rtol).max(0.03_f64.min(self.rtol.sqrt()));

        let spacing = if self.direction > 0.0 {
            self.t.next_up() - self.t
        } else {
            self.t - self.t.next_down()
        };
        let min_step = 10.0 * spacing.abs();

        let mut h_abs = self.h.abs().min(self.max_step).max(min_step);
        let mut rejected = false;

        loop {
            if h_abs < min_step {
                self.state = OdeSolverState::Failed;
                return Err(StepFailure::StepSizeTooSmall);
            }
            let mut h = h_abs * self.direction;
            let mut t_new = self.t + h;
            let reached_bound = self.direction * (t_new - self.t_bound) > 0.0;
            if reached_bound {
                t_new = self.t_bound;
            }
            h = t_new - self.t;
            h_abs = h.abs();
            if h_abs < min_step {
                self.state = OdeSolverState::Failed;
                return Err(StepFailure::StepSizeTooSmall);
            }

            // Ensure a Jacobian (reused across steps; recomputed only on Newton
            // failure). The two factorizations depend on h, which changes every
            // step, so they are rebuilt each attempt.
            let jac_fresh = self.jac.is_none();
            if jac_fresh {
                let f_cur = self.f.clone();
                let y_cur = self.y.clone();
                let jac = self.compute_jacobian(fun, self.t, &y_cur, &f_cur);
                self.jac = Some(jac);
            }
            let jac = self.jac.clone().expect("jacobian present");

            // M_3n = I_{3n} − h (A ⊗ J); M_real = (mu_real/h) I − J.
            let mut m = DMatrix::<f64>::zeros(3 * n, 3 * n);
            for bi in 0..3 {
                for bj in 0..3 {
                    let coef = h * self.a[bi][bj];
                    for r in 0..n {
                        for col in 0..n {
                            let mut val = -coef * jac[(r, col)];
                            if bi == bj && r == col {
                                val += 1.0;
                            }
                            m[(bi * n + r, bj * n + col)] = val;
                        }
                    }
                }
            }
            let lu_3n = m.lu();
            let m_real = DMatrix::<f64>::identity(n, n) * (self.mu_real / h) - &jac;
            let lu_real = m_real.lu();
            self.nlu += 2;

            // Simplified Newton on the stage corrections Z (3 × n), initial guess 0.
            let mut z = vec![vec![0.0; n]; 3];
            let scale: Vec<f64> = (0..n)
                .map(|j| self.atol[j] + self.rtol * self.y[j].abs())
                .collect();
            let mut converged = false;
            let mut dz_norm_old: Option<f64> = None;
            let mut bad = false;
            for k in 0..NEWTON_MAXITER {
                // Stage derivatives F_i = f(t + c_i h, y + Z_i).
                let mut fstage = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
                let mut finite = true;
                for i in 0..3 {
                    let yi: Vec<f64> = (0..n).map(|j| self.y[j] + z[i][j]).collect();
                    let fi = fun(t_new - h + self.c[i] * h, &yi);
                    self.nfev += 1;
                    if !fi.iter().all(|v| v.is_finite()) {
                        finite = false;
                    }
                    fstage[i] = fi;
                }
                if !finite {
                    bad = true;
                    break;
                }
                // Residual G_i = Z_i − h Σ_j A_ij F_j  (we solve M·ΔZ = −G).
                let mut rhs = DVector::<f64>::zeros(3 * n);
                for i in 0..3 {
                    for j in 0..n {
                        let mut acc = z[i][j];
                        for l in 0..3 {
                            acc -= h * self.a[i][l] * fstage[l][j];
                        }
                        rhs[i * n + j] = -acc;
                    }
                }
                let Some(dz) = lu_3n.solve(&rhs) else {
                    bad = true;
                    break;
                };
                let mut dz_scaled = vec![0.0; 3 * n];
                for i in 0..3 {
                    for j in 0..n {
                        let d = dz[i * n + j];
                        z[i][j] += d;
                        dz_scaled[i * n + j] = d / scale[j];
                    }
                }
                let dz_norm = rms_norm(&dz_scaled);
                let rate = dz_norm_old.map(|old| if old > 0.0 { dz_norm / old } else { 0.0 });
                if let Some(r) = rate
                    && (r >= 1.0
                        || r.powi((NEWTON_MAXITER - k) as i32) / (1.0 - r) * dz_norm > newton_tol)
                {
                    break;
                }
                if dz_norm == 0.0 || rate.is_some_and(|r| r / (1.0 - r) * dz_norm < newton_tol) {
                    converged = true;
                    break;
                }
                dz_norm_old = Some(dz_norm);
            }

            if bad || !converged {
                if jac_fresh {
                    h_abs *= 0.5;
                    rejected = true;
                    continue;
                }
                // Stale Jacobian — refresh and retry at the same h.
                self.jac = None;
                continue;
            }

            // Solution and embedded error estimate.
            let y_new: Vec<f64> = (0..n).map(|j| self.y[j] + z[2][j]).collect();
            let ze: Vec<f64> = (0..n)
                .map(|j| (self.e[0] * z[0][j] + self.e[1] * z[1][j] + self.e[2] * z[2][j]) / h)
                .collect();
            let err_scale: Vec<f64> = (0..n)
                .map(|j| self.atol[j] + self.rtol * self.y[j].abs().max(y_new[j].abs()))
                .collect();
            let mut err_rhs = DVector::<f64>::from_iterator(n, (0..n).map(|j| self.f[j] + ze[j]));
            let mut error = lu_real
                .solve(&err_rhs)
                .map(|v| (0..n).map(|j| v[j]).collect::<Vec<_>>())
                .unwrap_or_else(|| vec![f64::NAN; n]);
            let mut error_norm =
                rms_norm(&(0..n).map(|j| error[j] / err_scale[j]).collect::<Vec<_>>());

            // Stabilised estimate after a rejection (scipy): re-solve with f(t, y+error).
            if rejected && error_norm > 1.0 {
                let yp: Vec<f64> = (0..n).map(|j| self.y[j] + error[j]).collect();
                let fp = fun(self.t, &yp);
                self.nfev += 1;
                err_rhs = DVector::<f64>::from_iterator(n, (0..n).map(|j| fp[j] + ze[j]));
                if let Some(v) = lu_real.solve(&err_rhs) {
                    error = (0..n).map(|j| v[j]).collect();
                    error_norm =
                        rms_norm(&(0..n).map(|j| error[j] / err_scale[j]).collect::<Vec<_>>());
                }
            }

            if error_norm.is_nan() || error_norm > 1.0 {
                let factor = if error_norm.is_nan() {
                    0.5
                } else {
                    MIN_FACTOR.max(0.9 * error_norm.powf(ERR_EXP))
                };
                h_abs *= factor;
                rejected = true;
                continue;
            }

            // Accept.
            self.t_old = Some(self.t);
            self.y_old = Some(self.y.clone());
            self.f_old = Some(self.f.clone());
            self.t = t_new;
            self.y = y_new.clone();
            self.f = fun(t_new, &y_new);
            self.nfev += 1;

            let mut factor = MAX_FACTOR.min(0.9 * error_norm.max(1e-10).powf(ERR_EXP));
            if rejected {
                factor = factor.min(1.0);
            }
            self.h = (h_abs * factor) * self.direction;

            let state = if reached_bound {
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
    }
}
