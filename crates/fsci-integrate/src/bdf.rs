#![forbid(unsafe_code)]

//! Backward Differentiation Formula (BDF) solver for stiff ODEs.
//!
//! Genuine variable-order (1-5) BDF with adaptive step size, a faithful port of
//! `scipy.integrate.solve_ivp(method='BDF')` (`scipy/integrate/_ivp/bdf.py`):
//! predictor from the backward-difference array `D`, modified-Newton corrector on
//! `(I − c·J)` with a lazily-refreshed finite-difference Jacobian, and combined
//! error/step/order control via `change_D`. Order and step are reconsidered after
//! `n_equal_steps >= order + 1` accepted steps by comparing the local error at
//! orders `k-1`, `k`, `k+1` (frankenscipy-3y5p9). `SolverKind::Radau` is still
//! routed here as a BDF alias pending a true Radau IIA method.

use crate::solver::{OdeSolverState, StepFailure, StepOutcome};
use crate::validation::{ToleranceValue, validate_first_step, validate_max_step, validate_tol};
use fsci_runtime::RuntimeMode;
use nalgebra::{DMatrix, DVector, Dyn, LU};

/// Maximum BDF order.
const MAX_ORDER: usize = 5;
/// Maximum Newton iterations per step (scipy `NEWTON_MAXITER`).
const NEWTON_MAXITER: usize = 4;
/// Minimum step-size reduction factor on rejection.
const MIN_FACTOR: f64 = 0.2;
/// Maximum step-size growth factor on acceptance.
const MAX_FACTOR: f64 = 10.0;

/// Empirical `kappa` constants (scipy `_bdf.py`), indices 0..=5.
const KAPPA: [f64; 6] = [0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0];
/// `gamma[i] = Σ_{k=1}^{i} 1/k`, `gamma[0] = 0`.
const GAMMA_C: [f64; 6] = [0.0, 1.0, 1.5, 11.0 / 6.0, 25.0 / 12.0, 137.0 / 60.0];
/// `alpha[i] = (1 - kappa[i]) * gamma[i]` — the BDF leading coefficient.
const ALPHA_C: [f64; 6] = [
    (1.0 - KAPPA[0]) * GAMMA_C[0],
    (1.0 - KAPPA[1]) * GAMMA_C[1],
    (1.0 - KAPPA[2]) * GAMMA_C[2],
    (1.0 - KAPPA[3]) * GAMMA_C[3],
    (1.0 - KAPPA[4]) * GAMMA_C[4],
    (1.0 - KAPPA[5]) * GAMMA_C[5],
];
/// `error_const[i] = kappa[i]*gamma[i] + 1/(i+1)` — local error coefficient.
const ERR_C: [f64; 6] = [
    KAPPA[0] * GAMMA_C[0] + 1.0,
    KAPPA[1] * GAMMA_C[1] + 1.0 / 2.0,
    KAPPA[2] * GAMMA_C[2] + 1.0 / 3.0,
    KAPPA[3] * GAMMA_C[3] + 1.0 / 4.0,
    KAPPA[4] * GAMMA_C[4] + 1.0 / 5.0,
    KAPPA[5] * GAMMA_C[5] + 1.0 / 6.0,
];

/// RMS norm `sqrt(mean(x²))` (scipy `norm`).
fn rms_norm(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let s: f64 = x.iter().map(|&v| v * v).sum();
    (s / x.len() as f64).sqrt()
}

/// scipy `compute_R(order, factor)` — the `(order+1)×(order+1)` step-change
/// matrix whose columns are cumulative products down the rows.
fn compute_r(order: usize, factor: f64) -> DMatrix<f64> {
    let m = order + 1;
    let mut mat = DMatrix::<f64>::zeros(m, m);
    // Row 0 is all ones.
    for j in 0..m {
        mat[(0, j)] = 1.0;
    }
    // M[i,j] = (i - 1 - factor*j)/i for i,j >= 1; column 0 (j=0) stays 0.
    for i in 1..m {
        for j in 1..m {
            mat[(i, j)] = ((i as f64) - 1.0 - factor * (j as f64)) / (i as f64);
        }
    }
    // Cumulative product down each column (axis 0).
    for j in 0..m {
        for i in 1..m {
            mat[(i, j)] *= mat[(i - 1, j)];
        }
    }
    mat
}

/// scipy `change_D(D, order, factor)` — rescale the difference array `d[0..=order]`
/// in place when the step size changes by `factor`.
fn change_d(d: &mut [Vec<f64>], order: usize, factor: f64, n: usize) {
    let r = compute_r(order, factor);
    let u = compute_r(order, 1.0);
    let ru = &r * &u; // (order+1)×(order+1)
    let m = order + 1;
    // new D[i] = Σ_k (RU.T)[i,k] * D[k] = Σ_k RU[k,i] * D[k].
    let mut new_d = vec![vec![0.0; n]; m];
    for i in 0..m {
        for k in 0..m {
            let w = ru[(k, i)];
            if w != 0.0 {
                for col in 0..n {
                    new_d[i][col] += w * d[k][col];
                }
            }
        }
    }
    d[..m].clone_from_slice(&new_d[..m]);
}

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
    max_order: usize,
    /// Consecutive accepted steps at the current order/step (scipy `n_equal_steps`).
    n_equal_steps: usize,
    state: OdeSolverState,
    nfev: usize,
    njev: usize,
    nlu: usize,
    mode: RuntimeMode,

    f: Vec<f64>,
    f_old: Option<Vec<f64>>,

    // Nordsieck-style array: d[k] for k = 0..order
    d: Vec<Vec<f64>>,

    // Newton solver state
    current_jac: Option<DMatrix<f64>>,
    lu: Option<LU<f64, Dyn, Dyn>>,
    /// The value of `c = h/alpha[order]` for which `lu` was factorized.
    lu_c: Option<f64>,

    // Previous step values for interpolation
    t_old: Option<f64>,
    y_old: Option<Vec<f64>>,
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

        // Input validation mirrors RkSolver::new (per frankenscipy-ljmg):
        // previously BdfSolver::new did ZERO validation — a caller passing
        // rtol=NaN, max_step=NaN, first_step=Some(NaN) etc. started a solver
        // that then burned Newton iterations until StepSizeTooSmall.
        let _validated_tol = validate_tol(
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
            )
            .min(config.max_step),
        };
        let h = h_mag * direction;

        let y0 = config.y0.to_vec();
        let f0 = fun(config.t0, &y0);

        // Backward-difference array D[0..=MAX_ORDER+2]: D[0] = y, D[1] = h*f, rest 0.
        let mut d = vec![vec![0.0; n]; MAX_ORDER + 3];
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
            max_order: config.max_order.min(MAX_ORDER),
            n_equal_steps: 0,
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
            lu_c: None,
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

    // Index-aligned array arithmetic over the difference array reads cleaner with
    // explicit `j`/`k` indices than with zipped iterators here.
    #[allow(clippy::needless_range_loop)]
    fn bdf_step_impl<F>(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        // Faithful variable-order (1-5) BDF (scipy `_bdf.py::_step_impl`):
        // predictor from the backward-difference array D, modified-Newton corrector
        // on (I − c·J) with a lazy Jacobian, error/step/order control via change_D.
        let n = self.n;
        let newton_tol = (10.0 * f64::EPSILON / self.rtol).max(0.03_f64.min(self.rtol.sqrt()));

        let spacing = if self.direction > 0.0 {
            self.t.next_up() - self.t
        } else {
            self.t - self.t.next_down()
        };
        let min_step = 10.0 * spacing.abs();

        let mut h_abs = self.h.abs();
        if h_abs > self.max_step {
            let factor = self.max_step / h_abs;
            h_abs = self.max_step;
            change_d(&mut self.d, self.order, factor, n);
            self.n_equal_steps = 0;
            self.lu = None;
        } else if h_abs < min_step {
            let factor = min_step / h_abs;
            h_abs = min_step;
            change_d(&mut self.d, self.order, factor, n);
            self.n_equal_steps = 0;
            self.lu = None;
        }

        let mut order = self.order;
        let mut t_new;
        let mut y_new = vec![0.0; n];
        let mut d_corr = vec![0.0; n];
        let mut scale = vec![0.0; n];
        let mut n_iter = 1usize;
        let mut reached_bound;

        loop {
            if h_abs < min_step {
                self.state = OdeSolverState::Failed;
                return Err(StepFailure::StepSizeTooSmall);
            }
            let mut h = h_abs * self.direction;
            t_new = self.t + h;
            reached_bound = self.direction * (t_new - self.t_bound) > 0.0;
            if reached_bound {
                t_new = self.t_bound;
                let factor = (t_new - self.t).abs() / h_abs;
                change_d(&mut self.d, order, factor, n);
                self.n_equal_steps = 0;
                self.lu = None;
            }
            h = t_new - self.t;
            h_abs = h.abs();

            // Predictor and history terms.
            let mut y_predict = vec![0.0; n];
            for dk in self.d.iter().take(order + 1) {
                for (yp, &dkj) in y_predict.iter_mut().zip(dk.iter()) {
                    *yp += dkj;
                }
            }
            for j in 0..n {
                scale[j] = self.atol[j] + self.rtol * y_predict[j].abs();
            }
            let inv_alpha = 1.0 / ALPHA_C[order];
            let mut psi = vec![0.0; n];
            for k in 1..=order {
                let g = GAMMA_C[k] * inv_alpha;
                for (p, &dkj) in psi.iter_mut().zip(self.d[k].iter()) {
                    *p += g * dkj;
                }
            }
            let c = h * inv_alpha;

            // Modified-Newton with lazy Jacobian refresh.
            let mut converged = false;
            let mut jac_recomputed = false;
            loop {
                if self.current_jac.is_none() {
                    let f_pred = fun(t_new, &y_predict);
                    self.nfev += 1;
                    let jac = self.compute_jacobian(fun, t_new, &y_predict, &f_pred);
                    self.current_jac = Some(jac);
                    self.lu = None;
                    jac_recomputed = true;
                }
                if self.lu.is_none() || self.lu_c != Some(c) {
                    let jac = self.current_jac.as_ref().expect("jacobian present");
                    let system = DMatrix::<f64>::identity(n, n) - jac.scale(c);
                    self.lu = Some(system.lu());
                    self.lu_c = Some(c);
                    self.nlu += 1;
                }
                match self.newton_bdf(fun, t_new, &y_predict, c, &psi, &scale, newton_tol) {
                    Some((iters, y_sol, d_sol)) => {
                        converged = true;
                        n_iter = iters;
                        y_new = y_sol;
                        d_corr = d_sol;
                        break;
                    }
                    None => {
                        if jac_recomputed {
                            break; // Jacobian already fresh — give up, shrink step.
                        }
                        self.current_jac = None; // force recompute next pass.
                    }
                }
            }

            if !converged {
                let factor = 0.5;
                h_abs *= factor;
                change_d(&mut self.d, order, factor, n);
                self.n_equal_steps = 0;
                self.lu = None;
                continue;
            }

            let safety = 0.9 * (2.0 * NEWTON_MAXITER as f64 + 1.0)
                / (2.0 * NEWTON_MAXITER as f64 + n_iter as f64);
            for j in 0..n {
                scale[j] = self.atol[j] + self.rtol * y_new[j].abs();
            }
            let error: Vec<f64> = (0..n)
                .map(|j| ERR_C[order] * d_corr[j] / scale[j])
                .collect();
            let error_norm = rms_norm(&error);

            if error_norm > 1.0 {
                let factor = MIN_FACTOR.max(safety * error_norm.powf(-1.0 / (order as f64 + 1.0)));
                h_abs *= factor;
                change_d(&mut self.d, order, factor, n);
                self.n_equal_steps = 0;
                self.lu = None;
            } else {
                // Step accepted.
                self.t_old = Some(self.t);
                self.y_old = Some(self.y.clone());
                self.f_old = Some(self.f.clone());

                self.n_equal_steps += 1;
                self.t = t_new;
                self.y = y_new.clone();
                self.h = h_abs * self.direction;
                self.f = fun(t_new, &y_new);
                self.nfev += 1;

                // Update the difference array.
                for j in 0..n {
                    self.d[order + 2][j] = d_corr[j] - self.d[order + 1][j];
                    self.d[order + 1][j] = d_corr[j];
                }
                for i in (0..=order).rev() {
                    for j in 0..n {
                        self.d[i][j] += self.d[i + 1][j];
                    }
                }

                // Order/step selection once enough equal steps have accumulated.
                if self.n_equal_steps > order {
                    let safety_sel = safety;
                    let err_m = if order > 1 {
                        let e: Vec<f64> = (0..n)
                            .map(|j| ERR_C[order - 1] * self.d[order][j] / scale[j])
                            .collect();
                        rms_norm(&e)
                    } else {
                        f64::INFINITY
                    };
                    let err_p = if order < self.max_order {
                        let e: Vec<f64> = (0..n)
                            .map(|j| ERR_C[order + 1] * self.d[order + 2][j] / scale[j])
                            .collect();
                        rms_norm(&e)
                    } else {
                        f64::INFINITY
                    };
                    let norms = [err_m, error_norm, err_p];
                    let mut best = 0usize;
                    let mut best_factor = f64::NEG_INFINITY;
                    for (idx, &en) in norms.iter().enumerate() {
                        // factor = en^(-1/(order-1+idx+1)) = en^(-1/(order+idx)).
                        let exp = -1.0 / (order as f64 + idx as f64);
                        let fac = if en == 0.0 {
                            f64::INFINITY
                        } else {
                            en.powf(exp)
                        };
                        if fac > best_factor {
                            best_factor = fac;
                            best = idx;
                        }
                    }
                    order = (order as isize + best as isize - 1) as usize;
                    self.order = order;
                    let factor = MAX_FACTOR.min(safety_sel * best_factor);
                    self.h = (h_abs * factor) * self.direction;
                    change_d(&mut self.d, order, factor, n);
                    self.n_equal_steps = 0;
                    self.lu = None;
                }

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

    /// Modified-Newton corrector for the BDF system at the current order
    /// (scipy `solve_bdf_system`). Solves `(I − c·J) Δ = c·f − ψ − d`, accumulating
    /// the correction `d`. Returns `Some((n_iter, y, d))` on convergence.
    #[allow(clippy::too_many_arguments)]
    fn newton_bdf<F>(
        &mut self,
        fun: &mut F,
        t_new: f64,
        y_predict: &[f64],
        c: f64,
        psi: &[f64],
        scale: &[f64],
        tol: f64,
    ) -> Option<(usize, Vec<f64>, Vec<f64>)>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = self.n;
        let mut d = vec![0.0; n];
        let mut y = y_predict.to_vec();
        let mut dy_norm_old: Option<f64> = None;
        let lu = self.lu.as_ref()?;
        for k in 0..NEWTON_MAXITER {
            let f = fun(t_new, &y);
            self.nfev += 1;
            if !f.iter().all(|v| v.is_finite()) {
                return None;
            }
            let rhs = DVector::from_iterator(n, (0..n).map(|j| c * f[j] - psi[j] - d[j]));
            let dy = lu.solve(&rhs)?;
            let dy_norm = rms_norm(&(0..n).map(|j| dy[j] / scale[j]).collect::<Vec<_>>());

            let rate = dy_norm_old.map(|old| if old > 0.0 { dy_norm / old } else { 0.0 });
            if let Some(r) = rate
                && (r >= 1.0 || r.powi((NEWTON_MAXITER - k) as i32) / (1.0 - r) * dy_norm > tol)
            {
                return None;
            }

            for j in 0..n {
                y[j] += dy[j];
                d[j] += dy[j];
            }

            if dy_norm == 0.0 || rate.is_some_and(|r| r / (1.0 - r) * dy_norm < tol) {
                return Some((k + 1, y, d));
            }
            dy_norm_old = Some(dy_norm);
        }
        None
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

    let max_d = if d1.is_nan() || d2.is_nan() {
        f64::NAN
    } else {
        d1.max(d2)
    };
    let h1 = if max_d <= 1e-15 || max_d.is_nan() {
        if h0.is_nan() {
            f64::NAN
        } else {
            (h0 * 1e-3).max(1e-6)
        }
    } else {
        (0.01 / max_d).powf(0.5)
    };

    if h0.is_nan() || h1.is_nan() {
        f64::NAN
    } else {
        (100.0 * h0).min(h1)
    }
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
    fn bdf_rejects_nan_max_step() {
        let mut fun = |_t: f64, y: &[f64]| vec![-y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::NAN,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let result = BdfSolver::new(&mut fun, config);
        assert!(matches!(
            result,
            Err(crate::IntegrateValidationError::NonFiniteMaxStep)
        ));
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
