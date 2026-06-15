//! Complex-valued ODE integration — `scipy.integrate.complex_ode`.
//!
//! SciPy's `complex_ode` integrates a complex system `dy/dt = f(t, y)` by
//! reformulating it as a real system of twice the dimension. We follow the same
//! recipe, layering a thin wrapper over the real [`solve_ivp`] integrator: the
//! complex state `y` of length `n` is packed into a real vector
//! `[Re(y_0), …, Re(y_{n-1}), Im(y_0), …, Im(y_{n-1})]`, integrated, and
//! unpacked back into complex form.
//!
//! FrankenSciPy is functional rather than class-based, so this is exposed as a
//! single call instead of SciPy's stateful `complex_ode(...).integrate(t)`
//! object; the numerical result matches SciPy's `complex_ode` to tolerance.

use crate::api::{SolveIvpOptions, SolverKind, solve_ivp};
use crate::validation::{IntegrateValidationError, ToleranceValue};

/// A complex number as a `(real, imaginary)` pair, matching the workspace
/// `Complex64` convention used elsewhere (e.g. `fsci-fft`).
pub type Complex64 = (f64, f64);

/// Result of [`complex_ode`].
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexOdeResult {
    /// Time points at which the solution was evaluated.
    pub t: Vec<f64>,
    /// Complex state at each time point; `y[i]` is the state vector at `t[i]`.
    pub y: Vec<Vec<Complex64>>,
    /// Whether the integration reached the end of the interval.
    pub success: bool,
    /// Solver status message.
    pub message: String,
    /// Number of (real) right-hand-side evaluations.
    pub nfev: usize,
}

/// Integrate a complex-valued initial value problem `dy/dt = f(t, y)`.
///
/// Mirrors `scipy.integrate.complex_ode`: `fun(t, y)` maps the current time and
/// complex state vector to the complex derivative, `y0` is the complex initial
/// state, and the system is integrated over `t_span = (t0, tf)`. If `t_eval` is
/// given, the solution is reported at those points; otherwise at the solver's
/// own steps. `rtol`/`atol` are the relative/absolute tolerances.
///
/// Uses an explicit Runge-Kutta (`RK45`) integrator on the real reformulation,
/// matching SciPy's `complex_ode(...).set_integrator('dopri5')` to tolerance.
///
/// # Errors
/// Propagates [`IntegrateValidationError`] from the underlying real solver
/// (e.g. empty initial state or a non-finite span).
pub fn complex_ode<F>(
    mut fun: F,
    y0: &[Complex64],
    t_span: (f64, f64),
    t_eval: Option<&[f64]>,
    rtol: f64,
    atol: f64,
) -> Result<ComplexOdeResult, IntegrateValidationError>
where
    F: FnMut(f64, &[Complex64]) -> Vec<Complex64>,
{
    let n = y0.len();

    // Pack the complex initial state into [Re.., Im..].
    let mut real_y0 = Vec::with_capacity(2 * n);
    real_y0.extend(y0.iter().map(|c| c.0));
    real_y0.extend(y0.iter().map(|c| c.1));

    // Wrap the complex RHS as a real RHS on the doubled state.
    let mut real_fun = |t: f64, u: &[f64]| -> Vec<f64> {
        let yc: Vec<Complex64> = (0..n).map(|j| (u[j], u[n + j])).collect();
        let dyc = fun(t, &yc);
        let mut out = vec![0.0; 2 * n];
        for (j, d) in dyc.iter().take(n).enumerate() {
            out[j] = d.0;
            out[n + j] = d.1;
        }
        out
    };

    let opts = SolveIvpOptions {
        t_span,
        y0: &real_y0,
        method: SolverKind::Rk45,
        t_eval,
        rtol,
        atol: ToleranceValue::Scalar(atol),
        ..Default::default()
    };

    let res = solve_ivp(&mut real_fun, &opts)?;

    // Unpack each real state row back into complex form.
    let y: Vec<Vec<Complex64>> = res
        .y
        .iter()
        .map(|row| (0..n).map(|j| (row[j], row[n + j])).collect())
        .collect();

    Ok(ComplexOdeResult {
        t: res.t,
        y,
        success: res.success,
        message: res.message,
        nfev: res.nfev,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmul(a: Complex64, b: Complex64) -> Complex64 {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    #[test]
    fn complex_ode_exponential_matches_analytic_and_scipy() {
        // y' = i*y, y(0) = 1 -> y(t) = exp(i t); at t = 2: (cos 2, sin 2).
        let f = |_t: f64, y: &[Complex64]| vec![cmul((0.0, 1.0), y[0])];
        let r = complex_ode(f, &[(1.0, 0.0)], (0.0, 2.0), Some(&[2.0]), 1e-9, 1e-12).unwrap();
        assert!(r.success);
        let y = r.y.last().unwrap()[0];
        let (re, im) = (2.0_f64.cos(), 2.0_f64.sin());
        assert!((y.0 - re).abs() < 1e-7, "re = {}, expected {re}", y.0);
        assert!((y.1 - im).abs() < 1e-7, "im = {}, expected {im}", y.1);
    }

    #[test]
    fn complex_ode_decaying_spiral() {
        // y' = (-1 + 2i)*y, y(0) = 1 -> exp((-1+2i) t); at t = 1.5.
        let f = |_t: f64, y: &[Complex64]| vec![cmul((-1.0, 2.0), y[0])];
        let r = complex_ode(f, &[(1.0, 0.0)], (0.0, 1.5), Some(&[1.5]), 1e-9, 1e-12).unwrap();
        let y = r.y.last().unwrap()[0];
        // exp(-1.5) * (cos 3, sin 3)
        let mag = (-1.5_f64).exp();
        assert!((y.0 - mag * 3.0_f64.cos()).abs() < 1e-7, "re = {}", y.0);
        assert!((y.1 - mag * 3.0_f64.sin()).abs() < 1e-7, "im = {}", y.1);
    }

    #[test]
    fn complex_ode_coupled_pair() {
        // y0' = i*y1, y1' = i*y0; y(0) = (1, 0). y0(t) = cos t, y1(t) = i sin t.
        let f = |_t: f64, y: &[Complex64]| vec![cmul((0.0, 1.0), y[1]), cmul((0.0, 1.0), y[0])];
        let r = complex_ode(f, &[(1.0, 0.0), (0.0, 0.0)], (0.0, 1.0), Some(&[1.0]), 1e-9, 1e-12)
            .unwrap();
        let last = r.y.last().unwrap();
        assert!((last[0].0 - 1.0_f64.cos()).abs() < 1e-7);
        assert!(last[0].1.abs() < 1e-7);
        assert!(last[1].0.abs() < 1e-7);
        assert!((last[1].1 - 1.0_f64.sin()).abs() < 1e-7);
    }
}
