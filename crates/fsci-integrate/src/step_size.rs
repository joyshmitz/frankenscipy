#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::{IntegrateValidationError, ToleranceValue};

/// Type alias for the right-hand side function used in step size selection.
pub type StepRhsFn = dyn FnMut(f64, &[f64]) -> Vec<f64>;

/// Request parameters for the initial step size heuristic.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialStepRequest<'a> {
    pub t0: f64,
    pub y0: &'a [f64],
    pub t_bound: f64,
    pub max_step: f64,
    pub f0: &'a [f64],
    pub direction: f64,
    pub order: f64,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub mode: RuntimeMode,
}

/// Compute RMS norm: ||x|| / sqrt(n).
fn rms_norm(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = x.iter().map(|v| v * v).sum();
    (sum_sq / x.len() as f64).sqrt()
}

/// Empirically select a good initial step size.
///
/// Implements the algorithm from Hairer, Norsett & Wanner,
/// "Solving Ordinary Differential Equations I: Nonstiff Problems", Sec. II.4.
///
/// # Contract (P2C-001-D2)
/// - Matches SciPy's `select_initial_step` from `_ivp/common.py`.
/// - Returns `Ok(h_abs)` with h_abs > 0 on success.
/// - Returns `Ok(f64::INFINITY)` for empty systems (n=0).
/// - Returns `Ok(0.0)` for zero-length intervals.
/// - In Hardened mode, validates that all inputs are finite.
pub fn select_initial_step<F>(
    fun: &mut F,
    request: &InitialStepRequest<'_>,
) -> Result<f64, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let n = request.y0.len();

    // Empty system: infinite step is fine.
    if n == 0 {
        return Ok(f64::INFINITY);
    }

    // Hardened mode: validate inputs are finite.
    if request.mode == RuntimeMode::Hardened {
        if !request.y0.iter().all(|v| v.is_finite()) {
            eprintln!(
                "{{\"function\":\"select_initial_step\",\"event\":\"non_finite_y0\",\"mode\":\"Hardened\"}}"
            );
            return Err(IntegrateValidationError::NotYetImplemented {
                function: "select_initial_step: non-finite y0 in Hardened mode",
            });
        }
        if !request.f0.iter().all(|v| v.is_finite()) {
            eprintln!(
                "{{\"function\":\"select_initial_step\",\"event\":\"non_finite_f0\",\"mode\":\"Hardened\"}}"
            );
            return Err(IntegrateValidationError::NotYetImplemented {
                function: "select_initial_step: non-finite f0 in Hardened mode",
            });
        }
    }

    let interval_length = (request.t_bound - request.t0).abs();
    if interval_length == 0.0 {
        return Ok(0.0);
    }

    // Compute scale = atol + |y0| * rtol
    let scale: Vec<f64> = match &request.atol {
        ToleranceValue::Scalar(atol) => request
            .y0
            .iter()
            .map(|y| atol + y.abs() * request.rtol)
            .collect(),
        ToleranceValue::Vector(atol_vec) => request
            .y0
            .iter()
            .zip(atol_vec.iter())
            .map(|(y, a)| a + y.abs() * request.rtol)
            .collect(),
    };

    // d0 = norm(y0 / scale), d1 = norm(f0 / scale)
    let scaled_y0: Vec<f64> = request
        .y0
        .iter()
        .zip(scale.iter())
        .map(|(y, s)| y / s)
        .collect();
    let scaled_f0: Vec<f64> = request
        .f0
        .iter()
        .zip(scale.iter())
        .map(|(f, s)| f / s)
        .collect();
    let d0 = rms_norm(&scaled_y0);
    let d1 = rms_norm(&scaled_f0);

    // Initial guess h0
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    let h0 = h0.min(interval_length);

    // Euler step: y1 = y0 + h0 * direction * f0
    let y1: Vec<f64> = request
        .y0
        .iter()
        .zip(request.f0.iter())
        .map(|(y, f)| y + h0 * request.direction * f)
        .collect();

    // Evaluate f1 = fun(t0 + h0 * direction, y1)
    let f1 = fun(request.t0 + h0 * request.direction, &y1);

    // d2 = norm((f1 - f0) / scale) / h0
    let diff_scaled: Vec<f64> = f1
        .iter()
        .zip(request.f0.iter())
        .zip(scale.iter())
        .map(|((f1v, f0v), s)| (f1v - f0v) / s)
        .collect();
    let d2 = rms_norm(&diff_scaled) / h0;

    // Compute h1
    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
        (1e-6_f64).max(h0 * 1e-3)
    } else {
        (0.01 / d1.max(d2)).powf(1.0 / (request.order + 1.0))
    };

    // Return min(100 * h0, h1, interval_length, max_step)
    Ok((100.0 * h0)
        .min(h1)
        .min(interval_length)
        .min(request.max_step))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_initial_step_empty_system() {
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &[],
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &[],
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, _y| vec![], &request).unwrap();
        assert!(h.is_infinite());
    }

    #[test]
    fn select_initial_step_zero_interval() {
        let request = InitialStepRequest {
            t0: 1.0,
            y0: &[1.0],
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &[0.5],
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, _y| vec![0.5], &request).unwrap();
        assert_eq!(h, 0.0);
    }

    #[test]
    fn select_initial_step_exponential_decay() {
        // y' = -0.5*y, y(0) = 1.0
        let y0 = [1.0];
        let f0 = [-0.5]; // fun(0, [1.0])
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &y0,
            t_bound: 10.0,
            max_step: f64::INFINITY,
            f0: &f0,
            direction: 1.0,
            order: 4.0, // RK45 error_estimator_order
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, y| vec![-0.5 * y[0]], &request).unwrap();
        assert!(h > 0.0, "step must be positive, got {h}");
        assert!(h <= 10.0, "step must not exceed interval, got {h}");
    }

    #[test]
    fn select_initial_step_respects_max_step() {
        let y0 = [1.0];
        let f0 = [-0.5];
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &y0,
            t_bound: 100.0,
            max_step: 0.001,
            f0: &f0,
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, y| vec![-0.5 * y[0]], &request).unwrap();
        assert!(h <= 0.001, "step must respect max_step, got {h}");
    }

    #[test]
    fn select_initial_step_backward_integration() {
        let y0 = [1.0];
        let f0 = [-0.5];
        let request = InitialStepRequest {
            t0: 10.0,
            y0: &y0,
            t_bound: 0.0,
            max_step: f64::INFINITY,
            f0: &f0,
            direction: -1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, y| vec![-0.5 * y[0]], &request).unwrap();
        assert!(h > 0.0, "h_abs must be positive");
        assert!(h <= 10.0, "h_abs must not exceed interval");
    }

    #[test]
    fn select_initial_step_vector_atol() {
        let y0 = [1.0, 100.0];
        let f0 = [-0.5, -50.0];
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &y0,
            t_bound: 10.0,
            max_step: f64::INFINITY,
            f0: &f0,
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Vector(vec![1e-6, 1e-4]),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, y| vec![-0.5 * y[0], -0.5 * y[1]], &request).unwrap();
        assert!(h > 0.0);
    }

    #[test]
    fn select_initial_step_small_derivatives() {
        // When d0 < 1e-5 and d1 < 1e-5, h0 = 1e-6
        let y0 = [1e-8];
        let f0 = [1e-8];
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &y0,
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &f0,
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, _y| vec![1e-8], &request).unwrap();
        assert!(h > 0.0);
    }
}
