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
            return Err(IntegrateValidationError::NonFiniteY0);
        }
        if !request.f0.iter().all(|v| v.is_finite()) {
            return Err(IntegrateValidationError::NonFiniteF0);
        }
    }

    let interval_length = (request.t_bound - request.t0).abs();
    if interval_length == 0.0 {
        return Ok(0.0);
    }

    // Compute scale = atol + |y0| * rtol and d0 = norm(y0 / scale).
    // This keeps the same formula as before, but avoids building temporary
    // scaled vectors solely for RMS computation.
    let mut scale = Vec::with_capacity(n);
    let mut scaled_y0_sum_sq = 0.0_f64;
    match &request.atol {
        ToleranceValue::Scalar(atol) => {
            for &y in request.y0 {
                let s = atol + y.abs() * request.rtol;
                scale.push(s);
                let scaled = y / s;
                scaled_y0_sum_sq += scaled * scaled;
            }
        }
        ToleranceValue::Vector(atol_vec) => {
            for (&y, &a) in request.y0.iter().zip(atol_vec.iter()) {
                let s = a + y.abs() * request.rtol;
                scale.push(s);
                let scaled = y / s;
                scaled_y0_sum_sq += scaled * scaled;
            }
        }
    }
    let d0 = if scale.is_empty() {
        0.0
    } else {
        (scaled_y0_sum_sq / scale.len() as f64).sqrt()
    };

    // d1 = norm(f0 / scale)
    let mut scaled_f0_sum_sq = 0.0_f64;
    let mut scaled_f0_len = 0usize;
    for (&f, &s) in request.f0.iter().zip(scale.iter()) {
        let scaled = f / s;
        scaled_f0_sum_sq += scaled * scaled;
        scaled_f0_len += 1;
    }
    let d1 = if scaled_f0_len == 0 {
        0.0
    } else {
        (scaled_f0_sum_sq / scaled_f0_len as f64).sqrt()
    };

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
    let mut diff_scaled_sum_sq = 0.0_f64;
    let mut diff_scaled_len = 0usize;
    for ((&f1v, &f0v), &s) in f1.iter().zip(request.f0.iter()).zip(scale.iter()) {
        let diff = (f1v - f0v) / s;
        diff_scaled_sum_sq += diff * diff;
        diff_scaled_len += 1;
    }
    let d2 = if diff_scaled_len == 0 {
        0.0
    } else {
        (diff_scaled_sum_sq / diff_scaled_len as f64).sqrt() / h0
    };

    // Compute h1
    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
        if h0.is_nan() {
            f64::NAN
        } else {
            (1e-6_f64).max(h0 * 1e-3)
        }
    } else {
        let max_d = if d1.is_nan() || d2.is_nan() {
            f64::NAN
        } else {
            d1.max(d2)
        };
        (0.01 / max_d).powf(1.0 / (request.order + 1.0))
    };

    // Return min(100 * h0, h1, interval_length, max_step)
    let min_h1 = if h0.is_nan() || h1.is_nan() {
        f64::NAN
    } else {
        (100.0 * h0).min(h1)
    };
    let min_interval = if min_h1.is_nan() || interval_length.is_nan() {
        f64::NAN
    } else {
        min_h1.min(interval_length)
    };
    let final_h = if min_interval.is_nan() || request.max_step.is_nan() {
        f64::NAN
    } else {
        min_interval.min(request.max_step)
    };
    Ok(final_h)
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

    #[test]
    fn select_initial_step_hardened_rejects_non_finite_y0() {
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &[f64::NAN],
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &[1.0],
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Hardened,
        };

        let err = select_initial_step(&mut |_t, _y| vec![], &request)
            .expect_err("non-finite y0 must fail in Hardened mode");
        assert_eq!(err, IntegrateValidationError::NonFiniteY0);
    }

    #[test]
    fn select_initial_step_hardened_rejects_non_finite_f0() {
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &[f64::INFINITY],
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Hardened,
        };

        let err = select_initial_step(&mut |_t, _y| vec![], &request)
            .expect_err("non-finite f0 must fail in Hardened mode");
        assert_eq!(err, IntegrateValidationError::NonFiniteF0);
    }
}
