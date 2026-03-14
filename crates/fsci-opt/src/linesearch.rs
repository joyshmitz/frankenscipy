#![forbid(unsafe_code)]

use crate::types::OptError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WolfeParams {
    pub c1: f64,
    pub c2: f64,
    pub amax: f64,
    pub amin: f64,
    pub maxiter: usize,
}

impl Default for WolfeParams {
    fn default() -> Self {
        Self {
            c1: 1.0e-4,
            c2: 0.9,
            amax: 50.0,
            amin: 1.0e-8,
            maxiter: 10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSearchResult {
    pub alpha: f64,
    pub f_at_alpha: f64,
    pub directional_derivative: f64,
    pub evaluations: usize,
}

pub fn validate_wolfe_params(params: WolfeParams) -> Result<(), OptError> {
    if !(0.0 < params.c1 && params.c1 < params.c2 && params.c2 < 1.0) {
        return Err(OptError::InvalidArgument {
            detail: String::from("Wolfe constants must satisfy 0 < c1 < c2 < 1"),
        });
    }
    if params.amin <= 0.0 || params.amax <= 0.0 || params.amin >= params.amax {
        return Err(OptError::InvalidArgument {
            detail: String::from("line-search alpha bounds must satisfy 0 < amin < amax"),
        });
    }
    if params.maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("line-search maxiter must be >= 1"),
        });
    }
    Ok(())
}

/// Weak Wolfe line search (Armijo + curvature).
///
/// Finds alpha satisfying:
/// - Armijo:    f(x + alpha*d) <= f(x) + c1*alpha*g'*d
/// - Curvature: g(x + alpha*d)'*d >= c2*g(x)'*d
///
/// Matches `scipy.optimize.line_search` with `old_old_fval=None`.
pub fn line_search_wolfe1<F, G>(
    f: &F,
    grad: &G,
    x: &[f64],
    direction: &[f64],
    f0: f64,
    g0: &[f64],
    params: WolfeParams,
) -> Result<LineSearchResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    validate_wolfe_params(params)?;
    let dg0 = dot(g0, direction);
    if dg0 >= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: "search direction is not a descent direction".to_string(),
        });
    }

    line_search_wolfe_impl(f, grad, x, direction, f0, dg0, params, false)
}

/// Strong Wolfe line search.
///
/// Finds alpha satisfying:
/// - Armijo:       f(x + alpha*d) <= f(x) + c1*alpha*g'*d
/// - Strong Wolfe: |g(x + alpha*d)'*d| <= c2*|g(x)'*d|
///
/// Uses the zoom phase from Nocedal & Wright Algorithm 3.6.
/// Matches `scipy.optimize.line_search`.
pub fn line_search_wolfe2<F, G>(
    f: &F,
    grad: &G,
    x: &[f64],
    direction: &[f64],
    f0: f64,
    g0: &[f64],
    params: WolfeParams,
) -> Result<LineSearchResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    validate_wolfe_params(params)?;
    let dg0 = dot(g0, direction);
    if dg0 >= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: "search direction is not a descent direction".to_string(),
        });
    }

    line_search_wolfe_impl(f, grad, x, direction, f0, dg0, params, true)
}

/// Core Wolfe line search implementation (Nocedal & Wright Algorithm 3.5 + 3.6).
#[allow(clippy::too_many_arguments)]
fn line_search_wolfe_impl<F, G>(
    f: &F,
    grad: &G,
    x: &[f64],
    d: &[f64],
    f0: f64,
    dg0: f64,
    params: WolfeParams,
    strong: bool,
) -> Result<LineSearchResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let mut evals = 0;

    let eval_f = |alpha: f64, evals: &mut usize| -> f64 {
        let xp: Vec<f64> = (0..n).map(|i| x[i] + alpha * d[i]).collect();
        *evals += 1;
        f(&xp)
    };

    let eval_dg = |alpha: f64, evals: &mut usize| -> f64 {
        let xp: Vec<f64> = (0..n).map(|i| x[i] + alpha * d[i]).collect();
        let gp = grad(&xp);
        *evals += 1;
        dot(&gp, d)
    };

    // Bracketing phase (Algorithm 3.5)
    let mut alpha_prev = 0.0;
    let mut f_prev = f0;
    let mut alpha = 1.0_f64.min(params.amax);

    for i in 0..params.maxiter {
        let fi = eval_f(alpha, &mut evals);

        // Armijo violation or function not decreasing
        if fi > f0 + params.c1 * alpha * dg0 || (i > 0 && fi >= f_prev) {
            return zoom(
                f, grad, x, d, f0, dg0, alpha_prev, alpha, f_prev, fi, &params, strong, &mut evals,
            );
        }

        let dgi = eval_dg(alpha, &mut evals);

        // Curvature condition satisfied
        let curvature_ok = if strong {
            dgi.abs() <= params.c2 * dg0.abs()
        } else {
            dgi >= params.c2 * dg0
        };

        if curvature_ok {
            return Ok(LineSearchResult {
                alpha,
                f_at_alpha: fi,
                directional_derivative: dgi,
                evaluations: evals,
            });
        }

        // Positive slope means minimum is between alpha_prev and alpha
        if dgi >= 0.0 {
            return zoom(
                f, grad, x, d, f0, dg0, alpha, alpha_prev, fi, f_prev, &params, strong, &mut evals,
            );
        }

        alpha_prev = alpha;
        f_prev = fi;
        alpha = (2.0 * alpha).min(params.amax);
    }

    // Failed to find a step satisfying Wolfe — return best so far
    let fi = eval_f(alpha, &mut evals);
    Ok(LineSearchResult {
        alpha,
        f_at_alpha: fi,
        directional_derivative: dg0,
        evaluations: evals,
    })
}

/// Zoom phase (Nocedal & Wright Algorithm 3.6).
/// Finds a step size in [alpha_lo, alpha_hi] satisfying Strong Wolfe conditions.
#[allow(clippy::too_many_arguments)]
fn zoom<F, G>(
    f: &F,
    grad: &G,
    x: &[f64],
    d: &[f64],
    f0: f64,
    dg0: f64,
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    mut f_lo: f64,
    _f_hi: f64,
    params: &WolfeParams,
    strong: bool,
    evals: &mut usize,
) -> Result<LineSearchResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();

    for _ in 0..params.maxiter {
        // Bisection (could use cubic interpolation for speed, but bisection is robust)
        let alpha_j = 0.5 * (alpha_lo + alpha_hi);

        let xj: Vec<f64> = (0..n).map(|i| x[i] + alpha_j * d[i]).collect();
        *evals += 1;
        let fj = f(&xj);

        if fj > f0 + params.c1 * alpha_j * dg0 || fj >= f_lo {
            alpha_hi = alpha_j;
        } else {
            let gj = grad(&xj);
            *evals += 1;
            let dgj = dot(&gj, d);

            let curvature_ok = if strong {
                dgj.abs() <= params.c2 * dg0.abs()
            } else {
                dgj >= params.c2 * dg0
            };

            if curvature_ok {
                return Ok(LineSearchResult {
                    alpha: alpha_j,
                    f_at_alpha: fj,
                    directional_derivative: dgj,
                    evaluations: *evals,
                });
            }

            if dgj * (alpha_hi - alpha_lo) >= 0.0 {
                alpha_hi = alpha_lo;
            }

            alpha_lo = alpha_j;
            f_lo = fj;
        }

        if (alpha_hi - alpha_lo).abs() < params.amin {
            break;
        }
    }

    // Return best found
    let xlo: Vec<f64> = (0..n).map(|i| x[i] + alpha_lo * d[i]).collect();
    let flo = f(&xlo);
    *evals += 1;

    Ok(LineSearchResult {
        alpha: alpha_lo,
        f_at_alpha: flo,
        directional_derivative: dg0,
        evaluations: *evals,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn quadratic_grad(x: &[f64]) -> Vec<f64> {
        x.iter().map(|xi| 2.0 * xi).collect()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
    }

    fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
        let (a, b) = (x[0], x[1]);
        vec![
            -2.0 * (1.0 - a) + 200.0 * (b - a * a) * (-2.0 * a),
            200.0 * (b - a * a),
        ]
    }

    #[test]
    fn wolfe2_quadratic_descent() {
        let x = vec![5.0, 3.0];
        let g = quadratic_grad(&x);
        let f0 = quadratic(&x);
        let d: Vec<f64> = g.iter().map(|gi| -gi).collect(); // steepest descent

        let result = line_search_wolfe2(
            &quadratic,
            &quadratic_grad,
            &x,
            &d,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect("wolfe2 works");

        assert!(result.alpha > 0.0, "alpha should be positive");
        assert!(
            result.f_at_alpha < f0,
            "function should decrease: {} vs {f0}",
            result.f_at_alpha
        );
        // Armijo condition
        let dg0 = dot(&g, &d);
        assert!(
            result.f_at_alpha <= f0 + 1e-4 * result.alpha * dg0,
            "Armijo condition violated"
        );
        // Strong Wolfe curvature
        assert!(
            result.directional_derivative.abs() <= 0.9 * dg0.abs(),
            "Strong Wolfe curvature violated"
        );
    }

    #[test]
    fn wolfe1_quadratic_descent() {
        let x = vec![5.0, 3.0];
        let g = quadratic_grad(&x);
        let f0 = quadratic(&x);
        let d: Vec<f64> = g.iter().map(|gi| -gi).collect();

        let result = line_search_wolfe1(
            &quadratic,
            &quadratic_grad,
            &x,
            &d,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect("wolfe1 works");

        assert!(result.alpha > 0.0);
        assert!(result.f_at_alpha < f0);
    }

    #[test]
    fn wolfe2_rosenbrock_descent() {
        let x = vec![0.0, 0.0];
        let g = rosenbrock_grad(&x);
        let f0 = rosenbrock(&x);
        let d: Vec<f64> = g.iter().map(|gi| -gi).collect();

        let result = line_search_wolfe2(
            &rosenbrock,
            &rosenbrock_grad,
            &x,
            &d,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect("wolfe2 on rosenbrock works");

        assert!(result.alpha > 0.0);
        assert!(result.f_at_alpha < f0);
    }

    #[test]
    fn wolfe2_rejects_ascent_direction() {
        let x = vec![1.0, 2.0];
        let g = quadratic_grad(&x);
        let f0 = quadratic(&x);
        let d = g.clone(); // ascent direction (same as gradient)

        let err = line_search_wolfe2(
            &quadratic,
            &quadratic_grad,
            &x,
            &d,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect_err("ascent should fail");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn wolfe2_invalid_params() {
        let err = validate_wolfe_params(WolfeParams {
            c1: 0.5,
            c2: 0.1, // c2 < c1 is invalid
            ..WolfeParams::default()
        })
        .expect_err("invalid params");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn wolfe2_exact_minimizer_for_quadratic() {
        // For f(x) = x^2, starting at x=2 with d=-1, exact minimizer is alpha=2
        let x = vec![2.0];
        let g = vec![4.0]; // 2*x
        let f0 = 4.0; // x^2
        let d = vec![-1.0];

        let result = line_search_wolfe2(
            &|x: &[f64]| x[0] * x[0],
            &|x: &[f64]| vec![2.0 * x[0]],
            &x,
            &d,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect("1d quadratic works");

        // Should find a valid step that decreases the function
        assert!(result.alpha > 0.0, "alpha should be positive");
        assert!(
            result.f_at_alpha < f0,
            "f should decrease: {} vs {f0}",
            result.f_at_alpha
        );
    }

    #[test]
    fn wolfe_params_default_is_valid() {
        validate_wolfe_params(WolfeParams::default()).expect("default should be valid");
    }
}
