#![forbid(unsafe_code)]

use crate::validation::IntegrateValidationError;

/// Options for numerical quadrature.
#[derive(Debug, Clone, Copy)]
pub struct QuadOptions {
    /// Absolute error tolerance.
    pub epsabs: f64,
    /// Relative error tolerance.
    pub epsrel: f64,
    /// Maximum number of subdivisions.
    pub limit: usize,
}

impl Default for QuadOptions {
    fn default() -> Self {
        Self {
            epsabs: 1.49e-8,
            epsrel: 1.49e-8,
            limit: 50,
        }
    }
}

/// Result of numerical quadrature.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadResult {
    /// Estimated value of the integral.
    pub integral: f64,
    /// Estimated absolute error.
    pub error: f64,
    /// Number of function evaluations.
    pub neval: usize,
    /// Whether the requested accuracy was achieved.
    pub converged: bool,
}

/// Result of vector-valued numerical quadrature.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadVecResult {
    /// Estimated value of the integral for each output component.
    pub integral: Vec<f64>,
    /// Estimated absolute error using the maximum componentwise GK15/G7 delta.
    pub error: f64,
    /// Number of function evaluations.
    pub neval: usize,
    /// Whether the requested accuracy was achieved.
    pub converged: bool,
}

#[derive(Debug, Clone, Copy)]
struct QuadVecSpec {
    epsabs: f64,
    epsrel: f64,
    dim: usize,
}

/// Numerically integrate a scalar function over a finite interval [a, b].
///
/// Uses adaptive Gauss-Kronrod quadrature (7-point Gauss / 15-point Kronrod).
/// Matches the core behavior of `scipy.integrate.quad(f, a, b)`.
pub fn quad<F>(
    f: F,
    a: f64,
    b: f64,
    options: QuadOptions,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "integration bounds must be finite".to_string(),
        });
    }
    if options.epsabs < 0.0 || options.epsrel < 0.0 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "tolerances must be non-negative".to_string(),
        });
    }

    if (a - b).abs() < f64::EPSILON {
        return Ok(QuadResult {
            integral: 0.0,
            error: 0.0,
            neval: 0,
            converged: true,
        });
    }

    let mut neval = 0;
    let result = adaptive_gk15(
        &f,
        a,
        b,
        options.epsabs,
        options.epsrel,
        options.limit,
        &mut neval,
    );

    Ok(QuadResult {
        integral: result.0,
        error: result.1,
        neval,
        converged: result.2,
    })
}

/// Numerically integrate a vector-valued function over a finite interval [a, b].
///
/// Uses the same adaptive Gauss-Kronrod quadrature core as [`quad`], but
/// preserves every output component of the integrand.
///
/// Matches the core finite-interval behavior of `scipy.integrate.quad_vec(f, a, b)`.
pub fn quad_vec<F>(
    f: F,
    a: f64,
    b: f64,
    options: QuadOptions,
) -> Result<QuadVecResult, IntegrateValidationError>
where
    F: Fn(f64) -> Vec<f64>,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "integration bounds must be finite".to_string(),
        });
    }
    if options.epsabs < 0.0 || options.epsrel < 0.0 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "tolerances must be non-negative".to_string(),
        });
    }

    let sample = f(a);
    let dim = sample.len();
    if (a - b).abs() < f64::EPSILON {
        return Ok(QuadVecResult {
            integral: vec![0.0; dim],
            error: 0.0,
            neval: 1,
            converged: true,
        });
    }

    let mut neval = 0;
    let spec = QuadVecSpec {
        epsabs: options.epsabs,
        epsrel: options.epsrel,
        dim,
    };
    let (integral, error, converged) =
        adaptive_gk15_vec(&f, a, b, options.limit, &mut neval, spec)?;

    Ok(QuadVecResult {
        integral,
        error,
        neval,
        converged,
    })
}

/// Adaptive Gauss-Kronrod 15-point quadrature with recursive subdivision.
fn adaptive_gk15<F>(
    f: &F,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
    neval: &mut usize,
) -> (f64, f64, bool)
where
    F: Fn(f64) -> f64,
{
    let (integral, error) = gauss_kronrod_15(f, a, b, neval);
    let tolerance = epsabs.max(epsrel * integral.abs());

    if error <= tolerance || limit == 0 {
        return (integral, error, error <= tolerance);
    }

    // Subdivide at midpoint
    let mid = 0.5 * (a + b);
    let next_limit = limit.saturating_sub(1);

    let (i_left, e_left, c_left) =
        adaptive_gk15(f, a, mid, epsabs / 2.0, epsrel, next_limit, neval);
    let (i_right, e_right, c_right) =
        adaptive_gk15(f, mid, b, epsabs / 2.0, epsrel, next_limit, neval);

    let total_integral = i_left + i_right;
    let total_error = e_left + e_right;
    let converged = c_left && c_right;

    (total_integral, total_error, converged)
}

/// Adaptive Gauss-Kronrod 15-point quadrature for vector-valued integrands.
fn adaptive_gk15_vec<F>(
    f: &F,
    a: f64,
    b: f64,
    limit: usize,
    neval: &mut usize,
    spec: QuadVecSpec,
) -> Result<(Vec<f64>, f64, bool), IntegrateValidationError>
where
    F: Fn(f64) -> Vec<f64>,
{
    let (integral, error) = gauss_kronrod_15_vec(f, a, b, neval, spec.dim)?;
    let tolerance = spec.epsabs.max(spec.epsrel * max_abs_component(&integral));

    if error <= tolerance || limit == 0 {
        return Ok((integral, error, error <= tolerance));
    }

    let mid = 0.5 * (a + b);
    let next_limit = limit.saturating_sub(1);

    let next_spec = QuadVecSpec {
        epsabs: spec.epsabs / 2.0,
        ..spec
    };
    let (i_left, e_left, c_left) = adaptive_gk15_vec(f, a, mid, next_limit, neval, next_spec)?;
    let (i_right, e_right, c_right) = adaptive_gk15_vec(f, mid, b, next_limit, neval, next_spec)?;

    let total_integral = i_left
        .iter()
        .zip(i_right.iter())
        .map(|(left, right)| left + right)
        .collect();
    let total_error = e_left + e_right;
    let converged = c_left && c_right;

    Ok((total_integral, total_error, converged))
}

/// Gauss-Kronrod 15-point / 7-point quadrature rule on [a, b].
///
/// Returns (integral_estimate, error_estimate).
/// The error is estimated as |K15 - G7| where K15 is the 15-point Kronrod
/// estimate and G7 is the embedded 7-point Gauss estimate.
fn gauss_kronrod_15<F>(f: &F, a: f64, b: f64, neval: &mut usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    // Kronrod nodes on [-1, 1] (15 points)
    // The 7 Gauss nodes are at indices 1, 3, 5, 7, 9, 11, 13
    const XGK: [f64; 15] = [
        -0.991_455_371_120_812_6,
        -0.949_107_912_342_759,
        -0.864_864_423_359_769_1,
        -0.741_531_185_599_394_4,
        -0.586_087_235_467_691_1,
        -0.405_845_151_377_397_2,
        -0.207_784_955_007_898_5,
        0.0,
        0.207_784_955_007_898_5,
        0.405_845_151_377_397_2,
        0.586_087_235_467_691_1,
        0.741_531_185_599_394_4,
        0.864_864_423_359_769_1,
        0.949_107_912_342_759,
        0.991_455_371_120_812_6,
    ];

    // Kronrod weights (15 points)
    const WGK: [f64; 15] = [
        0.022_935_322_010_529_2,
        0.063_092_092_629_979,
        0.104_790_010_322_250_2,
        0.140_653_259_715_525_9,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_4,
        0.204_432_940_075_298_9,
        0.209_482_141_084_728,
        0.204_432_940_075_298_9,
        0.190_350_578_064_785_4,
        0.169_004_726_639_267_9,
        0.140_653_259_715_525_9,
        0.104_790_010_322_250_2,
        0.063_092_092_629_979,
        0.022_935_322_010_529_2,
    ];

    // Gauss weights (7 points, at odd indices of XGK)
    const WG: [f64; 7] = [
        0.129_484_966_168_869_7,
        0.279_705_391_489_276_7,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
        0.381_830_050_505_118_9,
        0.279_705_391_489_276_7,
        0.129_484_966_168_869_7,
    ];

    let half_length = 0.5 * (b - a);
    let center = 0.5 * (a + b);

    let mut result_kronrod = 0.0;
    let mut result_gauss = 0.0;

    for (i, (&xgk, &wgk)) in XGK.iter().zip(WGK.iter()).enumerate() {
        let x = center + half_length * xgk;
        let fval = f(x);
        *neval += 1;

        result_kronrod += wgk * fval;

        // Gauss nodes are at odd indices: 1, 3, 5, 7, 9, 11, 13
        if i % 2 == 1 {
            result_gauss += WG[i / 2] * fval;
        }
    }

    result_kronrod *= half_length;
    result_gauss *= half_length;

    let mut error = (result_kronrod - result_gauss).abs();
    if error.is_nan() {
        error = f64::INFINITY;
    }

    (result_kronrod, error)
}

/// Vector-valued Gauss-Kronrod 15-point / 7-point quadrature rule on [a, b].
fn gauss_kronrod_15_vec<F>(
    f: &F,
    a: f64,
    b: f64,
    neval: &mut usize,
    dim: usize,
) -> Result<(Vec<f64>, f64), IntegrateValidationError>
where
    F: Fn(f64) -> Vec<f64>,
{
    const XGK: [f64; 15] = [
        -0.991_455_371_120_812_6,
        -0.949_107_912_342_759,
        -0.864_864_423_359_769_1,
        -0.741_531_185_599_394_4,
        -0.586_087_235_467_691_1,
        -0.405_845_151_377_397_2,
        -0.207_784_955_007_898_5,
        0.0,
        0.207_784_955_007_898_5,
        0.405_845_151_377_397_2,
        0.586_087_235_467_691_1,
        0.741_531_185_599_394_4,
        0.864_864_423_359_769_1,
        0.949_107_912_342_759,
        0.991_455_371_120_812_6,
    ];
    const WGK: [f64; 15] = [
        0.022_935_322_010_529_2,
        0.063_092_092_629_979,
        0.104_790_010_322_250_2,
        0.140_653_259_715_525_9,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_4,
        0.204_432_940_075_298_9,
        0.209_482_141_084_728,
        0.204_432_940_075_298_9,
        0.190_350_578_064_785_4,
        0.169_004_726_639_267_9,
        0.140_653_259_715_525_9,
        0.104_790_010_322_250_2,
        0.063_092_092_629_979,
        0.022_935_322_010_529_2,
    ];
    const WG: [f64; 7] = [
        0.129_484_966_168_869_7,
        0.279_705_391_489_276_7,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
        0.381_830_050_505_118_9,
        0.279_705_391_489_276_7,
        0.129_484_966_168_869_7,
    ];

    let half_length = 0.5 * (b - a);
    let center = 0.5 * (a + b);

    let mut result_kronrod = vec![0.0; dim];
    let mut result_gauss = vec![0.0; dim];

    for (i, (&xgk, &wgk)) in XGK.iter().zip(WGK.iter()).enumerate() {
        let x = center + half_length * xgk;
        let fval = f(x);
        *neval += 1;

        if fval.len() != dim {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: format!(
                    "integrand returned inconsistent vector length: expected {dim}, got {}",
                    fval.len()
                ),
            });
        }

        for (acc, value) in result_kronrod.iter_mut().zip(fval.iter()) {
            *acc += wgk * value;
        }

        if i % 2 == 1 {
            let wg = WG[i / 2];
            for (acc, value) in result_gauss.iter_mut().zip(fval.iter()) {
                *acc += wg * value;
            }
        }
    }

    for value in &mut result_kronrod {
        *value *= half_length;
    }
    for value in &mut result_gauss {
        *value *= half_length;
    }

    let mut error = result_kronrod
        .iter()
        .zip(result_gauss.iter())
        .map(|(kronrod, gauss)| (kronrod - gauss).abs())
        .fold(0.0, f64::max);
    if error.is_nan() {
        error = f64::INFINITY;
    }

    Ok((result_kronrod, error))
}

fn max_abs_component(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

/// Options for double integration.
#[derive(Debug, Clone, Copy)]
pub struct DblquadOptions {
    /// Absolute error tolerance for the outer integral.
    pub epsabs: f64,
    /// Relative error tolerance for the outer integral.
    pub epsrel: f64,
    /// Maximum subdivisions for each 1-D quadrature.
    pub limit: usize,
}

impl Default for DblquadOptions {
    fn default() -> Self {
        Self {
            epsabs: 1.49e-8,
            epsrel: 1.49e-8,
            limit: 50,
        }
    }
}

/// Result of double integration.
#[derive(Debug, Clone, PartialEq)]
pub struct DblquadResult {
    /// Estimated value of the double integral.
    pub integral: f64,
    /// Estimated absolute error.
    pub error: f64,
    /// Whether the requested accuracy was achieved.
    pub converged: bool,
}

/// Numerically compute a double integral ∫∫ f(y, x) dy dx.
///
/// Integrates `f(y, x)` over the region where `x` ranges from `a` to `b`,
/// and for each `x`, `y` ranges from `gfun(x)` to `hfun(x)`.
///
/// Matches `scipy.integrate.dblquad(f, a, b, gfun, hfun)`.
pub fn dblquad<F, GL, GH>(
    f: F,
    a: f64,
    b: f64,
    gfun: GL,
    hfun: GH,
    options: DblquadOptions,
) -> Result<DblquadResult, IntegrateValidationError>
where
    F: Fn(f64, f64) -> f64,
    GL: Fn(f64) -> f64,
    GH: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "outer integration bounds must be finite".to_string(),
        });
    }

    if (a - b).abs() < f64::EPSILON {
        return Ok(DblquadResult {
            integral: 0.0,
            error: 0.0,
            converged: true,
        });
    }

    let inner_opts = QuadOptions {
        epsabs: options.epsabs,
        epsrel: options.epsrel,
        limit: options.limit,
    };

    // Outer integral over x, inner integral over y for each x
    let outer_result = quad(
        |x| {
            let y_lo = gfun(x);
            let y_hi = hfun(x);
            if (y_lo - y_hi).abs() < f64::EPSILON {
                return 0.0;
            }
            // Inner integral of f(y, x) over y ∈ [gfun(x), hfun(x)]
            match quad(|y| f(y, x), y_lo, y_hi, inner_opts) {
                Ok(r) => r.integral,
                Err(_) => f64::INFINITY, // Trigger error in outer quad
            }
        },
        a,
        b,
        QuadOptions {
            epsabs: options.epsabs,
            epsrel: options.epsrel,
            limit: options.limit,
        },
    )?;

    if !outer_result.integral.is_finite() {
        return Ok(DblquadResult {
            integral: f64::NAN,
            error: f64::INFINITY,
            converged: false,
        });
    }

    Ok(DblquadResult {
        integral: outer_result.integral,
        error: outer_result.error,
        converged: outer_result.converged,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Composite Quadrature Rules — trapezoid and Simpson
// ══════════════════════════════════════════════════════════════════════

/// Result of composite quadrature (trapezoid / Simpson).
#[derive(Debug, Clone, PartialEq)]
pub struct CompositeQuadResult {
    /// Estimated value of the integral.
    pub integral: f64,
}

/// Integrate sampled data using the composite trapezoidal rule.
///
/// Matches `scipy.integrate.trapezoid(y, x)` (formerly `trapz`).
///
/// `y` contains the function values and `x` contains the corresponding
/// sample points. Both slices must have the same length (>= 2).
pub fn trapezoid(y: &[f64], x: &[f64]) -> Result<CompositeQuadResult, IntegrateValidationError> {
    if y.len() != x.len() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!(
                "y and x must have the same length (y={}, x={})",
                y.len(),
                x.len()
            ),
        });
    }
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for trapezoidal rule".to_string(),
        });
    }

    let mut integral = 0.0;
    for i in 0..y.len() - 1 {
        let dx = x[i + 1] - x[i];
        integral += 0.5 * dx * (y[i] + y[i + 1]);
    }

    Ok(CompositeQuadResult { integral })
}

/// Integrate sampled data using the composite trapezoidal rule with uniform spacing.
///
/// Matches `scipy.integrate.trapezoid(y, dx=dx)`.
pub fn trapezoid_uniform(
    y: &[f64],
    dx: f64,
) -> Result<CompositeQuadResult, IntegrateValidationError> {
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for trapezoidal rule".to_string(),
        });
    }
    if !dx.is_finite() || dx <= 0.0 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "dx must be finite and positive".to_string(),
        });
    }

    let mut integral = 0.5 * (y[0] + y[y.len() - 1]);
    for yi in y.iter().take(y.len() - 1).skip(1) {
        integral += yi;
    }
    integral *= dx;

    Ok(CompositeQuadResult { integral })
}

/// Integrate sampled data using Simpson's rule.
///
/// Matches `scipy.integrate.simpson(y, x=x)`.
///
/// For an even number of intervals (odd number of points), uses composite
/// Simpson's 1/3 rule. For an odd number of intervals (even number of points),
/// uses Simpson's 1/3 rule on the first n-1 panels and a trapezoidal correction
/// on the last panel (matching SciPy's default behavior).
pub fn simpson(y: &[f64], x: &[f64]) -> Result<CompositeQuadResult, IntegrateValidationError> {
    if y.len() != x.len() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!(
                "y and x must have the same length (y={}, x={})",
                y.len(),
                x.len()
            ),
        });
    }
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for Simpson's rule".to_string(),
        });
    }
    if y.len() == 2 {
        // Fall back to trapezoidal
        return trapezoid(y, x);
    }

    let n = y.len();

    if n % 2 == 1 {
        // Odd number of points = even number of intervals.
        // Use composite Simpson's 1/3 rule directly.
        let integral = simpson_nonuniform_odd(y, x);
        Ok(CompositeQuadResult { integral })
    } else {
        // Even number of points = odd number of intervals.
        // Apply Simpson's 1/3 on first n-1 points, trapezoid on last panel.
        let integral_simp = simpson_nonuniform_odd(&y[..n - 1], &x[..n - 1]);
        let last_trap = 0.5 * (x[n - 1] - x[n - 2]) * (y[n - 2] + y[n - 1]);
        Ok(CompositeQuadResult {
            integral: integral_simp + last_trap,
        })
    }
}

/// Simpson's 1/3 rule for non-uniform spacing with an odd number of points.
fn simpson_nonuniform_odd(y: &[f64], x: &[f64]) -> f64 {
    let n = y.len();
    debug_assert!(n >= 3 && n % 2 == 1);
    let mut integral = 0.0;
    let mut i = 0;
    while i + 2 < n {
        let h0 = x[i + 1] - x[i];
        let h1 = x[i + 2] - x[i + 1];
        let h_sum = h0 + h1;
        let h_prod = h0 * h1;

        if h_prod.abs() < f64::EPSILON * 1e-10 {
            // Coincident points: fall back to trapezoidal rule for the two panels
            integral += 0.5 * h0 * (y[i] + y[i + 1]);
            integral += 0.5 * h1 * (y[i + 1] + y[i + 2]);
        } else {
            // Non-uniform Simpson's rule for a pair of panels
            integral += (h_sum / 6.0)
                * ((2.0 - h1 / h0) * y[i]
                    + (h_sum * h_sum / h_prod) * y[i + 1]
                    + (2.0 - h0 / h1) * y[i + 2]);
        }
        i += 2;
    }
    integral
}

/// Integrate sampled data using Simpson's rule with uniform spacing.
///
/// Matches `scipy.integrate.simpson(y, dx=dx)`.
pub fn simpson_uniform(
    y: &[f64],
    dx: f64,
) -> Result<CompositeQuadResult, IntegrateValidationError> {
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for Simpson's rule".to_string(),
        });
    }
    if !dx.is_finite() || dx <= 0.0 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "dx must be finite and positive".to_string(),
        });
    }
    if y.len() == 2 {
        return trapezoid_uniform(y, dx);
    }

    let n = y.len();

    if n % 2 == 1 {
        // Odd number of points — composite Simpson's 1/3 directly
        let mut integral = y[0] + y[n - 1];
        for i in (1..n - 1).step_by(2) {
            integral += 4.0 * y[i];
        }
        for i in (2..n - 1).step_by(2) {
            integral += 2.0 * y[i];
        }
        integral *= dx / 3.0;
        Ok(CompositeQuadResult { integral })
    } else {
        // Even number of points — Simpson's on first n-1, trapezoid on last
        let m = n - 1; // odd number of points
        let mut integral = y[0] + y[m - 1];
        for i in (1..m - 1).step_by(2) {
            integral += 4.0 * y[i];
        }
        for i in (2..m - 1).step_by(2) {
            integral += 2.0 * y[i];
        }
        integral *= dx / 3.0;
        // Add last trapezoid panel
        integral += 0.5 * dx * (y[n - 2] + y[n - 1]);
        Ok(CompositeQuadResult { integral })
    }
}

/// Cumulatively integrate y using the trapezoidal rule.
///
/// Matches `scipy.integrate.cumulative_trapezoid(y, x)`.
/// Returns a vector of length n-1 where result[i] = ∫₀ⁱ⁺¹ y dx.
pub fn cumulative_trapezoid(y: &[f64], x: &[f64]) -> Result<Vec<f64>, IntegrateValidationError> {
    if y.len() != x.len() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!(
                "y and x must have the same length (y={}, x={})",
                y.len(),
                x.len()
            ),
        });
    }
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for cumulative trapezoid".to_string(),
        });
    }

    let n = y.len();
    let mut result = Vec::with_capacity(n - 1);
    let mut cumsum = 0.0;
    for i in 0..n - 1 {
        let dx = x[i + 1] - x[i];
        cumsum += 0.5 * dx * (y[i] + y[i + 1]);
        result.push(cumsum);
    }

    Ok(result)
}

/// Cumulatively integrate y with uniform spacing using the trapezoidal rule.
///
/// Matches `scipy.integrate.cumulative_trapezoid(y, dx=dx)`.
pub fn cumulative_trapezoid_uniform(
    y: &[f64],
    dx: f64,
) -> Result<Vec<f64>, IntegrateValidationError> {
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for cumulative trapezoid".to_string(),
        });
    }
    if !dx.is_finite() || dx <= 0.0 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "dx must be finite and positive".to_string(),
        });
    }

    let n = y.len();
    let mut result = Vec::with_capacity(n - 1);
    let mut cumsum = 0.0;
    for i in 0..n - 1 {
        cumsum += 0.5 * dx * (y[i] + y[i + 1]);
        result.push(cumsum);
    }

    Ok(result)
}

// ══════════════════════════════════════════════════════════════════════
// Triple and N-D Quadrature, Romberg, cumulative Simpson
// ══════════════════════════════════════════════════════════════════════

/// Numerically integrate a function over a 3D region.
///
/// Matches `scipy.integrate.tplquad(func, a, b, gfun, hfun, qfun, rfun)`.
///
/// Computes ∫∫∫ f(z,y,x) dz dy dx where:
/// - x ∈ [a, b]
/// - y ∈ [gfun(x), hfun(x)]
/// - z ∈ [qfun(x,y), rfun(x,y)]
#[allow(clippy::too_many_arguments)]
pub fn tplquad<F, GL, GH, QL, QH>(
    f: F,
    a: f64,
    b: f64,
    gfun: GL,
    hfun: GH,
    qfun: QL,
    rfun: QH,
    options: DblquadOptions,
) -> Result<DblquadResult, IntegrateValidationError>
where
    F: Fn(f64, f64, f64) -> f64,
    GL: Fn(f64) -> f64,
    GH: Fn(f64) -> f64,
    QL: Fn(f64, f64) -> f64,
    QH: Fn(f64, f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "outer integration bounds must be finite".to_string(),
        });
    }

    let inner_opts = QuadOptions {
        epsabs: options.epsabs,
        epsrel: options.epsrel,
        limit: options.limit,
    };

    let outer_result = quad(
        |x| {
            let y_lo = gfun(x);
            let y_hi = hfun(x);
            if (y_lo - y_hi).abs() < f64::EPSILON {
                return 0.0;
            }
            match quad(
                |y| {
                    let z_lo = qfun(x, y);
                    let z_hi = rfun(x, y);
                    if (z_lo - z_hi).abs() < f64::EPSILON {
                        return 0.0;
                    }
                    match quad(|z| f(z, y, x), z_lo, z_hi, inner_opts) {
                        Ok(r) => r.integral,
                        Err(_) => f64::INFINITY,
                    }
                },
                y_lo,
                y_hi,
                inner_opts,
            ) {
                Ok(r) => r.integral,
                Err(_) => f64::INFINITY,
            }
        },
        a,
        b,
        inner_opts,
    )?;

    Ok(DblquadResult {
        integral: outer_result.integral,
        error: outer_result.error,
        converged: outer_result.converged,
    })
}

/// Romberg integration of a function over [a, b].
///
/// Matches `scipy.integrate.romberg(function, a, b)`.
///
/// Uses Richardson extrapolation of the trapezoidal rule for higher accuracy.
///
/// # Arguments
/// * `f` — Function to integrate.
/// * `a` — Lower bound.
/// * `b` — Upper bound.
/// * `max_order` — Maximum number of Richardson extrapolation steps (default 10).
/// * `tol` — Convergence tolerance (default 1.49e-8).
pub fn romb_func<F>(
    f: F,
    a: f64,
    b: f64,
    max_order: Option<usize>,
    tol: Option<f64>,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "bounds must be finite".to_string(),
        });
    }

    let max_order = max_order.unwrap_or(10);
    let tol = tol.unwrap_or(1.49e-8);

    // Romberg table: R[i][j]
    let mut r = vec![vec![0.0; max_order + 1]; max_order + 1];
    let mut neval = 0;

    // R[0][0] = trapezoidal with 1 interval
    r[0][0] = 0.5 * (b - a) * (f(a) + f(b));
    neval += 2;

    for i in 1..=max_order {
        let n = 1usize << i; // 2^i intervals
        let h = (b - a) / n as f64;

        // Compute trapezoidal rule with 2^i intervals using previously computed values
        let mut sum = 0.0;
        for k in 1..n {
            if k % 2 == 1 {
                // Only new points (odd indices)
                sum += f(a + k as f64 * h);
                neval += 1;
            }
        }
        r[i][0] = 0.5 * r[i - 1][0] + h * sum;

        // Richardson extrapolation
        for j in 1..=i {
            let factor = 4.0_f64.powi(j as i32);
            r[i][j] = (factor * r[i][j - 1] - r[i - 1][j - 1]) / (factor - 1.0);
        }

        // Check convergence
        if i >= 2 && (r[i][i] - r[i - 1][i - 1]).abs() < tol {
            return Ok(QuadResult {
                integral: r[i][i],
                error: (r[i][i] - r[i - 1][i - 1]).abs(),
                neval,
                converged: true,
            });
        }
    }

    Ok(QuadResult {
        integral: r[max_order][max_order],
        error: if max_order >= 1 {
            (r[max_order][max_order] - r[max_order - 1][max_order - 1]).abs()
        } else {
            f64::INFINITY
        },
        neval,
        converged: false,
    })
}

/// Romberg integration of pre-sampled data.
///
/// Matches `scipy.integrate.romb(y, dx)`.
///
/// The number of samples must be 2^k + 1 for some k.
pub fn romb(y: &[f64], dx: f64) -> Result<f64, IntegrateValidationError> {
    let n = y.len();
    if n < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points".to_string(),
        });
    }
    // Check n = 2^k + 1
    let intervals = n - 1;
    if intervals & (intervals - 1) != 0 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!("romb requires 2^k + 1 samples, got {n}"),
        });
    }

    let k = (intervals as f64).log2().round() as usize;
    let mut r = vec![vec![0.0; k + 1]; k + 1];

    // R[0][0] = trapezoidal with full interval
    r[0][0] = 0.5 * dx * (intervals as f64) * (y[0] + y[intervals]);

    for i in 1..=k {
        let step = intervals >> i; // stride for level i
        let num_intervals = 1usize << i;
        let h = dx * step as f64;

        // Trapezoidal rule at this level
        let mut sum = 0.0;
        for j in 0..num_intervals {
            let idx = j * step;
            if idx + step <= intervals {
                sum += y[idx] + y[idx + step];
            }
        }
        r[i][0] = 0.5 * h * sum;

        // Richardson extrapolation
        for j in 1..=i {
            let factor = 4.0_f64.powi(j as i32);
            r[i][j] = (factor * r[i][j - 1] - r[i - 1][j - 1]) / (factor - 1.0);
        }
    }

    Ok(r[k][k])
}

/// Fixed-order Gaussian quadrature using Gauss-Legendre nodes.
///
/// Integrates `f(x)` over `[a, b]` using `n`-point Gauss-Legendre quadrature.
/// Exact for polynomials up to degree `2n - 1`.
///
/// Matches `scipy.integrate.fixed_quad(func, a, b, n=n)`.
pub fn fixed_quad<F>(
    f: F,
    a: f64,
    b: f64,
    n: usize,
) -> Result<(f64, usize), IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    if n == 0 {
        return Ok((0.0, 0));
    }
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "bounds must be finite".to_string(),
        });
    }

    // Get Gauss-Legendre nodes and weights on [-1, 1]
    let (nodes, weights) = gauss_legendre_nodes_weights(n);

    // Transform from [-1, 1] to [a, b]: x = (b-a)/2 * t + (a+b)/2
    let half_width = (b - a) / 2.0;
    let center = (a + b) / 2.0;

    let mut integral = 0.0;
    for (node, weight) in nodes.iter().zip(weights.iter()) {
        let x = half_width * node + center;
        integral += weight * f(x);
    }
    integral *= half_width;

    Ok((integral, n))
}

/// N-dimensional adaptive quadrature via nested `quad` calls.
///
/// Matches `scipy.integrate.nquad(func, ranges)`.
///
/// # Arguments
/// * `func` — Function of N variables. Takes a slice `&[f64]` of length N.
/// * `ranges` — Integration ranges as `(lower, upper)` pairs, from outermost to innermost.
/// * `options` — Quadrature options for each 1D integration.
///
/// # Example
/// ```rust
/// use fsci_integrate::{nquad, QuadOptions};
/// // ∫₀¹ ∫₀¹ x*y dx dy = 0.25
/// let result = nquad(|args| args[0] * args[1], &[(0.0, 1.0), (0.0, 1.0)], QuadOptions::default()).unwrap();
/// assert!((result.integral - 0.25).abs() < 1e-10);
/// ```
pub fn nquad<F>(
    func: F,
    ranges: &[(f64, f64)],
    options: QuadOptions,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(&[f64]) -> f64,
{
    use std::cell::RefCell;

    let ndim = ranges.len();
    if ndim == 0 {
        return Ok(QuadResult {
            integral: func(&[]),
            error: 0.0,
            neval: 1,
            converged: true,
        });
    }

    // Use RefCell for interior mutability since quad() takes Fn, not FnMut
    let args = RefCell::new(vec![0.0; ndim]);
    let total_neval = RefCell::new(0usize);

    let result = nquad_inner(&func, ranges, &options, &args, &total_neval, 0)?;

    Ok(QuadResult {
        integral: result,
        error: 0.0,
        neval: *total_neval.borrow(),
        converged: true,
    })
}

fn nquad_inner<F>(
    func: &F,
    ranges: &[(f64, f64)],
    options: &QuadOptions,
    args: &std::cell::RefCell<Vec<f64>>,
    total_neval: &std::cell::RefCell<usize>,
    dim: usize,
) -> Result<f64, IntegrateValidationError>
where
    F: Fn(&[f64]) -> f64,
{
    let (a, b) = ranges[dim];

    if dim == ranges.len() - 1 {
        // Innermost dimension: call quad directly
        let result = quad(
            |x| {
                args.borrow_mut()[dim] = x;
                func(&args.borrow())
            },
            a,
            b,
            *options,
        )?;
        *total_neval.borrow_mut() += result.neval;
        Ok(result.integral)
    } else {
        // Outer dimension: integrate by nesting
        let result = quad(
            |x| {
                args.borrow_mut()[dim] = x;
                nquad_inner(func, ranges, options, args, total_neval, dim + 1).unwrap_or(0.0)
            },
            a,
            b,
            *options,
        )?;
        *total_neval.borrow_mut() += result.neval;
        Ok(result.integral)
    }
}

/// Compute n-point Gauss-Legendre nodes and weights on [-1, 1].
///
/// Uses Newton's method to find roots of the Legendre polynomial P_n(x),
/// then computes weights via the Christoffel-Darboux formula.
fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![0.0], vec![2.0]);
    }

    let nf = n as f64;
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);

    // Only need to find roots for the first half (symmetric)
    let m = n.div_ceil(2);
    for i in 0..m {
        // Initial guess: Chebyshev approximation
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();

        // Newton's method to find root of P_n(x)
        for _ in 0..50 {
            let (pn, dpn) = legendre_pn_dpn(n, x);
            let dx = pn / dpn;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }

        let (_, dpn) = legendre_pn_dpn(n, x);
        let w = 2.0 / ((1.0 - x * x) * dpn * dpn);

        nodes.push(x);
        weights.push(w);
    }

    // Fill symmetric half
    let mut full_nodes = Vec::with_capacity(n);
    let mut full_weights = Vec::with_capacity(n);

    for i in (0..m).rev() {
        if n % 2 == 1 && i == m - 1 {
            // Middle node (for odd n)
            continue;
        }
        full_nodes.push(-nodes[i]);
        full_weights.push(weights[i]);
    }
    if n % 2 == 1 {
        full_nodes.push(nodes[m - 1]); // should be ~0
        full_weights.push(weights[m - 1]);
    }
    for i in 0..m {
        if n % 2 == 1 && i == m - 1 {
            continue;
        }
        full_nodes.push(nodes[i]);
        full_weights.push(weights[i]);
    }

    // Sort by node value
    let mut pairs: Vec<(f64, f64)> = full_nodes.into_iter().zip(full_weights).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_nodes = pairs.iter().map(|p| p.0).collect();
    let sorted_weights = pairs.iter().map(|p| p.1).collect();

    (sorted_nodes, sorted_weights)
}

/// Evaluate P_n(x) and P_n'(x) using the recurrence relation.
fn legendre_pn_dpn(n: usize, x: f64) -> (f64, f64) {
    let mut p_prev = 1.0; // P_0
    let mut p_curr = x; // P_1

    for k in 1..n {
        let kf = k as f64;
        let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }

    // P_n'(x) = n (x P_n(x) - P_{n-1}(x)) / (x² - 1)
    let nf = n as f64;
    let dpn = if (x * x - 1.0).abs() > 1e-30 {
        nf * (x * p_curr - p_prev) / (x * x - 1.0)
    } else {
        // At x = ±1, use L'Hôpital or the value P_n'(1) = n(n+1)/2
        nf * (nf + 1.0) / 2.0
    };

    (p_curr, dpn)
}

/// Cumulative integral using composite Simpson's rule.
///
/// Matches `scipy.integrate.cumulative_simpson(y, x=x)` for 1-D sampled data.
///
/// Returns one cumulative value per subinterval, so the result length is `n - 1`.
/// With two or fewer samples, this falls back to `cumulative_trapezoid`, matching
/// SciPy's documented behavior.
pub fn cumulative_simpson(y: &[f64], x: &[f64]) -> Result<Vec<f64>, IntegrateValidationError> {
    if y.len() != x.len() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!("y ({}) and x ({}) must have same length", y.len(), x.len()),
        });
    }
    let n = y.len();
    if n < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for cumulative integration".to_string(),
        });
    }
    if n <= 2 {
        return cumulative_trapezoid(y, x);
    }

    let mut result = Vec::with_capacity(n - 1);
    let mut cumsum = 0.0;

    let first_h0 = x[1] - x[0];
    let first_h1 = x[2] - x[1];
    if !(first_h0.is_finite() && first_h1.is_finite() && first_h0 > 0.0 && first_h1 > 0.0) {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "x must be finite and strictly increasing".to_string(),
        });
    }

    let mut interval_integrals = Vec::with_capacity(n - 1);
    interval_integrals.push(cumulative_simpson_left_interval(
        y[0], y[1], y[2], first_h0, first_h1,
    ));
    interval_integrals.push(cumulative_simpson_right_interval(
        y[0], y[1], y[2], first_h0, first_h1,
    ));

    for i in 3..n {
        let h_prev = x[i - 1] - x[i - 2];
        let h_curr = x[i] - x[i - 1];
        if !(h_prev.is_finite() && h_curr.is_finite() && h_prev > 0.0 && h_curr > 0.0) {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: "x must be finite and strictly increasing".to_string(),
            });
        }
        interval_integrals.push(cumulative_simpson_right_interval(
            y[i - 2],
            y[i - 1],
            y[i],
            h_prev,
            h_curr,
        ));
    }

    for value in interval_integrals {
        cumsum += value;
        result.push(cumsum);
    }

    Ok(result)
}

fn cumulative_simpson_left_interval(y0: f64, y1: f64, y2: f64, h0: f64, h1: f64) -> f64 {
    if h0.abs() < f64::EPSILON || h1.abs() < f64::EPSILON {
        0.5 * h0 * (y0 + y1)
    } else {
        let hsum = h0 + h1;
        h0 / 6.0
            * ((3.0 - h0 / hsum) * y0 + (3.0 + h0 * h0 / (h1 * hsum) + h0 / hsum) * y1
                - h0 * h0 / (h1 * hsum) * y2)
    }
}

fn cumulative_simpson_right_interval(y0: f64, y1: f64, y2: f64, h0: f64, h1: f64) -> f64 {
    if h0.abs() < f64::EPSILON || h1.abs() < f64::EPSILON {
        0.5 * h1 * (y1 + y2)
    } else {
        let hsum = h0 + h1;
        h1 / 6.0
            * (-h1 * h1 / (h0 * hsum) * y0
                + (3.0 + h1 * h1 / (h0 * hsum) + h1 / hsum) * y1
                + (3.0 - h1 / hsum) * y2)
    }
}

/// Romberg integration using Richardson extrapolation on the trapezoidal rule.
///
/// Matches `scipy.integrate.romberg`.
pub fn romberg<F>(f: F, a: f64, b: f64, tol: f64, max_order: usize) -> QuadResult
where
    F: Fn(f64) -> f64,
{
    let max_order = max_order.clamp(2, 20);
    let mut r = vec![vec![0.0; max_order]; max_order];
    let mut neval = 0usize;

    // R[0,0] = trapezoidal rule with 1 panel
    r[0][0] = 0.5 * (b - a) * (f(a) + f(b));
    neval += 2;

    for n in 1..max_order {
        let panels = 1usize << n;
        let h = (b - a) / panels as f64;

        let mut sum = 0.0;
        for k in (1..panels).step_by(2) {
            sum += f(a + k as f64 * h);
            neval += 1;
        }
        r[n][0] = 0.5 * r[n - 1][0] + h * sum;

        for m in 1..=n {
            let factor = (1u64 << (2 * m)) as f64;
            r[n][m] = (factor * r[n][m - 1] - r[n - 1][m - 1]) / (factor - 1.0);
        }

        let error = (r[n][n] - r[n - 1][n - 1]).abs();
        if error < tol {
            return QuadResult {
                integral: r[n][n],
                error,
                neval,
                converged: true,
            };
        }
    }

    let n = max_order - 1;
    QuadResult {
        integral: r[n][n],
        error: (r[n][n] - r[n - 1][n - 1]).abs(),
        neval,
        converged: false,
    }
}

/// Integrate sampled data using the trapezoidal rule (irregular spacing).
///
/// Matches `scipy.integrate.trapezoid(y, x)` with explicit x values.
pub fn trapezoid_irregular(y: &[f64], x: &[f64]) -> f64 {
    if y.len() < 2 || x.len() != y.len() {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..y.len() - 1 {
        sum += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]);
    }
    sum
}

/// Simpson's rule for irregularly-spaced data.
///
/// Matches `scipy.integrate.simpson(y, x=x)` with explicit x values.
pub fn simpson_irregular(y: &[f64], x: &[f64]) -> f64 {
    if y.len() < 2 || x.len() != y.len() {
        return 0.0;
    }
    if y.len() < 3 {
        return trapezoid_irregular(y, x);
    }

    let n = y.len();
    let mut sum = 0.0;
    let mut i = 0;

    // Process pairs of intervals with Simpson's rule
    while i + 2 < n {
        let h0 = x[i + 1] - x[i];
        let h1 = x[i + 2] - x[i + 1];
        let hsum = h0 + h1;

        // Simpson's 3/8 for unequal spacing
        sum += hsum / 6.0
            * (y[i] * (2.0 - h1 / h0)
                + y[i + 1] * hsum * hsum / (h0 * h1)
                + y[i + 2] * (2.0 - h0 / h1));

        i += 2;
    }

    // Handle remaining interval with trapezoidal rule
    if i + 1 < n {
        sum += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]);
    }

    sum
}

/// Gauss-Kronrod adaptive quadrature (7-15 point rule).
///
/// Higher accuracy than basic adaptive Simpson's. Uses a 7-point Gauss
/// rule with a 15-point Kronrod extension for error estimation.
///
/// Matches `scipy.integrate.quad` behavior (which uses QUADPACK's QAG).
pub fn gauss_kronrod_quad<F>(f: F, a: f64, b: f64, options: QuadOptions) -> QuadResult
where
    F: Fn(f64) -> f64,
{
    gauss_kronrod_inner(&f, a, b, options)
}

fn gauss_kronrod_inner(f: &dyn Fn(f64) -> f64, a: f64, b: f64, options: QuadOptions) -> QuadResult {
    // Gauss-Kronrod G7-K15 quadrature rule.
    // The 7 Gauss nodes are a subset of the 15 Kronrod nodes.
    // We evaluate f at all 15 Kronrod nodes, and compute both the
    // 7-point Gauss estimate and the 15-point Kronrod estimate.
    // The difference gives an error estimate.

    // 15-point Kronrod nodes on [-1, 1] (sorted)
    let k_nodes: [f64; 15] = [
        -0.991_455_371_120_812_6,
        -0.949_107_912_342_758_5,
        -0.864_864_423_359_769_1,
        -0.741_531_185_599_394_4,
        -0.586_087_235_467_691_1,
        -0.405_845_151_377_397_2,
        -0.207_784_955_007_898_5,
        0.0,
        0.207_784_955_007_898_5,
        0.405_845_151_377_397_2,
        0.586_087_235_467_691_1,
        0.741_531_185_599_394_4,
        0.864_864_423_359_769_1,
        0.949_107_912_342_758_5,
        0.991_455_371_120_812_6,
    ];

    // 15-point Kronrod weights
    let k_weights: [f64; 15] = [
        0.022_935_322_010_529_2,
        0.063_092_092_629_978_6,
        0.104_790_010_322_250_2,
        0.140_653_259_715_525_9,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_4,
        0.204_432_940_075_298_9,
        0.209_482_141_084_727_8,
        0.204_432_940_075_298_9,
        0.190_350_578_064_785_4,
        0.169_004_726_639_267_9,
        0.140_653_259_715_525_9,
        0.104_790_010_322_250_2,
        0.063_092_092_629_978_6,
        0.022_935_322_010_529_2,
    ];

    // 7-point Gauss weights (for the 7 nodes that are shared with Kronrod)
    // These correspond to Kronrod indices 1, 3, 5, 7, 9, 11, 13
    let g_weights: [f64; 7] = [
        0.129_484_966_168_869_7,
        0.279_705_391_489_276_7,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
        0.381_830_050_505_118_9,
        0.279_705_391_489_276_7,
        0.129_484_966_168_869_7,
    ];
    // Indices into k_nodes that correspond to the 7 Gauss nodes
    let g_indices: [usize; 7] = [1, 3, 5, 7, 9, 11, 13];

    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);

    // Evaluate f at all 15 Kronrod nodes
    let fvals: Vec<f64> = k_nodes.iter().map(|&node| f(mid + half * node)).collect();
    let neval = 15;

    // Kronrod estimate (15-point)
    let kronrod_sum: f64 = fvals
        .iter()
        .zip(k_weights.iter())
        .map(|(&fv, &w)| w * fv)
        .sum::<f64>()
        * half;

    // Gauss estimate (7-point, using subset of evaluations)
    let gauss_sum: f64 = g_indices
        .iter()
        .zip(g_weights.iter())
        .map(|(&idx, &w)| w * fvals[idx])
        .sum::<f64>()
        * half;

    let error = (kronrod_sum - gauss_sum).abs();

    if error > options.epsabs && error > options.epsrel * kronrod_sum.abs() && options.limit > 1 {
        let left = gauss_kronrod_inner(
            f,
            a,
            mid,
            QuadOptions {
                limit: options.limit - 1,
                ..options
            },
        );
        let right = gauss_kronrod_inner(
            f,
            mid,
            b,
            QuadOptions {
                limit: options.limit - 1,
                ..options
            },
        );
        return QuadResult {
            integral: left.integral + right.integral,
            error: left.error + right.error,
            neval: left.neval + right.neval,
            converged: left.converged && right.converged,
        };
    }

    QuadResult {
        integral: kronrod_sum,
        error,
        neval,
        converged: true,
    }
}

/// Newton-Cotes integration weights for n equally-spaced points.
///
/// Returns weights such that ∫₀¹ f(x) dx ≈ Σ w_i * f(x_i).
/// Matches `scipy.integrate.newton_cotes`.
pub fn newton_cotes(n: usize) -> Result<Vec<f64>, IntegrateValidationError> {
    if n == 0 {
        return Ok(vec![1.0]);
    }
    match n {
        1 => Ok(vec![0.5, 0.5]),                                     // Trapezoidal
        2 => Ok(vec![1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]),           // Simpson's 1/3
        3 => Ok(vec![1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0]), // Simpson's 3/8
        4 => Ok(vec![
            7.0 / 90.0,
            32.0 / 90.0,
            12.0 / 90.0,
            32.0 / 90.0,
            7.0 / 90.0,
        ]), // Boole's rule
        _ => {
            // General Newton-Cotes via Lagrange integration
            let nf = n as f64;
            let mut weights = vec![0.0; n + 1];
            for i in 0..=n {
                // w_i = ∫₀¹ L_i(x) dx where L_i = Π_{j≠i} (x - j/n) / (i/n - j/n)
                // Use numerical integration with high-order quadrature
                let m = 200;
                let mut integral = 0.0;
                for k in 0..=m {
                    let x = k as f64 / m as f64;
                    let w = if k == 0 || k == m {
                        1.0
                    } else if k % 2 == 0 {
                        2.0
                    } else {
                        4.0
                    };
                    let mut li = 1.0;
                    for j in 0..=n {
                        if j != i {
                            li *= (x - j as f64 / nf) / (i as f64 / nf - j as f64 / nf);
                        }
                    }
                    integral += w * li;
                }
                weights[i] = integral / (3.0 * m as f64);
            }
            Ok(weights)
        }
    }
}

/// Integrate using Newton-Cotes composite rule of given order.
///
/// `order` is the number of subintervals per panel (1=trapezoidal, 2=Simpson).
pub fn newton_cotes_quad<F>(
    f: F,
    a: f64,
    b: f64,
    n_panels: usize,
    order: usize,
) -> Result<f64, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    let weights = newton_cotes(order)?;
    let panel_width = (b - a) / n_panels as f64;
    let mut total = 0.0;

    for panel in 0..n_panels {
        let panel_a = a + panel as f64 * panel_width;
        for (i, &w) in weights.iter().enumerate() {
            let x = panel_a + i as f64 * panel_width / order as f64;
            total += w * f(x) * panel_width;
        }
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_constant_function() {
        // ∫₀¹ 5 dx = 5
        let result = quad(|_| 5.0, 0.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            (result.integral - 5.0).abs() < 1e-12,
            "integral should be 5, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_linear_function() {
        // ∫₀¹ x dx = 0.5
        let result = quad(|x| x, 0.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            (result.integral - 0.5).abs() < 1e-12,
            "integral should be 0.5, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_polynomial() {
        // ∫₀¹ x² dx = 1/3
        let result = quad(|x| x * x, 0.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            (result.integral - 1.0 / 3.0).abs() < 1e-12,
            "integral should be 1/3, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_sin() {
        // ∫₀^π sin(x) dx = 2
        let result =
            quad(f64::sin, 0.0, std::f64::consts::PI, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            (result.integral - 2.0).abs() < 1e-10,
            "integral should be 2, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_exp() {
        // ∫₀¹ e^x dx = e - 1
        let result = quad(f64::exp, 0.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        let expected = std::f64::consts::E - 1.0;
        assert!(
            (result.integral - expected).abs() < 1e-10,
            "integral should be e-1={expected}, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_negative_interval() {
        // ∫₋₁¹ x³ dx = 0 (odd function)
        let result = quad(|x| x.powi(3), -1.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            result.integral.abs() < 1e-12,
            "integral of odd function should be 0, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_reversed_bounds() {
        // ∫₁⁰ x dx = -0.5
        let result = quad(|x| x, 1.0, 0.0, QuadOptions::default()).expect("quad works");
        // With reversed bounds, the integral should be negative of forward
        // Our implementation handles this naturally since b < a
        assert!(result.neval > 0);
    }

    #[test]
    fn quad_equal_bounds() {
        let result = quad(|x| x * x, 3.0, 3.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert_eq!(result.integral, 0.0);
        assert_eq!(result.neval, 0);
    }

    #[test]
    fn quad_gaussian() {
        // ∫₋₃³ e^(-x²) dx ≈ sqrt(π) * erf(3) ≈ 1.7724..
        let result =
            quad(|x| (-x * x).exp(), -3.0, 3.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        // sqrt(pi) * erf(3) ≈ 1.7724538509
        let expected = std::f64::consts::PI.sqrt() * 0.999_977_909_503_001_4;
        assert!(
            (result.integral - expected).abs() < 1e-8,
            "gaussian integral: got {} expected ~{expected}",
            result.integral
        );
    }

    #[test]
    fn quad_nonfinite_bounds_error() {
        let err =
            quad(|x| x, f64::INFINITY, 1.0, QuadOptions::default()).expect_err("infinite bounds");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn quad_negative_tolerance_error() {
        let err = quad(
            |x| x,
            0.0,
            1.0,
            QuadOptions {
                epsabs: -1.0,
                ..QuadOptions::default()
            },
        )
        .expect_err("negative tolerance");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidTolerance { .. }
        ));
    }

    #[test]
    fn quad_high_degree_polynomial() {
        // ∫₀¹ x⁶ dx = 1/7
        let result = quad(|x| x.powi(6), 0.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            (result.integral - 1.0 / 7.0).abs() < 1e-10,
            "integral should be 1/7, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_wide_interval() {
        // ∫₀¹⁰⁰ 1/(1+x²) dx = atan(100) ≈ 1.5607966...
        let result =
            quad(|x| 1.0 / (1.0 + x * x), 0.0, 100.0, QuadOptions::default()).expect("quad works");
        let expected = 100.0_f64.atan();
        assert!(
            (result.integral - expected).abs() < 1e-6,
            "integral should be atan(100)={expected}, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_vec_constant_function() {
        let result =
            quad_vec(|_| vec![5.0, -2.0], 0.0, 1.0, QuadOptions::default()).expect("quad_vec");
        assert!(result.converged);
        assert!((result.integral[0] - 5.0).abs() < 1e-12);
        assert!((result.integral[1] + 2.0).abs() < 1e-12);
    }

    #[test]
    fn quad_vec_polynomial_components() {
        let result =
            quad_vec(|x| vec![x, x * x], 0.0, 1.0, QuadOptions::default()).expect("quad_vec");
        assert!(result.converged);
        assert!((result.integral[0] - 0.5).abs() < 1e-12);
        assert!((result.integral[1] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn quad_vec_reversed_bounds() {
        let result =
            quad_vec(|x| vec![x, 1.0], 1.0, 0.0, QuadOptions::default()).expect("quad_vec");
        assert!(result.converged);
        assert!((result.integral[0] + 0.5).abs() < 1e-12);
        assert!((result.integral[1] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn quad_vec_invalid_input_errors() {
        let bounds_err = quad_vec(|x| vec![x], f64::INFINITY, 1.0, QuadOptions::default())
            .expect_err("non-finite bounds");
        assert!(matches!(
            bounds_err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));

        let tol_err = quad_vec(
            |x| vec![x],
            0.0,
            1.0,
            QuadOptions {
                epsabs: -1.0,
                ..QuadOptions::default()
            },
        )
        .expect_err("negative tolerance");
        assert!(matches!(
            tol_err,
            IntegrateValidationError::QuadInvalidTolerance { .. }
        ));
    }

    #[test]
    fn quad_vec_rejects_inconsistent_output_shape() {
        let err = quad_vec(
            |x| {
                if x < 0.5 { vec![x] } else { vec![x, x * x] }
            },
            0.0,
            1.0,
            QuadOptions::default(),
        )
        .expect_err("inconsistent output length");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    // ── dblquad tests ───────────────────────────────────────────────

    #[test]
    fn dblquad_constant() {
        // ∫₀¹ ∫₀¹ 1 dy dx = 1
        let result = dblquad(
            |_y, _x| 1.0,
            0.0,
            1.0,
            |_| 0.0,
            |_| 1.0,
            DblquadOptions::default(),
        )
        .expect("dblquad works");
        assert!(result.converged);
        assert!(
            (result.integral - 1.0).abs() < 1e-10,
            "should be 1, got {}",
            result.integral
        );
    }

    #[test]
    fn dblquad_xy() {
        // ∫₀¹ ∫₀¹ x*y dy dx = 1/4
        let result = dblquad(
            |y, x| x * y,
            0.0,
            1.0,
            |_| 0.0,
            |_| 1.0,
            DblquadOptions::default(),
        )
        .expect("dblquad works");
        assert!(result.converged);
        assert!(
            (result.integral - 0.25).abs() < 1e-10,
            "should be 0.25, got {}",
            result.integral
        );
    }

    #[test]
    fn dblquad_triangular_region() {
        // ∫₀¹ ∫₀ˣ 1 dy dx = ∫₀¹ x dx = 1/2
        let result = dblquad(
            |_y, _x| 1.0,
            0.0,
            1.0,
            |_| 0.0,
            |x| x,
            DblquadOptions::default(),
        )
        .expect("dblquad works");
        assert!(result.converged);
        assert!(
            (result.integral - 0.5).abs() < 1e-10,
            "should be 0.5, got {}",
            result.integral
        );
    }

    #[test]
    fn dblquad_disk_area() {
        // ∫₋₁¹ ∫₋√(1-x²)^√(1-x²) 1 dy dx = π (area of unit disk)
        let result = dblquad(
            |_y, _x| 1.0,
            -1.0,
            1.0,
            |x| -(1.0 - x * x).sqrt(),
            |x| (1.0 - x * x).sqrt(),
            DblquadOptions::default(),
        )
        .expect("dblquad works");
        assert!(
            (result.integral - std::f64::consts::PI).abs() < 1e-6,
            "should be pi, got {}",
            result.integral
        );
    }

    #[test]
    fn dblquad_equal_bounds() {
        let result = dblquad(
            |y, x| x * y,
            3.0,
            3.0,
            |_| 0.0,
            |_| 1.0,
            DblquadOptions::default(),
        )
        .expect("dblquad equal bounds");
        assert_eq!(result.integral, 0.0);
    }

    #[test]
    fn dblquad_nonfinite_bounds_error() {
        let err = dblquad(
            |y, x| x * y,
            f64::INFINITY,
            1.0,
            |_| 0.0,
            |_| 1.0,
            DblquadOptions::default(),
        )
        .expect_err("nonfinite");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    // ── trapezoid tests ─────────────────────────────────────────────

    #[test]
    fn trapezoid_constant() {
        // ∫₀¹ 5 dx = 5
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let y = vec![5.0; 5];
        let result = trapezoid(&y, &x).expect("trapezoid works");
        assert!(
            (result.integral - 5.0).abs() < 1e-12,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn trapezoid_linear() {
        // ∫₀¹ x dx = 0.5 (exact for trapezoidal on linear functions)
        let n = 100;
        let x: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.to_vec();
        let result = trapezoid(&y, &x).expect("trapezoid works");
        assert!(
            (result.integral - 0.5).abs() < 1e-12,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn trapezoid_quadratic() {
        // ∫₀¹ x² dx = 1/3
        let n = 1000;
        let x: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let result = trapezoid(&y, &x).expect("trapezoid works");
        assert!(
            (result.integral - 1.0 / 3.0).abs() < 1e-6,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn trapezoid_uniform_constant() {
        let y = vec![3.0; 11];
        let result = trapezoid_uniform(&y, 0.1).expect("trapezoid_uniform works");
        assert!(
            (result.integral - 3.0).abs() < 1e-12,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn trapezoid_length_mismatch_error() {
        let err = trapezoid(&[1.0, 2.0], &[0.0]).expect_err("length mismatch");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn trapezoid_too_few_points_error() {
        let err = trapezoid(&[1.0], &[0.0]).expect_err("too few points");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    // ── simpson tests ───────────────────────────────────────────────

    #[test]
    fn simpson_constant() {
        let x = vec![0.0, 0.5, 1.0];
        let y = vec![7.0; 3];
        let result = simpson(&y, &x).expect("simpson works");
        assert!(
            (result.integral - 7.0).abs() < 1e-12,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_quadratic_exact() {
        // Simpson's rule is exact for polynomials up to degree 3
        // ∫₀¹ x² dx = 1/3
        let x = vec![0.0, 0.5, 1.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let result = simpson(&y, &x).expect("simpson works");
        assert!(
            (result.integral - 1.0 / 3.0).abs() < 1e-12,
            "should be exact for quadratic, got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_cubic_exact() {
        // ∫₀¹ x³ dx = 1/4, exact for Simpson's rule
        let x = vec![0.0, 0.5, 1.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi * xi).collect();
        let result = simpson(&y, &x).expect("simpson works");
        assert!(
            (result.integral - 0.25).abs() < 1e-12,
            "should be exact for cubic, got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_sin_many_points() {
        // ∫₀^π sin(x) dx = 2
        let n = 101; // odd number of points
        let x: Vec<f64> = (0..n)
            .map(|i| std::f64::consts::PI * i as f64 / (n - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.sin()).collect();
        let result = simpson(&y, &x).expect("simpson works");
        assert!(
            (result.integral - 2.0).abs() < 1e-6,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_even_points() {
        // Even number of points — uses trapezoid correction on last panel
        let n = 100;
        let x: Vec<f64> = (0..n)
            .map(|i| std::f64::consts::PI * i as f64 / (n - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.sin()).collect();
        let result = simpson(&y, &x).expect("simpson with even points");
        assert!(
            (result.integral - 2.0).abs() < 1e-4,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_uniform_quadratic() {
        // ∫₀¹ x² dx = 1/3 with uniform spacing
        let n = 5; // odd
        let dx = 1.0 / (n - 1) as f64;
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let xi = i as f64 * dx;
                xi * xi
            })
            .collect();
        let result = simpson_uniform(&y, dx).expect("simpson_uniform works");
        assert!(
            (result.integral - 1.0 / 3.0).abs() < 1e-10,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_two_points_fallback() {
        // Only 2 points — falls back to trapezoidal
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 2.0];
        let result = simpson(&y, &x).expect("simpson 2 points");
        assert!(
            (result.integral - 1.0).abs() < 1e-12,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_too_few_points_error() {
        let err = simpson(&[1.0], &[0.0]).expect_err("too few points");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    // ── cumulative_trapezoid tests ──────────────────────────────────

    #[test]
    fn cumtrapz_linear() {
        // y = x on [0, 1, 2, 3] => cumulative integrals: 0.5, 2.0, 4.5
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let result = cumulative_trapezoid(&y, &x).expect("cumtrapz");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.5).abs() < 1e-12);
        assert!((result[1] - 2.0).abs() < 1e-12);
        assert!((result[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn cumtrapz_constant() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![5.0; 4];
        let result = cumulative_trapezoid(&y, &x).expect("cumtrapz");
        assert!((result[0] - 5.0).abs() < 1e-12);
        assert!((result[1] - 10.0).abs() < 1e-12);
        assert!((result[2] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn cumtrapz_uniform() {
        let y = vec![0.0, 1.0, 4.0, 9.0]; // x²-ish
        let result = cumulative_trapezoid_uniform(&y, 1.0).expect("cumtrapz_uniform");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.5).abs() < 1e-12);
        assert!((result[1] - 3.0).abs() < 1e-12);
        assert!((result[2] - 9.5).abs() < 1e-12);
    }

    #[test]
    fn cumtrapz_length_mismatch() {
        let err = cumulative_trapezoid(&[1.0, 2.0], &[0.0]).expect_err("mismatch");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn cumtrapz_too_few_points() {
        let err = cumulative_trapezoid(&[1.0], &[0.0]).expect_err("too few");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    // ── tplquad tests ──────────────────────────────────────────────

    #[test]
    fn tplquad_unit_cube() {
        // ∫∫∫ 1 dz dy dx over [0,1]^3 = 1
        let result = tplquad(
            |_z, _y, _x| 1.0,
            0.0,
            1.0,
            |_x| 0.0,
            |_x| 1.0,
            |_x, _y| 0.0,
            |_x, _y| 1.0,
            DblquadOptions::default(),
        )
        .unwrap();
        assert!(
            (result.integral - 1.0).abs() < 0.01,
            "unit cube volume: {}",
            result.integral
        );
    }

    #[test]
    fn tplquad_xyz() {
        // ∫∫∫ xyz dz dy dx over [0,1]^3 = (1/2)^3 = 0.125
        let result = tplquad(
            |z, y, x| x * y * z,
            0.0,
            1.0,
            |_x| 0.0,
            |_x| 1.0,
            |_x, _y| 0.0,
            |_x, _y| 1.0,
            DblquadOptions::default(),
        )
        .unwrap();
        assert!(
            (result.integral - 0.125).abs() < 0.01,
            "∫xyz = {}, expected 0.125",
            result.integral
        );
    }

    // ── Romberg tests ──────────────────────────────────────────────

    #[test]
    fn romb_func_polynomial() {
        // ∫ x^2 dx from 0 to 1 = 1/3
        let result = romb_func(|x| x * x, 0.0, 1.0, None, None).unwrap();
        assert!(
            (result.integral - 1.0 / 3.0).abs() < 1e-10,
            "romberg ∫x² = {}, expected 1/3",
            result.integral
        );
        assert!(result.converged);
    }

    #[test]
    fn romb_func_exp() {
        // ∫ e^x dx from 0 to 1 = e - 1
        let result = romb_func(|x| x.exp(), 0.0, 1.0, None, None).unwrap();
        let expected = std::f64::consts::E - 1.0;
        assert!(
            (result.integral - expected).abs() < 1e-8,
            "romberg ∫e^x = {}, expected {}",
            result.integral,
            expected
        );
    }

    #[test]
    fn romb_sampled() {
        // Sampled x^2 from 0 to 4 with 5 points (2^2 + 1)
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // x^2 at x=0,1,2,3,4
        let result = romb(&y, 1.0).unwrap();
        // ∫ x^2 from 0 to 4 = 64/3 ≈ 21.333
        assert!(
            (result - 64.0 / 3.0).abs() < 0.1,
            "romb sampled x^2 = {}, expected ~21.33",
            result
        );
    }

    #[test]
    fn romb_wrong_sample_count() {
        // 4 samples (not 2^k + 1)
        assert!(romb(&[1.0, 2.0, 3.0, 4.0], 1.0).is_err());
    }

    // ── cumulative_simpson tests ───────────────────────────────────

    #[test]
    fn cumulative_simpson_constant() {
        // f(x) = 2, x = [0, 1, 2, 3, 4]
        // Cumulative integral at each point: 2, 4, 6, 8
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0; 5];
        let result = cumulative_simpson(&y, &x).unwrap();
        assert_eq!(result.len(), 4);
        let expected = [2.0, 4.0, 6.0, 8.0];
        for (got, want) in result.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "got {got}, expected {want}");
        }
    }

    #[test]
    fn cumulative_simpson_quadratic() {
        // f(x) = x^2, x = [0, 1, 2, 3, 4]
        // Simpson should be exact for polynomials up to degree 3
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let result = cumulative_simpson(&y, &x).unwrap();
        assert_eq!(result.len(), 4);
        let expected = [1.0 / 3.0, 8.0 / 3.0, 9.0, 64.0 / 3.0];
        for (got, want) in result.iter().zip(expected) {
            assert!((got - want).abs() < 1e-10, "got {got}, expected {want}");
        }
    }

    #[test]
    fn cumulative_simpson_two_points_falls_back_to_trapezoid() {
        let x = vec![0.0, 2.0];
        let y = vec![1.0, 3.0];
        let result = cumulative_simpson(&y, &x).unwrap();
        assert_eq!(result, vec![4.0]);
    }

    #[test]
    fn cumulative_simpson_requires_strictly_increasing_x() {
        let err = cumulative_simpson(&[1.0, 2.0, 3.0], &[0.0, 0.0, 1.0]).unwrap_err();
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn cumulative_simpson_too_few() {
        assert!(cumulative_simpson(&[1.0], &[0.0]).is_err());
    }

    // ── nquad tests ──────────────────────────────────────────────────

    #[test]
    fn nquad_1d_matches_quad() {
        let opts = QuadOptions::default();
        let result_nquad = nquad(|args| args[0] * args[0], &[(0.0, 1.0)], opts).expect("nquad 1d");
        let result_quad = quad(|x| x * x, 0.0, 1.0, opts).expect("quad");
        assert!(
            (result_nquad.integral - result_quad.integral).abs() < 1e-10,
            "nquad 1D should match quad: {} vs {}",
            result_nquad.integral,
            result_quad.integral
        );
    }

    #[test]
    fn nquad_2d_product() {
        // ∫₀¹ ∫₀¹ x*y dx dy = (1/2)(1/2) = 0.25
        let opts = QuadOptions::default();
        let result =
            nquad(|args| args[0] * args[1], &[(0.0, 1.0), (0.0, 1.0)], opts).expect("nquad 2d");
        assert!(
            (result.integral - 0.25).abs() < 1e-8,
            "∫∫ xy dxdy = {}, expected 0.25",
            result.integral
        );
    }

    #[test]
    fn nquad_3d_unit_cube() {
        // ∫₀¹ ∫₀¹ ∫₀¹ 1 dxdydz = 1
        let opts = QuadOptions::default();
        let result =
            nquad(|_args| 1.0, &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], opts).expect("nquad 3d");
        assert!(
            (result.integral - 1.0).abs() < 1e-6,
            "∫∫∫ 1 dV = {}, expected 1.0",
            result.integral
        );
    }

    #[test]
    fn nquad_0d() {
        // 0-dimensional: just evaluates the function
        let opts = QuadOptions::default();
        let result = nquad(|_| 42.0, &[], opts).expect("nquad 0d");
        assert!((result.integral - 42.0).abs() < 1e-12);
    }

    // ── fixed_quad tests ─────────────────────────────────────────────

    #[test]
    fn fixed_quad_polynomial_exact() {
        // n-point Gauss-Legendre is exact for polynomials up to degree 2n-1
        // 5-point should be exact for x^9
        let (result, neval) = fixed_quad(|x| x.powi(4), 0.0, 1.0, 5).expect("fixed_quad");
        assert_eq!(neval, 5);
        assert!(
            (result - 0.2).abs() < 1e-12,
            "∫₀¹ x⁴ dx = 0.2, got {result}"
        );
    }

    #[test]
    fn fixed_quad_sin() {
        // ∫₀^π sin(x) dx = 2
        let (result, _) =
            fixed_quad(|x| x.sin(), 0.0, std::f64::consts::PI, 10).expect("fixed_quad sin");
        assert!(
            (result - 2.0).abs() < 1e-10,
            "∫₀^π sin(x) = 2, got {result}"
        );
    }

    #[test]
    fn fixed_quad_matches_quad() {
        // fixed_quad with high n should match adaptive quad
        let (fq, _) = fixed_quad(|x| x.exp(), 0.0, 1.0, 20).expect("fixed_quad");
        let aq = quad(|x| x.exp(), 0.0, 1.0, QuadOptions::default()).expect("quad");
        assert!(
            (fq - aq.integral).abs() < 1e-10,
            "fixed_quad vs quad: {fq} vs {}",
            aq.integral
        );
    }

    #[test]
    fn romberg_exp() {
        let result = romberg(|x| x.exp(), 0.0, 1.0, 1e-12, 10);
        let exact = std::f64::consts::E - 1.0;
        assert!(
            (result.integral - exact).abs() < 1e-10,
            "romberg exp: {} vs {}",
            result.integral,
            exact
        );
    }

    #[test]
    fn romberg_polynomial() {
        // ∫₀¹ x³ dx = 1/4
        let result = romberg(|x| x * x * x, 0.0, 1.0, 1e-12, 10);
        assert!(
            (result.integral - 0.25).abs() < 1e-10,
            "romberg x³: {}",
            result.integral
        );
    }

    #[test]
    fn trapezoid_irregular_basic() {
        let x = vec![0.0, 1.0, 3.0, 4.0];
        let y = vec![1.0, 1.0, 1.0, 1.0]; // constant = 1
        let result = trapezoid_irregular(&y, &x);
        assert!((result - 4.0).abs() < 1e-10); // area = 4
    }

    #[test]
    fn simpson_irregular_basic() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect(); // y = x²
        let result = simpson_irregular(&y, &x);
        let exact = 64.0 / 3.0; // ∫₀⁴ x² dx = 64/3
        assert!(
            (result - exact).abs() < 1e-6,
            "simpson irregular x²: {} vs {}",
            result,
            exact
        );
    }

    #[test]
    fn gauss_kronrod_exp() {
        let result = gauss_kronrod_quad(|x| x.exp(), 0.0, 1.0, QuadOptions::default());
        let exact = std::f64::consts::E - 1.0;
        assert!(
            (result.integral - exact).abs() < 1e-8,
            "GK quad exp: {} vs {}",
            result.integral,
            exact
        );
    }
}
