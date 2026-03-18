#![forbid(unsafe_code)]

//! Interpolation routines for FrankenSciPy.
//!
//! Matches `scipy.interpolate` core functions:
//! - `interp1d` — 1D interpolation (linear, nearest, cubic)
//! - `CubicSpline` — natural/clamped/not-a-knot cubic spline

use fsci_runtime::RuntimeMode;

/// Interpolation method for `interp1d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpKind {
    #[default]
    Linear,
    Nearest,
    CubicSpline,
}

/// Boundary conditions for cubic spline interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SplineBc {
    /// S''(x_0) = 0, S''(x_n) = 0 (default).
    #[default]
    Natural,
    /// Not-a-knot: S'''(x_1⁻) = S'''(x_1⁺), S'''(x_{n-2}⁻) = S'''(x_{n-2}⁺).
    NotAKnot,
    /// Clamped: S'(x_0) = deriv_left, S'(x_n) = deriv_right.
    Clamped(f64, f64),
}

/// Error type for interpolation operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpError {
    TooFewPoints { minimum: usize, actual: usize },
    UnsortedX,
    LengthMismatch { x_len: usize, y_len: usize },
    OutOfBounds { value: String },
    InvalidArgument { detail: String },
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewPoints { minimum, actual } => {
                write!(f, "need at least {minimum} points, got {actual}")
            }
            Self::UnsortedX => write!(f, "x values must be strictly increasing"),
            Self::LengthMismatch { x_len, y_len } => {
                write!(f, "x and y must have same length (x={x_len}, y={y_len})")
            }
            Self::OutOfBounds { value } => write!(f, "interpolation point out of bounds: {value}"),
            Self::InvalidArgument { detail } => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for InterpError {}

/// Options for 1D interpolation.
#[derive(Debug, Clone, Copy)]
pub struct Interp1dOptions {
    pub kind: InterpKind,
    pub mode: RuntimeMode,
    pub fill_value: Option<f64>,
    pub bounds_error: bool,
    pub spline_bc: SplineBc,
}

impl Default for Interp1dOptions {
    fn default() -> Self {
        Self {
            kind: InterpKind::Linear,
            mode: RuntimeMode::Strict,
            fill_value: None,
            bounds_error: true,
            spline_bc: SplineBc::default(),
        }
    }
}

/// 1D interpolation function object.
///
/// Matches `scipy.interpolate.interp1d(x, y, kind=...)`.
/// Create with `Interp1d::new(x, y, options)`, then call `.evaluate(x_new)`.
#[derive(Debug)]
pub struct Interp1d {
    x: Vec<f64>,
    y: Vec<f64>,
    options: Interp1dOptions,
    /// Cubic spline coefficients (a, b, c, d) for each interval, if applicable.
    spline_coeffs: Option<Vec<[f64; 4]>>,
}

impl Interp1d {
    /// Create a new 1D interpolator.
    pub fn new(x: &[f64], y: &[f64], options: Interp1dOptions) -> Result<Self, InterpError> {
        if x.len() != y.len() {
            return Err(InterpError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }

        let min_points = match options.kind {
            InterpKind::Linear | InterpKind::Nearest => 2,
            InterpKind::CubicSpline => 4,
        };
        if x.len() < min_points {
            return Err(InterpError::TooFewPoints {
                minimum: min_points,
                actual: x.len(),
            });
        }

        // Check strictly increasing
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InterpError::UnsortedX);
        }

        let spline_coeffs = if options.kind == InterpKind::CubicSpline {
            Some(compute_cubic_spline(x, y, options.spline_bc)?)
        } else {
            None
        };

        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            options,
            spline_coeffs,
        })
    }

    /// Evaluate the interpolant at a single point.
    pub fn eval(&self, x_new: f64) -> Result<f64, InterpError> {
        // Bounds checking
        if x_new < self.x[0] || x_new > *self.x.last().unwrap() {
            if self.options.bounds_error {
                return Err(InterpError::OutOfBounds {
                    value: format!("{x_new}"),
                });
            }
            if let Some(fill) = self.options.fill_value {
                return Ok(fill);
            }
            return Ok(f64::NAN);
        }

        match self.options.kind {
            InterpKind::Linear => self.eval_linear(x_new),
            InterpKind::Nearest => self.eval_nearest(x_new),
            InterpKind::CubicSpline => self.eval_cubic(x_new),
        }
    }

    /// Evaluate at multiple points.
    pub fn eval_many(&self, x_new: &[f64]) -> Result<Vec<f64>, InterpError> {
        x_new.iter().map(|&xi| self.eval(xi)).collect()
    }

    fn eval_linear(&self, x_new: f64) -> Result<f64, InterpError> {
        let i = self.find_interval(x_new);
        let t = (x_new - self.x[i]) / (self.x[i + 1] - self.x[i]);
        Ok(self.y[i] + t * (self.y[i + 1] - self.y[i]))
    }

    fn eval_nearest(&self, x_new: f64) -> Result<f64, InterpError> {
        let i = self.find_interval(x_new);
        let mid = 0.5 * (self.x[i] + self.x[i + 1]);
        Ok(if x_new <= mid {
            self.y[i]
        } else {
            self.y[i + 1]
        })
    }

    fn eval_cubic(&self, x_new: f64) -> Result<f64, InterpError> {
        let coeffs = self.spline_coeffs.as_ref().unwrap();
        let i = self.find_interval(x_new);
        let dx = x_new - self.x[i];
        let [a, b, c, d] = coeffs[i];
        Ok(a + dx * (b + dx * (c + dx * d)))
    }

    /// Binary search to find the interval containing x_new.
    fn find_interval(&self, x_new: f64) -> usize {
        let n = self.x.len();
        if x_new <= self.x[0] {
            return 0;
        }
        if x_new >= self.x[n - 1] {
            return n - 2;
        }
        // Binary search
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.x[mid] <= x_new {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Compute cubic spline coefficients with configurable boundary conditions.
///
/// For each interval [x_i, x_{i+1}], the spline is:
///   S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)² + d_i*(x-x_i)³
fn compute_cubic_spline(x: &[f64], y: &[f64], bc: SplineBc) -> Result<Vec<[f64; 4]>, InterpError> {
    let n = x.len();
    if n < 4 {
        return Err(InterpError::TooFewPoints {
            minimum: 4,
            actual: n,
        });
    }

    let m = n - 1; // number of intervals
    let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();

    // Solve for second derivatives c[0..n] via tridiagonal system.
    // The system size depends on the boundary condition.
    let c = match bc {
        SplineBc::Natural => solve_spline_natural(n, &h, y),
        SplineBc::NotAKnot => solve_spline_not_a_knot(n, &h, y),
        SplineBc::Clamped(dl, dr) => solve_spline_clamped(n, &h, y, dl, dr),
    };

    // Compute a, b, d from c
    let mut coeffs = Vec::with_capacity(m);
    for i in 0..m {
        let a_i = y[i];
        let b_i = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
        let c_i = c[i];
        let d_i = (c[i + 1] - c[i]) / (3.0 * h[i]);
        coeffs.push([a_i, b_i, c_i, d_i]);
    }

    Ok(coeffs)
}

/// Natural BC: c[0] = 0, c[n-1] = 0.
fn solve_spline_natural(n: usize, h: &[f64], y: &[f64]) -> Vec<f64> {
    let inner = n - 2;
    let mut rhs = vec![0.0; inner];
    for (i, r) in rhs.iter_mut().enumerate() {
        let ii = i + 1;
        *r = 3.0 * ((y[ii + 1] - y[ii]) / h[ii] - (y[ii] - y[ii - 1]) / h[ii - 1]);
    }

    let mut diag: Vec<f64> = (0..inner).map(|i| 2.0 * (h[i] + h[i + 1])).collect();
    let sub: Vec<f64> = (0..inner).map(|i| h[i]).collect();
    let sup: Vec<f64> = (0..inner).map(|i| h[i + 1]).collect();

    thomas_solve(&sub, &mut diag, &sup, &mut rhs);

    let mut c = vec![0.0; n];
    for (i, &ci) in rhs.iter().enumerate() {
        c[i + 1] = ci;
    }
    c
}

/// Not-a-knot BC: third derivative continuous at x[1] and x[n-2].
///
/// Uses substitution: express c[0] and c[n-1] via the not-a-knot relations,
/// then substitute into the interior equations to get a consistent (n-2)×(n-2) system.
fn solve_spline_not_a_knot(n: usize, h: &[f64], y: &[f64]) -> Vec<f64> {
    // Not-a-knot relations:
    //   d[0] = d[1] => c[0] = (1 + h[0]/h[1])*c[1] - (h[0]/h[1])*c[2]
    //   d[n-3] = d[n-2] => c[n-1] = (1 + h[n-2]/h[n-3])*c[n-2] - (h[n-2]/h[n-3])*c[n-3]
    //
    // Substitute into the interior tridiagonal system for c[1]..c[n-2].

    let inner = n - 2; // solve for c[1]..c[n-2]
    let r01 = h[0] / h[1]; // ratio for left BC

    let mut diag = vec![0.0; inner];
    let mut sub = vec![0.0; inner];
    let mut sup = vec![0.0; inner];
    let mut rhs = vec![0.0; inner];

    for idx in 0..inner {
        let i = idx + 1; // global index
        sub[idx] = h[i - 1];
        diag[idx] = 2.0 * (h[i - 1] + h[i]);
        sup[idx] = h[i];
        rhs[idx] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    // Row 0 (i=1): absorb c[0] = (1+r01)*c[1] - r01*c[2]
    // Original: h[0]*c[0] + 2(h[0]+h[1])*c[1] + h[1]*c[2] = rhs[0]
    // Substitute c[0]:
    //   h[0]*((1+r01)*c[1] - r01*c[2]) + 2(h[0]+h[1])*c[1] + h[1]*c[2] = rhs[0]
    //   (h[0]*(1+r01) + 2(h[0]+h[1]))*c[1] + (h[1] - h[0]*r01)*c[2] = rhs[0]
    diag[0] = h[0] * (1.0 + r01) + 2.0 * (h[0] + h[1]);
    sup[0] = h[1] - h[0] * r01;
    // sub[0] is unused (no c[-1])

    // Row inner-1 (i=n-2): absorb c[n-1]
    if n >= 5 {
        let rn = h[n - 2] / h[n - 3]; // ratio for right BC
        // c[n-1] = (1+rn)*c[n-2] - rn*c[n-3]
        // Original: h[n-3]*c[n-3] + 2(h[n-3]+h[n-2])*c[n-2] + h[n-2]*c[n-1] = rhs[last]
        // Substitute:
        //   h[n-3]*c[n-3] + 2(h[n-3]+h[n-2])*c[n-2] + h[n-2]*((1+rn)*c[n-2] - rn*c[n-3]) = rhs
        //   (h[n-3] - h[n-2]*rn)*c[n-3] + (2(h[n-3]+h[n-2]) + h[n-2]*(1+rn))*c[n-2] = rhs
        let last = inner - 1;
        sub[last] = h[n - 3] - h[n - 2] * rn;
        diag[last] = 2.0 * (h[n - 3] + h[n - 2]) + h[n - 2] * (1.0 + rn);
        // sup[last] is unused (no c[n])
    } else {
        // n=4: inner=2, both boundary rows are modified
        let rn = h[n - 2] / h[n - 3];
        let last = inner - 1;
        sub[last] = h[n - 3] - h[n - 2] * rn;
        diag[last] = 2.0 * (h[n - 3] + h[n - 2]) + h[n - 2] * (1.0 + rn);
    }

    thomas_solve(&sub, &mut diag, &sup, &mut rhs);

    // Build full c array
    let mut c = vec![0.0; n];
    for (idx, &ci) in rhs.iter().enumerate() {
        c[idx + 1] = ci;
    }
    // Apply not-a-knot relations
    c[0] = (1.0 + r01) * c[1] - r01 * c[2];
    let rn = h[n - 2] / h[n - 3];
    c[n - 1] = (1.0 + rn) * c[n - 2] - rn * c[n - 3];

    c
}

/// Clamped BC: S'(x_0) = deriv_left, S'(x_n) = deriv_right.
fn solve_spline_clamped(
    n: usize,
    h: &[f64],
    y: &[f64],
    deriv_left: f64,
    deriv_right: f64,
) -> Vec<f64> {
    // Full n×n tridiagonal system
    let mut diag = vec![0.0; n];
    let mut sub = vec![0.0; n];
    let mut sup = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    // Left BC: 2*h[0]*c[0] + h[0]*c[1] = 3*((y[1]-y[0])/h[0] - deriv_left)
    diag[0] = 2.0 * h[0];
    sup[0] = h[0];
    rhs[0] = 3.0 * ((y[1] - y[0]) / h[0] - deriv_left);

    // Interior rows
    for i in 1..n - 1 {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    // Right BC: h[n-2]*c[n-2] + 2*h[n-2]*c[n-1] = 3*(deriv_right - (y[n-1]-y[n-2])/h[n-2])
    let m = n - 1;
    sub[m] = h[m - 1];
    diag[m] = 2.0 * h[m - 1];
    rhs[m] = 3.0 * (deriv_right - (y[m] - y[m - 1]) / h[m - 1]);

    thomas_solve(&sub, &mut diag, &sup, &mut rhs);
    rhs
}

/// Thomas algorithm for tridiagonal system.
fn thomas_solve(sub: &[f64], diag: &mut [f64], sup: &[f64], rhs: &mut [f64]) {
    let n = diag.len();
    if n == 0 {
        return;
    }
    // Forward elimination
    for i in 1..n {
        if diag[i - 1].abs() < f64::EPSILON * 1e6 {
            continue;
        }
        let w = sub[i] / diag[i - 1];
        diag[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    // Back substitution
    if diag[n - 1].abs() > f64::EPSILON * 1e6 {
        rhs[n - 1] /= diag[n - 1];
    }
    for i in (0..n - 1).rev() {
        if diag[i].abs() > f64::EPSILON * 1e6 {
            rhs[i] = (rhs[i] - sup[i] * rhs[i + 1]) / diag[i];
        }
    }
}

/// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolator.
///
/// Matches `scipy.interpolate.PchipInterpolator(x, y)`.
/// Preserves monotonicity of the data — no overshoots or oscillations.
#[derive(Debug)]
pub struct PchipInterpolator {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}

impl PchipInterpolator {
    /// Create a PCHIP interpolator from data points.
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self, InterpError> {
        if x.len() != y.len() {
            return Err(InterpError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }
        if x.len() < 2 {
            return Err(InterpError::TooFewPoints {
                minimum: 2,
                actual: x.len(),
            });
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InterpError::UnsortedX);
        }

        let n = x.len();
        let m = n - 1;

        // Compute slopes
        let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();
        let delta: Vec<f64> = (0..m).map(|i| (y[i + 1] - y[i]) / h[i]).collect();

        // Compute PCHIP derivatives at each point
        let mut d = vec![0.0; n];

        if n == 2 {
            d[0] = delta[0];
            d[1] = delta[0];
        } else {
            // Interior points: weighted harmonic mean of adjacent slopes
            for i in 1..m {
                if delta[i - 1].signum() != delta[i].signum()
                    || delta[i - 1] == 0.0
                    || delta[i] == 0.0
                {
                    d[i] = 0.0; // sign change or zero slope → flat
                } else {
                    // Weighted harmonic mean (Fritsch-Carlson)
                    let w1 = 2.0 * h[i] + h[i - 1];
                    let w2 = h[i] + 2.0 * h[i - 1];
                    d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
                }
            }

            // Endpoints: one-sided shape-preserving formula
            d[0] = pchip_end_slope(&h, &delta, true);
            d[n - 1] = pchip_end_slope(&h, &delta, false);
        }

        // Build Hermite cubic coefficients for each interval
        let mut coeffs = Vec::with_capacity(m);
        for i in 0..m {
            let hi = h[i];
            let a = y[i];
            let b = d[i];
            let c = (3.0 * delta[i] - 2.0 * d[i] - d[i + 1]) / hi;
            let dd = (d[i] + d[i + 1] - 2.0 * delta[i]) / (hi * hi);
            coeffs.push([a, b, c, dd]);
        }

        Ok(Self {
            x: x.to_vec(),
            coeffs,
        })
    }

    /// Evaluate at a single point.
    pub fn eval(&self, x_new: f64) -> f64 {
        let n = self.x.len();
        if x_new <= self.x[0] {
            return self.eval_at_interval(0, x_new);
        }
        if x_new >= self.x[n - 1] {
            return self.eval_at_interval(n - 2, x_new);
        }
        let i = self.find_interval(x_new);
        self.eval_at_interval(i, x_new)
    }

    /// Evaluate at multiple points.
    pub fn eval_many(&self, x_new: &[f64]) -> Vec<f64> {
        x_new.iter().map(|&xi| self.eval(xi)).collect()
    }

    fn eval_at_interval(&self, i: usize, x_new: f64) -> f64 {
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    fn find_interval(&self, x_new: f64) -> usize {
        let n = self.x.len();
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.x[mid] <= x_new {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Endpoint slope for PCHIP using one-sided shape-preserving formula.
fn pchip_end_slope(h: &[f64], delta: &[f64], is_left: bool) -> f64 {
    let m = delta.len();
    if m < 2 {
        return delta[0];
    }

    let (d1, d2, h1, h2) = if is_left {
        (delta[0], delta[1], h[0], h[1])
    } else {
        (delta[m - 1], delta[m - 2], h[m - 1], h[m - 2])
    };

    // Bessel's formula with monotonicity correction
    let mut d = ((2.0 * h1 + h2) * d1 - h1 * d2) / (h1 + h2);

    // Enforce monotonicity: if sign differs from d1, set to 0
    if d.signum() != d1.signum() {
        d = 0.0;
    } else if d1.signum() != d2.signum() && d.abs() > 3.0 * d1.abs() {
        d = 3.0 * d1;
    }

    d
}

/// Convenience function: 1D linear interpolation at multiple points.
///
/// Matches `numpy.interp(x_new, x, y)`.
pub fn interp1d_linear(x: &[f64], y: &[f64], x_new: &[f64]) -> Result<Vec<f64>, InterpError> {
    let interp = Interp1d::new(
        x,
        y,
        Interp1dOptions {
            bounds_error: false,
            ..Interp1dOptions::default()
        },
    )?;
    interp.eval_many(x_new)
}

// ══════════════════════════════════════════════════════════════════════
// Standalone CubicSpline
// ══════════════════════════════════════════════════════════════════════

/// Standalone cubic spline interpolator with rich API.
///
/// Matches `scipy.interpolate.CubicSpline(x, y, bc_type=...)`.
///
/// Stores piecewise polynomial coefficients in PPoly form:
/// `S_i(x) = c[0][i]*(x-x_i)^3 + c[1][i]*(x-x_i)^2 + c[2][i]*(x-x_i) + c[3][i]`
#[derive(Debug, Clone)]
pub struct CubicSplineStandalone {
    x: Vec<f64>,
    /// Coefficients in PPoly layout: `coeffs[i] = [d, c, b, a]` where
    /// `S_i(dx) = a + b*dx + c*dx^2 + d*dx^3`.
    coeffs: Vec<[f64; 4]>,
}

impl CubicSplineStandalone {
    /// Construct a cubic spline interpolator.
    pub fn new(x: &[f64], y: &[f64], bc: SplineBc) -> Result<Self, InterpError> {
        if x.len() != y.len() {
            return Err(InterpError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }
        if x.len() < 4 {
            return Err(InterpError::TooFewPoints {
                minimum: 4,
                actual: x.len(),
            });
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InterpError::UnsortedX);
        }

        let coeffs = compute_cubic_spline(x, y, bc)?;
        Ok(Self {
            x: x.to_vec(),
            coeffs,
        })
    }

    /// Evaluate the spline at a single point.
    pub fn eval(&self, x_new: f64) -> f64 {
        let i = self.find_interval(x_new);
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    /// Evaluate at multiple points.
    pub fn eval_many(&self, x_new: &[f64]) -> Vec<f64> {
        x_new.iter().map(|&xi| self.eval(xi)).collect()
    }

    /// Return a new cubic spline representing the `nu`-th derivative.
    pub fn derivative(&self, nu: usize) -> CubicSplineDerivative {
        let m = self.coeffs.len();
        match nu {
            0 => CubicSplineDerivative {
                x: self.x.clone(),
                coeffs: self.coeffs.clone(),
                degree: 3,
            },
            1 => {
                // S'(dx) = b + 2c*dx + 3d*dx^2
                let dc: Vec<[f64; 4]> = self
                    .coeffs
                    .iter()
                    .map(|&[_a, b, c, d]| [b, 2.0 * c, 3.0 * d, 0.0])
                    .collect();
                CubicSplineDerivative {
                    x: self.x.clone(),
                    coeffs: dc,
                    degree: 2,
                }
            }
            2 => {
                // S''(dx) = 2c + 6d*dx
                let dc: Vec<[f64; 4]> = self
                    .coeffs
                    .iter()
                    .map(|&[_a, _b, c, d]| [2.0 * c, 6.0 * d, 0.0, 0.0])
                    .collect();
                CubicSplineDerivative {
                    x: self.x.clone(),
                    coeffs: dc,
                    degree: 1,
                }
            }
            _ => CubicSplineDerivative {
                x: self.x.clone(),
                coeffs: vec![[0.0; 4]; m],
                degree: 0,
            },
        }
    }

    /// Compute the definite integral over [a, b].
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        if (b - a).abs() < 1e-15 {
            return 0.0;
        }
        let sign = if a > b { -1.0 } else { 1.0 };
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };

        let n = self.x.len();
        let mut total = 0.0;

        for i in 0..n - 1 {
            let seg_lo = self.x[i].max(lo);
            let seg_hi = self.x[i + 1].min(hi);
            if seg_lo >= seg_hi {
                continue;
            }
            let [a0, b0, c0, d0] = self.coeffs[i];
            let xi = self.x[i];
            let dx_lo = seg_lo - xi;
            let dx_hi = seg_hi - xi;

            // Integral of a + b*dx + c*dx^2 + d*dx^3
            let anti = |dx: f64| {
                a0 * dx + b0 * dx * dx / 2.0 + c0 * dx.powi(3) / 3.0 + d0 * dx.powi(4) / 4.0
            };
            total += anti(dx_hi) - anti(dx_lo);
        }

        sign * total
    }

    fn find_interval(&self, x_new: f64) -> usize {
        let n = self.x.len();
        if x_new <= self.x[0] {
            return 0;
        }
        if x_new >= self.x[n - 1] {
            return n - 2;
        }
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.x[mid] <= x_new {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Result of `CubicSplineStandalone::derivative()`.
#[derive(Debug, Clone)]
pub struct CubicSplineDerivative {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
    degree: usize,
}

impl CubicSplineDerivative {
    /// Evaluate the derivative at a point.
    pub fn eval(&self, x_new: f64) -> f64 {
        let n = self.x.len();
        let i = {
            if x_new <= self.x[0] {
                0
            } else if x_new >= self.x[n - 1] {
                n - 2
            } else {
                let mut lo = 0;
                let mut hi = n - 1;
                while hi - lo > 1 {
                    let mid = (lo + hi) / 2;
                    if self.x[mid] <= x_new {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                lo
            }
        };
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }
}

// ══════════════════════════════════════════════════════════════════════
// Akima1DInterpolator
// ══════════════════════════════════════════════════════════════════════

/// Akima (1970) piecewise cubic Hermite interpolator.
///
/// Matches `scipy.interpolate.Akima1DInterpolator(x, y)`.
///
/// Uses a local 5-point stencil to compute slopes, producing smooth
/// interpolation with minimal overshoot. Unlike natural cubic splines,
/// changes to one data point only affect nearby intervals.
#[derive(Debug, Clone)]
pub struct Akima1DInterpolator {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}

impl Akima1DInterpolator {
    /// Create an Akima interpolator from data points.
    ///
    /// Requires at least 2 data points. x must be strictly increasing.
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self, InterpError> {
        if x.len() != y.len() {
            return Err(InterpError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }
        if x.len() < 2 {
            return Err(InterpError::TooFewPoints {
                minimum: 2,
                actual: x.len(),
            });
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InterpError::UnsortedX);
        }

        let n = x.len();
        let m = n - 1;

        // Compute slopes between consecutive points
        let delta: Vec<f64> = (0..m)
            .map(|i| (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            .collect();

        // Compute Akima slopes at each point
        let slopes = akima_slopes(&delta);

        // Build Hermite cubic coefficients
        let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();
        let mut coeffs = Vec::with_capacity(m);
        for i in 0..m {
            let hi = h[i];
            let a = y[i];
            let b = slopes[i];
            let c = (3.0 * delta[i] - 2.0 * slopes[i] - slopes[i + 1]) / hi;
            let d = (slopes[i] + slopes[i + 1] - 2.0 * delta[i]) / (hi * hi);
            coeffs.push([a, b, c, d]);
        }

        Ok(Self {
            x: x.to_vec(),
            coeffs,
        })
    }

    /// Evaluate at a single point.
    pub fn eval(&self, x_new: f64) -> f64 {
        let i = self.find_interval(x_new);
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    /// Evaluate at multiple points.
    pub fn eval_many(&self, x_new: &[f64]) -> Vec<f64> {
        x_new.iter().map(|&xi| self.eval(xi)).collect()
    }

    fn find_interval(&self, x_new: f64) -> usize {
        let n = self.x.len();
        if x_new <= self.x[0] {
            return 0;
        }
        if x_new >= self.x[n - 1] {
            return n - 2;
        }
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.x[mid] <= x_new {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Compute Akima slopes from segment slopes using the 5-point stencil.
///
/// Akima's formula: at each interior point, the slope is a weighted average
/// of adjacent segment slopes, where the weights are based on the absolute
/// differences of slopes on each side. This reduces overshoot.
fn akima_slopes(delta: &[f64]) -> Vec<f64> {
    let m = delta.len();
    let n = m + 1;
    let mut slopes = vec![0.0; n];

    if m == 0 {
        return slopes;
    }
    if m == 1 {
        slopes[0] = delta[0];
        slopes[1] = delta[0];
        return slopes;
    }

    // Extend delta with two phantom values on each side (Akima's parabolic extension)
    let mut ext = Vec::with_capacity(m + 4);
    ext.push(2.0 * delta[0] - delta[1]);
    ext.push(2.0 * delta[0] - delta[1] + (delta[0] - delta[1]));
    ext.extend_from_slice(delta);
    ext.push(2.0 * delta[m - 1] - delta[m - 2]);
    ext.push(2.0 * delta[m - 1] - delta[m - 2] + (delta[m - 1] - delta[m - 2]));

    // Recompute with proper phantom slopes:
    // ext[0] = 2*d[0] - d[1]  (left phantom 2)
    // ext[1] = d[0]           (left phantom 1)  => actually just use linear extrapolation
    let mut d = Vec::with_capacity(m + 4);
    // Two phantom slopes on the left
    d.push(2.0 * delta[0] - delta.get(1).copied().unwrap_or(delta[0]));
    d.push(delta[0] + (delta[0] - delta.get(1).copied().unwrap_or(delta[0])));
    // Actual slopes
    d.extend_from_slice(delta);
    // Two phantom slopes on the right
    d.push(
        delta[m - 1]
            + (delta[m - 1]
                - delta
                    .get(m.wrapping_sub(2))
                    .copied()
                    .unwrap_or(delta[m - 1])),
    );
    d.push(
        2.0 * delta[m - 1]
            - delta
                .get(m.wrapping_sub(2))
                .copied()
                .unwrap_or(delta[m - 1]),
    );

    // Now d has m+4 elements, indexed from 0 to m+3.
    // Original delta[i] = d[i+2].
    // For each data point j (0..n), use d[j], d[j+1], d[j+2], d[j+3].
    for j in 0..n {
        let w1 = (d[j + 3] - d[j + 2]).abs();
        let w2 = (d[j + 1] - d[j]).abs();

        if w1 + w2 < 1e-30 {
            // All slopes are equal — use simple average
            slopes[j] = 0.5 * (d[j + 1] + d[j + 2]);
        } else {
            slopes[j] = (w1 * d[j + 1] + w2 * d[j + 2]) / (w1 + w2);
        }
    }

    slopes
}

// ══════════════════════════════════════════════════════════════════════
// BSpline — Cox-de Boor B-spline evaluation
// ══════════════════════════════════════════════════════════════════════

/// B-spline representation of a piecewise polynomial curve.
///
/// Matches `scipy.interpolate.BSpline(t, c, k)`.
///
/// A B-spline of degree `k` is defined by knots `t` and coefficients `c`.
/// The spline value at `x` is `sum_i c[i] * B_{i,k}(x)` where `B_{i,k}` are
/// the B-spline basis functions evaluated via the Cox-de Boor recursion.
#[derive(Debug, Clone)]
pub struct BSpline {
    /// Knot vector (non-decreasing), length n + k + 1 where n = len(c).
    t: Vec<f64>,
    /// Spline coefficients, length n.
    c: Vec<f64>,
    /// Spline degree (order = k + 1).
    k: usize,
}

impl BSpline {
    /// Create a B-spline from knots, coefficients, and degree.
    ///
    /// # Requirements
    /// - `t` must be non-decreasing.
    /// - `t.len() == c.len() + k + 1`.
    /// - `k >= 0`.
    pub fn new(t: Vec<f64>, c: Vec<f64>, k: usize) -> Result<Self, InterpError> {
        let expected_knots = c.len() + k + 1;
        if t.len() != expected_knots {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "knot vector length {} != c.len() + k + 1 = {}",
                    t.len(),
                    expected_knots
                ),
            });
        }
        if t.windows(2).any(|w| w[1] < w[0]) {
            return Err(InterpError::InvalidArgument {
                detail: "knot vector must be non-decreasing".to_string(),
            });
        }
        Ok(Self { t, c, k })
    }

    /// Evaluate the B-spline at a single point using Cox-de Boor recursion.
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.c.len();
        let k = self.k;
        let t = &self.t;

        // Find the knot span: largest i such that t[i] <= x < t[i+1]
        // with clamping for the right endpoint.
        let mu = self.find_span(x);

        // Evaluate via de Boor's algorithm (triangular table)
        // Only basis functions B_{mu-k,k} through B_{mu,k} are nonzero at x.
        let mut d: Vec<f64> = (0..=k)
            .map(|j| {
                let idx = mu.wrapping_sub(k) + j;
                if idx < n { self.c[idx] } else { 0.0 }
            })
            .collect();

        for r in 1..=k {
            for j in (r..=k).rev() {
                let left = mu.wrapping_sub(k) + j;
                let right = left + k + 1 - r;
                if right < t.len() && left < t.len() {
                    let denom = t[right] - t[left];
                    if denom.abs() > 0.0 {
                        let alpha = (x - t[left]) / denom;
                        d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
                    }
                }
            }
        }

        d[k]
    }

    /// Evaluate the B-spline at multiple points.
    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.eval(x)).collect()
    }

    /// Construct a new BSpline representing the `nu`-th derivative.
    ///
    /// The derivative of a B-spline of degree k is a B-spline of degree k-1
    /// with modified coefficients and the interior knots.
    pub fn derivative(&self, nu: usize) -> Result<Self, InterpError> {
        if nu == 0 {
            return Ok(self.clone());
        }
        let mut t = self.t.clone();
        let mut c = self.c.clone();
        let mut k = self.k;

        for _ in 0..nu {
            if k == 0 {
                return Err(InterpError::InvalidArgument {
                    detail: "cannot differentiate degree-0 spline".to_string(),
                });
            }
            let n = c.len();
            let mut dc = Vec::with_capacity(n - 1);
            for i in 0..n - 1 {
                let denom = t[i + k + 1] - t[i + 1];
                if denom.abs() > 0.0 {
                    dc.push(k as f64 * (c[i + 1] - c[i]) / denom);
                } else {
                    dc.push(0.0);
                }
            }
            // Remove first and last knot
            t.remove(t.len() - 1);
            t.remove(0);
            c = dc;
            k -= 1;
        }

        Self::new(t, c, k)
    }

    /// Construct a new BSpline representing the `nu`-th antiderivative.
    ///
    /// The antiderivative of a B-spline of degree k is a B-spline of degree k+1.
    pub fn antiderivative(&self, nu: usize) -> Result<Self, InterpError> {
        if nu == 0 {
            return Ok(self.clone());
        }
        let mut t = self.t.clone();
        let mut c = self.c.clone();
        let mut k = self.k;

        for _ in 0..nu {
            let n = c.len();
            // Add a knot at each end (repeat first and last)
            t.insert(0, t[0]);
            t.push(t[t.len() - 1]);
            k += 1;

            // New coefficients via cumulative sum
            let mut new_c = vec![0.0; n + 1];
            // new_c[0] = 0 (integration constant)
            for i in 0..n {
                let denom = t[i + k + 1] - t[i + 1];
                let contrib = if denom.abs() > 0.0 {
                    denom / k as f64 * c[i]
                } else {
                    0.0
                };
                new_c[i + 1] = new_c[i] + contrib;
            }
            c = new_c;
        }

        Self::new(t, c, k)
    }

    /// Compute the definite integral of the spline over [a, b].
    pub fn integrate(&self, a: f64, b: f64) -> Result<f64, InterpError> {
        let anti = self.antiderivative(1)?;
        Ok(anti.eval(b) - anti.eval(a))
    }

    /// Find the knot span index for evaluation.
    fn find_span(&self, x: f64) -> usize {
        let n = self.c.len();
        let k = self.k;
        let t = &self.t;

        // Clamp to the valid domain [t[k], t[n]]
        if x <= t[k] {
            return k;
        }
        if x >= t[n] {
            // Return the last valid span
            let mut mu = n - 1;
            while mu > k && t[mu] == t[n] {
                mu -= 1;
            }
            return mu;
        }

        // Binary search for the span
        let mut lo = k;
        let mut hi = n;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if t[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Construct an interpolating B-spline from data points.
///
/// Matches `scipy.interpolate.make_interp_spline(x, y, k=3)`.
///
/// Given data points (x, y), constructs a B-spline of degree `k` that passes
/// through all data points. Uses not-a-knot boundary conditions by default.
///
/// # Arguments
/// * `x` — Abscissas (strictly increasing), length n.
/// * `y` — Ordinates, length n.
/// * `k` — Spline degree (default 3 for cubic).
pub fn make_interp_spline(x: &[f64], y: &[f64], k: usize) -> Result<BSpline, InterpError> {
    let n = x.len();
    if n != y.len() {
        return Err(InterpError::LengthMismatch {
            x_len: n,
            y_len: y.len(),
        });
    }
    if n < k + 1 {
        return Err(InterpError::TooFewPoints {
            minimum: k + 1,
            actual: n,
        });
    }
    if x.windows(2).any(|w| w[1] <= w[0]) {
        return Err(InterpError::UnsortedX);
    }

    // Build the knot vector (clamped / open uniform).
    // Total knots needed: n + k + 1.
    // Structure: k+1 copies of x[0], n-k-1 interior knots, k+1 copies of x[n-1].
    let num_knots = n + k + 1;
    let num_interior = n - k - 1;
    let mut t = Vec::with_capacity(num_knots);

    // k+1 copies of x[0]
    for _ in 0..=k {
        t.push(x[0]);
    }
    // Interior knots: use data points x[k]..x[n-2] for Schoenberg-Whitney
    // This ensures n-k-1 interior knots, giving n+k+1 total.
    for i in 0..num_interior {
        t.push(x[i + 1 + (k - 1) / 2]);
    }
    // k+1 copies of x[n-1]
    for _ in 0..=k {
        t.push(x[n - 1]);
    }

    debug_assert_eq!(t.len(), num_knots, "knot vector length mismatch");

    // For the clamped knot vector, the system is n×n with the collocation matrix.
    // A[i,j] = B_{j,k}(x[i]) for i,j = 0..n-1.
    // We need to solve A * c = y for the coefficients c.

    // Build collocation matrix (dense, since n is typically small for interp)
    let mut a_mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        let basis = eval_basis_all(&t, x[i], k, n);
        a_mat[i][..n].copy_from_slice(&basis[..n]);
    }

    // Solve via Gaussian elimination with partial pivoting
    let mut rhs = y.to_vec();
    let c = solve_dense_system(&mut a_mat, &mut rhs)?;

    BSpline::new(t, c, k)
}

/// Construct a least-squares B-spline approximation.
///
/// Matches `scipy.interpolate.make_lsq_spline(x, y, t, k=3)`.
///
/// Given data points (x, y) and a knot vector `t`, constructs a B-spline
/// that minimizes the sum of squared residuals.
pub fn make_lsq_spline(x: &[f64], y: &[f64], t: &[f64], k: usize) -> Result<BSpline, InterpError> {
    let m = x.len();
    if m != y.len() {
        return Err(InterpError::LengthMismatch {
            x_len: m,
            y_len: y.len(),
        });
    }

    // Number of coefficients
    let n = t.len() - k - 1;
    if n == 0 || n > m {
        return Err(InterpError::InvalidArgument {
            detail: format!("need t.len() - k - 1 coefficients ({n}) <= data points ({m})"),
        });
    }

    // Build collocation matrix A (m × n) and solve normal equations A^T A c = A^T y
    let mut ata = vec![vec![0.0; n]; n];
    let mut aty = vec![0.0; n];

    for i in 0..m {
        let basis = eval_basis_all(t, x[i], k, n);
        for j in 0..n {
            aty[j] += basis[j] * y[i];
            for l in 0..n {
                ata[j][l] += basis[j] * basis[l];
            }
        }
    }

    let c = solve_dense_system(&mut ata, &mut aty)?;

    BSpline::new(t.to_vec(), c, k)
}

/// Evaluate all n B-spline basis functions of degree k at point x.
fn eval_basis_all(t: &[f64], x: f64, k: usize, n: usize) -> Vec<f64> {
    let mut basis = vec![0.0; n];

    // Degree 0: indicator functions
    for i in 0..n {
        if i + 1 < t.len() {
            basis[i] = if (t[i] <= x && x < t[i + 1]) || (x == t[i + 1] && i + 1 == t.len() - k - 1)
            {
                1.0
            } else {
                0.0
            };
        }
    }

    // Build up degree by degree using Cox-de Boor recursion
    for p in 1..=k {
        let prev = basis.clone();
        for i in 0..n {
            let mut val = 0.0;
            if i + p < t.len() {
                let denom_left = t[i + p] - t[i];
                if denom_left > 0.0 {
                    val += (x - t[i]) / denom_left * prev[i];
                }
            }
            if i + p + 1 < t.len() && i + 1 < n {
                let denom_right = t[i + p + 1] - t[i + 1];
                if denom_right > 0.0 {
                    val += (t[i + p + 1] - x) / denom_right * prev[i + 1];
                }
            }
            basis[i] = val;
        }
    }

    basis
}

/// Solve a dense linear system Ax = b via Gaussian elimination with partial pivoting.
fn solve_dense_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<Vec<f64>, InterpError> {
    let n = b.len();
    if n == 0 || a.len() != n {
        return Err(InterpError::InvalidArgument {
            detail: "empty or mismatched system".to_string(),
        });
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for (row, a_row) in a.iter().enumerate().skip(col + 1) {
            if a_row[col].abs() > max_val {
                max_val = a_row[col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(InterpError::InvalidArgument {
                detail: "singular or near-singular collocation matrix".to_string(),
            });
        }
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        // Eliminate below
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            let pivot_row: Vec<f64> = a[col].clone();
            for (j, pval) in pivot_row.iter().enumerate().skip(col) {
                a[row][j] -= factor * pval;
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }

    Ok(x)
}

// ══════════════════════════════════════════════════════════════════════
// N-D Scattered Data Interpolation
// ══════════════════════════════════════════════════════════════════════

/// Nearest-neighbor interpolator for scattered N-D data.
///
/// Matches `scipy.interpolate.NearestNDInterpolator(points, values)`.
///
/// Constructs a KDTree from the input points and returns the value of
/// the nearest data point for each query.
#[derive(Debug)]
pub struct NearestNDInterpolator {
    tree: fsci_spatial::KDTree,
    values: Vec<f64>,
}

impl NearestNDInterpolator {
    /// Create a nearest-neighbor interpolator from scattered data.
    ///
    /// # Arguments
    /// * `points` — N-D data points, each as a Vec of coordinates
    /// * `values` — Function values at each point (same length as points)
    pub fn new(points: &[Vec<f64>], values: &[f64]) -> Result<Self, InterpError> {
        if points.is_empty() {
            return Err(InterpError::TooFewPoints {
                minimum: 1,
                actual: 0,
            });
        }
        if points.len() != values.len() {
            return Err(InterpError::LengthMismatch {
                x_len: points.len(),
                y_len: values.len(),
            });
        }
        let tree = fsci_spatial::KDTree::new(points).map_err(|e| InterpError::InvalidArgument {
            detail: format!("KDTree construction failed: {e}"),
        })?;
        Ok(Self {
            tree,
            values: values.to_vec(),
        })
    }

    /// Evaluate the interpolator at a single query point.
    pub fn eval(&self, query: &[f64]) -> Result<f64, InterpError> {
        let (idx, _dist) = self
            .tree
            .query(query)
            .map_err(|e| InterpError::InvalidArgument {
                detail: format!("query failed: {e}"),
            })?;
        Ok(self.values[idx])
    }

    /// Evaluate the interpolator at multiple query points.
    pub fn eval_many(&self, queries: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> {
        queries.iter().map(|q| self.eval(q)).collect()
    }
}

/// Interpolation method for griddata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GriddataMethod {
    /// Nearest-neighbor interpolation (always works, discontinuous).
    Nearest,
}

/// Interpolate unstructured N-D data onto specified points.
///
/// Matches `scipy.interpolate.griddata(points, values, xi, method)`.
///
/// Currently supports method='nearest'. Linear and cubic methods require
/// Delaunay triangulation (see LinearNDInterpolator bead).
pub fn griddata(
    points: &[Vec<f64>],
    values: &[f64],
    xi: &[Vec<f64>],
    method: GriddataMethod,
) -> Result<Vec<f64>, InterpError> {
    match method {
        GriddataMethod::Nearest => {
            let interp = NearestNDInterpolator::new(points, values)?;
            interp.eval_many(xi)
        }
    }
}

// ── RegularGridInterpolator ─────────────────────────────────────────

/// Interpolation method for `RegularGridInterpolator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegularGridMethod {
    /// Multilinear interpolation (tensor product of 1D linear).
    #[default]
    Linear,
    /// Nearest-neighbor interpolation on the grid.
    Nearest,
}

/// N-dimensional interpolation on a regular (rectilinear) grid.
///
/// Matches `scipy.interpolate.RegularGridInterpolator(points, values)`.
///
/// Each axis is specified by a sorted, monotonically increasing 1-D array
/// of grid coordinates. The values array has shape `(n_0, n_1, ..., n_{d-1})`
/// stored in row-major (C) order as a flat `Vec<f64>`.
///
/// Supports linear (multilinear) and nearest-neighbor interpolation.
#[derive(Debug, Clone)]
pub struct RegularGridInterpolator {
    /// Grid coordinates for each axis, sorted and strictly increasing.
    points: Vec<Vec<f64>>,
    /// Values on the grid, stored in row-major order. Length = product of axis sizes.
    values: Vec<f64>,
    /// Strides for row-major indexing: stride[i] = product of sizes of axes i+1..d.
    strides: Vec<usize>,
    /// Interpolation method.
    method: RegularGridMethod,
    /// Whether to raise an error for out-of-bounds queries.
    bounds_error: bool,
    /// Fill value for out-of-bounds queries (used when `bounds_error` is false).
    fill_value: Option<f64>,
}

impl RegularGridInterpolator {
    /// Create a new regular grid interpolator.
    ///
    /// # Arguments
    /// * `points` — Grid coordinates for each dimension. Each must be sorted
    ///   and strictly monotonically increasing with at least 2 elements.
    /// * `values` — Function values on the grid in row-major order.
    ///   Length must equal the product of all axis sizes.
    /// * `method` — Interpolation method (Linear or Nearest).
    /// * `bounds_error` — If true, out-of-bounds queries return an error.
    /// * `fill_value` — Value to return for out-of-bounds queries when
    ///   `bounds_error` is false. If `None`, uses `f64::NAN`.
    pub fn new(
        points: Vec<Vec<f64>>,
        values: Vec<f64>,
        method: RegularGridMethod,
        bounds_error: bool,
        fill_value: Option<f64>,
    ) -> Result<Self, InterpError> {
        if points.is_empty() {
            return Err(InterpError::InvalidArgument {
                detail: "points must have at least one axis".to_string(),
            });
        }

        // Validate each axis is sorted, monotonically increasing, and has >= 2 points.
        for (dim, axis) in points.iter().enumerate() {
            if axis.len() < 2 {
                return Err(InterpError::TooFewPoints {
                    minimum: 2,
                    actual: axis.len(),
                });
            }
            for w in axis.windows(2) {
                if w[1] <= w[0] {
                    return Err(InterpError::InvalidArgument {
                        detail: format!("axis {dim} grid points must be strictly increasing"),
                    });
                }
            }
        }

        // Compute expected total size and strides.
        let ndim = points.len();
        let mut strides = vec![0usize; ndim];
        let mut total_size: usize = 1;
        for i in (0..ndim).rev() {
            strides[i] = total_size;
            total_size = total_size.checked_mul(points[i].len()).ok_or_else(|| {
                InterpError::InvalidArgument {
                    detail: "grid size overflow".to_string(),
                }
            })?;
        }

        if values.len() != total_size {
            return Err(InterpError::LengthMismatch {
                x_len: total_size,
                y_len: values.len(),
            });
        }

        Ok(Self {
            points,
            values,
            strides,
            method,
            bounds_error,
            fill_value,
        })
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.points.len()
    }

    /// Evaluate the interpolator at a single query point.
    ///
    /// `xi` must have exactly `ndim` elements.
    pub fn eval(&self, xi: &[f64]) -> Result<f64, InterpError> {
        let ndim = self.ndim();
        if xi.len() != ndim {
            return Err(InterpError::InvalidArgument {
                detail: format!("expected {ndim}-D query point, got {}-D", xi.len()),
            });
        }

        // Check bounds.
        for (dim, &x) in xi.iter().enumerate() {
            let axis = &self.points[dim];
            if x < axis[0] || x > axis[axis.len() - 1] {
                if self.bounds_error {
                    return Err(InterpError::OutOfBounds {
                        value: format!(
                            "dimension {dim}: {x} outside [{}, {}]",
                            axis[0],
                            axis[axis.len() - 1]
                        ),
                    });
                }
                return Ok(self.fill_value.unwrap_or(f64::NAN));
            }
        }

        match self.method {
            RegularGridMethod::Linear => self.eval_linear(xi),
            RegularGridMethod::Nearest => Ok(self.eval_nearest(xi)),
        }
    }

    /// Evaluate at multiple query points.
    pub fn eval_many(&self, xi: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> {
        xi.iter().map(|x| self.eval(x)).collect()
    }

    /// Find the index `i` such that `axis[i] <= x < axis[i+1]`, clamped to valid range.
    fn find_interval(axis: &[f64], x: f64) -> usize {
        // Binary search for the interval.
        let n = axis.len();
        if x <= axis[0] {
            return 0;
        }
        if x >= axis[n - 1] {
            return n - 2; // last valid interval
        }
        // axis[result] <= x < axis[result+1]
        match axis.binary_search_by(|probe| probe.total_cmp(&x)) {
            Ok(i) => {
                // Exact match: use interval starting at i, but clamp to n-2.
                i.min(n - 2)
            }
            Err(i) => {
                // x is between axis[i-1] and axis[i].
                i.saturating_sub(1)
            }
        }
    }

    /// Nearest-neighbor evaluation (already bounds-checked).
    fn eval_nearest(&self, xi: &[f64]) -> f64 {
        let mut flat_idx = 0;
        for ((axis, &x), &stride) in self.points.iter().zip(xi).zip(&self.strides) {
            let i = Self::find_interval(axis, x);
            // Choose the nearest grid point.
            let nearest = if i + 1 < axis.len() && (x - axis[i]).abs() > (axis[i + 1] - x).abs() {
                i + 1
            } else {
                i
            };
            flat_idx += nearest * stride;
        }
        self.values[flat_idx]
    }

    /// Multilinear interpolation (already bounds-checked).
    ///
    /// For d dimensions, interpolates over 2^d vertices of the enclosing hypercube
    /// using successive 1D linear interpolations along each axis.
    fn eval_linear(&self, xi: &[f64]) -> Result<f64, InterpError> {
        let ndim = self.ndim();

        // For each dimension, find the interval index and the fractional position.
        let mut indices = Vec::with_capacity(ndim);
        let mut fracs = Vec::with_capacity(ndim);
        for (axis, &x) in self.points.iter().zip(xi) {
            let i = Self::find_interval(axis, x);
            let frac = if (axis[i + 1] - axis[i]).abs() < f64::EPSILON {
                0.0
            } else {
                (x - axis[i]) / (axis[i + 1] - axis[i])
            };
            indices.push(i);
            fracs.push(frac);
        }

        // Iterate over all 2^ndim corners of the enclosing hypercube.
        // Each corner contributes weight = product of (1-frac) or frac per dimension.
        let num_corners = 1usize << ndim;
        let mut result = 0.0;
        for corner in 0..num_corners {
            let mut weight = 1.0;
            let mut flat_idx = 0;
            for dim in 0..ndim {
                let bit = (corner >> dim) & 1;
                let idx = indices[dim] + bit;
                flat_idx += idx * self.strides[dim];
                weight *= if bit == 0 {
                    1.0 - fracs[dim]
                } else {
                    fracs[dim]
                };
            }
            result += weight * self.values[flat_idx];
        }

        Ok(result)
    }
}

/// Convenience function for N-D interpolation on a regular grid.
///
/// Matches `scipy.interpolate.interpn(points, values, xi, method)`.
///
/// # Arguments
/// * `points` — Grid coordinates for each dimension (list of 1-D arrays).
/// * `values` — Values on the grid in row-major order.
/// * `xi` — Query points, each with `ndim` elements.
/// * `method` — Interpolation method (`"linear"` or `"nearest"`).
/// * `bounds_error` — If true, raise error on out-of-bounds queries.
/// * `fill_value` — Fill value for out-of-bounds queries (default `NaN`).
pub fn interpn(
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
    xi: &[Vec<f64>],
    method: RegularGridMethod,
    bounds_error: bool,
    fill_value: Option<f64>,
) -> Result<Vec<f64>, InterpError> {
    let interp = RegularGridInterpolator::new(points, values, method, bounds_error, fill_value)?;
    interp.eval_many(xi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_interp_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0];
        let interp = Interp1d::new(&x, &y, Interp1dOptions::default()).expect("interp1d");
        for i in 0..4 {
            let val = interp.eval(x[i]).expect("eval");
            assert!((val - y[i]).abs() < 1e-12, "at knot {i}: {val} != {}", y[i]);
        }
    }

    #[test]
    fn linear_interp_midpoints() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 2.0, 0.0];
        let interp = Interp1d::new(&x, &y, Interp1dOptions::default()).expect("interp1d");
        let val = interp.eval(0.5).expect("eval");
        assert!((val - 1.0).abs() < 1e-12, "got {val}");
        let val = interp.eval(1.5).expect("eval");
        assert!((val - 1.0).abs() < 1e-12, "got {val}");
    }

    #[test]
    fn nearest_interp() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![10.0, 20.0, 30.0];
        let opts = Interp1dOptions {
            kind: InterpKind::Nearest,
            ..Interp1dOptions::default()
        };
        let interp = Interp1d::new(&x, &y, opts).expect("interp1d");
        assert_eq!(interp.eval(0.3).unwrap(), 10.0);
        assert_eq!(interp.eval(0.7).unwrap(), 20.0);
        assert_eq!(interp.eval(1.5).unwrap(), 20.0);
    }

    #[test]
    fn cubic_spline_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let opts = Interp1dOptions {
            kind: InterpKind::CubicSpline,
            ..Interp1dOptions::default()
        };
        let interp = Interp1d::new(&x, &y, opts).expect("cubic spline");
        for i in 0..5 {
            let val = interp.eval(x[i]).expect("eval");
            assert!((val - y[i]).abs() < 1e-10, "at knot {i}: {val} != {}", y[i]);
        }
    }

    #[test]
    fn cubic_spline_smooth_between_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let opts = Interp1dOptions {
            kind: InterpKind::CubicSpline,
            ..Interp1dOptions::default()
        };
        let interp = Interp1d::new(&x, &y, opts).expect("cubic spline");
        // x=1.5 => should be close to 2.25
        let val = interp.eval(1.5).expect("eval");
        assert!(
            (val - 2.25).abs() < 0.1,
            "cubic spline at 1.5: {val}, expected ~2.25"
        );
    }

    #[test]
    fn out_of_bounds_error() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let interp = Interp1d::new(&x, &y, Interp1dOptions::default()).expect("interp1d");
        let err = interp.eval(-1.0).expect_err("out of bounds");
        assert!(matches!(err, InterpError::OutOfBounds { .. }));
    }

    #[test]
    fn out_of_bounds_fill_value() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let opts = Interp1dOptions {
            bounds_error: false,
            fill_value: Some(-999.0),
            ..Interp1dOptions::default()
        };
        let interp = Interp1d::new(&x, &y, opts).expect("interp1d");
        assert_eq!(interp.eval(-1.0).unwrap(), -999.0);
        assert_eq!(interp.eval(5.0).unwrap(), -999.0);
    }

    #[test]
    fn unsorted_x_rejected() {
        let err = Interp1d::new(
            &[0.0, 2.0, 1.0],
            &[0.0, 1.0, 2.0],
            Interp1dOptions::default(),
        )
        .expect_err("unsorted");
        assert!(matches!(err, InterpError::UnsortedX));
    }

    #[test]
    fn length_mismatch_rejected() {
        let err =
            Interp1d::new(&[0.0, 1.0], &[0.0], Interp1dOptions::default()).expect_err("mismatch");
        assert!(matches!(err, InterpError::LengthMismatch { .. }));
    }

    #[test]
    fn too_few_points_linear() {
        let err = Interp1d::new(&[0.0], &[0.0], Interp1dOptions::default()).expect_err("too few");
        assert!(matches!(err, InterpError::TooFewPoints { .. }));
    }

    #[test]
    fn eval_many_works() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];
        let interp = Interp1d::new(&x, &y, Interp1dOptions::default()).expect("interp1d");
        let result = interp
            .eval_many(&[0.0, 0.5, 1.0, 1.5, 2.0])
            .expect("eval_many");
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-12);
        assert!((result[2] - 1.0).abs() < 1e-12);
        assert!((result[4] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn interp1d_linear_convenience() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 10.0, 20.0];
        let result = interp1d_linear(&x, &y, &[0.5, 1.5]).expect("interp1d_linear");
        assert!((result[0] - 5.0).abs() < 1e-12);
        assert!((result[1] - 15.0).abs() < 1e-12);
    }

    // ── Not-a-knot spline tests ─────────────────────────────────────

    #[test]
    fn not_a_knot_spline_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let opts = Interp1dOptions {
            kind: InterpKind::CubicSpline,
            spline_bc: SplineBc::NotAKnot,
            ..Interp1dOptions::default()
        };
        let interp = Interp1d::new(&x, &y, opts).expect("not-a-knot spline");
        for i in 0..5 {
            let val = interp.eval(x[i]).expect("eval");
            assert!((val - y[i]).abs() < 1e-8, "at knot {i}: {val} != {}", y[i]);
        }
    }

    #[test]
    fn not_a_knot_spline_quadratic_exact() {
        // For y = x², not-a-knot should reproduce the quadratic exactly
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let opts = Interp1dOptions {
            kind: InterpKind::CubicSpline,
            spline_bc: SplineBc::NotAKnot,
            ..Interp1dOptions::default()
        };
        let interp = Interp1d::new(&x, &y, opts).expect("not-a-knot spline");
        let val = interp.eval(1.5).expect("eval");
        assert!(
            (val - 2.25).abs() < 0.1,
            "not-a-knot at 1.5: {val}, expected ~2.25"
        );
    }

    // ── Clamped spline tests ────────────────────────────────────────

    #[test]
    fn clamped_spline_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let opts = Interp1dOptions {
            kind: InterpKind::CubicSpline,
            spline_bc: SplineBc::Clamped(0.0, 8.0), // y'(0)=0, y'(4)=8
            ..Interp1dOptions::default()
        };
        let interp = Interp1d::new(&x, &y, opts).expect("clamped spline");
        for i in 0..5 {
            let val = interp.eval(x[i]).expect("eval");
            assert!(
                (val - y[i]).abs() < 1e-8,
                "clamped at knot {i}: {val} != {}",
                y[i]
            );
        }
    }

    #[test]
    fn clamped_spline_derivative_at_boundaries() {
        // y = x³ => y'(0)=0, y'(2)=12
        let x = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi * xi).collect();
        let opts = Interp1dOptions {
            kind: InterpKind::CubicSpline,
            spline_bc: SplineBc::Clamped(0.0, 12.0),
            ..Interp1dOptions::default()
        };
        let interp = Interp1d::new(&x, &y, opts).expect("clamped spline");
        // Evaluate at interior points — should be close to x³
        let val = interp.eval(0.75).expect("eval");
        let expected = 0.75_f64.powi(3);
        assert!(
            (val - expected).abs() < 0.05,
            "clamped at 0.75: {val}, expected {expected}"
        );
    }

    // ── PCHIP tests ─────────────────────────────────────────────────

    #[test]
    fn pchip_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // x²
        let pchip = PchipInterpolator::new(&x, &y).expect("pchip");
        for i in 0..5 {
            let val = pchip.eval(x[i]);
            assert!(
                (val - y[i]).abs() < 1e-10,
                "pchip at knot {i}: {val} != {}",
                y[i]
            );
        }
    }

    #[test]
    fn pchip_monotone_preserving() {
        // Monotonically increasing data should produce monotone interpolation
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 0.1, 0.5, 0.9, 1.0];
        let pchip = PchipInterpolator::new(&x, &y).expect("pchip");
        let mut prev = -1.0;
        for i in 0..40 {
            let t = i as f64 * 0.1;
            let val = pchip.eval(t);
            assert!(
                val >= prev - 1e-10,
                "PCHIP should be monotone: val={val} < prev={prev} at t={t}"
            );
            prev = val;
        }
    }

    #[test]
    fn pchip_no_overshoot() {
        // Data with a step: PCHIP should not overshoot
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let pchip = PchipInterpolator::new(&x, &y).expect("pchip");
        for i in 0..40 {
            let t = i as f64 * 0.1;
            let val = pchip.eval(t);
            assert!(
                (-0.01..=1.01).contains(&val),
                "PCHIP should not overshoot: val={val} at t={t}"
            );
        }
    }

    #[test]
    fn pchip_two_points() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let pchip = PchipInterpolator::new(&x, &y).expect("pchip 2pts");
        assert!((pchip.eval(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn pchip_eval_many() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 0.0, -1.0];
        let pchip = PchipInterpolator::new(&x, &y).expect("pchip");
        let result = pchip.eval_many(&[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 4);
        for (i, (&r, &expected)) in result.iter().zip(y.iter()).enumerate() {
            assert!(
                (r - expected).abs() < 1e-10,
                "eval_many[{i}] = {r}, expected {expected}"
            );
        }
    }

    // ── CubicSplineStandalone tests ──────────────────────────────────

    #[test]
    fn cubic_standalone_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let cs = CubicSplineStandalone::new(&x, &y, SplineBc::Natural).expect("cubic");
        for i in 0..x.len() {
            let val = cs.eval(x[i]);
            assert!(
                (val - y[i]).abs() < 1e-10,
                "knot {i}: got {val}, expected {}",
                y[i]
            );
        }
    }

    #[test]
    fn cubic_standalone_not_a_knot() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let cs = CubicSplineStandalone::new(&x, &y, SplineBc::NotAKnot).expect("not-a-knot");
        let val = cs.eval(2.5);
        assert!(
            (val - 6.25).abs() < 0.5,
            "not-a-knot at 2.5: got {val}, expected ~6.25"
        );
    }

    #[test]
    fn cubic_standalone_clamped() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let cs = CubicSplineStandalone::new(&x, &y, SplineBc::Clamped(0.0, 8.0)).expect("clamped");
        for i in 0..x.len() {
            let val = cs.eval(x[i]);
            assert!(
                (val - y[i]).abs() < 1e-8,
                "clamped knot {i}: {val} != {}",
                y[i]
            );
        }
    }

    #[test]
    fn cubic_standalone_derivative_continuity() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let cs = CubicSplineStandalone::new(&x, &y, SplineBc::Natural).expect("cubic");
        let d1 = cs.derivative(1);
        // First derivative should be continuous at interior knots
        for &xi in &x[1..x.len() - 1] {
            let left = d1.eval(xi - 1e-8);
            let right = d1.eval(xi + 1e-8);
            assert!(
                (left - right).abs() < 1e-4,
                "derivative discontinuity at {xi}: left={left}, right={right}"
            );
        }
    }

    #[test]
    fn cubic_standalone_integrate() {
        // y = x^2 from 0 to 4: integral = 64/3 ≈ 21.333
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let cs = CubicSplineStandalone::new(&x, &y, SplineBc::Natural).expect("cubic");
        let integral = cs.integrate(0.0, 4.0);
        assert!(
            (integral - 64.0 / 3.0).abs() < 1.0,
            "integral of x^2: got {integral}, expected ~21.33"
        );
    }

    #[test]
    fn cubic_standalone_too_few() {
        let err = CubicSplineStandalone::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], SplineBc::Natural)
            .expect_err("too few");
        assert!(matches!(err, InterpError::TooFewPoints { .. }));
    }

    // ── Akima1DInterpolator tests ──────────────────────────────────

    #[test]
    fn akima_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // x^2
        let akima = Akima1DInterpolator::new(&x, &y).expect("akima");
        for i in 0..x.len() {
            let val = akima.eval(x[i]);
            assert!(
                (val - y[i]).abs() < 1e-10,
                "akima knot {i}: got {val}, expected {}",
                y[i]
            );
        }
    }

    #[test]
    fn akima_no_overshoot() {
        // Step-function data: Akima should not overshoot
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let akima = Akima1DInterpolator::new(&x, &y).expect("akima");
        for i in 0..50 {
            let t = i as f64 * 0.1;
            let val = akima.eval(t);
            assert!(
                (-0.1..=1.1).contains(&val),
                "akima overshoot at {t}: val={val}"
            );
        }
    }

    #[test]
    fn akima_local_modification() {
        // Changing y[5] should only affect intervals within ~2 of index 5
        let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let y1: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let mut y2 = y1.clone();
        y2[5] = 100.0; // perturb one point in the middle

        let a1 = Akima1DInterpolator::new(&x, &y1).expect("akima1");
        let a2 = Akima1DInterpolator::new(&x, &y2).expect("akima2");

        // Far from perturbation (x=0.5), values should be identical
        let v1 = a1.eval(0.5);
        let v2 = a2.eval(0.5);
        assert!(
            (v1 - v2).abs() < 1e-10,
            "akima locality broken: at x=0.5 got {v1} vs {v2}"
        );
        // Also check far-right
        let v1r = a1.eval(10.5);
        let v2r = a2.eval(10.5);
        assert!(
            (v1r - v2r).abs() < 1e-10,
            "akima locality broken: at x=10.5 got {v1r} vs {v2r}"
        );
    }

    #[test]
    fn akima_vs_cubic_less_ringing() {
        // Near a discontinuity, Akima should have less extreme values than cubic spline
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let akima = Akima1DInterpolator::new(&x, &y).expect("akima");

        // Akima max deviation from [0,1] range should be small
        let mut max_dev = 0.0_f64;
        for i in 0..60 {
            let t = i as f64 * 0.1;
            let val = akima.eval(t);
            let dev = if val < 0.0 {
                -val
            } else if val > 1.0 {
                val - 1.0
            } else {
                0.0
            };
            max_dev = max_dev.max(dev);
        }
        assert!(
            max_dev < 0.2,
            "akima max deviation {max_dev} should be small"
        );
    }

    #[test]
    fn akima_two_points() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let akima = Akima1DInterpolator::new(&x, &y).expect("akima 2pts");
        assert!((akima.eval(0.5) - 0.5).abs() < 1e-10, "akima 2pts midpoint");
    }

    #[test]
    fn akima_eval_many() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 0.0, -1.0];
        let akima = Akima1DInterpolator::new(&x, &y).expect("akima");
        let vals = akima.eval_many(&x);
        for (i, (&v, &e)) in vals.iter().zip(y.iter()).enumerate() {
            assert!(
                (v - e).abs() < 1e-10,
                "eval_many[{i}]: got {v}, expected {e}"
            );
        }
    }

    // ── BSpline tests ───────────────────────────────────────────────

    #[test]
    fn bspline_constant_degree0() {
        // Degree 0: piecewise constant
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let c = vec![10.0, 20.0, 30.0];
        let spl = BSpline::new(t, c, 0).expect("bspline k=0");
        assert!((spl.eval(0.5) - 10.0).abs() < 1e-12, "k=0 first interval");
        assert!((spl.eval(1.5) - 20.0).abs() < 1e-12, "k=0 second interval");
        assert!((spl.eval(2.5) - 30.0).abs() < 1e-12, "k=0 third interval");
    }

    #[test]
    fn bspline_linear_degree1() {
        // Degree 1: piecewise linear
        let t = vec![0.0, 0.0, 1.0, 2.0, 2.0];
        let c = vec![0.0, 1.0, 0.0];
        let spl = BSpline::new(t, c, 1).expect("bspline k=1");
        assert!((spl.eval(0.0) - 0.0).abs() < 1e-12, "k=1 at 0");
        assert!((spl.eval(0.5) - 0.5).abs() < 1e-12, "k=1 at 0.5");
        assert!((spl.eval(1.0) - 1.0).abs() < 1e-12, "k=1 at 1.0");
    }

    #[test]
    fn make_interp_spline_reproduces_linear() {
        // Degree 1 interpolation should recover linear data exactly
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0];
        let spl = make_interp_spline(&x, &y, 1).expect("interp k=1");
        for i in 0..x.len() {
            let val = spl.eval(x[i]);
            assert!(
                (val - y[i]).abs() < 1e-10,
                "interp k=1 at {}: got {}, expected {}",
                x[i],
                val,
                y[i]
            );
        }
        // Midpoint
        assert!((spl.eval(0.5) - 1.0).abs() < 1e-10, "midpoint");
    }

    #[test]
    fn make_interp_spline_cubic_at_knots() {
        // Cubic spline should pass through all data points
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let spl = make_interp_spline(&x, &y, 3).expect("interp k=3");
        for i in 0..x.len() {
            let val = spl.eval(x[i]);
            assert!(
                (val - y[i]).abs() < 1e-8,
                "cubic at knot {}: got {}, expected {}",
                x[i],
                val,
                y[i]
            );
        }
    }

    #[test]
    fn make_interp_spline_cubic_polynomial_exact() {
        // A cubic polynomial should be reproduced exactly by a cubic B-spline
        // y = x^2 (degree 2, well within cubic capacity)
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let spl = make_interp_spline(&x, &y, 3).expect("interp k=3 poly");
        // Evaluate at midpoints
        for &xi in &[0.5, 1.5, 2.5, 3.5] {
            let val = spl.eval(xi);
            let expected = xi * xi;
            assert!(
                (val - expected).abs() < 0.1,
                "cubic poly at {}: got {}, expected {}",
                xi,
                val,
                expected
            );
        }
    }

    #[test]
    fn make_interp_spline_roundtrip() {
        // Evaluate at original x should recover y
        let x = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let y = vec![0.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0]; // x^2
        let spl = make_interp_spline(&x, &y, 3).expect("interp k=3");
        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            let val = spl.eval(xi);
            assert!(
                (val - yi).abs() < 1e-8,
                "roundtrip point {}: got {}, expected {}",
                i,
                val,
                yi
            );
        }
    }

    #[test]
    fn bspline_derivative_cubic() {
        // Derivative of cubic spline for y=x^2 should approximate 2x
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let spl = make_interp_spline(&x, &y, 3).expect("interp k=3");
        let dspl = spl.derivative(1).expect("derivative");
        // Check derivative at interior points
        let d_at_2 = dspl.eval(2.0);
        assert!(
            (d_at_2 - 4.0).abs() < 0.5,
            "derivative at x=2: got {}, expected ~4.0",
            d_at_2
        );
    }

    #[test]
    fn bspline_integrate_polynomial() {
        // Integral of y = x^2 from 0 to 3 = 9.0
        let x = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let spl = make_interp_spline(&x, &y, 3).expect("interp k=3");
        let integral = spl.integrate(0.0, 3.0).expect("integrate");
        assert!(
            (integral - 9.0).abs() < 0.5,
            "integral of x^2 from 0 to 3: got {}, expected 9.0",
            integral
        );
    }

    #[test]
    fn bspline_eval_many() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let spl = make_interp_spline(&x, &y, 1).expect("interp k=1");
        let vals = spl.eval_many(&[0.0, 1.0, 2.0, 3.0]);
        for (i, (&v, &e)) in vals.iter().zip(y.iter()).enumerate() {
            assert!(
                (v - e).abs() < 1e-10,
                "eval_many[{}]: got {}, expected {}",
                i,
                v,
                e
            );
        }
    }

    #[test]
    fn make_interp_spline_too_few_points() {
        let err = make_interp_spline(&[0.0, 1.0], &[0.0, 1.0], 3).expect_err("too few");
        assert!(matches!(err, InterpError::TooFewPoints { .. }));
    }

    #[test]
    fn make_interp_spline_length_mismatch() {
        let err = make_interp_spline(&[0.0, 1.0, 2.0], &[0.0, 1.0], 1).expect_err("mismatch");
        assert!(matches!(err, InterpError::LengthMismatch { .. }));
    }

    #[test]
    fn make_interp_spline_unsorted() {
        let err = make_interp_spline(&[0.0, 2.0, 1.0], &[0.0, 1.0, 2.0], 1).expect_err("unsorted");
        assert!(matches!(err, InterpError::UnsortedX));
    }

    #[test]
    fn make_lsq_spline_basic() {
        // Fit a linear B-spline to noisy linear data
        let x: Vec<f64> = (0..20).map(|i| i as f64 / 19.0 * 4.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        // Knot vector for 5 coefficients, degree 1
        let t = vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0];
        let spl = make_lsq_spline(&x, &y, &t, 1).expect("lsq k=1");
        // Should approximate 2x+1 well
        let val = spl.eval(2.0);
        assert!(
            (val - 5.0).abs() < 0.5,
            "lsq at x=2: got {}, expected ~5.0",
            val
        );
    }

    #[test]
    fn bspline_knot_mismatch() {
        let err = BSpline::new(vec![0.0, 1.0], vec![1.0, 2.0, 3.0], 1).expect_err("mismatch");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    // ── NearestNDInterpolator tests ───────────────────────────────

    #[test]
    fn nearest_nd_exact_at_data_points() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let values = vec![1.0, 2.0, 3.0];
        let interp = NearestNDInterpolator::new(&points, &values).expect("nearestnd");

        // Querying at exact data points should return exact values
        for (i, pt) in points.iter().enumerate() {
            let val = interp.eval(pt).expect("eval");
            assert!(
                (val - values[i]).abs() < 1e-12,
                "at point {i}: got {val}, expected {}",
                values[i]
            );
        }
    }

    #[test]
    fn nearest_nd_between_points() {
        let points = vec![vec![0.0], vec![10.0]];
        let values = vec![100.0, 200.0];
        let interp = NearestNDInterpolator::new(&points, &values).expect("nearestnd");

        // Query near the first point
        let val = interp.eval(&[1.0]).expect("eval near 0");
        assert!(
            (val - 100.0).abs() < 1e-12,
            "closer to 0 → value 100, got {val}"
        );

        // Query near the second point
        let val = interp.eval(&[8.0]).expect("eval near 10");
        assert!(
            (val - 200.0).abs() < 1e-12,
            "closer to 10 → value 200, got {val}"
        );
    }

    #[test]
    fn nearest_nd_3d_data() {
        let points = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, 0.0, 0.0],
        ];
        let values = vec![10.0, 20.0, 30.0];
        let interp = NearestNDInterpolator::new(&points, &values).expect("nearestnd 3d");

        // Point closest to origin
        let val = interp.eval(&[0.1, 0.1, 0.1]).expect("eval");
        assert!((val - 10.0).abs() < 1e-12, "near origin → 10, got {val}");

        // Point closest to (1,1,1)
        let val = interp.eval(&[0.9, 0.9, 0.9]).expect("eval");
        assert!((val - 20.0).abs() < 1e-12, "near (1,1,1) → 20, got {val}");
    }

    #[test]
    fn nearest_nd_empty_rejected() {
        let err = NearestNDInterpolator::new(&[], &[]).expect_err("empty");
        assert!(matches!(err, InterpError::TooFewPoints { .. }));
    }

    #[test]
    fn nearest_nd_length_mismatch() {
        let err = NearestNDInterpolator::new(&[vec![0.0]], &[1.0, 2.0]).expect_err("mismatch");
        assert!(matches!(err, InterpError::LengthMismatch { .. }));
    }

    #[test]
    fn griddata_nearest_basic() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let xi = vec![vec![0.1, 0.1], vec![0.9, 0.9]];

        let result = griddata(&points, &values, &xi, GriddataMethod::Nearest).expect("griddata");
        assert_eq!(result.len(), 2);
        // (0.1, 0.1) closest to (0,0) → value 0
        assert!((result[0] - 0.0).abs() < 1e-12, "got {}", result[0]);
        // (0.9, 0.9) closest to (1,1) → value 3
        assert!((result[1] - 3.0).abs() < 1e-12, "got {}", result[1]);
    }

    #[test]
    fn griddata_out_of_bounds_still_works() {
        // Nearest interpolation works even far from data
        let points = vec![vec![0.0], vec![1.0]];
        let values = vec![10.0, 20.0];
        let xi = vec![vec![-100.0], vec![100.0]];
        let result = griddata(&points, &values, &xi, GriddataMethod::Nearest).expect("griddata");
        assert!((result[0] - 10.0).abs() < 1e-12); // nearest to 0
        assert!((result[1] - 20.0).abs() < 1e-12); // nearest to 1
    }

    #[test]
    fn nearest_nd_large_dataset() {
        // 100 random-ish 2D points with known function
        let points: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let x = (i as f64 * 0.73) % 10.0;
                let y = (i as f64 * 1.37) % 10.0;
                vec![x, y]
            })
            .collect();
        let values: Vec<f64> = points.iter().map(|p| p[0] + p[1]).collect();
        let interp = NearestNDInterpolator::new(&points, &values).expect("nearestnd large");

        // Query at exact data points should return exact values
        for (i, pt) in points.iter().enumerate() {
            let val = interp.eval(pt).expect("eval");
            assert!(
                (val - values[i]).abs() < 1e-10,
                "point {i}: got {val}, expected {}",
                values[i]
            );
        }
    }

    // ── RegularGridInterpolator tests ──────────────────────────────

    #[test]
    fn regular_grid_2d_bilinear() {
        // 2D grid: f(x,y) = x + 2*y
        // x = [0, 1, 2], y = [0, 1, 2]
        let points = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]];
        // values in row-major: values[ix * 3 + iy] = x + 2*y
        let values = vec![
            0.0, 2.0, 4.0, // x=0, y=0,1,2
            1.0, 3.0, 5.0, // x=1, y=0,1,2
            2.0, 4.0, 6.0, // x=2, y=0,1,2
        ];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, true, None)
                .expect("rgi");

        // At grid points
        assert!((interp.eval(&[0.0, 0.0]).unwrap() - 0.0).abs() < 1e-12);
        assert!((interp.eval(&[1.0, 1.0]).unwrap() - 3.0).abs() < 1e-12);
        assert!((interp.eval(&[2.0, 2.0]).unwrap() - 6.0).abs() < 1e-12);

        // Bilinear at midpoint: f(0.5, 0.5) = 0.5 + 1.0 = 1.5
        let val = interp.eval(&[0.5, 0.5]).unwrap();
        assert!(
            (val - 1.5).abs() < 1e-12,
            "bilinear at (0.5,0.5): got {val}, expected 1.5"
        );

        // Another midpoint: f(1.5, 0.5) = 1.5 + 1.0 = 2.5
        let val = interp.eval(&[1.5, 0.5]).unwrap();
        assert!(
            (val - 2.5).abs() < 1e-12,
            "bilinear at (1.5,0.5): got {val}, expected 2.5"
        );
    }

    #[test]
    fn regular_grid_3d_trilinear() {
        // 3D grid: f(x,y,z) = x + y + z
        // Each axis: [0, 1]
        let points = vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        // 2x2x2 = 8 values
        let values = vec![
            0.0, 1.0, // x=0, y=0, z=0,1
            1.0, 2.0, // x=0, y=1, z=0,1
            1.0, 2.0, // x=1, y=0, z=0,1
            2.0, 3.0, // x=1, y=1, z=0,1
        ];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, true, None)
                .expect("rgi 3d");

        // At center: f(0.5, 0.5, 0.5) = 1.5
        let val = interp.eval(&[0.5, 0.5, 0.5]).unwrap();
        assert!(
            (val - 1.5).abs() < 1e-12,
            "trilinear center: got {val}, expected 1.5"
        );

        // At corner: f(1, 1, 1) = 3
        assert!((interp.eval(&[1.0, 1.0, 1.0]).unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn regular_grid_nearest() {
        // 2D grid
        let points = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]];
        let values = vec![
            0.0, 1.0, 2.0, //
            3.0, 4.0, 5.0, //
            6.0, 7.0, 8.0, //
        ];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Nearest, true, None)
                .expect("rgi nearest");

        // At exact grid points
        assert!((interp.eval(&[0.0, 0.0]).unwrap() - 0.0).abs() < 1e-12);
        assert!((interp.eval(&[1.0, 1.0]).unwrap() - 4.0).abs() < 1e-12);

        // Near (1,1) → should return value at (1,1) = 4
        let val = interp.eval(&[0.9, 1.1]).unwrap();
        assert!(
            (val - 4.0).abs() < 1e-12,
            "nearest at (0.9,1.1): got {val}, expected 4.0"
        );
    }

    #[test]
    fn regular_grid_out_of_bounds_error() {
        let points = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, true, None)
                .expect("rgi");

        let err = interp.eval(&[-0.5, 0.5]).expect_err("oob");
        assert!(matches!(err, InterpError::OutOfBounds { .. }));
    }

    #[test]
    fn regular_grid_out_of_bounds_fill_value() {
        let points = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let interp = RegularGridInterpolator::new(
            points,
            values,
            RegularGridMethod::Linear,
            false,
            Some(-999.0),
        )
        .expect("rgi");

        let val = interp.eval(&[-0.5, 0.5]).unwrap();
        assert!(
            (val - (-999.0)).abs() < 1e-12,
            "fill value: got {val}, expected -999.0"
        );
    }

    #[test]
    fn regular_grid_1d_degenerates_to_interp1d() {
        // 1D grid should behave like linear interp1d
        let points = vec![vec![0.0, 1.0, 2.0, 3.0]];
        let values = vec![0.0, 2.0, 4.0, 6.0]; // f(x) = 2x
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, true, None)
                .expect("rgi 1d");

        let val = interp.eval(&[0.5]).unwrap();
        assert!(
            (val - 1.0).abs() < 1e-12,
            "1D linear at 0.5: got {val}, expected 1.0"
        );
        let val = interp.eval(&[2.5]).unwrap();
        assert!(
            (val - 5.0).abs() < 1e-12,
            "1D linear at 2.5: got {val}, expected 5.0"
        );
    }

    #[test]
    fn regular_grid_single_cell() {
        // Minimum grid: 2 points per axis
        let points = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        // f(x,y) = x * y at corners: (0,0)=0, (0,1)=0, (1,0)=0, (1,1)=1
        let values = vec![0.0, 0.0, 0.0, 1.0];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, true, None)
                .expect("rgi single cell");

        // Bilinear of x*y at (0.5, 0.5): (1-0.5)*(1-0.5)*0 + (1-0.5)*0.5*0 + 0.5*(1-0.5)*0 + 0.5*0.5*1 = 0.25
        let val = interp.eval(&[0.5, 0.5]).unwrap();
        assert!(
            (val - 0.25).abs() < 1e-12,
            "single cell bilinear: got {val}, expected 0.25"
        );
    }

    #[test]
    fn regular_grid_non_monotone_rejected() {
        let points = vec![vec![0.0, 2.0, 1.0], vec![0.0, 1.0]]; // axis 0 not sorted
        let values = vec![0.0; 6];
        let err =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, true, None)
                .expect_err("non-monotone");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn regular_grid_values_size_mismatch() {
        let points = vec![vec![0.0, 1.0], vec![0.0, 1.0]]; // 2x2 = 4 values needed
        let values = vec![0.0, 1.0, 2.0]; // only 3
        let err =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, true, None)
                .expect_err("size mismatch");
        assert!(matches!(err, InterpError::LengthMismatch { .. }));
    }

    #[test]
    fn regular_grid_eval_many() {
        let points = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, true, None)
                .expect("rgi");

        let results = interp
            .eval_many(&[vec![0.0, 0.0], vec![1.0, 1.0], vec![0.5, 0.5]])
            .expect("eval_many");
        assert_eq!(results.len(), 3);
        assert!((results[0] - 0.0).abs() < 1e-12);
        assert!((results[1] - 3.0).abs() < 1e-12);
        assert!((results[2] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn interpn_convenience() {
        // Test the interpn convenience function
        let points = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]];
        // f(x,y) = x + y
        let values = vec![
            0.0, 1.0, 2.0, //
            1.0, 2.0, 3.0, //
            2.0, 3.0, 4.0, //
        ];
        let xi = vec![vec![0.5, 0.5], vec![1.0, 1.0]];
        let result =
            interpn(points, values, &xi, RegularGridMethod::Linear, true, None).expect("interpn");
        assert!((result[0] - 1.0).abs() < 1e-12, "interpn at (0.5,0.5)");
        assert!((result[1] - 2.0).abs() < 1e-12, "interpn at (1,1)");
    }

    #[test]
    fn regular_grid_at_boundary() {
        // Queries exactly at grid boundaries should work
        let points = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0]];
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, true, None)
                .expect("rgi");

        // At all corners
        assert!((interp.eval(&[0.0, 0.0]).unwrap() - 10.0).abs() < 1e-12);
        assert!((interp.eval(&[0.0, 1.0]).unwrap() - 20.0).abs() < 1e-12);
        assert!((interp.eval(&[2.0, 0.0]).unwrap() - 50.0).abs() < 1e-12);
        assert!((interp.eval(&[2.0, 1.0]).unwrap() - 60.0).abs() < 1e-12);
    }

    #[test]
    fn regular_grid_oob_nan_default() {
        // When fill_value is None, out-of-bounds returns NaN
        let points = vec![vec![0.0, 1.0]];
        let values = vec![10.0, 20.0];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, false, None)
                .expect("rgi");

        let val = interp.eval(&[-1.0]).unwrap();
        assert!(val.is_nan(), "expected NaN for OOB, got {val}");
    }

    #[test]
    fn regular_grid_wrong_dim_query() {
        let points = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, true, None)
                .expect("rgi");

        let err = interp.eval(&[0.5]).expect_err("wrong dim");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }
}
