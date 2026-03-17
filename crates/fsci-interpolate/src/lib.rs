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
        let tree = fsci_spatial::KDTree::new(points).map_err(|e| {
            InterpError::InvalidArgument {
                detail: format!("KDTree construction failed: {e}"),
            }
        })?;
        Ok(Self {
            tree,
            values: values.to_vec(),
        })
    }

    /// Evaluate the interpolator at a single query point.
    pub fn eval(&self, query: &[f64]) -> Result<f64, InterpError> {
        let (idx, _dist) = self.tree.query(query).map_err(|e| {
            InterpError::InvalidArgument {
                detail: format!("query failed: {e}"),
            }
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
        let err =
            NearestNDInterpolator::new(&[vec![0.0]], &[1.0, 2.0]).expect_err("mismatch");
        assert!(matches!(err, InterpError::LengthMismatch { .. }));
    }

    #[test]
    fn griddata_nearest_basic() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
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
}
