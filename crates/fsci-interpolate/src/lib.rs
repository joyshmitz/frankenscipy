#![forbid(unsafe_code)]

//! Interpolation routines for FrankenSciPy.
//!
//! Matches `scipy.interpolate` core functions:
//! - `interp1d` — 1D interpolation (linear, nearest, cubic)
//! - `CubicSpline` — natural/clamped/not-a-knot cubic spline
//! - `BSpline` — B-spline representation and evaluation
//! - `Akima1DInterpolator` — Akima piecewise cubic Hermite interpolator
//! - `PchipInterpolator` — Monotonicity-preserving cubic Hermite interpolator
//! - `RegularGridInterpolator` — N-D interpolation on regular grids
//! - `NearestNDInterpolator` — Nearest-neighbor for scattered N-D data
//! - `LinearNDInterpolator` — Linear interpolation for scattered 2D data

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
        if x_new.is_nan() {
            return Ok(f64::NAN);
        }
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

        let i = find_interval_helper(&self.x, x_new);
        self.eval_at_interval(i, x_new)
    }

    /// Evaluate at multiple points.
    ///
    /// Optimized for sorted `x_new` to achieve O(N+M) performance.
    pub fn eval_many(&self, x_new: &[f64]) -> Result<Vec<f64>, InterpError> {
        if x_new.is_empty() {
            return Ok(Vec::new());
        }

        // Check if x_new is sorted to enable linear sweep optimization
        let is_sorted = x_new.windows(2).all(|w| w[0] <= w[1]);

        if is_sorted {
            let mut results = Vec::with_capacity(x_new.len());
            let mut i = 0;
            let n = self.x.len();
            for &xi in x_new {
                // Bounds checking
                if xi < self.x[0] || xi > self.x[n - 1] {
                    if self.options.bounds_error {
                        return Err(InterpError::OutOfBounds {
                            value: format!("{xi}"),
                        });
                    }
                    results.push(self.options.fill_value.unwrap_or(f64::NAN));
                    continue;
                }

                // Advance i while xi is beyond the current interval
                while i < n - 2 && xi >= self.x[i + 1] {
                    i += 1;
                }
                results.push(self.eval_at_interval(i, xi)?);
            }
            Ok(results)
        } else {
            x_new.iter().map(|&xi| self.eval(xi)).collect()
        }
    }

    fn eval_at_interval(&self, i: usize, x_new: f64) -> Result<f64, InterpError> {
        match self.options.kind {
            InterpKind::Linear => {
                let t = (x_new - self.x[i]) / (self.x[i + 1] - self.x[i]);
                Ok(self.y[i] + t * (self.y[i + 1] - self.y[i]))
            }
            InterpKind::Nearest => {
                let mid = 0.5 * (self.x[i] + self.x[i + 1]);
                Ok(if x_new <= mid {
                    self.y[i]
                } else {
                    self.y[i + 1]
                })
            }
            InterpKind::CubicSpline => {
                let coeffs = self.spline_coeffs.as_ref().unwrap();
                let dx = x_new - self.x[i];
                let [a, b, c, d] = coeffs[i];
                Ok(a + dx * (b + dx * (c + dx * d)))
            }
        }
    }
}

/// Binary search to find the interval containing x_new in a sorted array.
/// Returns index i such that array[i] <= x_new < array[i+1].
fn find_interval_helper(array: &[f64], x_new: f64) -> usize {
    let n = array.len();
    if x_new <= array[0] {
        return 0;
    }
    if x_new >= array[n - 1] {
        return n - 2;
    }
    // Binary search
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if array[mid] <= x_new {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Compute cubic spline coefficients with configurable boundary conditions.
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

    let c = match bc {
        SplineBc::Natural => solve_spline_natural(n, &h, y),
        SplineBc::NotAKnot => solve_spline_not_a_knot(n, &h, y),
        SplineBc::Clamped(dl, dr) => solve_spline_clamped(n, &h, y, dl, dr),
    };

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

fn solve_spline_not_a_knot(n: usize, h: &[f64], y: &[f64]) -> Vec<f64> {
    let inner = n - 2;
    let r01 = h[0] / h[1];

    let mut diag = vec![0.0; inner];
    let mut sub = vec![0.0; inner];
    let mut sup = vec![0.0; inner];
    let mut rhs = vec![0.0; inner];

    for idx in 0..inner {
        let i = idx + 1;
        sub[idx] = h[i - 1];
        diag[idx] = 2.0 * (h[i - 1] + h[i]);
        sup[idx] = h[i];
        rhs[idx] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    diag[0] = h[0] * (1.0 + r01) + 2.0 * (h[0] + h[1]);
    sup[0] = h[1] - h[0] * r01;

    let rn = h[n - 2] / h[n - 3];
    let last = inner - 1;
    sub[last] = h[n - 3] - h[n - 2] * rn;
    diag[last] = 2.0 * (h[n - 3] + h[n - 2]) + h[n - 2] * (1.0 + rn);

    thomas_solve(&sub, &mut diag, &sup, &mut rhs);

    let mut c = vec![0.0; n];
    for (idx, &ci) in rhs.iter().enumerate() {
        c[idx + 1] = ci;
    }
    c[0] = (1.0 + r01) * c[1] - r01 * c[2];
    c[n - 1] = (1.0 + rn) * c[n - 2] - rn * c[n - 3];

    c
}

fn solve_spline_clamped(
    n: usize,
    h: &[f64],
    y: &[f64],
    deriv_left: f64,
    deriv_right: f64,
) -> Vec<f64> {
    let mut diag = vec![0.0; n];
    let mut sub = vec![0.0; n];
    let mut sup = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    diag[0] = 2.0 * h[0];
    sup[0] = h[0];
    rhs[0] = 3.0 * ((y[1] - y[0]) / h[0] - deriv_left);

    for i in 1..n - 1 {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    let m = n - 1;
    sub[m] = h[m - 1];
    diag[m] = 2.0 * h[m - 1];
    rhs[m] = 3.0 * (deriv_right - (y[m] - y[m - 1]) / h[m - 1]);

    thomas_solve(&sub, &mut diag, &sup, &mut rhs);
    rhs
}

fn thomas_solve(sub: &[f64], diag: &mut [f64], sup: &[f64], rhs: &mut [f64]) {
    let n = diag.len();
    if n == 0 { return; }
    for i in 1..n {
        if diag[i - 1].abs() < 1e-18 { continue; }
        let w = sub[i] / diag[i - 1];
        diag[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    if diag[n - 1].abs() > 1e-18 {
        rhs[n - 1] /= diag[n - 1];
    }
    for i in (0..n - 1).rev() {
        if diag[i].abs() > 1e-18 {
            rhs[i] = (rhs[i] - sup[i] * rhs[i + 1]) / diag[i];
        }
    }
}

/// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolator.
#[derive(Debug)]
pub struct PchipInterpolator {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}

impl PchipInterpolator {
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self, InterpError> {
        if x.len() != y.len() {
            return Err(InterpError::LengthMismatch { x_len: x.len(), y_len: y.len() });
        }
        if x.len() < 2 {
            return Err(InterpError::TooFewPoints { minimum: 2, actual: x.len() });
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InterpError::UnsortedX);
        }

        let n = x.len();
        let m = n - 1;
        let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();
        let delta: Vec<f64> = (0..m).map(|i| (y[i + 1] - y[i]) / h[i]).collect();

        let mut d = vec![0.0; n];
        if n == 2 {
            d[0] = delta[0];
            d[1] = delta[0];
        } else {
            for i in 1..m {
                if delta[i - 1].signum() != delta[i].signum() || delta[i - 1] == 0.0 || delta[i] == 0.0 {
                    d[i] = 0.0;
                } else {
                    let w1 = 2.0 * h[i] + h[i - 1];
                    let w2 = h[i] + 2.0 * h[i - 1];
                    d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
                }
            }
            d[0] = pchip_end_slope(&h, &delta, true);
            d[n - 1] = pchip_end_slope(&h, &delta, false);
        }

        let mut coeffs = Vec::with_capacity(m);
        for i in 0..m {
            let hi = h[i];
            coeffs.push([
                y[i],
                d[i],
                (3.0 * delta[i] - 2.0 * d[i] - d[i + 1]) / hi,
                (d[i] + d[i + 1] - 2.0 * delta[i]) / (hi * hi),
            ]);
        }

        Ok(Self { x: x.to_vec(), coeffs })
    }

    pub fn eval(&self, x_new: f64) -> f64 {
        if x_new.is_nan() { return f64::NAN; }
        let n = self.x.len();
        let i = if x_new <= self.x[0] { 0 }
                else if x_new >= self.x[n - 1] { n - 2 }
                else { find_interval_helper(&self.x, x_new) };
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    pub fn eval_many(&self, x_new: &[f64]) -> Vec<f64> {
        x_new.iter().map(|&xi| self.eval(xi)).collect()
    }
}

fn pchip_end_slope(h: &[f64], delta: &[f64], is_left: bool) -> f64 {
    let m = delta.len();
    if m < 2 { return delta[0]; }
    let (d1, d2, h1, h2) = if is_left { (delta[0], delta[1], h[0], h[1]) }
                           else { (delta[m - 1], delta[m - 2], h[m - 1], h[m - 2]) };
    let mut d = ((2.0 * h1 + h2) * d1 - h1 * d2) / (h1 + h2);
    if d.signum() != d1.signum() { d = 0.0; }
    else if d1.signum() != d2.signum() && d.abs() > 3.0 * d1.abs() { d = 3.0 * d1; }
    d
}

pub fn interp1d_linear(x: &[f64], y: &[f64], x_new: &[f64]) -> Result<Vec<f64>, InterpError> {
    let interp = Interp1d::new(x, y, Interp1dOptions { bounds_error: false, ..Default::default() })?;
    interp.eval_many(x_new)
}

/// Standalone cubic spline interpolator.
#[derive(Debug, Clone)]
pub struct CubicSplineStandalone {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}

impl CubicSplineStandalone {
    pub fn new(x: &[f64], y: &[f64], bc: SplineBc) -> Result<Self, InterpError> {
        if x.len() != y.len() { return Err(InterpError::LengthMismatch { x_len: x.len(), y_len: y.len() }); }
        if x.len() < 4 { return Err(InterpError::TooFewPoints { minimum: 4, actual: x.len() }); }
        if x.windows(2).any(|w| w[1] <= w[0]) { return Err(InterpError::UnsortedX); }
        let coeffs = compute_cubic_spline(x, y, bc)?;
        Ok(Self { x: x.to_vec(), coeffs })
    }

    pub fn eval(&self, x_new: f64) -> f64 {
        if x_new.is_nan() { return f64::NAN; }
        let n = self.x.len();
        let i = if x_new <= self.x[0] { 0 }
                else if x_new >= self.x[n - 1] { n - 2 }
                else { find_interval_helper(&self.x, x_new) };
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    pub fn eval_many(&self, x_new: &[f64]) -> Vec<f64> {
        x_new.iter().map(|&xi| self.eval(xi)).collect()
    }

    pub fn derivative(&self, nu: usize) -> CubicSplineDerivative {
        let m = self.coeffs.len();
        match nu {
            0 => CubicSplineDerivative { x: self.x.clone(), coeffs: self.coeffs.clone() },
            1 => {
                let dc: Vec<[f64; 4]> = self.coeffs.iter().map(|&[_a, b, c, d]| [b, 2.0 * c, 3.0 * d, 0.0]).collect();
                CubicSplineDerivative { x: self.x.clone(), coeffs: dc }
            }
            2 => {
                let dc: Vec<[f64; 4]> = self.coeffs.iter().map(|&[_a, _b, c, d]| [2.0 * c, 6.0 * d, 0.0, 0.0]).collect();
                CubicSplineDerivative { x: self.x.clone(), coeffs: dc }
            }
            _ => CubicSplineDerivative { x: self.x.clone(), coeffs: vec![[0.0; 4]; m] },
        }
    }

    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        if (b - a).abs() < 1e-15 { return 0.0; }
        let sign = if a > b { -1.0 } else { 1.0 };
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let n = self.x.len();
        let mut total = 0.0;
        for i in 0..n - 1 {
            let seg_lo = self.x[i].max(lo);
            let seg_hi = self.x[i + 1].min(hi);
            if seg_lo >= seg_hi { continue; }
            let [a0, b0, c0, d0] = self.coeffs[i];
            let dx_lo = seg_lo - self.x[i];
            let dx_hi = seg_hi - self.x[i];
            let anti = |dx: f64| a0 * dx + b0 * dx * dx / 2.0 + c0 * dx.powi(3) / 3.0 + d0 * dx.powi(4) / 4.0;
            total += anti(dx_hi) - anti(dx_lo);
        }
        sign * total
    }
}

#[derive(Debug, Clone)]
pub struct CubicSplineDerivative {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}

impl CubicSplineDerivative {
    pub fn eval(&self, x_new: f64) -> f64 {
        if x_new.is_nan() { return f64::NAN; }
        let n = self.x.len();
        let i = if x_new <= self.x[0] { 0 }
                else if x_new >= self.x[n - 1] { n - 2 }
                else { find_interval_helper(&self.x, x_new) };
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }
}

/// Akima (1970) piecewise cubic Hermite interpolator.
#[derive(Debug, Clone)]
pub struct Akima1DInterpolator {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}

impl Akima1DInterpolator {
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self, InterpError> {
        if x.len() != y.len() { return Err(InterpError::LengthMismatch { x_len: x.len(), y_len: y.len() }); }
        if x.len() < 2 { return Err(InterpError::TooFewPoints { minimum: 2, actual: x.len() }); }
        if x.windows(2).any(|w| w[1] <= w[0]) { return Err(InterpError::UnsortedX); }
        let n = x.len();
        let m = n - 1;
        let delta: Vec<f64> = (0..m).map(|i| (y[i + 1] - y[i]) / (x[i + 1] - x[i])).collect();
        let slopes = akima_slopes(&delta);
        let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();
        let mut coeffs = Vec::with_capacity(m);
        for i in 0..m {
            let hi = h[i];
            coeffs.push([
                y[i],
                slopes[i],
                (3.0 * delta[i] - 2.0 * slopes[i] - slopes[i + 1]) / hi,
                (slopes[i] + slopes[i + 1] - 2.0 * delta[i]) / (hi * hi),
            ]);
        }
        Ok(Self { x: x.to_vec(), coeffs })
    }

    pub fn eval(&self, x_new: f64) -> f64 {
        let i = find_interval_helper(&self.x, x_new);
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    pub fn eval_many(&self, x_new: &[f64]) -> Vec<f64> {
        x_new.iter().map(|&xi| self.eval(xi)).collect()
    }
}

fn akima_slopes(delta: &[f64]) -> Vec<f64> {
    let m = delta.len();
    let n = m + 1;
    let mut slopes = vec![0.0; n];
    if m == 0 { return slopes; }
    if m == 1 { slopes[0] = delta[0]; slopes[1] = delta[0]; return slopes; }
    let mut d = Vec::with_capacity(m + 4);
    let d0 = delta[0];
    let d1 = delta.get(1).copied().unwrap_or(d0);
    d.push(3.0 * d0 - 2.0 * d1); d.push(2.0 * d0 - d1);
    d.extend_from_slice(delta);
    let d_last = delta[m - 1];
    let d_prev = delta.get(m.wrapping_sub(2)).copied().unwrap_or(d_last);
    d.push(2.0 * d_last - d_prev); d.push(3.0 * d_last - 2.0 * d_prev);
    for j in 0..n {
        let w1 = (d[j + 3] - d[j + 2]).abs();
        let w2 = (d[j + 1] - d[j]).abs();
        slopes[j] = if w1 + w2 < 1e-30 { 0.5 * (d[j + 1] + d[j + 2]) }
                    else { (w1 * d[j + 1] + w2 * d[j + 2]) / (w1 + w2) };
    }
    slopes
}

#[derive(Debug, Clone)]
pub struct BSpline {
    t: Vec<f64>,
    c: Vec<f64>,
    k: usize,
    pub extrapolate: bool,
}

impl BSpline {
    pub fn new(t: Vec<f64>, c: Vec<f64>, k: usize) -> Result<Self, InterpError> {
        let expected_knots = c.len() + k + 1;
        if t.len() != expected_knots {
            return Err(InterpError::InvalidArgument { detail: format!("knot length {} != c.len()+k+1={}", t.len(), expected_knots) });
        }
        if t.windows(2).any(|w| w[1] < w[0]) {
            return Err(InterpError::InvalidArgument { detail: "knots must be non-decreasing".to_string() });
        }
        Ok(Self { t, c, k, extrapolate: true })
    }

    pub fn knots(&self) -> &[f64] { &self.t }
    pub fn coeffs(&self) -> &[f64] { &self.c }
    pub fn degree(&self) -> usize { self.k }

    pub fn eval(&self, x: f64) -> f64 {
        let mut d = vec![0.0; self.k + 1];
        self.eval_into(x, &mut d)
    }

    pub fn eval_into(&self, x: f64, d: &mut [f64]) -> f64 {
        let n = self.c.len();
        let k = self.k;
        let t = &self.t;
        if !self.extrapolate && (x < t[k] || x > t[n]) { return f64::NAN; }
        let mu = self.find_span(x);
        for (j, value) in d.iter_mut().enumerate().take(k + 1) {
            let idx = mu.wrapping_sub(k) + j;
            *value = if idx < n { self.c[idx] } else { 0.0 };
        }
        for r in 1..=k {
            for j in (r..=k).rev() {
                let left = mu.wrapping_sub(k) + j;
                let right = left + k + 1 - r;
                if right < t.len() {
                    let denom = t[right] - t[left];
                    if denom > 0.0 {
                        let alpha = (x - t[left]) / denom;
                        d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
                    } else { d[j] = d[j - 1]; }
                }
            }
        }
        d[k]
    }

    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        if xs.is_empty() { return Vec::new(); }
        let mut d = vec![0.0; self.k + 1];
        let is_sorted = xs.windows(2).all(|w| w[0] <= w[1]);
        if is_sorted {
            let mut results = Vec::with_capacity(xs.len());
            let mut mu = self.k;
            let n = self.c.len();
            let t = &self.t;
            for &x in xs {
                while mu < n - 1 && x >= t[mu + 1] { mu += 1; }
                results.push(self.eval_into_with_span(x, mu, &mut d));
            }
            results
        } else {
            xs.iter().map(|&x| self.eval_into(x, &mut d)).collect()
        }
    }

    fn eval_into_with_span(&self, x: f64, mu: usize, d: &mut [f64]) -> f64 {
        let n = self.c.len();
        let k = self.k;
        let t = &self.t;
        if !self.extrapolate && (x < t[k] || x > t[n]) { return f64::NAN; }
        for (j, value) in d.iter_mut().enumerate().take(k + 1) {
            let idx = mu.wrapping_sub(k) + j;
            *value = if idx < n { self.c[idx] } else { 0.0 };
        }
        for r in 1..=k {
            for j in (r..=k).rev() {
                let left = mu.wrapping_sub(k) + j;
                let right = left + k + 1 - r;
                if right < t.len() {
                    let denom = t[right] - t[left];
                    if denom > 0.0 {
                        let alpha = (x - t[left]) / denom;
                        d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
                    } else { d[j] = d[j - 1]; }
                }
            }
        }
        d[k]
    }

    pub fn derivative(&self, nu: usize) -> Result<Self, InterpError> {
        if nu == 0 { return Ok(self.clone()); }
        let mut t = self.t.clone();
        let mut c = self.c.clone();
        let mut k = self.k;
        for _ in 0..nu {
            if k == 0 { return Err(InterpError::InvalidArgument { detail: "cannot differentiate degree-0 spline".to_string() }); }
            let n = c.len();
            let mut dc = Vec::with_capacity(n - 1);
            for i in 0..n - 1 {
                let denom = t[i + k + 1] - t[i + 1];
                dc.push(if denom > 0.0 { k as f64 * (c[i + 1] - c[i]) / denom } else { 0.0 });
            }
            t.remove(t.len() - 1); t.remove(0);
            c = dc; k -= 1;
        }
        Self::new(t, c, k)
    }

    pub fn antiderivative(&self, nu: usize) -> Result<Self, InterpError> {
        if nu == 0 { return Ok(self.clone()); }
        let mut t = self.t.clone();
        let mut c = self.c.clone();
        let mut k = self.k;
        for _ in 0..nu {
            let n = c.len();
            t.insert(0, t[0]); t.push(*t.last().unwrap());
            k += 1;
            let mut new_c = vec![0.0; n + 1];
            for i in 0..n {
                let denom = t[i + k + 1] - t[i + 1];
                new_c[i + 1] = new_c[i] + c[i] * denom / k as f64;
            }
            while new_c.len() + k + 1 < t.len() { new_c.push(*new_c.last().unwrap()); }
            c = new_c;
        }
        Self::new(t, c, k)
    }

    pub fn integrate(&self, a: f64, b: f64) -> Result<f64, InterpError> {
        let anti = self.antiderivative(1)?;
        Ok(anti.eval(b) - anti.eval(a))
    }

    fn find_span(&self, x: f64) -> usize {
        let n = self.c.len();
        let k = self.k;
        let t = &self.t;
        if x <= t[k] { return k; }
        if x >= t[n] {
            let mut mu = n - 1;
            while mu > k && t[mu] == t[n] { mu -= 1; }
            return mu;
        }
        let mut lo = k;
        let mut hi = n;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if t[mid] <= x { lo = mid; } else { hi = mid; }
        }
        lo
    }
}

pub fn make_interp_spline(x: &[f64], y: &[f64], k: usize) -> Result<BSpline, InterpError> {
    let n = x.len();
    if n != y.len() { return Err(InterpError::LengthMismatch { x_len: n, y_len: y.len() }); }
    if n < k + 1 { return Err(InterpError::TooFewPoints { minimum: k + 1, actual: n }); }
    if x.windows(2).any(|w| w[1] <= w[0]) { return Err(InterpError::UnsortedX); }
    let num_knots = n + k + 1;
    let num_interior = n - k - 1;
    let mut t = Vec::with_capacity(num_knots);
    for _ in 0..=k { t.push(x[0]); }
    for i in 0..num_interior { t.push(x[i + 1 + (k - 1) / 2]); }
    for _ in 0..=k { t.push(x[n - 1]); }
    let mut a_mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        let basis = eval_basis_all(&t, x[i], k, n);
        a_mat[i][..n].copy_from_slice(&basis[..n]);
    }
    let mut rhs = y.to_vec();
    let c = solve_dense_system(&mut a_mat, &mut rhs)?;
    BSpline::new(t, c, k)
}

pub fn make_lsq_spline(x: &[f64], y: &[f64], t: &[f64], k: usize) -> Result<BSpline, InterpError> {
    let m = x.len();
    if m != y.len() { return Err(InterpError::LengthMismatch { x_len: m, y_len: y.len() }); }
    let n = t.len() - k - 1;
    if n == 0 || n > m { return Err(InterpError::InvalidArgument { detail: format!("need t.len()-k-1 coeffs ({n}) <= points ({m})") }); }
    let mut ata = vec![vec![0.0; n]; n];
    let mut aty = vec![0.0; n];
    for i in 0..m {
        let basis = eval_basis_all(t, x[i], k, n);
        for j in 0..n {
            aty[j] += basis[j] * y[i];
            for l in 0..n { ata[j][l] += basis[j] * basis[l]; }
        }
    }
    let c = solve_dense_system(&mut ata, &mut aty)?;
    BSpline::new(t.to_vec(), c, k)
}

fn eval_basis_all(t: &[f64], x: f64, k: usize, n: usize) -> Vec<f64> {
    let mut basis = vec![0.0; n];
    for i in 0..n {
        if i + 1 < t.len() {
            basis[i] = if (t[i] <= x && x < t[i + 1]) || (x == t[i + 1] && i + 1 == t.len() - k - 1) { 1.0 } else { 0.0 };
        }
    }
    for p in 1..=k {
        let prev = basis.clone();
        for i in 0..n {
            let mut val = 0.0;
            if i + p < t.len() {
                let denom_left = t[i + p] - t[i];
                if denom_left > 0.0 { val += (x - t[i]) / denom_left * prev[i]; }
            }
            if i + p + 1 < t.len() && i + 1 < n {
                let denom_right = t[i + p + 1] - t[i + 1];
                if denom_right > 0.0 { val += (t[i + p + 1] - x) / denom_right * prev[i + 1]; }
            }
            basis[i] = val;
        }
    }
    basis
}

fn solve_dense_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<Vec<f64>, InterpError> {
    let n = b.len();
    if n == 0 || a.len() != n { return Err(InterpError::InvalidArgument { detail: "empty or mismatched system".to_string() }); }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for (row, a_row) in a.iter().enumerate().skip(col + 1) {
            if a_row[col].abs() > max_val { max_val = a_row[col].abs(); max_row = row; }
        }
        if max_val < 1e-14 { return Err(InterpError::InvalidArgument { detail: "singular matrix".to_string() }); }
        if max_row != col { a.swap(col, max_row); b.swap(col, max_row); }
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            let pivot_row = a[col].clone();
            for (j, pval) in pivot_row.iter().enumerate().skip(col) { a[row][j] -= factor * pval; }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n { s -= a[i][j] * x[j]; }
        x[i] = s / a[i][i];
    }
    Ok(x)
}

#[derive(Debug)]
pub struct NearestNDInterpolator {
    tree: fsci_spatial::KDTree,
    values: Vec<f64>,
}

impl NearestNDInterpolator {
    pub fn new(points: &[Vec<f64>], values: &[f64]) -> Result<Self, InterpError> {
        if points.is_empty() { return Err(InterpError::TooFewPoints { minimum: 1, actual: 0 }); }
        if points.len() != values.len() { return Err(InterpError::LengthMismatch { x_len: points.len(), y_len: values.len() }); }
        let tree = fsci_spatial::KDTree::new(points).map_err(|e| InterpError::InvalidArgument { detail: format!("KDTree error: {e}") })?;
        Ok(Self { tree, values: values.to_vec() })
    }
    pub fn eval(&self, query: &[f64]) -> Result<f64, InterpError> {
        let (idx, _dist) = self.tree.query(query).map_err(|e| InterpError::InvalidArgument { detail: format!("query error: {e}") })?;
        Ok(self.values[idx])
    }
    pub fn eval_many(&self, queries: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> {
        queries.iter().map(|q| self.eval(q)).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GriddataMethod { Nearest, Linear }

pub fn griddata(points: &[Vec<f64>], values: &[f64], xi: &[Vec<f64>], method: GriddataMethod) -> Result<Vec<f64>, InterpError> {
    match method {
        GriddataMethod::Nearest => NearestNDInterpolator::new(points, values)?.eval_many(xi),
        GriddataMethod::Linear => LinearNDInterpolator::new(points, values)?.eval_many(xi),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegularGridMethod { #[default] Linear, Nearest }

#[derive(Debug, Clone)]
pub struct RegularGridInterpolator {
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
    strides: Vec<usize>,
    method: RegularGridMethod,
    bounds_error: bool,
    fill_value: Option<f64>,
}

impl RegularGridInterpolator {
    pub fn new(points: Vec<Vec<f64>>, values: Vec<f64>, method: RegularGridMethod, bounds_error: bool, fill_value: Option<f64>) -> Result<Self, InterpError> {
        if points.is_empty() { return Err(InterpError::InvalidArgument { detail: "points empty".to_string() }); }
        for (dim, axis) in points.iter().enumerate() {
            if axis.len() < 2 { return Err(InterpError::TooFewPoints { minimum: 2, actual: axis.len() }); }
            if axis.windows(2).any(|w| w[1] <= w[0]) { return Err(InterpError::InvalidArgument { detail: format!("axis {dim} not strictly increasing") }); }
        }
        let ndim = points.len();
        let mut strides = vec![0usize; ndim];
        let mut total_size: usize = 1;
        for i in (0..ndim).rev() {
            strides[i] = total_size;
            total_size = total_size.checked_mul(points[i].len()).ok_or_else(|| InterpError::InvalidArgument { detail: "grid overflow".to_string() })?;
        }
        if values.len() != total_size { return Err(InterpError::LengthMismatch { x_len: total_size, y_len: values.len() }); }
        Ok(Self { points, values, strides, method, bounds_error, fill_value })
    }

    pub fn ndim(&self) -> usize { self.points.len() }

    pub fn eval(&self, xi: &[f64]) -> Result<f64, InterpError> {
        let ndim = self.ndim();
        if xi.len() != ndim { return Err(InterpError::InvalidArgument { detail: format!("expected {ndim}D, got {}D", xi.len()) }); }
        for (dim, &x) in xi.iter().enumerate() {
            let axis = &self.points[dim];
            if x < axis[0] || x > axis[axis.len() - 1] {
                if self.bounds_error { return Err(InterpError::OutOfBounds { value: format!("dim {dim}: {x} outside [{}, {}]", axis[0], axis[axis.len()-1]) }); }
                return Ok(self.fill_value.unwrap_or(f64::NAN));
            }
        }
        match self.method {
            RegularGridMethod::Linear => self.eval_linear(xi),
            RegularGridMethod::Nearest => Ok(self.eval_nearest(xi)),
        }
    }

    pub fn eval_many(&self, xi: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> { xi.iter().map(|x| self.eval(x)).collect() }

    fn find_interval(axis: &[f64], x: f64) -> usize {
        let n = axis.len();
        if x <= axis[0] { return 0; }
        if x >= axis[n - 1] { return n - 2; }
        match axis.binary_search_by(|probe| probe.total_cmp(&x)) { Ok(i) => i.min(n - 2), Err(i) => i.saturating_sub(1) }
    }

    fn eval_nearest(&self, xi: &[f64]) -> f64 {
        let mut flat_idx = 0;
        for ((axis, &x), &stride) in self.points.iter().zip(xi).zip(&self.strides) {
            let i = Self::find_interval(axis, x);
            let nearest = if i + 1 < axis.len() && (x - axis[i]).abs() > (axis[i + 1] - x).abs() { i + 1 } else { i };
            flat_idx += nearest * stride;
        }
        self.values[flat_idx]
    }

    fn eval_linear(&self, xi: &[f64]) -> Result<f64, InterpError> {
        let ndim = self.ndim();
        let mut indices = Vec::with_capacity(ndim);
        let mut fracs = Vec::with_capacity(ndim);
        for (axis, &x) in self.points.iter().zip(xi) {
            let i = Self::find_interval(axis, x);
            let denom = axis[i + 1] - axis[i];
            indices.push(i);
            fracs.push(if denom == 0.0 { 0.0 } else { (x - axis[i]) / denom });
        }
        let mut result = 0.0;
        for corner in 0..(1usize << ndim) {
            let mut weight = 1.0;
            let mut flat_idx = 0;
            for dim in 0..ndim {
                let bit = (corner >> dim) & 1;
                flat_idx += (indices[dim] + bit) * self.strides[dim];
                weight *= if bit == 0 { 1.0 - fracs[dim] } else { fracs[dim] };
            }
            result += weight * self.values[flat_idx];
        }
        Ok(result)
    }
}

pub fn interpn(points: Vec<Vec<f64>>, values: Vec<f64>, xi: &[Vec<f64>], method: RegularGridMethod, bounds_error: bool, fill_value: Option<f64>) -> Result<Vec<f64>, InterpError> {
    RegularGridInterpolator::new(points, values, method, bounds_error, fill_value)?.eval_many(xi)
}

#[derive(Debug, Clone)]
pub struct Delaunay2D { pub points: Vec<(f64, f64)>, pub simplices: Vec<(usize, usize, usize)> }

impl Delaunay2D {
    pub fn new(points: &[(f64, f64)]) -> Result<Self, InterpError> {
        let n = points.len();
        if n < 3 { return Err(InterpError::TooFewPoints { minimum: 3, actual: n }); }
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &(x, y) in points { min_x = min_x.min(x); min_y = min_y.min(y); max_x = max_x.max(x); max_y = max_y.max(y); }
        let (dx, dy) = ((max_x - min_x).max(1e-10), (max_y - min_y).max(1e-10));
        let margin = 10.0;
        let mut all_points = points.to_vec();
        all_points.push((min_x - margin * dx, min_y - margin * dy));
        all_points.push((max_x + margin * dx, min_y - margin * dy));
        all_points.push(((min_x + max_x) / 2.0, max_y + margin * dy));
        let mut triangles = vec![(n, n + 1, n + 2)];
        for p_idx in 0..n {
            let p = all_points[p_idx];
            let mut bad = Vec::new();
            for (t_idx, &(a, b, c)) in triangles.iter().enumerate() { if in_circumcircle(all_points[a], all_points[b], all_points[c], p) { bad.push(t_idx); } }
            let mut boundary = Vec::new();
            for &t_idx in &bad {
                let (a, b, c) = triangles[t_idx];
                for &(e0, e1) in &[(a, b), (b, c), (c, a)] {
                    if !bad.iter().any(|&o| o != t_idx && triangle_has_edge(triangles[o].0, triangles[o].1, triangles[o].2, e0, e1)) { boundary.push((e0, e1)); }
                }
            }
            bad.sort_unstable(); for &idx in bad.iter().rev() { triangles.swap_remove(idx); }
            for &(e0, e1) in &boundary { triangles.push((p_idx, e0, e1)); }
        }
        Ok(Self { points: points.to_vec(), simplices: triangles.into_iter().filter(|&(a, b, c)| a < n && b < n && c < n).collect() })
    }
    pub fn find_simplex(&self, query: (f64, f64)) -> Option<(usize, f64, f64, f64)> {
        for (idx, &(a, b, c)) in self.simplices.iter().enumerate() {
            let (l1, l2, l3) = barycentric(self.points[a], self.points[b], self.points[c], query);
            if l1 >= -1e-10 && l2 >= -1e-10 && l3 >= -1e-10 { return Some((idx, l1, l2, l3)); }
        }
        None
    }
}

fn in_circumcircle(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool {
    let (ax, ay, bx, by, cx, cy) = (a.0 - d.0, a.1 - d.1, b.0 - d.0, b.1 - d.1, c.0 - d.0, c.1 - d.1);
    let det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by)) - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) + (ax * ax + ay * ay) * (bx * cy - by * cx);
    let orient = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
    if orient > 0.0 { det > 0.0 } else { det < 0.0 }
}

fn triangle_has_edge(a: usize, b: usize, c: usize, e0: usize, e1: usize) -> bool {
    [(a, b), (b, c), (c, a)].iter().any(|&(x, y)| (x == e0 && y == e1) || (x == e1 && y == e0))
}

fn barycentric(a: (f64, f64), b: (f64, f64), c: (f64, f64), p: (f64, f64)) -> (f64, f64, f64) {
    let (v0x, v0y, v1x, v1y, v2x, v2y) = (b.0 - a.0, b.1 - a.1, c.0 - a.0, c.1 - a.1, p.0 - a.0, p.1 - a.1);
    let (d00, d01, d11, d20, d21) = (v0x * v0x + v0y * v0y, v0x * v1x + v0y * v1y, v1x * v1x + v1y * v1y, v2x * v0x + v2y * v0y, v2x * v1x + v2y * v1y);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-30 { return (f64::NAN, f64::NAN, f64::NAN); }
    let l2 = (d11 * d20 - d01 * d21) / denom;
    let l3 = (d00 * d21 - d01 * d20) / denom;
    (1.0 - l2 - l3, l2, l3)
}

#[derive(Debug, Clone)]
pub struct LinearNDInterpolator { delaunay: Delaunay2D, values: Vec<f64> }

impl LinearNDInterpolator {
    pub fn new(points: &[Vec<f64>], values: &[f64]) -> Result<Self, InterpError> {
        if points.is_empty() { return Err(InterpError::TooFewPoints { minimum: 3, actual: 0 }); }
        if points[0].len() != 2 { return Err(InterpError::InvalidArgument { detail: "LinearND only 2D".to_string() }); }
        if points.len() != values.len() { return Err(InterpError::LengthMismatch { x_len: points.len(), y_len: values.len() }); }
        let pts: Vec<(f64, f64)> = points.iter().map(|p| (p[0], p[1])).collect();
        Ok(Self { delaunay: Delaunay2D::new(&pts)?, values: values.to_vec() })
    }
    pub fn eval(&self, query: &[f64]) -> Result<f64, InterpError> {
        if query.len() != 2 { return Err(InterpError::InvalidArgument { detail: "query must be 2D".to_string() }); }
        match self.delaunay.find_simplex((query[0], query[1])) {
            Some((idx, l1, l2, l3)) => { let (a, b, c) = self.delaunay.simplices[idx]; Ok(l1 * self.values[a] + l2 * self.values[b] + l3 * self.values[c]) }
            None => Ok(f64::NAN),
        }
    }
    pub fn eval_many(&self, queries: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> { queries.iter().map(|q| self.eval(q)).collect() }
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
        let opts = Interp1dOptions { kind: InterpKind::Nearest, ..Interp1dOptions::default() };
        let interp = Interp1d::new(&x, &y, opts).expect("interp1d");
        assert_eq!(interp.eval(0.4).unwrap(), 10.0);
        assert_eq!(interp.eval(0.6).unwrap(), 20.0);
        assert_eq!(interp.eval(1.5).unwrap(), 20.0);
    }

    #[test]
    fn bspline_basic() {
        let t = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let c = vec![1.0, 2.0, 3.0];
        let k = 2;
        let spl = BSpline::new(t, c, k).expect("bspline");
        assert_eq!(spl.eval(0.0), 1.0);
        assert_eq!(spl.eval(1.0), 3.0);
    }

    #[test]
    fn bspline_antiderivative() {
        let t = vec![0.0, 0.0, 1.0, 1.0];
        let c = vec![1.0, 1.0];
        let k = 1;
        let spl = BSpline::new(t, c, k).unwrap();
        let anti = spl.antiderivative(1).unwrap();
        assert_eq!(anti.degree(), 2);
        assert_eq!(anti.eval(0.0), 0.0);
        assert_eq!(anti.eval(1.0), 1.0);
    }
}
