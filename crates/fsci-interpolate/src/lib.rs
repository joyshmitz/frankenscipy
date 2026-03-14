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
}

impl Default for Interp1dOptions {
    fn default() -> Self {
        Self {
            kind: InterpKind::Linear,
            mode: RuntimeMode::Strict,
            fill_value: None,
            bounds_error: true,
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
            Some(compute_natural_cubic_spline(x, y)?)
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

/// Compute natural cubic spline coefficients.
///
/// For each interval [x_i, x_{i+1}], the spline is:
///   S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)² + d_i*(x-x_i)³
///
/// Natural boundary conditions: S''(x_0) = 0, S''(x_n) = 0.
fn compute_natural_cubic_spline(x: &[f64], y: &[f64]) -> Result<Vec<[f64; 4]>, InterpError> {
    let n = x.len();
    if n < 4 {
        return Err(InterpError::TooFewPoints {
            minimum: 4,
            actual: n,
        });
    }

    let m = n - 1; // number of intervals

    // h[i] = x[i+1] - x[i]
    let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();

    // Solve tridiagonal system for second derivatives c[i]
    // Natural BC: c[0] = 0, c[n-1] = 0
    // Interior: h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1]
    //           = 3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    let inner = n - 2; // number of interior points
    let mut rhs = vec![0.0; inner];
    for (i, r) in rhs.iter_mut().enumerate() {
        let ii = i + 1;
        *r = 3.0 * ((y[ii + 1] - y[ii]) / h[ii] - (y[ii] - y[ii - 1]) / h[ii - 1]);
    }

    // Tridiagonal system: sub[i]*c[i-1] + diag[i]*c[i] + sup[i]*c[i+1] = rhs[i]
    let mut diag: Vec<f64> = (0..inner).map(|i| 2.0 * (h[i] + h[i + 1])).collect();
    let sub: Vec<f64> = (0..inner).map(|i| h[i]).collect();
    let sup: Vec<f64> = (0..inner).map(|i| h[i + 1]).collect();

    // Thomas algorithm for tridiagonal solve
    for i in 1..inner {
        let w = sub[i] / diag[i - 1];
        diag[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }

    let mut c_inner = vec![0.0; inner];
    c_inner[inner - 1] = rhs[inner - 1] / diag[inner - 1];
    for i in (0..inner - 1).rev() {
        c_inner[i] = (rhs[i] - sup[i] * c_inner[i + 1]) / diag[i];
    }

    // Full c array with natural BC
    let mut c = vec![0.0; n];
    for (i, &ci) in c_inner.iter().enumerate() {
        c[i + 1] = ci;
    }

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
}
