#![forbid(unsafe_code)]

//! Orthogonal distance regression support for FrankenSciPy.
//!
//! The public surface mirrors the durable pieces of `scipy.odr`: data
//! containers, model containers, an `ODR` runner, an `Output` result, a
//! low-level `odr` helper, and the standard model factories. The solver is a
//! conservative explicit-model implementation with a local damped
//! Gauss-Newton/Levenberg-Marquardt loop: it estimates both fit parameters and
//! input corrections, so weighted errors in `x` and `y` participate in the same
//! objective.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// Callable shape used by `Model`: `f(beta, x) -> y`.
pub type ModelFn = Arc<dyn Fn(&[f64], &[f64]) -> Vec<f64> + Send + Sync + 'static>;

/// Optional Jacobian callback shape. Rows correspond to observations.
pub type JacobianFn = Arc<dyn Fn(&[f64], &[f64]) -> Vec<Vec<f64>> + Send + Sync + 'static>;

/// Optional parameter-estimate callback.
pub type EstimateFn = Arc<dyn Fn(&Data) -> Vec<f64> + Send + Sync + 'static>;

/// Warning marker matching SciPy's `OdrWarning` symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OdrWarning {
    pub detail: String,
}

/// Stop marker matching SciPy's `OdrStop` symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OdrStop {
    pub detail: String,
}

/// Error type matching SciPy's `OdrError` role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OdrError {
    InvalidArgument { detail: String },
    NonFiniteInput { detail: String },
    SolverFailure { detail: String },
    Unsupported { detail: String },
}

impl fmt::Display for OdrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument { detail }
            | Self::NonFiniteInput { detail }
            | Self::SolverFailure { detail }
            | Self::Unsupported { detail } => f.write_str(detail),
        }
    }
}

impl std::error::Error for OdrError {}

/// Data to fit, equivalent to `scipy.odr.Data` for explicit one-response data.
#[derive(Debug, Clone, PartialEq)]
pub struct Data {
    pub x: Vec<f64>,
    pub y: Option<Vec<f64>>,
    pub we: Vec<f64>,
    pub wd: Vec<f64>,
    pub fix: Option<Vec<bool>>,
    pub meta: BTreeMap<String, String>,
}

impl Data {
    /// Construct explicit response data with unit weights.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, OdrError> {
        validate_finite_slice("x", &x)?;
        validate_finite_slice("y", &y)?;
        if x.is_empty() {
            return Err(OdrError::InvalidArgument {
                detail: String::from("x must contain at least one observation"),
            });
        }
        if y.is_empty() {
            return Err(OdrError::InvalidArgument {
                detail: String::from("y must contain at least one observation"),
            });
        }
        Ok(Self {
            we: vec![1.0; y.len()],
            wd: vec![1.0; x.len()],
            x,
            y: Some(y),
            fix: None,
            meta: BTreeMap::new(),
        })
    }

    /// Construct implicit data. The current runner rejects implicit models, but
    /// the container is exposed so callers can represent the SciPy shape.
    pub fn implicit(x: Vec<f64>) -> Result<Self, OdrError> {
        validate_finite_slice("x", &x)?;
        if x.is_empty() {
            return Err(OdrError::InvalidArgument {
                detail: String::from("x must contain at least one observation"),
            });
        }
        Ok(Self {
            wd: vec![1.0; x.len()],
            x,
            y: None,
            we: Vec::new(),
            fix: None,
            meta: BTreeMap::new(),
        })
    }

    /// Set response weights `we`.
    pub fn with_response_weights(mut self, we: Vec<f64>) -> Result<Self, OdrError> {
        let y_len = self.response()?.len();
        validate_weight_vec("we", &we, y_len)?;
        self.we = we;
        Ok(self)
    }

    /// Set input weights `wd`.
    pub fn with_input_weights(mut self, wd: Vec<f64>) -> Result<Self, OdrError> {
        validate_weight_vec("wd", &wd, self.x.len())?;
        self.wd = wd;
        Ok(self)
    }

    /// Set input correction freedom flags. `true` means free, `false` means
    /// fixed; this follows the semantics of SciPy's positive/free `ifixx`.
    pub fn with_input_free(mut self, free: Vec<bool>) -> Result<Self, OdrError> {
        if free.len() != self.x.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "fix/free mask length must match x length (got {} and {})",
                    free.len(),
                    self.x.len()
                ),
            });
        }
        self.fix = Some(free);
        Ok(self)
    }

    pub fn response(&self) -> Result<&[f64], OdrError> {
        self.y.as_deref().ok_or_else(|| OdrError::Unsupported {
            detail: String::from("implicit ODR data has no explicit response vector"),
        })
    }

    pub fn set_meta(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.meta.insert(key.into(), value.into());
    }
}

/// Data with actual standard deviations, equivalent to `scipy.odr.RealData`.
#[derive(Debug, Clone, PartialEq)]
pub struct RealData {
    pub data: Data,
    pub sx: Option<Vec<f64>>,
    pub sy: Option<Vec<f64>>,
}

impl RealData {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, OdrError> {
        Ok(Self {
            data: Data::new(x, y)?,
            sx: None,
            sy: None,
        })
    }

    /// Construct `RealData` from standard deviations. As in SciPy, standard
    /// deviations are converted to weights with `1 / sigma^2`.
    pub fn from_stddev(
        x: Vec<f64>,
        y: Vec<f64>,
        sx: Option<Vec<f64>>,
        sy: Option<Vec<f64>>,
    ) -> Result<Self, OdrError> {
        let mut data = Data::new(x, y)?;
        if let Some(values) = sx.as_ref() {
            let x_len = data.x.len();
            let wd = stddev_to_weights("sx", values, x_len)?;
            data = data.with_input_weights(wd)?;
        }
        if let Some(values) = sy.as_ref() {
            let y_len = data.response()?.len();
            let we = stddev_to_weights("sy", values, y_len)?;
            data = data.with_response_weights(we)?;
        }
        Ok(Self { data, sx, sy })
    }
}

impl From<RealData> for Data {
    fn from(value: RealData) -> Self {
        value.data
    }
}

/// Model container matching `scipy.odr.Model`.
#[derive(Clone)]
pub struct Model {
    pub name: String,
    pub fcn: ModelFn,
    pub fjacb: Option<JacobianFn>,
    pub fjacd: Option<JacobianFn>,
    pub estimate: Option<EstimateFn>,
    pub implicit: bool,
    pub scalar_separable: bool,
    pub parameter_count: Option<usize>,
    pub meta: BTreeMap<String, String>,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.name)
            .field("has_fjacb", &self.fjacb.is_some())
            .field("has_fjacd", &self.fjacd.is_some())
            .field("has_estimate", &self.estimate.is_some())
            .field("implicit", &self.implicit)
            .field("scalar_separable", &self.scalar_separable)
            .field("parameter_count", &self.parameter_count)
            .field("meta", &self.meta)
            .finish_non_exhaustive()
    }
}

impl Model {
    pub fn new<F>(fcn: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> Vec<f64> + Send + Sync + 'static,
    {
        Self {
            name: String::from("custom"),
            fcn: Arc::new(fcn),
            fjacb: None,
            fjacd: None,
            estimate: None,
            implicit: false,
            scalar_separable: false,
            parameter_count: None,
            meta: BTreeMap::new(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_parameter_count(mut self, parameter_count: usize) -> Self {
        self.parameter_count = Some(parameter_count);
        self
    }

    /// Declare that the model evaluates each observation independently: output
    /// point `i` depends only on input point `i` and `beta`.
    pub fn with_scalar_separable(mut self, scalar_separable: bool) -> Self {
        self.scalar_separable = scalar_separable;
        self
    }

    /// Whether the model was declared [`Self::with_scalar_separable`].
    pub fn is_scalar_separable(&self) -> bool {
        self.scalar_separable
    }

    pub fn with_estimate<F>(mut self, estimate: F) -> Self
    where
        F: Fn(&Data) -> Vec<f64> + Send + Sync + 'static,
    {
        self.estimate = Some(Arc::new(estimate));
        self
    }

    pub fn with_fjacb<F>(mut self, fjacb: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> Vec<Vec<f64>> + Send + Sync + 'static,
    {
        self.fjacb = Some(Arc::new(fjacb));
        self
    }

    pub fn with_fjacd<F>(mut self, fjacd: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> Vec<Vec<f64>> + Send + Sync + 'static,
    {
        self.fjacd = Some(Arc::new(fjacd));
        self
    }

    pub fn implicit(mut self, implicit: bool) -> Self {
        self.implicit = implicit;
        self
    }

    pub fn set_meta(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.meta.insert(key.into(), value.into());
    }

    pub fn evaluate(&self, beta: &[f64], x: &[f64]) -> Vec<f64> {
        (self.fcn)(beta, x)
    }
}

/// Solver options for `ODR::run`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdrOptions {
    pub maxit: usize,
    pub sstol: f64,
    pub partol: f64,
    pub diff_step: f64,
    pub fit_type: FitType,
}

impl Default for OdrOptions {
    fn default() -> Self {
        Self {
            maxit: 50,
            sstol: f64::EPSILON.sqrt(),
            partol: f64::EPSILON.cbrt(),
            diff_step: 1.490_116_119_384_765_6e-8,
            fit_type: FitType::Odr,
        }
    }
}

/// Fit mode. `Ols` fixes input corrections at zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitType {
    Odr,
    Ols,
}

/// Coordinates an ODR fit, equivalent to `scipy.odr.ODR`.
#[derive(Debug, Clone)]
pub struct ODR {
    pub data: Data,
    pub model: Model,
    pub beta0: Vec<f64>,
    pub delta0: Option<Vec<f64>>,
    pub ifixb: Option<Vec<bool>>,
    pub ifixx: Option<Vec<bool>>,
    pub options: OdrOptions,
}

impl ODR {
    pub fn new(data: Data, model: Model, beta0: Vec<f64>) -> Result<Self, OdrError> {
        validate_beta0(&beta0, model.parameter_count)?;
        Ok(Self {
            data,
            model,
            beta0,
            delta0: None,
            ifixb: None,
            ifixx: None,
            options: OdrOptions::default(),
        })
    }

    pub fn with_options(mut self, options: OdrOptions) -> Result<Self, OdrError> {
        validate_options(options)?;
        self.options = options;
        Ok(self)
    }

    pub fn with_delta0(mut self, delta0: Vec<f64>) -> Result<Self, OdrError> {
        validate_finite_slice("delta0", &delta0)?;
        if delta0.len() != self.data.x.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "delta0 length must match x length (got {} and {})",
                    delta0.len(),
                    self.data.x.len()
                ),
            });
        }
        self.delta0 = Some(delta0);
        Ok(self)
    }

    /// Set beta freedom flags. `true` means free, `false` means fixed.
    pub fn with_beta_free(mut self, free: Vec<bool>) -> Result<Self, OdrError> {
        if free.len() != self.beta0.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "ifixb/free mask length must match beta0 length (got {} and {})",
                    free.len(),
                    self.beta0.len()
                ),
            });
        }
        self.ifixb = Some(free);
        Ok(self)
    }

    /// Set input correction freedom flags. `true` means free, `false` means fixed.
    pub fn with_input_free(mut self, free: Vec<bool>) -> Result<Self, OdrError> {
        if free.len() != self.data.x.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "ifixx/free mask length must match x length (got {} and {})",
                    free.len(),
                    self.data.x.len()
                ),
            });
        }
        self.ifixx = Some(free);
        Ok(self)
    }

    pub fn set_job(&mut self, fit_type: FitType) {
        self.options.fit_type = fit_type;
    }

    pub fn run(&self) -> Result<Output, OdrError> {
        self.run_impl(true)
    }

    /// Reference solver that forms the full `(n_beta + n_free_delta)`-dimensional
    /// finite-difference Jacobian and normal equations (the pre-structural
    /// `O(n^3)`-per-iteration path). Retained as an in-tree oracle: the structured
    /// solver reproduces its `beta`/`delta`/covariance to convergence tolerance.
    #[doc(hidden)]
    pub fn run_dense_reference(&self) -> Result<Output, OdrError> {
        self.run_impl(false)
    }

    fn run_impl(&self, use_structured: bool) -> Result<Output, OdrError> {
        if self.model.implicit {
            return Err(OdrError::Unsupported {
                detail: String::from("implicit ODR models are represented but not solved yet"),
            });
        }
        validate_options(self.options)?;
        let y = self.data.response()?;
        if y.len() > self.data.x.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "explicit ODR requires at least as many x values as y observations (got {} and {})",
                    self.data.x.len(),
                    y.len()
                ),
            });
        }

        let beta_free = freedom_mask(self.ifixb.as_deref(), self.beta0.len());
        let data_free = self
            .ifixx
            .as_deref()
            .or(self.data.fix.as_deref())
            .map_or_else(|| vec![true; self.data.x.len()], ToOwned::to_owned);
        let delta_free = match self.options.fit_type {
            FitType::Odr => data_free,
            FitType::Ols => vec![false; self.data.x.len()],
        };

        let free_beta_indices = free_indices(&beta_free);
        let free_delta_indices = free_indices(&delta_free);
        if free_beta_indices.is_empty() && free_delta_indices.is_empty() {
            return Err(OdrError::InvalidArgument {
                detail: String::from("at least one beta or delta variable must be free"),
            });
        }

        let delta0 = self
            .delta0
            .clone()
            .unwrap_or_else(|| vec![0.0; self.data.x.len()]);
        let variable0 = pack_variables(
            &self.beta0,
            &delta0,
            &free_beta_indices,
            &free_delta_indices,
        );
        let model = self.model.clone();
        let data = self.data.clone();
        let beta_template = self.beta0.clone();
        let delta_template = delta0.clone();
        let residual_beta_indices = free_beta_indices.clone();
        let residual_delta_indices = free_delta_indices.clone();
        let residuals = move |variables: &[f64]| {
            let (beta, delta) = unpack_variables(
                variables,
                &beta_template,
                &delta_template,
                &residual_beta_indices,
                &residual_delta_indices,
            );
            weighted_residuals(&data, &model, &beta, &delta).unwrap_or_else(|_| {
                vec![f64::INFINITY; data.response().map_or(data.x.len(), |resp| resp.len())]
            })
        };
        let ctx = OdrStruct {
            model: &self.model,
            data: &self.data,
            y,
            beta_template: &self.beta0,
            delta_template: &delta0,
            free_beta: &free_beta_indices,
            free_delta: &free_delta_indices,
            diff_step: self.options.diff_step,
        };
        // The structured (Schur-eliminated) path is valid only for scalar
        // pointwise models: response `i` depends on `x[i]` and not on any other
        // input coordinate. Arbitrary custom closures can legally couple output
        // rows, so they stay on the dense reference path unless the model is
        // explicitly marked scalar-separable.
        let can_structured =
            use_structured && self.data.x.len() == y.len() && self.model.is_scalar_separable();
        let solved = if can_structured {
            let sr = solve_odr_structured(&ctx, &residuals, &variable0, self.options)?;
            SolvedFit {
                x: sr.x,
                cost: sr.cost,
                success: sr.success,
                message: sr.message,
                nfev: sr.nfev,
                njev: sr.njev,
                nit: sr.nit,
                cov: CovSource::Struct(sr.jac),
            }
        } else {
            let lr = solve_least_squares(&residuals, &variable0, self.options)?;
            SolvedFit {
                x: lr.x,
                cost: lr.cost,
                success: lr.success,
                message: lr.message,
                nfev: lr.nfev,
                njev: lr.njev,
                nit: lr.nit,
                cov: CovSource::Dense(lr.jac),
            }
        };
        let result = solved;
        let (beta, delta) = unpack_variables(
            &result.x,
            &self.beta0,
            &delta0,
            &free_beta_indices,
            &free_delta_indices,
        );
        let xplus = add_slices(&self.data.x, &delta);
        let yfit = self.model.evaluate(&beta, &xplus);
        if yfit.len() != y.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "model output length must match y length (got {} and {})",
                    yfit.len(),
                    y.len()
                ),
            });
        }
        validate_finite_slice("model output", &yfit)?;
        let eps = y
            .iter()
            .zip(yfit.iter())
            .map(|(observed, fitted)| observed - fitted)
            .collect::<Vec<_>>();
        let sum_square_eps = weighted_sum_square(&eps, &self.data.we);
        let sum_square_delta = weighted_sum_square(&delta, &self.data.wd);
        let sum_square = sum_square_eps + sum_square_delta;
        let dof = y.len().saturating_sub(beta.len()).max(1);
        let res_var = sum_square / dof as f64;
        let cov_beta = match &result.cov {
            CovSource::Dense(jac) => {
                covariance_from_jacobian(jac, beta.len(), &free_beta_indices, res_var)
            }
            CovSource::Struct(jac) => ctx.covariance_beta(jac, beta.len(), res_var),
        };
        let sd_beta = cov_beta
            .iter()
            .enumerate()
            .map(|(idx, row)| row.get(idx).copied().unwrap_or(f64::NAN).max(0.0).sqrt())
            .collect::<Vec<_>>();
        let inv_condnum = reciprocal_condition_proxy(&cov_beta);
        Ok(Output {
            beta,
            sd_beta,
            cov_beta,
            delta,
            eps,
            xplus,
            y: yfit,
            res_var,
            sum_square,
            sum_square_delta,
            sum_square_eps,
            inv_condnum,
            rel_error: result.cost.abs() * f64::EPSILON,
            info: if result.success { 1 } else { 4 },
            stopreason: vec![result.message],
            nfev: result.nfev,
            njev: result.njev,
            nit: result.nit,
            success: result.success,
        })
    }

    pub fn restart(&self, additional_iterations: usize) -> Result<Output, OdrError> {
        let mut restarted = self.clone();
        restarted.options.maxit = self.options.maxit.saturating_add(additional_iterations);
        restarted.run()
    }
}

/// Output from an ODR fit, equivalent to `scipy.odr.Output`.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub beta: Vec<f64>,
    pub sd_beta: Vec<f64>,
    pub cov_beta: Vec<Vec<f64>>,
    pub delta: Vec<f64>,
    pub eps: Vec<f64>,
    pub xplus: Vec<f64>,
    pub y: Vec<f64>,
    pub res_var: f64,
    pub sum_square: f64,
    pub sum_square_delta: f64,
    pub sum_square_eps: f64,
    pub inv_condnum: f64,
    pub rel_error: f64,
    pub info: i32,
    pub stopreason: Vec<String>,
    pub nfev: usize,
    pub njev: usize,
    pub nit: usize,
    pub success: bool,
}

impl Output {
    pub fn pprint(&self) -> String {
        format!(
            "beta={:?}\nsd_beta={:?}\nres_var={:.6e}\nsum_square={:.6e}\ninfo={}\nstopreason={:?}",
            self.beta, self.sd_beta, self.res_var, self.sum_square, self.info, self.stopreason
        )
    }
}

/// Low-level helper matching `scipy.odr.odr`.
pub fn odr<F>(fcn: F, beta0: Vec<f64>, y: Vec<f64>, x: Vec<f64>) -> Result<Output, OdrError>
where
    F: Fn(&[f64], &[f64]) -> Vec<f64> + Send + Sync + 'static,
{
    ODR::new(Data::new(x, y)?, Model::new(fcn), beta0)?.run()
}

/// Names of the `scipy.odr` public surface tracked by the docs census.
pub fn public_api_symbols() -> &'static [&'static str] {
    &[
        "Data",
        "RealData",
        "Model",
        "ODR",
        "Output",
        "odr",
        "OdrWarning",
        "OdrError",
        "OdrStop",
        "polynomial",
        "exponential",
        "multilinear",
        "unilinear",
        "quadratic",
        "models",
        "odrpack",
    ]
}

pub fn unilinear() -> Model {
    Model::new(|beta, x| {
        let slope = beta.first().copied().unwrap_or(0.0);
        let intercept = beta.get(1).copied().unwrap_or(0.0);
        x.iter().map(|value| slope * value + intercept).collect()
    })
    .with_name("unilinear")
    .with_parameter_count(2)
    .with_scalar_separable(true)
}

pub fn quadratic() -> Model {
    polynomial(2).with_name("quadratic")
}

pub fn polynomial(order: usize) -> Model {
    Model::new(move |beta, x| {
        x.iter()
            .map(|value| {
                beta.iter()
                    .take(order + 1)
                    .rev()
                    .fold(0.0, |acc, coeff| acc * value + coeff)
            })
            .collect()
    })
    .with_name(format!("polynomial({order})"))
    .with_parameter_count(order + 1)
    .with_scalar_separable(true)
}

pub fn exponential() -> Model {
    Model::new(|beta, x| {
        let amplitude = beta.first().copied().unwrap_or(1.0);
        let rate = beta.get(1).copied().unwrap_or(1.0);
        let offset = beta.get(2).copied().unwrap_or(0.0);
        x.iter()
            .map(|value| amplitude * (rate * value).exp() + offset)
            .collect()
    })
    .with_name("exponential")
    .with_parameter_count(3)
    .with_scalar_separable(true)
}

pub fn multilinear(input_dim: usize) -> Model {
    Model::new(move |beta, x| {
        if input_dim == 0 {
            return Vec::new();
        }
        x.chunks_exact(input_dim)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold(beta.first().copied().unwrap_or(0.0), |acc, (idx, value)| {
                        acc + beta.get(idx + 1).copied().unwrap_or(0.0) * value
                    })
            })
            .collect()
    })
    .with_name(format!("multilinear({input_dim})"))
    .with_parameter_count(input_dim + 1)
    .with_scalar_separable(true)
}

fn validate_beta0(beta0: &[f64], expected: Option<usize>) -> Result<(), OdrError> {
    validate_finite_slice("beta0", beta0)?;
    if beta0.is_empty() {
        return Err(OdrError::InvalidArgument {
            detail: String::from("beta0 must contain at least one parameter"),
        });
    }
    if let Some(expected) = expected
        && beta0.len() != expected
    {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "beta0 length must match model parameter count (got {} and {expected})",
                beta0.len()
            ),
        });
    }
    Ok(())
}

fn validate_options(options: OdrOptions) -> Result<(), OdrError> {
    if options.maxit == 0 {
        return Err(OdrError::InvalidArgument {
            detail: String::from("maxit must be at least 1"),
        });
    }
    for (name, value) in [
        ("sstol", options.sstol),
        ("partol", options.partol),
        ("diff_step", options.diff_step),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(OdrError::InvalidArgument {
                detail: format!("{name} must be positive and finite"),
            });
        }
    }
    Ok(())
}

fn validate_finite_slice(name: &str, values: &[f64]) -> Result<(), OdrError> {
    if let Some((idx, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(OdrError::NonFiniteInput {
            detail: format!("{name}[{idx}] must be finite, got {value}"),
        });
    }
    Ok(())
}

fn validate_weight_vec(name: &str, values: &[f64], expected_len: usize) -> Result<(), OdrError> {
    if values.len() != expected_len {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "{name} length must match expected length (got {} and {expected_len})",
                values.len()
            ),
        });
    }
    for (idx, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(OdrError::InvalidArgument {
                detail: format!("{name}[{idx}] must be finite and non-negative, got {value}"),
            });
        }
    }
    Ok(())
}

fn stddev_to_weights(
    name: &str,
    values: &[f64],
    expected_len: usize,
) -> Result<Vec<f64>, OdrError> {
    if values.len() != expected_len {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "{name} length must match expected length (got {} and {expected_len})",
                values.len()
            ),
        });
    }
    values
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| {
            if !value.is_finite() || value <= 0.0 {
                Err(OdrError::InvalidArgument {
                    detail: format!("{name}[{idx}] must be finite and positive, got {value}"),
                })
            } else {
                Ok(1.0 / (value * value))
            }
        })
        .collect()
}

fn freedom_mask(mask: Option<&[bool]>, len: usize) -> Vec<bool> {
    mask.map_or_else(|| vec![true; len], ToOwned::to_owned)
}

fn free_indices(mask: &[bool]) -> Vec<usize> {
    mask.iter()
        .copied()
        .enumerate()
        .filter_map(|(idx, free)| free.then_some(idx))
        .collect()
}

fn pack_variables(
    beta: &[f64],
    delta: &[f64],
    beta_indices: &[usize],
    delta_indices: &[usize],
) -> Vec<f64> {
    beta_indices
        .iter()
        .map(|&idx| beta[idx])
        .chain(delta_indices.iter().map(|&idx| delta[idx]))
        .collect()
}

fn unpack_variables(
    variables: &[f64],
    beta_template: &[f64],
    delta_template: &[f64],
    beta_indices: &[usize],
    delta_indices: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let mut beta = beta_template.to_vec();
    let mut delta = delta_template.to_vec();
    for (position, &idx) in beta_indices.iter().enumerate() {
        beta[idx] = variables[position];
    }
    let delta_offset = beta_indices.len();
    for (position, &idx) in delta_indices.iter().enumerate() {
        delta[idx] = variables[delta_offset + position];
    }
    (beta, delta)
}

fn weighted_residuals(
    data: &Data,
    model: &Model,
    beta: &[f64],
    delta: &[f64],
) -> Result<Vec<f64>, OdrError> {
    let y = data.response()?;
    let xplus = add_slices(&data.x, delta);
    let prediction = model.evaluate(beta, &xplus);
    if prediction.len() != y.len() {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "model output length must match y length (got {} and {})",
                prediction.len(),
                y.len()
            ),
        });
    }
    if prediction.iter().any(|value| !value.is_finite()) {
        return Err(OdrError::NonFiniteInput {
            detail: String::from("model returned a non-finite prediction"),
        });
    }
    Ok(y.iter()
        .zip(prediction.iter())
        .zip(data.we.iter())
        .map(|((observed, fitted), weight)| weight.sqrt() * (observed - fitted))
        .chain(
            delta
                .iter()
                .zip(data.wd.iter())
                .map(|(correction, weight)| weight.sqrt() * correction),
        )
        .collect())
}

fn add_slices(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect()
}

fn weighted_sum_square(values: &[f64], weights: &[f64]) -> f64 {
    values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| weight * value * value)
        .sum()
}

#[derive(Debug, Clone, PartialEq)]
struct LocalLeastSquaresResult {
    x: Vec<f64>,
    cost: f64,
    success: bool,
    message: String,
    nfev: usize,
    njev: usize,
    nit: usize,
    jac: Vec<Vec<f64>>,
}

fn solve_least_squares<F>(
    residuals: F,
    x0: &[f64],
    options: OdrOptions,
) -> Result<LocalLeastSquaresResult, OdrError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if x0.is_empty() {
        return Err(OdrError::InvalidArgument {
            detail: String::from("least-squares variable vector must not be empty"),
        });
    }
    let mut x = x0.to_vec();
    let mut r = residuals(&x);
    let mut nfev = 1usize;
    if r.len() < x.len() {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "number of residuals ({}) must be >= number of variables ({})",
                r.len(),
                x.len()
            ),
        });
    }
    validate_finite_slice("initial residuals", &r)?;
    let mut cost = 0.5 * dot(&r, &r);
    let mut damping = 1.0e-3;
    let mut jac = finite_diff_jacobian(&residuals, &x, &r, options.diff_step)?;
    nfev += x.len();
    let mut njev = 1usize;
    for nit in 0..options.maxit {
        let gradient = jt_residual(&jac, &r);
        if max_abs(&gradient) <= options.sstol {
            return Ok(LocalLeastSquaresResult {
                x,
                cost,
                success: true,
                message: String::from("gradient tolerance reached"),
                nfev,
                njev,
                nit,
                jac,
            });
        }

        let mut accepted = false;
        for _ in 0..8 {
            let step = solve_lm_step(&jac, &r, damping)?;
            if max_abs(&step) <= options.partol * (1.0 + max_abs(&x)) {
                return Ok(LocalLeastSquaresResult {
                    x,
                    cost,
                    success: true,
                    message: String::from("parameter tolerance reached"),
                    nfev,
                    njev,
                    nit,
                    jac,
                });
            }
            let candidate = x
                .iter()
                .zip(step.iter())
                .map(|(value, delta)| value + delta)
                .collect::<Vec<_>>();
            let candidate_r = residuals(&candidate);
            nfev += 1;
            if candidate_r.iter().any(|value| !value.is_finite()) {
                damping *= 10.0;
                continue;
            }
            let candidate_cost = 0.5 * dot(&candidate_r, &candidate_r);
            if candidate_cost < cost {
                let rel_change = (cost - candidate_cost).abs() / cost.max(1.0);
                x = candidate;
                r = candidate_r;
                cost = candidate_cost;
                jac = finite_diff_jacobian(&residuals, &x, &r, options.diff_step)?;
                nfev += x.len();
                njev += 1;
                damping = (damping * 0.3).max(1.0e-12);
                accepted = true;
                if rel_change <= options.sstol {
                    return Ok(LocalLeastSquaresResult {
                        x,
                        cost,
                        success: true,
                        message: String::from("sum-of-squares tolerance reached"),
                        nfev,
                        njev,
                        nit: nit + 1,
                        jac,
                    });
                }
                break;
            }
            damping *= 10.0;
        }
        if !accepted {
            damping *= 10.0;
        }
    }
    Ok(LocalLeastSquaresResult {
        x,
        cost,
        success: false,
        message: String::from("maximum iterations reached"),
        nfev,
        njev,
        nit: options.maxit,
        jac,
    })
}

// ---------------------------------------------------------------------------
// Structured ODR solver.
//
// The explicit-ODR least-squares problem packs the model parameters `beta`
// (p free entries) and one x-error `delta` per data point (m free entries) into
// a single variable vector; the residual is `[ sqrt(we)·(y - f(beta, x+delta)) ;
// sqrt(wd)·delta ]` (length 2n). The dense reference path forms the full
// (p+m)-column finite-difference Jacobian and (p+m)² normal equations, so each
// iteration costs O(n·(p+m)²) ~ O(n³) with m ~ n.
//
// Because observation i's prediction depends only on x_i + delta_i, the entire
// delta block of the Jacobian is a single diagonal (response rows) plus the
// analytic diagonal sqrt(wd) (penalty rows). That lets us
//   * build the whole delta diagonal from ONE batched model evaluation, and
//   * eliminate the delta variables from the LM step via the Schur complement of
//     the diagonal delta–delta block, leaving a p×p system in beta,
// dropping the per-iteration cost to O(n·p²) — linear in the data size. The step
// and covariance are mathematically identical to the dense path (equal up to
// floating-point roundoff), so the fit converges to the same solution.
// ---------------------------------------------------------------------------

/// Structured finite-difference Jacobian for an explicit ODR problem.
/// Response residual rows are `0..n`; delta-penalty rows are `n..2n`.
struct StructJac {
    /// `a[i][k]` = ∂r_resp[i] / ∂(free β_k), the dense response-block β columns.
    a: Vec<Vec<f64>>,
    /// `d_diag[j]` = ∂r_resp[dj] / ∂δ_dj, the response-block δ diagonal
    /// (`dj = free_delta[j]`).
    d_diag: Vec<f64>,
    /// `b[j]` = sqrt(wd[dj]), the analytic δ-penalty diagonal.
    b: Vec<f64>,
}

struct OdrStruct<'a> {
    model: &'a Model,
    data: &'a Data,
    y: &'a [f64],
    beta_template: &'a [f64],
    delta_template: &'a [f64],
    free_beta: &'a [usize],
    free_delta: &'a [usize],
    diff_step: f64,
}

struct StructResult {
    x: Vec<f64>,
    cost: f64,
    success: bool,
    message: String,
    nfev: usize,
    njev: usize,
    nit: usize,
    jac: StructJac,
}

impl OdrStruct<'_> {
    fn n(&self) -> usize {
        self.data.x.len()
    }
    fn p(&self) -> usize {
        self.free_beta.len()
    }
    fn m(&self) -> usize {
        self.free_delta.len()
    }

    /// Structured finite-difference Jacobian at packed variables `x` with base
    /// residual `r`. The β columns reproduce the dense path bit-for-bit; the δ
    /// diagonal comes from a single all-δ-perturbed model evaluation (each output
    /// point depends only on its own input, so the batched value equals the
    /// one-at-a-time perturbation).
    fn jac(&self, x: &[f64], r: &[f64]) -> Result<StructJac, OdrError> {
        let (n, p, m) = (self.n(), self.p(), self.m());
        let (beta, delta) = unpack_variables(
            x,
            self.beta_template,
            self.delta_template,
            self.free_beta,
            self.free_delta,
        );
        let xplus = add_slices(&self.data.x, &delta);

        let mut a = vec![vec![0.0; p]; n];
        for k in 0..p {
            let h = self.diff_step * x[k].abs().max(1.0);
            let mut beta_pert = beta.clone();
            beta_pert[self.free_beta[k]] += h;
            let pred = self.model.evaluate(&beta_pert, &xplus);
            if pred.len() != n {
                return Err(jac_length_error(pred.len(), n));
            }
            for i in 0..n {
                let r_pert = self.data.we[i].sqrt() * (self.y[i] - pred[i]);
                if !r_pert.is_finite() {
                    return Err(jac_nonfinite_error());
                }
                a[i][k] = (r_pert - r[i]) / h;
            }
        }

        let mut d_diag = vec![0.0; m];
        let mut b = vec![0.0; m];
        if m > 0 {
            let mut xpert = xplus.clone();
            let mut hs = vec![0.0; m];
            for (j, &dj) in self.free_delta.iter().enumerate() {
                let h = self.diff_step * x[p + j].abs().max(1.0);
                hs[j] = h;
                xpert[dj] += h;
                b[j] = self.data.wd[dj].sqrt();
            }
            let pred = self.model.evaluate(&beta, &xpert);
            if pred.len() != n {
                return Err(jac_length_error(pred.len(), n));
            }
            for (j, &dj) in self.free_delta.iter().enumerate() {
                let r_pert = self.data.we[dj].sqrt() * (self.y[dj] - pred[dj]);
                if !r_pert.is_finite() {
                    return Err(jac_nonfinite_error());
                }
                d_diag[j] = (r_pert - r[dj]) / hs[j];
            }
        }
        Ok(StructJac { a, d_diag, b })
    }

    /// Number of model evaluations the structured Jacobian consumes: one full
    /// evaluation per free β column, plus one batched evaluation for all δ.
    fn jac_evals(&self) -> usize {
        self.p() + usize::from(self.m() > 0)
    }

    /// `max_abs(Jᵀr)`, matching the dense gradient-tolerance convergence test.
    fn gradient_maxabs(&self, sj: &StructJac, r: &[f64]) -> f64 {
        let (n, p) = (self.n(), self.p());
        let mut worst = 0.0_f64;
        for k in 0..p {
            let mut g = 0.0;
            for (i, &ri) in r.iter().take(n).enumerate() {
                g += sj.a[i][k] * ri;
            }
            worst = worst.max(g.abs());
        }
        for (j, &dj) in self.free_delta.iter().enumerate() {
            let g = sj.d_diag[j] * r[dj] + sj.b[j] * r[n + dj];
            worst = worst.max(g.abs());
        }
        worst
    }

    /// LM step `(JᵀJ + μI) s = -Jᵀr` solved by eliminating the diagonal δ block:
    /// reduce to a p×p system in the β step, then back-substitute each δ step.
    /// Equal to the dense `solve_lm_step` up to roundoff.
    fn lm_step(&self, sj: &StructJac, r: &[f64], damping: f64) -> Option<Vec<f64>> {
        let (n, p, m) = (self.n(), self.p(), self.m());

        // g_beta = Aᵀ r_resp (the β part of Jᵀr); rhs_beta = -g_beta.
        let mut g_beta = vec![0.0; p];
        for (i, &ri) in r.iter().take(n).enumerate() {
            for (k, g) in g_beta.iter_mut().enumerate() {
                *g += sj.a[i][k] * ri;
            }
        }

        // Per-row Schur coefficient c_i (1 for fixed-δ rows, (b²+μ)/g for free-δ
        // rows) and the δ-elimination contribution to the reduced right-hand side.
        let mut cvec = vec![1.0; n];
        let mut red_rhs: Vec<f64> = g_beta.iter().map(|g| -g).collect();
        for (j, &dj) in self.free_delta.iter().enumerate() {
            let a_j = sj.d_diag[j];
            let b_j = sj.b[j];
            let g_j = a_j * a_j + b_j * b_j + damping;
            if g_j.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                return None;
            }
            cvec[dj] = (b_j * b_j + damping) / g_j;
            let rhs_delta_j = -(a_j * r[dj] + b_j * r[n + dj]);
            let coeff = (a_j / g_j) * rhs_delta_j;
            for (k, value) in red_rhs.iter_mut().enumerate() {
                *value -= coeff * sj.a[dj][k];
            }
        }

        // Reduced β system R = μI + Σ_i c_i a_i a_iᵀ.
        let mut rmat = vec![vec![0.0; p]; p];
        for (k, row) in rmat.iter_mut().enumerate() {
            row[k] = damping;
        }
        for (i, &ci) in cvec.iter().enumerate().take(n) {
            if ci == 0.0 {
                continue;
            }
            let ai = &sj.a[i];
            for lhs in 0..p {
                let scaled = ai[lhs] * ci;
                if scaled == 0.0 {
                    continue;
                }
                let row = &mut rmat[lhs];
                for rhs in 0..p {
                    row[rhs] += scaled * ai[rhs];
                }
            }
        }

        let s_beta = if p > 0 {
            gaussian_solve(rmat, red_rhs)?
        } else {
            Vec::new()
        };

        let mut step = Vec::with_capacity(p + m);
        step.extend_from_slice(&s_beta);
        for (j, &dj) in self.free_delta.iter().enumerate() {
            let a_j = sj.d_diag[j];
            let b_j = sj.b[j];
            let g_j = a_j * a_j + b_j * b_j + damping;
            let rhs_delta_j = -(a_j * r[dj] + b_j * r[n + dj]);
            let a_dot_s: f64 = sj.a[dj].iter().zip(&s_beta).map(|(a, s)| a * s).sum();
            step.push((rhs_delta_j - a_j * a_dot_s) / g_j);
        }
        Some(step)
    }

    /// β covariance = res_var · (Schur complement of the δ–δ block of JᵀJ)⁻¹.
    /// This equals the β-block of the inverse of the full (p+m)² normal matrix
    /// the dense `covariance_from_jacobian` extracts, computed in O(n·p²).
    fn covariance_beta(&self, sj: &StructJac, beta_len: usize, res_var: f64) -> Vec<Vec<f64>> {
        let (n, p) = (self.n(), self.p());
        let nan_cov = || vec![vec![f64::NAN; beta_len]; beta_len];
        if beta_len == 0 {
            return Vec::new();
        }
        if p == 0 {
            return vec![vec![0.0; beta_len]; beta_len];
        }

        // c0_i = 1 for fixed-δ rows, b²/(a²+b²) for free-δ rows (μ = 0).
        let mut c0 = vec![1.0; n];
        for (j, &dj) in self.free_delta.iter().enumerate() {
            let a_j = sj.d_diag[j];
            let b_j = sj.b[j];
            let denom = a_j * a_j + b_j * b_j;
            c0[dj] = if denom > 0.0 { b_j * b_j / denom } else { 0.0 };
        }
        let mut schur = vec![vec![0.0; p]; p];
        for (i, &ci) in c0.iter().enumerate().take(n) {
            if ci == 0.0 {
                continue;
            }
            let ai = &sj.a[i];
            for lhs in 0..p {
                let scaled = ai[lhs] * ci;
                if scaled == 0.0 {
                    continue;
                }
                let row = &mut schur[lhs];
                for rhs in 0..p {
                    row[rhs] += scaled * ai[rhs];
                }
            }
        }

        invert_matrix(schur).map_or_else(nan_cov, |inverse| {
            let mut covariance = vec![vec![0.0; beta_len]; beta_len];
            for (lhs_var, &lhs_beta) in self.free_beta.iter().enumerate() {
                for (rhs_var, &rhs_beta) in self.free_beta.iter().enumerate() {
                    covariance[lhs_beta][rhs_beta] = inverse[lhs_var][rhs_var] * res_var;
                }
            }
            covariance
        })
    }
}

fn jac_length_error(got: usize, expected: usize) -> OdrError {
    OdrError::InvalidArgument {
        detail: format!(
            "residual length changed during finite differences (got {got} and {expected})"
        ),
    }
}

fn jac_nonfinite_error() -> OdrError {
    OdrError::NonFiniteInput {
        detail: String::from("finite-difference residuals[..] must be finite"),
    }
}

/// Structured Levenberg–Marquardt loop for explicit ODR. Mirrors the control
/// flow of [`solve_least_squares`] (identical damping schedule, tolerances, and
/// accept/reject logic) but uses the structured Jacobian and Schur-eliminated
/// step, so it costs O(n·p²) per iteration instead of O(n³).
fn solve_odr_structured<F>(
    ctx: &OdrStruct,
    residuals: &F,
    x0: &[f64],
    options: OdrOptions,
) -> Result<StructResult, OdrError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if x0.is_empty() {
        return Err(OdrError::InvalidArgument {
            detail: String::from("least-squares variable vector must not be empty"),
        });
    }
    let mut x = x0.to_vec();
    let mut r = residuals(&x);
    let mut nfev = 1usize;
    if r.len() < x.len() {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "number of residuals ({}) must be >= number of variables ({})",
                r.len(),
                x.len()
            ),
        });
    }
    validate_finite_slice("initial residuals", &r)?;
    let mut cost = 0.5 * dot(&r, &r);
    let mut damping = 1.0e-3;
    let mut jac = ctx.jac(&x, &r)?;
    nfev += ctx.jac_evals();
    let mut njev = 1usize;
    for nit in 0..options.maxit {
        if ctx.gradient_maxabs(&jac, &r) <= options.sstol {
            return Ok(StructResult {
                x,
                cost,
                success: true,
                message: String::from("gradient tolerance reached"),
                nfev,
                njev,
                nit,
                jac,
            });
        }

        let mut accepted = false;
        for _ in 0..8 {
            let step = ctx
                .lm_step(&jac, &r, damping)
                .ok_or_else(|| OdrError::SolverFailure {
                    detail: String::from("normal equations are singular"),
                })?;
            if max_abs(&step) <= options.partol * (1.0 + max_abs(&x)) {
                return Ok(StructResult {
                    x,
                    cost,
                    success: true,
                    message: String::from("parameter tolerance reached"),
                    nfev,
                    njev,
                    nit,
                    jac,
                });
            }
            let candidate = x
                .iter()
                .zip(step.iter())
                .map(|(value, delta)| value + delta)
                .collect::<Vec<_>>();
            let candidate_r = residuals(&candidate);
            nfev += 1;
            if candidate_r.iter().any(|value| !value.is_finite()) {
                damping *= 10.0;
                continue;
            }
            let candidate_cost = 0.5 * dot(&candidate_r, &candidate_r);
            if candidate_cost < cost {
                let rel_change = (cost - candidate_cost).abs() / cost.max(1.0);
                x = candidate;
                r = candidate_r;
                cost = candidate_cost;
                jac = ctx.jac(&x, &r)?;
                nfev += ctx.jac_evals();
                njev += 1;
                damping = (damping * 0.3).max(1.0e-12);
                accepted = true;
                if rel_change <= options.sstol {
                    return Ok(StructResult {
                        x,
                        cost,
                        success: true,
                        message: String::from("sum-of-squares tolerance reached"),
                        nfev,
                        njev,
                        nit: nit + 1,
                        jac,
                    });
                }
                break;
            }
            damping *= 10.0;
        }
        if !accepted {
            damping *= 10.0;
        }
    }
    Ok(StructResult {
        x,
        cost,
        success: false,
        message: String::from("maximum iterations reached"),
        nfev,
        njev,
        nit: options.maxit,
        jac,
    })
}

/// Result of an LM solve, carrying whichever Jacobian representation the chosen
/// solver produced so the covariance can be assembled the matching way.
struct SolvedFit {
    x: Vec<f64>,
    cost: f64,
    success: bool,
    message: String,
    nfev: usize,
    njev: usize,
    nit: usize,
    cov: CovSource,
}

enum CovSource {
    Dense(Vec<Vec<f64>>),
    Struct(StructJac),
}

fn finite_diff_jacobian<F>(
    residuals: &F,
    x: &[f64],
    r0: &[f64],
    step: f64,
) -> Result<Vec<Vec<f64>>, OdrError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut jac = vec![vec![0.0; x.len()]; r0.len()];
    for col in 0..x.len() {
        let mut x_plus = x.to_vec();
        let h = step * x[col].abs().max(1.0);
        x_plus[col] += h;
        let r_plus = residuals(&x_plus);
        if r_plus.len() != r0.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "residual length changed during finite differences (got {} and {})",
                    r_plus.len(),
                    r0.len()
                ),
            });
        }
        validate_finite_slice("finite-difference residuals", &r_plus)?;
        for row in 0..r0.len() {
            jac[row][col] = (r_plus[row] - r0[row]) / h;
        }
    }
    Ok(jac)
}

fn solve_lm_step(
    jacobian: &[Vec<f64>],
    residuals: &[f64],
    damping: f64,
) -> Result<Vec<f64>, OdrError> {
    let n = jacobian.first().map_or(0, Vec::len);
    let mut normal = vec![vec![0.0; n]; n];
    let mut rhs = vec![0.0; n];
    for (row_idx, row) in jacobian.iter().enumerate() {
        for lhs in 0..n {
            rhs[lhs] -= row[lhs] * residuals[row_idx];
            for rhs_idx in 0..n {
                normal[lhs][rhs_idx] += row[lhs] * row[rhs_idx];
            }
        }
    }
    for (idx, row) in normal.iter_mut().enumerate() {
        row[idx] += damping;
    }
    // Solve `normal · step = rhs` directly instead of forming the full inverse
    // and multiplying: the LM step needs only the one solution vector, and a
    // direct Gauss-Jordan solve does ~half the O(n^3) work (it transforms the
    // single RHS column, not an n-column identity) while being the numerically
    // preferable route. For ODR the solve dimension is n_params + n_free_delta
    // (one delta per data point), so this is the per-iteration bottleneck.
    gaussian_solve(normal, rhs).ok_or_else(|| OdrError::SolverFailure {
        detail: String::from("normal equations are singular"),
    })
}

/// Solve `matrix · x = rhs` via Gauss-Jordan elimination with partial pivoting.
///
/// Uses the same pivot selection and singularity guard as [`invert_matrix`], but
/// transforms only the single RHS column rather than an n-column identity, so it
/// costs ~half the arithmetic of inverting then multiplying. Result agrees with
/// `invert_matrix(matrix)` applied to `rhs` to within floating-point roundoff.
fn gaussian_solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = matrix.len();
    if n == 0 || rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    for pivot in 0..n {
        let best = (pivot..n).max_by(|&lhs, &rhs_row| {
            matrix[lhs][pivot]
                .abs()
                .total_cmp(&matrix[rhs_row][pivot].abs())
        })?;
        if matrix[best][pivot].abs() <= 1.0e-14 {
            return None;
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let scale = matrix[pivot][pivot];
        for value in &mut matrix[pivot] {
            *value /= scale;
        }
        rhs[pivot] /= scale;
        let pivot_row = matrix[pivot].clone();
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for (value, pivot_value) in matrix[row].iter_mut().zip(&pivot_row) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
}

fn jt_residual(jacobian: &[Vec<f64>], residuals: &[f64]) -> Vec<f64> {
    let n = jacobian.first().map_or(0, Vec::len);
    let mut gradient = vec![0.0; n];
    for (row_idx, row) in jacobian.iter().enumerate() {
        for col in 0..n {
            gradient[col] += row[col] * residuals[row_idx];
        }
    }
    gradient
}

fn dot(values: &[f64], rhs: &[f64]) -> f64 {
    values
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn covariance_from_jacobian(
    jacobian: &[Vec<f64>],
    beta_len: usize,
    free_beta_indices: &[usize],
    res_var: f64,
) -> Vec<Vec<f64>> {
    let variable_len = jacobian.first().map_or(0, Vec::len);
    let nan_covariance = || vec![vec![f64::NAN; beta_len]; beta_len];
    if beta_len == 0 {
        return Vec::new();
    }
    if variable_len == 0
        || free_beta_indices
            .iter()
            .any(|&beta_idx| beta_idx >= beta_len)
    {
        return nan_covariance();
    }

    let mut normal = vec![vec![0.0; variable_len]; variable_len];
    for row in jacobian {
        if row.len() != variable_len {
            return nan_covariance();
        }
        for lhs in 0..variable_len {
            for rhs in 0..variable_len {
                normal[lhs][rhs] += row[lhs] * row[rhs];
            }
        }
    }
    invert_matrix(normal).map_or_else(nan_covariance, |inverse| {
        let mut covariance = vec![vec![0.0; beta_len]; beta_len];
        for (lhs_var, &lhs_beta) in free_beta_indices.iter().enumerate() {
            for (rhs_var, &rhs_beta) in free_beta_indices.iter().enumerate() {
                covariance[lhs_beta][rhs_beta] = inverse[lhs_var][rhs_var] * res_var;
            }
        }
        covariance
    })
}

fn invert_matrix(mut matrix: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut inverse = vec![vec![0.0; n]; n];
    for (idx, row) in inverse.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    for pivot in 0..n {
        let best = (pivot..n).max_by(|&lhs, &rhs| {
            matrix[lhs][pivot]
                .abs()
                .total_cmp(&matrix[rhs][pivot].abs())
        })?;
        if matrix[best][pivot].abs() <= 1.0e-14 {
            return None;
        }
        matrix.swap(pivot, best);
        inverse.swap(pivot, best);
        let scale = matrix[pivot][pivot];
        for col in 0..n {
            matrix[pivot][col] /= scale;
            inverse[pivot][col] /= scale;
        }
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for col in 0..n {
                matrix[row][col] -= factor * matrix[pivot][col];
                inverse[row][col] -= factor * inverse[pivot][col];
            }
        }
    }
    Some(inverse)
}

fn reciprocal_condition_proxy(matrix: &[Vec<f64>]) -> f64 {
    let diag = matrix
        .iter()
        .enumerate()
        .filter_map(|(idx, row)| row.get(idx).copied())
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if diag.is_empty() {
        return 0.0;
    }
    let min = diag
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, value| acc.min(value));
    let max = diag
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value));
    if max == 0.0 { 0.0 } else { min / max }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        assert!(
            (lhs - rhs).abs() <= tol,
            "expected {lhs} ~= {rhs} within {tol}"
        );
    }

    fn assert_vec_close(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (left, right) in lhs.iter().zip(rhs) {
            let scale = right.abs().max(1.0);
            assert!(
                (left - right).abs() / scale <= tol,
                "expected {left} ~= {right} within relative tol {tol}"
            );
        }
    }

    fn assert_output_close(lhs: &Output, rhs: &Output, tol: f64) {
        assert_vec_close(&lhs.beta, &rhs.beta, tol);
        assert_vec_close(&lhs.delta, &rhs.delta, tol);
        assert_vec_close(&lhs.sd_beta, &rhs.sd_beta, tol);
        assert_close(
            lhs.sum_square,
            rhs.sum_square,
            tol * rhs.sum_square.abs().max(1.0),
        );
        assert_eq!(lhs.cov_beta.len(), rhs.cov_beta.len());
        for (left, right) in lhs.cov_beta.iter().zip(&rhs.cov_beta) {
            assert_vec_close(left, right, tol);
        }
    }

    #[test]
    fn public_api_matches_documented_scipy_odr_symbols() {
        assert_eq!(public_api_symbols().len(), 16);
        for symbol in [
            "Data",
            "RealData",
            "Model",
            "ODR",
            "Output",
            "odr",
            "OdrWarning",
            "OdrError",
            "OdrStop",
            "polynomial",
            "exponential",
            "multilinear",
            "unilinear",
            "quadratic",
            "models",
            "odrpack",
        ] {
            assert!(public_api_symbols().contains(&symbol));
        }
    }

    #[test]
    fn realdata_stddevs_convert_to_inverse_variance_weights() -> Result<(), OdrError> {
        let real = RealData::from_stddev(
            vec![0.0, 1.0],
            vec![1.0, 3.0],
            Some(vec![2.0, 4.0]),
            Some(vec![0.5, 0.25]),
        )?;
        assert_eq!(real.data.wd, vec![0.25, 0.0625]);
        assert_eq!(real.data.we, vec![4.0, 16.0]);
        Ok(())
    }

    #[test]
    fn odr_recovers_exact_unilinear_parameters() -> Result<(), OdrError> {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let y = x.iter().map(|value| 2.5 * value - 1.25).collect();
        let output = ODR::new(Data::new(x, y)?, unilinear(), vec![0.0, 0.0])?.run()?;
        assert!(output.success);
        assert_close(output.beta[0], 2.5, 1.0e-6);
        assert_close(output.beta[1], -1.25, 1.0e-6);
        assert!(output.sum_square < 1.0e-10);
        assert!(output.delta.iter().all(|value| value.abs() < 1.0e-6));
        Ok(())
    }

    #[test]
    fn odr_matches_scipy_odr_on_noisy_data() -> Result<(), OdrError> {
        // Golden values from scipy.odr.ODR(Data(x,y), unilinear, beta0).run()
        // (SciPy 1.17.1, ODRPACK) on data where x and y both carry error, so the
        // orthogonal-distance fit genuinely differs from OLS. Independent Rust
        // and Fortran ODRPACK implementations agree to convergence tolerance.
        struct Case {
            x: Vec<f64>,
            y: Vec<f64>,
            beta0: Vec<f64>,
            beta: [f64; 2],
            sd_beta: [f64; 2],
            res_var: f64,
            sum_square: f64,
        }
        let cases = [
            Case {
                x: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                y: vec![1.1, 2.9, 5.2, 6.8, 9.1, 10.9],
                beta0: vec![1.0, 0.0],
                beta: [1.979694827136702e0, 1.050762934064172e0],
                sd_beta: [3.983626321501051e-2, 1.205905103654928e-1],
                res_var: 5.639702685253586e-3,
                sum_square: 2.255881074101434e-2,
            },
            Case {
                x: vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                y: vec![-8.5, -5.1, -3.2, 0.3, 2.7, 5.4, 8.9],
                beta0: vec![1.0, 0.0],
                beta: [2.833504940851501e0, 7.142907832097929e-2],
                sd_beta: [7.371963337556588e-2, 1.472423725008963e-1],
                res_var: 1.680877299337602e-2,
                sum_square: 8.404386496688011e-2,
            },
        ];
        for c in cases {
            let out = ODR::new(Data::new(c.x, c.y)?, unilinear(), c.beta0)?.run()?;
            assert!(out.success);
            // Parameters agree to ODR convergence tolerance (~1e-5).
            assert_close(out.beta[0], c.beta[0], 1.0e-5);
            assert_close(out.beta[1], c.beta[1], 1.0e-5);
            assert_close(out.sd_beta[0], c.sd_beta[0], 1.0e-5);
            assert_close(out.sd_beta[1], c.sd_beta[1], 1.0e-5);
            // The achieved objective matches much more tightly (both find the
            // same minimum even where the flat objective loosely pins beta).
            assert_close(out.res_var, c.res_var, 1.0e-8);
            assert_close(out.sum_square, c.sum_square, 1.0e-8);
        }
        Ok(())
    }

    #[test]
    fn odr_ols_mode_reduces_to_response_residual_fit() -> Result<(), OdrError> {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let mut odr = ODR::new(Data::new(x, y)?, unilinear(), vec![1.0, 0.0])?;
        odr.set_job(FitType::Ols);
        let output = odr.run()?;
        assert_close(output.beta[0], 2.0, 1.0e-6);
        assert_close(output.beta[1], 1.0, 1.0e-6);
        assert!(output.delta.iter().all(|value| *value == 0.0));
        Ok(())
    }

    #[test]
    fn fixed_beta_flag_holds_parameter_constant() -> Result<(), OdrError> {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 4.0, 7.0, 10.0];
        let output = ODR::new(Data::new(x, y)?, unilinear(), vec![3.0, 0.0])?
            .with_beta_free(vec![false, true])?
            .run()?;
        assert_close(output.beta[0], 3.0, 0.0);
        assert_close(output.beta[1], 1.0, 1.0e-5);
        assert_close(output.cov_beta[0][0], 0.0, 0.0);
        assert_close(output.sd_beta[0], 0.0, 0.0);
        Ok(())
    }

    #[test]
    fn structured_scalar_odr_matches_dense_reference() -> Result<(), OdrError> {
        let x = (0..24)
            .map(|idx| idx as f64 * 0.05 + (idx % 5) as f64 * 0.002)
            .collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|value| 0.7 + 1.3 * value + 0.4 * (0.9 * value).sin())
            .collect::<Vec<_>>();
        let model = Model::new(|beta: &[f64], x: &[f64]| {
            x.iter()
                .map(|value| beta[0] + beta[1] * value + beta[2] * (beta[3] * value).sin())
                .collect()
        })
        .with_parameter_count(4)
        .with_scalar_separable(true);
        let odr = ODR::new(Data::new(x, y)?, model, vec![0.5, 1.5, 0.2, 0.8])?;

        let structured = odr.run()?;
        let dense = odr.run_dense_reference()?;

        assert_output_close(&structured, &dense, 1.0e-6);
        Ok(())
    }

    #[test]
    fn custom_coupled_model_uses_dense_reference_path() -> Result<(), OdrError> {
        let x = vec![0.2, 0.4, 0.8, 1.6, 3.2];
        let y = vec![0.44, 0.88, 1.76, 3.52, 7.04];
        let model = Model::new(|beta: &[f64], x: &[f64]| {
            x.iter()
                .enumerate()
                .map(|(idx, value)| {
                    let neighbor = x[(idx + 1) % x.len()];
                    beta[0] * value + beta[1] * neighbor
                })
                .collect()
        })
        .with_parameter_count(2);
        let odr = ODR::new(Data::new(x, y)?, model, vec![1.0, 0.1])?;

        let default = odr.run()?;
        let dense = odr.run_dense_reference()?;

        assert_eq!(default, dense);
        Ok(())
    }

    #[test]
    fn structured_covariance_is_zero_when_all_beta_fixed() -> Result<(), OdrError> {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let output = ODR::new(Data::new(x, y)?, unilinear(), vec![2.0, 1.0])?
            .with_beta_free(vec![false, false])?
            .run()?;

        assert_eq!(output.cov_beta, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert_eq!(output.sd_beta, vec![0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn covariance_from_jacobian_uses_full_normal_beta_block() {
        let jacobian = vec![vec![1.0, 0.0], vec![1.0, 1.0]];
        let covariance = covariance_from_jacobian(&jacobian, 1, &[0], 3.0);

        // The full normal matrix is [[2, 1], [1, 1]]. Its inverse has
        // beta-block [[1]], while the old beta-only inverse was [[0.5]].
        assert_close(covariance[0][0], 3.0, 1.0e-12);
    }

    #[test]
    fn polynomial_model_recovers_quadratic_coefficients() -> Result<(), OdrError> {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = x
            .iter()
            .map(|value| 1.0 - 2.0 * value + 0.5 * value * value)
            .collect();
        let mut odr = ODR::new(Data::new(x, y)?, quadratic(), vec![0.0, 0.0, 0.0])?;
        odr.set_job(FitType::Ols);
        let output = odr.run()?;
        assert_close(output.beta[0], 1.0, 1.0e-6);
        assert_close(output.beta[1], -2.0, 1.0e-6);
        assert_close(output.beta[2], 0.5, 1.0e-6);
        Ok(())
    }

    #[test]
    fn multilinear_model_uses_chunked_input_rows() -> Result<(), OdrError> {
        let x = vec![1.0, 2.0, 2.0, 1.0, -1.0, 3.0, 0.0, -2.0];
        let y = x
            .chunks_exact(2)
            .map(|row| 0.5 + 2.0 * row[0] - 3.0 * row[1])
            .collect();
        let mut odr = ODR::new(Data::new(x, y)?, multilinear(2), vec![0.0, 0.0, 0.0])?;
        odr.set_job(FitType::Ols);
        let output = odr.run()?;
        assert_close(output.beta[0], 0.5, 1.0e-6);
        assert_close(output.beta[1], 2.0, 1.0e-6);
        assert_close(output.beta[2], -3.0, 1.0e-6);
        Ok(())
    }

    #[test]
    fn invalid_shapes_and_nonfinite_inputs_fail_closed() {
        assert!(Data::new(vec![0.0], vec![f64::NAN]).is_err());
        assert!(RealData::from_stddev(vec![0.0], vec![1.0], Some(vec![0.0]), None).is_err());
        let data = match Data::new(vec![0.0], vec![1.0]) {
            Ok(data) => data,
            Err(error) => return assert!(matches!(error, OdrError::InvalidArgument { .. })),
        };
        let err = ODR::new(data, unilinear(), vec![0.0]);
        assert!(err.is_err());
    }

    #[test]
    fn odr_linear_matches_scipy_reference_values() -> Result<(), OdrError> {
        // scipy.odr with linear model on y = 2*x + noise
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.1, 7.9, 10.1];
        let data = Data::new(x, y)?;
        let odr = ODR::new(data, unilinear(), vec![1.0, 0.0])?;
        let output = odr.run()?;
        // scipy reference: beta = [2.00192, 0.01424]
        assert_close(output.beta[0], 2.00192, 0.01);
        assert_close(output.beta[1], 0.01424, 0.1);
        Ok(())
    }

    #[test]
    fn odr_quadratic_matches_scipy_reference_values() -> Result<(), OdrError> {
        // scipy.odr with quadratic model on y = x^2 + noise
        // Polynomial params are [c, b, a] for ax^2 + bx + c (Horner form)
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.1, 4.0, 9.1, 15.9, 25.0];
        let data = Data::new(x, y)?;
        // Initial guess: c=0, b=0, a=1
        let odr = ODR::new(data, quadratic(), vec![0.0, 0.0, 1.0])?;
        let output = odr.run()?;
        // Expect quadratic coeff (beta[2]) ≈ 1
        assert_close(output.beta[2], 1.0, 0.1);
        Ok(())
    }

    #[test]
    fn odr_polynomial_degree2_matches_scipy_reference_values() -> Result<(), OdrError> {
        // scipy.odr with polynomial model
        // Polynomial params are [c, b, a] for ax^2 + bx + c
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2
        let data = Data::new(x, y)?;
        // Initial guess: c=0, b=0, a=1
        let odr = ODR::new(data, polynomial(2), vec![0.0, 0.0, 1.0])?;
        let output = odr.run()?;
        // Should fit x^2: beta[2]=1, beta[1]=0, beta[0]=0
        assert_close(output.beta[2], 1.0, 0.1);
        assert!(output.beta[1].abs() < 0.2, "linear term should be near 0");
        assert!(output.beta[0].abs() < 0.2, "constant term should be near 0");
        Ok(())
    }

    #[test]
    fn odr_exponential_matches_scipy_reference_values() -> Result<(), OdrError> {
        // scipy.odr with exponential model y = a * exp(b * x) + c
        // Test data: y = 2 * exp(0.5 * x) + 0
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * (0.5_f64 * xi).exp()).collect();
        let data = Data::new(x, y)?;
        // Initial guess: amplitude=1, rate=1, offset=0
        let odr = ODR::new(data, exponential(), vec![1.0, 1.0, 0.0])?;
        let output = odr.run()?;
        // Should fit: beta[0]=amplitude≈2, beta[1]=rate≈0.5, beta[2]=offset≈0
        assert_close(output.beta[0], 2.0, 0.5);
        assert_close(output.beta[1], 0.5, 0.3);
        assert!(output.beta[2].abs() < 0.5, "offset should be near 0");
        Ok(())
    }

    #[test]
    fn odr_unilinear_matches_scipy_reference_values() -> Result<(), OdrError> {
        // scipy.odr with unilinear model y = a * x + b
        // Test data: y = 2x + 1
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let data = Data::new(x, y)?;
        // Initial guess: a=1, b=0
        let odr = ODR::new(data, unilinear(), vec![1.0, 0.0])?;
        let output = odr.run()?;
        // Should fit: beta[0]=a≈2, beta[1]=b≈1
        assert_close(output.beta[0], 2.0, 0.1);
        assert_close(output.beta[1], 1.0, 0.1);
        Ok(())
    }
}
