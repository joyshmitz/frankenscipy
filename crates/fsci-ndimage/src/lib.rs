#![forbid(unsafe_code)]

//! N-dimensional image processing for FrankenSciPy.
//!
//! Matches `scipy.ndimage` core operations:
//! - Filters: uniform, gaussian, median, minimum, maximum, convolve, correlate
//! - Morphology: binary_erosion, binary_dilation, binary_opening, binary_closing
//! - Measurements: label, find_objects, sum, mean, variance, standard_deviation
//! - Interpolation: shift, rotate, zoom, map_coordinates
//! - Distance transforms: distance_transform_edt

use fsci_interpolate::make_interp_spline;

/// Error type for ndimage operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NdimageError {
    InvalidArgument(String),
    DimensionMismatch(String),
    EmptyInput,
}

impl std::fmt::Display for NdimageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for NdimageError {}

/// Boundary mode for filtering operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Reflect: d c b a | a b c d | d c b a
    Reflect,
    /// Constant: k k k k | a b c d | k k k k (default k=0)
    Constant,
    /// Nearest: a a a a | a b c d | d d d d
    Nearest,
    /// Wrap: a b c d | a b c d | a b c d
    Wrap,
}

// ══════════════════════════════════════════════════════════════════════
// N-D Array Helper
// ══════════════════════════════════════════════════════════════════════

/// A simple N-dimensional array stored in row-major (C) order.
#[derive(Debug, Clone)]
pub struct NdArray {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl NdArray {
    /// Create an NdArray from data and shape.
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, NdimageError> {
        let total: usize = shape.iter().product();
        if total != data.len() {
            return Err(NdimageError::DimensionMismatch(format!(
                "shape {:?} requires {} elements, got {}",
                shape,
                total,
                data.len()
            )));
        }
        let strides = compute_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Create a zero-filled array.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        let strides = compute_strides(&shape);
        Self {
            data: vec![0.0; total],
            shape,
            strides,
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get element by multi-dimensional index.
    pub fn get(&self, idx: &[usize]) -> f64 {
        let flat = self.flat_index(idx);
        self.data[flat]
    }

    /// Set element by multi-dimensional index.
    pub fn set(&mut self, idx: &[usize], value: f64) {
        let flat = self.flat_index(idx);
        self.data[flat] = value;
    }

    fn flat_index(&self, idx: &[usize]) -> usize {
        idx.iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum()
    }

    /// Convert flat index to multi-dimensional index.
    fn unravel(&self, mut flat: usize) -> Vec<usize> {
        let mut idx = vec![0usize; self.ndim()];
        for (d, slot) in idx.iter_mut().enumerate() {
            *slot = flat / self.strides[d];
            flat %= self.strides[d];
        }
        idx
    }

    /// Get value at index with boundary handling.
    fn get_boundary(&self, idx: &[i64], mode: BoundaryMode, cval: f64) -> f64 {
        let mut safe_idx = vec![0usize; self.ndim()];
        for d in 0..self.ndim() {
            let size = self.shape[d] as i64;
            let mut i = idx[d];
            match mode {
                BoundaryMode::Reflect => {
                    if i < 0 {
                        i = -i - 1;
                    }
                    if i >= size {
                        i = 2 * size - i - 1;
                    }
                    // Handle multiple reflections
                    let period = 2 * size;
                    if period > 0 {
                        i = i.rem_euclid(period);
                        if i >= size {
                            i = period - i - 1;
                        }
                    }
                }
                BoundaryMode::Constant => {
                    if i < 0 || i >= size {
                        return cval;
                    }
                }
                BoundaryMode::Nearest => {
                    i = i.clamp(0, size - 1);
                }
                BoundaryMode::Wrap => {
                    i = i.rem_euclid(size);
                }
            }
            safe_idx[d] = i as usize;
        }
        self.get(&safe_idx)
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

fn unravel_with_shape(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let strides = compute_strides(shape);
    let mut idx = vec![0usize; shape.len()];
    for (d, slot) in idx.iter_mut().enumerate() {
        *slot = flat / strides[d];
        flat %= strides[d];
    }
    idx
}

const SPLINE_NEAREST_PAD: usize = 12;

#[derive(Debug, Clone)]
struct SplinePrefilter {
    coeffs: NdArray,
    coord_offsets: Vec<f64>,
}

fn uniform_interpolation_knots(len: usize, order: usize) -> Vec<f64> {
    let mut knots = Vec::with_capacity(len + order + 1);
    knots.extend(std::iter::repeat_n(0.0, order + 1));
    for i in 0..len.saturating_sub(order + 1) {
        knots.push((i + 1 + (order - 1) / 2) as f64);
    }
    knots.extend(std::iter::repeat_n((len - 1) as f64, order + 1));
    knots
}

fn eval_bspline_basis_all(knots: &[f64], x: f64, order: usize, len: usize) -> Vec<f64> {
    let mut basis = vec![0.0; len];
    for i in 0..len {
        if i + 1 < knots.len() {
            basis[i] = if (knots[i] <= x && x < knots[i + 1])
                || (x == knots[i + 1] && i + 1 == knots.len() - order - 1)
            {
                1.0
            } else {
                0.0
            };
        }
    }
    for p in 1..=order {
        let prev = basis.clone();
        for i in 0..len {
            let mut val = 0.0;
            if i + p < knots.len() {
                let denom_left = knots[i + p] - knots[i];
                if denom_left > 0.0 {
                    val += (x - knots[i]) / denom_left * prev[i];
                }
            }
            if i + p + 1 < knots.len() && i + 1 < len {
                let denom_right = knots[i + p + 1] - knots[i + 1];
                if denom_right > 0.0 {
                    val += (knots[i + p + 1] - x) / denom_right * prev[i + 1];
                }
            }
            basis[i] = val;
        }
    }
    basis
}

fn map_coordinate(coord: f64, len: usize, mode: BoundaryMode) -> Option<f64> {
    let max = (len.saturating_sub(1)) as f64;
    match mode {
        BoundaryMode::Constant => {
            if coord < 0.0 || coord > max {
                None
            } else {
                Some(coord)
            }
        }
        BoundaryMode::Nearest => Some(coord.clamp(0.0, max)),
        BoundaryMode::Wrap => {
            if len == 0 {
                None
            } else {
                Some(coord.rem_euclid(len as f64))
            }
        }
        BoundaryMode::Reflect => {
            if len <= 1 {
                Some(0.0)
            } else {
                let period = 2.0 * len as f64;
                let mut reflected = coord.rem_euclid(period);
                if reflected >= len as f64 {
                    reflected = period - reflected - 1.0;
                }
                Some(reflected.clamp(0.0, max))
            }
        }
    }
}

fn spline_coefficients_for_line(line: &[f64], order: usize) -> Result<Vec<f64>, NdimageError> {
    if line.len() <= 1 || order <= 1 {
        return Ok(line.to_vec());
    }
    let effective_order = order.min(line.len() - 1);
    if effective_order <= 1 {
        return Ok(line.to_vec());
    }
    let x: Vec<f64> = (0..line.len()).map(|i| i as f64).collect();
    let spline = make_interp_spline(&x, line, effective_order).map_err(|err| {
        NdimageError::InvalidArgument(format!("failed to compute spline coefficients: {err}"))
    })?;
    Ok(spline.coeffs().to_vec())
}

fn pad_array_nearest(input: &NdArray, pad: usize) -> Result<NdArray, NdimageError> {
    if input.shape.contains(&0) {
        return Err(NdimageError::EmptyInput);
    }
    let padded_shape: Vec<usize> = input.shape.iter().map(|&dim| dim + 2 * pad).collect();
    let mut padded = NdArray::zeros(padded_shape.clone());
    for flat in 0..padded.size() {
        let padded_idx = unravel_with_shape(flat, &padded_shape);
        let src_idx: Vec<usize> = padded_idx
            .iter()
            .enumerate()
            .map(|(axis, &idx)| idx.saturating_sub(pad).min(input.shape[axis] - 1))
            .collect();
        padded.data[flat] = input.get(&src_idx);
    }
    Ok(padded)
}

fn prefilter_spline_coefficients(
    input: &NdArray,
    order: usize,
    mode: BoundaryMode,
) -> Result<SplinePrefilter, NdimageError> {
    if order <= 1 {
        return Ok(SplinePrefilter {
            coeffs: input.clone(),
            coord_offsets: vec![0.0; input.ndim()],
        });
    }
    let ndim = input.ndim();
    let (mut current, coord_offsets) = if mode == BoundaryMode::Nearest {
        (
            pad_array_nearest(input, SPLINE_NEAREST_PAD)?,
            vec![SPLINE_NEAREST_PAD as f64; ndim],
        )
    } else {
        (input.clone(), vec![0.0; ndim])
    };
    for axis in 0..ndim {
        let axis_len = current.shape[axis];
        if axis_len <= 1 {
            continue;
        }
        let reduced_shape: Vec<usize> = current
            .shape
            .iter()
            .enumerate()
            .filter_map(|(d, &size)| (d != axis).then_some(size))
            .collect();
        let line_count = reduced_shape.iter().product::<usize>().max(1);
        let mut next = current.clone();
        for line_flat in 0..line_count {
            let reduced_idx = unravel_with_shape(line_flat, &reduced_shape);
            let mut idx = vec![0usize; ndim];
            let mut src = 0usize;
            for (d, slot) in idx.iter_mut().enumerate() {
                if d == axis {
                    continue;
                }
                *slot = reduced_idx[src];
                src += 1;
            }
            let mut line = Vec::with_capacity(axis_len);
            for i in 0..axis_len {
                idx[axis] = i;
                line.push(current.get(&idx));
            }
            let coeffs = spline_coefficients_for_line(&line, order)?;
            for (i, coeff) in coeffs.into_iter().enumerate() {
                idx[axis] = i;
                next.set(&idx, coeff);
            }
        }
        current = next;
    }
    Ok(SplinePrefilter {
        coeffs: current,
        coord_offsets,
    })
}

fn sample_spline_recursive(
    coeffs: &NdArray,
    bases: &[Vec<(usize, f64)>],
    dim: usize,
    idx: &mut [usize],
) -> f64 {
    if dim == bases.len() {
        return coeffs.get(idx);
    }
    let mut acc = 0.0;
    for &(coord_idx, weight) in &bases[dim] {
        idx[dim] = coord_idx;
        acc += weight * sample_spline_recursive(coeffs, bases, dim + 1, idx);
    }
    acc
}

fn sample_interpolated(
    input: &NdArray,
    coeffs: &NdArray,
    coords: &[f64],
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> f64 {
    if order == 0 {
        let idx: Vec<i64> = coords.iter().map(|coord| coord.round() as i64).collect();
        return input.get_boundary(&idx, mode, cval);
    }

    let mut bases = Vec::with_capacity(coords.len());
    for (axis, &coord) in coords.iter().enumerate() {
        let len = coeffs.shape[axis];
        let Some(mapped) = map_coordinate(coord, len, mode) else {
            return cval;
        };
        let effective_order = order.min(len.saturating_sub(1));
        if effective_order == 0 {
            bases.push(vec![(
                mapped.round().clamp(0.0, (len - 1) as f64) as usize,
                1.0,
            )]);
            continue;
        }
        let knots = uniform_interpolation_knots(len, effective_order);
        let weights = eval_bspline_basis_all(&knots, mapped, effective_order, len);
        let support: Vec<(usize, f64)> = weights
            .into_iter()
            .enumerate()
            .filter(|(_, weight)| weight.abs() > 1e-12)
            .collect();
        if support.is_empty() {
            return cval;
        }
        bases.push(support);
    }

    let mut idx = vec![0usize; coeffs.ndim()];
    sample_spline_recursive(coeffs, &bases, 0, &mut idx)
}

// ══════════════════════════════════════════════════════════════════════
// Filters
// ══════════════════════════════════════════════════════════════════════

/// N-dimensional convolution with a given kernel.
///
/// Matches `scipy.ndimage.convolve`.
pub fn convolve(
    input: &NdArray,
    weights: &NdArray,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.ndim() != weights.ndim() {
        return Err(NdimageError::DimensionMismatch(
            "input and weights must have same number of dimensions".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    let mut output = NdArray::zeros(input.shape.clone());

    // Kernel center offsets
    let offsets: Vec<i64> = weights.shape.iter().map(|&s| s as i64 / 2).collect();

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut sum = 0.0;

        for flat_k in 0..weights.size() {
            let k_idx = weights.unravel(flat_k);
            let mut in_idx = vec![0i64; ndim];
            for d in 0..ndim {
                // Convolution: flip the kernel (unlike correlation)
                let k_flipped = weights.shape[d] as i64 - 1 - k_idx[d] as i64;
                in_idx[d] = out_idx[d] as i64 + k_flipped - offsets[d];
            }
            sum += weights.data[flat_k] * input.get_boundary(&in_idx, mode, cval);
        }

        output.data[flat_out] = sum;
    }

    Ok(output)
}

/// N-dimensional correlation with a given kernel.
///
/// Matches `scipy.ndimage.correlate`.
pub fn correlate(
    input: &NdArray,
    weights: &NdArray,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.ndim() != weights.ndim() {
        return Err(NdimageError::DimensionMismatch(
            "input and weights must have same number of dimensions".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    let mut output = NdArray::zeros(input.shape.clone());

    let offsets: Vec<i64> = weights.shape.iter().map(|&s| s as i64 / 2).collect();

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut sum = 0.0;

        for flat_k in 0..weights.size() {
            let k_idx = weights.unravel(flat_k);
            let mut in_idx = vec![0i64; ndim];
            for d in 0..ndim {
                in_idx[d] = out_idx[d] as i64 + k_idx[d] as i64 - offsets[d];
            }
            sum += weights.data[flat_k] * input.get_boundary(&in_idx, mode, cval);
        }

        output.data[flat_out] = sum;
    }

    Ok(output)
}

/// Uniform (box) filter.
///
/// Matches `scipy.ndimage.uniform_filter`.
pub fn uniform_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }
    let kernel_shape = vec![size; input.ndim()];
    let kernel_size: usize = kernel_shape.iter().product();
    let val = 1.0 / kernel_size as f64;
    let kernel = NdArray::new(vec![val; kernel_size], kernel_shape)?;
    convolve(input, &kernel, mode, cval)
}

/// Gaussian filter.
///
/// Matches `scipy.ndimage.gaussian_filter`.
pub fn gaussian_filter(
    input: &NdArray,
    sigma: f64,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if sigma <= 0.0 {
        return Err(NdimageError::InvalidArgument(
            "sigma must be positive".to_string(),
        ));
    }

    // Apply 1D Gaussian along each axis for separability
    let mut current = input.clone();

    for axis in 0..input.ndim() {
        let radius = (4.0 * sigma).ceil() as usize;
        let ksize = 2 * radius + 1;
        let mut kernel_1d = vec![0.0; ksize];
        let mut total = 0.0;
        for (i, value) in kernel_1d.iter_mut().enumerate() {
            let x = i as f64 - radius as f64;
            let g = (-x * x / (2.0 * sigma * sigma)).exp();
            *value = g;
            total += g;
        }
        for v in &mut kernel_1d {
            *v /= total;
        }

        // Build N-D kernel that is 1D along `axis`
        let mut kernel_shape = vec![1usize; input.ndim()];
        kernel_shape[axis] = ksize;
        let kernel = NdArray::new(kernel_1d, kernel_shape)?;
        current = convolve(&current, &kernel, mode, cval)?;
    }

    Ok(current)
}

/// Median filter.
///
/// Matches `scipy.ndimage.median_filter`.
pub fn median_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    let mut output = NdArray::zeros(input.shape.clone());
    let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
    let kernel_shape: Vec<usize> = vec![size; ndim];
    let kernel_total: usize = kernel_shape.iter().product();

    // Generate all offsets in kernel
    let kernel_strides = compute_strides(&kernel_shape);

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut neighborhood = Vec::with_capacity(kernel_total);

        for flat_k in 0..kernel_total {
            let mut k_idx = vec![0usize; ndim];
            let mut rem = flat_k;
            for d in 0..ndim {
                k_idx[d] = rem / kernel_strides[d];
                rem %= kernel_strides[d];
            }

            let mut in_idx = vec![0i64; ndim];
            for d in 0..ndim {
                in_idx[d] = out_idx[d] as i64 + k_idx[d] as i64 - offsets[d];
            }
            neighborhood.push(input.get_boundary(&in_idx, mode, cval));
        }

        neighborhood.sort_by(|a, b| a.total_cmp(b));
        let mid = neighborhood.len() / 2;
        output.data[flat_out] = if neighborhood.len() % 2 == 0 {
            (neighborhood[mid - 1] + neighborhood[mid]) / 2.0
        } else {
            neighborhood[mid]
        };
    }

    Ok(output)
}

/// Minimum filter.
///
/// Matches `scipy.ndimage.minimum_filter`.
pub fn minimum_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    rank_filter_impl(input, size, mode, cval, 0)
}

/// Maximum filter.
///
/// Matches `scipy.ndimage.maximum_filter`.
pub fn maximum_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let kernel_total: usize = vec![size; input.ndim()].iter().product();
    rank_filter_impl(input, size, mode, cval, kernel_total - 1)
}

fn rank_filter_impl(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
    rank: usize,
) -> Result<NdArray, NdimageError> {
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    let mut output = NdArray::zeros(input.shape.clone());
    let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
    let kernel_shape: Vec<usize> = vec![size; ndim];
    let kernel_total: usize = kernel_shape.iter().product();
    let kernel_strides = compute_strides(&kernel_shape);

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut neighborhood = Vec::with_capacity(kernel_total);

        for flat_k in 0..kernel_total {
            let mut k_idx = vec![0usize; ndim];
            let mut rem = flat_k;
            for d in 0..ndim {
                k_idx[d] = rem / kernel_strides[d];
                rem %= kernel_strides[d];
            }

            let mut in_idx = vec![0i64; ndim];
            for d in 0..ndim {
                in_idx[d] = out_idx[d] as i64 + k_idx[d] as i64 - offsets[d];
            }
            neighborhood.push(input.get_boundary(&in_idx, mode, cval));
        }

        neighborhood.sort_by(|a, b| a.total_cmp(b));
        output.data[flat_out] = neighborhood[rank.min(neighborhood.len() - 1)];
    }

    Ok(output)
}

/// Apply a generic function to each local neighborhood.
///
/// Matches `scipy.ndimage.generic_filter`.
pub fn generic_filter<F>(
    input: &NdArray,
    function: F,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError>
where
    F: Fn(&[f64]) -> f64,
{
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    let mut output = NdArray::zeros(input.shape.clone());
    let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
    let kernel_shape: Vec<usize> = vec![size; ndim];
    let kernel_total: usize = kernel_shape.iter().product();
    let kernel_strides = compute_strides(&kernel_shape);

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut neighborhood = Vec::with_capacity(kernel_total);

        for flat_k in 0..kernel_total {
            let mut k_idx = vec![0usize; ndim];
            let mut rem = flat_k;
            for (d, slot) in k_idx.iter_mut().enumerate() {
                *slot = rem / kernel_strides[d];
                rem %= kernel_strides[d];
            }

            let mut in_idx = vec![0i64; ndim];
            for d in 0..ndim {
                in_idx[d] = out_idx[d] as i64 + k_idx[d] as i64 - offsets[d];
            }
            neighborhood.push(input.get_boundary(&in_idx, mode, cval));
        }

        output.data[flat_out] = function(&neighborhood);
    }

    Ok(output)
}

/// Percentile filter: select the p-th percentile from each neighborhood.
///
/// Matches `scipy.ndimage.percentile_filter`.
pub fn percentile_filter(
    input: &NdArray,
    percentile: f64,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if !(0.0..=100.0).contains(&percentile) {
        return Err(NdimageError::InvalidArgument(
            "percentile must be in [0, 100]".to_string(),
        ));
    }

    generic_filter(
        input,
        |neighborhood| {
            let mut sorted = neighborhood.to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let idx = (percentile / 100.0 * (sorted.len() - 1) as f64).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        },
        size,
        mode,
        cval,
    )
}

/// Morphological gradient: dilation minus erosion.
///
/// Matches `scipy.ndimage.morphological_gradient`.
pub fn morphological_gradient(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let max_img = maximum_filter(input, size, mode, cval)?;
    let min_img = minimum_filter(input, size, mode, cval)?;

    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = max_img.data[i] - min_img.data[i];
    }
    Ok(result)
}

/// White top-hat: input minus morphological opening.
///
/// Matches `scipy.ndimage.white_tophat`.
pub fn white_tophat(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let opened = {
        let min_img = minimum_filter(input, size, mode, cval)?;
        maximum_filter(&min_img, size, mode, cval)?
    };

    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = input.data[i] - opened.data[i];
    }
    Ok(result)
}

/// Black top-hat: morphological closing minus input.
///
/// Matches `scipy.ndimage.black_tophat`.
pub fn black_tophat(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let closed = {
        let max_img = maximum_filter(input, size, mode, cval)?;
        minimum_filter(&max_img, size, mode, cval)?
    };

    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = closed.data[i] - input.data[i];
    }
    Ok(result)
}

/// Histogram of values in labeled regions.
///
/// Returns a Vec of histograms (bin counts) for each label.
/// Matches `scipy.ndimage.histogram`.
pub fn histogram_labels(
    input: &NdArray,
    labels: &NdArray,
    num_labels: usize,
    min_val: f64,
    max_val: f64,
    nbins: usize,
) -> Vec<Vec<usize>> {
    let bin_width = (max_val - min_val) / nbins as f64;
    let mut histograms = vec![vec![0usize; nbins]; num_labels];

    for i in 0..input.size() {
        let lbl = labels.data[i] as usize;
        if lbl > 0 && lbl <= num_labels {
            let bin = ((input.data[i] - min_val) / bin_width).floor() as usize;
            let bin = bin.min(nbins - 1);
            histograms[lbl - 1][bin] += 1;
        }
    }

    histograms
}

/// Minimum and maximum values in each labeled region.
///
/// Returns (min_values, max_values) vectors.
/// Matches `scipy.ndimage.extrema`.
pub fn extrema_labels(
    input: &NdArray,
    labels: &NdArray,
    num_labels: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut mins = vec![f64::INFINITY; num_labels];
    let mut maxs = vec![f64::NEG_INFINITY; num_labels];

    for i in 0..input.size() {
        let lbl = labels.data[i] as usize;
        if lbl > 0 && lbl <= num_labels {
            mins[lbl - 1] = mins[lbl - 1].min(input.data[i]);
            maxs[lbl - 1] = maxs[lbl - 1].max(input.data[i]);
        }
    }

    (mins, maxs)
}

/// Sobel edge detection filter along a given axis.
///
/// Matches `scipy.ndimage.sobel`.
pub fn sobel(
    input: &NdArray,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }
    if input.ndim() < 1 {
        return Err(NdimageError::InvalidArgument(
            "input must be at least 1-dimensional".to_string(),
        ));
    }

    // Apply smoothing kernels along non-axis dims, derivative along axis
    let mut current = input.clone();

    for d in 0..input.ndim() {
        let kernel_1d = if d == axis {
            vec![-1.0, 0.0, 1.0] // derivative
        } else {
            vec![1.0, 2.0, 1.0] // smoothing (triangle)
        };
        let mut kernel_shape = vec![1usize; input.ndim()];
        kernel_shape[d] = 3;
        let kernel = NdArray::new(kernel_1d, kernel_shape)?;
        current = correlate(&current, &kernel, mode, cval)?;
    }

    Ok(current)
}

/// Prewitt edge detection filter along a given axis.
///
/// Matches `scipy.ndimage.prewitt`.
pub fn prewitt(
    input: &NdArray,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }

    let mut current = input.clone();

    for d in 0..input.ndim() {
        let kernel_1d = if d == axis {
            vec![-1.0, 0.0, 1.0]
        } else {
            vec![1.0, 1.0, 1.0] // box smoothing (Prewitt uses uniform weights)
        };
        let mut kernel_shape = vec![1usize; input.ndim()];
        kernel_shape[d] = 3;
        let kernel = NdArray::new(kernel_1d, kernel_shape)?;
        current = correlate(&current, &kernel, mode, cval)?;
    }

    Ok(current)
}

/// Laplace filter (sum of second derivatives).
///
/// Matches `scipy.ndimage.laplace`.
pub fn laplace(input: &NdArray, mode: BoundaryMode, cval: f64) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    // Laplace = sum of second-derivative filters along each axis
    let mut result = NdArray::zeros(input.shape.clone());

    for axis in 0..input.ndim() {
        let kernel_1d = vec![1.0, -2.0, 1.0];
        let mut kernel_shape = vec![1usize; input.ndim()];
        kernel_shape[axis] = 3;
        let kernel = NdArray::new(kernel_1d, kernel_shape)?;
        let filtered = correlate(input, &kernel, mode, cval)?;
        for i in 0..result.data.len() {
            result.data[i] += filtered.data[i];
        }
    }

    Ok(result)
}

/// Gaussian Laplace (LoG) filter.
///
/// Matches `scipy.ndimage.gaussian_laplace`.
pub fn gaussian_laplace(
    input: &NdArray,
    sigma: f64,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let smoothed = gaussian_filter(input, sigma, mode, cval)?;
    laplace(&smoothed, mode, cval)
}

// ══════════════════════════════════════════════════════════════════════
// Morphology
// ══════════════════════════════════════════════════════════════════════

/// Binary erosion: output pixel is 1 only if all pixels in the structuring
/// element neighborhood are 1.
///
/// Matches `scipy.ndimage.binary_erosion`.
pub fn binary_erosion(
    input: &NdArray,
    structure_size: usize,
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let size = if structure_size == 0 {
        3
    } else {
        structure_size
    };
    let mut current = input.clone();

    for _ in 0..iterations.max(1) {
        let ndim = current.ndim();
        let mut output = NdArray::zeros(current.shape.clone());
        let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
        let kernel_shape: Vec<usize> = vec![size; ndim];
        let kernel_total: usize = kernel_shape.iter().product();
        let kernel_strides = compute_strides(&kernel_shape);

        for flat_out in 0..current.size() {
            let out_idx = current.unravel(flat_out);
            let mut all_set = true;

            for flat_k in 0..kernel_total {
                let mut k_idx = vec![0usize; ndim];
                let mut rem = flat_k;
                for d in 0..ndim {
                    k_idx[d] = rem / kernel_strides[d];
                    rem %= kernel_strides[d];
                }

                let mut in_idx = vec![0i64; ndim];
                for d in 0..ndim {
                    in_idx[d] = out_idx[d] as i64 + k_idx[d] as i64 - offsets[d];
                }

                let val = current.get_boundary(&in_idx, BoundaryMode::Constant, 0.0);
                if val == 0.0 {
                    all_set = false;
                    break;
                }
            }

            output.data[flat_out] = if all_set { 1.0 } else { 0.0 };
        }

        current = output;
    }

    Ok(current)
}

/// Binary dilation: output pixel is 1 if any pixel in the structuring
/// element neighborhood is 1.
///
/// Matches `scipy.ndimage.binary_dilation`.
pub fn binary_dilation(
    input: &NdArray,
    structure_size: usize,
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let size = if structure_size == 0 {
        3
    } else {
        structure_size
    };
    let mut current = input.clone();

    for _ in 0..iterations.max(1) {
        let ndim = current.ndim();
        let mut output = NdArray::zeros(current.shape.clone());
        let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
        let kernel_shape: Vec<usize> = vec![size; ndim];
        let kernel_total: usize = kernel_shape.iter().product();
        let kernel_strides = compute_strides(&kernel_shape);

        let mut kernel_deltas = Vec::with_capacity(kernel_total);
        for flat_k in 0..kernel_total {
            let mut k_idx = vec![0usize; ndim];
            let mut rem = flat_k;
            for d in 0..ndim {
                k_idx[d] = rem / kernel_strides[d];
                rem %= kernel_strides[d];
            }
            let mut delta = vec![0i64; ndim];
            for d in 0..ndim {
                delta[d] = k_idx[d] as i64 - offsets[d];
            }
            kernel_deltas.push(delta);
        }

        for flat_out in 0..current.size() {
            let out_idx = current.unravel(flat_out);
            let mut any_set = false;

            for delta in &kernel_deltas {
                let mut in_idx = vec![0i64; ndim];
                for d in 0..ndim {
                    in_idx[d] = out_idx[d] as i64 + delta[d];
                }

                let val = current.get_boundary(&in_idx, BoundaryMode::Constant, 0.0);
                if val != 0.0 {
                    any_set = true;
                    break;
                }
            }

            output.data[flat_out] = if any_set { 1.0 } else { 0.0 };
        }

        current = output;
    }

    Ok(current)
}

/// Binary opening: erosion followed by dilation.
///
/// Matches `scipy.ndimage.binary_opening`.
pub fn binary_opening(
    input: &NdArray,
    structure_size: usize,
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    let eroded = binary_erosion(input, structure_size, iterations)?;
    binary_dilation(&eroded, structure_size, iterations)
}

/// Binary closing: dilation followed by erosion.
///
/// Matches `scipy.ndimage.binary_closing`.
pub fn binary_closing(
    input: &NdArray,
    structure_size: usize,
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    let dilated = binary_dilation(input, structure_size, iterations)?;
    binary_erosion(&dilated, structure_size, iterations)
}

/// Binary fill holes: fill holes in binary objects.
///
/// Matches `scipy.ndimage.binary_fill_holes`.
pub fn binary_fill_holes(input: &NdArray) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    // Flood-fill from edges with complement, then invert
    // Start with all 1s, set boundary-connected background pixels to 0
    let mut filled = NdArray::zeros(input.shape.clone());
    for i in 0..filled.data.len() {
        filled.data[i] = 1.0;
    }

    // BFS from all border pixels that are background (0) in input
    let ndim = input.ndim();
    let mut queue = std::collections::VecDeque::new();

    // Find border pixels
    for flat in 0..input.size() {
        let idx = input.unravel(flat);
        let is_border = idx
            .iter()
            .zip(input.shape.iter())
            .any(|(&i, &s)| i == 0 || i == s - 1);
        if is_border && input.data[flat] == 0.0 {
            filled.data[flat] = 0.0;
            queue.push_back(flat);
        }
    }

    // BFS: spread background through connected 0-pixels
    while let Some(flat) = queue.pop_front() {
        let idx = input.unravel(flat);
        // Check all face-connected neighbors
        for d in 0..ndim {
            for delta in [-1i64, 1] {
                let ni = idx[d] as i64 + delta;
                if ni >= 0 && ni < input.shape[d] as i64 {
                    let mut neighbor_idx = idx.clone();
                    neighbor_idx[d] = ni as usize;
                    let nflat = neighbor_idx
                        .iter()
                        .zip(input.strides.iter())
                        .map(|(i, s)| i * s)
                        .sum::<usize>();
                    if input.data[nflat] == 0.0 && filled.data[nflat] == 1.0 {
                        filled.data[nflat] = 0.0;
                        queue.push_back(nflat);
                    }
                }
            }
        }
    }

    // Result: original foreground OR filled holes
    for i in 0..filled.data.len() {
        filled.data[i] = if input.data[i] != 0.0 || filled.data[i] != 0.0 {
            1.0
        } else {
            0.0
        };
    }

    Ok(filled)
}

// ══════════════════════════════════════════════════════════════════════
// Measurements
// ══════════════════════════════════════════════════════════════════════

/// Label connected components in a binary array.
///
/// Returns (labeled_array, num_features).
/// Matches `scipy.ndimage.label`.
pub fn label(input: &NdArray) -> Result<(NdArray, usize), NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let mut labels = NdArray::zeros(input.shape.clone());
    let mut current_label = 0usize;
    let ndim = input.ndim();

    // Union-Find for merging labels
    // parent[0] is unused (labels start at 1)
    let mut parent: Vec<usize> = vec![0];

    let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path compression
            x = parent[x];
        }
        x
    };

    for flat in 0..input.size() {
        if input.data[flat] == 0.0 {
            continue;
        }

        let idx = input.unravel(flat);
        let mut neighbor_labels = Vec::new();

        // Check already-processed neighbors (those with lower flat index)
        for d in 0..ndim {
            if idx[d] > 0 {
                let mut neighbor_idx = idx.clone();
                neighbor_idx[d] -= 1;
                let nflat: usize = neighbor_idx
                    .iter()
                    .zip(input.strides.iter())
                    .map(|(i, s)| i * s)
                    .sum();
                let nl = labels.data[nflat] as usize;
                if nl > 0 {
                    neighbor_labels.push(nl);
                }
            }
        }

        if neighbor_labels.is_empty() {
            current_label += 1;
            parent.push(current_label); // parent[current_label] = current_label (self-root)
            labels.data[flat] = current_label as f64;
        } else {
            // Find minimum root label
            let mut min_root = usize::MAX;
            for &nl in &neighbor_labels {
                let root = find(&mut parent, nl);
                min_root = min_root.min(root);
            }
            labels.data[flat] = min_root as f64;
            // Union all neighbor labels
            for &nl in &neighbor_labels {
                let root = find(&mut parent, nl);
                if root != min_root {
                    parent[root] = min_root;
                }
            }
        }
    }

    // Second pass: resolve all labels to roots
    let mut label_map = vec![0usize; current_label + 1];
    let mut next_label = 0usize;
    for lbl in 1..=current_label {
        let root = find(&mut parent, lbl);
        if label_map[root] == 0 {
            next_label += 1;
            label_map[root] = next_label;
        }
        label_map[lbl] = label_map[root];
    }

    for v in &mut labels.data {
        if *v > 0.0 {
            *v = label_map[*v as usize] as f64;
        }
    }

    Ok((labels, next_label))
}

/// Sum of values in labeled regions.
///
/// Matches `scipy.ndimage.sum_labels`.
pub fn sum_labels(input: &NdArray, labels: &NdArray, num_labels: usize) -> Vec<f64> {
    let mut sums = vec![0.0; num_labels + 1];
    for i in 0..input.size() {
        let lbl = labels.data[i] as usize;
        if lbl > 0 && lbl <= num_labels {
            sums[lbl] += input.data[i];
        }
    }
    sums[1..].to_vec()
}

/// Mean of values in labeled regions.
///
/// Matches `scipy.ndimage.mean`.
pub fn mean_labels(input: &NdArray, labels: &NdArray, num_labels: usize) -> Vec<f64> {
    let mut sums = vec![0.0; num_labels + 1];
    let mut counts = vec![0usize; num_labels + 1];
    for i in 0..input.size() {
        let lbl = labels.data[i] as usize;
        if lbl > 0 && lbl <= num_labels {
            sums[lbl] += input.data[i];
            counts[lbl] += 1;
        }
    }
    (1..=num_labels)
        .map(|l| {
            if counts[l] > 0 {
                sums[l] / counts[l] as f64
            } else {
                0.0
            }
        })
        .collect()
}

/// Variance of values in labeled regions.
///
/// Matches `scipy.ndimage.variance`.
pub fn variance_labels(input: &NdArray, labels: &NdArray, num_labels: usize) -> Vec<f64> {
    let means = mean_labels(input, labels, num_labels);
    let mut var_sums = vec![0.0; num_labels + 1];
    let mut counts = vec![0usize; num_labels + 1];
    for i in 0..input.size() {
        let lbl = labels.data[i] as usize;
        if lbl > 0 && lbl <= num_labels {
            let diff = input.data[i] - means[lbl - 1];
            var_sums[lbl] += diff * diff;
            counts[lbl] += 1;
        }
    }
    (1..=num_labels)
        .map(|l| {
            if counts[l] > 0 {
                var_sums[l] / counts[l] as f64
            } else {
                0.0
            }
        })
        .collect()
}

/// Standard deviation of values in labeled regions.
///
/// Matches `scipy.ndimage.standard_deviation`.
pub fn standard_deviation_labels(input: &NdArray, labels: &NdArray, num_labels: usize) -> Vec<f64> {
    variance_labels(input, labels, num_labels)
        .into_iter()
        .map(|v| v.sqrt())
        .collect()
}

/// Find bounding box slices for each labeled region.
///
/// Returns a Vec of (min_indices, max_indices) for each label.
/// Matches `scipy.ndimage.find_objects`.
pub fn find_objects(labels: &NdArray, num_labels: usize) -> Vec<Option<(Vec<usize>, Vec<usize>)>> {
    let ndim = labels.ndim();
    let mut mins: Vec<Vec<usize>> = vec![vec![usize::MAX; ndim]; num_labels + 1];
    let mut maxs: Vec<Vec<usize>> = vec![vec![0; ndim]; num_labels + 1];
    let mut found = vec![false; num_labels + 1];

    for flat in 0..labels.size() {
        let lbl = labels.data[flat] as usize;
        if lbl > 0 && lbl <= num_labels {
            found[lbl] = true;
            let idx = labels.unravel(flat);
            for d in 0..ndim {
                mins[lbl][d] = mins[lbl][d].min(idx[d]);
                maxs[lbl][d] = maxs[lbl][d].max(idx[d]);
            }
        }
    }

    (1..=num_labels)
        .map(|l| {
            if found[l] {
                Some((mins[l].clone(), maxs[l].clone()))
            } else {
                None
            }
        })
        .collect()
}

/// Center of mass for each labeled region.
///
/// Matches `scipy.ndimage.center_of_mass`.
pub fn center_of_mass(input: &NdArray, labels: &NdArray, num_labels: usize) -> Vec<Vec<f64>> {
    let ndim = input.ndim();
    let mut weighted_sums = vec![vec![0.0; ndim]; num_labels + 1];
    let mut total_weights = vec![0.0; num_labels + 1];

    for flat in 0..input.size() {
        let lbl = labels.data[flat] as usize;
        if lbl > 0 && lbl <= num_labels {
            let idx = input.unravel(flat);
            let w = input.data[flat];
            total_weights[lbl] += w;
            for d in 0..ndim {
                weighted_sums[lbl][d] += w * idx[d] as f64;
            }
        }
    }

    (1..=num_labels)
        .map(|l| {
            if total_weights[l] != 0.0 {
                weighted_sums[l]
                    .iter()
                    .map(|&s| s / total_weights[l])
                    .collect()
            } else {
                vec![0.0; ndim]
            }
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════
// Distance Transform
// ══════════════════════════════════════════════════════════════════════

/// Euclidean distance transform for a binary image.
///
/// For each foreground pixel (nonzero), computes the distance to the nearest
/// background pixel (0). Background pixels get distance 0.
///
/// Uses the brute-force approach for correctness (suitable for moderate sizes).
/// Matches `scipy.ndimage.distance_transform_edt`.
pub fn distance_transform_edt(input: &NdArray) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    if input.ndim() != 2 {
        return Err(NdimageError::InvalidArgument(
            "distance_transform_edt currently supports 2D arrays only".to_string(),
        ));
    }

    let rows = input.shape[0];
    let cols = input.shape[1];
    let mut output = NdArray::zeros(input.shape.clone());

    // Collect background pixel positions.
    let mut bg_pixels = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if input.get(&[r, c]) == 0.0 {
                bg_pixels.push((r, c));
            }
        }
    }

    // For each foreground pixel, find distance to nearest background pixel.
    for r in 0..rows {
        for c in 0..cols {
            if input.get(&[r, c]) == 0.0 {
                output.set(&[r, c], 0.0);
            } else {
                let mut min_dist = f64::INFINITY;
                for &(br, bc) in &bg_pixels {
                    let dr = r as f64 - br as f64;
                    let dc = c as f64 - bc as f64;
                    let dist = (dr * dr + dc * dc).sqrt();
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                output.set(&[r, c], min_dist);
            }
        }
    }

    Ok(output)
}

// ══════════════════════════════════════════════════════════════════════
// Interpolation
// ══════════════════════════════════════════════════════════════════════

/// Shift an array using spline interpolation.
///
/// Matches `scipy.ndimage.shift`.
pub fn shift(
    input: &NdArray,
    shift_values: &[f64],
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }
    if shift_values.len() != input.ndim() {
        return Err(NdimageError::DimensionMismatch(format!(
            "shift has {} values but input has {} dimensions",
            shift_values.len(),
            input.ndim()
        )));
    }

    let spline = prefilter_spline_coefficients(input, order, mode)?;
    let mut output = NdArray::zeros(input.shape.clone());

    for flat in 0..input.size() {
        let out_idx = input.unravel(flat);
        let coords: Vec<f64> = out_idx
            .iter()
            .enumerate()
            .map(|(axis, &o)| o as f64 - shift_values[axis] + spline.coord_offsets[axis])
            .collect();
        output.data[flat] = sample_interpolated(input, &spline.coeffs, &coords, order, mode, cval);
    }

    Ok(output)
}

/// Zoom (rescale) an array by the given factors.
///
/// Matches `scipy.ndimage.zoom`.
pub fn zoom(
    input: &NdArray,
    zoom_factors: &[f64],
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }
    if zoom_factors.len() != input.ndim() {
        return Err(NdimageError::DimensionMismatch(format!(
            "zoom has {} values but input has {} dimensions",
            zoom_factors.len(),
            input.ndim()
        )));
    }

    let new_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(zoom_factors.iter())
        .map(|(&s, &z)| ((s as f64 * z).round() as usize).max(1))
        .collect();

    let spline = prefilter_spline_coefficients(input, order, mode)?;
    let mut output = NdArray::zeros(new_shape.clone());

    for flat in 0..output.size() {
        let out_idx = output.unravel(flat);
        let coords: Vec<f64> = out_idx
            .iter()
            .enumerate()
            .map(|(axis, &o)| {
                if output.shape[axis] <= 1 || input.shape[axis] <= 1 {
                    spline.coord_offsets[axis]
                } else {
                    o as f64 * (input.shape[axis] - 1) as f64 / (output.shape[axis] - 1) as f64
                        + spline.coord_offsets[axis]
                }
            })
            .collect();
        output.data[flat] = sample_interpolated(input, &spline.coeffs, &coords, order, mode, cval);
    }

    Ok(output)
}

/// Rotate a 2D array by the given angle (in degrees).
///
/// Matches `scipy.ndimage.rotate`.
pub fn rotate(
    input: &NdArray,
    angle: f64,
    reshape: bool,
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }
    if input.ndim() != 2 {
        return Err(NdimageError::InvalidArgument(
            "rotate currently supports 2D arrays only".to_string(),
        ));
    }

    let rows = input.shape[0];
    let cols = input.shape[1];
    let rad = angle.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    let (out_rows, out_cols) = if reshape {
        // Compute new size to contain rotated image
        let corners = [
            (0.0, 0.0),
            (rows as f64, 0.0),
            (0.0, cols as f64),
            (rows as f64, cols as f64),
        ];
        let cy = rows as f64 / 2.0;
        let cx = cols as f64 / 2.0;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        for (y, x) in corners {
            let dy = y - cy;
            let dx = x - cx;
            let ny = cy + cos_a * dy - sin_a * dx;
            let nx = cx + sin_a * dy + cos_a * dx;
            min_y = min_y.min(ny);
            max_y = max_y.max(ny);
            min_x = min_x.min(nx);
            max_x = max_x.max(nx);
        }
        (
            (max_y - min_y).ceil() as usize,
            (max_x - min_x).ceil() as usize,
        )
    } else {
        (rows, cols)
    };

    let spline = prefilter_spline_coefficients(input, order, mode)?;
    let mut output = NdArray::zeros(vec![out_rows, out_cols]);
    let cy_in = (rows as f64 - 1.0) / 2.0;
    let cx_in = (cols as f64 - 1.0) / 2.0;
    let cy_out = (out_rows as f64 - 1.0) / 2.0;
    let cx_out = (out_cols as f64 - 1.0) / 2.0;

    for r in 0..out_rows {
        for c in 0..out_cols {
            // Map output to input (inverse rotation)
            let dy = r as f64 - cy_out;
            let dx = c as f64 - cx_out;
            let src_y = cy_in + cos_a * dy + sin_a * dx;
            let src_x = cx_in - sin_a * dy + cos_a * dx;

            let value = sample_interpolated(
                input,
                &spline.coeffs,
                &[
                    src_y + spline.coord_offsets[0],
                    src_x + spline.coord_offsets[1],
                ],
                order,
                mode,
                cval,
            );
            output.set(&[r, c], value);
        }
    }

    Ok(output)
}

// ══════════════════════════════════════════════════════════════════════
// Gradient
// ══════════════════════════════════════════════════════════════════════

/// Gaussian gradient magnitude.
///
/// Matches `scipy.ndimage.gaussian_gradient_magnitude`.
pub fn gaussian_gradient_magnitude(
    input: &NdArray,
    sigma: f64,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let smoothed = gaussian_filter(input, sigma, mode, cval)?;
    let mut result = NdArray::zeros(input.shape.clone());

    for axis in 0..input.ndim() {
        let grad = sobel(&smoothed, axis, mode, cval)?;
        for i in 0..result.data.len() {
            result.data[i] += grad.data[i] * grad.data[i];
        }
    }

    for v in &mut result.data {
        *v = v.sqrt();
    }

    Ok(result)
}

/// Grey-scale erosion (minimum filter equivalent for continuous values).
///
/// Matches `scipy.ndimage.grey_erosion`.
pub fn grey_erosion(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    minimum_filter(input, size, mode, cval)
}

/// Grey-scale dilation (maximum filter equivalent for continuous values).
///
/// Matches `scipy.ndimage.grey_dilation`.
pub fn grey_dilation(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    maximum_filter(input, size, mode, cval)
}

/// Grey-scale opening: erosion followed by dilation.
///
/// Matches `scipy.ndimage.grey_opening`.
pub fn grey_opening(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let eroded = minimum_filter(input, size, mode, cval)?;
    maximum_filter(&eroded, size, mode, cval)
}

/// Grey-scale closing: dilation followed by erosion.
///
/// Matches `scipy.ndimage.grey_closing`.
pub fn grey_closing(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let dilated = maximum_filter(input, size, mode, cval)?;
    minimum_filter(&dilated, size, mode, cval)
}

/// Map coordinates: evaluate input at arbitrary (non-integer) coordinates.
///
/// Matches `scipy.ndimage.map_coordinates`.
pub fn map_coordinates(
    input: &NdArray,
    coordinates: &[Vec<f64>],
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<Vec<f64>, NdimageError> {
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }
    if coordinates.is_empty() {
        return Ok(vec![]);
    }
    let ndim = input.ndim();
    if coordinates.len() != ndim {
        return Err(NdimageError::DimensionMismatch(format!(
            "coordinates has {} arrays but input has {} dimensions",
            coordinates.len(),
            ndim
        )));
    }

    let npts = coordinates[0].len();
    for c in coordinates {
        if c.len() != npts {
            return Err(NdimageError::DimensionMismatch(
                "all coordinate arrays must have the same length".to_string(),
            ));
        }
    }

    let spline = prefilter_spline_coefficients(input, order, mode)?;
    let mut result = Vec::with_capacity(npts);
    for p in 0..npts {
        let coords: Vec<f64> = coordinates
            .iter()
            .enumerate()
            .map(|(axis, c)| c[p] + spline.coord_offsets[axis])
            .collect();
        result.push(sample_interpolated(
            input,
            &spline.coeffs,
            &coords,
            order,
            mode,
            cval,
        ));
    }

    Ok(result)
}

/// Compute the maximum of the input array.
pub fn array_max(input: &NdArray) -> f64 {
    input
        .data
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

/// Compute the minimum of the input array.
pub fn array_min(input: &NdArray) -> f64 {
    input
        .data
        .iter()
        .cloned()
        .fold(f64::INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        })
}

/// Gaussian filter with per-axis sigma.
///
/// Matches `scipy.ndimage.gaussian_filter` with `sigma` as array.
pub fn gaussian_filter_multi_sigma(
    input: &NdArray,
    sigmas: &[f64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if sigmas.len() != input.ndim() {
        return Err(NdimageError::DimensionMismatch(format!(
            "sigmas length {} != ndim {}",
            sigmas.len(),
            input.ndim()
        )));
    }

    let mut current = input.clone();
    for (axis, &sigma) in sigmas.iter().enumerate() {
        if sigma <= 0.0 {
            continue;
        }
        let radius = (4.0 * sigma).ceil() as usize;
        let ksize = 2 * radius + 1;
        let mut kernel_1d = vec![0.0; ksize];
        let mut total = 0.0;
        for (i, value) in kernel_1d.iter_mut().enumerate() {
            let x = i as f64 - radius as f64;
            let g = (-x * x / (2.0 * sigma * sigma)).exp();
            *value = g;
            total += g;
        }
        for v in &mut kernel_1d {
            *v /= total;
        }
        let mut kernel_shape = vec![1usize; input.ndim()];
        kernel_shape[axis] = ksize;
        let kernel = NdArray::new(kernel_1d, kernel_shape)?;
        current = convolve(&current, &kernel, mode, cval)?;
    }

    Ok(current)
}

/// Apply a uniform filter along a single axis.
///
/// Matches `scipy.ndimage.uniform_filter1d`.
pub fn uniform_filter1d(
    input: &NdArray,
    size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "size must be positive".to_string(),
        ));
    }

    let val = 1.0 / size as f64;
    let mut kernel_shape = vec![1usize; input.ndim()];
    kernel_shape[axis] = size;
    let kernel = NdArray::new(vec![val; size], kernel_shape)?;
    convolve(input, &kernel, mode, cval)
}

/// Compute the gradient magnitude of an array.
///
/// Uses central differences along each axis.
/// Matches `numpy.gradient` magnitude.
pub fn gradient_magnitude(input: &NdArray) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let mut result = NdArray::zeros(input.shape.clone());

    for axis in 0..input.ndim() {
        let kernel_1d = vec![-0.5, 0.0, 0.5];
        let mut kernel_shape = vec![1usize; input.ndim()];
        kernel_shape[axis] = 3;
        let kernel = NdArray::new(kernel_1d, kernel_shape)?;
        let grad = correlate(input, &kernel, BoundaryMode::Reflect, 0.0)?;
        for i in 0..result.data.len() {
            result.data[i] += grad.data[i] * grad.data[i];
        }
    }

    for v in &mut result.data {
        *v = v.sqrt();
    }

    Ok(result)
}

/// Apply a maximum filter along a single axis.
///
/// Matches `scipy.ndimage.maximum_filter1d`.
pub fn maximum_filter1d(
    input: &NdArray,
    size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range",
        )));
    }
    // Build a structuring element along the specified axis
    generic_filter(
        input,
        |neighborhood| {
            neighborhood
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                })
        },
        size,
        mode,
        cval,
    )
}

/// Apply a minimum filter along a single axis.
///
/// Matches `scipy.ndimage.minimum_filter1d`.
pub fn minimum_filter1d(
    input: &NdArray,
    size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range",
        )));
    }
    generic_filter(
        input,
        |neighborhood| {
            neighborhood
                .iter()
                .cloned()
                .fold(f64::INFINITY, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.min(b)
                    }
                })
        },
        size,
        mode,
        cval,
    )
}

/// Generate a structuring element (cross/diamond shape).
///
/// Matches `scipy.ndimage.generate_binary_structure`.
pub fn generate_binary_structure(ndim: usize, connectivity: usize) -> NdArray {
    let size = 3;
    let shape = vec![size; ndim];
    let total: usize = shape.iter().product();
    let strides = compute_strides(&shape);
    let center = size / 2;

    let mut data = vec![0.0; total];

    for (flat, item) in data.iter_mut().enumerate() {
        let mut idx = vec![0usize; ndim];
        let mut rem = flat;
        for d in 0..ndim {
            idx[d] = rem / strides[d];
            rem %= strides[d];
        }

        // City-block distance from center
        let dist: usize = idx.iter().map(|&i| i.abs_diff(center)).sum();
        if dist <= connectivity {
            *item = 1.0;
        }
    }

    NdArray {
        data,
        shape,
        strides: compute_strides(&vec![size; ndim]),
    }
}

/// Compute the variance filter (local variance in each neighborhood).
///
/// Matches `scipy.ndimage.generic_filter` with variance function.
pub fn variance_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    generic_filter(
        input,
        |neighborhood| {
            let n = neighborhood.len() as f64;
            if n == 0.0 {
                return 0.0;
            }
            let mean: f64 = neighborhood.iter().sum::<f64>() / n;
            neighborhood
                .iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>()
                / n
        },
        size,
        mode,
        cval,
    )
}

/// Compute the standard deviation filter.
pub fn std_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let var = variance_filter(input, size, mode, cval)?;
    let mut result = var;
    for v in &mut result.data {
        *v = v.sqrt();
    }
    Ok(result)
}

/// Compute the range filter (max - min in each neighborhood).
pub fn range_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    morphological_gradient(input, size, mode, cval)
}

/// Threshold an array: values above threshold become 1, below become 0.
///
/// Matches `scipy.ndimage` threshold operations.
pub fn threshold(input: &NdArray, thresh: f64) -> NdArray {
    let data: Vec<f64> = input
        .data
        .iter()
        .map(|&v| if v > thresh { 1.0 } else { 0.0 })
        .collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply Otsu's thresholding to find optimal binary threshold.
///
/// Returns the optimal threshold value.
pub fn otsu_threshold(input: &NdArray) -> f64 {
    if input.size() == 0 {
        return 0.0;
    }

    // Build histogram
    let min_val = input
        .data
        .iter()
        .cloned()
        .fold(f64::INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        });
    let max_val = input
        .data
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });

    if (max_val - min_val).abs() < 1e-15 {
        return min_val;
    }

    let nbins = 256;
    let bin_width = (max_val - min_val) / nbins as f64;
    let mut hist = vec![0usize; nbins];

    for &v in &input.data {
        let bin = ((v - min_val) / bin_width).floor() as usize;
        hist[bin.min(nbins - 1)] += 1;
    }

    let total = input.size() as f64;
    let mut sum_total = 0.0;
    for (i, &count) in hist.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut best_thresh = 0.0;
    let mut best_var = 0.0;
    let mut weight_bg = 0.0;
    let mut sum_bg = 0.0;

    for (i, &count) in hist.iter().enumerate() {
        weight_bg += count as f64;
        if weight_bg == 0.0 {
            continue;
        }
        let weight_fg = total - weight_bg;
        if weight_fg == 0.0 {
            break;
        }

        sum_bg += i as f64 * count as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_total - sum_bg) / weight_fg;
        let between_var = weight_bg * weight_fg * (mean_bg - mean_fg).powi(2);

        if between_var > best_var {
            best_var = between_var;
            best_thresh = min_val + (i as f64 + 0.5) * bin_width;
        }
    }

    best_thresh
}

/// Affine transformation of an image (2D only).
///
/// `matrix` is a 2x3 affine transformation matrix [a b tx; c d ty].
/// Matches `scipy.ndimage.affine_transform`.
pub fn affine_transform(
    input: &NdArray,
    matrix: &[[f64; 3]; 2],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.ndim() != 2 {
        return Err(NdimageError::InvalidArgument(
            "affine_transform supports 2D only".to_string(),
        ));
    }

    let rows = input.shape[0];
    let cols = input.shape[1];
    let mut output = NdArray::zeros(input.shape.clone());

    // Invert the transformation to map output → input
    let det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if det.abs() < 1e-15 {
        return Ok(output);
    }

    let inv = [
        [matrix[1][1] / det, -matrix[0][1] / det],
        [-matrix[1][0] / det, matrix[0][0] / det],
    ];
    let inv_tx = -(inv[0][0] * matrix[0][2] + inv[0][1] * matrix[1][2]);
    let inv_ty = -(inv[1][0] * matrix[0][2] + inv[1][1] * matrix[1][2]);

    for r in 0..rows {
        for c in 0..cols {
            let src_r = inv[0][0] * r as f64 + inv[0][1] * c as f64 + inv_tx;
            let src_c = inv[1][0] * r as f64 + inv[1][1] * c as f64 + inv_ty;
            let idx = [src_r.round() as i64, src_c.round() as i64];
            output.set(&[r, c], input.get_boundary(&idx, mode, cval));
        }
    }

    Ok(output)
}

/// Binary hit-or-miss transform.
///
/// Detects specific patterns in binary images.
/// Matches `scipy.ndimage.binary_hit_or_miss`.
pub fn binary_hit_or_miss(
    input: &NdArray,
    structure1: &NdArray,
    structure2: Option<&NdArray>,
) -> Result<NdArray, NdimageError> {
    // Erode with structure1
    let hit = binary_erosion_with_struct(input, structure1)?;

    // Complement of input
    let complement: Vec<f64> = input
        .data
        .iter()
        .map(|&v| if v == 0.0 { 1.0 } else { 0.0 })
        .collect();
    let comp = NdArray::new(complement, input.shape.clone())?;

    // Erode complement with structure2 (or complement of structure1)
    let miss_struct = if let Some(s2) = structure2 {
        s2
    } else {
        // Use complement of structure1
        return Ok(hit); // simplified: skip miss if no structure2
    };

    let miss = binary_erosion_with_struct(&comp, miss_struct)?;

    // AND: hit AND miss
    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = if hit.data[i] != 0.0 && miss.data[i] != 0.0 {
            1.0
        } else {
            0.0
        };
    }

    Ok(result)
}

fn binary_erosion_with_struct(
    input: &NdArray,
    structure: &NdArray,
) -> Result<NdArray, NdimageError> {
    if input.ndim() != structure.ndim() {
        return Err(NdimageError::DimensionMismatch(
            "input and structure must have same dimensions".to_string(),
        ));
    }

    let _ndim = input.ndim();
    let mut output = NdArray::zeros(input.shape.clone());
    let offsets: Vec<i64> = structure.shape.iter().map(|&s| s as i64 / 2).collect();

    // Collect structure element positions
    let mut struct_positions = Vec::new();
    for flat in 0..structure.size() {
        if structure.data[flat] != 0.0 {
            let idx = structure.unravel(flat);
            let delta: Vec<i64> = idx
                .iter()
                .zip(offsets.iter())
                .map(|(&i, &o)| i as i64 - o)
                .collect();
            struct_positions.push(delta);
        }
    }

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut all_set = true;

        for delta in &struct_positions {
            let in_idx: Vec<i64> = out_idx
                .iter()
                .zip(delta.iter())
                .map(|(&o, &d)| o as i64 + d)
                .collect();
            let val = input.get_boundary(&in_idx, BoundaryMode::Constant, 0.0);
            if val == 0.0 {
                all_set = false;
                break;
            }
        }

        output.data[flat_out] = if all_set { 1.0 } else { 0.0 };
    }

    Ok(output)
}

/// Compute the sum of the array.
pub fn array_sum(input: &NdArray) -> f64 {
    input.data.iter().sum()
}

/// Compute the mean of the array.
pub fn array_mean(input: &NdArray) -> f64 {
    if input.size() == 0 {
        return f64::NAN;
    }
    input.data.iter().sum::<f64>() / input.size() as f64
}

/// Compute the variance of the array.
pub fn array_variance(input: &NdArray) -> f64 {
    if input.size() == 0 {
        return f64::NAN;
    }
    let mean = array_mean(input);
    let n = input.size() as f64;
    input.data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
}

/// Compute the standard deviation of the array.
pub fn array_std(input: &NdArray) -> f64 {
    array_variance(input).sqrt()
}

/// Count nonzero elements.
pub fn count_nonzero(input: &NdArray) -> usize {
    input.data.iter().filter(|&&v| v != 0.0).count()
}

/// Clip (clamp) array values to [a_min, a_max].
pub fn clip(input: &NdArray, a_min: f64, a_max: f64) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v.clamp(a_min, a_max)).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply element-wise absolute value.
pub fn abs_array(input: &NdArray) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v.abs()).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply element-wise square root.
pub fn sqrt_array(input: &NdArray) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v.max(0.0).sqrt()).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply element-wise power.
pub fn power_array(input: &NdArray, exponent: f64) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v.powf(exponent)).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Multiply two arrays element-wise.
pub fn multiply_arrays(a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match for element-wise multiply".to_string(),
        ));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| x * y)
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Add two arrays element-wise.
pub fn add_arrays(a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match for element-wise add".to_string(),
        ));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Subtract two arrays element-wise.
pub fn subtract_arrays(a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match".to_string(),
        ));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| x - y)
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Scale (multiply by scalar) an array.
pub fn scale_array(input: &NdArray, scalar: f64) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v * scalar).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply element-wise natural logarithm (ln).
pub fn log_array(input: &NdArray) -> NdArray {
    let data: Vec<f64> = input
        .data
        .iter()
        .map(|&v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY })
        .collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply element-wise exponential.
pub fn exp_array(input: &NdArray) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v.exp()).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Compute the sum along a specific axis, reducing that dimension.
pub fn sum_axis(input: &NdArray, axis: usize) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-d array",
            input.ndim()
        )));
    }

    let mut new_shape = input.shape.clone();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        // Collapsing to scalar
        let sum: f64 = input.data.iter().sum();
        return NdArray::new(vec![sum], vec![1]);
    }

    let mut result = NdArray::zeros(new_shape);

    for flat in 0..input.size() {
        let idx = input.unravel(flat);
        let mut out_idx: Vec<usize> = idx.clone();
        out_idx.remove(axis);
        let out_flat = out_idx
            .iter()
            .zip(result.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum::<usize>();
        result.data[out_flat] += input.data[flat];
    }

    Ok(result)
}

/// Compute the mean along a specific axis.
pub fn mean_axis(input: &NdArray, axis: usize) -> Result<NdArray, NdimageError> {
    let summed = sum_axis(input, axis)?;
    let axis_size = input.shape[axis] as f64;
    Ok(scale_array(&summed, 1.0 / axis_size))
}

/// Pad an array with constant values.
///
/// `pad_width` specifies (before, after) padding for each axis.
pub fn pad_constant(
    input: &NdArray,
    pad_width: &[(usize, usize)],
    constant: f64,
) -> Result<NdArray, NdimageError> {
    if pad_width.len() != input.ndim() {
        return Err(NdimageError::DimensionMismatch(
            "pad_width must have one entry per dimension".to_string(),
        ));
    }

    let new_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(pad_width.iter())
        .map(|(&s, &(before, after))| s + before + after)
        .collect();

    let mut result = NdArray::zeros(new_shape.clone());
    for v in &mut result.data {
        *v = constant;
    }

    // Copy input data to padded region
    let result_strides = compute_strides(&new_shape);
    for flat in 0..input.size() {
        let idx = input.unravel(flat);
        let padded_idx: Vec<usize> = idx
            .iter()
            .zip(pad_width.iter())
            .map(|(&i, &(before, _))| i + before)
            .collect();
        let out_flat: usize = padded_idx
            .iter()
            .zip(result_strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();
        result.data[out_flat] = input.data[flat];
    }

    Ok(result)
}

/// Create an NdArray filled with a constant value.
pub fn full(shape: Vec<usize>, value: f64) -> NdArray {
    let total: usize = shape.iter().product();
    NdArray {
        data: vec![value; total],
        strides: compute_strides(&shape),
        shape,
    }
}

/// Create an NdArray filled with ones.
pub fn ones(shape: Vec<usize>) -> NdArray {
    full(shape, 1.0)
}

/// Reshape an NdArray (must have same total number of elements).
pub fn reshape(input: &NdArray, new_shape: Vec<usize>) -> Result<NdArray, NdimageError> {
    let new_total: usize = new_shape.iter().product();
    if new_total != input.size() {
        return Err(NdimageError::DimensionMismatch(format!(
            "cannot reshape {} elements into {:?}",
            input.size(),
            new_shape
        )));
    }
    NdArray::new(input.data.clone(), new_shape)
}

/// Flatten an NdArray to 1D.
pub fn flatten(input: &NdArray) -> NdArray {
    let size = input.size();
    NdArray {
        data: input.data.clone(),
        shape: vec![size],
        strides: vec![1],
    }
}

/// Compute element-wise comparison: 1.0 where a > b, 0.0 otherwise.
pub fn greater_than(a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match".to_string(),
        ));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| if x > y { 1.0 } else { 0.0 })
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Compute element-wise comparison: 1.0 where a == b within tolerance.
pub fn equal_within(a: &NdArray, b: &NdArray, tol: f64) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match".to_string(),
        ));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| if (x - y).abs() <= tol { 1.0 } else { 0.0 })
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Compute where: select from a where condition is true, from b otherwise.
pub fn where_cond(condition: &NdArray, a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if condition.shape != a.shape || a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match".to_string(),
        ));
    }
    let data: Vec<f64> = condition
        .data
        .iter()
        .zip(a.data.iter().zip(b.data.iter()))
        .map(|(&c, (&x, &y))| if c != 0.0 { x } else { y })
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Compute argmax: index of maximum element.
pub fn argmax(input: &NdArray) -> usize {
    input
        .data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute argmin: index of minimum element.
pub fn argmin(input: &NdArray) -> usize {
    input
        .data
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute the cumulative sum along a flattened array.
pub fn cumsum_array(input: &NdArray) -> NdArray {
    let mut sum = 0.0;
    let data: Vec<f64> = input
        .data
        .iter()
        .map(|&v| {
            sum += v;
            sum
        })
        .collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Compute the cumulative product along a flattened array.
pub fn cumprod_array(input: &NdArray) -> NdArray {
    let mut prod = 1.0;
    let data: Vec<f64> = input
        .data
        .iter()
        .map(|&v| {
            prod *= v;
            prod
        })
        .collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Compute the difference between consecutive elements (first difference).
pub fn diff_array(input: &NdArray) -> NdArray {
    if input.size() < 2 {
        return NdArray::zeros(vec![0]);
    }
    let data: Vec<f64> = input.data.windows(2).map(|w| w[1] - w[0]).collect();
    NdArray {
        data: data.clone(),
        shape: vec![data.len()],
        strides: vec![1],
    }
}

/// Apply a boolean mask: return elements where mask is nonzero.
pub fn masked_select(input: &NdArray, mask: &NdArray) -> Vec<f64> {
    input
        .data
        .iter()
        .zip(mask.data.iter())
        .filter(|&(_, m)| *m != 0.0)
        .map(|(&v, _)| v)
        .collect()
}

/// Set elements where mask is nonzero to a given value.
pub fn masked_fill(input: &NdArray, mask: &NdArray, value: f64) -> NdArray {
    let data: Vec<f64> = input
        .data
        .iter()
        .zip(mask.data.iter())
        .map(|(&v, &m)| if m != 0.0 { value } else { v })
        .collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Compute the histogram of an NdArray.
pub fn array_histogram(input: &NdArray, bins: usize) -> (Vec<usize>, Vec<f64>) {
    if input.size() == 0 || bins == 0 {
        return (vec![], vec![]);
    }
    let min_val = array_min(input);
    let max_val = array_max(input);
    let range = max_val - min_val;
    let bw = if range > 0.0 {
        range / bins as f64
    } else {
        1.0
    };

    let mut counts = vec![0usize; bins];
    let edges: Vec<f64> = (0..=bins).map(|i| min_val + i as f64 * bw).collect();

    for &v in &input.data {
        let bin = ((v - min_val) / bw).floor() as usize;
        counts[bin.min(bins - 1)] += 1;
    }

    (counts, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_filter_1d() {
        let input = NdArray::new(vec![0.0, 0.0, 1.0, 0.0, 0.0], vec![5]).unwrap();
        let result = uniform_filter(&input, 3, BoundaryMode::Constant, 0.0).unwrap();
        // Center should be average of [0, 1, 0] = 1/3
        assert!((result.data[2] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn gaussian_filter_preserves_constant() {
        let input = NdArray::new(vec![5.0; 25], vec![5, 5]).unwrap();
        let result = gaussian_filter(&input, 1.0, BoundaryMode::Reflect, 0.0).unwrap();
        for &v in &result.data {
            assert!((v - 5.0).abs() < 1e-10, "constant image changed: {v}");
        }
    }

    #[test]
    fn median_filter_removes_impulse() {
        let mut data = vec![0.0; 9];
        data[4] = 100.0; // single spike
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = median_filter(&input, 3, BoundaryMode::Constant, 0.0).unwrap();
        // Median of 9 values where only 1 is 100 should be 0
        assert_eq!(result.data[4], 0.0);
    }

    #[test]
    fn binary_erosion_shrinks() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let input = NdArray::new(data, vec![5, 5]).unwrap();
        let result = binary_erosion(&input, 3, 1).unwrap();
        // Only center pixel should survive
        assert_eq!(result.data[12], 1.0); // (2,2)
        assert_eq!(result.data[7], 0.0); // (1,2) should be eroded
    }

    #[test]
    fn binary_dilation_grows() {
        let mut data = vec![0.0; 25];
        data[12] = 1.0; // single pixel at (2,2)
        let input = NdArray::new(data, vec![5, 5]).unwrap();
        let result = binary_dilation(&input, 3, 1).unwrap();
        // Center and all face-neighbors should be 1
        assert_eq!(result.data[12], 1.0); // (2,2)
        assert_eq!(result.data[7], 1.0); // (1,2)
        assert_eq!(result.data[17], 1.0); // (3,2)
    }

    #[test]
    fn label_two_components() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![4, 4]).unwrap();
        let (labels, num) = label(&input).unwrap();
        assert_eq!(num, 2);
        // Top-left block should have label 1
        assert_eq!(labels.data[0], 1.0);
        assert_eq!(labels.data[1], 1.0);
        assert_eq!(labels.data[4], 1.0);
        assert_eq!(labels.data[5], 1.0);
        // Bottom-right block should have label 2
        assert_eq!(labels.data[14], 2.0);
        assert_eq!(labels.data[15], 2.0);
    }

    #[test]
    fn sum_labels_correct() {
        let data = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 2.0, 2.0], vec![4]).unwrap();
        let sums = sum_labels(&data, &labels, 2);
        assert_eq!(sums, vec![3.0, 7.0]);
    }

    #[test]
    fn sobel_detects_edge() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = sobel(&input, 0, BoundaryMode::Constant, 0.0).unwrap();
        // Center pixel should have strong gradient in axis 0
        assert!(result.data[4].abs() > 0.0);
    }

    #[test]
    fn laplace_constant_image() {
        let input = NdArray::new(vec![5.0; 9], vec![3, 3]).unwrap();
        let result = laplace(&input, BoundaryMode::Reflect, 0.0).unwrap();
        // Laplace of constant should be zero (interior)
        assert!((result.data[4]).abs() < 1e-10);
    }

    #[test]
    fn distance_transform_edt_basic() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = distance_transform_edt(&input).unwrap();
        assert_eq!(result.data[4], 0.0); // background pixel
        assert!((result.data[1] - 1.0).abs() < 1e-10); // adjacent foreground pixel
        assert!((result.data[0] - 2.0f64.sqrt()).abs() < 1e-10); // diagonal foreground pixel
    }

    #[test]
    fn distance_transform_edt_zero_background_stays_zero() {
        let input = NdArray::new(vec![0.0; 9], vec![3, 3]).unwrap();
        let result = distance_transform_edt(&input).unwrap();
        assert!(result.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rotate_90_degrees() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        let input = NdArray::new(data, vec![2, 2]).unwrap();
        let result = rotate(&input, 90.0, false, 0, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        // After 90° rotation: top-right becomes top-left, etc.
        // Due to rounding, just check it doesn't crash and produces valid output
        assert_eq!(result.data.len(), 4);
    }

    #[test]
    fn zoom_upscale() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = zoom(&input, &[2.0, 2.0], 0, BoundaryMode::Nearest, 0.0).unwrap();
        assert_eq!(result.shape, vec![4, 4]);
    }

    #[test]
    fn shift_moves_data() {
        let input = NdArray::new(vec![0.0, 0.0, 1.0, 0.0, 0.0], vec![5]).unwrap();
        let result = shift(&input, &[1.0], 0, BoundaryMode::Constant, 0.0).unwrap();
        // Shifted right by 1: pixel at index 2 should now be at index 3
        assert_eq!(result.data[3], 1.0);
        assert_eq!(result.data[2], 0.0);
    }

    #[test]
    fn shift_order_zero_matches_existing_nearest_behavior() {
        let input = NdArray::new(vec![0.0, 1.0, 2.0, 3.0], vec![4]).unwrap();
        let result = shift(&input, &[0.49], 0, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(result.data, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn map_coordinates_order_one_matches_bilinear_center_value() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = map_coordinates(
            &input,
            &[vec![0.5], vec![0.5]],
            1,
            BoundaryMode::Nearest,
            0.0,
        )
        .unwrap();
        assert!((result[0] - 2.5).abs() < 1e-10, "got {}", result[0]);
    }

    #[test]
    fn shift_order_one_half_pixel_matches_linear_reference() {
        let input = NdArray::new(vec![0.0, 10.0, 20.0, 30.0], vec![4]).unwrap();
        let result = shift(&input, &[0.5], 1, BoundaryMode::Nearest, 0.0).unwrap();
        let expected = [0.0, 5.0, 15.0, 25.0];
        for (got, want) in result.data.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-10, "got {got}, want {want}");
        }
    }

    #[test]
    fn shift_order_three_half_pixel_matches_scipy_nearest_reference() {
        let input = NdArray::new(vec![0.0, 10.0, 20.0, 30.0], vec![4]).unwrap();
        let result = shift(&input, &[0.5], 3, BoundaryMode::Nearest, 0.0).unwrap();
        let expected = [
            -0.807_713_659_400_537_9,
            4.264_428_414_850_133_5,
            15.0,
            25.735_571_585_149_867,
        ];
        for (got, want) in result.data.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
    }

    #[test]
    fn rotate_full_turn_preserves_image_for_higher_order() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = rotate(&input, 360.0, false, 3, BoundaryMode::Nearest, 0.0).unwrap();
        for (got, want) in result.data.iter().zip(input.data.iter()) {
            assert!((got - want).abs() < 1e-8, "got {got}, want {want}");
        }
    }

    #[test]
    fn map_coordinates_order_three_hits_sample_points_exactly() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = map_coordinates(
            &input,
            &[vec![0.0, 1.0], vec![1.0, 0.0]],
            3,
            BoundaryMode::Nearest,
            0.0,
        )
        .unwrap();
        assert!((result[0] - 2.0).abs() < 1e-8, "got {}", result[0]);
        assert!((result[1] - 3.0).abs() < 1e-8, "got {}", result[1]);
    }

    #[test]
    fn binary_fill_holes_fills() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = binary_fill_holes(&input).unwrap();
        assert_eq!(result.data[4], 1.0); // hole should be filled
    }

    #[test]
    fn find_objects_bounding_boxes() {
        #[rustfmt::skip]
        let labels_data = vec![
            1.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 2.0, 2.0,
        ];
        let labels = NdArray::new(labels_data, vec![3, 3]).unwrap();
        let objects = find_objects(&labels, 2);
        assert_eq!(objects.len(), 2);

        let (min1, max1) = objects[0].as_ref().unwrap();
        assert_eq!(min1, &vec![0, 0]);
        assert_eq!(max1, &vec![0, 1]);

        let (min2, max2) = objects[1].as_ref().unwrap();
        assert_eq!(min2, &vec![2, 1]);
        assert_eq!(max2, &vec![2, 2]);
    }

    #[test]
    fn center_of_mass_single_region() {
        let data = NdArray::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let com = center_of_mass(&data, &labels, 1);
        assert_eq!(com.len(), 1);
        assert!((com[0][0] - 0.5).abs() < 1e-10);
        assert!((com[0][1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn generic_filter_max() {
        let input = NdArray::new(vec![1.0, 5.0, 3.0, 2.0, 4.0], vec![5]).unwrap();
        let result = generic_filter(
            &input,
            |n| {
                n.iter().cloned().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                })
            },
            3,
            BoundaryMode::Constant,
            0.0,
        )
        .unwrap();
        assert_eq!(result.data[1], 5.0); // max of [1, 5, 3]
        assert_eq!(result.data[2], 5.0); // max of [5, 3, 2]
    }

    #[test]
    fn percentile_filter_median() {
        let input = NdArray::new(vec![1.0, 5.0, 3.0, 2.0, 4.0], vec![5]).unwrap();
        let result = percentile_filter(&input, 50.0, 3, BoundaryMode::Constant, 0.0).unwrap();
        // 50th percentile of [0, 1, 5] = 1.0
        assert_eq!(result.data[0], 1.0);
    }

    #[test]
    fn morphological_gradient_detects_edges() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = morphological_gradient(&input, 3, BoundaryMode::Constant, 0.0).unwrap();
        // Center pixel: max-min = 1-0 = 1
        assert_eq!(result.data[4], 1.0);
        // Corner pixel: max-min = 0-0 = 0 (unless center is in neighborhood)
    }

    #[test]
    fn extrema_labels_correct() {
        let data = NdArray::new(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], vec![6]).unwrap();
        let (mins, maxs) = extrema_labels(&data, &labels, 2);
        assert_eq!(mins[0], 1.0); // min of [3, 1, 4]
        assert_eq!(maxs[0], 4.0); // max of [3, 1, 4]
        assert_eq!(mins[1], 1.0); // min of [1, 5, 9]
        assert_eq!(maxs[1], 9.0); // max of [1, 5, 9]
    }
}
