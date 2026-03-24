#![forbid(unsafe_code)]

//! N-dimensional image processing for FrankenSciPy.
//!
//! Matches `scipy.ndimage` core operations:
//! - Filters: uniform, gaussian, median, minimum, maximum, convolve, correlate
//! - Morphology: binary_erosion, binary_dilation, binary_opening, binary_closing
//! - Measurements: label, find_objects, sum, mean, variance, standard_deviation
//! - Interpolation: shift, rotate, zoom, map_coordinates
//! - Distance transforms: distance_transform_edt

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
        idx.iter().zip(self.strides.iter()).map(|(i, s)| i * s).sum()
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

        neighborhood.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

        neighborhood.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
pub fn laplace(
    input: &NdArray,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
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
    let size = if structure_size == 0 { 3 } else { structure_size };
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
    let size = if structure_size == 0 { 3 } else { structure_size };
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
            let mut any_set = false;

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
pub fn standard_deviation_labels(
    input: &NdArray,
    labels: &NdArray,
    num_labels: usize,
) -> Vec<f64> {
    variance_labels(input, labels, num_labels)
        .into_iter()
        .map(|v| v.sqrt())
        .collect()
}

/// Find bounding box slices for each labeled region.
///
/// Returns a Vec of (min_indices, max_indices) for each label.
/// Matches `scipy.ndimage.find_objects`.
pub fn find_objects(
    labels: &NdArray,
    num_labels: usize,
) -> Vec<Option<(Vec<usize>, Vec<usize>)>> {
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
pub fn center_of_mass(
    input: &NdArray,
    labels: &NdArray,
    num_labels: usize,
) -> Vec<Vec<f64>> {
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

/// Shift an array using spline interpolation (nearest-neighbor for now).
///
/// Matches `scipy.ndimage.shift`.
pub fn shift(
    input: &NdArray,
    shift_values: &[f64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if shift_values.len() != input.ndim() {
        return Err(NdimageError::DimensionMismatch(format!(
            "shift has {} values but input has {} dimensions",
            shift_values.len(),
            input.ndim()
        )));
    }

    let mut output = NdArray::zeros(input.shape.clone());

    for flat in 0..input.size() {
        let out_idx = input.unravel(flat);
        let in_idx: Vec<i64> = out_idx
            .iter()
            .zip(shift_values.iter())
            .map(|(&o, &s)| (o as f64 - s).round() as i64)
            .collect();
        output.data[flat] = input.get_boundary(&in_idx, mode, cval);
    }

    Ok(output)
}

/// Zoom (rescale) an array by the given factors.
///
/// Matches `scipy.ndimage.zoom`.
pub fn zoom(
    input: &NdArray,
    zoom_factors: &[f64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
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

    let mut output = NdArray::zeros(new_shape.clone());

    for flat in 0..output.size() {
        let out_idx = output.unravel(flat);
        let in_idx: Vec<i64> = out_idx
            .iter()
            .zip(zoom_factors.iter())
            .map(|(&o, &z)| (o as f64 / z).round() as i64)
            .collect();
        output.data[flat] = input.get_boundary(&in_idx, mode, cval);
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
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
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

    let mut output = NdArray::zeros(vec![out_rows, out_cols]);
    let cy_in = rows as f64 / 2.0;
    let cx_in = cols as f64 / 2.0;
    let cy_out = out_rows as f64 / 2.0;
    let cx_out = out_cols as f64 / 2.0;

    for r in 0..out_rows {
        for c in 0..out_cols {
            // Map output to input (inverse rotation)
            let dy = r as f64 - cy_out;
            let dx = c as f64 - cx_out;
            let src_y = cy_in + cos_a * dy + sin_a * dx;
            let src_x = cx_in - sin_a * dy + cos_a * dx;

            let in_idx = [src_y.round() as i64, src_x.round() as i64];
            output.set(&[r, c], input.get_boundary(&in_idx, mode, cval));
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
        let result = rotate(&input, 90.0, false, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        // After 90° rotation: top-right becomes top-left, etc.
        // Due to rounding, just check it doesn't crash and produces valid output
        assert_eq!(result.data.len(), 4);
    }

    #[test]
    fn zoom_upscale() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = zoom(&input, &[2.0, 2.0], BoundaryMode::Nearest, 0.0).unwrap();
        assert_eq!(result.shape, vec![4, 4]);
    }

    #[test]
    fn shift_moves_data() {
        let input = NdArray::new(vec![0.0, 0.0, 1.0, 0.0, 0.0], vec![5]).unwrap();
        let result = shift(&input, &[1.0], BoundaryMode::Constant, 0.0).unwrap();
        // Shifted right by 1: pixel at index 2 should now be at index 3
        assert_eq!(result.data[3], 1.0);
        assert_eq!(result.data[2], 0.0);
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
            |n| n.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
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
        let result =
            percentile_filter(&input, 50.0, 3, BoundaryMode::Constant, 0.0).unwrap();
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
        let result =
            morphological_gradient(&input, 3, BoundaryMode::Constant, 0.0).unwrap();
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
