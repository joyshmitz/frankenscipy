#![forbid(unsafe_code)]

//! Spatial data structures and algorithms for FrankenSciPy.
//!
//! Matches `scipy.spatial` core types:
//! - `KDTree` — k-d tree for fast nearest-neighbor queries
//! - `Rectangle` — hyperrectangle utility used by SciPy spatial search
//! - `HalfspaceIntersection` — bounded N-D halfspace intersection
//! - `distance` — pairwise distance computations

use std::collections::BTreeMap;

use fsci_sparse::{CooMatrix, DokMatrix, Shape2D};

/// Error raised by Qhull-backed spatial operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QhullError {
    message: String,
}

impl QhullError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for QhullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for QhullError {}

/// Error type for spatial operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpatialError {
    EmptyData,
    DimensionMismatch { expected: usize, actual: usize },
    InvalidArgument(String),
    Qhull(QhullError),
}

impl std::fmt::Display for SpatialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "empty data"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidArgument(msg) => write!(f, "{msg}"),
            Self::Qhull(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for SpatialError {}

impl From<QhullError> for SpatialError {
    fn from(value: QhullError) -> Self {
        Self::Qhull(value)
    }
}

fn sparse_error_to_spatial(err: impl std::fmt::Display) -> SpatialError {
    SpatialError::InvalidArgument(err.to_string())
}

// ══════════════════════════════════════════════════════════════════════
// Distance Functions
// ══════════════════════════════════════════════════════════════════════

/// Euclidean distance between two points.
pub fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi) * (ai - bi))
        .sum::<f64>()
        .sqrt()
}

/// Squared Euclidean distance (avoids sqrt for comparisons).
pub fn sqeuclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi) * (ai - bi))
        .sum()
}

/// Manhattan (L1) distance.
pub fn cityblock(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).abs()).sum()
}

/// Chebyshev (L∞) distance.
pub fn chebyshev(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).abs())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

/// Cosine distance: 1 - cosine_similarity(a, b).
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
    let norm_a: f64 = a.iter().map(|ai| ai * ai).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|bi| bi * bi).sum::<f64>().sqrt();
    let denom = norm_a * norm_b;
    if denom == 0.0 {
        return 0.0;
    }
    1.0 - dot / denom
}

/// Minkowski distance of order `p`.
pub fn minkowski(a: &[f64], b: &[f64], p: f64) -> f64 {
    if p == f64::INFINITY {
        return chebyshev(a, b);
    }
    if p == 1.0 {
        return cityblock(a, b);
    }
    if p == 2.0 {
        return euclidean(a, b);
    }
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

/// Correlation distance: 1 - pearson_r(a, b).
///
/// Treats a and b as samples and computes 1 minus the Pearson correlation.
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if a.len() < 2 {
        return 0.0;
    }
    let mean_a: f64 = a.iter().sum::<f64>() / n;
    let mean_b: f64 = b.iter().sum::<f64>() / n;
    let mut ssab = 0.0;
    let mut ssa = 0.0;
    let mut ssb = 0.0;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        ssab += da * db;
        ssa += da * da;
        ssb += db * db;
    }
    let denom = (ssa * ssb).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    1.0 - ssab / denom
}

/// Hamming distance: fraction of elements that differ.
///
/// Matches `scipy.spatial.distance.hamming(u, v)`.
/// For real-valued vectors, counts positions where u_i != v_i.
pub fn hamming(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let mismatches = a
        .iter()
        .zip(b.iter())
        .filter(|&(&ai, &bi)| ai != bi)
        .count();
    mismatches as f64 / a.len() as f64
}

/// Jaccard dissimilarity for binary vectors.
///
/// Matches `scipy.spatial.distance.jaccard(u, v)`.
/// J(u,v) = (c_TF + c_FT) / (c_TT + c_TF + c_FT) where c_TT is count
/// of positions where both are nonzero, etc.
pub fn jaccard(a: &[f64], b: &[f64]) -> f64 {
    let mut tt = 0usize;
    let mut tf = 0usize;
    let mut ft = 0usize;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let a_nz = ai != 0.0;
        let b_nz = bi != 0.0;
        match (a_nz, b_nz) {
            (true, true) => tt += 1,
            (true, false) => tf += 1,
            (false, true) => ft += 1,
            (false, false) => {}
        }
    }
    let denom = tt + tf + ft;
    if denom == 0 {
        return 0.0;
    }
    (tf + ft) as f64 / denom as f64
}

/// Canberra distance.
///
/// Matches `scipy.spatial.distance.canberra(u, v)`.
/// d(u,v) = Σ|u_i - v_i| / (|u_i| + |v_i|)
pub fn canberra(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let denom = ai.abs() + bi.abs();
            if denom == 0.0 {
                0.0
            } else {
                (ai - bi).abs() / denom
            }
        })
        .sum()
}

/// Bray-Curtis dissimilarity.
///
/// Matches `scipy.spatial.distance.braycurtis(u, v)`.
/// d(u,v) = Σ|u_i - v_i| / Σ|u_i + v_i|
pub fn braycurtis(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).abs())
        .sum();
    let den: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai + bi).abs())
        .sum();
    if den == 0.0 { 0.0 } else { num / den }
}

fn relative_entropy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * (x / y).ln() }
}

/// Jensen-Shannon distance between two probability vectors.
///
/// Matches `scipy.spatial.distance.jensenshannon(p, q, base=None)` for
/// one-dimensional inputs. The input vectors are normalized to sum to 1.
pub fn jensenshannon(p: &[f64], q: &[f64], base: Option<f64>) -> f64 {
    let sum_p: f64 = p.iter().sum();
    let sum_q: f64 = q.iter().sum();

    let normalized_p: Vec<f64> = p.iter().map(|value| value / sum_p).collect();
    let normalized_q: Vec<f64> = q.iter().map(|value| value / sum_q).collect();

    let js = normalized_p
        .iter()
        .zip(normalized_q.iter())
        .map(|(&left, &right)| {
            let mean = (left + right) / 2.0;
            relative_entropy(left, mean) + relative_entropy(right, mean)
        })
        .sum::<f64>();

    let scaled = match base {
        Some(log_base) => js / log_base.ln(),
        None => js,
    };
    (scaled / 2.0).sqrt()
}

/// Distance metric identifiers for use with `pdist` and `cdist_metric`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    SqEuclidean,
    Cityblock,
    Chebyshev,
    Cosine,
    Correlation,
    Hamming,
    Jaccard,
    Canberra,
    Braycurtis,
}

/// Compute the distance between two points using the specified metric.
pub fn metric_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => euclidean(a, b),
        DistanceMetric::SqEuclidean => sqeuclidean(a, b),
        DistanceMetric::Cityblock => cityblock(a, b),
        DistanceMetric::Chebyshev => chebyshev(a, b),
        DistanceMetric::Cosine => cosine(a, b),
        DistanceMetric::Correlation => correlation(a, b),
        DistanceMetric::Hamming => hamming(a, b),
        DistanceMetric::Jaccard => jaccard(a, b),
        DistanceMetric::Canberra => canberra(a, b),
        DistanceMetric::Braycurtis => braycurtis(a, b),
    }
}

/// Compute pairwise distances between observations in condensed form.
///
/// Matches `scipy.spatial.distance.pdist(X, metric)`.
///
/// Returns a condensed distance vector of length n*(n-1)/2, containing
/// the upper-triangular entries in row order.
pub fn pdist(x: &[Vec<f64>], metric: DistanceMetric) -> Result<Vec<f64>, SpatialError> {
    let n = x.len();
    if n < 2 {
        return Err(SpatialError::InvalidArgument(
            "pdist requires at least 2 observations".to_string(),
        ));
    }
    let dim = x[0].len();
    for row in x.iter() {
        if row.len() != dim {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual: row.len(),
            });
        }
    }

    let mut result = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            result.push(metric_distance(&x[i], &x[j], metric));
        }
    }
    Ok(result)
}

/// Convert a condensed distance vector to a square distance matrix.
///
/// Matches `scipy.spatial.distance.squareform(y)` where y is condensed.
///
/// Input: condensed vector of length n*(n-1)/2.
/// Output: symmetric n×n matrix with zeros on diagonal.
pub fn squareform_to_matrix(condensed: &[f64]) -> Result<Vec<Vec<f64>>, SpatialError> {
    // Solve n*(n-1)/2 = len for n
    let len = condensed.len();
    let n = ((1.0 + (1.0 + 8.0 * len as f64).sqrt()) / 2.0).round() as usize;
    if n * (n - 1) / 2 != len {
        return Err(SpatialError::InvalidArgument(format!(
            "condensed vector length {len} does not correspond to a valid square matrix"
        )));
    }

    let mut matrix = vec![vec![0.0; n]; n];
    let mut idx = 0;
    #[allow(clippy::needless_range_loop)]
    for row in 0..n {
        for col in (row + 1)..n {
            let val = condensed[idx];
            matrix[row][col] = val;
            matrix[col][row] = val;
            idx += 1;
        }
    }
    Ok(matrix)
}

/// Convert a square distance matrix to condensed form.
///
/// Matches `scipy.spatial.distance.squareform(X)` where X is a matrix.
///
/// Input: symmetric n×n distance matrix.
/// Output: condensed vector of length n*(n-1)/2.
pub fn squareform_to_condensed(matrix: &[Vec<f64>]) -> Result<Vec<f64>, SpatialError> {
    let n = matrix.len();
    if n < 2 {
        return Err(SpatialError::InvalidArgument(
            "matrix must be at least 2×2".to_string(),
        ));
    }

    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for (i, row) in matrix.iter().enumerate() {
        for &val in &row[i + 1..] {
            condensed.push(val);
        }
    }
    Ok(condensed)
}

/// Compute pairwise Euclidean distance matrix.
///
/// Matches `scipy.spatial.distance.cdist(XA, XB, 'euclidean')`.
/// Returns a matrix where result[i][j] = distance(xa[i], xb[j]).
pub fn cdist(xa: &[Vec<f64>], xb: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SpatialError> {
    if xa.is_empty() || xb.is_empty() {
        return Err(SpatialError::EmptyData);
    }
    let dim = xa[0].len();
    if xb[0].len() != dim {
        return Err(SpatialError::DimensionMismatch {
            expected: dim,
            actual: xb[0].len(),
        });
    }

    cdist_metric(xa, xb, DistanceMetric::Euclidean)
}

/// Compute pairwise distance matrix with a specified metric.
///
/// Matches `scipy.spatial.distance.cdist(XA, XB, metric)`.
pub fn cdist_metric(
    xa: &[Vec<f64>],
    xb: &[Vec<f64>],
    metric: DistanceMetric,
) -> Result<Vec<Vec<f64>>, SpatialError> {
    if xa.is_empty() || xb.is_empty() {
        return Err(SpatialError::EmptyData);
    }
    let dim = xa[0].len();
    if xb[0].len() != dim {
        return Err(SpatialError::DimensionMismatch {
            expected: dim,
            actual: xb[0].len(),
        });
    }
    for row in xa.iter().skip(1) {
        if row.len() != dim {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual: row.len(),
            });
        }
    }
    for row in xb.iter().skip(1) {
        if row.len() != dim {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual: row.len(),
            });
        }
    }

    let result: Vec<Vec<f64>> = xa
        .iter()
        .map(|a| xb.iter().map(|b| metric_distance(a, b, metric)).collect())
        .collect();
    Ok(result)
}

/// Compute the full Euclidean distance matrix between two point sets.
///
/// Matches `scipy.spatial.distance_matrix(x, y)`.
///
/// This is an explicit convenience wrapper around Euclidean `cdist`.
pub fn distance_matrix(x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SpatialError> {
    cdist_metric(x, y, DistanceMetric::Euclidean)
}

fn rectangle_norm(components: &[f64], p: f64) -> Result<f64, SpatialError> {
    if components.iter().any(|value| !value.is_finite()) {
        return Err(SpatialError::InvalidArgument(
            "rectangle distances require finite components".to_owned(),
        ));
    }
    if p == f64::INFINITY {
        return Ok(components
            .iter()
            .copied()
            .fold(0.0_f64, |a: f64, b: f64| a.max(b.abs())));
    }
    if !p.is_finite() || p <= 0.0 {
        return Err(SpatialError::InvalidArgument(
            "rectangle distances require p > 0 or infinity".to_owned(),
        ));
    }
    Ok(components
        .iter()
        .map(|value| value.abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p))
}

// ══════════════════════════════════════════════════════════════════════
// Rectangle
// ══════════════════════════════════════════════════════════════════════

/// Hyperrectangle utility matching `scipy.spatial.Rectangle`.
///
/// Represents a Cartesian product of intervals with normalized bounds:
/// `mins[i] <= maxes[i]` for every axis.
#[derive(Debug, Clone, PartialEq)]
pub struct Rectangle {
    pub maxes: Vec<f64>,
    pub mins: Vec<f64>,
    pub m: usize,
}

impl Rectangle {
    /// Construct a hyperrectangle from per-axis maxima and minima.
    pub fn new(maxes: &[f64], mins: &[f64]) -> Result<Self, SpatialError> {
        if maxes.is_empty() || mins.is_empty() {
            return Err(SpatialError::EmptyData);
        }
        if maxes.len() != mins.len() {
            return Err(SpatialError::DimensionMismatch {
                expected: maxes.len(),
                actual: mins.len(),
            });
        }
        if maxes
            .iter()
            .chain(mins.iter())
            .any(|value| !value.is_finite())
        {
            return Err(SpatialError::InvalidArgument(
                "rectangle bounds must be finite".to_owned(),
            ));
        }

        let mut norm_maxes = Vec::with_capacity(maxes.len());
        let mut norm_mins = Vec::with_capacity(maxes.len());
        for (&max_v, &min_v) in maxes.iter().zip(mins.iter()) {
            norm_maxes.push(max_v.max(min_v));
            norm_mins.push(max_v.min(min_v));
        }

        Ok(Self {
            m: norm_maxes.len(),
            maxes: norm_maxes,
            mins: norm_mins,
        })
    }

    /// Total volume of the hyperrectangle.
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.maxes
            .iter()
            .zip(self.mins.iter())
            .map(|(max_v, min_v)| max_v - min_v)
            .product()
    }

    /// Split the hyperrectangle along axis `d` at coordinate `split`.
    pub fn split(&self, d: usize, split: f64) -> Result<(Self, Self), SpatialError> {
        if d >= self.m {
            return Err(SpatialError::InvalidArgument(format!(
                "split axis {d} out of bounds for dimension {}",
                self.m
            )));
        }
        if !split.is_finite() {
            return Err(SpatialError::InvalidArgument(
                "split coordinate must be finite".to_owned(),
            ));
        }

        let mut lower_maxes = self.maxes.clone();
        lower_maxes[d] = split;
        let lower = Self::new(&lower_maxes, &self.mins)?;

        let mut upper_mins = self.mins.clone();
        upper_mins[d] = split;
        let upper = Self::new(&self.maxes, &upper_mins)?;

        Ok((lower, upper))
    }

    /// Minimum distance between the rectangle and a point under Minkowski `p`.
    pub fn min_distance_point(&self, x: &[f64], p: f64) -> Result<f64, SpatialError> {
        if x.len() != self.m {
            return Err(SpatialError::DimensionMismatch {
                expected: self.m,
                actual: x.len(),
            });
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(
                "point coordinates must be finite".to_owned(),
            ));
        }

        let components: Vec<f64> = self
            .mins
            .iter()
            .zip(self.maxes.iter())
            .zip(x.iter())
            .map(|((&min_v, &max_v), &coord)| {
                if coord < min_v {
                    min_v - coord
                } else if coord > max_v {
                    coord - max_v
                } else {
                    0.0
                }
            })
            .collect();
        rectangle_norm(&components, p)
    }

    /// Maximum distance between the rectangle and a point under Minkowski `p`.
    pub fn max_distance_point(&self, x: &[f64], p: f64) -> Result<f64, SpatialError> {
        if x.len() != self.m {
            return Err(SpatialError::DimensionMismatch {
                expected: self.m,
                actual: x.len(),
            });
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(
                "point coordinates must be finite".to_owned(),
            ));
        }

        let components: Vec<f64> = self
            .mins
            .iter()
            .zip(self.maxes.iter())
            .zip(x.iter())
            .map(|((&min_v, &max_v), &coord)| (max_v - coord).abs().max((coord - min_v).abs()))
            .collect();
        rectangle_norm(&components, p)
    }

    /// Minimum distance between two rectangles under Minkowski `p`.
    pub fn min_distance_rectangle(&self, other: &Self, p: f64) -> Result<f64, SpatialError> {
        if self.m != other.m {
            return Err(SpatialError::DimensionMismatch {
                expected: self.m,
                actual: other.m,
            });
        }

        let components: Vec<f64> = self
            .mins
            .iter()
            .zip(self.maxes.iter())
            .zip(other.mins.iter().zip(other.maxes.iter()))
            .map(|((&self_min, &self_max), (&other_min, &other_max))| {
                0.0_f64.max((self_min - other_max).max(other_min - self_max))
            })
            .collect();
        rectangle_norm(&components, p)
    }

    /// Maximum distance between two rectangles under Minkowski `p`.
    pub fn max_distance_rectangle(&self, other: &Self, p: f64) -> Result<f64, SpatialError> {
        if self.m != other.m {
            return Err(SpatialError::DimensionMismatch {
                expected: self.m,
                actual: other.m,
            });
        }

        let components: Vec<f64> = self
            .mins
            .iter()
            .zip(self.maxes.iter())
            .zip(other.mins.iter().zip(other.maxes.iter()))
            .map(|((&self_min, &self_max), (&other_min, &other_max))| {
                (self_max - other_min)
                    .abs()
                    .max((other_max - self_min).abs())
            })
            .collect();
        rectangle_norm(&components, p)
    }
}

// ══════════════════════════════════════════════════════════════════════
// KDTree
// ══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
pub struct SparseDistanceMatrixRecord {
    pub i: usize,
    pub j: usize,
    pub v: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SparseDistanceMatrixOutput {
    DokMatrix(DokMatrix),
    CooMatrix(CooMatrix),
    Dict(BTreeMap<(usize, usize), f64>),
    Ndarray(Vec<SparseDistanceMatrixRecord>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SparseDistanceMatrixOutputType {
    DokMatrix,
    CooMatrix,
    Dict,
    Ndarray,
}

impl SparseDistanceMatrixOutputType {
    fn parse(output_type: &str) -> Result<Self, SpatialError> {
        match output_type {
            "dok_matrix" => Ok(Self::DokMatrix),
            "coo_matrix" => Ok(Self::CooMatrix),
            "dict" => Ok(Self::Dict),
            "ndarray" => Ok(Self::Ndarray),
            _ => Err(SpatialError::InvalidArgument(
                "Invalid output type".to_string(),
            )),
        }
    }
}

fn sparse_distance_matrix_output_from_triplets(
    shape: Shape2D,
    triplets: &[(usize, usize, f64)],
    output_type: SparseDistanceMatrixOutputType,
) -> Result<SparseDistanceMatrixOutput, SpatialError> {
    let mut rows = Vec::with_capacity(triplets.len());
    let mut cols = Vec::with_capacity(triplets.len());
    let mut values = Vec::with_capacity(triplets.len());
    for &(row, col, value) in triplets {
        rows.push(row);
        cols.push(col);
        values.push(value);
    }

    match output_type {
        SparseDistanceMatrixOutputType::DokMatrix => {
            let matrix = DokMatrix::from_triplets(shape, values, rows, cols)
                .map_err(sparse_error_to_spatial)?;
            Ok(SparseDistanceMatrixOutput::DokMatrix(matrix))
        }
        SparseDistanceMatrixOutputType::CooMatrix => {
            let matrix = CooMatrix::from_triplets(shape, values, rows, cols, false)
                .map_err(sparse_error_to_spatial)?;
            Ok(SparseDistanceMatrixOutput::CooMatrix(matrix))
        }
        SparseDistanceMatrixOutputType::Dict => {
            let entries = triplets
                .iter()
                .map(|&(row, col, value)| ((row, col), value))
                .collect();
            Ok(SparseDistanceMatrixOutput::Dict(entries))
        }
        SparseDistanceMatrixOutputType::Ndarray => {
            let entries = triplets
                .iter()
                .map(|&(i, j, v)| SparseDistanceMatrixRecord { i, j, v })
                .collect();
            Ok(SparseDistanceMatrixOutput::Ndarray(entries))
        }
    }
}

/// A k-d tree for efficient nearest-neighbor search.
///
/// Matches `scipy.spatial.KDTree(data)`.
#[derive(Debug, Clone)]
pub struct KDTree {
    nodes: Vec<KDNode>,
    dim: usize,
}

#[derive(Debug, Clone)]
struct KDNode {
    point: Vec<f64>,
    index: usize,
    left: Option<usize>,
    right: Option<usize>,
    split_dim: usize,
}

impl KDTree {
    /// Build a k-d tree from a set of points.
    ///
    /// Each point is a `dim`-dimensional vector.
    pub fn new(data: &[Vec<f64>]) -> Result<Self, SpatialError> {
        if data.is_empty() {
            return Err(SpatialError::EmptyData);
        }
        let dim = data[0].len();
        if dim == 0 {
            return Err(SpatialError::InvalidArgument(
                "points must have at least 1 dimension".to_string(),
            ));
        }
        if let Some(actual) = data.iter().map(Vec::len).find(|&len| len != dim) {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual,
            });
        }
        if data.iter().flatten().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(
                "points must be finite".to_string(),
            ));
        }

        let mut indices: Vec<usize> = (0..data.len()).collect();
        let mut nodes = Vec::with_capacity(data.len());
        build_kdtree(data, &mut indices, 0, &mut nodes, dim);

        Ok(Self { nodes, dim })
    }

    /// Find the nearest neighbor to `query`.
    /// Returns `(index, distance)`.
    pub fn query(&self, query: &[f64]) -> Result<(usize, f64), SpatialError> {
        if query.len() != self.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(
                "query must be finite".to_string(),
            ));
        }
        if self.nodes.is_empty() {
            return Err(SpatialError::EmptyData);
        }

        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        nn_search(&self.nodes, 0, query, &mut best_idx, &mut best_dist);

        Ok((best_idx, best_dist.sqrt()))
    }

    /// Find the k nearest neighbors to `query`.
    /// Returns vectors of (index, distance) sorted by distance.
    pub fn query_k(&self, query: &[f64], k: usize) -> Result<Vec<(usize, f64)>, SpatialError> {
        if query.len() != self.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(
                "query must be finite".to_string(),
            ));
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let k = k.min(self.nodes.len());
        let mut results: Vec<(usize, f64)> = Vec::with_capacity(k);

        // Simple approach: find k nearest via repeated queries with exclusion
        // For a proper implementation, use a max-heap bounded to k elements.
        knn_search(&self.nodes, 0, query, k, &mut results);

        // Sort by distance and convert from squared to actual distance
        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        for r in &mut results {
            r.1 = r.1.sqrt();
        }

        Ok(results)
    }

    /// Number of points in the tree.
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Find all points within distance `r` of a query point.
    ///
    /// Matches `scipy.spatial.KDTree.query_ball_point(x, r)`.
    ///
    /// Returns indices of all points within Euclidean distance `r`.
    pub fn query_ball_point(&self, query: &[f64], r: f64) -> Result<Vec<usize>, SpatialError> {
        if query.len() != self.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(
                "query must be finite".to_string(),
            ));
        }
        if !r.is_finite() || r < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "radius must be finite and nonnegative".to_string(),
            ));
        }
        if self.nodes.is_empty() {
            return Ok(vec![]);
        }

        let r_sq = r * r;
        let mut results = Vec::new();
        ball_search(&self.nodes, 0, query, r_sq, &mut results);
        results.sort_unstable();
        Ok(results)
    }

    /// Find all cross-tree neighbors within distance `r`.
    ///
    /// Matches `scipy.spatial.KDTree.query_ball_tree(other, r)`.
    ///
    /// Returns one neighbor list per point in `self`, where each inner vector
    /// contains indices from `other` within Euclidean distance `r`.
    pub fn query_ball_tree(&self, other: &KDTree, r: f64) -> Result<Vec<Vec<usize>>, SpatialError> {
        if self.dim != other.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: other.dim,
            });
        }
        if !r.is_finite() || r < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "radius must be finite and nonnegative".to_string(),
            ));
        }
        if self.nodes.is_empty() {
            return Ok(vec![]);
        }

        let mut results = vec![Vec::new(); self.nodes.len()];
        for node in &self.nodes {
            results[node.index] = other.query_ball_point(&node.point, r)?;
        }
        Ok(results)
    }

    /// Count pairs of points within distance `r` between this tree and another.
    ///
    /// Matches `scipy.spatial.KDTree.count_neighbors(other, r)`.
    pub fn count_neighbors(&self, other: &KDTree, r: f64) -> Result<usize, SpatialError> {
        if self.dim != other.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: other.dim,
            });
        }
        if !r.is_finite() || r < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "radius must be finite and nonnegative".to_string(),
            ));
        }
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return Ok(0);
        }
        let r_sq = r * r;
        let mut count = 0;
        for other_node in &other.nodes {
            count += ball_search_count(&self.nodes, 0, &other_node.point, r_sq);
        }
        Ok(count)
    }

    /// Compute a sparse cross-distance matrix for pairs within `max_distance`.
    ///
    /// Matches the default `scipy.spatial.KDTree.sparse_distance_matrix`
    /// container semantics by returning a DOK-style sparse matrix.
    pub fn sparse_distance_matrix(
        &self,
        other: &KDTree,
        max_distance: f64,
    ) -> Result<DokMatrix, SpatialError> {
        let output =
            self.sparse_distance_matrix_with_output_type(other, max_distance, "dok_matrix")?;
        match output {
            SparseDistanceMatrixOutput::DokMatrix(matrix) => Ok(matrix),
            _ => unreachable!("dok_matrix output type must produce a DokMatrix"),
        }
    }

    /// Compute a sparse cross-distance matrix using SciPy's `output_type` modes.
    ///
    /// Supported output types are `"dok_matrix"`, `"coo_matrix"`, `"dict"`,
    /// and `"ndarray"`. Distances use the Euclidean metric.
    pub fn sparse_distance_matrix_with_output_type(
        &self,
        other: &KDTree,
        max_distance: f64,
        output_type: &str,
    ) -> Result<SparseDistanceMatrixOutput, SpatialError> {
        let parsed_output_type = SparseDistanceMatrixOutputType::parse(output_type)?;
        let triplets = self.sparse_distance_matrix_triplets(other, max_distance)?;
        sparse_distance_matrix_output_from_triplets(
            Shape2D::new(self.size(), other.size()),
            &triplets,
            parsed_output_type,
        )
    }

    fn sparse_distance_matrix_triplets(
        &self,
        other: &KDTree,
        max_distance: f64,
    ) -> Result<Vec<(usize, usize, f64)>, SpatialError> {
        if self.dim != other.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: other.dim,
            });
        }
        if !max_distance.is_finite() || max_distance < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "max_distance must be finite and nonnegative".to_string(),
            ));
        }
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let mut other_points = vec![None; other.nodes.len()];
        for node in &other.nodes {
            other_points[node.index] = Some(node.point.as_slice());
        }

        let mut entries = Vec::new();
        for node in &self.nodes {
            for other_index in other.query_ball_point(&node.point, max_distance)? {
                let Some(other_point) = other_points[other_index] else {
                    return Err(SpatialError::InvalidArgument(
                        "kdtree internal point index mapping was inconsistent".to_string(),
                    ));
                };
                entries.push((
                    node.index,
                    other_index,
                    sqeuclidean(&node.point, other_point).sqrt(),
                ));
            }
        }
        entries.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
        Ok(entries)
    }
}

fn ball_search_count(nodes: &[KDNode], node_idx: usize, query: &[f64], r_sq: f64) -> usize {
    let node = &nodes[node_idx];
    let dist_sq = sqeuclidean(query, &node.point);

    let mut count = if dist_sq <= r_sq { 1 } else { 0 };

    let diff = query[node.split_dim] - node.point[node.split_dim];
    let (near, far) = if diff <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(near_idx) = near {
        count += ball_search_count(nodes, near_idx, query, r_sq);
    }

    if diff * diff <= r_sq
        && let Some(far_idx) = far
    {
        count += ball_search_count(nodes, far_idx, query, r_sq);
    }

    count
}

fn build_kdtree(
    data: &[Vec<f64>],
    indices: &mut [usize],
    depth: usize,
    nodes: &mut Vec<KDNode>,
    dim: usize,
) -> Option<usize> {
    if indices.is_empty() {
        return None;
    }

    let split_dim = depth % dim;

    let median = indices.len() / 2;

    // Partition by the split dimension to find the median in O(N) time
    indices.select_nth_unstable_by(median, |&a, &b| {
        data[a][split_dim].total_cmp(&data[b][split_dim])
    });

    let node_idx = nodes.len();
    let point_idx = indices[median];

    nodes.push(KDNode {
        point: data[point_idx].clone(),
        index: point_idx,
        left: None,
        right: None,
        split_dim,
    });

    let (left_indices, right_part) = indices.split_at_mut(median);
    let right_indices = &mut right_part[1..]; // skip median

    let left = build_kdtree(data, left_indices, depth + 1, nodes, dim);
    let right = build_kdtree(data, right_indices, depth + 1, nodes, dim);

    nodes[node_idx].left = left;
    nodes[node_idx].right = right;

    Some(node_idx)
}

fn nn_search(
    nodes: &[KDNode],
    node_idx: usize,
    query: &[f64],
    best_idx: &mut usize,
    best_dist_sq: &mut f64,
) {
    let node = &nodes[node_idx];
    let dist_sq = sqeuclidean(query, &node.point);

    if dist_sq < *best_dist_sq {
        *best_dist_sq = dist_sq;
        *best_idx = node.index;
    }

    let diff = query[node.split_dim] - node.point[node.split_dim];
    let (near, far) = if diff <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(near_idx) = near {
        nn_search(nodes, near_idx, query, best_idx, best_dist_sq);
    }

    // Check if the other subtree could contain a closer point
    if diff * diff < *best_dist_sq
        && let Some(far_idx) = far
    {
        nn_search(nodes, far_idx, query, best_idx, best_dist_sq);
    }
}

fn knn_search(
    nodes: &[KDNode],
    node_idx: usize,
    query: &[f64],
    k: usize,
    results: &mut Vec<(usize, f64)>,
) {
    let node = &nodes[node_idx];
    let dist_sq = sqeuclidean(query, &node.point);

    // Insert if we have room or this is closer than the worst
    let is_better = results
        .last()
        .is_none_or(|&(_, worst_dist)| dist_sq < worst_dist);
    if results.len() < k || is_better {
        let pos = results
            .binary_search_by(|probe| probe.1.total_cmp(&dist_sq))
            .unwrap_or_else(|e| e);
        if results.len() < k {
            results.insert(pos, (node.index, dist_sq));
        } else if pos < k {
            results.pop();
            results.insert(pos, (node.index, dist_sq));
        }
    }

    let diff = query[node.split_dim] - node.point[node.split_dim];
    let (near, far) = if diff <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(near_idx) = near {
        knn_search(nodes, near_idx, query, k, results);
    }

    let worst_dist = if results.len() < k {
        f64::INFINITY
    } else {
        results[k - 1].1
    };

    if diff * diff < worst_dist
        && let Some(far_idx) = far
    {
        knn_search(nodes, far_idx, query, k, results);
    }
}

fn ball_search(
    nodes: &[KDNode],
    node_idx: usize,
    query: &[f64],
    r_sq: f64,
    results: &mut Vec<usize>,
) {
    let node = &nodes[node_idx];
    let dist_sq = sqeuclidean(query, &node.point);

    if dist_sq <= r_sq {
        results.push(node.index);
    }

    let diff = query[node.split_dim] - node.point[node.split_dim];
    let (near, far) = if diff <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(near_idx) = near {
        ball_search(nodes, near_idx, query, r_sq, results);
    }

    if diff * diff <= r_sq
        && let Some(far_idx) = far
    {
        ball_search(nodes, far_idx, query, r_sq, results);
    }
}

// ══════════════════════════════════════════════════════════════════════
// Convex Hull (2D)
// ══════════════════════════════════════════════════════════════════════

/// Result of a 2D convex hull computation.
///
/// Matches `scipy.spatial.ConvexHull(points)` for 2D point sets.
#[derive(Debug, Clone)]
pub struct ConvexHull {
    /// Indices of input points that form the convex hull (in CCW order).
    pub vertices: Vec<usize>,
    /// Hull edges as pairs of vertex indices (into the original points array).
    pub simplices: Vec<(usize, usize)>,
    /// Area enclosed by the convex hull.
    pub area: f64,
    /// Perimeter of the convex hull (called "volume" in SciPy for consistency
    /// with higher dimensions, but for 2D it's the perimeter).
    pub perimeter: f64,
}

impl ConvexHull {
    /// Compute the convex hull of a set of 2D points using Andrew's monotone chain algorithm.
    ///
    /// Time complexity: O(n log n).
    ///
    /// # Arguments
    /// * `points` — Slice of (x, y) coordinate pairs.
    pub fn new(points: &[(f64, f64)]) -> Result<Self, SpatialError> {
        if points.len() < 3 {
            return Err(qhull_error(
                "QH6214 qhull input error: not enough points to construct initial simplex",
            ));
        }

        let n = points.len();

        // Sort points by x, then by y (lexicographic)
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            points[a]
                .0
                .total_cmp(&points[b].0)
                .then(points[a].1.total_cmp(&points[b].1))
        });

        // Build lower hull
        let mut lower: Vec<usize> = Vec::new();
        for &idx in &sorted_indices {
            while lower.len() >= 2
                && cross(
                    points[lower[lower.len() - 2]],
                    points[lower[lower.len() - 1]],
                    points[idx],
                ) <= 0.0
            {
                lower.pop();
            }
            lower.push(idx);
        }

        // Build upper hull
        let mut upper: Vec<usize> = Vec::new();
        for &idx in sorted_indices.iter().rev() {
            while upper.len() >= 2
                && cross(
                    points[upper[upper.len() - 2]],
                    points[upper[upper.len() - 1]],
                    points[idx],
                ) <= 0.0
            {
                upper.pop();
            }
            upper.push(idx);
        }

        // Remove last point of each half because it's repeated
        lower.pop();
        upper.pop();

        let mut vertices: Vec<usize> = lower;
        vertices.extend(upper);

        // Handle degenerate cases (all collinear)
        if vertices.len() < 3 {
            return Err(qhull_error(
                "QH6154 Qhull precision error: initial simplex is flat",
            ));
        }

        // Compute simplices (edges)
        let nv = vertices.len();
        let simplices: Vec<(usize, usize)> = (0..nv)
            .map(|i| (vertices[i], vertices[(i + 1) % nv]))
            .collect();

        // Compute area using the shoelace formula
        let mut area = 0.0;
        for i in 0..nv {
            let j = (i + 1) % nv;
            let pi = points[vertices[i]];
            let pj = points[vertices[j]];
            area += pi.0 * pj.1 - pj.0 * pi.1;
        }
        area = area.abs() / 2.0;

        // Compute perimeter
        let mut perimeter = 0.0;
        for &(a, b) in &simplices {
            let dx = points[b].0 - points[a].0;
            let dy = points[b].1 - points[a].1;
            perimeter += (dx * dx + dy * dy).sqrt();
        }

        Ok(Self {
            vertices,
            simplices,
            area,
            perimeter,
        })
    }
}

// ══════════════════════════════════════════════════════════════════════
// Halfspace Intersection
// ══════════════════════════════════════════════════════════════════════

type Point2 = (f64, f64);
type Halfspace2 = [f64; 3];
type Equation2 = [f64; 3];
type NdVertices = Vec<Vec<f64>>;
type NdFacets = Vec<Vec<usize>>;

/// Halfspace intersection result for bounded N-D halfspaces.
///
/// Mirrors the core public attributes of
/// `scipy.spatial.HalfspaceIntersection(halfspaces, interior_point)`. The 2D
/// constructor retains the richer dual-hull metadata surface; `from_nd`
/// generalizes bounded intersections to higher dimensions with vector-backed
/// fields while leaving `dual_area`, `dual_volume`, and `dual_equations`
/// populated only for the 2D fast path.
#[derive(Debug, Clone)]
pub struct HalfspaceIntersection {
    /// Input halfspaces in SciPy row format `[a_0, ..., a_{n-1}, b]`.
    pub halfspaces: Vec<Vec<f64>>,
    /// Feasible point that must be strictly inside every halfspace.
    pub interior_point: Vec<f64>,
    /// Primal intersection vertices.
    pub intersections: Vec<Vec<f64>>,
    /// Qhull-style dual points `-A / (A * interior_point + b)`.
    pub dual_points: Vec<Vec<f64>>,
    /// Indices of dual points forming each dual convex-hull facet.
    pub dual_facets: Vec<Vec<usize>>,
    /// Indices of halfspaces that form the dual convex-hull vertices.
    pub dual_vertices: Vec<usize>,
    /// Dual hull facet equations. Populated only for the 2D fast path.
    pub dual_equations: Vec<Vec<f64>>,
    /// Perimeter of the 2D dual convex hull, matching SciPy's `dual_area`.
    /// `NaN` for higher-dimensional bounded intersections.
    pub dual_area: f64,
    /// Area of the 2D dual convex hull, matching SciPy's `dual_volume`.
    /// `NaN` for higher-dimensional bounded intersections.
    pub dual_volume: f64,
    /// Spatial dimension.
    pub ndim: usize,
    /// Number of input inequalities.
    pub nineq: usize,
    /// Whether the primal feasible region is bounded.
    pub is_bounded: bool,
}

impl HalfspaceIntersection {
    /// Compute the intersection of 2D halfspaces.
    pub fn new(halfspaces: &[Halfspace2], interior_point: Point2) -> Result<Self, SpatialError> {
        validate_halfspace_intersection_inputs(halfspaces, interior_point)?;

        let dual_points = halfspace_dual_points(halfspaces, interior_point);
        let dual_hull = ConvexHull::new(&dual_points).map_err(|err| match err {
            SpatialError::Qhull(qhull) => SpatialError::Qhull(qhull),
            other => qhull_error(format!(
                "Qhull failed to construct halfspace dual hull: {other}"
            )),
        })?;

        let dual_facets = dual_hull
            .simplices
            .iter()
            .map(|&(a, b)| vec![a, b])
            .collect::<Vec<_>>();
        let dual_equations = dual_hull
            .simplices
            .iter()
            .map(|&(a, b)| dual_edge_equation(dual_points[a], dual_points[b]))
            .collect::<Vec<_>>();
        let intersections = dual_equations
            .iter()
            .map(|equation| intersection_from_dual_equation(*equation, interior_point))
            .collect::<Vec<_>>();

        Ok(Self {
            halfspaces: halfspaces.iter().map(|row| row.to_vec()).collect(),
            interior_point: vec![interior_point.0, interior_point.1],
            intersections: intersections.into_iter().map(|(x, y)| vec![x, y]).collect(),
            dual_points: dual_points.into_iter().map(|(x, y)| vec![x, y]).collect(),
            dual_facets,
            dual_vertices: dual_hull.vertices,
            dual_equations: dual_equations.into_iter().map(|eq| eq.to_vec()).collect(),
            dual_area: dual_hull.perimeter,
            dual_volume: dual_hull.area,
            ndim: 2,
            nineq: halfspaces.len(),
            is_bounded: halfspace_region_is_bounded(halfspaces),
        })
    }

    /// Construct from SciPy-shaped rows.
    pub fn from_nd(halfspaces: &[Vec<f64>], interior_point: &[f64]) -> Result<Self, SpatialError> {
        validate_halfspace_intersection_inputs_nd(halfspaces, interior_point)?;
        let ndim = interior_point.len();
        if ndim == 2 {
            let rows = halfspaces
                .iter()
                .map(|row| [row[0], row[1], row[2]])
                .collect::<Vec<_>>();
            return Self::new(&rows, (interior_point[0], interior_point[1]));
        }

        if !halfspace_region_is_bounded_nd(halfspaces, ndim) {
            return Err(SpatialError::InvalidArgument(
                "HalfspaceIntersection currently supports only bounded regions for ndim > 2"
                    .to_string(),
            ));
        }

        let dual_points = halfspace_dual_points_nd(halfspaces, interior_point);
        let (intersections, dual_facets) = enumerate_halfspace_vertices_nd(halfspaces, ndim)?;
        let dual_vertices = collect_dual_vertices(&dual_facets, halfspaces.len());

        Ok(Self {
            halfspaces: halfspaces.to_vec(),
            interior_point: interior_point.to_vec(),
            intersections,
            dual_points,
            dual_facets,
            dual_vertices,
            dual_equations: Vec::new(),
            dual_area: f64::NAN,
            dual_volume: f64::NAN,
            ndim,
            nineq: halfspaces.len(),
            is_bounded: true,
        })
    }

    /// Recompute the intersection after appending halfspaces.
    ///
    /// The current pure-Rust implementation does not retain a live Qhull handle;
    /// `restart` is accepted for SciPy surface parity and recomputation is always
    /// deterministic from the complete halfspace set.
    pub fn add_halfspaces(
        &mut self,
        halfspaces: &[Vec<f64>],
        _restart: bool,
    ) -> Result<(), SpatialError> {
        let mut combined = self.halfspaces.clone();
        combined.extend_from_slice(halfspaces);
        *self = Self::from_nd(&combined, &self.interior_point)?;
        Ok(())
    }

    /// No-op parity method for SciPy's incremental object lifecycle.
    pub fn close(&mut self) {}
}

// ══════════════════════════════════════════════════════════════════════
// Delaunay Triangulation (2D)
// ══════════════════════════════════════════════════════════════════════

/// Result of a 2D Delaunay triangulation.
///
/// Matches the core surface of `scipy.spatial.Delaunay(points)` for 2D point
/// sets: the original points, the triangle simplices, and simplex lookup.
#[derive(Debug, Clone)]
pub struct Delaunay {
    /// Input points in the original order.
    pub points: Vec<(f64, f64)>,
    /// Triangle simplices as triples of indices into `points`.
    pub simplices: Vec<(usize, usize, usize)>,
}

impl Delaunay {
    /// Compute a 2D Delaunay triangulation with a Bowyer-Watson sweep.
    pub fn new(points: &[(f64, f64)]) -> Result<Self, SpatialError> {
        let n = points.len();
        if n < 3 {
            return Err(qhull_error(
                "QH6214 qhull input error: not enough points to construct initial simplex",
            ));
        }
        if points
            .iter()
            .any(|&(x, y)| !x.is_finite() || !y.is_finite())
        {
            return Err(SpatialError::InvalidArgument(
                "delaunay triangulation requires finite points".to_string(),
            ));
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for &(x, y) in points {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        let dx = (max_x - min_x).max(1e-10);
        let dy = (max_y - min_y).max(1e-10);
        let margin = 10.0;
        let mut all_points = points.to_vec();
        all_points.push((min_x - margin * dx, min_y - margin * dy));
        all_points.push((max_x + margin * dx, min_y - margin * dy));
        all_points.push(((min_x + max_x) / 2.0, max_y + margin * dy));

        let mut triangles = vec![(n, n + 1, n + 2)];
        for p_idx in 0..n {
            let point = all_points[p_idx];
            let mut bad = Vec::new();
            for (t_idx, &(a, b, c)) in triangles.iter().enumerate() {
                if point_in_circumcircle(all_points[a], all_points[b], all_points[c], point) {
                    bad.push(t_idx);
                }
            }

            let mut boundary = Vec::new();
            for &t_idx in &bad {
                let (a, b, c) = triangles[t_idx];
                for &(e0, e1) in &[(a, b), (b, c), (c, a)] {
                    if !bad.iter().any(|&other_idx| {
                        other_idx != t_idx
                            && triangle_has_edge(
                                triangles[other_idx].0,
                                triangles[other_idx].1,
                                triangles[other_idx].2,
                                e0,
                                e1,
                            )
                    }) {
                        boundary.push((e0, e1));
                    }
                }
            }

            bad.sort_unstable();
            for &idx in bad.iter().rev() {
                triangles.swap_remove(idx);
            }
            for &(e0, e1) in &boundary {
                triangles.push((p_idx, e0, e1));
            }
        }

        let simplices: Vec<(usize, usize, usize)> = triangles
            .into_iter()
            .filter(|&(a, b, c)| a < n && b < n && c < n)
            .collect();
        if simplices.is_empty() {
            return Err(qhull_error(
                "QH6154 Qhull precision error: initial simplex is flat",
            ));
        }

        Ok(Self {
            points: points.to_vec(),
            simplices,
        })
    }

    /// Find the simplex containing a query point.
    ///
    /// Returns the simplex index and its barycentric coordinates when the point
    /// lies in or on a triangle, or `None` if it falls outside the triangulation.
    pub fn find_simplex(&self, query: (f64, f64)) -> Option<(usize, f64, f64, f64)> {
        for (idx, &(a, b, c)) in self.simplices.iter().enumerate() {
            let (l1, l2, l3) =
                barycentric_2d(self.points[a], self.points[b], self.points[c], query);
            if l1 >= -1e-10 && l2 >= -1e-10 && l3 >= -1e-10 {
                return Some((idx, l1, l2, l3));
            }
        }
        None
    }
}

/// Cross product of vectors OA and OB where O, A, B are 2D points.
/// Positive = counter-clockwise, negative = clockwise, zero = collinear.
fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

fn point_in_circumcircle(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool {
    let (ax, ay) = (a.0 - d.0, a.1 - d.1);
    let (bx, by) = (b.0 - d.0, b.1 - d.1);
    let (cx, cy) = (c.0 - d.0, c.1 - d.1);
    let det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
        - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by))
        + (ax * ax + ay * ay) * (bx * cy - by * cx);
    let orient = cross(a, b, c);
    if orient > 0.0 { det > 0.0 } else { det < 0.0 }
}

fn triangle_has_edge(a: usize, b: usize, c: usize, e0: usize, e1: usize) -> bool {
    [(a, b), (b, c), (c, a)]
        .iter()
        .any(|&(x, y)| (x == e0 && y == e1) || (x == e1 && y == e0))
}

fn barycentric_2d(a: (f64, f64), b: (f64, f64), c: (f64, f64), p: (f64, f64)) -> (f64, f64, f64) {
    let (v0x, v0y) = (b.0 - a.0, b.1 - a.1);
    let (v1x, v1y) = (c.0 - a.0, c.1 - a.1);
    let (v2x, v2y) = (p.0 - a.0, p.1 - a.1);
    let d00 = v0x * v0x + v0y * v0y;
    let d01 = v0x * v1x + v0y * v1y;
    let d11 = v1x * v1x + v1y * v1y;
    let d20 = v2x * v0x + v2y * v0y;
    let d21 = v2x * v1x + v2y * v1y;
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-30 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let l2 = (d11 * d20 - d01 * d21) / denom;
    let l3 = (d00 * d21 - d01 * d20) / denom;
    (1.0 - l2 - l3, l2, l3)
}

fn qhull_error(message: impl Into<String>) -> SpatialError {
    SpatialError::Qhull(QhullError::new(message))
}

fn validate_halfspace_intersection_inputs(
    halfspaces: &[Halfspace2],
    interior_point: Point2,
) -> Result<(), SpatialError> {
    if halfspaces.len() < 3 {
        return Err(qhull_error(format!(
            "QH6214 qhull input error: not enough halfspaces({}) to construct initial simplex (need 3)",
            halfspaces.len()
        )));
    }
    if !interior_point.0.is_finite() || !interior_point.1.is_finite() {
        return Err(SpatialError::InvalidArgument(
            "interior_point must contain finite coordinates".to_string(),
        ));
    }

    for (idx, &[a, b, offset]) in halfspaces.iter().enumerate() {
        if !a.is_finite() || !b.is_finite() || !offset.is_finite() {
            return Err(SpatialError::InvalidArgument(format!(
                "halfspace row {idx} contains a non-finite coefficient"
            )));
        }
        let normal_norm = a.hypot(b);
        if normal_norm <= f64::EPSILON {
            return Err(qhull_error(format!(
                "QH6154 Qhull precision error: halfspace row {idx} has zero normal"
            )));
        }
        let signed_distance = a * interior_point.0 + b * interior_point.1 + offset;
        let clear_inside_tol = 1e-12 * normal_norm.max(1.0);
        if signed_distance >= -clear_inside_tol {
            return Err(qhull_error(
                "QH6023 qhull input error: feasible point is not clearly inside halfspace",
            ));
        }
    }

    Ok(())
}

fn validate_halfspace_intersection_inputs_nd(
    halfspaces: &[Vec<f64>],
    interior_point: &[f64],
) -> Result<(), SpatialError> {
    let ndim = interior_point.len();
    if ndim == 0 {
        return Err(SpatialError::InvalidArgument(
            "interior_point must not be empty".to_string(),
        ));
    }
    if halfspaces.len() < ndim + 1 {
        return Err(qhull_error(format!(
            "QH6214 qhull input error: not enough halfspaces({}) to construct initial simplex (need {})",
            halfspaces.len(),
            ndim + 1
        )));
    }
    if interior_point.iter().any(|value| !value.is_finite()) {
        return Err(SpatialError::InvalidArgument(
            "interior_point must contain finite coordinates".to_string(),
        ));
    }

    for (idx, row) in halfspaces.iter().enumerate() {
        if row.len() != ndim + 1 {
            return Err(SpatialError::DimensionMismatch {
                expected: ndim + 1,
                actual: row.len(),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(format!(
                "halfspace row {idx} contains a non-finite coefficient"
            )));
        }
        let normal_norm = row[..ndim]
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if normal_norm <= f64::EPSILON {
            return Err(qhull_error(format!(
                "QH6154 Qhull precision error: halfspace row {idx} has zero normal"
            )));
        }
        let signed_distance = row[..ndim]
            .iter()
            .zip(interior_point.iter())
            .map(|(a, x)| a * x)
            .sum::<f64>()
            + row[ndim];
        let clear_inside_tol = 1e-12 * normal_norm.max(1.0);
        if signed_distance >= -clear_inside_tol {
            return Err(qhull_error(
                "QH6023 qhull input error: feasible point is not clearly inside halfspace",
            ));
        }
    }

    Ok(())
}

fn halfspace_dual_points(halfspaces: &[Halfspace2], interior_point: Point2) -> Vec<Point2> {
    halfspaces
        .iter()
        .map(|&[a, b, offset]| {
            let distance = a * interior_point.0 + b * interior_point.1 + offset;
            (-a / distance, -b / distance)
        })
        .collect()
}

fn halfspace_dual_points_nd(halfspaces: &[Vec<f64>], interior_point: &[f64]) -> Vec<Vec<f64>> {
    let ndim = interior_point.len();
    halfspaces
        .iter()
        .map(|row| {
            let distance = row[..ndim]
                .iter()
                .zip(interior_point.iter())
                .map(|(a, x)| a * x)
                .sum::<f64>()
                + row[ndim];
            row[..ndim].iter().map(|value| -value / distance).collect()
        })
        .collect()
}

fn dual_edge_equation(lhs: Point2, rhs: Point2) -> Equation2 {
    let dx = rhs.0 - lhs.0;
    let dy = rhs.1 - lhs.1;
    let length = dx.hypot(dy);
    if length <= f64::EPSILON {
        return [f64::NAN, f64::NAN, f64::NAN];
    }

    let mut normal_x = -dy / length;
    let mut normal_y = dx / length;
    let mut offset = -(normal_x * lhs.0 + normal_y * lhs.1);
    if offset > 0.0 {
        normal_x = -normal_x;
        normal_y = -normal_y;
        offset = -offset;
    }
    [normal_x, normal_y, offset]
}

fn intersection_from_dual_equation(equation: Equation2, interior_point: Point2) -> Point2 {
    let denominator = -equation[2];
    (
        interior_point.0 + equation[0] / denominator,
        interior_point.1 + equation[1] / denominator,
    )
}

fn halfspace_region_is_bounded(halfspaces: &[Halfspace2]) -> bool {
    let mut angles = halfspaces
        .iter()
        .map(|&[a, b, _]| {
            let mut angle = b.atan2(a);
            if angle < 0.0 {
                angle += std::f64::consts::TAU;
            }
            angle
        })
        .collect::<Vec<_>>();
    angles.sort_by(f64::total_cmp);
    angles.dedup_by(|lhs, rhs| (*lhs - *rhs).abs() < 1e-12);
    if angles.len() < 3 {
        return false;
    }

    for idx in 0..angles.len() {
        let next = if idx + 1 == angles.len() {
            angles[0] + std::f64::consts::TAU
        } else {
            angles[idx + 1]
        };
        if next - angles[idx] >= std::f64::consts::PI - 1e-12 {
            return false;
        }
    }
    true
}

fn halfspace_region_is_bounded_nd(halfspaces: &[Vec<f64>], ndim: usize) -> bool {
    let normals = halfspaces
        .iter()
        .map(|row| row[..ndim].to_vec())
        .collect::<Vec<_>>();
    let mut combos = Vec::new();
    combinations_recursive(normals.len(), ndim + 1, 0, &mut Vec::new(), &mut combos);

    combos.into_iter().any(|combo| {
        let mut matrix = vec![vec![0.0; ndim + 1]; ndim + 1];
        let mut rhs = vec![0.0; ndim + 1];
        rhs[ndim] = 1.0;

        for (col, &normal_idx) in combo.iter().enumerate() {
            for row in 0..ndim {
                matrix[row][col] = normals[normal_idx][row];
            }
            matrix[ndim][col] = 1.0;
        }

        solve_linear_system(&matrix, &rhs, 1e-10)
            .is_some_and(|weights| weights.iter().all(|value| *value > 1e-9))
    })
}

fn enumerate_halfspace_vertices_nd(
    halfspaces: &[Vec<f64>],
    ndim: usize,
) -> Result<(NdVertices, NdFacets), SpatialError> {
    let mut combos = Vec::new();
    combinations_recursive(halfspaces.len(), ndim, 0, &mut Vec::new(), &mut combos);

    let mut vertices = Vec::<Vec<f64>>::new();
    let mut facets = Vec::<Vec<usize>>::new();
    for combo in combos {
        let mut matrix = vec![vec![0.0; ndim]; ndim];
        let mut rhs = vec![0.0; ndim];
        for (row_idx, &halfspace_idx) in combo.iter().enumerate() {
            matrix[row_idx].copy_from_slice(&halfspaces[halfspace_idx][..ndim]);
            rhs[row_idx] = -halfspaces[halfspace_idx][ndim];
        }

        let Some(solution) = solve_linear_system(&matrix, &rhs, 1e-10) else {
            continue;
        };
        if !point_satisfies_halfspaces(&solution, halfspaces, 1e-8) {
            continue;
        }

        if let Some(existing_idx) = vertices
            .iter()
            .position(|vertex| points_approx_eq(vertex, &solution, 1e-8))
        {
            merge_sorted_unique(&mut facets[existing_idx], &combo);
        } else {
            vertices.push(solution);
            facets.push(combo);
        }
    }

    if vertices.is_empty() {
        return Err(qhull_error(
            "QH6154 Qhull precision error: initial simplex is flat",
        ));
    }

    Ok((vertices, facets))
}

fn collect_dual_vertices(facets: &[Vec<usize>], total_halfspaces: usize) -> Vec<usize> {
    let mut present = vec![false; total_halfspaces];
    for facet in facets {
        for &index in facet {
            present[index] = true;
        }
    }
    present
        .into_iter()
        .enumerate()
        .filter_map(|(index, used)| used.then_some(index))
        .collect()
}

fn point_satisfies_halfspaces(point: &[f64], halfspaces: &[Vec<f64>], tol: f64) -> bool {
    let ndim = point.len();
    halfspaces.iter().all(|row| {
        row[..ndim]
            .iter()
            .zip(point.iter())
            .map(|(a, x)| a * x)
            .sum::<f64>()
            + row[ndim]
            <= tol
    })
}

fn points_approx_eq(lhs: &[f64], rhs: &[f64], tol: f64) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| (left - right).abs() <= tol)
}

fn merge_sorted_unique(target: &mut Vec<usize>, incoming: &[usize]) {
    for &value in incoming {
        if !target.contains(&value) {
            target.push(value);
        }
    }
    target.sort_unstable();
}

fn combinations_recursive(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        output.push(current.clone());
        return;
    }
    if start >= n {
        return;
    }
    for next in start..=n - (k - current.len()) {
        current.push(next);
        combinations_recursive(n, k, next + 1, current, output);
        current.pop();
    }
}

fn solve_linear_system(matrix: &[Vec<f64>], rhs: &[f64], tol: f64) -> Option<Vec<f64>> {
    let n = matrix.len();
    if n == 0 || rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    let mut augmented = matrix
        .iter()
        .zip(rhs.iter())
        .map(|(row, rhs_value)| {
            let mut combined = row.clone();
            combined.push(*rhs_value);
            combined
        })
        .collect::<Vec<_>>();

    for pivot in 0..n {
        let mut best_row = pivot;
        let mut best_value = augmented[pivot][pivot].abs();
        for (row_idx, row) in augmented.iter().enumerate().skip(pivot + 1) {
            let candidate = row[pivot].abs();
            if candidate > best_value {
                best_value = candidate;
                best_row = row_idx;
            }
        }
        if best_value <= tol {
            return None;
        }
        if best_row != pivot {
            augmented.swap(best_row, pivot);
        }

        let pivot_value = augmented[pivot][pivot];
        for value in augmented[pivot].iter_mut().skip(pivot) {
            *value /= pivot_value;
        }

        let pivot_row = augmented[pivot].clone();
        for (row_idx, row) in augmented.iter_mut().enumerate() {
            if row_idx == pivot {
                continue;
            }
            let factor = row[pivot];
            if factor.abs() <= tol {
                continue;
            }
            for (value, pivot_value) in row.iter_mut().zip(pivot_row.iter()).skip(pivot) {
                *value -= factor * *pivot_value;
            }
        }
    }

    Some(augmented.into_iter().map(|row| row[n]).collect())
}

// ══════════════════════════════════════════════════════════════════════
// Voronoi Diagram (2D)
// ══════════════════════════════════════════════════════════════════════

/// Result of a 2D Voronoi diagram construction.
///
/// Matches the core SciPy surface for `scipy.spatial.Voronoi(points)`:
/// input points, Voronoi vertices, ridge connectivity, and point regions.
#[derive(Debug, Clone)]
pub struct Voronoi {
    /// Input points in the original order.
    pub points: Vec<(f64, f64)>,
    /// Voronoi vertices (circumcenters of Delaunay simplices).
    pub vertices: Vec<(f64, f64)>,
    /// Input-point pairs whose dual Delaunay edge defines each Voronoi ridge.
    pub ridge_points: Vec<(usize, usize)>,
    /// Voronoi vertex pairs for each ridge. `-1` denotes an unbounded ray.
    pub ridge_vertices: Vec<(isize, isize)>,
    /// Region vertex indices for each Voronoi region. `-1` denotes infinity.
    pub regions: Vec<Vec<isize>>,
    /// Region index for each input point.
    pub point_region: Vec<usize>,
}

impl Voronoi {
    /// Compute a 2D Voronoi diagram via the dual of the Delaunay triangulation.
    pub fn new(points: &[(f64, f64)]) -> Result<Self, SpatialError> {
        let delaunay = Delaunay::new(points)?;
        let n = points.len();

        let mut vertices = Vec::with_capacity(delaunay.simplices.len());
        for &(a, b, c) in &delaunay.simplices {
            vertices.push(circumcenter_2d(points[a], points[b], points[c])?);
        }

        let mut edge_to_triangles: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();
        let mut point_to_triangles = vec![Vec::new(); n];
        for (tri_idx, &(a, b, c)) in delaunay.simplices.iter().enumerate() {
            point_to_triangles[a].push(tri_idx);
            point_to_triangles[b].push(tri_idx);
            point_to_triangles[c].push(tri_idx);
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let edge = canonical_edge(u, v);
                edge_to_triangles.entry(edge).or_default().push(tri_idx);
            }
        }

        let mut ridge_points = Vec::with_capacity(edge_to_triangles.len());
        let mut ridge_vertices = Vec::with_capacity(edge_to_triangles.len());
        let mut point_is_unbounded = vec![false; n];

        for (&(u, v), triangles) in &edge_to_triangles {
            ridge_points.push((u, v));
            match triangles.as_slice() {
                [tri_idx] => {
                    point_is_unbounded[u] = true;
                    point_is_unbounded[v] = true;
                    ridge_vertices.push((-1, *tri_idx as isize));
                }
                [left, right] => {
                    ridge_vertices.push((*left as isize, *right as isize));
                }
                _ => {
                    return Err(qhull_error(
                        "Qhull topology error: Voronoi construction encountered a non-manifold Delaunay edge",
                    ));
                }
            }
        }

        let mut regions = vec![Vec::new()];
        let mut point_region = vec![0usize; n];
        for point_idx in 0..n {
            let point = points[point_idx];
            let mut incident = point_to_triangles[point_idx].clone();
            incident.sort_by(|&lhs, &rhs| {
                let a = vertices[lhs];
                let b = vertices[rhs];
                let angle_a = (a.1 - point.1).atan2(a.0 - point.0);
                let angle_b = (b.1 - point.1).atan2(b.0 - point.0);
                angle_a.total_cmp(&angle_b)
            });

            let mut region: Vec<isize> = incident.into_iter().map(|idx| idx as isize).collect();
            if point_is_unbounded[point_idx] {
                region.insert(0, -1);
            }

            point_region[point_idx] = regions.len();
            regions.push(region);
        }

        Ok(Self {
            points: points.to_vec(),
            vertices,
            ridge_points,
            ridge_vertices,
            regions,
            point_region,
        })
    }
}

fn canonical_edge(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn circumcenter_2d(
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
) -> Result<(f64, f64), SpatialError> {
    let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
    if d.abs() < 1e-30 {
        return Err(qhull_error(
            "QH6154 Qhull precision error: initial simplex is flat",
        ));
    }

    let a2 = a.0 * a.0 + a.1 * a.1;
    let b2 = b.0 * b.0 + b.1 * b.1;
    let c2 = c.0 * c.0 + c.1 * c.1;
    let ux = (a2 * (b.1 - c.1) + b2 * (c.1 - a.1) + c2 * (a.1 - b.1)) / d;
    let uy = (a2 * (c.0 - b.0) + b2 * (a.0 - c.0) + c2 * (b.0 - a.0)) / d;
    Ok((ux, uy))
}

// ══════════════════════════════════════════════════════════════════════
// Spherical Voronoi Diagram (3D)
// ══════════════════════════════════════════════════════════════════════

/// Result of a 3D spherical Voronoi diagram for points on a common sphere.
///
/// Matches the core SciPy surface for `scipy.spatial.SphericalVoronoi`:
/// input points, Voronoi vertices, and per-point vertex regions.
#[derive(Debug, Clone)]
pub struct SphericalVoronoi {
    /// Input points in original order.
    pub points: Vec<[f64; 3]>,
    /// Voronoi vertices on the sphere.
    pub vertices: Vec<[f64; 3]>,
    /// For each input point, the ordered vertex indices of its Voronoi region.
    pub regions: Vec<Vec<usize>>,
    /// Sphere center.
    pub center: [f64; 3],
    /// Sphere radius.
    pub radius: f64,
}

impl SphericalVoronoi {
    /// Construct a spherical Voronoi diagram for 3D points on a common sphere.
    pub fn new(points: &[[f64; 3]], center: [f64; 3], radius: f64) -> Result<Self, SpatialError> {
        if points.len() < 4 {
            return Err(SpatialError::InvalidArgument(
                "spherical voronoi requires at least 4 points".to_string(),
            ));
        }
        if points.iter().flatten().any(|v| !v.is_finite()) || !center.iter().all(|v| v.is_finite())
        {
            return Err(SpatialError::InvalidArgument(
                "spherical voronoi points and center must be finite".to_string(),
            ));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(SpatialError::InvalidArgument(
                "spherical voronoi requires a positive finite radius".to_string(),
            ));
        }

        let tol = 1e-8;
        for (i, &point) in points.iter().enumerate() {
            let dist = norm3(sub3(point, center));
            if (dist - radius).abs() > tol * radius.max(1.0) {
                return Err(SpatialError::InvalidArgument(format!(
                    "point {i} is not on the specified sphere"
                )));
            }
        }
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                if norm3(sub3(points[i], points[j])) <= tol * radius.max(1.0) {
                    return Err(SpatialError::InvalidArgument(
                        "spherical voronoi requires distinct points".to_string(),
                    ));
                }
            }
        }

        let mut vertices = Vec::new();
        let mut face_indices = Vec::new();
        let n = points.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let pi = points[i];
                    let pj = points[j];
                    let pk = points[k];
                    let mut normal = cross3(sub3(pj, pi), sub3(pk, pi));
                    let normal_norm = norm3(normal);
                    if normal_norm <= 1e-12 {
                        continue;
                    }
                    if dot3(normal, sub3(pi, center)) < 0.0 {
                        normal = scale3(normal, -1.0);
                    }

                    let mut is_face = true;
                    for (idx, &point) in points.iter().enumerate() {
                        if idx == i || idx == j || idx == k {
                            continue;
                        }
                        let signed = dot3(normal, sub3(point, pi));
                        if signed > tol * radius.max(1.0) {
                            is_face = false;
                            break;
                        }
                    }
                    if !is_face {
                        continue;
                    }

                    let unit = scale3(normal, 1.0 / norm3(normal));
                    let vertex = add3(center, scale3(unit, radius));
                    if vertices
                        .iter()
                        .any(|&existing| norm3(sub3(existing, vertex)) <= tol * radius.max(1.0))
                    {
                        return Err(SpatialError::InvalidArgument(
                            "spherical voronoi requires non-coplanar generators on the sphere"
                                .to_string(),
                        ));
                    }
                    vertices.push(vertex);
                    face_indices.push((i, j, k));
                }
            }
        }

        if vertices.len() < 4 {
            return Err(SpatialError::InvalidArgument(
                "spherical voronoi requires non-degenerate 3D point configuration".to_string(),
            ));
        }

        let mut regions = vec![Vec::new(); n];
        for (vertex_idx, &(i, j, k)) in face_indices.iter().enumerate() {
            regions[i].push(vertex_idx);
            regions[j].push(vertex_idx);
            regions[k].push(vertex_idx);
        }

        for (point_idx, region) in regions.iter_mut().enumerate() {
            if region.len() < 3 {
                return Err(SpatialError::InvalidArgument(format!(
                    "point {point_idx} does not have a valid spherical Voronoi region"
                )));
            }
            sort_spherical_region(region, points[point_idx], &vertices, center);
        }

        Ok(Self {
            points: points.to_vec(),
            vertices,
            regions,
            center,
            radius,
        })
    }
}

fn sort_spherical_region(
    region: &mut [usize],
    point: [f64; 3],
    vertices: &[[f64; 3]],
    center: [f64; 3],
) {
    let point_dir = normalize3(sub3(point, center));
    let first = normalize3(project_to_tangent(
        sub3(vertices[region[0]], center),
        point_dir,
    ));
    let ortho = cross3(point_dir, first);
    region.sort_by(|&lhs, &rhs| {
        let lhs_dir = normalize3(project_to_tangent(sub3(vertices[lhs], center), point_dir));
        let rhs_dir = normalize3(project_to_tangent(sub3(vertices[rhs], center), point_dir));
        let lhs_angle = dot3(lhs_dir, ortho).atan2(dot3(lhs_dir, first));
        let rhs_angle = dot3(rhs_dir, ortho).atan2(dot3(rhs_dir, first));
        lhs_angle.total_cmp(&rhs_angle)
    });
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(a: [f64; 3], factor: f64) -> [f64; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let n = norm3(a);
    if n <= 1e-15 { a } else { scale3(a, 1.0 / n) }
}

fn project_to_tangent(v: [f64; 3], normal: [f64; 3]) -> [f64; 3] {
    sub3(v, scale3(normal, dot3(v, normal)))
}

// ══════════════════════════════════════════════════════════════════════
// Procrustes Analysis
// ══════════════════════════════════════════════════════════════════════

/// Result of Procrustes analysis.
#[derive(Debug, Clone)]
pub struct ProcrustesResult {
    /// Standardized version of data1.
    pub mtx1: Vec<Vec<f64>>,
    /// Standardized and transformed version of data2.
    pub mtx2: Vec<Vec<f64>>,
    /// Residual disparity (sum of squared differences after alignment).
    pub disparity: f64,
}

/// Procrustes analysis: find optimal rotation/reflection + scaling to align two point sets.
///
/// Both inputs must have the same shape (n × d). The data are centered, scaled
/// to unit Frobenius norm, then the optimal orthogonal transformation is found via SVD.
///
/// Matches `scipy.spatial.procrustes(data1, data2)`.
pub fn procrustes(
    data1: &[Vec<f64>],
    data2: &[Vec<f64>],
) -> Result<ProcrustesResult, SpatialError> {
    let n = data1.len();
    if n == 0 || data2.len() != n {
        return Err(SpatialError::InvalidArgument(
            "inputs must be non-empty and have same number of rows".to_string(),
        ));
    }
    let d = data1[0].len();
    if data2[0].len() != d {
        return Err(SpatialError::DimensionMismatch {
            expected: d,
            actual: data2[0].len(),
        });
    }

    // Center both datasets
    let mut mtx1 = center_matrix(data1);
    let mut mtx2 = center_matrix(data2);

    // Scale to unit Frobenius norm
    let norm1 = frobenius_norm(&mtx1);
    let norm2 = frobenius_norm(&mtx2);
    if norm1 > 0.0 {
        scale_matrix(&mut mtx1, 1.0 / norm1);
    }
    if norm2 > 0.0 {
        scale_matrix(&mut mtx2, 1.0 / norm2);
    }

    // Find optimal rotation: minimize ||mtx1 - mtx2 R||²
    // Solution: R = V U^T where mtx2^T mtx1 = U S V^T (SVD)
    // Compute M = mtx2^T mtx1 (d × d matrix)
    let mut m = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..n {
                m[i][j] += mtx2[k][i] * mtx1[k][j];
            }
        }
    }

    // Compute R = optimal rotation via M^T M eigendecomposition
    // For small d (typically 2 or 3), use iterative power method on M^T M
    // R = V U^T where M = U S V^T
    // Simpler approach: R = M (M^T M)^{-1/2}
    // Even simpler for Procrustes: use polar decomposition M = R H where R orthogonal
    // R = M (M^T M)^{-1/2}
    let mut mtm = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            for row_k in m.iter().take(d) {
                mtm[i][j] += row_k[i] * row_k[j]; // M^T M
            }
        }
    }

    // R = M * (M^T M)^{-1/2} via Denman-Beavers iteration for matrix square root inverse
    // Y_{k+1} = 0.5 * Y_k * (3I - B Y_k²) where B = M^T M, converges to B^{-1/2}
    let mut y = vec![vec![0.0; d]; d];
    for (i, row) in y.iter_mut().enumerate().take(d) {
        row[i] = 1.0;
    }

    for _ in 0..30 {
        let y2 = mat_mul(&y, &y, d);
        let by2 = mat_mul(&mtm, &y2, d);
        let mut rhs = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..d {
                rhs[i][j] = if i == j { 3.0 } else { 0.0 } - by2[i][j];
            }
        }
        let y_new = mat_mul(&y, &rhs, d);
        let mut max_diff = 0.0_f64;
        for i in 0..d {
            for j in 0..d {
                max_diff = max_diff.max((y_new[i][j] * 0.5 - y[i][j]).abs());
                y[i][j] = y_new[i][j] * 0.5;
            }
        }
        if max_diff < 1e-14 {
            break;
        }
    }
    let rotation = mat_mul(&m, &y, d);

    // Apply rotation: mtx2_aligned = mtx2 * R
    let mut aligned = vec![vec![0.0; d]; n];
    for i in 0..n {
        for j in 0..d {
            for k in 0..d {
                aligned[i][j] += mtx2[i][k] * rotation[k][j];
            }
        }
    }

    // Compute disparity: sum of squared differences
    let mut disparity = 0.0;
    for i in 0..n {
        for j in 0..d {
            disparity += (mtx1[i][j] - aligned[i][j]).powi(2);
        }
    }

    Ok(ProcrustesResult {
        mtx1,
        mtx2: aligned,
        disparity,
    })
}

fn center_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = data.len();
    let d = data[0].len();
    let mut means = vec![0.0; d];
    for row in data {
        for (j, &val) in row.iter().enumerate() {
            means[j] += val;
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }
    data.iter()
        .map(|row| row.iter().enumerate().map(|(j, &v)| v - means[j]).collect())
        .collect()
}

fn frobenius_norm(data: &[Vec<f64>]) -> f64 {
    data.iter()
        .flat_map(|row| row.iter())
        .map(|&v| v * v)
        .sum::<f64>()
        .sqrt()
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn scale_matrix(data: &mut [Vec<f64>], factor: f64) {
    for row in data.iter_mut() {
        for v in row.iter_mut() {
            *v *= factor;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Geometric SLERP
// ══════════════════════════════════════════════════════════════════════

/// Spherical linear interpolation (SLERP) between two N-dimensional unit vectors.
///
/// Interpolates along the great circle arc on the unit sphere.
///
/// Matches `scipy.spatial.geometric_slerp(start, end, t)`.
///
/// # Arguments
/// * `start` — Starting point on the unit sphere.
/// * `end` — Ending point on the unit sphere.
/// * `t` — Interpolation parameters in [0, 1]. Can be multiple values.
pub fn geometric_slerp(
    start: &[f64],
    end: &[f64],
    t_values: &[f64],
) -> Result<Vec<Vec<f64>>, SpatialError> {
    if start.len() != end.len() {
        return Err(SpatialError::DimensionMismatch {
            expected: start.len(),
            actual: end.len(),
        });
    }
    let d = start.len();

    // Compute angle between start and end
    let dot: f64 = start.iter().zip(end.iter()).map(|(&a, &b)| a * b).sum();
    let dot = dot.clamp(-1.0, 1.0);
    let omega = dot.acos();

    let mut results = Vec::with_capacity(t_values.len());

    if omega.abs() < 1e-10 {
        // Points are the same: all interpolations = start
        for _ in t_values {
            results.push(start.to_vec());
        }
        return Ok(results);
    }

    let sin_omega = omega.sin();

    for &t in t_values {
        let a = ((1.0 - t) * omega).sin() / sin_omega;
        let b = (t * omega).sin() / sin_omega;
        let point: Vec<f64> = (0..d).map(|i| a * start[i] + b * end[i]).collect();
        results.push(point);
    }

    Ok(results)
}

// ══════════════════════════════════════════════════════════════════════
// Coordinate Transforms
// ══════════════════════════════════════════════════════════════════════

/// Convert spherical coordinates (r, θ, φ) to Cartesian (x, y, z).
///
/// θ = polar angle from z-axis (0 to π), φ = azimuthal angle in x-y plane (0 to 2π).
pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> (f64, f64, f64) {
    (
        r * theta.sin() * phi.cos(),
        r * theta.sin() * phi.sin(),
        r * theta.cos(),
    )
}

/// Convert Cartesian (x, y, z) to spherical (r, θ, φ).
pub fn cartesian_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x * x + y * y + z * z).sqrt();
    let theta = if r > 0.0 {
        (z / r).clamp(-1.0, 1.0).acos()
    } else {
        0.0
    };
    let phi = y.atan2(x);
    (r, theta, phi)
}

/// Convert cylindrical coordinates (r, θ, z) to Cartesian (x, y, z).
pub fn cylindrical_to_cartesian(rho: f64, theta: f64, z: f64) -> (f64, f64, f64) {
    (rho * theta.cos(), rho * theta.sin(), z)
}

/// Convert Cartesian (x, y, z) to cylindrical (ρ, θ, z).
pub fn cartesian_to_cylindrical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let rho = (x * x + y * y).sqrt();
    let theta = y.atan2(x);
    (rho, theta, z)
}

// ══════════════════════════════════════════════════════════════════════
// Rotation
// ══════════════════════════════════════════════════════════════════════

/// 3D rotation matrix from axis-angle representation.
///
/// Rodrigues' rotation formula. `axis` must be a unit vector.
/// `angle` is in radians.
///
/// Matches `scipy.spatial.transform.Rotation.from_rotvec`.
pub fn rotation_matrix(axis: &[f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;
    let [x, y, z] = *axis;

    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

/// Apply a 3x3 rotation matrix to a 3D point.
pub fn rotate_point(r: &[[f64; 3]; 3], p: &[f64; 3]) -> [f64; 3] {
    [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2],
    ]
}

/// Compute the angle between two vectors.
pub fn angle_between(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let na: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0).acos()
}

/// Cross product of two 3D vectors.
pub fn cross_3d(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two vectors.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Normalize a vector to unit length.
pub fn normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|&x| x / norm).collect()
}

/// Directed Hausdorff distance between two point sets.
///
/// Returns max_{a in A} min_{b in B} ||a - b||.
/// Matches `scipy.spatial.distance.directed_hausdorff`.
pub fn directed_hausdorff(xa: &[Vec<f64>], xb: &[Vec<f64>]) -> Result<f64, SpatialError> {
    if xa.is_empty() || xb.is_empty() {
        return Err(SpatialError::EmptyData);
    }
    let dim = xa[0].len();
    if xb[0].len() != dim {
        return Err(SpatialError::DimensionMismatch {
            expected: dim,
            actual: xb[0].len(),
        });
    }
    if xa.iter().flatten().any(|v| !v.is_finite()) || xb.iter().flatten().any(|v| !v.is_finite()) {
        return Err(SpatialError::InvalidArgument(
            "hausdorff distance requires finite points".to_string(),
        ));
    }

    let mut max_dist = 0.0f64;
    for a in xa {
        let mut min_dist = f64::INFINITY;
        for b in xb {
            let d = euclidean(a, b);
            min_dist = min_dist.min(d);
        }
        max_dist = max_dist.max(min_dist);
    }
    Ok(max_dist)
}

/// Hausdorff distance between two point sets (symmetric).
pub fn hausdorff_distance(xa: &[Vec<f64>], xb: &[Vec<f64>]) -> Result<f64, SpatialError> {
    let d1 = directed_hausdorff(xa, xb)?;
    let d2 = directed_hausdorff(xb, xa)?;
    Ok(d1.max(d2))
}

/// Mahalanobis distance between two vectors given an inverse covariance matrix.
///
/// d = sqrt((x-y)^T * VI * (x-y))
/// Matches `scipy.spatial.distance.mahalanobis`.
pub fn mahalanobis(x: &[f64], y: &[f64], vi: &[Vec<f64>]) -> f64 {
    let n = x.len();
    let diff: Vec<f64> = x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect();

    let mut vi_diff = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            vi_diff[i] += vi[i][j] * diff[j];
        }
    }

    let result: f64 = diff
        .iter()
        .zip(vi_diff.iter())
        .map(|(&d, &vd)| d * vd)
        .sum();
    result.max(0.0).sqrt()
}

/// Standardized Euclidean distance.
///
/// d = sqrt(sum((x-y)² / v)) where v is the per-component variance.
/// Matches `scipy.spatial.distance.seuclidean`.
pub fn seuclidean(x: &[f64], y: &[f64], v: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .zip(v.iter())
        .map(|((&xi, &yi), &vi)| (xi - yi).powi(2) / vi)
        .sum::<f64>()
        .sqrt()
}

/// Weighted Minkowski distance.
///
/// Matches `scipy.spatial.distance.wminkowski`.
pub fn wminkowski(x: &[f64], y: &[f64], p: f64, w: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .zip(w.iter())
        .map(|((&xi, &yi), &wi)| (wi * (xi - yi).abs()).powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

/// Yule dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.yule`.
pub fn yule(u: &[bool], v: &[bool]) -> f64 {
    let mut ctf = 0usize; // true-false
    let mut cft = 0usize; // false-true
    let mut ctt = 0usize; // true-true
    let mut cff = 0usize; // false-false

    for (&a, &b) in u.iter().zip(v.iter()) {
        match (a, b) {
            (true, true) => ctt += 1,
            (true, false) => ctf += 1,
            (false, true) => cft += 1,
            (false, false) => cff += 1,
        }
    }

    let r = ctf * cft;
    let s = ctt * cff;
    if r + s == 0 {
        0.0
    } else {
        2.0 * r as f64 / (r + s) as f64
    }
}

/// Dice dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.dice`.
pub fn dice(u: &[bool], v: &[bool]) -> f64 {
    let mut ctf = 0usize;
    let mut cft = 0usize;
    let mut ctt = 0usize;

    for (&a, &b) in u.iter().zip(v.iter()) {
        match (a, b) {
            (true, true) => ctt += 1,
            (true, false) => ctf += 1,
            (false, true) => cft += 1,
            _ => {}
        }
    }

    let ntt = ctf + cft;
    if ntt + 2 * ctt == 0 {
        0.0
    } else {
        ntt as f64 / (ntt + 2 * ctt) as f64
    }
}

/// Kulsinski dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.kulsinski`.
pub fn kulsinski(u: &[bool], v: &[bool]) -> f64 {
    let n = u.len() as f64;
    let mut ctf = 0usize;
    let mut cft = 0usize;
    let mut ctt = 0usize;

    for (&a, &b) in u.iter().zip(v.iter()) {
        match (a, b) {
            (true, true) => ctt += 1,
            (true, false) => ctf += 1,
            (false, true) => cft += 1,
            _ => {}
        }
    }

    let ntt = (ctf + cft) as f64;
    if ctt == 0 && ntt == 0.0 {
        0.0
    } else {
        (ntt + n - ctt as f64) / (ntt + n)
    }
}

/// Rogerstanimoto dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.rogerstanimoto`.
pub fn rogerstanimoto(u: &[bool], v: &[bool]) -> f64 {
    let mut ndiff = 0usize;
    let mut nsame = 0usize;

    for (&a, &b) in u.iter().zip(v.iter()) {
        if a == b {
            nsame += 1;
        } else {
            ndiff += 1;
        }
    }

    let r = 2 * ndiff;
    (r as f64) / (nsame + r) as f64
}

/// Russell-Rao dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.russellrao`.
pub fn russellrao(u: &[bool], v: &[bool]) -> f64 {
    let n = u.len();
    if n == 0 {
        return 0.0;
    }
    let ctt = u.iter().zip(v.iter()).filter(|&(a, b)| *a && *b).count();
    (n - ctt) as f64 / n as f64
}

/// Sokal-Michener dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.sokalmichener`.
pub fn sokalmichener(u: &[bool], v: &[bool]) -> f64 {
    rogerstanimoto(u, v) // Same formula
}

/// Sokal-Sneath dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.sokalsneath`.
pub fn sokalsneath(u: &[bool], v: &[bool]) -> f64 {
    let mut ctf = 0usize;
    let mut cft = 0usize;
    let mut ctt = 0usize;

    for (&a, &b) in u.iter().zip(v.iter()) {
        match (a, b) {
            (true, true) => ctt += 1,
            (true, false) => ctf += 1,
            (false, true) => cft += 1,
            _ => {}
        }
    }

    let r = 2 * (ctf + cft);
    if r + ctt == 0 {
        0.0
    } else {
        r as f64 / (r + ctt) as f64
    }
}

/// Matching dissimilarity for boolean vectors.
///
/// Fraction of positions where both vectors agree.
/// Matches `scipy.spatial.distance.matching` (same as Hamming for boolean).
pub fn matching(u: &[bool], v: &[bool]) -> f64 {
    let n = u.len();
    if n == 0 {
        return 0.0;
    }
    let ndiff = u.iter().zip(v.iter()).filter(|&(a, b)| *a != *b).count();
    ndiff as f64 / n as f64
}

/// Compute all pairwise distances using a specified metric function.
///
/// Returns condensed distance vector.
pub fn pdist_func<F>(data: &[Vec<f64>], metric: F) -> Vec<f64>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let n = data.len();
    let mut dists = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            dists.push(metric(&data[i], &data[j]));
        }
    }
    dists
}

/// Compute pairwise distances between two sets using a metric function.
///
/// Returns an m×n matrix.
pub fn cdist_func<F>(xa: &[Vec<f64>], xb: &[Vec<f64>], metric: F) -> Vec<Vec<f64>>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    xa.iter()
        .map(|a| xb.iter().map(|b| metric(a, b)).collect())
        .collect()
}

/// Compute the nearest neighbor for each point in a dataset.
///
/// Returns (indices, distances) of nearest neighbors.
/// For empty input, returns empty vectors.
/// For single-element input, returns `(vec![None], vec![])` conceptually,
/// but since each point needs at least one other point to have a neighbor,
/// the index is undefined and the distance is `INFINITY`.
pub fn nearest_neighbors(data: &[Vec<f64>]) -> (Vec<Option<usize>>, Vec<f64>) {
    let n = data.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    let mut indices = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);

    for i in 0..n {
        let mut min_dist = f64::INFINITY;
        let mut min_idx: Option<usize> = None;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = euclidean(&data[i], &data[j]);
            if d < min_dist {
                min_dist = d;
                min_idx = Some(j);
            }
        }
        indices.push(min_idx);
        distances.push(min_dist);
    }

    (indices, distances)
}

/// Compute the k nearest neighbors for each point.
///
/// Returns (indices, distances) where each inner vec has k elements.
pub fn k_nearest_neighbors(data: &[Vec<f64>], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let n = data.len();
    let mut all_indices = Vec::with_capacity(n);
    let mut all_distances = Vec::with_capacity(n);

    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean(&data[i], &data[j])))
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));

        let k_actual = k.min(dists.len());
        all_indices.push(dists[..k_actual].iter().map(|&(idx, _)| idx).collect());
        all_distances.push(dists[..k_actual].iter().map(|&(_, d)| d).collect());
    }

    (all_indices, all_distances)
}

/// Compute the centroid of a set of points.
pub fn centroid(points: &[Vec<f64>]) -> Vec<f64> {
    if points.is_empty() {
        return vec![];
    }
    let d = points[0].len();
    let n = points.len() as f64;
    let mut center = vec![0.0; d];
    for p in points {
        for (j, &v) in p.iter().enumerate() {
            center[j] += v;
        }
    }
    for v in &mut center {
        *v /= n;
    }
    center
}

/// Compute the medoid (point minimizing sum of distances) of a set.
///
/// Returns `None` for an empty point set.
pub fn medoid(points: &[Vec<f64>]) -> Option<usize> {
    let n = points.len();
    if n == 0 {
        return None;
    }

    let mut best = 0;
    let mut best_total = f64::INFINITY;

    for i in 0..n {
        let total: f64 = (0..n)
            .filter(|&j| j != i)
            .map(|j| euclidean(&points[i], &points[j]))
            .sum();
        if total < best_total {
            best_total = total;
            best = i;
        }
    }

    Some(best)
}

/// Compute the diameter of a point set (maximum pairwise distance).
pub fn diameter(points: &[Vec<f64>]) -> f64 {
    let n = points.len();
    let mut max_d = 0.0f64;
    for i in 0..n {
        for j in i + 1..n {
            let d = euclidean(&points[i], &points[j]);
            max_d = if max_d.is_nan() || d.is_nan() {
                f64::NAN
            } else {
                max_d.max(d)
            };
        }
    }
    max_d
}

/// Compute the spread (average distance from centroid) of a point set.
pub fn spread(points: &[Vec<f64>]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let center = centroid(points);
    let n = points.len() as f64;
    points.iter().map(|p| euclidean(p, &center)).sum::<f64>() / n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point_set_contains(points: &[Vec<f64>], expected: &[f64]) -> bool {
        points
            .iter()
            .any(|point| points_approx_eq(point, expected, 1e-10))
    }

    #[test]
    fn euclidean_distance() {
        assert!((euclidean(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-12);
        assert!((euclidean(&[1.0], &[1.0])).abs() < 1e-12);
    }

    #[test]
    fn cityblock_distance() {
        assert!((cityblock(&[0.0, 0.0], &[3.0, 4.0]) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn chebyshev_distance() {
        assert!((chebyshev(&[0.0, 0.0], &[3.0, 4.0]) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn cdist_basic() {
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let d = cdist(&xa, &xb).expect("cdist");
        assert!((d[0][0] - 1.0).abs() < 1e-12); // (0,0) to (0,1)
        assert!((d[0][1] - 2.0_f64.sqrt()).abs() < 1e-12); // (0,0) to (1,1)
        assert!((d[1][0] - 2.0_f64.sqrt()).abs() < 1e-12); // (1,0) to (0,1)
        assert!((d[1][1] - 1.0).abs() < 1e-12); // (1,0) to (1,1)
    }

    #[test]
    fn kdtree_nearest_neighbor() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let tree = KDTree::new(&data).expect("kdtree");
        let (idx, dist) = tree.query(&[0.6, 0.6]).expect("query");
        assert_eq!(idx, 4); // closest to (0.5, 0.5)
        assert!(dist < 0.2);
    }

    #[test]
    fn kdtree_exact_match() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let (idx, dist) = tree.query(&[1.0, 2.0, 3.0]).expect("query");
        assert_eq!(idx, 0);
        assert!(dist < 1e-12);
    }

    #[test]
    fn kdtree_k_nearest() {
        let data = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let results = tree.query_k(&[1.5], 3).expect("query_k");
        assert_eq!(results.len(), 3);
        // Nearest to 1.5: 1.0 (dist=0.5), 2.0 (dist=0.5), 0.0 (dist=1.5) or 3.0 (dist=1.5)
        assert!(results[0].1 < 0.6);
        assert!(results[1].1 < 0.6);
    }

    #[test]
    fn kdtree_query_k_zero_returns_empty() {
        let data = vec![vec![0.0], vec![1.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let results = tree.query_k(&[0.0], 0).expect("query_k");
        assert!(results.is_empty());
    }

    #[test]
    fn kdtree_empty_rejected() {
        let err = KDTree::new(&[]).expect_err("empty");
        assert!(matches!(err, SpatialError::EmptyData));
    }

    #[test]
    fn kdtree_rejects_mixed_input_dimensions() {
        let err = KDTree::new(&[vec![0.0, 1.0], vec![2.0]])
            .expect_err("mixed-dimension points should be rejected");
        assert!(matches!(
            err,
            SpatialError::DimensionMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn kdtree_rejects_non_finite_points() {
        let err = KDTree::new(&[vec![0.0, 1.0], vec![f64::NAN, 2.0]])
            .expect_err("non-finite points should be rejected");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn kdtree_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let err = tree.query(&[1.0]).expect_err("dim mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn kdtree_rejects_non_finite_query() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let err = tree.query(&[f64::NAN, 0.0]).expect_err("nan query");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
        let err = tree
            .query_k(&[f64::INFINITY, 0.0], 1)
            .expect_err("inf query");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
        let err = tree
            .query_ball_point(&[0.0, f64::NEG_INFINITY], 1.0)
            .expect_err("inf query");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn kdtree_single_point() {
        let data = vec![vec![5.0, 5.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let (idx, _) = tree.query(&[0.0, 0.0]).expect("query");
        assert_eq!(idx, 0);
    }

    // ── New distance metrics ───────────────────────────────────

    #[test]
    fn cosine_distance_orthogonal() {
        // Orthogonal vectors: cosine distance = 1
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_distance_identical() {
        // Same direction: cosine distance = 0
        assert!((cosine(&[1.0, 2.0], &[2.0, 4.0])).abs() < 1e-12);
    }

    #[test]
    fn cosine_distance_opposite() {
        // Opposite direction: cosine distance = 2
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn minkowski_p1_is_cityblock() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 0.0, 1.0];
        assert!((minkowski(&a, &b, 1.0) - cityblock(&a, &b)).abs() < 1e-12);
    }

    #[test]
    fn minkowski_p2_is_euclidean() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 0.0, 1.0];
        assert!((minkowski(&a, &b, 2.0) - euclidean(&a, &b)).abs() < 1e-12);
    }

    #[test]
    fn minkowski_pinf_is_chebyshev() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 0.0, 1.0];
        assert!((minkowski(&a, &b, f64::INFINITY) - chebyshev(&a, &b)).abs() < 1e-12);
    }

    #[test]
    fn correlation_distance_perfect() {
        // Perfectly correlated: distance = 0
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 4.0, 6.0, 8.0];
        assert!(correlation(&a, &b).abs() < 1e-12);
    }

    #[test]
    fn correlation_distance_anticorrelated() {
        // Anti-correlated: distance = 2
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [8.0, 6.0, 4.0, 2.0];
        assert!((correlation(&a, &b) - 2.0).abs() < 1e-12);
    }

    // ── pdist and squareform ──────────────────────────────────

    #[test]
    fn pdist_euclidean_three_points() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let d = pdist(&x, DistanceMetric::Euclidean).expect("pdist");
        // 3 points -> 3 distances: (0,1), (0,2), (1,2)
        assert_eq!(d.len(), 3);
        assert!((d[0] - 1.0).abs() < 1e-12); // dist(0,1)
        assert!((d[1] - 1.0).abs() < 1e-12); // dist(0,2)
        assert!((d[2] - 2.0_f64.sqrt()).abs() < 1e-12); // dist(1,2)
    }

    #[test]
    fn pdist_cityblock() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 0.0]];
        let d = pdist(&x, DistanceMetric::Cityblock).expect("pdist");
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-12); // |1|+|1|
        assert!((d[1] - 2.0).abs() < 1e-12); // |2|+|0|
        assert!((d[2] - 2.0).abs() < 1e-12); // |1|+|1|
    }

    #[test]
    fn squareform_roundtrip() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let condensed = pdist(&x, DistanceMetric::Euclidean).expect("pdist");
        let matrix = squareform_to_matrix(&condensed).expect("to_matrix");
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        // Diagonal is zero
        assert!((matrix[0][0]).abs() < 1e-12);
        assert!((matrix[1][1]).abs() < 1e-12);
        assert!((matrix[2][2]).abs() < 1e-12);
        // Symmetric
        assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-12);
        // Roundtrip back to condensed
        let condensed2 = squareform_to_condensed(&matrix).expect("to_condensed");
        for (a, b) in condensed.iter().zip(condensed2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn squareform_invalid_length() {
        assert!(squareform_to_matrix(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn pdist_too_few_points() {
        assert!(pdist(&[vec![1.0]], DistanceMetric::Euclidean).is_err());
    }

    #[test]
    fn cdist_metric_cosine() {
        let xa = vec![vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let d = cdist_metric(&xa, &xb, DistanceMetric::Cosine).expect("cdist cosine");
        assert!((d[0][0] - 1.0).abs() < 1e-12); // orthogonal
        assert!((d[0][1]).abs() < 1e-12); // identical direction
    }

    #[test]
    fn distance_matrix_matches_cdist() {
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let dm = distance_matrix(&xa, &xb).expect("distance_matrix");
        let cd = cdist(&xa, &xb).expect("cdist");
        assert_eq!(dm.len(), cd.len());
        for (row_dm, row_cd) in dm.iter().zip(cd.iter()) {
            assert_eq!(row_dm.len(), row_cd.len());
            for (&a, &b) in row_dm.iter().zip(row_cd.iter()) {
                assert!(
                    (a - b).abs() < 1e-12,
                    "distance_matrix mismatch: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn rectangle_normalizes_bounds_and_volume() {
        let rect = Rectangle::new(&[0.0, 1.0], &[1.0, 0.0]).expect("rectangle");
        assert_eq!(rect.mins, vec![0.0, 0.0]);
        assert_eq!(rect.maxes, vec![1.0, 1.0]);
        assert_eq!(rect.m, 2);
        assert!((rect.volume() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rectangle_split_matches_scipy_semantics() {
        let rect = Rectangle::new(&[1.0, 2.0], &[0.0, -1.0]).expect("rectangle");
        let (lower, upper) = rect.split(0, 0.5).expect("split");
        assert_eq!(lower.mins, vec![0.0, -1.0]);
        assert_eq!(lower.maxes, vec![0.5, 2.0]);
        assert_eq!(upper.mins, vec![0.5, -1.0]);
        assert_eq!(upper.maxes, vec![1.0, 2.0]);

        let (outside_lower, outside_upper) = rect.split(0, -1.0).expect("outside split");
        assert_eq!(outside_lower.mins, vec![-1.0, -1.0]);
        assert_eq!(outside_lower.maxes, vec![0.0, 2.0]);
        assert_eq!(outside_upper.mins, vec![-1.0, -1.0]);
        assert_eq!(outside_upper.maxes, vec![1.0, 2.0]);
    }

    #[test]
    fn rectangle_point_distances_match_scipy() {
        let rect = Rectangle::new(&[1.0, 2.0], &[0.0, -1.0]).expect("rectangle");
        assert!(rect.min_distance_point(&[0.5, 0.0], 2.0).unwrap().abs() < 1e-12);
        assert!(
            (rect.max_distance_point(&[0.5, 0.0], 2.0).unwrap() - 2.0615528128088303).abs() < 1e-12
        );
        assert!((rect.min_distance_point(&[3.0, 0.0], 1.0).unwrap() - 2.0).abs() < 1e-12);
        assert!((rect.max_distance_point(&[3.0, 0.0], f64::INFINITY).unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rectangle_rectangle_distances_match_scipy() {
        let rect = Rectangle::new(&[1.0, 2.0], &[0.0, -1.0]).expect("rectangle");
        let other = Rectangle::new(&[2.0, 1.0], &[1.5, -2.0]).expect("other");
        assert!((rect.min_distance_rectangle(&other, 2.0).unwrap() - 0.5).abs() < 1e-12);
        assert!(
            (rect.max_distance_rectangle(&other, 2.0).unwrap() - 4.47213595499958).abs() < 1e-12
        );

        let far = Rectangle::new(&[4.0, 4.0], &[2.0, 1.0]).expect("far");
        assert!((rect.min_distance_rectangle(&far, 1.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((rect.max_distance_rectangle(&far, f64::INFINITY).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn rectangle_rejects_invalid_inputs() {
        let err = Rectangle::new(&[1.0, 2.0], &[0.0]).expect_err("dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));

        let rect = Rectangle::new(&[1.0], &[0.0]).expect("rectangle");
        let err = rect.split(2, 0.5).expect_err("bad axis");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));

        let err = rect
            .min_distance_point(&[0.0], 0.0)
            .expect_err("invalid p should be rejected");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn cdist_metric_dimension_mismatch_in_later_row() {
        let xa = vec![vec![0.0, 0.0], vec![1.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let err = cdist_metric(&xa, &xb, DistanceMetric::Euclidean)
            .expect_err("later row dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn kdtree_large_random() {
        // Verify KDTree gives same result as brute force for 100 points
        let data: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![(i as f64 * 0.73) % 10.0, (i as f64 * 1.37) % 10.0])
            .collect();
        let tree = KDTree::new(&data).expect("kdtree");
        let query = [5.0, 5.0];

        let (tree_idx, tree_dist) = tree.query(&query).expect("query");

        // Brute force
        let (brute_idx, brute_dist) = data
            .iter()
            .enumerate()
            .map(|(i, p)| (i, euclidean(p, &query)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();

        assert_eq!(tree_idx, brute_idx);
        assert!((tree_dist - brute_dist).abs() < 1e-10);
    }

    // ── New distance metric tests ──────────────────────────────────

    #[test]
    fn hamming_basic() {
        assert!((hamming(&[1.0, 0.0, 1.0], &[1.0, 1.0, 1.0]) - 1.0 / 3.0).abs() < 1e-12);
        assert!((hamming(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])).abs() < 1e-12);
    }

    #[test]
    fn jaccard_basic() {
        // u=[1,0,1], v=[1,1,0]: TT=1, TF=1, FT=1 → J = 2/3
        assert!((jaccard(&[1.0, 0.0, 1.0], &[1.0, 1.0, 0.0]) - 2.0 / 3.0).abs() < 1e-12);
        // identical nonzero → J = 0
        assert!(jaccard(&[1.0, 1.0], &[1.0, 1.0]).abs() < 1e-12);
    }

    #[test]
    fn jaccard_all_zeros() {
        // Both zero → J = 0 (no nonzero elements)
        assert!(jaccard(&[0.0, 0.0], &[0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn canberra_basic() {
        // |1-2|/(1+2) + |3-4|/(3+4) = 1/3 + 1/7
        let expected = 1.0 / 3.0 + 1.0 / 7.0;
        assert!((canberra(&[1.0, 3.0], &[2.0, 4.0]) - expected).abs() < 1e-12);
    }

    #[test]
    fn canberra_zeros() {
        // Both zero at a position → contribution is 0
        assert!((canberra(&[0.0, 1.0], &[0.0, 2.0]) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn braycurtis_basic() {
        // |1-2| + |3-4| = 2; |1+2| + |3+4| = 10 → 2/10 = 0.2
        assert!((braycurtis(&[1.0, 3.0], &[2.0, 4.0]) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn braycurtis_identical() {
        assert!(braycurtis(&[5.0, 10.0], &[5.0, 10.0]).abs() < 1e-12);
    }

    #[test]
    fn jensenshannon_matches_scipy_default_base() {
        let distance = jensenshannon(&[1.0, 0.0], &[0.5, 0.5], None);
        assert!((distance - 0.464501404022459).abs() < 1e-15);
    }

    #[test]
    fn jensenshannon_respects_base_kwarg() {
        let distance = jensenshannon(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], Some(2.0));
        assert!((distance - 1.0).abs() < 1e-15);
    }

    #[test]
    fn jensenshannon_normalizes_input_masses() {
        let normalized = jensenshannon(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], Some(2.0));
        let unnormalized = jensenshannon(&[2.0, 0.0, 0.0], &[0.0, 4.0, 0.0], Some(2.0));
        assert!((normalized - unnormalized).abs() < 1e-15);
    }

    #[test]
    fn jensenshannon_zero_mass_vectors_match_scipy_nan() {
        let distance = jensenshannon(&[0.0, 0.0], &[0.0, 0.0], None);
        assert!(distance.is_nan());
    }

    #[test]
    fn pdist_with_canberra() {
        let x = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let d = pdist(&x, DistanceMetric::Canberra).unwrap();
        assert_eq!(d.len(), 3);
        // All distances should be positive
        for &di in &d {
            assert!(di >= 0.0);
        }
    }

    // ── KDTree extension tests ─────────────────────────────────────

    #[test]
    fn kdtree_query_ball_point() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
        ];
        let tree = KDTree::new(&data).unwrap();

        // Radius 1.5 from origin should include points 0, 1, 2 but not 3
        let mut result = tree.query_ball_point(&[0.0, 0.0], 1.5).unwrap();
        result.sort();
        assert_eq!(result, vec![0, 1, 2]);

        // Radius 0.5 from origin should include only point 0
        let result = tree.query_ball_point(&[0.0, 0.0], 0.5).unwrap();
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn kdtree_query_ball_point_empty() {
        let data = vec![vec![10.0, 10.0]];
        let tree = KDTree::new(&data).unwrap();
        let result = tree.query_ball_point(&[0.0, 0.0], 1.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn kdtree_query_ball_point_negative_radius_rejected() {
        let tree = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let err = tree
            .query_ball_point(&[0.0, 0.0], -1.0)
            .expect_err("negative radius");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn kdtree_count_neighbors() {
        let data1 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let data2 = vec![vec![0.5, 0.0], vec![10.0, 10.0]];
        let tree1 = KDTree::new(&data1).unwrap();
        let tree2 = KDTree::new(&data2).unwrap();

        // Within radius 1.0: (0,0)↔(0.5,0) and (1,0)↔(0.5,0) = 2 pairs
        let count = tree1.count_neighbors(&tree2, 1.0).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn kdtree_count_neighbors_dimension_mismatch() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.0, 0.0, 0.0]]).unwrap();
        let err = tree1
            .count_neighbors(&tree2, 1.0)
            .expect_err("dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn kdtree_count_neighbors_negative_radius_rejected() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.5, 0.0]]).unwrap();
        let err = tree1
            .count_neighbors(&tree2, -1.0)
            .expect_err("negative radius");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn kdtree_query_ball_tree() {
        let data1 = vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![5.0, 5.0]];
        let data2 = vec![vec![0.5, 0.0], vec![2.5, 0.0], vec![9.0, 9.0]];
        let tree1 = KDTree::new(&data1).unwrap();
        let tree2 = KDTree::new(&data2).unwrap();

        let neighbors = tree1.query_ball_tree(&tree2, 0.75).unwrap();
        assert_eq!(neighbors.len(), data1.len());
        assert_eq!(neighbors[0], vec![0]);
        assert_eq!(neighbors[1], vec![1]);
        assert!(neighbors[2].is_empty());
    }

    #[test]
    fn kdtree_query_ball_tree_dimension_mismatch() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.0, 0.0, 0.0]]).unwrap();
        let err = tree1
            .query_ball_tree(&tree2, 1.0)
            .expect_err("dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn kdtree_query_ball_tree_negative_radius_rejected() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.5, 0.0]]).unwrap();
        let err = tree1
            .query_ball_tree(&tree2, -1.0)
            .expect_err("negative radius");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn kdtree_sparse_distance_matrix_basic() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0], vec![2.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.5, 0.0], vec![2.5, 0.0], vec![9.0, 9.0]]).unwrap();

        let entries = tree1.sparse_distance_matrix(&tree2, 0.75).unwrap();
        assert_eq!(
            entries.entries(),
            &BTreeMap::from([((0, 0), 0.5), ((1, 1), 0.5)]),
            "default DOK output should preserve original point indices"
        );
        assert_eq!(entries.shape(), Shape2D::new(2, 3));
    }

    #[test]
    fn kdtree_sparse_distance_matrix_empty() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![10.0, 10.0]]).unwrap();
        let entries = tree1.sparse_distance_matrix(&tree2, 1.0).unwrap();
        assert!(entries.entries().is_empty());
        assert_eq!(entries.shape(), Shape2D::new(1, 1));
    }

    #[test]
    fn kdtree_sparse_distance_matrix_dimension_mismatch() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.0, 0.0, 0.0]]).unwrap();
        let err = tree1
            .sparse_distance_matrix(&tree2, 1.0)
            .expect_err("dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn kdtree_sparse_distance_matrix_output_type_variants() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0], vec![2.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.5, 0.0], vec![2.5, 0.0], vec![9.0, 9.0]]).unwrap();

        let dict_output = tree1
            .sparse_distance_matrix_with_output_type(&tree2, 0.75, "dict")
            .unwrap();
        assert_eq!(
            dict_output,
            SparseDistanceMatrixOutput::Dict(BTreeMap::from([((0, 0), 0.5), ((1, 1), 0.5)]))
        );

        let ndarray_output = tree1
            .sparse_distance_matrix_with_output_type(&tree2, 0.75, "ndarray")
            .unwrap();
        assert_eq!(
            ndarray_output,
            SparseDistanceMatrixOutput::Ndarray(vec![
                SparseDistanceMatrixRecord { i: 0, j: 0, v: 0.5 },
                SparseDistanceMatrixRecord { i: 1, j: 1, v: 0.5 },
            ])
        );

        let coo_output = tree1
            .sparse_distance_matrix_with_output_type(&tree2, 0.75, "coo_matrix")
            .unwrap();
        assert!(matches!(
            coo_output,
            SparseDistanceMatrixOutput::CooMatrix(_)
        ));
        if let SparseDistanceMatrixOutput::CooMatrix(matrix) = coo_output {
            assert_eq!(matrix.shape(), Shape2D::new(2, 3));
            assert_eq!(matrix.row_indices(), &[0, 1]);
            assert_eq!(matrix.col_indices(), &[0, 1]);
            assert_eq!(matrix.data(), &[0.5, 0.5]);
        }
    }

    #[test]
    fn kdtree_sparse_distance_matrix_rejects_invalid_output_type() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.5, 0.0]]).unwrap();
        let err = tree1
            .sparse_distance_matrix_with_output_type(&tree2, 1.0, "bad")
            .expect_err("invalid output type");
        assert!(
            matches!(err, SpatialError::InvalidArgument(message) if message == "Invalid output type")
        );
    }

    // ── ConvexHull tests ─────────────────────────────────────────────

    #[test]
    fn convex_hull_square() {
        let points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let hull = ConvexHull::new(&points).expect("hull");
        assert_eq!(hull.vertices.len(), 4, "square has 4 hull vertices");
        assert!((hull.area - 1.0).abs() < 1e-10, "area = {}", hull.area);
        assert!(
            (hull.perimeter - 4.0).abs() < 1e-10,
            "perimeter = {}",
            hull.perimeter
        );
    }

    #[test]
    fn convex_hull_triangle() {
        let points = [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)];
        let hull = ConvexHull::new(&points).expect("hull");
        assert_eq!(hull.vertices.len(), 3, "triangle has 3 hull vertices");
        // Area of triangle with base 4, height 3 = 6
        assert!((hull.area - 6.0).abs() < 1e-10, "area = {}", hull.area);
    }

    #[test]
    fn convex_hull_interior_points_excluded() {
        // Square with interior point
        let points = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (1.0, 1.0), // interior
        ];
        let hull = ConvexHull::new(&points).expect("hull");
        assert_eq!(
            hull.vertices.len(),
            4,
            "interior point should be excluded: {:?}",
            hull.vertices
        );
        assert!(!hull.vertices.contains(&4), "point 4 is interior");
    }

    #[test]
    fn convex_hull_simplices_form_closed_polygon() {
        let points = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let hull = ConvexHull::new(&points).expect("hull");
        // Each vertex appears exactly twice in simplices (as start and end of adjacent edges)
        let mut vertex_count = std::collections::HashMap::new();
        for &(a, b) in &hull.simplices {
            *vertex_count.entry(a).or_insert(0) += 1;
            *vertex_count.entry(b).or_insert(0) += 1;
        }
        for &v in &hull.vertices {
            assert_eq!(
                vertex_count.get(&v),
                Some(&2),
                "vertex {v} should appear exactly twice in simplices"
            );
        }
    }

    #[test]
    fn convex_hull_too_few_points() {
        let points = [(0.0, 0.0), (1.0, 1.0)];
        let err = ConvexHull::new(&points).expect_err("too few");
        assert!(matches!(err, SpatialError::Qhull(_)));
    }

    #[test]
    fn convex_hull_collinear_points_rejected() {
        let points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let err = ConvexHull::new(&points).expect_err("collinear");
        assert!(matches!(err, SpatialError::Qhull(_)));
    }

    #[test]
    fn convex_hull_many_points() {
        // Generate points on a circle (all should be on hull) + center
        let n = 20;
        let mut points: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (theta.cos(), theta.sin())
            })
            .collect();
        points.push((0.0, 0.0)); // center point
        let hull = ConvexHull::new(&points).expect("hull");
        assert_eq!(hull.vertices.len(), n, "circle points all on hull");
        assert!(!hull.vertices.contains(&n), "center point excluded");
        // Area should approximate π
        assert!(
            (hull.area - std::f64::consts::PI).abs() < 0.1,
            "area ≈ π: {}",
            hull.area
        );
    }

    // ── HalfspaceIntersection tests ──────────────────────────────────

    #[test]
    fn qhull_error_variant_preserves_message() {
        let err = QhullError::new("QH6154 Qhull precision error");
        assert!(err.message().contains("QH6154"));
        let spatial: SpatialError = err.clone().into();
        assert_eq!(spatial.to_string(), err.to_string());
        assert!(matches!(spatial, SpatialError::Qhull(_)));
    }

    #[test]
    fn halfspace_intersection_square_matches_dual_surface() {
        let halfspaces = [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
        ];
        let hs = HalfspaceIntersection::new(&halfspaces, (0.5, 0.5)).expect("halfspaces");

        assert_eq!(hs.ndim, 2);
        assert_eq!(hs.nineq, 4);
        assert!(hs.is_bounded);
        assert_eq!(hs.dual_vertices.len(), 4);
        assert_eq!(hs.dual_facets.len(), 4);
        assert!((hs.dual_points[0][0] + 2.0).abs() < 1e-10);
        assert!((hs.dual_points[1][1] + 2.0).abs() < 1e-10);
        assert!((hs.dual_volume - 8.0).abs() < 1e-10);
        assert!((hs.dual_area - 8.0 * 2.0_f64.sqrt()).abs() < 1e-10);

        for expected in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
            assert!(
                point_set_contains(&hs.intersections, &expected),
                "missing intersection {expected:?}: {:?}",
                hs.intersections
            );
        }
    }

    #[test]
    fn halfspace_intersection_rejects_boundary_feasible_point_as_qhull_error() {
        let halfspaces = [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
        ];
        let err = HalfspaceIntersection::new(&halfspaces, (0.0, 0.5))
            .expect_err("boundary feasible point");
        assert!(matches!(
            err,
            SpatialError::Qhull(ref qhull) if qhull.message().contains("QH6023")
        ));
    }

    #[test]
    fn halfspace_intersection_unbounded_region_is_marked() {
        let halfspaces = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, -1.0]];
        let hs = HalfspaceIntersection::new(&halfspaces, (0.5, 0.5)).expect("halfspaces");

        assert!(!hs.is_bounded);
        assert!(point_set_contains(&hs.intersections, &[0.0, 0.0]));
        assert!(point_set_contains(&hs.intersections, &[1.0, 0.0]));
        assert!(
            hs.intersections
                .iter()
                .any(|row| row.iter().any(|value| !value.is_finite())),
            "unbounded dual edge should surface a non-finite intersection"
        );
    }

    #[test]
    fn halfspace_intersection_from_nd_supports_bounded_3d_tetrahedron() {
        let halfspaces = vec![
            vec![-1.0, 0.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, -1.0, 0.0],
            vec![1.0, 1.0, 1.0, -1.0],
        ];
        let hs = HalfspaceIntersection::from_nd(&halfspaces, &[0.2, 0.2, 0.2]).expect("3D bounded");
        assert_eq!(hs.ndim, 3);
        assert!(hs.is_bounded);
        assert_eq!(hs.dual_vertices, vec![0, 1, 2, 3]);
        for expected in [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ] {
            assert!(
                point_set_contains(&hs.intersections, &expected),
                "missing 3D intersection {expected:?}: {:?}",
                hs.intersections
            );
        }
        assert!(hs.dual_area.is_nan());
        assert!(hs.dual_volume.is_nan());
        assert!(hs.dual_equations.is_empty());
    }

    #[test]
    fn halfspace_intersection_from_nd_rejects_invalid_higher_dimensional_inputs() {
        let too_few_halfspaces = vec![vec![-1.0, 0.0, 0.0, 0.0]; 4];
        let err = HalfspaceIntersection::from_nd(&too_few_halfspaces, &[0.25, 0.25, 0.25, 0.25])
            .expect_err("4D input needs at least 5 halfspaces");
        assert!(matches!(err, SpatialError::Qhull(_)));

        let bad_rows = vec![vec![-1.0, 0.0, 0.0], vec![0.0, -1.0], vec![1.0, 1.0, -1.0]];
        let err = HalfspaceIntersection::from_nd(&bad_rows, &[0.25, 0.25])
            .expect_err("row shape mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn halfspace_intersection_add_halfspaces_recomputes_region() {
        let mut hs = HalfspaceIntersection::new(
            &[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, -1.0]],
            (0.5, 0.5),
        )
        .expect("initial halfspaces");
        assert!(!hs.is_bounded);

        hs.add_halfspaces(&[vec![0.0, 1.0, -1.0]], false)
            .expect("append y <= 1");
        assert!(hs.is_bounded);
        assert_eq!(hs.nineq, 4);
        assert!(point_set_contains(&hs.intersections, &[1.0, 1.0]));
    }

    // ── Delaunay tests ──────────────────────────────────────────────

    #[test]
    fn delaunay_square_triangulates() {
        let points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let tri = Delaunay::new(&points).expect("triangulation");
        assert_eq!(tri.points, points);
        assert_eq!(
            tri.simplices.len(),
            2,
            "square should split into 2 triangles"
        );
        for &(a, b, c) in &tri.simplices {
            assert!(a < points.len() && b < points.len() && c < points.len());
            assert_ne!(cross(points[a], points[b], points[c]), 0.0);
        }
    }

    #[test]
    fn delaunay_find_simplex_inside_triangle() {
        let points = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)];
        let tri = Delaunay::new(&points).expect("triangulation");
        let (_, l1, l2, l3) = tri.find_simplex((0.25, 0.5)).expect("inside simplex");
        assert!(l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0);
        assert!(((l1 + l2 + l3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn delaunay_find_simplex_outside_returns_none() {
        let points = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)];
        let tri = Delaunay::new(&points).expect("triangulation");
        assert!(tri.find_simplex((3.0, 3.0)).is_none());
    }

    #[test]
    fn delaunay_collinear_points_rejected() {
        let points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let err = Delaunay::new(&points).expect_err("collinear");
        assert!(matches!(err, SpatialError::Qhull(_)));
    }

    #[test]
    fn delaunay_too_few_points_rejected() {
        let points = [(0.0, 0.0), (1.0, 0.0)];
        let err = Delaunay::new(&points).expect_err("too few");
        assert!(matches!(err, SpatialError::Qhull(_)));
    }

    #[test]
    fn delaunay_non_finite_points_rejected() {
        let points = [(0.0, 0.0), (1.0, 0.0), (f64::NAN, f64::NAN)];
        let err = Delaunay::new(&points).expect_err("non-finite points");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    // ── Voronoi tests ───────────────────────────────────────────────

    #[test]
    fn voronoi_triangle_has_single_vertex_and_infinite_ridges() {
        let points = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)];
        let vor = Voronoi::new(&points).expect("voronoi");

        assert_eq!(vor.points, points);
        assert_eq!(vor.vertices.len(), 1);
        assert!((vor.vertices[0].0 - 1.0).abs() < 1e-10);
        assert!((vor.vertices[0].1 - 0.75).abs() < 1e-10);
        assert_eq!(vor.ridge_points.len(), 3);
        assert_eq!(vor.ridge_vertices.len(), 3);
        assert!(vor.ridge_vertices.iter().all(|&(a, b)| a == -1 && b == 0));
        assert_eq!(vor.point_region.len(), 3);
        for &region_idx in &vor.point_region {
            let region = &vor.regions[region_idx];
            assert!(region.contains(&-1));
            assert!(region.contains(&0));
        }
    }

    #[test]
    fn voronoi_center_point_gets_finite_region() {
        let points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)];
        let vor = Voronoi::new(&points).expect("voronoi");

        assert_eq!(vor.vertices.len(), 4);
        let center_region = &vor.regions[vor.point_region[4]];
        assert_eq!(center_region.len(), 4);
        assert!(center_region.iter().all(|&idx| idx >= 0));

        let expected = [(2.0, 1.0), (1.0, 0.0), (1.0, 2.0), (0.0, 1.0)];
        for &(ex, ey) in &expected {
            assert!(
                vor.vertices
                    .iter()
                    .any(|&(vx, vy)| (vx - ex).abs() < 1e-10 && (vy - ey).abs() < 1e-10),
                "missing Voronoi vertex ({ex}, {ey})"
            );
        }

        for &vertex_idx in center_region {
            let vertex = vor.vertices[vertex_idx as usize];
            assert!(
                expected
                    .iter()
                    .any(|&(ex, ey)| (vertex.0 - ex).abs() < 1e-10 && (vertex.1 - ey).abs() < 1e-10),
                "unexpected center-region vertex ({}, {})",
                vertex.0,
                vertex.1
            );
        }
    }

    // ── Spherical Voronoi tests ─────────────────────────────────────

    #[test]
    fn spherical_voronoi_tetrahedron_has_four_vertices() {
        let scale = 1.0 / 3.0_f64.sqrt();
        let points = [
            [scale, scale, scale],
            [scale, -scale, -scale],
            [-scale, scale, -scale],
            [-scale, -scale, scale],
        ];
        let sv = SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 1.0).expect("spherical voronoi");

        assert_eq!(sv.points, points);
        assert_eq!(sv.vertices.len(), 4);
        assert_eq!(sv.regions.len(), 4);
        for region in &sv.regions {
            assert_eq!(region.len(), 3);
            for &vertex_idx in region {
                assert!(vertex_idx < sv.vertices.len());
            }
        }
        for vertex in &sv.vertices {
            assert!((norm3(*vertex) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn spherical_voronoi_rejects_wrong_radius() {
        let points = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ];
        let err = SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 2.0).expect_err("radius");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn spherical_voronoi_rejects_duplicate_points() {
        let points = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let err = SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 1.0).expect_err("duplicate");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn spherical_voronoi_rejects_coplanar_great_circle_points() {
        let points = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let err = SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 1.0).expect_err("degenerate");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    // ── Procrustes tests ─────────────────────────────────────────────

    #[test]
    fn procrustes_identical() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = procrustes(&data, &data).expect("procrustes");
        assert!(
            result.disparity < 1e-10,
            "identical data should have 0 disparity: {}",
            result.disparity
        );
    }

    #[test]
    fn procrustes_rotated() {
        // data2 is data1 rotated 90 degrees
        let data1 = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let data2 = vec![vec![0.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0]];
        let result = procrustes(&data1, &data2).expect("procrustes");
        assert!(
            result.disparity < 0.01,
            "rotated data should align: disparity = {}",
            result.disparity
        );
    }

    #[test]
    fn procrustes_scaled() {
        // data2 is data1 scaled by 2 (should align after normalization)
        let data1 = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let data2: Vec<Vec<f64>> = data1
            .iter()
            .map(|p| p.iter().map(|&v| v * 2.0).collect())
            .collect();
        let result = procrustes(&data1, &data2).expect("procrustes");
        assert!(
            result.disparity < 1e-10,
            "scaled data should align: disparity = {}",
            result.disparity
        );
    }

    // ── Geometric SLERP tests ────────────────────────────────────────

    #[test]
    fn slerp_endpoints() {
        let start = vec![1.0, 0.0, 0.0];
        let end = vec![0.0, 1.0, 0.0];
        let result = geometric_slerp(&start, &end, &[0.0, 1.0]).expect("slerp");
        assert_eq!(result.len(), 2);
        // t=0 → start
        assert!((result[0][0] - 1.0).abs() < 1e-10);
        // t=1 → end
        assert!((result[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn slerp_midpoint_on_sphere() {
        let start = vec![1.0, 0.0];
        let end = vec![0.0, 1.0];
        let result = geometric_slerp(&start, &end, &[0.5]).expect("slerp");
        // Midpoint should be on unit circle
        let norm: f64 = result[0].iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "midpoint should be on unit sphere: norm = {norm}"
        );
        // Should be at 45 degrees
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (result[0][0] - expected).abs() < 1e-10,
            "slerp midpoint x: {}",
            result[0][0]
        );
    }

    #[test]
    fn slerp_dimension_mismatch() {
        let err = geometric_slerp(&[1.0, 0.0], &[1.0, 0.0, 0.0], &[0.5]).expect_err("dim");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    // ── Distance metric edge case tests ───────────────────────────

    #[test]
    fn cosine_zero_vector_returns_zero() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine(&[1.0, 2.0], &[0.0, 0.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn euclidean_identical_points() {
        assert_eq!(euclidean(&[3.0, 4.0], &[3.0, 4.0]), 0.0);
    }

    #[test]
    fn hamming_identical_vectors() {
        assert_eq!(hamming(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn hamming_completely_different() {
        assert!((hamming(&[1.0, 2.0], &[3.0, 4.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn chebyshev_single_dimension() {
        assert!((chebyshev(&[5.0], &[2.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn minkowski_special_cases() {
        let a = [1.0, 2.0];
        let b = [4.0, 6.0];
        // p=1 should equal cityblock
        assert!((minkowski(&a, &b, 1.0) - cityblock(&a, &b)).abs() < 1e-12);
        // p=2 should equal euclidean
        assert!((minkowski(&a, &b, 2.0) - euclidean(&a, &b)).abs() < 1e-12);
        // p=inf should equal chebyshev
        assert!((minkowski(&a, &b, f64::INFINITY) - chebyshev(&a, &b)).abs() < 1e-12);
    }

    #[test]
    fn canberra_single_zero_pair() {
        // When both elements are zero at same index, that term contributes 0
        assert_eq!(canberra(&[0.0], &[0.0]), 0.0);
    }

    // ── Hausdorff distance validation tests ─────────────────────────────

    #[test]
    fn directed_hausdorff_basic() {
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let d = directed_hausdorff(&xa, &xb).expect("valid input");
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn directed_hausdorff_empty_xa_rejected() {
        let err = directed_hausdorff(&[], &[vec![1.0]]).expect_err("empty xa");
        assert!(matches!(err, SpatialError::EmptyData));
    }

    #[test]
    fn directed_hausdorff_empty_xb_rejected() {
        let err = directed_hausdorff(&[vec![1.0]], &[]).expect_err("empty xb");
        assert!(matches!(err, SpatialError::EmptyData));
    }

    #[test]
    fn directed_hausdorff_nan_rejected() {
        let xa = vec![vec![1.0, 2.0]];
        let xb = vec![vec![3.0, f64::NAN]];
        let err = directed_hausdorff(&xa, &xb).expect_err("NaN input");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn directed_hausdorff_dimension_mismatch_rejected() {
        let xa = vec![vec![1.0, 2.0]];
        let xb = vec![vec![3.0]];
        let err = directed_hausdorff(&xa, &xb).expect_err("dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn hausdorff_distance_basic() {
        let xa = vec![vec![0.0, 0.0]];
        let xb = vec![vec![3.0, 4.0]];
        let d = hausdorff_distance(&xa, &xb).expect("valid input");
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn hausdorff_distance_symmetric() {
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 2.0]];
        let d1 = hausdorff_distance(&xa, &xb).expect("valid");
        let d2 = hausdorff_distance(&xb, &xa).expect("valid");
        assert!((d1 - d2).abs() < 1e-10);
    }

    // ── Medoid validation tests ─────────────────────────────────────────

    #[test]
    fn medoid_basic() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.0]];
        // Point 2 (0.5, 0) minimizes total distance
        let idx = medoid(&points).expect("non-empty");
        assert_eq!(idx, 2);
    }

    #[test]
    fn medoid_empty_returns_none() {
        let points: Vec<Vec<f64>> = vec![];
        assert!(medoid(&points).is_none());
    }

    #[test]
    fn medoid_single_point() {
        let points = vec![vec![5.0, 5.0]];
        let idx = medoid(&points).expect("single point");
        assert_eq!(idx, 0);
    }

    // ── Nearest neighbors validation tests ──────────────────────────────

    #[test]
    fn nearest_neighbors_basic() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![10.0, 0.0]];
        let (indices, distances) = nearest_neighbors(&data);
        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
        // Point 0's nearest is point 1
        assert_eq!(indices[0], Some(1));
        assert!((distances[0] - 1.0).abs() < 1e-10);
        // Point 1's nearest is point 0
        assert_eq!(indices[1], Some(0));
        // Point 2's nearest is point 1
        assert_eq!(indices[2], Some(1));
    }

    #[test]
    fn nearest_neighbors_empty() {
        let (indices, distances) = nearest_neighbors(&[]);
        assert!(indices.is_empty());
        assert!(distances.is_empty());
    }

    #[test]
    fn nearest_neighbors_single_point() {
        let data = vec![vec![5.0, 5.0]];
        let (indices, distances) = nearest_neighbors(&data);
        assert_eq!(indices.len(), 1);
        assert!(indices[0].is_none());
        assert_eq!(distances[0], f64::INFINITY);
    }
}
