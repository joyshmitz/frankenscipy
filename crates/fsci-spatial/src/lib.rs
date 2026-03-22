#![forbid(unsafe_code)]

//! Spatial data structures and algorithms for FrankenSciPy.
//!
//! Matches `scipy.spatial` core types:
//! - `KDTree` — k-d tree for fast nearest-neighbor queries
//! - `distance` — pairwise distance computations

/// Error type for spatial operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpatialError {
    EmptyData,
    DimensionMismatch { expected: usize, actual: usize },
    InvalidArgument(String),
}

impl std::fmt::Display for SpatialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "empty data"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidArgument(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for SpatialError {}

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
        .fold(0.0_f64, f64::max)
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

// ══════════════════════════════════════════════════════════════════════
// KDTree
// ══════════════════════════════════════════════════════════════════════

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

        let k = k.min(self.nodes.len());
        let mut results: Vec<(usize, f64)> = Vec::with_capacity(k);

        // Simple approach: find k nearest via repeated queries with exclusion
        // For a proper implementation, use a max-heap bounded to k elements.
        knn_search(&self.nodes, 0, query, k, &mut results);

        // Sort by distance and convert from squared to actual distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
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
    pub fn count_neighbors(&self, other: &KDTree, r: f64) -> usize {
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return 0;
        }
        let r_sq = r * r;
        let mut count = 0;
        for other_node in &other.nodes {
            count += ball_search_count(&self.nodes, 0, &other_node.point, r_sq);
        }
        count
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

    // Sort by the split dimension
    indices.sort_by(|&a, &b| {
        data[a][split_dim]
            .partial_cmp(&data[b][split_dim])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let median = indices.len() / 2;
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
    if results.len() < k {
        results.push((node.index, dist_sq));
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    } else if dist_sq < results[k - 1].1 {
        results[k - 1] = (node.index, dist_sq);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
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
            return Err(SpatialError::InvalidArgument(
                "convex hull requires at least 3 points".to_string(),
            ));
        }

        let n = points.len();

        // Sort points by x, then by y (lexicographic)
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            points[a]
                .0
                .partial_cmp(&points[b].0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    points[a]
                        .1
                        .partial_cmp(&points[b].1)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
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
            return Err(SpatialError::InvalidArgument(
                "convex hull requires at least 3 non-collinear points".to_string(),
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

/// Cross product of vectors OA and OB where O, A, B are 2D points.
/// Positive = counter-clockwise, negative = clockwise, zero = collinear.
fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
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
            for k in 0..d {
                mtm[i][j] += m[k][i] * m[k][j]; // M^T M
            }
        }
    }

    // For 2D case (most common), solve directly
    // For general case, use iterative square root inverse
    let rotation = if d == 2 {
        // 2D: compute polar decomposition analytically
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        let trace_mtm = mtm[0][0] + mtm[1][1];
        let det_mtm = mtm[0][0] * mtm[1][1] - mtm[0][1] * mtm[1][0];
        let s = (trace_mtm + 2.0 * det_mtm.max(0.0).sqrt()).max(0.0).sqrt();
        if s > 1e-15 {
            // R = M / s (scaled)
            vec![
                vec![m[0][0] / s + m[1][1] / s, m[0][1] / s - m[1][0] / s],
                vec![m[1][0] / s - m[0][1] / s, m[0][0] / s + m[1][1] / s],
            ]
        } else {
            // Degenerate: use identity
            vec![vec![1.0, 0.0], vec![0.0, 1.0]]
        }
    } else {
        // General: Newton iteration for (M^T M)^{-1/2}, then R = M * result
        // Start with identity as initial guess
        let mut y = vec![vec![0.0; d]; d];
        for i in 0..d { y[i][i] = 1.0; }

        for _ in 0..20 {
            // Y_new = 0.5 * Y * (3I - M^T M Y² )
            let mut y2 = mat_mul(&y, &y, d);
            let mut my2 = mat_mul(&mtm, &y2, d);
            for i in 0..d {
                for j in 0..d {
                    my2[i][j] = if i == j { 3.0 } else { 0.0 } - my2[i][j];
                }
            }
            let y_new = mat_mul(&y, &my2, d);
            let mut max_diff = 0.0_f64;
            for i in 0..d {
                for j in 0..d {
                    max_diff = max_diff.max((y_new[i][j] * 0.5 - y[i][j]).abs());
                    y[i][j] = y_new[i][j] * 0.5;
                }
            }
            if max_diff < 1e-14 { break; }
        }
        mat_mul(&m, &y, d)
    };

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
pub fn geometric_slerp(start: &[f64], end: &[f64], t_values: &[f64]) -> Result<Vec<Vec<f64>>, SpatialError> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn kdtree_empty_rejected() {
        let err = KDTree::new(&[]).expect_err("empty");
        assert!(matches!(err, SpatialError::EmptyData));
    }

    #[test]
    fn kdtree_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let err = tree.query(&[1.0]).expect_err("dim mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
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
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
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
        let count = tree1.count_neighbors(&tree2, 1.0);
        assert_eq!(count, 2);
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
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn convex_hull_collinear_points_rejected() {
        let points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let err = ConvexHull::new(&points).expect_err("collinear");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
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
        let data2: Vec<Vec<f64>> = data1.iter().map(|p| p.iter().map(|&v| v * 2.0).collect()).collect();
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
}
