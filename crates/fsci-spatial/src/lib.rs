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

/// Distance metric identifiers for use with `pdist` and `cdist_metric`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    SqEuclidean,
    Cityblock,
    Chebyshev,
    Cosine,
    Correlation,
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

    let result: Vec<Vec<f64>> = xa
        .iter()
        .map(|a| xb.iter().map(|b| metric_distance(a, b, metric)).collect())
        .collect();
    Ok(result)
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
}
