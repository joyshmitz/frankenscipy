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

    let result: Vec<Vec<f64>> = xa
        .iter()
        .map(|a| xb.iter().map(|b| euclidean(a, b)).collect())
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
