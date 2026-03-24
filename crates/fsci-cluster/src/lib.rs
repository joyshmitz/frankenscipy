#![forbid(unsafe_code)]

//! Clustering algorithms for FrankenSciPy.
//!
//! Matches `scipy.cluster` core operations:
//! - `scipy.cluster.vq` — K-means clustering and vector quantization
//! - `scipy.cluster.hierarchy` — Hierarchical/agglomerative clustering

/// Error type for clustering operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterError {
    InvalidArgument(String),
    EmptyData,
    ConvergenceFailed(String),
}

impl std::fmt::Display for ClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::EmptyData => write!(f, "empty data"),
            Self::ConvergenceFailed(msg) => write!(f, "convergence failed: {msg}"),
        }
    }
}

impl std::error::Error for ClusterError {}

// ══════════════════════════════════════════════════════════════════════
// K-Means Clustering
// ══════════════════════════════════════════════════════════════════════

/// Result of K-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Cluster centroids, shape (k, d).
    pub centroids: Vec<Vec<f64>>,
    /// Cluster labels for each data point.
    pub labels: Vec<usize>,
    /// Sum of squared distances to nearest centroid (inertia).
    pub inertia: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
}

/// K-means clustering using Lloyd's algorithm.
///
/// Matches `scipy.cluster.vq.kmeans2`.
///
/// # Arguments
/// * `data` — Input data points, each a Vec of features.
/// * `k` — Number of clusters.
/// * `max_iter` — Maximum iterations.
/// * `seed` — Random seed for initialization.
pub fn kmeans(
    data: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Result<KMeansResult, ClusterError> {
    let n = data.len();
    if n == 0 {
        return Err(ClusterError::EmptyData);
    }
    if k == 0 || k > n {
        return Err(ClusterError::InvalidArgument(format!(
            "k={k} must be in [1, n={n}]"
        )));
    }
    let d = data[0].len();

    // K-means++ initialization
    let mut centroids = kmeans_plusplus_init(data, k, seed);
    let mut labels = vec![0usize; n];
    let mut inertia = f64::INFINITY;

    for iter in 0..max_iter {
        // Assignment step
        let mut new_inertia = 0.0;
        for (i, point) in data.iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_c = 0;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = sq_dist(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_c = c;
                }
            }
            labels[i] = best_c;
            new_inertia += min_dist;
        }

        // Check convergence
        if (inertia - new_inertia).abs() < 1e-12 * inertia.abs().max(1.0) {
            return Ok(KMeansResult {
                centroids,
                labels,
                inertia: new_inertia,
                n_iter: iter + 1,
            });
        }
        inertia = new_inertia;

        // Update step
        let mut counts = vec![0usize; k];
        let mut new_centroids = vec![vec![0.0; d]; k];
        for (i, point) in data.iter().enumerate() {
            let c = labels[i];
            counts[c] += 1;
            for (j, &val) in point.iter().enumerate() {
                new_centroids[c][j] += val;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for val in &mut new_centroids[c][..d] {
                    *val /= counts[c] as f64;
                }
            } else {
                // Empty cluster: keep old centroid
                new_centroids[c] = centroids[c].clone();
            }
        }
        centroids = new_centroids;
    }

    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        n_iter: max_iter,
    })
}

/// K-means++ initialization.
fn kmeans_plusplus_init(data: &[Vec<f64>], k: usize, seed: u64) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut rng = seed;
    let next_rng = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut centroids = Vec::with_capacity(k);

    // First centroid: random
    let idx = (next_rng(&mut rng) * n as f64) as usize % n;
    centroids.push(data[idx].clone());

    // Remaining centroids: probability proportional to D²
    for _ in 1..k {
        let mut dists = vec![f64::INFINITY; n];
        for (i, point) in data.iter().enumerate() {
            for c in &centroids {
                let d = sq_dist(point, c);
                dists[i] = dists[i].min(d);
            }
        }

        let total: f64 = dists.iter().sum();
        if total <= 0.0 {
            // All points are at existing centroids; pick randomly
            let idx = (next_rng(&mut rng) * n as f64) as usize % n;
            centroids.push(data[idx].clone());
            continue;
        }

        let threshold = next_rng(&mut rng) * total;
        let mut cumsum = 0.0;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }

    centroids
}

/// Mini-batch K-means for large datasets.
///
/// Matches `sklearn.cluster.MiniBatchKMeans` (used alongside scipy).
pub fn mini_batch_kmeans(
    data: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    batch_size: usize,
    seed: u64,
) -> Result<KMeansResult, ClusterError> {
    let n = data.len();
    if n == 0 {
        return Err(ClusterError::EmptyData);
    }
    if k == 0 || k > n {
        return Err(ClusterError::InvalidArgument(format!(
            "k={k} must be in [1, n={n}]"
        )));
    }
    let d = data[0].len();
    let batch = batch_size.min(n);

    let mut centroids = kmeans_plusplus_init(data, k, seed);
    let mut counts = vec![0usize; k];
    let mut rng = seed.wrapping_add(12345);

    for _ in 0..max_iter {
        // Sample mini-batch
        let mut batch_indices = Vec::with_capacity(batch);
        for _ in 0..batch {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            batch_indices.push((rng >> 33) as usize % n);
        }

        // Assignment
        let mut batch_labels = Vec::with_capacity(batch);
        for &idx in &batch_indices {
            let point = &data[idx];
            let mut min_dist = f64::INFINITY;
            let mut best_c = 0;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = sq_dist(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_c = c;
                }
            }
            batch_labels.push(best_c);
        }

        // Update with learning rate
        for (bi, &idx) in batch_indices.iter().enumerate() {
            let c = batch_labels[bi];
            counts[c] += 1;
            let lr = 1.0 / counts[c] as f64;
            for j in 0..d {
                centroids[c][j] += lr * (data[idx][j] - centroids[c][j]);
            }
        }
    }

    // Final assignment
    let mut labels = vec![0usize; n];
    let mut inertia = 0.0;
    for (i, point) in data.iter().enumerate() {
        let mut min_dist = f64::INFINITY;
        let mut best_c = 0;
        for (c, centroid) in centroids.iter().enumerate() {
            let dist = sq_dist(point, centroid);
            if dist < min_dist {
                min_dist = dist;
                best_c = c;
            }
        }
        labels[i] = best_c;
        inertia += min_dist;
    }

    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        n_iter: max_iter,
    })
}

/// Vector quantization: assign each observation to the nearest centroid.
///
/// Matches `scipy.cluster.vq.vq`.
pub fn vq(data: &[Vec<f64>], centroids: &[Vec<f64>]) -> (Vec<usize>, Vec<f64>) {
    let mut labels = Vec::with_capacity(data.len());
    let mut dists = Vec::with_capacity(data.len());

    for point in data {
        let mut min_dist = f64::INFINITY;
        let mut best_c = 0;
        for (c, centroid) in centroids.iter().enumerate() {
            let d = sq_dist(point, centroid).sqrt();
            if d < min_dist {
                min_dist = d;
                best_c = c;
            }
        }
        labels.push(best_c);
        dists.push(min_dist);
    }

    (labels, dists)
}

/// Whiten observations by dividing by per-feature standard deviation.
///
/// Matches `scipy.cluster.vq.whiten`.
pub fn whiten(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return vec![];
    }
    let n = data.len();
    let d = data[0].len();

    // Compute per-feature std
    let mut means = vec![0.0; d];
    for point in data {
        for (j, &v) in point.iter().enumerate() {
            means[j] += v;
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }

    let mut vars = vec![0.0; d];
    for point in data {
        for (j, &v) in point.iter().enumerate() {
            vars[j] += (v - means[j]).powi(2);
        }
    }

    let stds: Vec<f64> = vars.iter().map(|&v| (v / n as f64).sqrt()).collect();

    data.iter()
        .map(|point| {
            point
                .iter()
                .zip(stds.iter())
                .map(|(&v, &s)| if s > 0.0 { v / s } else { v })
                .collect()
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════
// Hierarchical Clustering
// ══════════════════════════════════════════════════════════════════════

/// Linkage methods for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkageMethod {
    Single,
    Complete,
    Average,
    Ward,
}

/// Hierarchical clustering linkage matrix.
///
/// Each row [i, j, dist, count] represents a merge: clusters i and j
/// merged at distance dist, producing a cluster with count observations.
///
/// Matches `scipy.cluster.hierarchy.linkage`.
pub fn linkage(data: &[Vec<f64>], method: LinkageMethod) -> Result<Vec<[f64; 4]>, ClusterError> {
    let n = data.len();
    if n < 2 {
        return Err(ClusterError::InvalidArgument(
            "need at least 2 observations".to_string(),
        ));
    }

    // Compute pairwise distance matrix
    let mut dist_mat = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n {
        dist_mat[i][i] = 0.0;
        for j in i + 1..n {
            let d = sq_dist(&data[i], &data[j]).sqrt();
            dist_mat[i][j] = d;
            dist_mat[j][i] = d;
        }
    }

    // Active cluster tracking
    let mut active = vec![true; 2 * n - 1];
    let mut cluster_size = vec![1usize; 2 * n - 1];
    let mut result = Vec::with_capacity(n - 1);

    // Extend dist_mat to handle new clusters
    let total = 2 * n - 1;
    let mut inter_dist = vec![vec![f64::INFINITY; total]; total];
    for i in 0..n {
        for j in 0..n {
            inter_dist[i][j] = dist_mat[i][j];
        }
    }

    for step in 0..n - 1 {
        let new_id = n + step;

        // Find closest pair of active clusters
        let mut min_d = f64::INFINITY;
        let mut mi = 0;
        let mut mj = 0;
        for i in 0..new_id {
            if !active[i] {
                continue;
            }
            for j in i + 1..new_id {
                if !active[j] {
                    continue;
                }
                if inter_dist[i][j] < min_d {
                    min_d = inter_dist[i][j];
                    mi = i;
                    mj = j;
                }
            }
        }

        let new_size = cluster_size[mi] + cluster_size[mj];
        result.push([mi as f64, mj as f64, min_d, new_size as f64]);

        active[mi] = false;
        active[mj] = false;
        active[new_id] = true;
        cluster_size[new_id] = new_size;

        // Update distances from new cluster to all remaining active clusters
        for k in 0..new_id {
            if !active[k] || k == new_id {
                continue;
            }
            let d_ki = inter_dist[k][mi];
            let d_kj = inter_dist[k][mj];
            let new_dist = match method {
                LinkageMethod::Single => d_ki.min(d_kj),
                LinkageMethod::Complete => d_ki.max(d_kj),
                LinkageMethod::Average => {
                    let ni = cluster_size[mi] as f64;
                    let nj = cluster_size[mj] as f64;
                    (ni * d_ki + nj * d_kj) / (ni + nj)
                }
                LinkageMethod::Ward => {
                    let ni = cluster_size[mi] as f64;
                    let nj = cluster_size[mj] as f64;
                    let nk = cluster_size[k] as f64;
                    let nt = ni + nj + nk;
                    (((nk + ni) * d_ki * d_ki + (nk + nj) * d_kj * d_kj - nk * min_d * min_d) / nt)
                        .max(0.0)
                        .sqrt()
                }
            };
            inter_dist[k][new_id] = new_dist;
            inter_dist[new_id][k] = new_dist;
        }
    }

    Ok(result)
}

/// Cut a linkage tree to form flat clusters.
///
/// Matches `scipy.cluster.hierarchy.fcluster` with criterion='maxclust'.
pub fn fcluster(z: &[[f64; 4]], max_clusters: usize) -> Vec<usize> {
    let n = z.len() + 1;
    if max_clusters >= n || max_clusters == 0 {
        return (0..n).collect();
    }

    // Each leaf is its own cluster initially
    let mut cluster_of = vec![0usize; 2 * n - 1];
    for (i, cluster) in cluster_of.iter_mut().enumerate().take(n) {
        *cluster = i;
    }

    // Process merges in order, stopping when we have max_clusters
    let n_merges = n - max_clusters;
    for (step, row) in z.iter().enumerate().take(n_merges) {
        let new_id = n + step;
        let ci = row[0] as usize;
        let cj = row[1] as usize;
        // Assign the new cluster label to both merged clusters
        let label = cluster_of[ci].min(cluster_of[cj]);
        // Propagate labels
        let old_ci = cluster_of[ci];
        let old_cj = cluster_of[cj];
        for v in cluster_of.iter_mut().take(new_id + 1) {
            if *v == old_ci || *v == old_cj {
                *v = label;
            }
        }
        cluster_of[new_id] = label;
    }

    // Renumber labels to be contiguous 0..k-1
    let leaf_labels: Vec<usize> = cluster_of[..n].to_vec();
    let mut unique: Vec<usize> = leaf_labels.clone();
    unique.sort();
    unique.dedup();
    leaf_labels
        .iter()
        .map(|&l| unique.iter().position(|&u| u == l).unwrap())
        .collect()
}

/// Compute cophenetic distances from a linkage matrix.
///
/// Returns the cophenetic distance matrix (condensed form).
/// Matches `scipy.cluster.hierarchy.cophenet`.
pub fn cophenet(z: &[[f64; 4]]) -> Vec<f64> {
    let n = z.len() + 1;
    let mut membership = vec![vec![]; 2 * n - 1];
    for (i, mem) in membership.iter_mut().enumerate().take(n) {
        *mem = vec![i];
    }

    let mut condensed = vec![0.0; n * (n - 1) / 2];

    for (step, row) in z.iter().enumerate() {
        let ci = row[0] as usize;
        let cj = row[1] as usize;
        let dist = row[2];
        let new_id = n + step;

        // Set cophenetic distance for all pairs (one from ci, one from cj)
        for &a in &membership[ci] {
            for &b in &membership[cj] {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let idx = n * lo - lo * (lo + 1) / 2 + hi - lo - 1;
                condensed[idx] = dist;
            }
        }

        // Merge membership lists
        let mut merged = membership[ci].clone();
        merged.extend_from_slice(&membership[cj]);
        membership[new_id] = merged;
    }

    condensed
}

/// Compute the inconsistency statistic for each merge.
///
/// Matches `scipy.cluster.hierarchy.inconsistent`.
pub fn inconsistent(z: &[[f64; 4]], depth: usize) -> Vec<[f64; 4]> {
    let n = z.len() + 1;
    let mut result = Vec::with_capacity(z.len());

    for (step, row) in z.iter().enumerate() {
        // Collect distances of all merges within `depth` levels below this merge
        let mut dists = Vec::new();
        collect_depths(z, n + step, n, depth, &mut dists);

        let count = dists.len() as f64;
        let mean = if count > 0.0 {
            dists.iter().sum::<f64>() / count
        } else {
            row[2]
        };
        let std = if count > 1.0 {
            let var = dists.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / (count - 1.0);
            var.sqrt()
        } else {
            0.0
        };
        let incon = if std > 0.0 {
            (row[2] - mean) / std
        } else {
            0.0
        };

        result.push([mean, std, count, incon]);
    }

    result
}

fn collect_depths(z: &[[f64; 4]], node: usize, n: usize, depth: usize, dists: &mut Vec<f64>) {
    if depth == 0 || node < n {
        return;
    }
    let step = node - n;
    if step < z.len() {
        dists.push(z[step][2]);
        collect_depths(z, z[step][0] as usize, n, depth - 1, dists);
        collect_depths(z, z[step][1] as usize, n, depth - 1, dists);
    }
}

// ══════════════════════════════════════════════════════════════════════
// DBSCAN
// ══════════════════════════════════════════════════════════════════════

/// Result of DBSCAN clustering.
#[derive(Debug, Clone)]
pub struct DbscanResult {
    /// Cluster labels (-1 = noise).
    pub labels: Vec<i64>,
    /// Indices of core samples.
    pub core_sample_indices: Vec<usize>,
    /// Number of clusters found.
    pub n_clusters: usize,
}

/// DBSCAN density-based clustering.
///
/// Matches `sklearn.cluster.DBSCAN`.
pub fn dbscan(
    data: &[Vec<f64>],
    eps: f64,
    min_samples: usize,
) -> Result<DbscanResult, ClusterError> {
    let n = data.len();
    if n == 0 {
        return Err(ClusterError::EmptyData);
    }

    let eps2 = eps * eps;
    let mut labels = vec![-1i64; n];
    let mut visited = vec![false; n];
    let mut core_samples = Vec::new();
    let mut cluster_id = 0i64;

    // Find neighbors for each point
    let neighbors = |idx: usize| -> Vec<usize> {
        (0..n)
            .filter(|&j| sq_dist(&data[idx], &data[j]) <= eps2)
            .collect()
    };

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        let nbrs = neighbors(i);
        if nbrs.len() < min_samples {
            // Noise (may be reclassified later)
            continue;
        }

        // Core point: start new cluster
        core_samples.push(i);
        labels[i] = cluster_id;

        let mut queue = std::collections::VecDeque::from(nbrs);
        while let Some(j) = queue.pop_front() {
            if labels[j] == -1 {
                labels[j] = cluster_id; // was noise, now border
            }
            if visited[j] {
                continue;
            }
            visited[j] = true;

            let j_nbrs = neighbors(j);
            if j_nbrs.len() >= min_samples {
                core_samples.push(j);
                for &nb in &j_nbrs {
                    if labels[nb] == -1 {
                        queue.push_back(nb);
                    }
                }
            }
        }

        cluster_id += 1;
    }

    core_samples.sort();
    core_samples.dedup();

    Ok(DbscanResult {
        labels,
        core_sample_indices: core_samples,
        n_clusters: cluster_id as usize,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
        .sum()
}

/// Silhouette score: measure of cluster quality.
///
/// Returns mean silhouette coefficient over all samples.
/// Matches `sklearn.metrics.silhouette_score`.
pub fn silhouette_score(data: &[Vec<f64>], labels: &[usize]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }

    let k = labels.iter().cloned().max().unwrap_or(0) + 1;
    let mut total = 0.0;

    for i in 0..n {
        let li = labels[i];

        // a(i) = mean distance to same-cluster points
        let mut a_sum = 0.0;
        let mut a_count = 0;
        for j in 0..n {
            if i != j && labels[j] == li {
                a_sum += sq_dist(&data[i], &data[j]).sqrt();
                a_count += 1;
            }
        }
        let a = if a_count > 0 {
            a_sum / a_count as f64
        } else {
            0.0
        };

        // b(i) = min over other clusters of mean distance
        let mut b = f64::INFINITY;
        for c in 0..k {
            if c == li {
                continue;
            }
            let mut c_sum = 0.0;
            let mut c_count = 0;
            for j in 0..n {
                if labels[j] == c {
                    c_sum += sq_dist(&data[i], &data[j]).sqrt();
                    c_count += 1;
                }
            }
            if c_count > 0 {
                b = b.min(c_sum / c_count as f64);
            }
        }

        let s = if a.max(b) > 0.0 {
            (b - a) / a.max(b)
        } else {
            0.0
        };
        total += s;
    }

    total / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_two_clusters() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.0],
        ];
        let result = kmeans(&data, 2, 100, 42).unwrap();
        assert_eq!(result.labels.len(), 6);
        // First 3 should be in one cluster, last 3 in another
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn kmeans_single_cluster() {
        let data = vec![vec![1.0, 2.0], vec![1.1, 2.1], vec![0.9, 1.9]];
        let result = kmeans(&data, 1, 10, 42).unwrap();
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn vq_assigns_nearest() {
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let data = vec![vec![1.0, 1.0], vec![9.0, 9.0]];
        let (labels, dists) = vq(&data, &centroids);
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);
        assert!(dists[0] < 2.0);
    }

    #[test]
    fn whiten_normalizes_std() {
        let data = vec![vec![1.0, 100.0], vec![2.0, 200.0], vec![3.0, 300.0]];
        let whitened = whiten(&data);
        // After whitening, std of each column should be ~1
        let n = whitened.len() as f64;
        for col in 0..2 {
            let mean: f64 = whitened.iter().map(|r| r[col]).sum::<f64>() / n;
            let var: f64 = whitened
                .iter()
                .map(|r| (r[col] - mean).powi(2))
                .sum::<f64>()
                / n;
            assert!(
                (var.sqrt() - 1.0).abs() < 0.01,
                "column {col} std = {}",
                var.sqrt()
            );
        }
    }

    #[test]
    fn linkage_single() {
        let data = vec![vec![0.0], vec![1.0], vec![5.0]];
        let z = linkage(&data, LinkageMethod::Single).unwrap();
        assert_eq!(z.len(), 2);
        // First merge: 0 and 1 (distance 1)
        assert_eq!(z[0][0], 0.0);
        assert_eq!(z[0][1], 1.0);
        assert!((z[0][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn fcluster_two_groups() {
        let data = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
        let z = linkage(&data, LinkageMethod::Complete).unwrap();
        let labels = fcluster(&z, 2);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn dbscan_two_clusters() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
        ];
        let result = dbscan(&data, 0.5, 2).unwrap();
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn dbscan_noise_detection() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![100.0, 100.0], // isolated noise point
        ];
        let result = dbscan(&data, 0.5, 2).unwrap();
        assert_eq!(result.labels[3], -1); // should be noise
    }

    #[test]
    fn silhouette_perfect_clusters() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels);
        assert!(score > 0.9, "silhouette = {score}, expected > 0.9");
    }

    #[test]
    fn cophenet_basic() {
        let data = vec![vec![0.0], vec![1.0], vec![5.0]];
        let z = linkage(&data, LinkageMethod::Single).unwrap();
        let d = cophenet(&z);
        assert_eq!(d.len(), 3); // C(3,2)
        // cophenetic distance between 0 and 1 = merge distance = 1.0
        assert!((d[0] - 1.0).abs() < 1e-10);
    }
}
