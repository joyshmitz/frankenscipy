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
    unique.sort_unstable();
    unique.dedup();
    leaf_labels
        .iter()
        .map(|&l| unique.binary_search(&l).unwrap_or(0))
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
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "dbscan input must be finite".to_string(),
        ));
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

/// Calinski-Harabasz index: ratio of between-cluster to within-cluster dispersion.
///
/// Higher is better. Matches `sklearn.metrics.calinski_harabasz_score`.
pub fn calinski_harabasz_score(data: &[Vec<f64>], labels: &[usize]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let d = data[0].len();
    let k = labels.iter().cloned().max().unwrap_or(0) + 1;
    if k < 2 || n == k {
        return 0.0;
    }

    // Global centroid
    let mut global_centroid = vec![0.0; d];
    for point in data {
        for (j, &v) in point.iter().enumerate() {
            global_centroid[j] += v;
        }
    }
    for v in &mut global_centroid {
        *v /= n as f64;
    }

    // Cluster centroids and counts
    let mut centroids = vec![vec![0.0; d]; k];
    let mut counts = vec![0usize; k];
    for (i, point) in data.iter().enumerate() {
        let c = labels[i];
        counts[c] += 1;
        for (j, &v) in point.iter().enumerate() {
            centroids[c][j] += v;
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for v in &mut centroids[c] {
                *v /= counts[c] as f64;
            }
        }
    }

    // Between-cluster dispersion
    let mut bg = 0.0;
    for c in 0..k {
        bg += counts[c] as f64 * sq_dist(&centroids[c], &global_centroid);
    }

    // Within-cluster dispersion
    let mut wg = 0.0;
    for (i, point) in data.iter().enumerate() {
        wg += sq_dist(point, &centroids[labels[i]]);
    }

    if wg == 0.0 {
        return f64::INFINITY;
    }

    (bg / (k - 1) as f64) / (wg / (n - k) as f64)
}

/// Davies-Bouldin index: average similarity of clusters.
///
/// Lower is better. Matches `sklearn.metrics.davies_bouldin_score`.
pub fn davies_bouldin_score(data: &[Vec<f64>], labels: &[usize]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let d = data[0].len();
    let k = labels.iter().cloned().max().unwrap_or(0) + 1;
    if k < 2 {
        return 0.0;
    }

    // Cluster centroids
    let mut centroids = vec![vec![0.0; d]; k];
    let mut counts = vec![0usize; k];
    for (i, point) in data.iter().enumerate() {
        let c = labels[i];
        counts[c] += 1;
        for (j, &v) in point.iter().enumerate() {
            centroids[c][j] += v;
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for v in &mut centroids[c] {
                *v /= counts[c] as f64;
            }
        }
    }

    // Average within-cluster distance for each cluster
    let mut s = vec![0.0; k];
    for (i, point) in data.iter().enumerate() {
        s[labels[i]] += sq_dist(point, &centroids[labels[i]]).sqrt();
    }
    for c in 0..k {
        if counts[c] > 0 {
            s[c] /= counts[c] as f64;
        }
    }

    // Davies-Bouldin index
    let mut db = 0.0;
    for i in 0..k {
        let mut max_r = 0.0f64;
        for j in 0..k {
            if i != j {
                let d_ij = sq_dist(&centroids[i], &centroids[j]).sqrt();
                if d_ij > 0.0 {
                    max_r = max_r.max((s[i] + s[j]) / d_ij);
                }
            }
        }
        db += max_r;
    }

    db / k as f64
}

/// Adjusted Rand Index: similarity between two clusterings, adjusted for chance.
///
/// Matches `sklearn.metrics.adjusted_rand_score`.
pub fn adjusted_rand_score(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    let n = labels_true.len();
    if n != labels_pred.len() || n < 2 {
        return 0.0;
    }

    let k1 = labels_true.iter().cloned().max().unwrap_or(0) + 1;
    let k2 = labels_pred.iter().cloned().max().unwrap_or(0) + 1;

    // Contingency table
    let mut contingency = vec![vec![0i64; k2]; k1];
    for i in 0..n {
        contingency[labels_true[i]][labels_pred[i]] += 1;
    }

    // Row and column sums
    let row_sums: Vec<i64> = contingency.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<i64> = (0..k2)
        .map(|j| contingency.iter().map(|r| r[j]).sum())
        .collect();

    // Compute index using C(n,2) counts
    let comb2 = |x: i64| -> i64 { x * (x - 1) / 2 };

    let sum_comb_nij: i64 = contingency
        .iter()
        .flat_map(|r| r.iter())
        .map(|&v| comb2(v))
        .sum();
    let sum_comb_a: i64 = row_sums.iter().map(|&v| comb2(v)).sum();
    let sum_comb_b: i64 = col_sums.iter().map(|&v| comb2(v)).sum();
    let comb_n = comb2(n as i64);

    let expected = sum_comb_a as f64 * sum_comb_b as f64 / comb_n as f64;
    let max_index = (sum_comb_a + sum_comb_b) as f64 / 2.0;

    if (max_index - expected).abs() < 1e-15 {
        return if sum_comb_nij as f64 == expected {
            1.0
        } else {
            0.0
        };
    }

    (sum_comb_nij as f64 - expected) / (max_index - expected)
}

/// Normalized Mutual Information between two clusterings.
///
/// Matches `sklearn.metrics.normalized_mutual_info_score`.
pub fn normalized_mutual_info(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    let n = labels_true.len();
    if n != labels_pred.len() || n == 0 {
        return 0.0;
    }

    let k1 = labels_true.iter().cloned().max().unwrap_or(0) + 1;
    let k2 = labels_pred.iter().cloned().max().unwrap_or(0) + 1;
    let nf = n as f64;

    // Contingency table
    let mut contingency = vec![vec![0usize; k2]; k1];
    for i in 0..n {
        contingency[labels_true[i]][labels_pred[i]] += 1;
    }

    let row_sums: Vec<usize> = contingency.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<usize> = (0..k2)
        .map(|j| contingency.iter().map(|r| r[j]).sum())
        .collect();

    // Mutual information
    let mut mi = 0.0;
    for i in 0..k1 {
        for j in 0..k2 {
            if contingency[i][j] > 0 && row_sums[i] > 0 && col_sums[j] > 0 {
                let pij = contingency[i][j] as f64 / nf;
                let pi = row_sums[i] as f64 / nf;
                let pj = col_sums[j] as f64 / nf;
                mi += pij * (pij / (pi * pj)).ln();
            }
        }
    }

    // Entropies
    let h1: f64 = row_sums
        .iter()
        .filter(|&&s| s > 0)
        .map(|&s| {
            let p = s as f64 / nf;
            -p * p.ln()
        })
        .sum();
    let h2: f64 = col_sums
        .iter()
        .filter(|&&s| s > 0)
        .map(|&s| {
            let p = s as f64 / nf;
            -p * p.ln()
        })
        .sum();

    let denom = ((h1 + h2) / 2.0).max(1e-15);
    (mi / denom).clamp(0.0, 1.0)
}

/// Homogeneity score: each cluster contains only members of a single class.
///
/// Matches `sklearn.metrics.homogeneity_score`.
pub fn homogeneity_score(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    let n = labels_true.len();
    if n != labels_pred.len() || n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let k1 = labels_true.iter().cloned().max().unwrap_or(0) + 1;
    let k2 = labels_pred.iter().cloned().max().unwrap_or(0) + 1;

    let mut contingency = vec![vec![0usize; k2]; k1];
    for i in 0..n {
        contingency[labels_true[i]][labels_pred[i]] += 1;
    }

    let row_sums: Vec<usize> = contingency.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<usize> = (0..k2)
        .map(|j| contingency.iter().map(|r| r[j]).sum())
        .collect();

    // H(C|K) = conditional entropy of true labels given predicted
    let mut hck = 0.0;
    for j in 0..k2 {
        if col_sums[j] == 0 {
            continue;
        }
        for row in contingency.iter().take(k1) {
            if row[j] > 0 {
                let p = row[j] as f64 / nf;
                let pj = col_sums[j] as f64 / nf;
                hck -= p * (p / pj).ln();
            }
        }
    }

    // H(C) = entropy of true labels
    let hc: f64 = row_sums
        .iter()
        .filter(|&&s| s > 0)
        .map(|&s| {
            let p = s as f64 / nf;
            -p * p.ln()
        })
        .sum();

    if hc < 1e-15 {
        return 1.0;
    }
    1.0 - hck / hc
}

/// Completeness score: all members of a class are assigned to the same cluster.
///
/// Matches `sklearn.metrics.completeness_score`.
pub fn completeness_score(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    // Completeness = homogeneity with args swapped
    homogeneity_score(labels_pred, labels_true)
}

/// V-measure: harmonic mean of homogeneity and completeness.
///
/// Matches `sklearn.metrics.v_measure_score`.
pub fn v_measure_score(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    let h = homogeneity_score(labels_true, labels_pred);
    let c = completeness_score(labels_true, labels_pred);
    if h + c < 1e-15 {
        return 0.0;
    }
    2.0 * h * c / (h + c)
}

/// Fowlkes-Mallows index: geometric mean of precision and recall.
///
/// Matches `sklearn.metrics.fowlkes_mallows_score`.
pub fn fowlkes_mallows_score(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    let n = labels_true.len();
    if n < 2 {
        return 0.0;
    }

    // Count pairs
    let mut tp = 0u64; // true positive pairs (same class, same cluster)
    let mut fp = 0u64; // false positive (diff class, same cluster)
    let mut fn_ = 0u64; // false negative (same class, diff cluster)

    for i in 0..n {
        for j in i + 1..n {
            let same_true = labels_true[i] == labels_true[j];
            let same_pred = labels_pred[i] == labels_pred[j];
            match (same_true, same_pred) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_ += 1,
                _ => {}
            }
        }
    }

    if tp == 0 {
        return 0.0;
    }

    let precision = tp as f64 / (tp + fp) as f64;
    let recall = tp as f64 / (tp + fn_) as f64;
    (precision * recall).sqrt()
}

/// Elbow method helper: compute within-cluster sum of squares for k=1..max_k.
///
/// Returns a vector of inertia values useful for the elbow method.
pub fn elbow_inertias(data: &[Vec<f64>], max_k: usize, seed: u64) -> Vec<f64> {
    (1..=max_k.min(data.len()))
        .map(|k| {
            kmeans(data, k, 50, seed.wrapping_add(k as u64))
                .map(|r| r.inertia)
                .unwrap_or(f64::INFINITY)
        })
        .collect()
}

/// Mean-shift clustering: find cluster centers via kernel density gradient ascent.
///
/// Matches `sklearn.cluster.MeanShift`.
pub fn mean_shift(
    data: &[Vec<f64>],
    bandwidth: f64,
    max_iter: usize,
) -> Result<(Vec<Vec<f64>>, Vec<usize>), ClusterError> {
    let n = data.len();
    if n == 0 {
        return Err(ClusterError::EmptyData);
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "mean_shift input must be finite".to_string(),
        ));
    }
    let d = data[0].len();
    let bw2 = bandwidth * bandwidth;

    // Start each point as its own center candidate
    let mut centers: Vec<Vec<f64>> = data.to_vec();

    for _ in 0..max_iter {
        let mut shifted = false;
        for center in &mut centers {
            let mut new_center = vec![0.0; d];
            let mut total_weight = 0.0;

            for point in data {
                let dist2: f64 = center
                    .iter()
                    .zip(point.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                let weight = (-dist2 / (2.0 * bw2)).exp();
                total_weight += weight;
                for (j, &pj) in point.iter().enumerate() {
                    new_center[j] += weight * pj;
                }
            }

            if total_weight > 0.0 {
                for v in &mut new_center {
                    *v /= total_weight;
                }
            }

            let shift: f64 = center
                .iter()
                .zip(new_center.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if shift > 1e-6 {
                shifted = true;
            }
            *center = new_center;
        }

        if !shifted {
            break;
        }
    }

    // Merge nearby centers
    let merge_threshold = bandwidth / 2.0;
    let mut unique_centers: Vec<Vec<f64>> = Vec::new();
    let mut center_map = vec![0usize; n];

    for (i, center) in centers.iter().enumerate() {
        let mut found = false;
        for (j, uc) in unique_centers.iter().enumerate() {
            let dist: f64 = center
                .iter()
                .zip(uc.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < merge_threshold {
                center_map[i] = j;
                found = true;
                break;
            }
        }
        if !found {
            center_map[i] = unique_centers.len();
            unique_centers.push(center.clone());
        }
    }

    // Assign each original point to nearest unique center
    let mut labels = vec![0usize; n];
    for (i, point) in data.iter().enumerate() {
        let mut min_dist = f64::INFINITY;
        let mut best = 0;
        for (j, uc) in unique_centers.iter().enumerate() {
            let dist: f64 = point
                .iter()
                .zip(uc.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum();
            if dist < min_dist {
                min_dist = dist;
                best = j;
            }
        }
        labels[i] = best;
    }

    Ok((unique_centers, labels))
}

/// Agglomerative clustering with precomputed distance matrix.
///
/// Matches `scipy.cluster.hierarchy.linkage` with precomputed distances.
pub fn linkage_from_distances(
    condensed_dist: &[f64],
    n: usize,
    method: LinkageMethod,
) -> Result<Vec<[f64; 4]>, ClusterError> {
    if n < 2 {
        return Err(ClusterError::InvalidArgument(
            "need at least 2 observations".to_string(),
        ));
    }

    // Build full distance matrix from condensed form
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    let mut idx = 0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        dist[i][i] = 0.0;
        for j in i + 1..n {
            if idx < condensed_dist.len() {
                dist[i][j] = condensed_dist[idx];
                dist[j][i] = condensed_dist[idx];
                idx += 1;
            }
        }
    }

    // Convert to "data" format for linkage
    // Use the distance matrix directly in the agglomerative algorithm
    let total = 2 * n - 1;
    let mut active = vec![true; total];
    let mut cluster_size = vec![1usize; total];
    let mut inter_dist = vec![vec![f64::INFINITY; total]; total];

    for i in 0..n {
        for j in 0..n {
            inter_dist[i][j] = dist[i][j];
        }
    }

    let mut result = Vec::with_capacity(n - 1);

    for step in 0..n - 1 {
        let new_id = n + step;

        let mut min_d = f64::INFINITY;
        let mut mi = 0;
        let mut mj = 0;
        for i in 0..new_id {
            if !active[i] {
                continue;
            }
            for j in i + 1..new_id {
                if active[j] && inter_dist[i][j] < min_d {
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

/// Maximal cliques in a proximity graph (for small graphs).
///
/// Given data and an epsilon threshold, find all maximal cliques.
pub fn proximity_cliques(data: &[Vec<f64>], eps: f64) -> Vec<Vec<usize>> {
    let n = data.len();
    let eps2 = eps * eps;

    // Build adjacency
    let mut adj = vec![vec![]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d: f64 = data[i]
                .iter()
                .zip(data[j].iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum();
            if d <= eps2 {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    // Bron-Kerbosch for maximal cliques
    let mut cliques = Vec::new();
    bron_kerbosch(
        &adj,
        &mut vec![],
        &mut (0..n).collect(),
        &mut vec![],
        &mut cliques,
    );
    cliques
}

fn bron_kerbosch(
    adj: &[Vec<usize>],
    r: &mut Vec<usize>,
    p: &mut Vec<usize>,
    x: &mut Vec<usize>,
    cliques: &mut Vec<Vec<usize>>,
) {
    if p.is_empty() && x.is_empty() {
        if !r.is_empty() {
            cliques.push(r.clone());
        }
        return;
    }

    let p_copy = p.clone();
    for &v in &p_copy {
        r.push(v);
        let neighbors: std::collections::HashSet<usize> = adj[v].iter().cloned().collect();
        let mut new_p: Vec<usize> = p
            .iter()
            .filter(|&&u| neighbors.contains(&u))
            .cloned()
            .collect();
        let mut new_x: Vec<usize> = x
            .iter()
            .filter(|&&u| neighbors.contains(&u))
            .cloned()
            .collect();
        bron_kerbosch(adj, r, &mut new_p, &mut new_x, cliques);
        r.pop();
        p.retain(|&u| u != v);
        x.push(v);
    }
}

/// Compute silhouette coefficients for each sample.
///
/// Matches `sklearn.metrics.silhouette_samples`.
pub fn silhouette_samples(data: &[Vec<f64>], labels: &[usize]) -> Vec<f64> {
    let n = data.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let k = labels.iter().cloned().max().unwrap_or(0) + 1;

    (0..n)
        .map(|i| {
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

            if a.max(b) > 0.0 {
                (b - a) / a.max(b)
            } else {
                0.0
            }
        })
        .collect()
}

/// Gap statistic: compare within-cluster dispersion to reference.
///
/// Returns gap values for k=1..max_k.
pub fn gap_statistic(data: &[Vec<f64>], max_k: usize, n_ref: usize, seed: u64) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    let d = data[0].len();

    // Find data bounds
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for p in data {
        for (j, &v) in p.iter().enumerate() {
            mins[j] = mins[j].min(v);
            maxs[j] = maxs[j].max(v);
        }
    }

    let mut gaps = Vec::with_capacity(max_k);

    for k in 1..=max_k.min(n) {
        // Within-cluster dispersion for real data
        let real_result = kmeans(data, k, 50, seed);
        let log_wk = real_result
            .map(|r| r.inertia.max(1e-30).ln())
            .unwrap_or(0.0);

        // Average over reference datasets
        let mut ref_log_wks = Vec::with_capacity(n_ref);
        for r in 0..n_ref {
            let ref_seed = seed.wrapping_add(1000 * r as u64 + k as u64);
            let mut rng = ref_seed;

            // Generate uniform reference data
            let ref_data: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    (0..d)
                        .map(|j| {
                            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let u = (rng >> 11) as f64 / (1u64 << 53) as f64;
                            mins[j] + u * (maxs[j] - mins[j])
                        })
                        .collect()
                })
                .collect();

            let ref_result = kmeans(&ref_data, k, 30, ref_seed);
            let ref_wk = ref_result.map(|r| r.inertia.max(1e-30).ln()).unwrap_or(0.0);
            ref_log_wks.push(ref_wk);
        }

        let mean_ref = ref_log_wks.iter().sum::<f64>() / n_ref as f64;
        gaps.push(mean_ref - log_wk);
    }

    gaps
}

/// K-medoids (PAM) clustering.
///
/// Similar to K-means but uses actual data points as centers.
pub fn kmedoids(
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

    // Initialize medoids randomly
    let mut rng = seed;
    let mut medoid_indices: Vec<usize> = Vec::with_capacity(k);
    let mut used = vec![false; n];
    for _ in 0..k {
        loop {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng >> 33) as usize % n;
            if !used[idx] {
                used[idx] = true;
                medoid_indices.push(idx);
                break;
            }
        }
    }

    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign to nearest medoid
        for i in 0..n {
            let mut min_dist = f64::INFINITY;
            for (c, &med) in medoid_indices.iter().enumerate() {
                let d = sq_dist(&data[i], &data[med]);
                if d < min_dist {
                    min_dist = d;
                    labels[i] = c;
                }
            }
        }

        // Update medoids: for each cluster, find the point that minimizes total distance
        let mut changed = false;
        for (c, medoid_index) in medoid_indices.iter_mut().enumerate().take(k) {
            let members: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
            if members.is_empty() {
                continue;
            }

            let mut best_med = *medoid_index;
            let mut best_cost: f64 = members
                .iter()
                .map(|&i| sq_dist(&data[i], &data[best_med]).sqrt())
                .sum();

            for &candidate in &members {
                let cost: f64 = members
                    .iter()
                    .map(|&i| sq_dist(&data[i], &data[candidate]).sqrt())
                    .sum();
                if cost < best_cost {
                    best_cost = cost;
                    best_med = candidate;
                }
            }

            if best_med != *medoid_index {
                *medoid_index = best_med;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Compute final inertia
    let inertia: f64 = (0..n)
        .map(|i| sq_dist(&data[i], &data[medoid_indices[labels[i]]]))
        .sum();

    let centroids: Vec<Vec<f64>> = medoid_indices.iter().map(|&i| data[i].clone()).collect();

    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        n_iter: max_iter,
    })
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
    fn linkage_from_distances_skips_inactive_clusters_after_merge() {
        let condensed = vec![10.0, 4.0, 9.0, 8.0, 8.0, 1.0];
        let z = linkage_from_distances(&condensed, 4, LinkageMethod::Complete)
            .expect("complete linkage from condensed distances");
        assert_eq!(z.len(), 3);

        assert_eq!(z[0][0], 2.0);
        assert_eq!(z[0][1], 3.0);
        assert!((z[0][2] - 1.0).abs() < 1e-12);

        assert_eq!(z[1][0], 1.0);
        assert_eq!(z[1][1], 4.0);
        assert!((z[1][2] - 8.0).abs() < 1e-12);

        assert_eq!(z[2][0], 0.0);
        assert_eq!(z[2][1], 5.0);
        assert!((z[2][2] - 10.0).abs() < 1e-12);
    }

    /// Regression: verify that `linkage` never picks a merged (inactive) cluster
    /// as a merge candidate. With 5 collinear points at [0, 1, 3, 6, 10] using
    /// single linkage, each step must merge the closest *active* pair. If inactive
    /// clusters leaked into the nearest-pair search, merge distances would be wrong.
    ///
    /// Expected SciPy-equivalent merge sequence (single linkage):
    ///   step 0: (0,1) dist=1, size=2          -> cluster 5
    ///   step 1: (2,5) dist=2, size=3          -> cluster 6  (cluster 5 active, not 0 or 1)
    ///   step 2: (3,6) dist=3, size=4          -> cluster 7
    ///   step 3: (4,7) dist=4, size=5          -> cluster 8
    #[test]
    fn linkage_single_inactive_cluster_regression() {
        let data = vec![vec![0.0], vec![1.0], vec![3.0], vec![6.0], vec![10.0]];
        let z = linkage(&data, LinkageMethod::Single).unwrap();
        assert_eq!(z.len(), 4);

        // Step 0: merge points 0 and 1 (distance 1)
        assert_eq!(z[0][0], 0.0);
        assert_eq!(z[0][1], 1.0);
        assert!((z[0][2] - 1.0).abs() < 1e-12, "step 0 dist = {}", z[0][2]);
        assert_eq!(z[0][3], 2.0);

        // Step 1: merge point 2 with new cluster 5={0,1} (distance 2)
        assert_eq!(z[1][0], 2.0);
        assert_eq!(z[1][1], 5.0);
        assert!((z[1][2] - 2.0).abs() < 1e-12, "step 1 dist = {}", z[1][2]);
        assert_eq!(z[1][3], 3.0);

        // Step 2: merge point 3 with cluster 6={0,1,2} (distance 3)
        assert_eq!(z[2][0], 3.0);
        assert_eq!(z[2][1], 6.0);
        assert!((z[2][2] - 3.0).abs() < 1e-12, "step 2 dist = {}", z[2][2]);
        assert_eq!(z[2][3], 4.0);

        // Step 3: merge point 4 with cluster 7={0,1,2,3} (distance 4)
        assert_eq!(z[3][0], 4.0);
        assert_eq!(z[3][1], 7.0);
        assert!((z[3][2] - 4.0).abs() < 1e-12, "step 3 dist = {}", z[3][2]);
        assert_eq!(z[3][3], 5.0);
    }

    /// Regression: complete linkage with 5 points where merged-cluster distances
    /// differ significantly from original point distances. If inactive clusters
    /// pollute the search, the wrong pair gets picked.
    ///
    /// Points: [0, 1, 10, 11, 50]
    /// Complete linkage uses max distance, so merged cluster distances grow.
    #[test]
    fn linkage_complete_inactive_cluster_regression() {
        let data = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0], vec![50.0]];
        let z = linkage(&data, LinkageMethod::Complete).unwrap();
        assert_eq!(z.len(), 4);

        // Step 0: closest pair is (0,1) dist=1 or (2,3) dist=1; either valid
        assert!((z[0][2] - 1.0).abs() < 1e-12, "step 0 dist = {}", z[0][2]);
        assert_eq!(z[0][3], 2.0);

        // Step 1: next closest active pair, also dist=1
        assert!((z[1][2] - 1.0).abs() < 1e-12, "step 1 dist = {}", z[1][2]);
        assert_eq!(z[1][3], 2.0);

        // Step 2: merge the two size-2 clusters; complete linkage dist = max(11-0)=11
        assert!((z[2][2] - 11.0).abs() < 1e-12, "step 2 dist = {}", z[2][2]);
        assert_eq!(z[2][3], 4.0);

        // Step 3: merge point 4 with the size-4 cluster; complete dist = max(50-0)=50
        assert!((z[3][2] - 50.0).abs() < 1e-12, "step 3 dist = {}", z[3][2]);
        assert_eq!(z[3][3], 5.0);
    }

    /// Regression: `linkage_from_distances` with average method; ensures
    /// that after each merge step, only active clusters participate in
    /// distance updates and nearest-pair selection.
    #[test]
    fn linkage_from_distances_average_inactive_regression() {
        // 4 points with condensed distances: d(0,1)=2, d(0,2)=6, d(0,3)=10,
        //                                    d(1,2)=4, d(1,3)=8, d(2,3)=4
        let condensed = vec![2.0, 6.0, 10.0, 4.0, 8.0, 4.0];
        let z = linkage_from_distances(&condensed, 4, LinkageMethod::Average).unwrap();
        assert_eq!(z.len(), 3);

        // Step 0: closest pair is (0,1) dist=2
        assert_eq!(z[0][0], 0.0);
        assert_eq!(z[0][1], 1.0);
        assert!((z[0][2] - 2.0).abs() < 1e-12);
        assert_eq!(z[0][3], 2.0);

        // Step 1: cluster 4={0,1}. Average distances:
        //   d(4, 2) = (1*6 + 1*4)/2 = 5.0
        //   d(4, 3) = (1*10 + 1*8)/2 = 9.0
        //   d(2, 3) = 4.0  <- minimum among active pairs
        assert_eq!(z[1][0], 2.0);
        assert_eq!(z[1][1], 3.0);
        assert!((z[1][2] - 4.0).abs() < 1e-12);
        assert_eq!(z[1][3], 2.0);

        // Step 2: merge cluster 4={0,1} with cluster 5={2,3}
        //   d(4, 5) = average of (d(0,2), d(0,3), d(1,2), d(1,3)) via Lance-Williams:
        //   = (2*5.0 + 2*9.0) / (2+2) = 7.0
        // But Lance-Williams UPGMA: d(4,5) = (n4*d(4,2_merged) + n4_already?
        // Actually: d(new_cluster_4, k) was computed as average above.
        // d(4, 5={2,3}) = average(d(4,2), d(4,3)) weighted = (2*5 + 2*9)/4 = 7.0
        assert_eq!(z[2][0], 4.0);
        assert_eq!(z[2][1], 5.0);
        assert!((z[2][2] - 7.0).abs() < 1e-12, "step 2 dist = {}", z[2][2]);
        assert_eq!(z[2][3], 4.0);
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
