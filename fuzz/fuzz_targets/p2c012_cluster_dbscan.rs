#![no_main]

use arbitrary::Arbitrary;
use fsci_cluster::dbscan;
use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;

// DBSCAN clustering oracle:
// Tests dbscan(data, eps, min_samples) for correctness properties:
//
// 1. Label count matches input point count
// 2. Labels are -1 (noise) or >= 0 (cluster id)
// 3. Core points have at least min_samples neighbors within eps
// 4. Each cluster has at least one core point
// 5. All points in a cluster are reachable from a core point

const MAX_POINTS: usize = 64;
const MAX_DIMS: usize = 8;

#[derive(Debug, Arbitrary)]
struct DbscanInput {
    points: Vec<Vec<f64>>,
    eps_raw: f64,
    min_samples_raw: u8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let delta = ai - bi;
            delta * delta
        })
        .sum()
}

fuzz_target!(|input: DbscanInput| {
    let n = input.points.len().min(MAX_POINTS);
    if n < 2 {
        return;
    }

    let dim = input.points.first().map_or(0, |p| p.len().min(MAX_DIMS));
    if dim == 0 {
        return;
    }

    let data: Vec<Vec<f64>> = input
        .points
        .iter()
        .take(n)
        .map(|p| {
            p.iter()
                .take(dim)
                .map(|&v| sanitize(v))
                .chain(std::iter::repeat(0.0))
                .take(dim)
                .collect()
        })
        .collect();

    let eps = sanitize(input.eps_raw).abs().clamp(1e-10, 1e6);
    let eps2 = eps * eps;
    let min_samples = (input.min_samples_raw as usize) % (n + 2);

    let result = match dbscan(&data, eps, min_samples) {
        Ok(r) => r,
        Err(_) => return,
    };

    let labels = &result.labels;

    if labels.len() != n {
        panic!(
            "DBSCAN label count mismatch: got {} expected {} (n={}, eps={}, min_samples={})",
            labels.len(),
            n,
            n,
            eps,
            min_samples
        );
    }

    let max_cluster = labels.iter().copied().max().unwrap_or(-1);

    for (i, &label) in labels.iter().enumerate() {
        if label < -1 {
            panic!(
                "DBSCAN invalid label at index {}: {} (should be >= -1)",
                i, label
            );
        }
        if label > max_cluster {
            panic!(
                "DBSCAN label {} at index {} exceeds max cluster {}",
                label, i, max_cluster
            );
        }
    }

    if result.n_clusters as i64 != max_cluster + 1 && max_cluster >= 0 {
        panic!(
            "DBSCAN n_clusters mismatch: reported {} but max label is {} (expected {})",
            result.n_clusters,
            max_cluster,
            max_cluster + 1
        );
    }

    let core_indices: HashSet<usize> = result.core_sample_indices.iter().copied().collect();
    if core_indices.len() != result.core_sample_indices.len() {
        panic!(
            "DBSCAN core_sample_indices contains duplicates: {:?}",
            result.core_sample_indices
        );
    }

    for &core_idx in &result.core_sample_indices {
        if core_idx >= n {
            panic!(
                "DBSCAN core_sample_indices contains invalid index {} (n={})",
                core_idx, n
            );
        }
        if labels[core_idx] == -1 {
            panic!(
                "DBSCAN core point at index {} is labeled as noise",
                core_idx
            );
        }
        let neighbor_count = data
            .iter()
            .filter(|point| squared_distance(&data[core_idx], point) <= eps2)
            .count();
        if neighbor_count < min_samples {
            panic!(
                "DBSCAN core point {} has {} eps-neighbors but min_samples={}",
                core_idx, neighbor_count, min_samples
            );
        }
    }

    for cluster_id in 0..=max_cluster {
        let members: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l == cluster_id)
            .map(|(i, _)| i)
            .collect();

        if members.is_empty() {
            continue;
        }

        let cluster_core_indices: Vec<usize> = members
            .iter()
            .copied()
            .filter(|idx| core_indices.contains(idx))
            .collect();

        if cluster_core_indices.is_empty() {
            panic!(
                "DBSCAN cluster {} with {} members has no core point",
                cluster_id,
                members.len()
            );
        }

        for &idx in &members {
            if core_indices.contains(&idx) {
                continue;
            }
            let touches_core = cluster_core_indices
                .iter()
                .any(|&core_idx| squared_distance(&data[idx], &data[core_idx]) <= eps2);
            if !touches_core {
                panic!(
                    "DBSCAN cluster {} member {} is not within eps of any core point",
                    cluster_id, idx
                );
            }
        }
    }

    if max_cluster >= 0 {
        let cluster_ids: HashSet<i64> = labels.iter().copied().filter(|&l| l >= 0).collect();
        for expected in 0..=max_cluster {
            if !cluster_ids.contains(&expected) {
                panic!(
                    "DBSCAN cluster IDs not contiguous: missing cluster {} (max={})",
                    expected, max_cluster
                );
            }
        }
    }
});
