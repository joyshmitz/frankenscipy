//! Metamorphic tests for `fsci-cluster`.
//!
//! Each test asserts an invariant of clustering that holds regardless
//! of the specific input — e.g. labels lie in [0, k), `vq` reproduces
//! `kmeans` labels for the same centroids, linkage matrix monotonicity.
//!
//! Run with: `cargo test -p fsci-cluster --test metamorphic_tests`

use fsci_cluster::{
    LinkageMethod, fcluster, is_valid_linkage, kmeans, linkage, silhouette_score, vq,
};

fn small_dataset() -> Vec<Vec<f64>> {
    // Three clusters in 2D, well separated.
    vec![
        // Cluster 1
        vec![0.0, 0.0],
        vec![0.5, 0.2],
        vec![0.1, -0.3],
        vec![-0.2, 0.4],
        // Cluster 2
        vec![5.0, 5.0],
        vec![5.4, 5.1],
        vec![4.8, 4.7],
        vec![5.2, 4.9],
        // Cluster 3
        vec![10.0, 0.0],
        vec![9.7, -0.2],
        vec![10.3, 0.1],
        vec![9.8, 0.4],
    ]
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — kmeans labels are in [0, k); inertia is non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kmeans_label_range_and_inertia() {
    let data = small_dataset();
    for &k in &[1, 2, 3, 5] {
        if k > data.len() {
            continue;
        }
        let result = kmeans(&data, k, 200, 42).unwrap();
        assert_eq!(result.labels.len(), data.len(), "labels length");
        for &lab in &result.labels {
            assert!(lab < k, "MR1 label {lab} out of range for k={k}");
        }
        assert!(
            result.inertia.is_finite() && result.inertia >= 0.0,
            "MR1 inertia non-finite or negative: {} (k={k})",
            result.inertia
        );
        assert_eq!(result.centroids.len(), k, "centroid count");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — vq(data, kmeans.centroids) returns the same labels as kmeans.
//
// This is a structural sanity: the vq operator is the assignment step
// of Lloyd's algorithm, so a converged kmeans must satisfy
// vq(centroids) == kmeans.labels at convergence (up to ties on
// equidistant points).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_vq_matches_kmeans_assignment() {
    let data = small_dataset();
    let k = 3;
    let result = kmeans(&data, k, 500, 7).unwrap();
    let (vq_labels, dists) = vq(&data, &result.centroids).unwrap();
    assert_eq!(vq_labels.len(), result.labels.len());
    for (i, (kl, vl)) in result.labels.iter().zip(&vq_labels).enumerate() {
        assert_eq!(
            *kl, *vl,
            "MR2 label disagreement at i={i}: kmeans={kl} vq={vl}"
        );
        assert!(dists[i].is_finite() && dists[i] >= 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — linkage produces a valid linkage matrix (per is_valid_linkage)
//        and the merge distances are non-decreasing for single, complete,
//        and ward methods (these are guaranteed monotone).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_linkage_validity_and_monotonicity() {
    let data = small_dataset();
    for &method in &[
        LinkageMethod::Single,
        LinkageMethod::Complete,
        LinkageMethod::Average,
        LinkageMethod::Ward,
    ] {
        let z = linkage(&data, method).unwrap();
        assert!(is_valid_linkage(&z), "MR3 invalid linkage for {method:?}");
        if matches!(
            method,
            LinkageMethod::Single | LinkageMethod::Complete | LinkageMethod::Ward
        ) {
            for i in 1..z.len() {
                let d_prev = z[i - 1][2];
                let d_curr = z[i][2];
                assert!(
                    d_curr >= d_prev - 1e-12,
                    "MR3 {method:?} not monotone at i={i}: {d_prev} → {d_curr}"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — fcluster respects max_clusters: the number of unique labels
//        in the output is at most max_clusters.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fcluster_respects_max_clusters() {
    let data = small_dataset();
    let z = linkage(&data, LinkageMethod::Ward).unwrap();
    for &max_k in &[1usize, 2, 3, 5, 10] {
        let labels = fcluster(&z, max_k).unwrap();
        assert_eq!(labels.len(), data.len());
        let mut sorted_unique: Vec<usize> = labels.iter().copied().collect();
        sorted_unique.sort_unstable();
        sorted_unique.dedup();
        assert!(
            sorted_unique.len() <= max_k,
            "MR4 fcluster gave {} clusters but max_clusters={max_k}",
            sorted_unique.len()
        );
        // Labels must be 1-based integers (matches scipy convention).
        assert!(*sorted_unique.first().unwrap() >= 1, "labels must be ≥ 1");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — silhouette_score is in [-1, 1] for any clustering with k ≥ 2.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_silhouette_in_unit_signed_interval() {
    let data = small_dataset();
    for &k in &[2, 3] {
        let res = kmeans(&data, k, 200, 42).unwrap();
        let s = silhouette_score(&data, &res.labels).unwrap();
        assert!(
            (-1.0..=1.0).contains(&s),
            "MR5 silhouette score out of [-1, 1]: {s} (k={k})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — kmeans is deterministic across calls with the same seed.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kmeans_deterministic_seeded() {
    let data = small_dataset();
    let r1 = kmeans(&data, 3, 200, 12345).unwrap();
    let r2 = kmeans(&data, 3, 200, 12345).unwrap();
    assert_eq!(r1.labels, r2.labels, "MR6 kmeans labels differ across runs");
    assert_eq!(
        r1.centroids.len(),
        r2.centroids.len(),
        "centroid count differs"
    );
    for (c1, c2) in r1.centroids.iter().zip(&r2.centroids) {
        for (a, b) in c1.iter().zip(c2) {
            assert!((a - b).abs() < 1e-12, "centroid drift: {a} vs {b}");
        }
    }
    assert!((r1.inertia - r2.inertia).abs() < 1e-12, "inertia drift");
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — kmeans inertia computation matches Σ‖x_i − c_{label_i}‖².
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kmeans_inertia_matches_definition() {
    let data = small_dataset();
    let res = kmeans(&data, 3, 500, 7).unwrap();
    let mut expected = 0.0;
    for (i, point) in data.iter().enumerate() {
        let centroid = &res.centroids[res.labels[i]];
        let d2: f64 = point
            .iter()
            .zip(centroid)
            .map(|(p, c)| (p - c).powi(2))
            .sum();
        expected += d2;
    }
    let rel_err = ((res.inertia - expected) / expected.abs().max(1e-12)).abs();
    assert!(
        rel_err < 1e-9,
        "MR7 inertia mismatch: reported={} expected={} rel_err={rel_err:e}",
        res.inertia,
        expected
    );
}
