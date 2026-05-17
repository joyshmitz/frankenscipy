#![forbid(unsafe_code)]
//! Property-based coverage for fsci_cluster::kmeans.
//!
//! Resolves [frankenscipy-0c79z]. K-means is stochastic in its
//! initialization (k-means++), so direct numerical parity is brittle.
//! The meaningful invariants:
//!   * Number of returned centroids == k
//!   * On well-separated 3-cluster data, each ground-truth cluster
//!     gets a unique label
//!   * Inertia equals sum over all points of ||x - centroid[label]||²
//!     (definition; must hold exactly modulo FP rounding)
//!   * Each centroid is the mean of its assigned points (definition
//!     after Lloyd's algorithm converges)
//!   * Error paths: empty data, non-finite input, k=0, k>n

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::kmeans;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create kmeans diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[test]
fn diff_cluster_kmeans_properties() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // Three well-separated 2D clusters around (0,0), (10,10), (-5, 8)
    let data: Vec<Vec<f64>> = vec![
        // cluster A around (0, 0)
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![-0.1, 0.05],
        vec![0.0, -0.05],
        // cluster B around (10, 10)
        vec![10.0, 10.0],
        vec![10.1, 10.0],
        vec![10.0, 10.1],
        vec![9.9, 9.95],
        // cluster C around (-5, 8)
        vec![-5.0, 8.0],
        vec![-5.05, 8.05],
        vec![-4.95, 7.95],
        vec![-5.0, 8.1],
    ];
    let k = 3;
    let r = kmeans(&data, k, 100, 42).expect("kmeans");

    // === 1. n_centroids == k ===
    check(
        "n_centroids_eq_k",
        r.centroids.len() == k,
        format!("got {}", r.centroids.len()),
    );

    // === 2. Each ground-truth cluster gets a unique label ===
    let labels_a: HashSet<usize> = r.labels[..4].iter().copied().collect();
    let labels_b: HashSet<usize> = r.labels[4..8].iter().copied().collect();
    let labels_c: HashSet<usize> = r.labels[8..].iter().copied().collect();
    check(
        "cluster_a_single_label",
        labels_a.len() == 1,
        format!("labels={labels_a:?}"),
    );
    check(
        "cluster_b_single_label",
        labels_b.len() == 1,
        format!("labels={labels_b:?}"),
    );
    check(
        "cluster_c_single_label",
        labels_c.len() == 1,
        format!("labels={labels_c:?}"),
    );
    let label_a = *labels_a.iter().next().unwrap();
    let label_b = *labels_b.iter().next().unwrap();
    let label_c = *labels_c.iter().next().unwrap();
    check(
        "three_distinct_labels",
        label_a != label_b && label_b != label_c && label_a != label_c,
        format!("a={label_a} b={label_b} c={label_c}"),
    );

    // === 3. Inertia equals sum of squared distances to assigned centroid ===
    let computed_inertia: f64 = data
        .iter()
        .zip(r.labels.iter())
        .map(|(point, &lbl)| sq_dist(point, &r.centroids[lbl]))
        .sum();
    check(
        "inertia_matches_definition",
        (r.inertia - computed_inertia).abs() <= 1.0e-9 * r.inertia.abs().max(1.0),
        format!("reported={} computed={}", r.inertia, computed_inertia),
    );

    // === 4. Centroids are means of assigned points ===
    {
        let d = data[0].len();
        for c in 0..k {
            let assigned: Vec<&Vec<f64>> = data
                .iter()
                .zip(r.labels.iter())
                .filter(|&(_, &lbl)| lbl == c)
                .map(|(p, _)| p)
                .collect();
            assert!(!assigned.is_empty(), "centroid {c} has no assigned points");
            let mut mean = vec![0.0_f64; d];
            for p in &assigned {
                for (i, v) in p.iter().enumerate() {
                    mean[i] += v;
                }
            }
            for v in &mut mean {
                *v /= assigned.len() as f64;
            }
            let dist = sq_dist(&mean, &r.centroids[c]).sqrt();
            check(
                &format!("centroid_{c}_is_mean"),
                dist < 1.0e-9,
                format!("dist={dist}"),
            );
        }
    }

    // === 5. Reasonable centroids: should be near the ground-truth centers ===
    {
        let truth = vec![vec![0.0, 0.0], vec![10.0, 10.0], vec![-5.0, 8.0]];
        let mut all_found = true;
        for t in &truth {
            let min_dist = r
                .centroids
                .iter()
                .map(|c| sq_dist(c, t).sqrt())
                .fold(f64::INFINITY, f64::min);
            if min_dist > 0.5 {
                all_found = false;
                break;
            }
        }
        check(
            "centroids_near_truth_centers",
            all_found,
            format!("centroids={:?}", r.centroids),
        );
    }

    // === Error paths ===
    {
        let r = kmeans(&[], 2, 100, 1);
        check(
            "empty_data_errors",
            r.is_err(),
            String::new(),
        );
    }
    {
        let bad: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![f64::NAN, 1.0]];
        let r = kmeans(&bad, 1, 100, 1);
        check(
            "non_finite_data_errors",
            r.is_err(),
            String::new(),
        );
    }
    {
        let r = kmeans(&data, 0, 100, 1);
        check(
            "k_zero_errors",
            r.is_err(),
            String::new(),
        );
    }
    {
        let r = kmeans(&data, data.len() + 1, 100, 1);
        check(
            "k_too_large_errors",
            r.is_err(),
            String::new(),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_cluster_kmeans_properties".into(),
        category: "fsci_cluster::kmeans property-based coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("kmeans mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "kmeans property coverage failed: {} cases",
        diffs.len()
    );
}
