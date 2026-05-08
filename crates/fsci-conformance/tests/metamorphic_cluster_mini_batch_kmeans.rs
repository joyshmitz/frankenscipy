#![forbid(unsafe_code)]
//! Metamorphic invariants for `fsci_cluster::mini_batch_kmeans`.
//!
//! Resolves [frankenscipy-f0hcv]. fsci uses an LCG-seeded
//! sampler for mini-batches; sklearn's `MiniBatchKMeans` uses
//! PCG64 (and lives in sklearn, not scipy — sklearn isn't even
//! installed in this environment). Direct centroid parity is
//! impossible, so this harness verifies invariants any sensible
//! mini-batch k-means must satisfy:
//!
//!   1. Output: centroids shape (k, d).
//!   2. labels.len() == n.
//!   3. Each label ∈ [0, k).
//!   4. Determinism under fixed seed.
//!   5. Each point's label is the nearest centroid by Euclidean
//!      distance (assignment consistency).
//!   6. Inertia ≥ 0.
//!   7. Well-separated cluster recovery: when fixtures contain
//!      visually obvious clusters, points within a true cluster
//!      should share the same label, and points across true
//!      clusters should not.
//!
//! 4 fixtures × variable invariants ≈ 24 cases.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::mini_batch_kmeans;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-012";

#[derive(Debug, Clone, Serialize)]
struct CaseLog {
    case_id: String,
    invariant: String,
    detail: String,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct MetamorphicLog {
    test_id: String,
    case_count: usize,
    pass_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseLog>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("fixtures/artifacts/{PACKET_ID}/metamorphic"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir())
        .expect("create mini_batch_kmeans metamorphic output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &MetamorphicLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log)
        .expect("serialize mini_batch_kmeans metamorphic log");
    fs::write(path, json).expect("write mini_batch_kmeans metamorphic log");
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn nearest_center(point: &[f64], centers: &[Vec<f64>]) -> usize {
    let mut best = 0;
    let mut best_d = f64::INFINITY;
    for (i, c) in centers.iter().enumerate() {
        let d = sq_dist(point, c);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

#[test]
fn metamorphic_cluster_mini_batch_kmeans() {
    let start = Instant::now();
    let mut cases = Vec::new();

    // (label, data, k, true_groups). true_groups[i] = which canonical
    // cluster point i belongs to (for cluster-recovery invariant).
    let make_blob = |centers: &[(f64, f64)], per: usize| -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut points = Vec::new();
        let mut groups = Vec::new();
        for (g, (cx, cy)) in centers.iter().enumerate() {
            for k in 0..per {
                let off_x = ((k as f64) - (per as f64 - 1.0) * 0.5) * 0.05;
                let off_y = ((k as f64) - (per as f64 - 1.0) * 0.5) * 0.07;
                points.push(vec![cx + off_x, cy + off_y]);
                groups.push(g);
            }
        }
        (points, groups)
    };

    let (b2, g2) = make_blob(&[(0.0, 0.0), (10.0, 10.0)], 8);
    let (b3, g3) = make_blob(&[(0.0, 0.0), (10.0, 0.0), (5.0, 8.66)], 7);
    let (b4, g4) = make_blob(&[(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)], 6);
    let single_cluster: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![(i as f64) * 0.05, (i as f64) * 0.03])
        .collect();
    let g_single: Vec<usize> = vec![0; 20];

    let fixtures: Vec<(&str, Vec<Vec<f64>>, usize, Vec<usize>)> = vec![
        ("blobs_k2", b2, 2, g2),
        ("blobs_k3", b3, 3, g3),
        ("blobs_k4", b4, 4, g4),
        ("single_cluster_k1", single_cluster, 1, g_single),
    ];

    let max_iter = 100_usize;
    let batch_size = 8_usize;
    let seed = 42_u64;

    for (name, data, k, groups) in &fixtures {
        let n = data.len();
        let d = data.first().map(|p| p.len()).unwrap_or(0);
        let r = match mini_batch_kmeans(data, *k, max_iter, batch_size, seed) {
            Ok(v) => v,
            Err(e) => {
                cases.push(CaseLog {
                    case_id: name.to_string(),
                    invariant: "function_returns_ok".into(),
                    detail: format!("error: {e:?}"),
                    pass: false,
                });
                continue;
            }
        };

        // 1. centroid shape.
        let centroid_shape_ok =
            r.centroids.len() == *k && r.centroids.iter().all(|c| c.len() == d);
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "centroids_shape_k_by_d".into(),
            detail: format!(
                "centroids.len()={}, expected_k={k}, d={d}",
                r.centroids.len()
            ),
            pass: centroid_shape_ok,
        });

        // 2. labels.len() == n.
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "labels_len_eq_n".into(),
            detail: format!("labels.len()={}, n={n}", r.labels.len()),
            pass: r.labels.len() == n,
        });

        // 3. labels ∈ [0, k).
        let labels_in_range = r.labels.iter().all(|&l| l < *k);
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "labels_in_0_k".into(),
            detail: format!("max_label={}, k={k}", r.labels.iter().copied().max().unwrap_or(0)),
            pass: labels_in_range,
        });

        // 4. Determinism.
        let r2 = mini_batch_kmeans(data, *k, max_iter, batch_size, seed)
            .expect("MBK rerun");
        let determ = r.centroids == r2.centroids && r.labels == r2.labels;
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "deterministic_same_seed".into(),
            detail: format!("equal={determ}"),
            pass: determ,
        });

        // 5. Each label is the nearest center.
        let assignment_ok = data.iter().enumerate().all(|(i, p)| {
            r.labels.get(i).copied() == Some(nearest_center(p, &r.centroids))
        });
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "labels_match_nearest_center".into(),
            detail: format!("ok={assignment_ok}"),
            pass: assignment_ok,
        });

        // 6. Inertia ≥ 0.
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "inertia_nonneg".into(),
            detail: format!("inertia={}", r.inertia),
            pass: r.inertia.is_finite() && r.inertia >= 0.0,
        });

        // 7. Well-separated cluster recovery — within-group same
        //    label; cross-group different label. Skip for k=1.
        if *k > 1 {
            let mut within_ok = true;
            let mut cross_ok = true;
            for i in 0..n {
                for j in (i + 1)..n {
                    let same_group = groups[i] == groups[j];
                    let same_label = r.labels[i] == r.labels[j];
                    if same_group && !same_label {
                        within_ok = false;
                    }
                    if !same_group && same_label {
                        cross_ok = false;
                    }
                }
            }
            cases.push(CaseLog {
                case_id: name.to_string(),
                invariant: "well_separated_recovery".into(),
                detail: format!("within_same_label={within_ok}, cross_diff_label={cross_ok}"),
                pass: within_ok && cross_ok,
            });
        }
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = MetamorphicLog {
        test_id: "metamorphic_cluster_mini_batch_kmeans".into(),
        case_count: cases.len(),
        pass_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: cases.clone(),
    };

    emit_log(&log);

    for c in &cases {
        if !c.pass {
            eprintln!(
                "mini_batch_kmeans metamorphic fail: {} {} — {}",
                c.case_id, c.invariant, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "mini_batch_kmeans metamorphic failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
