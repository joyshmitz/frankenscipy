#![forbid(unsafe_code)]
//! Metamorphic invariants for `fsci_cluster::mean_shift`.
//!
//! Resolves [frankenscipy-x2950]. fsci's mean_shift is
//! deterministic (no seed) but its merge tolerance and
//! convergence criteria differ from sklearn.cluster.MeanShift
//! (sklearn isn't installed in this env). This harness
//! verifies invariants any sensible mean-shift implementation
//! must satisfy:
//!
//!   1. Returns (centers, labels) with labels.len() == n.
//!   2. Each label indexes a valid center (∈ [0, centers.len()).
//!   3. centers.len() ≥ 1 for non-empty data.
//!   4. Determinism: identical input → identical output.
//!   5. Well-separated cluster recovery: within true-group same
//!      label, cross true-group different label.
//!   6. Centers and points have the same dimensionality.
//!
//! 4 fixtures × variable invariants ≈ 24 cases.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::mean_shift;
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
    fs::create_dir_all(output_dir()).expect("create mean_shift metamorphic output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &MetamorphicLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json =
        serde_json::to_string_pretty(log).expect("serialize mean_shift metamorphic log");
    fs::write(path, json).expect("write mean_shift metamorphic log");
}

fn make_blob(centers: &[(f64, f64)], per: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut points = Vec::new();
    let mut groups = Vec::new();
    for (g, (cx, cy)) in centers.iter().enumerate() {
        for k in 0..per {
            let t = (k as f64) - (per as f64 - 1.0) * 0.5;
            points.push(vec![cx + t * 0.05, cy + t * 0.07]);
            groups.push(g);
        }
    }
    (points, groups)
}

#[test]
fn metamorphic_cluster_mean_shift() {
    let start = Instant::now();
    let mut cases = Vec::new();

    let (b1, g1) = make_blob(&[(0.0, 0.0)], 12);
    let (b2, g2) = make_blob(&[(0.0, 0.0), (10.0, 10.0)], 8);
    let (b3, g3) = make_blob(&[(0.0, 0.0), (10.0, 0.0), (5.0, 8.66)], 7);
    let (b4, g4) =
        make_blob(&[(0.0, 0.0), (12.0, 0.0), (0.0, 12.0), (12.0, 12.0)], 6);

    // Bandwidth chosen larger than within-cluster spread (~0.3-0.5 max
    // axis-aligned offset given per ≤ 12) but much smaller than the
    // 10-unit between-cluster distance.
    let fixtures: Vec<(&str, Vec<Vec<f64>>, f64, Vec<usize>)> = vec![
        ("single_cluster_n12", b1, 1.5, g1),
        ("two_clusters_n16", b2, 2.0, g2),
        ("three_clusters_n21", b3, 2.0, g3),
        ("four_clusters_n24", b4, 2.0, g4),
    ];
    let max_iter = 200_usize;

    for (name, data, bw, groups) in &fixtures {
        let n = data.len();
        let d = data.first().map(|p| p.len()).unwrap_or(0);
        let (centers, labels) = match mean_shift(data, *bw, max_iter) {
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

        // 1. labels.len() == n.
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "labels_len_eq_n".into(),
            detail: format!("labels.len()={}, n={n}", labels.len()),
            pass: labels.len() == n,
        });

        // 2. each label indexes a valid center.
        let labels_in_range = labels.iter().all(|&l| l < centers.len());
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "labels_in_centers_range".into(),
            detail: format!(
                "centers={}, max_label={}",
                centers.len(),
                labels.iter().copied().max().unwrap_or(0)
            ),
            pass: labels_in_range,
        });

        // 3. at least one center for non-empty data.
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "centers_nonempty".into(),
            detail: format!("centers.len()={}", centers.len()),
            pass: !centers.is_empty(),
        });

        // 4. dimensionality matches input.
        let dim_ok = centers.iter().all(|c| c.len() == d);
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "centers_dim_eq_input".into(),
            detail: format!(
                "input_d={d}, all_centers_d={}",
                centers.iter().all(|c| c.len() == d)
            ),
            pass: dim_ok,
        });

        // 5. determinism.
        let (centers2, labels2) = mean_shift(data, *bw, max_iter).expect("MS rerun");
        let determ = centers == centers2 && labels == labels2;
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "deterministic_same_input".into(),
            detail: format!("equal={determ}"),
            pass: determ,
        });

        // 6. well-separated cluster recovery.
        let unique_groups: std::collections::HashSet<usize> = groups.iter().copied().collect();
        if unique_groups.len() > 1 {
            let mut within_ok = true;
            let mut cross_ok = true;
            for i in 0..n {
                for j in (i + 1)..n {
                    let same_group = groups[i] == groups[j];
                    let same_label = labels[i] == labels[j];
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
                detail: format!(
                    "within_same_label={within_ok}, cross_diff_label={cross_ok}"
                ),
                pass: within_ok && cross_ok,
            });
        }
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = MetamorphicLog {
        test_id: "metamorphic_cluster_mean_shift".into(),
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
                "mean_shift metamorphic fail: {} {} — {}",
                c.case_id, c.invariant, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "mean_shift metamorphic failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
