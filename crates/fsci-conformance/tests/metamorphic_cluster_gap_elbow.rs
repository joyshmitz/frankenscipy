#![forbid(unsafe_code)]
//! Metamorphic invariants for two stochastic, uncovered
//! `fsci_cluster` selectors-of-k:
//!
//!   • `gap_statistic(data, max_k, n_ref, seed)` — Tibshirani's
//!     gap = E*[log(W_k_ref)] − log(W_k). fsci uses LCG-seeded
//!     uniform reference data; sklearn-style PCG64 references
//!     can't agree at any seed.
//!   • `elbow_inertias(data, max_k, seed)` — k-means inertia at
//!     each k ∈ [1, max_k]. Sklearn idiom, no scipy equivalent.
//!
//! Resolves [frankenscipy-ygjot]. Verifies invariants any
//! sensible impl must satisfy:
//!
//! gap_statistic:
//!   1. len(out) == min(max_k, n).
//!   2. All entries finite.
//!   3. Determinism under fixed seed.
//!
//! elbow_inertias:
//!   4. len(out) == min(max_k, n).
//!   5. Inertia at k=1 equals total within-cluster sum-of-
//!      squared-distances from the global mean.
//!   6. Monotone non-increasing in k (refining k cannot
//!      increase inertia at the optimum; k-means may not find
//!      it but reasonable seed should).
//!   7. inertia ≥ 0 throughout.
//!   8. Determinism under fixed seed.
//!   9. Inertia at k=n is exactly 0 (each point is its own
//!      cluster).
//!
//! 4 fixtures × variable invariants ≈ 30 cases.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::{elbow_inertias, gap_statistic};
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
        .expect("create gap_elbow metamorphic output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &MetamorphicLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize gap_elbow metamorphic log");
    fs::write(path, json).expect("write gap_elbow metamorphic log");
}

fn make_blob(centers: &[(f64, f64)], per: usize) -> Vec<Vec<f64>> {
    let mut points = Vec::new();
    for (cx, cy) in centers {
        for k in 0..per {
            let t = (k as f64) - (per as f64 - 1.0) * 0.5;
            points.push(vec![cx + t * 0.05, cy + t * 0.07]);
        }
    }
    points
}

fn k1_inertia(data: &[Vec<f64>]) -> f64 {
    let n = data.len();
    let d = data.first().map(|p| p.len()).unwrap_or(0);
    if n == 0 {
        return 0.0;
    }
    let mut mean = vec![0.0; d];
    for p in data {
        for (i, x) in p.iter().enumerate() {
            mean[i] += x;
        }
    }
    for v in &mut mean {
        *v /= n as f64;
    }
    data.iter()
        .map(|p| p.iter().zip(mean.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>())
        .sum()
}

#[test]
fn metamorphic_cluster_gap_elbow() {
    let start = Instant::now();
    let mut cases = Vec::new();

    let blobs2 = make_blob(&[(0.0, 0.0), (10.0, 10.0)], 8);
    let blobs3 = make_blob(&[(0.0, 0.0), (10.0, 0.0), (5.0, 8.66)], 7);
    let blobs4 = make_blob(&[(0.0, 0.0), (12.0, 0.0), (0.0, 12.0), (12.0, 12.0)], 6);
    let single: Vec<Vec<f64>> = (0..15)
        .map(|i| vec![(i as f64) * 0.05, (i as f64) * 0.04])
        .collect();

    let fixtures: Vec<(&str, Vec<Vec<f64>>, usize)> = vec![
        ("blobs2_n16", blobs2, 5),
        ("blobs3_n21", blobs3, 5),
        ("blobs4_n24", blobs4, 6),
        ("single_n15", single, 4),
    ];
    let n_ref = 10_usize;
    let seed = 42_u64;

    for (name, data, max_k) in &fixtures {
        let n = data.len();

        // ───────────────── gap_statistic ─────────────────
        let g = gap_statistic(data, *max_k, n_ref, seed);

        // 1. shape
        let expected = (*max_k).min(n);
        cases.push(CaseLog {
            case_id: format!("gap_{name}"),
            invariant: "gap_len_eq_min_maxk_n".into(),
            detail: format!("len={}, expected={expected}", g.len()),
            pass: g.len() == expected,
        });

        // 2. finite
        let all_finite = g.iter().all(|v| v.is_finite());
        cases.push(CaseLog {
            case_id: format!("gap_{name}"),
            invariant: "gap_all_finite".into(),
            detail: format!("all_finite={all_finite}"),
            pass: all_finite,
        });

        // 3. determinism
        let g2 = gap_statistic(data, *max_k, n_ref, seed);
        cases.push(CaseLog {
            case_id: format!("gap_{name}"),
            invariant: "gap_deterministic_same_seed".into(),
            detail: format!("equal={}", g == g2),
            pass: g == g2,
        });

        // ───────────────── elbow_inertias ─────────────────
        let e = elbow_inertias(data, *max_k, seed);

        // 4. shape
        cases.push(CaseLog {
            case_id: format!("elbow_{name}"),
            invariant: "elbow_len_eq_min_maxk_n".into(),
            detail: format!("len={}, expected={expected}", e.len()),
            pass: e.len() == expected,
        });

        // 5. inertia at k=1 equals SST.
        if !e.is_empty() {
            let sst = k1_inertia(data);
            let abs_diff = (e[0] - sst).abs();
            // k=1 should be within numerical precision of SST.
            cases.push(CaseLog {
                case_id: format!("elbow_{name}"),
                invariant: "elbow_k1_eq_total_sumsq".into(),
                detail: format!("e[0]={}, sst={}, abs_diff={}", e[0], sst, abs_diff),
                pass: abs_diff <= 1.0e-9 * sst.max(1.0),
            });
        }

        // 6. monotone non-increasing in k.
        let monotone = e.windows(2).all(|w| w[1] <= w[0] + 1.0e-9 * w[0].max(1.0));
        cases.push(CaseLog {
            case_id: format!("elbow_{name}"),
            invariant: "elbow_monotone_nonincreasing".into(),
            detail: format!("seq={:?}", e),
            pass: monotone,
        });

        // 7. nonneg.
        let nonneg = e.iter().all(|v| v.is_finite() && *v >= 0.0);
        cases.push(CaseLog {
            case_id: format!("elbow_{name}"),
            invariant: "elbow_nonneg".into(),
            detail: format!("min={}", e.iter().cloned().fold(f64::INFINITY, f64::min)),
            pass: nonneg,
        });

        // 8. determinism
        let e2 = elbow_inertias(data, *max_k, seed);
        cases.push(CaseLog {
            case_id: format!("elbow_{name}"),
            invariant: "elbow_deterministic_same_seed".into(),
            detail: format!("equal={}", e == e2),
            pass: e == e2,
        });

        // 9. inertia at k=n is exactly 0 (only when max_k ≥ n).
        if *max_k >= n && !e.is_empty() && e.len() == n {
            let last = *e.last().unwrap();
            cases.push(CaseLog {
                case_id: format!("elbow_{name}"),
                invariant: "elbow_k_eq_n_inertia_zero".into(),
                detail: format!("last={last}"),
                pass: last.abs() < 1.0e-9,
            });
        }
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = MetamorphicLog {
        test_id: "metamorphic_cluster_gap_elbow".into(),
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
                "gap_elbow metamorphic fail: {} {} — {}",
                c.case_id, c.invariant, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "gap_elbow metamorphic failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
