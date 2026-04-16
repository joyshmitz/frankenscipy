#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-010 (Spatial).
//!
//! Implements conformance tests for scipy.spatial parity:
//!   Happy-path (1-5): distance functions, pdist, cdist, squareform, KDTree
//!   Edge cases (6-8): empty data, dimension mismatch, NaN handling
//!   Cross-op consistency (9-11): metric relationships, KDTree vs brute-force
//!   Performance boundary (12-14): large datasets
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-010/e2e/`.

use fsci_conformance::PacketFamily;
use fsci_spatial::{
    DistanceMetric, KDTree, cdist, cdist_metric, chebyshev, cityblock, cosine, distance_matrix,
    euclidean, minkowski, pdist, sqeuclidean, squareform_to_condensed, squareform_to_matrix,
};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// ───────────────────────── Forensic log types ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    artifacts: Vec<ArtifactRef>,
    environment: EnvironmentInfo,
    spatial_metadata: Option<SpatialMetadata>,
    overall: OverallResult,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicStep {
    step_id: usize,
    step_name: String,
    action: String,
    input_summary: String,
    output_summary: String,
    duration_ns: u128,
    mode: String,
    outcome: String,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactRef {
    path: String,
    blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
    total_memory_mb: String,
}

#[derive(Debug, Clone, Serialize)]
struct SpatialMetadata {
    operation: String,
    n_points: usize,
    n_dims: usize,
    metric: String,
}

#[derive(Debug, Clone, Serialize)]
struct OverallResult {
    status: String,
    total_duration_ns: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

// ───────────────────────── Helpers ─────────────────────────

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts")
        .join(PacketFamily::Spatial.packet_id())
        .join("e2e")
}

fn make_env() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: String::from(env!("CARGO_PKG_VERSION")),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        total_memory_mb: String::from("unknown"),
    }
}

fn replay_cmd(scenario_id: &str) -> String {
    format!(
        "rch exec -- cargo test -p fsci-conformance --test e2e_spatial -- {scenario_id} --nocapture"
    )
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir).ok();
    let path = dir.join(format!("{scenario_id}.json"));
    if let Ok(json) = serde_json::to_string_pretty(bundle) {
        fs::write(path, json).ok();
    }
}

// ───────────────────────── Scenario Runner ─────────────────────────

struct ScenarioRunner {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    start: Instant,
    step_counter: usize,
    spatial_metadata: Option<SpatialMetadata>,
    status: String,
    error_chain: Option<String>,
}

impl ScenarioRunner {
    fn new(scenario_id: &str) -> Self {
        Self {
            scenario_id: scenario_id.to_string(),
            steps: Vec::new(),
            start: Instant::now(),
            step_counter: 0,
            spatial_metadata: None,
            status: "pass".to_string(),
            error_chain: None,
        }
    }

    fn set_spatial_meta(&mut self, operation: &str, n_points: usize, n_dims: usize, metric: &str) {
        self.spatial_metadata = Some(SpatialMetadata {
            operation: operation.to_string(),
            n_points,
            n_dims,
            metric: metric.to_string(),
        });
    }

    fn step<F>(&mut self, name: &str, action: &str, input: &str, mode: &str, f: F)
    where
        F: FnOnce() -> Result<String, String>,
    {
        self.step_counter += 1;
        let step_start = Instant::now();
        let result = f();
        let duration_ns = step_start.elapsed().as_nanos();

        let (outcome, output_summary) = match result {
            Ok(out) => ("pass".to_string(), out),
            Err(e) => {
                self.status = "fail".to_string();
                if self.error_chain.is_none() {
                    self.error_chain = Some(e.clone());
                }
                ("fail".to_string(), e)
            }
        };

        self.steps.push(ForensicStep {
            step_id: self.step_counter,
            step_name: name.to_string(),
            action: action.to_string(),
            input_summary: input.to_string(),
            output_summary,
            duration_ns,
            mode: mode.to_string(),
            outcome,
        });
    }

    fn finish(self) -> ForensicLogBundle {
        ForensicLogBundle {
            scenario_id: self.scenario_id.clone(),
            steps: self.steps,
            artifacts: vec![],
            environment: make_env(),
            spatial_metadata: self.spatial_metadata,
            overall: OverallResult {
                status: self.status,
                total_duration_ns: self.start.elapsed().as_nanos(),
                replay_command: replay_cmd(&self.scenario_id),
                error_chain: self.error_chain,
            },
        }
    }
}

// ───────────────────────── Test Data Generators ─────────────────────────

fn generate_points(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    // Simple deterministic PRNG for reproducibility
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as f64 / (1u64 << 31) as f64 * 10.0 - 5.0
    };
    (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol || (a.is_nan() && b.is_nan())
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIOS 1-5: HAPPY-PATH
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 1: Distance function validation
/// Tests all basic distance metrics against known values
#[test]
fn scenario_01_distance_functions() {
    let mut runner = ScenarioRunner::new("scenario_01_distance_functions");
    runner.set_spatial_meta("distance", 2, 3, "multiple");

    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    // Euclidean: sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
    runner.step(
        "euclidean",
        "euclidean(a, b)",
        "[1,2,3] vs [4,5,6]",
        "Strict",
        || {
            let d = euclidean(&a, &b);
            let expected = 27.0_f64.sqrt();
            if approx_eq(d, expected, 1e-10) {
                Ok(format!("euclidean={d:.6}"))
            } else {
                Err(format!("expected {expected}, got {d}"))
            }
        },
    );

    // Squared euclidean: 27.0
    runner.step(
        "sqeuclidean",
        "sqeuclidean(a, b)",
        "[1,2,3] vs [4,5,6]",
        "Strict",
        || {
            let d = sqeuclidean(&a, &b);
            if approx_eq(d, 27.0, 1e-10) {
                Ok(format!("sqeuclidean={d:.6}"))
            } else {
                Err(format!("expected 27.0, got {d}"))
            }
        },
    );

    // Manhattan: |4-1| + |5-2| + |6-3| = 9.0
    runner.step(
        "cityblock",
        "cityblock(a, b)",
        "[1,2,3] vs [4,5,6]",
        "Strict",
        || {
            let d = cityblock(&a, &b);
            if approx_eq(d, 9.0, 1e-10) {
                Ok(format!("cityblock={d:.6}"))
            } else {
                Err(format!("expected 9.0, got {d}"))
            }
        },
    );

    // Chebyshev: max(|3|, |3|, |3|) = 3.0
    runner.step(
        "chebyshev",
        "chebyshev(a, b)",
        "[1,2,3] vs [4,5,6]",
        "Strict",
        || {
            let d = chebyshev(&a, &b);
            if approx_eq(d, 3.0, 1e-10) {
                Ok(format!("chebyshev={d:.6}"))
            } else {
                Err(format!("expected 3.0, got {d}"))
            }
        },
    );

    // Minkowski p=3
    runner.step(
        "minkowski_p3",
        "minkowski(a, b, 3.0)",
        "[1,2,3] vs [4,5,6]",
        "Strict",
        || {
            let d = minkowski(&a, &b, 3.0);
            let expected = (3.0_f64.powi(3) + 3.0_f64.powi(3) + 3.0_f64.powi(3)).powf(1.0 / 3.0);
            if approx_eq(d, expected, 1e-10) {
                Ok(format!("minkowski_p3={d:.6}"))
            } else {
                Err(format!("expected {expected}, got {d}"))
            }
        },
    );

    // Cosine distance
    runner.step(
        "cosine",
        "cosine(a, b)",
        "[1,2,3] vs [4,5,6]",
        "Strict",
        || {
            let d = cosine(&a, &b);
            // cos_sim = (1*4 + 2*5 + 3*6) / (sqrt(14) * sqrt(77)) = 32 / sqrt(1078)
            let dot = 4.0 + 10.0 + 18.0;
            let expected = 1.0 - dot / (14.0_f64.sqrt() * 77.0_f64.sqrt());
            if approx_eq(d, expected, 1e-10) {
                Ok(format!("cosine={d:.6}"))
            } else {
                Err(format!("expected {expected}, got {d}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_01_distance_functions", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_01 failed");
}

/// Scenario 2: pdist - pairwise distances in condensed form
#[test]
fn scenario_02_pdist() {
    let mut runner = ScenarioRunner::new("scenario_02_pdist");
    runner.set_spatial_meta("pdist", 4, 2, "euclidean");

    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];

    runner.step(
        "pdist_euclidean",
        "pdist(points, Euclidean)",
        "4 points in 2D",
        "Strict",
        || {
            let condensed =
                pdist(&points, DistanceMetric::Euclidean).map_err(|e| format!("{e}"))?;
            // For 4 points: n*(n-1)/2 = 6 distances
            if condensed.len() != 6 {
                return Err(format!("expected 6 distances, got {}", condensed.len()));
            }
            // d(0,1) = 1.0, d(0,2) = 1.0, d(0,3) = sqrt(2), d(1,2) = sqrt(2), d(1,3) = 1.0, d(2,3) = 1.0
            let expected = [1.0, 1.0, 2.0_f64.sqrt(), 2.0_f64.sqrt(), 1.0, 1.0];
            for (i, (&c, &e)) in condensed.iter().zip(expected.iter()).enumerate() {
                if !approx_eq(c, e, 1e-10) {
                    return Err(format!("index {i}: expected {e}, got {c}"));
                }
            }
            Ok(format!("condensed_len={}", condensed.len()))
        },
    );

    runner.step(
        "pdist_cityblock",
        "pdist(points, Cityblock)",
        "4 points in 2D",
        "Strict",
        || {
            let condensed =
                pdist(&points, DistanceMetric::Cityblock).map_err(|e| format!("{e}"))?;
            // d(0,1) = 1.0, d(0,2) = 1.0, d(0,3) = 2.0, d(1,2) = 2.0, d(1,3) = 1.0, d(2,3) = 1.0
            let expected = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0];
            for (i, (&c, &e)) in condensed.iter().zip(expected.iter()).enumerate() {
                if !approx_eq(c, e, 1e-10) {
                    return Err(format!("index {i}: expected {e}, got {c}"));
                }
            }
            Ok(format!("cityblock distances verified"))
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_02_pdist", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_02 failed");
}

/// Scenario 3: squareform conversions
#[test]
fn scenario_03_squareform() {
    let mut runner = ScenarioRunner::new("scenario_03_squareform");
    runner.set_spatial_meta("squareform", 4, 2, "N/A");

    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];

    runner.step(
        "condensed_to_matrix",
        "squareform_to_matrix(pdist(...))",
        "4 points",
        "Strict",
        || {
            let condensed =
                pdist(&points, DistanceMetric::Euclidean).map_err(|e| format!("{e}"))?;
            let matrix = squareform_to_matrix(&condensed).map_err(|e| format!("{e}"))?;

            // Check dimensions
            if matrix.len() != 4 || matrix.iter().any(|row| row.len() != 4) {
                return Err("matrix not 4x4".to_string());
            }

            // Check diagonal is zero
            for (i, row) in matrix.iter().enumerate() {
                if row[i] != 0.0 {
                    return Err(format!("diagonal[{i}] not zero: {}", row[i]));
                }
            }

            // Check symmetry
            for i in 0..4 {
                for j in 0..4 {
                    if !approx_eq(matrix[i][j], matrix[j][i], 1e-10) {
                        return Err(format!("not symmetric at [{i}][{j}]"));
                    }
                }
            }

            Ok("4x4 symmetric matrix with zero diagonal".to_string())
        },
    );

    runner.step(
        "roundtrip",
        "condensed -> matrix -> condensed",
        "verify roundtrip",
        "Strict",
        || {
            let original = pdist(&points, DistanceMetric::Euclidean).map_err(|e| format!("{e}"))?;
            let matrix = squareform_to_matrix(&original).map_err(|e| format!("{e}"))?;
            let back = squareform_to_condensed(&matrix).map_err(|e| format!("{e}"))?;

            if original.len() != back.len() {
                return Err("length mismatch after roundtrip".to_string());
            }
            for (i, (&o, &b)) in original.iter().zip(back.iter()).enumerate() {
                if !approx_eq(o, b, 1e-10) {
                    return Err(format!("mismatch at index {i}"));
                }
            }
            Ok("roundtrip verified".to_string())
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_03_squareform", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_03 failed");
}

/// Scenario 4: cdist - cross-distance matrix
#[test]
fn scenario_04_cdist() {
    let mut runner = ScenarioRunner::new("scenario_04_cdist");
    runner.set_spatial_meta("cdist", 6, 2, "euclidean");

    let xa = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let xb = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![2.0, 2.0]];

    runner.step(
        "cdist_euclidean",
        "cdist(xa, xb)",
        "2 vs 3 points",
        "Strict",
        || {
            let dm = cdist(&xa, &xb).map_err(|e| format!("{e}"))?;

            if dm.len() != 2 || dm[0].len() != 3 {
                return Err(format!(
                    "expected 2x3 matrix, got {}x{}",
                    dm.len(),
                    dm[0].len()
                ));
            }

            // dm[0][0] = d([0,0], [1,0]) = 1.0
            // dm[0][1] = d([0,0], [0,1]) = 1.0
            // dm[0][2] = d([0,0], [2,2]) = sqrt(8) ≈ 2.828
            // dm[1][0] = d([1,1], [1,0]) = 1.0
            // dm[1][1] = d([1,1], [0,1]) = 1.0
            // dm[1][2] = d([1,1], [2,2]) = sqrt(2) ≈ 1.414
            let expected = [[1.0, 1.0, 8.0_f64.sqrt()], [1.0, 1.0, 2.0_f64.sqrt()]];

            for (i, row) in dm.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    if !approx_eq(val, expected[i][j], 1e-10) {
                        return Err(format!(
                            "dm[{i}][{j}]: expected {}, got {val}",
                            expected[i][j]
                        ));
                    }
                }
            }

            Ok(format!("2x3 distance matrix verified"))
        },
    );

    runner.step(
        "cdist_metric_cityblock",
        "cdist_metric(xa, xb, Cityblock)",
        "2 vs 3 points",
        "Strict",
        || {
            let dm =
                cdist_metric(&xa, &xb, DistanceMetric::Cityblock).map_err(|e| format!("{e}"))?;
            // dm[0][2] = |0-2| + |0-2| = 4
            // dm[1][2] = |1-2| + |1-2| = 2
            if !approx_eq(dm[0][2], 4.0, 1e-10) {
                return Err(format!("dm[0][2]: expected 4.0, got {}", dm[0][2]));
            }
            if !approx_eq(dm[1][2], 2.0, 1e-10) {
                return Err(format!("dm[1][2]: expected 2.0, got {}", dm[1][2]));
            }
            Ok("cityblock cdist verified".to_string())
        },
    );

    runner.step(
        "distance_matrix_alias",
        "distance_matrix(xa, xb)",
        "alias for cdist euclidean",
        "Strict",
        || {
            let dm1 = cdist(&xa, &xb).map_err(|e| format!("{e}"))?;
            let dm2 = distance_matrix(&xa, &xb).map_err(|e| format!("{e}"))?;
            for (row1, row2) in dm1.iter().zip(dm2.iter()) {
                for (&v1, &v2) in row1.iter().zip(row2.iter()) {
                    if !approx_eq(v1, v2, 1e-15) {
                        return Err("distance_matrix != cdist".to_string());
                    }
                }
            }
            Ok("distance_matrix == cdist(euclidean)".to_string())
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_04_cdist", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_04 failed");
}

/// Scenario 5: KDTree basic operations
#[test]
fn scenario_05_kdtree_basic() {
    let mut runner = ScenarioRunner::new("scenario_05_kdtree_basic");
    runner.set_spatial_meta("KDTree", 10, 2, "euclidean");

    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
        vec![2.0, 2.0],
        vec![3.0, 0.0],
        vec![0.0, 3.0],
        vec![2.0, 1.0],
        vec![1.0, 2.0],
    ];

    runner.step(
        "build_kdtree",
        "KDTree::new(points)",
        "10 points in 2D",
        "Strict",
        || {
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;
            if tree.size() != 10 {
                return Err(format!("expected size 10, got {}", tree.size()));
            }
            Ok(format!("tree.size()={}", tree.size()))
        },
    );

    runner.step(
        "query_1nn",
        "tree.query([0.1, 0.1])",
        "nearest to [0.1, 0.1]",
        "Strict",
        || {
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;
            let (idx, dist) = tree.query(&[0.1, 0.1]).map_err(|e| format!("{e}"))?;
            // Nearest should be [0,0] at index 0
            if idx != 0 {
                return Err(format!("expected index 0, got {idx}"));
            }
            let expected_dist = (0.1_f64.powi(2) + 0.1_f64.powi(2)).sqrt();
            if !approx_eq(dist, expected_dist, 1e-10) {
                return Err(format!("expected dist {expected_dist}, got {dist}"));
            }
            Ok(format!("idx={idx}, dist={dist:.6}"))
        },
    );

    runner.step(
        "query_knn",
        "tree.query_k([0.5, 0.5], 3)",
        "3 nearest to center",
        "Strict",
        || {
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;
            let results = tree.query_k(&[0.5, 0.5], 3).map_err(|e| format!("{e}"))?;
            if results.len() != 3 {
                return Err(format!("expected 3 results, got {}", results.len()));
            }
            // Nearest is [0.5, 0.5] itself at index 4 with distance 0
            if results[0].0 != 4 || results[0].1 != 0.0 {
                return Err(format!("first result not center: {:?}", results[0]));
            }
            // Verify sorted by distance
            for i in 1..results.len() {
                if results[i].1 < results[i - 1].1 {
                    return Err("results not sorted by distance".to_string());
                }
            }
            Ok(format!(
                "3-NN indices: {:?}",
                results.iter().map(|r| r.0).collect::<Vec<_>>()
            ))
        },
    );

    runner.step(
        "query_ball_point",
        "tree.query_ball_point([0.5, 0.5], 1.0)",
        "all within radius 1.0",
        "Strict",
        || {
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;
            let indices = tree
                .query_ball_point(&[0.5, 0.5], 1.0)
                .map_err(|e| format!("{e}"))?;
            // Points within distance 1.0 of [0.5, 0.5]:
            // [0,0]: d=0.707, [1,0]: d=0.707, [0,1]: d=0.707, [1,1]: d=0.707, [0.5,0.5]: d=0
            // So should include indices 0,1,2,3,4
            if indices.len() < 5 {
                return Err(format!("expected at least 5 points, got {}", indices.len()));
            }
            for expected in [0, 1, 2, 3, 4] {
                if !indices.contains(&expected) {
                    return Err(format!("missing index {expected}"));
                }
            }
            Ok(format!("found {} points in ball", indices.len()))
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_05_kdtree_basic", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_05 failed");
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIOS 6-8: EDGE CASES AND ERROR HANDLING
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 6: Empty data handling
#[test]
fn scenario_06_empty_data() {
    let mut runner = ScenarioRunner::new("scenario_06_empty_data");
    runner.set_spatial_meta("error_handling", 0, 0, "N/A");

    runner.step(
        "pdist_too_few",
        "pdist([single_point], Euclidean)",
        "1 point only",
        "Strict",
        || {
            let result = pdist(&[vec![1.0, 2.0]], DistanceMetric::Euclidean);
            if result.is_err() {
                Ok("correctly rejected single point".to_string())
            } else {
                Err("should have rejected single point".to_string())
            }
        },
    );

    runner.step(
        "kdtree_empty",
        "KDTree::new([])",
        "empty data",
        "Strict",
        || {
            let result = KDTree::new(&[]);
            if result.is_err() {
                Ok("correctly rejected empty data".to_string())
            } else {
                Err("should have rejected empty data".to_string())
            }
        },
    );

    runner.step(
        "cdist_empty_xa",
        "cdist([], xb)",
        "empty first set",
        "Strict",
        || {
            let empty: Vec<Vec<f64>> = vec![];
            let xb = vec![vec![1.0, 2.0]];
            let result = cdist(&empty, &xb);
            if result.is_err() {
                Ok("correctly rejected empty xa".to_string())
            } else {
                Err("should have rejected empty xa".to_string())
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_06_empty_data", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_06 failed");
}

/// Scenario 7: Dimension mismatch handling
#[test]
fn scenario_07_dimension_mismatch() {
    let mut runner = ScenarioRunner::new("scenario_07_dimension_mismatch");
    runner.set_spatial_meta("error_handling", 0, 0, "N/A");

    runner.step(
        "pdist_dim_mismatch",
        "pdist([[1,2], [1,2,3]], ...)",
        "inconsistent dimensions",
        "Strict",
        || {
            let points = vec![vec![1.0, 2.0], vec![1.0, 2.0, 3.0]];
            let result = pdist(&points, DistanceMetric::Euclidean);
            if result.is_err() {
                Ok("correctly rejected dimension mismatch".to_string())
            } else {
                Err("should have rejected dimension mismatch".to_string())
            }
        },
    );

    runner.step(
        "cdist_dim_mismatch",
        "cdist(2D, 3D)",
        "xa and xb different dims",
        "Strict",
        || {
            let xa = vec![vec![1.0, 2.0]];
            let xb = vec![vec![1.0, 2.0, 3.0]];
            let result = cdist(&xa, &xb);
            if result.is_err() {
                Ok("correctly rejected dimension mismatch".to_string())
            } else {
                Err("should have rejected dimension mismatch".to_string())
            }
        },
    );

    runner.step(
        "kdtree_query_dim_mismatch",
        "tree.query(wrong_dim)",
        "query with wrong dimension",
        "Strict",
        || {
            let points = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;
            let result = tree.query(&[1.0, 2.0, 3.0]);
            if result.is_err() {
                Ok("correctly rejected query dimension mismatch".to_string())
            } else {
                Err("should have rejected query dimension mismatch".to_string())
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_07_dimension_mismatch", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_07 failed");
}

/// Scenario 8: Non-finite value handling
#[test]
fn scenario_08_nonfinite_values() {
    let mut runner = ScenarioRunner::new("scenario_08_nonfinite_values");
    runner.set_spatial_meta("error_handling", 0, 0, "N/A");

    runner.step(
        "kdtree_reject_nan",
        "KDTree::new([[NaN, 1.0], ...])",
        "NaN in data",
        "Strict",
        || {
            let points = vec![vec![f64::NAN, 1.0], vec![2.0, 3.0]];
            let result = KDTree::new(&points);
            if result.is_err() {
                Ok("correctly rejected NaN in data".to_string())
            } else {
                Err("should have rejected NaN in data".to_string())
            }
        },
    );

    runner.step(
        "kdtree_reject_inf",
        "KDTree::new([[Inf, 1.0], ...])",
        "Inf in data",
        "Strict",
        || {
            let points = vec![vec![f64::INFINITY, 1.0], vec![2.0, 3.0]];
            let result = KDTree::new(&points);
            if result.is_err() {
                Ok("correctly rejected Inf in data".to_string())
            } else {
                Err("should have rejected Inf in data".to_string())
            }
        },
    );

    runner.step(
        "kdtree_query_reject_nan",
        "tree.query([NaN, 1.0])",
        "NaN in query",
        "Strict",
        || {
            let points = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;
            let result = tree.query(&[f64::NAN, 1.0]);
            if result.is_err() {
                Ok("correctly rejected NaN in query".to_string())
            } else {
                Err("should have rejected NaN in query".to_string())
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_08_nonfinite_values", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_08 failed");
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIOS 9-11: CROSS-OP CONSISTENCY
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 9: Metric relationships (triangle inequality, etc.)
#[test]
fn scenario_09_metric_properties() {
    let mut runner = ScenarioRunner::new("scenario_09_metric_properties");
    runner.set_spatial_meta("metric_properties", 3, 3, "multiple");

    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let c = vec![2.0, 3.0, 4.0];

    // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    runner.step(
        "triangle_inequality",
        "d(a,c) <= d(a,b) + d(b,c)",
        "for euclidean metric",
        "Strict",
        || {
            let dab = euclidean(&a, &b);
            let dbc = euclidean(&b, &c);
            let dac = euclidean(&a, &c);
            if dac <= dab + dbc + 1e-10 {
                Ok(format!("d(a,c)={dac:.4} <= d(a,b)+d(b,c)={:.4}", dab + dbc))
            } else {
                Err("triangle inequality violated".to_string())
            }
        },
    );

    // Symmetry: d(a,b) == d(b,a)
    runner.step(
        "symmetry",
        "d(a,b) == d(b,a)",
        "for all metrics",
        "Strict",
        || {
            for metric in [
                DistanceMetric::Euclidean,
                DistanceMetric::Cityblock,
                DistanceMetric::Chebyshev,
                DistanceMetric::Cosine,
            ] {
                let dab = fsci_spatial::metric_distance(&a, &b, metric);
                let dba = fsci_spatial::metric_distance(&b, &a, metric);
                if !approx_eq(dab, dba, 1e-10) {
                    return Err(format!("{metric:?} not symmetric"));
                }
            }
            Ok("all metrics symmetric".to_string())
        },
    );

    // Identity: d(a,a) == 0
    runner.step(
        "identity_of_indiscernibles",
        "d(a,a) == 0",
        "for all metrics",
        "Strict",
        || {
            for metric in [
                DistanceMetric::Euclidean,
                DistanceMetric::Cityblock,
                DistanceMetric::Chebyshev,
            ] {
                let daa = fsci_spatial::metric_distance(&a, &a, metric);
                if daa != 0.0 {
                    return Err(format!("{metric:?} d(a,a) != 0: {daa}"));
                }
            }
            Ok("d(a,a) = 0 for standard metrics".to_string())
        },
    );

    // Minkowski specializations
    runner.step(
        "minkowski_specializations",
        "minkowski(p=1) == cityblock, minkowski(p=2) == euclidean",
        "verify specializations",
        "Strict",
        || {
            let m1 = minkowski(&a, &b, 1.0);
            let cb = cityblock(&a, &b);
            if !approx_eq(m1, cb, 1e-10) {
                return Err(format!("minkowski(1) != cityblock: {m1} vs {cb}"));
            }

            let m2 = minkowski(&a, &b, 2.0);
            let eu = euclidean(&a, &b);
            if !approx_eq(m2, eu, 1e-10) {
                return Err(format!("minkowski(2) != euclidean: {m2} vs {eu}"));
            }

            let minf = minkowski(&a, &b, f64::INFINITY);
            let ch = chebyshev(&a, &b);
            if !approx_eq(minf, ch, 1e-10) {
                return Err(format!("minkowski(inf) != chebyshev: {minf} vs {ch}"));
            }

            Ok("minkowski specializations verified".to_string())
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_09_metric_properties", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_09 failed");
}

/// Scenario 10: KDTree vs brute-force consistency
#[test]
fn scenario_10_kdtree_vs_bruteforce() {
    let mut runner = ScenarioRunner::new("scenario_10_kdtree_vs_bruteforce");
    runner.set_spatial_meta("consistency", 50, 3, "euclidean");

    let points = generate_points(50, 3, 12345);
    let queries = generate_points(10, 3, 67890);

    runner.step(
        "1nn_consistency",
        "kdtree.query vs brute force",
        "10 queries on 50 points",
        "Strict",
        || {
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;

            for (qi, query) in queries.iter().enumerate() {
                let (tree_idx, tree_dist) = tree.query(query).map_err(|e| format!("{e}"))?;

                // Brute force
                let mut best_idx = 0;
                let mut best_dist = f64::INFINITY;
                for (i, pt) in points.iter().enumerate() {
                    let d = euclidean(pt, query);
                    if d < best_dist {
                        best_dist = d;
                        best_idx = i;
                    }
                }

                if tree_idx != best_idx {
                    // Distance should at least be equal (there could be ties)
                    if !approx_eq(tree_dist, best_dist, 1e-10) {
                        return Err(format!(
                            "query {qi}: tree found idx={tree_idx} dist={tree_dist}, brute found idx={best_idx} dist={best_dist}"
                        ));
                    }
                }
            }

            Ok("all 10 queries matched".to_string())
        },
    );

    runner.step(
        "knn_consistency",
        "kdtree.query_k vs brute force",
        "k=5 for each query",
        "Strict",
        || {
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;
            let k = 5;

            for (qi, query) in queries.iter().enumerate() {
                let tree_results = tree.query_k(query, k).map_err(|e| format!("{e}"))?;

                // Brute force k-NN
                let mut dists: Vec<(usize, f64)> = points
                    .iter()
                    .enumerate()
                    .map(|(i, pt)| (i, euclidean(pt, query)))
                    .collect();
                dists.sort_by(|a, b| a.1.total_cmp(&b.1));
                let brute_results: Vec<(usize, f64)> = dists.into_iter().take(k).collect();

                // Compare distances (indices might differ for ties)
                for (ti, (&tree_r, &brute_r)) in
                    tree_results.iter().zip(brute_results.iter()).enumerate()
                {
                    if !approx_eq(tree_r.1, brute_r.1, 1e-10) {
                        return Err(format!(
                            "query {qi} neighbor {ti}: tree dist={}, brute dist={}",
                            tree_r.1, brute_r.1
                        ));
                    }
                }
            }

            Ok("all k-NN queries matched".to_string())
        },
    );

    runner.step(
        "ball_query_consistency",
        "kdtree.query_ball_point vs brute force",
        "radius=3.0 for each query",
        "Strict",
        || {
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;
            let radius = 3.0;

            for (qi, query) in queries.iter().enumerate() {
                let tree_indices = tree
                    .query_ball_point(query, radius)
                    .map_err(|e| format!("{e}"))?;

                // Brute force
                let brute_indices: Vec<usize> = points
                    .iter()
                    .enumerate()
                    .filter(|(_, pt)| euclidean(pt, query) <= radius)
                    .map(|(i, _)| i)
                    .collect();

                let mut tree_sorted = tree_indices.clone();
                tree_sorted.sort();
                let mut brute_sorted = brute_indices.clone();
                brute_sorted.sort();

                if tree_sorted != brute_sorted {
                    return Err(format!(
                        "query {qi}: tree found {} points, brute found {}",
                        tree_indices.len(),
                        brute_indices.len()
                    ));
                }
            }

            Ok("all ball queries matched".to_string())
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_10_kdtree_vs_bruteforce", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_10 failed");
}

/// Scenario 11: pdist/cdist consistency
#[test]
fn scenario_11_pdist_cdist_consistency() {
    let mut runner = ScenarioRunner::new("scenario_11_pdist_cdist_consistency");
    runner.set_spatial_meta("consistency", 10, 3, "euclidean");

    let points = generate_points(10, 3, 54321);

    runner.step(
        "pdist_vs_cdist_same_set",
        "pdist(X) vs cdist(X, X)",
        "should give same distances",
        "Strict",
        || {
            let condensed =
                pdist(&points, DistanceMetric::Euclidean).map_err(|e| format!("{e}"))?;
            let matrix = cdist(&points, &points).map_err(|e| format!("{e}"))?;

            // Convert condensed to matrix and compare
            let n = points.len();
            let mut idx = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let cdist_val = matrix[i][j];
                    let pdist_val = condensed[idx];
                    if !approx_eq(cdist_val, pdist_val, 1e-10) {
                        return Err(format!(
                            "mismatch at [{i}][{j}]: pdist={pdist_val}, cdist={cdist_val}"
                        ));
                    }
                    idx += 1;
                }
            }

            Ok("pdist and cdist upper triangle match".to_string())
        },
    );

    runner.step(
        "squareform_vs_cdist",
        "squareform(pdist(X)) vs cdist(X, X)",
        "full matrix comparison",
        "Strict",
        || {
            let condensed =
                pdist(&points, DistanceMetric::Euclidean).map_err(|e| format!("{e}"))?;
            let from_pdist = squareform_to_matrix(&condensed).map_err(|e| format!("{e}"))?;
            let from_cdist = cdist(&points, &points).map_err(|e| format!("{e}"))?;

            for (i, (row_p, row_c)) in from_pdist.iter().zip(from_cdist.iter()).enumerate() {
                for (j, (&p, &c)) in row_p.iter().zip(row_c.iter()).enumerate() {
                    if !approx_eq(p, c, 1e-10) {
                        return Err(format!("mismatch at [{i}][{j}]: pdist={p}, cdist={c}"));
                    }
                }
            }

            Ok("squareform(pdist) == cdist".to_string())
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_11_pdist_cdist_consistency", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_11 failed");
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIOS 12-14: PERFORMANCE BOUNDARY
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 12: Large pdist computation
#[test]
fn scenario_12_large_pdist() {
    let mut runner = ScenarioRunner::new("scenario_12_large_pdist");
    let n = 500;
    runner.set_spatial_meta("pdist", n, 10, "euclidean");

    let points = generate_points(n, 10, 11111);

    runner.step(
        "pdist_500_points_10d",
        "pdist(500 points, 10D, euclidean)",
        &format!("{n} points, 10 dims"),
        "Strict",
        || {
            let start = Instant::now();
            let condensed =
                pdist(&points, DistanceMetric::Euclidean).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();

            let expected_len = n * (n - 1) / 2;
            if condensed.len() != expected_len {
                return Err(format!(
                    "expected {} distances, got {}",
                    expected_len,
                    condensed.len()
                ));
            }

            if elapsed.as_millis() > 5000 {
                return Err(format!("too slow: {:?}", elapsed));
            }

            Ok(format!(
                "time={:?}, {} distances computed",
                elapsed,
                condensed.len()
            ))
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_12_large_pdist", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_12 failed");
}

/// Scenario 13: Large KDTree operations
#[test]
fn scenario_13_large_kdtree() {
    let mut runner = ScenarioRunner::new("scenario_13_large_kdtree");
    let n = 10000;
    runner.set_spatial_meta("KDTree", n, 3, "euclidean");

    let points = generate_points(n, 3, 22222);
    let queries = generate_points(100, 3, 33333);

    runner.step(
        "build_10k_tree",
        "KDTree::new(10000 points, 3D)",
        "construction time",
        "Strict",
        || {
            let start = Instant::now();
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();

            if tree.size() != n {
                return Err(format!("expected size {n}, got {}", tree.size()));
            }

            if elapsed.as_millis() > 5000 {
                return Err(format!("construction too slow: {:?}", elapsed));
            }

            Ok(format!("build time={:?}", elapsed))
        },
    );

    runner.step(
        "100_queries_10k_tree",
        "100 nearest-neighbor queries",
        "query throughput",
        "Strict",
        || {
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;
            let start = Instant::now();

            for query in &queries {
                let _ = tree.query(query).map_err(|e| format!("{e}"))?;
            }

            let elapsed = start.elapsed();

            if elapsed.as_millis() > 5000 {
                return Err(format!("queries too slow: {:?}", elapsed));
            }

            Ok(format!(
                "100 queries in {:?}, {:.2} us/query",
                elapsed,
                elapsed.as_micros() as f64 / 100.0
            ))
        },
    );

    runner.step(
        "knn_k20_queries",
        "100 k=20 nearest neighbor queries",
        "kNN throughput",
        "Strict",
        || {
            let tree = KDTree::new(&points).map_err(|e| format!("{e}"))?;
            let start = Instant::now();

            for query in &queries {
                let results = tree.query_k(query, 20).map_err(|e| format!("{e}"))?;
                if results.len() != 20 {
                    return Err(format!("expected 20 neighbors, got {}", results.len()));
                }
            }

            let elapsed = start.elapsed();

            if elapsed.as_millis() > 5000 {
                return Err(format!("k-NN queries too slow: {:?}", elapsed));
            }

            Ok(format!(
                "100 20-NN queries in {:?}, {:.2} us/query",
                elapsed,
                elapsed.as_micros() as f64 / 100.0
            ))
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_13_large_kdtree", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_13 failed");
}

/// Scenario 14: Large cdist computation
#[test]
fn scenario_14_large_cdist() {
    let mut runner = ScenarioRunner::new("scenario_14_large_cdist");
    runner.set_spatial_meta("cdist", 1000, 5, "euclidean");

    let xa = generate_points(200, 5, 44444);
    let xb = generate_points(300, 5, 55555);

    runner.step(
        "cdist_200x300_5d",
        "cdist(200 points, 300 points, 5D)",
        "200x300 distance matrix",
        "Strict",
        || {
            let start = Instant::now();
            let dm = cdist(&xa, &xb).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();

            if dm.len() != 200 || dm[0].len() != 300 {
                return Err(format!(
                    "expected 200x300 matrix, got {}x{}",
                    dm.len(),
                    dm[0].len()
                ));
            }

            if elapsed.as_millis() > 5000 {
                return Err(format!("too slow: {:?}", elapsed));
            }

            Ok(format!("time={:?}, 60000 distances computed", elapsed))
        },
    );

    runner.step(
        "cdist_multiple_metrics",
        "cdist with different metrics",
        "compare metric performance",
        "Strict",
        || {
            let metrics = [
                DistanceMetric::Euclidean,
                DistanceMetric::Cityblock,
                DistanceMetric::Chebyshev,
                DistanceMetric::Cosine,
            ];

            let mut results = Vec::new();
            for metric in metrics {
                let start = Instant::now();
                let _ = cdist_metric(&xa, &xb, metric).map_err(|e| format!("{e}"))?;
                let elapsed = start.elapsed();
                results.push((format!("{metric:?}"), elapsed));
            }

            Ok(format!(
                "metrics: {}",
                results
                    .iter()
                    .map(|(m, t)| format!("{m}={:?}", t))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_14_large_cdist", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_14 failed");
}
