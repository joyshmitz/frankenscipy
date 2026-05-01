//! Metamorphic tests for `fsci-spatial`.
//!
//! Each relation is oracle-free: distance symmetry, identity of
//! indiscernibles, triangle inequality, pdist/squareform round-trip,
//! cdist symmetry, and KDTree nearest-distance correctness.
//!
//! Run with: `cargo test -p fsci-spatial --test metamorphic_tests`

use fsci_spatial::{
    ConvexHull, DistanceMetric, KDTree, cdist_metric, euclidean, metric_distance, pdist,
    squareform_to_condensed, squareform_to_matrix,
};

const ATOL: f64 = 1e-12;
const RTOL: f64 = 1e-10;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

const ALL_METRICS: &[DistanceMetric] = &[
    DistanceMetric::Euclidean,
    DistanceMetric::SqEuclidean,
    DistanceMetric::Cityblock,
    DistanceMetric::Chebyshev,
    DistanceMetric::Cosine,
    DistanceMetric::Correlation,
    DistanceMetric::Hamming,
    DistanceMetric::Jaccard,
    DistanceMetric::Canberra,
    DistanceMetric::Braycurtis,
];

fn sample_points() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4.0, 1.0, 2.0, 5.0],
        vec![-1.0, 0.0, 5.0, 2.0],
        vec![3.0, 3.0, 3.0, 3.0],
        vec![0.0, -2.0, 4.0, 1.0],
        vec![2.5, 1.5, 0.5, 3.5],
        vec![-3.0, 4.0, -2.0, 0.0],
    ]
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — distance symmetry: d(a, b) = d(b, a) for every metric
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_distance_symmetric() {
    let pts = sample_points();
    for &metric in ALL_METRICS {
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                let dab = metric_distance(&pts[i], &pts[j], metric);
                let dba = metric_distance(&pts[j], &pts[i], metric);
                assert!(
                    close(dab, dba),
                    "MR1 {metric:?} d({i},{j})={dab} d({j},{i})={dba}"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — identity of indiscernibles: d(a, a) = 0 for every metric
//        (Hamming/Jaccard return 0 for identical points)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_distance_self_zero() {
    let pts = sample_points();
    for &metric in ALL_METRICS {
        for (i, p) in pts.iter().enumerate() {
            let d = metric_distance(p, p, metric);
            assert!(
                close(d, 0.0),
                "MR2 {metric:?} d({i},{i}) = {d}, expected 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — triangle inequality for true metrics: d(a,c) ≤ d(a,b) + d(b,c)
//        Holds for Euclidean, Cityblock, Chebyshev (and Minkowski-p≥1)
//        but NOT for Correlation/Cosine/Bray-Curtis in general — those
//        are excluded from this relation.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_triangle_inequality_metric_distances() {
    let pts = sample_points();
    let metrics = [
        DistanceMetric::Euclidean,
        DistanceMetric::Cityblock,
        DistanceMetric::Chebyshev,
    ];
    let n = pts.len();
    for &metric in &metrics {
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let dab = metric_distance(&pts[i], &pts[j], metric);
                    let dbc = metric_distance(&pts[j], &pts[k], metric);
                    let dac = metric_distance(&pts[i], &pts[k], metric);
                    // Allow a small relative slack for f64 round-off when
                    // dac is close to the sum.
                    let bound = (dab + dbc) * (1.0 + 1e-12) + 1e-15;
                    assert!(
                        dac <= bound,
                        "MR3 {metric:?} triangle violated for ({i},{j},{k}): d_ac={dac} > d_ab+d_bc={}, bound={bound}",
                        dab + dbc
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — pdist ↔ squareform round-trip: condensed → square → condensed
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pdist_squareform_roundtrip() {
    let pts = sample_points();
    let condensed = pdist(&pts, DistanceMetric::Euclidean).unwrap();
    let square = squareform_to_matrix(&condensed).unwrap();
    let back = squareform_to_condensed(&square).unwrap();

    assert_eq!(back.len(), condensed.len(), "MR4 round-trip length");
    for (i, (got, want)) in back.iter().zip(&condensed).enumerate() {
        assert!(
            close(*got, *want),
            "MR4 round-trip element {i}: got={got} expected={want}"
        );
    }
    // Square matrix must be symmetric with zero diagonal.
    let n = pts.len();
    assert_eq!(square.len(), n);
    for i in 0..n {
        assert!((square[i][i]).abs() < 1e-15, "diagonal not zero at {i}");
        for j in (i + 1)..n {
            assert!(
                close(square[i][j], square[j][i]),
                "square not symmetric at ({i},{j})"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — cdist(X, X) is symmetric and diagonal == 0 for true metrics.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdist_self_symmetric_zero_diagonal() {
    let pts = sample_points();
    let m = cdist_metric(&pts, &pts, DistanceMetric::Euclidean).unwrap();
    let n = pts.len();
    assert_eq!(m.len(), n);
    for i in 0..n {
        assert_eq!(m[i].len(), n);
        assert!(
            m[i][i].abs() < 1e-12,
            "MR5 cdist diagonal not zero at {i}: {}",
            m[i][i]
        );
        for j in (i + 1)..n {
            assert!(
                close(m[i][j], m[j][i]),
                "MR5 cdist not symmetric at ({i},{j}): {} vs {}",
                m[i][j],
                m[j][i]
            );
            assert!(
                m[i][j] >= 0.0,
                "MR5 cdist negative at ({i},{j}): {}",
                m[i][j]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — KDTree.query returns the true Euclidean nearest neighbor.
//
// Build a tree, query each input point — the nearest must be the
// point itself (distance 0). Then query a perturbed point and check
// its returned distance equals the brute-force minimum.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kdtree_query_matches_brute_force() {
    let pts = sample_points();
    let tree = KDTree::new(&pts).unwrap();

    // 1. Self-query: each input must find itself with distance 0.
    for (i, p) in pts.iter().enumerate() {
        let (idx, d) = tree.query(p).unwrap();
        assert_eq!(idx, i, "MR6 self-query: expected {i}, got {idx}");
        assert!(d.abs() < 1e-12, "MR6 self-query distance: {d}");
    }

    // 2. Perturbed-query: distance must equal the true nearest.
    let q = vec![1.5, 2.5, 2.5, 3.5];
    let (kd_idx, kd_dist) = tree.query(&q).unwrap();
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for (i, p) in pts.iter().enumerate() {
        let d = euclidean(&q, p);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    assert_eq!(
        kd_idx, best_idx,
        "MR6 KDTree returned wrong index: {kd_idx} vs brute {best_idx}"
    );
    assert!(
        close(kd_dist, best_dist),
        "MR6 KDTree distance: {kd_dist} vs brute {best_dist}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — KDTree.query_ball_point returns *all* points within radius r.
//
// Brute-force the ball and verify both returned-vs-true sets agree
// after sorting.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kdtree_query_ball_point_matches_brute_force() {
    let pts = sample_points();
    let tree = KDTree::new(&pts).unwrap();
    let q = vec![1.5, 2.5, 2.5, 3.5];
    let r = 4.0;

    let mut from_tree = tree.query_ball_point(&q, r).unwrap();
    let mut from_brute: Vec<usize> = pts
        .iter()
        .enumerate()
        .filter(|(_, p)| euclidean(&q, p) <= r)
        .map(|(i, _)| i)
        .collect();

    from_tree.sort();
    from_brute.sort();
    assert_eq!(
        from_tree, from_brute,
        "MR7 KDTree ball: tree={from_tree:?} brute={from_brute:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — squared Euclidean equals Euclidean squared (per element).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sqeuclidean_equals_euclidean_squared() {
    let pts = sample_points();
    for i in 0..pts.len() {
        for j in 0..pts.len() {
            let e = metric_distance(&pts[i], &pts[j], DistanceMetric::Euclidean);
            let sq = metric_distance(&pts[i], &pts[j], DistanceMetric::SqEuclidean);
            assert!(
                close(sq, e * e),
                "MR8 sqeuclidean({i},{j})={sq} vs euclidean²={}",
                e * e
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — ConvexHull encloses every input point: each point must be on
// or inside the hull (signed distance from each hull edge ≥ −ε).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convex_hull_encloses_all_inputs() {
    // 7 points forming a square plus interior points.
    let pts: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 3.0),
        (0.0, 3.0),
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 1.5),
    ];
    let hull = ConvexHull::new(&pts).unwrap();

    // Get hull vertices in CCW order.
    let hull_pts: Vec<(f64, f64)> = hull.vertices.iter().map(|&i| pts[i]).collect();
    let m = hull_pts.len();
    assert!(m >= 3, "convex hull must have at least 3 vertices");

    // For every input point, signed distance from each hull edge must be
    // ≥ 0 (point is on the same side as the interior, or on the edge).
    // For a CCW-oriented polygon, interior points yield positive cross
    // products for every edge.
    for (idx, &p) in pts.iter().enumerate() {
        for k in 0..m {
            let a = hull_pts[k];
            let b = hull_pts[(k + 1) % m];
            let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            assert!(
                cross >= -1e-12,
                "MR9 point {idx} ({p:?}) outside hull edge ({a:?} → {b:?}): cross={cross}"
            );
        }
    }

    // Hull area for the 4x3 rectangle must be 12.
    let expected_area = 12.0_f64;
    assert!(
        (hull.area - expected_area).abs() < 1e-10,
        "MR9 hull area: {} vs {expected_area}",
        hull.area
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — ConvexHull of a triangle returns the triangle (3 vertices)
// and the area of the triangle.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convex_hull_triangle() {
    let pts = vec![(0.0_f64, 0.0), (4.0, 0.0), (0.0, 3.0)];
    let hull = ConvexHull::new(&pts).unwrap();
    assert_eq!(hull.vertices.len(), 3, "triangle should have 3 hull vertices");
    let expected_area = 6.0_f64; // 1/2 · base · height = 1/2 · 4 · 3
    assert!(
        (hull.area - expected_area).abs() < 1e-10,
        "MR10 triangle area: {} vs {expected_area}",
        hull.area
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — ConvexHull is invariant under translation: shifting every
// point by (Δx, Δy) leaves area and the set of hull vertices unchanged.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convex_hull_translation_invariance() {
    let base: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (5.0, 0.0),
        (5.0, 4.0),
        (0.0, 4.0),
        (2.0, 2.0),
        (3.5, 1.0),
    ];
    let hull_a = ConvexHull::new(&base).unwrap();
    let dx = 7.5_f64;
    let dy = -3.2_f64;
    let shifted: Vec<(f64, f64)> = base.iter().map(|&(x, y)| (x + dx, y + dy)).collect();
    let hull_b = ConvexHull::new(&shifted).unwrap();
    assert!(
        (hull_a.area - hull_b.area).abs() < 1e-10,
        "MR11 area changed under translation: {} vs {}",
        hull_a.area,
        hull_b.area
    );
    assert_eq!(
        hull_a.vertices.len(),
        hull_b.vertices.len(),
        "MR11 vertex count differs"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — KDTree.query_k returns k results sorted by ascending distance
// and contains the brute-force k-nearest set.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kdtree_query_k_sorted_and_correct() {
    let pts = sample_points();
    let tree = KDTree::new(&pts).unwrap();
    let q = vec![1.5_f64, 2.5, 2.5, 3.5];
    let k = 3;
    let result = tree.query_k(&q, k).unwrap();
    assert_eq!(result.len(), k, "MR12 query_k length");

    // Distances are non-decreasing.
    for w in result.windows(2) {
        assert!(
            w[0].1 <= w[1].1 + 1e-12,
            "MR12 distances not sorted ascending: {} > {}",
            w[0].1,
            w[1].1
        );
    }

    // Cross-check: sort all brute-force distances and compare top-k.
    let mut brute: Vec<(usize, f64)> =
        pts.iter().enumerate().map(|(i, p)| (i, euclidean(&q, p))).collect();
    brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for i in 0..k {
        assert!(
            close(result[i].1, brute[i].1),
            "MR12 distance #{i}: tree={} brute={}",
            result[i].1,
            brute[i].1
        );
        assert_eq!(
            result[i].0, brute[i].0,
            "MR12 index #{i}: tree={} brute={}",
            result[i].0, brute[i].0
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — KDTree.size matches input length.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kdtree_size_matches_input() {
    let pts = sample_points();
    let tree = KDTree::new(&pts).unwrap();
    assert_eq!(tree.size(), pts.len(), "MR13 KDTree.size mismatch");
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — cdist row count matches xa.len() and column count matches
// xb.len(); all entries are non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdist_dimensions_and_nonneg() {
    let xa = vec![
        vec![0.0_f64, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ];
    let xb = vec![
        vec![5.0_f64, 5.0],
        vec![5.0, 6.0],
    ];
    let m = cdist_metric(&xa, &xb, DistanceMetric::Euclidean).unwrap();
    assert_eq!(m.len(), xa.len(), "MR14 cdist row count");
    for (i, row) in m.iter().enumerate() {
        assert_eq!(row.len(), xb.len(), "MR14 cdist row {i} col count");
        for &v in row {
            assert!(v >= 0.0, "MR14 cdist negative entry: {v}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — cdist is translation-invariant: shifting both xa and xb by
// the same (Δ, Δ) leaves the distance matrix unchanged.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdist_translation_invariance() {
    let xa = vec![vec![0.0_f64, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let xb = vec![vec![5.0_f64, 5.0], vec![5.0, 6.0]];
    let m_base = cdist_metric(&xa, &xb, DistanceMetric::Euclidean).unwrap();

    let dx = 7.5_f64;
    let dy = -3.2_f64;
    let xa_t: Vec<Vec<f64>> = xa.iter().map(|p| vec![p[0] + dx, p[1] + dy]).collect();
    let xb_t: Vec<Vec<f64>> = xb.iter().map(|p| vec![p[0] + dx, p[1] + dy]).collect();
    let m_shift = cdist_metric(&xa_t, &xb_t, DistanceMetric::Euclidean).unwrap();
    for (r1, r2) in m_base.iter().zip(&m_shift) {
        for (a, b) in r1.iter().zip(r2) {
            assert!(close(*a, *b), "MR15 cdist translation: {a} vs {b}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — cdist between points and themselves equals pdist as a square
// matrix on the off-diagonal entries.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdist_self_matches_pdist() {
    let pts = sample_points();
    let cd = cdist_metric(&pts, &pts, DistanceMetric::Euclidean).unwrap();
    let condensed = pdist(&pts, DistanceMetric::Euclidean).unwrap();
    let n = pts.len();
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let from_cdist = cd[i][j];
            let from_pdist = condensed[idx];
            assert!(
                close(from_cdist, from_pdist),
                "MR16 cdist[{i},{j}]={from_cdist} pdist[{idx}]={from_pdist}"
            );
            idx += 1;
        }
    }
}
