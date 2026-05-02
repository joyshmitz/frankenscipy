//! Metamorphic tests for `fsci-spatial`.
//!
//! Each relation is oracle-free: distance symmetry, identity of
//! indiscernibles, triangle inequality, pdist/squareform round-trip,
//! cdist symmetry, and KDTree nearest-distance correctness.
//!
//! Run with: `cargo test -p fsci-spatial --test metamorphic_tests`

use fsci_spatial::{
    ConvexHull, DistanceMetric, KDTree, angle_between, cartesian_to_cylindrical,
    cartesian_to_spherical, cdist_metric, chebyshev, cityblock, cosine, cross_3d,
    cylindrical_to_cartesian, dot, euclidean, jensenshannon, metric_distance, minkowski, pdist,
    procrustes, rotate_point, rotation_matrix, spherical_to_cartesian, squareform_to_condensed,
    squareform_to_matrix,
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

// ─────────────────────────────────────────────────────────────────────
// MR17 — Minkowski with p = 2 equals Euclidean; with p = 1 equals
// Cityblock; with p → ∞ approaches Chebyshev.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minkowski_specializes_to_classic_metrics() {
    let pts = sample_points();
    for a in &pts {
        for b in &pts {
            let mp2 = minkowski(a, b, 2.0);
            let euc = euclidean(a, b);
            assert!(
                close(mp2, euc),
                "MR17 minkowski(p=2) = {mp2} != euclidean = {euc}"
            );
            let mp1 = minkowski(a, b, 1.0);
            let cb = cityblock(a, b);
            assert!(
                close(mp1, cb),
                "MR17 minkowski(p=1) = {mp1} != cityblock = {cb}"
            );
            // p = 50 should be very close to Chebyshev (within 5% relative on
            // non-zero distances).
            let mp_hi = minkowski(a, b, 50.0);
            let cheb = chebyshev(a, b);
            if cheb > 1e-6 {
                let rel = (mp_hi - cheb).abs() / cheb;
                assert!(
                    rel < 0.05,
                    "MR17 minkowski(p=50) = {mp_hi} vs chebyshev = {cheb} (rel diff {rel})"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — chebyshev ≤ euclidean ≤ cityblock for any pair (in Rᵈ).
// (Standard p-norm inequality between p=∞, p=2, p=1.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_metric_norm_ordering() {
    let pts = sample_points();
    for a in &pts {
        for b in &pts {
            let cheb = chebyshev(a, b);
            let euc = euclidean(a, b);
            let cb = cityblock(a, b);
            assert!(
                cheb <= euc + 1e-10,
                "MR18 chebyshev = {cheb} > euclidean = {euc}"
            );
            assert!(
                euc <= cb + 1e-10,
                "MR18 euclidean = {euc} > cityblock = {cb}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — Cosine distance is non-negative and bounded above by 2 for
// any pair of non-zero vectors, and is 0 for v vs c·v with c > 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cosine_distance_bounds_and_scale_invariance() {
    let pts = sample_points();
    for a in &pts {
        for b in &pts {
            let d = cosine(a, b);
            assert!(
                d >= -1e-12 && d <= 2.0 + 1e-12,
                "MR19 cosine = {d}, expected in [0, 2]"
            );
        }
        // a vs 3·a should be ≈ 0 (parallel).
        let scaled: Vec<f64> = a.iter().map(|&x| 3.0 * x).collect();
        if a.iter().any(|&x| x.abs() > 1e-9) {
            let d_par = cosine(a, &scaled);
            assert!(
                d_par.abs() < 1e-9,
                "MR19 cosine(a, 3a) = {d_par}, expected ≈ 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — Jensen-Shannon distance is symmetric and non-negative on two
// probability distributions.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_jensenshannon_symmetric_nonneg() {
    let dists = [
        vec![0.5_f64, 0.3, 0.2],
        vec![0.1_f64, 0.6, 0.3],
        vec![0.25_f64, 0.25, 0.5],
        vec![1.0_f64 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ];
    for p in &dists {
        for q in &dists {
            let pq = jensenshannon(p, q, None);
            let qp = jensenshannon(q, p, None);
            assert!(pq >= -1e-12, "MR20 JS = {pq} negative");
            assert!(
                close(pq, qp),
                "MR20 JS not symmetric: pq = {pq}, qp = {qp}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — procrustes(A, A) yields disparity ≈ 0 (Procrustes distance
// of a configuration to itself is zero).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_procrustes_self_disparity_zero() {
    let a: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
        vec![0.5, 0.5],
    ];
    let res = procrustes(&a, &a).expect("procrustes self");
    assert!(
        res.disparity < 1e-9,
        "MR21 procrustes(A, A) disparity = {}, expected ≈ 0",
        res.disparity
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — KDTree.query returns a self-match with distance 0 when
// querying a point already in the tree.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kdtree_query_self_match_distance_zero() {
    let pts = sample_points();
    let tree = KDTree::new(&pts).expect("kdtree");
    for (i, p) in pts.iter().enumerate() {
        let (idx, dist) = tree.query(p).expect("query");
        assert_eq!(idx, i, "MR22 self-match index for point {i}: got {idx}");
        assert!(
            dist < 1e-12,
            "MR22 self-match distance for point {i}: {dist}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — spherical→cartesian→spherical preserves r and yields the same
// (x, y, z) when round-tripped via cartesian_to_spherical.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_spherical_cartesian_roundtrip() {
    use std::f64::consts::PI;
    let cases = [
        (1.0_f64, PI / 4.0, 0.5),
        (3.5, PI / 3.0, 1.2),
        (2.0, PI / 2.0, 0.0),
        (5.0, PI / 6.0, -1.0),
    ];
    for &(r, theta, phi) in &cases {
        let (x, y, z) = spherical_to_cartesian(r, theta, phi);
        let (r2, theta2, phi2) = cartesian_to_spherical(x, y, z);
        assert!(
            (r - r2).abs() < 1e-12,
            "MR23 r mismatch: {r} vs {r2}"
        );
        assert!(
            (theta - theta2).abs() < 1e-12,
            "MR23 theta mismatch: {theta} vs {theta2}"
        );
        // φ may differ by ±2π — compare via cos and sin.
        assert!(
            (phi.cos() - phi2.cos()).abs() < 1e-9
                && (phi.sin() - phi2.sin()).abs() < 1e-9,
            "MR23 phi mismatch: {phi} vs {phi2}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — cylindrical→cartesian→cylindrical preserves ρ and z; θ
// matches modulo 2π.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cylindrical_cartesian_roundtrip() {
    use std::f64::consts::PI;
    let cases = [
        (1.0_f64, 0.0, 0.0),
        (2.5, PI / 4.0, 1.0),
        (3.0, PI / 2.0, -2.5),
        (0.5, PI, 5.0),
    ];
    for &(rho, theta, z) in &cases {
        let (x, y, zc) = cylindrical_to_cartesian(rho, theta, z);
        let (rho2, theta2, z2) = cartesian_to_cylindrical(x, y, zc);
        assert!(
            (rho - rho2).abs() < 1e-12,
            "MR24 rho mismatch: {rho} vs {rho2}"
        );
        assert!(
            (z - z2).abs() < 1e-12,
            "MR24 z mismatch: {z} vs {z2}"
        );
        assert!(
            (theta.cos() - theta2.cos()).abs() < 1e-9
                && (theta.sin() - theta2.sin()).abs() < 1e-9,
            "MR24 theta mismatch: {theta} vs {theta2}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — A rotation about a unit axis preserves the L2 norm of any
// vector it is applied to.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotation_preserves_norm() {
    use std::f64::consts::PI;
    let axes: &[[f64; 3]] = &[
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [
            1.0 / (3.0_f64).sqrt(),
            1.0 / (3.0_f64).sqrt(),
            1.0 / (3.0_f64).sqrt(),
        ],
    ];
    let points: &[[f64; 3]] = &[
        [1.0, 2.0, 3.0],
        [-1.0, 0.5, 4.0],
        [2.5, -3.0, 1.0],
    ];
    for axis in axes {
        for angle in &[0.0_f64, PI / 6.0, PI / 4.0, PI / 2.0, PI] {
            let r = rotation_matrix(axis, *angle);
            for p in points {
                let q = rotate_point(&r, p);
                let np: f64 = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                let nq: f64 = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
                assert!(
                    (np - nq).abs() < 1e-9,
                    "MR25 rotation broke norm: {np} vs {nq}"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — cross_3d is antisymmetric: a × b = -(b × a).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cross_3d_antisymmetric() {
    let pairs: &[([f64; 3], [f64; 3])] = &[
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ([-1.0, 0.5, 2.5], [3.0, -1.5, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
    ];
    for (a, b) in pairs {
        let ab = cross_3d(a, b);
        let ba = cross_3d(b, a);
        for k in 0..3 {
            assert!(
                (ab[k] + ba[k]).abs() < 1e-12,
                "MR26 a×b + b×a [{k}] = {} != 0",
                ab[k] + ba[k]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — dot(a, a) = ‖a‖² and dot is symmetric in its arguments.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dot_self_norm_squared_and_symmetry() {
    let vectors = sample_points();
    for a in &vectors {
        let aa = dot(a, a);
        let norm_sq: f64 = a.iter().map(|&v| v * v).sum();
        assert!(
            (aa - norm_sq).abs() < 1e-9,
            "MR27 dot(a, a) = {aa} vs ‖a‖² = {norm_sq}"
        );
        for b in &vectors {
            let ab = dot(a, b);
            let ba = dot(b, a);
            assert!(
                (ab - ba).abs() < 1e-12,
                "MR27 dot symmetry: {ab} vs {ba}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — angle_between is in [0, π] for any two non-zero vectors.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_angle_between_in_zero_pi() {
    use std::f64::consts::PI;
    let vectors = sample_points();
    for a in &vectors {
        for b in &vectors {
            let ang = angle_between(a, b);
            if ang.is_finite() {
                assert!(
                    ang >= -1e-12 && ang <= PI + 1e-12,
                    "MR28 angle_between = {ang} not in [0, π]"
                );
            }
        }
    }
}


