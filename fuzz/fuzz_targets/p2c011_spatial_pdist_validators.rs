#![no_main]

use arbitrary::Arbitrary;
use fsci_spatial::{
    DistanceMetric, is_valid_dm, is_valid_y, num_obs_dm, num_obs_y, pdist,
    squareform_to_condensed, squareform_to_matrix,
};
use libfuzzer_sys::fuzz_target;

// pdist + distance-matrix validator oracle:
//   For random X (rows × dim) and any DistanceMetric:
//     1. pdist(X) returns rows*(rows-1)/2 non-negative finite values
//     2. is_valid_y(pdist) is true
//     3. num_obs_y(pdist) == rows
//     4. squareform_to_matrix(pdist) is is_valid_dm(_, 0)
//     5. The reconstructed matrix has zero diagonal and is symmetric
//     6. squareform_to_condensed(matrix) recovers pdist exactly
//
// Catches regressions in pdist (any metric), the squareform pair, or
// the just-landed validators (frankenscipy-01528).

const MAX_ROWS: usize = 16;
const MAX_DIM: usize = 8;

#[derive(Debug, Arbitrary)]
struct PdistInput {
    raw_points: Vec<Vec<f64>>,
    metric_variant: u8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn pick_metric(v: u8) -> DistanceMetric {
    match v % 8 {
        0 => DistanceMetric::Euclidean,
        1 => DistanceMetric::SqEuclidean,
        2 => DistanceMetric::Cityblock,
        3 => DistanceMetric::Chebyshev,
        4 => DistanceMetric::Cosine,
        5 => DistanceMetric::Canberra,
        6 => DistanceMetric::Braycurtis,
        _ => DistanceMetric::Correlation,
    }
}

fuzz_target!(|input: PdistInput| {
    if input.raw_points.is_empty() {
        return;
    }
    let first_dim = input.raw_points[0].len();
    if first_dim == 0 || first_dim > MAX_DIM {
        return;
    }
    let points: Vec<Vec<f64>> = input
        .raw_points
        .iter()
        .take(MAX_ROWS)
        .filter(|p| p.len() == first_dim)
        .map(|p| p.iter().map(|&v| sanitize(v)).collect())
        .collect();
    if points.len() < 2 {
        return;
    }

    let metric = pick_metric(input.metric_variant);
    let Ok(y) = pdist(&points, metric) else {
        return;
    };

    // Property 1: length must equal n*(n-1)/2.
    let n = points.len();
    let expected_len = n * (n - 1) / 2;
    if y.len() != expected_len {
        panic!(
            "pdist length mismatch: got {} expected {} for n={n} metric={metric:?}",
            y.len(),
            expected_len
        );
    }

    // Property 2: all entries must be non-negative and finite (Cosine
    // and Correlation can return NaN if one of the inputs is the zero
    // vector — accept NaN there but reject negatives).
    for (i, &v) in y.iter().enumerate() {
        if v.is_finite() && v < -1e-12 {
            panic!("pdist[{i}] negative ({v}) for metric {metric:?}");
        }
    }

    // Property 3: is_valid_y must accept this length.
    if !is_valid_y(&y) {
        panic!("is_valid_y rejected pdist output of length {}", y.len());
    }

    // Property 4: num_obs_y inverts the pair count.
    if num_obs_y(&y) != n {
        panic!("num_obs_y returned {} but expected {n}", num_obs_y(&y));
    }

    // Property 5: roundtrip via squareform — only attempt when y is
    // free of NaN (cosine/correlation on zero vectors).
    if y.iter().all(|v| v.is_finite()) {
        let Ok(matrix) = squareform_to_matrix(&y) else {
            panic!("squareform_to_matrix failed on a is_valid_y vector");
        };
        if num_obs_dm(&matrix) != n {
            panic!("num_obs_dm returned {} but expected {n}", num_obs_dm(&matrix));
        }
        if !is_valid_dm(&matrix, 1e-9) {
            panic!("is_valid_dm rejected squareform(pdist) for metric {metric:?}");
        }
        // Diagonal must be zero
        for (i, row) in matrix.iter().enumerate() {
            if row[i].abs() > 1e-12 {
                panic!("matrix diagonal[{i}] = {} not zero", row[i]);
            }
        }
        // Reverse the roundtrip
        let Ok(recovered) = squareform_to_condensed(&matrix) else {
            panic!("squareform_to_condensed failed on a is_valid_dm matrix");
        };
        for (i, (a, b)) in y.iter().zip(recovered.iter()).enumerate() {
            if (a - b).abs() > 1e-12 + 1e-12 * a.abs().max(b.abs()) {
                panic!(
                    "pdist roundtrip mismatch at {i}: {a} vs {b} for metric {metric:?}"
                );
            }
        }
    }
});
