#![no_main]

use arbitrary::Arbitrary;
use fsci_opt::nnls;
use libfuzzer_sys::fuzz_target;

// NNLS (Non-Negative Least Squares) oracle:
// Tests nnls(A, b) for correctness properties:
//
// 1. Solution x must be non-negative (x[i] >= 0 for all i)
// 2. Returned residual matches ||Ax - b||_2
// 3. Residual is non-negative
// 4. Output length matches number of columns in A

const MAX_ROWS: usize = 16;
const MAX_COLS: usize = 16;

#[derive(Debug, Arbitrary)]
struct NnlsInput {
    matrix: Vec<Vec<f64>>,
    b: Vec<f64>,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fuzz_target!(|input: NnlsInput| {
    let m = input.matrix.len().min(MAX_ROWS);
    if m == 0 {
        return;
    }

    let n = input.matrix.first().map_or(0, |row| row.len().min(MAX_COLS));
    if n == 0 {
        return;
    }

    let a: Vec<Vec<f64>> = input
        .matrix
        .iter()
        .take(m)
        .map(|row| {
            row.iter()
                .take(n)
                .map(|&v| sanitize(v))
                .chain(std::iter::repeat(0.0))
                .take(n)
                .collect()
        })
        .collect();

    let b: Vec<f64> = input
        .b
        .iter()
        .take(m)
        .map(|&v| sanitize(v))
        .chain(std::iter::repeat(0.0))
        .take(m)
        .collect();

    let result = match nnls(&a, &b) {
        Ok(r) => r,
        Err(_) => return,
    };

    let (x, residual) = result;

    // Property 1: x length matches columns
    if x.len() != n {
        panic!(
            "NNLS solution length {} != columns {} (m={}, n={})",
            x.len(),
            n,
            m,
            n
        );
    }

    // Property 2: x must be non-negative
    for (i, &xi) in x.iter().enumerate() {
        if xi < -1e-10 {
            panic!(
                "NNLS solution x[{}] = {} is negative (should be >= 0)",
                i, xi
            );
        }
    }

    // Property 3: residual must be non-negative
    if residual < -1e-10 {
        panic!("NNLS residual {} is negative", residual);
    }

    // Property 4: verify residual = ||Ax - b||_2
    let mut ax = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            ax[i] += a[i][j] * x[j];
        }
    }

    let computed_residual: f64 = ax
        .iter()
        .zip(b.iter())
        .map(|(&axi, &bi)| (axi - bi).powi(2))
        .sum::<f64>()
        .sqrt();

    let residual_diff = (computed_residual - residual).abs();
    let residual_rtol = 1e-6 * computed_residual.max(residual).max(1.0);

    if residual_diff > residual_rtol {
        panic!(
            "NNLS residual mismatch: returned {} but computed ||Ax-b||_2 = {} (diff={})",
            residual, computed_residual, residual_diff
        );
    }
});
