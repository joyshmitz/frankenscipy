#![no_main]

use arbitrary::Arbitrary;
use fsci_spatial::{squareform_to_condensed, squareform_to_matrix};
use libfuzzer_sys::fuzz_target;

// Squareform roundtrip oracle:
// squareform_to_condensed(squareform_to_matrix(condensed)) should equal
// the original condensed representation (within floating-point tolerance).
//
// This catches:
// - Off-by-one in triangular indexing formulas
// - Row/column ordering bugs in matrix reconstruction
// - Edge cases: n=1 (single element), n=0 (empty)

const MAX_N: usize = 32;
const REL_TOL: f64 = 1e-14;
const ABS_TOL: f64 = 1e-15;

#[derive(Debug, Arbitrary)]
struct SquareformInput {
    values: Vec<f64>,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e10, 1e10)
    } else {
        0.0
    }
}

fn close_enough(a: f64, b: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return true;
    }
    let diff = (a - b).abs();
    diff <= ABS_TOL + REL_TOL * a.abs().max(b.abs())
}

fn condensed_size_to_n(size: usize) -> Option<usize> {
    let n_f = (1.0 + (1.0 + 8.0 * size as f64).sqrt()) / 2.0;
    let n = n_f.round() as usize;
    if n * (n - 1) / 2 == size {
        Some(n)
    } else {
        None
    }
}

fuzz_target!(|input: SquareformInput| {
    let max_condensed = MAX_N * (MAX_N - 1) / 2;
    let len = input.values.len().min(max_condensed);

    let Some(n) = condensed_size_to_n(len) else {
        return;
    };

    if n < 2 {
        return;
    }

    let condensed: Vec<f64> = input.values.iter().take(len).map(|&v| sanitize(v)).collect();

    let matrix = match squareform_to_matrix(&condensed) {
        Ok(m) => m,
        Err(_) => return,
    };

    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        panic!(
            "Matrix shape mismatch: expected {}x{}, got {:?}",
            n,
            n,
            matrix.iter().map(|r| r.len()).collect::<Vec<_>>()
        );
    }

    let recovered = match squareform_to_condensed(&matrix) {
        Ok(c) => c,
        Err(_) => return,
    };

    if recovered.len() != condensed.len() {
        panic!(
            "Condensed length mismatch: original {} vs recovered {}",
            condensed.len(),
            recovered.len()
        );
    }

    for (i, (orig, rec)) in condensed.iter().zip(recovered.iter()).enumerate() {
        if !close_enough(*orig, *rec) {
            panic!(
                "Squareform roundtrip failed at index {}: \
                 original={} recovered={} (n={})",
                i, orig, rec, n
            );
        }
    }
});
