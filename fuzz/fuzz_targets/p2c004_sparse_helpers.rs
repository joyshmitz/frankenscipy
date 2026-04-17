#![no_main]

use arbitrary::Arbitrary;
use fsci_sparse::{
    find, hstack, tril, triu, vstack, CooMatrix, CscMatrix, CsrMatrix, FormatConvertible,
    Shape2D, SparseError,
};
use libfuzzer_sys::fuzz_target;

const MAX_DIM: usize = 6;
const MAX_ENTRIES: usize = 24;

#[derive(Clone, Copy, Debug, Arbitrary)]
enum MatrixFormat {
    Coo,
    Csr,
    Csc,
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct Entry {
    row: u8,
    col: u8,
    value: i8,
}

#[derive(Clone, Debug, Arbitrary)]
struct SparseMatrixSpec {
    format: MatrixFormat,
    rows: u8,
    cols: u8,
    entries: Vec<Entry>,
}

#[derive(Clone, Debug, Arbitrary)]
struct SparseHelpersInput {
    primary: SparseMatrixSpec,
    secondary: SparseMatrixSpec,
    k: i8,
}

enum SparseInputMatrix {
    Coo(CooMatrix),
    Csr(CsrMatrix),
    Csc(CscMatrix),
}

impl SparseInputMatrix {
    fn as_format_convertible(&self) -> &dyn FormatConvertible {
        match self {
            Self::Coo(matrix) => matrix,
            Self::Csr(matrix) => matrix,
            Self::Csc(matrix) => matrix,
        }
    }
}

impl FormatConvertible for SparseInputMatrix {
    fn to_csr(&self) -> Result<CsrMatrix, SparseError> {
        match self {
            Self::Coo(matrix) => matrix.to_csr(),
            Self::Csr(matrix) => matrix.to_csr(),
            Self::Csc(matrix) => matrix.to_csr(),
        }
    }

    fn to_csc(&self) -> Result<CscMatrix, SparseError> {
        match self {
            Self::Coo(matrix) => matrix.to_csc(),
            Self::Csr(matrix) => matrix.to_csc(),
            Self::Csc(matrix) => matrix.to_csc(),
        }
    }

    fn to_coo(&self) -> Result<CooMatrix, SparseError> {
        match self {
            Self::Coo(matrix) => matrix.to_coo(),
            Self::Csr(matrix) => matrix.to_coo(),
            Self::Csc(matrix) => matrix.to_coo(),
        }
    }
}

fn clamp_dimension(raw: u8) -> usize {
    usize::from(raw) % MAX_DIM
}

fn normalize_value(raw: i8) -> f64 {
    f64::from(i16::from(raw).rem_euclid(9) - 4)
}

fn normalized_entries(spec: &SparseMatrixSpec) -> (Shape2D, Vec<f64>, Vec<usize>, Vec<usize>) {
    let rows = clamp_dimension(spec.rows);
    let cols = clamp_dimension(spec.cols);
    let shape = Shape2D::new(rows, cols);
    if rows == 0 || cols == 0 {
        return (shape, Vec::new(), Vec::new(), Vec::new());
    }

    let mut data = Vec::with_capacity(spec.entries.len().min(MAX_ENTRIES));
    let mut row_indices = Vec::with_capacity(spec.entries.len().min(MAX_ENTRIES));
    let mut col_indices = Vec::with_capacity(spec.entries.len().min(MAX_ENTRIES));
    for entry in spec.entries.iter().take(MAX_ENTRIES) {
        row_indices.push(usize::from(entry.row) % rows);
        col_indices.push(usize::from(entry.col) % cols);
        data.push(normalize_value(entry.value));
    }
    (shape, data, row_indices, col_indices)
}

fn build_matrix(spec: &SparseMatrixSpec) -> Option<SparseInputMatrix> {
    let (shape, data, row_indices, col_indices) = normalized_entries(spec);
    let coo_result = CooMatrix::from_triplets(shape, data, row_indices, col_indices, false);
    assert!(coo_result.is_ok(), "normalized fuzz spec should build COO");
    let Ok(coo) = coo_result else {
        return None;
    };
    match spec.format {
        MatrixFormat::Coo => Some(SparseInputMatrix::Coo(coo)),
        MatrixFormat::Csr => {
            let csr_result = coo.to_csr();
            assert!(csr_result.is_ok(), "COO should convert to CSR");
            let Ok(csr) = csr_result else {
                return None;
            };
            Some(SparseInputMatrix::Csr(csr))
        }
        MatrixFormat::Csc => {
            let csc_result = coo.to_csc();
            assert!(csc_result.is_ok(), "COO should convert to CSC");
            let Ok(csc) = csc_result else {
                return None;
            };
            Some(SparseInputMatrix::Csc(csc))
        }
    }
}

fn dense_from_spec(spec: &SparseMatrixSpec) -> Vec<Vec<f64>> {
    let (shape, data, row_indices, col_indices) = normalized_entries(spec);
    let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
    for idx in 0..data.len() {
        dense[row_indices[idx]][col_indices[idx]] += data[idx];
    }
    dense
}

fn dense_from_coo(coo: &CooMatrix) -> Vec<Vec<f64>> {
    let shape = coo.shape();
    let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
    for idx in 0..coo.nnz() {
        dense[coo.row_indices()[idx]][coo.col_indices()[idx]] += coo.data()[idx];
    }
    dense
}

fn find_triplets_from_dense(dense: &[Vec<f64>]) -> Vec<(usize, usize, f64)> {
    let mut triplets = Vec::new();
    for (row_idx, row) in dense.iter().enumerate() {
        for (col_idx, value) in row.iter().enumerate() {
            if *value != 0.0 {
                triplets.push((row_idx, col_idx, *value));
            }
        }
    }
    triplets
}

fn sorted_triplets(mut triplets: Vec<(usize, usize, f64)>) -> Vec<(usize, usize, f64)> {
    triplets.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.total_cmp(&right.2))
    });
    triplets
}

fn lower_triangle_dense(dense: &[Vec<f64>], k: isize) -> Vec<Vec<f64>> {
    dense
        .iter()
        .enumerate()
        .map(|(row_idx, row)| {
            row.iter()
                .enumerate()
                .map(|(col_idx, value)| {
                    let keep = if k >= 0 {
                        row_idx.saturating_add(k as usize) >= col_idx
                    } else {
                        match col_idx.checked_add(k.unsigned_abs()) {
                            Some(limit) => row_idx >= limit,
                            None => false,
                        }
                    };
                    if keep { *value } else { 0.0 }
                })
                .collect()
        })
        .collect()
}

fn upper_triangle_dense(dense: &[Vec<f64>], k: isize) -> Vec<Vec<f64>> {
    dense
        .iter()
        .enumerate()
        .map(|(row_idx, row)| {
            row.iter()
                .enumerate()
                .map(|(col_idx, value)| {
                    let keep = if k >= 0 {
                        match row_idx.checked_add(k as usize) {
                            Some(diagonal_col) => diagonal_col <= col_idx,
                            None => false,
                        }
                    } else {
                        match col_idx.checked_add(k.unsigned_abs()) {
                            Some(limit) => row_idx <= limit,
                            None => true,
                        }
                    };
                    if keep { *value } else { 0.0 }
                })
                .collect()
        })
        .collect()
}

fn vstack_dense(top: &[Vec<f64>], bottom: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let top_cols = top.first().map_or(0, Vec::len);
    let bottom_cols = bottom.first().map_or(0, Vec::len);
    if top_cols != bottom_cols {
        return None;
    }

    let mut stacked = Vec::with_capacity(top.len() + bottom.len());
    stacked.extend(top.iter().cloned());
    stacked.extend(bottom.iter().cloned());
    Some(stacked)
}

fn hstack_dense(left: &[Vec<f64>], right: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    if left.len() != right.len() {
        return None;
    }

    let mut stacked = Vec::with_capacity(left.len());
    for (left_row, right_row) in left.iter().zip(right.iter()) {
        let mut row = Vec::with_capacity(left_row.len() + right_row.len());
        row.extend_from_slice(left_row);
        row.extend_from_slice(right_row);
        stacked.push(row);
    }
    Some(stacked)
}

fn assert_dense_eq(actual: &[Vec<f64>], expected: &[Vec<f64>], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: row count mismatch");
    for (row_idx, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual_row.len(),
            expected_row.len(),
            "{context}: col count mismatch on row {row_idx}"
        );
        for (col_idx, (actual_value, expected_value)) in
            actual_row.iter().zip(expected_row.iter()).enumerate()
        {
            assert_eq!(
                actual_value, expected_value,
                "{context}: mismatch at ({row_idx}, {col_idx})"
            );
        }
    }
}

fuzz_target!(|input: SparseHelpersInput| {
    let Some(primary) = build_matrix(&input.primary) else {
        return;
    };
    let Some(secondary) = build_matrix(&input.secondary) else {
        return;
    };
    let primary_dense = dense_from_spec(&input.primary);
    let secondary_dense = dense_from_spec(&input.secondary);
    let diagonal_offset = isize::from(input.k).clamp(-8, 8);

    let find_result = find(&primary);
    assert!(find_result.is_ok(), "find should succeed");
    let Ok((rows, cols, data)) = find_result else {
        return;
    };
    let actual_find = sorted_triplets(
        rows.into_iter()
            .zip(cols)
            .zip(data)
            .map(|((row, col), value)| (row, col, value))
            .collect(),
    );
    let expected_find = sorted_triplets(find_triplets_from_dense(&primary_dense));
    assert_eq!(actual_find, expected_find, "find should match dense semantics");

    let tril_result = tril(&primary, diagonal_offset);
    assert!(tril_result.is_ok(), "tril should succeed");
    let Ok(actual_tril_sparse) = tril_result else {
        return;
    };
    let actual_tril = dense_from_coo(&actual_tril_sparse);
    let expected_tril = lower_triangle_dense(&primary_dense, diagonal_offset);
    assert_dense_eq(&actual_tril, &expected_tril, "tril");

    let triu_result = triu(&primary, diagonal_offset);
    assert!(triu_result.is_ok(), "triu should succeed");
    let Ok(actual_triu_sparse) = triu_result else {
        return;
    };
    let actual_triu = dense_from_coo(&actual_triu_sparse);
    let expected_triu = upper_triangle_dense(&primary_dense, diagonal_offset);
    assert_dense_eq(&actual_triu, &expected_triu, "triu");

    let blocks = [
        primary.as_format_convertible(),
        secondary.as_format_convertible(),
    ];

    match vstack_dense(&primary_dense, &secondary_dense) {
        Some(expected) => {
            let actual_result = vstack(&blocks);
            assert!(actual_result.is_ok(), "vstack should succeed");
            let Ok(actual) = actual_result else {
                return;
            };
            let actual_coo_result = actual.to_coo();
            assert!(actual_coo_result.is_ok(), "vstack result should convert to COO");
            let Ok(actual_coo) = actual_coo_result else {
                return;
            };
            assert_dense_eq(&dense_from_coo(&actual_coo), &expected, "vstack");
        }
        None => {
            assert!(
                matches!(vstack(&blocks), Err(SparseError::IncompatibleShape { .. })),
                "vstack should reject mismatched column counts"
            );
        }
    }

    match hstack_dense(&primary_dense, &secondary_dense) {
        Some(expected) => {
            let actual_result = hstack(&blocks);
            assert!(actual_result.is_ok(), "hstack should succeed");
            let Ok(actual) = actual_result else {
                return;
            };
            let actual_coo_result = actual.to_coo();
            assert!(actual_coo_result.is_ok(), "hstack result should convert to COO");
            let Ok(actual_coo) = actual_coo_result else {
                return;
            };
            assert_dense_eq(&dense_from_coo(&actual_coo), &expected, "hstack");
        }
        None => {
            assert!(
                matches!(hstack(&blocks), Err(SparseError::IncompatibleShape { .. })),
                "hstack should reject mismatched row counts"
            );
        }
    }
});
