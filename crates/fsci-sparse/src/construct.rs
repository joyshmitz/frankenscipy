use std::collections::HashSet;

use crate::formats::{CooMatrix, CsrMatrix, Shape2D, SparseError, SparseResult};
use crate::ops::FormatConvertible;

pub fn eye(size: usize) -> SparseResult<CsrMatrix> {
    let shape = Shape2D::new(size, size);
    let data = vec![1.0; size];
    let rows: Vec<usize> = (0..size).collect();
    let cols = rows.clone();
    let coo = CooMatrix::from_triplets(shape, data, rows, cols, false)?;
    coo.to_csr()
}

pub fn diags(
    diagonals: &[Vec<f64>],
    offsets: &[isize],
    shape: Option<Shape2D>,
) -> SparseResult<CsrMatrix> {
    if diagonals.len() != offsets.len() {
        return Err(SparseError::InvalidArgument {
            message: "diagonals and offsets lengths must match".to_string(),
        });
    }

    let mut seen = HashSet::new();
    for &offset in offsets {
        if !seen.insert(offset) {
            return Err(SparseError::InvalidArgument {
                message: "repeated diagonal offsets are not allowed".to_string(),
            });
        }
    }

    let inferred = infer_shape(diagonals, offsets)?;
    let shape = shape.unwrap_or(inferred);

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for (diag, &offset) in diagonals.iter().zip(offsets.iter()) {
        let start_row = if offset < 0 { (-offset) as usize } else { 0 };
        let start_col = if offset > 0 { offset as usize } else { 0 };
        for (k, &value) in diag.iter().enumerate() {
            let row = start_row + k;
            let col = start_col + k;
            if row >= shape.rows || col >= shape.cols {
                return Err(SparseError::InvalidShape {
                    message: "diagonal length exceeds matrix shape bounds".to_string(),
                });
            }
            rows.push(row);
            cols.push(col);
            data.push(value);
        }
    }

    let coo = CooMatrix::from_triplets(shape, data, rows, cols, true)?;
    coo.to_csr()
}

pub fn random(shape: Shape2D, density: f64, seed: u64) -> SparseResult<CooMatrix> {
    if !(0.0..=1.0).contains(&density) {
        return Err(SparseError::InvalidArgument {
            message: "density must be in [0.0, 1.0]".to_string(),
        });
    }
    let total = shape
        .rows
        .checked_mul(shape.cols)
        .ok_or_else(|| SparseError::IndexOverflow {
            message: "rows * cols overflows usize".to_string(),
        })?;

    let mut state = seed.max(1);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for index in 0..total {
        state = xorshift64(state);
        let sample = (state as f64) / (u64::MAX as f64);
        if sample <= density {
            let row = index / shape.cols.max(1);
            let col = index % shape.cols.max(1);
            state = xorshift64(state);
            let value = ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
            rows.push(row.min(shape.rows.saturating_sub(1)));
            cols.push(col.min(shape.cols.saturating_sub(1)));
            data.push(value);
        }
    }

    CooMatrix::from_triplets(shape, data, rows, cols, true)
}

fn infer_shape(diagonals: &[Vec<f64>], offsets: &[isize]) -> SparseResult<Shape2D> {
    let mut rows = 0usize;
    let mut cols = 0usize;
    for (diag, &offset) in diagonals.iter().zip(offsets.iter()) {
        let len = diag.len();
        if offset >= 0 {
            rows = rows.max(len);
            cols = cols.max(len + offset as usize);
        } else {
            let abs = (-offset) as usize;
            rows = rows.max(len + abs);
            cols = cols.max(len);
        }
    }
    if rows == 0 && cols == 0 {
        return Err(SparseError::InvalidShape {
            message: "cannot infer shape from empty diagonals".to_string(),
        });
    }
    Ok(Shape2D::new(rows, cols))
}

/// Construct a block diagonal sparse matrix from a list of sparse matrices.
///
/// Matches `scipy.sparse.block_diag(mats)`.
pub fn block_diag(matrices: &[&CsrMatrix]) -> SparseResult<CsrMatrix> {
    if matrices.is_empty() {
        return eye(0);
    }

    let total_rows: usize = matrices.iter().map(|m| m.shape().rows).sum();
    let total_cols: usize = matrices.iter().map(|m| m.shape().cols).sum();
    let shape = Shape2D::new(total_rows, total_cols);

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let mut row_offset = 0;
    let mut col_offset = 0;

    for &mat in matrices {
        let s = mat.shape();
        let indptr = mat.indptr();
        let indices = mat.indices();
        let mat_data = mat.data();
        for i in 0..s.rows {
            for idx in indptr[i]..indptr[i + 1] {
                rows.push(row_offset + i);
                cols.push(col_offset + indices[idx]);
                data.push(mat_data[idx]);
            }
        }
        row_offset += s.rows;
        col_offset += s.cols;
    }

    let coo = CooMatrix::from_triplets(shape, data, rows, cols, false)?;
    coo.to_csr()
}

/// Construct a sparse matrix from a block layout of sparse matrices.
///
/// `blocks` is a row-major grid where `None` entries represent zero blocks.
/// All blocks in the same row must have the same number of rows,
/// and all blocks in the same column must have the same number of columns.
/// Matches `scipy.sparse.bmat(blocks)`.
pub fn bmat(blocks: &[Vec<Option<&CsrMatrix>>]) -> SparseResult<CsrMatrix> {
    if blocks.is_empty() {
        return eye(0);
    }

    let n_block_rows = blocks.len();
    let n_block_cols = blocks[0].len();
    for row in blocks {
        if row.len() != n_block_cols {
            return Err(SparseError::InvalidArgument {
                message: "all block rows must have the same number of columns".to_string(),
            });
        }
    }

    // Determine row heights and column widths
    let mut row_heights = vec![0usize; n_block_rows];
    let mut col_widths = vec![0usize; n_block_cols];

    for (i, row) in blocks.iter().enumerate() {
        for (j, block) in row.iter().enumerate() {
            if let Some(mat) = block {
                let s = mat.shape();
                if row_heights[i] == 0 {
                    row_heights[i] = s.rows;
                } else if row_heights[i] != s.rows {
                    return Err(SparseError::IncompatibleShape {
                        message: format!(
                            "block row {i}: height mismatch {} vs {}",
                            row_heights[i], s.rows
                        ),
                    });
                }
                if col_widths[j] == 0 {
                    col_widths[j] = s.cols;
                } else if col_widths[j] != s.cols {
                    return Err(SparseError::IncompatibleShape {
                        message: format!(
                            "block col {j}: width mismatch {} vs {}",
                            col_widths[j], s.cols
                        ),
                    });
                }
            }
        }
    }

    let total_rows: usize = row_heights.iter().sum();
    let total_cols: usize = col_widths.iter().sum();
    let shape = Shape2D::new(total_rows, total_cols);

    let mut all_rows = Vec::new();
    let mut all_cols = Vec::new();
    let mut all_data = Vec::new();

    let mut row_offset = 0;
    for (i, block_row) in blocks.iter().enumerate() {
        let mut col_offset = 0;
        for (j, block) in block_row.iter().enumerate() {
            if let Some(mat) = block {
                let indptr = mat.indptr();
                let indices = mat.indices();
                let mat_data = mat.data();
                for r in 0..mat.shape().rows {
                    for idx in indptr[r]..indptr[r + 1] {
                        all_rows.push(row_offset + r);
                        all_cols.push(col_offset + indices[idx]);
                        all_data.push(mat_data[idx]);
                    }
                }
            }
            col_offset += col_widths[j];
        }
        row_offset += row_heights[i];
    }

    let coo = CooMatrix::from_triplets(shape, all_data, all_rows, all_cols, false)?;
    coo.to_csr()
}

/// Stack sparse matrices vertically (row wise).
///
/// Mirrors `scipy.sparse.vstack` for sparse inputs. The Rust API returns CSR
/// directly; call `.to_coo()` when coordinate output is desired.
pub fn vstack(blocks: &[&dyn FormatConvertible]) -> SparseResult<CsrMatrix> {
    stack_sparse_blocks(blocks, StackAxis::Rows)
}

/// Stack sparse matrices horizontally (column wise).
///
/// Mirrors `scipy.sparse.hstack` for sparse inputs. The Rust API returns CSR
/// directly; call `.to_coo()` when coordinate output is desired.
pub fn hstack(blocks: &[&dyn FormatConvertible]) -> SparseResult<CsrMatrix> {
    stack_sparse_blocks(blocks, StackAxis::Cols)
}

/// Kronecker product of two sparse matrices.
///
/// For A (m×n) and B (p×q), produces an (m*p × n*q) matrix.
/// Matches `scipy.sparse.kron(A, B)`.
pub fn kron(a: &CsrMatrix, b: &CsrMatrix) -> SparseResult<CsrMatrix> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let out_rows =
        a_shape
            .rows
            .checked_mul(b_shape.rows)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "kron output rows overflow".to_string(),
            })?;
    let out_cols =
        a_shape
            .cols
            .checked_mul(b_shape.cols)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "kron output cols overflow".to_string(),
            })?;
    let shape = Shape2D::new(out_rows, out_cols);

    let a_indptr = a.indptr();
    let a_indices = a.indices();
    let a_data = a.data();
    let b_indptr = b.indptr();
    let b_indices = b.indices();
    let b_data = b.data();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for ai in 0..a_shape.rows {
        for a_idx in a_indptr[ai]..a_indptr[ai + 1] {
            let aj = a_indices[a_idx];
            let a_val = a_data[a_idx];
            for bi in 0..b_shape.rows {
                for b_idx in b_indptr[bi]..b_indptr[bi + 1] {
                    let bj = b_indices[b_idx];
                    let b_val = b_data[b_idx];
                    rows.push(ai * b_shape.rows + bi);
                    cols.push(aj * b_shape.cols + bj);
                    data.push(a_val * b_val);
                }
            }
        }
    }

    let coo = CooMatrix::from_triplets(shape, data, rows, cols, false)?;
    coo.to_csr()
}

fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StackAxis {
    Rows,
    Cols,
}

fn stack_sparse_blocks(
    blocks: &[&dyn FormatConvertible],
    axis: StackAxis,
) -> SparseResult<CsrMatrix> {
    if blocks.is_empty() {
        return Err(SparseError::InvalidArgument {
            message: format!(
                "{} requires at least one block",
                match axis {
                    StackAxis::Rows => "vstack",
                    StackAxis::Cols => "hstack",
                }
            ),
        });
    }

    let first = blocks[0].to_coo()?;
    let mut total_rows = first.shape().rows;
    let mut total_cols = first.shape().cols;
    let mut row_offset = 0usize;
    let mut col_offset = 0usize;
    let shared_extent = match axis {
        StackAxis::Rows => first.shape().cols,
        StackAxis::Cols => first.shape().rows,
    };

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    append_coo_entries(
        &first, row_offset, col_offset, &mut rows, &mut cols, &mut data,
    );

    match axis {
        StackAxis::Rows => row_offset += first.shape().rows,
        StackAxis::Cols => col_offset += first.shape().cols,
    }

    for block in blocks.iter().skip(1) {
        let coo = block.to_coo()?;
        match axis {
            StackAxis::Rows => {
                if coo.shape().cols != shared_extent {
                    return Err(SparseError::IncompatibleShape {
                        message: "vstack requires all blocks to have matching column counts"
                            .to_string(),
                    });
                }
                total_rows += coo.shape().rows;
            }
            StackAxis::Cols => {
                if coo.shape().rows != shared_extent {
                    return Err(SparseError::IncompatibleShape {
                        message: "hstack requires all blocks to have matching row counts"
                            .to_string(),
                    });
                }
                total_cols += coo.shape().cols;
            }
        }

        append_coo_entries(
            &coo, row_offset, col_offset, &mut rows, &mut cols, &mut data,
        );

        match axis {
            StackAxis::Rows => row_offset += coo.shape().rows,
            StackAxis::Cols => col_offset += coo.shape().cols,
        }
    }

    let shape = Shape2D::new(total_rows, total_cols);
    let coo = CooMatrix::from_triplets(shape, data, rows, cols, false)?;
    coo.to_csr()
}

fn append_coo_entries(
    coo: &CooMatrix,
    row_offset: usize,
    col_offset: usize,
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    data: &mut Vec<f64>,
) {
    for idx in 0..coo.nnz() {
        rows.push(row_offset + coo.row_indices()[idx]);
        cols.push(col_offset + coo.col_indices()[idx]);
        data.push(coo.data()[idx]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::FormatConvertible;

    #[test]
    fn eye_constructs_expected_identity() {
        let id = eye(4).expect("identity");
        assert_eq!(id.shape(), Shape2D::new(4, 4));
        assert_eq!(id.nnz(), 4);

        let coo = id.to_coo().expect("csr->coo");
        for idx in 0..coo.nnz() {
            assert_eq!(coo.row_indices()[idx], coo.col_indices()[idx]);
            assert!((coo.data()[idx] - 1.0).abs() <= f64::EPSILON);
        }
    }

    #[test]
    fn eye_zero_size_is_empty() {
        let id = eye(0).expect("identity");
        assert_eq!(id.shape(), Shape2D::new(0, 0));
        assert_eq!(id.nnz(), 0);
    }

    #[test]
    fn diags_rejects_length_mismatch() {
        let err = diags(&[vec![1.0]], &[0, 1], None).expect_err("length mismatch");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn diags_rejects_repeated_offsets() {
        let err = diags(&[vec![1.0], vec![2.0]], &[0, 0], None).expect_err("repeated offsets");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn diags_rejects_empty_shape_inference() {
        let err = diags(&[], &[], None).expect_err("empty inference");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn diags_infers_shape_with_positive_and_negative_offsets() {
        let csr = diags(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0]], &[1, -2], None).expect("diags");
        assert_eq!(csr.shape(), Shape2D::new(4, 4));

        let dense = dense_from_csr(&csr);
        let expected = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 2.0, 0.0],
            vec![4.0, 0.0, 0.0, 3.0],
            vec![0.0, 5.0, 0.0, 0.0],
        ];
        assert_eq!(dense, expected);
    }

    #[test]
    fn diags_honors_explicit_shape() {
        let csr = diags(
            &[vec![1.0, 2.0], vec![3.0]],
            &[0, 2],
            Some(Shape2D::new(5, 5)),
        )
        .expect("diags");
        assert_eq!(csr.shape(), Shape2D::new(5, 5));
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn diags_rejects_out_of_bounds_for_explicit_shape() {
        let err = diags(&[vec![1.0, 2.0, 3.0]], &[1], Some(Shape2D::new(3, 3)))
            .expect_err("bounds violation");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn random_rejects_density_out_of_range() {
        let shape = Shape2D::new(2, 2);
        assert!(matches!(
            random(shape, -0.1, 7),
            Err(SparseError::InvalidArgument { .. })
        ));
        assert!(matches!(
            random(shape, 1.1, 7),
            Err(SparseError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn random_rejects_shape_overflow() {
        let err = random(Shape2D::new(usize::MAX, 2), 0.5, 9).expect_err("overflow");
        assert!(matches!(err, SparseError::IndexOverflow { .. }));
    }

    #[test]
    fn random_density_zero_returns_empty_matrix() {
        let coo = random(Shape2D::new(4, 5), 0.0, 11).expect("random");
        assert_eq!(coo.shape(), Shape2D::new(4, 5));
        assert_eq!(coo.nnz(), 0);
    }

    #[test]
    fn random_density_one_fills_every_position() {
        let coo = random(Shape2D::new(3, 2), 1.0, 11).expect("random");
        assert_eq!(coo.nnz(), 6);
        for idx in 0..coo.nnz() {
            assert!(coo.row_indices()[idx] < 3);
            assert!(coo.col_indices()[idx] < 2);
            assert!((-1.0..=1.0).contains(&coo.data()[idx]));
        }
    }

    #[test]
    fn random_is_deterministic_for_same_seed() {
        let first = random(Shape2D::new(5, 4), 0.35, 12345).expect("random first");
        let second = random(Shape2D::new(5, 4), 0.35, 12345).expect("random second");
        assert_eq!(first.row_indices(), second.row_indices());
        assert_eq!(first.col_indices(), second.col_indices());
        assert_eq!(first.data(), second.data());
    }

    #[test]
    fn random_zero_dimension_returns_empty() {
        let coo = random(Shape2D::new(0, 7), 1.0, 99).expect("random");
        assert_eq!(coo.nnz(), 0);
    }

    #[test]
    fn xorshift_changes_nonzero_values() {
        assert_eq!(xorshift64(0), 0);
        assert_ne!(xorshift64(1), 1);
    }

    // ── block_diag tests ─────────────────────────────────────────────

    #[test]
    fn block_diag_two_matrices() {
        let a = eye(2).expect("eye(2)");
        let b = diags(&[vec![3.0, 4.0]], &[0], None).expect("diags");
        let result = block_diag(&[&a, &b]).expect("block_diag");
        assert_eq!(result.shape(), Shape2D::new(4, 4));
        let dense = dense_from_csr(&result);
        assert_eq!(
            dense,
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 3.0, 0.0],
                vec![0.0, 0.0, 0.0, 4.0],
            ]
        );
    }

    #[test]
    fn block_diag_empty() {
        let result = block_diag(&[]).expect("block_diag empty");
        assert_eq!(result.shape(), Shape2D::new(0, 0));
    }

    #[test]
    fn block_diag_single() {
        let a = eye(3).expect("eye(3)");
        let result = block_diag(&[&a]).expect("block_diag single");
        assert_eq!(result.shape(), Shape2D::new(3, 3));
        assert_eq!(result.nnz(), 3);
    }

    // ── bmat tests ──────────────────────────────────────────────────

    #[test]
    fn bmat_2x2_grid() {
        let a = eye(2).expect("eye(2)");
        let b = diags(&[vec![5.0, 6.0]], &[0], None).expect("diags");
        let result = bmat(&[vec![Some(&a), None], vec![None, Some(&b)]]).expect("bmat");
        assert_eq!(result.shape(), Shape2D::new(4, 4));
        let dense = dense_from_csr(&result);
        assert_eq!(
            dense,
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 5.0, 0.0],
                vec![0.0, 0.0, 0.0, 6.0],
            ]
        );
    }

    #[test]
    fn bmat_with_off_diagonal_blocks() {
        let a = eye(2).expect("eye(2)");
        let result = bmat(&[vec![Some(&a), Some(&a)], vec![Some(&a), Some(&a)]]).expect("bmat");
        assert_eq!(result.shape(), Shape2D::new(4, 4));
        let dense = dense_from_csr(&result);
        assert_eq!(
            dense,
            vec![
                vec![1.0, 0.0, 1.0, 0.0],
                vec![0.0, 1.0, 0.0, 1.0],
                vec![1.0, 0.0, 1.0, 0.0],
                vec![0.0, 1.0, 0.0, 1.0],
            ]
        );
    }

    #[test]
    fn bmat_empty() {
        let result = bmat(&[]).expect("bmat empty");
        assert_eq!(result.shape(), Shape2D::new(0, 0));
    }

    #[test]
    fn bmat_ragged_block_rows_rejected() {
        let a = eye(2).expect("eye(2)");
        let err = bmat(&[vec![Some(&a), Some(&a)], vec![Some(&a)]]).expect_err("ragged");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn vstack_accepts_mixed_sparse_formats() {
        let top = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo");
        let bottom = CooMatrix::from_triplets(
            Shape2D::new(1, 2),
            vec![5.0, 6.0],
            vec![0, 0],
            vec![0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");

        let result = vstack(&[&top, &bottom]).expect("vstack");
        assert_eq!(result.shape(), Shape2D::new(3, 2));
        assert_eq!(
            dense_from_csr(&result),
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
        );
    }

    #[test]
    fn hstack_accepts_mixed_sparse_formats() {
        let left = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo");
        let right = CooMatrix::from_triplets(
            Shape2D::new(2, 1),
            vec![5.0, 6.0],
            vec![0, 1],
            vec![0, 0],
            false,
        )
        .expect("coo")
        .to_csc()
        .expect("csc");

        let result = hstack(&[&left, &right]).expect("hstack");
        assert_eq!(result.shape(), Shape2D::new(2, 3));
        assert_eq!(
            dense_from_csr(&result),
            vec![vec![1.0, 2.0, 5.0], vec![3.0, 4.0, 6.0]]
        );
    }

    #[test]
    fn vstack_rejects_empty_input() {
        let err = vstack(&[]).expect_err("empty vstack");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn vstack_rejects_mismatched_column_counts() {
        let left = eye(2).expect("eye");
        let right = eye(3).expect("eye");
        let err = vstack(&[&left, &right]).expect_err("mismatched cols");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn hstack_rejects_mismatched_row_counts() {
        let left = eye(2).expect("eye");
        let right = eye(3).expect("eye");
        let err = hstack(&[&left, &right]).expect_err("mismatched rows");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    // ── kron tests ──────────────────────────────────────────────────

    #[test]
    fn kron_identity_times_identity() {
        let i2 = eye(2).expect("eye(2)");
        let result = kron(&i2, &i2).expect("kron");
        assert_eq!(result.shape(), Shape2D::new(4, 4));
        // I2 ⊗ I2 = I4
        let dense = dense_from_csr(&result);
        for (i, row) in dense.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((val - expected).abs() < 1e-14, "kron I2⊗I2 at [{i}][{j}]");
            }
        }
    }

    #[test]
    fn kron_scalar_times_matrix() {
        // scalar [[2]] ⊗ [[1, 0], [0, 3]] = [[2, 0], [0, 6]]
        let scalar =
            CooMatrix::from_triplets(Shape2D::new(1, 1), vec![2.0], vec![0], vec![0], false)
                .expect("coo")
                .to_csr()
                .expect("csr");
        let b = diags(&[vec![1.0, 3.0]], &[0], None).expect("diags");
        let result = kron(&scalar, &b).expect("kron");
        assert_eq!(result.shape(), Shape2D::new(2, 2));
        let dense = dense_from_csr(&result);
        assert_eq!(dense, vec![vec![2.0, 0.0], vec![0.0, 6.0]]);
    }

    #[test]
    fn kron_known_result() {
        // [[1, 2], [3, 4]] ⊗ [[0, 5], [6, 7]]
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![5.0, 6.0, 7.0],
            vec![0, 1, 1],
            vec![1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = kron(&a, &b).expect("kron");
        assert_eq!(result.shape(), Shape2D::new(4, 4));
        let dense = dense_from_csr(&result);
        // Expected: [[0, 5, 0, 10], [6, 7, 12, 14], [0, 15, 0, 20], [18, 21, 24, 28]]
        let expected = [
            [0.0, 5.0, 0.0, 10.0],
            [6.0, 7.0, 12.0, 14.0],
            [0.0, 15.0, 0.0, 20.0],
            [18.0, 21.0, 24.0, 28.0],
        ];
        for (i, (d_row, e_row)) in dense.iter().zip(expected.iter()).enumerate() {
            for (j, (&d_val, &e_val)) in d_row.iter().zip(e_row.iter()).enumerate() {
                assert!(
                    (d_val - e_val).abs() < 1e-12,
                    "kron at [{i}][{j}]: {d_val} vs {e_val}"
                );
            }
        }
    }

    fn dense_from_csr(csr: &CsrMatrix) -> Vec<Vec<f64>> {
        let shape = csr.shape();
        let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
        for (row, row_dense) in dense.iter_mut().enumerate().take(shape.rows) {
            for idx in csr.indptr()[row]..csr.indptr()[row + 1] {
                row_dense[csr.indices()[idx]] += csr.data()[idx];
            }
        }
        dense
    }
}
