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

fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}
