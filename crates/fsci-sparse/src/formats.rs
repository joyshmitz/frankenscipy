use fsci_runtime::RuntimeMode;
use nalgebra::sparse::CsMatrix as NalgebraCsMatrix;
use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub type SparseResult<T> = Result<T, SparseError>;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SparseError {
    #[error("invalid shape: {message}")]
    InvalidShape { message: String },
    #[error("invalid sparse structure: {message}")]
    InvalidSparseStructure { message: String },
    #[error("index {index} out of bounds for {axis} with bound {bound}")]
    IndexOutOfBounds {
        axis: &'static str,
        index: usize,
        bound: usize,
    },
    #[error("incompatible shape: {message}")]
    IncompatibleShape { message: String },
    #[error("invalid argument: {message}")]
    InvalidArgument { message: String },
    #[error("index overflow: {message}")]
    IndexOverflow { message: String },
    #[error("unsupported operation: {feature}")]
    Unsupported { feature: String },
    #[error("non-finite input: {message}")]
    NonFiniteInput { message: String },
    #[error("singular matrix: {message}")]
    SingularMatrix { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    Csr,
    Csc,
    Coo,
    Dia,
    Dok,
    Lil,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape2D {
    pub rows: usize,
    pub cols: usize,
}

impl Shape2D {
    #[must_use]
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    #[must_use]
    pub const fn is_square(self) -> bool {
        self.rows == self.cols
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalMeta {
    pub sorted_indices: bool,
    pub deduplicated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstructionLogEntry {
    pub timestamp: String,
    pub operation_id: String,
    pub format: SparseFormat,
    pub shape: Shape2D,
    pub nnz: usize,
    pub mode: RuntimeMode,
    pub validation_result: String,
}

impl ConstructionLogEntry {
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"timestamp\":\"{}\",\"operation_id\":\"{}\",\"format\":\"{}\",\"shape\":[{},{}],\"nnz\":{},\"mode\":\"{}\",\"validation_result\":\"{}\"}}",
            self.timestamp,
            self.operation_id,
            format_label(self.format),
            self.shape.rows,
            self.shape.cols,
            self.nnz,
            mode_label(self.mode),
            self.validation_result
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) data: Vec<f64>,
    pub(crate) indices: Vec<usize>,
    pub(crate) indptr: Vec<usize>,
    pub(crate) canonical: CanonicalMeta,
}

impl CsrMatrix {
    pub fn from_components(
        shape: Shape2D,
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        canonicalize: bool,
    ) -> SparseResult<Self> {
        let canonical = validate_compressed(
            "CSR",
            shape.rows,
            shape.cols,
            &data,
            &indices,
            &indptr,
            canonicalize,
        )?;
        Ok(Self {
            shape,
            data,
            indices,
            indptr,
            canonical,
        })
    }

    /// Construct a CSR matrix without running `validate_compressed`, bypassing checks.
    pub(crate) fn from_components_unchecked(
        shape: Shape2D,
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
    ) -> Self {
        Self {
            shape,
            data,
            indices,
            indptr,
            canonical: CanonicalMeta {
                sorted_indices: false,
                deduplicated: false,
            },
        }
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn canonical_meta(&self) -> CanonicalMeta {
        self.canonical
    }

    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    #[must_use]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Csr,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CscMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) data: Vec<f64>,
    pub(crate) indices: Vec<usize>,
    pub(crate) indptr: Vec<usize>,
    pub(crate) canonical: CanonicalMeta,
}

impl CscMatrix {
    pub fn from_components(
        shape: Shape2D,
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        canonicalize: bool,
    ) -> SparseResult<Self> {
        let canonical = validate_compressed(
            "CSC",
            shape.cols,
            shape.rows,
            &data,
            &indices,
            &indptr,
            canonicalize,
        )?;
        Ok(Self {
            shape,
            data,
            indices,
            indptr,
            canonical,
        })
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn canonical_meta(&self) -> CanonicalMeta {
        self.canonical
    }

    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    #[must_use]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Csc,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) data: Vec<f64>,
    pub(crate) row_indices: Vec<usize>,
    pub(crate) col_indices: Vec<usize>,
}

impl CooMatrix {
    pub fn from_triplets(
        shape: Shape2D,
        data: Vec<f64>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        sum_duplicates: bool,
    ) -> SparseResult<Self> {
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(SparseError::IncompatibleShape {
                message: "COO data/row/col lengths must match".to_string(),
            });
        }
        for &row in &row_indices {
            if row >= shape.rows {
                return Err(SparseError::IndexOutOfBounds {
                    axis: "row",
                    index: row,
                    bound: shape.rows,
                });
            }
        }
        for &col in &col_indices {
            if col >= shape.cols {
                return Err(SparseError::IndexOutOfBounds {
                    axis: "col",
                    index: col,
                    bound: shape.cols,
                });
            }
        }

        if !sum_duplicates {
            return Ok(Self {
                shape,
                data,
                row_indices,
                col_indices,
            });
        }

        let mut triplets: Vec<(usize, usize, f64)> = row_indices
            .into_iter()
            .zip(col_indices)
            .zip(data)
            .map(|((r, c), v)| (r, c, v))
            .collect();
        triplets.sort_unstable_by_key(|(r, c, _)| (*r, *c));

        let mut merged_rows = Vec::with_capacity(triplets.len());
        let mut merged_cols = Vec::with_capacity(triplets.len());
        let mut merged_data = Vec::with_capacity(triplets.len());

        for (row, col, value) in triplets {
            let is_same_as_last = match (merged_rows.last(), merged_cols.last()) {
                (Some(&last_row), Some(&last_col)) => last_row == row && last_col == col,
                _ => false,
            };

            if is_same_as_last {
                if let Some(last) = merged_data.last_mut() {
                    *last += value;
                }
            } else {
                merged_rows.push(row);
                merged_cols.push(col);
                merged_data.push(value);
            }
        }

        Ok(Self {
            shape,
            data: merged_data,
            row_indices: merged_rows,
            col_indices: merged_cols,
        })
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[must_use]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    #[must_use]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Coo,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LilMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) row_indices: Vec<Vec<usize>>,
    pub(crate) row_data: Vec<Vec<f64>>,
}

impl LilMatrix {
    #[must_use]
    pub fn new(shape: Shape2D) -> Self {
        Self {
            shape,
            row_indices: vec![Vec::new(); shape.rows],
            row_data: vec![Vec::new(); shape.rows],
        }
    }

    pub fn from_rows(
        shape: Shape2D,
        row_indices: Vec<Vec<usize>>,
        row_data: Vec<Vec<f64>>,
    ) -> SparseResult<Self> {
        if row_indices.len() != shape.rows || row_data.len() != shape.rows {
            return Err(SparseError::InvalidShape {
                message: "LIL outer row containers must match shape.rows".to_string(),
            });
        }

        let mut canonical_indices = Vec::with_capacity(shape.rows);
        let mut canonical_data = Vec::with_capacity(shape.rows);

        for (row_cols, row_values) in row_indices.into_iter().zip(row_data) {
            let (cols, values) = canonicalize_lil_row(row_cols, row_values, shape.cols)?;
            canonical_indices.push(cols);
            canonical_data.push(values);
        }

        Ok(Self {
            shape,
            row_indices: canonical_indices,
            row_data: canonical_data,
        })
    }

    pub fn from_triplets(
        shape: Shape2D,
        data: Vec<f64>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
    ) -> SparseResult<Self> {
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(SparseError::IncompatibleShape {
                message: "LIL data/row/col lengths must match".to_string(),
            });
        }

        let mut row_maps = vec![BTreeMap::new(); shape.rows];
        for ((row, col), value) in row_indices.into_iter().zip(col_indices).zip(data) {
            validate_coordinate(shape, row, col)?;
            *row_maps[row].entry(col).or_insert(0.0) += value;
        }

        let mut canonical_indices = Vec::with_capacity(shape.rows);
        let mut canonical_data = Vec::with_capacity(shape.rows);
        for row_map in row_maps {
            let mut cols = Vec::with_capacity(row_map.len());
            let mut values = Vec::with_capacity(row_map.len());
            for (col, value) in row_map {
                cols.push(col);
                values.push(value);
            }
            canonical_indices.push(cols);
            canonical_data.push(values);
        }

        Ok(Self {
            shape,
            row_indices: canonical_indices,
            row_data: canonical_data,
        })
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.row_data.iter().map(Vec::len).sum()
    }

    #[must_use]
    pub fn row_indices(&self) -> &[Vec<usize>] {
        &self.row_indices
    }

    #[must_use]
    pub fn row_data(&self) -> &[Vec<f64>] {
        &self.row_data
    }

    pub fn get(&self, row: usize, col: usize) -> SparseResult<f64> {
        validate_coordinate(self.shape, row, col)?;
        let row_cols = &self.row_indices[row];
        Ok(match row_cols.binary_search(&col) {
            Ok(idx) => self.row_data[row][idx],
            Err(_) => 0.0,
        })
    }

    pub fn contains(&self, row: usize, col: usize) -> SparseResult<bool> {
        validate_coordinate(self.shape, row, col)?;
        Ok(self.row_indices[row].binary_search(&col).is_ok())
    }

    pub fn insert(&mut self, row: usize, col: usize, value: f64) -> SparseResult<Option<f64>> {
        validate_coordinate(self.shape, row, col)?;
        match self.row_indices[row].binary_search(&col) {
            Ok(idx) => {
                let previous = self.row_data[row][idx];
                self.row_data[row][idx] = value;
                Ok(Some(previous))
            }
            Err(idx) => {
                self.row_indices[row].insert(idx, col);
                self.row_data[row].insert(idx, value);
                Ok(None)
            }
        }
    }

    pub fn remove(&mut self, row: usize, col: usize) -> SparseResult<Option<f64>> {
        validate_coordinate(self.shape, row, col)?;
        Ok(match self.row_indices[row].binary_search(&col) {
            Ok(idx) => {
                self.row_indices[row].remove(idx);
                Some(self.row_data[row].remove(idx))
            }
            Err(_) => None,
        })
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Lil,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }

    pub fn to_coo(&self) -> SparseResult<CooMatrix> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        for (row, (row_cols, row_values)) in self
            .row_indices
            .iter()
            .zip(self.row_data.iter())
            .enumerate()
        {
            for (&col, &value) in row_cols.iter().zip(row_values.iter()) {
                rows.push(row);
                cols.push(col);
                data.push(value);
            }
        }

        CooMatrix::from_triplets(self.shape, data, rows, cols, false)
    }

    pub fn to_csr(&self) -> SparseResult<CsrMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csr()
    }

    pub fn to_csc(&self) -> SparseResult<CscMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csc()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DokMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) entries: BTreeMap<(usize, usize), f64>,
}

impl DokMatrix {
    #[must_use]
    pub fn new(shape: Shape2D) -> Self {
        Self {
            shape,
            entries: BTreeMap::new(),
        }
    }

    pub fn from_triplets(
        shape: Shape2D,
        data: Vec<f64>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
    ) -> SparseResult<Self> {
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(SparseError::IncompatibleShape {
                message: "DOK data/row/col lengths must match".to_string(),
            });
        }

        let mut entries = BTreeMap::new();
        for ((row, col), value) in row_indices.into_iter().zip(col_indices).zip(data) {
            validate_coordinate(shape, row, col)?;
            let key = (row, col);
            let updated = entries.get(&key).copied().unwrap_or(0.0) + value;
            if updated == 0.0 {
                entries.remove(&key);
            } else {
                entries.insert(key, updated);
            }
        }

        Ok(Self { shape, entries })
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn entries(&self) -> &BTreeMap<(usize, usize), f64> {
        &self.entries
    }

    pub fn get(&self, row: usize, col: usize) -> SparseResult<f64> {
        validate_coordinate(self.shape, row, col)?;
        Ok(self.entries.get(&(row, col)).copied().unwrap_or(0.0))
    }

    pub fn contains(&self, row: usize, col: usize) -> SparseResult<bool> {
        validate_coordinate(self.shape, row, col)?;
        Ok(self.entries.contains_key(&(row, col)))
    }

    pub fn insert(&mut self, row: usize, col: usize, value: f64) -> SparseResult<Option<f64>> {
        validate_coordinate(self.shape, row, col)?;
        let key = (row, col);
        let previous = self.entries.get(&key).copied();
        if value == 0.0 {
            self.entries.remove(&key);
        } else {
            self.entries.insert(key, value);
        }
        Ok(previous)
    }

    pub fn remove(&mut self, row: usize, col: usize) -> SparseResult<Option<f64>> {
        validate_coordinate(self.shape, row, col)?;
        Ok(self.entries.remove(&(row, col)))
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Dok,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }

    pub fn to_coo(&self) -> SparseResult<CooMatrix> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        for (&(row, col), &value) in &self.entries {
            rows.push(row);
            cols.push(col);
            data.push(value);
        }

        CooMatrix::from_triplets(self.shape, data, rows, cols, false)
    }

    pub fn to_csr(&self) -> SparseResult<CsrMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csr()
    }

    pub fn to_csc(&self) -> SparseResult<CscMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csc()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiaMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) offsets: Vec<isize>,
    pub(crate) data: Vec<Vec<f64>>,
}

impl DiaMatrix {
    pub fn from_diagonals(
        shape: Shape2D,
        offsets: Vec<isize>,
        data: Vec<Vec<f64>>,
    ) -> SparseResult<Self> {
        if offsets.len() != data.len() {
            return Err(SparseError::IncompatibleShape {
                message: "DIA offsets and data lengths must match".to_string(),
            });
        }

        let mut seen = BTreeSet::new();
        for (&offset, _diagonal) in offsets.iter().zip(data.iter()) {
            if !seen.insert(offset) {
                return Err(SparseError::InvalidArgument {
                    message: "DIA offsets must be unique".to_string(),
                });
            }

            let expected_len = diagonal_len(shape, offset);
            if expected_len == 0 {
                return Err(SparseError::InvalidArgument {
                    message: format!("DIA offset {offset} is out of bounds for matrix shape"),
                });
            }
            if _diagonal.len() != expected_len {
                return Err(SparseError::IncompatibleShape {
                    message: format!(
                        "DIA diagonal for offset {offset} has length {}, expected {expected_len}",
                        _diagonal.len()
                    ),
                });
            }
        }

        Ok(Self {
            shape,
            offsets,
            data,
        })
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        let mut count = 0;
        for diag in &self.data {
            for &val in diag {
                if val != 0.0 {
                    count += 1;
                }
            }
        }
        count
    }

    #[must_use]
    pub fn offsets(&self) -> &[isize] {
        &self.offsets
    }

    #[must_use]
    pub fn data(&self) -> &[Vec<f64>] {
        &self.data
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Dia,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }

    pub fn new(shape: Shape2D, data: Vec<Vec<f64>>, offsets: Vec<isize>) -> SparseResult<Self> {
        if data.len() != offsets.len() {
            return Err(SparseError::InvalidArgument {
                message: "data and offsets must have same length".to_string(),
            });
        }
        let mut exact_data = Vec::with_capacity(data.len());
        for (k, &off) in offsets.iter().enumerate() {
            let expected_len = diagonal_len(shape, off);
            let mut diag = Vec::with_capacity(expected_len);
            let mut r = if off < 0 { (-off) as usize } else { 0 };
            let mut c = if off > 0 { off as usize } else { 0 };
            while r < shape.rows && c < shape.cols {
                let val = if c < data[k].len() { data[k][c] } else { 0.0 };
                diag.push(val);
                r += 1;
                c += 1;
            }
            exact_data.push(diag);
        }
        Self::from_diagonals(shape, offsets, exact_data)
    }

    pub fn num_diags(&self) -> usize {
        self.offsets.len()
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        let offset_needed = col as isize - row as isize;
        for (k, &off) in self.offsets.iter().enumerate() {
            if off == offset_needed {
                let idx = if off >= 0 { row } else { col };
                if idx < self.data[k].len() {
                    return self.data[k][idx];
                }
            }
        }
        0.0
    }

    pub fn matvec(&self, x: &[f64]) -> SparseResult<Vec<f64>> {
        let (rows, cols) = (self.shape.rows, self.shape.cols);
        if x.len() != cols {
            return Err(SparseError::IncompatibleShape {
                message: format!("x length {} != cols {}", x.len(), cols),
            });
        }

        let mut y = vec![0.0; rows];
        for (k, &off) in self.offsets.iter().enumerate() {
            let diag = &self.data[k];
            for (i, yi) in y.iter_mut().enumerate() {
                let j = i as isize + off;
                if j >= 0 && (j as usize) < cols {
                    let j = j as usize;
                    let idx = if off >= 0 { i } else { j };
                    if idx < diag.len() {
                        *yi += diag[idx] * x[j];
                    }
                }
            }
        }
        Ok(y)
    }

    pub fn to_coo(&self) -> SparseResult<CooMatrix> {
        let (rows, cols) = (self.shape.rows, self.shape.cols);
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut data = Vec::new();

        for (k, &off) in self.offsets.iter().enumerate() {
            let diag = &self.data[k];
            for i in 0..rows {
                let j = i as isize + off;
                if j >= 0 && (j as usize) < cols {
                    let j = j as usize;
                    let idx = if off >= 0 { i } else { j };
                    if idx < diag.len() {
                        let val = diag[idx];
                        if val != 0.0 {
                            row_indices.push(i);
                            col_indices.push(j);
                            data.push(val);
                        }
                    }
                }
            }
        }

        CooMatrix::from_triplets(self.shape, data, row_indices, col_indices, true)
    }

    pub fn to_csr(&self) -> SparseResult<CsrMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csr()
    }

    pub fn to_csc(&self) -> SparseResult<CscMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csc()
    }
}

pub trait NalgebraBridge {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64>;
    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self>
    where
        Self: Sized;
}

impl NalgebraBridge for CooMatrix {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64> {
        NalgebraCsMatrix::from_triplet(
            self.shape.rows,
            self.shape.cols,
            &self.row_indices,
            &self.col_indices,
            &self.data,
        )
    }

    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self> {
        let (rows, cols) = matrix.shape();
        let dense: nalgebra::DMatrix<f64> = matrix.clone().into();
        let mut triplet_rows = Vec::new();
        let mut triplet_cols = Vec::new();
        let mut triplet_data = Vec::new();

        for row in 0..rows {
            for col in 0..cols {
                let value = dense[(row, col)];
                if value != 0.0 {
                    triplet_rows.push(row);
                    triplet_cols.push(col);
                    triplet_data.push(value);
                }
            }
        }

        Self::from_triplets(
            Shape2D::new(rows, cols),
            triplet_data,
            triplet_rows,
            triplet_cols,
            false,
        )
    }
}

impl NalgebraBridge for DokMatrix {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut vals = Vec::with_capacity(self.nnz());

        for (&(row, col), &value) in &self.entries {
            rows.push(row);
            cols.push(col);
            vals.push(value);
        }

        NalgebraCsMatrix::from_triplet(self.shape.rows, self.shape.cols, &rows, &cols, &vals)
    }

    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self> {
        let coo = CooMatrix::from_nalgebra_cs(matrix)?;
        Self::from_triplets(coo.shape, coo.data, coo.row_indices, coo.col_indices)
    }
}

impl NalgebraBridge for LilMatrix {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut vals = Vec::with_capacity(self.nnz());

        for (row, (row_cols, row_values)) in self
            .row_indices
            .iter()
            .zip(self.row_data.iter())
            .enumerate()
        {
            for (&col, &value) in row_cols.iter().zip(row_values.iter()) {
                rows.push(row);
                cols.push(col);
                vals.push(value);
            }
        }

        NalgebraCsMatrix::from_triplet(self.shape.rows, self.shape.cols, &rows, &cols, &vals)
    }

    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self> {
        let coo = CooMatrix::from_nalgebra_cs(matrix)?;
        Self::from_triplets(coo.shape, coo.data, coo.row_indices, coo.col_indices)
    }
}

impl NalgebraBridge for CsrMatrix {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut vals = Vec::with_capacity(self.nnz());

        for row in 0..self.shape.rows {
            for idx in self.indptr[row]..self.indptr[row + 1] {
                rows.push(row);
                cols.push(self.indices[idx]);
                vals.push(self.data[idx]);
            }
        }

        NalgebraCsMatrix::from_triplet(self.shape.rows, self.shape.cols, &rows, &cols, &vals)
    }

    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self> {
        let coo = CooMatrix::from_nalgebra_cs(matrix)?;
        let mut triplets: Vec<(usize, usize, f64)> = coo
            .row_indices
            .iter()
            .copied()
            .zip(coo.col_indices.iter().copied())
            .zip(coo.data.iter().copied())
            .map(|((r, c), v)| (r, c, v))
            .collect();
        triplets.sort_unstable_by_key(|(r, c, _)| (*r, *c));

        let mut dedup = Vec::with_capacity(triplets.len());
        for (r, c, v) in triplets {
            let should_merge = match dedup.last_mut() {
                Some((lr, lc, lv)) if *lr == r && *lc == c => {
                    *lv += v;
                    true
                }
                _ => false,
            };
            if !should_merge {
                dedup.push((r, c, v));
            }
        }

        let shape = coo.shape;
        let mut indptr = vec![0usize; shape.rows + 1];
        for (r, _, _) in &dedup {
            indptr[r + 1] += 1;
        }
        for row in 0..shape.rows {
            indptr[row + 1] += indptr[row];
        }

        let mut indices = Vec::with_capacity(dedup.len());
        let mut data = Vec::with_capacity(dedup.len());
        for (_, c, v) in dedup {
            indices.push(c);
            data.push(v);
        }

        Self::from_components(shape, data, indices, indptr, true)
    }
}

impl NalgebraBridge for CscMatrix {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut vals = Vec::with_capacity(self.nnz());

        for col in 0..self.shape.cols {
            for idx in self.indptr[col]..self.indptr[col + 1] {
                rows.push(self.indices[idx]);
                cols.push(col);
                vals.push(self.data[idx]);
            }
        }

        NalgebraCsMatrix::from_triplet(self.shape.rows, self.shape.cols, &rows, &cols, &vals)
    }

    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self> {
        let coo = CooMatrix::from_nalgebra_cs(matrix)?;
        let mut triplets: Vec<(usize, usize, f64)> = coo
            .row_indices
            .iter()
            .copied()
            .zip(coo.col_indices.iter().copied())
            .zip(coo.data.iter().copied())
            .map(|((r, c), v)| (r, c, v))
            .collect();
        triplets.sort_unstable_by_key(|(r, c, _)| (*c, *r));

        let mut dedup = Vec::with_capacity(triplets.len());
        for (r, c, v) in triplets {
            let should_merge = match dedup.last_mut() {
                Some((lr, lc, lv)) if *lr == r && *lc == c => {
                    *lv += v;
                    true
                }
                _ => false,
            };
            if !should_merge {
                dedup.push((r, c, v));
            }
        }

        let shape = coo.shape;
        let mut indptr = vec![0usize; shape.cols + 1];
        for (_, c, _) in &dedup {
            indptr[c + 1] += 1;
        }
        for col in 0..shape.cols {
            indptr[col + 1] += indptr[col];
        }

        let mut indices = Vec::with_capacity(dedup.len());
        let mut data = Vec::with_capacity(dedup.len());
        for (r, _, v) in dedup {
            indices.push(r);
            data.push(v);
        }

        Self::from_components(shape, data, indices, indptr, true)
    }
}

fn validate_compressed(
    label: &str,
    major_len: usize,
    minor_len: usize,
    data: &[f64],
    indices: &[usize],
    indptr: &[usize],
    canonicalize: bool,
) -> SparseResult<CanonicalMeta> {
    if data.len() != indices.len() {
        return Err(SparseError::IncompatibleShape {
            message: format!("{label} data and indices lengths differ"),
        });
    }
    if indptr.len() != major_len + 1 {
        return Err(SparseError::InvalidShape {
            message: format!("{label} indptr length must be major_len + 1"),
        });
    }
    if !indptr.windows(2).all(|w| w[0] <= w[1]) {
        return Err(SparseError::InvalidSparseStructure {
            message: format!("{label} indptr must be monotone non-decreasing"),
        });
    }
    if indptr.first().copied().unwrap_or_default() != 0 || indptr[major_len] != data.len() {
        return Err(SparseError::InvalidSparseStructure {
            message: format!(
                "{label} pointer endpoints must satisfy indptr[0]=0 and indptr[last]=nnz"
            ),
        });
    }
    for &idx in indices {
        if idx >= minor_len {
            return Err(SparseError::IndexOutOfBounds {
                axis: "minor",
                index: idx,
                bound: minor_len,
            });
        }
    }

    let (sorted, deduplicated) = detect_canonical(indptr, indices);
    Ok(CanonicalMeta {
        sorted_indices: canonicalize || sorted,
        deduplicated: canonicalize || deduplicated,
    })
}

fn validate_coordinate(shape: Shape2D, row: usize, col: usize) -> SparseResult<()> {
    if row >= shape.rows {
        return Err(SparseError::IndexOutOfBounds {
            axis: "row",
            index: row,
            bound: shape.rows,
        });
    }
    if col >= shape.cols {
        return Err(SparseError::IndexOutOfBounds {
            axis: "col",
            index: col,
            bound: shape.cols,
        });
    }
    Ok(())
}

fn canonicalize_lil_row(
    row_indices: Vec<usize>,
    row_data: Vec<f64>,
    col_bound: usize,
) -> SparseResult<(Vec<usize>, Vec<f64>)> {
    if row_indices.len() != row_data.len() {
        return Err(SparseError::IncompatibleShape {
            message: "LIL row indices/data lengths must match".to_string(),
        });
    }

    let mut pairs = Vec::with_capacity(row_indices.len());
    for (col, value) in row_indices.into_iter().zip(row_data) {
        if col >= col_bound {
            return Err(SparseError::IndexOutOfBounds {
                axis: "col",
                index: col,
                bound: col_bound,
            });
        }
        pairs.push((col, value));
    }

    pairs.sort_unstable_by_key(|(col, _)| *col);

    let mut canonical_indices = Vec::with_capacity(pairs.len());
    let mut canonical_data = Vec::with_capacity(pairs.len());
    for (col, value) in pairs {
        match canonical_indices.last().copied() {
            Some(last_col) if last_col == col => {
                if let Some(last_value) = canonical_data.last_mut() {
                    *last_value += value;
                }
            }
            _ => {
                canonical_indices.push(col);
                canonical_data.push(value);
            }
        }
    }

    Ok((canonical_indices, canonical_data))
}

fn diagonal_len(shape: Shape2D, offset: isize) -> usize {
    let start_row = if offset < 0 { (-offset) as usize } else { 0 };
    let start_col = if offset > 0 { offset as usize } else { 0 };
    shape
        .rows
        .saturating_sub(start_row)
        .min(shape.cols.saturating_sub(start_col))
}

fn detect_canonical(indptr: &[usize], indices: &[usize]) -> (bool, bool) {
    let mut sorted = true;
    let mut deduplicated = true;

    for window in indptr.windows(2) {
        let start = window[0];
        let end = window[1];
        let mut prev: Option<usize> = None;
        for &idx in &indices[start..end] {
            if let Some(prev_idx) = prev {
                if idx < prev_idx {
                    sorted = false;
                }
                if idx == prev_idx {
                    deduplicated = false;
                }
            }
            prev = Some(idx);
        }
    }

    (sorted, deduplicated)
}

fn now_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:09}Z", now.as_secs(), now.subsec_nanos())
}

fn format_label(format: SparseFormat) -> &'static str {
    match format {
        SparseFormat::Csr => "csr",
        SparseFormat::Csc => "csc",
        SparseFormat::Coo => "coo",
        SparseFormat::Dia => "dia",
        SparseFormat::Dok => "dok",
        SparseFormat::Lil => "lil",
    }
}

fn mode_label(mode: RuntimeMode) -> &'static str {
    match mode {
        RuntimeMode::Strict => "strict",
        RuntimeMode::Hardened => "hardened",
    }
}
