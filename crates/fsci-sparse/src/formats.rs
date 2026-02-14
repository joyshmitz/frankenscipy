use fsci_runtime::RuntimeMode;
use nalgebra::sparse::CsMatrix as NalgebraCsMatrix;
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
    }
}

fn mode_label(mode: RuntimeMode) -> &'static str {
    match mode {
        RuntimeMode::Strict => "strict",
        RuntimeMode::Hardened => "hardened",
    }
}
