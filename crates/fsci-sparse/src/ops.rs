use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;

use crate::formats::{
    CooMatrix, CscMatrix, CsrMatrix, Shape2D, SparseError, SparseFormat, SparseResult,
};

pub trait FormatConvertible {
    fn to_csr(&self) -> SparseResult<CsrMatrix>;
    fn to_csc(&self) -> SparseResult<CscMatrix>;
    fn to_coo(&self) -> SparseResult<CooMatrix>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionLogEntry {
    pub timestamp: String,
    pub operation_id: String,
    pub from_format: SparseFormat,
    pub to_format: SparseFormat,
    pub nnz_before: usize,
    pub nnz_after: usize,
}

impl ConversionLogEntry {
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"timestamp\":\"{}\",\"operation_id\":\"{}\",\"from_format\":\"{}\",\"to_format\":\"{}\",\"nnz_before\":{},\"nnz_after\":{}}}",
            self.timestamp,
            self.operation_id,
            format_label(self.from_format),
            format_label(self.to_format),
            self.nnz_before,
            self.nnz_after
        )
    }
}

impl FormatConvertible for CooMatrix {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        let mut triplets = canonical_triplets(self);
        triplets.sort_unstable_by_key(|(r, c, _)| (*r, *c));
        let (data, indices, indptr) = compress_triplets(self.shape(), &triplets, false);
        CsrMatrix::from_components(self.shape(), data, indices, indptr, true)
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        let mut triplets = canonical_triplets(self);
        triplets.sort_unstable_by_key(|(r, c, _)| (*c, *r));
        let (data, indices, indptr) = compress_triplets(self.shape(), &triplets, true);
        CscMatrix::from_components(self.shape(), data, indices, indptr, true)
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        Ok(self.clone())
    }
}

impl FormatConvertible for CsrMatrix {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        Ok(self.clone())
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        self.to_coo()?.to_csc()
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());
        for row in 0..self.shape().rows {
            for idx in self.indptr()[row]..self.indptr()[row + 1] {
                rows.push(row);
                cols.push(self.indices()[idx]);
                data.push(self.data()[idx]);
            }
        }
        CooMatrix::from_triplets(self.shape(), data, rows, cols, false)
    }
}

impl FormatConvertible for CscMatrix {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        self.to_coo()?.to_csr()
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        Ok(self.clone())
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());
        for col in 0..self.shape().cols {
            for idx in self.indptr()[col]..self.indptr()[col + 1] {
                rows.push(self.indices()[idx]);
                cols.push(col);
                data.push(self.data()[idx]);
            }
        }
        CooMatrix::from_triplets(self.shape(), data, rows, cols, false)
    }
}

pub fn coo_to_csr_with_mode(
    coo: &CooMatrix,
    mode: RuntimeMode,
    operation_id: impl Into<String>,
) -> SparseResult<(CsrMatrix, ConversionLogEntry)> {
    let mut csr = coo.to_csr()?;
    if mode == RuntimeMode::Hardened {
        csr.canonical.sorted_indices = true;
        csr.canonical.deduplicated = true;
    }
    let log = conversion_log(
        operation_id.into(),
        SparseFormat::Coo,
        SparseFormat::Csr,
        coo.nnz(),
        csr.nnz(),
    );
    Ok((csr, log))
}

pub fn csr_to_csc_with_mode(
    csr: &CsrMatrix,
    mode: RuntimeMode,
    operation_id: impl Into<String>,
) -> SparseResult<(CscMatrix, ConversionLogEntry)> {
    if mode == RuntimeMode::Hardened && !csr.canonical.sorted_indices {
        return Err(SparseError::InvalidSparseStructure {
            message: "hardened conversion requires sorted CSR indices".to_string(),
        });
    }
    let mut csc = csr.to_csc()?;
    if mode == RuntimeMode::Strict {
        csc.canonical = csr.canonical;
    } else {
        csc.canonical.sorted_indices = true;
        csc.canonical.deduplicated = true;
    }
    let log = conversion_log(
        operation_id.into(),
        SparseFormat::Csr,
        SparseFormat::Csc,
        csr.nnz(),
        csc.nnz(),
    );
    Ok((csc, log))
}

pub fn csc_to_csr_with_mode(
    csc: &CscMatrix,
    mode: RuntimeMode,
    operation_id: impl Into<String>,
) -> SparseResult<(CsrMatrix, ConversionLogEntry)> {
    if mode == RuntimeMode::Hardened && !csc.canonical.sorted_indices {
        return Err(SparseError::InvalidSparseStructure {
            message: "hardened conversion requires sorted CSC indices".to_string(),
        });
    }
    let mut csr = csc.to_csr()?;
    if mode == RuntimeMode::Strict {
        csr.canonical = csc.canonical;
    } else {
        csr.canonical.sorted_indices = true;
        csr.canonical.deduplicated = true;
    }
    let log = conversion_log(
        operation_id.into(),
        SparseFormat::Csc,
        SparseFormat::Csr,
        csc.nnz(),
        csr.nnz(),
    );
    Ok((csr, log))
}

pub fn spmv_csr(matrix: &CsrMatrix, vector: &[f64]) -> SparseResult<Vec<f64>> {
    if vector.len() != matrix.shape().cols {
        return Err(SparseError::IncompatibleShape {
            message: "spmv vector length must match matrix columns".to_string(),
        });
    }
    let mut result = vec![0.0; matrix.shape().rows];
    for (row, output) in result.iter_mut().enumerate().take(matrix.shape().rows) {
        for idx in matrix.indptr()[row]..matrix.indptr()[row + 1] {
            *output += matrix.data()[idx] * vector[matrix.indices()[idx]];
        }
    }
    Ok(result)
}

pub fn spmv_csc(matrix: &CscMatrix, vector: &[f64]) -> SparseResult<Vec<f64>> {
    if vector.len() != matrix.shape().cols {
        return Err(SparseError::IncompatibleShape {
            message: "spmv vector length must match matrix columns".to_string(),
        });
    }
    let mut result = vec![0.0; matrix.shape().rows];
    for (col, value) in vector.iter().enumerate().take(matrix.shape().cols) {
        for idx in matrix.indptr()[col]..matrix.indptr()[col + 1] {
            result[matrix.indices()[idx]] += matrix.data()[idx] * *value;
        }
    }
    Ok(result)
}

pub fn spmv_coo(matrix: &CooMatrix, vector: &[f64]) -> SparseResult<Vec<f64>> {
    if vector.len() != matrix.shape().cols {
        return Err(SparseError::IncompatibleShape {
            message: "spmv vector length must match matrix columns".to_string(),
        });
    }
    let mut result = vec![0.0; matrix.shape().rows];
    for idx in 0..matrix.nnz() {
        let row = matrix.row_indices()[idx];
        let col = matrix.col_indices()[idx];
        result[row] += matrix.data()[idx] * vector[col];
    }
    Ok(result)
}

pub fn add_csr(lhs: &CsrMatrix, rhs: &CsrMatrix) -> SparseResult<CsrMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    combine_coo(lhs.to_coo()?, rhs.to_coo()?, 1.0)?.to_csr()
}

pub fn sub_csr(lhs: &CsrMatrix, rhs: &CsrMatrix) -> SparseResult<CsrMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    combine_coo(lhs.to_coo()?, rhs.to_coo()?, -1.0)?.to_csr()
}

pub fn scale_csr(matrix: &CsrMatrix, alpha: f64) -> SparseResult<CsrMatrix> {
    let data: Vec<f64> = matrix.data().iter().map(|v| v * alpha).collect();
    let mut scaled = CsrMatrix::from_components(
        matrix.shape(),
        data,
        matrix.indices().to_vec(),
        matrix.indptr().to_vec(),
        false,
    )?;
    scaled.canonical = matrix.canonical;
    Ok(scaled)
}

pub fn add_csc(lhs: &CscMatrix, rhs: &CscMatrix) -> SparseResult<CscMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    combine_coo(lhs.to_coo()?, rhs.to_coo()?, 1.0)?.to_csc()
}

pub fn sub_csc(lhs: &CscMatrix, rhs: &CscMatrix) -> SparseResult<CscMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    combine_coo(lhs.to_coo()?, rhs.to_coo()?, -1.0)?.to_csc()
}

pub fn scale_csc(matrix: &CscMatrix, alpha: f64) -> SparseResult<CscMatrix> {
    let data: Vec<f64> = matrix.data().iter().map(|v| v * alpha).collect();
    let mut scaled = CscMatrix::from_components(
        matrix.shape(),
        data,
        matrix.indices().to_vec(),
        matrix.indptr().to_vec(),
        false,
    )?;
    scaled.canonical = matrix.canonical;
    Ok(scaled)
}

pub fn add_coo(lhs: &CooMatrix, rhs: &CooMatrix) -> SparseResult<CooMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    combine_coo(lhs.clone(), rhs.clone(), 1.0)
}

pub fn sub_coo(lhs: &CooMatrix, rhs: &CooMatrix) -> SparseResult<CooMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    combine_coo(lhs.clone(), rhs.clone(), -1.0)
}

pub fn scale_coo(matrix: &CooMatrix, alpha: f64) -> SparseResult<CooMatrix> {
    let data: Vec<f64> = matrix.data().iter().map(|v| v * alpha).collect();
    CooMatrix::from_triplets(
        matrix.shape(),
        data,
        matrix.row_indices().to_vec(),
        matrix.col_indices().to_vec(),
        false,
    )
}

fn combine_coo(lhs: CooMatrix, rhs: CooMatrix, rhs_scale: f64) -> SparseResult<CooMatrix> {
    let shape = lhs.shape();
    let mut accum = BTreeMap::<(usize, usize), f64>::new();
    for idx in 0..lhs.nnz() {
        let key = (lhs.row_indices()[idx], lhs.col_indices()[idx]);
        *accum.entry(key).or_insert(0.0) += lhs.data()[idx];
    }
    for idx in 0..rhs.nnz() {
        let key = (rhs.row_indices()[idx], rhs.col_indices()[idx]);
        *accum.entry(key).or_insert(0.0) += rhs_scale * rhs.data()[idx];
    }

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for ((row, col), value) in accum {
        if value != 0.0 {
            rows.push(row);
            cols.push(col);
            data.push(value);
        }
    }
    CooMatrix::from_triplets(shape, data, rows, cols, false)
}

fn ensure_same_shape(lhs: Shape2D, rhs: Shape2D) -> SparseResult<()> {
    if lhs != rhs {
        return Err(SparseError::IncompatibleShape {
            message: "sparse arithmetic requires matching shapes".to_string(),
        });
    }
    Ok(())
}

fn canonical_triplets(coo: &CooMatrix) -> Vec<(usize, usize, f64)> {
    let mut triplets: Vec<(usize, usize, f64)> = coo
        .row_indices()
        .iter()
        .copied()
        .zip(coo.col_indices().iter().copied())
        .zip(coo.data().iter().copied())
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

    dedup
}

fn compress_triplets(
    shape: Shape2D,
    triplets: &[(usize, usize, f64)],
    by_col: bool,
) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    if by_col {
        let mut indptr = vec![0usize; shape.cols + 1];
        for (_, col, _) in triplets {
            indptr[*col + 1] += 1;
        }
        for col in 0..shape.cols {
            indptr[col + 1] += indptr[col];
        }
        let mut data = Vec::with_capacity(triplets.len());
        let mut indices = Vec::with_capacity(triplets.len());
        for (row, _, value) in triplets {
            indices.push(*row);
            data.push(*value);
        }
        (data, indices, indptr)
    } else {
        let mut indptr = vec![0usize; shape.rows + 1];
        for (row, _, _) in triplets {
            indptr[*row + 1] += 1;
        }
        for row in 0..shape.rows {
            indptr[row + 1] += indptr[row];
        }
        let mut data = Vec::with_capacity(triplets.len());
        let mut indices = Vec::with_capacity(triplets.len());
        for (_, col, value) in triplets {
            indices.push(*col);
            data.push(*value);
        }
        (data, indices, indptr)
    }
}

fn conversion_log(
    operation_id: String,
    from_format: SparseFormat,
    to_format: SparseFormat,
    nnz_before: usize,
    nnz_after: usize,
) -> ConversionLogEntry {
    ConversionLogEntry {
        timestamp: now_timestamp(),
        operation_id,
        from_format,
        to_format,
        nnz_before,
        nnz_after,
    }
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
