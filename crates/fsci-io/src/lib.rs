#![forbid(unsafe_code)]

//! Input/Output routines for FrankenSciPy.
//!
//! Matches `scipy.io` core functions:
//! - `savemat` / `loadmat` — MATLAB .mat file v5 (Level 5) read/write
//! - `mmread` / `mmwrite` — Matrix Market format read/write
//! - `wavfile.read` / `wavfile.write` — WAV audio file read/write
//! - `netcdf_file` — NetCDF (simplified) read/write

/// Error type for I/O operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoError {
    InvalidFormat(String),
    IoFailed(String),
    UnsupportedFeature(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
            Self::IoFailed(msg) => write!(f, "I/O failed: {msg}"),
            Self::UnsupportedFeature(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for IoError {}

// ══════════════════════════════════════════════════════════════════════
// Matrix Market Format
// ══════════════════════════════════════════════════════════════════════

/// Matrix Market object type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmObject {
    Matrix,
    Vector,
}

/// Matrix Market format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmFormat {
    Coordinate,
    Array,
}

/// Matrix Market field type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmField {
    Real,
    Integer,
    Complex,
    Pattern,
}

/// Matrix Market symmetry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmSymmetry {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

/// Matrix Market header information.
#[derive(Debug, Clone)]
pub struct MmInfo {
    pub object: MmObject,
    pub format: MmFormat,
    pub field: MmField,
    pub symmetry: MmSymmetry,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
}

/// Dense matrix result from Matrix Market.
#[derive(Debug, Clone)]
pub struct MmMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
    pub info: MmInfo,
}

/// Read a Matrix Market file.
///
/// Matches `scipy.io.mmread`.
pub fn mmread(content: &str) -> Result<MmMatrix, IoError> {
    let mut lines = content.lines();

    // Parse header line
    let header = lines
        .next()
        .ok_or_else(|| IoError::InvalidFormat("empty file".to_string()))?;
    if !header.starts_with("%%MatrixMarket") {
        return Err(IoError::InvalidFormat(
            "missing %%MatrixMarket header".to_string(),
        ));
    }

    let parts: Vec<&str> = header.split_whitespace().collect();
    if parts.len() < 5 {
        return Err(IoError::InvalidFormat("incomplete header line".to_string()));
    }

    let object = match parts[1].to_lowercase().as_str() {
        "matrix" => MmObject::Matrix,
        "vector" => MmObject::Vector,
        other => {
            return Err(IoError::InvalidFormat(format!(
                "unknown object type: {other}"
            )));
        }
    };

    let format = match parts[2].to_lowercase().as_str() {
        "coordinate" => MmFormat::Coordinate,
        "array" => MmFormat::Array,
        other => return Err(IoError::InvalidFormat(format!("unknown format: {other}"))),
    };

    let field = match parts[3].to_lowercase().as_str() {
        "real" => MmField::Real,
        "integer" => MmField::Integer,
        "complex" => MmField::Complex,
        "pattern" => MmField::Pattern,
        other => {
            return Err(IoError::InvalidFormat(format!(
                "unknown field type: {other}"
            )));
        }
    };

    let symmetry = match parts[4].to_lowercase().as_str() {
        "general" => MmSymmetry::General,
        "symmetric" => MmSymmetry::Symmetric,
        "skew-symmetric" => MmSymmetry::SkewSymmetric,
        "hermitian" => MmSymmetry::Hermitian,
        other => return Err(IoError::InvalidFormat(format!("unknown symmetry: {other}"))),
    };

    // Skip comment lines
    let mut size_line = None;
    for line in lines.by_ref() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        size_line = Some(trimmed.to_string());
        break;
    }

    let size_str =
        size_line.ok_or_else(|| IoError::InvalidFormat("missing size line".to_string()))?;
    let size_parts: Vec<&str> = size_str.split_whitespace().collect();

    match format {
        MmFormat::Coordinate => {
            if size_parts.len() < 3 {
                return Err(IoError::InvalidFormat(
                    "coordinate format requires rows cols nnz".to_string(),
                ));
            }
            let rows: usize = size_parts[0]
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad rows: {e}")))?;
            let cols: usize = size_parts[1]
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad cols: {e}")))?;
            let nnz: usize = size_parts[2]
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad nnz: {e}")))?;

            let mut data = vec![0.0; rows * cols];

            for line in lines {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('%') {
                    continue;
                }
                let vals: Vec<&str> = trimmed.split_whitespace().collect();
                if vals.len() < 2 {
                    continue;
                }
                let r_one_based = vals[0]
                    .parse::<usize>()
                    .map_err(|e| IoError::InvalidFormat(format!("bad row index: {e}")))?;
                let c_one_based = vals[1]
                    .parse::<usize>()
                    .map_err(|e| IoError::InvalidFormat(format!("bad col index: {e}")))?;
                let r = r_one_based.checked_sub(1).ok_or_else(|| {
                    IoError::InvalidFormat(
                        "Matrix Market row indices must be 1-based and >= 1".to_string(),
                    )
                })?;
                let c = c_one_based.checked_sub(1).ok_or_else(|| {
                    IoError::InvalidFormat(
                        "Matrix Market col indices must be 1-based and >= 1".to_string(),
                    )
                })?;
                let v: f64 = if field == MmField::Pattern {
                    1.0
                } else if vals.len() >= 3 {
                    vals[2]
                        .parse()
                        .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?
                } else {
                    1.0
                };

                if r < rows && c < cols {
                    data[r * cols + c] = v;
                    if symmetry == MmSymmetry::Symmetric && r != c {
                        data[c * cols + r] = v;
                    } else if symmetry == MmSymmetry::SkewSymmetric && r != c {
                        data[c * cols + r] = -v;
                    }
                }
            }

            Ok(MmMatrix {
                rows,
                cols,
                data,
                info: MmInfo {
                    object,
                    format,
                    field,
                    symmetry,
                    rows,
                    cols,
                    nnz,
                },
            })
        }
        MmFormat::Array => {
            if size_parts.len() < 2 {
                return Err(IoError::InvalidFormat(
                    "array format requires rows cols".to_string(),
                ));
            }
            let rows: usize = size_parts[0]
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad rows: {e}")))?;
            let cols: usize = size_parts[1]
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad cols: {e}")))?;

            // Array format: column-major order
            let mut data = vec![0.0; rows * cols];
            let mut idx = 0;

            for line in lines {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('%') {
                    continue;
                }
                if idx >= rows * cols {
                    return Err(IoError::InvalidFormat(format!(
                        "array format has more than the declared {} values",
                        rows * cols
                    )));
                }
                let v: f64 = trimmed
                    .parse()
                    .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?;

                // Column-major to row-major conversion
                let col = idx / rows;
                let row = idx % rows;
                if row < rows && col < cols {
                    data[row * cols + col] = v;
                }
                idx += 1;
            }
            if idx != rows * cols {
                return Err(IoError::InvalidFormat(format!(
                    "array format expected {} values but found {idx}",
                    rows * cols
                )));
            }

            Ok(MmMatrix {
                rows,
                cols,
                data,
                info: MmInfo {
                    object,
                    format,
                    field,
                    symmetry,
                    rows,
                    cols,
                    nnz: rows * cols,
                },
            })
        }
    }
}

/// Write a dense matrix in Matrix Market format.
///
/// Matches `scipy.io.mmwrite`.
pub fn mmwrite(rows: usize, cols: usize, data: &[f64]) -> Result<String, IoError> {
    if data.len() != rows * cols {
        return Err(IoError::InvalidFormat(format!(
            "data length {} doesn't match {}x{}",
            data.len(),
            rows,
            cols
        )));
    }

    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix array real general\n");
    out.push_str(&format!("{rows} {cols}\n"));

    // Column-major order (Matrix Market convention)
    for c in 0..cols {
        for r in 0..rows {
            let v = data[r * cols + c];
            out.push_str(&format!("{v}\n"));
        }
    }

    Ok(out)
}

/// Write a sparse matrix in coordinate Matrix Market format.
pub fn mmwrite_sparse(
    rows: usize,
    cols: usize,
    entries: &[(usize, usize, f64)],
) -> Result<String, IoError> {
    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix coordinate real general\n");
    out.push_str(&format!("{rows} {cols} {}\n", entries.len()));

    for &(r, c, v) in entries {
        out.push_str(&format!("{} {} {v}\n", r + 1, c + 1));
    }

    Ok(out)
}

/// Read Matrix Market info (header only).
///
/// Matches `scipy.io.mminfo`.
pub fn mminfo(content: &str) -> Result<MmInfo, IoError> {
    let mat = mmread(content)?;
    Ok(mat.info)
}

// ══════════════════════════════════════════════════════════════════════
// WAV File Format
// ══════════════════════════════════════════════════════════════════════

/// WAV file data.
#[derive(Debug, Clone)]
pub struct WavData {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub data: Vec<f64>,
}

/// Read a WAV file from bytes.
///
/// Matches `scipy.io.wavfile.read`.
pub fn wav_read(bytes: &[u8]) -> Result<WavData, IoError> {
    if bytes.len() < 44 {
        return Err(IoError::InvalidFormat("WAV file too short".to_string()));
    }

    // RIFF header
    if &bytes[0..4] != b"RIFF" {
        return Err(IoError::InvalidFormat("missing RIFF header".to_string()));
    }
    if &bytes[8..12] != b"WAVE" {
        return Err(IoError::InvalidFormat(
            "missing WAVE identifier".to_string(),
        ));
    }

    // Find fmt chunk
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_format = 0u16;

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;

        if chunk_id == b"fmt " {
            if chunk_size < 16 || pos + 8 + chunk_size > bytes.len() {
                return Err(IoError::InvalidFormat("fmt chunk too small".to_string()));
            }
            let fmt = &bytes[pos + 8..];
            audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
            channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
        } else if chunk_id == b"data" {
            if pos + 8 + chunk_size > bytes.len() {
                return Err(IoError::InvalidFormat(
                    "data chunk extends past file".to_string(),
                ));
            }
            let data_bytes = &bytes[pos + 8..pos + 8 + chunk_size];

            if audio_format != 1 && audio_format != 3 {
                return Err(IoError::UnsupportedFeature(format!(
                    "unsupported audio format: {audio_format} (only PCM=1 and IEEE_FLOAT=3 supported)"
                )));
            }

            let samples = match bits_per_sample {
                8 => data_bytes
                    .iter()
                    .map(|&b| (b as f64 - 128.0) / 128.0)
                    .collect(),
                16 => data_bytes
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0)
                    .collect(),
                24 => data_bytes
                    .chunks_exact(3)
                    .map(|c| {
                        let sign = if c[2] & 0x80 != 0 { 0xFF } else { 0x00 };
                        let raw = i32::from_le_bytes([c[0], c[1], c[2], sign]);
                        raw as f64 / 8_388_608.0
                    })
                    .collect(),
                32 if audio_format == 3 => data_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                    .collect(),
                32 => data_bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64 / 2_147_483_648.0)
                    .collect(),
                _ => {
                    return Err(IoError::UnsupportedFeature(format!(
                        "unsupported bits per sample: {bits_per_sample}"
                    )));
                }
            };

            return Ok(WavData {
                sample_rate,
                channels,
                bits_per_sample,
                data: samples,
            });
        }

        pos += 8 + chunk_size;
        // Chunks are word-aligned
        if !chunk_size.is_multiple_of(2) {
            pos += 1;
        }
    }

    Err(IoError::InvalidFormat("no data chunk found".to_string()))
}

/// Write WAV file data as bytes (16-bit PCM).
///
/// Matches `scipy.io.wavfile.write`.
pub fn wav_write(sample_rate: u32, channels: u16, data: &[f64]) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let bytes_per_sample = bits_per_sample / 8;
    let data_size = (data.len() * bytes_per_sample as usize) as u32;
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(file_size as usize + 8);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * channels as u32 * bytes_per_sample as u32;
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels * bytes_per_sample;
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &sample in data {
        let clamped = sample.clamp(-1.0, 1.0);
        let val = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&val.to_le_bytes());
    }

    buf
}

// ══════════════════════════════════════════════════════════════════════
// MAT-file v4 (simple numeric arrays)
// ══════════════════════════════════════════════════════════════════════

/// A named array loaded from a MAT file.
#[derive(Debug, Clone)]
pub struct MatArray {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

/// Save arrays to a simple text-based format (similar to MATLAB ASCII).
///
/// This provides a basic `savemat`-like interface. Full .mat v5 binary
/// format requires extensive implementation; this provides a portable
/// text alternative.
pub fn savemat_text(arrays: &[MatArray]) -> String {
    let mut out = String::new();
    for arr in arrays {
        out.push_str(&format!(
            "# name: {}\n# type: matrix\n# rows: {}\n# columns: {}\n",
            arr.name, arr.rows, arr.cols
        ));
        for r in 0..arr.rows {
            for c in 0..arr.cols {
                if c > 0 {
                    out.push(' ');
                }
                out.push_str(&format!("{}", arr.data[r * arr.cols + c]));
            }
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

/// Load arrays from the text-based format.
pub fn loadmat_text(content: &str) -> Result<Vec<MatArray>, IoError> {
    let mut arrays = Vec::new();
    let mut lines = content.lines().peekable();

    while lines.peek().is_some() {
        // Find "# name:" line
        let mut name = None;
        let mut rows = 0usize;
        let mut cols = 0usize;

        loop {
            match lines.next() {
                None => return Ok(arrays),
                Some(line) => {
                    let trimmed = line.trim();
                    if let Some(stripped) = trimmed.strip_prefix("# name:") {
                        name = Some(stripped.trim().to_string());
                    } else if let Some(stripped) = trimmed.strip_prefix("# rows:") {
                        rows = stripped
                            .trim()
                            .parse()
                            .map_err(|e| IoError::InvalidFormat(format!("bad rows: {e}")))?;
                    } else if let Some(stripped) = trimmed.strip_prefix("# columns:") {
                        cols = stripped
                            .trim()
                            .parse()
                            .map_err(|e| IoError::InvalidFormat(format!("bad cols: {e}")))?;
                    } else if trimmed.starts_with("# type:") {
                        // Skip type line
                    } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
                        // Data line — we've hit the matrix data
                        if let Some(ref n) = name {
                            let mut data = Vec::with_capacity(rows * cols);
                            // Parse this line
                            for val_str in trimmed.split_whitespace() {
                                let v: f64 = val_str.parse().map_err(|e| {
                                    IoError::InvalidFormat(format!("bad value: {e}"))
                                })?;
                                data.push(v);
                            }
                            // Read remaining rows
                            for _ in 1..rows {
                                if let Some(line) = lines.next() {
                                    for val_str in line.split_whitespace() {
                                        let v: f64 = val_str.parse().map_err(|e| {
                                            IoError::InvalidFormat(format!("bad value: {e}"))
                                        })?;
                                        data.push(v);
                                    }
                                }
                            }
                            arrays.push(MatArray {
                                name: n.clone(),
                                rows,
                                cols,
                                data,
                            });
                            break;
                        }
                    }
                }
            }
        }
    }

    Ok(arrays)
}

// ══════════════════════════════════════════════════════════════════════
// IDL / Recarr Utility
// ══════════════════════════════════════════════════════════════════════

/// Load a whitespace-delimited text file as a matrix.
///
/// Like `numpy.loadtxt`.
pub fn loadtxt(content: &str) -> Result<(usize, usize, Vec<f64>), IoError> {
    let mut data = Vec::new();
    let mut cols = 0usize;
    let mut rows = 0usize;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('%') {
            continue;
        }

        let vals: Result<Vec<f64>, _> = trimmed
            .split_whitespace()
            .map(|s| s.parse::<f64>())
            .collect();
        let vals = vals.map_err(|e| IoError::InvalidFormat(format!("parse error: {e}")))?;

        if rows == 0 {
            cols = vals.len();
        } else if vals.len() != cols {
            return Err(IoError::InvalidFormat(format!(
                "row {rows} has {} columns, expected {cols}",
                vals.len()
            )));
        }

        data.extend_from_slice(&vals);
        rows += 1;
    }

    Ok((rows, cols, data))
}

/// Save a matrix as whitespace-delimited text.
///
/// Like `numpy.savetxt`.
pub fn savetxt(rows: usize, cols: usize, data: &[f64], delimiter: &str) -> String {
    let mut out = String::new();
    for r in 0..rows {
        for c in 0..cols {
            if c > 0 {
                out.push_str(delimiter);
            }
            out.push_str(&format!("{}", data[r * cols + c]));
        }
        out.push('\n');
    }
    out
}

/// Read a CSV file into rows of f64 values.
///
/// Simple CSV reader for numerical data.
pub type CsvResult = Result<(Option<Vec<String>>, Vec<Vec<f64>>), IoError>;

pub fn read_csv(content: &str, delimiter: char, has_header: bool) -> CsvResult {
    let mut lines = content.lines();
    let header = if has_header {
        lines.next().map(|h| {
            h.split(delimiter)
                .map(|s| s.trim().to_string())
                .collect()
        })
    } else {
        None
    };

    let mut data = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let row: Result<Vec<f64>, _> = trimmed
            .split(delimiter)
            .map(|s| s.trim().parse::<f64>())
            .collect();
        match row {
            Ok(r) => data.push(r),
            Err(e) => {
                return Err(IoError::InvalidFormat(format!("CSV parse error: {e}")));
            }
        }
    }

    Ok((header, data))
}

/// Write data to CSV format.
pub fn write_csv(
    header: Option<&[&str]>,
    data: &[Vec<f64>],
    delimiter: char,
) -> String {
    let mut out = String::new();
    if let Some(h) = header {
        out.push_str(&h.join(&delimiter.to_string()));
        out.push('\n');
    }
    for row in data {
        let row_str: Vec<String> = row.iter().map(|v| format!("{v}")).collect();
        out.push_str(&row_str.join(&delimiter.to_string()));
        out.push('\n');
    }
    out
}

/// Read a simple JSON array of numbers.
pub fn read_json_array(content: &str) -> Result<Vec<f64>, IoError> {
    let trimmed = content.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(IoError::InvalidFormat("expected JSON array".to_string()));
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    inner
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|e| IoError::InvalidFormat(format!("JSON parse error: {e}")))
        })
        .collect()
}

/// Write a vector as a JSON array.
pub fn write_json_array(data: &[f64]) -> String {
    let items: Vec<String> = data.iter().map(|v| format!("{v}")).collect();
    format!("[{}]", items.join(", "))
}

/// Read a simple NPY-like header (shape + dtype) from text representation.
///
/// Returns (shape, data).
pub fn read_npy_text(content: &str) -> Result<(Vec<usize>, Vec<f64>), IoError> {
    let mut lines = content.lines();

    // Read shape line
    let shape_line = lines
        .next()
        .ok_or_else(|| IoError::InvalidFormat("missing shape line".to_string()))?;
    let shape: Result<Vec<usize>, _> = shape_line
        .trim()
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>())
        .collect();
    let shape = shape.map_err(|e| IoError::InvalidFormat(format!("bad shape: {e}")))?;

    // Read data
    let mut data = Vec::new();
    for line in lines {
        for val in line.split_whitespace() {
            let v: f64 = val
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?;
            data.push(v);
        }
    }

    Ok((shape, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmread_coordinate() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 3\n\
                        1 1 1.0\n\
                        2 2 2.0\n\
                        3 3 3.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.rows, 3);
        assert_eq!(mat.cols, 3);
        assert_eq!(mat.data[0], 1.0); // (0,0)
        assert_eq!(mat.data[4], 2.0); // (1,1)
        assert_eq!(mat.data[8], 3.0); // (2,2)
        assert_eq!(mat.data[1], 0.0); // off-diagonal is zero
    }

    #[test]
    fn mmread_symmetric() {
        let content = "%%MatrixMarket matrix coordinate real symmetric\n\
                        3 3 2\n\
                        1 1 5.0\n\
                        2 1 3.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.data[0], 5.0); // (0,0)
        assert_eq!(mat.data[3], 3.0); // (1,0)
        assert_eq!(mat.data[1], 3.0); // (0,1) = symmetric
    }

    #[test]
    fn mmread_rejects_zero_based_coordinate_indices() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 1\n\
                        0 1 5.0\n";
        let err = mmread(content).expect_err("zero-based row index should be rejected");
        assert_eq!(
            err,
            IoError::InvalidFormat("Matrix Market row indices must be 1-based and >= 1".to_string())
        );
    }

    #[test]
    fn mmread_array() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n\
                        1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 3);
        // Column-major: 1,2 are col 0, 3,4 are col 1, 5,6 are col 2
        assert_eq!(mat.data[0], 1.0); // (0,0)
        assert_eq!(mat.data[3], 2.0); // (1,0)
        assert_eq!(mat.data[1], 3.0); // (0,1)
    }

    #[test]
    fn mmread_array_rejects_too_few_values() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n\
                        1.0\n2.0\n3.0\n4.0\n5.0\n";
        let err = mmread(content).expect_err("underfilled array payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array format expected 6 values but found 5".to_string())
        );
    }

    #[test]
    fn mmread_array_rejects_too_many_values() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n\
                        1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n7.0\n";
        let err = mmread(content).expect_err("overfilled array payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array format has more than the declared 6 values".to_string())
        );
    }

    #[test]
    fn mmwrite_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let content = mmwrite(2, 3, &data).unwrap();
        let mat = mmread(&content).unwrap();
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 3);
        for (i, (&orig, &read)) in data.iter().zip(mat.data.iter()).enumerate() {
            assert!(
                (orig - read).abs() < 1e-10,
                "mismatch at {i}: orig={orig}, read={read}"
            );
        }
    }

    #[test]
    fn wav_roundtrip() {
        let samples = vec![0.0, 0.5, 1.0, -1.0, -0.5, 0.0];
        let bytes = wav_write(44100, 1, &samples);
        let wav = wav_read(&bytes).unwrap();
        assert_eq!(wav.sample_rate, 44100);
        assert_eq!(wav.channels, 1);
        assert_eq!(wav.data.len(), samples.len());
        // 16-bit quantization: ~1/32768 precision
        for (i, (&orig, &read)) in samples.iter().zip(wav.data.iter()).enumerate() {
            assert!(
                (orig - read).abs() < 0.001,
                "sample {i}: orig={orig}, read={read}"
            );
        }
    }

    #[test]
    fn wav_read_24bit_pcm_sign_extends_negative_samples() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + 3u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&44100u32.to_le_bytes());
        bytes.extend_from_slice(&(44100u32 * 3).to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(&24u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&[0x00, 0x00, 0x80]); // -1.0 in signed 24-bit PCM
        bytes.push(0); // pad odd-sized chunk

        let wav = wav_read(&bytes).expect("24-bit wav should decode");
        assert_eq!(wav.bits_per_sample, 24);
        assert!(
            (wav.data[0] + 1.0).abs() < 1e-6,
            "expected -1.0 sample, got {}",
            wav.data[0]
        );
    }

    #[test]
    fn savemat_loadmat_roundtrip() {
        let arrays = vec![
            MatArray {
                name: "A".to_string(),
                rows: 2,
                cols: 2,
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
            MatArray {
                name: "b".to_string(),
                rows: 3,
                cols: 1,
                data: vec![10.0, 20.0, 30.0],
            },
        ];
        let text = savemat_text(&arrays);
        let loaded = loadmat_text(&text).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "A");
        assert_eq!(loaded[0].data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded[1].name, "b");
        assert_eq!(loaded[1].data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn loadtxt_basic() {
        let content = "# comment\n1 2 3\n4 5 6\n";
        let (rows, cols, data) = loadtxt(content).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(cols, 3);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn savetxt_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let text = savetxt(2, 2, &data, " ");
        assert_eq!(text, "1 2\n3 4\n");
    }

    #[test]
    fn mmwrite_sparse_format() {
        let entries = vec![(0, 0, 1.0), (1, 1, 2.0)];
        let content = mmwrite_sparse(3, 3, &entries).unwrap();
        assert!(content.contains("coordinate"));
        let mat = mmread(&content).unwrap();
        assert_eq!(mat.data[0], 1.0);
        assert_eq!(mat.data[4], 2.0);
    }
}
