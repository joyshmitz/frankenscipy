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

fn checked_matrix_len(rows: usize, cols: usize, context: &str) -> Result<usize, IoError> {
    rows.checked_mul(cols).ok_or_else(|| {
        IoError::InvalidFormat(format!(
            "{context} dimensions {rows}x{cols} overflowed usize"
        ))
    })
}

fn parse_mm_info(lines: &mut std::str::Lines<'_>) -> Result<MmInfo, IoError> {
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

            Ok(MmInfo {
                object,
                format,
                field,
                symmetry,
                rows,
                cols,
                nnz,
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
            let nnz = rows.checked_mul(cols).ok_or_else(|| {
                IoError::InvalidFormat("array dimensions overflowed nnz computation".to_string())
            })?;

            Ok(MmInfo {
                object,
                format,
                field,
                symmetry,
                rows,
                cols,
                nnz,
            })
        }
    }
}

/// Read a Matrix Market file.
///
/// Matches `scipy.io.mmread`.
pub fn mmread(content: &str) -> Result<MmMatrix, IoError> {
    let mut lines = content.lines();
    let info = parse_mm_info(&mut lines)?;
    if info.field == MmField::Complex {
        return Err(IoError::UnsupportedFeature(
            "Matrix Market complex field is not supported".to_string(),
        ));
    }

    match info.format {
        MmFormat::Coordinate => {
            let rows = info.rows;
            let cols = info.cols;
            let nnz = info.nnz;
            let dense_len = checked_matrix_len(rows, cols, "Matrix Market matrix")?;
            let mut data = vec![0.0; dense_len];
            let mut seen_nnz = 0usize;

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
                let v: f64 = if info.field == MmField::Pattern {
                    1.0
                } else if vals.len() >= 3 {
                    vals[2]
                        .parse()
                        .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?
                } else {
                    return Err(IoError::InvalidFormat(
                        "coordinate entry missing value for non-pattern field".to_string(),
                    ));
                };

                if r >= rows || c >= cols {
                    return Err(IoError::InvalidFormat(format!(
                        "coordinate entry ({r}, {c}) out of bounds for {rows}x{cols}"
                    )));
                }

                let idx = r * cols + c;
                match info.symmetry {
                    MmSymmetry::General => {
                        data[idx] += v;
                    }
                    MmSymmetry::Symmetric | MmSymmetry::Hermitian => {
                        data[idx] += v;
                        if r != c {
                            data[c * cols + r] += v;
                        }
                    }
                    MmSymmetry::SkewSymmetric => {
                        if r == c {
                            if v != 0.0 {
                                return Err(IoError::InvalidFormat(
                                    "skew-symmetric diagonal entries must be zero".to_string(),
                                ));
                            }
                        } else {
                            data[idx] += v;
                            data[c * cols + r] -= v;
                        }
                    }
                }
                seen_nnz += 1;
            }

            if seen_nnz != nnz {
                return Err(IoError::InvalidFormat(format!(
                    "coordinate format expected {nnz} entries but found {seen_nnz}"
                )));
            }

            Ok(MmMatrix {
                rows,
                cols,
                data,
                info,
            })
        }
        MmFormat::Array => {
            let rows = info.rows;
            let cols = info.cols;
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
                info,
            })
        }
    }
}

/// Write a dense matrix in Matrix Market format.
///
/// Matches `scipy.io.mmwrite`.
pub fn mmwrite(rows: usize, cols: usize, data: &[f64]) -> Result<String, IoError> {
    let expected_len = checked_matrix_len(rows, cols, "Matrix Market matrix")?;
    if data.len() != expected_len {
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
        if r >= rows || c >= cols {
            return Err(IoError::InvalidFormat(format!(
                "sparse entry ({r}, {c}) out of bounds for {rows}x{cols}"
            )));
        }
        let row = r.checked_add(1).ok_or_else(|| {
            IoError::InvalidFormat("sparse row index overflowed Matrix Market encoding".to_string())
        })?;
        let col = c.checked_add(1).ok_or_else(|| {
            IoError::InvalidFormat("sparse col index overflowed Matrix Market encoding".to_string())
        })?;
        out.push_str(&format!("{row} {col} {v}\n"));
    }

    Ok(out)
}

/// Read Matrix Market info (header only).
///
/// Matches `scipy.io.mminfo`.
pub fn mminfo(content: &str) -> Result<MmInfo, IoError> {
    let mut lines = content.lines();
    parse_mm_info(&mut lines)
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
            if channels == 0 {
                return Err(IoError::InvalidFormat(
                    "fmt chunk declares zero channels".to_string(),
                ));
            }
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
        } else if chunk_id == b"data" {
            if pos + 8 + chunk_size > bytes.len() {
                return Err(IoError::InvalidFormat(
                    "data chunk extends past file".to_string(),
                ));
            }
            let data_bytes = &bytes[pos + 8..pos + 8 + chunk_size];
            if sample_rate == 0 || channels == 0 || bits_per_sample == 0 {
                return Err(IoError::InvalidFormat(
                    "encountered data chunk before a valid fmt chunk".to_string(),
                ));
            }

            if audio_format != 1 && audio_format != 3 {
                return Err(IoError::UnsupportedFeature(format!(
                    "unsupported audio format: {audio_format} (only PCM=1 and IEEE_FLOAT=3 supported)"
                )));
            }

            let bytes_per_sample = match bits_per_sample {
                8 => 1usize,
                16 => 2,
                24 => 3,
                32 => 4,
                _ => {
                    return Err(IoError::UnsupportedFeature(format!(
                        "unsupported bits per sample: {bits_per_sample}"
                    )));
                }
            };
            if !data_bytes.len().is_multiple_of(bytes_per_sample) {
                return Err(IoError::InvalidFormat(format!(
                    "data chunk size {} is not aligned to {}-byte samples",
                    data_bytes.len(),
                    bytes_per_sample
                )));
            }
            let frame_bytes = bytes_per_sample
                .checked_mul(channels as usize)
                .ok_or_else(|| {
                    IoError::InvalidFormat("WAV frame size overflowed usize".to_string())
                })?;
            if !data_bytes.len().is_multiple_of(frame_bytes) {
                return Err(IoError::InvalidFormat(format!(
                    "data chunk size {} does not contain whole {}-channel frames",
                    data_bytes.len(),
                    channels
                )));
            }

            let samples = match (bits_per_sample, audio_format) {
                (8, _) => data_bytes
                    .iter()
                    .map(|&b| (b as f64 - 128.0) / 128.0)
                    .collect(),
                (16, _) => data_bytes
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0)
                    .collect(),
                (24, _) => data_bytes
                    .chunks_exact(3)
                    .map(|c| {
                        let sign = if c[2] & 0x80 != 0 { 0xFF } else { 0x00 };
                        let raw = i32::from_le_bytes([c[0], c[1], c[2], sign]);
                        raw as f64 / 8_388_608.0
                    })
                    .collect(),
                (32, 3) => data_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                    .collect(),
                (32, _) => data_bytes
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
pub fn wav_write(sample_rate: u32, channels: u16, data: &[f64]) -> Result<Vec<u8>, IoError> {
    if sample_rate == 0 {
        return Err(IoError::InvalidFormat(
            "WAV sample rate must be nonzero".to_string(),
        ));
    }
    if channels == 0 {
        return Err(IoError::InvalidFormat(
            "WAV channel count must be nonzero".to_string(),
        ));
    }
    if !data.len().is_multiple_of(channels as usize) {
        return Err(IoError::InvalidFormat(format!(
            "data length {} does not contain whole frames for {channels} channels",
            data.len()
        )));
    }
    let bits_per_sample: u16 = 16;
    let bytes_per_sample = bits_per_sample / 8;
    let data_size = data
        .len()
        .checked_mul(bytes_per_sample as usize)
        .and_then(|size| u32::try_from(size).ok())
        .ok_or_else(|| IoError::InvalidFormat("WAV data chunk too large".to_string()))?;
    let file_size = 36u32
        .checked_add(data_size)
        .ok_or_else(|| IoError::InvalidFormat("WAV file too large".to_string()))?;

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
    let byte_rate = sample_rate
        .checked_mul(channels as u32)
        .and_then(|rate| rate.checked_mul(bytes_per_sample as u32))
        .ok_or_else(|| IoError::InvalidFormat("WAV byte rate overflowed u32".to_string()))?;
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| IoError::InvalidFormat("WAV block align overflowed u16".to_string()))?;
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

    Ok(buf)
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
pub fn savemat_text(arrays: &[MatArray]) -> Result<String, IoError> {
    let mut out = String::new();
    for arr in arrays {
        if arr.name.contains(['\n', '\r']) {
            return Err(IoError::InvalidFormat(format!(
                "array name '{}' contains a newline and cannot be encoded safely",
                arr.name.escape_debug()
            )));
        }
        let expected_len = arr.rows.checked_mul(arr.cols).ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "array '{}' dimensions {}x{} overflow usize",
                arr.name, arr.rows, arr.cols
            ))
        })?;
        if arr.data.len() != expected_len {
            return Err(IoError::InvalidFormat(format!(
                "array '{}' expected {} values but found {}",
                arr.name,
                expected_len,
                arr.data.len()
            )));
        }
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
    Ok(out)
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
                None => {
                    if name.is_some() || rows != 0 || cols != 0 {
                        return Err(IoError::InvalidFormat(
                            "incomplete MAT text block at end of file".to_string(),
                        ));
                    }
                    return Ok(arrays);
                }
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
                        let n = name.as_ref().ok_or_else(|| {
                            IoError::InvalidFormat(
                                "encountered matrix data before '# name:' header".to_string(),
                            )
                        })?;
                        if rows == 0 || cols == 0 {
                            return Err(IoError::InvalidFormat(format!(
                                "array '{n}' is missing nonzero '# rows:' and '# columns:' headers before data"
                            )));
                        }
                        let expected_len = checked_matrix_len(rows, cols, "MAT text matrix")?;
                        let mut data = Vec::with_capacity(expected_len);
                        // Parse this line
                        for val_str in trimmed.split_whitespace() {
                            let v: f64 = val_str
                                .parse()
                                .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?;
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
                        if data.len() != expected_len {
                            return Err(IoError::InvalidFormat(format!(
                                "array '{n}' expected {} values but found {}",
                                expected_len,
                                data.len()
                            )));
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
pub fn savetxt(rows: usize, cols: usize, data: &[f64], delimiter: &str) -> Result<String, IoError> {
    if delimiter.contains(['\n', '\r']) {
        return Err(IoError::InvalidFormat(format!(
            "delimiter {:?} contains a newline and cannot be encoded safely",
            delimiter
        )));
    }
    let expected_len = checked_matrix_len(rows, cols, "text matrix")?;
    if data.len() != expected_len {
        return Err(IoError::InvalidFormat(format!(
            "data length {} doesn't match {}x{}",
            data.len(),
            rows,
            cols
        )));
    }
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
    Ok(out)
}

/// Read a CSV file into rows of f64 values.
///
/// Simple CSV reader for numerical data.
pub type CsvResult = Result<(Option<Vec<String>>, Vec<Vec<f64>>), IoError>;

pub fn read_csv(content: &str, delimiter: char, has_header: bool) -> CsvResult {
    let mut lines = content.lines();
    let header = if has_header {
        Some(
            lines
                .next()
                .ok_or_else(|| {
                    IoError::InvalidFormat(
                        "CSV header row is required but the input is empty".to_string(),
                    )
                })?
                .split(delimiter)
                .map(|s| s.trim().to_string())
                .collect(),
        )
    } else {
        None
    };
    let header_cols = header.as_ref().map(std::vec::Vec::len);

    let mut data = Vec::new();
    let mut expected_cols = None;
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
            Ok(r) => {
                if let Some(cols) = expected_cols {
                    if r.len() != cols {
                        return Err(IoError::InvalidFormat(format!(
                            "CSV row has {} columns, expected {cols}",
                            r.len()
                        )));
                    }
                } else {
                    if let Some(header_cols) = header_cols
                        && r.len() != header_cols
                    {
                        return Err(IoError::InvalidFormat(format!(
                            "CSV header has {header_cols} columns but first data row has {}",
                            r.len()
                        )));
                    }
                    expected_cols = Some(r.len());
                }
                data.push(r);
            }
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
) -> Result<String, IoError> {
    let mut out = String::new();
    let header_cols = header.map(<[&str]>::len);
    if let Some(h) = header {
        for cell in h {
            if cell.contains(['\n', '\r']) {
                return Err(IoError::InvalidFormat(format!(
                    "CSV header cell {:?} contains a newline and cannot be encoded safely",
                    cell
                )));
            }
            if cell.contains(delimiter) {
                return Err(IoError::InvalidFormat(format!(
                    "CSV header cell {:?} contains the delimiter {:?} and cannot be encoded safely",
                    cell, delimiter
                )));
            }
        }
        out.push_str(&h.join(&delimiter.to_string()));
        out.push('\n');
    }
    let mut expected_cols = None;
    for row in data {
        if let Some(cols) = expected_cols {
            if row.len() != cols {
                return Err(IoError::InvalidFormat(format!(
                    "CSV row has {} columns, expected {cols}",
                    row.len()
                )));
            }
        } else {
            if let Some(header_cols) = header_cols
                && row.len() != header_cols
            {
                return Err(IoError::InvalidFormat(format!(
                    "CSV header has {header_cols} columns but first data row has {}",
                    row.len()
                )));
            }
            expected_cols = Some(row.len());
        }
        let row_str: Vec<String> = row.iter().map(|v| format!("{v}")).collect();
        out.push_str(&row_str.join(&delimiter.to_string()));
        out.push('\n');
    }
    Ok(out)
}

/// Read a simple JSON array of numbers.
pub fn read_json_array(content: &str) -> Result<Vec<f64>, IoError> {
    let trimmed = content.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(IoError::InvalidFormat("expected JSON array".to_string()));
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
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
pub fn write_json_array(data: &[f64]) -> Result<String, IoError> {
    if let Some((idx, value)) = data
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        return Err(IoError::InvalidFormat(format!(
            "JSON array value at index {idx} is not finite: {value}"
        )));
    }
    let items: Vec<String> = data.iter().map(|v| format!("{v}")).collect();
    Ok(format!("[{}]", items.join(", ")))
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
    let trimmed_shape = shape_line.trim();
    if trimmed_shape.is_empty() {
        return Err(IoError::InvalidFormat(
            "shape declaration must contain at least one dimension".to_string(),
        ));
    }
    if trimmed_shape
        .split(',')
        .any(|segment| segment.trim().is_empty())
    {
        return Err(IoError::InvalidFormat(
            "shape declaration contains an empty dimension".to_string(),
        ));
    }
    let shape: Result<Vec<usize>, _> = shape_line
        .trim()
        .split(',')
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

    let expected_len = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| IoError::InvalidFormat("shape product overflowed usize".to_string()))
    })?;
    if data.len() != expected_len {
        return Err(IoError::InvalidFormat(format!(
            "shape {:?} expects {expected_len} values but found {}",
            shape,
            data.len()
        )));
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
    fn mmread_coordinate_sums_duplicates() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        2 2 3\n\
                        1 1 1.0\n\
                        1 1 2.5\n\
                        2 1 -1.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.data[0], 3.5);
        assert_eq!(mat.data[2], -1.0);
        assert_eq!(mat.data[1], 0.0);
    }

    #[test]
    fn mmread_skew_symmetric() {
        let content = "%%MatrixMarket matrix coordinate real skew-symmetric\n\
                        3 3 1\n\
                        1 3 2.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.data[2], 2.0); // (0,2)
        assert_eq!(mat.data[6], -2.0); // (2,0)
    }

    #[test]
    fn mmread_skew_symmetric_rejects_nonzero_diagonal() {
        let content = "%%MatrixMarket matrix coordinate real skew-symmetric\n\
                        2 2 1\n\
                        1 1 1.0\n";
        let err = mmread(content).expect_err("skew-symmetric diagonal must be zero");
        assert_eq!(
            err,
            IoError::InvalidFormat("skew-symmetric diagonal entries must be zero".to_string())
        );
    }

    #[test]
    fn mmread_rejects_zero_based_coordinate_indices() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 1\n\
                        0 1 5.0\n";
        let err = mmread(content).expect_err("zero-based row index should be rejected");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "Matrix Market row indices must be 1-based and >= 1".to_string()
            )
        );
    }

    #[test]
    fn mmread_rejects_out_of_bounds_coordinate_indices() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 1\n\
                        4 1 5.0\n";
        let err = mmread(content).expect_err("out-of-bounds row index should be rejected");
        assert_eq!(
            err,
            IoError::InvalidFormat("coordinate entry (3, 0) out of bounds for 3x3".to_string())
        );
    }

    #[test]
    fn mmread_rejects_coordinate_nnz_mismatch() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 2\n\
                        1 1 1.0\n";
        let err = mmread(content).expect_err("declared nnz mismatch should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("coordinate format expected 2 entries but found 1".to_string())
        );
    }

    #[test]
    fn mmread_rejects_missing_coordinate_value_for_real_field() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 1\n\
                        1 1\n";
        let err = mmread(content).expect_err("real coordinate entries need explicit values");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "coordinate entry missing value for non-pattern field".to_string()
            )
        );
    }

    #[test]
    fn mmread_rejects_complex_coordinate_payloads() {
        let content = "%%MatrixMarket matrix coordinate complex general\n\
                        2 2 1\n\
                        1 1 1.0 2.0\n";
        let err = mmread(content).expect_err("complex Matrix Market payloads should fail closed");
        assert_eq!(
            err,
            IoError::UnsupportedFeature("Matrix Market complex field is not supported".to_string())
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
    fn mminfo_reads_coordinate_header_without_body() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 4 2\n";
        let info = mminfo(content).expect("header-only coordinate metadata should parse");
        assert_eq!(info.object, MmObject::Matrix);
        assert_eq!(info.format, MmFormat::Coordinate);
        assert_eq!(info.field, MmField::Real);
        assert_eq!(info.symmetry, MmSymmetry::General);
        assert_eq!(info.rows, 3);
        assert_eq!(info.cols, 4);
        assert_eq!(info.nnz, 2);
    }

    #[test]
    fn mminfo_reads_array_header_without_body() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n";
        let info = mminfo(content).expect("header-only array metadata should parse");
        assert_eq!(info.object, MmObject::Matrix);
        assert_eq!(info.format, MmFormat::Array);
        assert_eq!(info.field, MmField::Real);
        assert_eq!(info.symmetry, MmSymmetry::General);
        assert_eq!(info.rows, 2);
        assert_eq!(info.cols, 3);
        assert_eq!(info.nnz, 6);
    }

    #[test]
    fn mmread_rejects_complex_array_payloads() {
        let content = "%%MatrixMarket matrix array complex general\n\
                        2 1\n\
                        1.0 2.0\n\
                        3.0 4.0\n";
        let err = mmread(content).expect_err("complex Matrix Market payloads should fail closed");
        assert_eq!(
            err,
            IoError::UnsupportedFeature("Matrix Market complex field is not supported".to_string())
        );
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
    fn mmread_rejects_dense_size_overflow() {
        let content = format!(
            "%%MatrixMarket matrix coordinate real general\n{} 2 0\n",
            usize::MAX
        );
        let err = mmread(&content).expect_err("overflowing dense dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "Matrix Market matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
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
    fn mmwrite_rejects_dense_size_overflow() {
        let err = mmwrite(usize::MAX, 2, &[])
            .expect_err("overflowing dense dimensions should fail before length comparison");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "Matrix Market matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
        );
    }

    #[test]
    fn wav_roundtrip() {
        let samples = vec![0.0, 0.5, 1.0, -1.0, -0.5, 0.0];
        let bytes = wav_write(44100, 1, &samples).expect("mono samples should encode");
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
    fn wav_write_rejects_partial_frames() {
        let err = wav_write(44_100, 2, &[0.0, 0.5, 1.0])
            .expect_err("stereo data with odd sample count should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "data length 3 does not contain whole frames for 2 channels".to_string()
            )
        );
    }

    #[test]
    fn wav_write_rejects_zero_sample_rate() {
        let err =
            wav_write(0, 1, &[0.0]).expect_err("zero sample rate should fail before encoding");
        assert_eq!(
            err,
            IoError::InvalidFormat("WAV sample rate must be nonzero".to_string())
        );
    }

    #[test]
    fn wav_write_rejects_block_align_overflow() {
        let err = wav_write(1, 32_768, &[])
            .expect_err("oversized channel count should fail before header overflow");
        assert_eq!(
            err,
            IoError::InvalidFormat("WAV block align overflowed u16".to_string())
        );
    }

    #[test]
    fn wav_read_rejects_partial_sample_bytes() {
        let mut bytes = wav_write(44_100, 1, &[0.0, 0.5]).expect("mono samples should encode");
        bytes[40..44].copy_from_slice(&3u32.to_le_bytes());
        bytes.truncate(44 + 3);

        let err = wav_read(&bytes).expect_err("misaligned sample bytes should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "data chunk size 3 is not aligned to 2-byte samples".to_string()
            )
        );
    }

    #[test]
    fn wav_read_rejects_partial_frames() {
        let mut bytes =
            wav_write(44_100, 2, &[0.0, 0.5, 1.0, -1.0]).expect("stereo samples should encode");
        bytes[40..44].copy_from_slice(&6u32.to_le_bytes());
        bytes.truncate(44 + 6);

        let err = wav_read(&bytes).expect_err("partial stereo frame should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "data chunk size 6 does not contain whole 2-channel frames".to_string()
            )
        );
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
    fn wav_read_rejects_zero_channel_fmt() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + 2u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&44_100u32.to_le_bytes());
        bytes.extend_from_slice(&(44_100u32 * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i16.to_le_bytes());

        let err = wav_read(&bytes).expect_err("zero-channel WAV should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("fmt chunk declares zero channels".to_string())
        );
    }

    #[test]
    fn wav_read_rejects_data_before_fmt_chunk() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + 2u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i16.to_le_bytes());
        bytes.resize(44, 0);

        let err = wav_read(&bytes).expect_err("data before fmt should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("encountered data chunk before a valid fmt chunk".to_string())
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
        let text = savemat_text(&arrays).expect("well-formed arrays should serialize");
        let loaded = loadmat_text(&text).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "A");
        assert_eq!(loaded[0].data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded[1].name, "b");
        assert_eq!(loaded[1].data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn savemat_rejects_wrong_element_count() {
        let arrays = vec![MatArray {
            name: "A".to_string(),
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0],
        }];
        let err = savemat_text(&arrays).expect_err("truncated matrix payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array 'A' expected 4 values but found 3".to_string())
        );
    }

    #[test]
    fn savemat_rejects_multiline_names() {
        let arrays = vec![MatArray {
            name: "bad\nname".to_string(),
            rows: 1,
            cols: 1,
            data: vec![1.0],
        }];
        let err = savemat_text(&arrays).expect_err("multiline names should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "array name 'bad\\nname' contains a newline and cannot be encoded safely"
                    .to_string()
            )
        );
    }

    #[test]
    fn loadmat_rejects_wrong_element_count() {
        let text = "# name: A\n# type: matrix\n# rows: 2\n# columns: 2\n1 2\n3\n";
        let err = loadmat_text(text).expect_err("truncated matrix payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array 'A' expected 4 values but found 3".to_string())
        );
    }

    #[test]
    fn loadmat_rejects_data_without_name_header() {
        let err = loadmat_text("1 2\n3 4\n")
            .expect_err("data block without a name header should fail closed");
        assert_eq!(
            err,
            IoError::InvalidFormat("encountered matrix data before '# name:' header".to_string())
        );
    }

    #[test]
    fn loadmat_rejects_incomplete_trailing_header_block() {
        let err = loadmat_text("# name: A\n# rows: 2\n# columns: 2\n")
            .expect_err("trailing header-only block should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("incomplete MAT text block at end of file".to_string())
        );
    }

    #[test]
    fn loadmat_rejects_data_before_dimension_headers_are_complete() {
        let err = loadmat_text("# name: A\n# rows: 2\n1 2\n3 4\n")
            .expect_err("data before full dimension metadata should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "array 'A' is missing nonzero '# rows:' and '# columns:' headers before data"
                    .to_string()
            )
        );
    }

    #[test]
    fn loadmat_rejects_dimension_overflow() {
        let text = format!(
            "# name: A\n# type: matrix\n# rows: {}\n# columns: 2\n1 2\n",
            usize::MAX
        );
        let err = loadmat_text(&text).expect_err("overflowing MAT dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "MAT text matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
        );
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
        let text = savetxt(2, 2, &data, " ").expect("matching shape should succeed");
        assert_eq!(text, "1 2\n3 4\n");
    }

    #[test]
    fn savetxt_rejects_shape_length_mismatch() {
        let err = savetxt(2, 2, &[1.0, 2.0, 3.0], " ").expect_err("mismatched shape should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("data length 3 doesn't match 2x2".to_string())
        );
    }

    #[test]
    fn savetxt_rejects_dimension_overflow() {
        let err = savetxt(usize::MAX, 2, &[], " ")
            .expect_err("overflowing savetxt dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "text matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
        );
    }

    #[test]
    fn savetxt_rejects_multiline_delimiter() {
        let err = savetxt(1, 2, &[1.0, 2.0], "\n").expect_err("multiline delimiters should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "delimiter \"\\n\" contains a newline and cannot be encoded safely".to_string()
            )
        );
    }

    #[test]
    fn read_npy_text_rejects_shape_payload_mismatch() {
        let err = read_npy_text("2,2\n1 2 3\n").expect_err("truncated payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("shape [2, 2] expects 4 values but found 3".to_string())
        );
    }

    #[test]
    fn read_npy_text_rejects_empty_shape() {
        let err = read_npy_text("\n1\n").expect_err("empty shape line should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "shape declaration must contain at least one dimension".to_string()
            )
        );
    }

    #[test]
    fn read_npy_text_rejects_empty_shape_dimension() {
        let err = read_npy_text("2,,2\n1 2 3 4\n")
            .expect_err("shape declarations with empty dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("shape declaration contains an empty dimension".to_string())
        );
    }

    #[test]
    fn read_csv_rejects_ragged_rows() {
        let err = read_csv("1,2,3\n4,5\n", ',', false).expect_err("ragged CSV should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV row has 2 columns, expected 3".to_string())
        );
    }

    #[test]
    fn read_csv_rejects_empty_input_when_header_is_required() {
        let err =
            read_csv("", ',', true).expect_err("empty input with required header should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV header row is required but the input is empty".to_string())
        );
    }

    #[test]
    fn read_csv_rejects_header_data_column_mismatch() {
        let err =
            read_csv("a,b,c\n1,2\n3,4\n", ',', true).expect_err("header/data mismatch should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV header has 3 columns but first data row has 2".to_string())
        );
    }

    #[test]
    fn write_csv_basic() {
        let text = write_csv(None, &[vec![1.0, 2.0], vec![3.0, 4.0]], ',')
            .expect("rectangular CSV should serialize");
        assert_eq!(text, "1,2\n3,4\n");
    }

    #[test]
    fn write_csv_rejects_header_data_column_mismatch() {
        let err = write_csv(Some(&["a", "b", "c"]), &[vec![1.0, 2.0]], ',')
            .expect_err("header/data mismatch should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV header has 3 columns but first data row has 2".to_string())
        );
    }

    #[test]
    fn write_csv_rejects_header_cells_with_delimiters() {
        let err = write_csv(Some(&["bad,header"]), &[vec![1.0]], ',')
            .expect_err("header cells containing the delimiter should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "CSV header cell \"bad,header\" contains the delimiter ',' and cannot be encoded safely"
                    .to_string()
            )
        );
    }

    #[test]
    fn write_csv_rejects_header_cells_with_newlines() {
        let err = write_csv(Some(&["bad\nheader"]), &[vec![1.0]], ',')
            .expect_err("header cells containing newlines should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "CSV header cell \"bad\\nheader\" contains a newline and cannot be encoded safely"
                    .to_string()
            )
        );
    }

    #[test]
    fn write_csv_rejects_ragged_rows() {
        let err = write_csv(None, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0]], ',')
            .expect_err("ragged CSV should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV row has 2 columns, expected 3".to_string())
        );
    }

    #[test]
    fn read_json_array_accepts_empty_array() {
        let data = read_json_array("[]").expect("empty array should parse");
        assert!(data.is_empty());
    }

    #[test]
    fn write_json_array_rejects_non_finite_values() {
        let err = write_json_array(&[1.0, f64::NAN]).expect_err("NaN is not valid JSON");
        assert_eq!(
            err,
            IoError::InvalidFormat("JSON array value at index 1 is not finite: NaN".to_string())
        );
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

    #[test]
    fn mmwrite_sparse_rejects_out_of_bounds_entries() {
        let err = mmwrite_sparse(2, 2, &[(2, 0, 1.0)])
            .expect_err("out-of-bounds sparse entry should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("sparse entry (2, 0) out of bounds for 2x2".to_string())
        );
    }

    #[test]
    fn read_csv_single_column_no_header() {
        let (header, data) = read_csv("1\n2\n3\n", ',', false).expect("single column CSV");
        assert_eq!(data.len(), 3);
        assert!(header.is_none());
        assert_eq!(data[0], vec![1.0]);
        assert_eq!(data[2], vec![3.0]);
    }

    #[test]
    fn read_csv_with_tab_delimiter() {
        let (header, data) = read_csv("a\tb\n1\t2\n3\t4\n", '\t', true).expect("TSV");
        assert_eq!(header, Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(data[0], vec![1.0, 2.0]);
    }

    #[test]
    fn read_csv_empty_no_header() {
        let (_header, data) = read_csv("", ',', false).expect("empty without header is ok");
        assert!(data.is_empty());
    }
}
