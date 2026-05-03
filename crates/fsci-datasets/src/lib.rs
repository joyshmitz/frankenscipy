#![forbid(unsafe_code)]

//! Sample datasets for FrankenSciPy.
//!
//! The public surface mirrors `scipy.datasets`: `ascent`, `face`,
//! `electrocardiogram`, `download_all`, and `clear_cache`. The arrays are
//! embedded deterministic fixtures with SciPy-compatible shapes and dtypes, so
//! callers get stable no-network example data for tests and demos.

/// ECG sampling rate used by SciPy's electrocardiogram dataset.
pub const ELECTROCARDIOGRAM_SAMPLE_RATE_HZ: f64 = 360.0;

const ASCENT_HEIGHT: usize = 512;
const ASCENT_WIDTH: usize = 512;
const FACE_HEIGHT: usize = 768;
const FACE_WIDTH: usize = 1024;
const FACE_CHANNELS: usize = 3;
const ECG_LEN: usize = 108_000;

/// Element type for a dataset fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetDType {
    U8,
    F64,
}

/// Public dataset family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetKind {
    Ascent,
    FaceRgb,
    FaceGray,
    Electrocardiogram,
}

/// Metadata for a stable fixture entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetFixture {
    pub name: &'static str,
    pub kind: DatasetKind,
    pub dtype: DatasetDType,
    pub shape: &'static [usize],
    pub element_count: usize,
}

/// Result of the embedded-cache compatibility helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheSummary {
    pub available: &'static [&'static str],
    pub downloaded: usize,
    pub removed: usize,
    pub cache_bytes: usize,
}

/// Dataset loader errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetError {
    UnknownFixture(String),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownFixture(name) => write!(f, "unknown dataset fixture: {name}"),
        }
    }
}

impl std::error::Error for DatasetError {}

/// Immutable 8-bit image dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageU8 {
    name: &'static str,
    height: usize,
    width: usize,
    channels: usize,
    data: Box<[u8]>,
}

impl ImageU8 {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn dtype(&self) -> DatasetDType {
        DatasetDType::U8
    }

    pub fn shape(&self) -> Vec<usize> {
        if self.channels == 1 {
            vec![self.height, self.width]
        } else {
            vec![self.height, self.width, self.channels]
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn pixel(&self, row: usize, col: usize, channel: usize) -> Option<u8> {
        if row >= self.height || col >= self.width || channel >= self.channels {
            return None;
        }
        let idx = ((row * self.width) + col) * self.channels + channel;
        self.data.get(idx).copied()
    }
}

/// Immutable one-dimensional floating-point signal dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalF64 {
    name: &'static str,
    sample_rate_hz: f64,
    data: Box<[f64]>,
}

impl SignalF64 {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn dtype(&self) -> DatasetDType {
        DatasetDType::F64
    }

    pub fn shape(&self) -> [usize; 1] {
        [self.data.len()]
    }

    pub fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    pub fn duration_seconds(&self) -> f64 {
        self.data.len() as f64 / self.sample_rate_hz
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }
}

/// Runtime-loaded dataset value.
#[derive(Debug, Clone, PartialEq)]
pub enum Dataset {
    ImageU8(ImageU8),
    SignalF64(SignalF64),
}

impl Dataset {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ImageU8(image) => image.name(),
            Self::SignalF64(signal) => signal.name(),
        }
    }

    pub fn dtype(&self) -> DatasetDType {
        match self {
            Self::ImageU8(image) => image.dtype(),
            Self::SignalF64(signal) => signal.dtype(),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Self::ImageU8(image) => image.shape(),
            Self::SignalF64(signal) => signal.shape().to_vec(),
        }
    }

    pub fn element_count(&self) -> usize {
        match self {
            Self::ImageU8(image) => image.data().len(),
            Self::SignalF64(signal) => signal.data().len(),
        }
    }
}

const ASCENT_SHAPE: &[usize] = &[ASCENT_HEIGHT, ASCENT_WIDTH];
const ASCENT_64_SHAPE: &[usize] = &[64, 64];
const FACE_RGB_SHAPE: &[usize] = &[FACE_HEIGHT, FACE_WIDTH, FACE_CHANNELS];
const FACE_RGB_128_SHAPE: &[usize] = &[128, 128, FACE_CHANNELS];
const FACE_GRAY_SHAPE: &[usize] = &[FACE_HEIGHT, FACE_WIDTH];
const FACE_GRAY_128_SHAPE: &[usize] = &[128, 128];
const ECG_SHAPE: &[usize] = &[ECG_LEN];
const ECG_1024_SHAPE: &[usize] = &[1024];
const ECG_4096_SHAPE: &[usize] = &[4096];

/// Ten stable fixture handles used by tests, examples, and fuzz targets.
pub const CANONICAL_FIXTURES: &[DatasetFixture] = &[
    DatasetFixture {
        name: "ascent",
        kind: DatasetKind::Ascent,
        dtype: DatasetDType::U8,
        shape: ASCENT_SHAPE,
        element_count: ASCENT_HEIGHT * ASCENT_WIDTH,
    },
    DatasetFixture {
        name: "ascent_top_left_64",
        kind: DatasetKind::Ascent,
        dtype: DatasetDType::U8,
        shape: ASCENT_64_SHAPE,
        element_count: 64 * 64,
    },
    DatasetFixture {
        name: "ascent_center_64",
        kind: DatasetKind::Ascent,
        dtype: DatasetDType::U8,
        shape: ASCENT_64_SHAPE,
        element_count: 64 * 64,
    },
    DatasetFixture {
        name: "face",
        kind: DatasetKind::FaceRgb,
        dtype: DatasetDType::U8,
        shape: FACE_RGB_SHAPE,
        element_count: FACE_HEIGHT * FACE_WIDTH * FACE_CHANNELS,
    },
    DatasetFixture {
        name: "face_gray",
        kind: DatasetKind::FaceGray,
        dtype: DatasetDType::U8,
        shape: FACE_GRAY_SHAPE,
        element_count: FACE_HEIGHT * FACE_WIDTH,
    },
    DatasetFixture {
        name: "face_center_rgb_128",
        kind: DatasetKind::FaceRgb,
        dtype: DatasetDType::U8,
        shape: FACE_RGB_128_SHAPE,
        element_count: 128 * 128 * FACE_CHANNELS,
    },
    DatasetFixture {
        name: "face_center_gray_128",
        kind: DatasetKind::FaceGray,
        dtype: DatasetDType::U8,
        shape: FACE_GRAY_128_SHAPE,
        element_count: 128 * 128,
    },
    DatasetFixture {
        name: "electrocardiogram",
        kind: DatasetKind::Electrocardiogram,
        dtype: DatasetDType::F64,
        shape: ECG_SHAPE,
        element_count: ECG_LEN,
    },
    DatasetFixture {
        name: "electrocardiogram_head_1024",
        kind: DatasetKind::Electrocardiogram,
        dtype: DatasetDType::F64,
        shape: ECG_1024_SHAPE,
        element_count: 1024,
    },
    DatasetFixture {
        name: "electrocardiogram_mid_4096",
        kind: DatasetKind::Electrocardiogram,
        dtype: DatasetDType::F64,
        shape: ECG_4096_SHAPE,
        element_count: 4096,
    },
];

/// Names of the five `scipy.datasets` public symbols.
pub fn public_api_symbols() -> &'static [&'static str] {
    &[
        "ascent",
        "electrocardiogram",
        "face",
        "download_all",
        "clear_cache",
    ]
}

/// Stable fixture registry.
pub fn canonical_fixtures() -> &'static [DatasetFixture] {
    CANONICAL_FIXTURES
}

/// Return the 512 x 512 8-bit grayscale ascent sample image.
pub fn ascent() -> ImageU8 {
    ascent_window("ascent", 0, 0, ASCENT_HEIGHT, ASCENT_WIDTH)
}

/// Return the default 768 x 1024 x 3 8-bit color face sample image.
pub fn face() -> ImageU8 {
    face_window("face", false, 0, 0, FACE_HEIGHT, FACE_WIDTH)
}

/// Return the 768 x 1024 8-bit grayscale face sample image.
pub fn face_gray() -> ImageU8 {
    face_window("face_gray", true, 0, 0, FACE_HEIGHT, FACE_WIDTH)
}

/// Return the 108000-point floating-point electrocardiogram sample signal.
pub fn electrocardiogram() -> SignalF64 {
    electrocardiogram_window("electrocardiogram", 0, ECG_LEN)
}

/// Compatibility no-op: datasets are embedded, so there is nothing to fetch.
pub fn download_all() -> CacheSummary {
    CacheSummary {
        available: public_api_symbols(),
        downloaded: 0,
        removed: 0,
        cache_bytes: 0,
    }
}

/// Compatibility no-op: datasets are embedded, so there is no external cache.
pub fn clear_cache() -> CacheSummary {
    CacheSummary {
        available: public_api_symbols(),
        downloaded: 0,
        removed: 0,
        cache_bytes: 0,
    }
}

/// Load one of the canonical fixture handles by name.
pub fn load_fixture(name: &str) -> Result<Dataset, DatasetError> {
    match name {
        "ascent" => Ok(Dataset::ImageU8(ascent())),
        "ascent_top_left_64" => Ok(Dataset::ImageU8(ascent_window(
            "ascent_top_left_64",
            0,
            0,
            64,
            64,
        ))),
        "ascent_center_64" => Ok(Dataset::ImageU8(ascent_window(
            "ascent_center_64",
            (ASCENT_HEIGHT - 64) / 2,
            (ASCENT_WIDTH - 64) / 2,
            64,
            64,
        ))),
        "face" => Ok(Dataset::ImageU8(face())),
        "face_gray" => Ok(Dataset::ImageU8(face_gray())),
        "face_center_rgb_128" => Ok(Dataset::ImageU8(face_window(
            "face_center_rgb_128",
            false,
            (FACE_HEIGHT - 128) / 2,
            (FACE_WIDTH - 128) / 2,
            128,
            128,
        ))),
        "face_center_gray_128" => Ok(Dataset::ImageU8(face_window(
            "face_center_gray_128",
            true,
            (FACE_HEIGHT - 128) / 2,
            (FACE_WIDTH - 128) / 2,
            128,
            128,
        ))),
        "electrocardiogram" => Ok(Dataset::SignalF64(electrocardiogram())),
        "electrocardiogram_head_1024" => Ok(Dataset::SignalF64(electrocardiogram_window(
            "electrocardiogram_head_1024",
            0,
            1024,
        ))),
        "electrocardiogram_mid_4096" => Ok(Dataset::SignalF64(electrocardiogram_window(
            "electrocardiogram_mid_4096",
            ECG_LEN / 2,
            4096,
        ))),
        other => Err(DatasetError::UnknownFixture(other.to_owned())),
    }
}

fn ascent_window(
    name: &'static str,
    row_start: usize,
    col_start: usize,
    height: usize,
    width: usize,
) -> ImageU8 {
    let mut data = Vec::with_capacity(height * width);
    for row in row_start..row_start + height {
        for col in col_start..col_start + width {
            data.push(ascent_value(row, col));
        }
    }
    ImageU8 {
        name,
        height,
        width,
        channels: 1,
        data: data.into_boxed_slice(),
    }
}

fn face_window(
    name: &'static str,
    gray: bool,
    row_start: usize,
    col_start: usize,
    height: usize,
    width: usize,
) -> ImageU8 {
    let channels = if gray { 1 } else { FACE_CHANNELS };
    let mut data = Vec::with_capacity(height * width * channels);
    for row in row_start..row_start + height {
        for col in col_start..col_start + width {
            let rgb = face_rgb_value(row, col);
            if gray {
                data.push(rgb_to_gray(rgb));
            } else {
                data.extend_from_slice(&rgb);
            }
        }
    }
    ImageU8 {
        name,
        height,
        width,
        channels,
        data: data.into_boxed_slice(),
    }
}

fn electrocardiogram_window(name: &'static str, start: usize, len: usize) -> SignalF64 {
    let data = (start..start + len)
        .map(electrocardiogram_value)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    SignalF64 {
        name,
        sample_rate_hz: ELECTROCARDIOGRAM_SAMPLE_RATE_HZ,
        data,
    }
}

fn ascent_value(row: usize, col: usize) -> u8 {
    let diagonal = (row * 3 + col * 5 + (row ^ col)) % 256;
    let ridge = ((row / 8 + col / 16) % 32) * 3;
    ((diagonal + ridge) % 256) as u8
}

fn face_rgb_value(row: usize, col: usize) -> [u8; FACE_CHANNELS] {
    let red = (col * 255 / (FACE_WIDTH - 1)) as u8;
    let green = (row * 255 / (FACE_HEIGHT - 1)) as u8;
    let blue = ((row * 7 + col * 11 + (row ^ col)) % 256) as u8;
    [red, green, blue]
}

fn rgb_to_gray([red, green, blue]: [u8; FACE_CHANNELS]) -> u8 {
    let weighted = 77_u16 * u16::from(red) + 150_u16 * u16::from(green) + 29_u16 * u16::from(blue);
    ((weighted + 128) >> 8) as u8
}

fn electrocardiogram_value(index: usize) -> f64 {
    let beat = index % 360;
    let t = index as f64 / ELECTROCARDIOGRAM_SAMPLE_RATE_HZ;
    let baseline = 0.05 * (std::f64::consts::TAU * t / 8.0).sin();
    let p_wave = triangular_pulse(beat, 70, 25, 0.12);
    let q_dip = -triangular_pulse(beat, 166, 8, 0.18);
    let r_peak = triangular_pulse(beat, 180, 10, 1.05);
    let s_dip = -triangular_pulse(beat, 194, 12, 0.28);
    let t_wave = triangular_pulse(beat, 270, 45, 0.32);
    let deterministic_noise = ((index * 37 % 101) as f64 - 50.0) * 0.0004;
    baseline + p_wave + q_dip + r_peak + s_dip + t_wave + deterministic_noise
}

fn triangular_pulse(beat: usize, center: usize, half_width: usize, amplitude: f64) -> f64 {
    let distance = beat.abs_diff(center);
    if distance > half_width {
        0.0
    } else {
        amplitude * (1.0 - distance as f64 / half_width as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_api_matches_scipy_datasets_symbols() {
        assert_eq!(
            public_api_symbols(),
            &[
                "ascent",
                "electrocardiogram",
                "face",
                "download_all",
                "clear_cache"
            ]
        );
    }

    #[test]
    fn core_loader_shapes_and_dtypes_match_documented_scipy_invariants() {
        let ascent = ascent();
        assert_eq!(ascent.shape(), vec![512, 512]);
        assert_eq!(ascent.dtype(), DatasetDType::U8);
        assert_eq!(ascent.data().len(), 512 * 512);

        let face = face();
        assert_eq!(face.shape(), vec![768, 1024, 3]);
        assert_eq!(face.dtype(), DatasetDType::U8);
        assert_eq!(face.data().len(), 768 * 1024 * 3);

        let face_gray = face_gray();
        assert_eq!(face_gray.shape(), vec![768, 1024]);
        assert_eq!(face_gray.channels(), 1);

        let ecg = electrocardiogram();
        assert_eq!(ecg.shape(), [108_000]);
        assert_eq!(ecg.dtype(), DatasetDType::F64);
        assert_eq!(ecg.sample_rate_hz(), 360.0);
        assert!((ecg.duration_seconds() - 300.0).abs() < 1e-12);
    }

    #[test]
    fn metamorphic_face_gray_matches_rgb_luma_projection() {
        let gray = face_gray();
        let probes = [
            (0, 0),
            (17, 29),
            (FACE_HEIGHT / 2, FACE_WIDTH / 2),
            (767, 1023),
        ];
        for (row, col) in probes {
            let expected = rgb_to_gray(face_rgb_value(row, col));
            assert_eq!(gray.pixel(row, col, 0), Some(expected));
        }
    }

    #[test]
    fn metamorphic_fixture_metadata_matches_loaded_values() -> Result<(), DatasetError> {
        assert_eq!(canonical_fixtures().len(), 10);
        for fixture in canonical_fixtures() {
            let loaded = load_fixture(fixture.name)?;
            assert_eq!(loaded.name(), fixture.name);
            assert_eq!(loaded.dtype(), fixture.dtype);
            assert_eq!(loaded.shape(), fixture.shape);
            assert_eq!(loaded.element_count(), fixture.element_count);
        }
        Ok(())
    }

    #[test]
    fn metamorphic_window_fixtures_match_full_dataset_slices() -> Result<(), DatasetError> {
        let ascent_full = ascent();
        let Dataset::ImageU8(ascent_head) = load_fixture("ascent_top_left_64")? else {
            return Err(DatasetError::UnknownFixture(
                "ascent_top_left_64".to_owned(),
            ));
        };
        for row in 0..64 {
            for col in 0..64 {
                assert_eq!(
                    ascent_head.pixel(row, col, 0),
                    ascent_full.pixel(row, col, 0)
                );
            }
        }

        let ecg_full = electrocardiogram();
        let Dataset::SignalF64(ecg_head) = load_fixture("electrocardiogram_head_1024")? else {
            return Err(DatasetError::UnknownFixture(
                "electrocardiogram_head_1024".to_owned(),
            ));
        };
        assert_eq!(ecg_head.data(), &ecg_full.data()[..1024]);
        Ok(())
    }

    #[test]
    fn loaders_return_fresh_immutable_buffers() {
        let left = ascent();
        let right = ascent();
        assert_eq!(left.data(), right.data());
        assert_ne!(left.data().as_ptr(), right.data().as_ptr());

        let left_ecg = electrocardiogram();
        let right_ecg = electrocardiogram();
        assert_eq!(left_ecg.data(), right_ecg.data());
        assert_ne!(left_ecg.data().as_ptr(), right_ecg.data().as_ptr());
    }

    #[test]
    fn embedded_cache_compatibility_helpers_are_noops() {
        let download = download_all();
        assert_eq!(download.available, public_api_symbols());
        assert_eq!(download.downloaded, 0);
        assert_eq!(download.cache_bytes, 0);

        let clear = clear_cache();
        assert_eq!(clear.available, public_api_symbols());
        assert_eq!(clear.removed, 0);
        assert_eq!(clear.cache_bytes, 0);
    }

    #[test]
    fn unknown_fixture_fails_closed() {
        assert_eq!(
            load_fixture("missing"),
            Err(DatasetError::UnknownFixture("missing".to_owned()))
        );
    }
}
